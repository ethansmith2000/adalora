#!/usr/bin/env python3
"""Offset-enabled token self-conditioning: bounded rate calibration then width trends."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

from dit_followup import private_supervisor
from train_cifar10 import (Config, build_model, learning_rate_scale, load_config, model_stats, write_json,
                          training_setup, make_optimizer, optimizer_group_manifest)
from train_dit_cifar10 import SOURCE_FILES

PILOT_SEEDS = (96, 97)
QUALITY_SEEDS = (98, 99)
HORIZON = 3520
PILOT_STEPS = 500
PROBES = (100, 250, 500)
FAMILIES = {
    'plain': dict(variant='plain'),
    'diagonal': dict(variant='diagonal'),
    'static_r8_diagonal': dict(variant='static_lora_diagonal'),
    'static_capacity': dict(variant='plain', mlp_ratio=1483/192),
    's12': dict(variant='dplr', n_basis=12),
    's24': dict(variant='dplr', n_basis=24),
    's48': dict(variant='dplr', n_basis=48),
    's24_detached': dict(variant='dplr', n_basis=24, detach_condition=True),
}
RATE_GRID = {f: (.25, 1., 4.) for f in ('s12', 's48', 's24_detached')}
RATE_GRID['static_r8_diagonal'] = (.25, 1., 4., 16.)
EXTRA_SOURCE = ('vit_token_study.py', 'test_vit_token_study.py', 'test_vit_factor_offsets.py',
                'vit_quality_diagnostics.py', 'dit_followup.py', 'summarize_dit_cifar10_screen.py', 'requirements.txt')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(family, seed, output, rate=1., pilot=False):
    features = FAMILIES[family]
    dynamic = features['variant'] == 'dplr'
    lowrank = dynamic or features['variant'] == 'static_lora_diagonal'
    # Forward research default: unrestricted zero-initialized factor offsets on.
    # Historical constructors/config APIs retain their old defaults for replay.
    return Config(**features, factor_offsets=dynamic, rank=8, width=192, heads=3, depth=6,
                  adapter_gain=1/math.sqrt(8) if lowrank else 1., seed=seed,
                  epochs=10, max_steps=PILOT_STEPS if pilot else HORIZON,
                  lr_schedule_steps=HORIZON, lr=5e-4, hyper_lr_multiplier=rate,
                  compile=True, num_workers=2, cpu_threads=2, log_every=100,
                  diagnostic_steps='0,1,2,3,5,10,50,100,250,500' if pilot else
                                   '0,1,2,3,5,10,50,100,250,500,1000,2000,3520',
                  output_dir=str(output))


def pilot_cases(output):
    cases = [('s24', seed, 1., make_config('s24', seed, output/f's24_lr1_s{seed}', pilot=True))
             for seed in PILOT_SEEDS]
    for family, rates in RATE_GRID.items():
        cases += [(family, seed, rate,
                   make_config(family, seed, output/f'{family}_lr{rate:g}_s{seed}', rate, pilot=True))
                  for rate in rates for seed in PILOT_SEEDS]
    return cases


def update(row, step):
    event = next(e for e in row['results']['token_diagnostics']['diagnostics'] if e['step'] == step)
    return statistics.mean(v['update_rms'] for v in event['fixed_lowrank_updates'])


def select_rates(rows):
    keyed = {(r['family'], r['seed'], r['rate']): r for r in rows}
    expected = {('s24', seed, 1.) for seed in PILOT_SEEDS} | {
        (family, seed, rate) for family, rates in RATE_GRID.items() for rate in rates for seed in PILOT_SEEDS}
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError('Require the exact 28-pilot grid; no incomplete or duplicate evidence')
    selections = {'s24': 1.}
    details = {}
    for family, rates in RATE_GRID.items():
        candidates = []
        for rate in rates:
            ratios = []
            for seed in PILOT_SEEDS:
                reference = keyed['s24', seed, 1.]
                candidate = keyed[family, seed, rate]
                ratios += [update(candidate, step)/update(reference, step) for step in PROBES]
            if not all(math.isfinite(r) and r > 0 for r in ratios):
                raise ValueError('Nonfinite or zero calibration updates')
            candidates.append(dict(rate=rate, ratios=ratios,
                                   within_bounds=all(.5 <= r <= 2. for r in ratios),
                                   score=statistics.mean(abs(math.log(r)) for r in ratios)))
        valid = [c for c in candidates if c['within_bounds']]
        details[family] = candidates
        if not valid:
            raise ValueError(f'No rate matches declared update bounds for {family}; withhold quality')
        chosen = min(valid, key=lambda c: (c['score'], c['rate']))
        selections[family] = chosen['rate']
    return dict(selected_rates=selections, candidates=details, reference='s24 offset-enabled, unit adapter LR',
                probe_steps=list(PROBES), ratio_bounds=[.5, 2.], seeds=list(PILOT_SEEDS),
                criterion='Closest geometric update magnitude using paired seeds and fixed-token intervals; no accuracy-based selection',
                limits='Early raw low-rank function updates only; no entire-network, diagonal or full-trajectory matching guarantee.')


def saved_activation_probe():
    import torch
    from conditioning import ConditionedLinear
    rows = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2035)
        for width in (12, 24, 48):
            layer = ConditionedLinear(torch.nn.Linear(192, 768), 'dplr', rank=8, n_basis=width,
                                      gain=1/math.sqrt(8), factor_offsets=True)
            params = {p.untyped_storage().data_ptr() for p in layer.parameters()}
            saved = {}
            def pack(t):
                key = t.untyped_storage().data_ptr()
                if key not in params:
                    saved[key] = t.untyped_storage().nbytes()
                return t
            x = torch.randn(2, 65, 192, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                layer(x, x).square().mean().backward()
            rows.append(dict(n_basis=width, unique_saved_nonparameter_storage_bytes=sum(saved.values()),
                             largest_activation_mixing_shape=[2, 65, width, 8]))
    return dict(rows=rows, dtype='FP32 CPU', input_shape=[2, 65, 192], output_width=768,
                definition='Unique saved tensor storage excluding parameter storage, including input/loss; one eager layer',
                limits='Not CUDA peak allocation or compiled saved activations; excludes backward transients and full-network storage.')


def prepare(output, launch=False):
    import torch
    torch.set_num_threads(2)
    output = Path(output).resolve(); root = Path(__file__).resolve().parent
    if output.exists(): raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free < 8*1024**3: raise RuntimeError('Require 8 GiB free for the 4 GiB bound and headroom')
    for path in Path('/tmp/vast_gpu_claims').glob('*.claim'):
        claim = json.loads(path.read_text())
        if claim.get('owner') == 'adalora': raise RuntimeError('Existing adalora GPU work must finish before this two-worker study')
    output.mkdir(parents=True); runner = output/'runner'; runner.mkdir()
    for name in sorted(set((*SOURCE_FILES, *EXTRA_SOURCE))): shutil.copy2(root/name, runner/name)
    parameters = {family: model_stats(build_model(make_config(family, 96, output/'unused')), make_config(family, 96, output/'unused'))['parameters'] for family in FAMILIES}
    assert abs(parameters['static_capacity']-parameters['s24']) <= 6*(2*192+1)
    protocol = dict(state='prepared', created_utc=now(), repo_root=str(root),
        purpose='Offset-enabled token self-conditioning width trend with separate bounded learning-rate calibration',
        forward_research_default='factor_offsets=True for new condition-generated U/V methods; explicit false ablations retained',
        source_sha256={p.name: digest(p) for p in runner.iterdir()}, families=FAMILIES, parameters=parameters,
        pilot_seeds=list(PILOT_SEEDS), quality_seeds=list(QUALITY_SEEDS), rate_grid=RATE_GRID,
        pilot_steps=PILOT_STEPS, train_steps=HORIZON, lr_schedule_steps=HORIZON, epochs=10,
        train_examples=45000, validation_examples=5000, split_seed=123, max_concurrent_workers=2,
        order_seed=2036, planned_runs=dict(preflight=2, calibration=28, quality=16),
        calibration_bounds=[.5, 2.], calibration_probes=list(PROBES),
        quality_gate='Both GPU preflights, exact pilot grid, source/config/data/gradient audit and every selected calibration ratio in bounds',
        resources=dict(estimated_gpu_hours='3–6 aggregate; approximately 1.5–3 wall hours at two GPUs, availability dependent',
                       max_our_gpus=2, disk_free_bytes_before_launch=free, retained_artifact_bound_bytes=4*1024**3,
                       artifact_purposes={'compact_evidence':'source/config/runtime/seed/split/metrics/logs/diagnostics/audits/reports',
                                          'compiler_cache':'compatibility/performance provenance; expected at most 3 GiB'},
                       retention='Compact evidence and declared compiler caches until explicitly removed'),
        fresh_initialization=True, weights_saved=False, periodic_checkpoints=False, test_evaluated=False,
        dedicated_speed_benchmarks=False, quality_selected_by_accuracy=False,
        limits=['Two fresh quality seeds and reused research holdout; sample SD is not a confidence interval.',
                'Offsets are the forward default by user preference; this study does not repeat the offset ablation.',
                's24 has the near-parameter-matched static-capacity control; s12/s48 are width trends with different budgets.',
                'All norms retain static affine parameters across controls; attention remains ordinary self-attention.',
                'Factor offsets are treated as biases with zero weight decay; learned factor bases retain weight decay .05.',
                'No AdaLN residual gate is introduced into this existing ViT; identity is the zero adapter and zero initial classifier.',
                'The adapter is parallel to the MLP input projection; normalized h drives both the branch and condition.',
                'Activation mixing avoids per-token generated factors but costs O(B*T*s*r) activations and more token-wise GEMM work.',
                'Calibration matches only early low-rank function updates; diagonal and full-network optimization may differ.',
                '500-step pilots use a prefix of the 3520-step schedule; quality uses a complete ten-epoch schedule.',
                'This is a short trend study, not an asymptotic quality comparison; concurrent throughput is not an isolated speed benchmark.'])
    write_json(output/'protocol.json', protocol)
    write_json(output/'saved_activation_probe.json', saved_activation_probe())
    planned = output/'planned_configs'; planned.mkdir()
    for family, seed, rate, cfg in pilot_cases(output/'calibration'):
        path = planned/(Path(cfg.output_dir).name+'.json'); write_json(path, asdict(cfg)); load_config(['--override_json', str(path)])
    if launch:
        private_supervisor(output, [('vit_token_controller', [sys.executable, '-u', str(runner/Path(__file__).name), 'run', '--output', str(output)])],
                           label='controller', autostart=True)
    return dict(output=str(output), launched=launch, parameters=parameters, max_our_gpus=2)


def execute_stage(output, name, cases, auditor=None):
    stage = output/name; stage.mkdir(); configs = stage/'configs'; configs.mkdir()
    jobs = []; programs = []
    for family, seed, rate, cfg in cases:
        path = configs/(Path(cfg.output_dir).name+'.json'); write_json(path, asdict(cfg)); load_config(['--override_json', str(path)])
        service = f'adalora_{output.name.replace("-", "_")}_{name}_{path.stem}'
        command = ['/workspace/bin/gpu-claim', 'run', '--owner', 'adalora', '--job', service, '--wait', '--',
                   'env', f'TORCHINDUCTOR_CACHE_DIR={output}/compile-cache/{name}', 'TORCHINDUCTOR_COMPILE_THREADS=2',
                   sys.executable, '-u', str(output/'runner/train_cifar10.py'), '--override_json', str(path)]
        jobs.append(dict(family=family, seed=seed, rate=rate, config=str(path), output_dir=cfg.output_dir, service=service))
        programs.append((service, command))
    random.Random(2036).shuffle(jobs)
    write_json(stage/'screen.json', dict(jobs=jobs, created_utc=now()))
    first = [j['service'] for j in jobs[:2]]; submitted = set(first); claims = {}; control = None
    try:
        control = private_supervisor(stage, programs, start_programs=first)
        while True:
            for path in Path('/tmp/vast_gpu_claims').glob('*.claim'):
                try: claim = json.loads(path.read_text())
                except (OSError, ValueError): continue
                if claim.get('job') in submitted: claims[claim['job']] = claim
            write_json(stage/'hardware_claims.json', claims)
            status = subprocess.run(['supervisorctl', '-c', control, 'status'], capture_output=True, text=True).stdout
            if not status.strip() or any(w in status for w in ('FATAL', 'BACKOFF')): raise RuntimeError(status)
            log = (stage/'supervisord.log').read_text()
            complete = {j['service'] for j in jobs if f'exited: {j["service"]} (exit status 0; expected)' in log}
            for line in status.splitlines():
                if line.split()[0] in submitted and 'EXITED' in line and line.split()[0] not in complete: raise RuntimeError('Failed worker: '+line)
            for job in [j for j in jobs if j['service'] not in submitted][:2-len(submitted-complete)]:
                subprocess.run(['supervisorctl', '-c', control, 'start', job['service']], check=True); submitted.add(job['service'])
            progress = dict(utc=now(), stage=name, completed=len(complete), submitted=len(submitted), total=len(jobs), active=len(submitted-complete))
            assert progress['active'] <= 2
            with (output/'progress.jsonl').open('a') as stream: stream.write(json.dumps(progress)+'\n')
            print(json.dumps(progress), flush=True)
            if len(complete) == len(jobs): break
            time.sleep(20)
    finally:
        if control: subprocess.run(['supervisorctl', '-c', control, 'shutdown'], check=True)
    rows = (auditor or audit_stage)(output, name)
    write_json(stage/'summary.json', dict(rows=rows, audit_passed=True))
    return rows


def audit_stage(output, name, config_factory=None, preflight_configs=None):
    import torch
    from vit_quality_diagnostics import parameter_digest
    torch.set_num_threads(2)
    p = json.loads((output/'protocol.json').read_text()); stage = output/name
    config_factory = config_factory or make_config
    screen = json.loads((stage/'screen.json').read_text()); claims = json.loads((stage/'hardware_claims.json').read_text())
    log = (stage/'supervisord.log').read_text(); assert not (stage/'supervisor.sock').exists()
    assert len(screen['jobs']) == p['planned_runs'][name] and set(claims) == {j['service'] for j in screen['jobs']}
    for filename, sha in p['source_sha256'].items(): assert digest(output/'runner'/filename) == sha
    rows = []; split = None; runtime = None; base = {}; streams = {}
    for job in screen['jobs']:
        out = Path(job['output_dir']); cfg = json.loads((out/'config.json').read_text()); result = json.loads((out/'results.json').read_text())
        recorded = json.loads(Path(job['config']).read_text()); assert cfg == recorded
        if name != 'preflight':
            assert cfg == asdict(config_factory(job['family'], job['seed'], out, job['rate'], name == 'calibration'))
        elif preflight_configs is not None:
            assert cfg == preflight_configs[job['family'], job['seed'], job['rate']]
        else:
            assert job['family'] in ('s48', 'static_capacity') and job['seed'] == 96 and job['rate'] == 1.
            expected = replace(make_config(job['family'],96,out), max_steps=12, lr_schedule_steps=12,
                               train_limit=328, warmup_ratio=2/12, log_every=6, diagnostic_steps='0,1,2,3,5,10,12')
            assert cfg == asdict(expected)
        metadata = json.loads((out/'dataset.json').read_text()); provenance = json.loads((out/'provenance.json').read_text())
        artifact = json.loads((out/'artifact_manifest.json').read_text())
        events = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        assert events[-1]['event'] == 'complete' and result['completed_steps'] == cfg['max_steps']
        assert f'exited: {job["service"]} (exit status 0; expected)' in log
        assert not artifact['weights_saved'] and not artifact['periodic_checkpoints'] and result['test'] is None
        assert not [f for f in out.rglob('*') if f.suffix in ('.pt', '.pth', '.ckpt', '.safetensors')]
        def finite(value):
            if isinstance(value, float): assert math.isfinite(value)
            if isinstance(value, dict):
                for v in value.values(): finite(v)
            if isinstance(value, list):
                for v in value: finite(v)
        finite(result); finite(events)
        train_indices, val_indices = metadata['train_indices'], metadata['validation_indices']
        assert len(train_indices) == (328 if name == 'preflight' else 45000) and len(val_indices) == 5000
        assert len(set(train_indices)) == len(train_indices) and len(set(val_indices)) == 5000 and not set(train_indices) & set(val_indices)
        assert all(0 <= i < 50000 for i in train_indices+val_indices)
        current_split = {k:metadata[k] for k in ('train_indices','validation_indices','source_fingerprint','location')}
        assert split is None or split == current_split; split = current_split
        current_runtime = {k:provenance[k] for k in ('python','torch','cuda_wheel','package_versions','source_sha256')}
        assert runtime is None or runtime == current_runtime; runtime = current_runtime
        for filename, sha in provenance['source_sha256'].items(): assert sha == p['source_sha256'][filename]
        claim = claims[job['service']]; diag = result['token_diagnostics']
        assert claim['owner'] == 'adalora' and claim['uuid'].removeprefix('GPU-') == diag['hardware']['device_uuid'].removeprefix('GPU-')
        assert [e['step'] for e in diag['diagnostics']] == [int(s) for s in cfg['diagnostic_steps'].split(',')]
        assert diag['diagnostics'] == [e for e in events if e['event']=='token_diagnostic']
        torch.manual_seed(cfg['seed']); model = build_model(Config(**cfg)); stats = model_stats(model, Config(**cfg))
        assert all(stats[k] == result[k] for k in stats)
        assert result['parameters'] == p['parameters'][job['family']]
        assert parameter_digest(model) == diag['base_initialization_sha256'] and parameter_digest(model, True) == diag['adapter_initialization_sha256']
        if cfg['mlp_ratio'] == 4.:
            base.setdefault((cfg['seed'], cfg.get('mlp_init', 'legacy')), []).append(diag['base_initialization_sha256'])
        from train_cifar10 import hyperparameter_ids
        ids = hyperparameter_ids(model); expected_gradients = {n for n,v in model.named_parameters() if id(v) in ids}
        assert set(diag['first_nonzero_gradient']) == expected_gradients and all(v <= 10 for v in diag['first_nonzero_gradient'].values())
        assert all(v['delta_rms'] == 0 for v in diag['diagnostics'][0]['fixed_lowrank_updates'])
        trace = [{k:e[k] for k in ('step','batch_images_sha256','batch_labels_sha256')} for e in diag['diagnostics'][1:]]
        streams.setdefault(cfg['seed'], []).append(trace)
        n = len(train_indices); q, remainder = divmod(cfg['max_steps'], math.ceil(n/cfg['batch_size']))
        assert result['processed_images'] == q*n+min(remainder*cfg['batch_size'], n)
        assert result['training_setup'] == training_setup(Config(**cfg),cfg['lr_schedule_steps'])
        if 'optimizer_groups' in result:
            assert result['optimizer_groups'] == optimizer_group_manifest(model, make_optimizer(model,Config(**cfg)))
        assert all(e['examples'] == 5000 for e in events if e['event']=='validation')
        assert [e['step'] for e in events if e['event']=='validation'] == sorted(set([min(k*math.ceil(n/cfg['batch_size']),cfg['max_steps']) for k in range(1,math.ceil(cfg['max_steps']/math.ceil(n/cfg['batch_size']))+1)]))
        for e in events:
            if e['event'] == 'train':
                scale = learning_rate_scale(Config(**cfg), e['step']-1, cfg['lr_schedule_steps'])
                assert math.isclose(e['lr'], cfg['lr']*scale, rel_tol=1e-12)
                if 'optimizer_lrs' in e:
                    expected = [g['initial_lr']*scale for g in result['optimizer_groups']]
                    assert len(e['optimizer_lrs']) == len(expected)
                    assert all(math.isclose(a,b,rel_tol=1e-12)for a,b in zip(e['optimizer_lrs'],expected))
        rows.append(evidence_row(job, cfg, result, out))
    assert all(len(set(values)) == 1 for values in base.values())
    assert all(all(v == traces[0] for v in traces) for traces in streams.values())
    return rows


def evidence_row(job, config, result, output):
    return {**job, 'launched_config_path': job['config'], 'config': config, 'results': result,
            'evidence_sha256': {f:digest(output/f) for f in ('config.json','results.json','metrics.jsonl','dataset.json','provenance.json','artifact_manifest.json')}}


def run(output, recover_preflight=False):
    output = Path(output).resolve(); p = json.loads((output/'protocol.json').read_text())
    if recover_preflight:
        if (p['state'] != 'failed' or p.get('failure_type') != 'TypeError'
                or p.get('failure') != "dict() got multiple values for keyword argument 'config'"):
            raise ValueError('Recovery requires the specific preflight report-assembly failure')
        backup = output/'preflight_failure_protocol.json'
        if backup.exists(): raise FileExistsError('Preserve the original recovery record; no implicit retry')
        shutil.copy2(output/'protocol.json',backup)
        p['preflight_reporting_recovery'] = dict(failed_protocol_sha256=digest(backup),
            orchestration_source_sha256=digest(Path(__file__)), original_worker_outputs_unchanged=True,
            preflight_rerun=False, description='Correct duplicate config field in audited row assembly; reuse successful preflight only')
    elif p['state'] != 'prepared': raise ValueError('Fresh study only; no implicit resume')
    try:
        if recover_preflight:
            preflight = audit_stage(output,'preflight')
            write_json(output/'preflight/summary.json', dict(rows=preflight,audit_passed=True))
        else:
            p.update(state='preflight', started_utc=now()); write_json(output/'protocol.json',p)
            cases = []
            for family in ('s48','static_capacity'):
                cfg = replace(make_config(family,96,output/'preflight'/family), max_steps=12, lr_schedule_steps=12,
                              train_limit=328, warmup_ratio=2/12, log_every=6, diagnostic_steps='0,1,2,3,5,10,12')
                cases.append((family,96,1.,cfg))
            preflight = execute_stage(output,'preflight',cases)
        p.update(state='calibration', preflight_complete_utc=now()); write_json(output/'protocol.json',p)
        pilots = execute_stage(output,'calibration',pilot_cases(output/'calibration'))
        selection = select_rates(pilots); write_json(output/'calibration.json',selection)
        p.update(state='quality', calibration_complete_utc=now(), calibration_sha256=digest(output/'calibration.json')); write_json(output/'protocol.json',p)
        cases = [(family,seed,selection['selected_rates'].get(family,1.),
                  make_config(family,seed,output/'quality'/f'{family}_s{seed}',selection['selected_rates'].get(family,1.)))
                 for family in FAMILIES for seed in QUALITY_SEEDS]
        quality = execute_stage(output,'quality',cases)
        write_json(output/'report.json',dict(preflight=preflight,calibration=pilots,quality=quality))
        p.update(state='complete',completed_utc=now()); write_json(output/'protocol.json',p)
        audit(output); write_summary(output,quality,p)
    except BaseException as error:
        p.update(state='failed', failed_utc=now(), failure_type=type(error).__name__, failure=str(error)); write_json(output/'protocol.json',p)
        raise


def audit(output):
    output = Path(output).resolve(); p = json.loads((output/'protocol.json').read_text()); assert p['state']=='complete'
    report = json.loads((output/'report.json').read_text())
    for name in ('preflight','calibration','quality'): assert audit_stage(output,name) == report[name]
    selection = select_rates(report['calibration']); assert selection == json.loads((output/'calibration.json').read_text())
    assert digest(output/'calibration.json') == p['calibration_sha256']
    rows = report['quality']; assert {(r['family'],r['seed']) for r in rows} == {(f,s) for f in FAMILIES for s in QUALITY_SEEDS}
    assert len(rows)==16 and all(r['rate']==selection['selected_rates'].get(r['family'],1.) for r in rows)
    progress = [json.loads(line) for line in (output/'progress.jsonl').read_text().splitlines()]
    assert all(e['active']==e['submitted']-e['completed']<=2 for e in progress)
    retained = sum(f.stat().st_size for f in output.rglob('*') if f.is_file())
    assert retained < p['resources']['retained_artifact_bound_bytes']
    checks = ('all_46_runs_complete_successful_worker_exits','unchanged_frozen_sources_runtime',
              'paired_base_initialization_and_observed_training_streams','zero_start_and_all_adapter_gradients_wake',
              'prefix_schedule_and_validation_denominators','gpu_claim_identity_and_two_worker_cap',
              'exact_calibration_grid_and_update_only_selection','separate_fresh_quality_seeds_and_frozen_rates',
              'no_weights_test_evaluation_and_storage_bound')
    result = dict(checks={k:True for k in checks}, report_sha256=digest(output/'report.json'), bytes_retained=retained)
    write_json(output/'audit.json',result); return result


def write_summary(output,rows,p):
    aggregates = []
    for family in FAMILIES:
        values = sorted([r for r in rows if r['family']==family], key=lambda r:r['seed'])
        metrics = {}
        for key, getter in {'accuracy_percent':lambda r:100*r['results']['validation']['accuracy'],
                            'training_peak_mib':lambda r:r['results']['peak_training_allocated_bytes']/2**20,
                            'loss':lambda r:r['results']['validation']['loss']}.items():
            xs = [getter(r) for r in values]; metrics[key] = dict(values=xs, mean=statistics.mean(xs), sd=statistics.stdev(xs))
        aggregates.append(dict(family=family,parameters=values[0]['results']['parameters'],rate=values[0]['rate'],metrics=metrics))
    write_json(output/'comparison.json',dict(aggregates=aggregates,limits=p['limits']))
    lines=['# ViT token self-conditioning','', 'Ten epochs; paired fresh seeds 98/99. Means ± sample SD; offsets on in dynamic methods.', '',
           '| Family | Parameters | Adapter LR multiplier | Accuracy (%) | Training peak MiB |','|---|---:|---:|---:|---:|']
    for row in aggregates:
        a=row['metrics']['accuracy_percent'];m=row['metrics']['training_peak_mib']
        lines.append(f"| {row['family']} | {row['parameters']:,} | {row['rate']:g} | {a['mean']:.2f} ± {a['sd']:.2f} | {m['mean']:.1f} ± {m['sd']:.1f} |")
    lines+=['','Limits: '+' '.join(p['limits']),'','[Evidence](report.json) · [Audit](audit.json) · [Calibration](calibration.json)','']
    (output/'summary.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--output',type=Path);parser.add_argument('--launch',action='store_true')
    parser.add_argument('--recover-completed-preflight',action='store_true');args=parser.parse_args()
    output=args.output or Path('runs')/('vit-token-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action=='prepare': print(json.dumps(prepare(output,args.launch)),flush=True)
    elif args.action=='run': run(output,args.recover_completed_preflight)
    else: print(json.dumps(audit(output),indent=2))

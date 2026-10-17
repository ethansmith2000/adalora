#!/usr/bin/env python3
"""Bounded magnitude/gradient-route diagnostic prefixes; no accuracy-based selection."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import sys

import torch
import vit_initializer_study as initializer
from dit_followup import private_supervisor
from train_cifar10 import build_model, load_config, model_stats, write_json
from train_dit_cifar10 import SOURCE_FILES

common = initializer.common
SEEDS = (110, 111)
PILOT_STEPS = 1200
FAMILIES = {
    'current': dict(condition_normalization='none', condition_gradient_scale=1.),
    'condition_rms': dict(condition_normalization='rms', condition_gradient_scale=1.),
    'half_condition_gradient': dict(condition_normalization='none', condition_gradient_scale=.5),
}
EXTRA_SOURCE = ('vit_condition_route_study.py', 'test_vit_condition_route_study.py', 'test_vit_condition_routes.py',
                'vit_initializer_study.py', 'vit_token_study.py', 'vit_quality_diagnostics.py', 'dit_followup.py',
                'summarize_dit_cifar10_screen.py', 'requirements.txt')


def make_config(family, seed, output, rate=1., pilot=False):
    if rate != 1.: raise ValueError('No LR search in this fixed diagnostic')
    features = (dict(condition_normalization='rms', condition_gradient_scale=.5)
                if family == 'combined_preflight' else FAMILIES[family])
    base = initializer.make_config('dplr__xavier_gelu', seed, output)
    return replace(base, **features, routing_diagnostics=True, max_steps=PILOT_STEPS,
                   diagnostic_steps='0,1,2,3,5,10,50,100,250,500,800,1200')


def pilot_cases(output):
    return [(f, s, 1., make_config(f, s, output/f'{f}_s{s}')) for f in FAMILIES for s in SEEDS]


def preflight_cases(output):
    return [(f, 110, 1., replace(make_config(f, 110, output/f), max_steps=12, lr_schedule_steps=12,
                                train_limit=328, warmup_ratio=2/12, log_every=6, diagnostic_steps='0,1,2,3,5,10,12'))
            for f in ('current', 'combined_preflight')]


def audit_stage(output, name):
    configs = {(f, s, r): asdict(c) for f, s, r, c in preflight_cases(output/'preflight')}
    rows = common.audit_stage(output, name, config_factory=make_config, preflight_configs=configs)
    for row in rows:
        events = row['results']['token_diagnostics']['diagnostics']
        for e in events:
            route = e['gradient_routing']; assert route['examples'] == 4 and len(route['blocks']) == 6
            assert all(b['reconstruction_relative_error'] < 1e-4 and b['forward_reconstruction_max_error'] < 1e-5 for b in route['blocks'])
        assert all(b['total_gradient_rms'] == 0 for b in events[0]['gradient_routing']['blocks'])
        assert any(b['condition_gradient_rms'] > 0 for b in events[-1]['gradient_routing']['blocks'])
        assert all(len(e['fixed_mlp_updates']) == len(e['live_branches']) == 6 for e in events)
    return rows


def prepare(output, launch=False):
    torch.set_num_threads(2); root = Path(__file__).resolve().parent; output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    initializer.ensure_no_active_work(); free = shutil.disk_usage(root).free
    if free < 8*1024**3: raise RuntimeError('Require 8 GiB free for the 4 GiB bound and headroom')
    output.mkdir(parents=True); runner = output/'runner'; runner.mkdir()
    for n in sorted(set((*SOURCE_FILES, *EXTRA_SOURCE))): shutil.copy2(root/n, runner/n)
    families = [*FAMILIES, 'combined_preflight']
    counts = {f: model_stats(build_model(make_config(f, 110, output/'unused')), make_config(f, 110, output/'unused'))['parameters'] for f in families}
    assert len(set(counts.values())) == 1 and counts['current'] == 4342378
    p = dict(state='prepared', created_utc=common.now(), repo_root=str(root),
             purpose='Measure condition magnitude and local task-loss gradient routes with two minimal interventions',
             source_sha256={f.name: common.digest(f) for f in runner.iterdir() if f.is_file()}, parameters=counts,
             families=FAMILIES, pilot_seeds=list(SEEDS), pilot_steps=PILOT_STEPS, lr_schedule_steps=10560,
             max_concurrent_workers=2, planned_runs=dict(preflight=2, calibration=6), recipe_selection=False,
             quality_runs=0, quality_gate='None: no automatic long training or accuracy-based candidate selection',
             weights_saved=False, periodic_checkpoints=False, test_evaluated=False,
             resources=dict(max_our_gpus=2, estimated_gpu_hours='0.2–0.6 aggregate; roughly 10–20 wall minutes including cold compilation',
                            disk_free_bytes_before_launch=free, retained_artifact_bound_bytes=4*1024**3,
                            artifact_purposes={'compact_evidence':'frozen sources/configs/runtime/seeds/splits/metrics/logs/audits and diagnostic route/magnitude records',
                                               'compiler_cache':'compatibility/performance provenance, expected at most 3 GiB'},
                            retention='Compact evidence and declared compiler caches until explicitly removed; historical retention unchanged'),
             limits=['Two new seeds, 1200-step prefixes of the same 30-epoch schedule; not a final-quality or generalization comparison.',
                     'Condition-only RMS normalization and gradient scale0.5 affect both affine and low-rank heads; routes are measured separately.',
                     'Normalization is parameter-free and preserves direction, while removing condition RMS information.',
                     'Gradient scaling preserves the forward function and parameter gradients at a fixed state, but changes the training trajectory.',
                     'VJPs use the actual smoothed task loss on four fixed validation images in FP32 eager eval-mode; not training-batch or whole-network conflict.',
                     'Raw and applied condition gradients are retained; zero-gradient alignment is undefined, rather than an apparent cosine0.',
                     'Measured cosine/opposition is descriptive; no theorem or rule that negative alignment harms optimization.',
                     'No gradient/magnitude threshold selects a recipe; no early accuracy winner or automatic longer follow-up.',
                     'Historical weights/runs are not reused; default training behavior and old frozen snapshots remain unchanged.'])
    write_json(output/'protocol.json', p); configs = output/'planned_configs'; configs.mkdir()
    for f, s, r, cfg in [*preflight_cases(output/'preflight'), *pilot_cases(output/'calibration')]:
        path = configs/(Path(cfg.output_dir).name+'.json'); write_json(path, asdict(cfg)); load_config(['--override_json', str(path)])
    assert not torch.cuda.is_initialized()
    if launch: launch_controller(output)
    return dict(output=str(output), launched=launch, planned_runs=p['planned_runs'], quality_runs=0, max_our_gpus=2)


def launch_controller(output):
    output = Path(output).resolve(); p = json.loads((output/'protocol.json').read_text())
    assert p['state'] == 'prepared'; initializer.ensure_no_active_work()
    assert all(common.digest(output/'runner'/n) == h for n, h in p['source_sha256'].items())
    return private_supervisor(output, [('vit_condition_route_controller', [sys.executable, '-u', str(output/'runner/vit_condition_route_study.py'), 'run', '--output', str(output)])], label='controller', autostart=True)


def verify_pairs(rows):
    expected = {(f, s, 1.) for f in FAMILIES for s in SEEDS}
    assert len(rows) == 6 and {(r['family'], r['seed'], r['rate']) for r in rows} == expected
    for seed in SEEDS:
        ds = [r['results']['token_diagnostics'] for r in rows if r['seed'] == seed]
        for k in ['base_initialization_sha256', 'adapter_initialization_sha256', 'probe_sha256', 'live_images_sha256',
                  'gradient_probe_images_sha256', 'gradient_probe_labels_sha256']:
            assert len({d[k] for d in ds}) == 1


def run(output):
    output = Path(output).resolve(); p = json.loads((output/'protocol.json').read_text()); assert p['state'] == 'prepared'
    try:
        p.update(state='preflight', started_utc=common.now()); write_json(output/'protocol.json', p)
        pf = common.execute_stage(output, 'preflight', preflight_cases(output/'preflight'), auditor=audit_stage)
        p.update(state='calibration', preflight_complete_utc=common.now()); write_json(output/'protocol.json', p)
        rows = common.execute_stage(output, 'calibration', pilot_cases(output/'calibration'), auditor=audit_stage)
        verify_pairs(rows); write_json(output/'report.json', dict(preflight=pf, diagnostic_pilots=rows))
        p.update(state='complete', completed_utc=common.now()); write_json(output/'protocol.json', p)
        audit(output); write_summary(output, rows, p)
    except BaseException as error:
        p.update(state='failed', failed_utc=common.now(), failure_type=type(error).__name__, failure=str(error)); write_json(output/'protocol.json', p); raise


def audit(output):
    output = Path(output).resolve(); p = json.loads((output/'protocol.json').read_text()); assert p['state'] == 'complete'
    pf = audit_stage(output, 'preflight'); rows = audit_stage(output, 'calibration'); verify_pairs(rows)
    assert json.loads((output/'report.json').read_text()) == dict(preflight=pf, diagnostic_pilots=rows)
    progress = [json.loads(s) for s in (output/'progress.jsonl').read_text().splitlines()]
    assert all(e['active'] == e['submitted']-e['completed'] <= 2 for e in progress)
    retained = sum(f.stat().st_size for f in output.rglob('*') if f.is_file()); assert retained < p['resources']['retained_artifact_bound_bytes']
    result = dict(checks={n: True for n in ('compiled_preflights_and_exact_six_prefixes', 'source_runtime_config_optimizer_split_and_observed_streams',
        'paired_initialization_and_fixed_probe_images_labels', 'local_task_loss_gradient_and_forward_reconstruction',
        'zero_start_and_finite_nonzero_condition_routes', 'two_gpu_worker_cap_and_claim_identity', 'no_recipe_selection_quality_weights_or_test_and_storage_bound')},
        report_sha256=common.digest(output/'report.json'), bytes_retained=retained)
    write_json(output/'audit.json', result); return result


def write_summary(output, rows, p):
    records = []
    for row in rows:
        for e in row['results']['token_diagnostics']['diagnostics']:
            if e['step'] not in (100, 500, 1200): continue
            blocks = e['gradient_routing']['blocks']
            cosine = [b['value_raw_condition_alignment']['cosine'] for b in blocks if b['value_raw_condition_alignment']['cosine'] is not None]
            records.append(dict(family=row['family'], seed=row['seed'], step=e['step'],
                                mean_input_rms=statistics.mean(b['input_rms'] for b in blocks), mean_condition_rms=statistics.mean(b['condition_rms'] for b in blocks),
                                mean_applied_condition_to_value_ratio=statistics.mean(b['condition_to_value_ratio'] for b in blocks),
                                mean_raw_condition_cosine=statistics.mean(cosine) if cosine else None))
    write_json(output/'comparison.json', dict(records=records, recipe_selection=False, limits=p['limits']))
    lines=['# ViT condition-route diagnostic prefixes','','1200 steps of a10560-step schedule; two seeds; no quality-based selection.','',
           '| Family | Seed | Step | Input RMS | Condition RMS | Applied condition/value gradient ratio | Raw condition/value cosine |','|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        cosine='undefined' if r['mean_raw_condition_cosine'] is None else f"{r['mean_raw_condition_cosine']:.3f}"
        lines.append(f"| {r['family']} | {r['seed']} | {r['step']} | {r['mean_input_rms']:.3f} | {r['mean_condition_rms']:.3f} | {r['mean_applied_condition_to_value_ratio']:.3f} | {cosine} |")
    lines += ['', 'Ratios and cosines are arithmetic means of six per-block measurements; raw per-block records are retained. They do not establish a causal optimization problem or predict final quality.', '', 'Limits: '+' '.join(p['limits']), '', '[Evidence](report.json) · [Audit](audit.json) · [Protocol](protocol.json)']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('prepare','launch','run','audit'))
    parser.add_argument('--output', type=Path); parser.add_argument('--launch', action='store_true'); args = parser.parse_args()
    output = args.output or Path('runs')/('vit-condition-routes-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action == 'prepare': print(json.dumps(prepare(output, args.launch)), flush=True)
    elif args.action == 'launch': print(launch_controller(output), flush=True)
    elif args.action == 'run': run(output)
    else: print(json.dumps(audit(output), indent=2))

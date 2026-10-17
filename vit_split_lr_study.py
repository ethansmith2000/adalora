#!/usr/bin/env python3
"""Split affine/low-rank rates, update-only calibration, then fresh 30-epoch ViT runs."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import vit_token_study as common
from dit_followup import private_supervisor
from train_cifar10 import Config, build_model, load_config, model_stats, write_json
from train_dit_cifar10 import SOURCE_FILES

PILOT_SEEDS = (100, 101)
QUALITY_SEEDS = (102, 103)
HORIZON = 10560
PILOT_STEPS = 1200
PROBES = (500, 800, 1200)
FAMILIES = {
    'plain': dict(variant='plain'),
    'diagonal': dict(variant='diagonal'),
    'static_capacity': dict(variant='plain', mlp_ratio=1483/192),
    's24': dict(variant='dplr', n_basis=24),
    's48': dict(variant='dplr', n_basis=48),
    'static_unit': dict(variant='static_lora_diagonal'),
    'static_split': dict(variant='static_lora_diagonal'),
    'static_joint_12': dict(variant='static_lora_diagonal'),
}
RATE_GRID = {'s48': (.25, .5, 1.), 'static_split': (4., 8., 12., 16.)}
FIXED_PILOTS = {'s24': 1., 'static_unit': 1., 'static_joint_12': 12.}
EXTRA_SOURCE = ('vit_token_study.py', 'vit_split_lr_study.py', 'test_vit_split_lr.py',
                'test_vit_token_study.py', 'test_vit_factor_offsets.py', 'vit_quality_diagnostics.py',
                'dit_followup.py', 'summarize_dit_cifar10_screen.py', 'requirements.txt')


def make_config(family, seed, output, rate=1., pilot=False):
    features = FAMILIES[family]
    variant = features['variant']; lowrank = variant in ('dplr', 'static_lora_diagonal')
    if family == 'static_joint_12' and rate != 12.:
        raise ValueError('Joint control requires its frozen multiplier 12')
    if family in ('plain', 'diagonal', 'static_capacity', 's24', 'static_unit') and rate != 1.:
        raise ValueError('Fixed family requires its declared unit multiplier')
    return Config(**features, factor_offsets=variant=='dplr', rank=8, width=192, heads=3, depth=6,
                  adapter_gain=1/math.sqrt(8) if lowrank else 1., seed=seed,
                  epochs=30, max_steps=PILOT_STEPS if pilot else HORIZON,
                  lr_schedule_steps=HORIZON, lr=5e-4, hyper_lr_multiplier=1.,
                  affine_lr_multiplier=(12. if family=='static_joint_12' else 1.) if variant!='plain' else None,
                  lowrank_lr_multiplier=rate if lowrank else None,
                  compile=True, num_workers=2, cpu_threads=2, log_every=100,
                  diagnostic_steps=('0,1,2,3,5,10,50,100,250,500,800,1200' if pilot else
                                    '0,1,2,3,5,10,50,100,250,500,800,1200,2000,3520,7040,10560'),
                  output_dir=str(output))


def pilot_cases(output):
    grid = {f:(r,) for f,r in FIXED_PILOTS.items()} | RATE_GRID
    return [(f,s,r,make_config(f,s,output/f'{f}_lr{r:g}_s{s}',r,True))
            for f,rates in grid.items() for r in rates for s in PILOT_SEEDS]


def preflight_cases(output):
    cases = []
    for family,rate in (('s48',.5),('static_split',12.)):
        cfg = replace(make_config(family,100,output/family,rate),max_steps=12,lr_schedule_steps=12,
                      train_limit=328,warmup_ratio=2/12,log_every=6,diagnostic_steps='0,1,2,3,5,10,12')
        cases.append((family,100,rate,cfg))
    return cases


def audit_stage(output, name):
    preflight = {(f,s,r):asdict(cfg) for f,s,r,cfg in preflight_cases(output/'preflight')}
    return common.audit_stage(output,name,config_factory=make_config,preflight_configs=preflight)


def select_rates(rows):
    expected = {(f,s,r)for f,r in FIXED_PILOTS.items() for s in PILOT_SEEDS} | {
        (f,s,r)for f,rates in RATE_GRID.items()for r in rates for s in PILOT_SEEDS}
    keyed = {(r['family'],r['seed'],r['rate']):r for r in rows}
    if len(rows)!=20 or set(keyed)!=expected:
        raise ValueError('Require exactly the declared 20 pilots, without duplicates')
    selected = dict(FIXED_PILOTS); details = {}; failed = []
    for family,rates in RATE_GRID.items():
        candidates = []
        for rate in rates:
            ratios = []
            for seed in PILOT_SEEDS:
                for step in PROBES:
                    a=common.update(keyed[family,seed,rate],step);b=common.update(keyed['s24',seed,1.],step)
                    if not math.isfinite(a) or not math.isfinite(b) or min(a,b)<=0:
                        raise ValueError('Nonfinite or zero calibration updates')
                    ratios.append(a/b)
            candidates.append(dict(rate=rate,ratios=ratios,within_bounds=all(.5<=v<=2. for v in ratios),
                                   score=statistics.mean(abs(math.log(v))for v in ratios)))
        details[family]=candidates;valid=[v for v in candidates if v['within_bounds']]
        if valid:selected[family]=min(valid,key=lambda v:(v['score'],v['rate']))['rate']
        else:failed.append(family)
    return dict(selected_rates=selected,candidates=details,failed_families=failed,quality_gate_passed=not failed,
                reference='s24 affine LR1, low-rank LR1, offsets enabled',seeds=list(PILOT_SEEDS),
                probe_steps=list(PROBES),ratio_bounds=[.5,2.],
                criterion='Closest geometric low-rank function update among rates satisfying every bound; no accuracy selection',
                limits='Static unit/joint12 are frozen diagnostic controls, not update-matched; affine/network updates are unmatched.')


def verify_pairs(rows):
    # Stronger than common audit: identical initial adapters as well as base
    # weights and complete observation streams in all static-rate variants.
    for seed in {r['seed']for r in rows}:
        static=[r for r in rows if r['seed']==seed and r['family'].startswith('static_') and r['family']!='static_capacity']
        if static:
            diagnostics=[r['results']['token_diagnostics']for r in static]
            assert len({d['adapter_initialization_sha256']for d in diagnostics})==1
        for row in [r for r in rows if r['seed']==seed]:
            cfg=row['config'];dynamic=cfg['variant']=='dplr'
            assert cfg['factor_offsets']==dynamic and cfg['hyper_lr_multiplier']==1.
            if cfg['variant']!='plain':
                assert cfg['affine_lr_multiplier']==(12. if row['family']=='static_joint_12' else 1.)
            if cfg['variant'] in ('dplr','static_lora_diagonal'):
                assert cfg['lowrank_lr_multiplier']==row['rate']


def prepare(output, launch=False):
    import torch
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent;output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    free=shutil.disk_usage(root).free
    if free<8*1024**3:raise RuntimeError('Require 8 GiB free for the declared 4 GiB bound and headroom')
    if any(json.loads(f.read_text()).get('owner')=='adalora' for f in Path('/tmp/vast_gpu_claims').glob('*.claim')):
        raise RuntimeError('Existing adalora workers must finish before this study')
    output.mkdir(parents=True);runner=output/'runner';runner.mkdir()
    for name in sorted(set((*SOURCE_FILES,*EXTRA_SOURCE))):shutil.copy2(root/name,runner/name)
    parameters={f:model_stats(build_model(make_config(f,100,output/'unused',FIXED_PILOTS.get(f,1.))),
                              make_config(f,100,output/'unused',FIXED_PILOTS.get(f,1.)))['parameters'] for f in FAMILIES}
    assert abs(parameters['static_capacity']-parameters['s24'])==606
    assert not torch.cuda.is_initialized(), 'Preparation must stay CPU-only'
    p=dict(state='prepared',created_utc=common.now(),repo_root=str(root),
           purpose='Separate affine/low-rank rates and extend the ViT comparison to 30 epochs',
           source_sha256={f.name:common.digest(f)for f in runner.iterdir() if f.is_file()},
           families=FAMILIES,parameters=parameters,pilot_seeds=list(PILOT_SEEDS),quality_seeds=list(QUALITY_SEEDS),
           rate_grid=RATE_GRID,fixed_pilots=FIXED_PILOTS,epochs=30,pilot_steps=PILOT_STEPS,train_steps=HORIZON,
           lr_schedule_steps=HORIZON,calibration_probes=list(PROBES),calibration_bounds=[.5,2.],
           planned_runs=dict(preflight=2,calibration=20,quality=16),max_concurrent_workers=2,
           train_examples=45000,validation_examples=5000,split_seed=123,
           quality_gate='Two new GPU preflights, all 20 audited pilots, independent affine rates, and all selected low-rank ratios within unchanged bounds',
           resources=dict(estimated_gpu_hours='3–5 aggregate; approximately 1.5–2.5 wall hours with two available GPUs',
                          max_our_gpus=2,disk_free_bytes_before_launch=free,retained_artifact_bound_bytes=4*1024**3,
                          artifact_purposes={'compact_evidence':'sources/configs/runtime/seeds/splits/optimizer groups/metrics/diagnostics/logs/audits/reports',
                                             'compiler_cache':'compatibility/performance provenance, expected at most 3 GiB'},
                          retention='Compact evidence and declared compiler caches until explicitly removed'),
           weights_saved=False,periodic_checkpoints=False,test_evaluated=False,quality_selected_by_accuracy=False,
           limits=['Two fresh paired quality seeds and reused research holdout; sample SD is not a confidence interval.',
                   '30 epochs is a separate fresh-init schedule; comparison to previous ten-epoch endpoints is not a continuation or paired duration-only ablation.',
                   'Pilot 1200-step prefixes use the full 10560-step horizon and 528 warmup steps; no old pilots are reused.',
                   'Affine LR1 is fixed except the declared static joint12 diagnostic; low-rank updates do not match whole-network or diagonal optimization.',
                   'Static unit and static joint12 remain frozen controls even if their updates miss calibration bounds.',
                   's24 is within 606 total parameters of the wider static MLP; s48 has a larger budget.',
                   'Offsets are enabled in dynamic methods; their initialization is zero and weight decay is zero.',
                   'Static affine norms and ordinary self-attention are unchanged; no AdaLN residual gate is introduced.',
                   'Compiled concurrent telemetry is not an isolated speed benchmark; no weights, checkpoints or official-test evaluation.',
                   'Grid and selection are fixed before quality; a failed gate withholds quality without automatic expansion.'])
    write_json(output/'protocol.json',p);configs=output/'planned_configs';configs.mkdir()
    for f,s,r,cfg in pilot_cases(output/'calibration'):
        path=configs/(Path(cfg.output_dir).name+'.json');write_json(path,asdict(cfg));load_config(['--override_json',str(path)])
    if launch:launch_controller(output)
    return dict(output=str(output),launched=launch,parameters=parameters,planned_runs=p['planned_runs'],max_our_gpus=2)


def launch_controller(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text())
    if p['state']!='prepared':raise ValueError('Controller launch requires a fresh prepared study')
    assert all(common.digest(output/'runner'/name)==sha for name,sha in p['source_sha256'].items())
    return private_supervisor(output,[('vit_split_lr_controller',[sys.executable,'-u',str(output/'runner/vit_split_lr_study.py'),'run','--output',str(output)])],label='controller',autostart=True)


def run(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text())
    if p['state']!='prepared':raise ValueError('Fresh study only; no implicit resume')
    try:
        p.update(state='preflight',started_utc=common.now());write_json(output/'protocol.json',p)
        preflight=common.execute_stage(output,'preflight',preflight_cases(output/'preflight'),auditor=audit_stage)
        p.update(state='calibration',preflight_complete_utc=common.now());write_json(output/'protocol.json',p)
        pilots=common.execute_stage(output,'calibration',pilot_cases(output/'calibration'),auditor=audit_stage)
        verify_pairs(pilots);selection=select_rates(pilots);write_json(output/'calibration.json',selection)
        if not selection['quality_gate_passed']:raise ValueError('Split-rate calibration misses fixed bounds; withhold quality')
        p.update(state='quality',calibration_complete_utc=common.now(),calibration_sha256=common.digest(output/'calibration.json'));write_json(output/'protocol.json',p)
        rates=selection['selected_rates'];cases=[(f,s,rates.get(f,1.),make_config(f,s,output/'quality'/f'{f}_s{s}',rates.get(f,1.)))for f in FAMILIES for s in QUALITY_SEEDS]
        quality=common.execute_stage(output,'quality',cases,auditor=audit_stage);verify_pairs(quality)
        write_json(output/'report.json',dict(preflight=preflight,calibration=pilots,quality=quality))
        p.update(state='complete',completed_utc=common.now());write_json(output/'protocol.json',p)
        audit(output);write_summary(output,quality,p)
    except BaseException as error:
        p.update(state='failed',failed_utc=common.now(),failure_type=type(error).__name__,failure=str(error));write_json(output/'protocol.json',p);raise


def audit(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='complete'
    report=json.loads((output/'report.json').read_text());rows={name:audit_stage(output,name)for name in ('preflight','calibration','quality')};assert report==rows
    verify_pairs(rows['calibration']);verify_pairs(rows['quality']);selection=select_rates(rows['calibration'])
    assert selection['quality_gate_passed'] and selection==json.loads((output/'calibration.json').read_text())
    assert common.digest(output/'calibration.json')==p['calibration_sha256']
    quality=rows['quality'];assert len(quality)==16 and {(r['family'],r['seed'])for r in quality}=={(f,s)for f in FAMILIES for s in QUALITY_SEEDS}
    assert all(r['rate']==selection['selected_rates'].get(r['family'],1.)for r in quality)
    progress=[json.loads(line)for line in(output/'progress.jsonl').read_text().splitlines()]
    assert all(e['active']==e['submitted']-e['completed']<=2 for e in progress)
    retained=sum(f.stat().st_size for f in output.rglob('*')if f.is_file());assert retained<p['resources']['retained_artifact_bound_bytes']
    checks=('all_38_runs_audited_and_successful','frozen_sources_runtime_configs_and_optimizer_groups',
            'paired_base_static_adapter_initialization_and_observed_streams','zero_start_and_gradient_wakeup',
            'independent_affine_rates_and_logged_group_schedules','exact_update_only_calibration_and_fixed_diagnostic_controls',
            'sixteen_fresh_30_epoch_runs_with_frozen_rates','two_gpu_claimed_worker_cap','no_weights_test_and_storage_bound')
    result=dict(checks={k:True for k in checks},report_sha256=common.digest(output/'report.json'),bytes_retained=retained)
    write_json(output/'audit.json',result);return result


def write_summary(output,rows,p):
    aggregates=[]
    for family in FAMILIES:
        values=sorted([r for r in rows if r['family']==family],key=lambda r:r['seed']);metrics={}
        for name,getter in {'accuracy_percent':lambda r:100*r['results']['validation']['accuracy'],
                            'training_peak_mib':lambda r:r['results']['peak_training_allocated_bytes']/2**20,
                            'loss':lambda r:r['results']['validation']['loss']}.items():
            xs=[getter(r)for r in values];metrics[name]=dict(values=xs,mean=statistics.mean(xs),sd=statistics.stdev(xs))
        aggregates.append(dict(family=family,parameters=values[0]['results']['parameters'],
                               affine_rate=values[0]['config']['affine_lr_multiplier'],lowrank_rate=values[0]['config']['lowrank_lr_multiplier'],metrics=metrics))
    write_json(output/'comparison.json',dict(aggregates=aggregates,limits=p['limits']))
    lines=['# ViT split learning rates: 30 epochs','','Fresh paired seeds102/103; means ± sample SD. Final research-holdout accuracy.','',
           '| Family | Parameters | Affine LR × | Low-rank LR × | Accuracy (%) | Training peak MiB |','|---|---:|---:|---:|---:|---:|']
    for row in aggregates:
        a=row['metrics']['accuracy_percent'];m=row['metrics']['training_peak_mib'];aff='—' if row['affine_rate']is None else f"{row['affine_rate']:g}";lr='—' if row['lowrank_rate']is None else f"{row['lowrank_rate']:g}"
        lines.append(f"| {row['family']} | {row['parameters']:,} | {aff} | {lr} | {a['mean']:.2f} ± {a['sd']:.2f} | {m['mean']:.1f} ± {m['sd']:.1f} |")
    lines+=['','Limits: '+' '.join(p['limits']),'','[Evidence](report.json) · [Audit](audit.json) · [Calibration](calibration.json)','']
    (output/'summary.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','run','audit'))
    parser.add_argument('--output',type=Path);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    output=args.output or Path('runs')/('vit-split-lr-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action=='prepare':print(json.dumps(prepare(output,args.launch)),flush=True)
    elif args.action=='launch':print(launch_controller(output),flush=True)
    elif args.action=='run':run(output)
    else:print(json.dumps(audit(output),indent=2))

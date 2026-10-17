#!/usr/bin/env python3
"""Equal-budget initializer/LR tuning and fresh confirmation of the ViT capacity control."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

import torch
import vit_token_study as common
from conditioning import project
from dit_followup import private_supervisor
from train_cifar10 import Config, build_model, load_config, model_stats, write_json
from train_dit_cifar10 import SOURCE_FILES
from vit_initialization import mlp_initialization_stds

METHODS = ('plain', 'wide', 'dplr')
INITIALIZERS = ('legacy', 'fanin_gelu', 'xavier_gelu', 'mixed_gelu')
RATES = (.5, 1., 2.)
TUNING_SEEDS = (104, 105)
QUALITY_SEEDS = (106, 107, 108, 109)
HORIZON = 10560
EXTRA_SOURCE = ('vit_initializer_study.py','test_vit_initializer_study.py','vit_token_study.py',
                'vit_quality_diagnostics.py','dit_followup.py','summarize_dit_cifar10_screen.py','requirements.txt')


def family_key(method,initializer):return method+'__'+initializer


def make_config(family,seed,output,rate=1.,pilot=False):
    method,initializer=family.split('__')
    if method not in METHODS or initializer not in (*INITIALIZERS,'reference_fanin'):
        raise ValueError('Invalid method or initializer')
    if initializer=='reference_fanin' and (method!='wide' or rate!=1.):
        raise ValueError('Reference-fanin is a frozen wide-only diagnostic at LR1')
    conditional=method=='dplr'
    return Config(variant='dplr' if conditional else 'plain',mlp_ratio=1483/192 if method=='wide' else 4.,
                  mlp_init=initializer,width=192,heads=3,depth=6,rank=8,n_basis=24,
                  factor_offsets=conditional,adapter_gain=1/math.sqrt(8) if conditional else 1.,seed=seed,
                  epochs=30,max_steps=HORIZON,lr_schedule_steps=HORIZON,lr=5e-4*rate,
                  hyper_lr_multiplier=1.,affine_lr_multiplier=1. if conditional else None,
                  lowrank_lr_multiplier=1. if conditional else None,compile=True,num_workers=2,cpu_threads=2,
                  log_every=100,diagnostic_steps='0,1,2,3,5,10,50,100,250,500,800,1200,2000,3520,7040,10560',
                  output_dir=str(output))


def tuning_cases(output):
    cases=[(family_key(m,i),s,r,make_config(family_key(m,i),s,output/f'{m}_{i}_lr{r:g}_s{s}',r))
           for m in METHODS for i in INITIALIZERS for r in RATES for s in TUNING_SEEDS]
    cases += [(family_key('wide','reference_fanin'),s,1.,make_config(family_key('wide','reference_fanin'),s,output/f'wide_reference_fanin_lr1_s{s}'))for s in TUNING_SEEDS]
    return cases


def preflight_cases(output):
    result=[]
    for method,initializer in (('wide','xavier_gelu'),('dplr','fanin_gelu')):
        f=family_key(method,initializer)
        cfg=replace(make_config(f,104,output/f,2.),max_steps=12,lr_schedule_steps=12,train_limit=328,
                    warmup_ratio=2/12,log_every=6,diagnostic_steps='0,1,2,3,5,10,12')
        result.append((f,104,2.,cfg))
    return result


def audit_stage(output,name):
    pf={(f,s,r):asdict(c)for f,s,r,c in preflight_cases(output/'preflight')}
    rows=common.audit_stage(output,name,config_factory=make_config,preflight_configs=pf)
    for row in rows:
        events=row['results']['token_diagnostics']['diagnostics']
        assert all(len(e['fixed_mlp_updates'])==6 for e in events)
        assert all(v['update_rms']==0 and v['output_rms']>0 for v in events[0]['fixed_mlp_updates'])
        assert any(v['update_rms']>0 for v in events[-1]['fixed_mlp_updates'])
    return rows


def select_recipes(rows):
    expected={(family_key(m,i),s,r)for m in METHODS for i in INITIALIZERS for r in RATES for s in TUNING_SEEDS}
    expected |= {(family_key('wide','reference_fanin'),s,1.)for s in TUNING_SEEDS}
    keyed={(r['family'],r['seed'],r['rate']):r for r in rows}
    if len(rows)!=74 or set(keyed)!=expected:raise ValueError('Require the exact 74-run tuning/diagnostic grid')
    selected={};details={}
    for method in METHODS:
        candidates=[]
        for initializer in INITIALIZERS:
            for rate in RATES:
                values=[keyed[family_key(method,initializer),s,rate]['results']['validation']for s in TUNING_SEEDS]
                accuracies=[v['accuracy']for v in values];losses=[v['loss']for v in values]
                if not all(math.isfinite(v)for v in (*accuracies,*losses)):raise ValueError('Nonfinite tuning results')
                candidates.append(dict(initializer=initializer,rate=rate,accuracies=accuracies,losses=losses,
                                       mean_accuracy=statistics.mean(accuracies),mean_loss=statistics.mean(losses)))
        chosen=min(candidates,key=lambda c:(-c['mean_accuracy'],c['mean_loss'],abs(math.log(c['rate'])),c['rate'],INITIALIZERS.index(c['initializer'])))
        selected[method]=dict(initializer=chosen['initializer'],rate=chosen['rate']);details[method]=candidates
    return dict(selected=selected,candidates=details,tuning_seeds=list(TUNING_SEEDS),
                candidates_per_method=12,runs_per_method=24,diagnostic_reference_fanin_selected=False,
                criterion='Final 30-epoch mean validation accuracy; ties use mean loss, closest unit LR, lower LR and declared initializer order',
                limits='Selection and confirmation reuse the research holdout; four fresh confirmation seeds separate seed overfitting, not holdout overfitting.')


def quality_cases(output,selection):
    cases=[]
    for method,recipe in selection['selected'].items():
        choices=[('selected',recipe['initializer'],recipe['rate'])]
        if recipe!=dict(initializer='legacy',rate=1.):choices.append(('anchor','legacy',1.))
        for role,initializer,rate in choices:
            f=family_key(method,initializer)
            cases += [(f,s,rate,make_config(f,s,output/f'{method}_{role}_{initializer}_lr{rate:g}_s{s}',rate))for s in QUALITY_SEEDS]
    return cases


def initialization_probe():
    rows=[]
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(1201);h=torch.nn.functional.layer_norm(torch.randn(2,65,192),(192,))
        for method in METHODS:
            for initializer in (*INITIALIZERS,*(['reference_fanin']if method=='wide'else [])):
                for seed in TUNING_SEEDS:
                    cfg=make_config(family_key(method,initializer),seed,Path('/unused'))
                    torch.manual_seed(seed);model=build_model(cfg);model.eval();blocks=[]
                    for index,block in enumerate(model.blocks):
                        z=project(block.mlp.fc1,h,h);post=block.mlp.act(z);y=block.mlp(h,h)
                        blocks.append(dict(block=index,preactivation_rms=z.square().mean().sqrt().item(),
                                           post_gelu_second_moment=post.square().mean().item(),output_rms=y.square().mean().sqrt().item()))
                    d,o=mlp_initialization_stds(192,int(192*cfg.mlp_ratio),initializer)
                    rows.append(dict(method=method,initializer=initializer,seed=seed,fc1_std=d,fc2_std=o,blocks=blocks))
    return dict(rows=rows,dtype='FP32 CPU',shape=[2,65,192],probe_seed=1201,
                definition='Standalone untrained MLP branches on identical normalized random tokens; no parameter snapshots',
                limits='Synthetic inputs, Gaussian gain model and two seeds; not an observed training improvement.')


def ensure_no_active_work():
    for path in Path('/tmp/vast_gpu_claims').glob('*.claim'):
        claim=json.loads(path.read_text())
        if claim.get('owner')=='adalora':
            try:os.kill(claim['pid'],0)
            except ProcessLookupError:continue
            raise RuntimeError('Existing adalora workers must finish; dead stale claims are left intact')


def prepare(output,launch=False):
    torch.set_num_threads(2);root=Path(__file__).resolve().parent;output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    ensure_no_active_work();free=shutil.disk_usage(root).free
    if free<8*1024**3:raise RuntimeError('Require 8 GiB free for the declared 4 GiB artifact bound and headroom')
    output.mkdir(parents=True);runner=output/'runner';runner.mkdir()
    for name in sorted(set((*SOURCE_FILES,*EXTRA_SOURCE))):shutil.copy2(root/name,runner/name)
    families={family_key(m,i):dict(method=m,initializer=i)for m in METHODS for i in INITIALIZERS}
    families[family_key('wide','reference_fanin')]=dict(method='wide',initializer='reference_fanin',diagnostic_only=True)
    parameters={f:model_stats(build_model(make_config(f,104,output/'unused')),make_config(f,104,output/'unused'))['parameters']for f in families}
    assert abs(parameters[family_key('wide','legacy')]-parameters[family_key('dplr','legacy')])==606
    probe=initialization_probe();assert not torch.cuda.is_initialized()
    p=dict(state='prepared',created_utc=common.now(),repo_root=str(root),
           purpose='Validate the wider MLP capacity control with equal initializer/LR tuning budgets and fresh paired confirmation',
           source_sha256={f.name:common.digest(f)for f in runner.iterdir()if f.is_file()},families=families,parameters=parameters,
           methods=list(METHODS),initializers=list(INITIALIZERS),rate_grid=list(RATES),tuning_seeds=list(TUNING_SEEDS),
           quality_seeds=list(QUALITY_SEEDS),epochs=30,train_steps=HORIZON,lr_schedule_steps=HORIZON,
           planned_runs=dict(preflight=2,calibration=74,quality=None),maximum_quality_runs=24,minimum_quality_runs=12,
           max_concurrent_workers=2,train_examples=45000,validation_examples=5000,split_seed=123,
           quality_gate='Two compiled GPU preflights, complete audited equal-budget grid, finite MLP updates and frozen selection before confirmation',
           resources=dict(estimated_gpu_hours='10–16 aggregate; approximately 5–8 wall hours with two available GPUs',max_our_gpus=2,
                          disk_free_bytes_before_launch=free,retained_artifact_bound_bytes=4*1024**3,
                          artifact_purposes={'compact_evidence':'frozen sources/configs/runtime/seeds/splits/initializer and optimizer records/metrics/diagnostics/logs/audits/reports',
                                             'compiler_cache':'compatibility/performance provenance, expected at most 3 GiB'},
                          retention='Compact evidence and declared compiler caches until explicitly removed'),
           weights_saved=False,periodic_checkpoints=False,test_evaluated=False,
           limits=['Three methods each receive the same 12 recipe candidates × two tuning seeds, all trained for 30 epochs.',
                   'Four fresh paired confirmation seeds; reused research holdout, not independent test confirmation; SD is not a confidence interval.',
                   'Gaussian GELU quadrature anchors expected narrow-model output RMS; actual branches and update magnitudes are measured.',
                   'Initializer policies rescale legacy normal draws, retaining their ±100 std truncation and consuming no extra RNG.',
                   'Only base MLP weights change; attention/embedding/norm/classifier initialization and adapter zero/RNG rules are preserved.',
                   'Width changes constructor RNG consumption; shared base initializations match between plain and DPLR within a policy, not across widths.',
                   'Reference-fanin wide LR1 has two diagnostic runs; it cannot enter selection and gives no extra tuning budget.',
                   'Global LR scales base/affine/low-rank rates together; relative adapter multipliers remain1, without function-update matching.',
                   'Fresh confirmation includes legacy-LR1 anchors unless identical to selected recipes; no duplicate training cases.',
                   'This targets the wider-control concern; s48 and the static low-rank baseline are not retuned in this cohort.',
                   'No weights/checkpoints/test evaluation; concurrent telemetry is not an isolated speed benchmark.'])
    write_json(output/'protocol.json',p);write_json(output/'initialization_probe.json',probe)
    planned=output/'planned_configs';planned.mkdir()
    for f,s,r,cfg in tuning_cases(output/'calibration'):
        path=planned/(Path(cfg.output_dir).name+'.json');write_json(path,asdict(cfg));load_config(['--override_json',str(path)])
    if launch:launch_controller(output)
    return dict(output=str(output),launched=launch,planned_runs=p['planned_runs'],quality_range=[12,24],max_our_gpus=2)


def launch_controller(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='prepared'
    ensure_no_active_work();assert all(common.digest(output/'runner'/n)==s for n,s in p['source_sha256'].items())
    return private_supervisor(output,[('vit_initializer_controller',[sys.executable,'-u',str(output/'runner/vit_initializer_study.py'),'run','--output',str(output)])],label='controller',autostart=True)


def run(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='prepared'
    try:
        p.update(state='preflight',started_utc=common.now());write_json(output/'protocol.json',p)
        preflight=common.execute_stage(output,'preflight',preflight_cases(output/'preflight'),auditor=audit_stage)
        p.update(state='tuning',preflight_complete_utc=common.now());write_json(output/'protocol.json',p)
        tuning=common.execute_stage(output,'calibration',tuning_cases(output/'calibration'),auditor=audit_stage)
        selection=select_recipes(tuning);write_json(output/'selection.json',selection)
        cases=quality_cases(output/'quality',selection)
        p.update(state='quality',tuning_complete_utc=common.now(),selection_sha256=common.digest(output/'selection.json'))
        p['planned_runs']['quality']=len(cases);write_json(output/'protocol.json',p)
        quality=common.execute_stage(output,'quality',cases,auditor=audit_stage)
        write_json(output/'report.json',dict(preflight=preflight,tuning=tuning,quality=quality))
        p.update(state='complete',completed_utc=common.now());write_json(output/'protocol.json',p)
        audit(output);write_summary(output,quality,selection,p)
    except BaseException as error:
        p.update(state='failed',failed_utc=common.now(),failure_type=type(error).__name__,failure=str(error));write_json(output/'protocol.json',p);raise


def audit(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='complete'
    report=json.loads((output/'report.json').read_text());pf=audit_stage(output,'preflight');tuning=audit_stage(output,'calibration');quality=audit_stage(output,'quality')
    assert report==dict(preflight=pf,tuning=tuning,quality=quality)
    selection=select_recipes(tuning);assert selection==json.loads((output/'selection.json').read_text())
    assert common.digest(output/'selection.json')==p['selection_sha256']
    cases=quality_cases(output/'quality',selection)
    assert {(r['family'],r['seed'],r['rate'],r['output_dir'])for r in quality}=={(f,s,r,str(c.output_dir))for f,s,r,c in cases}
    assert len(quality)==len(cases) and 12<=len(cases)<=24
    progress=[json.loads(l)for l in(output/'progress.jsonl').read_text().splitlines()]
    assert all(e['active']==e['submitted']-e['completed']<=2 for e in progress)
    retained=sum(f.stat().st_size for f in output.rglob('*')if f.is_file());assert retained<p['resources']['retained_artifact_bound_bytes']
    result=dict(checks={n:True for n in ('all_workers_sources_runtime_configs_optimizer_groups_and_splits_audited',
        'shared_shape_initializer_pairing_observed_streams_and_zero_gradient_rules',
        'equal_tuning_budgets_exact_74_grid_and_diagnostic_exclusion','full_30_epoch_selection_and_frozen_fresh_confirmation',
        'explicit_legacy_anchors_and_no_duplicate_quality_cases','fixed_mlp_functions_and_lowrank_update_diagnostics',
        'two_gpu_worker_cap_and_no_weights_test_storage_bound')},report_sha256=common.digest(output/'report.json'),bytes_retained=retained)
    write_json(output/'audit.json',result);return result


def write_summary(output,rows,selection,p):
    aggregates=[]
    for method,recipe in selection['selected'].items():
        for role in ('selected','anchor'):
            values=[r for r in rows if Path(r['output_dir']).name.startswith(method+'_'+role+'_')]
            if not values and role=='anchor':values=[r for r in rows if Path(r['output_dir']).name.startswith(method+'_selected_')]
            values.sort(key=lambda r:r['seed']);assert len(values)==4
            xs=[100*r['results']['validation']['accuracy']for r in values]
            aggregates.append(dict(method=method,role=role,initializer=values[0]['config']['mlp_init'],rate=values[0]['rate'],
                parameters=values[0]['results']['parameters'],accuracy_percent=dict(values=xs,mean=statistics.mean(xs),sd=statistics.stdev(xs)),
                anchor_reuses_selected=role=='anchor' and recipe==dict(initializer='legacy',rate=1.)))
    write_json(output/'comparison.json',dict(aggregates=aggregates,selection=selection,limits=p['limits']))
    lines=['# ViT initializer/LR confirmation','','30 epochs; fresh paired seeds106–109; mean ± sample SD. Final research-holdout accuracy.','',
           '| Method | Role | Initializer | LR × | Parameters | Accuracy (%) |','|---|---|---|---:|---:|---:|']
    for row in aggregates:
        a=row['accuracy_percent'];lines.append(f"| {row['method']} | {row['role']} | {row['initializer']} | {row['rate']:g} | {row['parameters']:,} | {a['mean']:.2f} ± {a['sd']:.2f} |")
    lines+=['','Limits: '+' '.join(p['limits']),'','[Evidence](report.json) · [Audit](audit.json) · [Selection](selection.json)','']
    (output/'summary.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','run','audit'))
    parser.add_argument('--output',type=Path);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    output=args.output or Path('runs')/('vit-initializers-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action=='prepare':print(json.dumps(prepare(output,args.launch)),flush=True)
    elif args.action=='launch':print(launch_controller(output),flush=True)
    elif args.action=='run':run(output)
    else:print(json.dumps(audit(output),indent=2))

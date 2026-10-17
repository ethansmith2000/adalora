#!/usr/bin/env python3
"""Equal global-LR tuning and fresh-seed confirmation of broadcast DPLR versus AdaLN."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import torch
from class_fidelity import classifier_provenance
from dit_followup import private_supervisor
from dit_optimizer_cfg import digest, execute_stage, mean_sd, now, same_stage_evidence
from summarize_dit_cifar10_screen import summarize
from train_dit_cifar10 import Config, make_model, load_config, write_json

REFERENCE = 'runs/dit-factor-offset-20261010T193609Z'
METHODS = ('plain', 'wide', 'dplr')
RATES = (.5, 1., 2.)
TUNING_SEEDS = (114, 115)
QUALITY_SEEDS = (116, 117, 118, 119)
HORIZON = 30000
DIAGNOSTICS = '0,1,3,5,10,25,50,100,250,500,1000,5000,10000,20000,30000'
PARAMETERS = dict(plain=9758768, wide=12594632, dplr=12593712)
OVERLAYS = ('dit_external_lr_study.py', 'test_dit_external_lr_study.py',
            'dit_optimizer_cfg.py', 'summarize_dit_cifar10_screen.py')


def family_key(method, rate):
    if method not in METHODS or rate not in RATES: raise ValueError('Unknown fixed method/rate')
    return f'{method}_lr{rate:g}'


def make_config(method, seed, output, classifier, rate=1.):
    family_key(method, rate)
    return Config(variant='dplr' if method == 'dplr' else 'adaln_zero', width=256, depth=8, heads=4,
                  patch_size=4, mlp_ratio=1715/256 if method == 'wide' else 4., rank=8, n_basis=32,
                  factor_offsets=method == 'dplr', adapter_input='normalized', adapter_gain=1/math.sqrt(8),
                  adapter_lr_multiplier=1., seed=seed, lr=4e-4*rate, max_steps=HORIZON, lr_schedule_steps=HORIZON,
                  warmup_steps=50, batch_size=128, eval_every=5000, log_every=500, cpu_threads=2, num_workers=2,
                  adam_beta1=.9, adam_beta2=.999, fused_adamw=True, compile_model=True, compile_mode='default',
                  compile_dynamic=True, class_dropout_prob=.1, guidance_scale=1., guidance_scales='1,2',
                  generative_eval_steps='10000,20000', sampling_steps=50, fid_images=5000, fid_batch_size=64,
                  sample_images=100, classifier_run=classifier, diagnostic_steps=DIAGNOSTICS, output_dir=str(output))


def tuning_cases(output, classifier):
    cases = [(family_key(m, r), s, make_config(m, s, output/f'{family_key(m,r)}_s{s}', classifier, r))
             for m in METHODS for r in RATES for s in TUNING_SEEDS]
    random.Random(2076).shuffle(cases)
    return cases


def preflight_cases(output, classifier):
    return [(family_key(m,1.), 114, replace(make_config(m,114,output/f'{family_key(m,1.)}_s114',classifier),
            max_steps=12, lr_schedule_steps=12, warmup_steps=2, train_limit=328, eval_every=6, log_every=6,
            diagnostic_steps='0,1,3,5,10,12', generative_eval_steps='3,6', sampling_steps=2,
            fid_images=20, fid_batch_size=10, sample_images=10)) for m in ('wide','dplr')]


def select_rates(rows):
    keyed = {(r['family'], r['config']['seed']): r for r in rows}
    expected = {(family_key(m,r), s) for m in METHODS for r in RATES for s in TUNING_SEEDS}
    if len(rows) != 18 or set(keyed) != expected: raise ValueError('Require the exact eighteen full-horizon tuning cases')
    selected, candidates = {}, {}
    for m in METHODS:
        cs=[]
        for rate in RATES:
            rs=[keyed[family_key(m,rate),s] for s in TUNING_SEEDS]
            if any(r['completed_steps'] != HORIZON for r in rs): raise ValueError('Incomplete tuning horizon')
            fids=[r['guidance_evaluations']['2.0']['fid'] for r in rs]
            mse=[r['ema_validation']['noise_mse'] for r in rs]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in [*fids,*mse]):
                raise ValueError('Nonfinite or absent selection metric')
            cs.append(dict(rate=rate, global_lr=4e-4*rate, cfg2_fid=mean_sd(fids), ema_mse=mean_sd(mse)))
        chosen=min(cs,key=lambda c:(c['cfg2_fid']['mean'],c['ema_mse']['mean'],abs(math.log(c['rate'])),c['rate']))
        selected[m]=chosen['rate'];candidates[m]=cs
    return dict(selected_rates=selected,candidates=candidates,tuning_seeds=list(TUNING_SEEDS),
                candidates_per_method=3,runs_per_method=6,
                criterion='Final 30k mean CFG2 EMA FID; ties use final EMA noise MSE, closest unit rate, lower rate',
                limits='Two tuning seeds and reused 5k research holdout/noise; no intermediate best-state selection')


def quality_cases(output, classifier, selection):
    cases=[(family_key(m,selection['selected_rates'][m]),s,
            make_config(m,s,output/f'{family_key(m,selection["selected_rates"][m])}_s{s}',classifier,selection['selected_rates'][m]))
           for m in METHODS for s in QUALITY_SEEDS]
    random.Random(2077).shuffle(cases)
    return cases


def ensure_inactive():
    for f in Path('/tmp/vast_gpu_claims').glob('*.claim'):
        claim=json.loads(f.read_text())
        if claim.get('owner')!='adalora':continue
        try:os.kill(claim['pid'],0)
        except ProcessLookupError:continue
        raise RuntimeError('Existing adalora work must finish before this two-worker workflow')


def prepare(output, launch=False):
    torch.set_num_threads(2);root=Path(__file__).resolve().parent;output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    ensure_inactive();free=shutil.disk_usage(root).free
    if free<8*1024**3:raise RuntimeError('Require 8 GiB free for the 4 GiB artifact bound and headroom')
    reference=root/REFERENCE; prior=json.loads((reference/'report.json').read_text())
    old_audit=json.loads((reference/'audit.json').read_text());assert all(old_audit['checks'].values())
    assert old_audit['report_sha256']==digest(reference/'report.json')
    original=json.loads((reference/'protocol.json').read_text())
    assert all(digest(reference/'runner'/n)==h for n,h in original['source_sha256'].items())
    # The old reporting failure/recovery is preserved; no old outcomes are reused as new cells.
    assert digest(root/'dit_cifar10.py')==digest(reference/'runner/dit_cifar10.py')
    assert digest(root/'class_fidelity.py')==digest(reference/'runner/class_fidelity.py')
    classifier=str(root/'runs/dit-followup-20261007T174446Z/classifier/classifier_config')
    auxiliary=classifier_provenance(classifier);evaluator=Path(Config().fid_weights)
    assert auxiliary==original['auxiliary_classifier'] and digest(evaluator)==original['evaluator_cache']['sha256']
    from train_dit_cifar10 import provenance
    runtime=provenance();assert {k:runtime[k] for k in ('package_versions','cuda_wheel')}==prior['quality']['runtime']
    output.mkdir(parents=True);runner=output/'runner'
    shutil.copytree(reference/'runner',runner,ignore=shutil.ignore_patterns('__pycache__'))
    for n in OVERLAYS:shutil.copy2(root/n,runner/n)
    frozen_training=prior['quality']['source_sha256']
    assert all(digest(runner/n)==h for n,h in frozen_training.items())
    for m in METHODS:
        c=make_config(m,114,output/'unused',classifier);assert make_model(c).statistics()['parameters']==PARAMETERS[m]
    p=dict(state='prepared',created_utc=now(),repo_root=str(root),purpose='Equal global-LR tuning of broadcast DPLR against plain and parameter-matched AdaLN',
           source_sha256={f.name:digest(f) for f in runner.iterdir() if f.is_file()},reference_workflow=str(reference),
           reference_files_sha256={n:digest(reference/n) for n in ('protocol.json','report.json','audit.json','reporting_recovery.json') if (reference/n).exists()},
           frozen_training_source_sha256=frozen_training,declared_orchestration_overlays=list(OVERLAYS),
           methods=list(METHODS),parameters=PARAMETERS,rate_grid=list(RATES),tuning_seeds=list(TUNING_SEEDS),quality_seeds=list(QUALITY_SEEDS),
           train_steps=HORIZON,lr_schedule_steps=HORIZON,planned_runs=dict(preflight=2,tuning=18,quality=12),
           max_concurrent_workers=2,classifier_run=classifier,auxiliary_classifier=auxiliary,evaluator_cache=original['evaluator_cache'],
           expected_storage_upper_bound_bytes=4*1024**3,disk_free_bytes_before_launch=free,estimated_gpu_hours='10–15 aggregate; about5–8 wall hours with two available GPUs',
           artifact_purposes={'compact_evidence':'frozen sources/configs/runtime/seeds/splits/metrics/diagnostics/logs/audits/selections and reports',
                              'compiler_cache':'compatibility/performance provenance; expected at most3GiB',
                              'diagnostic_grids':'thirty final100-image EMA CFG1 grids and two preflight10-image grids; no model weights'},
           retention='Declared compact evidence/cache/grids until explicitly removed; prior evidence/evaluator retention unchanged',
           weights_saved=False,periodic_checkpoints=False,test_evaluated=False,
           selection_metric='Final CFG2 FID only, with declared tie-breaks; selection frozen before four fresh confirmation seeds',
           limits=['Three global LRs × two complete 30k tuning seeds per method; no early/best-state accuracy/FID selection.',
                   'The same 30k schedule in tuning and confirmation; no resume or automatic grid expansion.',
                   'Four fresh confirmation seeds address seed variability, not overfitting to the reused research holdout/evaluation noise.',
                   'Baseline and DPLR share narrow base initialization; wider shape changes RNG/initialization, while each method is paired across rates.',
                   'Wider hidden1715 matches offset-enabled DPLR within920 parameters (0.0074%); no claim of FLOP or function equivalence.',
                   'All base Linear weights already use Xavier initialization; no new initializer or architecture search.',
                   'External condition is a normalized class-plus-timestep vector, broadcast over tokens; sequence-conditioned cross-attention is not tested.',
                   'DPLR r8/s32, offsets on, affine independent projector and gain1/sqrt(8), relative adapter LR1 stay fixed.',
                   'Primary selection uses CFG2; final CFG1 and CFG2 share one EMA state and sampling noise. All guided and unguided results are reported.',
                   '5k-image held-out-train FID and frozen auxiliary classifier agreement are research proxies, not official50k-image test metrics.',
                   'Historical training/evaluation records are context only; no historical quality cell is reused.',
                   'The complete original DiT trainer/model/data/diagnostic source snapshot is reused; recent ViT source changes are excluded.',
                   'Compiled concurrent throughput is descriptive; no dedicated speed benchmark or new evaluator/model-weight artifact.'])
    write_json(output/'protocol.json',p);configs=output/'planned_configs';configs.mkdir()
    for f,s,c in [*preflight_cases(output/'preflight',classifier),*tuning_cases(output/'tuning',classifier)]:
        name=('preflight_' if c.max_steps==12 else '')+Path(c.output_dir).name+'.json'
        path=configs/name;write_json(path,asdict(c));load_config(['--override_json',str(path)])
    assert not torch.cuda.is_initialized()
    if launch:launch_controller(output)
    return dict(output=str(output),launched=launch,planned_runs=p['planned_runs'],max_our_gpus=2)


def launch_controller(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='prepared';ensure_inactive()
    assert all(digest(output/'runner'/n)==h for n,h in p['source_sha256'].items())
    return private_supervisor(output,[('dit_external_lr_controller',[sys.executable,'-u',str(output/'runner/dit_external_lr_study.py'),'run','--output',str(output)])],label='controller',autostart=True)


def stage_cases(output,name,p):
    if name=='preflight':return preflight_cases(output/name,p['classifier_run'])
    if name=='tuning':return tuning_cases(output/name,p['classifier_run'])
    selection=json.loads((output/'selection.json').read_text());assert digest(output/'selection.json')==p['selection_sha256']
    return quality_cases(output/name,p['classifier_run'],selection)


def audit_stage(output,name):
    from benchmark_dit_design import digest_parameters
    from train_dit_cifar10 import learning_rate_scale
    torch.set_num_threads(2);output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text())
    replay=summarize(output/name/'screen.json');jobs=json.loads((output/name/'screen.json').read_text())['jobs']
    byname={j['name']:j for j in jobs}
    for row in replay['rows']:row['family']=byname[row['name']]['family']
    assert same_stage_evidence(replay,json.loads((output/name/'summary.json').read_text()))
    cases={(f,s):asdict(c) for f,s,c in stage_cases(output,name,p)}
    assert len(replay['rows'])==p['planned_runs'][name]==len(cases)
    assert {(r['family'],r['config']['seed']) for r in replay['rows']}==set(cases)
    assert not (output/name/'supervisor.sock').exists()
    for n,h in p['source_sha256'].items():assert digest(output/'runner'/n)==digest(output/name/'source'/n)==h
    assert replay['source_sha256']==p['frozen_training_source_sha256']
    assert classifier_provenance(p['classifier_run'])==p['auxiliary_classifier']
    claims=json.loads((output/name/'hardware_claims.json').read_text());assert set(claims)=={j['service'] for j in jobs}
    bases,adapters,traces,noise,label,dropped,probe={},{},{},{},{},{},{}
    real=[]
    for row in replay['rows']:
        cfg=Config(**row['config']);assert row['config']==cases[row['family'],cfg.seed]
        m=row['family'].split('_lr')[0];torch.manual_seed(cfg.seed);model=make_model(cfg);stats=model.statistics()
        assert row['parameters']==PARAMETERS[m]
        for k in ('parameters','adapter_parameters','forward_matmul_flops_per_image_estimate'):assert row[k]==stats[k]
        diag=row['design_diagnostics'];assert diag['base_initialization_sha256']==digest_parameters(model)
        assert diag['adapter_initialization_sha256']==digest_parameters(model,True)
        bases.setdefault((cfg.seed,'wide' if m=='wide' else 'narrow'),set()).add(diag['base_initialization_sha256'])
        adapters.setdefault((cfg.seed,m),set()).add(diag['adapter_initialization_sha256'])
        events=diag['diagnostics'];assert [e['step'] for e in events]==[int(s) for s in cfg.diagnostic_steps.split(',')]
        assert all(b['gate_rms']==b['delta_rms']==0 for b in events[0]['branches'])
        assert all(v<=10 for v in diag['adapter_first_nonzero_gradient_observed'].values())
        if m=='dplr':assert len([k for k in diag['adapter_first_nonzero_gradient_observed'] if k.endswith('_offset')])==16
        trace=[{k:e[k] for k in ('step','batch_images_sha256','batch_labels_sha256','noise_sha256','timesteps_sha256','noise_rng_state_sha256')} for e in events[1:]]
        traces.setdefault(cfg.seed,[]).append(trace);noise.setdefault(cfg.seed,set()).add(diag['noise_rng_final_state_sha256'])
        setup=row['training_setup'];assert setup['adam_betas']==[.9,.999] and setup['class_dropout_prob']==.1
        assert setup['fused_adamw'] and setup['model_compiled'] and setup['compile_dynamic']
        assert setup['label_dropout_rng_seed']==cfg.seed+30000 and 0<setup['dropped_training_labels']<row['processed_images']
        label.setdefault(cfg.seed,set()).add(setup['label_dropout_rng_final_state_sha256']);dropped.setdefault(cfg.seed,set()).add(setup['dropped_training_labels'])
        runtime=json.loads((Path(cfg.output_dir)/'provenance.json').read_text());claim=claims[byname[row['name']]['service']]
        assert claim['owner']=='adalora' and claim['uuid'].removeprefix('GPU-')==runtime['device_uuid']
        logs=[json.loads(x) for x in (Path(cfg.output_dir)/'metrics.jsonl').read_text().splitlines()]
        assert events==[e for e in logs if e['event']=='design_diagnostic']
        assert all(math.isclose(e['base_lr'],cfg.lr*learning_rate_scale(cfg,e['step']),rel_tol=1e-14) for e in logs if e['event']=='train')
        expected_gen=[3,6,12] if name=='preflight' else [10000,20000,30000]
        assert [e['step'] for e in row['generative_validation_curve']]==expected_gen
        assert set(row['guidance_evaluations'])=={'1.0','2.0'} and row['guidance_evaluations']['1.0']==row['generative_evaluation']
        for point in [*row['generative_validation_curve'],*row['guidance_evaluations'].values()]:
            assert point['model_state']=='EMA' and point['sampling_seed']==cfg.sampling_seed and point['sampling_steps']==cfg.sampling_steps
            assert point['generated_images']==point['fid_fake_images']==cfg.fid_images and point['fid_real_images']==5000
            assert point['balanced_requested_class_counts']==[cfg.fid_images//10]*10
            assert point['class_agreement']['classifier']==p['auxiliary_classifier'] and point['eta']==0
            for state,n in [('real',5000),('generated',cfg.fid_images)]:
                a=point['class_agreement'][state];mat=a['confusion_rows_requested_columns_predicted']
                assert a['examples']==n and a['class_counts']==[n//10]*10 and [sum(t) for t in mat]==[n//10]*10
                assert math.isclose(a['agreement'],sum(mat[i][i] for i in range(10))/n,abs_tol=1e-7)
            real.append(point['class_agreement']['real'])
        assert all(not e['sample_grid_labels'] for e in row['generative_validation_curve'][:-1])
        assert row['generative_evaluation']['sample_grid_labels']==[i%10 for i in range(cfg.sample_images)]
        assert not row['guidance_evaluations']['2.0']['sample_grid_labels']
    assert all(len(v)==1 for groups in (bases,adapters,noise,label,dropped) for v in groups.values())
    assert all(all(t==ts[0] for t in ts) for ts in traces.values()) and all(v==real[0] for v in real)
    return replay


def run(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='prepared'
    try:
        p.update(state='preflight',started_utc=now());write_json(output/'protocol.json',p)
        execute_stage(output,p,'preflight',preflight_cases(output/'preflight',p['classifier_run']),stage_examples=(328,5000),phase='screen')
        pf=audit_stage(output,'preflight')
        p.update(state='tuning',preflight_complete_utc=now());write_json(output/'protocol.json',p)
        execute_stage(output,p,'tuning',tuning_cases(output/'tuning',p['classifier_run']),phase='tuning')
        tuning=audit_stage(output,'tuning');selection=select_rates(tuning['rows']);write_json(output/'selection.json',selection)
        p.update(state='quality',tuning_complete_utc=now(),selection_sha256=digest(output/'selection.json'));write_json(output/'protocol.json',p)
        execute_stage(output,p,'quality',quality_cases(output/'quality',p['classifier_run'],selection),phase='confirmation')
        quality=audit_stage(output,'quality');write_json(output/'report.json',dict(preflight=pf,tuning=tuning,quality=quality))
        p.update(state='complete',completed_utc=now());write_json(output/'protocol.json',p);audit(output);write_summary(output,quality['rows'],p)
    except BaseException as error:
        p.update(state='failed',failed_utc=now(),failure_type=type(error).__name__,failure=str(error));write_json(output/'protocol.json',p);raise


def audit(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='complete'
    report=json.loads((output/'report.json').read_text());stages={n:audit_stage(output,n) for n in ('preflight','tuning','quality')}
    assert all(same_stage_evidence(stages[n],report[n]) for n in stages)
    assert len({json.dumps(v['runtime'],sort_keys=True) for v in stages.values()})==1
    selection=select_rates(stages['tuning']['rows']);assert selection==json.loads((output/'selection.json').read_text())
    assert digest(output/'selection.json')==p['selection_sha256']
    assert all(digest(Path(p['reference_workflow'])/n)==h for n,h in p['reference_files_sha256'].items())
    progress=[json.loads(s) for s in (output/'progress.jsonl').read_text().splitlines()]
    assert all(e['active']==e['submitted']-e['completed']<=2 for e in progress)
    assert progress[-1]['completed']==12 and progress[-1]['active']==0
    retained=sum(f.stat().st_size for f in output.rglob('*') if f.is_file());assert retained<p['expected_storage_upper_bound_bytes']
    result=dict(checks={n:True for n in ('all_thirty_two_workers_sources_configs_runtime_and_exits_audited','original_dit_training_snapshot_and_evaluators_preserved',
       'equal_eighteen_case_full_horizon_lr_grid','frozen_final_cfg2_selection_and_twelve_fresh_seed_cases','paired_initialization_data_noise_and_class_dropout_streams',
       'zero_start_gradient_wakeup_and_function_diagnostics','same_state_guidance_fid_classifier_and_validation_budgets','two_gpu_cap_and_claim_identity','no_weights_checkpoints_test_or_storage_overrun')},
       report_sha256=digest(output/'report.json'),bytes_retained=retained)
    write_json(output/'audit.json',result);return result


def write_summary(output,rows,p):
    groups=[];keyed={(r['family'].split('_lr')[0],r['config']['seed']):r for r in rows}
    for m in METHODS:
        rs=[keyed[m,s] for s in QUALITY_SEEDS];guided={}
        for scale in ('1.0','2.0'):
            guided[scale]=dict(fid=mean_sd([r['guidance_evaluations'][scale]['fid'] for r in rs]),
                              agreement_percent=mean_sd([100*r['guidance_evaluations'][scale]['class_agreement']['generated']['agreement'] for r in rs]))
        groups.append(dict(method=m,parameters=PARAMETERS[m],global_lr=rs[0]['config']['lr'],final_guidance=guided))
    contrasts={}
    for control in ('plain','wide'):
        contrasts[control]={}
        for scale in ('1.0','2.0'):
            pairs=[(keyed['dplr',s]['guidance_evaluations'][scale],keyed[control,s]['guidance_evaluations'][scale]) for s in QUALITY_SEEDS]
            contrasts[control][scale]=dict(fid_dplr_minus_control=mean_sd([a['fid']-b['fid'] for a,b in pairs]),
                 agreement_pp_dplr_minus_control=mean_sd([100*(a['class_agreement']['generated']['agreement']-b['class_agreement']['generated']['agreement']) for a,b in pairs]))
    write_json(output/'comparison.json',dict(aggregates=groups,paired_contrasts=contrasts,selection=json.loads((output/'selection.json').read_text()),limits=p['limits']))
    lines=['# Broadcast conditioning after equal LR tuning','','Four fresh paired seeds; final30k EMA state. Means ± sample SD, not confidence intervals.','',
           '| Method | Parameters | Selected global LR | CFG1 FID | CFG2 FID | CFG2 class agreement |','|---|---:|---:|---:|---:|---:|']
    for g in groups:
        a,b=g['final_guidance']['1.0']['fid'],g['final_guidance']['2.0']['fid'];c=g['final_guidance']['2.0']['agreement_percent']
        lines.append(f"| {g['method']} | {g['parameters']:,} | {g['global_lr']:.2g} | {a['mean']:.2f} ± {a['sd']:.2f} | {b['mean']:.2f} ± {b['sd']:.2f} | {c['mean']:.2f} ± {c['sd']:.2f}% |")
    lines+=['','Both guided/unguided contrasts and all tuning candidates remain in comparison.json. Fresh seeds address seed variability, while the same research holdout/noise was reused.','',
            'Limits: '+' '.join(p['limits']),'','[Evidence](report.json) · [Audit](audit.json) · [Frozen selection](selection.json) · [Protocol](protocol.json)']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','run','audit'))
    parser.add_argument('--output',type=Path);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    output=args.output or Path('runs')/('dit-external-lr-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action=='prepare':print(json.dumps(prepare(output,args.launch)),flush=True)
    elif args.action=='launch':print(launch_controller(output),flush=True)
    elif args.action=='run':run(output)
    else:print(json.dumps(audit(output),indent=2))

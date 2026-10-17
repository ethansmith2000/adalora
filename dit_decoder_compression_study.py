#!/usr/bin/env python3
"""Bounded CP factor-decoder screen with dense controls and no model weights."""
import argparse
from dataclasses import asdict, replace
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
from train_dit_cifar10 import Config, make_model, load_config, write_json, SOURCE_FILES

REFERENCE = 'runs/dit-external-lr-20261013T070213Z'
DESIGNS = {'dense_s32': (32, 0), 'dense_s64': (64, 0),
           'cp_s32_p64': (32, 64), 'cp_s32_p128': (32, 128), 'cp_s64_p128': (64, 128)}
SEEDS = (120, 121)
HORIZON = 30000
PARAMETERS = dict(dense_s32=12593712, dense_s64=15346736, cp_s32_p64=10668592,
                  cp_s32_p128=11364912, cp_s64_p128=11562032)
OVERLAYS = ('dit_cifar10.py', 'train_dit_cifar10.py', 'dit_quality_diagnostics.py',
            'benchmark_dit_design.py', 'summarize_dit_cifar10_screen.py', 'vit_initialization.py',
            'dit_decoder_compression_study.py', 'test_dit_decoder_compression_study.py', 'test_dit_cp_decoders.py')


def make_config(family, seed, output, classifier):
    s, p = DESIGNS[family]
    return Config(variant='dplr', width=256, depth=8, heads=4, patch_size=4, mlp_ratio=4.,
                  rank=8, n_basis=s, decoder_cp_width=p, factor_offsets=True,
                  adapter_input='normalized', adapter_gain=1/math.sqrt(8), adapter_lr_multiplier=1.,
                  seed=seed, lr=8e-4, max_steps=HORIZON, lr_schedule_steps=HORIZON, warmup_steps=50,
                  batch_size=128, eval_every=5000, log_every=500, cpu_threads=2, num_workers=2,
                  adam_beta1=.9, adam_beta2=.999, fused_adamw=True, compile_model=True,
                  compile_mode='default', compile_dynamic=True, class_dropout_prob=.1,
                  guidance_scale=1., guidance_scales='1,2', generative_eval_steps='10000,20000',
                  sampling_steps=50, fid_images=5000, fid_batch_size=64, sample_images=100,
                  classifier_run=classifier, diagnostic_steps='0,1,3,5,10,25,50,100,250,500,1000,5000,10000,20000,30000',
                  output_dir=str(output))


def quality_cases(output, classifier):
    cases = [(f, s, make_config(f, s, output/f'{f}_s{s}', classifier)) for f in DESIGNS for s in SEEDS]
    random.Random(2078).shuffle(cases)
    return cases


def preflight_cases(output, classifier):
    return [(f, 120, replace(make_config(f,120,output/f'{f}_s120',classifier), max_steps=12,
             lr_schedule_steps=12,warmup_steps=2,train_limit=328,eval_every=6,log_every=6,
             diagnostic_steps='0,1,3,5,10,12',generative_eval_steps='3,6',sampling_steps=2,
             fid_images=20,fid_batch_size=10,sample_images=10)) for f in DESIGNS]


def ensure_inactive():
    for f in Path('/tmp/vast_gpu_claims').glob('*.claim'):
        try: claim = json.loads(f.read_text())
        except (OSError,ValueError): continue
        if claim.get('owner') != 'adalora': continue
        try: os.kill(claim['pid'],0)
        except ProcessLookupError: continue
        raise RuntimeError('Existing adalora work must finish before this two-worker study')


def prepare(output, launch=False):
    torch.set_num_threads(2); root=Path(__file__).resolve().parent; output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    ensure_inactive(); free=shutil.disk_usage(root).free
    if free < 8*1024**3: raise RuntimeError('Require 8 GiB free for 4 GiB bound and headroom')
    ref=root/REFERENCE; old=json.loads((ref/'protocol.json').read_text()); oldaudit=json.loads((ref/'audit.json').read_text())
    assert old['state']=='complete' and all(oldaudit['checks'].values()) and oldaudit['report_sha256']==digest(ref/'report.json')
    assert all(digest(ref/'runner'/n)==h for n,h in old['source_sha256'].items())
    classifier=old['classifier_run']; assert classifier_provenance(classifier)==old['auxiliary_classifier']
    assert digest(Path(old['evaluator_cache']['path']))==old['evaluator_cache']['sha256']
    output.mkdir(parents=True); runner=output/'runner'
    shutil.copytree(ref/'runner',runner,ignore=shutil.ignore_patterns('__pycache__'))
    for n in OVERLAYS: shutil.copy2(root/n,runner/n)
    shutil.copyfile(ref/'runner/dit_cifar10.py',runner/'cp_baseline_model.py')
    unchanged={n:h for n,h in old['frozen_training_source_sha256'].items() if n not in OVERLAYS}
    assert all(digest(runner/n)==h for n,h in unchanged.items())
    sizes={}
    for f in DESIGNS:
        torch.manual_seed(120); stats=make_model(make_config(f,120,output/'unused',classifier)).statistics()
        assert stats['parameters']==PARAMETERS[f]; sizes[f]=stats
    p=dict(state='prepared',created_utc=now(),repo_root=str(root),purpose='CP decoder compression at fixed transform rank, with dense generator-width controls',
           reference_workflow=str(ref),reference_files_sha256={n:digest(ref/n) for n in ('protocol.json','report.json','audit.json','selection.json')},
           source_sha256={f.name:digest(f) for f in runner.iterdir() if f.is_file()},
           training_source_sha256={n:digest(runner/n) for n in (*SOURCE_FILES,'dit_quality_diagnostics.py','benchmark_dit_design.py')},
           unchanged_training_source_sha256=unchanged,declared_overlays=list(OVERLAYS),designs=DESIGNS,parameters=PARAMETERS,
           model_statistics=sizes,seeds=list(SEEDS),train_steps=HORIZON,lr_schedule_steps=HORIZON,
           planned_runs=dict(preflight=5,quality=10),max_concurrent_workers=2,classifier_run=classifier,
           auxiliary_classifier=old['auxiliary_classifier'],evaluator_cache=old['evaluator_cache'],
           expected_storage_upper_bound_bytes=4*1024**3,disk_free_bytes_before_launch=free,
           estimated_gpu_hours='4–7 aggregate GPU-hours; approximately 2–3.5 wall hours with two available GPUs',
           weights_saved=False,periodic_checkpoints=False,test_evaluated=False,selection_metric=None,
           artifact_purposes=dict(compact_evidence='configs/sources/runtime/seeds/splits/metrics/diagnostics/logs/audits/reports',
              compiler_cache='compatibility/performance provenance, expected at most 3 GiB',
              diagnostic_grids='ten final 100-image EMA CFG1 grids plus five 10-image preflight grids'),
           retention='Declared evidence, caches and grids until explicitly removed; prior evidence/evaluator unchanged',
           initialization=dict(cp_input_entry_variance='1/d in expectation over independent CP initialization draws, matching dense U marginal variance',
              zero_factor='V_channel; other CP companions nonzero, U/V offsets zero',
              cp_generator_seed='training seed + 100000 + adapter index; isolated CPU generator',
              pairing='same base/null initialization across designs; identical coefficient heads/RNG trajectory within each s',
              temporary_dense_initialization='dense U/V are created then discarded on CPU to preserve head/global RNG; never moved to CUDA'),
           limits=['Five fixed designs × two full 30k seeds; no automatic selection, LR calibration, longer confirmation or grid expansion.',
              'Transform rank r8, coefficient width s32/s64 and CP tensor width p64/p128 are separate axes.',
              'CP reduces decoder parameters by changing its function class; generated factors have rank at most r, including offsets.',
              'CP adds multiplicative gauges and changes optimizer geometry; matching marginal initial variance does not establish variance preservation or matched function updates.',
              'Global LR8e-4 was selected in the prior bounded grid and is not established optimal; all designs use this same fixed recipe.',
              'Both dense widths are new paired controls; no historical quality cells reused. The research holdout/evaluation noise are reused.',
              'Generation at 10k/20k tracks convergence; primary comparisons use final30k CFG2 FID, with CFG1 and agreement also reported.',
              'Whole-training allocation and concurrent throughput are descriptive, not isolated speed benchmarks; CP can increase decoder FLOPs despite parameter savings.',
              'External class/timestep condition is broadcast; this is not a per-token memory, causal retrieval or cross-attention study.',
              'No model weights/checkpoints, official test evaluation, new evaluator downloads or automatic causal-LM launch.'])
    write_json(output/'protocol.json',p); configs=output/'planned_configs';configs.mkdir()
    for f,s,cfg in [*preflight_cases(output/'preflight',classifier),*quality_cases(output/'quality',classifier)]:
        n=('preflight_' if cfg.max_steps==12 else '')+Path(cfg.output_dir).name+'.json'
        path=configs/n;write_json(path,asdict(cfg));load_config(['--override_json',str(path)])
    assert not torch.cuda.is_initialized()
    if launch: launch_controller(output)
    return dict(output=str(output),launched=launch,planned_runs=p['planned_runs'],max_our_gpus=2)


def launch_controller(output):
    output=Path(output).resolve(); p=json.loads((output/'protocol.json').read_text())
    assert p['state']=='prepared';ensure_inactive()
    assert all(digest(output/'runner'/n)==h for n,h in p['source_sha256'].items())
    return private_supervisor(output,[('dit_decoder_compression_controller',
        [sys.executable,'-u',str(output/'runner/dit_decoder_compression_study.py'),'run','--output',str(output)])],label='controller',autostart=True)


def stage_cases(output,name,p):
    if name=='preflight': return preflight_cases(output/name,p['classifier_run'])
    if name=='quality': return quality_cases(output/name,p['classifier_run'])
    raise ValueError('Unknown fixed stage')


def digest_parameters_by_prefix(model, pattern):
    import hashlib
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if pattern in n:
            h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


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
    assert replay['source_sha256']==p['training_source_sha256']
    assert classifier_provenance(p['classifier_run'])==p['auxiliary_classifier']
    claims=json.loads((output/name/'hardware_claims.json').read_text());assert set(claims)=={j['service'] for j in jobs}
    bases,adapters,traces,noise,label,dropped,heads={},{},{},{},{},{},{}
    real=[]
    for row in replay['rows']:
        cfg=Config(**row['config']);assert row['config']==cases[row['family'],cfg.seed]
        m=row['family'];torch.manual_seed(cfg.seed);model=make_model(cfg);stats=model.statistics()
        assert row['parameters']==PARAMETERS[m]
        for k in ('parameters','adapter_parameters','forward_matmul_flops_per_image_estimate'):assert row[k]==stats[k]
        diag=row['design_diagnostics'];assert diag['base_initialization_sha256']==digest_parameters(model)
        assert diag['adapter_initialization_sha256']==digest_parameters(model,True)
        bases.setdefault(cfg.seed,set()).add(diag['base_initialization_sha256'])
        adapters.setdefault((cfg.seed,m),set()).add(diag['adapter_initialization_sha256'])
        events=diag['diagnostics'];assert [e['step'] for e in events]==[int(s) for s in cfg.diagnostic_steps.split(',')]
        assert all(b['gate_rms']==b['delta_rms']==0 for b in events[0]['branches'])
        assert all(v<=10 for v in diag['adapter_first_nonzero_gradient_observed'].values())
        assert len([k for k in diag['adapter_first_nonzero_gradient_observed'] if k.endswith('_offset')])==16
        expected_tensors=80 if cfg.decoder_cp_width else 48
        assert len(diag['adapter_first_nonzero_gradient_observed'])==expected_tensors
        assert cfg.decoder_cp_width==DESIGNS[m][1] and cfg.n_basis==DESIGNS[m][0]
        head_hash=digest_parameters_by_prefix(model, '.coefficients.')
        heads.setdefault((cfg.seed,cfg.n_basis),set()).add(head_hash)
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
    assert all(len(v)==1 for groups in (bases,adapters,noise,label,dropped,heads) for v in groups.values())
    assert all(all(t==ts[0] for t in ts) for ts in traces.values()) and all(v==real[0] for v in real)
    return replay


def run(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='prepared'
    try:
        p.update(state='preflight',started_utc=now());write_json(output/'protocol.json',p)
        execute_stage(output,p,'preflight',preflight_cases(output/'preflight',p['classifier_run']),stage_examples=(328,5000),phase='screen')
        pf=audit_stage(output,'preflight')
        p.update(state='quality',preflight_complete_utc=now());write_json(output/'protocol.json',p)
        execute_stage(output,p,'quality',quality_cases(output/'quality',p['classifier_run']),phase='screen')
        q=audit_stage(output,'quality');write_json(output/'report.json',dict(preflight=pf,quality=q))
        p.update(state='complete',completed_utc=now());write_json(output/'protocol.json',p)
        audit(output);write_summary(output,q['rows'],p)
    except BaseException as error:
        p.update(state='failed',failed_utc=now(),failure_type=type(error).__name__,failure=str(error))
        write_json(output/'protocol.json',p);raise


def audit(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='complete'
    report=json.loads((output/'report.json').read_text());stages={n:audit_stage(output,n) for n in ('preflight','quality')}
    assert all(same_stage_evidence(stages[n],report[n]) for n in stages)
    assert stages['preflight']['runtime']==stages['quality']['runtime']
    assert all(digest(Path(p['reference_workflow'])/n)==h for n,h in p['reference_files_sha256'].items())
    assert all(digest(output/'runner'/n)==h for n,h in p['unchanged_training_source_sha256'].items())
    progress=[json.loads(s) for s in (output/'progress.jsonl').read_text().splitlines()]
    assert all(e['active']==e['submitted']-e['completed']<=2 for e in progress)
    assert progress[-1]['completed']==10 and progress[-1]['active']==0
    retained=sum(f.stat().st_size for f in output.rglob('*') if f.is_file());assert retained<p['expected_storage_upper_bound_bytes']
    result=dict(checks={n:True for n in ('all_fifteen_workers_sources_configs_and_exits_audited',
       'unchanged_training_helpers_and_evaluators_preserved','exact_five_design_two_seed_full_horizon_grid',
       'paired_base_heads_data_noise_and_dropout_streams','zero_start_and_all_decoder_gradients_wake',
       'same_state_guidance_and_equal_evaluation_budgets','two_gpu_cap_and_claim_identity',
       'no_weights_checkpoints_test_or_storage_overrun')},report_sha256=digest(output/'report.json'),bytes_retained=retained)
    write_json(output/'audit.json',result);return result


def write_summary(output,rows,p):
    keyed={(r['family'],r['config']['seed']):r for r in rows};groups=[];contrasts={}
    for f in DESIGNS:
        rs=[keyed[f,s] for s in SEEDS]
        groups.append(dict(family=f,parameters=PARAMETERS[f],statistics=p['model_statistics'][f],
             final_guidance={scale:dict(fid=mean_sd([r['guidance_evaluations'][scale]['fid'] for r in rs]),
              agreement_percent=mean_sd([100*r['guidance_evaluations'][scale]['class_agreement']['generated']['agreement'] for r in rs]))
              for scale in ('1.0','2.0')},
             mean_train_images_per_second=mean_sd([r['mean_train_images_per_second_excluding_warmup'] for r in rs]),
             peak_allocated_bytes=mean_sd([r['peak_allocated_bytes'] for r in rs])))
        if f.startswith('cp'):
            controls=['dense_s32']+(['dense_s64'] if DESIGNS[f][0]==64 else [])
            contrasts[f]={ctrl:{scale:mean_sd([keyed[f,s]['guidance_evaluations'][scale]['fid']-
                keyed[ctrl,s]['guidance_evaluations'][scale]['fid'] for s in SEEDS]) for scale in ('1.0','2.0')} for ctrl in controls}
    write_json(output/'comparison.json',dict(aggregates=groups,paired_fid_contrasts=contrasts,limits=p['limits']))
    lines=['# CP decoder compression screen','','Two full 30k seeds; mean ± sample SD, not confidence intervals. No recipe selection.','',
           '| Design | Parameters | CFG1 FID | CFG2 FID | CFG2 class agreement |','|---|---:|---:|---:|---:|']
    for x in groups:
        g=x['final_guidance'];a,b=g['1.0']['fid'],g['2.0']['fid'];v=g['2.0']['agreement_percent']
        lines.append(f"| {x['family']} | {x['parameters']:,} | {a['mean']:.2f} ± {a['sd']:.2f} | {b['mean']:.2f} ± {b['sd']:.2f} | {v['mean']:.2f} ± {v['sd']:.2f}% |")
    lines+=['','Limits: '+' '.join(p['limits']),'','[Evidence](report.json) · [Audit](audit.json) · [Comparison](comparison.json) · [Protocol](protocol.json)','']
    (output/'summary.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','launch','run','audit'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    fn={'prepare':lambda:prepare(args.output,args.launch),'launch':lambda:launch_controller(args.output),
        'run':lambda:run(args.output),'audit':lambda:audit(args.output)}[args.action]
    print(json.dumps(fn(),indent=2),flush=True)


if __name__=='__main__':main()

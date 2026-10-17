#!/usr/bin/env python3
"""Channel-modulated static factors with static/dense/CP controls and equal adapter-LR budgets."""
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

REFERENCE = 'runs/dit-decoder-core-20261014T071618Z'
DESIGNS = {'dense_s32': (32, 0, 0, 'none'), 'cp_s32_p128': (32, 128, 0, 'none'),
           'channel_static': (32, 0, 0, 'static'), 'channel_s32': (32, 0, 0, 'modulated'),
           'channel_s64': (64, 0, 0, 'modulated')}
RATES = (1., 4.)
SEEDS = (124, 125)

HORIZON = 30000
PARAMETERS = dict(dense_s32=12593712, cp_s32_p128=11364912, channel_static=9922608,
                  channel_s32=10381872, channel_s64=10841136)
OVERLAYS = ('dit_cifar10.py', 'train_dit_cifar10.py', 'dit_quality_diagnostics.py',
            'benchmark_dit_design.py', 'summarize_dit_cifar10_screen.py',
            'dit_channel_factor_study.py', 'test_dit_channel_factor_study.py', 'test_dit_channel_factors.py')



def family_key(design, rate):
    if design not in DESIGNS or rate not in RATES:
        raise ValueError('Unknown fixed decoder design or adapter LR')
    return f'{design}_a{rate:g}'



def make_config(design, seed, output, classifier, rate=1.):
    family_key(design, rate)
    s, p, t, mode = DESIGNS[design]
    return Config(variant='dplr', width=256, depth=8, heads=4, patch_size=4, mlp_ratio=4.,
                  rank=8, n_basis=s, decoder_cp_width=p, decoder_tucker_width=t, decoder_channel_mode=mode, factor_offsets=True,
                  adapter_input='normalized', adapter_gain=1/math.sqrt(8), adapter_lr_multiplier=rate,
                  seed=seed, lr=8e-4, max_steps=HORIZON, lr_schedule_steps=HORIZON, warmup_steps=50,
                  batch_size=128, eval_every=5000, log_every=500, cpu_threads=2, num_workers=2,
                  adam_beta1=.9, adam_beta2=.999, fused_adamw=True, compile_model=True,
                  compile_mode='default', compile_dynamic=True, class_dropout_prob=.1,
                  guidance_scale=1., guidance_scales='1,2', generative_eval_steps='10000,20000',
                  sampling_steps=50, fid_images=5000, fid_batch_size=64, sample_images=100,
                  classifier_run=classifier, diagnostic_steps='0,1,3,5,10,25,50,100,250,500,1000,5000,10000,20000,30000',
                  output_dir=str(output))


def quality_cases(output, classifier):
    cases = [(family_key(f,r), s, make_config(f,s,output/f'{family_key(f,r)}_s{s}',classifier,r))
             for f in DESIGNS for r in RATES for s in SEEDS]
    random.Random(2080).shuffle(cases)
    return cases


def preflight_cases(output, classifier):
    seed=SEEDS[0]
    return [(family_key(f,4.),seed,replace(make_config(f,seed,output/f'{family_key(f,4.)}_s{seed}',classifier,4.),
             max_steps=12,lr_schedule_steps=12,warmup_steps=2,train_limit=328,eval_every=6,log_every=6,
             diagnostic_steps='0,1,3,5,10,12',generative_eval_steps='3,6',sampling_steps=2,
             fid_images=20,fid_batch_size=10,sample_images=10)) for f in DESIGNS]


def verify_initial_channel_pairing(seed):
    torch.manual_seed(seed);static=make_model(make_config('channel_static',seed,Path('/unused'),None))
    generator=torch.Generator().manual_seed(2032)
    c=torch.nn.functional.layer_norm(torch.randn(2,256,generator=generator),(256,));records=[]
    for design in ('channel_s32','channel_s64'):
        torch.manual_seed(seed);model=make_model(make_config(design,seed,Path('/unused'),None))
        errors=[]
        for a,b in zip((x.adapter for x in static.blocks),(x.adapter for x in model.blocks)):
            assert torch.equal(a.U_static,b.U_static) and torch.equal(a.V_static,b.V_static)
            assert torch.equal(a.U_offset,b.U_offset) and torch.equal(a.V_offset,b.V_offset)
            gu,gv=b.channel_scales(c);assert torch.equal(gu,torch.ones_like(gu)) and torch.equal(gv,torch.ones_like(gv))
            for enabled in (False,True):
                with torch.autocast('cpu',dtype=torch.bfloat16,enabled=enabled):
                    u,v=a.factors(c);uu,vv=b.factors(c)
                assert torch.equal(u,uu) and torch.equal(v,vv)
                errors.append((u-uu).abs().max().item())
        records.append(dict(design=design,initial_scales_one=True,max_input_factor_absolute_error=max(errors)))
    return dict(seed=seed,paired_static_factor_and_offset_slots=True,records=records,
                precision='FP32 and BF16 eager CPU factors; compiled CUDA parity is not asserted')


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
    shutil.copyfile(ref/'runner/dit_cifar10.py',runner/'channel_factor_baseline_model.py')
    unchanged={n:h for n,h in old['training_source_sha256'].items() if n not in OVERLAYS}
    assert all(digest(runner/n)==h for n,h in unchanged.items())
    sizes={}
    for f in DESIGNS:
        torch.manual_seed(SEEDS[0]); stats=make_model(make_config(f,SEEDS[0],output/'unused',classifier)).statistics()
        assert stats['parameters']==PARAMETERS[f]; sizes[f]=stats
    p=dict(state='prepared',created_utc=now(),repo_root=str(root),purpose='Affordable row/column channel modulation of static low-rank factors with static/dense/CP controls',
           reference_workflow=str(ref),reference_files_sha256={n:digest(ref/n) for n in ('protocol.json','report.json','audit.json','comparison.json')},
           source_sha256={f.name:digest(f) for f in runner.iterdir() if f.is_file()},
           training_source_sha256={n:digest(runner/n) for n in (*SOURCE_FILES,'dit_quality_diagnostics.py','benchmark_dit_design.py')},
           unchanged_training_source_sha256=unchanged,declared_overlays=list(OVERLAYS),designs=DESIGNS,parameters=PARAMETERS,
           model_statistics=sizes,seeds=list(SEEDS),adapter_lr_grid=list(RATES),train_steps=HORIZON,lr_schedule_steps=HORIZON,
           planned_runs=dict(preflight=5,quality=20),max_concurrent_workers=2,classifier_run=classifier,
           auxiliary_classifier=old['auxiliary_classifier'],evaluator_cache=old['evaluator_cache'],
           expected_storage_upper_bound_bytes=4*1024**3,disk_free_bytes_before_launch=free,
           estimated_gpu_hours='8–12 aggregate GPU-hours; approximately 4–6 wall hours with two available GPUs',
           weights_saved=False,periodic_checkpoints=False,test_evaluated=False,selection_metric=None,
           artifact_purposes=dict(compact_evidence='configs/sources/runtime/seeds/splits/metrics/diagnostics/logs/audits/reports',
              compiler_cache='compatibility/performance provenance, expected at most 3 GiB',
              diagnostic_grids='twenty final 100-image EMA CFG1 grids plus five 10-image preflight grids'),
           retention='Declared evidence, caches and grids until explicitly removed; prior evidence/evaluator unchanged',
           initialization=dict(static_input_factor='independent normal variance1/d; static output factor zero; offsets zero',
              channel_scales='1 + a(c) @ zero U_scale_basis and 1 + b(c) @ zero V_scale_basis',
              coefficient_heads='nonzero default affine heads; do not zero both projection stages',
              static_control='same static factor/offset slots, condition heads removed; common AdaLN stays conditional',
              generator_seed='training seed +100000+adapter index; isolated CPU generator; dense temporary tensors discarded before CUDA',
              pairing='identical base/null weights; static/modulated initial factors/scales paired; same-s non-static coefficient heads paired; dense/CP initial factor functions are not paired to channel factors'),
           initial_channel_pairing=[verify_initial_channel_pairing(seed) for seed in SEEDS],
           limits=['Five designs × two adapter LRs × two fresh seeds × full30k; no automatic selection, longer confirmation or grid expansion.',
              'Produced transform rank8, normalized value input and parallel MLP delta remain; channel scales act only inside the adapter, not on the base scale/shift output.',
              'Input/output channel scales move static factor subspaces with condition, unlike rank-only gains; this is a different restriction from dense/CP factor generation.',
              'Static control is condition-independent only in its transform; the common AdaLN and upstream hidden states remain class/time conditioned.',
              'Static offsets stay enabled in all designs. Static control has redundant factor/offset slots to match the modulated branch initialization and shared parameter routes.',
              'Identity channel scales and zero delta do not establish trained variance preservation or matched initial Jacobians against dense/CP; multiplicative and offset gauges remain.',
              'Global LR8e-4, gain1/sqrt(8), rank8 fixed; adapter multipliers1/4 apply to all adapter parameters, including heads, scales and offsets.',
              'All designs get equal two-point adapter-LR budgets. The dense/CP/static/modulated models are not parameter or FLOP matched.',
              'Every cell trains anew. Two seeds and a reused research holdout/evaluation noise limit conclusions; historical quality outcomes are context only.',
              'Primary comparisons use final30k CFG2 EMA FID; CFG1/class agreement and intermediate convergence also reported. No best-intermediate selection.',
              'Whole-training allocation and concurrent throughput are descriptive. No per-token execution or dedicated speed benchmark is launched.',
              'No output normalization, new gain, muP rule, weights/checkpoints, official test, evaluator downloads, causal retrieval/LM launch or implicit resume.'])
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
    return private_supervisor(output,[('dit_channel_factor_controller',
        [sys.executable,'-u',str(output/'runner/dit_channel_factor_study.py'),'run','--output',str(output)])],label='controller',autostart=True)


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
        m=row['family'].rsplit('_a',1)[0];torch.manual_seed(cfg.seed);model=make_model(cfg);stats=model.statistics()
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
        expected_tensors=32 if cfg.decoder_channel_mode=='static' else 64 if cfg.decoder_channel_mode=='modulated' else 80 if cfg.decoder_cp_width else 48
        assert len(diag['adapter_first_nonzero_gradient_observed'])==expected_tensors
        assert cfg.decoder_cp_width==DESIGNS[m][1] and cfg.n_basis==DESIGNS[m][0] and cfg.decoder_tucker_width==DESIGNS[m][2] and cfg.decoder_channel_mode==DESIGNS[m][3]
        assert cfg.adapter_lr_multiplier==float(row['family'].rsplit('_a',1)[1])
        if row['coefficient_parameters']:
            head_hash=digest_parameters_by_prefix(model, '.coefficients.')
            heads.setdefault((cfg.seed,cfg.n_basis),set()).add(head_hash)
        elif cfg.decoder_channel_mode=='static':
            assert not any('.coefficients.' in n for n,_ in model.named_parameters())
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
        from train_dit_cifar10 import make_optimizer, optimizer_group_metadata
        actual_groups=setup['optimizer_groups']
        assert actual_groups==optimizer_group_metadata(model,make_optimizer(model,replace(cfg,device='cpu')))
        assert sum(g['parameter_count'] for g in actual_groups if g['is_adapter'])==row['adapter_parameters']
        for e in logs:
            if e['event']=='train':
                assert len(e['optimizer_group_lrs'])==len(actual_groups)
                assert all(math.isclose(lr,e['base_lr']*(cfg.adapter_lr_multiplier if g['is_adapter'] else 1),rel_tol=1e-14)
                           for lr,g in zip(e['optimizer_group_lrs'],actual_groups))
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
    for seed in {cfg['seed'] for cfg in cases.values()}:
        verify_initial_channel_pairing(seed)
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
    assert progress[-1]['completed']==20 and progress[-1]['active']==0
    retained=sum(f.stat().st_size for f in output.rglob('*') if f.is_file());assert retained<p['expected_storage_upper_bound_bytes']
    result=dict(checks={n:True for n in ('all_twenty_five_workers_sources_configs_and_exits_audited',
       'unchanged_training_helpers_and_evaluators_preserved','exact_five_design_two_lr_two_seed_full_horizon_grid',
       'paired_base_heads_static_factor_functions_data_noise_and_dropout_streams','actual_optimizer_group_rates_and_ownership','zero_start_and_all_decoder_gradients_wake',
       'same_state_guidance_and_equal_evaluation_budgets','two_gpu_cap_and_claim_identity',
       'no_weights_checkpoints_test_or_storage_overrun')},report_sha256=digest(output/'report.json'),bytes_retained=retained)
    write_json(output/'audit.json',result);return result


def write_summary(output,rows,p):
    keyed={(r['family'],r['config']['seed']):r for r in rows};groups=[];contrasts={};sensitivity={}
    for design in DESIGNS:
        sensitivity[design]={}
        for rate in RATES:
            family=family_key(design,rate);rs=[keyed[family,s] for s in SEEDS]
            groups.append(dict(design=design,adapter_lr_multiplier=rate,family=family,parameters=PARAMETERS[design],statistics=p['model_statistics'][design],
                final_guidance={scale:dict(fid=mean_sd([r['guidance_evaluations'][scale]['fid'] for r in rs]),
                 agreement_percent=mean_sd([100*r['guidance_evaluations'][scale]['class_agreement']['generated']['agreement'] for r in rs]))
                 for scale in ('1.0','2.0')},
                mean_train_images_per_second=mean_sd([r['mean_train_images_per_second_excluding_warmup'] for r in rs]),
                peak_allocated_bytes=mean_sd([r['peak_allocated_bytes'] for r in rs])))
            controls=['dense_s32']+(['channel_static','cp_s32_p128'] if design.startswith('channel_s') and design!='channel_static' else [])
            if design!='dense_s32':
                contrasts[family]={family_key(ctrl,rate):{scale:mean_sd([keyed[family,s]['guidance_evaluations'][scale]['fid']-
                    keyed[family_key(ctrl,rate),s]['guidance_evaluations'][scale]['fid'] for s in SEEDS]) for scale in ('1.0','2.0')} for ctrl in controls}
        for scale in ('1.0','2.0'):
            sensitivity[design][scale]=mean_sd([keyed[family_key(design,4.),s]['guidance_evaluations'][scale]['fid']-
                keyed[family_key(design,1.),s]['guidance_evaluations'][scale]['fid'] for s in SEEDS])
    write_json(output/'comparison.json',dict(aggregates=groups,paired_fid_contrasts_at_same_rate=contrasts,
                adapter_lr_four_minus_unit_fid=sensitivity,limits=p['limits']))
    lines=['# Channel-modulated static factors and adapter-LR screen','','Two full30k seeds per design/rate; mean ± sample SD, not confidence intervals. All rates reported; no automatic recipe selection.','',
           '| Design | Adapter LR | Parameters | CFG1 FID | CFG2 FID | CFG2 class agreement |','|---|---:|---:|---:|---:|---:|']
    for x in groups:
        g=x['final_guidance'];a,b=g['1.0']['fid'],g['2.0']['fid'];v=g['2.0']['agreement_percent']
        lines.append(f"| {x['design']} | {x['adapter_lr_multiplier']:g} | {x['parameters']:,} | {a['mean']:.2f} ± {a['sd']:.2f} | {b['mean']:.2f} ± {b['sd']:.2f} | {v['mean']:.2f} ± {v['sd']:.2f}% |")
    lines+=['','Limits: '+' '.join(p['limits']),'','[Evidence](report.json) · [Audit](audit.json) · [Comparison](comparison.json) · [Protocol](protocol.json)','']
    (output/'summary.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','launch','run','audit'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    fn={'prepare':lambda:prepare(args.output,args.launch),'launch':lambda:launch_controller(args.output),
        'run':lambda:run(args.output),'audit':lambda:audit(args.output)}[args.action]
    print(json.dumps(fn(),indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Bounded refinement of the audited ViT calibration grid, with unchanged training."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import vit_token_study as study
from dit_followup import private_supervisor
from train_cifar10 import load_config, write_json

EXTRA_RATES = {'s12': (2.,), 's48': (.5,), 'static_r8_diagonal': (8., 12.)}
REFERENCE = 'runs/vit-token-20261011T070503Z'


def reference_rows(output, protocol):
    reference = Path(protocol['reference_workflow'])
    assert study.digest(reference/'protocol.json') == protocol['reference_protocol_sha256']
    assert study.digest(reference/'calibration/summary.json') == protocol['reference_calibration_sha256']
    # Preserve and verify the old failed gate rather than declaring it successful.
    old = json.loads((reference/'protocol.json').read_text())
    assert old['state']=='failed' and old['failure_type']=='ValueError'
    assert old['failure']=='No rate matches declared update bounds for s12; withhold quality'
    assert not (reference/'recovery-controller.sock').exists()
    rows = study.audit_stage(reference,'calibration')
    assert rows == json.loads((reference/'calibration/summary.json').read_text())['rows']
    study.audit_stage(reference,'preflight')
    expected = {('s24',s,1.) for s in study.PILOT_SEEDS} | {
        (f,s,r) for f,rates in study.RATE_GRID.items() for r in rates for s in study.PILOT_SEEDS}
    assert len(rows)==28 and {(r['family'],r['seed'],r['rate'])for r in rows}==expected
    return rows


def refinement_cases(output):
    return [(family,seed,rate,study.make_config(family,seed,output/f'{family}_lr{rate:g}_s{seed}',rate,pilot=True))
            for family,rates in EXTRA_RATES.items() for rate in rates for seed in study.PILOT_SEEDS]


def evaluate_rates(rows):
    grid = {f:tuple(sorted((*rates,*EXTRA_RATES.get(f,())))) for f,rates in study.RATE_GRID.items()}
    expected = {('s24',s,1.) for s in study.PILOT_SEEDS} | {
        (f,s,r) for f,rates in grid.items() for r in rates for s in study.PILOT_SEEDS}
    keyed = {(r['family'],r['seed'],r['rate']):r for r in rows}
    if len(rows)!=36 or set(keyed)!=expected:
        raise ValueError('Require exactly the 28 original and eight new pilot cases')
    selections={'s24':1.}; details={}; failed=[]
    for family,rates in grid.items():
        candidates=[]
        for rate in rates:
            ratios=[]
            for seed in study.PILOT_SEEDS:
                candidate=keyed[family,seed,rate]; reference=keyed['s24',seed,1.]
                for step in study.PROBES:
                    numerator=study.update(candidate,step); denominator=study.update(reference,step)
                    if not math.isfinite(numerator) or not math.isfinite(denominator) or min(numerator,denominator)<=0:
                        raise ValueError('Nonfinite or zero calibration update')
                    ratios.append(numerator/denominator)
            candidates.append(dict(rate=rate,ratios=ratios,within_bounds=all(.5<=r<=2. for r in ratios),
                                   score=statistics.mean(abs(math.log(r))for r in ratios)))
        details[family]=candidates; valid=[r for r in candidates if r['within_bounds']]
        if valid:selections[family]=min(valid,key=lambda r:(r['score'],r['rate']))['rate']
        else:failed.append(family)
    return dict(selected_rates=selections,candidates=details,failed_families=failed,quality_gate_passed=not failed,
                reference='s24 offset-enabled, unit adapter LR',seeds=list(study.PILOT_SEEDS),
                probe_steps=list(study.PROBES),ratio_bounds=[.5,2.],
                criterion='Closest geometric update magnitude among rates meeting every per-seed/per-probe bound; no accuracy selection',
                limits='Early low-rank function updates only; diagonal/whole-network updates and full trajectory remain unmatched.')


def verify_reused_evidence(old,new):
    expected={(f,s,r)for f,rates in EXTRA_RATES.items()for r in rates for s in study.PILOT_SEEDS}
    assert len(new)==8 and {(r['family'],r['seed'],r['rate'])for r in new}==expected
    for candidate in new:
        seed=candidate['seed']; anchor=next(r for r in old if r['family']=='s24' and r['seed']==seed)
        family_anchor=next(r for r in old if r['family']==candidate['family'] and r['seed']==seed and r['rate']==1.)
        a,b=candidate['results']['token_diagnostics'],family_anchor['results']['token_diagnostics']
        assert a['base_initialization_sha256']==b['base_initialization_sha256']==anchor['results']['token_diagnostics']['base_initialization_sha256']
        assert a['adapter_initialization_sha256']==b['adapter_initialization_sha256']
        assert a['probe_sha256']==b['probe_sha256'] and a['live_images_sha256']==b['live_images_sha256']
        def stream(row):
            return [{k:e[k]for k in ('step','batch_images_sha256','batch_labels_sha256')}
                    for e in row['results']['token_diagnostics']['diagnostics'][1:]]
        assert stream(candidate)==stream(anchor)
        assert {k:v for k,v in candidate['config'].items()if k not in ('hyper_lr_multiplier','output_dir')}=={
            k:v for k,v in family_anchor['config'].items()if k not in ('hyper_lr_multiplier','output_dir')}
        for name in ('provenance.json','dataset.json'):
            ca=json.loads((Path(candidate['output_dir'])/name).read_text());cb=json.loads((Path(anchor['output_dir'])/name).read_text())
            if name=='provenance.json':
                assert all(ca[k]==cb[k]for k in ('python','torch','cuda_wheel','package_versions','source_sha256'))
            else:assert ca==cb


def prepare(output,launch=False):
    import torch
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent;reference=root/REFERENCE;output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    free=shutil.disk_usage(root).free
    if free<8*1024**3:raise RuntimeError('Require 8 GiB free for the declared 4 GiB bound and headroom')
    for path in Path('/tmp/vast_gpu_claims').glob('*.claim'):
        if json.loads(path.read_text()).get('owner')=='adalora':raise RuntimeError('Existing adalora workers must finish')
    prior=json.loads((reference/'protocol.json').read_text())
    p={**prior,'state':'prepared','created_utc':study.now(),'repo_root':str(root),
       'purpose':'Refine only missing intermediate calibration rates; retain the original bounds and original failed gate',
       'reference_workflow':str(reference),'reference_protocol_sha256':study.digest(reference/'protocol.json'),
       'reference_calibration_sha256':study.digest(reference/'calibration/summary.json'),
       'extra_rates':EXTRA_RATES,'planned_runs':dict(calibration=8,quality=16),
       'resources':dict(estimated_gpu_hours='2–4 aggregate, approximately 1–2 wall hours with two available GPUs',max_our_gpus=2,
           disk_free_bytes_before_launch=free,retained_artifact_bound_bytes=4*1024**3,
           artifact_purposes={'compact_evidence':'frozen configs/sources/runtime/seed/split/metrics/logs/diagnostics/audits/reports',
                              'compiler_cache':'compatibility/performance provenance, expected at most 3 GiB'},
           retention='Compact evidence and declared compiler caches until explicitly removed'),
       'limits':[*prior['limits'],'Eight additional rates are frozen after observing coarse-grid updates; no quality-based selection or relaxed bounds.',
                 'The original two preflights and 28 pilots are reused only after full audit and paired source/runtime/initialization/stream checks.']}
    for k in ('failed_utc','failure_type','failure','started_utc','preflight_reporting_recovery','calibration_complete_utc','calibration_sha256'):
        p.pop(k,None)
    reference_rows(output,p)
    output.mkdir(parents=True);runner=output/'runner'
    shutil.copytree(reference/'preflight-reporting-recovery/runner',runner,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('vit_token_refinement.py','test_vit_token_refinement.py'):shutil.copy2(root/name,runner/name)
    p['source_sha256']={f.name:study.digest(f)for f in runner.iterdir()if f.is_file()}
    # All trainer/model/diagnostic sources are identical to the first pilot cohort.
    source=json.loads((Path(json.loads((reference/'calibration/summary.json').read_text())['rows'][0]['output_dir'])/'provenance.json').read_text())['source_sha256']
    assert all(p['source_sha256'][name]==sha for name,sha in source.items())
    write_json(output/'protocol.json',p);planned=output/'planned_configs';planned.mkdir()
    for _,_,_,cfg in refinement_cases(output/'calibration'):
        path=planned/(Path(cfg.output_dir).name+'.json');write_json(path,asdict(cfg));load_config(['--override_json',str(path)])
    if launch:
        private_supervisor(output,[('vit_refinement_controller',[sys.executable,'-u',str(runner/Path(__file__).name),'run','--output',str(output)])],label='controller',autostart=True)
    return dict(output=str(output),launched=launch,new_pilots=8,reused_pilots=28,gated_quality_runs=16,max_our_gpus=2)


def run(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text())
    if p['state']!='prepared':raise ValueError('Fresh refinement only; no implicit resume')
    try:
        old=reference_rows(output,p)
        p.update(state='calibration',started_utc=study.now());write_json(output/'protocol.json',p)
        new=study.execute_stage(output,'calibration',refinement_cases(output/'calibration'))
        verify_reused_evidence(old,new);selection=evaluate_rates([*old,*new]);write_json(output/'calibration.json',selection)
        if not selection['quality_gate_passed']:raise ValueError('Refined calibration still misses fixed update bounds; withhold quality')
        p.update(state='quality',calibration_complete_utc=study.now(),calibration_sha256=study.digest(output/'calibration.json'));write_json(output/'protocol.json',p)
        rates=selection['selected_rates'];cases=[(f,s,rates.get(f,1.),study.make_config(f,s,output/'quality'/f'{f}_s{s}',rates.get(f,1.)))for f in study.FAMILIES for s in study.QUALITY_SEEDS]
        quality=study.execute_stage(output,'quality',cases)
        write_json(output/'report.json',dict(reference_rows=old,refinement_rows=new,quality_rows=quality))
        p.update(state='complete',completed_utc=study.now());write_json(output/'protocol.json',p)
        audit(output);study.write_summary(output,quality,p)
    except BaseException as error:
        p.update(state='failed',failed_utc=study.now(),failure_type=type(error).__name__,failure=str(error));write_json(output/'protocol.json',p);raise


def audit(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='complete'
    report=json.loads((output/'report.json').read_text());old=reference_rows(output,p);new=study.audit_stage(output,'calibration');quality=study.audit_stage(output,'quality')
    assert report==dict(reference_rows=old,refinement_rows=new,quality_rows=quality)
    verify_reused_evidence(old,new);selection=evaluate_rates([*old,*new]);assert selection['quality_gate_passed']
    assert selection==json.loads((output/'calibration.json').read_text()) and study.digest(output/'calibration.json')==p['calibration_sha256']
    assert len(quality)==16 and {(r['family'],r['seed'])for r in quality}=={(f,s)for f in study.FAMILIES for s in study.QUALITY_SEEDS}
    assert all(r['rate']==selection['selected_rates'].get(r['family'],1.)for r in quality)
    progress=[json.loads(line)for line in(output/'progress.jsonl').read_text().splitlines()]
    assert all(e['active']==e['submitted']-e['completed']<=2 for e in progress)
    retained=sum(f.stat().st_size for f in output.rglob('*')if f.is_file());assert retained<p['resources']['retained_artifact_bound_bytes']
    checks=('original_gate_and_audited_pilots_preserved','exact_eight_new_rates_and_36_case_selection',
            'unchanged_training_sources_runtime_split_initialization_and_streams','all_new_workers_completed_with_claim_identity',
            'unchanged_per_seed_per_probe_bounds_and_update_only_selection','sixteen_fresh_quality_runs_with_frozen_rates',
            'two_gpu_worker_cap','no_weights_test_evaluation_and_storage_bound')
    result=dict(checks={k:True for k in checks},report_sha256=study.digest(output/'report.json'),bytes_retained=retained)
    write_json(output/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--output',type=Path);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    output=args.output or Path('runs')/('vit-token-refinement-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action=='prepare':print(json.dumps(prepare(output,args.launch)),flush=True)
    elif args.action=='run':run(output)
    else:print(json.dumps(audit(output),indent=2))

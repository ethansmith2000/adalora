#!/usr/bin/env python3
"""Complete the frozen split-LR study after a container interruption, without weights."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import vit_split_lr_study as study
from dit_followup import private_supervisor
from train_cifar10 import provenance, write_json

REFERENCE = 'runs/vit-split-lr-20261011T214507Z'


def alive(pid):
    try:os.kill(int(pid),0)
    except ProcessLookupError:return False
    return True


def completed_jobs(reference):
    screen=json.loads((reference/'quality/screen.json').read_text());log=(reference/'quality/supervisord.log').read_text()
    expected={(f,s)for f in study.FAMILIES for s in study.QUALITY_SEEDS}
    assert len(screen['jobs'])==16 and {(j['family'],j['seed'])for j in screen['jobs']}==expected
    completed=[]
    for job in screen['jobs']:
        output=Path(job['output_dir']);result=output/'results.json';metrics=output/'metrics.jsonl'
        if not result.exists() or not metrics.exists():continue
        events=[json.loads(line)for line in metrics.read_text().splitlines()]
        if (events and events[-1]['event']=='complete' and
                json.loads(result.read_text())['completed_steps']==study.HORIZON and
                f"exited: {job['service']} (exit status 0; expected)" in log):
            completed.append(job)
    if not completed or len(completed)==16:
        raise ValueError('Recovery requires an interrupted study with a strict subset of completed runs')
    return completed


def check_reference_inactive(reference):
    for name in ('controller.pid','quality/supervisor.pid'):
        path=reference/name
        if path.exists() and alive(path.read_text().strip()):
            raise RuntimeError('Original supervisor is still active; do not duplicate workers')
    for path in Path('/tmp/vast_gpu_claims').glob('*.claim'):
        claim=json.loads(path.read_text())
        if claim.get('owner')=='adalora' and alive(claim['pid']):
            raise RuntimeError('Existing adalora workers must finish; stale dead claims are left intact')


def reference_projection(output,reference,jobs):
    """A declared audit view of completed workers; all worker files stay in place.

    The original stage's live socket and incomplete jobs are not represented.
    The protocol count, screen and claims are restricted to the identical subset;
    source/config/output/exit records are audited by the original frozen auditor.
    """
    proxy=output/'reference-check';proxy.mkdir();(proxy/'runner').symlink_to(reference/'runner',target_is_directory=True)
    p=json.loads((reference/'protocol.json').read_text());p['planned_runs']={**p['planned_runs'],'quality':len(jobs)}
    write_json(proxy/'protocol.json',p);stage=proxy/'quality';stage.mkdir()
    write_json(stage/'screen.json',dict(jobs=jobs,audit_view_only=True,original_screen_sha256=study.common.digest(reference/'quality/screen.json')))
    claims=json.loads((reference/'quality/hardware_claims.json').read_text())
    write_json(stage/'hardware_claims.json',{j['service']:claims[j['service']]for j in jobs})
    shutil.copyfile(reference/'quality/supervisord.log',stage/'supervisord.log')
    return proxy


def reference_rows(output,p):
    reference=Path(p['reference_workflow'])
    assert all(study.common.digest(reference/name)==sha for name,sha in p['reference_files_sha256'].items())
    original=json.loads((reference/'protocol.json').read_text())
    assert original['state']=='quality' and all(study.common.digest(reference/'runner'/n)==s for n,s in original['source_sha256'].items())
    check_reference_inactive(reference)
    for stage in ('preflight','calibration'):
        audited=study.audit_stage(reference,stage)
        assert audited==json.loads((reference/stage/'summary.json').read_text())['rows']
    selection=study.select_rates(json.loads((reference/'calibration/summary.json').read_text())['rows'])
    assert selection['quality_gate_passed'] and selection==json.loads((reference/'calibration.json').read_text())
    assert study.common.digest(reference/'calibration.json')==original['calibration_sha256']
    jobs=completed_jobs(reference);assert jobs==p['reused_jobs']
    proxy=output/'reference-check'
    assert json.loads((proxy/'quality/screen.json').read_text())['jobs']==jobs
    assert (proxy/'quality/supervisord.log').read_bytes()==(reference/'quality/supervisord.log').read_bytes()
    expected=json.loads((reference/'protocol.json').read_text());expected['planned_runs']={**expected['planned_runs'],'quality':len(jobs)}
    assert json.loads((proxy/'protocol.json').read_text())==expected
    original_claims=json.loads((reference/'quality/hardware_claims.json').read_text())
    assert json.loads((proxy/'quality/hardware_claims.json').read_text())=={j['service']:original_claims[j['service']]for j in jobs}
    rows=study.audit_stage(proxy,'quality')
    assert {(r['family'],r['seed'])for r in rows}==set(map(tuple,p['reused_cases']))
    assert all(r['rate']==selection['selected_rates'].get(r['family'],1.)for r in rows)
    return rows


def remaining_cases(output,p):
    assert study.common.digest(output/'calibration.json')==p['calibration_sha256']
    reused=set(map(tuple,p['reused_cases']));rates=json.loads((output/'calibration.json').read_text())['selected_rates']
    return [(f,s,rates.get(f,1.),study.make_config(f,s,output/'quality'/f'{f}_s{s}',rates.get(f,1.)))
            for f in study.FAMILIES for s in study.QUALITY_SEEDS if (f,s)not in reused]


def verify_combined(rows):
    assert len(rows)==16 and {(r['family'],r['seed'])for r in rows}=={(f,s)for f in study.FAMILIES for s in study.QUALITY_SEEDS}
    study.verify_pairs(rows)
    # Pairing must hold across the pre-interruption and replacement cohorts too.
    runtime=None;dataset=None
    for seed in study.QUALITY_SEEDS:
        values=[r for r in rows if r['seed']==seed]
        assert len({r['results']['token_diagnostics']['base_initialization_sha256']for r in values if r['config']['mlp_ratio']==4.})==1
        traces=[]
        for row in values:
            output=Path(row['output_dir']);prov=json.loads((output/'provenance.json').read_text())
            current={k:prov[k]for k in ('python','torch','cuda_wheel','package_versions','source_sha256')}
            assert runtime is None or runtime==current;runtime=current
            metadata=json.loads((output/'dataset.json').read_text())
            assert dataset is None or dataset==metadata;dataset=metadata
            traces.append([{k:e[k]for k in ('step','batch_images_sha256','batch_labels_sha256')}
                           for e in row['results']['token_diagnostics']['diagnostics'][1:]])
        assert all(t==traces[0]for t in traces)


def prepare(output,launch=False):
    import torch
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent;reference=(root/REFERENCE).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    check_reference_inactive(reference)
    free=shutil.disk_usage(root).free
    if free<8*1024**3:raise RuntimeError('Require 8 GiB free for the 4 GiB artifact bound and headroom')
    original=json.loads((reference/'protocol.json').read_text());assert original['state']=='quality'
    assert study.common.digest(Path(study.__file__))==original['source_sha256']['vit_split_lr_study.py']
    jobs=completed_jobs(reference);remaining=16-len(jobs)
    output.mkdir(parents=True);runner=output/'runner'
    shutil.copytree(reference/'runner',runner,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('recover_vit_split_lr.py','test_recover_vit_split_lr.py'):shutil.copy2(root/name,runner/name)
    assert all(study.common.digest(runner/n)==sha for n,sha in original['source_sha256'].items())
    reference_projection(output,reference,jobs)
    ref_names=('protocol.json','calibration.json','preflight/summary.json','calibration/summary.json',
               'quality/screen.json','quality/hardware_claims.json','quality/supervisord.log','progress.jsonl')
    p={**original,'state':'prepared','created_utc':study.common.now(),'repo_root':str(root),
       'purpose':'Finish only the remaining frozen split-LR quality cases after container interruption; preserve completed and partial outputs',
       'source_sha256':{f.name:study.common.digest(f)for f in runner.iterdir()if f.is_file()},
       'reference_workflow':str(reference),'reference_files_sha256':{n:study.common.digest(reference/n)for n in ref_names},
       'reused_jobs':jobs,'reused_cases':[[j['family'],j['seed']]for j in jobs],
       'planned_runs':dict(quality=remaining),'combined_quality_runs':16,
       'resources':dict(estimated_gpu_hours='2–4 aggregate; approximately 1–2 wall hours with two available GPUs',max_our_gpus=2,
                        disk_free_bytes_before_launch=free,retained_artifact_bound_bytes=4*1024**3,
                        artifact_purposes={'compact_evidence':'sources/configs/runtime/seeds/splits/optimizer groups/metrics/logs/audits/reports and completed-worker audit view',
                                           'compiler_cache':'compatibility/performance provenance, expected at most 3 GiB'},
                        retention='Compact evidence and declared compiler caches until explicitly removed; original retention unchanged'),
       'limits':[*original['limits'],'Recovery reuses only fully completed, audited cases; remaining cases restart from original seeds without weights.',
                 'Container interruption spans cohorts; trainer/model sources, packages, split, initialization and observed streams are audited across cohorts.']}
    for key in ('started_utc','completed_utc','failure','failure_type','failed_utc'):p.pop(key,None)
    write_json(output/'protocol.json',p);shutil.copyfile(reference/'calibration.json',output/'calibration.json')
    old=reference_rows(output,p);write_json(output/'reused_quality.json',dict(rows=old,audit_passed=True))
    current=provenance();prior=json.loads((Path(old[0]['output_dir'])/'provenance.json').read_text())
    assert all(current[k]==prior[k]for k in ('python','torch','cuda_wheel','package_versions','source_sha256'))
    assert not torch.cuda.is_initialized()
    if launch:launch_controller(output)
    return dict(output=str(output),launched=launch,reused_quality=len(old),remaining_quality=remaining,max_our_gpus=2)


def launch_controller(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='prepared'
    assert all(study.common.digest(output/'runner'/n)==s for n,s in p['source_sha256'].items())
    check_reference_inactive(Path(p['reference_workflow']))
    return private_supervisor(output,[('vit_split_lr_recovery',[sys.executable,'-u',str(output/'runner/recover_vit_split_lr.py'),'run','--output',str(output)])],label='controller',autostart=True)


def run(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='prepared'
    try:
        old=reference_rows(output,p);assert old==json.loads((output/'reused_quality.json').read_text())['rows']
        p.update(state='quality',started_utc=study.common.now());write_json(output/'protocol.json',p)
        new=study.common.execute_stage(output,'quality',remaining_cases(output,p),auditor=study.audit_stage)
        verify_combined([*old,*new]);write_json(output/'report.json',dict(reused_quality=old,replacement_quality=new))
        p.update(state='complete',completed_utc=study.common.now());write_json(output/'protocol.json',p)
        audit(output);study.write_summary(output,[*old,*new],p)
    except BaseException as error:
        p.update(state='failed',failed_utc=study.common.now(),failure_type=type(error).__name__,failure=str(error));write_json(output/'protocol.json',p);raise


def audit(output):
    output=Path(output).resolve();p=json.loads((output/'protocol.json').read_text());assert p['state']=='complete'
    old=reference_rows(output,p);assert old==json.loads((output/'reused_quality.json').read_text())['rows']
    new=study.audit_stage(output,'quality');verify_combined([*old,*new])
    assert json.loads((output/'report.json').read_text())==dict(reused_quality=old,replacement_quality=new)
    expected={(f,s,r)for f,s,r,cfg in remaining_cases(output,p)}
    assert {(r['family'],r['seed'],r['rate'])for r in new}==expected and len(new)==p['planned_runs']['quality']
    assert study.common.digest(output/'calibration.json')==p['calibration_sha256']
    progress=[json.loads(line)for line in(output/'progress.jsonl').read_text().splitlines()]
    assert all(e['active']==e['submitted']-e['completed']<=2 for e in progress)
    retained=sum(f.stat().st_size for f in output.rglob('*')if f.is_file());assert retained<p['resources']['retained_artifact_bound_bytes']
    result=dict(checks={name:True for name in ('original_protocol_partial_records_and_completed_workers_preserved',
        'original_preflights_and_calibration_reaudited_without_new_selection','exact_remaining_cases_restart_from_seed',
        'unchanged_training_sources_runtime_split_and_cross_cohort_initialization_streams',
        'sixteen_combined_quality_cases_and_original_group_rates','all_new_workers_and_two_gpu_cap','no_weights_test_and_storage_bound')},
        report_sha256=study.common.digest(output/'report.json'),bytes_retained=retained)
    write_json(output/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','run','audit'))
    parser.add_argument('--output',type=Path);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    output=args.output or Path('runs')/('vit-split-lr-recovery-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action=='prepare':print(json.dumps(prepare(output,args.launch)),flush=True)
    elif args.action=='launch':print(launch_controller(output),flush=True)
    elif args.action=='run':run(output)
    else:print(json.dumps(audit(output),indent=2))

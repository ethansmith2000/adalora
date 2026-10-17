"""Recovery preserves complete evidence and restarts only the exact remaining cases."""
from dataclasses import asdict
import json

import pytest

import recover_vit_split_lr as recovery


def test_completion_requires_full_budget_final_event_and_successful_worker_exit(tmp_path):
    stage=tmp_path/'quality';stage.mkdir();jobs=[];exits=[]
    for index,(family,seed) in enumerate((f,s)for f in recovery.study.FAMILIES for s in recovery.study.QUALITY_SEEDS):
        out=stage/f'{family}_s{seed}';out.mkdir();service=f'worker_{index}'
        jobs.append(dict(family=family,seed=seed,output_dir=str(out),service=service))
        if index<=3:
            (out/'results.json').write_text(json.dumps({'completed_steps':10560 if index!=1 else 100}))
            (out/'metrics.jsonl').write_text(json.dumps({'event':'complete' if index!=2 else 'train'})+'\n')
        if index!=3:exits.append(f'exited: {service} (exit status 0; expected)')
    (stage/'screen.json').write_text(json.dumps({'jobs':jobs}));(stage/'supervisord.log').write_text('\n'.join(exits))
    assert recovery.completed_jobs(tmp_path)==[jobs[0]]
    (stage/'supervisord.log').write_text('')
    with pytest.raises(ValueError,match='strict subset'):recovery.completed_jobs(tmp_path)


def test_remaining_cases_are_exact_complement_with_original_rates_and_fresh_full_budget(tmp_path):
    rates={'s24':1.,'s48':.5,'static_split':16.,'static_unit':1.,'static_joint_12':12.}
    path=tmp_path/'calibration.json';path.write_text(json.dumps({'selected_rates':rates}))
    reused=[['static_capacity',102],['static_capacity',103],['static_joint_12',103],['static_split',102]]
    p={'reused_cases':reused,'calibration_sha256':recovery.study.common.digest(path)}
    cases=recovery.remaining_cases(tmp_path,p)
    assert len(cases)==12 and not {(f,s)for f,s,r,c in cases}&set(map(tuple,reused))
    assert {(f,s)for f,s,r,c in cases}|set(map(tuple,reused))=={(f,s)for f in recovery.study.FAMILIES for s in (102,103)}
    for f,s,r,cfg in cases:
        old=recovery.study.make_config(f,s,tmp_path/'old',rates.get(f,1.))
        assert {k:v for k,v in asdict(cfg).items()if k!='output_dir'}=={k:v for k,v in asdict(old).items()if k!='output_dir'}
        assert cfg.max_steps==cfg.lr_schedule_steps==10560 and cfg.epochs==30
        assert not cfg.save_weights and not cfg.evaluate_test
    path.write_text(json.dumps({'selected_rates':{**rates,'s48':1.}}))
    with pytest.raises(AssertionError):recovery.remaining_cases(tmp_path,p)


def test_reference_verification_failure_prevents_all_new_gpu_work(tmp_path,monkeypatch):
    path=tmp_path/'protocol.json';path.write_text(json.dumps({'state':'prepared'}));calls=[]
    def verify(*args):raise AssertionError('Original evidence mismatch')
    monkeypatch.setattr(recovery,'reference_rows',verify)
    monkeypatch.setattr(recovery.study.common,'execute_stage',lambda *args,**kwargs:calls.append(args))
    with pytest.raises(AssertionError,match='evidence mismatch'):recovery.run(tmp_path)
    assert not calls and json.loads(path.read_text())['state']=='failed'


def test_inactive_reference_check_rejects_live_supervisor_and_leaves_stale_pid_files(tmp_path,monkeypatch):
    path=tmp_path/'controller.pid';path.write_text('123456789')
    monkeypatch.setattr(recovery,'alive',lambda _:False);recovery.check_reference_inactive(tmp_path)
    assert path.read_text()=='123456789'
    monkeypatch.setattr(recovery,'alive',lambda _:True)
    with pytest.raises(RuntimeError,match='still active'):recovery.check_reference_inactive(tmp_path)


def paired_rows(tmp_path):
    rows=[]
    for family in recovery.study.FAMILIES:
        for seed in recovery.study.QUALITY_SEEDS:
            rate=12. if family=='static_joint_12' else 16. if family=='static_split' else .5 if family=='s48' else 1.
            cfg=recovery.study.make_config(family,seed,tmp_path/f'{family}_{seed}',rate);out=__import__('pathlib').Path(cfg.output_dir);out.mkdir()
            (out/'provenance.json').write_text(json.dumps({'python':'same','torch':'same','cuda_wheel':'same','package_versions':{},'source_sha256':{'trainer':'same'}}))
            (out/'dataset.json').write_text(json.dumps({'split':'same'}))
            diag={'base_initialization_sha256':f'base_{seed}'if family!='static_capacity'else f'wide_{seed}',
                  'adapter_initialization_sha256':f'static_{seed}',
                  'diagnostics':[{'step':0},{'step':500,'batch_images_sha256':str(seed),'batch_labels_sha256':str(seed)}]}
            rows.append(dict(family=family,seed=seed,rate=rate,config=asdict(cfg),output_dir=str(out),results={'token_diagnostics':diag}))
    return rows


@pytest.mark.parametrize('broken',['duplicate','base_initialization','adapter_initialization','runtime','split','stream'])
def test_cross_cohort_pairing_detects_each_incompatible_reuse(tmp_path,broken):
    rows=paired_rows(tmp_path);recovery.verify_combined(rows)
    row=next(r for r in rows if r['family']=='static_unit' and r['seed']==102)
    if broken=='duplicate':rows[-1]=rows[0]
    elif broken=='base_initialization':row['results']['token_diagnostics']['base_initialization_sha256']='changed'
    elif broken=='adapter_initialization':row['results']['token_diagnostics']['adapter_initialization_sha256']='changed'
    elif broken=='runtime':
        path=__import__('pathlib').Path(row['output_dir'])/'provenance.json';d=json.loads(path.read_text());d['torch']='changed';path.write_text(json.dumps(d))
    elif broken=='split':(__import__('pathlib').Path(row['output_dir'])/'dataset.json').write_text(json.dumps({'split':'changed'}))
    else:row['results']['token_diagnostics']['diagnostics'][-1]['batch_images_sha256']='changed'
    with pytest.raises(AssertionError):recovery.verify_combined(rows)

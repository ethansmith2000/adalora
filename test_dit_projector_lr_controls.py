"""Bounded calibration, paired update arithmetic and exact rate-only recipes."""
from dataclasses import asdict
import json

import pytest

import dit_projector_lr_controls as study
from dit_projector_confirmation import make_config as base_config
from train_dit_cifar10 import load_config


def fake_pilots():
    result=[]
    specs={f:(f,1.) for f in study.CONTROLS}|study.trial_specs()
    response={'affine_s16':1.,'linear_s32_h64':2.,'affine_s32':2.,'affine_s64':4.,'silu_s32_h64':.5}
    for trial,(family,rate) in specs.items():
        for seed in study.CAL_SEEDS:
            events=[dict(step=step,fixed_token_condition_function_updates=[{'update_rms':response[family]*rate}]*8)
                    for step in study.OBSERVATIONS]
            result.append(dict(family=trial,config={'seed':seed,'adapter_lr_multiplier':rate},
                               design_diagnostics={'diagnostics':events},fid=float('nan')))
    return result


def test_exact_grid_selects_using_paired_updates_and_ignores_quality():
    selected=study.select_rates(fake_pilots())
    assert selected['all_within_tolerance'] and selected['selection_uses_quality'] is False
    assert {f:v['adapter_lr_multiplier'] for f,v in selected['selected'].items()}=={
        'affine_s32':.5,'affine_s64':.25,'silu_s32_h64':4.}
    assert all(v['paired_ratios']==[1.]*8 for v in selected['selected'].values())


def test_rejects_missing_duplicate_wrong_rate_and_nonpositive_updates():
    rows=fake_pilots()
    with pytest.raises(ValueError):study.select_rates(rows[:-1])
    with pytest.raises(ValueError):study.select_rates([*rows[:-1],rows[0]])
    rows=fake_pilots();rows[-1]['config']['adapter_lr_multiplier']=99
    with pytest.raises(ValueError):study.select_rates(rows)
    rows=fake_pilots();rows[-1]['design_diagnostics']['diagnostics'][0]['fixed_token_condition_function_updates']=[{'update_rms':0}]*8
    with pytest.raises(ValueError):study.select_rates(rows)


def test_per_observation_guard_detects_unmatched_seed_even_with_matching_center():
    rows=fake_pilots()
    for r in rows:
        if r['family'].startswith('affine_s32_trial'):
            factor=.25 if r['config']['seed']==90 else 4.
            for e in r['design_diagnostics']['diagnostics']:
                e['fixed_token_condition_function_updates']=[{'update_rms':v['update_rms']*factor} for v in e['fixed_token_condition_function_updates']]
    selected=study.select_rates(rows)
    assert selected['selected']['affine_s32']['geometric_mean_ratio']==pytest.approx(1.)
    assert not selected['selected']['affine_s32']['within_tolerance']
    assert not selected['all_within_tolerance']


def test_valid_guarded_candidate_wins_over_closer_unmatched_center():
    rows=fake_pilots()
    for r in rows:
        if r['family']=='affine_s32_trial1':
            factor=.25 if r['config']['seed']==90 else 4.
        elif r['family']=='affine_s32_trial0':
            factor=2.2
        else:continue
        for e in r['design_diagnostics']['diagnostics']:
            e['fixed_token_condition_function_updates']=[{'update_rms':v['update_rms']*factor} for v in e['fixed_token_condition_function_updates']]
    selection=study.select_rates(rows)
    assert selection['selected']['affine_s32']['adapter_lr_multiplier']==.25
    assert selection['selected']['affine_s32']['within_tolerance']


def test_quality_changes_only_adapter_rate_and_output(tmp_path):
    for family in study.TARGETS:
        baseline=base_config(family,86,tmp_path/'base',None)
        candidate=study.make_config(family,86,.25,tmp_path/'new',None)
        differences={k for k in asdict(candidate) if asdict(candidate)[k]!=asdict(baseline)[k]}
        assert differences=={'adapter_lr_multiplier','output_dir'}
        pilot=study.make_config(family,90,.25,tmp_path/'pilot',None,True)
        assert pilot.max_steps==500 and pilot.lr_schedule_steps==30000
        assert pilot.compile_model and pilot.fused_adamw and pilot.class_dropout_prob==.1
        assert pilot.sample_images==pilot.fid_images==0 and pilot.classifier_run is None
        assert pilot.guidance_scales==pilot.generative_eval_steps==''
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(pilot)))
        assert asdict(load_config(['--override_json',str(p)]))==asdict(pilot)


def test_failed_bounded_calibration_withholds_quality(tmp_path,monkeypatch):
    p={'state':'prepared','classifier_run':None}
    (tmp_path/'protocol.json').write_text(json.dumps(p))
    rows=fake_pilots()
    for r in rows:
        r.pop('fid')
        if r['family'].startswith('affine_s64_trial'):
            for e in r['design_diagnostics']['diagnostics']:
                e['fixed_token_condition_function_updates']=[{'update_rms':v['update_rms']*100} for v in e['fixed_token_condition_function_updates']]
    calls=[]
    def execute(output,protocol,stage,cases):
        calls.append(stage);assert stage=='pilot' and len(cases)==22
        return {'rows':rows}
    monkeypatch.setattr(study,'execute_stage',execute)
    study.run(tmp_path)
    assert calls==['pilot']
    assert json.loads((tmp_path/'protocol.json').read_text())['state']=='calibration_failed'
    assert not (tmp_path/'recipes.json').exists() and not (tmp_path/'quality').exists()


def test_running_reference_cannot_prepare_or_launch(tmp_path,monkeypatch):
    ref=tmp_path/'ref';ref.mkdir();(ref/'protocol.json').write_text(json.dumps({'state':'training'}))
    monkeypatch.setattr(study,'REFERENCE',str(ref))
    with pytest.raises(RuntimeError,match='must finish and pass'):
        study.prepare(tmp_path/'future',launch=True)
    assert not (tmp_path/'future').exists()


def test_waiter_starts_only_after_audit_and_worker_shutdown(tmp_path,monkeypatch):
    ref=tmp_path/'ref';ref.mkdir();(ref/'quality').mkdir();(ref/'quality/supervisor.sock').touch()
    (ref/'protocol.json').write_text(json.dumps({'state':'complete'}));(ref/'report.json').write_text('{}')
    (ref/'audit.json').write_text(json.dumps({'checks':{'ok':True},'report_sha256':study.digest(ref/'report.json')}))
    out=tmp_path/'queue';out.mkdir();dest=out/'study'
    (out/'queue_protocol.json').write_text(json.dumps({'reference_workflow':str(ref),'study_output':str(dest),
        'maximum_wait_hours':8,'source_sha256':{}}))
    calls=[]
    def sleep(seconds):
        assert calls==[] and seconds==20
        (ref/'quality/supervisor.sock').unlink()
    def prepare(path,reference=None):
        assert not (ref/'quality/supervisor.sock').exists();calls.append('prepare');Path(path).mkdir()
    def run(path):
        calls.append('run');(Path(path)/'protocol.json').write_text(json.dumps({'state':'complete'}))
    from pathlib import Path
    monkeypatch.setattr(study.time,'sleep',sleep);monkeypatch.setattr(study,'prepare',prepare);monkeypatch.setattr(study,'run',run)
    study.wait_run(out)
    assert calls==['prepare','run'] and json.loads((out/'queue_protocol.json').read_text())['state']=='complete'


def test_failed_reference_aborts_waiter_without_gpu_work(tmp_path,monkeypatch):
    ref=tmp_path/'ref';ref.mkdir();(ref/'protocol.json').write_text(json.dumps({'state':'failed'}))
    out=tmp_path/'queue';out.mkdir();(out/'queue_protocol.json').write_text(json.dumps({'reference_workflow':str(ref),'maximum_wait_hours':8}))
    monkeypatch.setattr(study,'prepare',lambda *a,**k:pytest.fail('Failed prerequisite must not prepare GPU work'))
    with pytest.raises(RuntimeError,match='Prerequisite failed'):study.wait_run(out)
    assert json.loads((out/'queue_protocol.json').read_text())['state']=='failed'

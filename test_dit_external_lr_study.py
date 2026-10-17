"""Full-horizon selection, equal budgets, frozen confirmation and failure gates."""
from dataclasses import asdict
import json

import pytest
import torch

import dit_external_lr_study as study
from train_dit_cifar10 import load_config, make_model
from summarize_dit_cifar10_screen import common_config


@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_exact_full_horizon_equal_lr_grid_and_matching_offset_capacity(tmp_path):
    cases=study.tuning_cases(tmp_path,None);assert len(cases)==18
    for m in study.METHODS:assert len([c for f,s,c in cases if f.startswith(m+'_lr')])==6
    assert {(f,s) for f,s,c in cases}=={(study.family_key(m,r),s) for m in study.METHODS for r in study.RATES for s in (114,115)}
    for f,s,c in cases:
        assert c.max_steps==c.lr_schedule_steps==30000 and c.warmup_steps==50
        assert c.lr in (2e-4,4e-4,8e-4) and c.guidance_scales=='1,2' and c.fid_images==5000
        assert c.generative_eval_steps=='10000,20000' and c.class_dropout_prob==.1
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(c)));assert asdict(load_config(['--override_json',str(p)]))==asdict(c)
    counts={m:make_model(study.make_config(m,114,tmp_path/m,'/classifier')).statistics()['parameters'] for m in study.METHODS}
    assert counts==study.PARAMETERS and counts['wide']-counts['dplr']==920
    assert set(study.TUNING_SEEDS).isdisjoint(study.QUALITY_SEEDS)


def tuning_rows():
    return [dict(family=f,config=asdict(c),completed_steps=30000,
                 guidance_evaluations={'2.0':{'fid':50.}},ema_validation={'noise_mse':.1})
            for f,s,c in study.tuning_cases(__import__('pathlib').Path('/tmp/tuning'),'/classifier')]


def test_selection_requires_exact_budget_uses_only_final_cfg2_and_has_fixed_ties():
    rows=tuning_rows();assert study.select_rates(rows)['selected_rates']=={m:1. for m in study.METHODS}
    for row in rows:
        row['best_intermediate_fid']=0.
        row['guidance_evaluations']['1.0']={'fid':0. if row['config']['lr']==8e-4 else 100.}
        if row['family']=='plain_lr0.5':row['guidance_evaluations']['2.0']['fid']=49.
    assert study.select_rates(rows)['selected_rates']['plain']==.5
    with pytest.raises(ValueError,match='eighteen'):study.select_rates(rows[:-1])
    with pytest.raises(ValueError,match='eighteen'):study.select_rates([*rows[:-1],rows[0]])
    rows[0]['completed_steps']=1000
    with pytest.raises(ValueError,match='horizon'):study.select_rates(rows)
    rows[0]['completed_steps']=30000;rows[0]['guidance_evaluations']['2.0']['fid']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):study.select_rates(rows)


def test_confirmation_is_twelve_fresh_cases_with_preselected_rates(tmp_path):
    rates={'plain':.5,'wide':1.,'dplr':2.};cases=study.quality_cases(tmp_path,'/classifier',{'selected_rates':rates})
    assert len(cases)==12 and {s for f,s,c in cases}=={116,117,118,119}
    assert len({(f,s) for f,s,c in cases})==12
    for f,s,c in cases:assert c.lr==4e-4*rates[f.split('_lr')[0]] and c.max_steps==c.lr_schedule_steps==30000


def test_summarizer_allows_lr_and_seed_but_rejects_changed_training_budget(tmp_path):
    configs=[asdict(c) for f,s,c in study.tuning_cases(tmp_path,'/classifier')]
    common_config(configs,'tuning')
    with pytest.raises(ValueError,match='settings differ'):common_config(configs,'optimizer_cfg')
    configs[-1]['max_steps']=1200
    with pytest.raises(ValueError,match='settings differ'):common_config(configs,'tuning')


def test_preflights_cover_wide_and_dplr_with_same_real_eval_and_ragged_batch(tmp_path):
    cases=study.preflight_cases(tmp_path,'/classifier');assert len(cases)==2
    assert {f for f,s,c in cases}=={'wide_lr1','dplr_lr1'}
    for f,s,c in cases:
        assert c.max_steps==c.lr_schedule_steps==12 and c.train_limit==328 and c.warmup_steps==2
        assert c.fid_images==20 and c.sampling_steps==2 and c.sample_images==10 and c.guidance_scales=='1,2'


@pytest.mark.parametrize('failure',['preflight','tuning'])
def test_failed_preflight_or_incomplete_tuning_withholds_confirmation(tmp_path,monkeypatch,failure):
    p={'state':'prepared','classifier_run':'/classifier'};(tmp_path/'protocol.json').write_text(json.dumps(p));calls=[]
    def execute(output,p,name,cases,**kwargs):
        calls.append(name)
        if failure=='preflight':raise RuntimeError('bad preflight')
    monkeypatch.setattr(study,'execute_stage',execute);monkeypatch.setattr(study,'audit_stage',lambda *a:{'rows':[]})
    with pytest.raises((RuntimeError,ValueError)):study.run(tmp_path)
    assert 'quality' not in calls and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'


def test_frozen_selection_hash_must_match_before_confirmation(tmp_path):
    s=tmp_path/'selection.json';s.write_text(json.dumps({'selected_rates':{m:1. for m in study.METHODS}}))
    p={'classifier_run':'/classifier','selection_sha256':study.digest(s)}
    assert len(study.stage_cases(tmp_path,'quality',p))==12
    s.write_text(json.dumps({'selected_rates':{'plain':.5,'wide':1.,'dplr':1.}}))
    with pytest.raises(AssertionError):study.stage_cases(tmp_path,'quality',p)

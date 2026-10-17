"""Diagnostic-budget and fail-before-pilot execution guards."""
from dataclasses import asdict
import json

import pytest
import vit_condition_route_study as study
from train_cifar10 import load_config


def test_fixed_six_prefixes_keep_long_schedule_without_tuning(tmp_path):
    cases=study.pilot_cases(tmp_path); assert len(cases)==6
    assert {(f,s,r) for f,s,r,c in cases}=={(f,s,1.) for f in study.FAMILIES for s in (110,111)}
    for f,s,r,c in cases:
        assert c.max_steps==1200 and c.lr_schedule_steps==10560 and c.lr==5e-4 and c.epochs==30
        assert c.routing_diagnostics and c.diagnostic_steps.endswith(',1200') and not c.save_weights and not c.evaluate_test
        p=tmp_path/'config.json';p.write_text(json.dumps(asdict(c)));assert asdict(load_config(['--override_json',str(p)]))==asdict(c)
    with pytest.raises(ValueError,match='No LR'): study.make_config('current',110,tmp_path,.5)


def test_combined_preflight_exercises_both_interventions(tmp_path):
    cases=study.preflight_cases(tmp_path);assert len(cases)==2
    c=next(c for f,s,r,c in cases if f=='combined_preflight')
    assert c.condition_normalization=='rms' and c.condition_gradient_scale==.5 and c.routing_diagnostics
    assert c.max_steps==c.lr_schedule_steps==12 and c.train_limit==328


def test_failed_preflight_withholds_pilots(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared'}));calls=[]
    def execute(output,name,cases,**kwargs):calls.append(name);raise AssertionError('bad gradient reconstruction')
    monkeypatch.setattr(study.common,'execute_stage',execute)
    with pytest.raises(AssertionError,match='bad gradient'):study.run(tmp_path)
    assert calls==['preflight'] and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'

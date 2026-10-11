"""Paired recipes, actual budgets, contrast arithmetic and failure gating."""
from dataclasses import asdict
import json

import pytest

import dit_factor_offset_study as study
from train_dit_cifar10 import load_config, make_model


def test_offset_pairs_change_only_feature_and_output(tmp_path):
    for candidate,control in study.PAIRS:
        for seed in study.SEEDS:
            a=study.make_config(candidate,seed,tmp_path/candidate,None)
            b=study.make_config(control,seed,tmp_path/control,None)
            assert {k for k in asdict(a) if asdict(a)[k]!=asdict(b)[k]}=={'factor_offsets','output_dir'}
            assert a.factor_offsets is True and b.factor_offsets is False
            assert a.max_steps==a.lr_schedule_steps==30000 and a.rank==8
            assert a.adapter_lr_multiplier==1 and a.guidance_scales=='1,2'
            p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(a)))
            assert asdict(load_config(['--override_json',str(p)]))==asdict(a)
    assert set(study.SEEDS).isdisjoint({86,87,90,91,92,93})


def test_actual_model_counts_match_declared_offset_increment(tmp_path):
    import torch
    torch.set_num_threads(2)
    for family in study.FAMILIES:
        m=make_model(study.make_config(family,94,tmp_path/family,None))
        assert m.statistics()['parameters']==study.PARAMETERS[family]
    assert all(study.PARAMETERS[a]-study.PARAMETERS[b]==81920 for a,b in study.PAIRS)


def test_paired_contrasts_preserve_seed_order_and_require_complete_grid():
    rows=[]
    for family in study.FAMILIES:
        for seed in study.SEEDS:
            fid=30+(seed-94)*3-(seed-93 if family.endswith('offsets') else 0)
            point={'fid':fid,'class_agreement':{'generated':{'agreement':.8}}}
            rows.append(dict(family=family,config={'seed':seed},parameters=study.PARAMETERS[family],
                peak_allocated_bytes=1024**2,guidance_evaluations={str(s):point for s in study.SCALES},
                generative_validation_curve=[{'step':s,'fid':fid} for s in study.STEPS],
                validation_curve=[{'step':s,'ema':{'noise_mse':.1}} for s in study.STEPS]))
    result=study.aggregate(list(reversed(rows)))
    for candidate,control in study.PAIRS:
        assert result['paired_offset_differences'][candidate+'_minus_'+control]['2.0']['fid']['values']==[-1,-2]
    with pytest.raises(ValueError):study.aggregate(rows[:-1])
    with pytest.raises(ValueError):study.aggregate([*rows[:-1],rows[0]])


def test_incomplete_reference_withholds_launch(tmp_path,monkeypatch):
    ref=tmp_path/'ref';ref.mkdir();(ref/'protocol.json').write_text(json.dumps({'state':'training'}))
    monkeypatch.setattr(study,'REFERENCE',str(ref))
    with pytest.raises(RuntimeError,match='must complete'):study.prepare(tmp_path/'future',launch=True)
    assert not (tmp_path/'future').exists()


def test_failed_preflight_withholds_quality(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','classifier_run':None}))
    calls=[]
    def fail(output,protocol,stage,cases,**kwargs):
        calls.append(stage)
        assert stage=='preflight' and len(cases)==2
        assert all(cfg.factor_offsets and cfg.max_steps==12 for _,_,cfg in cases)
        raise RuntimeError('failed preflight')
    monkeypatch.setattr(study,'execute_stage',fail)
    with pytest.raises(RuntimeError,match='failed preflight'):study.run(tmp_path)
    assert calls==['preflight']
    assert json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'

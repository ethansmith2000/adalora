"""Guard the longer comparison's schedules, capacity contrast and launch gates."""
from dataclasses import asdict
import json

import pytest

import dit_longer_confirmation as study
from dit_projector_confirmation import make_config as prior_config
from train_dit_cifar10 import learning_rate_scale, load_config, make_model


def test_fresh_60k_recipe_preserves_model_and_optimizer(tmp_path):
    for family in study.FAMILIES:
        old=asdict(prior_config(family,92,tmp_path/'old',None))
        cfg=study.make_config(family,92,tmp_path/'new',None)
        new=asdict(cfg)
        assert {k for k in new if new[k]!=old[k]} == {
            'max_steps','lr_schedule_steps','generative_eval_steps','diagnostic_steps','output_dir'}
        assert cfg.max_steps==cfg.lr_schedule_steps==60000
        assert learning_rate_scale(cfg,30000) > .49
        assert learning_rate_scale(cfg,60000)==pytest.approx(.01)
        assert cfg.adapter_lr_multiplier==1. and cfg.rank==8
        assert cfg.n_basis==(32 if family=='affine_s32' else 16)
        p=tmp_path/(family+'.json');p.write_text(json.dumps(new))
        assert asdict(load_config(['--override_json',str(p)]))==new
    assert set(study.SEEDS).isdisjoint({86,87,90,91})


def test_static_capacity_match_is_real_and_plain_base_matches_adapter(tmp_path):
    import torch
    from benchmark_dit_design import digest_parameters
    torch.set_num_threads(2)
    bases={}
    for family in study.FAMILIES:
        torch.manual_seed(92)
        model=make_model(study.make_config(family,92,tmp_path/family,None))
        assert model.statistics()['parameters']==study.PARAMETERS[family]
        bases[family]=digest_parameters(model)
    assert bases['plain']==bases['affine_s32']
    assert abs(study.PARAMETERS['static_s32']/study.PARAMETERS['affine_s32']-1)<.00039


def synthetic_rows():
    rows=[]
    # Different per-seed control offsets make incorrect pairing observable.
    for family,offset in [('plain',5),('static_s32',2),('affine_s32',0)]:
        for seed in study.SEEDS:
            fid=30+offset+(seed-92)*3
            rows.append(dict(family=family,config={'seed':seed},parameters=study.PARAMETERS[family],
                adapter_parameters=0,forward_matmul_flops_per_image_estimate=100,peak_allocated_bytes=1024**2,
                guidance_evaluations={str(s):{'fid':fid,'class_agreement':{'generated':{'agreement':.8}}} for s in study.SCALES},
                generative_validation_curve=[{'step':s,'fid':fid} for s in study.STEPS],
                validation_curve=[{'step':s,'ema':{'noise_mse':.1}} for s in study.STEPS]))
    return rows


def test_contrasts_pair_seeds_and_reject_incomplete_or_duplicate_grid():
    rows=synthetic_rows();result=study.aggregate(list(reversed(rows)))
    assert result['paired_dplr_differences']['static_s32']['2.0']['fid_dplr_minus_control']['values']==[-2,-2]
    assert result['paired_dplr_differences']['plain']['1.0']['fid_dplr_minus_control']['values']==[-5,-5]
    with pytest.raises(ValueError):study.aggregate(rows[:-1])
    with pytest.raises(ValueError):study.aggregate([*rows[:-1],rows[0]])


def test_incomplete_reference_cannot_launch(tmp_path,monkeypatch):
    ref=tmp_path/'reference';ref.mkdir()
    (ref/'protocol.json').write_text(json.dumps({'state':'training'}))
    monkeypatch.setattr(study,'REFERENCE',str(ref))
    with pytest.raises(RuntimeError,match='must complete'):study.prepare(tmp_path/'future',launch=True)
    assert not (tmp_path/'future').exists()


def test_failed_preflight_withholds_long_training(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','classifier_run':None}))
    calls=[]
    def fail(output,protocol,stage,cases,**kwargs):
        calls.append(stage)
        assert stage=='preflight' and len(cases)==2
        assert all(cfg.max_steps==12 and cfg.lr_schedule_steps==60000 for _,_,cfg in cases)
        raise RuntimeError('preflight failure')
    monkeypatch.setattr(study,'execute_stage',fail)
    with pytest.raises(RuntimeError,match='preflight failure'):study.run(tmp_path)
    assert calls==['preflight']
    assert json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'

"""Capacity matching, recipe isolation and paired contrast arithmetic."""
from dataclasses import asdict
import json
import math

import pytest
import torch

from benchmark_dit_design import digest_parameters
from dit_capacity import FAMILIES, PARAMETERS, SEEDS, SCALES, STEPS, aggregate, make_config
from train_dit_cifar10 import load_config, make_model


def test_parameter_matching_and_shared_base_initialization(tmp_path):
    torch.set_num_threads(2);base={}
    for family in FAMILIES:
        cfg=make_config(family,SEEDS[0],tmp_path/family,None);torch.manual_seed(SEEDS[0]);model=make_model(cfg)
        assert model.statistics()['parameters']==PARAMETERS[family]
        base[family]=digest_parameters(model)
    assert base['plain']==base['dplr']
    assert base['plain_ffn_budget']!=base['plain']
    assert PARAMETERS['plain_ffn_budget']-PARAMETERS['dplr']==2432
    assert abs(PARAMETERS['plain_ffn_budget']/PARAMETERS['dplr']-1)<.00022


def test_capacity_grid_changes_only_declared_architecture_and_seed(tmp_path):
    rows=[]
    for family in FAMILIES:
        for seed in SEEDS:
            cfg=make_config(family,seed,tmp_path/f'{family}_{seed}',None)
            path=tmp_path/'cfg.json';path.write_text(json.dumps(asdict(cfg)))
            assert asdict(load_config(['--override_json',str(path)]))==asdict(cfg)
            assert cfg.width==256 and cfg.depth==8 and cfg.heads==4
            assert cfg.adam_beta1==.9 and cfg.class_dropout_prob==.1 and cfg.guidance_scales=='1,2'
            assert cfg.adapter_gain==1/math.sqrt(8) and cfg.adapter_lr_multiplier==1
            assert cfg.train_limit is cfg.val_limit is None
            assert cfg.max_steps==cfg.lr_schedule_steps==30000 and cfg.warmup_steps==50
            rows.append({k:v for k,v in asdict(cfg).items() if k not in ('variant','mlp_ratio','seed','output_dir')})
    assert len(rows)==6 and all(r==rows[0] for r in rows)


def fake_rows():
    rows=[]
    for family in FAMILIES:
        for i,seed in enumerate(SEEDS):
            def point(scale):
                return dict(fid=60+i*2+scale-(5 if family=='dplr' else 0),
                            class_agreement={'generated':{'agreement':.5+(0.1 if family=='dplr' else 0)}})
            rows.append(dict(family=family,config={'seed':seed},parameters=PARAMETERS[family],adapter_parameters=0,
                             forward_matmul_flops_per_image_estimate=1000,peak_allocated_bytes=1024**2,
                             guidance_evaluations={str(s):point(s) for s in SCALES},
                             generative_validation_curve=[dict(step=s,**point(1)) for s in STEPS],
                             validation_curve=[dict(step=s,ema={'noise_mse':.06}) for s in STEPS]))
    return rows


def test_paired_capacity_contrasts_preserve_direction_and_seed_order():
    result=aggregate(fake_rows())
    for control in ('plain','plain_ffn_budget'):
        for scale in SCALES:
            point=result['paired_dplr_differences'][control][str(scale)]
            assert point['fid_dplr_minus_control']['values']==[-5,-5]
            assert point['agreement_pp_dplr_minus_control']['values']==pytest.approx([10,10])
    rows=fake_rows()
    with pytest.raises(ValueError):aggregate(rows[:-1])
    with pytest.raises(ValueError):aggregate([*rows[:-1],rows[0]])

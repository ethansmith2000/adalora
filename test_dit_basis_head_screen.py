"""Predeclared shared-projector screen selection and budgets."""
from dataclasses import asdict
import math
from pathlib import Path

import pytest
from dit_basis_head_screen import (CAL_SEEDS,CONFIRM_SEEDS,FAMILIES,CAL_FAMILIES,OBSERVATIONS,
                                  aggregate_candidate,next_rates,make_config,update_metric)
from train_dit_cifar10 import load_config


def rows(value):
    return [dict(config=dict(seed=seed),generative_evaluation=dict(fid=float('nan')),
                 design_diagnostics=dict(diagnostics=[dict(step=step,fixed_token_condition_function_updates=[dict(update_rms=value)]*6) for step in OBSERVATIONS])) for seed in CAL_SEEDS]


def test_fresh_seeds_and_exact_screen_grid():
    assert set(CAL_SEEDS).isdisjoint(CONFIRM_SEEDS) and min(CAL_SEEDS)>68
    assert len(FAMILIES)==6 and CAL_FAMILIES==tuple(FAMILIES)
    assert all(v['rank']==8 and v['adapter_gain']==1/math.sqrt(8) for v in FAMILIES.values())


@pytest.mark.parametrize('family',list(FAMILIES))
def test_serialized_calibration_prefix_and_no_large_artifacts(tmp_path,family):
    import json
    for calibration in (True,False):
        cfg=make_config(family,CAL_SEEDS[0] if calibration else CONFIRM_SEEDS[0],1.2,tmp_path/family,'/existing/classifier',calibration)
        path=tmp_path/'config.json';path.write_text(json.dumps(asdict(cfg)))
        # Classifier existence is independently verified in prelaunch; validate calibration here.
        if calibration:assert asdict(load_config(['--override_json',str(path)]))==asdict(cfg)
        assert cfg.max_steps==(500 if calibration else 5000) and cfg.lr_schedule_steps==30000 and cfg.warmup_steps==50
        assert cfg.sample_images==0 and cfg.train_limit is cfg.val_limit is None
        assert cfg.projector_groups==FAMILIES[family]['projector_groups']
        assert cfg.coefficient_activation==FAMILIES[family]['coefficient_activation']
        assert cfg.decoder_groups==FAMILIES[family]['decoder_groups']
        assert cfg.coefficient_hidden==FAMILIES[family]['coefficient_hidden']


def test_update_only_bounded_selection_and_mandatory_response():
    candidate=aggregate_candidate('mlp_silu',0,1.,rows(.5),.5)
    assert candidate['within_tolerance'] and candidate['metric']==.5
    rates={'independent_linear':1.,'mlp_silu':1.};candidates={family:[candidate] for family in rates}
    assert next_rates(0,rates,candidates,.5)=={'mlp_silu':1.}
    assert next_rates(1,rates,candidates,.5)=={}
    low=aggregate_candidate('mlp_silu',0,1.,rows(.001),.5)
    assert next_rates(0,{'mlp_silu':1.},{'mlp_silu':[low]},.5)=={'mlp_silu':16.}
    assert next_rates(3,{'mlp_silu':1.},{'mlp_silu':[low]},.5)=={}
    with pytest.raises(ValueError):aggregate_candidate('mlp_silu',0,1.,rows(.5)[:1],.5)
    with pytest.raises(ValueError):update_metric(rows(0)[0])

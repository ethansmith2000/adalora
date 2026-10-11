"""Regression for the offset study's post-training configuration check."""
from dataclasses import asdict

import pytest

from dit_factor_offset_study import FAMILIES, SEEDS, make_config
from summarize_dit_cifar10_screen import common_config


def configs():
    return [asdict(make_config(family, seed, f"/tmp/{family}_{seed}", None))
            for family in FAMILIES for seed in SEEDS]


def test_offset_grid_is_an_architecture_comparison():
    common = common_config(configs(), "optimizer_cfg")
    assert "factor_offsets" not in common
    assert common["max_steps"] == 30000 and common["guidance_scales"] == "1,2"


@pytest.mark.parametrize("key,value", [("lr", 1e-3), ("adam_beta2", .95),
                                      ("max_steps", 60000), ("split_seed", 999),
                                      ("fid_images", 1000), ("sampling_steps", 20)])
def test_unplanned_training_or_evaluation_difference_is_rejected(key, value):
    rows = configs()
    rows[-1][key] = value
    with pytest.raises(ValueError, match="Training or evaluation settings differ"):
        common_config(rows, "optimizer_cfg")

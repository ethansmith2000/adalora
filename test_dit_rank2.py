"""Matched schedule prefixes and quality-independent full-loader selection."""
from dataclasses import asdict
import json
import math

import pytest

from dit_rank2_confirmation import aggregate_candidate, make_config, OBSERVATIONS, FAMILIES
from train_dit_cifar10 import Config, learning_rate_scale, load_config, main


def test_schedule_default_and_long_prefix():
    cfg=Config(max_steps=30000,warmup_steps=50)
    pilot=Config(max_steps=500,warmup_steps=50,lr_schedule_steps=30000)
    for step in (1,25,50,100,250,500):
        assert learning_rate_scale(pilot,step)==learning_rate_scale(cfg,step)
    assert learning_rate_scale(cfg,30000)==cfg.min_lr_ratio
    for step in (1,50,100,500,30000):
        progress=max(0,step-cfg.warmup_steps)/max(1,cfg.max_steps-cfg.warmup_steps)
        old=step/max(1,cfg.warmup_steps) if step<=cfg.warmup_steps else cfg.min_lr_ratio+(1-cfg.min_lr_ratio)*(1+math.cos(math.pi*progress))/2
        assert learning_rate_scale(cfg,step)==old


@pytest.mark.parametrize("horizon",[499,0,-1,500.5,"30000"])
def test_invalid_schedule_horizon(tmp_path,horizon):
    path=tmp_path/"cfg.json"
    path.write_text(json.dumps(dict(max_steps=500,warmup_steps=50,lr_schedule_steps=horizon)))
    with pytest.raises(ValueError):
        load_config(["--override_json",str(path)])


def test_selection_uses_both_seeds_and_no_quality():
    def row(seed,value,quality):
        return dict(config=dict(seed=seed),generative_evaluation=dict(fid=quality),
                    design_diagnostics=dict(diagnostics=[dict(step=s,fixed_token_condition_function_updates=[dict(update_rms=value)]*6) for s in OBSERVATIONS]))
    a=aggregate_candidate("basis_r2",0,1.,[row(48,1.,999),row(49,3.,-99)],2.)
    b=aggregate_candidate("basis_r2",0,1.,[row(48,1.,-99),row(49,3.,999)],2.)
    assert a==b and a["metric"]==2. and a["within_tolerance"]
    with pytest.raises(ValueError):
        aggregate_candidate("basis_r2",0,1.,[row(48,1.,999)],2.)


def test_frozen_recipes_and_calibration_schedule(tmp_path):
    for family in FAMILIES:
        pilot=make_config(family,48,.75,tmp_path/family,"teacher",True)
        confirm=make_config(family,50,.75,tmp_path/family,"teacher",False)
        assert pilot.lr_schedule_steps==confirm.lr_schedule_steps==30000
        assert not pilot.fid_images and not pilot.sample_images and not pilot.classifier_run
        assert pilot.train_limit is None and pilot.val_limit is None and pilot.num_workers==2
        for key in ("variant","width","rank","adapter_gain","adapter_lr_multiplier","warmup_steps","lr"):
            assert getattr(pilot,key)==getattr(confirm,key)


def test_short_run_has_exact_long_run_prefix(tmp_path):
    events=[]
    for maximum in (12,18):
        output=tmp_path/str(maximum)
        cfg=Config(variant="conditional_bottleneck",width=24,depth=2,patch_size=8,rank=2,
                   max_steps=maximum,lr_schedule_steps=18,warmup_steps=2,batch_size=4,
                   synthetic=True,synthetic_samples=16,val_limit=8,device="cpu",precision="fp32",
                   cpu_threads=2,num_workers=0,log_every=12,eval_every=12,benchmark_warmup_steps=2,
                   sample_images=0,output_dir=str(output),
                   diagnostic_steps="0,1,3,5,10,12"+(",18" if maximum==18 else ""))
        path=tmp_path/f"{maximum}.json"; path.write_text(json.dumps(asdict(cfg)))
        main(["--override_json",str(path)])
        rows=[json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
        for row in rows:
            if row["event"]=="validation":
                for state in ("raw","ema"):
                    row[state].pop("seconds")
        events.append([r for r in rows if r["event"] in ("design_diagnostic","validation") and r["step"]<=12])
    assert events[0]==events[1]
    assert not list(tmp_path.rglob("*.pt"))

"""Quality diagnostics must preserve training and the frozen recipe transfer."""

import json
from dataclasses import asdict

import pytest
import torch

from dit_quality_diagnostics import parse_steps
from train_dit_cifar10 import Config, main


@pytest.mark.parametrize("value", ["1,0", "0,1,1", "-1,2", "0,21", "0,x"])
def test_diagnostic_steps_validate(value):
    with pytest.raises(ValueError):
        parse_steps(value,20)


@pytest.mark.parametrize("variant", ["dplr", "conditional_bottleneck", "conditional_core", "conditional_dplr_core"])
def test_diagnostics_preserve_training_and_ema(tmp_path,variant):
    rows=[]
    for enabled in (False,True):
        folder=tmp_path/("on" if enabled else "off")
        cfg=Config(variant=variant,width=24,depth=2,patch_size=8,rank=2,
                   max_steps=12,warmup_steps=2,batch_size=4,synthetic=True,synthetic_samples=16,
                   val_limit=8,device="cpu",precision="fp32",cpu_threads=2,num_workers=0,
                   log_every=12,eval_every=12,benchmark_warmup_steps=2,sample_images=0,
                   output_dir=str(folder),diagnostic_steps="0,1,3,5,10,12" if enabled else "")
        config=tmp_path/("on.json" if enabled else "off.json")
        config.write_text(json.dumps(asdict(cfg)))
        rows.append(main(["--override_json",str(config)]))
    for key in ("raw_validation","ema_validation"):
        assert rows[0][key]["noise_mse"]==rows[1][key]["noise_mse"]
        assert rows[0][key]["class_prediction_change_rms"]==rows[1][key]["class_prediction_change_rms"]
    diag=rows[1]["design_diagnostics"]
    assert [d["step"] for d in diag["diagnostics"]]==[0,1,3,5,10,12]
    assert diag["adapter_first_nonzero_gradient_observed"]
    assert all(b["delta_rms"]==0 and b["gate_rms"]==0 for b in diag["diagnostics"][0]["branches"])
    assert not list(tmp_path.rglob("*.pt"))

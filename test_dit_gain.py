"""Gain affects the function/gradient, not parameter initialization or identity."""

import copy
import math

import pytest
import torch

from benchmark_dit_design import factor_probe, token_adapter
from benchmark_dit_gain import choose_candidate, proposed_rate, update_metric
from dit_cifar10 import BroadcastLowRank, CIFAR10DiT
from train_dit_cifar10 import load_config, make_model


@pytest.mark.parametrize("variant", ["dplr", "conditional_bottleneck", "static_lora", "exact",
                                      "conditional_core", "conditional_dplr_core"])
def test_fixed_gain_scales_output_and_all_gradients(variant):
    torch.manual_seed(6)
    layer = BroadcastLowRank(8,12,6,variant,2,3).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=.2)
    scaled = copy.deepcopy(layer)
    scaled.gain = .3
    x,c = torch.randn(2,5,8,dtype=torch.double,requires_grad=True), torch.randn(2,6,dtype=torch.double,requires_grad=True)
    xx,cc = x.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
    out, other = layer(x,c), scaled(xx,cc)
    torch.testing.assert_close(other,.3*out)
    upstream = torch.randn_like(out)
    (out*upstream).sum().backward(); (other*upstream).sum().backward()
    for a,b in [(x,xx),(c,cc),*zip(layer.parameters(),scaled.parameters())]:
        if a.grad is None:
            assert b.grad is None  # static LoRA does not use the condition
        else:
            torch.testing.assert_close(b.grad,.3*a.grad)


@pytest.mark.parametrize("variant", ["dplr", "conditional_bottleneck"])
def test_scaled_token_path_and_fixed_probe_use_same_gain(variant):
    torch.set_num_threads(2)
    torch.manual_seed(8)
    model = CIFAR10DiT(width=24,depth=2,patch_size=8,variant=variant,rank=2,adapter_gain=.25)
    with torch.no_grad():
        for block in model.blocks:
            for p in block.adapter.parameters():
                p.normal_(std=.1)
    x,c = torch.randn(2,5,24), torch.randn(2,24)
    for block, probe in zip(model.blocks,factor_probe(model,x,c)):
        expected = block.adapter(x,c)
        torch.testing.assert_close(probe,expected)
        torch.testing.assert_close(token_adapter(x,c[:,None].expand(-1,5,-1),block.adapter),expected)


def test_gain_preserves_seeded_parameters_and_zero_identity():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    a = CIFAR10DiT(width=24,depth=2,patch_size=8,variant="dplr",rank=2)
    torch.manual_seed(42)
    b = CIFAR10DiT(width=24,depth=2,patch_size=8,variant="dplr",rank=2,adapter_gain=1/math.sqrt(2))
    for key,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
    x,c = torch.randn(2,5,24),torch.randn(2,24)
    assert all(block.adapter(x,c).count_nonzero()==0 for block in b.blocks)
    cfg=load_config(["--adapter-gain","0.5","--width","24","--depth","2","--patch-size","8","--variant","dplr"])
    assert make_model(cfg).blocks[0].adapter.gain==.5


@pytest.mark.parametrize("gain", [0,-1,float("inf"),float("nan")])
def test_gain_rejects_dead_or_nonfinite_multiplier(gain):
    with pytest.raises(ValueError,match="gain"):
        BroadcastLowRank(8,12,6,"dplr",gain=gain)
    with pytest.raises(ValueError,match="gain"):
        load_config(["--adapter-gain",str(gain)])


def test_calibration_uses_function_updates_and_bounded_rate():
    row={"diagnostics":[{"step":s,"fixed_token_condition_function_updates":[{"update_rms":x} for x in xs]}
                        for s,xs in [(10,[1,3]),(25,[2,4])]]}
    assert update_metric(row,[10,25])==2.5
    assert proposed_rate(1,.25,1)==4
    assert proposed_rate(16,.001,1)==16
    assert proposed_rate(.1,100,1)==1/16
    candidates=[{"metric":.5,"loss":.01},{"metric":1.1,"loss":100}]
    assert choose_candidate(candidates,1)==candidates[1]


def test_function_probe_restores_matmul_setting_even_on_failure():
    original=torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with pytest.raises(AttributeError):
            factor_probe(None,torch.randn(2,5,8),torch.randn(2,6))
        assert torch.get_float32_matmul_precision()=="high"
        model=CIFAR10DiT(width=24,depth=2,patch_size=8,variant="dplr",rank=2)
        factor_probe(model,torch.randn(2,5,24),torch.randn(2,24))
        assert torch.get_float32_matmul_precision()=="high"
    finally:
        torch.set_float32_matmul_precision(original)

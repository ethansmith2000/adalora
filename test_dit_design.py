"""Input-routing, conditional bottleneck oracle, and diagnostic regression tests."""

import copy

import pytest
import torch

from benchmark_dit_design import BranchDiagnostics, ProjectionProbe, factor_probe, token_adapter
from dit_cifar10 import BroadcastLowRank, CIFAR10DiT, DiTBlock


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_bottleneck_forward_and_all_gradients_match_generated_matrix():
    torch.manual_seed(12)
    layer = BroadcastLowRank(8, 12, 6, "conditional_bottleneck", 2).double()
    with torch.no_grad():
        layer.gains.weight.normal_(std=.1)
    oracle = copy.deepcopy(layer)
    x,c = torch.randn(2,5,8,dtype=torch.double,requires_grad=True), torch.randn(2,6,dtype=torch.double,requires_grad=True)
    xx,cc = x.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
    actual = layer(x,c)
    u = oracle.U.unsqueeze(0) * oracle.gains(cc).unsqueeze(1)
    expected = torch.bmm(xx,u) @ oracle.V
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    upstream=torch.randn_like(actual)
    (actual*upstream).sum().backward(); (expected*upstream).sum().backward()
    for a,b in [(x,xx),(c,cc),*zip(layer.parameters(),oracle.parameters())]:
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-10)


def test_bottleneck_zero_init_opens_gain_head_not_static_factors():
    layer=BroadcastLowRank(8,12,6,"conditional_bottleneck",2)
    x,c=torch.randn(2,5,8),torch.randn(2,6)
    out=layer(x,c)
    assert out.count_nonzero()==0
    (out*torch.randn_like(out)).sum().backward()
    assert layer.gains.weight.grad.norm()>0 and layer.gains.bias.grad.norm()>0
    assert layer.U.grad.count_nonzero()==0 and layer.V.grad.count_nonzero()==0


@pytest.mark.parametrize("mode",["normalized","modulated"])
def test_projection_routes_expected_adapter_input(mode):
    block=DiTBlock(12,3,4,mode)
    block.adapter=BroadcastLowRank(12,48,12,"dplr",2,3)
    with torch.no_grad():
        block.adapter.V.normal_(std=.1)
    x,c=torch.randn(2,5,12),torch.randn(2,12)
    h=block.norm_mlp(x)
    shift,scale=block.modulation(c).unsqueeze(1).chunk(6,-1)[3:5]
    hm=h*(1+scale)+shift
    expected=block.mlp_in(hm)+block.adapter(h if mode=="normalized" else hm,c)
    torch.testing.assert_close(ProjectionProbe(block)(x,c),expected)


def test_paired_input_variants_preserve_initial_parameters():
    torch.manual_seed(42)
    a=CIFAR10DiT(width=24,depth=2,patch_size=8,variant="dplr",rank=2)
    torch.manual_seed(42)
    b=CIFAR10DiT(width=24,depth=2,patch_size=8,variant="dplr",rank=2,adapter_input="modulated")
    for key,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)


def test_branch_hooks_cleanup_and_fixed_factor_probe_cpu():
    model=CIFAR10DiT(width=24,depth=2,patch_size=8,variant="conditional_bottleneck",rank=2)
    hooks=BranchDiagnostics(model)
    model(torch.randn(2,3,32,32),torch.tensor([20,80]),torch.tensor([1,3]))
    rows=hooks.compute(); hooks.close()
    assert all(r["delta_rms"]==0 and r["gate_rms"]==0 for r in rows)
    assert not any(module._forward_hooks for module in model.modules())
    probes=factor_probe(model,torch.randn(2,16,24),torch.randn(2,24))
    assert all(p.device.type=="cpu" and p.count_nonzero()==0 for p in probes)


@pytest.mark.parametrize("variant",["dplr","conditional_bottleneck"])
def test_token_adapter_matches_individual_sample_factors_and_gradients(variant):
    torch.manual_seed(7)
    adapter=BroadcastLowRank(8,12,6,variant,2,3).double()
    with torch.no_grad():
        adapter.V.normal_(std=.1)
        if variant=="conditional_bottleneck":
            adapter.gains.weight.normal_(std=.1)
    oracle=copy.deepcopy(adapter)
    x,c=torch.randn(2,5,8,dtype=torch.double,requires_grad=True),torch.randn(2,5,6,dtype=torch.double,requires_grad=True)
    xx,cc=x.detach().clone().requires_grad_(),c.detach().clone().requires_grad_()
    actual=token_adapter(x,c,adapter)
    expected=oracle(xx.reshape(10,1,8),cc.reshape(10,6)).reshape(2,5,12)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    upstream=torch.randn_like(actual)
    (actual*upstream).sum().backward(); (expected*upstream).sum().backward()
    for a,b in [(x,xx),(c,cc),*zip(adapter.parameters(),oracle.parameters())]:
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-10)

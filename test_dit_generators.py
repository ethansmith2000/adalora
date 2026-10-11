"""Dense matrix/gradient oracles and zero-init checks for conditional cores."""

import copy
from dataclasses import asdict
import json

import pytest
import torch

from benchmark_dit_design import token_adapter
from benchmark_dit_generators import generator_cases
from dit_cifar10 import BroadcastLowRank, CIFAR10DiT
from train_dit_cifar10 import Config, load_config, main, make_model


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("variant", ["conditional_core", "conditional_dplr_core"])
@pytest.mark.parametrize("token_condition", [False, True])
def test_core_forward_and_gradients_against_full_matrix(variant, token_condition):
    torch.manual_seed(61)
    layer = BroadcastLowRank(8,12,6,variant,rank=4,gain=.3,core_rank=2).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=.2)
    oracle = copy.deepcopy(layer)
    x = torch.randn(2,5,8,dtype=torch.double,requires_grad=True)
    c = torch.randn(*((2,5,6) if token_condition else (2,6)),dtype=torch.double,requires_grad=True)
    xx,cc = x.detach().clone().requires_grad_(),c.detach().clone().requires_grad_()
    actual = token_adapter(x,c,layer) if token_condition else layer(x,c)
    if variant == "conditional_core":
        core = oracle.core(cc).unflatten(-1,(4,4))
    else:
        core = torch.diag_embed(oracle.gains(cc)) + (oracle.core_left(cc).unflatten(-1,(4,2)) @ oracle.core_right(cc).unflatten(-1,(2,4)))
    matrix = oracle.U @ core @ oracle.V
    expected = ((xx.unsqueeze(-2) @ matrix).squeeze(-2) if token_condition else xx @ matrix) * oracle.gain
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    upstream = torch.randn_like(actual)
    (actual*upstream).sum().backward()
    (expected*upstream).sum().backward()
    for first,second in [(x,xx),(c,cc),*zip(layer.parameters(),oracle.parameters())]:
        torch.testing.assert_close(first.grad,second.grad,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize("variant", ["conditional_core", "conditional_dplr_core"])
def test_zero_core_opens_correct_heads_before_static_factors(variant):
    layer = BroadcastLowRank(8,12,6,variant,rank=4,core_rank=2)
    x,c = torch.randn(2,5,8),torch.randn(2,6)
    out = layer(x,c)
    assert out.count_nonzero() == 0
    (out*torch.randn_like(out)).sum().backward()
    assert layer.U.grad.count_nonzero() == layer.V.grad.count_nonzero() == 0
    if variant == "conditional_core":
        assert layer.core.weight.grad.norm() > 0 and layer.core.bias.grad.norm() > 0
    else:
        assert layer.gains.weight.grad.norm() > 0 and layer.gains.bias.grad.norm() > 0
        assert layer.core_right.weight.grad.norm() > 0 and layer.core_right.bias.grad.norm() > 0
        assert layer.core_left.weight.grad.count_nonzero() == layer.core_left.bias.grad.count_nonzero() == 0


def test_real_width_generator_budgets_and_ffn_control():
    stats = {}
    for spec in generator_cases():
        model = CIFAR10DiT(**{key:value for key,value in spec.items() if key != "name"})
        stats[spec["name"]] = model.statistics()
    assert stats["ffn_r2_budget"]["parameters"] - stats["basis_r2_s16"]["parameters"] == 384
    assert stats["core_r8"]["adapter_parameters"] == 120192
    assert stats["dplr_core_r8_p2"]["adapter_parameters"] == 92400
    assert stats["basis_r16_q32"]["adapter_parameters"] == 566856
    assert [spec["n_basis"] for spec in generator_cases() if spec["name"] in ("basis_r8_q4","basis_r8_q16","basis_r8_q32")] == [48,12,6]


@pytest.mark.parametrize("variant", ["conditional_core", "conditional_dplr_core"])
def test_core_trainer_cli_and_opt_in_diagnostics_no_weights(tmp_path, variant):
    cfg = Config(variant=variant,width=24,depth=2,patch_size=8,rank=4,core_rank=2,
                 max_steps=10,warmup_steps=2,batch_size=4,num_workers=0,cpu_threads=2,
                 synthetic=True,synthetic_samples=16,val_limit=8,device="cpu",precision="fp32",
                 sample_images=0,log_every=10,eval_every=10,benchmark_warmup_steps=2,
                 diagnostic_steps="0,1,3,5,10",output_dir=str(tmp_path/variant))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(cfg)))
    loaded = load_config(["--override_json",str(path)])
    assert make_model(loaded).blocks[0].adapter.core_rank == 2
    result = main(["--override_json",str(path)])
    assert result["design_diagnostics"]["adapter_first_nonzero_gradient_observed"]
    assert result["completed_steps"] == 10 and not result["weights_saved"]
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("factor_rank", [0,5])
def test_invalid_core_rank_rejected(factor_rank):
    with pytest.raises(ValueError,match="Core factor rank|core_rank"):
        load_config(["--variant","conditional_dplr_core","--rank","4","--core-rank",str(factor_rank)])

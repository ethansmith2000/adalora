"""Numerical broadcast oracles and end-to-end diffusion integration checks."""

import copy
import json

import pytest
import torch

from condlr import BasisMixLowRank, ExactCondLowRank
from dit_cifar10 import BroadcastLowRank, CIFAR10DiT, VARIANTS
from train_dit_cifar10 import (Config, denoising_loss, main, make_noise_scheduler, sample_batches, update_ema)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("variant", ["exact", "dplr"])
def test_broadcast_matches_token_oracle_and_gradients(variant):
    torch.manual_seed(9)
    layer = BroadcastLowRank(8, 12, 6, variant, rank=2, n_basis=3).double()
    if variant == "exact":
        oracle = ExactCondLowRank(8, 12, 6, rank=2, chunk=4).double()
        with torch.no_grad():
            layer.B.normal_(std=0.1)
            oracle.A.copy_(layer.A)
            oracle.B.copy_(layer.B)
        parameter_pairs = [(layer.A, oracle.A), (layer.B, oracle.B)]
    else:
        oracle = BasisMixLowRank(8, 12, 6, rank=2, n_basis=3, mix="activations").double()
        with torch.no_grad():
            layer.V.normal_(std=0.1)
            oracle.U.copy_(layer.U)
            oracle.V.copy_(layer.V)
            oracle.to_coef.load_state_dict(layer.coefficients.state_dict())
        parameter_pairs = [(layer.U, oracle.U), (layer.V, oracle.V),
                           (layer.coefficients.weight, oracle.to_coef.weight),
                           (layer.coefficients.bias, oracle.to_coef.bias)]
    with torch.no_grad():
        oracle.base.weight.zero_()
        oracle.base.bias.zero_()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    c = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    xx, cc = x.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
    y = layer(x, c)
    reference = oracle(xx, cc.unsqueeze(1).expand(-1, 5, -1))
    torch.testing.assert_close(y, reference, rtol=1e-10, atol=1e-10)
    upstream = torch.randn_like(y)
    (y * upstream).sum().backward()
    (reference * upstream).sum().backward()
    for first, second in [(x, xx), (c, cc)] + parameter_pairs:
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-9, atol=1e-9)
    u, v = layer.factors(c)
    assert u.shape == (2, 8, 2) and v.shape == (2, 2, 12)


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_bf16_float32_weights_have_finite_nonzero_gradients(variant):
    layer = BroadcastLowRank(16, 24, 16, variant, rank=2, n_basis=4)
    with torch.no_grad():
        (layer.B if variant == "exact" else layer.V).normal_(std=0.05)
        if variant == "conditional_bottleneck":
            layer.gains.weight.normal_(std=0.05)
        elif variant == "conditional_core":
            layer.core.weight.normal_(std=0.05)
        elif variant == "conditional_dplr_core":
            layer.gains.weight.normal_(std=0.05)
            layer.core_right.weight.normal_(std=0.05)
    x = torch.randn(2, 9, 16, requires_grad=True)
    c = torch.randn(2, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(x, c)
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert x.grad.abs().sum() > 0
    if variant != "static_lora":
        assert c.grad.abs().sum() > 0
    for p in layer.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_variants_preserve_base_initialization(variant):
    torch.manual_seed(12)
    plain = CIFAR10DiT(width=24, depth=2, heads=3, patch_size=8)
    torch.manual_seed(12)
    modified = CIFAR10DiT(width=24, depth=2, heads=3, patch_size=8, variant=variant, rank=2, n_basis=3)
    actual = {k: v for k, v in modified.state_dict().items() if ".adapter." not in k}
    assert actual.keys() == plain.state_dict().keys()
    for name, value in plain.state_dict().items():
        torch.testing.assert_close(value, actual[name], rtol=0, atol=0)
    for block in modified.blocks:
        x, c = torch.randn(2, 16, 24), torch.randn(2, 24)
        torch.testing.assert_close(block.adapter(x, c), torch.zeros(2, 16, 96), rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_zero_initialization_wakes_up_conditioning_and_adapters(variant):
    torch.manual_seed(42)
    model = CIFAR10DiT(width=24, depth=2, heads=3, patch_size=8, variant=variant, rank=2, n_basis=3)
    images, labels, timesteps = torch.randn(2, 3, 32, 32), torch.tensor([1, 8]), torch.tensor([20, 700])
    assert model(images, timesteps, labels).abs().sum() == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.randn_like(images)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        (model(images, timesteps, labels) - target).square().mean().backward()
        optimizer.step()
    assert model.class_embedding.weight.grad.abs().sum() > 0
    assert model.time_embedding[0].weight.grad.abs().sum() > 0
    if variant != "adaln_zero":
        for block in model.blocks:
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in block.adapter.parameters())
    prediction = model(images, timesteps, labels)
    assert (prediction - model(images, timesteps, (labels + 1) % 10)).abs().sum() > 0
    assert (prediction - model(images, timesteps + 50, labels)).abs().sum() > 0


def test_ddpm_loss_is_zero_for_known_noise_and_ema_is_in_memory():
    cfg = Config(num_train_timesteps=10, sampling_steps=3)
    scheduler = make_noise_scheduler(cfg)
    images, noise = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    timesteps, labels = torch.tensor([0, 9]), torch.tensor([0, 1])
    class KnownNoise(torch.nn.Module):
        def forward(self, noisy, timestep, label):
            assert not torch.equal(noisy, images)
            return noise
    assert denoising_loss(KnownNoise(), scheduler, images, labels, noise, timesteps) == 0
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        model.weight.fill_(2)
    update_ema(ema, model, 0.9)
    torch.testing.assert_close(ema.weight, torch.tensor([[0.2]]))
    assert ema.weight.data_ptr() != model.weight.data_ptr()


def test_sampling_reproducible_balanced_and_bounded():
    class ToyPredictor(torch.nn.Module):
        def forward(self, images, timestep, labels):
            return images * 0.1 + labels[:, None, None, None] * 0.01
    cfg = Config(num_train_timesteps=10, sampling_steps=3, precision="fp32")
    scheduler = make_noise_scheduler(cfg)
    model = ToyPredictor()
    first = list(sample_batches(model, scheduler, cfg, torch.device("cpu"), 14, 5))
    second = list(sample_batches(model, scheduler, cfg, torch.device("cpu"), 14, 5))
    for (a, labels), (b, _) in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert torch.cat([labels for _, labels in first]).tolist() == [i % 10 for i in range(14)]


def test_trainer_cpu_synthetic_records_and_no_weights(tmp_path):
    result = main(["--synthetic", "--device", "cpu", "--precision", "fp32", "--variant", "exact",
                   "--width", "24", "--depth", "1", "--heads", "3", "--patch-size", "8", "--rank", "2",
                   "--max-steps", "8", "--warmup-steps", "2", "--batch-size", "4", "--num-workers", "0",
                   "--synthetic-samples", "16", "--val-limit", "8", "--log-every", "2", "--eval-every", "4",
                   "--sample-images", "10", "--sample-batch-size", "5", "--sampling-steps", "3",
                   "--num-train-timesteps", "10", "--output-dir", str(tmp_path)])
    assert result["completed_steps"] == 8 and result["processed_images"] == 32
    assert result["ema_validation"]["examples"] == 8 and not result["weights_saved"]
    assert (tmp_path / "samples_ema.png").is_file()
    events = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events if e["event"] == "validation"] == [0, 4, 8]
    assert not list(tmp_path.glob("*.pt"))

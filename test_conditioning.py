"""Research-critical checks: gradient parity, initialization and AMP integration."""

import json
from pathlib import Path

import pytest
import torch

from conditioning import ConditionedLinear, VARIANTS
from condlr import ExactCondLowRank, NaiveCondLowRank
from train_cifar10 import Config, build_model, main, stratified_split


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def test_exact_shared_condition_gradient_and_tail_chunk():
    torch.manual_seed(4)
    m = ExactCondLowRank(5, 7, 5, rank=3, chunk=4).double()
    with torch.no_grad():
        m.B.normal_(std=0.1)
    x = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
    params = list(m.parameters())
    exact = m(x, x)
    grads = torch.autograd.grad(exact.square().mean(), [x, *params])
    oracle = NaiveCondLowRank(m)(x, x)
    ref = torch.autograd.grad(oracle.square().mean(), [x, *params])
    torch.testing.assert_close(exact, oracle)
    for actual, expected in zip(grads, ref):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_bf16_activations_fp32_weights(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(5)
    m = ExactCondLowRank(16, 24, 16, rank=2, chunk=7).to(device)
    with torch.no_grad():
        m.B.normal_(std=0.01)
    x = torch.randn(2, 9, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    c = torch.randn_like(x, requires_grad=True)
    inputs = [x, c, *m.parameters()]
    with torch.autocast(device, dtype=torch.bfloat16):
        y = m(x, c)
        loss = y.float().square().mean()
    grads = torch.autograd.grad(loss, inputs)
    with torch.autocast(device, dtype=torch.bfloat16):
        reference = NaiveCondLowRank(m)(x, c)
        loss_ref = reference.float().square().mean()
    refs = torch.autograd.grad(loss_ref, inputs)
    torch.testing.assert_close(y.float(), reference.float(), atol=0.008, rtol=0.03)
    for actual, expected in zip(grads, refs):
        assert torch.isfinite(actual).all()
        # Different contraction orders round differently in bf16. Bound global
        # error rather than demanding small relative error at near-zero entries.
        error = (actual.float() - expected.float()).norm()
        assert error <= 0.04 * expected.float().norm() + 1e-5


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_projection_starts_at_base_and_conditional_branch_learns(variant):
    torch.manual_seed(8)
    base = torch.nn.Linear(8, 16)
    layer = ConditionedLinear(base, variant, rank=2, n_basis=4, chunk=3)
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(layer(x, x), base(x))
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    # First update wakes the zero output side; the second reaches the generator.
    for _ in range(2):
        optimizer.zero_grad()
        layer(x, x).square().mean().backward()
        optimizer.step()
    if variant == "exact":
        assert layer.transform.A.grad.abs().sum() > 0
    elif variant in ("basis", "dplr"):
        lowrank = layer.transform.lowrank if variant == "dplr" else layer.transform
        assert lowrank.to_coef.weight.grad.abs().sum() > 0
    elif variant == "static_lora_diagonal":
        assert layer.U.grad.abs().sum() > 0
    else:
        assert layer.to_affine.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_vit_identical_base_initialization(variant):
    cfg = Config(width=24, heads=3, depth=2, rank=2, n_basis=4, placement="both")
    torch.manual_seed(12)
    plain = build_model(cfg)
    cfg.variant = variant
    torch.manual_seed(12)
    conditioned = build_model(cfg)
    # Avoid the zero classifier making this an uninformative equality test.
    with torch.no_grad():
        plain.head.weight.normal_(std=0.1)
        conditioned.head.load_state_dict(plain.head.state_dict())
    images = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(conditioned(images), plain(images), rtol=1e-5, atol=1e-6)


def test_stratified_split_has_no_leakage():
    labels = [i % 10 for i in range(100)]
    train, val = stratified_split(labels, 0.2, 42)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(100))
    assert [sum(labels[i] == c for i in val) for c in range(10)] == [2] * 10
    assert (train, val) == stratified_split(labels, 0.2, 42)


@pytest.mark.parametrize("variant", ["exact", "basis", "dplr"])
def test_checkpointed_vit_learns_with_self_condition(variant):
    cfg = Config(width=24, heads=3, depth=1, rank=2, n_basis=4, variant=variant,
                 gradient_checkpointing=True, placement="both")
    model = build_model(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    images, labels = torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = model(images, labels)
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)
    projection = model.blocks[0].mlp.fc1.transform
    parameter = projection.B if variant == "exact" else projection.lowrank.V if variant == "dplr" else projection.V
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_trainer_compact_records_without_weights(tmp_path):
    main(["--synthetic", "--device", "cpu", "--width", "24", "--depth", "1", "--heads", "3",
          "--variant", "exact", "--chunk", "7", "--max-steps", "3", "--batch-size", "4",
          "--num-workers", "0", "--synthetic-samples", "12", "--benchmark-warmup-steps", "1",
          "--log-every", "2", "--output-dir", str(tmp_path)])
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["completed_steps"] == 3
    assert results["benchmark_images"] == 8
    assert results["train_images_per_second"] > 0
    assert results["test"] is None
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["weights_saved"] is False
    assert not any(p.suffix in (".pt", ".bin", ".safetensors") for p in Path(tmp_path).iterdir())

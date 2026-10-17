#!/usr/bin/env python3
"""Shared-stream DiT design engineering, with live diagnostics and no weights."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from dit_cifar10 import CIFAR10DiT
from train_dit_cifar10 import Config, build_datasets, evaluate, make_noise_scheduler, provenance, write_json
from train_cifar10 import make_loader, synchronize


CASES = [
    ("adaln", "adaln_zero", 8, "normalized"),
    ("dplr_r8_reference", "dplr", 8, "normalized"),
    ("basis_r2_normalized", "dplr", 2, "normalized"),
    ("basis_r2_modulated", "dplr", 2, "modulated"),
    ("basis_r4_normalized", "dplr", 4, "normalized"),
    ("basis_r4_modulated", "dplr", 4, "modulated"),
    ("bottleneck_r2", "conditional_bottleneck", 2, "normalized"),
    ("bottleneck_r4", "conditional_bottleneck", 4, "normalized"),
]


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def digest_parameters(model, adapters=False):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if (".adapter." in name) == adapters:
            digest.update(name.encode())
            digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def rms(tensor):
    return tensor.float().square().mean().sqrt().item()


def gradient_groups(model):
    groups = {name: [] for name in ("base", "adapter_input", "adapter_output", "adapter_generator")}
    for name, p in model.named_parameters():
        key = "base"
        if ".adapter." in name:
            key = "adapter_input" if name.endswith((".U", ".U_offset", ".U_channel", ".U_basis", ".U_rank", ".U_core", ".U_static", ".U_scale_basis")) else "adapter_output" if name.endswith((".V", ".V_offset", ".V_channel", ".V_basis", ".V_rank", ".V_core", ".V_static", ".V_scale_basis")) else "adapter_generator"
            if name.endswith((".input_basis", ".input_offset", ".middle_basis", ".middle_offset")):
                key = "adapter_input"
            elif name.endswith((".output_basis", ".output_offset")):
                key = "adapter_output"
        groups[key].append((name, p))
    return groups


def gradient_record(groups):
    result = {}
    for name, params in groups.items():
        if not params:
            continue
        square, elements, nonzero = 0.0, 0, []
        param_square = 0.0
        for key, p in params:
            param_square += p.detach().float().square().sum().item()
            if p.grad is not None:
                norm2 = p.grad.detach().float().square().sum().item()
                if not math.isfinite(norm2):
                    raise ValueError("Nonfinite parameter gradient")
                square += norm2
                elements += p.numel()
                if norm2 > 0:
                    nonzero.append(key)
        result[name] = {"gradient_l2": math.sqrt(square), "gradient_rms": math.sqrt(square / max(elements, 1)),
                        "parameter_l2": math.sqrt(param_square), "parameter_tensors": len(params),
                        "nonzero_gradient_tensors": nonzero}
    return result


class BranchDiagnostics:
    """Hooks active only for an untimed fixed-image diagnostic forward."""
    def __init__(self, model):
        self.states = [dict() for _ in model.blocks]
        self.handles = []
        for block, state in zip(model.blocks, self.states):
            def norm_hook(module, args, out, state=state):
                state.update(tokens=args[0].detach(), h=out.detach())
            def mod_hook(module, args, out, state=state):
                shift, scale, gate = out.chunk(6, -1)[3:]
                state.update(shift=shift.detach(), scale=scale.detach(), gate=gate.detach())
            def base_hook(module, args, out, state=state, block=block):
                if getattr(block.adapter, "is_square", False):
                    h_mod = state["h"]*(1+state["scale"][:,None])+state["shift"][:,None]
                    base = F.linear(h_mod, module.weight, module.bias).detach()
                    delta = F.linear(state["square_delta"], module.weight, None).detach()
                    state.update(base=base, delta=delta, h_mod=h_mod,
                                 actual_pre_activation=out.detach(),
                                 projection_rounding_residual_rms=rms(out.float()-base.float()-delta.float()))
                else:
                    state.update(base=out.detach(), h_mod=args[0].detach())
            def delta_hook(module, args, out, state=state, block=block):
                if getattr(block.adapter, "is_square", False):
                    state.update(square_delta=out.detach(), adapter_input=state["h"])
                    return
                inp = (state["h"] if block.adapter_input == "normalized" else state["h_mod"]) if hasattr(block, "adapter_delta") else args[0].detach()
                state.update(delta=out.detach(), adapter_input=inp)
            def out_hook(module, args, out, state=state):
                state["out"] = out.detach()
            for module, hook in [(block.norm_mlp, norm_hook), (block.modulation, mod_hook),
                                 (block.mlp_in, base_hook), (block.mlp_out, out_hook)]:
                self.handles.append(module.register_forward_hook(hook))
            if block.adapter is not None:
                self.handles.append(getattr(block, "adapter_delta", block.adapter).register_forward_hook(delta_hook))

    def compute(self):
        records = []
        for state in self.states:
            base = state["base"].float()
            delta = state.get("delta", torch.zeros_like(base)).float()
            bc, dc = base - base.mean(), delta - delta.mean()
            denom = bc.square().sum().sqrt() * dc.square().sum().sqrt()
            records.append({"normalized_input_rms": rms(state["h"]), "modulated_input_rms": rms(state["h_mod"]),
                            "adapter_input_rms": rms(state.get("adapter_input", state["h"])),
                            "base_rms": rms(base), "delta_rms": rms(delta),
                            "pre_activation_rms": rms(state.get("actual_pre_activation", base + delta)),
                            "combined_to_base_rms": rms(state.get("actual_pre_activation", base + delta)) / max(rms(base), 1e-12),
                            "delta_mean": delta.mean().item(),
                            "delta_to_base_rms": rms(delta) / max(rms(base), 1e-12),
                            "base_delta_correlation": (bc * dc).sum().item() / denom.item() if denom.item() > 0 else None,
                            "scale_rms": rms(state["scale"]), "shift_rms": rms(state["shift"]),
                            "gate_rms": rms(state["gate"]),
                            "gated_residual_to_input_rms": rms(state["gate"][:, None] * state["out"]) / max(rms(state["tokens"]), 1e-12)})
            if "square_delta" in state:
                records[-1].update(square_feature_delta_rms=rms(state["square_delta"]),
                                   projection_rounding_residual_rms=state["projection_rounding_residual_rms"])
        return records

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.states.clear()


@torch.no_grad()
def branch_snapshot(model, noisy, steps, labels, device, precision):
    hooks = BranchDiagnostics(model)
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            model(noisy, steps, labels)
        return hooks.compute()
    finally:
        hooks.close()


@torch.no_grad()
def factor_probe(model, h, c):
    """Hold both normalized token and condition fixed across optimizer steps."""
    precision = torch.get_float32_matmul_precision()
    try:
        # TF32 rounding can dominate small late-stage function differences.
        torch.set_float32_matmul_precision("highest")
        return _factor_probe(model, h, c)
    finally:
        torch.set_float32_matmul_precision(precision)


def _factor_probe(model, h, c):
    deltas = []
    for block in model.blocks:
        if block.adapter is None:
            deltas.append(torch.zeros(*h.shape[:-1], block.mlp_in.out_features))
            continue
        inp = h
        if block.adapter_input == "modulated":
            shift, scale = block.modulation(c).chunk(6, -1)[3:5]
            inp = h * (1 + scale[:, None]) + shift[:, None]
        delta = block.adapter(inp, c)
        if getattr(block.adapter, "is_square", False):
            delta = F.linear(delta, block.mlp_in.weight, None)
        deltas.append(delta.float().cpu())
    return deltas


class ProjectionProbe(nn.Module):
    """Actual trained MLP input path, excluding GELU/output/gate/attention."""
    def __init__(self, block):
        super().__init__()
        self.norm = block.norm_mlp
        self.modulation = block.modulation
        self.base = block.mlp_in
        self.adapter = block.adapter
        self.adapter_input = block.adapter_input

    def forward(self, x, c):
        modulation = self.modulation(c)
        if c.ndim == 2:
            modulation = modulation.unsqueeze(1)
        shift, scale = modulation.chunk(6, -1)[3:5]
        h = self.norm(x)
        h_mod = h * (1 + scale) + shift
        if getattr(self.adapter, "is_square", False):
            if c.ndim != 2:
                raise ValueError("Square operator token execution is not implemented")
            return self.base(h_mod + self.adapter(h,c))
        out = self.base(h_mod)
        if self.adapter is not None:
            inp = h if self.adapter_input == "normalized" else h_mod
            out = out + (self.adapter(inp, c) if c.ndim == 2 else token_adapter(inp, c, self.adapter))
        return out


def token_adapter(x, c, adapter):
    """Per-token contraction of the same factors, without materializing U/V."""
    if adapter.variant == "conditional_bottleneck":
        delta = ((x @ adapter.U) * adapter.gains(c)) @ adapter.V
        return delta if adapter.gain == 1.0 else delta * adapter.gain
    if adapter.variant in ("conditional_core", "conditional_dplr_core"):
        z = x @ adapter.U
        if adapter.variant == "conditional_core":
            delta = (z.unsqueeze(-2) @ adapter.core_components(c)).squeeze(-2) @ adapter.V
        else:
            gains, left, right = adapter.core_components(c)
            mixed = ((z.unsqueeze(-2) @ left) @ right).squeeze(-2)
            delta = (z * gains + mixed) @ adapter.V
        return delta if adapter.gain == 1.0 else delta * adapter.gain
    if adapter.variant != "dplr":
        raise ValueError("Token engineering probe supports basis or conditional bottleneck")
    a,b = adapter.coefficient_values(c).chunk(2,-1)
    p = (x @ adapter.U).unflatten(-1,(adapter.n_basis,adapter.rank))
    z = (p*a.unsqueeze(-1)).sum(-2)
    g = (z.unsqueeze(-2)*b.unsqueeze(-1)).flatten(-2)
    delta = g @ adapter.V
    return delta if adapter.gain == 1.0 else delta * adapter.gain


def microbenchmark(block, settings, device, condition_mode="sample"):
    probe = ProjectionProbe(copy.deepcopy(block)).to(device)
    b, t, d = settings["batch_size"], (32 // settings["patch_size"]) ** 2, settings["width"]
    rng = torch.Generator(device=device).manual_seed(2040)
    x = torch.randn(b, t, d, generator=rng, device=device, requires_grad=True)
    shape = (b,d) if condition_mode == "sample" else (b,t,d)
    c = F.layer_norm(torch.randn(*shape, generator=rng, device=device), (d,)).requires_grad_()
    upstream = torch.randn(b, t, probe.base.out_features, generator=rng, device=device)

    def step(module):
        probe.zero_grad(set_to_none=True)
        x.grad, c.grad = None, None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=settings["precision"] == "bf16"):
            out = module(x, c)
        (out.float() * upstream).mean().backward()
        return out

    modes, reference = [], None
    for mode in ("eager", "compiled") if settings["compile_layer"] else ("eager",):
        module = probe if mode == "eager" else torch.compile(probe, fullgraph=True)
        synchronize(device)
        cold_start = time.perf_counter()
        out = step(module)
        synchronize(device)
        cold_seconds = time.perf_counter() - cold_start
        grads = [x.grad.detach().clone(), c.grad.detach().clone()] + [p.grad.detach().clone() for p in probe.parameters()]
        parity = None
        if reference is None:
            reference = (out.detach().clone(), grads)
        else:
            errors = []
            for actual, expected in [(out.detach(), reference[0]), *zip(grads, reference[1])]:
                error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
                errors.append(error.item())
            parity = max(errors)
            if parity > (0.03 if settings["precision"] == "bf16" else 2e-4):
                raise ValueError(f"Compiled output/gradient relative-L2 discrepancy: {parity}")
        del out, grads
        for _ in range(10):
            step(module)
        synchronize(device)
        timings = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(settings["layer_repeats"]):
            synchronize(device)
            start = time.perf_counter()
            for _ in range(settings["layer_steps"]):
                step(module)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000 / settings["layer_steps"])
        modes.append({"mode": mode, "cold_first_forward_backward_seconds": cold_seconds,
                      "milliseconds_per_forward_backward": statistics.median(timings), "repeat_ms": timings,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      "compiled_max_relative_l2_output_gradient_error": parity})

    # Saved-storage accounting is eager-only; compile may save/recompute differently.
    probe.zero_grad(set_to_none=True)
    x.grad, c.grad = None, None
    original = {p.untyped_storage().data_ptr() for p in probe.parameters()}
    saved = {}
    def pack(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in original:
            saved[storage.data_ptr()] = storage.nbytes()
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=settings["precision"] == "bf16"):
            value = probe(x, c)
    result = {"scope": "trained first-block normalized/modulated MLP input projection only; no optimizer step",
              "condition_mode": condition_mode,
              "token_probe_note": "per-token extension of the trained weights with synthetic normalized conditions; not a trained token-conditioned model" if condition_mode == "token" else None,
              "parameters": sum(p.numel() for p in probe.parameters()), "modes": modes,
              "eager_saved_nonparameter_storage_bytes": sum(saved.values()),
              "saved_storage_note": "unique backing storages; includes inputs/autocast copies; excludes original parameters",
              "precision": settings["precision"], "batch_size": b, "tokens": t}
    del value, module, probe, reference, x, c, upstream
    if settings["compile_layer"]:
        torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_case(case, settings, images_bank, labels_bank, validation, scheduler, fixed, output, runtime):
    specification = case if isinstance(case, dict) else dict(zip(("name", "variant", "rank", "adapter_input"), case))
    name, variant, rank, adapter_input = (specification[k] for k in ("name", "variant", "rank", "adapter_input"))
    n_basis = specification.get("n_basis", 16)
    core_rank = specification.get("core_rank", 2)
    mlp_ratio = specification.get("mlp_ratio", 4.)
    gain = specification.get("adapter_gain", settings.get("adapter_gain", 1.0))
    adapter_lr = settings.get("adapter_lr_multiplier", 1.0)
    if not math.isfinite(adapter_lr) or adapter_lr <= 0:
        raise ValueError("Adapter LR multiplier must be finite and positive")
    folder = output / name
    folder.mkdir()
    device = torch.device(settings["device"])
    torch.manual_seed(42)
    model = CIFAR10DiT(settings["width"], settings["depth"], settings["heads"], settings["patch_size"],
                      variant=variant, rank=rank, n_basis=n_basis, adapter_input=adapter_input, adapter_gain=gain,
                      mlp_ratio=mlp_ratio, core_rank=core_rank)
    base_hash, adapter_hash = digest_parameters(model), digest_parameters(model, True)
    model = model.to(device)
    cfg = Config(variant=variant, rank=rank, n_basis=n_basis, mlp_ratio=mlp_ratio, core_rank=core_rank,
                 adapter_input=adapter_input, adapter_gain=gain, width=settings["width"],
                 depth=settings["depth"], heads=settings["heads"], patch_size=settings["patch_size"],
                 batch_size=settings["batch_size"], max_steps=settings["steps"], warmup_steps=settings["warmup"],
                 lr=settings["lr"], device=settings["device"], precision=settings["precision"],
                 val_limit=settings["val_examples"], num_workers=0, sample_images=0)
    groups = gradient_groups(model)
    optimizer = torch.optim.AdamW([{"params": [p for _, p in ps], "lr": cfg.lr,
                                    "lr_multiplier": 1.0 if key == "base" else adapter_lr, "group_name": key}
                                   for key, ps in groups.items() if ps],
                                 betas=(0.9, 0.999), weight_decay=0, fused=device.type == "cuda")
    cpu_rng = torch.Generator().manual_seed(2030)
    noise_rng = torch.Generator(device=device).manual_seed(2031)
    probe_rng = torch.Generator(device=device).manual_seed(2032)
    probe_h = F.layer_norm(torch.randn(8, 64, cfg.width, generator=probe_rng, device=device), (cfg.width,))
    probe_c = F.layer_norm(torch.randn(8, cfg.width, generator=probe_rng, device=device), (cfg.width,))
    noisy, ts, labs = fixed
    diagnostic_steps = sorted({0, 1, 2, 3, 5, 10, 25, 50, 100, 250, settings["steps"]} & set(range(settings["steps"] + 1)))
    stream = hashlib.sha256()
    first_noise_hash = None
    first_nonzero = {}
    step_seconds, losses, diagnostic_records = [], [], []
    stats = model.statistics()
    write_json(folder / "config.json", {**settings, "name": name, "variant": variant, "rank": rank,
                                         "adapter_input": adapter_input, "n_basis": n_basis, "mlp_ratio": mlp_ratio,
                                         "core_rank": core_rank, "adapter_gain": gain, "seed": 42})
    metrics = (folder / "metrics.jsonl").open("w")
    def record(value):
        metrics.write(json.dumps(value, allow_nan=False) + "\n")
        metrics.flush()
    initial_branches = branch_snapshot(model, noisy, ts, labs, device, cfg.precision)
    if any(row["delta_rms"] != 0 or row["gate_rms"] != 0 for row in initial_branches):
        raise ValueError("Initialization is not a zero-delta identity block")
    initial = {"event": "diagnostic", "step": 0, "branches": initial_branches}
    diagnostic_records.append(initial)
    record(initial)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    training_peak = 0
    for step in range(1, settings["steps"] + 1):
        diagnose = step in diagnostic_steps
        before = factor_probe(model, probe_h, probe_c) if diagnose else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        start = time.perf_counter()
        indices = torch.randint(len(images_bank), (cfg.batch_size,), generator=cpu_rng)
        flips = torch.rand(cfg.batch_size, generator=cpu_rng) < .5
        images = images_bank[indices].clone()
        images[flips] = images[flips].flip(-1)
        labels = labels_bank[indices].to(device)
        images = images.to(device)
        timesteps = torch.randint(1000, (cfg.batch_size,), generator=noise_rng, device=device)
        noise = torch.randn(images.shape, generator=noise_rng, device=device)
        noisy_batch = scheduler.add_noise(images, noise, timesteps)
        progress = max(0, step - cfg.warmup_steps) / max(1, settings.get("schedule_steps", cfg.max_steps) - cfg.warmup_steps)
        lr_scale = step / max(1, cfg.warmup_steps) if step <= cfg.warmup_steps else .01 + .99 * (1 + math.cos(math.pi * progress)) / 2
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * lr_scale * group["lr_multiplier"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
            prediction = model(noisy_batch, timesteps, labels)
        loss = F.mse_loss(prediction.float(), noise.float())
        loss.backward()
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        step_seconds.append(elapsed)
        losses.append(loss.item())
        if device.type == "cuda":
            training_peak = max(training_peak, torch.cuda.max_memory_allocated(device))
        stream.update(indices.numpy().tobytes())
        stream.update(flips.numpy().tobytes())
        if first_noise_hash is None:
            first_noise_hash = digest_tensor(noise)
        if diagnose:
            gradients = gradient_record(groups)
            for group in gradients.values():
                for key in group["nonzero_gradient_tensors"]:
                    first_nonzero.setdefault(key, step)
            after = factor_probe(model, probe_h, probe_c)
            changes = [{"before_rms": rms(a), "after_rms": rms(b), "update_rms": rms(b - a)} for a, b in zip(before, after)]
            event = {"event": "diagnostic", "step": step, "loss": loss.item(), "pre_clip_total_gradient_l2": gradnorm.item(),
                     "lr_by_group": {g["group_name"]: g["lr"] for g in optimizer.param_groups},
                     "training_stream_sha256": stream.hexdigest(),
                     "noise_rng_state_sha256": digest_tensor(noise_rng.get_state()),
                     "gradient_groups_after_clip": gradients, "fixed_token_condition_function_updates": changes,
                     "branches": branch_snapshot(model, noisy, ts, labs, device, cfg.precision)}
            diagnostic_records.append(event)
            record(event)
            del before, after
        if step % 50 == 0 or step == settings["steps"]:
            record({"event": "progress", "step": step, "loss": loss.item(), "step_seconds": elapsed})
            print(json.dumps({"case": name, "step": step, "loss": round(loss.item(), 5)}), flush=True)
        del images, labels, noise, noisy_batch, prediction, loss
    validation_loader = make_loader(validation, cfg, False)
    model.eval()
    validation_result = evaluate(model, validation_loader, scheduler, cfg, device)
    trained_block = copy.deepcopy(model.blocks[0]).cpu() if settings.get("projection_probes", True) else None
    adapter_names = [key for key, _ in model.named_parameters() if ".adapter." in key]
    missing = sorted(set(adapter_names) - set(first_nonzero))
    if missing:
        raise ValueError(f"Adapter parameters never showed nonzero gradients at diagnostic steps: {missing}")
    noise_state_hash = digest_tensor(noise_rng.get_state())
    del optimizer, groups, model, probe_h, probe_c
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    projection = microbenchmark(trained_block, settings, device) if trained_block is not None else None
    token_projection = microbenchmark(trained_block, settings, device, "token") if trained_block is not None else None
    measured = step_seconds[settings.get("timing_warmup", settings["warmup"]):]
    result = {"name": name, "variant": variant, "rank": rank, "adapter_input": adapter_input,
              "n_basis": n_basis, "mlp_ratio": mlp_ratio, "core_rank": core_rank,
              "adapter_lr_multiplier": adapter_lr,
              "fixed_function_probe_precision": "FP32 tensors and highest FP32 matmul precision; no autocast",
              **stats, "completed_steps": settings["steps"], "processed_images": settings["steps"] * cfg.batch_size,
              "train_images_per_second": cfg.batch_size * len(measured) / sum(measured),
              "train_step_ms_median": statistics.median(measured) * 1000,
              "train_step_ms_p10_p90": [sorted(measured)[int((len(measured)-1)*q)] * 1000 for q in (.1,.9)],
              "train_timing_scope": "synchronized steps including cached CPU batch selection/flip, H2D, DDPM noise, forward/backward/optimizer; diagnostics/validation excluded",
              "peak_training_allocated_bytes": training_peak if device.type == "cuda" else None,
              "training_memory_scope": "model, gradients, AdamW, training activations, fixed diagnostic inputs; no EMA; diagnostic forwards excluded",
              "stream_sha256": stream.hexdigest(), "first_noise_sha256": first_noise_hash,
              "noise_rng_final_state_sha256": noise_state_hash,
              "base_initialization_sha256": base_hash, "adapter_initialization_sha256": adapter_hash,
              "adapter_first_nonzero_gradient_observed": {key: first_nonzero[key] for key in adapter_names},
              "diagnostic_observation_steps": diagnostic_steps, "diagnostics": diagnostic_records,
              "last50_mean_training_loss": statistics.mean(losses[-50:]), "raw_validation": validation_result,
              "projection_microbenchmark": projection, "token_projection_microbenchmark": token_projection, "runtime": runtime,
              "weights_saved": False, "test_evaluated": False, "ema_used": False,
              "quality_ranking_supported": False}
    record({"event": "complete", "step": settings["steps"]})
    metrics.close()
    write_json(folder / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    output = Path(settings["output_dir"])
    if (output / "report.json").exists():
        raise FileExistsError("Completed engineering evidence already exists")
    torch.set_num_threads(4)
    device = torch.device(settings["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
    cfg = Config(synthetic=settings["synthetic"], synthetic_samples=settings["bank_examples"],
                 train_limit=settings["bank_examples"], val_limit=settings["val_examples"], num_workers=0,
                 batch_size=settings["batch_size"], device=settings["device"], precision=settings["precision"])
    training, validation, metadata = build_datasets(cfg)
    if not settings["synthetic"]:
        training.transform = validation.transform  # cache unaugmented pixels; apply shared-stream flips later
    bank = [training[i] for i in range(len(training))]
    images_bank = torch.stack([image for image, _ in bank])
    labels_bank = torch.tensor([label for _, label in bank])
    del bank
    metadata.update(bank_images_sha256=digest_tensor(images_bank), bank_labels_sha256=digest_tensor(labels_bank),
                    protocol="sample with replacement from fixed train-only bank; reproducible per-step horizontal flips")
    write_json(output / "dataset.json", metadata)
    scheduler = make_noise_scheduler(cfg)
    val_count = min(8, len(validation))
    sample = [validation[i] for i in range(val_count)]
    images = torch.stack([x for x, _ in sample]).to(device)
    labels = torch.tensor([label for _, label in sample], device=device)
    rng = torch.Generator(device=device).manual_seed(2027)
    ts = torch.randint(1000, (val_count,), generator=rng, device=device)
    noisy = scheduler.add_noise(images, torch.randn(images.shape, generator=rng, device=device), ts)
    del images, sample
    runtime = provenance()
    runtime["source_sha256"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in settings.get("extra_provenance_sources", []):
        runtime["source_sha256"][name] = hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
    runtime.update(device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   device_uuid=str(torch.cuda.get_device_properties(device).uuid) if device.type == "cuda" else None)
    if settings.get("case_specifications"):
        runtime["gpu_claim"] = {key: os.environ.get(key) for key in ("GPU_CLAIM_INDICES", "CUDA_VISIBLE_DEVICES")}
    write_json(output / "runtime.json", runtime)
    rows = []
    cases = settings.get("case_specifications", CASES)
    by_name = {case["name"] if isinstance(case, dict) else case[0]: case for case in cases}
    for name in settings["run_order"]:
        row = run_case(by_name[name], settings, images_bank, labels_bank, validation, scheduler,
                       (noisy, ts, labels), output, runtime)
        rows.append(row)
        write_json(output / "progress.json", {"completed": [r["name"] for r in rows], "total": len(settings["run_order"])})
    report = {"settings": settings, "dataset": metadata, "runtime": runtime, "rows": rows,
              "quality_ranking_supported": False,
              "limits": ["single seed, short engineering budget, subset bank sampled with replacement; no quality selection",
                         "same sequential claimed GPU; other projects still share CPU/host resources",
                         "whole-model training eager; compilation applies only to trained first-block MLP input projection",
                         "token projection probe is a synthetic per-token extension, not quality evaluation of a token-conditioned model",
                         "no EMA/FID/generated-class assessment; validation MSE is a learning diagnostic",
                         "different generators have different identity initializations; gain/rate scaling is not yet derived",
                         "sequence-valued conditioning and cross-attention are not tested"]}
    if settings.get("case_specifications"):
        report["limits"].append("FFN control changes MLP shapes and base initialization; equal-shape variants share identical base hashes")
        report["limits"].append("fixed 1/sqrt(rank) gains except raw reference; new generator/core recipes not function-update calibrated")
    write_json(output / "report.json", report)
    protocol = json.loads((output / "protocol.json").read_text())
    protocol.update(state="complete", completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "suite_complete", "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()

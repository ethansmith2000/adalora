"""Opt-in, untimed raw-model gain/update diagnostics during quality training."""

import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark_dit_design import (branch_snapshot, digest_parameters, digest_tensor,
                                  factor_probe, gradient_groups, gradient_record, rms)


def parse_steps(value, maximum):
    if not value:
        return []
    try:
        steps = [int(v) for v in value.split(",")]
    except ValueError as exc:
        raise ValueError("Diagnostic steps must be comma-separated integers") from exc
    if steps != sorted(set(steps)) or min(steps) < 0 or max(steps) > maximum:
        raise ValueError("Diagnostic steps must be unique, increasing, and within the training budget")
    return steps


class QualityDiagnostics:
    def __init__(self, model, validation, scheduler, cfg, device):
        self.model, self.cfg, self.device = model, cfg, device
        self.steps = set(parse_steps(cfg.diagnostic_steps, cfg.max_steps))
        self.groups = gradient_groups(model)
        self.projector_groups = model.grouped_adapters()
        names = {id(p):name for name,p in model.named_parameters()}
        self.projector_gradients = {str(i):[(names[id(p)],p) for p in adapter.parameters()]
                                   for i,(_,adapter) in enumerate(self.projector_groups)}
        self.decoder_gradients = {str(i):[(names[id(adapter.U)],adapter.U),(names[id(adapter.V)],adapter.V)]
                                  for i,(_,adapter) in enumerate(model.grouped_decoders())}
        for i,(_,adapter) in enumerate(model.grouped_decoders()):
            if adapter.factor_offsets:
                self.decoder_gradients[str(i)].extend(
                    (names[id(p)],p) for p in (adapter.U_offset,adapter.V_offset))
        self.head_gradients = {str(i):[(names[id(p)],p) for p in adapter.coefficients.parameters()]
                              for i,(_,adapter) in enumerate(self.projector_groups) if adapter.variant == "dplr"}
        self.base_hash = digest_parameters(model)
        self.adapter_hash = digest_parameters(model, True)
        self.first_nonzero = {}
        self.events = []
        generator = torch.Generator(device=device).manual_seed(2032)
        self.h = F.layer_norm(torch.randn(8, (32 // cfg.patch_size)**2, cfg.width,
                                         generator=generator, device=device), (cfg.width,))
        self.c = F.layer_norm(torch.randn(8, cfg.width, generator=generator, device=device), (cfg.width,))
        samples = [validation[i] for i in range(min(8, len(validation)))]
        images = torch.stack([x for x, _ in samples]).to(device)
        self.labels = torch.tensor([label for _, label in samples], device=device)
        generator = torch.Generator(device=device).manual_seed(cfg.evaluation_seed)
        self.ts = torch.randint(cfg.num_train_timesteps, (len(samples),), generator=generator, device=device)
        noise = torch.randn(images.shape, generator=generator, device=device)
        self.noisy = scheduler.add_noise(images, noise, self.ts)
        self.before = None

    def initial(self):
        branches = branch_snapshot(self.model, self.noisy, self.ts, self.labels, self.device, self.cfg.precision)
        if any(b["delta_rms"] != 0 or b["gate_rms"] != 0 for b in branches):
            raise ValueError("Expected zero adapter/gates at initialization")
        event = {"event": "design_diagnostic", "step": 0, "branches": branches,
                 "fixed_condition_projector_magnitudes": self.projector_magnitudes()}
        self.events.append(event)
        return event

    @torch.no_grad()
    def projector_magnitudes(self):
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            result = []
            for index,(layers,adapter) in enumerate(self.projector_groups):
                if adapter.variant not in ("dplr", "exact"):
                    continue
                u,v = adapter.factors(self.c)
                norm2 = ((u.transpose(-1,-2) @ u) * (v @ v.transpose(-1,-2))).sum((-2,-1)).clamp_min(0)
                entry = dict(group=index,layers=layers,input_factor_rms=rms(u),output_factor_rms=rms(v),
                             transform_frobenius_mean=adapter.gain*norm2.sqrt().mean().item())
                if adapter.variant == "dplr":
                    a,b = adapter.coefficient_values(self.c).chunk(2,-1)
                    entry.update(input_coefficient_rms=rms(a),output_coefficient_rms=rms(b),
                                 input_basis_parameter_rms=rms(adapter.U),output_basis_parameter_rms=rms(adapter.V))
                    if adapter.factor_offsets:
                        entry.update(input_factor_offset_rms=rms(adapter.U_offset),
                                     output_factor_offset_rms=rms(adapter.V_offset))
                result.append(entry)
            return result
        finally:
            torch.set_float32_matmul_precision(precision)

    def begin(self):
        self.before = factor_probe(self.model, self.h, self.c)

    def finish(self, step, images, labels, noise, timesteps, generator):
        after = factor_probe(self.model, self.h, self.c)
        grads = gradient_record(self.groups)
        for group in grads.values():
            for name in group["nonzero_gradient_tensors"]:
                self.first_nonzero.setdefault(name, step)
        event = {"event": "design_diagnostic", "step": step,
                 "branches": branch_snapshot(self.model, self.noisy, self.ts, self.labels, self.device, self.cfg.precision),
                 "gradient_groups_after_clip": grads,
                 "projector_gradient_groups_after_clip": gradient_record(self.projector_gradients),
                 "decoder_gradient_groups_after_clip": gradient_record(self.decoder_gradients),
                 "coefficient_head_gradient_groups_after_clip": gradient_record(self.head_gradients),
                 "fixed_condition_projector_magnitudes": self.projector_magnitudes(),
                 "fixed_token_condition_function_updates": [
                     {"before_rms": rms(a), "after_rms": rms(b), "update_rms": rms(b-a)}
                     for a,b in zip(self.before,after)],
                 "batch_images_sha256": digest_tensor(images), "batch_labels_sha256": digest_tensor(labels),
                 "noise_sha256": digest_tensor(noise), "timesteps_sha256": digest_tensor(timesteps),
                 "noise_rng_state_sha256": digest_tensor(generator.get_state())}
        self.before = None
        self.events.append(event)
        return event

    def summary(self, generator):
        names = [name for name,_ in self.model.named_parameters() if ".adapter." in name]
        missing = sorted(set(names) - set(self.first_nonzero))
        if missing:
            raise ValueError(f"Adapter gradients never observed active: {missing}")
        return {"base_initialization_sha256": self.base_hash, "adapter_initialization_sha256": self.adapter_hash,
                "fixed_function_probe_precision": "FP32 tensors and highest FP32 matmul precision; no autocast",
                "fixed_input_seed": 2032, "diagnostic_observation_steps": sorted(self.steps), "model_state": "raw",
                "noise_rng_final_state_sha256": digest_tensor(generator.get_state()),
                "adapter_first_nonzero_gradient_observed": {name:self.first_nonzero[name] for name in names},
                "projector_group_layers": [layers for layers,_ in self.projector_groups],
                "coefficient_activation": self.model.coefficient_activation,
                "coefficient_hidden": self.model.coefficient_hidden,
                "factor_offsets": self.model.factor_offsets,
                "decoder_group_layers": [layers for layers,_ in self.model.grouped_decoders()],
                "projector_gradient_scope": "per adapter; shared decoder tensors repeat across adapters; separate decoder/head groups count unique tensors",
                "diagnostics": self.events,
                "scope": "fixed independent normalized inputs/conditions; live diagnostic validation images; not EMA updates"}


def diagnostic_source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ("dit_quality_diagnostics.py", "benchmark_dit_design.py")}

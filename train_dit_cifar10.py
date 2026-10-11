#!/usr/bin/env python3
"""Class/timestep-conditioned CIFAR-10 DDPM benchmark with a small DiT.

Ports the conventional epsilon objective from the U-Net reference trainer,
without its external optimizer/logger dependencies or checkpoint defaults.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from dit_cifar10 import ADAPTER_INPUTS, CIFAR10DiT, VARIANTS
from train_cifar10 import (Config as DataConfig, SyntheticImages, TransformedSubset, _hf_split,
                          make_loader, memory_stats, stratified_split, synchronize)


SOURCE_FILES = ("train_dit_cifar10.py", "dit_cifar10.py", "train_cifar10.py", "vit.py", "conditioning.py", "condlr.py", "class_fidelity.py")


@dataclass
class Config:
    variant: str = "adaln_zero"
    width: int = 192
    depth: int = 6
    heads: int = 3
    patch_size: int = 4
    mlp_ratio: float = 4.0
    rank: int = 8
    n_basis: int = 16
    core_rank: int = 2
    adapter_input: str = "normalized"
    adapter_gain: float = 1.0
    projector_groups: int = 0
    coefficient_activation: str = "linear"
    decoder_groups: int = 0
    coefficient_hidden: int = 0
    factor_offsets: bool = False
    max_steps: int = 10000
    batch_size: int = 128
    lr: float = 2e-4
    adapter_lr_multiplier: float = 1.0
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    fused_adamw: bool = True
    compile_model: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool = True
    class_dropout_prob: float = 0.0
    guidance_scale: float = 1.0
    guidance_scales: str = ""
    generative_eval_steps: str = ""
    warmup_steps: int = 500
    lr_schedule_steps: int | None = None
    min_lr_ratio: float = 0.01
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    seed: int = 42
    split_seed: int = 123
    evaluation_seed: int = 2027
    sampling_seed: int = 2028
    hf_home: str = "/workspace/.hf_home"
    download: bool = False
    num_workers: int = 4
    cpu_threads: int = 4
    train_limit: int | None = None
    val_limit: int | None = None
    synthetic: bool = False
    synthetic_samples: int = 256
    precision: str = "bf16"
    device: str = "cuda:0"
    log_every: int = 20
    eval_every: int = 2500
    benchmark_warmup_steps: int = 10
    sample_images: int = 100
    sample_batch_size: int = 100
    sampling_steps: int = 50
    fid_images: int = 0
    fid_batch_size: int = 64
    fid_weights: str = "/root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
    classifier_run: str | None = None
    output_dir: str | None = None
    diagnostic_steps: str = ""


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--override_json", type=Path)
    for name, value in asdict(Config()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            kind = int if name in ("train_limit", "val_limit", "lr_schedule_steps") else type(value) if value is not None else str
            parser.add_argument(flag, type=kind, default=None)
    args = vars(parser.parse_args(argv))
    overrides = args.pop("override_json")
    values = asdict(Config())
    if overrides:
        supplied = json.loads(overrides.read_text())
        if not isinstance(supplied, dict) or set(supplied) - set(values):
            raise ValueError("Only known Config fields are accepted")
        values.update(supplied)
    values.update({k: v for k, v in args.items() if v is not None})
    cfg = Config(**values)
    if cfg.diagnostic_steps:
        from dit_quality_diagnostics import parse_steps
        steps = parse_steps(cfg.diagnostic_steps, cfg.max_steps)
        if 0 not in steps or cfg.max_steps not in steps:
            raise ValueError("Enabled diagnostics must include step zero and the final step")
    for field in ("width", "depth", "heads", "patch_size", "rank", "n_basis", "core_rank", "max_steps", "batch_size",
                  "cpu_threads", "log_every", "eval_every", "sample_batch_size", "sampling_steps", "fid_batch_size"):
        if getattr(cfg, field) < 1:
            raise ValueError(f"{field} must be positive")
    if cfg.variant not in VARIANTS or cfg.adapter_input not in ADAPTER_INPUTS or cfg.width % cfg.heads or cfg.width % 4 or 32 % cfg.patch_size:
        raise ValueError("Invalid model variant or dimensions")
    if cfg.variant == "conditional_dplr_core" and cfg.core_rank > cfg.rank:
        raise ValueError("Core factor rank must be no larger than the outer rank")
    if cfg.mlp_ratio <= 0 or cfg.lr <= 0 or cfg.adapter_lr_multiplier <= 0 or cfg.max_grad_norm <= 0:
        raise ValueError("Invalid model/optimizer setting")
    if not all(math.isfinite(v) and 0 <= v < 1 for v in (cfg.adam_beta1, cfg.adam_beta2, cfg.class_dropout_prob)):
        raise ValueError("Adam betas and class dropout must be finite values in [0,1)")
    if cfg.compile_mode not in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        raise ValueError("Unsupported model compile mode")
    for name in ("fused_adamw", "compile_model", "compile_dynamic"):
        if type(getattr(cfg,name)) is not bool:
            raise ValueError(f"{name} must be boolean")
    if type(cfg.factor_offsets) is not bool or (cfg.factor_offsets and cfg.variant != "dplr"):
        raise ValueError("Factor offsets require DPLR and a boolean setting")
    scales = evaluation_guidance_scales(cfg)
    generation_steps = intermediate_generation_steps(cfg)
    if generation_steps and not (cfg.sample_images or cfg.fid_images):
        raise ValueError("Intermediate generation requires a positive sample or FID count")
    if any(scale != 1 for scale in scales) and cfg.class_dropout_prob == 0:
        raise ValueError("CFG requires class-dropout training with a learned null class")
    if not math.isfinite(cfg.adapter_gain) or cfg.adapter_gain <= 0:
        raise ValueError("Adapter gain must be finite and positive")
    if (not isinstance(cfg.projector_groups, int) or isinstance(cfg.projector_groups, bool)
            or cfg.projector_groups < 0 or cfg.projector_groups > cfg.depth
            or (cfg.projector_groups and cfg.depth % cfg.projector_groups)
            or cfg.coefficient_activation not in ("linear", "silu")
            or (cfg.variant != "dplr" and (cfg.projector_groups or cfg.coefficient_activation != "linear"))):
        raise ValueError("Projector groups must divide depth; shared/nonlinear projectors require DPLR")
    if (not isinstance(cfg.decoder_groups, int) or isinstance(cfg.decoder_groups, bool)
            or cfg.decoder_groups < 0 or cfg.decoder_groups > cfg.depth
            or (cfg.decoder_groups and cfg.depth % cfg.decoder_groups)
            or (cfg.decoder_groups and cfg.projector_groups not in (0, cfg.depth))
            or not isinstance(cfg.coefficient_hidden, int) or isinstance(cfg.coefficient_hidden, bool)
            or cfg.coefficient_hidden < 0
            or (cfg.variant != "dplr" and (cfg.decoder_groups or cfg.coefficient_hidden))):
        raise ValueError("Decoder sharing requires independent DPLR heads and hidden width must be nonnegative")
    if not 0 <= cfg.warmup_steps < cfg.max_steps or not 0 <= cfg.ema_decay < 1 or not 0 <= cfg.min_lr_ratio <= 1:
        raise ValueError("Invalid schedule or EMA")
    if cfg.lr_schedule_steps is not None and (type(cfg.lr_schedule_steps) is not int or cfg.lr_schedule_steps < cfg.max_steps):
        raise ValueError("LR schedule horizon must be an integer at least max_steps")
    if cfg.precision not in ("fp32", "bf16") or cfg.num_workers < 0 or cfg.weight_decay < 0:
        raise ValueError("Invalid precision, worker count or weight decay")
    if cfg.num_train_timesteps < 2 or not 1 <= cfg.sampling_steps <= cfg.num_train_timesteps:
        raise ValueError("Invalid diffusion timestep budget")
    if min(cfg.sample_images, cfg.fid_images, cfg.benchmark_warmup_steps) < 0:
        raise ValueError("Artifact and benchmark counts cannot be negative")
    if any(getattr(cfg, k) is not None and getattr(cfg, k) < 1 for k in ("train_limit", "val_limit")):
        raise ValueError("Data limits must be positive or null")
    if cfg.fid_images and (cfg.fid_images < 2 or cfg.synthetic):
        raise ValueError("FID requires real data and at least two generated images")
    if cfg.fid_images and not Path(cfg.fid_weights).is_file():
        raise FileNotFoundError("The declared FID evaluator weights must already be cached")
    if cfg.classifier_run and (cfg.synthetic or not cfg.fid_images or not (Path(cfg.classifier_run) / "model.pt").is_file()):
        raise ValueError("Class agreement requires real-data FID and an existing frozen evaluator")
    return cfg


def build_datasets(cfg):
    if cfg.synthetic:
        class BoundedSynthetic(SyntheticImages):
            def __getitem__(self, index):
                image, label = super().__getitem__(index)
                return image.tanh(), label
        return (BoundedSynthetic(cfg.synthetic_samples, cfg.seed + 100000),
                BoundedSynthetic(min(cfg.synthetic_samples, cfg.val_limit or cfg.synthetic_samples), cfg.seed + 200000),
                {"source": "synthetic", "quality_metrics_meaningful": False})
    source, location = _hf_split(DataConfig(hf_home=cfg.hf_home, download=cfg.download), "train")
    train_indices, validation_indices = stratified_split(source["label"], 0.1, cfg.split_seed)
    train_indices, validation_indices = train_indices[:cfg.train_limit], validation_indices[:cfg.val_limit]
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    training = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    evaluation = transforms.Compose([transforms.ToTensor(), normalize])
    return (TransformedSubset(source, train_indices, training, True),
            TransformedSubset(source, validation_indices, evaluation, True),
            {"source": "huggingface", "location": location, "source_fingerprint": source._fingerprint,
             "train_indices": train_indices, "validation_indices": validation_indices,
             "train_examples": len(train_indices), "validation_examples": len(validation_indices),
             "test_evaluated": False, "normalization": "pixel RGB [-1,1]", "training_augmentation": "horizontal flip only",
             "class_names": source.features["label"].names})


def make_model(cfg):
    return CIFAR10DiT(cfg.width, cfg.depth, cfg.heads, cfg.patch_size, cfg.mlp_ratio, cfg.variant, cfg.rank, cfg.n_basis,
                      adapter_input=cfg.adapter_input, adapter_gain=cfg.adapter_gain, core_rank=cfg.core_rank,
                      projector_groups=cfg.projector_groups, coefficient_activation=cfg.coefficient_activation,
                      decoder_groups=cfg.decoder_groups, coefficient_hidden=cfg.coefficient_hidden,
                      enable_unconditional=cfg.class_dropout_prob > 0, factor_offsets=cfg.factor_offsets)


def adapter_ids(model):
    return {id(p) for block in model.blocks if block.adapter is not None for p in block.adapter.parameters()}


def make_optimizer(model, cfg):
    ids = adapter_ids(model)
    groups = []
    for extra in (False, True):
        for decay in (False, True):
            params = [p for p in model.parameters() if (id(p) in ids) == extra and (p.ndim >= 2) == decay]
            if params:
                groups.append({"params": params, "lr": cfg.lr * (cfg.adapter_lr_multiplier if extra else 1),
                               "weight_decay": cfg.weight_decay if decay else 0})
    return torch.optim.AdamW(groups, betas=(cfg.adam_beta1, cfg.adam_beta2),
                            fused=cfg.fused_adamw and cfg.device.startswith("cuda"))


def training_callable(model, cfg):
    # Keep ownership/EMA/diagnostics on the raw module: compiled wrappers rename parameters.
    return (torch.compile(model, mode=cfg.compile_mode, dynamic=cfg.compile_dynamic, fullgraph=True)
            if cfg.compile_model else model)


def drop_class_labels(labels, probability, generator):
    if probability == 0:
        return labels
    mask = torch.rand(labels.shape, device=labels.device, generator=generator) < probability
    return torch.where(mask, 10, labels)


def evaluation_guidance_scales(cfg):
    if not isinstance(cfg.guidance_scales, str):
        raise ValueError("Guidance scales must be a comma-separated string")
    try:
        scales = [float(v) for v in cfg.guidance_scales.split(",")] if cfg.guidance_scales else [cfg.guidance_scale]
    except (ValueError, TypeError) as exc:
        raise ValueError("Guidance scales must be comma-separated numbers") from exc
    if (not scales or len(set(scales)) != len(scales) or cfg.guidance_scale not in scales
            or any(not math.isfinite(v) or v < 0 for v in [cfg.guidance_scale,*scales])):
        raise ValueError("Guidance scales must be finite, nonnegative, unique and include the primary scale")
    return scales


def intermediate_generation_steps(cfg):
    """Explicit intermediate states; final generation is handled separately."""
    if not isinstance(cfg.generative_eval_steps, str):
        raise ValueError("Generation steps must be a comma-separated string")
    if not cfg.generative_eval_steps:
        return []
    try:
        steps = [int(v) for v in cfg.generative_eval_steps.split(",")]
    except ValueError as exc:
        raise ValueError("Generation steps must be comma-separated integers") from exc
    if steps != sorted(set(steps)) or not all(0 < v < cfg.max_steps for v in steps):
        raise ValueError("Generation steps must be unique, increasing, positive and before the final step")
    return steps


def guided_prediction(model, images, timestep, labels, scale):
    if scale == 1:
        return model(images, timestep, labels)
    if not getattr(model, "enable_unconditional", False):
        raise ValueError("Guidance needs a model with a trained null-class path")
    both = model(torch.cat((images,images)), timestep, torch.cat((labels,torch.full_like(labels,10))))
    conditional, unconditional = both.float().chunk(2)
    return unconditional + scale * (conditional - unconditional)


def make_noise_scheduler(cfg):
    from diffusers import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps, beta_schedule=cfg.beta_schedule,
                         prediction_type="epsilon", clip_sample=True)


def denoising_loss(model, scheduler, images, labels, noise, timesteps):
    prediction = model(scheduler.add_noise(images, noise, timesteps), timesteps, labels)
    return F.mse_loss(prediction.float(), noise.float())


@torch.no_grad()
def update_ema(ema, model, decay):
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)


@torch.no_grad()
def evaluate(model, loader, scheduler, cfg, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(cfg.evaluation_seed)
    total, wrong_total, effect_total, count = 0.0, 0.0, 0.0, 0
    bucket_sum = torch.zeros(10, device=device, dtype=torch.float64)
    bucket_count = torch.zeros(10, device=device, dtype=torch.int64)
    synchronize(device)
    start = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        timesteps = torch.randint(cfg.num_train_timesteps, (len(images),), generator=generator, device=device)
        noise = torch.randn(images.shape, generator=generator, device=device)
        noisy = scheduler.add_noise(images, noise, timesteps)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
            prediction = model(noisy, timesteps, labels).float()
            wrong = model(noisy, timesteps, (labels + 1) % 10).float()
        loss = (prediction - noise).square().mean(dim=(1, 2, 3))
        total += loss.sum().item()
        wrong_total += (wrong - noise).square().mean(dim=(1, 2, 3)).sum().item()
        effect_total += (prediction - wrong).square().mean(dim=(1, 2, 3)).sum().item()
        count += len(images)
        bucket = timesteps * 10 // cfg.num_train_timesteps
        bucket_sum.index_add_(0, bucket, loss.double())
        bucket_count += torch.bincount(bucket, minlength=10)
    synchronize(device)
    return {"noise_mse": total / count, "wrong_class_noise_mse": wrong_total / count,
            "wrong_minus_correct_class_noise_mse": (wrong_total - total) / count,
            "class_prediction_change_rms": math.sqrt(effect_total / count), "examples": count,
            "noise_seed": cfg.evaluation_seed, "timestep_decile_counts": bucket_count.tolist(),
            "timestep_decile_noise_mse": [(s / n if n else None) for s, n in zip(bucket_sum.tolist(), bucket_count.tolist())],
            "seconds": time.perf_counter() - start}


@torch.no_grad()
def sample_batches(model, scheduler, cfg, device, num_images, batch_size):
    from diffusers import DDIMScheduler
    sampler = DDIMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(cfg.sampling_steps, device=device)
    generator = torch.Generator(device=device).manual_seed(cfg.sampling_seed)
    model.eval()
    for offset in range(0, num_images, batch_size):
        count = min(batch_size, num_images - offset)
        labels = torch.arange(offset, offset + count, device=device) % 10
        images = torch.randn((count, 3, 32, 32), generator=generator, device=device)
        for timestep in sampler.timesteps:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
                prediction = guided_prediction(model, images, timestep, labels, cfg.guidance_scale)
            images = sampler.step(prediction.float(), timestep, images, eta=0).prev_sample
        yield images.clamp(-1, 1).add(1).mul(0.5), labels


@torch.no_grad()
def generative_evaluation(model, scheduler, loader, cfg, device, output, save_grid=True):
    synchronize(device)
    start = time.perf_counter()
    metric = None
    agreement = None
    if cfg.classifier_run:
        from class_fidelity import ClassAgreement
        agreement = ClassAgreement.from_run(cfg.classifier_run, device, loader.dataset.indices)
    if cfg.fid_images:
        from torchmetrics.image.fid import FrechetInceptionDistance
        # The installed metric uses float32 Inception and float64 moment sums.
        metric = FrechetInceptionDistance(feature=2048, normalize=False,
                                         feature_extractor_weights_path=cfg.fid_weights, antialias=True).to(device)
        for images, labels in loader:
            metric.update(images.to(device).add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8), real=True)
            if agreement is not None:
                agreement.update(images.add(1).mul(0.5), labels, "real")
    num_images = cfg.fid_images or cfg.sample_images
    batch_size = cfg.fid_batch_size if metric is not None else cfg.sample_batch_size
    grid, labels_saved = [], []
    for images, labels in sample_batches(model, scheduler, cfg, device, num_images, batch_size):
        if metric is not None:
            metric.update(images.mul(255).round().clamp(0, 255).to(torch.uint8), real=False)
        if agreement is not None:
            agreement.update(images, labels, "generated")
        remaining = (cfg.sample_images if save_grid else 0) - sum(len(part) for part in grid)
        if remaining > 0:
            grid.append(images[:remaining].cpu())
            labels_saved.extend(labels[:remaining].cpu().tolist())
    result = {"model_state": "EMA", "sampler": "DDIM", "eta": 0, "sampling_steps": cfg.sampling_steps,
              "sampling_seed": cfg.sampling_seed, "classifier_free_guidance": cfg.guidance_scale != 1,
              "guidance_scale": cfg.guidance_scale, "class_dropout_prob": cfg.class_dropout_prob,
              "generated_images": num_images, "sample_batch_size": batch_size,
              "balanced_requested_class_counts": [num_images // 10 + int(i < num_images % 10) for i in range(10)],
              "fid": float(metric.compute().item()) if metric is not None else None,
              "fid_feature_dimension": 2048 if metric is not None else None,
              "fid_real_images": len(loader.dataset) if metric is not None else None,
              "fid_fake_images": cfg.fid_images, "fid_reference": "held-out training split; official test untouched",
              "fid_feature_input": "uint8 RGB [0,255]; float32 Inception, float64 moment accumulation; antialias=True",
              "fid_interpretation": "finite-sample pilot; not directly comparable to published 50k-image FID",
              "sample_grid_labels": labels_saved, "saved_generated_dataset": False}
    if agreement is not None:
        result["class_agreement"] = agreement.compute()
    if grid:
        save_image(torch.cat(grid), output / "samples_ema.png", nrow=10)
    synchronize(device)
    result["seconds"] = time.perf_counter() - start
    return result


def evaluate_generation_state(model, scheduler, loader, cfg, device, output, all_scales=True, save_grid=True):
    """In-memory evaluation with evaluator construction isolated from training RNG."""
    if not (cfg.sample_images or cfg.fid_images):
        return None, {}
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            primary = generative_evaluation(model, scheduler, loader, cfg, device, output, save_grid=save_grid)
            guided = {}
            if all_scales and cfg.guidance_scales:
                for scale in evaluation_guidance_scales(cfg):
                    guided[str(scale)] = (primary if scale == cfg.guidance_scale else
                        generative_evaluation(model, scheduler, loader, replace(cfg, guidance_scale=scale),
                                              device, output, save_grid=False))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return primary, guided


def provenance():
    repo = Path(__file__).resolve().parent
    versions = {}
    for package in ("torch", "torchvision", "diffusers", "datasets", "torchmetrics", "torch-fidelity"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
            "package_versions": versions, "cuda_wheel": torch.version.cuda}


def learning_rate_scale(cfg, step):
    """A short calibration run can execute the exact prefix of a longer schedule."""
    horizon = cfg.lr_schedule_steps if cfg.lr_schedule_steps is not None else cfg.max_steps
    progress = max(0, step - cfg.warmup_steps) / max(1, horizon - cfg.warmup_steps)
    return (step / max(1, cfg.warmup_steps) if step <= cfg.warmup_steps else
            cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2)


def main(argv=None):
    cfg = load_config(argv)
    generation_steps = set(intermediate_generation_steps(cfg))
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.set_float32_matmul_precision("high")
    output = Path(cfg.output_dir or f"runs/dit-{cfg.variant}-s{cfg.seed}-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise FileExistsError("Run records already exist; use a new output directory")
    cfg.output_dir = str(output)
    free = shutil.disk_usage(output).free
    if free < 20 * 1024**2:
        raise RuntimeError("Need at least 20 MiB free for declared compact evidence")
    write_json(output / "config.json", asdict(cfg))
    runtime = provenance()
    if cfg.diagnostic_steps:
        from dit_quality_diagnostics import diagnostic_source_hashes
        runtime["source_sha256"].update(diagnostic_source_hashes())
    runtime.update(device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   device_uuid=str(torch.cuda.get_device_properties(device).uuid) if device.type == "cuda" else None)
    write_json(output / "provenance.json", runtime)
    evaluator = None
    if cfg.fid_images:
        path = Path(cfg.fid_weights)
        evaluator = {"purpose": "pretrained Inception feature extractor required by FID; not a training checkpoint",
                     "path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "retention": "existing shared evaluator cache; this workflow does not delete or replace it"}
    auxiliary = None
    if cfg.classifier_run:
        from class_fidelity import classifier_provenance
        auxiliary = classifier_provenance(cfg.classifier_run)
    write_json(output / "artifact_manifest.json", {"purpose": "external broadcast-conditioning optimization and generative-quality pilot",
               "disk_free_bytes_before_training": free, "expected_compact_record_bytes_upper_bound": 20 * 1024**2,
               "weights_saved": False, "periodic_checkpoints": False, "ema": "in memory only; evaluated before process exits",
               "evaluator": evaluator, "auxiliary_classifier": auxiliary, "sample_artifact": "one small balanced-class EMA image grid",
               "retention": "configs, split indices, source provenance, metrics, logs, summary and sample grid until explicitly removed"})
    train, val, metadata = build_datasets(cfg)
    write_json(output / "dataset.json", metadata)
    train_loader, val_loader = make_loader(train, cfg, True), make_loader(val, cfg, False)
    model = make_model(cfg).to(device)
    stats = model.statistics()
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = make_optimizer(model, cfg)
    train_model = training_callable(model, cfg)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scheduler = make_noise_scheduler(cfg)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 10000)
    label_generator = torch.Generator(device=device).manual_seed(cfg.seed + 30000) if cfg.class_dropout_prob else None
    dropped_count = torch.zeros((), dtype=torch.int64, device=device) if label_generator is not None else None
    diagnostics = None
    if cfg.diagnostic_steps:
        from dit_quality_diagnostics import QualityDiagnostics
        diagnostics = QualityDiagnostics(model, val, scheduler, cfg, device)
    ids = adapter_ids(model)
    peak = {"peak_allocated_bytes": 0 if device.type == "cuda" else None,
            "peak_reserved_bytes": 0 if device.type == "cuda" else None}
    total_images, measured_images, measured_seconds = 0, 0, 0.0
    generation_curve = []
    with (output / "metrics.jsonl").open("w", buffering=1) as metrics:
        def record(event):
            line = json.dumps(event, allow_nan=False)
            metrics.write(line + "\n")
            print(line, flush=True)

        initial = evaluate(model, val_loader, scheduler, cfg, device)
        record({"event": "validation", "step": 0, "raw": initial, "ema": initial})
        if diagnostics:
            record(diagnostics.initial())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        loop_start = time.perf_counter()
        iterator = iter(train_loader)
        window_start, window_images = time.perf_counter(), 0
        excluded_diagnostic_seconds = 0.0
        for step in range(1, cfg.max_steps + 1):
            model.train()
            diagnose = diagnostics is not None and step in diagnostics.steps
            if diagnose:
                synchronize(device)
                diagnostic_start = time.perf_counter()
                current = memory_stats(device)
                for key in peak:
                    if current[key] is not None:
                        peak[key] = max(peak[key], current[key])
                diagnostics.begin()
                synchronize(device)
                excluded_diagnostic_seconds += time.perf_counter() - diagnostic_start
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, labels = next(iterator)
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            scale = learning_rate_scale(cfg, step)
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * scale
            noise = torch.randn(images.shape, generator=generator, device=device)
            timesteps = torch.randint(cfg.num_train_timesteps, (len(images),), generator=generator, device=device)
            training_labels = drop_class_labels(labels, cfg.class_dropout_prob, label_generator)
            if dropped_count is not None:
                dropped_count += (training_labels == 10).sum()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
                loss = denoising_loss(train_model, scheduler, images, training_labels, noise, timesteps)
            loss.backward()
            report = step == 1 or step % cfg.log_every == 0 or step % cfg.eval_every == 0 or step == cfg.max_steps
            adapter_norm = None
            if report and ids:
                values = [p.grad.detach().float().square().sum() for p in model.parameters() if id(p) in ids and p.grad is not None]
                adapter_norm = torch.stack(values).sum().sqrt().item() if values else 0.0
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            update_ema(ema, model, cfg.ema_decay)
            if diagnose:
                synchronize(device)
                current = memory_stats(device)
                for key in peak:
                    if current[key] is not None:
                        peak[key] = max(peak[key], current[key])
                diagnostic_start = time.perf_counter()
                record(diagnostics.finish(step, images, training_labels, noise, timesteps, generator))
                synchronize(device)
                excluded_diagnostic_seconds += time.perf_counter() - diagnostic_start
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            total_images += len(images)
            window_images += len(images)
            if step == cfg.benchmark_warmup_steps:
                synchronize(device)
                window_start, window_images = time.perf_counter(), 0
                excluded_diagnostic_seconds = 0.0
            if report:
                synchronize(device)
                duration = time.perf_counter() - window_start - excluded_diagnostic_seconds
                measured = step > cfg.benchmark_warmup_steps
                if measured:
                    measured_images += window_images
                    measured_seconds += duration
                record({"event": "train", "step": step, "noise_mse": float(loss.detach()),
                        "grad_norm_before_clip": float(grad), "adapter_grad_norm_before_clip": adapter_norm,
                        "base_lr": optimizer.param_groups[0]["lr"], "images_per_second": window_images / max(duration, 1e-9),
                        "benchmark_window": measured, "processed_images": total_images})
                window_start, window_images = time.perf_counter(), 0
                excluded_diagnostic_seconds = 0.0
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                current = memory_stats(device)
                for key in peak:
                    if current[key] is not None:
                        peak[key] = max(peak[key], current[key])
                raw, smoothed = evaluate(model, val_loader, scheduler, cfg, device), evaluate(ema, val_loader, scheduler, cfg, device)
                record({"event": "validation", "step": step, "raw": raw, "ema": smoothed})
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                window_start, window_images = time.perf_counter(), 0
                excluded_diagnostic_seconds = 0.0
            if step in generation_steps:
                current = memory_stats(device)
                for key in peak:
                    if current[key] is not None:
                        peak[key] = max(peak[key], current[key])
                generated, _ = evaluate_generation_state(ema, scheduler, val_loader, cfg, device, output,
                                                         all_scales=False, save_grid=False)
                generation_curve.append({"step": step, **generated})
                record({"event": "generative_evaluation", "step": step, **generated})
                if device.type == "cuda":
                    # Release evaluator-only allocator reservations before training peaks resume.
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                window_start, window_images = time.perf_counter(), 0
                excluded_diagnostic_seconds = 0.0
        synchronize(device)
        loop_seconds = time.perf_counter() - loop_start
        generated, guidance_evaluations = evaluate_generation_state(ema, scheduler, val_loader, cfg, device, output)
        if generated:
            record({"event": "generative_evaluation", "step": cfg.max_steps, **generated})
        result = {**stats, "variant": cfg.variant, "completed_steps": cfg.max_steps, "processed_images": total_images,
                  "raw_validation": raw, "ema_validation": smoothed, "initial_validation": initial,
                  "generative_evaluation": generated, "loop_seconds": loop_seconds,
                  "mean_train_images_per_second_excluding_warmup": measured_images / measured_seconds if measured_seconds else None,
                  "training_memory_includes": "model, in-memory EMA, gradients, optimizer and training activations; excludes FID",
                  **peak, "weights_saved": False, "test_evaluated": False}
        result["training_setup"] = {"adam_betas": [cfg.adam_beta1,cfg.adam_beta2],
                                   "fused_adamw": optimizer.defaults["fused"],
                                   "model_compiled": cfg.compile_model, "compile_mode": cfg.compile_mode,
                                   "compile_dynamic": cfg.compile_dynamic,
                                   "compile_scope": "training forward/backward; raw module owns optimizer, EMA and eager diagnostics/evaluation",
                                   "class_dropout_prob": cfg.class_dropout_prob,
                                   "dropped_training_labels": int(dropped_count.item()) if dropped_count is not None else 0,
                                   "label_dropout_rng_seed": cfg.seed+30000 if label_generator is not None else None,
                                   "label_dropout_rng_final_state_sha256":
                                       hashlib.sha256(label_generator.get_state().cpu().numpy().tobytes()).hexdigest()
                                       if label_generator is not None else None}
        if guidance_evaluations:
            result["guidance_evaluations"] = guidance_evaluations
        if generation_steps:
            result["generative_validation_curve"] = [*generation_curve, {"step": cfg.max_steps, **generated}]
            result["intermediate_generation_seconds"] = sum(r["seconds"] for r in generation_curve)
            result["loop_seconds_includes"] = "training, validation, diagnostics, compilation and intermediate generation"
        if diagnostics:
            result["design_diagnostics"] = diagnostics.summary(generator)
            result["training_memory_includes"] += "; fixed diagnostic inputs; diagnostic forwards excluded from peaks"
            result["diagnostic_timing"] = "diagnostic probes and logging excluded from train windows; included in loop_seconds"
        write_json(output / "results.json", result)
        record({"event": "complete", **result})
    return result


if __name__ == "__main__":
    main()

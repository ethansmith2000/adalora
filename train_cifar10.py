#!/usr/bin/env python3
"""CIFAR-10 ViT screening for token-conditioned channel transforms.

Uses AdamW and a seeded train/validation split. The official test split is only
used with --evaluate-test. No checkpoints or final weights are saved by default.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from conditioning import ConditionedLinear, VARIANTS
from vit import ViT


@dataclass
class Config:
    variant: str = "plain"
    placement: str = "mlp_in"
    rank: int = 2
    n_basis: int = 16
    chunk: int = 2048
    detach_condition: bool = False
    adapter_gain: float = 1.0
    factor_offsets: bool = False
    diagnostic_steps: str = ""
    width: int = 192
    depth: int = 6
    heads: int = 3
    patch_size: int = 4
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    drop_rate: float = 0.0
    gradient_checkpointing: bool = False
    epochs: int = 100
    max_steps: int | None = None
    lr_schedule_steps: int | None = None
    batch_size: int = 128
    lr: float = 5e-4
    hyper_lr_multiplier: float = 1.0
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.01
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    split_seed: int = 123
    val_fraction: float = 0.1
    num_workers: int = 4
    cpu_threads: int = 4
    data_source: str = "huggingface"
    data_dir: str = "data"
    hf_home: str = os.environ.get("HF_HOME", "/workspace/.hf_home")
    download: bool = False
    train_limit: int | None = None
    val_limit: int | None = None
    synthetic: bool = False
    synthetic_samples: int = 256
    precision: str = "bf16"
    device: str = "cuda:0"
    compile: bool = False
    log_every: int = 20
    benchmark_warmup_steps: int = 10
    evaluate_test: bool = False
    output_dir: str | None = None
    save_weights: bool = False
    artifact_purpose: str | None = None


def load_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--override_json", type=Path)
    defaults = Config()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            kind = int if name in ("max_steps", "lr_schedule_steps", "train_limit", "val_limit") else type(value) if value is not None else str
            parser.add_argument(flag, type=kind, default=None)
    args = vars(parser.parse_args(argv))
    path = args.pop("override_json")
    values = asdict(defaults)
    if path:
        overrides = json.loads(path.read_text())
        if not isinstance(overrides, dict) or set(overrides) - set(values):
            raise ValueError("Config must be an object containing only known Config keys")
        values.update(overrides)
    values.update({k: v for k, v in args.items() if v is not None})
    cfg = Config(**values)
    if cfg.variant not in VARIANTS or cfg.placement not in ("mlp_in", "qkv", "both"):
        raise ValueError("Invalid variant or placement")
    for name in ("width", "depth", "heads", "patch_size", "rank", "n_basis", "chunk", "epochs",
                 "batch_size", "cpu_threads", "log_every", "synthetic_samples"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_steps", "lr_schedule_steps", "train_limit", "val_limit"):
        if getattr(cfg, name) is not None and getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive or null")
    if cfg.width % cfg.heads or 32 % cfg.patch_size or cfg.mlp_ratio <= 0:
        raise ValueError("Invalid model dimensions")
    if cfg.precision not in ("fp32", "bf16") or cfg.data_source not in ("huggingface", "torchvision"):
        raise ValueError("Invalid precision or data_source")
    if not 0 < cfg.val_fraction < 1 or not 0 <= cfg.warmup_ratio < 1:
        raise ValueError("Invalid validation or warmup fraction")
    if not 0 <= cfg.min_lr_ratio <= 1 or not 0 <= cfg.label_smoothing < 1:
        raise ValueError("Invalid LR floor or label smoothing")
    if cfg.lr <= 0 or cfg.hyper_lr_multiplier <= 0 or cfg.max_grad_norm <= 0 or cfg.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if not math.isfinite(cfg.adapter_gain) or cfg.adapter_gain <= 0:
        raise ValueError("Adapter gain must be finite and positive")
    if cfg.variant in ("plain", "diagonal", "exact") and cfg.adapter_gain != 1:
        raise ValueError("Non-unit adapter gain requires a low-rank basis/DPLR/static variant")
    if type(cfg.factor_offsets) is not bool or (cfg.factor_offsets and cfg.variant not in ("basis", "dplr")):
        raise ValueError("Factor offsets require a basis/DPLR variant and a boolean setting")
    if cfg.diagnostic_steps:
        steps = [int(step) for step in cfg.diagnostic_steps.split(",")]
        if steps != sorted(set(steps)) or not steps or steps[0] != 0:
            raise ValueError("Diagnostic steps must be sorted, unique and include zero")
    if cfg.num_workers < 0 or cfg.benchmark_warmup_steps < 0:
        raise ValueError("Worker count and benchmark warmup must be nonnegative")
    if cfg.compile and cfg.variant == "exact":
        raise ValueError("Exact custom-autograd chunk loops stay eager; compile factorized variants instead")
    if cfg.save_weights and not cfg.artifact_purpose:
        raise ValueError("--save-weights requires --artifact-purpose")
    return cfg


def stratified_split(labels, fraction, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    labels = np.asarray(labels)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, round(len(indices) * fraction))
        if count >= len(indices):
            raise ValueError("Each class needs at least two examples for a held-out split")
        val.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class TransformedSubset(Dataset):
    def __init__(self, source, indices, transform, huggingface=False):
        self.source, self.indices, self.transform, self.huggingface = source, indices, transform, huggingface

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        example = self.source[self.indices[index]]
        image, label = (example["img"], example["label"]) if self.huggingface else example
        return self.transform(image), int(label)


class SyntheticImages(Dataset):
    """Deterministic random data for engineering checks; no quality conclusions."""
    def __init__(self, count, seed):
        self.count, self.seed = count, seed

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = torch.Generator().manual_seed(self.seed + index)
        return torch.randn(3, 32, 32, generator=rng), index % 10


def _hf_split(cfg, split):
    from datasets import Dataset as HFDataset, load_dataset

    # Read an existing Arrow cache directly to avoid Hub metadata/network calls.
    cache_root = Path(cfg.hf_home) / "datasets" / "uoft-cs___cifar10"
    candidates = sorted(cache_root.glob(f"**/cifar10-{split}.arrow"))
    if len(candidates) == 1:
        path = candidates[0]
        return HFDataset.from_file(str(path)), str(path)
    if not cfg.download:
        raise FileNotFoundError(f"Expected one cached CIFAR-10 {split} Arrow file under {cache_root}; "
                                "use --hf-home, --data-source torchvision, or --download")
    source = load_dataset("uoft-cs/cifar10", split=split, cache_dir=str(Path(cfg.hf_home) / "datasets"))
    return source, source._fingerprint


def build_datasets(cfg):
    if cfg.synthetic:
        train = SyntheticImages(cfg.synthetic_samples, cfg.seed + 100_000)
        val = SyntheticImages(min(cfg.synthetic_samples, cfg.val_limit or cfg.synthetic_samples), cfg.seed + 200_000)
        return train, val, None, {"source": "synthetic", "quality_metrics_meaningful": False}
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    training = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), normalize])
    evaluation = transforms.Compose([transforms.ToTensor(), normalize])
    hf = cfg.data_source == "huggingface"
    if hf:
        source, location = _hf_split(cfg, "train")
        labels = source["label"]
    else:
        source = datasets.CIFAR10(cfg.data_dir, train=True, download=cfg.download)
        location, labels = str(Path(cfg.data_dir).resolve()), source.targets
    train_idx, val_idx = stratified_split(labels, cfg.val_fraction, cfg.split_seed)
    train_idx, val_idx = train_idx[:cfg.train_limit], val_idx[:cfg.val_limit]
    train = TransformedSubset(source, train_idx, training, hf)
    val = TransformedSubset(source, val_idx, evaluation, hf)
    test = None
    if cfg.evaluate_test:
        test_source = _hf_split(cfg, "test")[0] if hf else datasets.CIFAR10(cfg.data_dir, train=False, download=cfg.download)
        test = TransformedSubset(test_source, list(range(len(test_source))), evaluation, hf)
    metadata = {"source": cfg.data_source, "location": location, "train_examples": len(train),
                "validation_examples": len(val), "test_evaluated": cfg.evaluate_test,
                "train_indices": train_idx, "validation_indices": val_idx,
                "source_fingerprint": getattr(source, "_fingerprint", None)}
    return train, val, test, metadata


def build_model(cfg):
    return ViT(img_size=32, patch_size=cfg.patch_size, num_classes=10, embed_dim=cfg.width,
               depth=cfg.depth, heads=cfg.heads, mlp_ratio=cfg.mlp_ratio, qk_norm=cfg.qk_norm,
               drop_rate=cfg.drop_rate, gradient_checkpointing=cfg.gradient_checkpointing,
               conditioning=cfg.variant, conditioning_placement=cfg.placement,
               condition_rank=cfg.rank, condition_n_basis=cfg.n_basis, condition_chunk=cfg.chunk,
               detach_condition=cfg.detach_condition, condition_gain=cfg.adapter_gain,
               condition_factor_offsets=cfg.factor_offsets)


def hyperparameter_ids(model):
    ids = set()
    for module in model.modules():
        if isinstance(module, ConditionedLinear):
            if module.variant in ("diagonal", "static_lora_diagonal"):
                base = module.base
            elif module.variant == "dplr":
                base = module.transform.lowrank.base
            else:
                base = module.transform.base
            ids.update({id(p) for p in module.parameters()} - {id(p) for p in base.parameters()})
    return ids


def model_stats(model, cfg):
    hyper_ids = hyperparameter_ids(model)
    params = sum(p.numel() for p in model.parameters())
    extra = sum(p.numel() for p in model.parameters() if id(p) in hyper_ids)
    n, d, h = (32 // cfg.patch_size) ** 2 + 1, cfg.width, int(cfg.width * cfg.mlp_ratio)
    base_flops = ((n - 1) * 2 * d * 3 * cfg.patch_size ** 2
                  + cfg.depth * (8 * n * d * d + 4 * n * d * h + 4 * n * n * d) + 2 * d * 10)
    extra_flops = n * sum(m.additional_matmul_flops_per_token() for m in model.modules()
                         if isinstance(m, ConditionedLinear))
    return {"parameters": params, "extra_parameters": extra, "base_parameters": params - extra,
            "tokens_per_image": n, "forward_matmul_flops_per_image_estimate": base_flops + extra_flops,
            "extra_forward_matmul_flops_per_token_estimate": extra_flops / n,
            "flop_estimate_excludes": "elementwise operations, reductions, normalization, backward"}


def make_optimizer(model, cfg):
    hyper_ids = hyperparameter_ids(model)
    factor_bias_ids = {id(p) for name, p in model.named_parameters()
                       if name.endswith(('U_offset', 'V_offset'))}
    groups = []
    for hyper in (False, True):
        for decay in (False, True):
            params = [p for p in model.parameters() if (id(p) in hyper_ids) == hyper
                      and (p.ndim >= 2 and id(p) not in factor_bias_ids) == decay]
            if params:
                groups.append({"params": params, "lr": cfg.lr * (cfg.hyper_lr_multiplier if hyper else 1),
                               "weight_decay": cfg.weight_decay if decay else 0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=cfg.device.startswith("cuda"))


def learning_rate_scale(cfg, step, schedule_steps):
    """Zero-based step; a pilot can retain the future training schedule horizon."""
    warmup_steps = int(schedule_steps * cfg.warmup_ratio)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, schedule_steps - warmup_steps - 1)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def _seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, cfg, training):
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=training, num_workers=cfg.num_workers,
                      pin_memory=cfg.device.startswith("cuda"), persistent_workers=cfg.num_workers > 0,
                      worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(cfg.seed + int(training)))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device):
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    losses, correct, count = 0.0, 0, 0
    synchronize(device)
    start = time.perf_counter()
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), targets, reduction="sum")
        losses += loss.item()
        correct += (logits.argmax(-1) == targets).sum().item()
        count += len(images)
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {"loss": losses / count, "accuracy": correct / count, "examples": count,
            "seconds": elapsed, "images_per_second": count / elapsed,
            "tokens_per_second": count * ((32 // cfg.patch_size) ** 2 + 1) / elapsed}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def provenance():
    root = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except subprocess.CalledProcessError:
        revision = None
    source_names = ["train_cifar10.py", "vit.py", "conditioning.py", "condlr.py"]
    if (root / "vit_quality_diagnostics.py").exists():
        source_names.append("vit_quality_diagnostics.py")
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "datasets", "pillow")}
    return {"git_revision": revision, "python": platform.python_version(), "torch": torch.__version__,
            "package_versions": packages,
            "cuda_wheel": torch.version.cuda, "source_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in source_names}}


def main(argv=None):
    cfg = load_config(argv)
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.set_float32_matmul_precision("high")
    output = Path(cfg.output_dir or f"runs/{cfg.variant}-r{cfg.rank}-s{cfg.seed}-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise FileExistsError(f"Run records already exist at {output}; choose a new output directory")
    cfg.output_dir = str(output)
    model = build_model(cfg)
    stats = model_stats(model, cfg)
    disk = shutil.disk_usage(output)
    weights_estimate = 4 * stats["parameters"]
    if disk.free < (weights_estimate * 2 if cfg.save_weights else 10 * 1024 ** 2):
        raise RuntimeError("Insufficient disk capacity for this run's declared artifacts")
    manifest = {"purpose": cfg.artifact_purpose or "configs, metrics and engineering/quality evidence",
                "disk_free_bytes_at_start": disk.free, "periodic_checkpoints": False,
                "weights_saved": cfg.save_weights, "expected_weight_bytes": weights_estimate if cfg.save_weights else 0,
                "weight_retention": "one final model until explicitly removed" if cfg.save_weights else "none",
                "compact_record_expected_bytes": "under 10 MiB for default run",
                "files": ["config.json", "provenance.json", "artifact_manifest.json", "dataset.json",
                          "metrics.jsonl", "results.json"] + (["model.pt"] if cfg.save_weights else [])}
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "provenance.json", provenance())
    write_json(output / "artifact_manifest.json", manifest)
    train, val, test, metadata = build_datasets(cfg)
    write_json(output / "dataset.json", metadata)
    train_loader, val_loader = make_loader(train, cfg, True), make_loader(val, cfg, False)
    planned_steps = min(cfg.max_steps or cfg.epochs * len(train_loader), cfg.epochs * len(train_loader))
    schedule_steps = cfg.lr_schedule_steps or planned_steps
    if schedule_steps < planned_steps:
        raise ValueError("LR schedule horizon must cover the actual training budget")
    model = model.to(device)
    optimizer = make_optimizer(model, cfg)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    hyper_ids = hyperparameter_ids(model)
    if cfg.compile:
        model = torch.compile(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stats["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    diagnostic = None
    if cfg.diagnostic_steps:
        from vit_quality_diagnostics import TokenDiagnostics
        diagnostic = TokenDiagnostics(model, cfg, val, device)
        if diagnostic.steps[-1] != planned_steps:
            raise ValueError("Diagnostics must include the actual final training step")
    print(json.dumps({"output_dir": str(output), "planned_steps": planned_steps, **stats}), flush=True)
    step, processed, benchmark_images, benchmark_seconds = 0, 0, 0, 0.0
    best_accuracy, last_val = -1.0, None
    training_peak = {"peak_training_allocated_bytes": None, "peak_training_reserved_bytes": None}
    run_start = time.perf_counter()
    tokens = stats["tokens_per_image"]
    with (output / "metrics.jsonl").open("w", buffering=1) as log:
        def record(value):
            log.write(json.dumps(value, allow_nan=False) + "\n")
            print(json.dumps(value), flush=True)

        if diagnostic:
            record(diagnostic.collect(model, 0))

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            window_images, window_loss, window_correct = 0, 0.0, 0
            synchronize(device)
            window_start = time.perf_counter()
            for images, targets in train_loader:
                factor = learning_rate_scale(cfg, step, schedule_steps)
                for group, lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = lr * factor
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
                    logits = model(images)
                    loss = F.cross_entropy(logits.float(), targets, label_smoothing=cfg.label_smoothing)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                loss.backward()
                report = ((step + 1) % cfg.log_every == 0 or step + 1 == planned_steps)
                hyper_grad_norm = None
                if report and hyper_ids:
                    squares = [p.grad.float().square().sum() for p in model.parameters()
                               if id(p) in hyper_ids and p.grad is not None]
                    hyper_grad_norm = torch.stack(squares).sum().sqrt().item() if squares else 0.0
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                processed += len(images)
                if diagnostic:
                    diagnostic.observe_gradients(model, step)
                    if step in diagnostic.steps:
                        if device.type == "cuda":
                            training_peak["peak_training_allocated_bytes"] = max(training_peak["peak_training_allocated_bytes"] or 0, torch.cuda.max_memory_allocated(device))
                            training_peak["peak_training_reserved_bytes"] = max(training_peak["peak_training_reserved_bytes"] or 0, torch.cuda.max_memory_reserved(device))
                        synchronize(device)
                        diagnostic_start = time.perf_counter()
                        record(diagnostic.collect(model, step, images, targets))
                        synchronize(device)
                        window_start += time.perf_counter() - diagnostic_start
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                window_images += len(images)
                window_loss += loss.item() * len(images)
                window_correct += (logits.argmax(-1) == targets).sum().item()
                # Warmup exclusion uses an explicit boundary, including dataloader
                # and optimizer costs. Validation is outside all training windows.
                if step == cfg.benchmark_warmup_steps:
                    synchronize(device)
                    window_start = time.perf_counter()
                    window_images, window_loss, window_correct = 0, 0.0, 0
                if report and window_images:
                    synchronize(device)
                    elapsed = time.perf_counter() - window_start
                    measured = step > cfg.benchmark_warmup_steps
                    if measured:
                        benchmark_images += window_images
                        benchmark_seconds += elapsed
                    record({"event": "train", "step": step, "epoch": epoch,
                            "loss": window_loss / window_images, "accuracy": window_correct / window_images,
                            "window_images": window_images, "window_seconds": elapsed,
                            "images_per_second": window_images / elapsed,
                            "tokens_per_second": window_images * tokens / elapsed,
                            "benchmark_window": measured, "lr": optimizer.param_groups[0]["lr"],
                            "grad_norm_before_clip": grad_norm.item(), "hyper_grad_norm_before_clip": hyper_grad_norm,
                            **memory_stats(device)})
                    window_images, window_loss, window_correct = 0, 0.0, 0
                    window_start = time.perf_counter()
                if step >= planned_steps:
                    break
            # Capture any tail training window before validation interrupts it.
            if window_images and step > cfg.benchmark_warmup_steps:
                synchronize(device)
                elapsed = time.perf_counter() - window_start
                benchmark_images += window_images
                benchmark_seconds += elapsed
            if device.type == "cuda":
                training_peak["peak_training_allocated_bytes"] = max(
                    training_peak["peak_training_allocated_bytes"] or 0, torch.cuda.max_memory_allocated(device))
                training_peak["peak_training_reserved_bytes"] = max(
                    training_peak["peak_training_reserved_bytes"] or 0, torch.cuda.max_memory_reserved(device))
            last_val = evaluate(model, val_loader, cfg, device)
            best_accuracy = max(best_accuracy, last_val["accuracy"])
            record({"event": "validation", "step": step, "epoch": epoch, **last_val})
            if step >= planned_steps:
                break
        test_metrics = evaluate(model, make_loader(test, cfg, False), cfg, device) if test is not None else None
        if cfg.save_weights:
            raw = getattr(model, "_orig_mod", model)
            torch.save(raw.state_dict(), output / "model.pt")
        results = {"variant": cfg.variant, "rank": cfg.rank, "n_basis": cfg.n_basis,
                   "factor_offsets": cfg.factor_offsets,
                   "training_setup": {"lr_schedule_steps": schedule_steps,
                                      "warmup_steps": int(schedule_steps * cfg.warmup_ratio),
                                      "adam_betas": [0.9, 0.999], "fused_adamw": cfg.device.startswith("cuda"),
                                      "model_compiled": cfg.compile},
                   "seed": cfg.seed, "completed_steps": step, "processed_images": processed,
                   "elapsed_seconds": time.perf_counter() - run_start, "validation": last_val,
                   "best_validation_accuracy": best_accuracy, "test": test_metrics,
                   "synthetic": cfg.synthetic, "benchmark_warmup_steps": cfg.benchmark_warmup_steps,
                   "benchmark_images": benchmark_images, "benchmark_seconds": benchmark_seconds,
                   "train_images_per_second": benchmark_images / benchmark_seconds if benchmark_seconds else None,
                   "train_tokens_per_second": benchmark_images * tokens / benchmark_seconds if benchmark_seconds else None,
                   **stats, **training_peak}
        if diagnostic:
            results["token_diagnostics"] = diagnostic.summary()
        write_json(output / "results.json", results)
        record({"event": "complete", **results})


if __name__ == "__main__":
    main()

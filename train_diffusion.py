#!/usr/bin/env python3
"""Class-conditional DDPM training for CIFAR-10 and ImageNet-64.

The script intentionally keeps the diffusion experiment conventional: a
diffusers UNet2DModel predicts epsilon for a DDPMScheduler. Configuration is
loaded from ``--override_json`` so paper launchers can produce reviewed,
reproducible run files.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from csv_logger import CSVLogger
from snr_adam import SNRAdam


LOGGER = logging.getLogger(__name__)


def get_default_config() -> dict[str, Any]:
    """Return defaults shared by manual runs and generated paper configs."""
    return {
        # Data and model. Dataset-specific architecture defaults are resolved
        # after JSON overrides have been applied.
        "dataset_name": "cifar10",  # cifar10 | imagenet64 | synthetic
        "data_path": "./data",
        "cifar_download": False,
        "cifar_source": "huggingface",  # huggingface | torchvision
        "image_size": None,
        "num_classes": None,
        "num_workers": 4,
        "prefetch_factor": 2,
        "max_train_samples": None,
        "synthetic_num_samples": 128,
        "block_out_channels": None,
        "down_block_types": None,
        "up_block_types": None,
        "layers_per_block": 2,
        "norm_num_groups": None,
        "attention_head_dim": 8,
        "dropout": 0.0,
        "gradient_checkpointing": False,
        "compile": False,
        "compile_mode": "default",
        "compile_optimizer": False,
        # Diffusion objective.
        "prediction_type": "epsilon",
        "num_train_timesteps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        # Training.
        "per_device_train_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "max_train_steps": 200_000,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "max_grad_norm": 1.0,
        "num_warmup_steps": 5_000,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.0,
        "mixed_precision": "bf16",
        "cpu": False,
        "seed": 123,
        "mode": "adamw",  # adamw | snr_adam
        "fused_adamw": True,
        # Retained SNRAdam surface, kept in parity with train_llm.py.
        "boost_min": 0.25,
        "boost_max": 4.0,
        "normalize_mode": "global",
        "snr_mode": "coherence",
        "score_beta": None,
        "boost_center_mode": "mean_log",
        "boost_center_scope": "global",
        "boost_center_every": 10,
        "snr_whiten_stat": "winsor",
        "snr_whiten_every": 10,
        "grad_z_clip": 4.0,
        "momentum_mode": "snr_adaptive",
        "adaptive_beta1_fast": 0.9,
        "adaptive_beta1_slow": 0.99,
        "power_warmup_steps": 100,
        "slow_add_alpha": 0.5,
        "track_null_diagnostics": True,
        "diagnostics_every": 10,
        "log_snr_metrics_every_n": 100,
        # EMA is deliberately outside optimizer selection so it is updated at
        # exactly the same point for AdamW and every SNRAdam configuration.
        "use_ema": True,
        "ema_decay": 0.9999,
        "ema_update_after_step": 0,
        # Logging, output, and checkpoints.
        "base_output_dir": "model-output/diffusion",
        "run_name": None,
        "with_tracking": False,
        "report_to": "wandb",
        "wandb_project": "SNRAdam-Diffusion",
        "wandb_run_id": None,
        "wandb_resume": "allow",
        "csv_logging": True,
        "csv_flush_every": 100,
        "log_every_n": 10,
        "checkpointing_steps": 10_000,
        "checkpoints_total_limit": 1,
        "resume_from_checkpoint": None,
        "auto_resume": False,
        # Deterministic qualitative and quantitative evaluation.
        "sample_interval": 5_000,
        "num_sample_images": 64,
        "sample_batch_size": 64,
        "sample_inference_steps": 100,
        "sample_seed": 2027,
        "evaluate_fid_is": False,
        "require_fid_dependencies": False,
        "eval_interval": 0,  # 0 means final-only
        "eval_num_images": 50_000,
        "eval_reference_num_images": None,  # None uses the full reference dataset.
        "eval_batch_size": 128,
        "eval_num_workers": 4,
        "eval_seed": 2027,
        "eval_sampler": "ddpm",
        "eval_inference_steps": 100,
        "eval_inception_splits": 10,
        "fid_real_stats_cache_dir": None,  # Defaults to <data_path>/fid_stats.
        "paper_phase": None,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--override_json", type=Path, default=None)
    return parser.parse_args(argv)


def load_config(argv: Sequence[str] | None = None) -> dict[str, Any]:
    cli = _parse_args(argv)
    cfg = get_default_config()
    if cli.override_json is not None:
        with cli.override_json.open() as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            raise ValueError("--override_json must contain a JSON object")
        cfg.update(overrides)
        if not cfg.get("run_name"):
            cfg["run_name"] = cli.override_json.stem
    return resolve_config(cfg)


def resolve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve dataset-specific defaults and reject unsupported experiment drift."""
    cfg = dict(config)
    dataset_name = str(cfg["dataset_name"]).lower()
    profiles = {
        "cifar10": {
            "image_size": 32,
            "num_classes": 10,
            "norm_num_groups": 32,
            "block_out_channels": [128, 128, 256, 256],
            "down_block_types": [
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ],
            "up_block_types": [
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
        },
        "imagenet64": {
            "image_size": 64,
            "num_classes": 1000,
            "norm_num_groups": 32,
            "block_out_channels": [128, 128, 256, 256, 512],
            "down_block_types": [
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ],
            "up_block_types": [
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
        },
        "synthetic": {
            "image_size": 8,
            "num_classes": 4,
            "norm_num_groups": 8,
            "block_out_channels": [16, 32],
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
        },
    }
    if dataset_name not in profiles:
        raise ValueError("dataset_name must be one of: cifar10, imagenet64, synthetic")
    cfg["dataset_name"] = dataset_name
    for key, value in profiles[dataset_name].items():
        if cfg.get(key) is None:
            cfg[key] = value

    if cfg["prediction_type"] != "epsilon":
        raise ValueError("This staged experiment supports epsilon prediction only")
    if cfg["mode"] == "base":
        cfg["mode"] = "adamw"
    if cfg["mode"] not in {"adamw", "snr_adam"}:
        raise ValueError("mode must be adamw or snr_adam")
    if len(cfg["block_out_channels"]) != len(cfg["down_block_types"]):
        raise ValueError("block_out_channels and down_block_types must have equal lengths")
    if len(cfg["block_out_channels"]) != len(cfg["up_block_types"]):
        raise ValueError("block_out_channels and up_block_types must have equal lengths")
    if cfg["dataset_name"] == "imagenet64" and not cfg.get("data_path"):
        raise ValueError("ImageNet-64 requires data_path pointing to an ImageFolder root")
    if int(cfg["max_train_steps"]) < 1:
        raise ValueError("max_train_steps must be positive")
    if (
        cfg.get("checkpoints_total_limit") is not None
        and int(cfg["checkpoints_total_limit"]) < 1
    ):
        raise ValueError("checkpoints_total_limit must be positive or None")
    if cfg.get("eval_sampler") != "ddpm":
        raise ValueError("eval_sampler must be ddpm for this experiment")
    if int(cfg["eval_num_images"]) < 1:
        raise ValueError("eval_num_images must be positive")
    if int(cfg["eval_inference_steps"]) < 1:
        raise ValueError("eval_inference_steps must be positive")
    if cfg.get("require_fid_dependencies") and not cfg.get("evaluate_fid_is"):
        raise ValueError(
            "require_fid_dependencies=True requires evaluate_fid_is=True"
        )
    if cfg.get("paper_phase"):
        if int(cfg["eval_num_images"]) != 50_000:
            raise ValueError("Paper phases require exactly 50,000 generated images")
        if int(cfg["eval_interval"]) <= 0:
            raise ValueError("Paper phases require checkpoint-interval FID/IS")
        if not cfg.get("require_fid_dependencies"):
            raise ValueError("Paper phases must fail loudly when FID/IS is unavailable")
    if cfg.get("run_name") is None:
        cfg["run_name"] = (
            f"{dataset_name}-{cfg['mode']}-lr{cfg['learning_rate']}-s{cfg['seed']}"
        )
    cfg["output_dir"] = str(Path(cfg["base_output_dir"]) / str(cfg["run_name"]))
    return cfg


class SyntheticClassDataset(Dataset):
    """Deterministic no-network dataset used only for tests and smoke debugging."""

    def __init__(self, length: int, image_size: int, num_classes: int, seed: int):
        self.length = int(length)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(self.seed + int(index))
        image = torch.rand(
            3, self.image_size, self.image_size, generator=generator
        ).mul_(2).sub_(1)
        return image, int(index) % self.num_classes


class HuggingFaceCIFAR10(Dataset):
    """CIFAR-10 backed by the maintained Hugging Face parquet mirror."""

    def __init__(self, root: str | Path, transform: Any):
        from datasets import load_dataset

        self.dataset = load_dataset(
            "uoft-cs/cifar10",
            "plain_text",
            split="train",
            cache_dir=str(Path(root) / "huggingface"),
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        example = self.dataset[int(index)]
        return self.transform(example["img"]), int(example["label"])


def build_image_transform(cfg: Mapping[str, Any], training: bool) -> transforms.Compose:
    operations: list[Any] = []
    image_size = int(cfg["image_size"])
    if cfg["dataset_name"] == "imagenet64":
        operations.extend(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
            ]
        )
    if training:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transforms.Compose(operations)


def build_dataset(cfg: Mapping[str, Any], training: bool = True) -> Dataset:
    name = cfg["dataset_name"]
    if name == "synthetic":
        dataset: Dataset = SyntheticClassDataset(
            cfg["synthetic_num_samples"],
            cfg["image_size"],
            cfg["num_classes"],
            cfg["seed"] + (0 if training else 1_000_000),
        )
    elif name == "cifar10":
        # The 50k train split is also the standard real reference for FID-50k.
        transform = build_image_transform(cfg, training=training)
        if cfg.get("cifar_source", "huggingface") == "huggingface":
            dataset = HuggingFaceCIFAR10(cfg["data_path"], transform)
        else:
            dataset = datasets.CIFAR10(
                root=str(cfg["data_path"]),
                train=True,
                transform=transform,
                download=bool(cfg["cifar_download"]),
            )
    else:
        root = Path(str(cfg["data_path"]))
        if not root.is_dir():
            raise FileNotFoundError(f"ImageNet-64 ImageFolder root not found: {root}")
        dataset = datasets.ImageFolder(
            root=str(root), transform=build_image_transform(cfg, training=training)
        )
        discovered_classes = len(dataset.classes)
        if discovered_classes != int(cfg["num_classes"]):
            raise ValueError(
                f"Expected {cfg['num_classes']} ImageFolder classes, found "
                f"{discovered_classes} under {root}"
            )

    max_samples = cfg.get("max_train_samples") if training else None
    if max_samples is not None and int(max_samples) < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(int(max_samples)))
    return dataset


def build_dataloader(
    dataset: Dataset,
    cfg: Mapping[str, Any],
    *,
    training: bool,
) -> DataLoader:
    workers = int(cfg["num_workers"] if training else cfg["eval_num_workers"])
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(
            cfg["per_device_train_batch_size"] if training else cfg["eval_batch_size"]
        ),
        "shuffle": training,
        "drop_last": training,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available() and not bool(cfg.get("cpu", False)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(cfg["prefetch_factor"]))
    return DataLoader(**kwargs)


def build_model(cfg: Mapping[str, Any]) -> torch.nn.Module:
    try:
        from diffusers import UNet2DModel
    except ImportError as exc:
        raise ImportError(
            "train_diffusion.py requires diffusers; install requirements.txt"
        ) from exc

    model = UNet2DModel(
        sample_size=int(cfg["image_size"]),
        in_channels=3,
        out_channels=3,
        layers_per_block=int(cfg["layers_per_block"]),
        block_out_channels=tuple(int(x) for x in cfg["block_out_channels"]),
        down_block_types=tuple(cfg["down_block_types"]),
        up_block_types=tuple(cfg["up_block_types"]),
        num_class_embeds=int(cfg["num_classes"]),
        norm_num_groups=int(cfg["norm_num_groups"]),
        attention_head_dim=int(cfg["attention_head_dim"]),
        dropout=float(cfg["dropout"]),
    )
    if cfg["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    return model


def build_noise_scheduler(cfg: Mapping[str, Any]):
    try:
        from diffusers import DDPMScheduler
    except ImportError as exc:
        raise ImportError(
            "train_diffusion.py requires diffusers; install requirements.txt"
        ) from exc
    return DDPMScheduler(
        num_train_timesteps=int(cfg["num_train_timesteps"]),
        beta_schedule=str(cfg["beta_schedule"]),
        prediction_type="epsilon",
        clip_sample=bool(cfg["clip_sample"]),
    )


def make_parameter_groups(
    model: torch.nn.Module, weight_decay: float
) -> tuple[list[dict[str, Any]], dict[torch.nn.Parameter, str]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    param_to_name: dict[torch.nn.Parameter, str] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_to_name[parameter] = name
        if parameter.ndim == 1 or "embed" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return (
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        param_to_name,
    )


def create_optimizer(
    model: torch.nn.Module, cfg: Mapping[str, Any]
) -> tuple[torch.optim.Optimizer, dict[torch.nn.Parameter, str]]:
    groups, param_to_name = make_parameter_groups(model, float(cfg["weight_decay"]))
    common = {
        "lr": float(cfg["learning_rate"]),
        "betas": (float(cfg["beta1"]), float(cfg["beta2"])),
        "eps": float(cfg["eps"]),
        "weight_decay": float(cfg["weight_decay"]),
    }
    if cfg["mode"] == "adamw":
        fused = (
            bool(cfg["fused_adamw"])
            and torch.cuda.is_available()
            and not bool(cfg.get("cpu", False))
        )
        return torch.optim.AdamW(groups, fused=fused, **common), param_to_name

    optimizer = SNRAdam(
        groups,
        boost_min=float(cfg["boost_min"]),
        boost_max=float(cfg["boost_max"]),
        normalize_mode=str(cfg["normalize_mode"]),
        snr_mode=str(cfg["snr_mode"]),
        score_beta=cfg.get("score_beta"),
        boost_center_mode=str(cfg["boost_center_mode"]),
        boost_center_scope=str(cfg["boost_center_scope"]),
        boost_center_every=int(cfg["boost_center_every"]),
        snr_whiten_stat=str(cfg["snr_whiten_stat"]),
        snr_whiten_every=int(cfg["snr_whiten_every"]),
        grad_z_clip=float(cfg["grad_z_clip"]),
        momentum_mode=str(cfg["momentum_mode"]),
        adaptive_beta1_fast=float(cfg["adaptive_beta1_fast"]),
        adaptive_beta1_slow=float(cfg["adaptive_beta1_slow"]),
        warmup_steps=int(cfg["power_warmup_steps"]),
        slow_add_alpha=float(cfg["slow_add_alpha"]),
        track_null_diagnostics=bool(cfg["track_null_diagnostics"]),
        diagnostics_every=int(cfg["diagnostics_every"]),
        compile_optimizer=bool(cfg.get("compile_optimizer", False)),
        **common,
    )
    return optimizer, param_to_name


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any]
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = int(cfg["num_warmup_steps"])
    total = int(cfg["max_train_steps"])
    minimum = float(cfg["min_lr_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = min(1.0, (step - warmup) / float(max(1, total - warmup)))
        schedule = str(cfg["lr_scheduler_type"])
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return max(minimum, 1.0 - progress)
        if schedule == "cosine":
            return max(minimum, 0.5 * (1.0 + math.cos(math.pi * progress)))
        raise ValueError(f"Unknown lr_scheduler_type: {schedule}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Small checkpointable EMA with a temporary-weights context manager."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float,
        update_after_step: int = 0,
    ):
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema_decay must satisfy 0 <= decay < 1")
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.optimization_step = 0
        self.shadow_params = [
            parameter.detach().clone() for parameter in parameters if parameter.requires_grad
        ]
        self._stored_params: list[torch.Tensor] | None = None

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        values = [parameter for parameter in parameters if parameter.requires_grad]
        if len(values) != len(self.shadow_params):
            raise ValueError("EMA parameter set changed after construction")
        self.optimization_step += 1
        decay = self.decay if self.optimization_step > self.update_after_step else 0.0
        for shadow, parameter in zip(self.shadow_params, values):
            shadow.lerp_(parameter.detach().to(dtype=shadow.dtype), 1.0 - decay)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        values = [parameter for parameter in parameters if parameter.requires_grad]
        for parameter, shadow in zip(values, self.shadow_params):
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))

    @contextlib.contextmanager
    def average_parameters(
        self, parameters: Iterable[torch.nn.Parameter]
    ) -> Iterator[None]:
        values = [parameter for parameter in parameters if parameter.requires_grad]
        self._stored_params = [parameter.detach().clone() for parameter in values]
        self.copy_to(values)
        try:
            yield
        finally:
            with torch.no_grad():
                for parameter, stored in zip(values, self._stored_params):
                    parameter.copy_(stored)
            self._stored_params = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "optimization_step": self.optimization_step,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.decay = float(state_dict["decay"])
        self.update_after_step = int(state_dict["update_after_step"])
        self.optimization_step = int(state_dict["optimization_step"])
        if len(state_dict["shadow_params"]) != len(self.shadow_params):
            raise ValueError("EMA checkpoint does not match the model")
        for current, loaded in zip(self.shadow_params, state_dict["shadow_params"]):
            current.copy_(loaded.to(device=current.device, dtype=current.dtype))


def diffusion_loss(
    model: torch.nn.Module,
    noise_scheduler: Any,
    clean_images: torch.Tensor,
    class_labels: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the conventional epsilon-prediction denoising objective."""
    if noise is None:
        noise = torch.randn_like(clean_images)
    if timesteps is None:
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
            dtype=torch.long,
        )
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    prediction = model(
        noisy_images, timesteps, class_labels=class_labels
    ).sample
    return F.mse_loss(prediction.float(), noise.float(), reduction="mean")


@torch.no_grad()
def sample_images(
    model: torch.nn.Module,
    noise_scheduler: Any,
    *,
    num_images: int,
    batch_size: int,
    num_classes: int,
    image_size: int,
    num_inference_steps: int,
    seed: int,
    device: torch.device,
    class_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic, balanced-class samples in [0, 1] on CPU."""
    from diffusers import DDPMScheduler

    sampler = DDPMScheduler.from_config(noise_scheduler.config)
    sampler.set_timesteps(int(num_inference_steps), device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    was_training = model.training
    model.eval()
    output: list[torch.Tensor] = []
    output_labels: list[torch.Tensor] = []
    for offset in range(0, int(num_images), int(batch_size)):
        current_batch = min(int(batch_size), int(num_images) - offset)
        labels = (
            torch.arange(
                class_offset + offset,
                class_offset + offset + current_batch,
                device=device,
            )
            % int(num_classes)
        ).long()
        images = torch.randn(
            current_batch,
            3,
            int(image_size),
            int(image_size),
            generator=generator,
            device=device,
        )
        for timestep in sampler.timesteps:
            prediction = model(
                images, timestep, class_labels=labels
            ).sample
            images = sampler.step(
                prediction, timestep, images, generator=generator
            ).prev_sample
        output.append(images.clamp(-1, 1).add(1).div(2).cpu())
        output_labels.append(labels.cpu())
    model.train(was_training)
    return torch.cat(output), torch.cat(output_labels)


def _ema_context(
    ema: ModelEMA | None, model: torch.nn.Module
) -> contextlib.AbstractContextManager:
    if ema is None:
        return contextlib.nullcontext()
    return ema.average_parameters(model.parameters())


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP and, for sampling/export, a torch.compile OptimizedModule."""
    unwrapped = accelerator.unwrap_model(model)
    return getattr(unwrapped, "_orig_mod", unwrapped)


def save_sample_grid(
    model: torch.nn.Module,
    noise_scheduler: Any,
    cfg: Mapping[str, Any],
    output_dir: Path,
    step: int,
    device: torch.device,
    ema: ModelEMA | None,
) -> Path:
    with _ema_context(ema, model):
        images, _ = sample_images(
            model,
            noise_scheduler,
            num_images=int(cfg["num_sample_images"]),
            batch_size=int(cfg["sample_batch_size"]),
            num_classes=int(cfg["num_classes"]),
            image_size=int(cfg["image_size"]),
            num_inference_steps=int(cfg["sample_inference_steps"]),
            seed=int(cfg["sample_seed"]),
            device=device,
        )
    path = output_dir / "samples" / f"step_{step}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=max(1, int(math.sqrt(len(images)))))
    return path


def log_sample_grid(
    accelerator: Accelerator,
    path: Path,
    step: int,
) -> None:
    """Log a saved sample grid to supported experiment trackers."""
    for tracker in accelerator.trackers:
        if getattr(tracker, "name", "") != "wandb":
            continue
        try:
            import wandb

            tracker.log(
                {
                    "eval/sample_grid": wandb.Image(
                        str(path), caption=f"EMA samples at step {step}"
                    )
                },
                step=step,
            )
        except Exception as exc:
            LOGGER.warning("Unable to log sample grid to W&B: %s", exc)


def _cycled_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


_FID_REAL_STATE_NAMES = (
    "real_features_sum",
    "real_features_cov_sum",
    "real_features_num_samples",
)


def _optional_package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _fid_real_cache_metadata(
    real_dataset: Dataset,
    cfg: Mapping[str, Any],
    reference_target: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_name": str(cfg["dataset_name"]),
        "cifar_source": str(cfg.get("cifar_source", "")),
        "dataset_size": len(real_dataset),
        "reference_num_images": reference_target,
        "image_size": int(cfg["image_size"]),
        "feature": 2048,
        "normalize": True,
        "input_transform": "minus_one_one_to_zero_one",
        "torchmetrics_version": _optional_package_version("torchmetrics"),
        "torch_fidelity_version": _optional_package_version("torch-fidelity"),
    }


def _fid_real_cache_path(
    real_dataset: Dataset,
    cfg: Mapping[str, Any],
    reference_target: int,
) -> tuple[Path, dict[str, Any]]:
    metadata = _fid_real_cache_metadata(real_dataset, cfg, reference_target)
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    configured_dir = cfg.get("fid_real_stats_cache_dir")
    cache_dir = (
        Path(str(configured_dir))
        if configured_dir
        else Path(str(cfg["data_path"])) / "fid_stats"
    )
    dataset_name = str(cfg["dataset_name"]).replace("/", "-")
    return cache_dir / f"{dataset_name}-fid2048-{digest}.pt", metadata


def _restore_fid_real_state(
    fid: Any,
    cache_path: Path,
    expected_metadata: Mapping[str, Any],
    device: torch.device,
) -> bool:
    if not cache_path.is_file():
        return False
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("metadata") != dict(expected_metadata):
            return False
        states = payload["states"]
        for name in _FID_REAL_STATE_NAMES:
            current = getattr(fid, name)
            loaded = states[name]
            if current.shape != loaded.shape:
                return False
            current.copy_(loaded.to(device=device, dtype=current.dtype))
        return True
    except Exception as exc:
        LOGGER.warning("Ignoring invalid FID real-stat cache %s: %s", cache_path, exc)
        return False


def _save_fid_real_state(
    fid: Any,
    cache_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
    )
    payload = {
        "metadata": dict(metadata),
        "states": {
            name: getattr(fid, name).detach().cpu()
            for name in _FID_REAL_STATE_NAMES
        },
    }
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _create_fid_metrics(
    cfg: Mapping[str, Any], device: torch.device
) -> tuple[Any, Any]:
    """Construct the exact paper metrics, raising on any missing dependency."""
    try:
        import torch_fidelity  # noqa: F401 - required backend for torchmetrics
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
    except Exception as exc:
        raise RuntimeError(
            "FID/IS requires torchmetrics[image] and torch-fidelity; "
            "install the paper requirements before launching"
        ) from exc
    return (
        FrechetInceptionDistance(feature=2048, normalize=True).to(device),
        InceptionScore(
            normalize=True, splits=int(cfg["eval_inception_splits"])
        ).to(device),
    )


def verify_fid_dependencies(cfg: Mapping[str, Any]) -> None:
    """Fail before training if a required paper metric cannot be constructed."""
    if not cfg.get("evaluate_fid_is"):
        return
    try:
        _create_fid_metrics(cfg, torch.device("cpu"))
    except Exception:
        if cfg.get("require_fid_dependencies"):
            raise
        LOGGER.warning("Optional FID/IS dependencies are unavailable", exc_info=True)


@torch.no_grad()
def evaluate_generative_metrics(
    model: torch.nn.Module,
    noise_scheduler: Any,
    real_dataset: Dataset,
    cfg: Mapping[str, Any],
    device: torch.device,
    ema: ModelEMA | None,
) -> tuple[dict[str, float], str | None]:
    """Stream the fixed FID/IS protocol, failing loudly for paper phases."""
    try:
        fid, inception = _create_fid_metrics(cfg, device)
        target = int(cfg["eval_num_images"])
        configured_reference_target = cfg.get("eval_reference_num_images")
        reference_target = (
            int(configured_reference_target)
            if configured_reference_target is not None
            else len(real_dataset)
        )
        cache_path, cache_metadata = _fid_real_cache_path(
            real_dataset, cfg, reference_target
        )
        cache_hit = _restore_fid_real_state(
            fid, cache_path, cache_metadata, device
        )
        if cache_hit:
            LOGGER.info("Loaded cached FID real statistics from %s", cache_path)
        else:
            real_loader = build_dataloader(real_dataset, cfg, training=False)
            real_batches = _cycled_batches(real_loader)
            seen = 0
            while seen < reference_target:
                real_images, _ = next(real_batches)
                real_images = real_images[: reference_target - seen].to(device)
                real_images = real_images.clamp(-1, 1).add(1).div(2)
                fid.update(real_images, real=True)
                seen += real_images.shape[0]
            _save_fid_real_state(fid, cache_path, cache_metadata)
            LOGGER.info("Cached FID real statistics at %s", cache_path)

        generated = 0
        with _ema_context(ema, model):
            while generated < target:
                count = min(int(cfg["eval_batch_size"]), target - generated)
                fake_images, _ = sample_images(
                    model,
                    noise_scheduler,
                    num_images=count,
                    batch_size=count,
                    num_classes=int(cfg["num_classes"]),
                    image_size=int(cfg["image_size"]),
                    num_inference_steps=int(cfg["eval_inference_steps"]),
                    seed=int(cfg["eval_seed"]) + generated,
                    device=device,
                    class_offset=generated,
                )
                fake_images = fake_images.to(device)
                fid.update(fake_images, real=False)
                inception.update(fake_images)
                generated += count
        fid_value = float(fid.compute().item())
        is_mean, is_std = inception.compute()
        fid_label = (
            f"{target // 1000}k" if target % 1000 == 0 else str(target)
        )
        return {
            f"eval/fid_{fid_label}": fid_value,
            "eval/inception_score": float(is_mean.item()),
            "eval/inception_score_std": float(is_std.item()),
            "eval/num_images": float(target),
            "eval/seed": float(cfg["eval_seed"]),
            "eval/inference_steps": float(cfg["eval_inference_steps"]),
        }, None
    except Exception as exc:
        if cfg.get("require_fid_dependencies"):
            raise RuntimeError("Required paper FID/IS evaluation failed") from exc
        return {}, f"FID/IS evaluation unavailable: {exc}"


def collect_snr_diagnostics(
    optimizer: torch.optim.Optimizer,
    param_to_name: Mapping[torch.nn.Parameter, str],
    cfg: Mapping[str, Any],
) -> dict[str, float]:
    """Collect retained score, boost, momentum, clipping, and slow metrics."""
    del param_to_name, cfg
    opt = getattr(optimizer, "optimizer", optimizer)
    logs: dict[str, float] = {}
    if hasattr(opt, "get_diagnostics"):
        for param_key, diagnostic in opt.get_diagnostics().items():
            for key, value in diagnostic.items():
                if not isinstance(value, (int, float)) or value is None:
                    continue
                if key.startswith("null_"):
                    prefix = "snr_null"
                elif key.startswith("beta1_eff_"):
                    prefix = "snr_momentum"
                elif key == "grad_clip_fraction":
                    prefix = "snr_clipping"
                elif key.startswith("slow_") or key == "final_update_rms":
                    prefix = "snr_slow"
                elif key.startswith("signal_") or key.startswith("score_"):
                    prefix = "snr_score"
                elif key.startswith("boost_") or key.startswith("global_"):
                    prefix = "snr_boost"
                else:
                    continue
                logs[f"{prefix}/{key}/{param_key}"] = float(value)
    return logs


def summarize_diagnostics(logs: Mapping[str, float]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for key, value in logs.items():
        parts = key.split("/")
        if len(parts) >= 2 and math.isfinite(value):
            groups["/".join(parts[:2])].append(value)
    return {
        f"summary/{key}": sum(values) / len(values)
        for key, values in groups.items()
        if values
    }


def _checkpoint_step(path: Path) -> int:
    state_path = path / "training_state.json"
    if state_path.is_file():
        return int(json.loads(state_path.read_text())["completed_steps"])
    if path.name.startswith("step_"):
        return int(path.name.removeprefix("step_"))
    raise ValueError(f"Cannot infer completed step from checkpoint: {path}")


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    state_path = path / "training_state.json"
    if not state_path.is_file():
        return {"completed_steps": _checkpoint_step(path)}
    payload = json.loads(state_path.read_text())
    if "completed_steps" not in payload:
        raise ValueError(f"Checkpoint metadata has no completed_steps: {state_path}")
    return payload


def latest_checkpoint(output_dir: Path, maximum_step: int) -> Path | None:
    """Return the newest valid checkpoint at or below the requested horizon."""
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("step_*"):
        if not path.is_dir():
            continue
        try:
            step = _checkpoint_step(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if step <= int(maximum_step):
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def prune_checkpoints(output_dir: Path, total_limit: int | None) -> list[Path]:
    """Remove oldest resumable checkpoints while retaining the newest states."""
    if total_limit is None:
        return []
    limit = int(total_limit)
    if limit < 1:
        raise ValueError("total_limit must be positive or None")

    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("step_*"):
        if not path.is_dir():
            continue
        try:
            candidates.append((_checkpoint_step(path), path))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue

    removed: list[Path] = []
    for _, path in sorted(candidates)[:-limit]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def resolve_resume_checkpoint(
    cfg: Mapping[str, Any], output_dir: Path
) -> Path | None:
    requested = cfg.get("resume_from_checkpoint")
    if requested and str(requested).lower() not in {"auto", "latest"}:
        path = Path(str(requested))
        if not path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path
    if requested or cfg.get("auto_resume"):
        return latest_checkpoint(output_dir, int(cfg["max_train_steps"]))
    return None


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Count unique tensor storage used by initialized optimizer state."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    seen: set[int] = set()
    total = 0

    def visit(value: Any) -> None:
        nonlocal total
        if torch.is_tensor(value):
            identity = id(value)
            if identity not in seen:
                seen.add(identity)
                total += value.numel() * value.element_size()
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    visit(getattr(optimizer, "state", {}))
    return total


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    completed_steps: int,
    *,
    elapsed_seconds: float = 0.0,
    processed_images: int = 0,
    peak_allocated_bytes: int = 0,
    peak_reserved_bytes: int = 0,
    optimizer_bytes: int = 0,
    checkpoints_total_limit: int | None = 1,
) -> Path:
    checkpoint_dir = output_dir / f"step_{completed_steps}"
    accelerator.save_state(str(checkpoint_dir))
    if accelerator.is_main_process:
        (checkpoint_dir / "training_state.json").write_text(
            json.dumps(
                {
                    "completed_steps": completed_steps,
                    "elapsed_seconds": elapsed_seconds,
                    "processed_images": processed_images,
                    "peak_allocated_bytes": peak_allocated_bytes,
                    "peak_reserved_bytes": peak_reserved_bytes,
                    "optimizer_state_bytes": optimizer_bytes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        removed = prune_checkpoints(output_dir, checkpoints_total_limit)
        for path in removed:
            LOGGER.info("Removed superseded checkpoint %s", path)
    accelerator.wait_for_everyone()
    return checkpoint_dir


def _memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    gib = 1024**3
    return {
        "memory/allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "memory/reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "memory/max_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "memory/max_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def _peak_memory_bytes(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _load_evaluated_steps(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    steps: set[int] = set()
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            steps.add(int(row["step"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            LOGGER.warning("Ignoring invalid evaluation record in %s", path)
    return steps


def _record_evaluation(path: Path, step: int, metrics: Mapping[str, float]) -> None:
    row = {"step": int(step), **dict(metrics)}
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _numeric_for_csv(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value) and value.numel() == 1:
            value = value.detach().item()
        if isinstance(value, (int, float)):
            result[key] = value
    return result


def _resume_safe_csv_logger(path: Path, flush_every: int) -> CSVLogger:
    """Continue an existing metrics file without truncating its prior rows."""
    logger = CSVLogger(path, flush_every=flush_every)
    if not path.is_file() or path.stat().st_size == 0:
        return logger
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        logger.columns = list(reader.fieldnames or [])
        logger.rows = [dict(row) for row in reader]
    return logger


def main(argv: Sequence[str] | None = None) -> None:
    cfg = load_config(argv)
    output_dir = Path(cfg["output_dir"])
    session_wall_started = time.perf_counter()
    resume_checkpoint = resolve_resume_checkpoint(cfg, output_dir)
    cfg["resume_from_checkpoint"] = (
        str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    accelerator_kwargs: dict[str, Any] = {}
    if cfg["with_tracking"]:
        accelerator_kwargs.update(
            log_with=cfg["report_to"], project_dir=str(output_dir)
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        mixed_precision=str(cfg["mixed_precision"]),
        cpu=bool(cfg["cpu"]),
        **accelerator_kwargs,
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_seed(int(cfg["seed"]))
    verify_fid_dependencies(cfg)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True) + "\n"
        )
    accelerator.wait_for_everyone()

    train_dataset = build_dataset(cfg, training=True)
    train_loader = build_dataloader(train_dataset, cfg, training=True)
    real_dataset = (
        build_dataset(cfg, training=False) if cfg["evaluate_fid_is"] else None
    )
    model = build_model(cfg)
    optimizer, param_to_name = create_optimizer(model, cfg)
    lr_scheduler = create_lr_scheduler(optimizer, cfg)
    noise_scheduler = build_noise_scheduler(cfg)

    resume_path = cfg.get("resume_from_checkpoint")
    compile_after_resume = bool(cfg["compile"] and resume_path)
    if cfg["compile"] and not compile_after_resume:
        model = torch.compile(model, mode=str(cfg["compile_mode"]))
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    unwrapped_model = unwrap_model(accelerator, model)
    ema = (
        ModelEMA(
            unwrapped_model.parameters(),
            float(cfg["ema_decay"]),
            int(cfg["ema_update_after_step"]),
        )
        if cfg["use_ema"]
        else None
    )
    if ema is not None:
        accelerator.register_for_checkpointing(ema)

    if cfg["with_tracking"]:
        wandb_init_kwargs: dict[str, Any] = {"name": cfg["run_name"]}
        if cfg.get("wandb_run_id"):
            wandb_init_kwargs.update(
                {
                    "id": str(cfg["wandb_run_id"]),
                    "resume": str(cfg.get("wandb_resume", "allow")),
                }
            )
        accelerator.init_trackers(
            str(cfg["wandb_project"]),
            config=cfg,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
    csv_log = (
        _resume_safe_csv_logger(
            output_dir / "metrics.csv",
            flush_every=max(1, int(cfg["csv_flush_every"])),
        )
        if accelerator.is_main_process and cfg["csv_logging"]
        else None
    )

    completed_steps = 0
    previous_elapsed_seconds = 0.0
    processed_images = 0
    previous_peak_allocated_bytes = 0
    previous_peak_reserved_bytes = 0
    if resume_path:
        checkpoint_path = Path(str(resume_path))
        accelerator.load_state(str(checkpoint_path))
        checkpoint_metadata = _checkpoint_metadata(checkpoint_path)
        completed_steps = int(checkpoint_metadata["completed_steps"])
        previous_elapsed_seconds = float(
            checkpoint_metadata.get("elapsed_seconds", 0.0)
        )
        processed_images = int(checkpoint_metadata.get("processed_images", 0))
        previous_peak_allocated_bytes = int(
            checkpoint_metadata.get("peak_allocated_bytes", 0)
        )
        previous_peak_reserved_bytes = int(
            checkpoint_metadata.get("peak_reserved_bytes", 0)
        )
        accelerator.print(
            f"Resumed from {checkpoint_path} at optimizer step {completed_steps}"
        )
    if compile_after_resume:
        model = torch.compile(model, mode=str(cfg["compile_mode"]))

    updates_per_epoch = math.ceil(
        len(train_loader) / int(cfg["gradient_accumulation_steps"])
    )
    starting_epoch = completed_steps // updates_per_epoch
    resume_batches = (
        completed_steps % updates_per_epoch
    ) * int(cfg["gradient_accumulation_steps"])
    num_epochs = math.ceil(int(cfg["max_train_steps"]) / updates_per_epoch)
    total_batch_size = (
        int(cfg["per_device_train_batch_size"])
        * accelerator.num_processes
        * int(cfg["gradient_accumulation_steps"])
    )
    accelerator.print(
        f"Dataset={cfg['dataset_name']} examples={len(train_dataset)} "
        f"model_params={sum(p.numel() for p in unwrapped_model.parameters()):,} "
        f"effective_batch={total_batch_size} max_steps={cfg['max_train_steps']}"
    )

    log_started = time.perf_counter()
    samples_since_log = 0
    loss_since_log = 0.0
    updates_since_log = 0
    eval_records_path = output_dir / "eval_metrics.jsonl"
    evaluated_steps = (
        _load_evaluated_steps(eval_records_path)
        if accelerator.is_main_process
        else set()
    )
    last_eval_step = max(evaluated_steps, default=-1)
    for epoch in range(starting_epoch, num_epochs):
        active_loader = train_loader
        if epoch == starting_epoch and resume_batches:
            active_loader = accelerator.skip_first_batches(train_loader, resume_batches)
        for clean_images, class_labels in active_loader:
            with accelerator.accumulate(model):
                clean_images = clean_images.to(
                    accelerator.device, non_blocking=True
                )
                class_labels = class_labels.to(
                    accelerator.device, non_blocking=True
                ).long()
                with accelerator.autocast():
                    loss = diffusion_loss(
                        model,
                        noise_scheduler,
                        clean_images,
                        class_labels,
                    )
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients and cfg["max_grad_norm"] is not None:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), float(cfg["max_grad_norm"])
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    completed_steps += 1
                    if ema is not None:
                        ema.step(unwrap_model(accelerator, model).parameters())

            gathered_loss = accelerator.gather(loss.detach().float()).mean().item()
            batch_images = clean_images.shape[0] * accelerator.num_processes
            samples_since_log += batch_images
            processed_images += batch_images
            loss_since_log += gathered_loss
            if accelerator.sync_gradients:
                updates_since_log += 1

            should_log = (
                accelerator.sync_gradients
                and completed_steps % int(cfg["log_every_n"]) == 0
            )
            if should_log:
                elapsed = max(time.perf_counter() - log_started, 1.0e-9)
                cumulative_elapsed = previous_elapsed_seconds + (
                    time.perf_counter() - session_wall_started
                )
                optimizer_bytes = optimizer_state_bytes(optimizer)
                metrics: dict[str, Any] = {
                    "step": completed_steps,
                    "train/denoising_loss": loss_since_log
                    / max(1, updates_since_log),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm,
                    "throughput/images_per_second": samples_since_log / elapsed,
                    "throughput/cumulative_images_per_second": processed_images
                    / max(cumulative_elapsed, 1.0e-9),
                    "time/elapsed_seconds": cumulative_elapsed,
                    "system/optimizer_state_bytes": optimizer_bytes,
                    "system/optimizer_state_gib": optimizer_bytes / (1024**3),
                    **_memory_metrics(accelerator.device),
                }
                if cfg["with_tracking"]:
                    accelerator.log(_numeric_for_csv(metrics), step=completed_steps)
                if csv_log is not None:
                    csv_log.log(_numeric_for_csv(metrics))
                accelerator.print(
                    f"step={completed_steps} loss={metrics['train/denoising_loss']:.5f} "
                    f"lr={metrics['train/lr']:.3e} "
                    f"images/s={metrics['throughput/images_per_second']:.1f}"
                )
                log_started = time.perf_counter()
                samples_since_log = 0
                loss_since_log = 0.0
                updates_since_log = 0

            if (
                cfg["mode"] == "snr_adam"
                and accelerator.sync_gradients
                and completed_steps > 0
                and completed_steps % int(cfg["log_snr_metrics_every_n"]) == 0
            ):
                diagnostic_logs = collect_snr_diagnostics(
                    optimizer, param_to_name, cfg
                )
                diagnostic_summary = summarize_diagnostics(diagnostic_logs)
                if cfg["with_tracking"] and diagnostic_summary:
                    accelerator.log(diagnostic_summary, step=completed_steps)
                if csv_log is not None and diagnostic_summary:
                    csv_log.log(
                        {
                            "step": completed_steps,
                            **diagnostic_summary,
                        }
                    )

            checkpointing_steps = cfg.get("checkpointing_steps")
            if (
                accelerator.sync_gradients
                and checkpointing_steps
                and completed_steps % int(checkpointing_steps) == 0
            ):
                current_peak_allocated, current_peak_reserved = _peak_memory_bytes(
                    accelerator.device
                )
                save_checkpoint(
                    accelerator,
                    output_dir,
                    completed_steps,
                    elapsed_seconds=previous_elapsed_seconds
                    + (time.perf_counter() - session_wall_started),
                    processed_images=processed_images,
                    peak_allocated_bytes=max(
                        previous_peak_allocated_bytes, current_peak_allocated
                    ),
                    peak_reserved_bytes=max(
                        previous_peak_reserved_bytes, current_peak_reserved
                    ),
                    optimizer_bytes=optimizer_state_bytes(optimizer),
                    checkpoints_total_limit=cfg["checkpoints_total_limit"],
                )

            if (
                accelerator.sync_gradients
                and int(cfg["sample_interval"]) > 0
                and completed_steps % int(cfg["sample_interval"]) == 0
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    sample_path = save_sample_grid(
                        unwrap_model(accelerator, model),
                        noise_scheduler,
                        cfg,
                        output_dir,
                        completed_steps,
                        accelerator.device,
                        ema,
                    )
                    if cfg["with_tracking"]:
                        log_sample_grid(
                            accelerator, sample_path, completed_steps
                        )
                accelerator.wait_for_everyone()

            eval_interval = int(cfg["eval_interval"])
            should_evaluate = (
                cfg["evaluate_fid_is"]
                and eval_interval > 0
                and accelerator.sync_gradients
                and completed_steps % eval_interval == 0
            )
            if should_evaluate:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if completed_steps in evaluated_steps:
                        LOGGER.info(
                            "Evaluation at step %d already recorded; skipping",
                            completed_steps,
                        )
                    else:
                        eval_metrics, skip_reason = evaluate_generative_metrics(
                            unwrap_model(accelerator, model),
                            noise_scheduler,
                            real_dataset,
                            cfg,
                            accelerator.device,
                            ema,
                        )
                        if skip_reason:
                            LOGGER.warning(skip_reason)
                        elif eval_metrics:
                            last_eval_step = completed_steps
                            evaluated_steps.add(completed_steps)
                            _record_evaluation(
                                eval_records_path, completed_steps, eval_metrics
                            )
                            LOGGER.info(
                                "Generative metrics at step %d: %s",
                                completed_steps,
                                eval_metrics,
                            )
                            if cfg["with_tracking"]:
                                accelerator.log(eval_metrics, step=completed_steps)
                            if csv_log is not None:
                                csv_log.log({"step": completed_steps, **eval_metrics})
                accelerator.wait_for_everyone()

            if completed_steps >= int(cfg["max_train_steps"]):
                break
        resume_batches = 0
        if completed_steps >= int(cfg["max_train_steps"]):
            break

    accelerator.wait_for_everyone()
    final_checkpoint = output_dir / f"step_{completed_steps}"
    if not (final_checkpoint / "training_state.json").is_file():
        current_peak_allocated, current_peak_reserved = _peak_memory_bytes(
            accelerator.device
        )
        save_checkpoint(
            accelerator,
            output_dir,
            completed_steps,
            elapsed_seconds=previous_elapsed_seconds
            + (time.perf_counter() - session_wall_started),
            processed_images=processed_images,
            peak_allocated_bytes=max(
                previous_peak_allocated_bytes, current_peak_allocated
            ),
            peak_reserved_bytes=max(
                previous_peak_reserved_bytes, current_peak_reserved
            ),
            optimizer_bytes=optimizer_state_bytes(optimizer),
            checkpoints_total_limit=cfg["checkpoints_total_limit"],
        )
    if accelerator.is_main_process:
        final_model = unwrap_model(accelerator, model)
        with _ema_context(ema, final_model):
            final_model.save_pretrained(output_dir / "unet")
        noise_scheduler.save_pretrained(output_dir / "scheduler")
        final_sample_path = save_sample_grid(
            final_model,
            noise_scheduler,
            cfg,
            output_dir,
            completed_steps,
            accelerator.device,
            ema,
        )
        if cfg["with_tracking"]:
            log_sample_grid(accelerator, final_sample_path, completed_steps)
    accelerator.wait_for_everyone()

    if cfg["evaluate_fid_is"] and last_eval_step != completed_steps:
        if accelerator.is_main_process:
            if completed_steps in evaluated_steps:
                last_eval_step = completed_steps
            else:
                eval_metrics, skip_reason = evaluate_generative_metrics(
                    unwrap_model(accelerator, model),
                    noise_scheduler,
                    real_dataset,
                    cfg,
                    accelerator.device,
                    ema,
                )
                if skip_reason:
                    LOGGER.warning(skip_reason)
                elif eval_metrics:
                    last_eval_step = completed_steps
                    evaluated_steps.add(completed_steps)
                    _record_evaluation(
                        eval_records_path, completed_steps, eval_metrics
                    )
                    LOGGER.info(
                        "Final generative metrics at step %d: %s",
                        completed_steps,
                        eval_metrics,
                    )
                    if cfg["with_tracking"]:
                        accelerator.log(eval_metrics, step=completed_steps)
                    if csv_log is not None:
                        csv_log.log({"step": completed_steps, **eval_metrics})
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        current_peak_allocated, current_peak_reserved = _peak_memory_bytes(
            accelerator.device
        )
        elapsed_seconds = previous_elapsed_seconds + (
            time.perf_counter() - session_wall_started
        )
        optimizer_bytes = optimizer_state_bytes(optimizer)
        summary = {
            "status": "completed",
            "completed_steps": completed_steps,
            "processed_images": processed_images,
            "elapsed_seconds": elapsed_seconds,
            "throughput_images_per_second": processed_images
            / max(elapsed_seconds, 1.0e-9),
            "peak_allocated_bytes": max(
                previous_peak_allocated_bytes, current_peak_allocated
            ),
            "peak_reserved_bytes": max(
                previous_peak_reserved_bytes, current_peak_reserved
            ),
            "optimizer_state_bytes": optimizer_bytes,
            "evaluation_protocol": {
                "fid_feature": 2048,
                "generated_images": int(cfg["eval_num_images"]),
                "reference_images": cfg["eval_reference_num_images"],
                "sampler": cfg["eval_sampler"],
                "inference_steps": int(cfg["eval_inference_steps"]),
                "seed": int(cfg["eval_seed"]),
                "batch_size": int(cfg["eval_batch_size"]),
                "inception_splits": int(cfg["eval_inception_splits"]),
                "real_stats_cache_dir": cfg["fid_real_stats_cache_dir"],
                "evaluated_steps": sorted(evaluated_steps),
            },
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        final_system_metrics = {
            "step": completed_steps,
            "time/elapsed_seconds": elapsed_seconds,
            "throughput/cumulative_images_per_second": summary[
                "throughput_images_per_second"
            ],
            "system/optimizer_state_bytes": optimizer_bytes,
            "memory/max_allocated_gib": summary["peak_allocated_bytes"] / (1024**3),
            "memory/max_reserved_gib": summary["peak_reserved_bytes"] / (1024**3),
        }
        if cfg["with_tracking"]:
            accelerator.log(final_system_metrics, step=completed_steps)
        if csv_log is not None:
            csv_log.log(final_system_metrics)

    if csv_log is not None:
        csv_log.close()
    if cfg["with_tracking"]:
        accelerator.end_training()
    accelerator.print(f"Training complete: {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])

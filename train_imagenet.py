"""
ViT ImageNet-1k Training Script

Supports:
- SNRAdam vs standard AdamW comparison
- torch.compile + bf16 for speed
- Distributed training via accelerate
- Standard ImageNet augmentation (RandAugment, Mixup, CutMix)
- HuggingFace datasets for ImageNet loading

Usage:
    python train_imagenet.py
    python train_imagenet.py --override_json config.json
"""

import inspect as _inspect
import os
import tempfile
from copy import deepcopy

# Use portable, user-writable cache paths.
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "triton"),
)
if "TMPDIR" in os.environ:
    tempfile.tempdir = os.environ["TMPDIR"]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _safe_getsourcelines(obj):
    """Work around Triton+Python 3.11 'source code not available' issues."""
    try:
        return _inspect._orig_getsourcelines(obj)
    except OSError as e:
        if "source code not available" in str(e):
            return ["# source code not available\n"], 0
        raise


if not hasattr(_inspect, "_orig_getsourcelines"):
    _inspect._orig_getsourcelines = _inspect.getsourcelines
    _inspect.getsourcelines = _safe_getsourcelines


import io
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import torch

# PyTorch performance settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

import torchvision.models as tv_models

from vit import vit_tiny, vit_small, vit_base, vit_large
from snr_adam import SNRAdam
from csv_logger import CSVLogger
from utils import ParamGradMetrics

try:
    import timm
    from timm.data import create_transform as timm_create_transform
except ImportError:
    timm = None
    timm_create_transform = None


def warmup_to_steps(warmup_value, total_steps):
    """
    Convert warmup config to actual steps.
    
    Args:
        warmup_value: None (no warmup), 0-1 (ratio of total_steps), or >1 (direct step count)
        total_steps: Total training steps for ratio calculation
    
    Returns:
        int or None: Warmup steps, or None if no warmup
    """
    if warmup_value is None:
        return None
    if warmup_value <= 1.0:
        # Interpret as ratio
        return int(warmup_value * total_steps)
    else:
        # Interpret as direct step count
        return int(warmup_value)


# =============================================================================
# Configuration
# =============================================================================

def get_config():
    """Default configuration - edit directly or override via JSON."""
    return {
        # Model
        "model": "deit_small_patch16_224",  # maintained timm path; legacy/local models remain available
        "img_size": 224,
        "gradient_checkpointing": False,
        "qk_norm": False,
        "stochastic_depth": 0.1,
        
        # Data (HuggingFace)
        "dataset_name": "ILSVRC/imagenet-1k",  # gated HuggingFace dataset
        "image_column": "image",
        "label_column": "label",
        "num_classes": 1000,
        "num_workers": 13,
        "prefetch_factor": 4,
        "hf_cache_dir": None,
        
        # Training
        "epochs": 300,
        "batch_size": 128,
        "gradient_accumulation_steps": 4,  # Effective single-GPU batch = 512
        "lr": 5e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "lr_warmup_mult": 1.0,
        "min_lr_ratio": 0.0,
        "max_grad_norm": None,  # ConvNeXt recipe does not clip gradients
        "label_smoothing": 0.1,
        
        # Optimizer
        "optimizer": "adamw",  # adamw or snr_adam
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        
        # SNRAdam specific
        "boost_min": 0.25,
        "boost_max": 4.0,
        "normalize_mode": "global",
        "snr_mode": "coherence",  # coherence, snr
        "score_beta": None,
        "boost_center_mode": "mean_log",
        "boost_center_scope": "global",
        "boost_center_every": 10,
        "snr_whiten_stat": "winsor",
        "snr_whiten_every": 10,
        "grad_z_clip": 4.0,
        "power_warmup_steps": 100,
        "slow_add_alpha": 0.5,
        "momentum_mode": "snr_adaptive",
        "adaptive_beta1_fast": 0.9,
        "adaptive_beta1_slow": 0.99,
        "track_null_diagnostics": False,
        "diagnostics_every": 10,
        
        # Augmentation
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "mixup_prob": 1.0,
        "mixup_switch_prob": 0.5,
        "drop_rate": 0.0,
        "random_erasing": 0.25,  # Random erasing probability
        "random_erasing_mode": "pixel",
        "random_erasing_count": 1,
        "color_jitter": 0.4,
        "auto_augment": "rand-m9-mstd0.5-inc1",
        "train_interpolation": "bicubic",
        "crop_pct": 0.875,

        # Applied identically to every optimizer.
        "model_ema": False,
        "model_ema_decay": 0.99996,
        
        # Compile (off by default -- conflicts with stochastic depth)
        "compile": False,
        "compile_mode": "reduce-overhead",
        "profile_timers": False,
        
        # Misc
        "seed": 42,
        "output_dir": "./vit_output",
        "log_interval": 10,
        "save_interval": 10,
        "save_best_model": True,
        "wandb": False,
        "wandb_project": "snradam_imagenet",
        "run_name": None,
        
        # SNR-Adam metrics logging
        "log_snr_metrics_every_n": 100,
        "max_train_batches": None,
        "max_val_batches": None,
    }


# =============================================================================
# HuggingFace Dataset Wrapper
# =============================================================================

class HuggingFaceImageDataset(Dataset):
    """Wrap a HuggingFace image dataset for PyTorch DataLoader."""
    
    def __init__(self, dataset, transform, image_column="image", label_column="label"):
        self.dataset = dataset
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
    
    def __len__(self):
        return len(self.dataset)
    
    def _to_pil(self, image):
        """Convert various image formats to PIL Image."""
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(io.BytesIO(image["bytes"]))
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception:
                raise ValueError(f"Unexpected image type: {type(image)}")
        return image.convert("RGB")
    
    def __getitem__(self, idx):
        record = self.dataset[idx]
        image = self._to_pil(record[self.image_column])
        pixel_values = self.transform(image)
        label = int(record[self.label_column])
        return pixel_values, label


def load_imagenet_dataset(cfg):
    """Load ImageNet with an actionable error for gated/auth failures."""
    try:
        return load_dataset(
            cfg["dataset_name"],
            cache_dir=cfg["hf_cache_dir"],
        )
    except Exception as error:
        message = str(error)
        error_name = type(error).__name__
        auth_failure = (
            error_name in {"GatedRepoError", "RepositoryNotFoundError"}
            or "401" in message
            or "403" in message
            or "gated" in message.lower()
            or "authentication" in message.lower()
        )
        guidance = (
            "Accept the dataset terms on Hugging Face and authenticate with "
            "`hf auth login` or a readable HF_TOKEN."
            if auth_failure
            else "Verify dataset_name, network access, and hf_cache_dir."
        )
        raise RuntimeError(
            f"Unable to load ImageNet dataset {cfg['dataset_name']!r}: "
            f"{error_name}: {message}. {guidance}"
        ) from error


# =============================================================================
# Data Augmentation
# =============================================================================

def get_train_transforms(
    img_size=224,
    random_erasing=0.25,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="bicubic",
    random_erasing_mode="pixel",
    random_erasing_count=1,
):
    """Standard ImageNet training augmentation with RandAugment + random erasing."""
    if timm_create_transform is not None:
        return timm_create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            re_prob=random_erasing,
            re_mode=random_erasing_mode,
            re_count=random_erasing_count,
        )

    t = [
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(color_jitter, color_jitter, color_jitter)
        if color_jitter
        else transforms.Lambda(lambda image: image),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if random_erasing > 0:
        t.append(transforms.RandomErasing(p=random_erasing))
    return transforms.Compose(t)


def get_val_transforms(img_size=224, crop_pct=0.875):
    """Standard ImageNet validation preprocessing."""
    resize_size = int(img_size / crop_pct)  # 256 for 224 at crop_pct=0.875
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class Mixup:
    """Mixup and CutMix augmentation."""
    def __init__(
        self,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        num_classes=1000,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
    
    def __call__(self, images, targets):
        if torch.rand(1).item() > self.prob:
            return images, F.one_hot(targets, self.num_classes).float()
        
        targets_onehot = F.one_hot(targets, self.num_classes).float()
        use_cutmix = torch.rand(1).item() < self.switch_prob and self.cutmix_alpha > 0
        
        if use_cutmix:
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            B, C, H, W = images.shape
            cut_ratio = math.sqrt(1 - lam)
            cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
            cy, cx = torch.randint(H, (1,)).item(), torch.randint(W, (1,)).item()
            y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
            x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
            
            indices = torch.randperm(B, device=images.device)
            images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
            lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
            targets_mixed = lam * targets_onehot + (1 - lam) * targets_onehot[indices]
        else:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            indices = torch.randperm(images.size(0), device=images.device)
            images = lam * images + (1 - lam) * images[indices]
            targets_mixed = lam * targets_onehot + (1 - lam) * targets_onehot[indices]
        
        return images, targets_mixed


def soft_cross_entropy(logits, targets, label_smoothing=0.0):
    """Cross entropy with soft targets and optional label smoothing."""
    log_probs = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0:
        n_classes = logits.size(-1)
        targets = targets * (1.0 - label_smoothing) + label_smoothing / n_classes
    return -(targets * log_probs).sum(dim=-1).mean()


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Training Loop
# =============================================================================

class ModelEMA:
    """Optimizer-agnostic exponential moving average of parameters and buffers."""

    def __init__(self, model, decay=0.99996):
        self.module = deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = getattr(model, "_orig_mod", model)
        source_state = source.state_dict()
        for name, value in self.module.state_dict().items():
            source_value = source_state[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                value.copy_(source_value)


def create_model(cfg):
    """Create maintained timm DeiT-S or one of the existing local architectures."""
    if cfg["model"] == "deit_small_patch16_224":
        if timm is not None:
            return timm.create_model(
                "deit_small_patch16_224",
                pretrained=False,
                num_classes=cfg["num_classes"],
                img_size=cfg["img_size"],
                drop_rate=cfg["drop_rate"],
                drop_path_rate=cfg.get("stochastic_depth", 0.1),
            )
        print(
            "WARNING: timm is unavailable; falling back to local vit_small. "
            "Install timm for the maintained DeiT-S paper path.",
            file=sys.stderr,
        )
        return vit_small(
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
            drop_rate=cfg["drop_rate"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
            qk_norm=cfg["qk_norm"],
        )

    vit_models = {
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
    }
    if cfg["model"] in vit_models:
        return vit_models[cfg["model"]](
            num_classes=cfg["num_classes"],
            img_size=cfg["img_size"],
            drop_rate=cfg["drop_rate"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
            qk_norm=cfg["qk_norm"],
        )
    if cfg["model"] == "convnext_tiny":
        return tv_models.convnext_tiny(
            num_classes=cfg["num_classes"],
            stochastic_depth_prob=cfg.get("stochastic_depth", 0.1),
        )
    if cfg["model"] == "convnext_small":
        return tv_models.convnext_small(
            num_classes=cfg["num_classes"],
            stochastic_depth_prob=cfg.get("stochastic_depth", 0.4),
        )
    if cfg["model"] == "convnext_base":
        return tv_models.convnext_base(
            num_classes=cfg["num_classes"],
            stochastic_depth_prob=cfg.get("stochastic_depth", 0.5),
        )
    if cfg["model"] == "resnet50":
        return tv_models.resnet50(num_classes=cfg["num_classes"])
    raise ValueError(
        f"Unknown model: {cfg['model']}. Choose from: deit_small_patch16_224, "
        "convnext_tiny, convnext_small, convnext_base, resnet50, "
        "vit_tiny, vit_small, vit_base, vit_large"
    )


def _optimizer_diagnostics(optimizer):
    logs = {}
    if not hasattr(optimizer, "get_diagnostics"):
        return logs
    for param_key, diagnostics in optimizer.get_diagnostics().items():
        for key, value in diagnostics.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith("null_"):
                prefix = "snr_null"
            elif key.startswith("beta1_eff_"):
                prefix = "snr_momentum"
            elif key == "grad_clip_fraction":
                prefix = "snr_clipping"
            elif key.startswith("slow_") or key == "final_update_rms":
                prefix = "snr_slow"
            else:
                prefix = "snr_diagnostics"
            logs[f"{prefix}/{key}/{param_key}"] = float(value)
    return logs


def optimizer_state_bytes(optimizer):
    """Count tensor storage held by optimizer state after initialization."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    seen = set()

    def count(value):
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            if identity in seen:
                return 0
            seen.add(identity)
            return storage.nbytes()
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(item) for item in value)
        return 0

    return count(optimizer.state)


def runtime_metadata():
    def version(distribution):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "timm_version": version("timm"),
        "datasets_version": version("datasets"),
        "accelerate_version": version("accelerate"),
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
    }


def train_one_epoch(model, train_loader, optimizer, scheduler, mixup_fn, accelerator, epoch, cfg, 
                    snr_metrics_tracker=None, param_to_name=None, csv_log=None, model_ema=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.perf_counter()
    processed_examples = 0
    max_batches = cfg.get("max_train_batches")
    
    profile_timers = bool(cfg.get("profile_timers", False) and torch.cuda.is_available())
    if profile_timers:
        data_start = torch.cuda.Event(enable_timing=True)
        data_end = torch.cuda.Event(enable_timing=True)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        optimizer_start = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
    
    dataloader_iter = iter(train_loader)
    
    # Unwrap optimizer for state access
    opt = getattr(optimizer, "optimizer", optimizer)
    
    label_smoothing = cfg.get("label_smoothing", 0.0)

    num_batches_to_run = min(len(train_loader), max_batches) if max_batches else len(train_loader)
    for batch_idx in range(num_batches_to_run):
        # Time data loading
        if profile_timers:
            data_start.record()
        images, targets = next(dataloader_iter)
        if profile_timers:
            data_end.record()
        
        images, targets_mixed = mixup_fn(images, targets)
        
        with accelerator.accumulate(model):
            # Time forward pass
            if profile_timers:
                forward_start.record()
            with accelerator.autocast():
                logits = model(images)
                loss = soft_cross_entropy(logits, targets_mixed, label_smoothing=label_smoothing)
            if profile_timers:
                forward_end.record()
            
            # Time backward pass
            if profile_timers:
                backward_start.record()
            accelerator.backward(loss)
            if profile_timers:
                backward_end.record()
            
            if cfg.get("max_grad_norm") is not None and cfg["max_grad_norm"] > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            
            # Time optimizer step
            if profile_timers:
                optimizer_start.record()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if model_ema is not None and accelerator.sync_gradients:
                model_ema.update(accelerator.unwrap_model(model))
            if profile_timers:
                optimizer_end.record()
        
        timing_metrics = {}
        if profile_timers:
            torch.cuda.synchronize()
            timing_metrics = {
                "timer/data_load_ms": data_start.elapsed_time(data_end),
                "timer/forward_ms": forward_start.elapsed_time(forward_end),
                "timer/backward_ms": backward_start.elapsed_time(backward_end),
                "timer/optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
            }
        
        total_loss += loss.item()
        num_batches += 1
        processed_examples += images.size(0) * accelerator.num_processes
        
        if batch_idx % cfg["log_interval"] == 0:
            elapsed = time.perf_counter() - start_time
            imgs_per_sec = processed_examples / max(elapsed, 1e-9)
            current_lr = scheduler.get_last_lr()[0]
            accelerator.print(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} LR: {current_lr:.6f} "
                f"Speed: {imgs_per_sec:.1f} img/s"
            )
            # Log metrics every log_interval steps
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            step_metrics = {
                "train/loss_step": loss.item(),
                "train/lr": current_lr,
                "train/imgs_per_sec": imgs_per_sec,
                "train/step": global_step,
                **timing_metrics,
            }
            if cfg["wandb"]:
                accelerator.log(step_metrics, step=global_step)
            if csv_log is not None:
                csv_log.log({"step": global_step, **step_metrics})
        
        # Log SNR-Adam metrics every N steps
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        if (snr_metrics_tracker is not None 
            and param_to_name is not None
            and global_step > 0 
            and global_step % cfg["log_snr_metrics_every_n"] == 0):
            snr_logs = {}
            num_params_logged = 0
            
            # Iterate over optimizer state directly (most reliable)
            for param, state in opt.state.items():
                if "m_matched" not in state and "exp_avg" not in state:
                    continue
                
                name = param_to_name.get(param, f"param_{id(param)}")
                
                # Use simpler method that only needs optimizer state (no grad needed)
                stats = snr_metrics_tracker.compute_snr_stats(
                    optimizer_state=state,
                    power=cfg.get("power", 1.5),
                    boost_min=cfg.get("boost_min", 0.25),
                    boost_max=cfg.get("boost_max", 8.0),
                    normalize=cfg.get("normalize", True),
                    beta2=cfg.get("beta2", 0.999),
                )
                if stats is None:
                    continue
                num_params_logged += 1
                
                # Log per-parameter metrics (group by metric, flatten param name)
                flat_name = name.replace(".", "_")
                snr_logs[f"snr/snr_mean/{flat_name}"] = stats["snr_mean"]
                snr_logs[f"snr/snr_std/{flat_name}"] = stats["snr_std"]
                snr_logs[f"snr/boost_median/{flat_name}"] = stats["boost_median"]
                snr_logs[f"snr/boost_std/{flat_name}"] = stats["boost_std"]
                snr_logs[f"snr/boost_p10/{flat_name}"] = stats["boost_p10"]
                snr_logs[f"snr/boost_p90/{flat_name}"] = stats["boost_p90"]
                snr_logs[f"snr/frac_min_clamp/{flat_name}"] = stats["boost_frac_at_min_clamp"]
                snr_logs[f"snr/frac_max_clamp/{flat_name}"] = stats["boost_frac_at_max_clamp"]
            
            # Debug on first log
            if global_step == cfg["log_snr_metrics_every_n"]:
                accelerator.print(f"[SNR Metrics] Logged {num_params_logged} params, {len(snr_logs)} metrics")
            
            snr_logs.update(_optimizer_diagnostics(opt))
            if snr_logs and cfg["wandb"]:
                accelerator.log(snr_logs, step=global_step)
            if snr_logs and csv_log is not None:
                csv_log.log({"step": global_step, **snr_logs})
    
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    elapsed = time.perf_counter() - start_time
    return (
        total_loss / max(num_batches, 1),
        processed_examples / max(elapsed, 1e-9),
        elapsed,
    )


@torch.no_grad()
def evaluate(model, val_loader, accelerator, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        with accelerator.autocast():
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        
        total_loss += loss.item() * images.size(0)
        top5 = logits.topk(k=min(5, logits.size(-1)), dim=-1).indices
        matches = top5.eq(targets.unsqueeze(-1))
        correct_top1 += matches[:, :1].sum().item()
        correct_top5 += matches.sum().item()
        total += images.size(0)
    
    metrics = torch.tensor(
        [correct_top1, correct_top5, total, total_loss],
        dtype=torch.float64,
        device=accelerator.device,
    )
    gathered = accelerator.reduce(metrics, reduction="sum")
    correct_top1, correct_top5, total, total_loss = (value.item() for value in gathered)
    
    if total == 0:
        raise RuntimeError("validation produced zero examples")
    return total_loss / total, correct_top1 / total, correct_top5 / total


def main():
    # Load config
    cfg = get_config()
    
    # Parse command line for override JSON
    if "--override_json" in sys.argv:
        idx = sys.argv.index("--override_json")
        with open(sys.argv[idx + 1], 'r') as f:
            cfg.update(json.load(f))
    
    # Generate run name if not provided
    if cfg["run_name"] is None:
        cfg["run_name"] = f"{cfg['model']}-{cfg['optimizer']}-lr{cfg['lr']}"
        if cfg["optimizer"] == "snr_adam":
            cfg["run_name"] = f"{cfg['model']}-SNRAdam-{cfg['normalize_mode']}"
            if cfg.get("snr_mode", "coherence") != "coherence":
                cfg["run_name"] += f"-{cfg['snr_mode']}"
            if cfg["boost_max"] != 4.0:
                cfg["run_name"] += f"-bmax{cfg['boost_max']}"
            if cfg.get("slow_add_alpha", 0) > 0:
                cfg["run_name"] += f"-slow{cfg['slow_add_alpha']}"
            if cfg.get("grad_z_clip", 0.0) > 0:
                cfg["run_name"] += f"-gzc{cfg['grad_z_clip']}"
            cfg["run_name"] += f"-lr{cfg['lr']}"
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        log_with="wandb" if cfg["wandb"] else None,
    )
    
    set_seed(cfg["seed"])
    
    # Create output directory
    output_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", 'w') as f:
            json.dump(cfg, f, indent=2)
    
    # Initialize wandb
    if cfg["wandb"] and accelerator.is_main_process:
        accelerator.init_trackers(
            cfg["wandb_project"],
            config=cfg,
            init_kwargs={"wandb": {"name": cfg["run_name"]}}
        )
    
    # The paper path uses timm's maintained DeiT-S implementation.
    model = create_model(cfg)
    
    accelerator.print(f"Model: {cfg['model']}, Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    model_ema = None
    if cfg.get("model_ema", False):
        model_ema = ModelEMA(model, decay=cfg.get("model_ema_decay", 0.99996))
        model_ema.module.to(accelerator.device)
        accelerator.print(f"Model EMA enabled: decay={model_ema.decay}")
    
    # Compile model (ViT is small enough to compile whole model)
    if cfg["compile"]:
        accelerator.print(f"Compiling model with mode={cfg['compile_mode']}...")
        model = torch.compile(model, mode=cfg["compile_mode"], fullgraph=True, dynamic=False)
    
    # Load dataset from HuggingFace
    accelerator.print(f"Loading dataset: {cfg['dataset_name']}")
    dataset = load_imagenet_dataset(cfg)
    missing_splits = {"train", "validation"}.difference(dataset)
    if missing_splits:
        raise RuntimeError(
            f"ImageNet dataset {cfg['dataset_name']!r} is missing required split(s): "
            f"{', '.join(sorted(missing_splits))}"
        )
    
    # Create wrapped datasets with transforms
    train_dataset = HuggingFaceImageDataset(
        dataset["train"],
        transform=get_train_transforms(
            cfg["img_size"],
            random_erasing=cfg.get("random_erasing", 0.25),
            color_jitter=cfg.get("color_jitter", 0.4),
            auto_augment=cfg.get("auto_augment", "rand-m9-mstd0.5-inc1"),
            interpolation=cfg.get("train_interpolation", "bicubic"),
            random_erasing_mode=cfg.get("random_erasing_mode", "pixel"),
            random_erasing_count=cfg.get("random_erasing_count", 1),
        ),
        image_column=cfg["image_column"],
        label_column=cfg["label_column"],
    )
    val_dataset = HuggingFaceImageDataset(
        dataset["validation"],
        transform=get_val_transforms(cfg["img_size"], crop_pct=cfg.get("crop_pct", 0.875)),
        image_column=cfg["image_column"],
        label_column=cfg["label_column"],
    )
    
    accelerator.print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    loader_workers = {"num_workers": cfg["num_workers"], "pin_memory": True}
    if cfg["num_workers"] > 0:
        loader_workers["prefetch_factor"] = cfg["prefetch_factor"]
        loader_workers["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, **loader_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"] * 2, shuffle=False,
        **loader_workers,
    )
    
    # Create parameter groups with proper weight decay exclusions
    # - 1D params: catches all biases and all normalization weights (LayerNorm, etc.)
    # - Embeddings: 2D but shouldn't have weight decay (includes pos_embed, cls_token)
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D params (biases, norm weights) or embeddings/cls_token -> no weight decay
        if param.dim() == 1 or "embed" in name.lower() or "cls_token" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    accelerator.print(f"Params with weight decay: {len(decay_params)}, without: {len(no_decay_params)}")
    
    optimizer_grouped_params = [
        {"params": decay_params, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    # Create optimizer
    if cfg["optimizer"] == "snr_adam":
        optimizer = SNRAdam(
            optimizer_grouped_params, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
            eps=cfg.get("eps", 1e-8),
            weight_decay=cfg["weight_decay"],
            boost_min=cfg["boost_min"], boost_max=cfg["boost_max"],
            normalize_mode=cfg.get("normalize_mode", "global"),
            snr_mode=cfg.get("snr_mode", "coherence"),
            score_beta=cfg.get("score_beta"),
            boost_center_mode=cfg.get("boost_center_mode", "mean_log"),
            boost_center_scope=cfg.get("boost_center_scope", "global"),
            boost_center_every=cfg.get("boost_center_every", 10),
            snr_whiten_stat=cfg.get("snr_whiten_stat", "winsor"),
            snr_whiten_every=cfg.get("snr_whiten_every", 10),
            grad_z_clip=cfg.get("grad_z_clip", 4.0),
            momentum_mode=cfg.get("momentum_mode", "snr_adaptive"),
            adaptive_beta1_fast=cfg.get("adaptive_beta1_fast", 0.9),
            adaptive_beta1_slow=cfg.get("adaptive_beta1_slow", 0.99),
            warmup_steps=cfg.get("power_warmup_steps", 100),
            slow_add_alpha=cfg.get("slow_add_alpha", 0.5),
            track_null_diagnostics=cfg.get("track_null_diagnostics", False),
            diagnostics_every=cfg.get("diagnostics_every", 10),
        )
        accelerator.print(f"SNRAdam: boost=[{cfg['boost_min']}, {cfg['boost_max']}], "
                          f"norm={cfg.get('normalize_mode', 'global')}, "
                          f"snr={cfg.get('snr_mode', 'coherence')}, "
                          f"stat={cfg.get('snr_whiten_stat', 'winsor')}, "
                          f"momentum={cfg.get('momentum_mode', 'snr_adaptive')}, "
                          f"slow_add_alpha={cfg.get('slow_add_alpha', 0.5)}")
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_params, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
            eps=cfg.get("eps", 1e-8),
        )
        accelerator.print(f"AdamW: lr={cfg['lr']}, betas=({cfg['beta1']}, {cfg['beta2']})")
    
    # Create scheduler
    train_batches_per_epoch = len(train_loader)
    if cfg.get("max_train_batches"):
        train_batches_per_epoch = min(
            train_batches_per_epoch, int(cfg["max_train_batches"])
        )
    updates_per_epoch = math.ceil(
        train_batches_per_epoch / cfg.get("gradient_accumulation_steps", 1)
    )
    num_training_steps = updates_per_epoch * cfg["epochs"]
    num_warmup_steps = updates_per_epoch * cfg["warmup_epochs"]
    lr_warmup_mult = float(getattr(optimizer, "param_groups", [{}])[0].get("lr_warmup_mult", cfg.get("lr_warmup_mult", 1.0)))
    num_warmup_steps = int(round(num_warmup_steps * lr_warmup_mult))
    num_warmup_steps = min(num_warmup_steps, int(num_training_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, cfg["min_lr_ratio"])
    
    # Create mixup
    mixup_fn = Mixup(
        cfg["mixup_alpha"],
        cfg["cutmix_alpha"],
        cfg["mixup_prob"],
        cfg.get("mixup_switch_prob", 0.5),
        cfg["num_classes"],
    )
    
    # Prepare for distributed
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    snr_metrics_tracker = ParamGradMetrics() if cfg["optimizer"] == "snr_adam" else None
    param_to_name = None
    if snr_metrics_tracker is not None:
        opt = getattr(optimizer, "optimizer", optimizer)
        named_params = {
            parameter: name
            for name, parameter in accelerator.unwrap_model(model).named_parameters()
        }
        param_to_name = {}
        fallback_index = 0
        for group in opt.param_groups:
            for parameter in group["params"]:
                param_to_name[parameter] = named_params.get(parameter, f"param_{fallback_index}")
                fallback_index += 1
    csv_log = CSVLogger(output_dir / "metrics.csv") if accelerator.is_main_process else None
    
    # Training loop
    best_acc = 0.0
    best_acc5 = 0.0
    best_ema_acc = 0.0
    best_ema_acc5 = 0.0
    final_metrics = {}
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    training_start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        accelerator.print(f"\n{'='*60}\nEpoch {epoch}/{cfg['epochs']}\n{'='*60}")
        
        train_loss, examples_per_second, epoch_train_seconds = train_one_epoch(
            model, train_loader, optimizer, scheduler, mixup_fn, accelerator, epoch, cfg,
            snr_metrics_tracker=snr_metrics_tracker, param_to_name=param_to_name,
            csv_log=csv_log, model_ema=model_ema,
        )
        val_loss, val_acc, val_acc5 = evaluate(
            model,
            val_loader,
            accelerator,
            max_batches=cfg.get("max_val_batches"),
        )
        
        accelerator.print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, "
            f"Top-1: {val_acc*100:.2f}%, Top-5: {val_acc5*100:.2f}%"
        )
        
        epoch_metrics = {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/top1": val_acc,
            "val/top5": val_acc5,
            "throughput/examples_per_second": examples_per_second,
            "train/epoch_seconds": epoch_train_seconds,
            "wall_time_seconds": time.perf_counter() - training_start,
            "optimizer/state_bytes": optimizer_state_bytes(optimizer),
            "epoch": epoch,
        }
        if accelerator.device.type == "cuda":
            epoch_metrics.update(
                {
                    "memory/peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                        accelerator.device
                    ),
                    "memory/peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
                        accelerator.device
                    ),
                    "memory/max_allocated_gb": torch.cuda.max_memory_allocated(
                        accelerator.device
                    )
                    / (1024**3),
                    "memory/max_reserved_gb": torch.cuda.max_memory_reserved(
                        accelerator.device
                    )
                    / (1024**3),
                }
            )
        if model_ema is not None:
            ema_loss, ema_acc, ema_acc5 = evaluate(
                model_ema.module,
                val_loader,
                accelerator,
                max_batches=cfg.get("max_val_batches"),
            )
            epoch_metrics.update({
                "ema_val/loss": ema_loss,
                "ema_val/top1": ema_acc,
                "ema_val/top5": ema_acc5,
            })
            accelerator.print(
                f"EMA Val Loss: {ema_loss:.4f}, Top-1: {ema_acc*100:.2f}%, "
                f"Top-5: {ema_acc5*100:.2f}%"
            )
        if cfg["wandb"]:
            accelerator.log(epoch_metrics)
        if csv_log is not None:
            csv_log.log(epoch_metrics)
        final_metrics = dict(epoch_metrics)
        
        if accelerator.is_main_process:
            if val_acc > best_acc:
                best_acc = val_acc
                best_acc5 = val_acc5
                if cfg.get("save_best_model", True):
                    accelerator.save(accelerator.unwrap_model(model).state_dict(), output_dir / "best_model.pt")
                accelerator.print(f"New best: {best_acc*100:.2f}%")
            if model_ema is not None and ema_acc > best_ema_acc:
                best_ema_acc = ema_acc
                best_ema_acc5 = ema_acc5
                if cfg.get("save_best_model", True):
                    accelerator.save(model_ema.module.state_dict(), output_dir / "best_model_ema.pt")
                accelerator.print(f"New EMA best: {best_ema_acc*100:.2f}%")
            
            if epoch % cfg["save_interval"] == 0:
                accelerator.save({
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "model_ema": model_ema.module.state_dict() if model_ema is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_acc": best_acc, "best_ema_acc": best_ema_acc,
                }, output_dir / f"checkpoint_epoch{epoch}.pt")
    
    if csv_log is not None:
        csv_log.close()

    if accelerator.is_main_process:
        summary_metrics = {
            "run_name": cfg["run_name"],
            "dataset_name": cfg["dataset_name"],
            "model": cfg["model"],
            "optimizer": cfg["optimizer"],
            "seed": cfg["seed"],
            "epochs": cfg["epochs"],
            "best_top1": best_acc,
            "best_top5": best_acc5,
            "best_ema_top1": best_ema_acc if model_ema is not None else None,
            "best_ema_top5": best_ema_acc5 if model_ema is not None else None,
            "final_top1": final_metrics.get("val/top1"),
            "final_top5": final_metrics.get("val/top5"),
            "final_ema_top1": final_metrics.get("ema_val/top1"),
            "final_ema_top5": final_metrics.get("ema_val/top5"),
            "examples_per_second": final_metrics.get(
                "throughput/examples_per_second"
            ),
            "elapsed_time_seconds": final_metrics.get("wall_time_seconds"),
            "peak_cuda_allocated_bytes": final_metrics.get(
                "memory/peak_cuda_allocated_bytes", 0
            ),
            "peak_cuda_reserved_bytes": final_metrics.get(
                "memory/peak_cuda_reserved_bytes", 0
            ),
            "optimizer_state_bytes": final_metrics.get("optimizer/state_bytes", 0),
            "recipe_metadata": cfg.get("recipe_metadata", {}),
            "runtime_metadata": runtime_metadata(),
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary_metrics, indent=2, sort_keys=True) + "\n"
        )
    
    summary = f"\nDone! Best accuracy: {best_acc*100:.2f}%"
    if model_ema is not None:
        summary += f", best EMA accuracy: {best_ema_acc*100:.2f}%"
    accelerator.print(summary)
    if cfg["wandb"]:
        accelerator.end_training()


if __name__ == "__main__":
    main()

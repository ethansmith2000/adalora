import os
import tempfile
import inspect as _inspect
from pathlib import Path

# Ensure Hugging Face datasets cache is set early to a user-writable path.
os.environ.setdefault("HF_HOME", "/fsx/scratch/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))

# Make Triton and TorchInductor use stable, user-writable cache/tmp dirs on the cluster
os.environ.setdefault("TRITON_CACHE_DIR", "/home/ethan/.triton/cache")
os.environ.setdefault("TMPDIR", "/home/ethan/tmp")
tempfile.tempdir = os.environ["TMPDIR"]

# When using datasets.map(num_proc=...) we rely on multiprocessing for parallelism,
# so we disable internal tokenizer threading to avoid oversubscription.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _safe_getsourcelines(obj):
    """
    Work around Triton+Python 3.11 \"source code not available\" issues by
    returning a dummy source snippet instead of raising, so torch.compile
    can proceed. This is only meant for tooling and does not affect numerics.
    """
    try:
        return _inspect._orig_getsourcelines(obj)  # type: ignore[attr-defined]
    except OSError as e:
        if "source code not available" in str(e):
            return ["# source code not available\n"], 0
        raise


if not hasattr(_inspect, "_orig_getsourcelines"):
    _inspect._orig_getsourcelines = _inspect.getsourcelines  # type: ignore[attr-defined]
    _inspect.getsourcelines = _safe_getsourcelines


import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.v8_api_enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # 
torch.backends.cuda.allow_tensor_float_32 = True


# torch._inductor.config.triton.cudagraphs = True   # or False if capture causes overhead
# torch._inductor.config.use_mixed_mm = True        # enables faster matmul codegen
# torch._inductor.config.triton.cudagraphs = True
# torch._inductor.config.triton.cudagraph_trees = True  # More aggressive cudagraph usage
# torch._inductor.config.triton.autotune_pointwise = True
# # torch._inductor.config.triton.dense_indexing = True
# torch._inductor.config.triton.max_tiles = 8  # Increase tiling options
# torch._inductor.config.aggressive_fusion = True
# torch._inductor.config.pattern_matcher = True
# torch._inductor.config.permute_fusion = True
# torch._inductor.config.max_autotune = True
# torch._inductor.config.max_autotune_gemm = True

torch.set_num_threads(12)
torch.set_num_interop_threads(2)

# torch._inductor.config.autotune_in_subproc = True            # instead of exporting TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC
# torch._inductor.config.autotune_multi_device = True          # mirrors TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE
# # torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"



import argparse
import copy
import json
import logging
import math
import os
import sys
from itertools import chain
from typing import Dict, Any

import datasets
import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from types import SimpleNamespace

import torch
import time
from transformer import Transformer


from utils import ParamGradMetrics
from csv_logger import CSVLogger
from score_diagnostics import (
    CompactJSONLWriter,
    classify_calibration_family,
    compute_score_diagnostics,
    distributed_reduce_records,
    parse_block_depth,
)
from snr_adam import SNRAdam
from gradient_probe import (
    GradientProbeCollector, ProbeConfig, validate_probe_log, validate_probe_runtime,
)
from snr_muon import SNRMuon, get_muon_param_groups as get_snr_muon_param_groups
from snr_soap import SNRSOAP


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

logger = get_logger(__name__)


REMOVED_SNRADAM_CONFIG_KEYS = {
    "agreement_tilt_log_max",
    "agreement_tilt_mode",
    "boost_map",
    "hierarchical_combine_mode",
    "match_norm",
    "mismatch_neutralize_strength",
    "mismatch_penalty_strength",
    "mismatch_route_strength",
    "null_gate_mode",
    "null_gate_scope",
    "null_gate_strength",
    "score_between_alpha",
    "score_between_metric",
    "score_between_mode",
    "score_between_strength",
    "score_between_temperature",
    "score_calibration_every",
    "score_calibration_stat",
    "score_coordinate_family_mode",
    "score_family_blend_component",
    "score_family_blend_weight",
    "score_family_target",
    "score_scope_blend_alpha",
    "score_scope_blend_mode",
    "score_transform",
    "score_transform_null_scale",
    "slow_blend_mode",
    "slow_m_alpha",
    "update_norm_every",
    "update_norm_metric",
    "update_norm_mode",
    "update_norm_p",
    "update_norm_robust_clip",
    "update_norm_scope",
    "update_norm_strength",
    "update_norm_value",
}


def _load_overrides():
    """
    Lightweight override loader to keep the script largely argument-free.
    Priority: CLI flag --override_json, then env OVERRIDE_JSON.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--override_json", type=str, default=os.environ.get("OVERRIDE_JSON"))
    parsed, _ = parser.parse_known_args()
    override_path = parsed.override_json
    if override_path:
        with open(override_path, "r") as f:
            data = json.load(f)
        # Stash metadata so downstream code can use the sweep entry name.
        data["_override_json_path"] = override_path
        data["_override_name"] = Path(override_path).stem
        return data
    return {}


def _build_activation_probe_batch(batch, limit):
    """Return a shallow copy of `batch` with tensors truncated along batch dim."""
    if limit is None:
        return batch
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.size(0) > limit:
            sliced[key] = value[:limit]
        else:
            sliced[key] = value
    return sliced


logger = get_logger(__name__)


def _compact_kwargs(kwargs: Dict[str, Any]) -> str:
    """Return a short, deterministic string for kwargs."""
    if not kwargs:
        return ""
    bits = []
    for k in sorted(kwargs.keys()):
        bits.append(f"{k}{kwargs[k]}")
    return "-".join(bits)


class SyntheticTokenDataset(Dataset):
    """Small deterministic token dataset for offline trainer smoke tests."""

    def __init__(self, num_samples: int, block_size: int, vocab_size: int, seed: int):
        if num_samples < 1:
            raise ValueError("synthetic_num_samples must be >= 1")
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        if vocab_size < 2:
            raise ValueError("synthetic_vocab_size must be >= 2")
        generator = torch.Generator().manual_seed(int(seed))
        self.input_ids = torch.randint(
            0,
            int(vocab_size),
            (int(num_samples), int(block_size)),
            generator=generator,
        )

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, index):
        # No labels column: targets are a shift of input_ids, so duplicating
        # them here would double the collated bytes for nothing.
        return {"input_ids": self.input_ids[index]}


def optimizer_state_bytes(optimizer) -> int:
    """Return logical bytes occupied by tensor optimizer state on this rank."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


SNR_DIAGNOSTIC_METRIC_PATHS = {
    "beta1_eff_mean": "snr_momentum/beta1_eff_mean",
    "beta1_eff_std": "snr_momentum/beta1_eff_std",
    "beta1_eff_min": "snr_momentum/beta1_eff_min",
    "beta1_eff_max": "snr_momentum/beta1_eff_max",
    "grad_clip_fraction": "snr_clipping/grad_clip_fraction",
    "eff_slow_add_alpha": "snr_slow/eff_slow_add_alpha",
    "final_update_rms": "snr_slow/final_update_rms",
    "slow_fast_rms": "snr_slow/fast_rms",
    "slow_branch_rms": "snr_slow/branch_rms",
    "slow_to_fast_ratio": "snr_slow/slow_to_fast_ratio",
    "slow_fast_cosine": "snr_slow/slow_fast_cosine",
    "signal_mean": "snr_score/signal_mean",
    "signal_std": "snr_score/signal_std",
    "signal_max": "snr_score/signal_max",
    "score_mean": "snr_score/calibration_mean",
    "score_std": "snr_score/calibration_std",
    "boost_mean_log": "snr_boost/boost_mean_log",
    "boost_k_star": "snr_boost/boost_k_star",
    "boost_redistribution_frac": "snr_boost/boost_redistribution_frac",
    "sign_rms_scale": "snr_geometry/sign_rms_scale",
    "sign_flip_frac": "snr_geometry/sign_flip_frac",
    "_global_score_loc": "snr_global/score_loc",
    "_global_score_scale": "snr_global/score_scale",
    "_global_mismatch_score_loc": "snr_global/mismatch_score_loc",
    "_global_mismatch_score_scale": "snr_global/mismatch_score_scale",
    "_family_score_loc": "snr_family/score_loc",
    "_family_score_scale": "snr_family/score_scale",
    "_global_boost_mean_log": "snr_global/boost_mean_log",
    "_global_sign_rms_scale": "snr_global/sign_rms_scale",
    "group_q_mean": "snr_null/group_q_mean",
    "group_null_ratio_mean": "snr_null/group_null_ratio_mean",
    "group_debiased_c2_mean": "snr_null/group_debiased_c2_mean",
    "mismatch_confidence_mean": "snr_mismatch/confidence_mean",
    "mismatch_active_frac": "snr_mismatch/active_fraction",
    "predictor_agreement_mean": "snr_agreement/predictor_mean",
    "final_agreement_mean": "snr_agreement/final_mean",
    "predictor_mismatch_mean": "snr_mismatch/predictor_mean",
    "final_mismatch_mean": "snr_mismatch/final_mean",
    "mismatch_feedback_delta": "snr_mismatch/feedback_delta",
    "momentum_gate_mean": "snr_mismatch/momentum_gate_mean",
    "boost_gate_mean": "snr_mismatch/boost_gate_mean",
    "mismatch_gate_delta_mean": "snr_mismatch/gate_delta_mean",
}

_SNR_NON_DIAGNOSTIC_SCALARS = {
    "step",
    "score_stats_mean_std_tensor_counter",
    "score_stats_mean_std_per_output_counter",
    "score_stats_winsor_tensor_counter",
    "score_stats_winsor_per_output_counter",
    "boost_center_tensor_counter",
    "boost_center_per_output_counter",
}


def aggregate_snr_diagnostics(optimizer) -> dict[str, float]:
    """Aggregate host scalars captured by SNRAdam without recomputing tensors."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    values_by_metric: dict[str, list[float]] = {}
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and value.numel() == 1:
                value = value.item()
            if (
                not isinstance(value, (int, float))
                or key in _SNR_NON_DIAGNOSTIC_SCALARS
                or key.endswith("_counter")
            ):
                continue
            metric = SNR_DIAGNOSTIC_METRIC_PATHS.get(
                key, f"snr_diagnostics/{key}"
            )
            values_by_metric.setdefault(metric, []).append(float(value))
    return {
        f"{metric}/mean": sum(values) / len(values)
        for metric, values in values_by_metric.items()
    }


def make_optimizer_lr_capturable(optimizer) -> None:
    """Keep scheduled LRs as device scalars so torch.compile sees no float guards."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    for group in optimizer.param_groups:
        lr = group["lr"]
        if torch.is_tensor(lr):
            continue
        parameter = next(
            (parameter for parameter in group["params"] if parameter is not None),
            None,
        )
        if parameter is None:
            continue
        group["lr"] = torch.tensor(
            float(lr), device=parameter.device, dtype=torch.float32
        )


def build_ademamix_optimizer(parameters, args, total_steps):
    """Construct the repository-local paper baseline implementation."""
    from ademamix import AdEMAMix

    beta3_warmup = warmup_to_steps(
        getattr(args, "ademamix_beta3_warmup", None), total_steps
    )
    alpha_warmup = warmup_to_steps(
        getattr(args, "ademamix_alpha_warmup", None), total_steps
    )
    return AdEMAMix(
        parameters,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2, args.ademamix_beta3),
        alpha=args.ademamix_alpha,
        beta3_warmup=beta3_warmup,
        alpha_warmup=alpha_warmup,
        eps=getattr(args, "eps", 1e-8),
        weight_decay=args.weight_decay,
    )



def main():
    override_args = _load_overrides()

    args = {
        "num_validation_batches": 25,
        "validate_every": 1000,
        # `huggingface_hub` now enforces `namespace/name` for `hf://...` URIs.
        # The OpenWebText dataset is hosted at `Skylion007/openwebtext`.
        "dataset_name": "Skylion007/openwebtext",
        "dataset_config_name": None,
        "synthetic_num_samples": 16,
        "synthetic_validation_samples": 4,
        "synthetic_vocab_size": 128,
        # "dataset_name": "wikitext",
        # "dataset_config_name": "wikitext-103-v1",
        "train_file": None,
        "validation_file": None,
        "validation_split_percentage": 5,
        "model_name_or_path": "openai-community/gpt2-medium",
        # "model_name_or_path": "openai-community/gpt2",
        "config_name": None,
        "tokenizer_name": None,
        "use_slow_tokenizer": False,
        "per_device_train_batch_size": 32,

        "num_train_epochs": 2,
        # "max_train_steps": 500_000,
        "max_train_steps": 100_000,
        # "max_train_steps": 125_000,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 100,
        "seed": 123,
        "model_type": None,
        "block_size": 1024,
        "preprocessing_num_workers": 180,
        "overwrite_cache": False,
        "no_keep_linebreaks": False,
        "trust_remote_code": False,
        "checkpointing_steps": None,
        "resume_from_checkpoint": None,
        "save_final_model": False,  # paper runs evaluate live; skip ~1GB weight dumps
        "with_tracking": True,
        "report_to": "wandb",
        "low_cpu_mem_usage": False,
        "max_grad_norm": 1.0,
        "hf_path": None,
        "base_output_dir": "model-output",



        "compile": True,
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": True,

        "gradient_checkpointing": True,
        "fsdp2": False,
        "fsdp2_reshard_after_forward": True,

        # Only deepcopy a second model when explicitly enabled (saves a full
        # model worth of VRAM on the default path).
        "activation_probe": False,

        "num_workers": 8,
        "prefetch_factor": 2,
        "profile_timers": False,
        # When profile_timers is on, record CUDA-event breakdown every N optimizer steps
        # (not every step — per-step synchronize destroys throughput).
        "profile_every": 10,
        "log_every_n": 10,
        "csv_logging": False,
        "csv_flush_every": 100,

        "log_params_every_n": 100,

        # optimizer stats logging
        "track_stats_every_n": 200,
        
        # SNR-Adam metrics logging — amortized; prefer aggregates over per-param spam
        "log_snr_metrics_every_n": 50,
        # Per-param W&B keys (~1k+/step) are expensive (D2H + sync + wandb). Default off.
        "log_snr_per_param": False,
        # Host-side optimizer diagnostic syncs (.item()) cadence
        "diagnostics_every": 50,
        # torch.compile SNRAdam.step (default on when CUDA is available)
        "compile_optimizer": True,

        # model parameters
        "hidden_size": 1024,
        "depth": 12,
        "n_head": 8,

        # optimizer / Combo configuration
        "mode": "muon",
        # Boost parameters (exponential: boost = boost_base ^ combo)
        "boost_base": 1.03,  # Conservative: combo=30 → 2.4x boost
        "boost_min": 1.0,  # 1.0 means no slowdown, only speedup
        "boost_max": 8.0,
        # Importance mode settings (use_importance=True)
        # weight_scale=0: pure importance mode - boost = importance ^ power
        # weight_scale>0: combo + importance mode
        "use_importance": False,
        "importance_power": 1.5,  # boost = importance ^ power (power > 1 amplifies differences)
        "weight_scale": 0.0,  # 0 = pure importance, >0 = combo + importance
        "optim_warmup_steps": 0,
        # Additive bonus scales for modifier signals (0 = disabled)
        "grad_snr_weight": 0.0,  # Scale for gradient SNR bonus
        "activity_weight": 0.0,  # Scale for activity bonus
        "use_alignment_gating": False,  # Gate bonuses by cos_sim(grad, exp_avg)
        # PowerAdamW settings (for power_adam mode)
        "power": 1.5,  # Exponent for importance (>1 amplifies differences)
        "normalize": True,  # Whether to normalize importance by mean
        "normalize_mode": "global",  # "tensor" | "per_output" | "local_shard" | "global"
        # Pure coordinate-family calibration; no global/family interpolation.
        "score_family_mode": "none",
        "snr_mode": "coherence",  # "coherence" | "snr"
        "score_beta": None,  # None = share Adam beta2; else independent score horizon
        "boost_center_mode": "mean_log",
        "boost_center_scope": "global",
        "boost_center_every": 10,
        "snr_whiten_stat": "winsor",  # "mean_std" | "winsor"
        "snr_whiten_every": 10,
        "track_null_diagnostics": False,
        # Moment-only distribution snapshots and independent held-out probes.
        "score_snapshot_every": 0,
        "score_probe_every": 0,
        # Independent-gradient audits: opt-in, bounded eager/single-process path.
        "gradient_probe_every": 0,
        "gradient_probe_parameters": [],
        "gradient_probe_max_events": 3,
        "gradient_probe_selection_views": 4,
        "gradient_probe_audit_views": 4,
        "gradient_probe_batch_size": 2,
        "gradient_probe_fast_beta": 0.9,
        "gradient_probe_slow_beta": 0.99,
        "gradient_probe_seed": 71237,
        "gradient_probe_max_coordinates": 262144,
        "score_probe_max_batches": 0,
        "score_diagnostics_jsonl": True,
        "grad_z_clip": 4.0,
        "momentum_mode": "snr_adaptive",
        "adaptive_beta1_fast": 0.9,
        "adaptive_beta1_slow": 0.99,
        "mismatch_score_stage": "none",
        "mismatch_momentum_gate": "separate",
        "mismatch_strength": 0.0,
        "importance_weight": 1.0,  # Exponent for importance in geometric mean (when using aux)
        # Adafactor-style beta2 decay (optional for power_adam mode)
        "beta2_decay": None,  # e.g., -0.8 for Adafactor-style: beta2 = 1 - step^beta2_decay
        "separate_second_moment": False,
        "adam_combo_mode": "exp_avg_consistency",  # for Adam branch in ComboMuon: "exp_avg_consistency" | "grad_exp_avg_match" | "grad_consistency"

        "beta1": 0.9,
        "beta2": 0.999,

        "beta3": 0.95,
        # AdEMAMix baseline (same package/API as finetune_llm.py).
        "ademamix_beta3": 0.9999,
        "ademamix_alpha": 5.0,
        # None=no warmup, <=1=ratio of total steps, >1=absolute steps.
        "ademamix_beta3_warmup": None,
        "ademamix_alpha_warmup": None,

        "slow_add_alpha": 0.5,
        "slow_blend_warmup_steps": 0,

        # Update geometry: 1.0 = Adam, 0.0 = signum scaled by the boost.
        "update_magnitude_power": 1.0,
        "sign_source": "fast",
        "sign_blend_alpha": 0.5,
        "sign_rms_match": "none",
        "sign_match_stat": "rms",

        "learning_rate": 0.7e-4,
        "weight_decay": 0.01,

        "qk_norm": True,
        # 1/sqrt(2*depth) on the two projections writing into the residual
        # stream. Moves the baseline, so it stays a flag for A/B.
        "depth_scaled_residual_init": True,


        "m_slow_alpha_start": 0.25,

        # Muon optimizer settings
        "muon_lr": 0.02,
        "muon_momentum": 0.95,
        "muon_weight_decay": 0.0,
        "muon_ns_steps": 5,
        "muon_nesterov": True,
        # Threshold for routing large tensors (embeddings, lm_head) to Adam
        "muon_large_tensor_threshold": 16384,
        # SNR Muon-specific settings (used when mode=snr_muon)
        "muon_snr_beta": 0.999,
        "muon_snr_mode": "post_ns",  # "pre_ns" or "post_ns"

        # SOAP optimizer settings
        "soap_shampoo_beta": -1,
        "soap_precondition_frequency": 10,
        "soap_max_precond_dim": 10000,
        "soap_merge_dims": False,
        "soap_precondition_1d": False,
        "soap_normalize_grads": False,
        "soap_correct_bias": True,
        "snr_beta": -1.0,

        "hf_cache_dir": "/fsx/scratch/huggingface/hub",
        "tokenized_dataset_path": "/fsx/scratch/huggingface/tokenized/gpt2-medium_openwebtext_1024",


        "lr_warmup_mult": 1.0,

        "denoising_adam_batch_ratio": 0.5,
        "denoise_target": "momentum",
    }

    # Optional overrides from a JSON file for sweep scripts or manual runs
    if override_args:
        requested_mode = override_args.get("mode", args["mode"])
        removed = sorted(
            REMOVED_SNRADAM_CONFIG_KEYS.intersection(override_args)
            if requested_mode in ("snr_adam", "power_adam")
            else ()
        )
        if removed:
            joined = ", ".join(removed)
            raise ValueError(
                f"Removed SNRAdam config key(s): {joined}. "
                "See RESEARCH_HISTORY.md for the corresponding experiments."
            )
        args.update(override_args)

    if args["normalize_mode"] == "family" and args["score_family_mode"] == "none":
        raise ValueError(
            "normalize_mode='family' requires score_family_mode"
        )

    synthetic_data = args["dataset_name"] == "synthetic"
    if synthetic_data:
        config = None
        vocab_size = int(args["synthetic_vocab_size"])
    else:
        config = AutoConfig.from_pretrained(
            args['model_name_or_path'],
            trust_remote_code=args['trust_remote_code'],
        )
        vocab_size = config.vocab_size

        # if args.get("_override_name"):
        #     # For sweep runs we trust the JSON entry's name completely.
        #     run_name = args["_override_name"]
        # else:
    if True:
        # For manual runs (no override JSON), build a compact, descriptive name
        # from the hyperparameters.
        mode = args["mode"]
        
        if mode == "adamw" or mode == "base":
            # AdamW baseline: name by lr and wd
            lr = args.get("learning_rate", 5e-4)
            wd = args.get("weight_decay", 0.01)
            beta2 = args.get("beta2", 0.999)
            beta2_decay = args.get("beta2_decay")
            run_name = f"adamw-lr{lr}-b2{beta2}"
            if beta2_decay is not None:
                run_name += f"-b2d{beta2_decay}"
            run_name += f"-wd{wd}"
        
        elif mode == "muon":
            name_bits = ["Muon"]
            hp = {
                "lr": args.get("muon_lr", 0.02),
                "mom": args.get("muon_momentum", 0.95),
            }
            adam_lr = args.get("learning_rate", 3e-4)
            hp["alr"] = adam_lr
            
            name_bits.append(_compact_kwargs(hp))
            run_name = "-".join([bit for bit in name_bits if bit])

        elif mode == "ademamix":
            hp = {
                "lr": args.get("learning_rate", 3e-4),
                "b3": args.get("ademamix_beta3", 0.9999),
                "a": args.get("ademamix_alpha", 5.0),
            }
            if args.get("ademamix_beta3_warmup") is not None:
                hp["b3w"] = args["ademamix_beta3_warmup"]
            if args.get("ademamix_alpha_warmup") is not None:
                hp["aw"] = args["ademamix_alpha_warmup"]
            run_name = f"AdEMAMix-{_compact_kwargs(hp)}"

        elif mode in ("power_adam", "snr_adam"):
            name_bits = ["SNRAdam"]
            hp = {
                "lr": args.get("learning_rate", 3e-4),
                "bmin": args.get("boost_min", 0.25),
                "bmax": args.get("boost_max", 4.0),
                "scope": args.get("normalize_mode", "global"),
                "score": args.get("snr_mode", "coherence"),
                "stat": args.get("snr_whiten_stat", "winsor"),
                "center": args.get("boost_center_mode", "mean_log"),
                "momentum": args.get("momentum_mode", "snr_adaptive"),
                "clip": args.get("grad_z_clip", 4.0),
                "slow": args.get("slow_add_alpha", 0.5),
            }
            if args.get("score_family_mode", "none") != "none":
                hp["family"] = args["score_family_mode"]
            if args.get("mismatch_score_stage", "none") != "none":
                hp["mstage"] = args["mismatch_score_stage"]
                hp["mgate"] = args["mismatch_momentum_gate"]
                hp["mstr"] = args["mismatch_strength"]
            if args.get("score_beta") is not None:
                hp["sbeta"] = args["score_beta"]
            name_bits.append(_compact_kwargs(hp))
            run_name = "-".join([bit for bit in name_bits if bit])

        elif mode == "snr_muon":
            name_bits = ["SNRMuon"]
            hp = {
                "lr": args.get("muon_lr", 0.02),
                "mom": args.get("muon_momentum", 0.95),
                "p": args.get("power", 1.5),
            }
            hp["alr"] = args.get("learning_rate", 3e-4)
            hp["snrb"] = args.get("muon_snr_beta", 0.999)
            snr_mode = args.get("muon_snr_mode", "post_ns")
            if snr_mode != "post_ns":
                hp["snrm"] = snr_mode
            boost_min = args.get("boost_min", 0.5)
            if boost_min != 0.5:
                hp["bmin"] = boost_min
            boost_max = args.get("boost_max", 8.0)
            if boost_max != 8.0:
                hp["bmax"] = boost_max
            match_norm = args.get("match_norm", True)
            if not match_norm:
                hp["mn"] = 0
            sma = args.get("slow_m_alpha", 0.0)
            if sma != 0.0:
                hp["sma"] = sma
            name_bits.append(_compact_kwargs(hp))
            run_name = "-".join([bit for bit in name_bits if bit])

        elif mode == "soap":
            name_bits = ["SOAP"]
            hp = {}
            lr = args.get("learning_rate", 3e-3)
            hp["lr"] = lr
            betas = (args.get("beta1", 0.95), args.get("beta2", 0.95))
            hp["b1"] = betas[0]
            hp["b2"] = betas[1]
            pf = args.get("soap_precondition_frequency", 10)
            if pf != 10:
                hp["pf"] = pf
            name_bits.append(_compact_kwargs(hp))
            run_name = "-".join([bit for bit in name_bits if bit])

        elif mode == "snr_soap":
            name_bits = ["SNRSOAP"]
            hp = {}
            lr = args.get("learning_rate", 3e-3)
            hp["lr"] = lr
            betas = (args.get("beta1", 0.95), args.get("beta2", 0.95))
            hp["b1"] = betas[0]
            hp["b2"] = betas[1]
            pf = args.get("soap_precondition_frequency", 10)
            if pf != 10:
                hp["pf"] = pf
            power = args.get("power", 1.5)
            if power != 1.5:
                hp["pw"] = power
            snr_beta = args.get("snr_beta", -1.0)
            if snr_beta >= 0.0:
                hp["snrb"] = snr_beta
            snr_warmup = args.get("warmup_steps", 100)
            if snr_warmup != 100:
                hp["wu"] = snr_warmup
            sma = args.get("slow_m_alpha", 0.0)
            if sma != 0.0:
                hp["sma"] = sma
            name_bits.append(_compact_kwargs(hp))
            run_name = "-".join([bit for bit in name_bits if bit])

        else:
            # Fallback for unknown modes
            lr = args.get("learning_rate", 5e-4)
            run_name = f"{mode}-lr{lr}"

    if args.get("_override_name"):
        # Generated sweep configs use the filename as the reviewed run identity.
        run_name = args["_override_name"]

    args["run_name"] = run_name
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"
    args["wandb_run_name"] = run_name

    # Drop metadata helpers before namespacing.
    args.pop("_override_json_path", None)
    args.pop("_override_name", None)

    args = SimpleNamespace(**args)

    print("Running with the following arguments:", flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.output_dir is None:
        args.output_dir = time.strftime("run_%Y%m%d_%H%M%S")

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    fsdp_plugin = None
    if getattr(args, "fsdp2", False):
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=getattr(args, "fsdp2_reshard_after_forward", True),
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["TransformerBlock"],
            activation_checkpointing=False,
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        fsdp_plugin=fsdp_plugin,
        # Accelerate's DataLoaderShard does send_to_device inside next(). Default
        # non_blocking=False makes that a device sync, so host timers around
        # next() look like "100ms data loading" when they are mostly waiting on
        # the previous step's GPU work. Keep H2D async; we still stage tensors
        # below with non_blocking=.to().
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
        **accelerator_log_kwargs,
    )

    gradient_probe_config = None
    if type(args.gradient_probe_every) is not int or args.gradient_probe_every < 0:
        raise ValueError("gradient_probe_every must be a nonnegative integer")
    if args.gradient_probe_every:
        validate_probe_runtime(
            num_processes=accelerator.num_processes,
            compiled=args.compile,
            accumulation_steps=args.gradient_accumulation_steps,
            scaler=accelerator.scaler,
            gradient_checkpointing=args.gradient_checkpointing,
            fsdp=args.fsdp2,
            num_workers=args.num_workers,
        )
        if args.score_probe_every:
            raise ValueError("disable the legacy score_probe_every path during gradient audits")
        if args.mode not in ("adamw", "base", "ademamix", "snr_adam", "power_adam"):
            raise ValueError("gradient probes currently require AdamW, AdEMAMix, or SNRAdam geometry")
        if not isinstance(args.gradient_probe_parameters, (list, tuple)):
            raise ValueError("gradient_probe_parameters must be a list of exact parameter names")
        gradient_probe_config = ProbeConfig(
            parameter_names=tuple(args.gradient_probe_parameters),
            every=args.gradient_probe_every,
            max_events=args.gradient_probe_max_events,
            selection_views=args.gradient_probe_selection_views,
            audit_views=args.gradient_probe_audit_views,
            batch_size=args.gradient_probe_batch_size,
            sequence_length=args.block_size,
            fast_beta=args.gradient_probe_fast_beta,
            slow_beta=args.gradient_probe_slow_beta,
            seed=args.gradient_probe_seed,
            max_coordinates=args.gradient_probe_max_coordinates,
            history_batch_size=args.per_device_train_batch_size,
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = None
    if not synthetic_data:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )


    print("Creating model...", flush=True)
    model = Transformer(
        dim=args.hidden_size,
        depth=args.depth,
        heads=args.n_head,
        ff_mult=4,
        vocab_size=vocab_size,
        max_seq_len=args.block_size,
        gradient_checkpointing=args.gradient_checkpointing,
        qk_norm=args.qk_norm,
        depth_scaled_residual_init=getattr(
            args, "depth_scaled_residual_init", True
        ),
    )

    print("num parameters", sum(p.numel() for p in model.parameters()), flush=True)
    activation_probe_model = None
    if getattr(args, "activation_probe", False):
        print("Creating activation probe model...", flush=True)
        activation_probe_model = copy.deepcopy(model)
        for param in activation_probe_model.parameters():
            param.requires_grad_(False)
        activation_probe_model = activation_probe_model.to(accelerator.device)
    model = model.to(accelerator.device)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.token_embedding.weight.shape[0]
    if tokenizer is not None and len(tokenizer) > embedding_size:
        print("resizing token embeddings", len(tokenizer), embedding_size)
        model.resize_token_embeddings(len(tokenizer))

    # ---- Load or build tokenized dataset ----
    tok_path = getattr(args, "tokenized_dataset_path", None)

    if synthetic_data:
        lm_datasets = {
            "train": SyntheticTokenDataset(
                args.synthetic_num_samples,
                args.block_size,
                vocab_size,
                args.seed,
            ),
            "validation": SyntheticTokenDataset(
                args.synthetic_validation_samples,
                args.block_size,
                vocab_size,
                args.seed + 1,
            ),
        }
    elif tok_path and os.path.isdir(tok_path):
        logger.info(f"Loading pre-tokenized dataset from {tok_path}")
        print(f"Loading pre-tokenized dataset from {tok_path}", flush=True)
        lm_datasets = datasets.load_from_disk(tok_path)
    else:
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split={
                "train": f"train[{args.validation_split_percentage}%:]",
                "validation": f"train[:{args.validation_split_percentage}%]",
            },
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.hf_cache_dir,
            num_proc=args.preprocessing_num_workers,
        )

        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, config.max_position_embeddings)
        else:
            if args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(args.block_size, tokenizer.model_max_length)

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        def tokenize_and_group(examples):
            tokenized = tokenize_function(examples)
            concatenated_examples = {k: list(chain(*tokenized[k])) for k in tokenized.keys()}
            total_length = len(concatenated_examples[list(tokenized.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            # Targets are derived as a shift of input_ids at consumption time.
            return result

        print("Starting dataset tokenization...", flush=True)
        with accelerator.main_process_first():
            lm_datasets = raw_datasets.map(
                tokenize_and_group,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Tokenize + group into {block_size}",
            )
        print("Dataset tokenization complete.", flush=True)

        if tok_path and accelerator.is_main_process:
            logger.info(f"Saving tokenized dataset to {tok_path}")
            lm_datasets.save_to_disk(tok_path)

    if getattr(args, "prepare_dataset_only", False):
        print(f"Dataset preparation complete at {tok_path}", flush=True)
        return

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    loader_common = {
        "collate_fn": default_data_collator,
        "batch_size": args.per_device_train_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = max(1, int(getattr(args, "prefetch_factor", 2)))

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_common,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    # Optimizer
    # Weight decay exclusions using 1D param strategy (robust to naming conventions)
    # - 1D params: catches all biases and all normalization weights (LayerNorm, RMSNorm, etc.)
    # - Embeddings: 2D but shouldn't have weight decay
    optimizer_grouped_parameters = []
    param_names = []
    n_decay, n_no_decay = 0, 0
    trainable_named_parameters = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    score_family_mode = getattr(args, "score_family_mode", "none")
    score_family_counts = {}
    for name, param in trainable_named_parameters:
        lr_mult = getattr(param, "lr_mult", 1.0)
        # 1D params (biases, norm weights) or embeddings -> no weight decay
        if param.dim() == 1 or "embed" in name.lower():
            wd = 0.0
            n_no_decay += 1
        else:
            wd = args.weight_decay
            n_decay += 1
        parameter_group = {
            "params": [param],
            "weight_decay": wd,
            "lr": args.learning_rate * lr_mult,
            "score_name": name,
            "score_depth": parse_block_depth(name),
        }
        score_family = classify_calibration_family(
            name, param.ndim, score_family_mode
        )
        if score_family is not None:
            parameter_group["score_family"] = score_family
            score_family_counts[score_family] = (
                score_family_counts.get(score_family, 0) + param.numel()
            )
        optimizer_grouped_parameters.append(parameter_group)
        param_names.append(name)
    print(f"Params with weight decay: {n_decay}, without: {n_no_decay}")
    if score_family_counts:
        print(f"Score calibration family coordinates: {score_family_counts}")


    if args.mode in ("adamw", "base"):
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=1e-8,
            weight_decay=args.weight_decay,
            fused=True
        )



    elif args.mode == "ademamix":
        configured_total_steps = (
            int(args.max_train_steps)
            if args.max_train_steps is not None
            else math.ceil(
                len(train_dataloader) / args.gradient_accumulation_steps
            )
            * int(args.num_train_epochs)
        )
        optimizer = build_ademamix_optimizer(
            optimizer_grouped_parameters, args, configured_total_steps
        )

    elif args.mode in ("snr_adam", "power_adam"):
        optimizer = SNRAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=getattr(args, "eps", 1e-8),
            weight_decay=args.weight_decay,
            boost_min=getattr(args, "boost_min", 0.25),
            boost_max=getattr(args, "boost_max", 4.0),
            normalize_mode=getattr(args, "normalize_mode", "global"),
            snr_mode=getattr(args, "snr_mode", "coherence"),
            score_beta=getattr(args, "score_beta", None),
            boost_center_mode=getattr(args, "boost_center_mode", "mean_log"),
            boost_center_scope=getattr(args, "boost_center_scope", "global"),
            boost_center_every=getattr(args, "boost_center_every", 10),
            snr_whiten_stat=getattr(args, "snr_whiten_stat", "winsor"),
            snr_whiten_every=getattr(args, "snr_whiten_every", 10),
            grad_z_clip=getattr(args, "grad_z_clip", 4.0),
            momentum_mode=getattr(args, "momentum_mode", "snr_adaptive"),
            adaptive_beta1_fast=getattr(args, "adaptive_beta1_fast", 0.9),
            adaptive_beta1_slow=getattr(args, "adaptive_beta1_slow", 0.99),
            mismatch_score_stage=getattr(
                args, "mismatch_score_stage", "none"
            ),
            mismatch_momentum_gate=getattr(
                args, "mismatch_momentum_gate", "separate"
            ),
            mismatch_strength=getattr(args, "mismatch_strength", 0.0),
            warmup_steps=getattr(args, "power_warmup_steps", 100),
            slow_add_alpha=getattr(args, "slow_add_alpha", 0.5),
            slow_blend_warmup_steps=getattr(args, "slow_blend_warmup_steps", 0),
            update_magnitude_power=getattr(
                args, "update_magnitude_power", 1.0
            ),
            sign_source=getattr(args, "sign_source", "fast"),
            sign_blend_alpha=getattr(args, "sign_blend_alpha", 0.5),
            sign_rms_match=getattr(args, "sign_rms_match", "none"),
            sign_match_stat=getattr(args, "sign_match_stat", "rms"),
            track_null_diagnostics=getattr(args, "track_null_diagnostics", False),
            diagnostics_every=getattr(args, "diagnostics_every", 10),
            compile_optimizer=getattr(args, "compile_optimizer", True),
        )

    elif args.mode in ("soap", "snr_soap"):
        soap_base_kwargs = dict(
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            shampoo_beta=getattr(args, "soap_shampoo_beta", -1),
            eps=getattr(args, "eps", 1e-8),
            weight_decay=args.weight_decay,
            precondition_frequency=getattr(args, "soap_precondition_frequency", 10),
            max_precond_dim=getattr(args, "soap_max_precond_dim", 10000),
            merge_dims=getattr(args, "soap_merge_dims", False),
            precondition_1d=getattr(args, "soap_precondition_1d", False),
            normalize_grads=getattr(args, "soap_normalize_grads", False),
            correct_bias=getattr(args, "soap_correct_bias", True),
        )
        if args.mode == "soap":
            snr_kwargs = dict(warmup_steps=999_999_999)
        else:
            snr_kwargs = dict(
                power=getattr(args, "power", 1.5),
                boost_min=getattr(args, "boost_min", 0.5),
                boost_max=getattr(args, "boost_max", 8.0),
                normalize_mode=getattr(args, "normalize_mode", "matrix"),
                block_size=getattr(args, "block_size", 512),
                match_norm=getattr(args, "match_norm", True),
                warmup_steps=getattr(args, "warmup_steps", 100),
                slow_m_alpha=getattr(args, "slow_m_alpha", 0.0),
            )
        optimizer = SNRSOAP(
            optimizer_grouped_parameters,
            **soap_base_kwargs,
            **snr_kwargs,
        )

    else:
        raise ValueError(f"Unknown optimizer mode: {args.mode}. "
                         "Choose from: adamw, ademamix, muon, snr_muon, "
                         "snr_adam, soap, snr_soap")

    # Print optimizer param group settings (excluding raw parameter tensors).
    print("Optimizer param groups:", flush=True)
    for idx, group in enumerate(optimizer.param_groups):
        group_view = {k: v for k, v in group.items() if k != "params"}
        print(f"  Group {idx}: {group_view}")

    if (
        args.mode in ("snr_adam", "power_adam")
        and bool(getattr(args, "compile_optimizer", True))
    ):
        # LambdaLR updates tensor LRs in-place. The compiled optimizer consumes
        # a stable scalar tensor rather than specializing on each Python float
        # emitted by the schedule.
        make_optimizer_lr_capturable(optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Allow optimizer to scale LR scheduler warmup without touching beta/alpha warmups.
    # e.g. num_warmup_steps=2000 and optimizer lr_warmup_mult=5.0 -> 10_000 warmup steps.
    lr_warmup_mult = float(optimizer.param_groups[0].get("lr_warmup_mult", 1.0))
    base_num_warmup_steps = int(args.num_warmup_steps or 0) * accelerator.num_processes
    scaled_num_warmup_steps = int(round(base_num_warmup_steps * lr_warmup_mult))

    # Clamp to avoid pathological schedules where warmup exceeds total training steps.
    total_sched_steps = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    scaled_num_warmup_steps = min(scaled_num_warmup_steps, int(total_sched_steps))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=scaled_num_warmup_steps,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator` first, then compile the
    # device-placed (and DDP-wrapped, if any) module. Compiling after prepare
    # generally yields a better graph for reduce-overhead / cudagraphs.
    print("Preparing accelerator...", flush=True)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    print("Accelerator ready.", flush=True)

    if args.compile:
        if (
            int(getattr(args, "gradient_accumulation_steps", 1)) > 1
            and args.compile_mode == "reduce-overhead"
        ):
            # CUDA Graphs reuse static buffers across captures. With grad-accum,
            # a later microbatch forward can overwrite activations still needed
            # by an earlier backward → RuntimeError on tok_emb / graph outputs.
            print(
                "[compile] gradient_accumulation_steps>1 is incompatible with "
                "reduce-overhead CUDAGraphs; falling back to compile_mode=default",
                flush=True,
            )
            args.compile_mode = "default"

        if args.compile_mode == "reduce-overhead":
            # The transformer lazily allocates RoPE tables on the first forward.
            # Build them before torch.compile so CUDAGraphs do not capture those
            # cache tensors as graph outputs that are reused on later steps.
            seq_len = max(1, int(args.block_size) - 1)
            dummy_input = torch.zeros((1, seq_len), dtype=torch.long, device=accelerator.device)
            dummy_targets = torch.zeros((1, seq_len), dtype=torch.long, device=accelerator.device)
            was_training = model.training
            model.eval()
            with torch.no_grad(), accelerator.autocast():
                _ = model(input_ids=dummy_input, targets=dummy_targets)
            if was_training:
                model.train()
            del dummy_input, dummy_targets
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(
            f"Compiling model (mode={args.compile_mode}, fullgraph={args.compile_fullgraph})...",
            flush=True,
        )
        model = torch.compile(
            model, mode=args.compile_mode, fullgraph=args.compile_fullgraph
        )

    cudagraph_mark_step_begin = None
    if args.compile and args.compile_mode == "reduce-overhead":
        cudagraph_mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        init_kwargs = {
            "wandb": {
                "name": args.wandb_run_name,
            }
        }
        mode = getattr(args, "mode", "base")
        project_name = "SNRAdam"
        print(f"Initializing wandb tracker ({project_name})...", flush=True)
        accelerator.init_trackers(project_name, experiment_config, init_kwargs=init_kwargs)
        print("Wandb initialized.", flush=True)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    gradient_collector = None
    if gradient_probe_config is not None:
        dataset_identity = getattr(train_dataset, "_fingerprint", None)
        if synthetic_data:
            dataset_identity = (
                f"synthetic:{args.seed}:{args.synthetic_num_samples}:"
                f"{args.synthetic_vocab_size}:{args.block_size}"
            )
        if dataset_identity is None:
            raise ValueError("gradient probes require a fingerprinted map-style training dataset")
        gradient_collector = GradientProbeCollector(
            accelerator.unwrap_model(model), gradient_probe_config,
            dataset_size=len(train_dataset), dataset_id=str(dataset_identity),
        )
        # Register BEFORE load_state: checkpoints without the histories/private
        # probe RNG cannot silently cold-start observers on a late trajectory.
        accelerator.register_for_checkpointing(gradient_collector)
    

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

        if gradient_collector is not None and resume_step not in (None, 0):
            raise ValueError(
                "gradient probes currently require epoch-boundary resume: "
                "the trainer's shuffled data order is not replayed exactly mid-epoch; "
                "use an epoch-boundary checkpoint or start a fresh audit run"
            )

    # update the progress_bar if load from checkpoint
    if gradient_collector is not None and gradient_collector.observed_steps != completed_steps:
        raise ValueError("gradient observer checkpoint does not match resumed training step")
    gradient_probe_writer = None
    if gradient_collector is not None:
        probe_log_path = os.path.join(args.output_dir, "gradient_probes.jsonl")
        validate_probe_log(probe_log_path, completed_steps)
        gradient_probe_writer = CompactJSONLWriter(
            probe_log_path,
            append=bool(completed_steps > 0), flush=True,
        )
    progress_bar.update(completed_steps)

    snr_metrics_tracker = ParamGradMetrics() if args.mode in ("power_adam", "snr_adam", "snr_soap") else None
    # Build param -> name mapping from optimizer param groups (most reliable way)
    if snr_metrics_tracker is not None:
        # Get unwrapped optimizer
        opt_for_names = getattr(optimizer, "optimizer", optimizer)
        # Build name map: we need to match params in optimizer to their names
        # Use the param_names list we built earlier (same order as optimizer_grouped_parameters)
        param_to_name = {}
        param_idx = 0
        for group in opt_for_names.param_groups:
            for p in group["params"]:
                if param_idx < len(param_names):
                    param_to_name[p] = param_names[param_idx]
                param_idx += 1
        print(f"[SNR Metrics] Built param_to_name map with {len(param_to_name)} params", flush=True)
    else:
        param_to_name = None

    score_snapshot_every = max(
        0, int(getattr(args, "score_snapshot_every", 0))
    )
    score_probe_every = max(0, int(getattr(args, "score_probe_every", 0)))
    score_probe_max_batches = max(
        0, int(getattr(args, "score_probe_max_batches", 0))
    )
    score_probe_count = 0
    score_probe_iter = iter(eval_dataloader) if score_probe_every > 0 else None
    score_diagnostic_writer = None
    if (
        (score_snapshot_every > 0 or score_probe_every > 0)
        and accelerator.is_main_process
        and bool(getattr(args, "score_diagnostics_jsonl", True))
    ):
        score_diagnostic_writer = CompactJSONLWriter(
            os.path.join(args.output_dir, "score_diagnostics.jsonl"),
            append=bool(completed_steps > 0),
            flush=True,
        )

    def collect_score_records(
        opt, diagnostic_step: int, *, include_heldout: bool
    ):
        local_records = []
        if param_to_name is None:
            return local_records
        for group in opt.param_groups:
            group_beta1, group_beta2 = group.get(
                "betas", (args.beta1, args.beta2)
            )
            group_score_beta = group.get("score_beta")
            for parameter in group["params"]:
                state = opt.state.get(parameter, {})
                if "m_matched" not in state:
                    continue
                if include_heldout and parameter.grad is None:
                    continue
                name = param_to_name.get(
                    parameter, group.get("score_name", f"param_{id(parameter)}")
                )
                record = compute_score_diagnostics(
                    name,
                    state,
                    beta2=float(group_beta2),
                    beta1=float(group_beta1),
                    score_beta=group_score_beta,
                    heldout_gradient=(
                        parameter.grad if include_heldout else None
                    ),
                )
                record.update({
                    "probe_step": int(diagnostic_step),
                    "record_type": (
                        "heldout_probe" if include_heldout else "snapshot"
                    ),
                    "batch_size": int(args.per_device_train_batch_size)
                    * int(accelerator.num_processes),
                    "model_depth": int(args.depth),
                    "seed": int(args.seed),
                    "normalization_scope": getattr(
                        args, "normalize_mode", "global"
                    ),
                    "calibration_stat": getattr(
                        args, "snr_whiten_stat", "winsor"
                    ),
                })
                local_records.append(record)
        return distributed_reduce_records(local_records)

    # Allocated and reserved memory (zero for CPU smoke tests).
    allocated_memory = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )
    reserved_memory = (
        torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    )
    progress_bar.set_postfix(vram=f"{reserved_memory / (1024 ** 3):.2f} GB")

    log_every_n = max(1, int(getattr(args, "log_every_n", 10)))
    profile_timers = bool(getattr(args, "profile_timers", False)) and torch.cuda.is_available()
    profile_every = max(1, int(getattr(args, "profile_every", log_every_n)))
    csv_log = (
        CSVLogger(
            os.path.join(args.output_dir, "metrics.csv"),
            flush_every=max(1, int(getattr(args, "csv_flush_every", 100))),
        )
        if accelerator.is_main_process and bool(getattr(args, "csv_logging", False))
        else None
    )
    train_wall_start = time.perf_counter()
    throughput_window_start = train_wall_start
    throughput_window_step = completed_steps
    training_start_step = completed_steps
    optimizer_state_bytes_cache = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # CUDA events only fire on profile_every steps; data load uses host time
    # because next(dataloader) is CPU-side and invisible to the GPU clock.
    if profile_timers:
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        optimizer_start = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        if accelerator.is_main_process:
            print(
                f"[profile] every {profile_every} optimizer steps: "
                f"input_stall=host (sync then next), "
                f"h2d/fwd/bwd/opt=cuda events. "
                f"Note: accelerate next() includes device H2D; "
                f"input_stall is the true post-GPU bubble.",
                flush=True,
            )

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        dataloader_iter = iter(active_dataloader)
        for step in range(len(active_dataloader)):
            model.train()

            # Profile the upcoming optimizer step (completed_steps increments after sync).
            do_profile = profile_timers and ((completed_steps + 1) % profile_every == 0)
            input_stall_ms = None
            if do_profile:
                # Drain prior GPU work first. Otherwise accelerate's (even async)
                # next()/H2D can look like "data loading" while it is really
                # waiting on the previous backward/optimizer kernels.
                torch.cuda.synchronize()
                stall_t0 = time.perf_counter()
            batch = next(dataloader_iter)
            if do_profile:
                input_stall_ms = (time.perf_counter() - stall_t0) * 1000.0
            
            with accelerator.accumulate(model):
                # Only time full update iterations (skip grad-accum microbatches).
                do_profile = do_profile and accelerator.sync_gradients

                if do_profile:
                    h2d_start.record()
                # Batch may already be on-device via accelerate; .to(non_blocking)
                # is cheap then. Slice stays on device. Targets are a shift of
                # the inputs, so one transfer feeds both.
                tokens = batch["input_ids"].to(accelerator.device, non_blocking=True)
                input_ids = tokens[:, :-1]
                targets = tokens[:, 1:]
                if do_profile:
                    h2d_end.record()
                    forward_start.record()

                if cudagraph_mark_step_begin is not None:
                    cudagraph_mark_step_begin()
                loss = model(input_ids=input_ids, targets=targets)
                if do_profile:
                    forward_end.record()
                
                # Sync loss across GPUs for accurate metrics (needed for multi-GPU training)
                synced_loss = loss.detach().float()
                if accelerator.num_processes > 1:
                    synced_loss = accelerator.gather(synced_loss).mean()
                
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += synced_loss
                
                if do_profile:
                    backward_start.record()
                accelerator.backward(loss)
                if do_profile:
                    backward_end.record()
                
                # clip the gradients
                mini_logs = {
                    "step_loss": synced_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }

                # Detect scout optimizer and whether scouting happens this step
                underlying_opt = getattr(optimizer, 'optimizer', optimizer)

                if do_profile:
                    optimizer_start.record()

                if gradient_collector is not None:
                    # BF16/no scaling, no accumulation on the supported path.
                    gradient_collector.observe(completed_steps + 1)
                if args.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    mini_logs["grad_norm"] = grad_norm

                optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if gradient_collector is not None:
                    gradient_probe_writer.write_many(gradient_collector.probe(
                        model, train_dataset, optimizer, completed_steps + 1,
                        autocast=accelerator.autocast,
                    ))

                # Moment-only snapshots are cheap. Held-out probes additionally
                # run one validation backward pass at the updated weights.
                diagnostic_step = completed_steps + 1
                should_snapshot = (
                    accelerator.sync_gradients
                    and score_snapshot_every > 0
                    and diagnostic_step % score_snapshot_every == 0
                )
                should_probe = (
                    accelerator.sync_gradients
                    and score_probe_every > 0
                    and diagnostic_step % score_probe_every == 0
                    and (
                        score_probe_max_batches == 0
                        or score_probe_count < score_probe_max_batches
                    )
                )
                diagnostic_records = []
                diagnostic_loss = None
                if should_probe:
                    try:
                        probe_batch = next(score_probe_iter)
                    except StopIteration:
                        score_probe_iter = iter(eval_dataloader)
                        probe_batch = next(score_probe_iter)
                    probe_tokens = probe_batch["input_ids"].to(
                        accelerator.device, non_blocking=True
                    )
                    probe_input_ids = probe_tokens[:, :-1]
                    probe_targets = probe_tokens[:, 1:]
                    probe_loss = model(
                        input_ids=probe_input_ids, targets=probe_targets
                    )
                    diagnostic_loss = probe_loss.detach()
                    accelerator.backward(
                        probe_loss
                        * max(1, int(args.gradient_accumulation_steps))
                    )
                    accelerator.unscale_gradients(optimizer)
                    probe_opt = getattr(optimizer, "optimizer", optimizer)
                    diagnostic_records = collect_score_records(
                        probe_opt, diagnostic_step, include_heldout=True
                    )
                    optimizer.zero_grad(set_to_none=True)
                    score_probe_count += 1
                elif should_snapshot:
                    snapshot_opt = getattr(optimizer, "optimizer", optimizer)
                    diagnostic_records = collect_score_records(
                        snapshot_opt, diagnostic_step, include_heldout=False
                    )

                if diagnostic_records:
                    if score_diagnostic_writer is not None:
                        score_diagnostic_writer.write_many(diagnostic_records)
                    if args.with_tracking:
                        family_values = {}
                        for record in diagnostic_records:
                            family = record["family"]
                            for key in (
                                "q",
                                "rgroup",
                                "q_debiased",
                                "heldout_projection",
                                "heldout_update_alignment",
                                "output_q_mean",
                                "output_rgroup_mean",
                                "output_q_debiased_mean",
                                "output_heldout_projection_mean",
                                "output_heldout_update_alignment_mean",
                            ):
                                if key not in record:
                                    continue
                                weight_key = (
                                    "output_group_count"
                                    if key.startswith("output_")
                                    else "numel"
                                )
                                family_values.setdefault(
                                    (family, key), []
                                ).append((
                                    float(record[key]),
                                    max(1, int(record.get(weight_key, 1))),
                                ))
                        diagnostic_logs = {}
                        for (family, key), values in family_values.items():
                            tensor_mean = (
                                sum(value for value, _ in values) / len(values)
                            )
                            total_coordinates = sum(
                                weight for _, weight in values
                            )
                            coordinate_mean = sum(
                                value * weight for value, weight in values
                            ) / total_coordinates
                            diagnostic_logs[
                                f"score_diagnostics/{family}/{key}"
                            ] = tensor_mean
                            diagnostic_logs[
                                f"score_diagnostics/{family}/{key}_"
                                f"{'output' if key.startswith('output_') else 'coordinate'}_weighted"
                            ] = coordinate_mean
                        if diagnostic_loss is not None:
                            diagnostic_logs["score_diagnostics/probe_loss"] = (
                                float(diagnostic_loss.float().item())
                            )
                        accelerator.log(
                            diagnostic_logs, step=diagnostic_step
                        )
                if do_profile:
                    optimizer_end.record()
                    # One sync per profiled step — never on the hot path.
                    torch.cuda.synchronize()
                    h2d_ms = h2d_start.elapsed_time(h2d_end)
                    fwd_ms = forward_start.elapsed_time(forward_end)
                    bwd_ms = backward_start.elapsed_time(backward_end)
                    opt_ms = optimizer_start.elapsed_time(optimizer_end)
                    compute_ms = h2d_ms + fwd_ms + bwd_ms + opt_ms
                    mini_logs.update(
                        {
                            "timer/input_stall_ms": float(input_stall_ms),
                            # Keep old key as alias so existing CSV/W&B panels keep working.
                            "timer/data_ms": float(input_stall_ms),
                            "timer/h2d_ms": float(h2d_ms),
                            "timer/forward_ms": float(fwd_ms),
                            "timer/backward_ms": float(bwd_ms),
                            "timer/optimizer_ms": float(opt_ms),
                            "timer/compute_ms": float(compute_ms),
                            "timer/step_ms": float(input_stall_ms) + float(compute_ms),
                        }
                    )
                    if accelerator.is_main_process:
                        print(
                            f"[profile] step={completed_steps + 1} "
                            f"stall={input_stall_ms:.1f}ms h2d={h2d_ms:.1f}ms "
                            f"fwd={fwd_ms:.1f}ms bwd={bwd_ms:.1f}ms "
                            f"opt={opt_ms:.1f}ms compute={compute_ms:.1f}ms "
                            f"step={input_stall_ms + compute_ms:.1f}ms",
                            flush=True,
                        )
                
                # Log update norm if available
                opt = getattr(optimizer, "optimizer", optimizer)  # unwrap Accelerate wrapper if present
                if hasattr(opt, "last_update_norm"):
                    mini_logs["optim/update_norm"] = opt.last_update_norm
                
                log_step = completed_steps + (1 if accelerator.sync_gradients else 0)
                should_log_step = (
                    accelerator.sync_gradients
                    and log_step > 0
                    and log_step % log_every_n == 0
                )
                if should_log_step:
                    now = time.perf_counter()
                    elapsed = max(now - throughput_window_start, 1e-9)
                    update_steps = max(log_step - throughput_window_step, 1)
                    sequences = update_steps * total_batch_size
                    tokens_per_sequence = max(1, int(args.block_size) - 1)
                    if optimizer_state_bytes_cache is None:
                        optimizer_state_bytes_cache = optimizer_state_bytes(optimizer)
                    sequences_per_second = sequences / elapsed
                    tokens_per_second = (
                        sequences * tokens_per_sequence / elapsed
                    )
                    peak_allocated = (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else 0
                    )
                    peak_reserved = (
                        torch.cuda.max_memory_reserved()
                        if torch.cuda.is_available()
                        else 0
                    )
                    mini_logs.update(
                        {
                            "wall_time_seconds": now - train_wall_start,
                            "throughput/sequences_per_second": sequences_per_second,
                            "throughput/tokens_per_second": tokens_per_second,
                            "memory/max_allocated_gb": peak_allocated / (1024 ** 3),
                            "memory/max_reserved_gb": peak_reserved / (1024 ** 3),
                            "efficiency/elapsed_wall_seconds": now - train_wall_start,
                            "efficiency/sequences_per_second": sequences_per_second,
                            "efficiency/tokens_per_second": tokens_per_second,
                            "efficiency/peak_cuda_memory_bytes": float(peak_allocated),
                            "efficiency/optimizer_state_bytes": float(
                                optimizer_state_bytes_cache
                            ),
                        }
                    )
                    throughput_window_start = now
                    throughput_window_step = log_step
                if args.with_tracking and should_log_step:
                    accelerator.log(
                        mini_logs,
                        step=log_step,
                    )
                elif (
                    do_profile
                    and args.with_tracking
                    and accelerator.sync_gradients
                ):
                    # profile_every may not divide log_every_n — still emit timers.
                    accelerator.log(
                        {k: v for k, v in mini_logs.items() if str(k).startswith("timer/")},
                        step=log_step,
                    )

                if csv_log is not None and should_log_step:
                    csv_metrics = {"step": log_step}
                    for key, value in mini_logs.items():
                        if isinstance(value, (int, float)):
                            csv_metrics[key] = float(value)
                        elif torch.is_tensor(value) and value.numel() == 1:
                            csv_metrics[key] = float(value.detach().item())
                    csv_log.log(csv_metrics)
                elif csv_log is not None and do_profile and accelerator.sync_gradients:
                    csv_metrics = {"step": log_step}
                    for key, value in mini_logs.items():
                        if str(key).startswith("timer/") and isinstance(value, (int, float)):
                            csv_metrics[key] = float(value)
                    csv_log.log(csv_metrics)

                # Log optimizer stats when gathered this step
                opt = getattr(optimizer, "optimizer", optimizer)  # unwrap Accelerate wrapper if present
                if getattr(opt, "just_gathered_stats", False):
                    stats_logs = {}
                    for name, stats in getattr(opt, "latest_stats", {}).items():
                        clean = name.replace(".", "/")
                        for k, v in stats.items():
                            if k == "name" or v is None:
                                continue
                            key = f"optim/{clean}/{k}"
                            if torch.is_tensor(v):
                                v = v.item()
                            if isinstance(v, (int, float)):
                                stats_logs[key] = float(v)
                    if stats_logs and args.with_tracking:
                        accelerator.log(stats_logs, step=completed_steps)
                
                # Log SNR-Adam metrics every N steps. Per-param mode emits ~1k W&B
                # keys and full-tensor diagnostics; aggregate mode keeps a handful.
                log_snr_per_param = bool(getattr(args, "log_snr_per_param", False))
                if (snr_metrics_tracker is not None 
                    and param_to_name is not None
                    and completed_steps > 0 
                    and completed_steps % args.log_snr_metrics_every_n == 0):
                    snr_logs = {}
                    num_params_logged = 0
                    agg = {
                        "snr_mean": [],
                        "snr_std": [],
                        "boost_median": [],
                        "boost_std": [],
                        "boost_p10": [],
                        "boost_p90": [],
                        "boost_frac_at_min_clamp": [],
                        "boost_frac_at_max_clamp": [],
                    }
                    
                    # Iterate over optimizer state directly (most reliable)
                    for param, state in opt.state.items():
                        # SNRAdam uses m/v/m_matched; older paths used exp_avg*
                        if "m_matched" not in state and "exp_avg" not in state:
                            continue
                        
                        # Get param name
                        name = param_to_name.get(param, f"param_{id(param)}")
                        
                        # Use simpler method that only needs optimizer state (no grad needed)
                        metrics_beta = args.beta2
                        if args.mode == "snr_soap" and getattr(args, "snr_beta", -1.0) >= 0:
                            metrics_beta = args.snr_beta
                        score_beta = getattr(args, "score_beta", None)
                        if score_beta is not None:
                            metrics_beta = float(score_beta)
                        stats = snr_metrics_tracker.compute_snr_stats(
                            optimizer_state=state,
                            power=args.power,
                            boost_min=args.boost_min,
                            boost_max=args.boost_max,
                            normalize=args.normalize,
                            beta2=metrics_beta,
                            score_beta=score_beta,
                        )
                        if stats is None:
                            continue
                        num_params_logged += 1

                        if log_snr_per_param:
                            flat_name = name.replace(".", "_")
                            snr_logs[f"snr/snr_mean/{flat_name}"] = stats["snr_mean"]
                            snr_logs[f"snr/snr_std/{flat_name}"] = stats["snr_std"]
                            snr_logs[f"snr/boost_median/{flat_name}"] = stats["boost_median"]
                            snr_logs[f"snr/boost_std/{flat_name}"] = stats["boost_std"]
                            snr_logs[f"snr/boost_p10/{flat_name}"] = stats["boost_p10"]
                            snr_logs[f"snr/boost_p90/{flat_name}"] = stats["boost_p90"]
                            snr_logs[f"snr/frac_min_clamp/{flat_name}"] = stats["boost_frac_at_min_clamp"]
                            snr_logs[f"snr/frac_max_clamp/{flat_name}"] = stats["boost_frac_at_max_clamp"]
                        else:
                            for key in agg:
                                agg[key].append(float(stats[key]))

                    if not log_snr_per_param and num_params_logged:
                        for key, values in agg.items():
                            snr_logs[f"snr_agg/{key}"] = sum(values) / len(values)
                        snr_logs["snr_agg/num_params"] = float(num_params_logged)
                    
                    # Debug: log how many params we tracked
                    if completed_steps == args.log_snr_metrics_every_n:
                        mode = "per-param" if log_snr_per_param else "aggregate"
                        print(
                            f"[SNR Metrics] {mode}: {num_params_logged} params, "
                            f"{len(snr_logs)} metrics",
                            flush=True,
                        )
                    
                    if snr_logs and args.with_tracking:
                        accelerator.log(snr_logs, step=completed_steps)

                if (
                    args.with_tracking
                    and args.mode in ("snr_adam", "power_adam")
                    and completed_steps > 0
                    and completed_steps % args.log_snr_metrics_every_n == 0
                ):
                    diag_logs = {}
                    if log_snr_per_param and hasattr(opt, "get_diagnostics"):
                        # Expensive: recomputes signal/score/null stats per tensor.
                        for param_key, diag in opt.get_diagnostics().items():
                            for key, value in diag.items():
                                if not isinstance(value, (int, float)):
                                    continue
                                if key.startswith("null_"):
                                    diag_logs[f"snr_null/{key}/{param_key}"] = float(value)
                                elif key.startswith("beta1_eff_"):
                                    diag_logs[f"snr_momentum/{key}/{param_key}"] = float(value)
                                elif key == "grad_clip_fraction":
                                    diag_logs[f"snr_clipping/{key}/{param_key}"] = float(value)
                                elif key.startswith("slow_") or key == "final_update_rms":
                                    diag_logs[f"snr_slow/{key}/{param_key}"] = float(value)
                                elif key in (
                                    "signal_mean",
                                    "signal_std",
                                    "signal_max",
                                    "score_mean",
                                    "score_std",
                                    "boost_mean_log",
                                    "boost_k_star",
                                    "boost_redistribution_frac",
                                    "score_beta_eff",
                                    "global_score_loc",
                                    "global_score_scale",
                                    "global_boost_mean_log",
                                    "group_q_mean",
                                    "group_null_ratio_mean",
                                    "group_debiased_c2_mean",
                                ):
                                    diag_logs[f"snr_boost/{key}/{param_key}"] = float(value)
                    else:
                        # Aggregate all cheap scalars captured by optimizer.step.
                        # Unknown future diagnostics get a generic key instead
                        # of being silently omitted from W&B.
                        diag_logs = aggregate_snr_diagnostics(opt)
                    if diag_logs:
                        accelerator.log(diag_logs, step=completed_steps)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    if gradient_collector is None:
                        accelerator.save_state(output_dir)
            if completed_steps % args.validate_every == 0:
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        tokens = batch["input_ids"].to(accelerator.device, non_blocking=True)
                        input_ids = tokens[:, :-1]
                        targets = tokens[:, 1:]
                        if cudagraph_mark_step_begin is not None:
                            cudagraph_mark_step_begin()
                        loss = model(input_ids=input_ids, targets=targets)
                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_train_batch_size)))
                    if args.num_validation_batches is not None:
                        if step >= args.num_validation_batches:
                            break

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

                if args.with_tracking:
                    eval_metrics = {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    }
                    accelerator.log(eval_metrics, step=completed_steps)

                    if csv_log is not None:
                        csv_log.log({"step": completed_steps, **{k: float(v) for k, v in eval_metrics.items()}})

            if (
                gradient_collector is not None
                and isinstance(checkpointing_steps, int)
                and completed_steps % checkpointing_steps == 0
            ):
                # The opt-in audit checkpoint includes any scheduled validation
                # and its RNG consumption. Otherwise an epoch-boundary resume
                # could omit validation that originally followed the checkpoint.
                output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    # One terminal synchronization makes the final paper metric include all GPU
    # work without putting a synchronization on the per-step hot path.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    final_wall_seconds = max(time.perf_counter() - train_wall_start, 1e-9)
    run_update_steps = max(completed_steps - training_start_step, 0)
    run_sequences = run_update_steps * total_batch_size
    run_tokens = run_sequences * max(1, int(args.block_size) - 1)
    if optimizer_state_bytes_cache is None:
        optimizer_state_bytes_cache = optimizer_state_bytes(optimizer)
    peak_cuda_memory_bytes = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    final_efficiency_metrics = {
        "efficiency/elapsed_wall_seconds": final_wall_seconds,
        "efficiency/sequences_per_second": run_sequences / final_wall_seconds,
        "efficiency/tokens_per_second": run_tokens / final_wall_seconds,
        "efficiency/peak_cuda_memory_bytes": float(peak_cuda_memory_bytes),
        "efficiency/optimizer_state_bytes": float(optimizer_state_bytes_cache),
    }
    if args.with_tracking:
        accelerator.log(final_efficiency_metrics, step=completed_steps)
    if accelerator.is_main_process:
        print(
            "[efficiency] "
            f"tokens/s={final_efficiency_metrics['efficiency/tokens_per_second']:.2f} "
            f"sequences/s={final_efficiency_metrics['efficiency/sequences_per_second']:.2f} "
            f"wall={final_wall_seconds:.2f}s "
            f"peak_cuda={peak_cuda_memory_bytes}B "
            f"optimizer_state={optimizer_state_bytes_cache}B",
            flush=True,
        )
    if csv_log is not None:
        csv_log.log({"step": completed_steps, **final_efficiency_metrics})
        csv_log.close()
    if score_diagnostic_writer is not None:
        score_diagnostic_writer.close()
    if gradient_probe_writer is not None:
        gradient_probe_writer.close()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        save_final_model = bool(getattr(args, "save_final_model", False))
        if save_final_model:
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_orig_mod"):
                unwrapped_model = unwrapped_model._orig_mod
            print("Saving model to", args.output_dir)
            save_pretrained = getattr(unwrapped_model, "save_pretrained", None)
            if callable(save_pretrained):
                save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
            else:
                # Local Transformer is a plain nn.Module, not a Hugging Face PreTrainedModel.
                state_dict = accelerator.get_state_dict(model)
                if accelerator.is_main_process:
                    accelerator.save(
                        state_dict,
                        os.path.join(args.output_dir, "pytorch_model.bin"),
                    )
                    model_config = {
                        "model_class": "Transformer",
                        "hidden_size": args.hidden_size,
                        "depth": args.depth,
                        "n_head": args.n_head,
                        "ff_mult": 4,
                        "vocab_size": vocab_size,
                        "max_seq_len": args.block_size,
                        "gradient_checkpointing": args.gradient_checkpointing,
                        "qk_norm": args.qk_norm,
                        "depth_scaled_residual_init": args.depth_scaled_residual_init,
                    }
                    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
                        json.dump(model_config, f, indent=2)
            if accelerator.is_main_process and tokenizer is not None:
                tokenizer.save_pretrained(args.output_dir)
        elif accelerator.is_main_process:
            print(
                "Skipping final model/tokenizer save "
                f"(save_final_model={save_final_model}); writing results only to",
                args.output_dir,
            )
            # Still write a tiny model_config for provenance when weights are skipped.
            model_config_path = os.path.join(args.output_dir, "model_config.json")
            if not os.path.exists(model_config_path):
                with open(model_config_path, "w") as f:
                    json.dump(
                        {
                            "model_class": "Transformer",
                            "hidden_size": args.hidden_size,
                            "depth": args.depth,
                            "n_head": args.n_head,
                            "ff_mult": 4,
                            "vocab_size": vocab_size,
                            "max_seq_len": args.block_size,
                            "gradient_checkpointing": args.gradient_checkpointing,
                            "qk_norm": args.qk_norm,
                            "depth_scaled_residual_init": args.depth_scaled_residual_init,
                            "weights_saved": False,
                        },
                        f,
                        indent=2,
                    )

        if accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(
                    {
                        "perplexity": locals().get("perplexity"),
                        **final_efficiency_metrics,
                    },
                    f,
                    indent=2,
                )

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()

"""
LLM Fine-tuning Script for Instruction Tuning

Supports:
- SNRAdam vs standard AdamW comparison
- Mistral-7B or LLaMA-2-7B fine-tuning
- Alpaca dataset (52k instruction examples)
- Full fine-tuning with bf16 + gradient checkpointing
- SDPA attention (no flash_attention_2 dependency)
- Per-layer compilation with reduce-overhead

Usage:
    python train_llm_finetune.py
    python train_llm_finetune.py --override_json config.json
"""

import inspect as _inspect
import os
import tempfile

# Set up cache directories early, before importing torch/transformers
# Don't override HF_HOME - let it use default ~/.cache/huggingface or env var
os.environ.setdefault("TRITON_CACHE_DIR", "/home/ethan/.triton/cache")
os.environ.setdefault("TMPDIR", "/home/ethan/tmp")
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

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

# For ROUGE evaluation - install with: pip install rouge-score
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available, ROUGE evaluation will be disabled")

# For benchmark evaluation - install with: pip install lm-eval
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    print("Warning: lm-eval not available, benchmark evaluation will be disabled")

from snr_adam import SNRAdam
from csv_logger import CSVLogger


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
        "model": "mistralai/Mistral-7B-v0.1",  # or meta-llama/Llama-2-7b-hf
        "gradient_checkpointing": True,  # Required for 7B on single GPU
        
        # Data
        "dataset": "tatsu-lab/alpaca",
        "max_length": 512,
        "val_split": 0.02,
        "num_workers": 4,
        
        # Training
        "epochs": 3,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,  # Effective batch = 32
        "lr": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        # Multiply LR scheduler warmup steps (keeps beta/alpha warmups untouched).
        # Example: warmup_ratio=0.03, lr_warmup_mult=5.0 -> 5x more warmup steps.
        "lr_warmup_mult": 1.0,
        "max_grad_norm": 1.0,
        
        # Optimizer
        "optimizer": "adamw",  # adamw or snr_adam
        "beta1": 0.9,
        "beta2": 0.95,
        
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
        "momentum_mode": "snr_adaptive",
        "power_warmup_steps": 100,
        "slow_add_alpha": 0.5,
        
        # Compile (disabled by default - conflicts with gradient checkpointing)
        "compile": True,
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": False,
        "compile_dynamic": True,
        
        # ROUGE evaluation (generation-based)
        "eval_rouge": True,  # Enable ROUGE evaluation
        "rouge_num_samples": 200,  # Number of samples for ROUGE eval
        "rouge_max_new_tokens": 128,  # Max tokens to generate
        
        # Benchmark evaluation (lm-evaluation-harness)
        "eval_benchmarks": True,  # Enable post-training benchmark eval
        "eval_benchmarks_per_epoch": False,  # Optional: lightweight eval each epoch (skips last)
        "benchmark_tasks_full": ["mmlu", "hellaswag"],  # Full eval tasks (post-training)
        "benchmark_tasks_light": ["hellaswag"],  # Light eval tasks (per-epoch, faster)
        "benchmark_num_fewshot": 5,  # Few-shot examples for benchmarks
        "benchmark_limit": None,  # Limit samples (None = full, set to 100 for faster per-epoch)
        "benchmark_limit_light": 200,  # Limit for per-epoch lightweight eval
        
        # Misc
        "seed": 42,
        "output_dir": "./llm_finetune_output",
        "log_interval": 10,
        "wandb": False,
        "wandb_project": "snradam_finetune",
        "run_name": None,
        "hf_cache_dir": None,
    }


# =============================================================================
# Dataset
# =============================================================================

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


class AlpacaDataset(Dataset):
    """Alpaca instruction-tuning dataset."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if item.get("input", "").strip():
            text = ALPACA_PROMPT.format(
                instruction=item["instruction"],
                input=item["input"],
                output=item["output"]
            )
        else:
            text = ALPACA_PROMPT_NO_INPUT.format(
                instruction=item["instruction"],
                output=item["output"]
            )
        
        encodings = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


# =============================================================================
# Training
# =============================================================================

def compile_model_layers(model, mode="reduce-overhead", fullgraph=False, dynamic=True):
    """Compile transformer layers individually for large models."""
    # Get the transformer layers (works for Mistral/LLaMA)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        print("Warning: Could not find transformer layers, skipping per-layer compile")
        return model
    
    print(f"Compiling {len(layers)} transformer layers with mode={mode}...")
    for i, layer in enumerate(layers):
        # Compile each layer with reduce-overhead for training
        # dynamic=True since sequence lengths may vary slightly
        layers[i] = torch.compile(layer, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    
    return model


def train_one_epoch(model, train_loader, optimizer, scheduler, accelerator, epoch, cfg, csv_log=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    
    # Create CUDA events for detailed timing
    data_start = torch.cuda.Event(enable_timing=True)
    data_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
    dataloader_iter = iter(progress_bar)
    
    for batch_idx in range(len(train_loader)):
        # Time data loading
        data_start.record()
        batch = next(dataloader_iter)
        data_end.record()
        
        with accelerator.accumulate(model):
            # Time forward pass
            forward_start.record()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            forward_end.record()
            
            # Time backward pass
            backward_start.record()
            accelerator.backward(loss)
            backward_end.record()
            
            if cfg["max_grad_norm"] > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            
            # Time optimizer step
            optimizer_start.record()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_end.record()
        
        # Synchronize to get accurate timings
        torch.cuda.synchronize()
        data_time = data_start.elapsed_time(data_end)
        forward_time = forward_start.elapsed_time(forward_end)
        backward_time = backward_start.elapsed_time(backward_end)
        optimizer_time = optimizer_start.elapsed_time(optimizer_end)
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * cfg["batch_size"] * accelerator.num_processes / elapsed
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
                "samples/s": f"{samples_per_sec:.1f}"
            })
            # Log to wandb every log_interval steps
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            step_metrics = {
                "train/loss_step": loss.item(),
                "train/lr": current_lr,
                "train/samples_per_sec": samples_per_sec,
                "train/step": global_step,
                "timer/data_load_ms": data_time,
                "timer/forward_ms": forward_time,
                "timer/backward_ms": backward_time,
                "timer/optimizer_ms": optimizer_time,
            }
            if cfg["wandb"]:
                accelerator.log(step_metrics, step=global_step)
            if csv_log is not None:
                csv_log.log({"step": global_step, **step_metrics})
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader, accelerator):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in val_loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        num_tokens = (batch["labels"] != -100).sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens
    
    metrics = torch.tensor([total_loss, total_tokens], device=accelerator.device)
    gathered = accelerator.gather(metrics)
    # Handle both single GPU (shape [2]) and multi-GPU (shape [N, 2]) cases
    if gathered.dim() == 1:
        total_loss, total_tokens = gathered[0].item(), gathered[1].item()
    else:
        summed = gathered.sum(dim=0)
        total_loss, total_tokens = summed[0].item(), summed[1].item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 20))
    
    return avg_loss, perplexity


# Prompt templates for generation (without the output)
ALPACA_PROMPT_GEN = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

ALPACA_PROMPT_NO_INPUT_GEN = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


@torch.no_grad()
def evaluate_rouge(model, tokenizer, val_data, accelerator, num_samples=200, max_new_tokens=128):
    """
    Evaluate model using ROUGE scores by generating responses and comparing to references.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        val_data: Raw validation data with instruction/input/output fields
        accelerator: Accelerator for distributed training
        num_samples: Number of samples to evaluate
        max_new_tokens: Maximum new tokens to generate
    
    Returns:
        Dictionary with rouge1, rouge2, rougeL scores (F1)
    """
    model.eval()
    
    # Sample indices (deterministic for reproducibility)
    indices = list(range(min(num_samples, len(val_data))))
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # Process in batches for efficiency
    batch_size = 4
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        
        prompts = []
        references = []
        
        for idx in batch_indices:
            item = val_data[idx]
            
            # Build prompt (without the response)
            if item.get("input", "").strip():
                prompt = ALPACA_PROMPT_GEN.format(
                    instruction=item["instruction"],
                    input=item["input"],
                )
            else:
                prompt = ALPACA_PROMPT_NO_INPUT_GEN.format(
                    instruction=item["instruction"],
                )
            
            prompts.append(prompt)
            references.append(item["output"])
        
        # Tokenize prompts
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=384,  # Leave room for generation
        ).to(accelerator.device)
        
        # Generate responses
        unwrapped_model = accelerator.unwrap_model(model)
        outputs = unwrapped_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Decode and extract generated parts
        for j, (prompt, ref, output) in enumerate(zip(prompts, references, outputs)):
            # Decode full output
            full_text = tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract generated response (after the prompt)
            if "### Response:" in full_text:
                generated = full_text.split("### Response:")[-1].strip()
            else:
                # Fallback: remove prompt from beginning
                generated = full_text[len(prompt):].strip() if full_text.startswith(prompt[:50]) else full_text
            
            # Compute ROUGE scores
            scores = scorer.score(ref, generated)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # Average scores
    results = {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }
    
    return results


@torch.no_grad()
def evaluate_benchmarks(model, tokenizer, accelerator, tasks, num_fewshot=5, limit=None):
    """
    Evaluate model on standard benchmarks using lm-evaluation-harness.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        accelerator: Accelerator for distributed training
        tasks: List of task names (e.g., ["mmlu", "hellaswag"])
        num_fewshot: Number of few-shot examples
        limit: Limit number of samples per task (None = full)
    
    Returns:
        Dictionary with task accuracies
    """
    if not LM_EVAL_AVAILABLE:
        return {}
    
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Create lm-eval model wrapper
    lm = HFLM(
        pretrained=unwrapped_model,
        tokenizer=tokenizer,
        batch_size=4,
    )
    
    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
    )
    
    # Extract accuracies
    scores = {}
    for task_name, task_results in results.get("results", {}).items():
        # Handle different metric names across tasks
        if "acc,none" in task_results:
            scores[task_name] = task_results["acc,none"]
        elif "acc_norm,none" in task_results:
            scores[task_name] = task_results["acc_norm,none"]
        elif "acc" in task_results:
            scores[task_name] = task_results["acc"]
        
    return scores


def main():
    # Load config
    cfg = get_config()
    
    # Parse command line for override JSON
    if "--override_json" in sys.argv:
        idx = sys.argv.index("--override_json")
        with open(sys.argv[idx + 1], 'r') as f:
            cfg.update(json.load(f))
    
    # Generate run name
    model_short = cfg["model"].split("/")[-1]
    if cfg["run_name"] is None:
        cfg["run_name"] = f"{model_short}-{cfg['optimizer']}-lr{cfg['lr']}"
        if cfg["optimizer"] == "snr_adam":
            cfg["run_name"] = f"{model_short}-SNRAdam-{cfg['normalize_mode']}"
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
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        log_with="wandb" if cfg["wandb"] else None,
    )
    
    set_seed(cfg["seed"])
    
    # Create output directory
    output_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", 'w') as f:
            json.dump(cfg, f, indent=2)
    
    if cfg["wandb"] and accelerator.is_main_process:
        accelerator.init_trackers(cfg["wandb_project"], config=cfg, init_kwargs={"wandb": {"name": cfg["run_name"]}})
    
    accelerator.print(f"Loading model: {cfg['model']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], cache_dir=cfg["hf_cache_dir"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with SDPA (default in recent transformers, no flash_attention_2 needed)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        cache_dir=cfg["hf_cache_dir"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use SDPA, no flash_attention_2 dependency
    )
    
    # Enable gradient checkpointing (required for 7B on single GPU)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    accelerator.print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # Compile per-layer for large models
    if cfg["compile"]:
        model = compile_model_layers(model, cfg["compile_mode"], cfg["compile_fullgraph"], cfg["compile_dynamic"])
    
    # Load dataset
    accelerator.print(f"Loading dataset: {cfg['dataset']}")
    dataset = load_dataset(cfg["dataset"], cache_dir=cfg["hf_cache_dir"])
    
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
        train_data, val_data = dataset["train"], dataset["test"]
    else:
        train_data, val_data = dataset["train"], dataset["validation"]
    
    accelerator.print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_dataset = AlpacaDataset(train_data, tokenizer, cfg["max_length"])
    val_dataset = AlpacaDataset(val_data, tokenizer, cfg["max_length"])
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=collate_fn, pin_memory=True,
    )
    
    # Create optimizer with weight decay grouping
    # Strategy: exclude 1D params (biases, norm weights) + embeddings
    # - 1D params: catches all biases and all normalization weights (LayerNorm, RMSNorm, etc.)
    # - Embeddings: 2D but shouldn't have weight decay (acts as regularization on rare tokens)
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 1D params: biases, LayerNorm/RMSNorm/etc weights (gamma)
        if param.dim() == 1:
            no_decay_params.append(param)
        # Embeddings by name (they're 2D so need explicit check)
        elif "embed" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    accelerator.print(f"Params with weight decay: {len(decay_params)}, without: {len(no_decay_params)}")
    
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    if cfg["optimizer"] == "snr_adam":
        optimizer = SNRAdam(
            optimizer_grouped_parameters, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
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
            warmup_steps=cfg.get("power_warmup_steps", 100),
            slow_add_alpha=cfg.get("slow_add_alpha", 0.5),
        )
        accelerator.print(f"SNRAdam: boost=[{cfg['boost_min']}, {cfg['boost_max']}], "
                          f"norm={cfg.get('normalize_mode', 'global')}, "
                          f"snr={cfg.get('snr_mode', 'coherence')}, "
                          f"stat={cfg.get('snr_whiten_stat', 'winsor')}, "
                          f"slow_add_alpha={cfg.get('slow_add_alpha', 0.5)}")
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        accelerator.print(f"AdamW: lr={cfg['lr']}, betas=({cfg['beta1']}, {cfg['beta2']})")
    
    # Create scheduler
    num_training_steps = len(train_loader) * cfg["epochs"] // cfg["gradient_accumulation_steps"]
    num_warmup_steps = int(num_training_steps * cfg["warmup_ratio"])
    # Optional warmup scaling (e.g. 2000 * 5.0 -> 10_000 warmup steps)
    lr_warmup_mult = float(getattr(optimizer, "param_groups", [{}])[0].get("lr_warmup_mult", cfg.get("lr_warmup_mult", 1.0)))
    num_warmup_steps = int(round(num_warmup_steps * lr_warmup_mult))
    num_warmup_steps = min(num_warmup_steps, int(num_training_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    accelerator.print(f"Training steps: {num_training_steps}, Warmup: {num_warmup_steps}")
    
    # Prepare for distributed
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Local CSV logger
    output_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    csv_log = CSVLogger(output_dir / "metrics.csv") if accelerator.is_main_process else None
    
    # Training loop
    best_ppl = float('inf')
    best_rouge = 0.0
    
    for epoch in range(1, cfg["epochs"] + 1):
        accelerator.print(f"\n{'='*60}\nEpoch {epoch}/{cfg['epochs']}\n{'='*60}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, accelerator, epoch, cfg, csv_log=csv_log)
        val_loss, val_ppl = evaluate(model, val_loader, accelerator)
        
        # ROUGE evaluation (generation-based)
        rouge_scores = None
        if cfg.get("eval_rouge", True) and ROUGE_AVAILABLE and accelerator.is_main_process:
            accelerator.print("Running ROUGE evaluation...")
            rouge_scores = evaluate_rouge(
                model, tokenizer, val_data, accelerator,
                num_samples=cfg.get("rouge_num_samples", 200),
                max_new_tokens=cfg.get("rouge_max_new_tokens", 128),
            )
            accelerator.print(f"ROUGE-1: {rouge_scores['rouge1']:.4f} | ROUGE-2: {rouge_scores['rouge2']:.4f} | ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        # Per-epoch lightweight benchmark eval (optional, skips last epoch)
        benchmark_scores = None
        is_last_epoch = (epoch == cfg["epochs"])
        if cfg.get("eval_benchmarks_per_epoch", False) and not is_last_epoch and LM_EVAL_AVAILABLE and accelerator.is_main_process:
            accelerator.print("Running lightweight benchmark evaluation...")
            benchmark_scores = evaluate_benchmarks(
                model, tokenizer, accelerator,
                tasks=cfg.get("benchmark_tasks_light", ["hellaswag"]),
                num_fewshot=cfg.get("benchmark_num_fewshot", 5),
                limit=cfg.get("benchmark_limit_light", 200),
            )
            for task, acc in benchmark_scores.items():
                accelerator.print(f"  {task}: {acc:.4f}")
        
        accelerator.print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")
        
        # Log metrics
        log_dict = {"train/loss": train_loss, "val/loss": val_loss, "val/perplexity": val_ppl, "epoch": epoch}
        if rouge_scores:
            log_dict.update({
                "val/rouge1": rouge_scores['rouge1'],
                "val/rouge2": rouge_scores['rouge2'],
                "val/rougeL": rouge_scores['rougeL'],
            })
        if benchmark_scores:
            for task, acc in benchmark_scores.items():
                log_dict[f"val/{task}_acc"] = acc
        if cfg["wandb"]:
            accelerator.log(log_dict)
        if csv_log is not None:
            csv_log.log(log_dict)
        
        # Track best metrics (no model saving to conserve storage)
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            accelerator.print(f"New best PPL: {best_ppl:.2f}")
        
        if rouge_scores and rouge_scores['rougeL'] > best_rouge:
            best_rouge = rouge_scores['rougeL']
            accelerator.print(f"New best ROUGE-L: {best_rouge:.4f}")
    
    # Post-training full benchmark evaluation
    final_benchmark_scores = None
    if cfg.get("eval_benchmarks", True) and LM_EVAL_AVAILABLE and accelerator.is_main_process:
        accelerator.print(f"\n{'='*60}\nRunning full benchmark evaluation...\n{'='*60}")
        final_benchmark_scores = evaluate_benchmarks(
            model, tokenizer, accelerator,
            tasks=cfg.get("benchmark_tasks_full", ["mmlu", "hellaswag"]),
            num_fewshot=cfg.get("benchmark_num_fewshot", 5),
            limit=cfg.get("benchmark_limit"),  # None = full evaluation
        )
        accelerator.print("\nFinal Benchmark Results:")
        for task, acc in final_benchmark_scores.items():
            accelerator.print(f"  {task}: {acc:.4f}")
        
        # Log final benchmark scores
        if cfg["wandb"]:
            final_log = {f"final/{task}_acc": acc for task, acc in final_benchmark_scores.items()}
            accelerator.log(final_log)
    
    if csv_log is not None:
        csv_log.close()
    
    accelerator.print(f"\nDone! Best perplexity: {best_ppl:.2f}, Best ROUGE-L: {best_rouge:.4f}")
    if cfg["wandb"]:
        accelerator.end_training()


if __name__ == "__main__":
    main()

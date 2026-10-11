#!/usr/bin/env python3
"""Audit a completed DiT screen and preserve compact comparative evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite numeric evidence")
    if isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    if isinstance(value, list):
        for item in value:
            check_finite(item)


def common_config(configs, phase):
    """Allow declared architecture axes while retaining training/evaluation checks."""
    allowed_differences = {"variant", "width", "rank", "n_basis", "mlp_ratio", "core_rank",
                           "adapter_lr_multiplier", "adapter_gain", "output_dir", "projector_groups", "coefficient_activation",
                           "decoder_groups", "coefficient_hidden", "factor_offsets"}
    if phase in ("tuning", "confirmation", "screen", "calibration"):
        allowed_differences.update(("lr", "seed"))
    if phase == "optimizer_cfg":
        allowed_differences.update(("seed", "adam_beta1"))
    common = {k: v for k, v in configs[0].items() if k not in allowed_differences}
    if any({k: v for k, v in cfg.items() if k not in allowed_differences} != common for cfg in configs):
        raise ValueError("Training or evaluation settings differ across variants")
    return common


def summarize(screen_path):
    screen_path = Path(screen_path).resolve()
    screen = json.loads(screen_path.read_text())
    supervisor_log = (screen_path.parent / "supervisord.log").read_text()
    cached_evaluator = Path(screen["evaluator_cache"]["path"])
    if hashlib.sha256(cached_evaluator.read_bytes()).hexdigest() != screen["evaluator_cache"]["sha256"]:
        raise ValueError("The shared FID evaluator cache changed")
    rows, configs, split, runtime, source = [], [], None, None, None
    for job in screen["jobs"]:
        out = Path(job["output_dir"])
        cfg = json.loads((out / "config.json").read_text())
        result = json.loads((out / "results.json").read_text())
        metadata = json.loads((out / "dataset.json").read_text())
        provenance = json.loads((out / "provenance.json").read_text())
        artifacts = json.loads((out / "artifact_manifest.json").read_text())
        events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
        check_finite(result)
        check_finite(events)
        if cfg != json.loads(Path(job["config"]).read_text()):
            raise ValueError("Launched and recorded configs differ")
        if result["completed_steps"] != cfg["max_steps"] or events[-1]["event"] != "complete":
            raise ValueError("Incomplete run")
        if f"exited: {job['service']} (exit status 0; expected)" not in supervisor_log:
            raise ValueError("Missing successful worker exit in supervisor evidence")
        count = len(metadata["train_indices"])
        steps_per_epoch = math.ceil(count / cfg["batch_size"])
        epochs, remaining = divmod(cfg["max_steps"], steps_per_epoch)
        expected_images = epochs * count + min(remaining * cfg["batch_size"], count)
        if result["processed_images"] != expected_images:
            raise ValueError("Processed-image budget differs from recorded data/steps")
        if result["weights_saved"] or result["test_evaluated"] or artifacts["weights_saved"] or artifacts["periodic_checkpoints"]:
            raise ValueError("Unexpected weights, checkpoint or test evaluation")
        unexpected = [p.name for p in out.rglob("*") if p.suffix in (".pt", ".pth", ".safetensors", ".ckpt")]
        if unexpected:
            raise ValueError(f"Unexpected large training artifacts: {unexpected}")
        current_split = {k: metadata[k] for k in ("train_indices", "validation_indices", "source_fingerprint")}
        train_indices, val_indices = current_split["train_indices"], current_split["validation_indices"]
        if (len(set(train_indices)) != len(train_indices) or len(set(val_indices)) != len(val_indices)
                or set(train_indices) & set(val_indices) or any(i < 0 or i >= 50000 for i in train_indices + val_indices)):
            raise ValueError("Duplicated, overlapping or invalid split indices")
        if len(train_indices) != screen["protocol"]["train_examples"] or len(val_indices) != screen["protocol"]["validation_examples"]:
            raise ValueError("Split count differs from declared protocol")
        current_source = provenance["source_sha256"]
        current_runtime = {k: provenance[k] for k in ("package_versions", "cuda_wheel")}
        for previous, current in ((split, current_split), (source, current_source), (runtime, current_runtime)):
            if previous is not None and previous != current:
                raise ValueError("Dataset, source or runtime mismatch across runs")
        split, source, runtime = current_split, current_source, current_runtime
        for name, digest in current_source.items():
            if hashlib.sha256((screen_path.parent / "source" / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Source snapshot hash differs")
        validations = [e for e in events if e["event"] == "validation"]
        expected_steps = [0] + list(range(cfg["eval_every"], cfg["max_steps"] + 1, cfg["eval_every"]))
        if expected_steps[-1] != cfg["max_steps"]:
            expected_steps.append(cfg["max_steps"])
        if [e["step"] for e in validations] != expected_steps:
            raise ValueError("Missing validation states")
        if any(e["ema"]["examples"] != len(metadata["validation_indices"]) for e in validations):
            raise ValueError("Wrong validation denominator")
        generated = result["generative_evaluation"]
        if cfg["fid_images"]:
            if generated["fid_fake_images"] != cfg["fid_images"] or generated["fid_real_images"] != len(metadata["validation_indices"]):
                raise ValueError("Wrong FID budget")
            if artifacts["evaluator"]["sha256"] != screen["evaluator_cache"]["sha256"]:
                raise ValueError("FID evaluator weights differ")
        train_events = [e for e in events if e["event"] == "train"]
        row = {"name": job["name"], "config": cfg, "output_dir": str(out), **result,
               "mean_train_tokens_per_second_excluding_warmup": result["mean_train_images_per_second_excluding_warmup"] * result["tokens_per_image"],
               "forward_matmul_flops_per_token_estimate": result["forward_matmul_flops_per_image_estimate"] / result["tokens_per_image"],
               "max_logged_grad_norm": max(e["grad_norm_before_clip"] for e in train_events),
               "max_logged_adapter_grad_norm": max((e["adapter_grad_norm_before_clip"] or 0) for e in train_events),
               "validation_curve": validations,
               "evidence_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                   for name in ("config.json", "dataset.json", "metrics.jsonl", "results.json", "provenance.json", "artifact_manifest.json") + (("samples_ema.png",) if cfg["sample_images"] else ())}}
        rows.append(row)
        configs.append(cfg)
    if len({r["initial_validation"]["noise_mse"] for r in rows}) != 1:
        raise ValueError("Initial validation noise differs across zero-output models")
    common = common_config(configs, screen["phase"])
    summary = {"screen": str(screen_path), "phase": screen["phase"], "protocol": screen["protocol"],
               "common_config": common, "runtime": runtime, "source_sha256": source,
               "evaluator_cache": screen["evaluator_cache"], "rows": rows,
               "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "audit_input_sha256": {name: hashlib.sha256((screen_path.parent / name).read_bytes()).hexdigest()
                                      for name in ("screen.json", "supervisord.log", "hardware_claims.json", "split_class_balance.json")
                                      if (screen_path.parent / name).is_file()},
               "audit": {"all_runs_complete": True, "successful_worker_exits": True, "finite_logged_metrics": True, "same_split": True,
                         "split_disjoint_and_unique": True,
                         "evaluator_cache_unchanged": True, "same_initial_validation_noise": True,
                         "same_source_runtime": True, "same_training_evaluation_budget": True,
                         "weights_saved": False, "test_evaluated": False,
                         "bytes_retained_under_screen": sum(p.stat().st_size for p in screen_path.parent.rglob("*") if p.is_file())},
               "limitations": ["One fixed validation split; seed spread is not a confidence interval or test-set estimate.",
                               "FID uses held-out train images and finite sample counts; not published 50k-image FID.",
                               "Wrong-class loss measures sensitivity/use, not generated class accuracy.",
                               "External condition is a single class/time vector; sequence-conditioned cross-attention is not evaluated.",
                               "Concurrent throughput shares CPU/storage; matrix FLOPs omit non-matmul work and backward.",
                               "Generative evaluation time includes sampling and FID feature/statistic computation; isolated inference throughput is not measured.",
                               "Each screen has its own declared LR/initialization recipe; optimal rates are not established without bracketing."]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.screen)
    output = args.output or args.screen.parent / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("model | params M | raw MSE | EMA MSE | FID | train img/s | peak GiB")
    for row in summary["rows"]:
        fid = row["generative_evaluation"]["fid"]
        peak = (row["peak_allocated_bytes"] or 0) / 1024**3
        print(f"{row['name']} | {row['parameters']/1e6:.3f} | {row['raw_validation']['noise_mse']:.5f} | "
              f"{row['ema_validation']['noise_mse']:.5f} | {fid if fid is not None else 'none'} | "
              f"{row['mean_train_images_per_second_excluding_warmup']:.0f} | {peak:.3f}")
    print(json.dumps({"summary": str(output), "audit": summary["audit"]}))


if __name__ == "__main__":
    main()

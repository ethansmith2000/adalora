#!/usr/bin/env python3
"""Bounded gain/LR calibration followed by common-stream 500-step diagnostics."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from benchmark_dit_design import digest_tensor, run_case
from train_dit_cifar10 import Config, build_datasets, make_noise_scheduler, provenance, write_json


FAMILIES = [("basis_r8", "dplr", 8), ("basis_r2", "dplr", 2),
            ("basis_r4", "dplr", 4), ("bottleneck_r2", "conditional_bottleneck", 2),
            ("bottleneck_r4", "conditional_bottleneck", 4)]
RATE_BOUNDS = (1 / 16, 16.0)
TOLERANCE = .25
MAX_CORRECTIONS = 2


def update_metric(row, observation_steps):
    """Arithmetic mean of fixed-input delta-update RMS across steps and blocks."""
    events = {e["step"]: e for e in row["diagnostics"]}
    values = [v["update_rms"] for step in observation_steps
              for v in events[step]["fixed_token_condition_function_updates"]]
    value = statistics.mean(values)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Calibration requires a finite, nonzero functional update")
    return value


def proposed_rate(rate, observed, target):
    return min(RATE_BOUNDS[1], max(RATE_BOUNDS[0], rate * target / observed))


def choose_candidate(candidates, target):
    return min(candidates, key=lambda c: abs(math.log(c["metric"] / target)))


def calibration_plan():
    return {"gain_rule": "1/sqrt(rank), fixed nontrainable multiplier on the delta",
            "target_family": "basis_r8", "target_adapter_lr_multiplier": 1.0,
            "metric": "arithmetic mean fixed-token/condition delta-update RMS across all blocks and observation steps; highest FP32 matmul precision",
            "selection": "closest absolute log metric/target; never uses loss or validation",
            "proposal": "clip(current_multiplier * target / observed, 1/16, 16)",
            "maximum_corrections_per_family": MAX_CORRECTIONS, "rate_bounds": list(RATE_BOUNDS),
            "relative_tolerance": TOLERANCE, "confirmation": "all five selected families plus raw-gain basis_r8 anchor",
            "limits": "scalar LR matching is approximate, optimizer/clip/path dependent, and not a muP derivation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    output = Path(settings["output_dir"])
    if (output / "report.json").exists():
        raise FileExistsError("Completed gain evidence already exists")
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
        training.transform = validation.transform
    bank = [training[i] for i in range(len(training))]
    images_bank = torch.stack([image for image, _ in bank])
    labels_bank = torch.tensor([label for _, label in bank])
    del bank
    metadata.update(bank_images_sha256=digest_tensor(images_bank), bank_labels_sha256=digest_tensor(labels_bank),
                    protocol="fixed train-only bank, replacement sampling and shared-stream flips")
    write_json(output / "dataset.json", metadata)
    scheduler = make_noise_scheduler(cfg)
    sample = [validation[i] for i in range(min(8, len(validation)))]
    images = torch.stack([x for x, _ in sample]).to(device)
    labels = torch.tensor([label for _, label in sample], device=device)
    rng = torch.Generator(device=device).manual_seed(2027)
    ts = torch.randint(1000, (len(sample),), generator=rng, device=device)
    noisy = scheduler.add_noise(images, torch.randn(images.shape, generator=rng, device=device), ts)
    del images, sample
    runtime = provenance()
    for name in ("benchmark_dit_design.py", Path(__file__).name):
        runtime["source_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    runtime.update(device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   device_uuid=str(torch.cuda.get_device_properties(device).uuid) if device.type == "cuda" else None)
    write_json(output / "runtime.json", runtime)
    pilot_dir, confirmation_dir = output / "calibration", output / "confirmation"
    pilot_dir.mkdir(); confirmation_dir.mkdir()
    pilots, selected, confirmations = [], {}, []
    target = None

    def run(name, variant, rank, gain, rate, steps, directory):
        case_settings = {**settings, "steps": steps, "schedule_steps": settings["steps"],
                         "timing_warmup": min(10, steps // 5),
                         "adapter_gain": gain, "adapter_lr_multiplier": rate,
                         "projection_probes": False}
        return run_case((name, variant, rank, "normalized"), case_settings, images_bank, labels_bank,
                        validation, scheduler, (noisy, ts, labels), directory, runtime)

    for family, variant, rank in FAMILIES:
        candidates = []
        rate, gain = 1.0, 1 / math.sqrt(rank)
        for attempt in range(MAX_CORRECTIONS + 1):
            name = f"{family}_candidate{attempt}"
            row = run(name, variant, rank, gain, rate, settings["calibration_steps"], pilot_dir)
            metric = update_metric(row, settings["calibration_observation_steps"])
            if target is None:
                target = metric
            candidate = {"name": name, "family": family, "attempt": attempt,
                         "gain": gain, "adapter_lr_multiplier": rate, "metric": metric,
                         "metric_to_target": metric / target, "within_tolerance": abs(metric / target - 1) <= TOLERANCE}
            candidates.append(candidate)
            pilots.append({"candidate": candidate, "row": row})
            print(json.dumps({"event": "calibration", **candidate}), flush=True)
            if candidate["within_tolerance"] or attempt == MAX_CORRECTIONS:
                break
            rate = proposed_rate(rate, metric, target)
            if rate == candidate["adapter_lr_multiplier"]:
                break
        selected[family] = choose_candidate(candidates, target)
        write_json(output / "calibration_progress.json", {"target": target, "selected": selected,
                                                        "candidates": [p["candidate"] for p in pilots]})
    write_json(output / "calibration_selection.json", {"target": target, "selected": selected,
                                                       "plan": calibration_plan()})
    # The selected rates are frozen before this longer confirmation begins.
    families = {name: (variant, rank) for name, variant, rank in FAMILIES}
    for family in settings["confirmation_order"]:
        raw = family == "basis_r8_raw"
        variant, rank = families["basis_r8" if raw else family]
        choice = {"gain": 1.0, "adapter_lr_multiplier": 1.0} if raw else selected[family]
        row = run(family, variant, rank, choice["gain"], choice["adapter_lr_multiplier"], settings["steps"], confirmation_dir)
        row["early_update_metric"] = update_metric(row, settings["calibration_observation_steps"])
        row["early_update_metric_to_target"] = row["early_update_metric"] / target
        row["calibration_candidate"] = None if raw else choice["name"]
        write_json(confirmation_dir / family / "results.json", row)
        confirmations.append(row)
        write_json(output / "confirmation_progress.json", {"completed": [r["name"] for r in confirmations]})
    report = {"settings": settings, "runtime": runtime, "dataset": metadata, "plan": calibration_plan(),
              "target_update_rms": target, "calibration": pilots, "selected": selected, "confirmation": confirmations,
              "quality_ranking_supported": False,
              "limits": ["one seed and short train-only bank engineering runs; no quality selection",
                         "calibration and confirmation deliberately share their early stream; later steps test transfer in time",
                         "functional probe has independent normalized fixed tokens/conditions, not the live learned condition distribution",
                         "uniform adapter LR multiplier; input/output/generator ratios are not separately tuned",
                         "fixed gain is not variance-preserving or a derivation of muP",
                         "sequence conditioning and cross-attention remain untested"]}
    write_json(output / "report.json", report)
    protocol = json.loads((output / "protocol.json").read_text())
    protocol.update(state="complete", completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "suite_complete", "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Managed, bounded LR tuning followed by frozen three-seed confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from summarize_cifar10_screen import summarize
from train_cifar10 import Config


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def family(cfg):
    return f"plain_w{cfg['width']}" if cfg["variant"] == "plain" and cfg["width"] != 192 else cfg["variant"]


def config_name(cfg):
    return f"{family(cfg)}-lr{cfg['lr']:g}-h{cfg['hyper_lr_multiplier']:g}-s{cfg['seed']}"


def prepare(output, launch):
    repo = Path(__file__).resolve().parent
    previous = json.loads((repo / "reports/cifar10-screen-20epochs.json").read_text())
    if output.exists():
        raise FileExistsError(output)
    disk_free = shutil.disk_usage(repo).free
    if disk_free < 1024**3:
        raise RuntimeError("Need at least 1 GiB free for this bounded evidence-only workflow")
    output.mkdir(parents=True)
    runner = output / "runner"
    runner.mkdir()
    for name in ("train_cifar10.py", "vit.py", "conditioning.py", "condlr.py", "requirements.txt",
                 "launch_cifar10_screen.py", "summarize_cifar10_screen.py", "cifar10_followup.py"):
        shutil.copy2(repo / name, runner / name)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in runner.iterdir()}
    centers = {}
    for row in previous["summary"]["rows"]:
        key = family(row["config"])
        if key in ("plain", "diagonal", "static_lora_diagonal", "dplr", "exact", "plain_w216"):
            centers[key] = row
    assert len(centers) == 6
    references = []
    for key, row in centers.items():
        provenance = json.loads((Path(row["output_dir"]) / "provenance.json").read_text())
        if any(hashes[k] != v for k, v in provenance["source_sha256"].items()):
            raise ValueError("Completed center runs cannot be reused after a training-source change")
        references.append({"name": config_name(row["config"]), "service": f"reference_{key}",
                           "config": str(Path(row["output_dir"]) / "config.json"),
                           "output_dir": row["output_dir"], "reused": True})
    paths = []
    configs = output / "tuning_configs"
    configs.mkdir()

    def add(cfg):
        cfg = dict(cfg)
        cfg.update(epochs=20, batch_size=128, seed=42, max_steps=None, train_limit=None, val_limit=None,
                   synthetic=False, download=False, evaluate_test=False, save_weights=False,
                   compile=False, output_dir=None)
        path = configs / f"{config_name(cfg)}.json"
        write_json(path, cfg)
        paths.append(str(path))

    for key in ("plain", "diagonal", "static_lora_diagonal", "dplr", "plain_w216"):
        for lr in (2e-4, 1e-3):
            add({**centers[key]["config"], "lr": lr})
    for multiplier in (0.3, 3.0):
        add({**centers["dplr"]["config"], "hyper_lr_multiplier": multiplier})
    for multiplier in (0.1, 0.3):
        add({**centers["exact"]["config"], "hyper_lr_multiplier": multiplier})
    for lr in (2e-4, 5e-4, 1e-3):
        add({**centers["plain"]["config"], "width": 240, "heads": 3, "lr": lr})
    write_json(output / "references.json", references)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "repo_root": str(repo),
        "purpose": "LR tuning and independent-seed confirmation of token-conditioned CIFAR-10 ViT",
        "source_sha256": hashes, "disk_free_bytes_before_launch": disk_free,
        "expected_total_storage_bytes_upper_bound": 400 * 1024**2,
        "artifact_retention": "retain source/configs, seeds, split indices, logs, metrics, selection, audit and summaries until explicitly removed",
        "weights_saved": False, "periodic_checkpoints": False, "test_evaluated": False,
        "base_lr_grid": [2e-4, 5e-4, 1e-3], "dplr_hyper_lr_multipliers_at_base_5e4": [0.3, 1.0, 3.0],
        "exact_hyper_lr_multipliers_at_base_5e4": [0.1, 0.3, 1.0],
        "tuning_epochs": 20, "tuning_seed": 42, "tuning_new_runs": len(paths), "tuning_reused_runs": len(references),
        "tuning_configs": paths, "selection_metric": "final validation accuracy at epoch 20; tie break by lower matmul FLOPs then lower LR",
        "confirmation_epochs": 50, "confirmation_seeds": [43, 44, 45],
        "confirmation_families": ["plain", "diagonal", "static_lora_diagonal", "dplr", "plain_w216", "plain_w240"],
        "exact_confirmation_rule": "include only if best tuning endpoint is within 1 percentage point of the best tuned plain endpoint",
        "confirmation_new_runs_min": 18, "confirmation_new_runs_max": 21,
        "selection_frozen_before_confirmation": True, "split_seed": 123, "batch_size": 128,
        "controls": "plain width 216 approximates diagonal/static-LoRA parameter budget; width 240 exceeds DPLR's budget slightly; report actual parameters and FLOPs",
        "selection_limit": "one fixed held-out validation split; confirmation seeds are independent of tuning seed; no official test or significance claim",
        "state": "prepared"}
    write_json(output / "protocol.json", protocol)
    if launch:
        service = "adalora_followup_" + output.name.replace("-", "_")
        wrapper = output / "controller.sh"
        wrapper.write_text("\n".join([
            "#!/bin/bash", "set -e", ". /opt/supervisor-scripts/utils/environment.sh",
            ". /opt/supervisor-scripts/utils/logging.sh", "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false",
            "exec " + shlex.join([sys.executable, "-u", str(runner / "cifar10_followup.py"), "run", "--output", str(output)])
            + f" > >(tee -a {shlex.quote(str(output / 'controller.log'))}) 2>&1", ""]))
        wrapper.chmod(0o755)
        control = output / "controller.conf"
        control.write_text(f"""[unix_http_server]
file={output}/controller.sock
chmod=0600
[supervisord]
logfile={output}/controller-supervisord.log
logfile_maxbytes=1048576
logfile_backups=1
pidfile={output}/controller.pid
childlogdir={output}
nodaemon=false
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{output}/controller.sock
[program:{service}]
environment=PROC_NAME="%(program_name)s"
command={wrapper}
autostart=true
autorestart=false
exitcodes=0
startsecs=3
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
""")
        protocol["controller_service"] = service
        protocol["controller_control"] = str(control)
        write_json(output / "protocol.json", protocol)
        subprocess.run(["supervisord", "-c", str(control)], check=True)
    print(json.dumps({"protocol": str(output / "protocol.json"), "new_tuning_runs": len(paths), "launched": launch}), flush=True)


def launch_stage(output, name, configs, epochs, references=None):
    command = [sys.executable, "-u", str(output / "runner/launch_cifar10_screen.py"), "--launch", "--epochs", str(epochs),
               "--batch-size", "128", "--output", str(output / name), "--purpose", f"CIFAR-10 followup {name}",
               "--configs", *configs]
    if references:
        command += ["--references", str(references)]
    subprocess.run(command, check=True)


def wait_stage(output, name):
    screen = output / name / "screen.json"
    claims = {}
    while True:
        summary = summarize(screen)
        for path in Path("/tmp/vast_gpu_claims").glob("*.claim"):
            try:
                metadata = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if any(r["service"] == metadata.get("job") for r in summary["rows"]):
                claims[metadata["job"]] = metadata
        write_json(screen.parent / "hardware_claims.json", claims)
        print(json.dumps({"stage": name, "completed": sum(r["state"] == "complete" for r in summary["rows"]),
                          "total": len(summary["rows"]), "running": sum(r["state"] == "running" for r in summary["rows"]),
                          "failed": [r["name"] for r in summary["rows"] if r["state"] in ("fatal", "failed_or_incomplete")]}), flush=True)
        if any(r["state"] in ("fatal", "failed_or_incomplete") for r in summary["rows"]):
            raise RuntimeError(f"A {name} run failed; inspect its preserved log")
        if summary["all_complete"]:
            return summary
        time.sleep(20)


def audit_stage(summary, epochs):
    if not summary["all_complete"] or not summary["same_data_split"] or not summary["same_source_snapshot"]:
        raise ValueError("Stage completeness, split or source audit failed")
    for row in summary["rows"]:
        result, cfg = row["results"], row["config"]
        if result["completed_steps"] != epochs * 352 or result["processed_images"] != epochs * 45000:
            raise ValueError("Training budget mismatch")
        if len(row["validation_curve"]) != epochs or result["test"] is not None or cfg["save_weights"]:
            raise ValueError("Evaluation or artifact policy mismatch")
        if not math.isfinite(result["validation"]["loss"]):
            raise ValueError("Nonfinite validation loss")
        directory = Path(row["output_dir"])
        if list(directory.glob("*.pt")) or list(directory.glob("*.safetensors")):
            raise ValueError("Unexpected weights saved")


def shutdown_stage(output, name):
    control = output / name / "supervisor.conf"
    # results.json is written just before worker cleanup; let each child exit.
    for _ in range(12):
        status = subprocess.run(["supervisorctl", "-c", str(control), "status"], capture_output=True, text=True)
        if not any("RUNNING" in line or "STARTING" in line for line in status.stdout.splitlines()):
            break
        time.sleep(1)
    subprocess.run(["supervisorctl", "-c", str(control), "shutdown"], check=True)


def run(output):
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["state"] = "tuning"
    write_json(output / "protocol.json", protocol)
    launch_stage(output, "tune", protocol["tuning_configs"], 20, output / "references.json")
    tuning = wait_stage(output, "tune")
    audit_stage(tuning, 20)
    shutdown_stage(output, "tune")
    selections = {}
    for row in tuning["rows"]:
        key = family(row["config"])
        rank = (row["results"]["validation"]["accuracy"], -row["results"]["forward_matmul_flops_per_image_estimate"], -row["config"]["lr"])
        if key not in selections or rank > selections[key][0]:
            selections[key] = (rank, row)
    selected = {key: row for key, (_, row) in selections.items()}
    include_exact = selected["exact"]["results"]["validation"]["accuracy"] >= selected["plain"]["results"]["validation"]["accuracy"] - 0.01
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "selection_metric": protocol["selection_metric"],
              "selected": selected, "include_exact_in_confirmation": include_exact,
              "confirmation_seeds": protocol["confirmation_seeds"]}
    write_json(output / "selection.json", frozen)
    print(json.dumps({"event": "selection_frozen", "settings": {k: {"lr": r["config"]["lr"],
                     "hyper_lr_multiplier": r["config"]["hyper_lr_multiplier"], "accuracy": r["results"]["validation"]["accuracy"]}
                     for k, r in selected.items()}, "include_exact": include_exact}), flush=True)
    configuration_dir = output / "confirmation_configs"
    configuration_dir.mkdir()
    families = protocol["confirmation_families"] + (["exact"] if include_exact else [])
    paths = []
    for key in families:
        for seed in protocol["confirmation_seeds"]:
            cfg = {**selected[key]["config"], "seed": seed, "epochs": 50, "output_dir": None}
            path = configuration_dir / f"{key}-s{seed}.json"
            write_json(path, cfg)
            paths.append(str(path))
    protocol["state"] = "confirmation"
    protocol["confirmation_configs"] = paths
    protocol["confirmation_families_selected"] = families
    write_json(output / "protocol.json", protocol)
    launch_stage(output, "confirm", paths, 50)
    confirmation = wait_stage(output, "confirm")
    audit_stage(confirmation, 50)
    shutdown_stage(output, "confirm")
    grouped = {}
    for row in confirmation["rows"]:
        grouped.setdefault(family(row["config"]), []).append(row)
    stats = {}
    plain_by_seed = {r["config"]["seed"]: r["results"]["validation"]["accuracy"] for r in grouped["plain"]}
    for key, rows in grouped.items():
        accuracies = [r["results"]["validation"]["accuracy"] for r in rows]
        deltas = [(r["results"]["validation"]["accuracy"] - plain_by_seed[r["config"]["seed"]]) * 100 for r in rows]
        stats[key] = {"mean_final_accuracy": statistics.mean(accuracies), "sample_sd_final_accuracy": statistics.stdev(accuracies),
                      "seed_accuracies": {r["config"]["seed"]: r["results"]["validation"]["accuracy"] for r in rows},
                      "paired_gain_vs_plain_percentage_points": deltas,
                      "mean_gain_vs_plain_percentage_points": statistics.mean(deltas),
                      "parameters": rows[0]["results"]["parameters"],
                      "forward_matmul_flops_per_image_estimate": rows[0]["results"]["forward_matmul_flops_per_image_estimate"],
                      "mean_loop_seconds": statistics.mean(r["results"]["elapsed_seconds"] for r in rows)}
    report = {"protocol": protocol, "selection": frozen, "tuning": tuning, "confirmation": confirmation,
              "confirmation_statistics": stats,
              "uncertainty": "sample SD over three initialization/data-order seeds; not a confidence interval or test-set estimate",
              "total_new_loop_gpu_hours": sum(r["results"]["elapsed_seconds"] / 3600 for stage in (tuning, confirmation)
                                          for r in stage["rows"] if not r["service"].startswith("reference_"))}
    write_json(output / "report.json", report)
    destination = Path(protocol["repo_root"]) / "reports" / f"cifar10-lr-confirmation-{output.name}.json"
    write_json(destination, report)
    protocol["state"] = "complete"
    protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["report"] = str(destination)
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "complete", "report": str(destination), "statistics": stats}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or Path(__file__).resolve().parent / "runs" / f"followup-{stamp}").resolve()
    if args.action == "prepare":
        prepare(output, args.launch)
    else:
        try:
            run(output)
        except Exception as exc:
            protocol = json.loads((output / "protocol.json").read_text())
            protocol["state"] = "failed"
            protocol["error"] = repr(exc)
            write_json(output / "protocol.json", protocol)
            raise


if __name__ == "__main__":
    main()

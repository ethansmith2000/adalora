#!/usr/bin/env python3
"""Managed LR tuning, frozen evaluator, and three-seed diffusion confirmation."""

import argparse
import hashlib
import json
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from summarize_dit_cifar10_screen import summarize
from train_dit_cifar10 import Config, SOURCE_FILES, load_config, provenance, write_json
from train_cifar10 import Config as ClassifierConfig, load_config as load_classifier_config


FAMILIES = {"adaln_d192": ("adaln_zero", 192), "adaln_d216": ("adaln_zero", 216),
            "adaln_d240": ("adaln_zero", 240), "dplr": ("dplr", 192)}


def family(cfg):
    return "dplr" if cfg["variant"] == "dplr" else f"adaln_d{cfg['width']}"


def private_supervisor(directory, programs, label="supervisor", autostart=False, start_programs=None):
    """Install only this workflow's finite services, without touching global config."""
    services = directory / f"{label}-services"
    services.mkdir()
    for name, command in programs:
        wrapper = services / f"{name}.sh"
        wrapper.write_text("\n".join([
            "#!/bin/bash", "set -e", ". /opt/supervisor-scripts/utils/environment.sh",
            ". /opt/supervisor-scripts/utils/logging.sh",
            "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
            "exec " + shlex.join(command) + f" > >(tee -a {shlex.quote(str(directory / (name + '.log')))}) 2>&1", ""]))
        wrapper.chmod(0o755)
        (services / f"{name}.conf").write_text(f"""[program:{name}]
environment=PROC_NAME="%(program_name)s"
command={wrapper}
autostart={'true' if autostart else 'false'}
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
    control = directory / f"{label}.conf"
    control.write_text(f"""[unix_http_server]
file={directory}/{label}.sock
chmod=0600
[supervisord]
logfile={directory}/{label if label != 'supervisor' else 'supervisord'}.log
logfile_maxbytes=1048576
logfile_backups=1
pidfile={directory}/{label}.pid
childlogdir={directory}
nodaemon=false
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{directory}/{label}.sock
[include]
files={services}/*.conf
""")
    subprocess.run(["supervisord", "-c", str(control)], check=True)
    if not autostart:
        names = [name for name,_ in programs] if start_programs is None else start_programs
        if names:
            subprocess.run(["supervisorctl", "-c", str(control), "start", *names], check=True)
    return str(control)


def prepare(output, launch):
    root = Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free < 1024**3:
        raise RuntimeError("Require 1 GiB free for declared compact evidence and one evaluator")
    output.mkdir(parents=True)
    runner = output / "runner"
    runner.mkdir()
    for name in (*SOURCE_FILES, "requirements.txt", "dit_followup.py", "summarize_dit_cifar10_screen.py"):
        shutil.copy2(root / name, runner / name)
    configs = output / "tuning_configs"
    configs.mkdir()
    paths = []
    for key, (variant, width) in FAMILIES.items():
        settings = [(lr, 1.) for lr in (1e-4, 2e-4, 4e-4)]
        if key == "dplr":
            settings += [(2e-4, multiplier) for multiplier in (.3, 3.)]
        for lr, multiplier in settings:
            cfg = Config(variant=variant, width=width, lr=lr, adapter_lr_multiplier=multiplier, fid_images=5000)
            path = configs / f"{key}-lr{lr:g}-h{multiplier:g}.json"
            write_json(path, asdict(cfg))
            load_config(["--override_json", str(path)])
            paths.append(str(path))
    classifier = ClassifierConfig(width=216, epochs=50, seed=46, save_weights=True,
                                 artifact_purpose="One frozen auxiliary class-agreement evaluator for diffusion comparisons; retain model and audit inputs until explicitly removed")
    write_json(output / "classifier_config.json", asdict(classifier))
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "repo_root": str(root), "state": "prepared",
                "purpose": "diffusion-specific LR tuning followed by independent-seed broadcast-conditioning confirmation",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in runner.iterdir()},
                "disk_free_bytes_before_launch": free, "expected_storage_upper_bound_bytes": 640 * 1024**2,
                "artifact_retention": "compact source/configs/splits/metrics/logs/reports/grids and one fixed evaluator until explicitly removed",
                "diffusion_weights_saved": False, "periodic_checkpoints": False, "test_evaluated": False,
                "classifier_artifact": {"purpose": classifier.artifact_purpose, "expected_weight_bytes": 4 * 3399634,
                                        "count": 1, "epochs": 50, "seed": 46, "validation_accuracy_is_a_proxy_limit": True},
                "classifier_real_validation_accuracy_floor": .75,
                "estimated_new_gpu_hours": "approximately 4–6 based on the pilot; wait for shared free GPUs",
                "families": list(FAMILIES), "base_lr_grid": [1e-4, 2e-4, 4e-4],
                "dplr_adapter_multipliers_at_base_2e4": [.3, 1., 3.], "tuning_configs": paths,
                "tuning_steps": 10000, "tuning_seed": 42, "new_tuning_runs": 14, "reused_runs": 0,
                "selection_metric": "lowest final EMA 5000-image FID; ties by lower EMA noise MSE, base LR, adapter multiplier",
                "confirmation_steps": 30000, "confirmation_eval_every": 5000, "confirmation_seeds": [43, 44, 45],
                "confirmation_runs": 12, "selection_frozen_before_confirmation": True,
                "split_seed": 123, "train_examples": 45000, "validation_examples": 5000,
                "fid_images": 5000, "sampling_seed": 2028, "sampling_steps": 50, "classifier_free_guidance": False,
                "selection_limits": "One fixed held-out split; unequal candidate counts (DPLR 5, AdaLN 3); transferring 10k selection to 30k; SD is not CI",
                "classifier_limits": "Fixed auxiliary classifier agreement, not ground-truth generated class accuracy; classifier errors and domain shift apply"}
    write_json(output / "protocol.json", protocol)
    if launch:
        protocol["controller_control"] = str(output / "controller.conf")
        write_json(output / "protocol.json", protocol)
        private_supervisor(output, [("adalora_dit_controller", [sys.executable, "-u", str(runner / "dit_followup.py"), "run", "--output", str(output)])], "controller", True)
    print(json.dumps({"protocol": str(output / "protocol.json"), "launched": launch}), flush=True)


def launch_stage(output, name, configs, classifier=False):
    stage = output / name
    stage.mkdir()
    source = stage / "source"
    shutil.copytree(output / "runner", source, ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "configs").mkdir()
    jobs, commands = [], []
    for path in configs:
        cfg = (load_classifier_config if classifier else load_config)(["--override_json", str(path)])
        cfg.output_dir = str(stage / Path(path).stem)
        resolved = stage / "configs" / Path(path).name
        write_json(resolved, asdict(cfg))
        service = "adalora_" + output.name.replace("-", "_") + "_" + name + "_" + Path(path).stem.replace("-", "_").replace(".", "_")
        script = "train_cifar10.py" if classifier else "train_dit_cifar10.py"
        command = ["/workspace/bin/gpu-claim", "run", "--owner", "adalora", "--job", service, "--wait", "--",
                   sys.executable, "-u", str(source / script), "--override_json", str(resolved)]
        commands.append((service, command))
        jobs.append({"name": Path(path).stem, "service": service, "config": str(resolved), "output_dir": cfg.output_dir})
    cache = Path(Config().fid_weights)
    manifest = {"phase": name, "jobs": jobs, "supervisor_control": str(stage / "supervisor.conf"),
                "protocol": {"train_examples": 45000, "validation_examples": 5000},
                "evaluator_cache": {"path": str(cache), "bytes": cache.stat().st_size,
                                    "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()},
                "provenance": provenance(), "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}}
    write_json(stage / "screen.json", manifest)
    private_supervisor(stage, commands)
    return stage / "screen.json"


def wait_stage(screen_path):
    screen_path = Path(screen_path)
    screen = json.loads(screen_path.read_text())
    claims = {}
    while True:
        for path in Path("/tmp/vast_gpu_claims").glob("*.claim"):
            try:
                claim = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if claim.get("job") in {j["service"] for j in screen["jobs"]}:
                claims[claim["job"]] = claim
        write_json(screen_path.parent / "hardware_claims.json", claims)
        status = subprocess.run(["supervisorctl", "-c", screen["supervisor_control"], "status"], capture_output=True, text=True).stdout
        log = (screen_path.parent / "supervisord.log").read_text()
        if any(term in status for term in ("FATAL", "BACKOFF")):
            raise RuntimeError(status)
        failed = [j["name"] for j in screen["jobs"] if "EXITED" in next((s for s in status.splitlines() if s.startswith(j["service"] + " ")), "")
                  and f"exited: {j['service']} (exit status 0; expected)" not in log]
        if failed:
            raise RuntimeError(f"Failed workers: {failed}")
        for job in screen["jobs"]:
            if f"exited: {job['service']} (exit status 0; expected)" in log and not (Path(job["output_dir"]) / "results.json").exists():
                raise RuntimeError(f"Worker exited without results: {job['name']}")
        completed = sum((Path(j["output_dir"]) / "results.json").exists() for j in screen["jobs"])
        print(json.dumps({"stage": screen["phase"], "completed": completed, "total": len(screen["jobs"])}), flush=True)
        if completed == len(screen["jobs"]) and all(f"exited: {j['service']} (exit status 0; expected)" in log for j in screen["jobs"]):
            subprocess.run(["supervisorctl", "-c", screen["supervisor_control"], "shutdown"], check=True)
            return
        time.sleep(20)


def select(rows):
    selected = {}
    for row in rows:
        key = family(row["config"])
        rank = (row["generative_evaluation"]["fid"], row["ema_validation"]["noise_mse"],
                row["config"]["lr"], row["config"]["adapter_lr_multiplier"])
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, row)
    return {key: row for key, (_, row) in selected.items()}


def statistics_from_rows(rows):
    grouped = {key: sorted([r for r in rows if family(r["config"]) == key], key=lambda r: r["config"]["seed"]) for key in FAMILIES}
    if any([r["config"]["seed"] for r in values] != [43, 44, 45] for values in grouped.values()):
        raise ValueError("Expected three fresh paired seeds per family")
    result = {}
    for key, values in grouped.items():
        metrics = {"fid": [r["generative_evaluation"]["fid"] for r in values],
                   "ema_noise_mse": [r["ema_validation"]["noise_mse"] for r in values],
                   "class_agreement": [r["generative_evaluation"]["class_agreement"]["generated"]["agreement"] for r in values]}
        result[key] = {name: {"mean": statistics.mean(nums), "sample_sd": statistics.stdev(nums),
                             "seed_values": dict(zip([43, 44, 45], nums))} for name, nums in metrics.items()}
        result[key].update(parameters=values[0]["parameters"], forward_matmul_flops_per_image_estimate=values[0]["forward_matmul_flops_per_image_estimate"],
                           mean_train_images_per_second=statistics.mean(r["mean_train_images_per_second_excluding_warmup"] for r in values))
        result[key]["paired_fid_change_vs_adaln_d192"] = [a - b for a, b in zip(metrics["fid"], [r["generative_evaluation"]["fid"] for r in grouped["adaln_d192"]])]
    result["dplr"]["paired_fid_change_vs_width_controls"] = {
        key: [a["generative_evaluation"]["fid"] - b["generative_evaluation"]["fid"] for a, b in zip(grouped["dplr"], grouped[key])]
        for key in ("adaln_d216", "adaln_d240")}
    return result


def run(output):
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["state"] = "tuning_and_classifier"
    write_json(output / "protocol.json", protocol)
    classifier_screen = launch_stage(output, "classifier", [str(output / "classifier_config.json")], True)
    tuning_screen = launch_stage(output, "tuning", protocol["tuning_configs"])
    wait_stage(tuning_screen)
    tuning = summarize(tuning_screen)
    write_json(output / "tuning_report.json", tuning)
    selected = select(tuning["rows"])
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "metric": protocol["selection_metric"], "selected": selected}
    write_json(output / "selection.json", frozen)
    print(json.dumps({"event": "selection_frozen", "settings": {k: {"lr": r["config"]["lr"], "adapter_multiplier": r["config"]["adapter_lr_multiplier"], "fid": r["generative_evaluation"]["fid"]} for k, r in selected.items()}}), flush=True)
    wait_stage(classifier_screen)
    classifier_job = json.loads(classifier_screen.read_text())["jobs"][0]
    classifier_run = Path(classifier_job["output_dir"])
    class_results = json.loads((classifier_run / "results.json").read_text())
    if class_results["validation"]["accuracy"] < protocol["classifier_real_validation_accuracy_floor"] or class_results["test"] is not None:
        raise ValueError("Classifier does not meet the predeclared 75% real-validation floor or touched the test set")
    configuration_dir = output / "confirmation_configs"
    configuration_dir.mkdir()
    paths = []
    for key in FAMILIES:
        for seed in protocol["confirmation_seeds"]:
            cfg = {**selected[key]["config"], "seed": seed, "max_steps": protocol["confirmation_steps"],
                   "eval_every": protocol["confirmation_eval_every"], "classifier_run": str(classifier_run), "output_dir": None}
            path = configuration_dir / f"{key}-s{seed}.json"
            write_json(path, cfg)
            paths.append(str(path))
    protocol.update(state="confirmation", confirmation_configs=paths, classifier_run=str(classifier_run))
    write_json(output / "protocol.json", protocol)
    confirmation_screen = launch_stage(output, "confirmation", paths)
    wait_stage(confirmation_screen)
    confirmation = summarize(confirmation_screen)
    for row in confirmation["rows"]:
        key = family(row["config"])
        wanted = dict(selected[key]["config"])
        wanted.update(seed=row["config"]["seed"], max_steps=30000, eval_every=5000,
                      classifier_run=str(classifier_run), output_dir=row["config"]["output_dir"])
        if row["config"] != wanted:
            raise ValueError("Confirmation differs from frozen selection")
    stats = statistics_from_rows(confirmation["rows"])
    report = {"protocol": protocol, "selection": frozen, "tuning": tuning, "confirmation": confirmation,
              "classifier": {"run": str(classifier_run), "results": class_results}, "statistics": stats,
              "total_new_loop_gpu_hours": sum(r["loop_seconds"] for stage in (tuning, confirmation) for r in stage["rows"]) / 3600,
              "uncertainty": "sample SD over three training seeds on one split with fixed sampling noise, not a confidence interval"}
    write_json(output / "report.json", report)
    destination = Path(protocol["repo_root"]) / "reports" / "dit-lr-confirmation.json"
    write_json(destination, report)
    protocol.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat(), report=str(destination))
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "complete", "report": str(destination), "statistics": stats}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or Path(__file__).resolve().parent / "runs" / f"dit-followup-{stamp}").resolve()
    if args.action == "prepare":
        prepare(output, args.launch)
    else:
        try:
            run(output)
        except Exception as exc:
            protocol = json.loads((output / "protocol.json").read_text())
            protocol.update(state="failed", error=repr(exc))
            write_json(output / "protocol.json", protocol)
            raise


if __name__ == "__main__":
    main()

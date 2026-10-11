#!/usr/bin/env python3
"""Prepare/launch one sequential GPU-claimed design suite under private supervisor."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shlex
import shutil
import subprocess
from datetime import datetime, timezone

from benchmark_dit_design import CASES
from benchmark_dit_gain import FAMILIES, calibration_plan
from benchmark_dit_generators import generator_cases
from train_dit_cifar10 import SOURCE_FILES, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--no-compile-layer", action="store_true")
    parser.add_argument("--gain-screen", action="store_true", help="calibrate functional updates, then confirm selected rates")
    parser.add_argument("--generator-screen", action="store_true", help="fractional generator widths, FFN control and conditional cores")
    args = parser.parse_args()
    if args.gain_screen and args.generator_screen:
        parser.error("Choose one screen")
    if args.steps < 10:
        parser.error("At least 10 steps are required for the initialization wake-up check")
    if args.gain_screen and not args.synthetic and args.steps < 100:
        parser.error("The real-data gain confirmation requires at least 100 steps")
    root = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workflow = "generators" if args.generator_screen else "gain" if args.gain_screen else "design"
    output = (args.output or root / "runs" / f"dit-{workflow}-{stamp}").resolve()
    if output.exists():
        raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free < 1024**3:
        raise RuntimeError("Require at least 1 GiB free for this compact workflow")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    for name in (*SOURCE_FILES, "benchmark_dit_design.py", "benchmark_dit_gain.py", "launch_dit_design.py",
                 "benchmark_dit_generators.py", "audit_dit_generators.py", "summarize_dit_generators.py", "test_dit_generators.py",
                 "audit_dit_design.py", "audit_dit_gain.py", "test_dit_cifar10.py", "test_dit_design.py", "test_dit_gain.py", "requirements.txt"):
        shutil.copy2(root / name, source / name)
    order = [row[0] for row in CASES]
    random.Random(77).shuffle(order)
    settings = {"output_dir": str(output), "steps": args.steps, "warmup": min(50, args.steps//5),
                "bank_examples": 4096 if not args.synthetic else 32,
                "val_examples": 256 if not args.synthetic else 16,
                "batch_size": 128 if not args.synthetic else 4,
                "width": 192 if not args.synthetic else 24, "depth": 6 if not args.synthetic else 2,
                "heads": 3, "patch_size": 4 if not args.synthetic else 8,
                "lr": 4e-4, "precision": "bf16" if args.device.startswith("cuda") else "fp32",
                "device": args.device, "synthetic": args.synthetic,
                "compile_layer": not args.no_compile_layer, "layer_steps": 30 if not args.synthetic else 2,
                "layer_repeats": 3 if not args.synthetic else 1, "run_order": order}
    if args.gain_screen:
        order = [row[0] for row in FAMILIES] + ["basis_r8_raw"]
        random.Random(78).shuffle(order)
        settings.update(projection_probes=False, compile_layer=False,
                        calibration_steps=10 if args.synthetic else 50,
                        calibration_observation_steps=[5,10] if args.synthetic else [10,25,50],
                        run_order=order, confirmation_order=order, calibration_plan=calibration_plan())
    if args.generator_screen:
        cases = generator_cases(settings["width"])
        order = [case["name"] for case in cases]
        random.Random(79).shuffle(order)
        settings.update(case_specifications=cases, run_order=order,
                        extra_provenance_sources=["benchmark_dit_generators.py"],
                        schedule_steps=30000, timing_warmup=min(50, args.steps//5))
    write_json(output / "settings.json", settings)
    service = f"adalora_dit_{workflow}_{stamp.lower()}"
    wrapper = Path("/opt/supervisor-scripts") / f"{service}.sh"
    control = output / "supervisor.conf"
    command = ["/workspace/bin/gpu-claim", "run", "--owner", "adalora", "--job", service, "--wait", "--",
               "/venv/main/bin/python", "-u", str(source / f"benchmark_dit_{workflow}.py"), "--config", str(output / "settings.json")]
    protocol = {"state": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "shared-stream initialization, gain, memory and timing engineering; no quality selection",
                "expected_storage_upper_bound_bytes": 64 * 1024**2, "disk_free_bytes_before_launch": free,
                "retention": "compact source/configs/seeds/metrics/logs/reports retained until explicitly removed",
                "weights_saved": False, "periodic_checkpoints": False, "test_evaluated": False, "ema": False,
                "estimated_gpu_hours": "0.15-0.4 including compilation; one sequential worker",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
                "repo_root": str(root), "settings": settings, "service": service,
                "supervisor_control": str(control), "wrapper": str(wrapper), "command": command,
                "stream_seeds": {"model": 42, "case_order": 77, "batch_selection_flip": 2030,
                                 "noise_timesteps": 2031, "fixed_factor_probe": 2032, "projection_probe": 2040,
                                 "diagnostic_images": 2027},
                "optimizer": "AdamW betas=(.9,.999), wd=0, all parameter groups same LR, global clip=1; warmup/cosine",
                "large_artifacts": "none; image bank and models are in memory only"}
    if args.gain_screen:
        protocol.update(purpose="fixed gain and bounded adapter-LR calibration on functional updates; 500-step engineering confirmation",
                        estimated_gpu_hours="0.05-0.2; at most 13 short calibration runs and six confirmations",
                        optimizer="AdamW betas=(.9,.999), wd=0, base LR=4e-4, uniform adapter LR multiplier, global clip=1; warmup/cosine",
                        calibration_plan=calibration_plan())
        protocol["stream_seeds"].pop("case_order")
        protocol["stream_seeds"].pop("projection_probe")
        protocol["stream_seeds"]["confirmation_order"] = 78
    if args.generator_screen:
        protocol.update(purpose="generator-width/rank budgets, dense/DPLR core correctness, learning, memory and timing; no quality ranking",
                        estimated_gpu_hours="0.2-0.6 including compilation; one sequential worker, 14 cases",
                        initialization="same base hash for equal MLP shapes; FFN control changes shapes and is seed-matched, not identical-base matched",
                        generator_width_rounding="max(1, condition_width // divisor), separately for each U/V head",
                        core_initialization="dense core head zero; DPLR gains/R heads zero, L head nonzero; static U/V nonzero",
                        scaling="fixed 1/sqrt(outer rank), LR multiplier 1; raw rank-8 basis control; not function-update calibrated",
                        schedule="500-step engineering prefix of 30k cosine horizon; warmup 50 at default budget",
                        expected_storage_upper_bound_bytes=128*1024**2)
        protocol["stream_seeds"]["case_order"] = 79
    write_json(output / "protocol.json", protocol)
    if args.launch:
        if wrapper.exists():
            raise FileExistsError(wrapper)
        wrapper.write_text("\n".join(["#!/bin/bash", "set -e",
            ". /opt/supervisor-scripts/utils/environment.sh", ". /opt/supervisor-scripts/utils/logging.sh",
            f"cd {shlex.quote(str(source))}",
            "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
            f"exec {shlex.join(command)} > >(tee -a {shlex.quote(str(output/'worker.log'))}) 2>&1", ""]))
        wrapper.chmod(0o755)
        control.write_text(f"""[unix_http_server]
file={output}/supervisor.sock
chmod=0600
[supervisord]
logfile={output}/supervisord.log
logfile_maxbytes=1048576
logfile_backups=1
pidfile={output}/supervisord.pid
childlogdir={output}
nodaemon=false
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{output}/supervisor.sock
[program:{service}]
environment=PROC_NAME="%(program_name)s"
command={wrapper}
directory={source}
autostart=false
autorestart=false
exitcodes=0
startsecs=1
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
""")
        subprocess.run(["supervisord", "-c", str(control)], check=True)
        subprocess.run(["supervisorctl", "-c", str(control), "start", service], check=True)
        protocol.update(state="launched", launched_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "protocol.json", protocol)
    print(json.dumps({"output": str(output), "state": protocol["state"]}))


if __name__ == "__main__":
    main()

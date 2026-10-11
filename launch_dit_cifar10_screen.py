#!/usr/bin/env python3
"""Launch a bounded DiT screen under a private supervisor and shared GPU claims."""

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from train_dit_cifar10 import SOURCE_FILES, load_config, provenance, write_json


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engineering", action="store_true", help="200-step GPU check, 1000 validation images, small grid")
    parser.add_argument("--fid-preflight", action="store_true", help="Engineering-only 128-image FID infrastructure check")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.fid_preflight and not args.engineering:
        parser.error("FID preflight is an engineering check")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    phase = "engineering" if args.engineering else "pilot"
    output = (args.output or root / "runs" / f"dit-{phase}-{stamp}").resolve()
    if output.exists():
        raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free < 1024**3:
        raise RuntimeError("Require 1 GiB free before launching this compact-evidence workflow")
    output.mkdir(parents=True)
    snapshot, configs, services = output / "source", output / "configs", output / "services"
    for directory in (snapshot, configs, services):
        directory.mkdir()
    for name in (*SOURCE_FILES, "requirements.txt", "launch_dit_cifar10_screen.py"):
        shutil.copy2(root / name, snapshot / name)
    jobs = []
    paths = args.configs or sorted((root / "configs" / "diffusion").glob("*.json"))
    if not paths:
        raise ValueError("No configs supplied")
    for path in paths:
        cfg = load_config(["--override_json", str(path)])
        if cfg.synthetic or cfg.download or cfg.train_limit is not None or cfg.val_limit is not None:
            raise ValueError("Source configs must use the full cached real train/validation split")
        if args.engineering:
            cfg.max_steps, cfg.warmup_steps, cfg.eval_every = 200, 20, 200
            cfg.val_limit, cfg.sample_images = 1000, 20
            cfg.fid_images = 128 if args.fid_preflight else 0
        elif cfg.fid_images != 5000 or cfg.max_steps != 10000:
            raise ValueError("This quality pilot requires the declared 10k-step/5k-FID budget")
        cfg.device = "cuda:0"
        cfg.output_dir = str(output / path.stem)
        resolved = configs / path.name
        write_json(resolved, asdict(cfg))
        load_config(["--override_json", str(resolved)])
        suffix = re.sub(r"[^a-zA-Z0-9_]", "_", path.stem)
        service = f"adalora_dit_{phase}_{stamp.lower()}_{suffix}"
        wrapper = Path("/opt/supervisor-scripts") / f"{service}.sh"
        command = ["/workspace/bin/gpu-claim", "run", "--owner", "adalora", "--job", service, "--wait", "--",
                   "/venv/main/bin/python", "-u", str(snapshot / "train_dit_cifar10.py"), "--override_json", str(resolved)]
        wrapper_text = "\n".join([
            "#!/bin/bash", "set -e", ". /opt/supervisor-scripts/utils/environment.sh",
            ". /opt/supervisor-scripts/utils/logging.sh", f"cd {shlex.quote(str(snapshot))}",
            "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
            f"exec {shlex.join(command)} > >(tee -a {shlex.quote(str(output / (path.stem + '.log')))}) 2>&1", ""])
        supervisor_text = f"""[program:{service}]
environment=PROC_NAME="%(program_name)s"
command={wrapper}
directory={snapshot}
autostart=false
autorestart=false
exitcodes=0
startsecs=3
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
"""
        (services / f"{service}.sh").write_text(wrapper_text)
        (services / f"{service}.conf").write_text(supervisor_text)
        jobs.append({"name": path.stem, "service": service, "config": str(resolved), "output_dir": cfg.output_dir,
                     "wrapper": str(wrapper), "command": command})
    cache = Path(cfg.fid_weights)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "phase": phase, "fid_preflight": args.fid_preflight,
        "output_dir": str(output), "jobs": jobs, "provenance": provenance(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()},
        "disk_free_bytes_before_launch": free, "expected_total_storage_bytes_upper_bound": len(jobs) * 20 * 1024**2,
        "purpose": "GPU numerical/throughput engineering check" if args.engineering else "external broadcast-conditioning quality pilot",
        "model_weights": False, "periodic_checkpoints": False, "ema": "in memory only; evaluate before process exits",
        "retention": "retain compact configs, sources, splits, metrics, logs, summaries and small sample grids until explicitly removed",
        "evaluator_cache": {"path": str(cache), "bytes": cache.stat().st_size, "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                            "purpose": "existing pretrained FID Inception evaluator; no training weights saved",
                            "retention": "shared existing cache; this workflow does not delete or replace it"},
        "gpu_allocation": "one shared lifetime GPU claim per run; wait for free GPUs; no preemption",
        "timing_limitation": "concurrent CPU/storage contention; pilot throughput is not an isolated efficiency benchmark",
        "supervisor_control": str(output / "supervisor.conf"), "launched": False,
        "protocol": {"dataset": "CIFAR-10 cached train only", "train_examples": 45000,
                     "validation_examples": 1000 if args.engineering else 5000, "test_evaluated": False,
                     "seed": 42, "split_seed": 123, "same_condition": "normalized learned class-plus-timestep, broadcast to all image tokens",
                     "reference": "common AdaLN-Zero-style heads and gates; extra low-rank transform at MLP input",
                     "sequence_conditioning_tested": False, "quality_claims": "single-seed pilot; requires LR tuning and multi-seed confirmation"}}
    write_json(output / "screen.json", manifest)
    if args.launch:
        for job in jobs:
            wrapper = Path(job["wrapper"])
            if wrapper.exists():
                raise FileExistsError(wrapper)
            wrapper.write_text((services / wrapper.name).read_text())
            wrapper.chmod(0o755)
        control = output / "supervisor.conf"
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
[include]
files={output}/services/*.conf
""")
        subprocess.run(["supervisord", "-c", str(control)], check=True)
        subprocess.run(["supervisorctl", "-c", str(control), "start", *[j["service"] for j in jobs]], check=True)
        manifest.update(launched=True, launched_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "screen.json", manifest)
    print(json.dumps({"screen": str(output / "screen.json"), "launched": manifest["launched"]}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare and launch a finite CIFAR-10 screen using supervisor and gpu-claim."""

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

from train_cifar10 import Config, load_config, provenance


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, help="Override config seeds; otherwise preserve each config's seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--configs", type=Path, nargs="+", help="Config files; defaults to the initial eight variants")
    parser.add_argument("--references", type=Path, help="JSON list of compatible completed reference runs")
    parser.add_argument("--purpose", default="CIFAR-10 learning-quality screen")
    parser.add_argument("--launch", action="store_true", help="Load and start only this screen's supervisor services")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or root / "runs" / f"cifar10-screen-{stamp}").resolve()
    if output.exists():
        raise FileExistsError(output)
    free_bytes = shutil.disk_usage(output.parent if output.parent.exists() else root).free
    if free_bytes < 100 * 1024**2:
        raise RuntimeError("This screen requires at least 100 MiB free for compact evidence")
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    for name in ("train_cifar10.py", "vit.py", "conditioning.py", "condlr.py", "requirements.txt"):
        shutil.copy2(root / name, snapshot / name)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()}
    (output / "configs").mkdir()
    jobs = []
    config_paths = args.configs or sorted((root / "configs" / "cifar10").glob("*.json"))
    seeds = set()
    for config_path in config_paths:
        cfg = load_config(["--override_json", str(config_path)])
        cfg.epochs, cfg.batch_size = args.epochs, args.batch_size
        if args.seed is not None:
            cfg.seed = args.seed
        seeds.add(cfg.seed)
        cfg.max_steps, cfg.train_limit, cfg.val_limit = None, None, None
        cfg.synthetic, cfg.download, cfg.evaluate_test, cfg.save_weights = False, False, False, False
        cfg.device, cfg.compile = "cuda:0", False
        cfg.output_dir = str(output / config_path.stem)
        resolved = output / "configs" / config_path.name
        write_json(resolved, asdict(cfg))
        # Validate the final serialized config, not just its original template.
        load_config(["--override_json", str(resolved)])
        suffix = re.sub(r"[^a-zA-Z0-9_]", "_", config_path.stem)
        service = f"adalora_cifar{args.epochs}_{stamp.lower()}_{suffix}"
        wrapper = Path("/opt/supervisor-scripts") / f"{service}.sh"
        supervisor_config = output / "services" / f"{service}.conf"
        command = ["/workspace/bin/gpu-claim", "run", "--owner", "adalora", "--job", service, "--wait", "--",
                   "/venv/main/bin/python", "-u", str(snapshot / "train_cifar10.py"), "--override_json", str(resolved)]
        # Finite research jobs do not need a portal/Caddy entry or web skip guard.
        wrapper_text = "\n".join([
            "#!/bin/bash", "set -e", ". /opt/supervisor-scripts/utils/environment.sh",
            ". /opt/supervisor-scripts/utils/logging.sh", f"cd {shlex.quote(str(snapshot))}",
            "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
            f"exec {shlex.join(command)} > >(tee -a {shlex.quote(str(output / (config_path.stem + '.log')))}) 2>&1", ""])
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
        jobs.append({"name": config_path.stem, "service": service, "config": str(resolved),
                     "output_dir": cfg.output_dir, "wrapper": str(wrapper), "supervisor_config": str(supervisor_config),
                     "command": command, "wrapper_text": wrapper_text, "supervisor_text": supervisor_text})
    references = json.loads(args.references.read_text()) if args.references else []
    for reference in references:
        cfg = json.loads(Path(reference["config"]).read_text())
        old_provenance = json.loads((Path(reference["output_dir"]) / "provenance.json").read_text())
        if not all(source_hashes[k] == v for k, v in old_provenance["source_sha256"].items()):
            raise ValueError("Reference source differs from the snapshot")
        if cfg["epochs"] != args.epochs or cfg["batch_size"] != args.batch_size:
            raise ValueError("Reference training budget differs")
        seeds.add(cfg["seed"])
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(output),
                "purpose": args.purpose,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "seed": next(iter(seeds)) if len(seeds) == 1 else None, "seeds": sorted(seeds), "split_seed": Config().split_seed,
                "training_examples": 45000, "validation_examples": 5000, "test_evaluated": False,
                "expected_steps_per_run": args.epochs * ((45000 + args.batch_size - 1) // args.batch_size),
                "source_sha256": source_hashes, "provenance": provenance(),
                "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "disk_free_bytes_before_launch": free_bytes, "expected_total_storage_bytes_upper_bound": len(jobs) * 10 * 1024**2,
                "artifact_retention": "retain configs, source snapshot, split indices, metrics, logs and summary until explicitly removed",
                "model_weights": False, "periodic_checkpoints": False,
                "gpu_allocation": "one lifetime GPU claim per run; wait for a free GPU; no forced preemption",
                "timing_limitation": "concurrent runs share CPU/storage; use isolated repeated measurements for final efficiency claims",
                "jobs": [{k: v for k, v in job.items() if k not in ("wrapper_text", "supervisor_text")} for job in jobs] + references,
                "supervisor_control": str(output / "supervisor.conf"),
                "launched": False}
    write_json(output / "screen.json", manifest)
    # Keep exact service definitions as audit inputs alongside the run files.
    (output / "services").mkdir()
    for job in jobs:
        (output / "services" / (job["service"] + ".sh")).write_text(job["wrapper_text"])
        (output / "services" / (job["service"] + ".conf")).write_text(job["supervisor_text"])
    if args.launch:
        if not Path("/workspace/bin/gpu-claim").is_file():
            raise FileNotFoundError("Shared GPU launcher /workspace/bin/gpu-claim is required")
        for job in jobs:
            wrapper = Path(job["wrapper"])
            if wrapper.exists():
                raise FileExistsError(f"Refusing to overwrite existing service {job['service']}")
            wrapper.write_text(job["wrapper_text"])
            wrapper.chmod(0o755)
        # Use a private UNIX socket and our own finite batch supervisor, so
        # unrelated instance configurations cannot block or restart our jobs.
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
        subprocess.run(["supervisorctl", "-c", str(control), "start", *[job["service"] for job in jobs]], check=True)
        manifest["launched"] = True
        manifest["launched_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "screen.json", manifest)
    print(json.dumps({"screen": str(output / "screen.json"), "launched": manifest["launched"],
                      "services": [job["service"] for job in jobs]}), flush=True)


if __name__ == "__main__":
    main()

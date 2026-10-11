#!/usr/bin/env python3
"""Run a bounded, sequential engineering screen and preserve compact evidence.

Short runs validate execution and cost; they do not establish quality rankings.
For sustained research jobs, run the trainer through the instance supervisor.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(f"runs/engineering-{time.time_ns()}"))
    parser.add_argument("--configs", type=Path, nargs="+", default=sorted((root / "configs/cifar10").glob("*.json")))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.val_limit) < 1 or args.num_workers < 0:
        parser.error("steps, batch size and validation limit must be positive; workers nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps({
        "purpose": "bounded execution, stability and efficiency screening; no quality ranking",
        "settings": {k: [str(p) for p in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items()},
        "expected_storage": "under 10 MiB for default eight-run screen; no model weights",
        "retention": "compact evidence retained until explicitly removed",
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    results = []
    for config in args.configs:
        run_dir = output / config.stem
        command = [sys.executable, str(root / "train_cifar10.py"), "--override_json", str(config.resolve()),
                   "--max-steps", str(args.steps), "--batch-size", str(args.batch_size),
                   "--val-limit", str(args.val_limit), "--num-workers", str(args.num_workers),
                   "--device", args.device, "--no-save-weights", "--no-evaluate-test",
                   "--output-dir", str(run_dir)]
        print(f"Starting {config.stem}: {args.steps} steps", flush=True)
        with (output / f"{config.stem}.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((run_dir / "results.json").read_text())
        result["config_name"] = config.stem
        result["run_dir"] = str(run_dir)
        results.append(result)
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("config_name", "parameters", "train_images_per_second",
                                                "peak_training_allocated_bytes", "validation")}), flush=True)
    print(f"Evidence: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

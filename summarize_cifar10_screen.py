#!/usr/bin/env python3
"""Read live or completed screen records and write a compact comparison."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_events(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A trainer can be writing its final line during a live snapshot.
            continue
    return records


def summarize(screen_path):
    screen = json.loads(screen_path.read_text())
    control = screen.get("supervisor_control")
    statuses = {}
    if control and Path(control).is_file():
        result = subprocess.run(["supervisorctl", "-c", control, "status"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                statuses[parts[0]] = parts[1]
    rows, split_hashes = [], set()
    for job in screen["jobs"]:
        directory = Path(job["output_dir"])
        events = read_events(directory / "metrics.jsonl")
        train = [e for e in events if e.get("event") == "train"]
        validation = [e for e in events if e.get("event") == "validation"]
        result_path = directory / "results.json"
        results = json.loads(result_path.read_text()) if result_path.is_file() else None
        state = "complete" if results else statuses.get(job["service"], "not_started").lower()
        if not results and state == "exited":
            state = "failed_or_incomplete"
        config = json.loads(Path(job["config"]).read_text())
        row = {"name": job["name"], "state": state, "service": job["service"],
               "step": max((e.get("step", 0) for e in events), default=0),
               "epochs_completed": len(validation), "latest_train_loss": train[-1]["loss"] if train else None,
               "latest_validation_accuracy": validation[-1]["accuracy"] if validation else None,
               "best_validation_accuracy": max((e["accuracy"] for e in validation), default=None),
               "validation_curve": [{k: e[k] for k in ("epoch", "step", "accuracy", "loss")} for e in validation],
               "latest_images_per_second": train[-1]["images_per_second"] if train else None,
               "recent_grad_norm_before_clip": train[-1]["grad_norm_before_clip"] if train else None,
               "recent_hyper_grad_norm_before_clip": train[-1]["hyper_grad_norm_before_clip"] if train else None,
               "output_dir": str(directory), "config": config}
        if results:
            row["results"] = results
            row["elapsed_gpu_hours"] = results["elapsed_seconds"] / 3600
        provenance = directory / "provenance.json"
        if provenance.exists():
            hashes = json.loads(provenance.read_text())["source_sha256"]
            row["source_matches_snapshot"] = all(screen["source_sha256"][k] == v for k, v in hashes.items())
        dataset = directory / "dataset.json"
        if dataset.exists():
            metadata = json.loads(dataset.read_text())
            value = {k: metadata[k] for k in ("train_indices", "validation_indices", "source_fingerprint")}
            row["split_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            row["train_examples"] = metadata["train_examples"]
            row["validation_examples"] = metadata["validation_examples"]
            split_hashes.add(row["split_sha256"])
        rows.append(row)
    summary = {"updated_utc": datetime.now(timezone.utc).isoformat(), "screen": str(screen_path.resolve()),
               "all_complete": all(r["state"] == "complete" for r in rows),
               "runs_started": sum(r["step"] > 0 for r in rows),
               "same_data_split": len(split_hashes) == 1 and all("split_sha256" in r for r in rows),
               "same_source_snapshot": all(r.get("source_matches_snapshot", False) for r in rows),
               "selection_limit": "tuning/confirmation uses a held-out training split; official test is not evaluated",
               "rows": rows}
    target = screen_path.parent / "summary.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temp.replace(target)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path, help="Path to the screen.json manifest")
    args = parser.parse_args()
    summary = summarize(args.screen)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    for row in summary["rows"]:
        print(json.dumps({k: row[k] for k in ("name", "state", "step", "epochs_completed", "latest_train_loss",
                                            "latest_validation_accuracy", "best_validation_accuracy", "latest_images_per_second")}))


if __name__ == "__main__":
    main()

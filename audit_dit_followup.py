#!/usr/bin/env python3
"""Independently verify budgets, selection, pairing and frozen evaluation inputs."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dit_followup import FAMILIES, family, select, statistics_from_rows
from summarize_dit_cifar10_screen import summarize, check_finite
from train_dit_cifar10 import write_json


def audit(output):
    output = Path(output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["state"] != "complete":
        raise ValueError("Workflow is not complete")
    tuning = summarize(output / "tuning/screen.json")
    confirmation = summarize(output / "confirmation/screen.json")
    selection = json.loads((output / "selection.json").read_text())
    if select(tuning["rows"]) != selection["selected"]:
        raise ValueError("Selection differs from the predeclared FID ranking")
    expected_counts = {key: 5 if key == "dplr" else 3 for key in FAMILIES}
    if dict(Counter(family(r["config"]) for r in tuning["rows"])) != expected_counts:
        raise ValueError("Wrong tuning candidate counts")
    for key in FAMILIES:
        combinations = {(r["config"]["lr"], r["config"]["adapter_lr_multiplier"])
                        for r in tuning["rows"] if family(r["config"]) == key}
        wanted = {(lr, 1.) for lr in protocol["base_lr_grid"]}
        if key == "dplr":
            wanted.update((2e-4, mult) for mult in (.3, 3.))
        if combinations != wanted:
            raise ValueError("Tuning grid differs from protocol")
    for row in tuning["rows"]:
        cfg = row["config"]
        if cfg["max_steps"] != 10000 or cfg["seed"] != 42 or cfg["classifier_run"] is not None:
            raise ValueError("Tuning budget, seed or evaluator changed")
    if tuning["source_sha256"] != confirmation["source_sha256"] or tuning["runtime"] != confirmation["runtime"]:
        raise ValueError("Source or runtime changed between stages")
    class_run = Path(protocol["classifier_run"])
    class_cfg = json.loads((class_run / "config.json").read_text())
    class_results = json.loads((class_run / "results.json").read_text())
    class_data = json.loads((class_run / "dataset.json").read_text())
    class_artifacts = json.loads((class_run / "artifact_manifest.json").read_text())
    check_finite(class_results)
    if (class_cfg["seed"] != 46 or class_cfg["epochs"] != 50 or class_results["completed_steps"] != 17600
            or class_results["processed_images"] != 2250000 or class_results["test"] is not None
            or class_results["validation"]["accuracy"] < protocol["classifier_real_validation_accuracy_floor"]
            or not class_artifacts["weights_saved"] or class_artifacts["periodic_checkpoints"]):
        raise ValueError("Invalid fixed classifier budget, quality floor or artifact declaration")
    class_source = json.loads((class_run / "provenance.json").read_text())["source_sha256"]
    if any(confirmation["source_sha256"][name] != digest for name, digest in class_source.items()):
        raise ValueError("Classifier implementation differs from the evaluator")
    class_screen = json.loads((output / "classifier/screen.json").read_text())
    class_log = (output / "classifier/supervisord.log").read_text()
    if any(f"exited: {j['service']} (exit status 0; expected)" not in class_log for j in class_screen["jobs"]):
        raise ValueError("Classifier did not exit successfully")
    expected_class_hashes = {name: hashlib.sha256((class_run / name).read_bytes()).hexdigest()
                             for name in ("model.pt", "config.json", "dataset.json", "provenance.json", "results.json")}
    seen_real = []
    for row in confirmation["rows"]:
        cfg = row["config"]
        wanted = dict(selection["selected"][family(cfg)]["config"])
        wanted.update(seed=cfg["seed"], max_steps=30000, eval_every=5000,
                      classifier_run=str(class_run), output_dir=cfg["output_dir"])
        if cfg != wanted:
            raise ValueError("Confirmation settings differ from frozen selection")
        data = json.loads((Path(row["output_dir"]) / "dataset.json").read_text())
        if any(data[k] != class_data[k] for k in ("train_indices", "validation_indices", "source_fingerprint")):
            raise ValueError("Classifier and diffusion data differ")
        gen = row["generative_evaluation"]
        agreement = gen["class_agreement"]
        if agreement["classifier"]["sha256"] != expected_class_hashes:
            raise ValueError("Frozen classifier provenance differs")
        for key in ("real", "generated"):
            metric = agreement[key]
            matrix = metric["confusion_rows_requested_columns_predicted"]
            if (metric["examples"] != 5000 or metric["class_counts"] != [500] * 10
                    or sum(sum(r) for r in matrix) != 5000
                    or abs(metric["agreement"] - sum(matrix[i][i] for i in range(10)) / 5000) > 1e-7):
                raise ValueError("Invalid class-agreement denominator or confusion matrix")
        real = agreement["real"]
        if abs(real["agreement"] - class_results["validation"]["accuracy"]) > .005:
            raise ValueError("FP32 real-data classifier accuracy differs materially from training evaluation")
        seen_real.append(real)
        if gen["sample_grid_labels"] != [i % 10 for i in range(100)]:
            raise ValueError("Sample-grid labels differ from balanced columns")
    if any(r != seen_real[0] for r in seen_real):
        raise ValueError("Real-data classifier predictions differ across confirmations")
    stats = statistics_from_rows(confirmation["rows"])
    report = json.loads((output / "report.json").read_text())
    if json.loads(json.dumps(stats)) != report["statistics"]:
        raise ValueError("Report statistics differ from recomputation")
    first_log_line = (output / "confirmation/supervisord.log").read_text().splitlines()[0]
    start = datetime.fromisoformat(first_log_line[:23].replace(",", ".")).replace(tzinfo=timezone.utc)
    if datetime.fromisoformat(selection["frozen_utc"]) >= start:
        raise ValueError("Selection was not frozen before confirmation")
    weights = [p.resolve() for p in output.rglob("*") if p.suffix in (".pt", ".pth", ".ckpt", ".safetensors")]
    if weights != [(class_run / "model.pt").resolve()]:
        raise ValueError("Expected exactly one declared auxiliary evaluator and no diffusion weights")
    if any((output / stage / "supervisor.sock").exists() for stage in ("classifier", "tuning", "confirmation")):
        raise ValueError("A stage supervisor remains active")
    return {"workflow": str(output), "created_utc": datetime.now(timezone.utc).isoformat(),
            "tuning": tuning["audit"], "confirmation": confirmation["audit"],
            "checks": {"selection_recomputed": True, "selection_frozen_before_confirmation": True,
                       "three_paired_seeds_per_family": True, "fixed_classifier_hashes": True,
                       "class_agreement_denominators": True, "same_real_classifier_predictions": True,
                       "only_declared_evaluator_weights": True, "no_test_evaluated": True,
                       "statistics_recomputed": True, "stage_supervisors_shutdown": True},
            "classifier_real_fp32_accuracy": seen_real[0]["agreement"],
            "classifier_weight_bytes": (class_run / "model.pt").stat().st_size,
            "bytes_retained_under_workflow": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "report_sha256": hashlib.sha256((output / "report.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/dit-lr-confirmation-audit.json"))
    args = parser.parse_args()
    result = audit(args.workflow)
    write_json(args.output, result)
    write_json(args.workflow / "audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

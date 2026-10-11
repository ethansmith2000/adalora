#!/usr/bin/env python3
"""Independently verify finished followup records without changing selection."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def family(cfg):
    return f"plain_w{cfg['width']}" if cfg["variant"] == "plain" and cfg["width"] != 192 else cfg["variant"]


def audit(root):
    protocol = read_json(root / "protocol.json")
    report = read_json(root / "report.json")
    selection = read_json(root / "selection.json")
    assert protocol["state"] == "complete", "Workflow is not complete"
    assert report["selection"] == selection, "Selection record changed"
    confirm_screen = read_json(root / "confirm/screen.json")
    assert datetime.fromisoformat(selection["frozen_utc"]) < datetime.fromisoformat(confirm_screen["created_utc"])
    source_hashes = {name: digest for name, digest in protocol["source_sha256"].items()
                     if name in ("train_cifar10.py", "vit.py", "conditioning.py", "condlr.py")}
    hardware = read_json(root / "hardware_inventory.json")
    known_uuids = {gpu["uuid"] for gpu in hardware["gpus"]}
    versions = set()
    split_hashes, common_configs, records = set(), [], []
    ignored = {"variant", "width", "heads", "rank", "n_basis", "lr", "hyper_lr_multiplier", "seed", "epochs", "output_dir"}
    grouped = {}
    for stage, epochs in (("tuning", 20), ("confirmation", 50)):
        summary = report[stage]
        assert summary["all_complete"] and summary["same_data_split"] and summary["same_source_snapshot"]
        stage_dir = root / ("tune" if stage == "tuning" else "confirm")
        supervisor_log = (stage_dir / "supervisord.log").read_text()
        claims = read_json(stage_dir / "hardware_claims.json")
        for row in summary["rows"]:
            directory = Path(row["output_dir"])
            cfg, result = read_json(directory / "config.json"), read_json(directory / "results.json")
            assert cfg == row["config"] and result == row["results"], "Summary differs from original evidence"
            common_configs.append({k: v for k, v in cfg.items() if k not in ignored})
            assert cfg["epochs"] == epochs and cfg["batch_size"] == 128
            assert not any(cfg[k] for k in ("synthetic", "evaluate_test", "save_weights", "compile", "download"))
            assert all(cfg[k] is None for k in ("max_steps", "train_limit", "val_limit"))
            assert result["completed_steps"] == 352 * epochs and result["processed_images"] == 45000 * epochs
            assert result["test"] is None
            provenance = read_json(directory / "provenance.json")
            assert provenance["source_sha256"] == source_hashes
            versions.add(tuple(provenance[k] for k in ("python", "torch", "cuda_wheel")))
            dataset = read_json(directory / "dataset.json")
            assert dataset["train_examples"] == 45000 and dataset["validation_examples"] == 5000
            train_indices, val_indices = dataset["train_indices"], dataset["validation_indices"]
            assert len(set(train_indices)) == 45000 and len(set(val_indices)) == 5000
            assert not set(train_indices).intersection(val_indices)
            split = {k: dataset[k] for k in ("train_indices", "validation_indices", "source_fingerprint")}
            split_hashes.add(hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest())
            manifest = read_json(directory / "artifact_manifest.json")
            assert not manifest["weights_saved"] and not manifest["periodic_checkpoints"]
            assert not any(p.suffix in (".pt", ".pth", ".safetensors", ".ckpt") for p in directory.rglob("*"))
            events = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            train = [e for e in events if e["event"] == "train"]
            validation = [e for e in events if e["event"] == "validation"]
            assert [e["epoch"] for e in validation] == list(range(1, epochs + 1))
            assert validation[-1]["accuracy"] == result["validation"]["accuracy"]
            assert sum(e["event"] == "complete" for e in events) == 1
            for event in train + validation:
                assert math.isfinite(event["loss"])
                for key in ("grad_norm_before_clip", "hyper_grad_norm_before_clip"):
                    if event.get(key) is not None:
                        assert math.isfinite(event[key])
                if "accuracy" in event:
                    assert 0 <= event["accuracy"] <= 1
            reused = row["service"].startswith("reference_")
            if not reused:
                assert f"exited: {row['service']} (exit status 0; expected)" in supervisor_log
                assert row["service"] in claims, "Physical GPU provenance missing"
                assert claims[row["service"]]["uuid"] in known_uuids
            if stage == "confirmation":
                selected_cfg = selection["selected"][family(cfg)]["config"]
                assert {k: v for k, v in cfg.items() if k not in ("seed", "epochs", "output_dir")} == {
                    k: v for k, v in selected_cfg.items() if k not in ("seed", "epochs", "output_dir")}
                grouped.setdefault(family(cfg), {})[cfg["seed"]] = result
            norms = [e["grad_norm_before_clip"] for e in train]
            records.append({"stage": stage, "name": row["name"], "reused": reused,
                            "logged_train_observations": len(train), "logged_max_grad_norm": max(norms),
                            "logged_fraction_grad_norm_above_clip": sum(n > cfg["max_grad_norm"] for n in norms) / len(norms)})
    assert len(split_hashes) == 1
    assert len(versions) == 1, "Runtime versions differ across runs"
    assert all(cfg == common_configs[0] for cfg in common_configs), "Shared training recipe mismatch"
    tuning_rows = report["tuning"]["rows"]
    for key, chosen in selection["selected"].items():
        candidates = [r for r in tuning_rows if family(r["config"]) == key]
        rank = lambda r: (r["results"]["validation"]["accuracy"],
                          -r["results"]["forward_matmul_flops_per_image_estimate"], -r["config"]["lr"])
        assert chosen == max(candidates, key=rank), "Selected configuration does not follow recorded rule"
    exact_accuracy = selection["selected"]["exact"]["results"]["validation"]["accuracy"]
    plain_accuracy = selection["selected"]["plain"]["results"]["validation"]["accuracy"]
    assert selection["include_exact_in_confirmation"] == (exact_accuracy >= plain_accuracy - 0.01)
    seeds = protocol["confirmation_seeds"]
    assert set(grouped) == set(protocol["confirmation_families_selected"])
    assert all(set(runs) == set(seeds) for runs in grouped.values())
    for key, runs in grouped.items():
        values = [runs[seed]["validation"]["accuracy"] for seed in seeds]
        stats = report["confirmation_statistics"][key]
        assert math.isclose(stats["mean_final_accuracy"], statistics.mean(values), abs_tol=1e-12)
        assert math.isclose(stats["sample_sd_final_accuracy"], statistics.stdev(values), abs_tol=1e-12)
        gains = [(runs[seed]["validation"]["accuracy"] - grouped["plain"][seed]["validation"]["accuracy"]) * 100
                 for seed in seeds]
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(gains, stats["paired_gain_vs_plain_percentage_points"], strict=True))
        assert math.isclose(stats["mean_gain_vs_plain_percentage_points"], statistics.mean(gains), abs_tol=1e-12)
    comparisons = {}
    for reference in ("plain", "diagonal", "static_lora_diagonal", "plain_w216", "plain_w240"):
        gains = [(grouped["dplr"][seed]["validation"]["accuracy"] -
                  grouped[reference][seed]["validation"]["accuracy"]) * 100 for seed in seeds]
        comparisons[reference] = {"seeds": seeds, "paired_dplr_gains_percentage_points": gains,
                                  "mean": statistics.mean(gains), "sample_sd": statistics.stdev(gains)}
    result = {"passed": True, "protocol_state": protocol["state"], "selection_frozen_before_confirmation": True,
              "training_source_sha256": source_hashes, "split_sha256": next(iter(split_hashes)),
              "runtime_versions": dict(zip(("python", "torch", "cuda_wheel"), next(iter(versions)), strict=True)),
              "hardware_inventory": hardware,
              "all_new_workers_exit_status_zero": True, "all_logged_losses_and_gradients_finite": True,
              "official_test_evaluated": False, "weights_or_checkpoints_saved": False,
              "common_recipe": common_configs[0], "tuning_endpoints": len(tuning_rows),
              "confirmation_runs": len(report["confirmation"]["rows"]),
              "dplr_paired_comparisons": comparisons, "stability_observations": records,
              "evidence_bytes_under_workflow": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
              "limitations": ["Three seeds on one fixed validation split; no confidence interval or test-set result.",
                              "DPLR received five tuning candidates; the other main families received three.",
                              "Parameter controls are approximate; FLOPs estimates exclude elementwise ops and backward.",
                              "Width 240 slightly exceeds DPLR's budget; no width-264 control is included in this 50-epoch confirmation of exact rank 2.",
                              "Concurrent shared-instance timings are not isolated efficiency measurements.",
                              "Gradient summaries cover logged steps, not every optimizer step."]}
    (root / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    args = parser.parse_args()
    result = audit(args.workflow)
    print(json.dumps({k: v for k, v in result.items() if k != "stability_observations"}, indent=2))


if __name__ == "__main__":
    main()

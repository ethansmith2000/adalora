#!/usr/bin/env python3
"""Independently validate design-suite completeness, common streams and evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmark_dit_design import CASES
from dit_cifar10 import CIFAR10DiT
from train_dit_cifar10 import write_json


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite evidence")
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    if isinstance(value, list):
        for item in value:
            finite(item)


def audit(output, require_exit=True):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    report = json.loads((output / "report.json").read_text())
    settings = json.loads((output / "settings.json").read_text())
    assert protocol["state"] == "complete" and settings == report["settings"] == protocol["settings"]
    assert not any(protocol[k] for k in ("weights_saved", "periodic_checkpoints", "test_evaluated", "ema"))
    finite(report)
    if require_exit:
        log = (output / "supervisord.log").read_text()
        assert f"exited: {protocol['service']} (exit status 0; expected)" in log
    for name, sha in protocol["source_sha256"].items():
        assert hashlib.sha256((output / "source" / name).read_bytes()).hexdigest() == sha
    for name, sha in report["runtime"]["source_sha256"].items():
        assert sha == protocol["source_sha256"][name]
    assert [r["name"] for r in report["rows"]] == settings["run_order"]
    assert set(settings["run_order"]) == {case[0] for case in CASES}
    rows = {r["name"]: r for r in report["rows"]}
    for key in ("stream_sha256", "first_noise_sha256", "noise_rng_final_state_sha256", "base_initialization_sha256"):
        assert len({r[key] for r in rows.values()}) == 1, key
    for rank in (2,4):
        a, b = rows[f"basis_r{rank}_normalized"], rows[f"basis_r{rank}_modulated"]
        assert a["adapter_initialization_sha256"] == b["adapter_initialization_sha256"]
    metadata = report["dataset"]
    if not settings["synthetic"]:
        assert len(metadata["train_indices"]) == settings["bank_examples"]
        assert len(metadata["validation_indices"]) == settings["val_examples"]
        assert len(set(metadata["train_indices"])) == settings["bank_examples"]
        assert not set(metadata["train_indices"]) & set(metadata["validation_indices"])
        assert metadata["test_evaluated"] is False
    for name, row in rows.items():
        cfg = json.loads((output / name / "config.json").read_text())
        assert cfg["name"] == name and row == json.loads((output / name / "results.json").read_text())
        assert row["runtime"] == report["runtime"]
        expected = next(case for case in CASES if case[0] == name)
        assert (row["variant"],row["rank"],row["adapter_input"]) == expected[1:]
        assert (cfg["variant"],cfg["rank"],cfg["adapter_input"]) == expected[1:]
        model = CIFAR10DiT(settings["width"],settings["depth"],settings["heads"],settings["patch_size"],
                           variant=row["variant"],rank=row["rank"],n_basis=16,adapter_input=row["adapter_input"])
        stats = model.statistics()
        assert all(row[key]==stats[key] for key in ("parameters","adapter_parameters","tokens_per_image","forward_matmul_flops_per_image_estimate"))
        expected_adapter_names = {key for key,_ in model.named_parameters() if ".adapter." in key}
        assert set(row["adapter_first_nonzero_gradient_observed"]) == expected_adapter_names
        assert row["completed_steps"] == settings["steps"]
        assert row["processed_images"] == settings["steps"] * settings["batch_size"]
        assert not any(row[k] for k in ("weights_saved", "test_evaluated", "ema_used", "quality_ranking_supported"))
        events = [json.loads(line) for line in (output / name / "metrics.jsonl").read_text().splitlines()]
        finite(events)
        assert events[-1] == {"event": "complete", "step": settings["steps"]}
        assert row["diagnostics"] == [e for e in events if e["event"] == "diagnostic"]
        assert [e["step"] for e in row["diagnostics"]] == row["diagnostic_observation_steps"]
        assert all(b["delta_rms"] == 0 and b["gate_rms"] == 0 for b in row["diagnostics"][0]["branches"])
        if row["variant"] != "adaln_zero":
            assert row["adapter_first_nonzero_gradient_observed"]
            final = row["diagnostics"][-1]["gradient_groups_after_clip"]
            assert all(len(v["nonzero_gradient_tensors"]) == v["parameter_tensors"] for k,v in final.items() if k != "base")
        assert row["train_images_per_second"] > 0
        expected_modes = ["eager", "compiled"] if settings["compile_layer"] else ["eager"]
        for scope in ("projection_microbenchmark","token_projection_microbenchmark"):
            assert [m["mode"] for m in row[scope]["modes"]] == expected_modes
            for mode in row[scope]["modes"]:
                assert mode["milliseconds_per_forward_backward"] > 0
                error = mode["compiled_max_relative_l2_output_gradient_error"]
                assert error is None or error <= (.03 if settings["precision"] == "bf16" else 2e-4)
    assert not [p for p in output.rglob("*") if p.suffix in (".pt", ".pth", ".ckpt", ".safetensors")]
    retained = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    assert retained < protocol["expected_storage_upper_bound_bytes"]
    result = {"checks": {"complete": True, "successful_worker_exit": require_exit,
                         "same_batches_noise_and_base_init": True, "paired_adapter_init": True,
                         "frozen_source_verified": True, "finite_metrics": True, "budgets_verified": True,
                         "zero_init_and_gradient_wakeup": True, "compiled_parity": settings["compile_layer"],
                         "weights_saved": False, "test_evaluated": False},
              "bytes_retained": retained, "report_sha256": hashlib.sha256((output / "report.json").read_bytes()).hexdigest()}
    write_json(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))


if __name__ == "__main__":
    main()

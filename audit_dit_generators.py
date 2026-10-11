#!/usr/bin/env python3
"""Audit generator/core engineering against frozen specifications and source."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_dit_design import finite
from benchmark_dit_generators import generator_cases
from dit_cifar10 import CIFAR10DiT
from train_dit_cifar10 import write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output, require_exit=True):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    report = json.loads((output / "report.json").read_text())
    settings = json.loads((output / "settings.json").read_text())
    assert protocol["state"] == "complete"
    assert settings == protocol["settings"] == report["settings"]
    assert settings["case_specifications"] == generator_cases(settings["width"])
    assert not any(protocol[k] for k in ("weights_saved", "periodic_checkpoints", "test_evaluated", "ema"))
    finite(report)
    if require_exit:
        assert f"exited: {protocol['service']} (exit status 0; expected)" in (output / "supervisord.log").read_text()
    for name, digest in protocol["source_sha256"].items():
        assert sha(output / "source" / name) == digest
    for name, digest in report["runtime"]["source_sha256"].items():
        assert digest == protocol["source_sha256"][name]
    if settings["device"].startswith("cuda"):
        claim = report["runtime"]["gpu_claim"]
        assert claim["GPU_CLAIM_INDICES"] and report["runtime"]["device_uuid"] in claim["CUDA_VISIBLE_DEVICES"]
    specs = {case["name"]: case for case in settings["case_specifications"]}
    assert [row["name"] for row in report["rows"]] == settings["run_order"]
    assert set(settings["run_order"]) == set(specs) and len(settings["run_order"]) == len(specs)
    rows = {row["name"]: row for row in report["rows"]}
    for key in ("stream_sha256", "first_noise_sha256", "noise_rng_final_state_sha256"):
        assert len({row[key] for row in rows.values()}) == 1, key
    equal_shape = [row for row in rows.values() if row["mlp_ratio"] == 4.]
    assert len({row["base_initialization_sha256"] for row in equal_shape}) == 1
    assert rows["basis_r8_s16"]["adapter_initialization_sha256"] == rows["basis_r8_s16_raw"]["adapter_initialization_sha256"]
    metadata = report["dataset"]
    if not settings["synthetic"]:
        assert len(set(metadata["train_indices"])) == settings["bank_examples"]
        assert len(set(metadata["validation_indices"])) == settings["val_examples"]
        assert not set(metadata["train_indices"]) & set(metadata["validation_indices"])
        assert not metadata["test_evaluated"]
    for name, row in rows.items():
        spec = specs[name]
        cfg = json.loads((output / name / "config.json").read_text())
        assert row == json.loads((output / name / "results.json").read_text())
        assert row["runtime"] == report["runtime"]
        for key, value in spec.items():
            assert cfg[key] == value and row[key] == value, (name, key)
        model = CIFAR10DiT(width=settings["width"], depth=settings["depth"], heads=settings["heads"],
                           patch_size=settings["patch_size"], **{key: value for key, value in spec.items() if key != "name"})
        stats = model.statistics()
        assert all(row[key] == stats[key] for key in ("parameters", "adapter_parameters", "tokens_per_image", "forward_matmul_flops_per_image_estimate"))
        names = {key for key, _ in model.named_parameters() if ".adapter." in key}
        assert set(row["adapter_first_nonzero_gradient_observed"]) == names
        assert row["completed_steps"] == settings["steps"]
        assert row["processed_images"] == settings["steps"] * settings["batch_size"]
        assert not any(row[key] for key in ("weights_saved", "test_evaluated", "ema_used", "quality_ranking_supported"))
        events = [json.loads(line) for line in (output / name / "metrics.jsonl").read_text().splitlines()]
        assert events[-1] == dict(event="complete", step=settings["steps"])
        assert row["diagnostics"] == [event for event in events if event["event"] == "diagnostic"]
        assert [event["step"] for event in row["diagnostics"]] == row["diagnostic_observation_steps"]
        assert all(branch["delta_rms"] == 0 and branch["gate_rms"] == 0 for branch in row["diagnostics"][0]["branches"])
        if names:
            groups = row["diagnostics"][-1]["gradient_groups_after_clip"]
            assert all(len(group["nonzero_gradient_tensors"]) == group["parameter_tensors"] for key, group in groups.items() if key != "base")
        assert row["train_images_per_second"] > 0 and row["raw_validation"]["examples"] == settings["val_examples"]
        for scope in ("projection_microbenchmark", "token_projection_microbenchmark"):
            probe = row[scope]
            assert [mode["mode"] for mode in probe["modes"]] == (["eager", "compiled"] if settings["compile_layer"] else ["eager"])
            assert probe["eager_saved_nonparameter_storage_bytes"] > 0
            for mode in probe["modes"]:
                assert mode["milliseconds_per_forward_backward"] > 0
                error = mode["compiled_max_relative_l2_output_gradient_error"]
                assert error is None or error <= (.03 if settings["precision"] == "bf16" else 2e-4)
    if settings["width"] == 192 and settings["depth"] == 6:
        assert rows["ffn_r2_budget"]["parameters"] - rows["basis_r2_s16"]["parameters"] == 384
        assert rows["core_r8"]["adapter_parameters"] == 120192
        assert rows["dplr_core_r8_p2"]["adapter_parameters"] == 92400
        assert rows["basis_r16_q32"]["adapter_parameters"] == 566856
    assert not [path for path in output.rglob("*") if path.suffix in (".pt", ".pth", ".ckpt", ".safetensors")]
    retained = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    assert retained < protocol["expected_storage_upper_bound_bytes"]
    result = dict(checks=dict(complete=True, successful_worker_exit=require_exit, frozen_source=True,
                             budgets=True, shared_data_noise=True, same_base_for_equal_shapes=True,
                             paired_raw_scaled_basis=True, zero_init_gradient_wakeup=True,
                             compiled_parity=settings["compile_layer"], weights_saved=False, quality_ranking=False),
                  bytes_retained=retained, report_sha256=sha(output / "report.json"))
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))

#!/usr/bin/env python3
"""Recompute core calibration, fresh paired seeds and FFN quality evidence."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path

import torch

from benchmark_dit_design import digest_parameters
from benchmark_dit_gain import choose_candidate, proposed_rate
from class_fidelity import classifier_provenance
from dit_core_confirmation import (CAL_FAMILIES, CAL_SEEDS, CONFIRM_SEEDS, FAMILIES,
                                   OBSERVATIONS, aggregate_candidate, digest, make_config)
from summarize_dit_cifar10_screen import summarize
from train_dit_cifar10 import Config, learning_rate_scale, make_model, write_json


def audit_stage(output, name, protocol, expected_rows, config_factory=make_config):
    directory=output/name
    screen=json.loads((directory/"screen.json").read_text())
    replay=summarize(directory/"screen.json")
    for row in replay["rows"]:
        row["family"]=next(job["family"] for job in screen["jobs"] if job["name"]==row["name"])
    assert replay["rows"]==expected_rows
    assert not (directory/"supervisor.sock").exists()
    claims=json.loads((directory/"hardware_claims.json").read_text())
    assert set(claims)=={job["service"] for job in screen["jobs"]}
    for job in screen["jobs"]:
        claim=claims[job["service"]]
        runtime=json.loads((Path(job["output_dir"])/"provenance.json").read_text())
        assert claim["owner"]=="adalora" and claim["job"]==job["service"] and claim["locker"]=="gpu-claim"
        assert claim["uuid"].removeprefix("GPU-")==runtime["device_uuid"]
    for filename,sha in protocol["source_sha256"].items():
        assert digest(directory/"source"/filename)==sha
    for seed in sorted({row["config"]["seed"] for row in expected_rows}):
        rows=[row for row in expected_rows if row["config"]["seed"]==seed]
        traces=[]; states=[]; bases=[]
        for row in rows:
            cfg=row["config"]
            expected=config_factory(row["family"],seed,cfg["adapter_lr_multiplier"],Path(cfg["output_dir"]),protocol["classifier_run"],name.startswith("calibration"))
            assert cfg==asdict(expected)
            assert cfg["train_limit"] is cfg["val_limit"] is None
            assert cfg["lr_schedule_steps"]==30000 and cfg["warmup_steps"]==50
            torch.manual_seed(seed)
            model=make_model(Config(**cfg))
            stats=model.statistics()
            assert all(row[key]==stats[key] for key in ("parameters","adapter_parameters","forward_matmul_flops_per_image_estimate"))
            diag=row["design_diagnostics"]
            assert diag["base_initialization_sha256"]==digest_parameters(model)
            assert diag["adapter_initialization_sha256"]==digest_parameters(model,True)
            assert diag["fixed_function_probe_precision"]=="FP32 tensors and highest FP32 matmul precision; no autocast"
            assert diag["model_state"]=="raw" and diag["fixed_input_seed"]==2032
            events=diag["diagnostics"]
            logs=[json.loads(line) for line in (Path(row["output_dir"])/"metrics.jsonl").read_text().splitlines()]
            assert events==[event for event in logs if event["event"]=="design_diagnostic"]
            assert [event["step"] for event in events]==[int(step) for step in cfg["diagnostic_steps"].split(",")]
            assert all(branch["delta_rms"]==branch["gate_rms"]==0 for branch in events[0]["branches"])
            names={key for key,_ in model.named_parameters() if ".adapter." in key}
            assert set(diag["adapter_first_nonzero_gradient_observed"])==names
            assert all(diag["adapter_first_nonzero_gradient_observed"][key]<=10 for key in names)
            assert all(len(group["nonzero_gradient_tensors"])==group["parameter_tensors"]
                       for key,group in events[-1]["gradient_groups_after_clip"].items() if key!="base")
            keys=("step","batch_images_sha256","batch_labels_sha256","noise_sha256","timesteps_sha256","noise_rng_state_sha256")
            traces.append([{key:event[key] for key in keys} for event in events[1:]])
            states.append(diag["noise_rng_final_state_sha256"])
            if cfg["mlp_ratio"]==4.:
                bases.append(diag["base_initialization_sha256"])
            assert all(math.isclose(event["base_lr"],cfg["lr"]*learning_rate_scale(Config(**cfg),event["step"]),rel_tol=1e-14)
                       for event in logs if event["event"]=="train")
        assert all(trace==traces[0] for trace in traces) and len(set(states))==1 and len(set(bases))<=1
        paired={row["family"]:row for row in rows}
        if "basis_r8_raw" in paired:
            assert paired["basis_r8"]["design_diagnostics"]["adapter_initialization_sha256"]==paired["basis_r8_raw"]["design_diagnostics"]["adapter_initialization_sha256"]
    return replay


def audit(output):
    torch.set_num_threads(2)
    output=Path(output).resolve()
    protocol=json.loads((output/"protocol.json").read_text())
    report=json.loads((output/"report.json").read_text())
    assert protocol["state"]=="complete" and protocol["max_concurrent_workers"]==6
    assert protocol["calibration_seeds"]==list(CAL_SEEDS) and protocol["confirmation_seeds"]==list(CONFIRM_SEEDS)
    assert set(CAL_SEEDS).isdisjoint(CONFIRM_SEEDS)
    assert protocol["observation_steps"]==list(OBSERVATIONS) and protocol["families"]==FAMILIES
    assert protocol["calibration_families"]==list(CAL_FAMILIES)
    assert not any(protocol[key] for key in ("weights_saved","periodic_checkpoints","test_evaluated"))
    for filename,sha in protocol["source_sha256"].items():
        assert digest(output/"runner"/filename)==sha
    calibration=json.loads((output/"calibration.json").read_text())
    assert calibration==report["calibration"]
    candidates={family:[] for family in CAL_FAMILIES}; rates={family:1. for family in CAL_FAMILIES}; target=None
    for attempt,summary in enumerate(calibration["rounds"]):
        rows=summary["rows"]
        audit_stage(output,f"calibration{attempt}",protocol,rows)
        assert {(row["family"],row["config"]["seed"]) for row in rows}=={(family,seed) for family in rates for seed in CAL_SEEDS}
        if target is None:
            target=aggregate_candidate("basis_r8",0,1.,[row for row in rows if row["family"]=="basis_r8"])["metric"]
        next_rates={}
        for family,rate in rates.items():
            assert all(row["config"]["adapter_lr_multiplier"]==rate for row in rows if row["family"]==family)
            candidate=aggregate_candidate(family,attempt,rate,[row for row in rows if row["family"]==family],target)
            candidates[family].append(candidate)
            if family!="basis_r8" and not candidate["within_tolerance"] and attempt<2:
                next_rates[family]=proposed_rate(rate,candidate["metric"],target)
        rates=next_rates
        if not rates:
            assert attempt==len(calibration["rounds"])-1
    selected={family:choose_candidate(values,target) for family,values in candidates.items()}
    assert target==calibration["target"] and candidates==calibration["candidates"] and selected==calibration["selected"]
    assert all(candidate["within_tolerance"] for candidate in selected.values())
    recipes=json.loads((output/"recipes.json").read_text())
    assert recipes==report["recipes"] and recipes["selected"]==selected and not recipes["selection_uses_quality"]
    assert recipes["calibration_sha256"]==digest(output/"calibration.json")
    assert protocol["recipes_sha256"]==digest(output/"recipes.json")
    screen=json.loads((output/"confirmation/screen.json").read_text())
    assert datetime.fromisoformat(recipes["frozen_utc"])<datetime.fromisoformat(screen["created_utc"])
    rows=report["confirmation"]["rows"]
    audit_stage(output,"confirmation",protocol,rows)
    assert len(rows)==21 and {(row["family"],row["config"]["seed"]) for row in rows}=={(family,seed) for family in FAMILIES for seed in CONFIRM_SEEDS}
    classifier=classifier_provenance(protocol["classifier_run"])
    assert classifier==protocol["auxiliary_classifier"]
    real=[]
    by_case={(row["family"],row["config"]["seed"]):row for row in rows}
    for seed in CONFIRM_SEEDS:
        assert by_case["ffn_r2_budget",seed]["parameters"]-by_case["basis_r2",seed]["parameters"]==384
    for row in rows:
        assert row["config"]["adapter_lr_multiplier"]==(selected[row["family"]]["adapter_lr_multiplier"] if row["family"] in selected else 1.)
        generated=row["generative_evaluation"]
        assert generated["class_agreement"]["classifier"]==classifier and generated["model_state"]=="EMA"
        assert generated["generated_images"]==generated["fid_real_images"]==generated["fid_fake_images"]==5000
        assert generated["sample_grid_labels"]==[i%10 for i in range(100)]
        for state in ("real","generated"):
            metric=generated["class_agreement"][state]
            matrix=metric["confusion_rows_requested_columns_predicted"]
            assert metric["examples"]==5000 and metric["class_counts"]==[500]*10
            assert [sum(row) for row in matrix]==[500]*10
            assert math.isclose(metric["agreement"],sum(matrix[i][i] for i in range(10))/5000,abs_tol=1e-7)
        real.append(generated["class_agreement"]["real"])
    assert all(value==real[0] for value in real)
    progress=[json.loads(line) for line in (output/"adalora_core_controller.log").read_text().splitlines() if line.startswith('{')]
    assert all(event["submitted"]-event["completed"]<=6 for event in progress if event.get("event")=="stage_progress")
    assert "exited: adalora_core_controller (exit status 0; expected)" in (output/"controller.log").read_text()
    assert not (output/"controller.sock").exists()
    assert not [path for path in output.rglob('*') if path.suffix in ('.pt','.pth','.ckpt','.safetensors')]
    retained=sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    assert retained<protocol["expected_storage_upper_bound_bytes"]
    checks=("complete","successful_worker_controller_exits","same_source_runtime_data","full_loader_schedule_prefix",
            "calibration_recomputed_without_quality_selection","frozen_rates_fresh_confirmation_seeds","paired_batch_noise_traces",
            "initialization_hashes_reproduced","equal_shape_base_init","raw_scaled_adapter_init","ffn_budget_within_384",
            "highest_fp32_probes","zero_init_gradient_wakeup","frozen_evaluators_denominators","bounded_concurrency",
            "claimed_hardware_matches_runtime","no_weights_test_evaluation","supervisors_closed","storage_within_bound")
    result=dict(checks={key:True for key in checks},report_sha256=digest(output/"report.json"),
                bytes_retained=retained,classifier_real_accuracy=real[0]["agreement"])
    write_json(output/"audit.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.output),indent=2))

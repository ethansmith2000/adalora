#!/usr/bin/env python3
"""Independently recompute full-loader calibration and paired confirmation evidence."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from benchmark_dit_gain import proposed_rate, choose_candidate
from class_fidelity import classifier_provenance
from dit_cifar10 import CIFAR10DiT
from dit_rank2_confirmation import (CAL_SEEDS, CONFIRM_SEEDS, FAMILIES, OBSERVATIONS,
                                    aggregate_candidate, digest, make_config)
from summarize_dit_cifar10_screen import summarize
from train_dit_cifar10 import write_json


def audit_stage(output, stage, protocol, expected_rows):
    directory=output/stage
    screen=json.loads((directory/"screen.json").read_text())
    replay=summarize(directory/"screen.json")
    for row in replay["rows"]:
        row["family"]=next(j["family"] for j in screen["jobs"] if j["name"]==row["name"])
    assert replay["rows"]==expected_rows
    assert not (directory/"supervisor.sock").exists()
    claims=json.loads((directory/"hardware_claims.json").read_text())
    assert {j["service"] for j in screen["jobs"]}==set(claims)
    for job in screen["jobs"]:
        claim=claims[job["service"]]
        runtime=json.loads((Path(job["output_dir"])/"provenance.json").read_text())
        assert claim["owner"]=="adalora" and claim["job"]==job["service"] and claim["locker"]=="gpu-claim"
        assert claim["uuid"].removeprefix("GPU-")==runtime["device_uuid"]
    for name,sha in protocol["source_sha256"].items():
        assert digest(directory/"source"/name)==sha
    for seed in sorted({r["config"]["seed"] for r in expected_rows}):
        rows=[r for r in expected_rows if r["config"]["seed"]==seed]
        traces=[]; states=[]; bases=[]
        for row in rows:
            cfg=row["config"]
            family=row["family"]
            expected=make_config(family,seed,cfg["adapter_lr_multiplier"],Path(cfg["output_dir"]),protocol["classifier_run"],stage.startswith("calibration"))
            from dataclasses import asdict
            assert cfg==asdict(expected)
            assert cfg["train_limit"] is cfg["val_limit"] is None
            assert cfg["lr_schedule_steps"]==30000 and cfg["warmup_steps"]==50
            variant,width,rank,gain=FAMILIES[family]
            model=CIFAR10DiT(width,cfg["depth"],cfg["heads"],cfg["patch_size"],variant=variant,rank=rank,adapter_gain=gain)
            stats=model.statistics()
            assert all(row[k]==stats[k] for k in ("parameters","adapter_parameters","forward_matmul_flops_per_image_estimate"))
            diagnostic=row["design_diagnostics"]
            assert diagnostic["fixed_function_probe_precision"]=="FP32 tensors and highest FP32 matmul precision; no autocast"
            assert diagnostic["model_state"]=="raw" and diagnostic["fixed_input_seed"]==2032
            events=diagnostic["diagnostics"]
            recorded=[json.loads(line) for line in (Path(row["output_dir"])/"metrics.jsonl").read_text().splitlines()]
            assert events==[r for r in recorded if r["event"]=="design_diagnostic"]
            assert [r["step"] for r in events]==[int(s) for s in cfg["diagnostic_steps"].split(",")]
            assert all(b["delta_rms"]==b["gate_rms"]==0 for b in events[0]["branches"])
            names={name for name,_ in model.named_parameters() if ".adapter." in name}
            assert set(diagnostic["adapter_first_nonzero_gradient_observed"])==names
            assert all(diagnostic["adapter_first_nonzero_gradient_observed"][name]<=10 for name in names)
            for group,v in events[-1]["gradient_groups_after_clip"].items():
                if group!="base":
                    assert len(v["nonzero_gradient_tensors"])==v["parameter_tensors"]
            traces.append([{k:r[k] for k in ("step","batch_images_sha256","batch_labels_sha256","noise_sha256","timesteps_sha256","noise_rng_state_sha256")} for r in events[1:]])
            states.append(diagnostic["noise_rng_final_state_sha256"])
            if width==192:
                bases.append(diagnostic["base_initialization_sha256"])
            trains=[r for r in recorded if r["event"]=="train"]
            from train_dit_cifar10 import Config,learning_rate_scale
            config=Config(**cfg)
            assert all(math.isclose(r["base_lr"],cfg["lr"]*learning_rate_scale(config,r["step"]),rel_tol=1e-14) for r in trains)
        assert all(t==traces[0] for t in traces) and len(set(states))==1 and len(set(bases))<=1
        paired={r["family"]:r for r in rows}
        if "dplr_r8_raw" in paired:
            assert paired["basis_r8"]["design_diagnostics"]["adapter_initialization_sha256"]==paired["dplr_r8_raw"]["design_diagnostics"]["adapter_initialization_sha256"]
    return replay


def audit(output):
    output=Path(output).resolve()
    protocol=json.loads((output/"protocol.json").read_text())
    report=json.loads((output/"report.json").read_text())
    assert protocol["state"]=="complete" and protocol["max_concurrent_workers"]==6
    assert protocol["calibration_seeds"]==list(CAL_SEEDS) and protocol["confirmation_seeds"]==list(CONFIRM_SEEDS)
    assert set(CAL_SEEDS).isdisjoint(CONFIRM_SEEDS)
    assert protocol["observation_steps"]==list(OBSERVATIONS)
    assert not any(protocol[k] for k in ("weights_saved","periodic_checkpoints","test_evaluated"))
    for name,sha in protocol["source_sha256"].items():
        assert digest(output/"runner"/name)==sha
    calibration=json.loads((output/"calibration.json").read_text())
    assert report["calibration"]==calibration
    candidates={f:[] for f in ("basis_r8","basis_r2","bottleneck_r2")}
    rates={f:1. for f in candidates}; target=None
    for attempt,summary in enumerate(calibration["rounds"]):
        rows=summary["rows"]
        audit_stage(output,f"calibration{attempt}",protocol,rows)
        assert {(r["family"],r["config"]["seed"]) for r in rows}=={(f,s) for f in rates for s in CAL_SEEDS}
        if target is None:
            target=aggregate_candidate("basis_r8",0,1.,[r for r in rows if r["family"]=="basis_r8"])["metric"]
        next_rates={}
        for family,rate in rates.items():
            assert all(r["config"]["adapter_lr_multiplier"]==rate for r in rows if r["family"]==family)
            candidate=aggregate_candidate(family,attempt,rate,[r for r in rows if r["family"]==family],target)
            candidates[family].append(candidate)
            if family!="basis_r8" and not candidate["within_tolerance"] and attempt<2:
                next_rates[family]=proposed_rate(rate,candidate["metric"],target)
        rates=next_rates
        if not rates:
            assert attempt==len(calibration["rounds"])-1
    selected={f:choose_candidate(values,target) for f,values in candidates.items()}
    assert target==calibration["target"] and candidates==calibration["candidates"] and selected==calibration["selected"]
    assert all(v["within_tolerance"] for v in selected.values())
    frozen=json.loads((output/"recipes.json").read_text())
    assert frozen==report["recipes"] and frozen["selected"]==selected and frozen["selection_uses_quality"] is False
    assert frozen["calibration_sha256"]==digest(output/"calibration.json")
    assert protocol["recipes_sha256"]==digest(output/"recipes.json")
    screen=json.loads((output/"confirmation/screen.json").read_text())
    assert datetime.fromisoformat(frozen["frozen_utc"])<datetime.fromisoformat(screen["created_utc"])
    rows=report["confirmation"]["rows"]
    audit_stage(output,"confirmation",protocol,rows)
    assert len(rows)==18 and {(r["family"],r["config"]["seed"]) for r in rows}=={(f,s) for f in FAMILIES for s in CONFIRM_SEEDS}
    classifier=classifier_provenance(protocol["classifier_run"])
    assert classifier==protocol["auxiliary_classifier"]
    real=[]
    for row in rows:
        cfg=row["config"]
        assert cfg["adapter_lr_multiplier"]==(selected[row["family"]]["adapter_lr_multiplier"] if row["family"] in selected else 1.)
        generated=row["generative_evaluation"]
        assert generated["class_agreement"]["classifier"]==classifier and generated["model_state"]=="EMA"
        assert generated["generated_images"]==generated["fid_real_images"]==generated["fid_fake_images"]==5000
        assert generated["sample_grid_labels"]==[i%10 for i in range(100)]
        for state in ("real","generated"):
            metric=generated["class_agreement"][state]
            matrix=metric["confusion_rows_requested_columns_predicted"]
            assert metric["examples"]==5000 and metric["class_counts"]==[500]*10
            assert [sum(v) for v in matrix]==[500]*10
            assert math.isclose(metric["agreement"],sum(matrix[i][i] for i in range(10))/5000,abs_tol=1e-7)
        real.append(generated["class_agreement"]["real"])
    assert all(v==real[0] for v in real)
    progress=[json.loads(line) for line in (output/"adalora_rank2_controller.log").read_text().splitlines() if line.startswith('{')]
    assert all(v["submitted"]-v["completed"]<=6 for v in progress if v.get("event")=="stage_progress")
    assert "exited: adalora_rank2_controller (exit status 0; expected)" in (output/"controller.log").read_text()
    assert not [p for p in output.rglob("*") if p.suffix in (".pt",".pth",".ckpt",".safetensors")]
    retained=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    assert retained<protocol["expected_storage_upper_bound_bytes"]
    result=dict(checks={k:True for k in ("complete","successful_worker_exits","same_source_runtime_data",
                "full_loader_and_matched_schedule","calibration_recomputed_without_quality_selection",
                "global_rates_frozen_before_independent_confirmation_seeds","same_paired_batch_noise_digests",
                "same_noise_end_states","same_paired_width192_initialization","paired_rank8_adapter_initialization",
                "highest_fp32_probes","zero_init_and_gradient_wakeup","same_frozen_evaluators_and_denominators",
                "bounded_concurrency","lifetime_claim_hardware_matches_runtime","no_new_weights_or_test_evaluation","workers_shutdown","storage_within_bound")},
                report_sha256=digest(output/"report.json"),bytes_retained=retained,
                classifier_real_accuracy=real[0]["agreement"])
    write_json(output/"audit.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.output),indent=2))

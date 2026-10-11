#!/usr/bin/env python3
"""Full-loader core update calibration, frozen recipes and paired quality controls."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

from benchmark_dit_gain import choose_candidate, proposed_rate
from class_fidelity import classifier_provenance
from dit_followup import private_supervisor
from dit_rank2_confirmation import digest, execute_stage, now
from train_dit_cifar10 import Config, SOURCE_FILES, write_json

CAL_SEEDS = (53,54)
CONFIRM_SEEDS = (55,56,57)
OBSERVATIONS = (50,100,250,500)
CAL_FAMILIES = ("basis_r8", "core_r8", "dplr_core_r8_p2")
FAMILIES = {
    "adaln": dict(variant="adaln_zero",adapter_gain=1.),
    "ffn_r2_budget": dict(variant="adaln_zero",mlp_ratio=4.5,adapter_gain=1.),
    "basis_r2": dict(variant="dplr",rank=2,adapter_gain=1/math.sqrt(2)),
    "basis_r8": dict(variant="dplr",rank=8,adapter_gain=1/math.sqrt(8)),
    "basis_r8_raw": dict(variant="dplr",rank=8,adapter_gain=1.),
    "core_r8": dict(variant="conditional_core",rank=8,adapter_gain=1/math.sqrt(8)),
    "dplr_core_r8_p2": dict(variant="conditional_dplr_core",rank=8,core_rank=2,adapter_gain=1/math.sqrt(8)),
}
EXTRA_SOURCE = ("dit_core_confirmation.py", "audit_dit_core.py", "summarize_dit_core.py", "test_dit_core.py",
                "dit_rank2_confirmation.py", "dit_quality_diagnostics.py", "benchmark_dit_design.py",
                "benchmark_dit_gain.py", "dit_followup.py", "summarize_dit_cifar10_screen.py",
                "test_dit_generators.py", "benchmark_dit_generators.py", "test_dit_quality.py",
                "test_dit_rank2.py", "test_dit_cifar10.py", "test_dit_design.py", "test_dit_gain.py", "requirements.txt")


def update_metric(row):
    events={event["step"]:event for event in row["design_diagnostics"]["diagnostics"]}
    value=statistics.mean(block["update_rms"] for step in OBSERVATIONS
                          for block in events[step]["fixed_token_condition_function_updates"])
    if not math.isfinite(value) or value<=0:
        raise ValueError("Expected positive finite fixed-input adapter updates")
    return value


def aggregate_candidate(family, attempt, multiplier, rows, target=None):
    if sorted(row["config"]["seed"] for row in rows)!=list(CAL_SEEDS):
        raise ValueError("Both declared calibration seeds are required")
    metrics={str(row["config"]["seed"]):update_metric(row) for row in rows}
    metric=statistics.mean(metrics.values())
    target=metric if target is None else target
    return dict(family=family,attempt=attempt,adapter_lr_multiplier=multiplier,
                metric=metric,per_seed_metrics=metrics,metric_to_target=metric/target,
                within_tolerance=abs(metric/target-1)<=.25)


def make_config(family, seed, multiplier, output, classifier, calibration=False):
    maximum=500 if calibration else 30000
    steps="0,1,3,5,10,25,50,100,250,500" if calibration else "0,1,3,5,10,25,50,100,250,500,1000,5000,10000,20000,30000"
    return Config(**FAMILIES[family],width=192,depth=6,heads=3,patch_size=4,n_basis=16,
                  adapter_lr_multiplier=multiplier,seed=seed,lr=4e-4,max_steps=maximum,
                  lr_schedule_steps=30000,warmup_steps=50,eval_every=500 if calibration else 5000,
                  log_every=100 if calibration else 500,num_workers=2,
                  classifier_run=None if calibration else classifier,
                  fid_images=0 if calibration else 5000,sample_images=0 if calibration else 100,
                  diagnostic_steps=steps,output_dir=str(output))


def prepare(output, launch):
    root=Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError(output)
    free=shutil.disk_usage(root).free
    if free<1024**3:
        raise RuntimeError("Require 1 GiB free for compact evidence")
    classifier=str(root/"runs/dit-followup-20261007T174446Z/classifier/classifier_config")
    auxiliary=classifier_provenance(classifier)
    evaluator_path=Path(Config().fid_weights)
    evaluator=dict(path=str(evaluator_path),bytes=evaluator_path.stat().st_size,sha256=digest(evaluator_path))
    output.mkdir(parents=True)
    runner=output/"runner"; runner.mkdir()
    for name in (*SOURCE_FILES,*EXTRA_SOURCE):
        shutil.copy2(root/name,runner/name)
    protocol=dict(state="prepared",created_utc=now(),repo_root=str(root),
                  purpose="full-loader two-seed conditional-core update calibration and seven-family paired quality confirmation including the FFN capacity control",
                  source_sha256={path.name:digest(path) for path in runner.iterdir()},
                  calibration_seeds=list(CAL_SEEDS),confirmation_seeds=list(CONFIRM_SEEDS),
                  calibration_families=list(CAL_FAMILIES),families=FAMILIES,
                  calibration_steps=500,train_steps=30000,lr_schedule_steps=30000,warmup_steps=50,
                  observation_steps=list(OBSERVATIONS),target_family="basis_r8",target_multiplier=1.,
                  initial_family_multipliers={family:1. for family in CAL_FAMILIES},
                  calibration_metric="arithmetic mean highest-FP32 fixed-input adapter-delta update RMS across 6 blocks, steps 50/100/250/500 and seeds 53/54",
                  selection="closest absolute log(mean update / target); excludes loss/validation/FID/class agreement",
                  correction="clip(previous multiplier * target / previous mean update, 1/16, 16)",
                  max_corrections=2,relative_tolerance=.25,
                  fixed_control_multiplier=1.,fixed_gains="1/sqrt(rank), except base/FFN/raw-r8 gain 1",
                  max_concurrent_workers=6,order_seed=81,train_examples=45000,validation_examples=5000,
                  split_seed=123,evaluation_seed=2027,sampling_seed=2028,sampling_steps=50,fid_images=5000,
                  evaluator_cache=evaluator,auxiliary_classifier=auxiliary,classifier_run=classifier,
                  disk_free_bytes_before_launch=free,expected_storage_upper_bound_bytes=512*1024**2,
                  estimated_new_gpu_hours="7–10 including validation and final 5,000-image evaluations",
                  weights_saved=False,periodic_checkpoints=False,test_evaluated=False,
                  ema="in-memory .999; existing evaluators reused, no new model/evaluator weights",
                  retention="compact source/configs/splits/metrics/logs/reports and one 100-image grid per confirmation until explicitly removed",
                  limits=["Scalar aggregate update matching does not match each seed/block/time or live branch magnitude.",
                          "Confirmation seeds 55/56/57 are independent of calibration 53/54 and previous screens; one fixed held-out split.",
                          "FFN capacity differs from basis r2 by 384 parameters, but its shape change also changes base initialization; seed-matched only.",
                          "Equal-shape variants have identical base initialization; raw/scaled r8 share adapter initialization.",
                          "Rates are selected on function updates; controls keep multiplier 1; no quality-rate optimization.",
                          "Three seeds and sample SD are not confidence intervals; finite 5k FID and one sampling seed.",
                          "Concurrent eager throughput is not an isolated speed comparison; equal steps are not equal wall-clock.",
                          "Common non-affine norms/AdaLN-Zero; normalized adapter input; broadcast class/time only; no sequence-context retrieval or cross-attention.",
                          "Gains and identity initialization are empirical choices, not variance preservation or a muP derivation."])
    write_json(output/"protocol.json",protocol)
    if launch:
        private_supervisor(output,[("adalora_core_controller",[sys.executable,"-u",str(runner/Path(__file__).name),"run","--output",str(output)])],"controller",True)
    print(json.dumps(dict(output=str(output),launched=launch)),flush=True)


def run(output):
    protocol=json.loads((output/"protocol.json").read_text())
    protocol.update(state="calibrating",started_utc=now()); write_json(output/"protocol.json",protocol)
    candidates={family:[] for family in CAL_FAMILIES}
    rates={family:1. for family in CAL_FAMILIES}; selected={}; rounds=[]; target=None
    for attempt in range(3):
        specs=[(family,seed,rate) for family,rate in rates.items() for seed in CAL_SEEDS]
        summary=execute_stage(output,protocol,f"calibration{attempt}",specs,config_factory=make_config)
        rounds.append(summary)
        if target is None:
            target=aggregate_candidate("basis_r8",0,1.,[row for row in summary["rows"] if row["family"]=="basis_r8"])["metric"]
        next_rates={}
        for family,rate in rates.items():
            candidate=aggregate_candidate(family,attempt,rate,[row for row in summary["rows"] if row["family"]==family],target)
            candidates[family].append(candidate)
            selected[family]=choose_candidate(candidates[family],target)
            if family!="basis_r8" and not candidate["within_tolerance"] and attempt<2:
                next_rates[family]=proposed_rate(rate,candidate["metric"],target)
        write_json(output/"calibration_progress.json",dict(target=target,candidates=candidates,selected=selected))
        rates=next_rates
        if not rates:
            break
    calibration=dict(target=target,candidates=candidates,selected=selected,rounds=rounds)
    write_json(output/"calibration.json",calibration)
    if not all(candidate["within_tolerance"] for candidate in selected.values()):
        raise RuntimeError("Bounded core calibration missed tolerance; confirmation withheld")
    frozen=dict(frozen_utc=now(),calibration_sha256=digest(output/"calibration.json"),selected=selected,
                selection_uses_quality=False,confirmation_seeds=list(CONFIRM_SEEDS),
                fixed_control_multiplier=1.)
    write_json(output/"recipes.json",frozen)
    protocol.update(state="confirming",recipes_sha256=digest(output/"recipes.json"),recipes_frozen_utc=frozen["frozen_utc"])
    write_json(output/"protocol.json",protocol)
    specs=[(family,seed,selected[family]["adapter_lr_multiplier"] if family in selected else 1.)
           for family in FAMILIES for seed in CONFIRM_SEEDS]
    confirmation=execute_stage(output,protocol,"confirmation",specs,config_factory=make_config)
    report=dict(protocol=protocol,calibration=calibration,recipes=frozen,confirmation=confirmation,
                limits=protocol["limits"],quality_winner_claimed=False)
    write_json(output/"report.json",report)
    protocol.update(state="complete",completed_utc=now()); write_json(output/"protocol.json",protocol)
    print(json.dumps(dict(event="suite_complete",report=str(output/"report.json"))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--launch",action="store_true")
    args=parser.parse_args()
    output=(args.output or Path(__file__).resolve().parent/"runs"/("dit-core-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    prepare(output,args.launch) if args.action=="prepare" else run(output)

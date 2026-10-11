#!/usr/bin/env python3
"""Full-loader update calibration, then frozen-rate paired-seed DiT confirmation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

from benchmark_dit_gain import proposed_rate, choose_candidate
from class_fidelity import classifier_provenance
from dit_followup import private_supervisor
from summarize_dit_cifar10_screen import summarize
from train_dit_cifar10 import Config, SOURCE_FILES, load_config, write_json

CAL_SEEDS = (48, 49)
CONFIRM_SEEDS = (50, 51, 52)
OBSERVATIONS = (50, 100, 250, 500)
FAMILIES = {
    "basis_r8": ("dplr", 192, 8, 1/math.sqrt(8)),
    "basis_r2": ("dplr", 192, 2, 1/math.sqrt(2)),
    "bottleneck_r2": ("conditional_bottleneck", 192, 2, 1/math.sqrt(2)),
    "dplr_r8_raw": ("dplr", 192, 8, 1.),
    "adaln_d192": ("adaln_zero", 192, 8, 1.),
    "adaln_d240": ("adaln_zero", 240, 8, 1.),
}
EXTRA_SOURCE = ("dit_rank2_confirmation.py", "dit_quality_diagnostics.py", "benchmark_dit_design.py",
                "benchmark_dit_gain.py", "dit_followup.py", "summarize_dit_cifar10_screen.py",
                "test_dit_quality.py", "test_dit_rank2.py", "requirements.txt")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def update_metric(row):
    events = {e["step"]: e for e in row["design_diagnostics"]["diagnostics"]}
    value = statistics.mean(v["update_rms"] for step in OBSERVATIONS
                            for v in events[step]["fixed_token_condition_function_updates"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected positive finite fixed-input transform updates")
    return value


def aggregate_candidate(family, attempt, multiplier, rows, target=None):
    if sorted(r["config"]["seed"] for r in rows) != list(CAL_SEEDS):
        raise ValueError("Both declared calibration seeds are required")
    metrics = {str(r["config"]["seed"]): update_metric(r) for r in rows}
    metric = statistics.mean(metrics.values())
    target = metric if target is None else target
    return dict(family=family, attempt=attempt, adapter_lr_multiplier=multiplier,
                metric=metric, per_seed_metrics=metrics, metric_to_target=metric/target,
                within_tolerance=abs(metric/target-1) <= .25)


def make_config(family, seed, multiplier, output, classifier, calibration=False):
    variant, width, rank, gain = FAMILIES[family]
    maximum = 500 if calibration else 30000
    diagnostics = "0,1,3,5,10,25,50,100,250,500" if calibration else "0,1,3,5,10,25,50,100,250,500,1000,5000,10000,20000,30000"
    return Config(variant=variant,width=width,rank=rank,adapter_gain=gain,
                  adapter_lr_multiplier=multiplier,seed=seed,lr=4e-4,max_steps=maximum,
                  lr_schedule_steps=30000,warmup_steps=50,eval_every=500 if calibration else 5000,
                  log_every=100 if calibration else 500,num_workers=2,
                  classifier_run=None if calibration else classifier,
                  fid_images=0 if calibration else 5000,sample_images=0 if calibration else 100,
                  diagnostic_steps=diagnostics,output_dir=str(output))


def prepare(output, launch):
    root = Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free < 1024**3:
        raise RuntimeError("Require 1 GiB free for compact evidence")
    classifier = str(root/"runs/dit-followup-20261007T174446Z/classifier/classifier_config")
    auxiliary = classifier_provenance(classifier)
    cache = Path(Config().fid_weights)
    evaluator = dict(path=str(cache),bytes=cache.stat().st_size,sha256=digest(cache))
    output.mkdir(parents=True)
    runner = output/"runner"; runner.mkdir()
    for name in (*SOURCE_FILES,*EXTRA_SOURCE):
        shutil.copy2(root/name,runner/name)
    protocol = dict(state="prepared",created_utc=now(),repo_root=str(root),
                    purpose="full-loader multi-seed functional-update calibration followed by longer paired-seed quality confirmation",
                    source_sha256={p.name:digest(p) for p in runner.iterdir()},
                    calibration_seeds=list(CAL_SEEDS),confirmation_seeds=list(CONFIRM_SEEDS),
                    calibration_steps=500,train_steps=30000,lr_schedule_steps=30000,warmup_steps=50,
                    observation_steps=list(OBSERVATIONS),target_family="basis_r8",target_multiplier=1.,
                    initial_family_multipliers={f:1. for f in ("basis_r2","bottleneck_r2")},
                    calibration_metric="arithmetic mean highest-FP32 fixed-input delta-update RMS across 6 blocks, steps 50/100/250/500, and seeds 48/49",
                    selection="closest absolute log(mean update / reference target); no loss, validation, FID or class agreement",
                    correction="clip(previous multiplier * target / previous mean update, 1/16, 16)",
                    max_corrections=2,relative_tolerance=.25,
                    families={f:dict(variant=v,width=w,rank=r,gain=g) for f,(v,w,r,g) in FAMILIES.items()},
                    max_concurrent_workers=6,order_seed=80,train_examples=45000,validation_examples=5000,
                    split_seed=123,evaluation_seed=2027,sampling_seed=2028,sampling_steps=50,fid_images=5000,
                    evaluator_cache=evaluator,auxiliary_classifier=auxiliary,classifier_run=classifier,
                    disk_free_bytes_before_launch=free,expected_storage_upper_bound_bytes=512*1024**2,
                    estimated_new_gpu_hours="6–9 including validation and final quality evaluation",
                    weights_saved=False,periodic_checkpoints=False,test_evaluated=False,
                    ema="in-memory .999 only; no new evaluator weights",
                    retention="compact source/configs/splits/metrics/logs/reports and one 100-image grid per confirmation until explicitly removed",
                    limits=["Calibration matches a scalar aggregate; individual seeds, blocks and later updates can drift.",
                            "Three training seeds and one held-out split; sample SD is not a confidence interval.",
                            "Finite 5,000-image FID; one fixed sampling seed; auxiliary classifier agreement is a proxy.",
                            "Rates are update-selected, not quality-optimized; gains are empirical choices, not a variance-preservation or muP guarantee.",
                            "Concurrent throughput is not an isolated speed comparison; equal steps are not equal wall-clock budgets.",
                            "Warmup 50 differs from the earlier 30k study's warmup 500; only current controls are recipe-matched.",
                            "Broadcast class/time conditioning only; sequence-valued cross-attention remains untested."])
    write_json(output/"protocol.json",protocol)
    if launch:
        private_supervisor(output,[("adalora_rank2_controller",[sys.executable,"-u",str(runner/Path(__file__).name),"run","--output",str(output)])],"controller",True)
    print(json.dumps(dict(output=str(output),launched=launch)),flush=True)


def execute_stage(output, protocol, name, specifications, config_factory=make_config):
    stage = output/name; stage.mkdir()
    source = stage/"source"
    shutil.copytree(output/"runner",source,ignore=shutil.ignore_patterns("__pycache__"))
    configs = stage/"configs"; configs.mkdir()
    jobs=[]; programs=[]
    for family,seed,multiplier in specifications:
        case = f"{family}_s{seed}"
        config = config_factory(family,seed,multiplier,stage/case,protocol["classifier_run"],name.startswith("calibration"))
        path = configs/(case+".json")
        write_json(path,asdict(config)); load_config(["--override_json",str(path)])
        service = f"adalora_{output.name.replace('-','_')}_{name}_{case}"
        command = ["/workspace/bin/gpu-claim","run","--owner","adalora","--job",service,"--wait","--",
                   sys.executable,"-u",str(source/"train_dit_cifar10.py"),"--override_json",str(path)]
        jobs.append(dict(name=case,family=family,config=str(path),output_dir=config.output_dir,service=service,command=command))
        programs.append((service,command))
    random.Random(protocol["order_seed"]+int(name[-1]) if name.startswith("calibration") else protocol["order_seed"]+3).shuffle(jobs)
    phase = ("calibration" if name.startswith("calibration") else protocol["quality_phase"]) if "quality_phase" in protocol else "confirmation"
    write_json(stage/"screen.json",dict(phase=phase,stage=name,created_utc=now(),jobs=jobs,evaluator_cache=protocol["evaluator_cache"],
               protocol=dict(train_examples=45000,validation_examples=5000,test_evaluated=False)))
    cap = protocol["max_concurrent_workers"]
    first_services = [j["service"] for j in jobs[:cap]]
    submitted = set(first_services)
    control = private_supervisor(stage,programs,start_programs=first_services)
    claims={}
    while True:
        for path in Path("/tmp/vast_gpu_claims").glob("*.claim"):
            try:
                claim=json.loads(path.read_text())
            except (OSError,ValueError):
                continue
            if claim.get("job") in submitted:
                claims[claim["job"]]=claim
        write_json(stage/"hardware_claims.json",claims)
        status=subprocess.run(["supervisorctl","-c",control,"status"],capture_output=True,text=True).stdout
        if not status.strip() or any(t in status for t in ("FATAL","BACKOFF")):
            raise RuntimeError(status or "No supervisor status")
        log=(stage/"supervisord.log").read_text()
        completed={j["service"] for j in jobs if f"exited: {j['service']} (exit status 0; expected)" in log}
        for line in status.splitlines():
            service=line.split()[0]
            if service in submitted and "EXITED" in line and service not in completed:
                raise RuntimeError(f"Failed worker: {line}; evidence retained")
        if any(not (Path(j["output_dir"])/"results.json").exists() for j in jobs if j["service"] in completed):
            raise RuntimeError("Successful worker has no results")
        pending=[j for j in jobs if j["service"] not in submitted]
        for j in pending[:cap-len(submitted-completed)]:
            subprocess.run(["supervisorctl","-c",control,"start",j["service"]],check=True)
            submitted.add(j["service"])
        print(json.dumps(dict(event="stage_progress",stage=name,completed=len(completed),submitted=len(submitted),total=len(jobs))),flush=True)
        if len(completed)==len(jobs):
            break
        time.sleep(20)
    subprocess.run(["supervisorctl","-c",control,"shutdown"],check=True)
    report=summarize(stage/"screen.json")
    for row in report["rows"]:
        row["family"]=next(j["family"] for j in jobs if j["name"]==row["name"])
    write_json(stage/"summary.json",report)
    return report


def run(output):
    protocol=json.loads((output/"protocol.json").read_text())
    protocol.update(state="calibrating",started_utc=now()); write_json(output/"protocol.json",protocol)
    candidates={f:[] for f in ("basis_r8","basis_r2","bottleneck_r2")}
    rates={f:1. for f in candidates}; selected={}; rounds=[]; target=None
    for attempt in range(3):
        specifications=[(f,s,rate) for f,rate in rates.items() for s in CAL_SEEDS]
        summary=execute_stage(output,protocol,f"calibration{attempt}",specifications)
        rounds.append(summary)
        if target is None:
            reference=aggregate_candidate("basis_r8",attempt,1.,[r for r in summary["rows"] if r["family"]=="basis_r8"])
            target=reference["metric"]
        next_rates={}
        for family,rate in rates.items():
            candidate=aggregate_candidate(family,attempt,rate,[r for r in summary["rows"] if r["family"]==family],target)
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
    if not all(c["within_tolerance"] for c in selected.values()):
        raise RuntimeError("Bounded calibration did not match aggregate updates; confirmation withheld")
    frozen=dict(frozen_utc=now(),calibration_sha256=digest(output/"calibration.json"),selected=selected,
                selection_uses_quality=False,confirmation_seeds=list(CONFIRM_SEEDS))
    write_json(output/"recipes.json",frozen)
    protocol.update(state="confirming",recipes_sha256=digest(output/"recipes.json"),recipes_frozen_utc=frozen["frozen_utc"])
    write_json(output/"protocol.json",protocol)
    specifications=[(f,s,selected[f]["adapter_lr_multiplier"] if f in selected else 1.) for f in FAMILIES for s in CONFIRM_SEEDS]
    confirmation=execute_stage(output,protocol,"confirmation",specifications)
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
    output=(args.output or Path(__file__).resolve().parent/"runs"/("dit-rank2-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    prepare(output,args.launch) if args.action=="prepare" else run(output)

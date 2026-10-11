#!/usr/bin/env python3
"""Two-GPU, paired-seed beta1/CFG study with in-memory convergence evaluation."""

import argparse
from dataclasses import asdict, replace
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

from class_fidelity import classifier_provenance
from dit_followup import private_supervisor
from summarize_dit_cifar10_screen import check_finite, summarize
from train_dit_cifar10 import Config, SOURCE_FILES, load_config, write_json

SEEDS = (86, 87)
FAMILIES = {
    "adaln_b09": dict(variant="adaln_zero", adam_beta1=.9),
    "adaln_b095": dict(variant="adaln_zero", adam_beta1=.95),
    "dplr_b09": dict(variant="dplr", adam_beta1=.9),
    "dplr_b095": dict(variant="dplr", adam_beta1=.95),
}
STEPS = (5000, 10000, 20000, 30000)
SCALES = (1., 1.5, 2.)
DIAGNOSTICS = "0,1,3,5,10,25,50,100,250,500,1000,5000,10000,20000,30000"
EXTRA_SOURCE = ("dit_optimizer_cfg.py", "test_dit_optimizer_cfg.py", "test_dit_training_options.py",
                "dit_quality_diagnostics.py", "benchmark_dit_design.py", "dit_followup.py",
                "summarize_dit_cifar10_screen.py", "requirements.txt")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(family, seed, output, classifier):
    return Config(**FAMILIES[family], width=192, depth=6, heads=3, patch_size=4,
                  rank=8, n_basis=16, adapter_gain=1/math.sqrt(8), adapter_lr_multiplier=1.,
                  seed=seed, lr=4e-4, max_steps=30000, lr_schedule_steps=30000,
                  warmup_steps=50, batch_size=128, eval_every=5000, log_every=500,
                  cpu_threads=2, num_workers=2, adam_beta2=.999, fused_adamw=True,
                  compile_model=True, compile_mode="default", compile_dynamic=True,
                  class_dropout_prob=.1, guidance_scale=1., guidance_scales="1,1.5,2",
                  generative_eval_steps="5000,10000,20000", sampling_steps=50,
                  fid_images=5000, fid_batch_size=64, sample_images=100,
                  classifier_run=classifier, diagnostic_steps=DIAGNOSTICS, output_dir=str(output))


def prepare(output, launch=False):
    root = Path(__file__).resolve().parent
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free < 4*1024**3:
        raise RuntimeError("Require 4 GiB free for the declared 2 GiB artifact bound and shared-space headroom")
    classifier = str(root/"runs/dit-followup-20261007T174446Z/classifier/classifier_config")
    auxiliary = classifier_provenance(classifier)
    evaluator = Path(Config().fid_weights)
    output.mkdir(parents=True)
    runner = output/"runner"; runner.mkdir()
    for name in (*SOURCE_FILES, *EXTRA_SOURCE):
        shutil.copy2(root/name, runner/name)
    seeds_order = [(86,"dplr"), (86,"adaln"), (87,"adaln"), (87,"dplr")]
    ordered = []
    ordering = random.Random(188)
    for seed, architecture in seeds_order:
        pair = [(architecture+"_b09",seed), (architecture+"_b095",seed)]
        ordering.shuffle(pair); ordered.extend(pair)
    protocol = dict(state="prepared", created_utc=now(), repo_root=str(root),
                    purpose="Current-size optimizer contrast and same-state CFG; convergence before capacity changes",
                    source_sha256={p.name:digest(p) for p in runner.iterdir()},
                    families=FAMILIES, seeds=list(SEEDS), ordered_cases=ordered,
                    max_concurrent_workers=2, train_steps=30000, lr_schedule_steps=30000,
                    warmup_steps=50, base_lr=4e-4, adapter_lr_multiplier=1.,
                    intermediate_generation_steps=list(STEPS[:-1]), intermediate_guidance_scale=1.,
                    final_guidance_scales=list(SCALES), class_dropout_prob=.1,
                    train_examples=45000, validation_examples=5000, split_seed=123,
                    evaluation_seed=2027, sampling_seed=2028, sampling_steps=50, fid_images=5000,
                    classifier_run=classifier, auxiliary_classifier=auxiliary,
                    evaluator_cache=dict(path=str(evaluator), bytes=evaluator.stat().st_size, sha256=digest(evaluator)),
                    preflight="12 compiled BF16 plain-model steps, full real loader/holdout, intermediate and guided final generation",
                    estimated_gpu_hours="3–5 aggregate GPU-hours; approximately 1.5–2.5 hours at two GPUs, availability dependent",
                    disk_free_bytes_before_launch=free, expected_storage_upper_bound_bytes=2*1024**3,
                    artifact_purposes={"compact_evidence":"Frozen source/configs/splits/metrics/logs/audits/reports",
                                       "compiler_cache":"Reproducible compilation and bounded cold-start overhead; at most 1.5 GiB expected",
                                       "diagnostic_grids":"One 100-image final EMA CFG1 grid per quality run; one 10-image preflight grid"},
                    retention="Declared evidence, diagnostic grids and compiler caches until explicitly removed; existing evaluators unchanged",
                    weights_saved=False, periodic_checkpoints=False, test_evaluated=False,
                    ema="in-memory .999; intermediate and final evaluations before the process exits",
                    limits=["Two paired seeds on one held-out split; sample SD is not a confidence interval.",
                            "All recipes train with10% label dropout; this does not isolate dropout versus the historical no-dropout regime.",
                            "Beta1 changes use a fixed base and adapter LR; no update recalibration or quality-based rate selection.",
                            "Plain and DPLR use the same width/depth, not equal total parameters; capacity remains a confound.",
                            "Guidance scales are evaluated on the same EMA state/noise; scale1 uses one forward and other scales use two.",
                            "Intermediate quality uses CFG1 only; the final state evaluates all three scales.",
                            "FID is finite-sample 5k-image held-out-train FID, not published50k-image FID.",
                            "Classifier agreement is an auxiliary proxy; real accuracy and per-class confusion are retained.",
                            "Compilation is common to all recipes; throughput is concurrent and not an isolated speed experiment.",
                            "Thirty thousand steps are a convergence probe, not proof of saturation; larger models are deferred."])
    write_json(output/"protocol.json", protocol)
    configs = output/"planned_configs"; configs.mkdir()
    for family,seed in ordered:
        path = configs/f"{family}_s{seed}.json"
        cfg = make_config(family,seed,output/"quality"/path.stem,classifier)
        write_json(path,asdict(cfg)); load_config(["--override_json",str(path)])
    if launch:
        private_supervisor(output,[("optimizer_cfg_controller",[sys.executable,"-u",str(runner/Path(__file__).name),
                                                               "run","--output",str(output)])],
                           label="controller",autostart=True)
    return dict(output=str(output), launched=launch, jobs=len(ordered), max_concurrent_workers=2)


def execute_stage(output, protocol, name, cases, stage_examples=(45000, 5000)):
    """Finite workers; at most two submissions hold or wait for GPU claims."""
    stage = output/name; stage.mkdir()
    source = stage/"source"; shutil.copytree(output/"runner",source,ignore=shutil.ignore_patterns("__pycache__"))
    configs = stage/"configs"; configs.mkdir()
    jobs,programs = [],[]
    for family,seed,cfg in cases:
        case = f"{family}_s{seed}"
        service = f"optimizer_cfg_{output.name}_{name}_{case}".replace("-","_")
        path = configs/(case+".json"); write_json(path,asdict(cfg)); load_config(["--override_json",str(path)])
        command = ["/workspace/bin/gpu-claim","run","--owner","adalora","--job",service,"--wait","--",
                   "env", "TORCHINDUCTOR_CACHE_DIR="+str(output/"compile-cache"/name/case),
                   "TORCHINDUCTOR_COMPILE_THREADS=2",sys.executable,"-u",str(source/"train_dit_cifar10.py"),
                   "--override_json",str(path)]
        job = dict(name=case,family=family,config=str(path),output_dir=cfg.output_dir,service=service,command=command)
        jobs.append(job);programs.append((service,command))
    write_json(stage/"screen.json",dict(phase="optimizer_cfg",stage=name,created_utc=now(),jobs=jobs,
               evaluator_cache=protocol["evaluator_cache"],
               protocol=dict(train_examples=stage_examples[0],validation_examples=stage_examples[1],test_evaluated=False)))
    cap = protocol["max_concurrent_workers"]
    first = [j["service"] for j in jobs[:cap]]
    submitted = set(first); claims = {}; control = None
    try:
        control = private_supervisor(stage,programs,start_programs=first)
        while True:
            for path in Path("/tmp/vast_gpu_claims").glob("*.claim"):
                try: claim = json.loads(path.read_text())
                except (OSError, ValueError): continue
                if claim.get("job") in submitted:
                    claims[claim["job"]] = claim
            write_json(stage/"hardware_claims.json",claims)
            status = subprocess.run(["supervisorctl","-c",control,"status"],capture_output=True,text=True).stdout
            if not status.strip() or any(t in status for t in ("FATAL","BACKOFF")):
                raise RuntimeError(status or "No supervisor status")
            log = (stage/"supervisord.log").read_text()
            completed = {j["service"] for j in jobs if f"exited: {j['service']} (exit status 0; expected)" in log}
            for line in status.splitlines():
                service = line.split()[0]
                if service in submitted and "EXITED" in line and service not in completed:
                    raise RuntimeError("Failed worker; evidence retained: "+line)
            for job in jobs:
                if job["service"] in completed and not (Path(job["output_dir"])/"results.json").exists():
                    raise RuntimeError("Successful worker has no results")
            pending = [j for j in jobs if j["service"] not in submitted]
            for job in pending[:cap-len(submitted-completed)]:
                subprocess.run(["supervisorctl","-c",control,"start",job["service"]],check=True)
                submitted.add(job["service"])
            progress = dict(event="stage_progress",utc=now(),stage=name,completed=len(completed),
                            submitted=len(submitted),total=len(jobs),active=len(submitted-completed))
            assert progress["active"]<=cap
            with (output/"progress.jsonl").open("a") as stream:
                stream.write(json.dumps(progress)+"\n")
            print(json.dumps(progress),flush=True)
            if len(completed)==len(jobs): break
            time.sleep(20)
    finally:
        if control is not None:
            subprocess.run(["supervisorctl","-c",control,"shutdown"],check=True)
    report = summarize(stage/"screen.json")
    for row in report["rows"]:
        row["family"] = next(j["family"] for j in jobs if j["name"]==row["name"])
    write_json(stage/"summary.json",report)
    return report


def run(output):
    output = Path(output).resolve()
    protocol = json.loads((output/"protocol.json").read_text())
    if protocol["state"]!="prepared":
        raise ValueError("Only a fresh prepared workflow may launch; no implicit resume")
    protocol.update(state="preflight",started_utc=now()); write_json(output/"protocol.json",protocol)
    try:
        family = "adaln_b095";seed = SEEDS[0]
        cfg = replace(make_config(family,seed,output/"preflight"/f"{family}_s{seed}",protocol["classifier_run"]),
                      max_steps=12,warmup_steps=2,eval_every=6,log_every=6,
                      diagnostic_steps="0,1,3,5,10,12",generative_eval_steps="3,6",
                      sampling_steps=2,fid_images=20,fid_batch_size=10,sample_images=10)
        preflight = execute_stage(output,protocol,"preflight",[(family,seed,cfg)])
        preflight_row = preflight["rows"][0]
        assert preflight_row["training_setup"]["dropped_training_labels"]>0
        assert [r["step"] for r in preflight_row["generative_validation_curve"]]==[3,6,12]
        assert set(preflight_row["guidance_evaluations"])=={"1.0","1.5","2.0"}
        assert preflight_row["generative_evaluation"]["class_agreement"]["real"]["examples"]==5000
        protocol.update(state="training",preflight_complete_utc=now()); write_json(output/"protocol.json",protocol)
        cases = [(family,seed,make_config(family,seed,output/"quality"/f"{family}_s{seed}",protocol["classifier_run"]))
                 for family,seed in protocol["ordered_cases"]]
        quality = execute_stage(output,protocol,"quality",cases)
        write_json(output/"report.json",dict(protocol=protocol,preflight=preflight,quality=quality))
        protocol.update(state="complete",completed_utc=now()); write_json(output/"protocol.json",protocol)
        audited = audit(output)
        write_summary(output,quality["rows"],audited,protocol)
        print(json.dumps(dict(event="complete",output=str(output),audit_passed=True)),flush=True)
    except BaseException as error:
        protocol.update(state="failed",failed_utc=now(),failure_type=type(error).__name__,failure=str(error))
        write_json(output/"protocol.json",protocol)
        raise


def mean_sd(values):
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else 0.,values=values)


def aggregate(rows):
    keyed = {(r["family"],r["config"]["seed"]):r for r in rows}
    if set(keyed)!={(family,seed) for family in FAMILIES for seed in SEEDS} or len(rows)!=8:
        raise ValueError("Exactly the declared four recipes and two paired seeds are required")
    groups = []
    for family in FAMILIES:
        paired = [keyed[family,seed] for seed in SEEDS]
        guided = {}
        for scale in SCALES:
            points = [r["guidance_evaluations"][str(scale)] for r in paired]
            guided[str(scale)] = dict(fid=mean_sd([r["fid"] for r in points]),
                                      agreement_percent=mean_sd([100*r["class_agreement"]["generated"]["agreement"] for r in points]))
        curve = {str(step): dict(fid=mean_sd([next(r["fid"] for r in row["generative_validation_curve"] if r["step"]==step) for row in paired]),
                                ema_noise_mse=mean_sd([next(r["ema"]["noise_mse"] for r in row["validation_curve"] if r["step"]==step) for row in paired]))
                 for step in STEPS}
        groups.append(dict(family=family,parameters=paired[0]["parameters"],adapter_parameters=paired[0]["adapter_parameters"],
                           final_guidance=guided,convergence=curve,ema_noise_mse=mean_sd([r["ema_validation"]["noise_mse"] for r in paired])))
    contrasts = {}
    for architecture in ("adaln","dplr"):
        contrasts[architecture] = {}
        for scale in SCALES:
            a = [keyed[architecture+"_b09",s]["guidance_evaluations"][str(scale)] for s in SEEDS]
            b = [keyed[architecture+"_b095",s]["guidance_evaluations"][str(scale)] for s in SEEDS]
            contrasts[architecture][str(scale)] = dict(fid_b095_minus_b09=mean_sd([y["fid"]-x["fid"] for x,y in zip(a,b)]),
                 agreement_pp_b095_minus_b09=mean_sd([100*(y["class_agreement"]["generated"]["agreement"]-x["class_agreement"]["generated"]["agreement"]) for x,y in zip(a,b)]))
    return dict(aggregates=groups,paired_beta_differences=contrasts)


def same_stage_evidence(replayed, recorded):
    """Ignore only a live disk tally, which grows when the summary is saved."""
    def stable(value):
        return {**value, "audit": {k:v for k,v in value["audit"].items()
                                  if k != "bytes_retained_under_screen"}}
    return stable(replayed) == stable(recorded)


def audit(output, postflight_recovery=False):
    """Replay compact evidence and verify paired training/evaluation controls."""
    import torch
    from benchmark_dit_design import digest_parameters
    from train_dit_cifar10 import learning_rate_scale, make_model
    torch.set_num_threads(2)
    output = Path(output).resolve();protocol = json.loads((output/"protocol.json").read_text())
    recoverable = (postflight_recovery and protocol["state"]=="failed"
                   and protocol.get("completed_utc") and protocol.get("failure_type")=="AssertionError")
    assert (protocol["state"]=="complete" or recoverable) and protocol["max_concurrent_workers"]==2
    assert protocol["seeds"]==list(SEEDS) and protocol["families"]==FAMILIES
    report = json.loads((output/"report.json").read_text())
    summary = summarize(output/"quality/screen.json")
    for row in summary["rows"]:
        row["family"] = next(j["family"] for j in json.loads((output/"quality/screen.json").read_text())["jobs"] if j["name"]==row["name"])
    assert same_stage_evidence(summary, report["quality"])
    aggregate(summary["rows"])
    for name,expected in protocol["source_sha256"].items():
        assert digest(output/"runner"/name)==expected
        for stage in ("preflight","quality"):
            assert digest(output/stage/"source"/name)==expected
    assert classifier_provenance(protocol["classifier_run"])==protocol["auxiliary_classifier"]
    for stage in ("preflight","quality"):
        assert not (output/stage/"supervisor.sock").exists()
        claims = json.loads((output/stage/"hardware_claims.json").read_text())
        screen = json.loads((output/stage/"screen.json").read_text())
        assert set(claims)=={j["service"] for j in screen["jobs"]}
        for job in screen["jobs"]:
            runtime=json.loads((Path(job["output_dir"])/"provenance.json").read_text());claim=claims[job["service"]]
            assert claim["owner"]=="adalora" and claim["uuid"].removeprefix("GPU-")==runtime["device_uuid"]
    real_metrics = []
    for seed in SEEDS:
        rows = [r for r in summary["rows"] if r["config"]["seed"]==seed]
        base_hashes, traces, noise_states, label_states, dropped = [],[],[],[],[]
        adapter_hashes = []
        for row in rows:
            cfg = Config(**row["config"])
            assert row["config"]==asdict(make_config(row["family"],seed,Path(cfg.output_dir),protocol["classifier_run"]))
            torch.manual_seed(seed);model = make_model(cfg);stats = model.statistics()
            for key in ("parameters","adapter_parameters","forward_matmul_flops_per_image_estimate"):
                assert row[key]==stats[key]
            diag = row["design_diagnostics"]
            assert diag["base_initialization_sha256"]==digest_parameters(model)
            assert diag["adapter_initialization_sha256"]==digest_parameters(model,True)
            base_hashes.append(diag["base_initialization_sha256"])
            if cfg.variant=="dplr": adapter_hashes.append(diag["adapter_initialization_sha256"])
            events = diag["diagnostics"]
            assert [r["step"] for r in events]==[int(v) for v in DIAGNOSTICS.split(",")]
            assert all(b["gate_rms"]==b["delta_rms"]==0 for b in events[0]["branches"])
            assert all(v<=10 for v in diag["adapter_first_nonzero_gradient_observed"].values())
            keys = ("step","batch_images_sha256","batch_labels_sha256","noise_sha256","timesteps_sha256","noise_rng_state_sha256")
            traces.append([{k:e[k] for k in keys} for e in events[1:]])
            noise_states.append(diag["noise_rng_final_state_sha256"])
            setup = row["training_setup"];dropped.append(setup["dropped_training_labels"])
            label_states.append(setup["label_dropout_rng_final_state_sha256"])
            assert setup["adam_betas"]==[cfg.adam_beta1,.999] and setup["class_dropout_prob"]==.1
            assert setup["fused_adamw"] and setup["model_compiled"] and setup["compile_dynamic"]
            assert setup["label_dropout_rng_seed"]==seed+30000 and 0<dropped[-1]<row["processed_images"]
            logs = [json.loads(s) for s in (Path(cfg.output_dir)/"metrics.jsonl").read_text().splitlines()]
            assert events==[e for e in logs if e["event"]=="design_diagnostic"]
            assert all(math.isclose(e["base_lr"],cfg.lr*learning_rate_scale(cfg,e["step"]),rel_tol=1e-14)
                       for e in logs if e["event"]=="train")
            generation = [e for e in logs if e["event"]=="generative_evaluation"]
            assert [e["step"] for e in generation]==list(STEPS)
            assert row["generative_validation_curve"]==[{k:v for k,v in e.items() if k!="event"} for e in generation]
            assert set(row["guidance_evaluations"])=={"1.0","1.5","2.0"}
            assert row["guidance_evaluations"]["1.0"]==row["generative_evaluation"]
            assert [e["step"] for e in row["generative_validation_curve"]]==list(STEPS)
            for point in [*row["generative_validation_curve"],*row["guidance_evaluations"].values()]:
                assert point["model_state"]=="EMA" and point["sampling_seed"]==2028 and point["sampling_steps"]==50
                assert point["generated_images"]==point["fid_fake_images"]==point["fid_real_images"]==5000
                assert point["class_dropout_prob"]==.1 and point["eta"]==0 and point["balanced_requested_class_counts"]==[500]*10
                assert point["class_agreement"]["classifier"]==protocol["auxiliary_classifier"]
                for state in ("real","generated"):
                    metric = point["class_agreement"][state];matrix = metric["confusion_rows_requested_columns_predicted"]
                    assert metric["examples"]==5000 and metric["class_counts"]==[500]*10 and [sum(r) for r in matrix]==[500]*10
                    assert math.isclose(metric["agreement"],sum(matrix[i][i] for i in range(10))/5000,abs_tol=1e-7)
                real_metrics.append(point["class_agreement"]["real"])
            assert all(not p["sample_grid_labels"] for p in row["generative_validation_curve"][:-1])
            assert row["generative_evaluation"]["sample_grid_labels"]==[i%10 for i in range(100)]
            assert all(not row["guidance_evaluations"][str(s)]["sample_grid_labels"] for s in (1.5,2.))
        assert len(set(base_hashes))==len(set(noise_states))==len(set(label_states))==len(set(dropped))==1
        assert len(set(adapter_hashes))==1 and all(trace==traces[0] for trace in traces)
    assert all(r==real_metrics[0] for r in real_metrics)
    progress = [json.loads(s) for s in (output/"progress.jsonl").read_text().splitlines()]
    assert progress and all(r["active"]==r["submitted"]-r["completed"]<=2 for r in progress)
    assert not [p for p in (output/"quality").rglob("*") if p.suffix in (".pt",".pth",".ckpt",".safetensors")]
    retained = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    assert retained<protocol["expected_storage_upper_bound_bytes"]
    checked = ("eight_completed_runs","successful_exits_and_closed_worker_supervisors","frozen_source_and_evaluators",
               "full_loader_common_recipe","paired_initialization_and_noise_label_traces","zero_start_gradient_wakeup",
               "same_state_guidance_and_denominators","intermediate_generation_curve","claimed_hardware_matches_runtime",
               "two_gpu_cap","no_checkpoints_or_test_evaluation","storage_within_bound")
    result = dict(checks={key:True for key in checked},bytes_retained=retained,report_sha256=digest(output/"report.json"),
                  classifier_real_accuracy=real_metrics[0]["agreement"])
    write_json(output/"audit.json",result)
    return result


def write_summary(output, rows, audited, protocol):
    comparison = aggregate(rows);check_finite(comparison)
    result = dict(raw_workflow=str(output),**comparison,audit=audited,limits=protocol["limits"],
                  gpu_hours_loops=sum(r["loop_seconds"] for r in rows)/3600,
                  gpu_hours_intermediate_generation=sum(r["intermediate_generation_seconds"] for r in rows)/3600,
                  gpu_hours_final_generation=sum(sum(p["seconds"] for p in r["guidance_evaluations"].values()) for r in rows)/3600)
    write_json(output/"comparison.json",result)
    lines = ["# Current-size beta1 and CFG comparison", "",
             "Eight 30k-step runs: four recipes and paired seeds86/87. Values are mean ± sample SD, not confidence intervals.", "",
             "| Recipe | Parameters | EMA MSE | CFG1 FID | CFG1.5 FID | CFG2 FID |", "|---|---:|---:|---:|---:|---:|"]
    def cell(m,d=2): return f"{m['mean']:.{d}f} ± {m['sd']:.{d}f}"
    for row in comparison["aggregates"]:
        lines.append(f"| {row['family']} | {row['parameters']:,} | {cell(row['ema_noise_mse'],5)} | "+
                     " | ".join(cell(row['final_guidance'][str(s)]['fid']) for s in SCALES)+" |")
    lines += ["", "| Recipe | CFG1 agreement | CFG1.5 agreement | CFG2 agreement |", "|---|---:|---:|---:|"]
    for row in comparison["aggregates"]:
        lines.append("| "+row['family']+" | "+" | ".join(cell(row['final_guidance'][str(s)]['agreement_percent'])+"%" for s in SCALES)+" |")
    lines += ["", "Paired beta differences below are beta1=.95 minus.9 in seed order86/87; negative FID and positive agreement differences favor.95.", "",
              "| Architecture | CFG | FID differences | Agreement differences (pp) |", "|---|---:|---|---|"]
    for architecture,values in comparison["paired_beta_differences"].items():
        for scale,point in values.items():
            lines.append(f"| {architecture} | {scale} | "+", ".join(f"{v:+.2f}" for v in point['fid_b095_minus_b09']['values'])+
                         " | "+", ".join(f"{v:+.2f}" for v in point['agreement_pp_b095_minus_b09']['values'])+" |")
    lines += ["", "![Convergence](convergence.png)", "", "![Guidance](guidance.png)", "",
              "All recipes use width192/depth6/heads3/patch4, LR4e-4, warmup50, 30k cosine schedule, batch128,10% label dropout, BF16, compiled training, fused AdamW, EMA.999 and50-step deterministic DDIM. DPLR is independent affine r8/s16 with gain1/sqrt(8) and adapter LR multiplier1. Intermediate generation is CFG1 only; final scales share a state/noise seed. Model weights remain in memory only.", "",
              "Limits: "+" ".join(protocol['limits']), "", "[Structured comparison](comparison.json) · [Full evidence](report.json) · [Audit](audit.json) · [Protocol](protocol.json)", ""]
    (output/"summary.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(adaln_b09="#667085",adaln_b095="#344054",dplr_b09="#168aad",dplr_b095="#d05b26")
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for row in comparison["aggregates"]:
        for axis,key in zip(axes,("fid","ema_noise_mse")):
            values = [row['convergence'][str(s)][key] for s in STEPS]
            axis.errorbar(STEPS,[v['mean'] for v in values],yerr=[v['sd'] for v in values],marker='o',capsize=3,
                          label=row['family'],color=colors[row['family']])
            axis.set_xlabel('Training step');axis.grid(alpha=.15)
    axes[0].set_ylabel('EMA CFG1 FID');axes[1].set_ylabel('EMA epsilon MSE');axes[0].legend(fontsize=8)
    for suffix in ('png','svg'):fig.savefig(output/f'convergence.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for row in comparison['aggregates']:
        for axis,key in zip(axes,('fid','agreement_percent')):
            values = [row['final_guidance'][str(s)][key] for s in SCALES]
            axis.errorbar(SCALES,[v['mean'] for v in values],yerr=[v['sd'] for v in values],marker='o',capsize=3,
                          label=row['family'],color=colors[row['family']])
            axis.set_xlabel('CFG scale');axis.set_xticks(SCALES);axis.grid(alpha=.15)
    axes[0].set_ylabel('Final EMA FID');axes[1].set_ylabel('Requested-class agreement (%)');axes[0].legend(fontsize=8)
    for suffix in ('png','svg'):fig.savefig(output/f'guidance.{suffix}',dpi=160)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=("prepare","run","audit"))
    parser.add_argument("--output",type=Path);parser.add_argument("--launch",action="store_true");args=parser.parse_args()
    output = args.output or Path("runs")/("dit-optimizer-cfg-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if args.action=="prepare":print(json.dumps(prepare(output,args.launch)),flush=True)
    elif args.action=="run":run(output)
    else:print(json.dumps(audit(output),indent=2))

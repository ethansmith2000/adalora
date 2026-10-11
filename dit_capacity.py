#!/usr/bin/env python3
"""Two-GPU capacity controls: larger AdaLN, wider static MLP and DPLR."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import sys

from class_fidelity import classifier_provenance
from dit_followup import private_supervisor
from dit_optimizer_cfg import (DIAGNOSTICS, STEPS, digest, execute_stage, mean_sd,
                               now, same_stage_evidence)
from summarize_dit_cifar10_screen import check_finite, summarize
from train_dit_cifar10 import Config, SOURCE_FILES, load_config, write_json

SEEDS = (86, 87)
SCALES = (1., 2.)
FAMILIES = {
    "plain": dict(variant="adaln_zero", mlp_ratio=4.),
    "plain_ffn_budget": dict(variant="adaln_zero", mlp_ratio=1360/256),
    "dplr": dict(variant="dplr", mlp_ratio=4.),
}
PARAMETERS = dict(plain=9758768, plain_ffn_budget=11137712, dplr=11135280)
REFERENCE = "runs/dit-optimizer-cfg-20261009T224730Z"
EXTRA_SOURCE = ("dit_capacity.py", "test_dit_capacity.py", "dit_optimizer_cfg.py",
                "test_dit_optimizer_cfg.py", "test_dit_training_options.py", "dit_quality_diagnostics.py",
                "benchmark_dit_design.py", "dit_followup.py", "summarize_dit_cifar10_screen.py", "requirements.txt")


def make_config(family, seed, output, classifier):
    return Config(**FAMILIES[family], width=256, depth=8, heads=4, patch_size=4,
                  rank=8, n_basis=16, adapter_gain=1/math.sqrt(8), adapter_lr_multiplier=1.,
                  seed=seed, lr=4e-4, max_steps=30000, lr_schedule_steps=30000, warmup_steps=50,
                  batch_size=128, eval_every=5000, log_every=500, cpu_threads=2, num_workers=2,
                  adam_beta1=.9, adam_beta2=.999, fused_adamw=True, compile_model=True,
                  compile_mode="default", compile_dynamic=True, class_dropout_prob=.1,
                  guidance_scale=1., guidance_scales="1,2", generative_eval_steps="5000,10000,20000",
                  sampling_steps=50, fid_images=5000, fid_batch_size=64, sample_images=100,
                  classifier_run=classifier, diagnostic_steps=DIAGNOSTICS, output_dir=str(output))


def prepare(output, launch=False):
    root = Path(__file__).resolve().parent;output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free<4*1024**3: raise RuntimeError("Require 4 GiB free for the declared 2 GiB bound and headroom")
    reference = root/REFERENCE
    assert json.loads((reference/"protocol.json").read_text())["state"]=="complete"
    reference_report = json.loads((reference/"report.json").read_text())
    classifier = str(root/"runs/dit-followup-20261007T174446Z/classifier/classifier_config")
    auxiliary = classifier_provenance(classifier);evaluator = Path(Config().fid_weights)
    # Old anchors remain independent frozen evidence; training implementation must match.
    for name,sha in reference_report["quality"]["source_sha256"].items():
        assert digest(root/name)==sha
    output.mkdir(parents=True);runner=output/"runner";runner.mkdir()
    for name in (*SOURCE_FILES,*EXTRA_SOURCE):shutil.copy2(root/name,runner/name)
    ordering=random.Random(189);ordered=[]
    for seed in SEEDS:
        families=list(FAMILIES);ordering.shuffle(families);ordered.extend((family,seed) for family in families)
    protocol=dict(state="prepared",created_utc=now(),purpose="Larger-model conditional-transform and static-capacity controls",
                  repo_root=str(root),families=FAMILIES,parameters=PARAMETERS,seeds=list(SEEDS),ordered_cases=ordered,
                  source_sha256={p.name:digest(p) for p in runner.iterdir()},max_concurrent_workers=2,
                  train_steps=30000,width=256,depth=8,heads=4,train_examples=45000,validation_examples=5000,
                  lr=4e-4,warmup_steps=50,lr_schedule_steps=30000,adapter_lr_multiplier=1.,class_dropout_prob=.1,
                  intermediate_generation_steps=list(STEPS[:-1]),intermediate_guidance_scale=1.,
                  final_guidance_scales=list(SCALES),sampling_seed=2028,evaluation_seed=2027,split_seed=123,
                  sampling_steps=50,fid_images=5000,classifier_run=classifier,auxiliary_classifier=auxiliary,
                  evaluator_cache=dict(path=str(evaluator),bytes=evaluator.stat().st_size,sha256=digest(evaluator)),
                  reference_workflow=str(reference),reference_report_sha256=digest(reference/"report.json"),
                  reference_audit_sha256=digest(reference/"audit.json"),
                  preflight="DPLR and wider-FFN 12-step compiled BF16 checks; 328 train examples produce128/128/72 batches; full5k holdout/classifier",
                  estimated_gpu_hours="4–6 aggregate GPU-hours; approximately2–3 hours at two GPUs, availability dependent",
                  disk_free_bytes_before_launch=free,expected_storage_upper_bound_bytes=2*1024**3,
                  artifact_purposes=dict(compact_evidence="Frozen sources/configs/splits/metrics/logs/audits/reports",
                                        compiler_cache="Compatibility and compilation evidence; expected at most1.5GiB",
                                        diagnostic_grids="One100-image final EMA CFG1 grid per quality run; one10-image grid per preflight"),
                  retention="Declared compact evidence, compiler caches and diagnostic grids until explicitly removed; existing evaluators unchanged",
                  weights_saved=False,periodic_checkpoints=False,test_evaluated=False,ema="In-memory .999; evaluations before process exit",
                  limits=["Two paired seeds and one held-out split; sample SD is not a confidence interval.",
                          "Wider plain MLP matches DPLR parameters within0.022%, not FLOPs, activation shapes or identical initialization.",
                          "Plain ratio4 and DPLR share the same base initialization; all three share observed data/noise/effective-label traces.",
                          "LR and adapter gain/rate are held fixed, not recalibrated or optimized for the larger models.",
                          "Earlier small-model anchors use the same seeds and training implementation but were run in a separate cohort.",
                          "Intermediate quality uses CFG1; final scales1/2 share one EMA state and initial-noise seed.",
                          "5k-image held-out-train FID is not published50k-image FID; classifier agreement is an auxiliary proxy.",
                          "Compiled concurrent training is not an isolated speed benchmark.",
                          "Thirty thousand steps do not establish convergence or saturation."])
    write_json(output/"protocol.json",protocol);configs=output/"planned_configs";configs.mkdir()
    for family,seed in ordered:
        path=configs/f"{family}_s{seed}.json"
        write_json(path,asdict(make_config(family,seed,output/"quality"/path.stem,classifier)))
        load_config(["--override_json",str(path)])
    if launch:
        private_supervisor(output,[("capacity_controller",[sys.executable,"-u",str(runner/Path(__file__).name),
                                                           "run","--output",str(output)])],label="controller",autostart=True)
    return dict(output=str(output),launched=launch,jobs=6,max_concurrent_workers=2)


def aggregate(rows):
    keyed={(r["family"],r["config"]["seed"]):r for r in rows}
    if len(rows)!=6 or set(keyed)!={(f,s) for f in FAMILIES for s in SEEDS}:
        raise ValueError("Require the exact three-family/two-seed capacity grid")
    groups=[]
    for family in FAMILIES:
        paired=[keyed[family,s] for s in SEEDS]
        guidance={str(scale):dict(fid=mean_sd([r["guidance_evaluations"][str(scale)]["fid"] for r in paired]),
                                  agreement_percent=mean_sd([100*r["guidance_evaluations"][str(scale)]["class_agreement"]["generated"]["agreement"] for r in paired])) for scale in SCALES}
        curve={str(step):dict(fid=mean_sd([next(p["fid"] for p in r["generative_validation_curve"] if p["step"]==step) for r in paired]),
                             ema_noise_mse=mean_sd([next(p["ema"]["noise_mse"] for p in r["validation_curve"] if p["step"]==step) for r in paired])) for step in STEPS}
        groups.append(dict(family=family,parameters=paired[0]["parameters"],adapter_parameters=paired[0]["adapter_parameters"],
                           forward_flops=paired[0]["forward_matmul_flops_per_image_estimate"],final_guidance=guidance,convergence=curve,
                           peak_allocated_mib=mean_sd([r["peak_allocated_bytes"]/1024**2 for r in paired])))
    paired_differences={}
    for control in ("plain","plain_ffn_budget"):
        paired_differences[control]={}
        for scale in SCALES:
            candidates=[keyed["dplr",s]["guidance_evaluations"][str(scale)] for s in SEEDS]
            controls=[keyed[control,s]["guidance_evaluations"][str(scale)] for s in SEEDS]
            paired_differences[control][str(scale)]=dict(fid_dplr_minus_control=mean_sd([a["fid"]-b["fid"] for a,b in zip(candidates,controls)]),
                 agreement_pp_dplr_minus_control=mean_sd([100*(a["class_agreement"]["generated"]["agreement"]-b["class_agreement"]["generated"]["agreement"]) for a,b in zip(candidates,controls)]))
    return dict(aggregates=groups,paired_dplr_differences=paired_differences)


def run(output):
    output=Path(output).resolve();protocol=json.loads((output/"protocol.json").read_text())
    if protocol["state"]!="prepared":raise ValueError("Fresh workflows only; no implicit resume")
    protocol.update(state="preflight",started_utc=now());write_json(output/"protocol.json",protocol)
    try:
        cases=[]
        for family in ("dplr","plain_ffn_budget"):
            cfg=replace(make_config(family,SEEDS[0],output/"preflight"/f"{family}_s{SEEDS[0]}",protocol["classifier_run"]),
                        max_steps=12,warmup_steps=2,eval_every=6,log_every=6,train_limit=328,
                        diagnostic_steps="0,1,3,5,10,12",generative_eval_steps="3,6",
                        sampling_steps=2,fid_images=20,fid_batch_size=10,sample_images=10)
            cases.append((family,SEEDS[0],cfg))
        preflight=execute_stage(output,protocol,"preflight",cases,stage_examples=(328,5000))
        for row in preflight["rows"]:
            assert row["processed_images"]==1312 and row["parameters"]==PARAMETERS[row["family"]]
            assert row["training_setup"]["dropped_training_labels"]>0 and set(row["guidance_evaluations"])=={"1.0","2.0"}
            assert [p["step"] for p in row["generative_validation_curve"]]==[3,6,12]
            assert row["generative_evaluation"]["class_agreement"]["real"]["examples"]==5000
        protocol.update(state="training",preflight_complete_utc=now());write_json(output/"protocol.json",protocol)
        cases=[(family,seed,make_config(family,seed,output/"quality"/f"{family}_s{seed}",protocol["classifier_run"]))
               for family,seed in protocol["ordered_cases"]]
        quality=execute_stage(output,protocol,"quality",cases)
        write_json(output/"report.json",dict(preflight=preflight,quality=quality))
        protocol.update(state="complete",completed_utc=now());write_json(output/"protocol.json",protocol)
        audited=audit(output);write_summary(output,quality["rows"],audited,protocol)
        print(json.dumps(dict(event="complete",output=str(output),audit_passed=True)),flush=True)
    except BaseException as error:
        protocol.update(state="failed",failed_utc=now(),failure_type=type(error).__name__,failure=str(error))
        write_json(output/"protocol.json",protocol);raise


def audit(output):
    import torch
    from benchmark_dit_design import digest_parameters
    from train_dit_cifar10 import make_model,learning_rate_scale
    torch.set_num_threads(2);output=Path(output).resolve();p=json.loads((output/"protocol.json").read_text())
    assert p["state"]=="complete" and p["max_concurrent_workers"]==2 and p["families"]==FAMILIES and p["seeds"]==list(SEEDS)
    report=json.loads((output/"report.json").read_text());replay=summarize(output/"quality/screen.json")
    screen=json.loads((output/"quality/screen.json").read_text())
    for row in replay["rows"]:row["family"]=next(j["family"] for j in screen["jobs"] if j["name"]==row["name"])
    assert same_stage_evidence(replay,report["quality"]);aggregate(replay["rows"])
    assert abs(PARAMETERS["plain_ffn_budget"]/PARAMETERS["dplr"]-1)<.00022
    reference=Path(p["reference_workflow"])
    assert digest(reference/"report.json")==p["reference_report_sha256"] and digest(reference/"audit.json")==p["reference_audit_sha256"]
    previous=json.loads((reference/"report.json").read_text())["quality"]
    assert replay["source_sha256"]==previous["source_sha256"] and replay["runtime"]==previous["runtime"]
    assert classifier_provenance(p["classifier_run"])==p["auxiliary_classifier"]
    for name,sha in p["source_sha256"].items():
        assert digest(output/"runner"/name)==sha
        for stage in ("preflight","quality"):assert digest(output/stage/"source"/name)==sha
    for stage in ("preflight","quality"):
        assert not (output/stage/"supervisor.sock").exists()
        claims=json.loads((output/stage/"hardware_claims.json").read_text());jobs=json.loads((output/stage/"screen.json").read_text())["jobs"]
        assert set(claims)=={j["service"] for j in jobs}
        for job in jobs:
            runtime=json.loads((Path(job["output_dir"])/"provenance.json").read_text());claim=claims[job["service"]]
            assert claim["owner"]=="adalora" and claim["uuid"].removeprefix("GPU-")==runtime["device_uuid"]
    real=[]
    for seed in SEEDS:
        rows=[r for r in replay["rows"] if r["config"]["seed"]==seed];traces=[];noise=[];labels=[];dropped=[];base={}
        def trace(row):
            keys=("step","batch_images_sha256","batch_labels_sha256","noise_sha256","timesteps_sha256","noise_rng_state_sha256")
            return [{k:e[k] for k in keys} for e in row["design_diagnostics"]["diagnostics"][1:]]
        for row in rows:
            cfg=Config(**row["config"]);assert row["config"]==asdict(make_config(row["family"],seed,Path(cfg.output_dir),p["classifier_run"]))
            torch.manual_seed(seed);model=make_model(cfg);stats=model.statistics();assert stats["parameters"]==PARAMETERS[row["family"]]
            for key in ("parameters","adapter_parameters","forward_matmul_flops_per_image_estimate"):assert row[key]==stats[key]
            diag=row["design_diagnostics"];assert diag["base_initialization_sha256"]==digest_parameters(model)
            assert diag["adapter_initialization_sha256"]==digest_parameters(model,True)
            base[row["family"]]=diag["base_initialization_sha256"];events=diag["diagnostics"]
            assert [e["step"] for e in events]==[int(v) for v in DIAGNOSTICS.split(",")]
            assert all(b["gate_rms"]==b["delta_rms"]==0 for b in events[0]["branches"])
            assert all(v<=10 for v in diag["adapter_first_nonzero_gradient_observed"].values())
            traces.append(trace(row));noise.append(diag["noise_rng_final_state_sha256"])
            setup=row["training_setup"];assert setup["adam_betas"]==[.9,.999] and setup["fused_adamw"] and setup["model_compiled"]
            assert setup["class_dropout_prob"]==.1 and setup["label_dropout_rng_seed"]==seed+30000
            labels.append(setup["label_dropout_rng_final_state_sha256"]);dropped.append(setup["dropped_training_labels"])
            logs=[json.loads(s) for s in (Path(cfg.output_dir)/"metrics.jsonl").read_text().splitlines()]
            assert events==[e for e in logs if e["event"]=="design_diagnostic"]
            assert all(math.isclose(e["base_lr"],cfg.lr*learning_rate_scale(cfg,e["step"]),rel_tol=1e-14) for e in logs if e["event"]=="train")
            generation=[e for e in logs if e["event"]=="generative_evaluation"]
            assert [e["step"] for e in generation]==list(STEPS)
            assert row["generative_validation_curve"]==[{k:v for k,v in e.items() if k!="event"} for e in generation]
            assert set(row["guidance_evaluations"])=={"1.0","2.0"} and row["guidance_evaluations"]["1.0"]==row["generative_evaluation"]
            for point in [*row["generative_validation_curve"],*row["guidance_evaluations"].values()]:
                assert point["model_state"]=="EMA" and point["sampling_seed"]==2028 and point["sampling_steps"]==50 and point["eta"]==0
                assert point["generated_images"]==point["fid_fake_images"]==point["fid_real_images"]==5000
                assert point["class_agreement"]["classifier"]==p["auxiliary_classifier"]
                for state in ("real","generated"):
                    metric=point["class_agreement"][state];matrix=metric["confusion_rows_requested_columns_predicted"]
                    assert metric["examples"]==5000 and metric["class_counts"]==[500]*10 and [sum(v) for v in matrix]==[500]*10
                    assert math.isclose(metric["agreement"],sum(matrix[i][i] for i in range(10))/5000,abs_tol=1e-7)
                real.append(point["class_agreement"]["real"])
            assert all(not e["sample_grid_labels"] for e in row["generative_validation_curve"][:-1])
            assert row["generative_evaluation"]["sample_grid_labels"]==[i%10 for i in range(100)] and not row["guidance_evaluations"]["2.0"]["sample_grid_labels"]
        assert base["plain"]==base["dplr"] and all(t==traces[0] for t in traces)
        assert len(set(noise))==len(set(labels))==len(set(dropped))==1 and 0<dropped[0]<rows[0]["processed_images"]
        anchors=[r for r in previous["rows"] if r["config"]["seed"]==seed and r["family"] in ("adaln_b09","dplr_b09")]
        assert len(anchors)==2 and all(trace(a)==traces[0] for a in anchors)
        assert all(a["design_diagnostics"]["noise_rng_final_state_sha256"]==noise[0] and a["training_setup"]["label_dropout_rng_final_state_sha256"]==labels[0] for a in anchors)
    assert all(v==real[0] for v in real)
    progress=[json.loads(s) for s in (output/"progress.jsonl").read_text().splitlines()];assert all(e["active"]==e["submitted"]-e["completed"]<=2 for e in progress)
    assert not [f for stage in ("preflight","quality") for f in (output/stage).rglob("*") if f.suffix in (".pt",".pth",".ckpt",".safetensors")]
    retained=sum(f.stat().st_size for f in output.rglob("*") if f.is_file());assert retained<p["expected_storage_upper_bound_bytes"]
    checks=("six_completed_runs","successful_exits_closed_worker_supervisors","frozen_sources_evaluators","same_training_implementation_as_anchors",
            "parameter_budget_matched","plain_dplr_equal_base_init","paired_data_noise_labels_including_anchors","zero_start_gradient_wakeup",
            "correct_curves_and_guidance_denominators","claimed_gpu_matches_runtime","two_gpu_cap","no_weights_test_evaluation","storage_within_bound")
    audited=dict(checks={c:True for c in checks},bytes_retained=retained,report_sha256=digest(output/"report.json"))
    write_json(output/"audit.json",audited);return audited


def write_summary(output,rows,audited,protocol):
    comparison=aggregate(rows);check_finite(comparison)
    anchors=json.loads((Path(protocol["reference_workflow"])/"report.json").read_text())["quality"]["rows"]
    anchors=[r for r in anchors if r["family"] in ("adaln_b09","dplr_b09")]
    result=dict(raw_workflow=str(output),**comparison,audit=audited,limits=protocol["limits"],
                small_model_anchors=[dict(family=r["family"],seed=r["config"]["seed"],parameters=r["parameters"],guidance={str(s):r["guidance_evaluations"][str(s)] for s in SCALES}) for r in anchors])
    write_json(output/"comparison.json",result)
    def cell(m):return f"{m['mean']:.2f} ± {m['sd']:.2f}"
    lines=["# Larger-model capacity controls","","Six30k-step runs, paired seeds86/87; mean ± sample SD, not confidence intervals.","",
           "| Model | Parameters | CFG1 FID | CFG2 FID | CFG2 agreement | Train peak allocated MiB |","|---|---:|---:|---:|---:|---:|"]
    for r in comparison["aggregates"]:
        lines.append(f"| {r['family']} | {r['parameters']:,} | {cell(r['final_guidance']['1.0']['fid'])} | {cell(r['final_guidance']['2.0']['fid'])} | {cell(r['final_guidance']['2.0']['agreement_percent'])}% | {cell(r['peak_allocated_mib'])} |")
    lines += ["","DPLR-minus-control differences below preserve seed order86/87. Negative FID and positive agreement differences favor DPLR.","",
              "| Control | CFG | FID differences | Agreement differences (pp) |","|---|---:|---|---|"]
    for control,scales in comparison["paired_dplr_differences"].items():
        for scale,p in scales.items():lines.append(f"| {control} | {scale} | "+", ".join(f"{v:+.2f}" for v in p['fid_dplr_minus_control']['values'])+" | "+", ".join(f"{v:+.2f}" for v in p['agreement_pp_dplr_minus_control']['values'])+" |")
    lines += ["","![Convergence](convergence.png)","","![Guidance](guidance.png)","",
              "All models use width256/depth8/heads4, beta1=.9,30k cosine LR4e-4/warmup50, batch128, BF16 compiled/fused training,10% label dropout and EMA.999. Static MLP hidden widths are1024 and1360; DPLR uses r8/s16, hidden1024 and gain1/sqrt(8). Evaluations use DDIM50,5k real/generated images and the fixed auxiliary classifier. Weights remain in memory only.","",
              "Limits: "+" ".join(protocol["limits"]),"","[Comparison](comparison.json) · [Evidence](report.json) · [Audit](audit.json) · [Protocol](protocol.json)",""]
    (output/"summary.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=dict(plain="#667085",plain_ffn_budget="#344054",dplr="#168aad")
    for kind in ("convergence","guidance"):
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
        for r in comparison["aggregates"]:
            keys=("fid","ema_noise_mse") if kind=="convergence" else ("fid","agreement_percent")
            xs=STEPS if kind=="convergence" else SCALES
            for ax,key in zip(axes,keys):
                points=[r["convergence" if kind=="convergence" else "final_guidance"][str(x)][key] for x in xs]
                ax.errorbar(xs,[v["mean"] for v in points],yerr=[v["sd"] for v in points],marker="o",capsize=3,label=r["family"],color=colors[r["family"]])
                ax.set_xlabel("Training step" if kind=="convergence" else "CFG scale");ax.grid(alpha=.15)
        axes[0].set_ylabel("EMA CFG1 FID" if kind=="convergence" else "Final EMA FID")
        axes[1].set_ylabel("EMA epsilon MSE" if kind=="convergence" else "Requested-class agreement (%)");axes[0].legend(fontsize=8)
        for suffix in ("png","svg"):fig.savefig(output/f"{kind}.{suffix}",dpi=160)
        plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=("prepare","run","audit"))
    parser.add_argument("--output",type=Path);parser.add_argument("--launch",action="store_true");args=parser.parse_args()
    out=args.output or Path("runs")/("dit-capacity-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if args.action=="prepare":print(json.dumps(prepare(out,args.launch)),flush=True)
    elif args.action=="run":run(out)
    else:print(json.dumps(audit(out),indent=2))

#!/usr/bin/env python3
"""Audited multi-seed rank-2 comparison, paired deltas and update trajectories."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

from audit_dit_rank2 import audit
from dit_rank2_confirmation import FAMILIES, CONFIRM_SEEDS, OBSERVATIONS, digest, update_metric
from train_dit_cifar10 import write_json

ORDER=("adaln_d192","bottleneck_r2","basis_r2","basis_r8","dplr_r8_raw","adaln_d240")
LABELS={"adaln_d192":"AdaLN 192","bottleneck_r2":"Bottleneck r2","basis_r2":"Basis r2",
        "basis_r8":"Basis r8 / sqrt(8)","dplr_r8_raw":"Raw DPLR r8","adaln_d240":"AdaLN 240"}
COLORS={f:c for f,c in zip(ORDER,("#64748b","#16a34a","#0891b2","#7c3aed","#ea580c","#334155"))}


def mean_sd(values):
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),values=list(values))


def event(row,step):
    return next(e for e in row["design_diagnostics"]["diagnostics"] if e["step"]==step)


def step_update(row,step):
    return statistics.mean(v["update_rms"] for v in event(row,step)["fixed_token_condition_function_updates"])


def live_ratio(row,step):
    return statistics.mean(b["delta_to_base_rms"] for b in event(row,step)["branches"])


def summarize(output,destination):
    output=Path(output).resolve(); destination=Path(destination).resolve()
    audited=audit(output)
    report=json.loads((output/"report.json").read_text())
    rows=report["confirmation"]["rows"]
    keyed={(r["family"],r["config"]["seed"]):r for r in rows}
    aggregates=[]; paired={}
    for family in ORDER:
        group=[keyed[family,s] for s in CONFIRM_SEEDS]
        metrics={"fid":[r["generative_evaluation"]["fid"] for r in group],
                 "agreement_percent":[100*r["generative_evaluation"]["class_agreement"]["generated"]["agreement"] for r in group],
                 "ema_noise_mse":[r["ema_validation"]["noise_mse"] for r in group],
                 "train_images_per_second":[r["mean_train_images_per_second_excluding_warmup"] for r in group],
                 "peak_allocated_mib":[r["peak_allocated_bytes"]/1024**2 for r in group],
                 "live_delta_to_base_30000":[live_ratio(r,30000) for r in group],
                 "combined_to_base_30000":[statistics.mean(b["combined_to_base_rms"] for b in event(r,30000)["branches"]) for r in group],
                 "gate_rms_30000":[statistics.mean(b["gate_rms"] for b in event(r,30000)["branches"]) for r in group]}
        if family in ("basis_r2","bottleneck_r2","basis_r8","dplr_r8_raw"):
            metrics["early_update_to_paired_reference"]=[update_metric(r)/update_metric(keyed["basis_r8",s]) for s,r in zip(CONFIRM_SEEDS,group)]
            metrics["final_update_to_paired_reference"]=[step_update(r,30000)/step_update(keyed["basis_r8",s],30000) for s,r in zip(CONFIRM_SEEDS,group)]
            metrics["base_delta_correlation_30000"]=[statistics.mean(b["base_delta_correlation"] for b in event(r,30000)["branches"]) for r in group]
        aggregate=dict(family=family,label=LABELS[family],parameters=group[0]["parameters"],
                       adapter_parameters=group[0]["adapter_parameters"],seeds=list(CONFIRM_SEEDS),
                       metrics={k:mean_sd(v) for k,v in metrics.items()})
        aggregates.append(aggregate)
        paired[family]={}
        for control in ("adaln_d192","adaln_d240","basis_r8","dplr_r8_raw"):
            if family==control: continue
            differences={"fid":[keyed[family,s]["generative_evaluation"]["fid"]-keyed[control,s]["generative_evaluation"]["fid"] for s in CONFIRM_SEEDS],
                         "agreement_pp":[100*(keyed[family,s]["generative_evaluation"]["class_agreement"]["generated"]["agreement"]-keyed[control,s]["generative_evaluation"]["class_agreement"]["generated"]["agreement"]) for s in CONFIRM_SEEDS]}
            paired[family][control]={k:mean_sd(v) for k,v in differences.items()}
    evidence=dict(raw_workflow=str(output),raw_report_sha256=digest(output/"report.json"),audit=audited,
                  calibration=report["calibration"]["selected"],aggregates=aggregates,paired_differences=paired,
                  gpu_hours_training_validation_diagnostics=sum(r["loop_seconds"] for r in rows)/3600,
                  gpu_hours_final_generation_evaluation=sum(r["generative_evaluation"]["seconds"] for r in rows)/3600,
                  limits=report["limits"]+["The controls use base width 192 and larger width 240; there is no exact parameter-matched FFN or static-adapter control in this study, so added capacity and conditioning are not isolated."],quality_winner_claimed=False)
    destination.mkdir(exist_ok=True,parents=True)
    prefix="dit-rank2-confirmation"
    for filename in (prefix+".json",prefix+"-audit.json",prefix+"-evidence.json"):
        if (destination/filename).exists(): raise FileExistsError(destination/filename)
    write_json(destination/(prefix+".json"),evidence)
    shutil.copy2(output/"audit.json",destination/(prefix+"-audit.json"))
    shutil.copy2(output/"report.json",destination/(prefix+"-evidence.json"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    for ax,key,title in zip(axes,("fid","agreement_percent"),("FID (lower is better)","Requested-class agreement (%)")):
        for i,a in enumerate(aggregates):
            metric=a["metrics"][key]; color=COLORS[a["family"]]
            ax.bar(i,metric["mean"],color=color,alpha=.7,yerr=metric["sd"],capsize=4)
            ax.scatter(i+np.array([-.12,0,.12]),metric["values"],c="black",s=19,zorder=3)
        ax.set_xticks(range(len(ORDER)),[LABELS[f] for f in ORDER],rotation=25,ha="right")
        ax.set_title(title); ax.grid(axis="y",alpha=.15)
    fig.suptitle("30,000 steps · three paired training seeds · mean ± sample SD")
    for suffix in ("png","svg"): fig.savefig(destination/(prefix+"."+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    for family in ("bottleneck_r2","basis_r2","basis_r8","dplr_r8_raw"):
        for seed in CONFIRM_SEEDS:
            row=keyed[family,seed]; reference=keyed["basis_r8",seed]
            steps=[e["step"] for e in row["design_diagnostics"]["diagnostics"] if e["step"]>=10]
            axes[0].plot(steps,[step_update(row,s)/step_update(reference,s) for s in steps],color=COLORS[family],alpha=.27,lw=1)
            axes[1].plot(steps,[live_ratio(row,s) for s in steps],color=COLORS[family],alpha=.27,lw=1)
        axes[0].plot(steps,[statistics.mean(step_update(keyed[family,seed],s)/step_update(keyed["basis_r8",seed],s) for seed in CONFIRM_SEEDS) for s in steps],color=COLORS[family],label=LABELS[family],lw=2)
        axes[1].plot(steps,[statistics.mean(live_ratio(keyed[family,seed],s) for seed in CONFIRM_SEEDS) for s in steps],color=COLORS[family],label=LABELS[family],lw=2)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("Training step"); ax.grid(alpha=.15)
    axes[0].set_title("Fixed-input update / paired scaled-r8 reference")
    axes[0].axhline(1,color="black",ls="--",lw=.7)
    axes[1].set_title("Live delta/base RMS (raw-model diagnostic)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Mean trajectories with individual seeds · highest-FP32 fixed probes")
    for suffix in ("png","svg"): fig.savefig(destination/(prefix+"-updates."+suffix),dpi=180)
    plt.close(fig)
    lines=["# Full-loader calibrated rank-2 diffusion confirmation","",
           "Three paired training seeds (50/51/52), 30,000 steps each. All values below are mean ± sample standard deviation, not confidence intervals.","",
           "| Variant | Parameters | Added adapter params | FID ↓ | Class agreement ↑ | EMA noise MSE | Train images/s |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    def cell(a,key,places):
        m=a["metrics"][key];return f"{m['mean']:.{places}f} ± {m['sd']:.{places}f}"
    for a in aggregates:
        lines.append(f"| {a['label']} | {a['parameters']:,} | {a['adapter_parameters']:,} | {cell(a,'fid',2)} | {cell(a,'agreement_percent',2)}% | {cell(a,'ema_noise_mse',5)} | {cell(a,'train_images_per_second',0)} |")
    lines += ["",f"![Quality comparison]({prefix}.png)","",
              "| Variant | Whole-training peak allocated MiB |","|---|---:|"]
    for a in aggregates:
        lines.append(f"| {a['label']} | {cell(a,'peak_allocated_mib',1)} |")
    lines += ["", "Peaks include the model, in-memory EMA, gradients, optimizer, training activations and resident fixed diagnostic inputs. Diagnostic forwards and final FID evaluation are excluded. These are whole-trainer peaks, not generated-factor storage or standalone-layer activation memory. Timing is concurrent eager training on the shared machine.","",
              "Calibration used the actual 45,000-example training loader on seeds 48/49 and a 500-step prefix of the 30,000-step schedule. Fixed gains are `1/sqrt(rank)` for scaled adapters; raw r8 retains gain 1. The metric averages highest-FP32 independent-input transform-update RMS over all six blocks, steps 50/100/250/500 and both seeds. The declared ±25% tolerance was met on the first attempt: basis r2 0.964×, bottleneck r2 1.034×. Both global adapter LR multipliers remain 1. No loss, validation or generated quality entered selection. Rates were frozen before confirmation.","",
              "| Adapter | Early update / paired r8 | Final update / paired r8 | Final live delta/base RMS | Final combined/base RMS | Base–delta correlation |",
              "|---|---:|---:|---:|---:|---:|"]
    for a in aggregates:
        if "early_update_to_paired_reference" in a["metrics"]:
            lines.append(f"| {a['label']} | {cell(a,'early_update_to_paired_reference',3)} | {cell(a,'final_update_to_paired_reference',3)} | {cell(a,'live_delta_to_base_30000',3)} | {cell(a,'combined_to_base_30000',3)} | {cell(a,'base_delta_correlation_30000',3)} |")
    outside=[f"{a['label']} seed {seed}: {ratio:.3f}×" for a in aggregates if a["family"] in ("basis_r2","bottleneck_r2")
             for seed,ratio in zip(CONFIRM_SEEDS,a["metrics"]["early_update_to_paired_reference"]["values"]) if abs(ratio-1)>.25]
    lines += ["", "Fresh-seed early matches outside the declared calibration tolerance: "+("; ".join(outside) if outside else "none")+". Confirmation rates were not adjusted in response."]
    lines += ["",f"![Update and live magnitude trajectories]({prefix}-updates.png)","",
              "Raw-model transform diagnostics and EMA quality measure different states. Calibration matches a scalar aggregate, not every block, seed or later update. The live RMS ratios describe the additive MLP-input branch; they are not variance-preservation guarantees. `combined/base` measures `rms(base + delta) / rms(base)` before GELU; the centered base–delta correlation helps distinguish reinforcing and cancelling contributions. These are averages over six blocks and a fixed eight-image diagnostic batch, not whole-dataset moments.","",
              "Paired differences are variant minus control, in training-seed order 50/51/52. Negative FID differences favor the variant; positive agreement differences favor it.","",
              "| Variant vs control | FID differences | Agreement differences (pp) |","|---|---|---|"]
    for family in ("bottleneck_r2","basis_r2","basis_r8","dplr_r8_raw"):
        controls=("adaln_d192","adaln_d240")+(("basis_r8","dplr_r8_raw") if family in ("bottleneck_r2","basis_r2") else ())
        for control in controls:
            d=paired[family][control]
            lines.append(f"| {LABELS[family]} vs {LABELS[control]} | {', '.join(f'{v:+.2f}' for v in d['fid']['values'])} | {', '.join(f'{v:+.2f}' for v in d['agreement_pp']['values'])} |")
    lines += ["","The recipe uses pixel-space epsilon prediction, width 192/depth 6/3 heads/patch 4 (64 tokens), class/time broadcast conditions, base LR 4e-4, warmup 50, AdamW (.9,.999), no weight decay, clipping 1, cosine floor .01, BF16 training and in-memory EMA .999. AdaLN-240 widens only the base. FID uses 5,000 held-out real and 5,000 generated images, 50-step DDIM without CFG, sampling seed 2028. Requested-class agreement uses the existing frozen FP32 classifier, with 500 examples per requested class; real held-out classifier accuracy is 80.72%.","",
              f"Training/validation/diagnostic loops used {evidence['gpu_hours_training_validation_diagnostics']:.3f} aggregate GPU-hours; final generation/evaluation used {evidence['gpu_hours_final_generation_evaluation']:.3f}. Calibration and startup are additional. Raw evidence is {audited['bytes_retained']/1024**2:.1f} MiB. Workers are managed with private supervisors and shared lifetime claims, at most six concurrent workers. No new weights/checkpoints, generated image datasets or official-test evaluations were saved/performed.","",
              "The independent audit recomputes calibration proposals/selection, frozen rates and source hashes, actual recorded schedules, full disjoint splits and counts, paired initializations and observed batch/noise digests, final noise RNG states, zero starts and gradient wake-up, evaluator hashes and denominators, finite results and successful exits. Observed diagnostic digests do not hash every training image.","",
              "Limits: "+" ".join(evidence["limits"]),"",
              f"[Structured comparison]({prefix}.json) · [Complete evidence]({prefix}-evidence.json) · [Audit]({prefix}-audit.json)","",
              "Raw workflow: `"+str(output)+"`.","",
              "```bash","/venv/main/bin/python dit_rank2_confirmation.py prepare --launch",
              "/venv/main/bin/python audit_dit_rank2.py runs/dit-rank2-YYYYMMDDTHHMMSSZ",
              "/venv/main/bin/python summarize_dit_rank2.py runs/dit-rank2-YYYYMMDDTHHMMSSZ","```",""]
    (destination/(prefix+".md")).write_text("\n".join(lines))
    analysis=output/"analysis_source";analysis.mkdir(exist_ok=True)
    for name in ("audit_dit_rank2.py",Path(__file__).name):
        shutil.copy2(Path(__file__).resolve().parent/name,analysis/name)
    manifest=dict(raw_workflow=str(output),raw_report_sha256=digest(output/"report.json"),
                  artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in destination.glob(prefix+"*") if p.is_file() and not p.name.endswith("-manifest.json")},
                  analysis_source={p.name:digest(p) for p in analysis.iterdir() if p.is_file()},weights_saved=False)
    write_json(destination/(prefix+"-manifest.json"),manifest)
    return evidence


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--destination",type=Path,default=Path("reports"))
    args=parser.parse_args()
    result=summarize(args.output,args.destination)
    print(json.dumps(result["aggregates"],indent=2))

#!/usr/bin/env python3
"""Export an audited single-seed quality screen and update-transfer figures."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics

from train_dit_cifar10 import write_json


ORDER=["adaln_d192","bottleneck_r2","bottleneck_r4","basis_r2","basis_r4","basis_r8",
       "dplr_r8_raw","adaln_d216","adaln_d240"]
LABELS=["AdaLN\n192","Bottleneck\nr2","Bottleneck\nr4","Basis\nr2","Basis\nr4","Basis\nr8",
        "DPLR\nraw r8","AdaLN\n216","AdaLN\n240"]


def event(row, step):
    return next(e for e in row["design_diagnostics"]["diagnostics"] if e["step"]==step)


def function_update(row, steps):
    return statistics.mean(v["update_rms"] for step in steps
                           for v in event(row,step)["fixed_token_condition_function_updates"])


def live_ratio(row, step):
    return statistics.mean(b["delta_to_base_rms"] for b in event(row,step)["branches"])


def figure_bytes(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"]="adalora-gain-quality"
    outputs={}
    colors=["#777777","#e76f51","#b56576","#2a9d8f","#7b8f37","#264653","#6d597a","#999999","#bbbbbb"]
    fig,axes=plt.subplots(1,2,figsize=(14,4.6))
    fid=[rows[n]["generative_evaluation"]["fid"] for n in ORDER]
    agreement=[100*rows[n]["generative_evaluation"]["class_agreement"]["generated"]["agreement"] for n in ORDER]
    for ax,values,title,ylabel in zip(axes,[fid,agreement],["Global FID (lower is better)","Requested-class agreement"],
                                    ["5,000-real / 5,000-fake FID","Auxiliary classifier agreement (%)"]):
        ax.bar(range(len(ORDER)),values,color=colors)
        ax.set_xticks(range(len(ORDER)),LABELS,fontsize=8)
        ax.set(title=title,ylabel=ylabel)
        ax.set_ylim(0,max(values)*1.18)
        ax.bar_label(ax.containers[0],fmt="%.1f",fontsize=8,padding=3)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("Calibrated-recipe quality screen: 5,000 steps, fresh seed 47",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95))
    for kind in ("png","svg"):
        buffer=io.BytesIO(); fig.savefig(buffer,format=kind,dpi=170,metadata={"Date":None} if kind=="svg" else None)
        outputs[kind]=buffer.getvalue()
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
    reference=rows["basis_r8"]
    for name,color in zip(ORDER[1:7],colors[1:7]):
        row=rows[name]
        steps=[e["step"] for e in row["design_diagnostics"]["diagnostics"] if e["step"]>=10]
        label=name.replace("_"," ")
        style="--" if name=="dplr_r8_raw" else "-"
        axes[0].plot(steps,[function_update(row,[s])/function_update(reference,[s]) for s in steps],
                     style,marker="o",ms=3,color=color,label=label)
        axes[1].plot(steps,[live_ratio(row,s) for s in steps],style,marker="o",ms=3,color=color)
    axes[0].axhline(1,lw=.8,color="#777777")
    axes[0].set(title="Full-data functional update transfer",ylabel="Raw-model update RMS / scaled basis r8")
    axes[1].set(title="Raw-model live branch magnitude",ylabel="Mean block delta / base RMS")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("Optimizer step")
        ax.set_xticks([10,50,100,500,1000,5000],labels=["10","50","100","500","1,000","5,000"])
        ax.minorticks_off(); ax.grid(alpha=.2); ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.suptitle("Raw diagnostics alongside EMA generative evaluation",fontsize=13)
    fig.tight_layout(rect=(0,.16,1,.94))
    for kind in ("png","svg"):
        buffer=io.BytesIO(); fig.savefig(buffer,format=kind,dpi=170,metadata={"Date":None} if kind=="svg" else None)
        outputs["-updates."+kind]=buffer.getvalue()
    plt.close(fig)
    return outputs


def export(output,prefix):
    output,prefix=Path(output).resolve(),Path(prefix).resolve()
    report=json.loads((output/"report.json").read_text())
    audit=json.loads((output/"audit.json").read_text())
    assert audit["report_sha256"]==hashlib.sha256((output/"report.json").read_bytes()).hexdigest()
    assert all(audit["checks"][k] for k in ("all_runs_complete","successful_worker_exits","full_fp32_probes","frozen_recipes_recomputed"))
    rows={r["name"]:r for r in report["rows"]}
    best_fid=min(rows.values(),key=lambda r:r["generative_evaluation"]["fid"])
    best_class=max(rows.values(),key=lambda r:r["generative_evaluation"]["class_agreement"]["generated"]["agreement"])
    lines=["# Calibrated DiT recipes: fresh-seed quality screen","",
           f"Audited source run: `{output}`. Nine configurations, one fresh training seed (47), 5,000 steps each.","",
           f"The lowest observed endpoint FID is {best_fid['generative_evaluation']['fid']:.2f} for `{best_fid['name']}`. The highest requested-class agreement is {100*best_class['generative_evaluation']['class_agreement']['generated']['agreement']:.2f}% for `{best_class['name']}`. These are descriptive short-screen endpoints, not confirmed winners.",
           "", "## Protocol", "",
           "The five fixed-gain recipes were frozen from the audited full-FP32 functional-update calibration before this screen launched. The raw-gain rank-8 DPLR anchor and AdaLN widths 192/216/240 provide fresh controls. No learning rate was selected using quality in this screen.",
           "", "All use CIFAR-10's cached training split: 45,000 training and 5,000 held-out images, split seed 123. Training seed 47 is fresh relative to calibration and earlier diffusion runs. Batch 128, LR 0.0004, 50-step warmup, cosine floor 1%, AdamW (.9,.999), no weight decay, global clip 1, in-memory EMA decay .999. Depth 6, three heads, patch 4, MLP ratio 4; adapter models have width 192. AdaLN scale/shift/gates and non-affine LayerNorm are common. Adapter input is unmodulated normalized tokens.",
           "", "Final evaluation uses EMA weights, DDIM 50 steps without guidance, common sampling seed 2028, 5,000 generated images and the same 5,000 real images. Requested-class agreement uses the existing frozen classifier on the same generated stream as FID. Its real-image accuracy is a diagnostic, not a ceiling on generated-image agreement. No evaluator was retrained.",
           "", "## Quality and cost", "",
           "| Family | Parameters | Extra adapter parameters | FID ↓ | Class agreement ↑ | EMA noise MSE ↓ | Train images/s |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        r=rows[name]; g=r["generative_evaluation"]
        lines.append(f"| {name} | {r['parameters']:,} | {r['adapter_parameters']:,} | {g['fid']:.2f} | {100*g['class_agreement']['generated']['agreement']:.2f}% | {r['ema_validation']['noise_mse']:.5f} | {r['mean_train_images_per_second_excluding_warmup']:,.0f} |")
    baseline=rows["adaln_d192"]
    bn=rows["bottleneck_r2"]
    small=rows["basis_r2"]
    raw=rows["dplr_r8_raw"]
    norm=rows["basis_r8"]
    fid=lambda r:r["generative_evaluation"]["fid"]
    agreement=lambda r:100*r["generative_evaluation"]["class_agreement"]["generated"]["agreement"]
    lines += ["", f"Bottleneck r2 adds only {bn['adapter_parameters']:,} parameters ({100*bn['adapter_parameters']/baseline['parameters']:.2f}% of the base model), with FID change {fid(bn)-fid(baseline):+.2f} and agreement change {agreement(bn)-agreement(baseline):+.2f} percentage points versus width-192 AdaLN. Basis r2 adds {small['adapter_parameters']:,} parameters, with FID change {fid(small)-fid(baseline):+.2f} and agreement change {agreement(small)-agreement(baseline):+.2f} points. These two smaller variants are useful candidates for longer paired-seed checks.",
              "", f"At identical rank-8 parameter counts, the scaled recipe changes FID by {fid(norm)-fid(raw):+.2f} and agreement by {agreement(norm)-agreement(raw):+.2f} points relative to raw gain. That is a quality/conditioning tradeoff, not dominance. Increasing rank or width does not give a monotonic endpoint improvement in this short screen."]
    lines += ["", f"The fixed classifier's real-image accuracy was {100*audit['classifier_real_accuracy']:.2f}%. Generated agreement is a proxy affected by classifier errors and domain shift. Concurrent train throughput excludes diagnostic probes and validation; it shares host resources and does not establish an isolated speed ordering.",
              "", f"![Single-seed quality endpoints]({prefix.name}.png)","",
              "## Did the update calibration transfer?", "",
              "| Adapter | Gain | Adapter LR multiplier | Early update / current scaled r8 | Step-5000 update / current scaled r8 | Raw live delta/base RMS at 5000 |",
              "|---|---:|---:|---:|---:|---:|"]
    ref=rows["basis_r8"]
    for name in ORDER[1:7]:
        r=rows[name]; cfg=r["config"]
        early=function_update(r,[10,25,50])/function_update(ref,[10,25,50])
        last=function_update(r,[5000])/function_update(ref,[5000])
        lines.append(f"| {name} | {cfg['adapter_gain']:.4f} | {cfg['adapter_lr_multiplier']:.4f} | {early:.3f} | {last:.3f} | {live_ratio(r,5000):.3f} |")
    lines += ["", "The bottleneck update match from the subset calibration does not persist on this full-data seed: both early and final update ratios are below the reference. Architectural and optimization differences therefore remain entangled. Recalibrate on the full-data loader across intended seeds, or explicitly track this drift, before treating a longer comparison as a test of function class at matched updates."]
    lines += ["", "These full-FP32 probes hold independent normalized tokens/conditions fixed and measure optimizer-induced movement of the raw adapter. Live image/condition magnitudes also use the raw model. Quality is evaluated with EMA; the two states serve different purposes. Transfer to a new seed and full-data loader is measured rather than assumed. Similar branch magnitudes and similar functional updates are different constraints.",
              "", f"![Functional-update transfer and branch magnitude]({prefix.name}-updates.png)","",
              "## Limits and retained evidence", "",
              "This screen has one seed and a short horizon. A low endpoint FID does not establish a multi-seed or long-budget advantage; sampling uncertainty and optimization differences remain. Base LR 0.0004 was previously at the upper tuning boundary, and these architectures have not received equal, separate quality-LR searches. Rank scaling is an empirical recipe, not a variance-preservation guarantee or a muP derivation.",
              "", "The audit verifies frozen recipes/source/evaluators, identical splits and budgets, common width-192 base initialization, paired raw/scaled rank-8 initialization, matching batch/noise digests at diagnostic steps, matching final noise RNG states, zero initialization and adapter gradient wake-up, quality denominators, finite records and successful managed exits. Diagnostic digests verify the observed batches; they are not a hash of every training image.",
              "", "No new model weights/checkpoints or generated image datasets were saved. Compact configs, split indices, sources, logs, metrics, summaries and 100-image grids are retained. Existing shared classifier/Inception artifacts are referenced and preserved. The official test split was untouched. Sequence-valued conditioning and cross-attention remain untested.",
              "", f"[Full evidence]({prefix.name}.json) · [Audit]({prefix.name}-audit.json) · [Vector quality figure]({prefix.name}.svg) · [Vector diagnostics]({prefix.name}-updates.svg)",""]
    files={Path(str(prefix)+".md"):"\n".join(lines).encode(),Path(str(prefix)+".json"):(output/"report.json").read_bytes(),
           Path(str(prefix)+"-audit.json"):(output/"audit.json").read_bytes()}
    for suffix,data in figure_bytes(rows).items():
        files[Path(str(prefix)+(suffix if suffix.startswith("-") else "."+suffix))]=data
    prefix.parent.mkdir(parents=True,exist_ok=True)
    for path,data in files.items():
        if path.exists() and path.read_bytes()!=data:
            raise FileExistsError(path)
        path.write_bytes(data)
    analysis=output/"analysis_source"; analysis.mkdir(exist_ok=True)
    target=analysis/Path(__file__).name
    if target.exists() and target.read_bytes()!=Path(__file__).read_bytes():
        raise FileExistsError(target)
    shutil.copy2(__file__,target)
    manifest={"source_run":str(output),"report_sha256":audit["report_sha256"],
              "analysis_source_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
              "outputs_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "weights_saved":False,"test_evaluated":False,"retention":"compact evidence until explicitly removed"}
    write_json(Path(str(prefix)+"-manifest.json"),manifest)
    return {"report":str(prefix)+".md","audited":True}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--prefix",type=Path,default=Path("reports/dit-gain-quality"))
    args=parser.parse_args()
    print(json.dumps(export(args.output,args.prefix)))

#!/usr/bin/env python3
"""Export audited gain/update evidence and a concise tensor-notation report."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics

from benchmark_dit_gain import FAMILIES
from train_dit_cifar10 import write_json


def figure_bytes(rows, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"]="adalora-dit-gain"
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3))
    order=[f[0] for f in FAMILIES]+["basis_r8_raw"]
    colors=["#264653","#2a9d8f","#7b8f37","#e76f51","#b56576","#6d597a"]
    labels=["Basis r8, scaled","Basis r2, scaled","Basis r4, scaled",
            "Bottleneck r2, scaled","Bottleneck r4, scaled","Basis r8, raw gain"]
    reference=rows["basis_r8"]
    for name,color,label in zip(order,colors,labels):
        row=rows[name]
        events=[e for e in row["diagnostics"] if 10<=e["step"]<=steps]
        xs=[e["step"] for e in events]
        updates=[mean_update(row,s)/mean_update(reference,s) for s in xs]
        live=[statistics.mean(b["delta_to_base_rms"] for b in e["branches"]) for e in events]
        style="--" if name=="basis_r8_raw" else "-"
        axes[0].plot(xs,updates,style,marker="o",ms=4,color=color,label=label)
        axes[1].plot(xs,live,style,marker="o",ms=4,color=color)
    axes[0].axhline(1,color="#777777",lw=.8,zorder=0)
    axes[0].set(title="Fixed-input functional update",ylabel="Update RMS / scaled basis r8")
    axes[1].set(title="Live branch magnitude",ylabel="Mean block delta / base RMS")
    ticks=sorted({s for s in (10,25,50,100,250,steps) if s<=steps})
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks,labels=[str(s) for s in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.suptitle("Gain and update controls: one-seed engineering screen",fontsize=13)
    fig.tight_layout(rect=(0,.16,1,.94))
    outputs={}
    for kind in ("png","svg"):
        buffer=io.BytesIO()
        fig.savefig(buffer,format=kind,dpi=180,metadata={"Date":None} if kind=="svg" else None)
        outputs[kind]=buffer.getvalue()
    plt.close(fig)
    return outputs


def mean_update(row, step):
    event=next(e for e in row["diagnostics"] if e["step"]==step)
    return statistics.mean(v["update_rms"] for v in event["fixed_token_condition_function_updates"])


def export(output, prefix):
    output,prefix=Path(output).resolve(),Path(prefix).resolve()
    report=json.loads((output/"report.json").read_text())
    audit=json.loads((output/"audit.json").read_text())
    assert audit["checks"]["complete"] and audit["report_sha256"]==hashlib.sha256((output/"report.json").read_bytes()).hexdigest()
    rows={r["name"]:r for r in report["confirmation"]}
    pilots={p["candidate"]["name"]:p for p in report["calibration"]}
    settings=report["settings"]
    ref=rows["basis_r8"]
    live_ratio=lambda row: statistics.mean(b["delta_to_base_rms"] for b in row["diagnostics"][-1]["branches"])
    basis_final=[live_ratio(rows[f"basis_r{rank}"]) for rank in (2,4,8)]
    bottleneck_final=[live_ratio(rows[f"bottleneck_r{rank}"]) for rank in (2,4)]
    final_updates=[mean_update(rows[f[0]],settings["steps"])/mean_update(ref,settings["steps"]) for f in FAMILIES]
    lines=["# DiT fixed-gain and functional-update screen", "",
           f"Source run: `{output}`. Audited {len(pilots)} calibration runs and six {settings['steps']}-step confirmations; one seed (42).",
           "", "This is an optimization diagnostic. No quality winner is selected, and there are no model weights, checkpoints, EMA, FID, or official-test evaluation.",
           "", f"All selected families meet the early ±25% update tolerance. At step {settings['steps']}, their fixed-input update ratios span {min(final_updates):.3f}–{max(final_updates):.3f} relative to the scaled rank-8 reference. Live delta/base RMS remains very different: basis ranks 2/4/8 give {', '.join(f'{x:.3f}' for x in basis_final)}, while bottleneck ranks 2/4 give {', '.join(f'{x:.3f}' for x in bottleneck_final)}. Matching function movement does not match the size of the learned branch.",
           "", "## Protocol", "", "```python", "h = norm_mlp(tokens)  # (B, T, d), affine=False",
           "base = mlp_in(h * (1 + scale[:, None]) + shift[:, None])",
           "delta = adapter_gain * conditional_lowrank(h, condition)  # (B, T, 4*d)",
           "pre_act = base + delta", "tokens = tokens + gate[:, None] * mlp_out(gelu(pre_act))", "```", "",
           "Gain is a fixed positive scalar. Basis mixing retains a zero output factor; the bottleneck retains a zero condition-to-rank gain head. The default gain of 1 preserves the previous code path.",
           "", f"All scaled candidates use `adapter_gain = 1 / sqrt(rank)`. Base LR is 4e-4. AdamW uses betas=(0.9, 0.999), no weight decay, global gradient clipping at 1, and {settings['warmup']}-step warmup then cosine decay. All adapter groups share the selected multiplier.",
           "", f"The target is the rank-8 basis model with scaled gain and multiplier 1. Calibration averages fixed-token/condition output-update RMS across all blocks at steps {settings['calibration_observation_steps']}. A rate correction multiplies the current rate by `target / observed`, bounded to [1/16, 16], with at most two corrections. Selection uses closest absolute log update ratio; loss and validation never enter the choice. ±25% is the declared tolerance.",
           "", "Calibration and confirmation deliberately share their early batches, noise, initialization, and LR schedule. The audit verifies that prefix replay. Later confirmation steps measure whether the early match persists; they are not independent-seed validation.",
           "", "## Calibration", "", "| Family | Fixed gain | Unit-rate update / target | Selected adapter LR multiplier | Selected update / target | Within ±25% |",
           "|---|---:|---:|---:|---:|---|" ]
    for family,_,_ in FAMILIES:
        s=report["selected"][family]
        first=pilots[f"{family}_candidate0"]["candidate"]
        lines.append(f"| {family} | {s['gain']:.4f} | {first['metric_to_target']:.3f} | {s['adapter_lr_multiplier']:.4f} | {s['metric_to_target']:.3f} | {'yes' if s['within_tolerance'] else 'no'} |")
    lines += ["", f"Target mean functional update RMS: `{report['target_update_rms']:.8g}`. This probe holds independent normalized tokens and conditions fixed, so it measures adapter function movement rather than raw parameter norms or a drift in inputs. It uses FP32 tensors, highest FP32 matmul precision, and no autocast, independently of BF16 training.",
              "", "## Confirmation", "", "| Family | Extra parameters | Early update / target | Final update / scaled rank-8 | Final live delta / base RMS | Final pre-activation / base RMS |",
              "|---|---:|---:|---:|---:|---:|"]
    for family in [f[0] for f in FAMILIES]+["basis_r8_raw"]:
        r=rows[family]
        branches=r["diagnostics"][-1]["branches"]
        ratio=statistics.mean(b["delta_to_base_rms"] for b in branches)
        combined=statistics.mean(b["combined_to_base_rms"] for b in branches) if "combined_to_base_rms" in branches[0] else None
        lines.append(f"| {family} | {r['adapter_parameters']:,} | {r['early_update_metric_to_target']:.3f} | {mean_update(r,settings['steps'])/mean_update(ref,settings['steps']):.3f} | {ratio:.3f} | {combined:.3f} |" if combined is not None else
                     f"| {family} | {r['adapter_parameters']:,} | {r['early_update_metric_to_target']:.3f} | {mean_update(r,settings['steps'])/mean_update(ref,settings['steps']):.3f} | {ratio:.3f} | not recorded |")
    checkpoints=[step for step in (100,250,settings["steps"]) if step<=settings["steps"]]
    lines += ["", "Later functional-update ratios relative to the scaled rank-8 reference:", "",
              "| Family | " + " | ".join(f"Step {s}" for s in dict.fromkeys(checkpoints)) + " |",
              "|---|"+"---:|"*len(dict.fromkeys(checkpoints))]
    for family,_,_ in FAMILIES:
        lines.append("| "+family+" | "+" | ".join(f"{mean_update(rows[family],step)/mean_update(ref,step):.3f}" for step in dict.fromkeys(checkpoints))+" |")
    lines += ["", "## Interpretation and limits", "",
              "An early match is a calibration result, not proof of variance preservation or comparable learning throughout training. Fixed probes, live branch magnitudes, and gate magnitudes measure different parts of the system. A shared scalar adapter rate cannot generally control all blocks, generator/factor groups, and steps.",
              "", "Basis and bottleneck architectures retain different function classes and zero-init paths. The bottleneck's static input/output subspaces are particularly restrictive at rank 2 or 4. This screen does not tune separate factor rates, change generator initialization, test orthogonal/structured transforms, or derive muP scaling.",
              "", f"Model/data: width {settings['width']}, depth {settings['depth']}, heads {settings['heads']}, patch {settings['patch_size']}, batch {settings['batch_size']}; fixed {settings['bank_examples']}-image train bank sampled with replacement and flips, {settings['val_examples']} held-out training-split examples. Validation MSE remains a learning diagnostic. Whole-model timing is eager and can vary with shared host load; compile timing is deliberately not repeated here.",
              "", "Sequence-valued conditioning and cross-attention remain untested.", ""]
    lines += [f"![Functional updates and live branch magnitude]({prefix.name}.png)", "",
              f"[Vector figure]({prefix.name}.svg). Each plotted update is a single diagnostic step; calibration matches the average over its declared early observation window.", ""]
    prefix.parent.mkdir(parents=True,exist_ok=True)
    files={Path(str(prefix)+".md"):"\n".join(lines).encode(),
           Path(str(prefix)+".json"):(output/"report.json").read_bytes(),
           Path(str(prefix)+"-audit.json"):(output/"audit.json").read_bytes()}
    files.update({Path(str(prefix)+"."+kind):data for kind,data in figure_bytes(rows,settings["steps"]).items()})
    for path,data in files.items():
        if path.exists() and path.read_bytes()!=data:
            raise FileExistsError(path)
        path.write_bytes(data)
    source=output/"analysis_source"
    source.mkdir(exist_ok=True)
    frozen=source/Path(__file__).name
    if frozen.exists() and frozen.read_bytes()!=Path(__file__).read_bytes():
        raise FileExistsError(frozen)
    shutil.copy2(__file__,frozen)
    manifest={"source_run":str(output),"report_sha256":audit["report_sha256"],
              "analysis_source_sha256":hashlib.sha256(frozen.read_bytes()).hexdigest(),
              "outputs_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "weights_saved":False,"test_evaluated":False,"retention":"compact research evidence until explicitly removed"}
    write_json(Path(str(prefix)+"-manifest.json"),manifest)
    return {"report":str(prefix)+".md","audited":True}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--prefix",type=Path,default=Path("reports/dit-gain-engineering"))
    args=parser.parse_args()
    print(json.dumps(export(args.output,args.prefix)))

#!/usr/bin/env python3
"""Scientific LR-tuning and paired-seed confirmation figures from audited evidence."""

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dit_followup import family


STYLES = {"adaln_d192": ("AdaLN d192", "#333333", "o"), "dplr": ("DPLR r8 K16", "#748544", "D"),
          "adaln_d216": ("AdaLN d216", "#BF6B35", "^"), "adaln_d240": ("AdaLN d240", "#376C9B", "s")}


def export(fig, prefix):
    for extension in ("png", "pdf"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


def decorate(axes, axis="y"):
    for ax in axes:
        ax.grid(axis=axis, color="#E5E5E5", linewidth=.7)
        ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    report = json.loads((workflow / "report.json").read_text())
    audit = json.loads((workflow / "audit.json").read_text())
    if not audit["checks"]["statistics_recomputed"]:
        raise ValueError("Completed audited evidence required")
    out = Path(report["protocol"]["repo_root"]) / "reports"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    prefix = out / "dit-lr-tuning"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.27, top=.70, wspace=.18)
    for key, (label, color, marker) in STYLES.items():
        rows = sorted((r for r in report["tuning"]["rows"] if family(r["config"]) == key
                       and r["config"]["adapter_lr_multiplier"] == 1), key=lambda r: r["config"]["lr"])
        axes[0].plot([r["config"]["lr"] for r in rows], [r["generative_evaluation"]["fid"] for r in rows],
                     label=label, color=color, marker=marker, linewidth=1.7)
    rows = sorted((r for r in report["tuning"]["rows"] if family(r["config"]) == "dplr"
                   and r["config"]["lr"] == 2e-4), key=lambda r: r["config"]["adapter_lr_multiplier"])
    axes[1].plot([r["config"]["adapter_lr_multiplier"] for r in rows], [r["generative_evaluation"]["fid"] for r in rows],
                 color=STYLES["dplr"][1], marker="D", linewidth=1.7)
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_xticks([1e-4, 2e-4, 4e-4], ["0.0001", "0.0002", "0.0004"])
    axes[1].set_xticks([.3, 1, 3], ["0.3", "1", "3"])
    axes[0].set_xlabel("Base LR · adapter multiplier 1")
    axes[1].set_xlabel("DPLR adapter LR multiplier · base LR 0.0002")
    axes[0].set_ylabel("Final EMA held-out FID · lower is better")
    for ax in axes: ax.minorticks_off()
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2, frameon=False, fontsize=10)
    decorate(axes)
    fig.suptitle("CIFAR-10 DiT · diffusion LR tuning", x=.08, y=.97, ha="left", fontsize=18)
    fig.text(.08, .90, "14 fresh runs · 10,000 steps · seed 42 · final 5k-image FID selects each family's settings", color="#555555")
    fig.text(.08, .13, "DPLR has 5 candidates; each AdaLN control has 3. One tuning seed and one fixed held-out split.", fontsize=10, color="#555555")
    fig.text(.08, .07, "Selection is frozen before fresh-seed confirmation. A boundary selection does not bracket the optimal LR.", fontsize=10, color="#555555")
    export(fig, prefix)

    prefix = out / "dit-lr-confirmation"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=.21, right=.97, bottom=.30, top=.78, wspace=.26)
    for i, (key, (label, color, marker)) in enumerate(STYLES.items()):
        stats = report["statistics"][key]
        for ax, metric, multiplier in ((axes[0], "fid", 1), (axes[1], "class_agreement", 100)):
            values = [v * multiplier for v in stats[metric]["seed_values"].values()]
            ax.errorbar(stats[metric]["mean"] * multiplier, i, xerr=stats[metric]["sample_sd"] * multiplier,
                        color=color, marker=marker, capsize=4, markersize=7, linewidth=1.7)
            ax.scatter(values, [i-.12, i, i+.12], color=color, s=24, alpha=.55)
    labels = [f"{style[0]}\n{report['statistics'][key]['parameters']/1e6:.2f}M parameters" for key, style in STYLES.items()]
    axes[0].set_yticks(range(4), labels)
    axes[1].set_yticks(range(4), ["" for _ in range(4)])
    for ax in axes: ax.invert_yaxis(); ax.margins(x=.15, y=.18)
    axes[0].set_xlabel("EMA FID · mean ± sample SD · lower is better")
    axes[1].set_xlabel("Requested-class agreement (%)\nmean ± sample SD · auxiliary classifier proxy")
    axes[1].axvline(10, color="#888888", linestyle="--", linewidth=1)
    decorate(axes, "x")
    real_accuracy = 100 * report["confirmation"]["rows"][0]["generative_evaluation"]["class_agreement"]["real"]["agreement"]
    fig.suptitle("CIFAR-10 DiT · three-seed confirmation", x=.06, y=.97, ha="left", fontsize=18)
    fig.text(.06, .90, "30,000 steps · seeds 43–45 · settings frozen after tuning · 5,000 real / 5,000 generated images", color="#555555")
    fig.text(.06, .14, "Small points: individual training seeds. Intervals: sample SD, not confidence intervals. Official test untouched.", fontsize=9, color="#555555")
    fig.text(.06, .08, f"Classifier held-out accuracy: {real_accuracy:.2f}%. Agreement is limited by classifier errors and domain shift; dashed line: 10% chance.", fontsize=9, color="#555555")
    fig.text(.06, .035, "Fixed DDIM-50 noise seed across runs; seed spread does not estimate sampling uncertainty or new-split generalization.", fontsize=9, color="#777777")
    export(fig, prefix)

    prefix = out / "dit-lr-confirmation-curves"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.21, top=.70, wspace=.20)
    for key, (label, color, marker) in STYLES.items():
        curves = [r["validation_curve"] for r in report["confirmation"]["rows"] if family(r["config"]) == key]
        steps = [v["step"] for v in curves[0] if v["step"]]
        for ax, state in zip(axes, ("raw", "ema")):
            series = [[v[state]["noise_mse"] for v in curve if v["step"]] for curve in curves]
            means = [statistics.mean(v) for v in zip(*series)]
            sd = [statistics.stdev(v) for v in zip(*series)]
            ax.plot(steps, means, color=color, label=label, marker=marker, markersize=4, linewidth=1.7)
            ax.fill_between(steps, [m-s for m,s in zip(means,sd)], [m+s for m,s in zip(means,sd)], color=color, alpha=.12, linewidth=0)
    for ax, title in zip(axes, ("Raw weights", "EMA weights")):
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Held-out noise MSE · lower is better")
        ax.set_xlabel("Optimizer steps")
        ax.set_xticks([5000, 15000, 30000])
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.09), ncol=2, frameon=False, fontsize=10)
    decorate(axes)
    fig.suptitle("CIFAR-10 DiT · confirmation learning curves", x=.08, y=.97, ha="left", fontsize=18)
    fig.text(.08, .90, "30,000 steps · seeds 43–45 · fixed validation noise/timestep seed 2027 · same 5,000-image split", color="#555555")
    fig.text(.08, .11, "Lines show three-seed means, bands ±1 sample SD. Step 0 omitted: all initial zero-output noise losses are approximately 1.", fontsize=9, color="#555555")
    fig.text(.08, .05, "LRs selected at 10k steps are transferred to 30k; longer-run optimal settings and sequence-conditioned attention remain untested.", fontsize=9, color="#777777")
    export(fig, prefix)
    figures = [out / f"{stem}.{ext}" for stem in ("dit-lr-tuning", "dit-lr-confirmation", "dit-lr-confirmation-curves") for ext in ("png", "pdf")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_report": str(workflow / "report.json"),
                "source_sha256": hashlib.sha256((workflow / "report.json").read_bytes()).hexdigest(),
                "audit_sha256": hashlib.sha256((workflow / "audit.json").read_bytes()).hexdigest(),
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "matplotlib_version": matplotlib.__version__,
                "uncertainty": "sample SD over three training seeds, not CI", "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in figures}}
    (out / "dit-lr-confirmation-figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out / "dit-lr-confirmation.png")


if __name__ == "__main__":
    main()

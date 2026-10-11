#!/usr/bin/env python3
"""Export tuning and three-seed confirmation figures from preserved evidence."""

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLES = {
    "plain": ("Plain, width 192", "#333333", "o"),
    "diagonal": ("Diagonal", "#376C9B", "s"),
    "static_lora_diagonal": ("Static LoRA r8 + diagonal", "#B0688B", "P"),
    "dplr": ("DPLR r8, K16", "#748544", "D"),
    "plain_w216": ("Plain, width 216", "#BF6B35", "^"),
    "plain_w240": ("Plain, width 240", "#B88B25", "v"),
    "exact": ("Exact r2", "#777777", "X"),
}


def family(cfg):
    return f"plain_w{cfg['width']}" if cfg["variant"] == "plain" and cfg["width"] != 192 else cfg["variant"]


def export(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def tuning_figure(report, directory):
    rows = report["tuning"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.69, wspace=0.15)
    for key, (label, color, marker) in STYLES.items():
        points = sorted((r for r in rows if family(r["config"]) == key and
                         r["config"]["hyper_lr_multiplier"] == 1), key=lambda r: r["config"]["lr"])
        if len(points) > 1:
            axes[0].plot([r["config"]["lr"] for r in points],
                         [100 * r["results"]["validation"]["accuracy"] for r in points],
                         color=color, marker=marker, label=label, linewidth=1.5, markersize=5)
    for key in ("dplr", "exact"):
        label, color, marker = STYLES[key]
        points = sorted((r for r in rows if family(r["config"]) == key and r["config"]["lr"] == 5e-4),
                        key=lambda r: r["config"]["hyper_lr_multiplier"])
        axes[1].plot([r["config"]["hyper_lr_multiplier"] for r in points],
                     [100 * r["results"]["validation"]["accuracy"] for r in points],
                     color=color, marker=marker, label=label, linewidth=1.5, markersize=5)
    axes[0].set_xscale("log")
    axes[0].set_xticks([2e-4, 5e-4, 1e-3], ["0.0002", "0.0005", "0.001"])
    axes[0].set_xlabel("Base learning rate · hypernetwork multiplier 1")
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
    axes[1].set_xlabel("Hypernetwork LR multiplier · base LR 0.0005")
    axes[0].set_ylabel("Final validation accuracy (%)")
    values = [100 * r["results"]["validation"]["accuracy"] for r in rows]
    for ax in axes:
        ax.set_ylim(min(values) - 2, max(values) + 2)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncol=2, frameon=False, fontsize=9)
        ax.minorticks_off()
    fig.suptitle("CIFAR-10 ViT · learning-rate tuning", x=0.08, y=0.97, ha="left", fontsize=18)
    fig.text(0.08, 0.91, "20 epochs · seed 42 · 45,000 train / 5,000 validation · final epoch selects settings", color="#555555")
    fig.text(0.08, 0.12, "23 endpoints: 17 new runs and 6 reused center runs with identical training source and split.", fontsize=10, color="#555555")
    fig.text(0.08, 0.07, "Single tuning seed; these endpoints select settings and do not estimate uncertainty. Official test untouched.", fontsize=10, color="#555555")
    export(fig, directory, "tuning")


def confirmation_figures(report, directory):
    stats = report["confirmation_statistics"]
    keys = [key for key in STYLES if key in stats]
    rows = report["confirmation"]["rows"]
    grouped = {key: sorted((r for r in rows if family(r["config"]) == key), key=lambda r: r["config"]["seed"])
               for key in keys}
    seeds = report["protocol"]["confirmation_seeds"]
    if len(seeds) != 3 or any(len(grouped[key]) != 3 for key in keys):
        raise ValueError("Expected exactly three confirmation seeds per family")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.21, top=0.78, wspace=0.30)
    offsets = [-0.12, 0, 0.12]
    for index, key in enumerate(keys):
        label, color, marker = STYLES[key]
        values = [100 * r["results"]["validation"]["accuracy"] for r in grouped[key]]
        deltas = stats[key]["paired_gain_vs_plain_percentage_points"]
        # All seed values remain visible; the interval describes sample spread.
        axes[0].errorbar(statistics.mean(values), index, xerr=statistics.stdev(values),
                         color=color, marker=marker, markersize=7, capsize=4, linewidth=1.6)
        axes[0].scatter(values, [index + offset for offset in offsets], color=color, s=20, alpha=0.55)
        axes[1].scatter(deltas, [index + offset for offset in offsets], color=color, s=28, alpha=0.65)
        axes[1].scatter([statistics.mean(deltas)], [index], color=color, marker=marker, s=70, zorder=3)
        axes[1].annotate(f"{statistics.mean(deltas):+.2f}", (statistics.mean(deltas), index),
                         xytext=(0, -17), textcoords="offset points", ha="center", fontsize=9, color=color)
    axes[0].set_yticks(range(len(keys)), [f"{STYLES[k][0]}\n{stats[k]['parameters'] / 1e6:.2f}M parameters" for k in keys])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Final validation accuracy (%) · mean ± sample SD")
    axes[1].set_xlabel("Paired gain versus plain width 192\n(percentage points)")
    axes[1].axvline(0, color="#777777", linewidth=1, linestyle="--", zorder=0)
    for ax in axes:
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.margins(x=0.16, y=0.12)
    fig.suptitle("CIFAR-10 ViT · independent-seed confirmation", x=0.06, y=0.97, ha="left", fontsize=18)
    fig.text(0.06, 0.90, "50 epochs · seeds 43–45 · settings frozen after tuning · 5,000-image validation split", color="#555555")
    fig.text(0.06, 0.11, "Small dots: individual seeds. Larger markers: means. SD is seed spread, not a confidence interval.", fontsize=10, color="#555555")
    fig.text(0.06, 0.06, "Controls approximate parameter budgets; see report for actual parameters and estimated FLOPs. Official test untouched.", fontsize=9, color="#555555")
    export(fig, directory, "confirmation")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.71, bottom=0.20, wspace=0.15)
    comparisons = [[k for k in keys if not k.startswith("plain_w")],
                   [k for k in ("plain", "plain_w216", "plain_w240") if k in keys]]
    for ax, comparison in zip(axes, comparisons):
        for key in comparison:
            label, color, marker = STYLES[key]
            curves = [r["validation_curve"] for r in grouped[key]]
            epochs = [p["epoch"] for p in curves[0]]
            if any([p["epoch"] for p in curve] != epochs for curve in curves):
                raise ValueError("Epoch alignment mismatch")
            mean = [statistics.mean(100 * curve[i]["accuracy"] for curve in curves) for i in range(len(epochs))]
            sd = [statistics.stdev(100 * curve[i]["accuracy"] for curve in curves) for i in range(len(epochs))]
            ax.plot(epochs, mean, color=color, label=label, linewidth=1.8,
                    marker=marker, markersize=4, markevery=5)
            ax.fill_between(epochs, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)],
                            color=color, alpha=0.12, linewidth=0)
        ax.set_xlim(1, 50)
        ax.set_ylim(0, 100)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_xlabel("Training epoch")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncol=2, frameon=False, fontsize=9)
    axes[0].set_ylabel("Validation accuracy (%)")
    fig.suptitle("CIFAR-10 ViT · confirmation learning curves", x=0.08, y=0.97, ha="left", fontsize=18)
    fig.text(0.08, 0.91, "50 epochs · seeds 43–45 · mean curves with ±1 sample SD bands", color="#555555")
    fig.text(0.08, 0.10, "Bands describe initialization/data-order seed spread, not confidence intervals. Same held-out split in every run.", fontsize=10, color="#555555")
    fig.text(0.08, 0.05, "Source: per-epoch validation events and final results in the preserved confirmation evidence.", fontsize=9, color="#777777")
    export(fig, directory, "confirmation_learning_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    args = parser.parse_args()
    report = json.loads((args.workflow / "report.json").read_text())
    if not all(report[stage]["all_complete"] for stage in ("tuning", "confirmation")):
        raise ValueError("Only completed, audited evidence can be plotted")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False})
    tuning_figure(report, args.workflow)
    confirmation_figures(report, args.workflow)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Scientific figures from completed tuning and confirmation evidence",
        "source_report": str((args.workflow / "report.json").resolve()),
        "source_report_sha256": hashlib.sha256((args.workflow / "report.json").read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "uncertainty": "sample SD across three seeds, not a confidence interval",
        "figures": {name: hashlib.sha256((args.workflow / name).read_bytes()).hexdigest()
                    for stem in ("tuning", "confirmation", "confirmation_learning_curves")
                    for extension in ("png", "pdf") for name in (f"{stem}.{extension}",)},
    }
    (args.workflow / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.workflow / "confirmation.png")


if __name__ == "__main__":
    main()

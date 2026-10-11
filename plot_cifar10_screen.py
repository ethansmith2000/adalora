#!/usr/bin/env python3
"""Export scientific learning-curve figures from a screen's real metrics."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from summarize_cifar10_screen import summarize


STYLES = {
    "plain": ("Plain, width 192", "#333333", "-", "o"),
    "diagonal": ("Diagonal", "#376C9B", "-", "s"),
    "basis-r8-k16": ("Basis r8, K16", "#B88B25", "-", "^"),
    "dplr-r8-k16": ("DPLR r8, K16", "#748544", "-", "D"),
    "exact-r2": ("Exact r2", "#BF6B35", "--", "v"),
    "static-lora-r8-diagonal": ("Static LoRA r8 + diagonal", "#B0688B", "--", "P"),
    "plain-w216": ("Plain, width 216", "#376C9B", "--", "s"),
    "plain-w264": ("Plain, width 264", "#B88B25", ":", "^"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    args = parser.parse_args()
    summary = summarize(args.screen)
    manifest = json.loads(args.screen.read_text())
    rows = {r["name"]: r for r in summary["rows"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 12, "axes.titlesize": 13,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.68, bottom=0.19, wspace=0.12)
    comparisons = [(["plain", "diagonal", "exact-r2", "basis-r8-k16", "dplr-r8-k16", "static-lora-r8-diagonal"],
                    "Conditioning variants · width 192"),
                   (["plain", "plain-w216", "plain-w264"], "Plain ViT capacity controls")]
    for ax, (names, title) in zip(axes, comparisons):
        for name in names:
            points = rows[name]["validation_curve"]
            label, color, style, marker = STYLES[name]
            ax.plot([p["epoch"] for p in points], [100 * p["accuracy"] for p in points],
                    label=label, color=color, linestyle=style, linewidth=1.8,
                    marker=marker, markersize=3.5)
        ax.set_title(title, loc="left", pad=70)
        ax.set_xlim(1, manifest["epochs"])
        ax.set_ylim(0, 100)
        if manifest["epochs"] == 20:
            ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("Training epoch")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.01), ncol=2,
                  frameon=False, fontsize=10, columnspacing=1.0, handlelength=2.2)
    axes[0].set_ylabel("Validation accuracy (%)")
    fig.suptitle("CIFAR-10 · token-conditioned ViT", x=0.08, y=0.97, ha="left", fontsize=18)
    state = "completed screen" if summary["all_complete"] else "live snapshot · incomplete curves"
    fig.text(0.08, 0.91, f"{manifest['epochs']} epochs · seed {manifest['seed']} · 45,000 train / 5,000 validation · {state}",
             fontsize=11, color="#555555")
    fig.text(0.08, 0.09, "Single-seed screening; no uncertainty intervals or final quality claim. Test set untouched.",
             fontsize=10, color="#555555")
    fig.text(0.08, 0.05, "Source: per-epoch validation events in this screen's metrics.jsonl files.",
             fontsize=9, color="#777777")
    # A small header-anchored blossom following the research chart style.
    for angle in range(0, 360, 72):
        radians = math.radians(angle)
        fig.add_artist(Circle((0.963 + 0.006 * math.cos(radians), 0.955 + 0.013 * math.sin(radians)),
                              0.006, transform=fig.transFigure, color="#B0688B", alpha=0.8))
    for extension in ("png", "pdf"):
        fig.savefig(args.screen.parent / f"learning_curves.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    print(args.screen.parent / "learning_curves.png")


if __name__ == "__main__":
    main()

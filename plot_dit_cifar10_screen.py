#!/usr/bin/env python3
"""Export scientific figures from an audited diffusion pilot."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLES = {
    "adaln_d192": ("AdaLN d192", "#333333", "o"),
    "static_r8_d192": ("Static LoRA r8", "#B0688B", "P"),
    "exact_r2_d192": ("Exact r2", "#777777", "X"),
    "dplr_r8_k16_d192": ("DPLR r8 K16", "#748544", "D"),
    "adaln_d216": ("AdaLN d216", "#BF6B35", "^"),
    "adaln_d240": ("AdaLN d240", "#376C9B", "s"),
}


def export(fig, prefix):
    for extension in ("png", "pdf"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=Path("reports/dit-cifar10-pilot"))
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    if report["phase"] != "pilot" or not report["audit"]["all_runs_complete"]:
        raise ValueError("Plot only the completed, audited quality pilot")
    rows = {row["name"]: row for row in report["rows"]}
    if set(rows) != set(STYLES):
        raise ValueError("Unexpected comparison set")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.23, top=0.67, wspace=0.23)
    for name, (label, color, marker) in STYLES.items():
        row = rows[name]
        for ax, x in ((axes[0], row["parameters"] / 1e6),
                      (axes[1], row["forward_matmul_flops_per_image_estimate"] / 1e6)):
            ax.scatter(x, row["generative_evaluation"]["fid"], label=label, color=color, marker=marker, s=65)
    axes[0].set_xlabel("Trainable parameters (millions)")
    axes[1].set_xlabel("Estimated forward matmul MFLOPs/image")
    for ax in axes:
        ax.set_ylabel("Held-out FID · lower is better")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.margins(x=0.12, y=0.15)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("CIFAR-10 DiT · broadcast conditioning pilot", x=0.08, y=0.97, ha="left", fontsize=18)
    fig.text(0.08, 0.90, "10,000 optimizer steps · seed 42 · class + timestep · common AdaLN-Zero-style blocks", color="#555555")
    fig.text(0.08, 0.14, "EMA sampling: DDIM 50, no CFG · 5,000 generated / 5,000 held-out real images · official test untouched", fontsize=10, color="#555555")
    fig.text(0.08, 0.08, "One run per model; no uncertainty estimate. FID is not directly comparable to published 50k-image evaluations.", fontsize=9, color="#555555")
    fig.text(0.08, 0.035, "FLOPs exclude elementwise work, normalization and backward. Width controls approximate parameter budgets.", fontsize=9, color="#777777")
    export(fig, prefix)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.67, wspace=0.23)
    for name, (label, color, marker) in STYLES.items():
        curve = rows[name]["validation_curve"]
        steps = [v["step"] for v in curve if v["step"]]
        for ax, state in zip(axes, ("raw", "ema")):
            ax.plot(steps, [v[state]["noise_mse"] for v in curve if v["step"]],
                    label=label, color=color, marker=marker, markersize=5, linewidth=1.6)
            ax.set_title("Raw weights" if state == "raw" else "EMA weights", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Optimizer steps")
        ax.set_ylabel("Held-out noise MSE · lower is better")
        ax.set_xticks([2500, 5000, 7500, 10000])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("CIFAR-10 DiT · denoising learning curves", x=0.08, y=0.97, ha="left", fontsize=18)
    fig.text(0.08, 0.90, "45,000 train / 5,000 validation · fixed validation noise and timestep seed 2027 · same split in every run", color="#555555")
    fig.text(0.08, 0.13, "Initial noise MSE is approximately 1 for every zero-output model; step 0 omitted to show the learning region.", fontsize=9, color="#555555")
    fig.text(0.08, 0.075, "Single seed and common untuned diffusion recipe. Exact adapters use LR multiplier 0.1 transferred from ViT diagnostics.", fontsize=9, color="#555555")
    fig.text(0.08, 0.03, "Denoising loss and global FID do not measure generated class accuracy or sequence-conditioned attention.", fontsize=9, color="#777777")
    curves_prefix = prefix.with_name(prefix.name + "-curves")
    export(fig, curves_prefix)
    figures = [p.with_suffix("." + ext) for p in (prefix, curves_prefix) for ext in ("png", "pdf")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_report": str(args.summary.resolve()),
                "source_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib_version": matplotlib.__version__, "uncertainty": "none; single-seed pilot",
                "figures": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in figures}}
    prefix.with_name(prefix.name + "-figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()

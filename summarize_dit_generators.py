#!/usr/bin/env python3
"""Export an audited generator/core comparison without selecting a quality winner."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from benchmark_dit_generators import generator_cases


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(workflow, prefix="dit-generator-engineering"):
    if Path(prefix).name != prefix:
        raise ValueError("Prefix must be a filename stem")
    workflow = Path(workflow).resolve()
    report = json.loads((workflow / "report.json").read_text())
    audit = json.loads((workflow / "audit.json").read_text())
    assert audit["checks"]["successful_worker_exit"] and audit["checks"]["complete"]
    assert audit["report_sha256"] == sha(workflow / "report.json")
    settings = report["settings"]
    root = Path(json.loads((workflow / "protocol.json").read_text())["repo_root"])
    target = root / "reports"
    target.mkdir(exist_ok=True)
    rows = {row["name"]: row for row in report["rows"]}
    order = [case["name"] for case in generator_cases(settings["width"])]
    base = rows["adaln"]["parameters"]
    def mib(value):
        return "n/a" if value is None else f"{value / 2**20:.2f}"
    def times(probe):
        return " / ".join(f"{mode['milliseconds_per_forward_backward']:.3f}" for mode in probe["modes"])
    lines = ["# DiT generator widths and conditional cores", "",
             f"{settings['steps']} steps/case, seed 42; {settings['bank_examples']:,} cached train images, "
             f"batch {settings['batch_size']}, width {settings['width']}, depth {settings['depth']}. "
             "One sequential claimed GPU. This screen measures learning and engineering costs, not quality rankings.", "",
             "All adapters use normalized input, common AdaLN-Zero and a fixed 1/sqrt(outer rank) gain except "
             "the explicit raw basis control. Adapter LR multiplier is 1; these new recipes are not function-update calibrated. "
             f"Warmup {settings['warmup']}, base LR {settings['lr']}, cosine horizon {settings['schedule_steps']:,} steps. "
             "Raw validation noise MSE is a short-run learning diagnostic; no EMA/FID/class-agreement evaluation.", "",
             "| Case | Total parameters | Added versus AdaLN | Peak training MiB | Train images/s | Final mean delta/base RMS |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in order:
        row = rows[name]
        branches = row["diagnostics"][-1]["branches"]
        magnitude = statistics.mean(branch["delta_to_base_rms"] for branch in branches)
        lines.append(f"| {name} | {row['parameters']:,} | {row['parameters']-base:,} | "
                     f"{mib(row['peak_training_allocated_bytes'])} | {row['train_images_per_second']:,.0f} | {magnitude:.3f} |")
    lines += ["", "The FFN control adds capacity through MLP hidden width, so its adapter parameter count is zero. "
              "At width 192/depth 6 its total parameter count exceeds basis r2 by only 384. "
              "Equal-shape variants have identical initial base hashes. The expanded FFN is seed-matched but changes "
              "initialization through different shapes; it does not have an identical base hash.", "",
              "Peak training allocation includes model, gradients, AdamW, training activations and fixed diagnostic inputs, "
              "and excludes EMA/diagnostic forwards. Timing includes synchronized cached CPU preparation, H2D, "
              "noise generation, forward/backward and optimizer; diagnostics and validation are outside the timer.", "",
              "| Case | Broadcast saved MiB | Token saved MiB | Broadcast eager / compiled ms | Token eager / compiled ms |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in order:
        row = rows[name]
        sample, token = row["projection_microbenchmark"], row["token_projection_microbenchmark"]
        lines.append(f"| {name} | {mib(sample['eager_saved_nonparameter_storage_bytes'])} | "
                     f"{mib(token['eager_saved_nonparameter_storage_bytes'])} | {times(sample)} | {times(token)} |")
    lines += ["", "Projection timings measure the trained first-block pre-activation path, including non-affine normalization, "
              "common modulation and base projection, with complete forward/backward but no optimizer. "
              "Token conditions are synthetic extensions of broadcast-trained weights, not a token-conditioned quality experiment. "
              "Saved storage counts unique eager nonparameter backing storage including inputs and autocast copies; "
              "it is not a whole-model peak or compiled activation measurement.", "",
              "Repeat timings, first-call latency and compiled output/gradient parity are retained in JSON. "
              "Compilation may reuse disk caches. Other projects share host resources even though this worker "
              "has one exclusively claimed GPU. A single order/seed and short training window do not establish "
              "a stable wall-clock winner or equal-time quality result.", "",
              "| Case | Final mean fixed-input adapter-delta update RMS | Raw validation noise MSE |",
              "| --- | ---: | ---: |"]
    for name in order:
        row = rows[name]
        update = statistics.mean(block["update_rms"] for block in row["diagnostics"][-1]["fixed_token_condition_function_updates"])
        lines.append(f"| {name} | {update:.6g} | {row['raw_validation']['noise_mse']:.6f} |")
    lines += ["", "The update column measures only the adapter delta, not the entire pre-activation or model. "
              "Zeros for AdaLN/FFN indicate an absent adapter; their base parameters still learn.", "",
              "Zero initial deltas and adapter gradient wake-up are audited. Distinct core/generator parameterizations "
              "have different gauges and initial optimization geometry. Scalar gains do not ensure comparable function updates "
              "or variance preservation. Compare these diagnostics before calibrating a longer quality study.", "",
              "No model weights, resume checkpoints, generated image datasets or official test evaluation. "
              "The evidence audit checks frozen source/configs, parameter/FLOP budgets, shared batches/noise, "
              "equal-shape initialization, numerical parity and successful completion.", "",
              f"[Full report]({prefix}.json) · [Audit]({prefix}-audit.json) · [Design notes](../DESIGN_NOTES.md)", ""]
    contents = {f"{prefix}.md": "\n".join(lines).encode(), f"{prefix}.json": (workflow / "report.json").read_bytes(),
                f"{prefix}-audit.json": (workflow / "audit.json").read_bytes()}
    for name, body in contents.items():
        path = target / name
        if path.exists() and path.read_bytes() != body:
            raise FileExistsError(f"Preserve existing evidence with another prefix: {path}")
        path.write_bytes(body)
    analysis = workflow / "analysis_source"
    analysis.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), analysis / Path(__file__).name)
    manifest = dict(workflow=str(workflow), report_sha256=sha(workflow / "report.json"),
                    audit_sha256=sha(workflow / "audit.json"), summary_script_sha256=sha(Path(__file__)),
                    artifacts={name: dict(sha256=sha(target / name), bytes=(target / name).stat().st_size) for name in contents})
    (target / f"{prefix}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return str(target / f"{prefix}.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--prefix", default="dit-generator-engineering")
    args = parser.parse_args()
    print(json.dumps(dict(summary=summarize(args.workflow, args.prefix))))

#!/usr/bin/env python3
"""Export an audited design engineering report and reproducible comparison tables."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ORDER = ["adaln", "bottleneck_r2", "bottleneck_r4", "basis_r2_normalized", "basis_r2_modulated",
         "basis_r4_normalized", "basis_r4_modulated", "dplr_r8_reference"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--prefix", default="dit-design-engineering")
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix:
        raise ValueError("Prefix must be a filename stem")
    workflow = args.workflow.resolve()
    report = json.loads((workflow / "report.json").read_text())
    audit = json.loads((workflow / "audit.json").read_text())
    assert audit["checks"]["successful_worker_exit"] and audit["checks"]["complete"]
    assert audit["report_sha256"] == sha(workflow / "report.json")
    root = Path(json.loads((workflow / "protocol.json").read_text())["repo_root"])
    target = root / "reports"
    rows = {r["name"]: r for r in report["rows"]}
    settings = report["settings"]
    lines = ["# DiT adapter design engineering", "",
             f"{settings['steps']} steps per case, seed 42, one sequential claimed GPU; "
             f"{settings['bank_examples']:,} cached train images sampled with replacement. "
             "This is an engineering screen, not a quality ranking.", "",
             "| Case | Extra parameters | Peak training MiB | Mean block delta/base RMS | Train images/s |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ORDER:
        r = rows[name]
        branches = r["diagnostics"][-1]["branches"]
        gain = sum(b["delta_to_base_rms"] for b in branches) / len(branches)
        lines.append(f"| {name} | {r['adapter_parameters']:,} | {r['peak_training_allocated_bytes']/2**20:.1f} | {gain:.3f} | {r['train_images_per_second']:,.0f} |")
    lines += ["", "Peak training allocation excludes diagnostic forwards and EMA. The gain column "
              "is the arithmetic mean of each block's fixed-image delta/base RMS ratio at the final step, "
              "not an operator-norm bound. Training steps include cached CPU preparation and are individually synchronized.", "",
              "| Case | Broadcast saved MiB | Token saved MiB | Broadcast eager / compiled ms | Token eager / compiled ms |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in ORDER:
        r = rows[name]
        sample, token = r["projection_microbenchmark"], r["token_projection_microbenchmark"]
        def times(probe):
            return " / ".join(f"{m['milliseconds_per_forward_backward']:.3f}" for m in probe['modes'])
        lines.append(f"| {name} | {sample['eager_saved_nonparameter_storage_bytes']/2**20:.2f} | {token['eager_saved_nonparameter_storage_bytes']/2**20:.2f} | {times(sample)} | {times(token)} |")
    lines += ["", "Projection timings are host-wall-clock medians of repeated complete forward/backward loops; "
              "repeat values and first-call latency are in the JSON. Compiler disk caches can be reused, so "
              "first-call latency is not necessarily a cold-cache build. Other projects share host resources. "
              "These values do not establish a stable speed ordering or end-to-end compiled training gain.", "",
              "Saved MiB count unique eager nonparameter backing storage, including inputs and autocast weight copies. "
              "They describe the first-block projection with the common six-output modulation calculation, "
              "not adapter-only activation storage or compiled peak memory. "
              "Token probes are synthetic extensions of trained broadcast weights, not trained self-conditioned models.", "",
              "The bottleneck reduces adapter parameters by about 93–94% relative to matched-rank basis mixing. "
              "Its output gain is substantially smaller under the shared LR/initialization recipe. "
              "Gain and functional-update controls are needed before attributing quality differences to capacity. "
              "No quality winner is selected from the short held-out noise-MSE diagnostic.", "",
              "All cases preserve zero initial deltas and open their adapter gradients. "
              "The audit verifies shared streams, paired initialization, frozen source, budgets, finite evidence, "
              "and compiled numerical checks. No model weights/checkpoints or official test evaluation.", "",
              f"[Full report]({args.prefix}.json) · [Audit]({args.prefix}-audit.json) · "
              "[Design notes](../DESIGN_NOTES.md)", ""]
    contents = {f"{args.prefix}.json": (workflow / "report.json").read_bytes(),
                f"{args.prefix}-audit.json": (workflow / "audit.json").read_bytes(),
                f"{args.prefix}.md": "\n".join(lines).encode()}
    for name, data in contents.items():
        path = target / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Use another --prefix to preserve existing evidence: {path}")
        path.write_bytes(data)
    analysis = workflow / "analysis_source"
    analysis.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), analysis / Path(__file__).name)
    manifest = {"workflow": str(workflow), "report_sha256": sha(workflow / "report.json"),
                "audit_sha256": sha(workflow / "audit.json"), "summary_script_sha256": sha(Path(__file__)),
                "artifacts": {name: {"sha256": sha(target / name), "bytes": (target / name).stat().st_size} for name in contents}}
    (target / f"{args.prefix}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"summary": str(target / f"{args.prefix}.md"), "manifest": str(target / f"{args.prefix}-manifest.json")}))


if __name__ == "__main__":
    main()

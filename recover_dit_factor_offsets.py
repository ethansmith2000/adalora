#!/usr/bin/env python3
"""Recover reports from the specific offset-flag comparison failure, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def recover(output):
    output = Path(output).resolve()
    original = json.loads((output / "protocol.json").read_text())
    if (original["state"] != "failed" or original.get("failure_type") != "ValueError"
            or original.get("failure") != "Training or evaluation settings differ across variants"):
        raise ValueError("Recovery requires the recorded offset comparison failure")
    if any((output / name).exists() for name in ("controller.sock", "preflight/supervisor.sock", "quality/supervisor.sock")):
        raise ValueError("Close this study's private supervisors before reporting recovery")
    recovery = output / "reporting-recovery"
    recovery.mkdir()  # Deliberately reject an implicit overwrite/retry.
    shutil.copy2(output / "protocol.json", recovery / "failed_protocol.json")
    for name in ("controller.log", "factor_offset_controller.log"):
        shutil.copy2(output / name, recovery / name)
    runner = recovery / "runner"
    shutil.copytree(output / "runner", runner, ignore=shutil.ignore_patterns("__pycache__"))
    source = Path(__file__).resolve().parent
    shutil.copy2(source / "summarize_dit_cifar10_screen.py", runner / "summarize_dit_cifar10_screen.py")
    for name in (Path(__file__).name, "test_dit_screen_config.py"):
        shutil.copy2(source / name, recovery / name)
    changed = [p.name for p in runner.iterdir() if p.is_file()
               and digest(p) != original["source_sha256"][p.name]]
    assert changed == ["summarize_dit_cifar10_screen.py"]
    record = dict(state="reporting", started_utc=datetime.now(timezone.utc).isoformat(),
                  purpose="Repair the architectural offset flag in the comparison checker; replay existing evidence only",
                  failure_protocol_sha256=digest(recovery / "failed_protocol.json"),
                  original_failure=dict(utc=original["failed_utc"], type=original["failure_type"], message=original["failure"]),
                  original_controller_exit_status=1, controller_closed=True,
                  reporting_only_changes=changed,
                  original_training_source_sha256=original["source_sha256"],
                  reporting_runner_sha256={p.name: digest(p) for p in runner.iterdir() if p.is_file()},
                  recovery_script_sha256=digest(recovery / Path(__file__).name),
                  regression_test_sha256=digest(recovery / "test_dit_screen_config.py"),
                  retrained=False, gpu_jobs_started=0, training_and_evaluation_artifacts_changed=False)
    write(recovery / "manifest.json", record)
    # Models and auditor remain the frozen versions; only reporting is patched.
    sys.path.insert(0, str(runner))
    import dit_factor_offset_study as study
    assert Path(study.__file__).resolve() == runner / "dit_factor_offset_study.py"
    import summarize_dit_cifar10_screen as summarizer
    assert Path(summarizer.__file__).resolve() == runner / "summarize_dit_cifar10_screen.py"
    try:
        reports = {}
        for stage in ("preflight", "quality"):
            summary = summarizer.summarize(output / stage / "screen.json")
            screen = json.loads((output / stage / "screen.json").read_text())
            for row in summary["rows"]:
                row["family"] = next(j["family"] for j in screen["jobs"] if j["name"] == row["name"])
            reports[stage] = summary
            write(recovery / f"{stage}_summary.json", summary)
        study.aggregate(reports["quality"]["rows"])
        write(output / "report.json", reports)
        protocol = {**original, "state": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "reporting_recovery": str(recovery / "manifest.json")}
        # As in the original workflow, failed audit restores a failed state.
        write(output / "protocol.json", protocol)
        audited = study.audit(output)
        study.write_summary(output, reports["quality"]["rows"], audited, protocol)
        assert all(audited["checks"].values())
        record.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                      audit_passed=True, report_sha256=digest(output / "report.json"),
                      audit_sha256=digest(output / "audit.json"))
        write(recovery / "manifest.json", record)
        print(json.dumps(dict(output=str(output), audit=audited)), flush=True)
    except BaseException as error:
        record.update(state="failed", recovery_failure_type=type(error).__name__, recovery_failure=str(error))
        write(recovery / "manifest.json", record)
        write(output / "protocol.json", {**original, "reporting_recovery": str(recovery / "manifest.json")})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    recover(parser.parse_args().output)

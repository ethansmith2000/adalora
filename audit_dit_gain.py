#!/usr/bin/env python3
"""Audit bounded calibration, confirmation replay, frozen sources and artifacts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_dit_design import finite
from benchmark_dit_gain import FAMILIES, MAX_CORRECTIONS, RATE_BOUNDS, TOLERANCE, calibration_plan
from dit_cifar10 import CIFAR10DiT
from train_dit_cifar10 import write_json


def metric(row, steps):
    events={e["step"]:e for e in row["diagnostics"]}
    return statistics.mean(v["update_rms"] for s in steps for v in events[s]["fixed_token_condition_function_updates"])


def audit(output, require_exit=True):
    output=Path(output)
    protocol=json.loads((output/"protocol.json").read_text())
    report=json.loads((output/"report.json").read_text())
    settings=json.loads((output/"settings.json").read_text())
    assert protocol["state"]=="complete"
    assert settings==protocol["settings"]==report["settings"]
    assert calibration_plan()==protocol["calibration_plan"]==settings["calibration_plan"]==report["plan"]
    finite(report)
    assert not any(protocol[k] for k in ("weights_saved","periodic_checkpoints","test_evaluated","ema"))
    if require_exit:
        assert f"exited: {protocol['service']} (exit status 0; expected)" in (output/"supervisord.log").read_text()
    for name,sha in protocol["source_sha256"].items():
        assert hashlib.sha256((output/"source"/name).read_bytes()).hexdigest()==sha
    assert all(protocol["source_sha256"][name]==sha for name,sha in report["runtime"]["source_sha256"].items())
    assert report["quality_ranking_supported"] is False
    assert len(report["confirmation"])==6
    assert [r["name"] for r in report["confirmation"]]==settings["confirmation_order"]
    assert set(settings["confirmation_order"])=={f[0] for f in FAMILIES}|{"basis_r8_raw"}
    pilots={p["candidate"]["name"]:p for p in report["calibration"]}
    assert len(pilots)==len(report["calibration"])<=1+(len(FAMILIES)-1)*(MAX_CORRECTIONS+1)
    target=report["target_update_rms"]
    assert target==metric(pilots["basis_r8_candidate0"]["row"],settings["calibration_observation_steps"])>0
    all_rows=[("calibration",p["row"]) for p in pilots.values()]+[("confirmation",r) for r in report["confirmation"]]
    for phase,row in all_rows:
        folder=output/phase/row["name"]
        assert row==json.loads((folder/"results.json").read_text())
        cfg=json.loads((folder/"config.json").read_text())
        assert cfg["adapter_gain"]==row["adapter_gain"] and cfg["adapter_lr_multiplier"]==row["adapter_lr_multiplier"]
        assert row["runtime"]==report["runtime"]
        assert row["fixed_function_probe_precision"]=="FP32 tensors and highest FP32 matmul precision; no autocast"
        steps=settings["calibration_steps"] if phase=="calibration" else settings["steps"]
        assert row["completed_steps"]==steps and row["processed_images"]==steps*settings["batch_size"]
        assert cfg["schedule_steps"]==settings["steps"] and cfg["warmup"]==settings["warmup"]
        assert not any(row[k] for k in ("weights_saved","test_evaluated","ema_used","quality_ranking_supported"))
        assert row["projection_microbenchmark"] is row["token_projection_microbenchmark"] is None
        model=CIFAR10DiT(settings["width"],settings["depth"],settings["heads"],settings["patch_size"],
                         variant=row["variant"],rank=row["rank"],adapter_gain=row["adapter_gain"])
        stats=model.statistics()
        assert all(row[k]==stats[k] for k in ("parameters","adapter_parameters","forward_matmul_flops_per_image_estimate"))
        assert set(row["adapter_first_nonzero_gradient_observed"])=={n for n,_ in model.named_parameters() if ".adapter." in n}
        events=[json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
        finite(events)
        assert events[-1]=={"event":"complete","step":steps}
        assert row["diagnostics"]==[e for e in events if e["event"]=="diagnostic"]
        assert all(b["delta_rms"]==b["gate_rms"]==0 for b in row["diagnostics"][0]["branches"])
        for e in row["diagnostics"][1:]:
            assert math.isclose(e["lr_by_group"]["adapter_input"]/e["lr_by_group"]["base"],row["adapter_lr_multiplier"],rel_tol=1e-12)
        final=row["diagnostics"][-1]["gradient_groups_after_clip"]
        assert all(v["parameter_tensors"]==len(v["nonzero_gradient_tensors"]) for k,v in final.items() if k!="base")
    assert len({r["base_initialization_sha256"] for _,r in all_rows})==1
    for phase in ("calibration","confirmation"):
        for k in ("stream_sha256","first_noise_sha256","noise_rng_final_state_sha256"):
            assert len({r[k] for p,r in all_rows if p==phase})==1
    for family,variant,rank in FAMILIES:
        candidates=sorted((p["candidate"] for p in pilots.values() if p["candidate"]["family"]==family),key=lambda c:c["attempt"])
        assert [c["attempt"] for c in candidates]==list(range(len(candidates)))
        assert candidates[0]["adapter_lr_multiplier"]==1
        for i,c in enumerate(candidates):
            p=pilots[c["name"]]["row"]
            assert (p["variant"],p["rank"],p["adapter_input"])==(variant,rank,"normalized")
            assert c["gain"]==p["adapter_gain"]==1/math.sqrt(rank)
            assert c["metric"]==metric(p,settings["calibration_observation_steps"])
            assert c["metric_to_target"]==c["metric"]/target
            assert c["within_tolerance"]==(abs(c["metric"]/target-1)<=TOLERANCE)
            if i:
                prev=candidates[i-1]
                assert not prev["within_tolerance"]
                assert c["adapter_lr_multiplier"]==min(RATE_BOUNDS[1],max(RATE_BOUNDS[0],prev["adapter_lr_multiplier"]*target/prev["metric"]))
        best=min(candidates,key=lambda c:abs(math.log(c["metric"]/target)))
        assert report["selected"][family]==best
        confirmation=next(r for r in report["confirmation"] if r["name"]==family)
        pilot=pilots[best["name"]]["row"]
        assert confirmation["adapter_gain"]==best["gain"] and confirmation["adapter_lr_multiplier"]==best["adapter_lr_multiplier"]
        assert confirmation["calibration_candidate"]==best["name"]
        assert confirmation["adapter_initialization_sha256"]==pilot["adapter_initialization_sha256"]
        assert math.isclose(confirmation["early_update_metric"],best["metric"],rel_tol=1e-6,abs_tol=1e-12)
        for step in settings["calibration_observation_steps"]:
            ce=next(e for e in confirmation["diagnostics"] if e["step"]==step)
            pe=next(e for e in pilot["diagnostics"] if e["step"]==step)
            assert ce["training_stream_sha256"]==pe["training_stream_sha256"]
            assert ce["noise_rng_state_sha256"]==pe["noise_rng_state_sha256"]
    raw=next(r for r in report["confirmation"] if r["name"]=="basis_r8_raw")
    assert raw["variant"]=="dplr" and raw["rank"]==8 and raw["adapter_gain"]==raw["adapter_lr_multiplier"]==1
    assert raw["adapter_initialization_sha256"]==pilots["basis_r8_candidate0"]["row"]["adapter_initialization_sha256"]
    metadata=report["dataset"]
    if not settings["synthetic"]:
        assert len(set(metadata["train_indices"]))==settings["bank_examples"]
        assert len(set(metadata["validation_indices"]))==settings["val_examples"]
        assert not set(metadata["train_indices"]) & set(metadata["validation_indices"])
        assert metadata["test_evaluated"] is False
    assert not [p for p in output.rglob("*") if p.suffix in (".pt",".pth",".ckpt",".safetensors")]
    retained=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    assert retained<protocol["expected_storage_upper_bound_bytes"]
    result={"checks":{"complete":True,"successful_worker_exit":require_exit,"source_verified":True,
                       "bounded_nonquality_calibration":True,"early_stream_replay":True,
                       "same_batches_noise_and_base_init":True,"finite_metrics":True,"zero_init_and_wakeup":True,
                       "weights_saved":False,"test_evaluated":False},
            "selected_within_tolerance":{k:v["within_tolerance"] for k,v in report["selected"].items()},
            "bytes_retained":retained,"report_sha256":hashlib.sha256((output/"report.json").read_bytes()).hexdigest()}
    write_json(output/"audit.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.output),indent=2))

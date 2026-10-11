#!/usr/bin/env python3
"""Audit fresh-seed recipe transfer, quality denominators and live diagnostics."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from class_fidelity import classifier_provenance
from dit_gain_quality import recipes
from dit_cifar10 import CIFAR10DiT
from summarize_dit_cifar10_screen import summarize
from train_dit_cifar10 import write_json


def audit(output):
    output=Path(output).resolve()
    protocol=json.loads((output/"protocol.json").read_text())
    assert protocol["state"]=="complete"
    assert protocol["max_concurrent_workers"]==3
    assert not any(protocol[k] for k in ("weights_saved","periodic_checkpoints","test_evaluated"))
    report=json.loads((output/"report.json").read_text())
    original=summarize(output/"quality/screen.json")
    assert report["rows"]==original["rows"]
    assert {k:v for k,v in report["audit"].items() if k!="bytes_retained_under_screen"}=={
        k:v for k,v in original["audit"].items() if k!="bytes_retained_under_screen"}
    assert report["quality_winner_confirmed"] is False
    frozen=json.loads((output/"recipes.json").read_text())
    assert frozen==report["recipe_provenance"]
    assert datetime.fromisoformat(frozen["frozen_utc"])<datetime.fromisoformat(protocol["started_utc"])
    calibration=Path(frozen["calibration_run"])
    assert hashlib.sha256((calibration/"report.json").read_bytes()).hexdigest()==frozen["calibration_report_sha256"]
    assert frozen["selected"]==json.loads((calibration/"report.json").read_text())["selected"]
    expected={name:(variant,width,rank,gain,multiplier) for name,variant,width,rank,gain,multiplier in recipes(frozen["selected"])}
    assert set(expected)=={r["name"] for r in report["rows"]} and len(report["rows"])==9
    for name,sha in protocol["source_sha256"].items():
        for folder in ("runner","quality/source"):
            assert hashlib.sha256((output/folder/name).read_bytes()).hexdigest()==sha
    current_classifier=classifier_provenance(protocol["auxiliary_classifier"]["run"])
    assert current_classifier==protocol["auxiliary_classifier"]
    real=[]; base_hashes=[]; noise_states=[]; batch_traces=[]
    for row in report["rows"]:
        cfg=row["config"]
        assert (cfg["variant"],cfg["width"],cfg["rank"],cfg["adapter_gain"],cfg["adapter_lr_multiplier"])==expected[row["name"]]
        assert cfg["seed"]==47 and cfg["max_steps"]==5000 and cfg["warmup_steps"]==50
        assert cfg["lr"]==4e-4 and cfg["adapter_input"]=="normalized" and cfg["num_workers"]==2
        assert cfg["sample_images"]==100 and cfg["fid_images"]==5000 and cfg["sampling_steps"]==50 and cfg["sampling_seed"]==2028
        assert cfg["train_limit"] is cfg["val_limit"] is None
        model=CIFAR10DiT(cfg["width"],cfg["depth"],cfg["heads"],cfg["patch_size"],variant=cfg["variant"],rank=cfg["rank"],adapter_gain=cfg["adapter_gain"])
        assert all(row[k]==model.statistics()[k] for k in ("parameters","adapter_parameters","forward_matmul_flops_per_image_estimate"))
        diagnostic=row["design_diagnostics"]
        assert diagnostic["fixed_function_probe_precision"]=="FP32 tensors and highest FP32 matmul precision; no autocast"
        assert diagnostic["model_state"]=="raw" and diagnostic["fixed_input_seed"]==2032
        events=diagnostic["diagnostics"]
        assert [d["step"] for d in events]==[int(s) for s in cfg["diagnostic_steps"].split(",")]
        recorded=[json.loads(line) for line in (Path(row["output_dir"])/"metrics.jsonl").read_text().splitlines()]
        assert events==[e for e in recorded if e["event"]=="design_diagnostic"]
        assert all(b["delta_rms"]==0 and b["gate_rms"]==0 for b in events[0]["branches"])
        names={name for name,_ in model.named_parameters() if ".adapter." in name}
        assert set(diagnostic["adapter_first_nonzero_gradient_observed"])==names
        for group,v in events[-1]["gradient_groups_after_clip"].items():
            if group!="base":
                assert len(v["nonzero_gradient_tensors"])==v["parameter_tensors"]
        if cfg["width"]==192:
            base_hashes.append(diagnostic["base_initialization_sha256"])
        noise_states.append(diagnostic["noise_rng_final_state_sha256"])
        batch_traces.append([{key:event[key] for key in ("step","batch_images_sha256","batch_labels_sha256","noise_sha256","timesteps_sha256","noise_rng_state_sha256")} for event in events[1:]])
        generated=row["generative_evaluation"]
        assert generated["class_agreement"]["classifier"]==current_classifier
        assert generated["generated_images"]==5000 and generated["fid_real_images"]==5000 and generated["fid_fake_images"]==5000
        assert generated["sample_grid_labels"]==[i%10 for i in range(100)]
        assert generated["model_state"]=="EMA"
        for state in ("real","generated"):
            metric=generated["class_agreement"][state]
            matrix=metric["confusion_rows_requested_columns_predicted"]
            assert metric["examples"]==5000 and metric["class_counts"]==[500]*10
            assert sum(sum(values) for values in matrix)==5000
            assert math.isclose(metric["agreement"],sum(matrix[i][i] for i in range(10))/5000,abs_tol=1e-7)
        real.append(generated["class_agreement"]["real"])
    assert len(set(base_hashes))==1 and len(set(noise_states))==1
    assert all(trace==batch_traces[0] for trace in batch_traces)
    assert all(r==real[0] for r in real)
    rows={r["name"]:r for r in report["rows"]}
    assert rows["basis_r8"]["design_diagnostics"]["adapter_initialization_sha256"]==rows["dplr_r8_raw"]["design_diagnostics"]["adapter_initialization_sha256"]
    assert not (output/"quality/supervisor.sock").exists()
    assert "exited: adalora_gain_quality_controller (exit status 0; expected)" in (output/"controller.log").read_text()
    assert not [p for p in output.rglob("*") if p.suffix in (".pt",".pth",".ckpt",".safetensors")]
    retained=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    assert retained<protocol["expected_storage_upper_bound_bytes"]
    result={"checks":{**report["audit"],"frozen_recipes_recomputed":True,"frozen_before_launch":True,
                      "same_diagnostic_batch_noise_digests":True,"same_noise_rng_end_state":True,
                      "same_width192_base_initialization":True,"paired_rank8_adapter_initialization":True,
                      "source_snapshots_verified":True,"classifier_hashes_and_denominators":True,
                      "same_real_classifier_predictions":True,"full_fp32_probes":True,"zero_init_and_wakeup":True},
            "classifier_real_accuracy":real[0]["agreement"],"bytes_retained":retained,
            "report_sha256":hashlib.sha256((output/"report.json").read_bytes()).hexdigest()}
    write_json(output/"audit.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.output),indent=2))

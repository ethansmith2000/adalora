#!/usr/bin/env python3
"""Frozen gain recipes, bounded fresh-seed quality screen, no model weights."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from class_fidelity import classifier_provenance
from dit_followup import private_supervisor
from train_dit_cifar10 import Config, SOURCE_FILES, load_config, provenance, write_json
from summarize_dit_cifar10_screen import summarize


EXTRA_SOURCE = ("dit_gain_quality.py", "dit_quality_diagnostics.py", "benchmark_dit_design.py",
                "benchmark_dit_gain.py", "audit_dit_gain.py", "audit_dit_gain_quality.py",
                "dit_followup.py", "summarize_dit_cifar10_screen.py", "test_dit_quality.py", "requirements.txt")


def recipes(selection):
    result = []
    for family, choice in selection.items():
        rank = int(family.rsplit("r",1)[1])
        variant = "conditional_bottleneck" if family.startswith("bottleneck") else "dplr"
        result.append((family,variant,192,rank,choice["gain"],choice["adapter_lr_multiplier"]))
    result += [("dplr_r8_raw","dplr",192,8,1.,1.)]
    result += [(f"adaln_d{width}","adaln_zero",width,8,1.,1.) for width in (192,216,240)]
    if len(result)!=9 or len({r[0] for r in result})!=9:
        raise ValueError("Expected five calibrated adapters, raw rank-8 anchor and three AdaLN controls")
    return result


def prepare(output, calibration, launch):
    root=Path(__file__).resolve().parent
    calibration=calibration.resolve()
    if output.exists():
        raise FileExistsError(output)
    audit=json.loads((calibration/"audit.json").read_text())
    report=json.loads((calibration/"report.json").read_text())
    if not audit["checks"]["successful_worker_exit"] or not audit["checks"]["complete"]:
        raise ValueError("Require the audited real-data calibration")
    if hashlib.sha256((calibration/"report.json").read_bytes()).hexdigest()!=audit["report_sha256"]:
        raise ValueError("Calibration report hash changed")
    if "highest FP32" not in report["plan"]["metric"]:
        raise ValueError("Use corrected full-FP32 calibration")
    if not all(c["within_tolerance"] for c in report["selected"].values()):
        raise ValueError("Every transferred family must satisfy declared calibration tolerance")
    free=shutil.disk_usage(root).free
    if free<1024**3:
        raise RuntimeError("Require 1 GiB free before the compact quality screen")
    prior=json.loads((root/"runs/dit-followup-20261007T174446Z/protocol.json").read_text())
    classifier=Path(prior["classifier_run"])
    auxiliary=classifier_provenance(classifier)
    cache=Path(Config().fid_weights)
    evaluator={"path":str(cache),"bytes":cache.stat().st_size,
               "sha256":hashlib.sha256(cache.read_bytes()).hexdigest(),"purpose":"existing frozen FID evaluator"}
    output.mkdir(parents=True)
    runner=output/"runner"; runner.mkdir()
    for name in (*SOURCE_FILES,*EXTRA_SOURCE):
        shutil.copy2(root/name,runner/name)
    frozen={"calibration_run":str(calibration),"calibration_report_sha256":audit["report_sha256"],
            "frozen_utc":datetime.now(timezone.utc).isoformat(),"selected":report["selected"],
            "selection_uses_quality":False,"transfer_limit":"subset seed-42 calibration to fresh full-data seed-47 quality training"}
    write_json(output/"recipes.json",frozen)
    configs=output/"configs"; configs.mkdir()
    jobs=[]
    diagnostics="0,1,3,5,10,25,50,100,500,1000,2500,5000"
    for name,variant,width,rank,gain,multiplier in recipes(frozen["selected"]):
        cfg=Config(variant=variant,width=width,rank=rank,adapter_gain=gain,adapter_lr_multiplier=multiplier,
                   seed=47,lr=4e-4,max_steps=5000,warmup_steps=50,eval_every=1000,log_every=100,
                   classifier_run=str(classifier),fid_images=5000,sample_images=100,num_workers=2,
                   diagnostic_steps=diagnostics,output_dir=str(output/"quality"/name))
        path=configs/f"{name}.json"
        write_json(path,asdict(cfg)); load_config(["--override_json",str(path)])
        jobs.append({"name":name,"config":str(path),"output_dir":cfg.output_dir})
    random.Random(79).shuffle(jobs)
    protocol={"state":"prepared","created_utc":datetime.now(timezone.utc).isoformat(),"repo_root":str(root),
              "purpose":"fresh-seed finite-budget generative quality comparison of gain/update-calibrated recipes",
              "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in runner.iterdir()},
              "calibration_run":str(calibration),"calibration_report_sha256":audit["report_sha256"],
              "recipes_frozen_before_launch":True,"jobs":jobs,"max_concurrent_workers":3,
              "train_steps":5000,"seed":47,"split_seed":123,"order_seed":79,
              "train_examples":45000,"validation_examples":5000,"fid_images":5000,
              "sampling_steps":50,"sampling_seed":2028,"warmup_steps":50,
              "evaluator_cache":evaluator,"auxiliary_classifier":auxiliary,
              "disk_free_bytes_before_launch":free,"expected_storage_upper_bound_bytes":256*1024**2,
              "weights_saved":False,"periodic_checkpoints":False,"test_evaluated":False,
              "ema":"in-memory only, decay .999","retention":"compact sources/configs/splits/metrics/logs/summaries/grids until explicitly removed",
              "large_artifacts":"none new; shared existing evaluator/classifier are referenced and preserved",
              "estimated_new_gpu_hours":"0.7–1.3 including validation/sampling; at most three claimed workers",
              "limitations":["one fresh training seed; short budget; screen does not establish a quality winner",
                             "all rates frozen from earlier studies; no architecture-specific quality LR tuning",
                             "subset update calibration may not transfer to full data/seed/schedule; measure rather than assume",
                             "concurrent throughput is not an isolated efficiency benchmark",
                             "broadcast class/time conditioning only; sequence-valued cross-attention remains untested"]}
    write_json(output/"protocol.json",protocol)
    if launch:
        private_supervisor(output,[("adalora_gain_quality_controller",[sys.executable,"-u",str(runner/"dit_gain_quality.py"),"run","--output",str(output)])],"controller",True)
    print(json.dumps({"output":str(output),"launched":launch}),flush=True)


def run(output):
    protocol=json.loads((output/"protocol.json").read_text())
    stage=output/"quality"; stage.mkdir()
    source=stage/"source"
    shutil.copytree(output/"runner",source,ignore=shutil.ignore_patterns("__pycache__"))
    jobs,programs=[],[]
    for job in protocol["jobs"]:
        service="adalora_gain_quality_"+output.name.replace("-","_")+"_"+job["name"]
        command=["/workspace/bin/gpu-claim","run","--owner","adalora","--job",service,"--wait","--",
                 sys.executable,"-u",str(source/"train_dit_cifar10.py"),"--override_json",job["config"]]
        jobs.append({**job,"service":service,"command":command})
        programs.append((service,command))
    screen={"phase":"gain_quality","jobs":jobs,"supervisor_control":str(stage/"supervisor.conf"),
            "evaluator_cache":protocol["evaluator_cache"],"source_sha256":protocol["source_sha256"],
            "protocol":{"train_examples":45000,"validation_examples":5000,"seed":47,"test_evaluated":False}}
    write_json(stage/"screen.json",screen)
    protocol.update(state="running",started_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/"protocol.json",protocol)
    cap=protocol["max_concurrent_workers"]
    submitted=set(j["service"] for j in jobs[:cap])
    control=private_supervisor(stage,programs,start_programs=[j["service"] for j in jobs[:cap]])
    claims={}
    while True:
        for path in Path("/tmp/vast_gpu_claims").glob("*.claim"):
            try:
                claim=json.loads(path.read_text())
            except (OSError,ValueError):
                continue
            if claim.get("job") in {j["service"] for j in jobs}:
                claims[claim["job"]]=claim
        write_json(stage/"hardware_claims.json",claims)
        status=subprocess.run(["supervisorctl","-c",control,"status"],capture_output=True,text=True).stdout
        if not status.strip():
            raise RuntimeError("Private supervisor returned no worker status")
        log=(stage/"supervisord.log").read_text()
        if any(term in status for term in ("FATAL","BACKOFF")):
            raise RuntimeError(status)
        completed={j["service"] for j in jobs if f"exited: {j['service']} (exit status 0; expected)" in log}
        failed=[j["service"] for j in jobs if j["service"] in submitted and
                "EXITED" in next((s for s in status.splitlines() if s.startswith(j["service"]+" ")),"") and j["service"] not in completed]
        if failed:
            raise RuntimeError(f"Workers failed; evidence preserved: {failed}")
        if any(not Path(j["output_dir"]).joinpath("results.json").exists() for j in jobs if j["service"] in completed):
            raise RuntimeError("A worker exited successfully without results")
        slots=cap-len(submitted-completed)
        pending=[j for j in jobs if j["service"] not in submitted]
        for job in pending[:slots]:
            subprocess.run(["supervisorctl","-c",control,"start",job["service"]],check=True)
            submitted.add(job["service"])
        print(json.dumps({"completed":len(completed),"submitted":len(submitted),"total":len(jobs)}),flush=True)
        if len(completed)==len(jobs):
            break
        time.sleep(20)
    subprocess.run(["supervisorctl","-c",control,"shutdown"],check=True)
    report=summarize(stage/"screen.json")
    report.update(recipe_provenance=json.loads((output/"recipes.json").read_text()),
                  quality_winner_confirmed=False,limits=protocol["limitations"])
    write_json(output/"report.json",report)
    protocol.update(state="complete",completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/"protocol.json",protocol)
    print(json.dumps({"event":"suite_complete","report":str(output/"report.json")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--calibration",type=Path,default=Path("runs/dit-gain-20261007T213229Z"))
    parser.add_argument("--launch",action="store_true")
    args=parser.parse_args()
    output=(args.output or Path(__file__).resolve().parent/"runs"/("dit-gain-quality-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    (prepare(output,args.calibration,args.launch) if args.action=="prepare" else run(output))

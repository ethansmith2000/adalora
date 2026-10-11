#!/usr/bin/env python3
"""Bounded adapter-LR controls, gated on the completed projector comparison."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

from class_fidelity import classifier_provenance
from dit_followup import private_supervisor
from dit_optimizer_cfg import DIAGNOSTICS, digest, execute_stage, mean_sd, now, same_stage_evidence
from dit_projector_confirmation import make_config as projector_config
from summarize_dit_cifar10_screen import summarize, check_finite
from train_dit_cifar10 import SOURCE_FILES, provenance, write_json

REFERENCE = "runs/dit-projector-confirmation-20261010T043230Z"
CAL_SEEDS = (90, 91)
QUALITY_SEEDS = (86, 87)
OBSERVATIONS = (50, 100, 250, 500)
CAL_DIAGNOSTICS = "0,1,3,5,10,25,50,100,250,500"
CONTROLS = ("affine_s16", "linear_s32_h64")
TARGETS = {"affine_s32": "affine_s16", "affine_s64": "affine_s16", "silu_s32_h64": "linear_s32_h64"}
RATE_GRID = {"affine_s32": (.25, .5, 1.), "affine_s64": (.125, .25, .5), "silu_s32_h64": (1., 2., 4.)}
TOLERANCE = .30
POINT_BOUNDS = (.5, 2.)
EXTRA_SOURCE = ("dit_projector_lr_controls.py", "test_dit_projector_lr_controls.py",
                "dit_projector_confirmation.py", "test_dit_projector_confirmation.py",
                "dit_capacity.py", "dit_optimizer_cfg.py", "dit_followup.py",
                "dit_quality_diagnostics.py", "benchmark_dit_design.py",
                "summarize_dit_cifar10_screen.py", "requirements.txt")


def trial_specs():
    return {f"{family}_trial{i}": (family, rate)
            for family, rates in RATE_GRID.items() for i, rate in enumerate(rates)}


def make_config(family, seed, rate, output, classifier, calibration=False):
    cfg = replace(projector_config(family, seed, output, classifier), adapter_lr_multiplier=rate)
    if calibration:
        cfg = replace(cfg, max_steps=500, eval_every=500, log_every=100,
                      diagnostic_steps=CAL_DIAGNOSTICS, fid_images=0, sample_images=0,
                      generative_eval_steps="", guidance_scales="", classifier_run=None)
    return cfg


def function_updates(row):
    events = {e["step"]: e for e in row["design_diagnostics"]["diagnostics"]}
    if len(events) != len(row["design_diagnostics"]["diagnostics"]):
        raise ValueError("Duplicate diagnostic observations")
    result = []
    for step in OBSERVATIONS:
        values = [v["update_rms"] for v in events[step]["fixed_token_condition_function_updates"]]
        if len(values) != 8 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Require eight finite, positive block function updates")
        result.append(statistics.mean(values))
    return result


def select_rates(rows):
    """Pair seeds/steps before averaging ratios; selection never reads quality metrics."""
    keyed = {(r["family"], r["config"]["seed"]): r for r in rows}
    expected = {(f, s) for f in (*CONTROLS, *trial_specs()) for s in CAL_SEEDS}
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError("Require the exact 22-run pilot grid")
    candidates = {f: [] for f in TARGETS}
    for trial, (family, rate) in trial_specs().items():
        ratios = []
        for seed in CAL_SEEDS:
            row = keyed[trial, seed]; target = keyed[TARGETS[family], seed]
            if row["config"]["adapter_lr_multiplier"] != rate or target["config"]["adapter_lr_multiplier"] != 1.:
                raise ValueError("Pilot rate differs from its declared trial")
            ratios.extend(a/b for a, b in zip(function_updates(row), function_updates(target)))
        center = math.exp(statistics.mean(math.log(v) for v in ratios))
        matched = abs(center-1) <= TOLERANCE and all(POINT_BOUNDS[0] <= v <= POINT_BOUNDS[1] for v in ratios)
        candidates[family].append(dict(trial=trial, adapter_lr_multiplier=rate, paired_ratios=ratios,
                                       geometric_mean_ratio=center, within_tolerance=matched))
    selected = {f: min([v for v in values if v["within_tolerance"]] or values,
                       key=lambda v: (abs(math.log(v["geometric_mean_ratio"])), v["adapter_lr_multiplier"]))
                for f, values in candidates.items()}
    return dict(candidates=candidates, selected=selected,
                all_within_tolerance=all(v["within_tolerance"] for v in selected.values()),
                selection_uses_quality=False)


def reference_rows(protocol):
    path = Path(protocol["reference_workflow"])
    if digest(path/"report.json") != protocol["reference_report_sha256"] or digest(path/"audit.json") != protocol["reference_audit_sha256"]:
        raise ValueError("Frozen reference report or audit changed")
    report = json.loads((path/"report.json").read_text())
    return report["quality"]["rows"] + report["anchors"]


def prepare(output, launch=False, reference=None):
    root = Path(__file__).resolve().parent; output = Path(output).resolve()
    reference = Path(reference).resolve() if reference is not None else root/REFERENCE
    if output.exists(): raise FileExistsError(output)
    previous = json.loads((reference/"protocol.json").read_text())
    if previous["state"] != "complete" or not (reference/"audit.json").exists():
        raise RuntimeError("The current projector study must finish and pass its audit before this study can be prepared or launched")
    prior_audit = json.loads((reference/"audit.json").read_text())
    if not all(prior_audit["checks"].values()) or prior_audit["report_sha256"] != digest(reference/"report.json"):
        raise ValueError("Reference audit failed or evidence changed")
    if (reference/"quality/supervisor.sock").exists(): raise RuntimeError("Reference workers are still managed by an active supervisor")
    prior_report = json.loads((reference/"report.json").read_text())
    runtime=provenance()
    if {k:runtime[k] for k in ('package_versions','cuda_wheel')} != prior_report['quality']['runtime']:
        raise ValueError('Training runtime changed since the paired references')
    if digest(Path(previous['evaluator_cache']['path'])) != previous['evaluator_cache']['sha256']:
        raise ValueError('Frozen FID evaluator changed before launch')
    for name, sha in prior_report["quality"]["source_sha256"].items():
        if digest(root/name) != sha: raise ValueError("Training/diagnostic source changed since the paired references: "+name)
    free = shutil.disk_usage(root).free
    if free < 8*1024**3: raise RuntimeError("Require 8 GiB shared free space for the 4 GiB artifact bound and headroom")
    output.mkdir(parents=True); runner = output/"runner"; runner.mkdir()
    for name in (*SOURCE_FILES, *EXTRA_SOURCE): shutil.copy2(root/name, runner/name)
    protocol = dict(state="prepared", created_utc=now(), purpose="Separate adapter LR effects from projector capacity/nonlinearity",
        reference_workflow=str(reference), reference_report_sha256=digest(reference/"report.json"), reference_audit_sha256=digest(reference/"audit.json"),
        source_sha256={f.name:digest(f) for f in runner.iterdir()}, reference_training_source_sha256=prior_report["quality"]["source_sha256"],
        reference_runtime=prior_report["quality"]["runtime"],
        calibration_seeds=list(CAL_SEEDS), quality_seeds=list(QUALITY_SEEDS), observation_steps=list(OBSERVATIONS),
        targets=TARGETS, rate_grid=RATE_GRID, calibration_steps=500, maximum_pilot_runs=22, quality_runs=6,
        aggregate_ratio_tolerance=TOLERANCE, per_observation_ratio_bounds=list(POINT_BOUNDS),
        selection="Closest absolute log geometric-mean paired function-update ratio among candidates satisfying both guards, or closest miss for diagnosis; no quality/loss selection; fixed grid, no extra corrections",
        adapter_gain=1/math.sqrt(8), base_lr=4e-4, rank=8, width=256, depth=8, train_steps=30000,
        max_concurrent_workers=2, train_examples=45000, validation_examples=5000,
        classifier_run=previous["classifier_run"], auxiliary_classifier=classifier_provenance(previous["classifier_run"]),
        evaluator_cache=previous["evaluator_cache"], disk_free_bytes_before_launch=free,
        estimated_gpu_hours="3–5 aggregate GPU-hours, approximately1.5–2.5 wall hours with two GPUs, availability dependent",
        expected_storage_upper_bound_bytes=4*1024**3,
        artifact_purposes=dict(evidence="Frozen sources/configs/splits/provenance, metrics and update diagnostics, recipes, logs and reports",
                               compiler_cache="Compiled execution evidence, expected up to3GiB", grids="One100-image final CFG1 grid per quality run; no pilot grids"),
        retention="Compact evidence, declared compiler caches and final grids until explicitly removed; existing reference/evaluator artifacts unchanged",
        weights_saved=False, periodic_checkpoints=False, test_evaluated=False,
        limits=["Quality seeds86/87 and evaluation split are reused for paired rate contrasts; this is an exploratory follow-up, not an independent generalization estimate.",
                "Calibration seeds90/91 are separate; rates are frozen before new quality runs and never selected using FID, agreement or loss.",
                "Function updates are FP32 raw-model fixed-input probes; matching is approximate and path/optimizer dependent, not a muP derivation.",
                "Quality-seed update transfer is measured and reported; a pilot match does not establish matching throughout training.",
                "Only adapter LR changes; base LR, gain, initialization, model/data/sampling recipe and 30k schedule remain fixed.",
                "Final CFG1/2 share one EMA state/noise; 5k-image held-out FID and classifier agreement remain pilot proxies."])
    write_json(output/"protocol.json", protocol)
    if launch:
        private_supervisor(output, [("projector_lr_controller", [sys.executable,"-u",str(runner/Path(__file__).name),"run","--output",str(output)])], label="controller", autostart=True)
    return dict(output=str(output), launched=launch, pilot_runs=22, quality_runs=6, max_concurrent_workers=2)


def queue(output):
    """A CPU-only managed waiter; freeze sources before waiting for the prerequisite."""
    root=Path(__file__).resolve().parent;output=Path(output).resolve();reference=root/REFERENCE
    if output.exists():raise FileExistsError(output)
    state=json.loads((reference/'protocol.json').read_text())['state']
    if state not in ('training','complete'):raise RuntimeError('Reference is not running or successfully complete')
    free=shutil.disk_usage(root).free
    if free<8*1024**3:raise RuntimeError('Require 8 GiB free before queueing the declared follow-up')
    output.mkdir(parents=True);runner=output/'queue-runner';runner.mkdir()
    for name in (*SOURCE_FILES,*EXTRA_SOURCE):shutil.copy2(root/name,runner/name)
    p=dict(state='waiting_for_reference',created_utc=now(),reference_workflow=str(reference),study_output=str(output/'study'),
           source_sha256={f.name:digest(f) for f in runner.iterdir()},max_concurrent_gpu_workers=2,waiting_gpu_workers=0,
           maximum_wait_hours=8,maximum_pilot_runs=22,pilot_steps=500,maximum_quality_runs=6,quality_steps=30000,
           targets=TARGETS,rate_grid=RATE_GRID,calibration_seeds=list(CAL_SEEDS),quality_seeds=list(QUALITY_SEEDS),
           expected_total_storage_upper_bound_bytes=5*1024**3,study_storage_upper_bound_bytes=4*1024**3,
           disk_free_bytes_before_queue=free,estimated_gpu_hours='3–5 aggregate; approximately1.5–2.5 wall hours after release of prerequisite',
           artifact_purposes='Frozen queue/study sources, configs/provenance, diagnostics/metrics, recipes, logs/reports, compiler caches up to3GiB and six final100-image grids',
           retention='Declared compact evidence, compiler caches and grids until explicitly removed; existing artifacts unchanged',
           weights_saved=False,periodic_checkpoints=False,test_evaluated=False,
           launch_gate='Successful completed reference audit and closed reference worker supervisor; recheck frozen sources and capacity at launch; quality requires pilot guards')
    write_json(output/'queue_protocol.json',p)
    control=private_supervisor(output,[('projector_lr_queue',[sys.executable,'-u',str(runner/Path(__file__).name),'wait','--output',str(output)])],label='controller',autostart=True)
    return dict(output=str(output),queued=True,controller=control,waiting_gpu_workers=0,max_concurrent_gpu_workers=2)


def wait_run(output):
    output=Path(output);p=json.loads((output/'queue_protocol.json').read_text());reference=Path(p['reference_workflow'])
    deadline=time.monotonic()+p['maximum_wait_hours']*3600
    try:
        while True:
            state=json.loads((reference/'protocol.json').read_text())['state']
            if state not in ('training','complete'):raise RuntimeError('Prerequisite failed or left its declared completion path: '+state)
            if state=='complete' and (reference/'audit.json').exists() and not (reference/'quality/supervisor.sock').exists():
                audited=json.loads((reference/'audit.json').read_text())
                if not all(audited['checks'].values()) or audited['report_sha256']!=digest(reference/'report.json'):
                    raise RuntimeError('Prerequisite audit failed or frozen evidence changed')
                break
            if time.monotonic()>=deadline:raise TimeoutError('Eight-hour prerequisite wait expired; no GPU work launched')
            time.sleep(20)
        for name,sha in p['source_sha256'].items():
            if digest(output/'queue-runner'/name)!=sha:raise ValueError('Frozen queue source changed: '+name)
        prepare(p['study_output'],reference=reference)
        p.update(state='study_running',released_utc=now(),reference_report_sha256=digest(reference/'report.json'));write_json(output/'queue_protocol.json',p)
        run(p['study_output'])
        result=json.loads((Path(p['study_output'])/'protocol.json').read_text())
        p.update(state=result['state'],completed_utc=now());write_json(output/'queue_protocol.json',p)
    except BaseException as error:
        p.update(state='failed',failed_utc=now(),failure_type=type(error).__name__,failure=str(error));write_json(output/'queue_protocol.json',p);raise


def run(output):
    output = Path(output).resolve(); p = json.loads((output/"protocol.json").read_text())
    if p["state"] != "prepared": raise ValueError("Fresh workflows only")
    p.update(state="calibrating", started_utc=now()); write_json(output/"protocol.json", p)
    try:
        cases = [(f,s,make_config(f,s,1.,output/"pilot"/f"{f}_s{s}",None,True)) for f in CONTROLS for s in CAL_SEEDS]
        trials = [(trial,s,make_config(f,s,rate,output/"pilot"/f"{trial}_s{s}",None,True))
                  for trial,(f,rate) in trial_specs().items() for s in CAL_SEEDS]
        random.Random(193).shuffle(trials); cases += trials
        pilot = execute_stage(output,p,"pilot",cases)
        selection = select_rates(pilot["rows"]); write_json(output/"calibration.json", selection)
        write_json(output/"pilot_report.json",pilot)
        if not selection["all_within_tolerance"]:
            p.update(state="calibration_failed", completed_utc=now(), reason="Bounded pilot grid missed update tolerance; quality withheld")
            write_json(output/"protocol.json",p); return
        recipes = dict(frozen_utc=now(), calibration_sha256=digest(output/"calibration.json"),
                       multipliers={f:v["adapter_lr_multiplier"] for f,v in selection["selected"].items()},
                       selection_uses_quality=False)
        write_json(output/"recipes.json",recipes); p.update(state="training", recipes_sha256=digest(output/"recipes.json")); write_json(output/"protocol.json",p)
        cases = [(f,s,make_config(f,s,recipes["multipliers"][f],output/"quality"/f"{f}_s{s}",p["classifier_run"]))
                 for f in TARGETS for s in QUALITY_SEEDS]
        random.Random(197).shuffle(cases); quality = execute_stage(output,p,"quality",cases)
        write_json(output/"report.json",dict(pilot=pilot,selection=selection,recipes=recipes,quality=quality))
        p.update(state="complete",completed_utc=now());write_json(output/"protocol.json",p)
        audited=audit(output);write_summary(output,quality["rows"],p,audited)
        print(json.dumps(dict(event="complete",output=str(output),audit_passed=True)),flush=True)
    except BaseException as error:
        p.update(state="failed",failed_utc=now(),failure_type=type(error).__name__,failure=str(error));write_json(output/"protocol.json",p);raise


def trace(row):
    fields=("step","batch_images_sha256","batch_labels_sha256","noise_sha256","timesteps_sha256","noise_rng_state_sha256")
    return [{k:e[k] for k in fields} for e in row["design_diagnostics"]["diagnostics"][1:]]


def audit(output):
    import torch
    from train_dit_cifar10 import make_model
    from benchmark_dit_design import digest_parameters
    torch.set_num_threads(2);output=Path(output);p=json.loads((output/"protocol.json").read_text())
    assert p["state"]=="complete" and p["max_concurrent_workers"]==2
    report=json.loads((output/"report.json").read_text());refs=reference_rows(p)
    assert classifier_provenance(p["classifier_run"])==p["auxiliary_classifier"]
    for name,sha in p["source_sha256"].items():
        assert digest(output/"runner"/name)==sha
        for stage in ("pilot","quality"):assert digest(output/stage/"source"/name)==sha
    replays={}
    for stage in ("pilot","quality"):
        replay=summarize(output/stage/"screen.json");jobs=json.loads((output/stage/"screen.json").read_text())["jobs"]
        for row in replay["rows"]:row["family"]=next(j["family"] for j in jobs if j["name"]==row["name"])
        assert same_stage_evidence(replay,report[stage]);replays[stage]=replay
        assert replay["source_sha256"]==p["reference_training_source_sha256"]
        assert replay["runtime"]==p["reference_runtime"]
        assert not (output/stage/"supervisor.sock").exists()
        claims=json.loads((output/stage/"hardware_claims.json").read_text());assert set(claims)=={j["service"] for j in jobs}
        for job in jobs:
            provenance=json.loads((Path(job["output_dir"])/"provenance.json").read_text())
            assert claims[job["service"]]["owner"]=="adalora" and claims[job["service"]]["uuid"].removeprefix("GPU-")==provenance["device_uuid"]
        for row in replay["rows"]:
            cfg=row["config"];family=row["family"]
            if stage=="pilot":base,rate=trial_specs().get(family,(family,1.))
            else:base=family;rate=report["recipes"]["multipliers"][base]
            expected=make_config(base,cfg["seed"],rate,cfg["output_dir"],p["classifier_run"],stage=="pilot")
            assert cfg==asdict(expected);torch.manual_seed(cfg["seed"]);model=make_model(expected)
            assert row["parameters"]==model.statistics()["parameters"]
            diag=row["design_diagnostics"]
            assert diag["base_initialization_sha256"]==digest_parameters(model) and diag["adapter_initialization_sha256"]==digest_parameters(model,True)
            assert all(v<=10 for v in diag["adapter_first_nonzero_gradient_observed"].values())
            assert all(v["gate_rms"]==v["delta_rms"]==0 for v in diag["diagnostics"][0]["branches"])
    assert select_rates(replays["pilot"]["rows"])==report["selection"] and report["selection"]["all_within_tolerance"]
    assert json.loads((output/"calibration.json").read_text())==report["selection"]
    assert digest(output/"recipes.json")==p["recipes_sha256"]
    assert report["recipes"]==json.loads((output/"recipes.json").read_text())
    assert report["recipes"]["calibration_sha256"]==digest(output/"calibration.json")
    assert report["recipes"]["multipliers"]=={f:v["adapter_lr_multiplier"] for f,v in report["selection"]["selected"].items()}
    quality_keys={(r["family"],r["config"]["seed"]) for r in replays["quality"]["rows"]}
    assert len(replays["quality"]["rows"])==6 and quality_keys=={(f,s) for f in TARGETS for s in QUALITY_SEEDS}
    for seed in CAL_SEEDS:
        rows=[r for r in replays["pilot"]["rows"] if r["config"]["seed"]==seed]
        assert all(trace(r)==trace(rows[0]) for r in rows)
    for row in replays["quality"]["rows"]:
        ref=next(r for r in refs if r["family"]==row["family"] and r["config"]["seed"]==row["config"]["seed"])
        assert trace(row)==trace(ref)
        assert all(row["design_diagnostics"][k]==ref["design_diagnostics"][k] for k in
                   ("base_initialization_sha256","adapter_initialization_sha256","noise_rng_final_state_sha256"))
        assert row["training_setup"]["label_dropout_rng_final_state_sha256"]==ref["training_setup"]["label_dropout_rng_final_state_sha256"]
        assert set(row["guidance_evaluations"])=={"1.0","2.0"}
        assert [e["step"] for e in row["generative_validation_curve"]]==[5000,10000,20000,30000]
        for metric in row["guidance_evaluations"].values():
            assert metric["generated_images"]==metric["fid_fake_images"]==metric["fid_real_images"]==5000
            assert metric["class_agreement"]["classifier"]==p["auxiliary_classifier"]
    progress=[json.loads(s) for s in (output/"progress.jsonl").read_text().splitlines()]
    assert all(e["active"]<=2 for e in progress)
    assert not [f for stage in ("pilot","quality") for f in (output/stage).rglob('*') if f.suffix in ('.pt','.pth','.ckpt','.safetensors')]
    retained=sum(f.stat().st_size for f in output.rglob('*') if f.is_file());assert retained<p["expected_storage_upper_bound_bytes"]
    audited=dict(checks={k:True for k in ('bounded_pilot_grid','update_only_selection','recipes_frozen','six_quality_runs','frozen_sources_evaluators',
          'same_initialization_and_observed_streams','zero_start_gradient_activation','successful_exits_closed_workers','two_gpu_cap','no_weights_test','storage_within_bound')},
          report_sha256=digest(output/'report.json'),bytes_retained=retained)
    write_json(output/'audit.json',audited);return audited


def write_summary(output,rows,p,audited):
    refs=reference_rows(p);comparisons=[]
    for family,target in TARGETS.items():
        group=[next(r for r in rows if r['family']==family and r['config']['seed']==s) for s in QUALITY_SEEDS]
        controls=[next(r for r in refs if r['family']==family and r['config']['seed']==s) for s in QUALITY_SEEDS]
        for scale in ('1.0','2.0'):
            a=[r['guidance_evaluations'][scale] for r in group];b=[r['guidance_evaluations'][scale] for r in controls]
            ratios=[]
            for r,s in zip(group,QUALITY_SEEDS):
                t=next(v for v in refs if v['family']==target and v['config']['seed']==s)
                ratios.append([x/y for x,y in zip(function_updates(r),function_updates(t))])
            comparisons.append(dict(family=family,target=target,scale=scale,adapter_lr_multiplier=group[0]['config']['adapter_lr_multiplier'],
                fid=mean_sd([v['fid'] for v in a]),unit_rate_fid=mean_sd([v['fid'] for v in b]),
                paired_fid_change=mean_sd([x['fid']-y['fid'] for x,y in zip(a,b)]),
                agreement_percent=mean_sd([100*v['class_agreement']['generated']['agreement'] for v in a]),
                paired_agreement_pp_change=mean_sd([100*(x['class_agreement']['generated']['agreement']-y['class_agreement']['generated']['agreement']) for x,y in zip(a,b)]),
                quality_seed_early_update_ratios=ratios))
    check_finite(comparisons);write_json(output/'comparison.json',dict(comparisons=comparisons,audit=audited,limits=p['limits']))
    lines=['# Projector adapter-LR controls','','Two paired quality seeds; means and sample SDs are exploratory estimates. Rate selection uses separate pilot seeds and function updates only.','',
           '| Recipe | Adapter LR multiplier | CFG | FID | Unit-rate FID | Paired FID change | Classifier agreement |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for v in comparisons:
        lines.append(f"| {v['family']} | {v['adapter_lr_multiplier']} | {v['scale']} | {v['fid']['mean']:.2f} ± {v['fid']['sd']:.2f} | {v['unit_rate_fid']['mean']:.2f} | {v['paired_fid_change']['mean']:+.2f} | {v['agreement_percent']['mean']:.2f}% |")
    lines+=['','Quality-seed early update ratios are included in comparison.json; pilot matching does not imply successful transfer. Keep these with the rate and quality comparisons.','','Limits: '+' '.join(p['limits']),'',
            '[Comparison](comparison.json) · [Calibration](calibration.json) · [Frozen recipes](recipes.json) · [Evidence](report.json) · [Audit](audit.json)','']
    (output/'summary.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run','audit','queue','wait'))
    parser.add_argument('--output',type=Path);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    output=args.output or Path('runs')/('dit-projector-lr-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action=='prepare':print(json.dumps(prepare(output,args.launch)),flush=True)
    elif args.action=='run':run(output)
    elif args.action=='queue':print(json.dumps(queue(output)),flush=True)
    elif args.action=='wait':wait_run(output)
    else:print(json.dumps(audit(output),indent=2))

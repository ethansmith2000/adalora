#!/usr/bin/env python3
"""Bounded gain/LR calibration followed by fresh-seed rank/width confirmation."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

from benchmark_dit_gain import choose_candidate, proposed_rate
from class_fidelity import classifier_provenance, frozen_classifier_model
from dit_followup import private_supervisor
from dit_rank2_confirmation import digest, execute_stage, now
from train_dit_cifar10 import Config, SOURCE_FILES, write_json

CAL_SEEDS = (60, 61)
CONFIRM_SEEDS = (62, 63, 64)
OBSERVATIONS = (50, 100, 250, 500)
TOLERANCE = .15
MAX_CORRECTIONS = 3
FAMILIES = {
    'r8_s16': dict(rank=8, n_basis=16, adapter_gain=1/math.sqrt(8)),
    'r8_s12_default': dict(rank=8, n_basis=12, adapter_gain=1/math.sqrt(8)),
    'r8_s12_lr': dict(rank=8, n_basis=12, adapter_gain=1/math.sqrt(8)),
    'r8_s12_gain_lr': dict(rank=8, n_basis=12, adapter_gain=(16/12)/math.sqrt(8)),
    'r16_s6_default': dict(rank=16, n_basis=6, adapter_gain=1/math.sqrt(16)),
    'r16_s6_lr': dict(rank=16, n_basis=6, adapter_gain=1/math.sqrt(16)),
    'r16_s6_gain_lr': dict(rank=16, n_basis=6, adapter_gain=(16/6)/math.sqrt(16)),
}
CAL_FAMILIES = ('r8_s16', 'r8_s12_lr', 'r8_s12_gain_lr', 'r16_s6_lr', 'r16_s6_gain_lr')
CAL_DIAGNOSTICS = '0,1,3,5,10,25,50,100,250,500'
CONFIRM_DIAGNOSTICS = CAL_DIAGNOSTICS + ',1000,5000,10000,20000,30000'
EXTRA_SOURCE = ('dit_width_calibration.py', 'audit_dit_width.py', 'summarize_dit_width.py',
                'audit_dit_core.py', 'dit_core_confirmation.py',
                'test_dit_width.py', 'dit_rank2_confirmation.py', 'dit_quality_diagnostics.py',
                'benchmark_dit_design.py', 'benchmark_dit_gain.py', 'dit_followup.py',
                'summarize_dit_cifar10_screen.py', 'test_dit_quality.py', 'test_dit_cifar10.py',
                'test_class_fidelity.py', 'test_rank_width_search.py', 'rank_width_search.py',
                'vit_quality_diagnostics.py', 'requirements.txt')


def update_metric(row):
    events = {event['step']: event for event in row['design_diagnostics']['diagnostics']}
    value = statistics.mean(block['update_rms'] for step in OBSERVATIONS
                            for block in events[step]['fixed_token_condition_function_updates'])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Expected finite positive fixed-input function updates')
    return value


def aggregate_candidate(family, attempt, multiplier, rows, target=None):
    if sorted(row['config']['seed'] for row in rows) != list(CAL_SEEDS):
        raise ValueError('Both declared calibration seeds are required')
    metrics = {str(row['config']['seed']): update_metric(row) for row in rows}
    value = statistics.mean(metrics.values())
    target = value if target is None else target
    return dict(family=family, attempt=attempt, adapter_lr_multiplier=multiplier,
                metric=value, per_seed_metrics=metrics, metric_to_target=value/target,
                within_tolerance=abs(value/target-1) <= TOLERANCE)


def next_rates(attempt, rates, candidates, target):
    # Verify one response to the proposed rate even if the first candidate is
    # already within tolerance; all later corrections depend only on updates.
    return {family: proposed_rate(rate, candidates[family][-1]['metric'], target)
            for family, rate in rates.items() if family != 'r8_s16'
            and attempt < MAX_CORRECTIONS
            and (attempt == 0 or not candidates[family][-1]['within_tolerance'])}


def make_config(family, seed, multiplier, output, classifier, calibration=False):
    return Config(variant='dplr', **FAMILIES[family], width=192, depth=6, heads=3, patch_size=4,
                  adapter_lr_multiplier=multiplier, seed=seed, lr=4e-4,
                  max_steps=500 if calibration else 30000, lr_schedule_steps=30000, warmup_steps=50,
                  eval_every=500 if calibration else 5000, log_every=100 if calibration else 500,
                  cpu_threads=2, num_workers=2, classifier_run=None if calibration else classifier,
                  fid_images=0 if calibration else 5000, sample_images=0,
                  diagnostic_steps=CAL_DIAGNOSTICS if calibration else CONFIRM_DIAGNOSTICS,
                  output_dir=str(output))


def prepare(output, launch):
    root = Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free < 1024**3:
        raise RuntimeError('Require 1 GiB free for compact evidence')
    classifier = str(root/'runs/dit-followup-20261007T174446Z/classifier/classifier_config')
    frozen_classifier_model(classifier)  # Reject missing/changed original source before any launch.
    auxiliary = classifier_provenance(classifier)
    cache = Path(Config().fid_weights)
    output.mkdir(parents=True)
    runner = output/'runner'; runner.mkdir()
    for name in dict.fromkeys((*SOURCE_FILES, *EXTRA_SOURCE)):
        shutil.copy2(root/name, runner/name)
    protocol = dict(state='prepared', created_utc=now(), repo_root=str(root),
                    purpose='separate adapter learning-rate matching from explicit gain changes for r8/s12 and r16/s6, then confirm quality against r8/s16',
                    source_sha256={p.name:digest(p) for p in runner.iterdir()},
                    calibration_seeds=list(CAL_SEEDS), confirmation_seeds=list(CONFIRM_SEEDS),
                    calibration_families=list(CAL_FAMILIES), families=FAMILIES,
                    calibration_steps=500, train_steps=30000, lr_schedule_steps=30000, warmup_steps=50,
                    observation_steps=list(OBSERVATIONS), target_family='r8_s16', target_multiplier=1.,
                    calibration_metric='mean highest-FP32 fixed-input adapter-delta single-step update RMS across six blocks, steps 50/100/250/500 and seeds 60/61',
                    selection='closest absolute log(mean update / target); excludes loss, validation, FID and class agreement',
                    correction='clip(previous multiplier * target / previous update metric, 1/16, 16)',
                    mandatory_first_correction=True, max_corrections=MAX_CORRECTIONS, relative_tolerance=TOLERANCE,
                    initial_family_multipliers={family:1. for family in CAL_FAMILIES},
                    gain_design='fixed 1/sqrt(r), or (16/s)/sqrt(r) as one empirical width-compensation alternative; same gain throughout each calibration/confirmation recipe',
                    fixed_control_multiplier=1., confirmation_runs=21, maximum_calibration_runs=34,
                    max_concurrent_workers=6, order_seed=92, train_examples=45000, validation_examples=5000,
                    split_seed=123, evaluation_seed=2027, sampling_seed=2028, sampling_steps=50, fid_images=5000,
                    classifier_run=classifier, auxiliary_classifier=auxiliary,
                    evaluator_cache=dict(path=str(cache),bytes=cache.stat().st_size,sha256=digest(cache)),
                    disk_free_bytes_before_launch=free, expected_storage_upper_bound_bytes=512*1024**2,
                    estimated_new_gpu_hours='7–10 aggregate including bounded calibration, validation, diagnostics and final evaluations; startup additional',
                    weights_saved=False, periodic_checkpoints=False, test_evaluated=False,
                    dedicated_speed_benchmarks=False, compile=False,
                    retention='compact source/configs/seeds/splits/metrics/logs/audits/plots until explicitly removed; no grids, model/evaluator weights or generated datasets',
                    limits=['One alternate gain per shape; this is targeted sensitivity, not exhaustive hyperparameter optimization.',
                            'Gain times 16/s is an empirical compensation heuristic, not a variance-preservation or muP derivation.',
                            'The rate changes every adapter parameter group; gain also changes function scaling. These are different optimization parameterizations.',
                            'Matching scalar early update RMS does not match every seed/block, later update, live magnitude or entire-network update.',
                            'Calibration seeds 60/61 and confirmation seeds 62/63/64 are independent of previous screens; no quality-based recipe selection.',
                            'Default-rate controls keep LR multiplier 1; calibrated gains/rates freeze before any quality run and never retune on quality.',
                            'All equal-shape models share base initialization; gain/rate variants of one rank/width share adapter initialization.',
                            'r8/s12 and r16/s6 approximately match decoder budget, but encoder/bias totals and training memory differ.',
                            'Three training seeds, one held-out split and sampling seed; sample SD is not a confidence interval.',
                            'FID uses 5,000 real/generated images and class agreement uses the existing frozen auxiliary classifier.',
                            'Concurrent eager training is not an isolated speed or equal-time quality comparison.',
                            'Broadcast class/time only; no new ViT/LM, cross-attention, sequence retrieval or structured-operator experiments.'])
    write_json(output/'protocol.json', protocol)
    if launch:
        private_supervisor(output, [('adalora_width_controller',
                           [sys.executable,'-u',str(runner/Path(__file__).name),'run','--output',str(output)])], 'controller', True)
    print(json.dumps(dict(output=str(output),launched=launch)),flush=True)


def run(output):
    protocol = json.loads((output/'protocol.json').read_text())
    protocol.update(state='calibrating',started_utc=now()); write_json(output/'protocol.json',protocol)
    candidates={family:[] for family in CAL_FAMILIES}
    rates={family:1. for family in CAL_FAMILIES}; rounds=[]; target=None
    for attempt in range(MAX_CORRECTIONS+1):
        specs=[(family,seed,rate) for family,rate in rates.items() for seed in CAL_SEEDS]
        summary=execute_stage(output,protocol,f'calibration{attempt}',specs,config_factory=make_config)
        rounds.append(summary)
        if target is None:
            target=aggregate_candidate('r8_s16',0,1.,[row for row in summary['rows'] if row['family']=='r8_s16'])['metric']
        for family,rate in rates.items():
            candidates[family].append(aggregate_candidate(family,attempt,rate,
                                      [row for row in summary['rows'] if row['family']==family],target))
        selected={family:choose_candidate(values,target) for family,values in candidates.items()}
        write_json(output/'calibration_progress.json',dict(target=target,candidates=candidates,selected=selected))
        rates=next_rates(attempt,rates,candidates,target)
        if not rates:break
    calibration=dict(target=target,candidates=candidates,selected=selected,rounds=rounds)
    write_json(output/'calibration.json',calibration)
    if not all(value['within_tolerance'] for value in selected.values()):
        protocol.update(state='calibration_failed',failure_utc=now());write_json(output/'protocol.json',protocol)
        raise RuntimeError('Bounded calibration missed tolerance; quality confirmation withheld')
    recipes=dict(frozen_utc=now(),calibration_sha256=digest(output/'calibration.json'),selected=selected,
                 selection_uses_quality=False,confirmation_seeds=list(CONFIRM_SEEDS),
                 gains={family:values['adapter_gain'] for family,values in FAMILIES.items()},
                 multipliers={family:selected[family]['adapter_lr_multiplier'] if family in selected else 1. for family in FAMILIES})
    write_json(output/'recipes.json',recipes)
    protocol.update(state='confirming',recipes_sha256=digest(output/'recipes.json'),recipes_frozen_utc=recipes['frozen_utc'])
    write_json(output/'protocol.json',protocol)
    specs=[(family,seed,recipes['multipliers'][family]) for family in FAMILIES for seed in CONFIRM_SEEDS]
    confirmation=execute_stage(output,protocol,'confirmation',specs,config_factory=make_config)
    write_json(output/'report.json',dict(protocol=protocol,calibration=calibration,recipes=recipes,
                                       confirmation=confirmation,limits=protocol['limits'],quality_winner_claimed=False))
    protocol.update(state='complete',completed_utc=now());write_json(output/'protocol.json',protocol)
    print(json.dumps(dict(event='suite_complete',report=str(output/'report.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--output',type=Path)
    parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    output=(args.output or Path(__file__).resolve().parent/'runs'/('dit-width-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    prepare(output,args.launch) if args.action=='prepare' else run(output)

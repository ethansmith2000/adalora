#!/usr/bin/env python3
"""Bounded update calibration and a fresh-seed shared/nonlinear projector screen."""

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

CAL_SEEDS = (65, 66)
CONFIRM_SEEDS = (67, 68)  # quality-screen seeds, not a longer confirmation
OBSERVATIONS = (50, 100, 250, 500)
TOLERANCE = .15
MAX_CORRECTIONS = 3
FAMILIES = {
    'independent_linear': dict(rank=8,n_basis=16,projector_groups=6,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
    'independent_silu': dict(rank=8,n_basis=16,projector_groups=6,coefficient_activation='silu',adapter_gain=1/math.sqrt(8)),
    'two_groups_linear': dict(rank=8,n_basis=48,projector_groups=2,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
    'two_groups_silu': dict(rank=8,n_basis=48,projector_groups=2,coefficient_activation='silu',adapter_gain=1/math.sqrt(8)),
    'all_shared_linear': dict(rank=8,n_basis=96,projector_groups=1,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
    'two_groups_s16': dict(rank=8,n_basis=16,projector_groups=2,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
}
CAL_FAMILIES = tuple(FAMILIES)
CAL_DIAGNOSTICS = '0,1,3,5,10,25,50,100,250,500'
CONFIRM_DIAGNOSTICS = CAL_DIAGNOSTICS + ',1000,2500,5000'
EXTRA_SOURCE = ('dit_projector_screen.py','audit_dit_projectors.py','summarize_dit_projectors.py',
                'test_dit_projectors.py','test_dit_projector_screen.py','audit_dit_core.py','dit_core_confirmation.py',
                'dit_rank2_confirmation.py','dit_quality_diagnostics.py','benchmark_dit_design.py',
                'benchmark_dit_gain.py','dit_followup.py','summarize_dit_cifar10_screen.py',
                'test_dit_quality.py','test_dit_cifar10.py','test_class_fidelity.py','requirements.txt')


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
            for family, rate in rates.items() if family != 'independent_linear'
            and attempt < MAX_CORRECTIONS
            and (attempt == 0 or not candidates[family][-1]['within_tolerance'])}


def make_config(family, seed, multiplier, output, classifier, calibration=False):
    return Config(variant='dplr', **FAMILIES[family], width=192, depth=6, heads=3, patch_size=4,
                  adapter_lr_multiplier=multiplier, seed=seed, lr=4e-4,
                  max_steps=500 if calibration else 5000, lr_schedule_steps=30000, warmup_steps=50,
                  eval_every=500 if calibration else 2500, log_every=100 if calibration else 500,
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
                    purpose='test nonlinear coefficient heads and shared wider projectors at equal parameter budget, with a fixed-width tying control',
                    source_sha256={p.name:digest(p) for p in runner.iterdir()},
                    calibration_seeds=list(CAL_SEEDS), confirmation_seeds=list(CONFIRM_SEEDS),
                    calibration_families=list(CAL_FAMILIES), families=FAMILIES,
                    calibration_steps=500, train_steps=5000, quality_phase='screen', lr_schedule_steps=30000, warmup_steps=50,
                    observation_steps=list(OBSERVATIONS), target_family='independent_linear', target_multiplier=1.,
                    calibration_metric='mean highest-FP32 fixed-input adapter-delta single-step update RMS across six blocks, steps 50/100/250/500 and seeds 65/66',
                    selection='closest absolute log(mean update / target); excludes loss, validation, FID and class agreement',
                    correction='clip(previous multiplier * target / previous update metric, 1/16, 16)',
                    mandatory_first_correction=True, max_corrections=MAX_CORRECTIONS, relative_tolerance=TOLERANCE,
                    initial_family_multipliers={family:1. for family in CAL_FAMILIES},
                    gain_design='fixed 1/sqrt(8) throughout; independent affine r8/s16 is the update anchor',
                    fixed_control_multiplier=1., confirmation_runs=12, maximum_calibration_runs=42,
                    max_concurrent_workers=6, order_seed=96, train_examples=45000, validation_examples=5000,
                    split_seed=123, evaluation_seed=2027, sampling_seed=2028, sampling_steps=50, fid_images=5000,
                    classifier_run=classifier, auxiliary_classifier=auxiliary,
                    evaluator_cache=dict(path=str(cache),bytes=cache.stat().st_size,sha256=digest(cache)),
                    disk_free_bytes_before_launch=free, expected_storage_upper_bound_bytes=512*1024**2,
                    estimated_new_gpu_hours='1.5–3 aggregate including bounded calibration, validation, diagnostics and final evaluations; startup additional',
                    weights_saved=False, periodic_checkpoints=False, test_evaluated=False,
                    dedicated_speed_benchmarks=False, compile=False,
                    retention='compact source/configs/seeds/splits/metrics/logs/audits/plots until explicitly removed; no grids, model/evaluator weights or generated datasets',
                    limits=['Two-seed 5k quality screen, not longer confirmation or statistical significance.',
                            'Independent affine/silu s16, two-group affine/silu s48, and all-shared affine s96 have equal adapter and total parameter counts.',
                            'Two-group affine s16 is a fixed-width tying control; the full sharing/width/nonlinearity factorial is not tested.',
                            'One shared module per group; unique parameters and per-layer diagnostics are separately audited.',
                            'The same broadcast condition ties the conditional transform within each group, while base blocks and AdaLN/gates remain independent.',
                            'At equal affine budget G*s is fixed; lower group count does not establish lower total generation FLOPs or speed.',
                            'Rates are selected only on aggregate early function updates, not loss or generated quality; matching may drift across seeds/layers/time.',
                            'Calibration seeds 65/66 and screen seeds 67/68 are fresh relative to previous quality studies.',
                            'FID uses 5k images and a fixed sampling seed; classifier agreement is a frozen proxy.',
                            'Raw-model diagnostics and EMA quality concern different states; fixed-probe operator Frobenius norms do not establish variance preservation.',
                            'No output normalization, larger MLP/trunk, layer embeddings, partial decoder sharing, new ViT/LM/retrieval or structured operators are tested.',
                            'Concurrent eager training is not isolated timing or an equal-time comparison; dedicated speed work stays deferred.',
                            'No automatic longer confirmation is launched by this screen.'])
    write_json(output/'protocol.json', protocol)
    if launch:
        private_supervisor(output, [('adalora_projector_controller',
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
            target=aggregate_candidate('independent_linear',0,1.,[row for row in summary['rows'] if row['family']=='independent_linear'])['metric']
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
    protocol.update(state='screening',recipes_sha256=digest(output/'recipes.json'),recipes_frozen_utc=recipes['frozen_utc'])
    write_json(output/'protocol.json',protocol)
    specs=[(family,seed,recipes['multipliers'][family]) for family in FAMILIES for seed in CONFIRM_SEEDS]
    confirmation=execute_stage(output,protocol,'screen',specs,config_factory=make_config)
    write_json(output/'report.json',dict(protocol=protocol,calibration=calibration,recipes=recipes,
                                       confirmation=confirmation,limits=protocol['limits'],quality_winner_claimed=False))
    protocol.update(state='complete',completed_utc=now());write_json(output/'protocol.json',protocol)
    print(json.dumps(dict(event='suite_complete',report=str(output/'report.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--output',type=Path)
    parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    output=(args.output or Path(__file__).resolve().parent/'runs'/('dit-projectors-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    prepare(output,args.launch) if args.action=='prepare' else run(output)

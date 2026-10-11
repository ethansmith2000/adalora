#!/usr/bin/env python3
"""Bounded update calibration and a fresh-seed shared-basis/layer-head and small-MLP screen."""

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

CAL_SEEDS = (70, 71)
CONFIRM_SEEDS = (72, 73)  # quality-screen seeds, not a longer confirmation
OBSERVATIONS = (50, 100, 250, 500)
TOLERANCE = .15
MAX_CORRECTIONS = 3
FAMILIES = {
    'independent_linear': dict(rank=8,n_basis=16,projector_groups=6,decoder_groups=0,coefficient_hidden=0,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
    'two_bases_s16': dict(rank=8,n_basis=16,projector_groups=6,decoder_groups=2,coefficient_hidden=0,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
    'two_bases_s44': dict(rank=8,n_basis=44,projector_groups=6,decoder_groups=2,coefficient_hidden=0,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
    'all_bases_s77': dict(rank=8,n_basis=77,projector_groups=6,decoder_groups=1,coefficient_hidden=0,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
    'mlp_linear': dict(rank=8,n_basis=16,projector_groups=6,decoder_groups=0,coefficient_hidden=48,coefficient_activation='linear',adapter_gain=1/math.sqrt(8)),
    'mlp_silu': dict(rank=8,n_basis=16,projector_groups=6,decoder_groups=0,coefficient_hidden=48,coefficient_activation='silu',adapter_gain=1/math.sqrt(8)),
}
CAL_FAMILIES = tuple(FAMILIES)
CAL_DIAGNOSTICS = '0,1,3,5,10,25,50,100,250,500'
CONFIRM_DIAGNOSTICS = CAL_DIAGNOSTICS + ',1000,2500,5000'
EXTRA_SOURCE = ('dit_basis_head_screen.py','audit_dit_basis_heads.py','summarize_dit_basis_heads.py',
                'test_dit_basis_heads.py','test_dit_basis_head_screen.py','test_dit_projectors.py','test_dit_projector_screen.py','test_dit_width.py','test_dit_design.py','audit_dit_core.py','dit_core_confirmation.py',
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
                    purpose='test shared factor decoders with layer-specific coefficient heads, plus hidden-linear/SiLU MLP heads with an exact parameter-matched activation control',
                    source_sha256={p.name:digest(p) for p in runner.iterdir()},
                    calibration_seeds=list(CAL_SEEDS), confirmation_seeds=list(CONFIRM_SEEDS),
                    calibration_families=list(CAL_FAMILIES), families=FAMILIES,
                    calibration_steps=500, train_steps=5000, quality_phase='screen', lr_schedule_steps=30000, warmup_steps=50,
                    observation_steps=list(OBSERVATIONS), target_family='independent_linear', target_multiplier=1.,
                    calibration_metric='mean highest-FP32 fixed-input adapter-delta single-step update RMS across six blocks, steps 50/100/250/500 and seeds 70/71',
                    selection='closest absolute log(mean update / target); excludes loss, validation, FID and class agreement',
                    correction='clip(previous multiplier * target / previous update metric, 1/16, 16)',
                    mandatory_first_correction=True, max_corrections=MAX_CORRECTIONS, relative_tolerance=TOLERANCE,
                    initial_family_multipliers={family:1. for family in CAL_FAMILIES},
                    gain_design='fixed 1/sqrt(8) throughout; independent affine r8/s16 is the update anchor',
                    fixed_control_multiplier=1., confirmation_runs=12, maximum_calibration_runs=42,
                    max_concurrent_workers=2, order_seed=103, train_examples=45000, validation_examples=5000,
                    split_seed=123, evaluation_seed=2027, sampling_seed=2028, sampling_steps=50, fid_images=5000,
                    classifier_run=classifier, auxiliary_classifier=auxiliary,
                    evaluator_cache=dict(path=str(cache),bytes=cache.stat().st_size,sha256=digest(cache)),
                    disk_free_bytes_before_launch=free, expected_storage_upper_bound_bytes=512*1024**2,
                    estimated_new_gpu_hours='1.5–3 aggregate including bounded calibration, validation, diagnostics and final evaluations; startup additional',
                    weights_saved=False, periodic_checkpoints=False, test_evaluated=False,
                    dedicated_speed_benchmarks=False, compile=False,
                    retention='compact source/configs/seeds/splits/metrics/logs/audits/plots until explicitly removed; no grids, model/evaluator weights or generated datasets',
                    limits=['Two fresh seeds and 5k steps of a 30k schedule; not longer confirmation or statistical significance.',
                            'Independent s16 versus two shared decoder groups s44 and one group s77 match whole-model parameters within 0.1%, not exactly.',
                            'Two shared decoder groups s16 is a fixed-width tying control; all adapters have independent layer-specific heads.',
                            'Hidden-linear versus SiLU MLP heads have exactly equal parameters; both add 0.56% whole-model parameters relative to the affine reference.',
                            'U/V decoders are shared, but independently generated per-layer factors cannot be reused as identical transforms.',
                            'Base blocks, AdaLN heads and gates remain independent; the produced rank stays eight.',
                            'Gradient records per adapter overlap shared decoder parameters; separate unique decoder and head groups describe ownership.',
                            'Rates are selected only on aggregate early function updates; fresh seeds/layers/later steps may drift and are not retuned.',
                            'Calibration seeds 70/71 and quality seeds 72/73 are fresh relative to previous studies.',
                            'FID uses5k images and one sampling seed; class agreement is a frozen proxy.',
                            'Fixed-probe operator norms and raw-model updates do not guarantee variance preservation or EMA quality.',
                            'No output normalization, shared nonlinear trunk, layer embeddings, new token-conditioning or structured operators are tested.',
                            'At most two claimed GPU workers including all stages; concurrent eager telemetry is not isolated speed evidence.',
                            'No weights/checkpoints and no automatic longer confirmation.'])
    write_json(output/'protocol.json', protocol)
    if launch:
        private_supervisor(output, [('adalora_basis_head_controller',
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
    output=(args.output or Path(__file__).resolve().parent/'runs'/('dit-basis-heads-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    prepare(output,args.launch) if args.action=='prepare' else run(output)

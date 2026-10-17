#!/usr/bin/env python3
"""Complete a fixed Xavier plain/DPLR by global-LR matrix, reusing audited cells."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import torch
import vit_initializer_study as study
from dit_followup import private_supervisor
from train_cifar10 import load_config, provenance, write_json

REFERENCE = 'runs/vit-initializers-20261012T070919Z'
METHODS = ('plain', 'dplr')
RATES = (.5, 1.)
SEEDS = (106, 107, 108, 109)
REUSED_RATES = {'plain': 1., 'dplr': .5}
RUNTIME_KEYS = ('python', 'torch', 'cuda_wheel', 'package_versions', 'source_sha256')
REFERENCE_FILES = ('protocol.json', 'report.json', 'audit.json', 'selection.json',
                   'quality/summary.json', 'quality/screen.json', 'quality/hardware_claims.json',
                   'quality/supervisord.log', 'progress.jsonl')


def case_key(row):
    return row['family'], row['seed'], row['rate']


def expected_keys():
    return {(study.family_key(m, 'xavier_gelu'), s, r) for m in METHODS for s in SEEDS for r in RATES}


def reused_keys():
    return {(study.family_key(m, 'xavier_gelu'), s, REUSED_RATES[m]) for m in METHODS for s in SEEDS}


def remaining_cases(output):
    return [(f, s, r, study.make_config(f, s, output/'quality'/f'{m}_xavier_gelu_lr{r:g}_s{s}', r))
            for m in METHODS for r in RATES for s in SEEDS
            if (f := study.family_key(m, 'xavier_gelu'), s, r) not in reused_keys()]


def runtime_record(value):
    return {k: value[k] for k in RUNTIME_KEYS}


def reference_rows(p):
    """Read-only replay of original quality audit; do not edit old evidence."""
    reference = Path(p['reference_workflow'])
    assert not (reference/'controller.sock').exists() and not (reference/'quality/supervisor.sock').exists()
    assert all(study.common.digest(reference/n) == h for n, h in p['reference_files_sha256'].items())
    original = json.loads((reference/'protocol.json').read_text())
    assert original['state'] == 'complete'
    assert all(study.common.digest(reference/'runner'/n) == h for n, h in original['source_sha256'].items())
    assert all(study.common.digest(Path(study.__file__).parent/n) == h for n, h in original['source_sha256'].items())
    audit = json.loads((reference/'audit.json').read_text())
    assert all(audit['checks'].values()) and audit['report_sha256'] == study.common.digest(reference/'report.json')
    assert original['selection_sha256'] == study.common.digest(reference/'selection.json')
    rows = study.audit_stage(reference, 'quality')
    assert rows == json.loads((reference/'report.json').read_text())['quality']
    assert rows == json.loads((reference/'quality/summary.json').read_text())['rows']
    selected = [r for r in rows if case_key(r) in reused_keys()]
    assert len(selected) == 8 and {case_key(r) for r in selected} == reused_keys()
    for row in selected:
        assert row['config'] == asdict(study.make_config(row['family'], row['seed'], Path(row['output_dir']), row['rate']))
    return selected


def verify_combined(rows):
    """Reject duplicate cells or incompatible initialization/runtime/data reuse."""
    assert len(rows) == 16 and {case_key(r) for r in rows} == expected_keys()
    runtime = dataset = None
    for seed in SEEDS:
        values = [r for r in rows if r['seed'] == seed]
        bases, adapters, traces, probes = set(), {}, [], set()
        for row in values:
            cfg = row['config']; out = Path(row['output_dir'])
            assert cfg == asdict(study.make_config(row['family'], seed, out, row['rate']))
            assert row['results']['completed_steps'] == study.HORIZON
            current = runtime_record(json.loads((out/'provenance.json').read_text()))
            assert runtime is None or current == runtime; runtime = current
            metadata = json.loads((out/'dataset.json').read_text())
            assert dataset is None or metadata == dataset; dataset = metadata
            diag = row['results']['token_diagnostics']
            bases.add(diag['base_initialization_sha256'])
            adapters.setdefault(row['family'], set()).add(diag['adapter_initialization_sha256'])
            probes.add((diag['probe_seed'], diag['probe_sha256'], diag['live_images_sha256']))
            traces.append([{k: e[k] for k in ('step', 'batch_images_sha256', 'batch_labels_sha256')}
                           for e in diag['diagnostics'][1:]])
        assert len(bases) == len(probes) == 1
        assert all(len(v) == 1 for v in adapters.values())
        assert all(t == traces[0] for t in traces)


def prepare(output, reference=None, launch=False):
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent; output = Path(output).resolve()
    reference = Path(reference or root/REFERENCE).resolve()
    if output.exists(): raise FileExistsError(output)
    study.ensure_no_active_work()
    free = shutil.disk_usage(root).free
    if free < 8*1024**3: raise RuntimeError('Require 8 GiB free for the 4 GiB artifact bound and headroom')
    p = dict(reference_workflow=str(reference),
             reference_files_sha256={n: study.common.digest(reference/n) for n in REFERENCE_FILES})
    reused = reference_rows(p)
    assert runtime_record(provenance()) == runtime_record(json.loads((Path(reused[0]['output_dir'])/'provenance.json').read_text()))
    output.mkdir(parents=True); runner = output/'runner'
    shutil.copytree(reference/'runner', runner, ignore=shutil.ignore_patterns('__pycache__'))
    for n in ('vit_matched_lr_study.py', 'test_vit_matched_lr_study.py'): shutil.copy2(root/n, runner/n)
    original = json.loads((reference/'protocol.json').read_text())
    assert all(study.common.digest(runner/n) == h for n, h in original['source_sha256'].items())
    p.update(state='prepared', created_utc=study.common.now(), repo_root=str(root),
             purpose='Fixed plain/DPLR by global-LR interaction diagnostic under the same Xavier initialization',
             source_sha256={f.name: study.common.digest(f) for f in runner.iterdir() if f.is_file()},
             parameters={study.family_key(m, 'xavier_gelu'): original['parameters'][study.family_key(m, 'xavier_gelu')] for m in METHODS},
             methods=list(METHODS), initializer='xavier_gelu', rate_grid=list(RATES), quality_seeds=list(SEEDS),
             epochs=30, train_steps=study.HORIZON, lr_schedule_steps=study.HORIZON,
             train_examples=45000, validation_examples=5000, split_seed=123,
             planned_runs=dict(quality=8), reused_cases=[list(case_key(r)) for r in reused], combined_quality_runs=16,
             max_concurrent_workers=2, weights_saved=False, periodic_checkpoints=False, test_evaluated=False,
             recipe_selection=False, reference_quality_audit_passed=True,
             preflight_policy='Reuse successful compiled preflights and full-budget cells; training sources and runtime must be identical',
             resources=dict(estimated_gpu_hours='1–2 aggregate; about 30–60 minutes with two available GPUs', max_our_gpus=2,
                            disk_free_bytes_before_launch=free, retained_artifact_bound_bytes=4*1024**3,
                            artifact_purposes={'compact_evidence':'frozen sources/configs/runtime/seeds/splits/metrics/diagnostics/logs/audits and reports, including reference hashes',
                                               'compiler_cache':'compatibility/performance provenance, expected at most 3 GiB'},
                            retention='Compact evidence and declared compiler caches until explicitly removed; original retention unchanged'),
             primary_outcomes=['Final validation accuracy DPLR minus plain at each fixed global LR, paired by seed',
                               'Interaction: (DPLR minus plain at LR1) minus (DPLR minus plain at LR0.5), paired by seed'],
             limits=['Follow-up proposed after inspection of the original confirmation; these seeds/holdout are not independent confirmation.',
                     'Reuse eight fully audited existing cells and train only the exact missing complement; no new winner or retuning.',
                     'Global LR changes base and conditional groups together; relative affine/low-rank multipliers stay1.',
                     'Same nominal LR is not matched function-update magnitude or architecture-specific optimal LR.',
                     'Thirty epochs, four seeds, one CIFAR10 research split; sample SD is not a confidence interval.',
                     'Shared-instance compute observations are not isolated speed benchmarks; no broader architecture claim.'])
    write_json(output/'protocol.json', p); write_json(output/'reused_quality.json', dict(rows=reused, audit_passed=True))
    configs = output/'planned_configs'; configs.mkdir()
    for f, s, r, cfg in remaining_cases(output):
        path = configs/(Path(cfg.output_dir).name+'.json'); write_json(path, asdict(cfg)); load_config(['--override_json', str(path)])
    assert not torch.cuda.is_initialized()
    if launch: launch_controller(output)
    return dict(output=str(output), launched=launch, reused_quality=8, planned_runs=p['planned_runs'], max_our_gpus=2)


def launch_controller(output):
    output = Path(output).resolve(); p = json.loads((output/'protocol.json').read_text())
    assert p['state'] == 'prepared'; study.ensure_no_active_work()
    assert all(study.common.digest(output/'runner'/n) == h for n, h in p['source_sha256'].items())
    assert reference_rows(p) == json.loads((output/'reused_quality.json').read_text())['rows']
    return private_supervisor(output, [('vit_matched_lr_controller', [sys.executable, '-u', str(output/'runner/vit_matched_lr_study.py'),
                              'run', '--output', str(output)])], label='controller', autostart=True)


def run(output):
    output = Path(output).resolve(); p = json.loads((output/'protocol.json').read_text()); assert p['state'] == 'prepared'
    try:
        reused = reference_rows(p); assert reused == json.loads((output/'reused_quality.json').read_text())['rows']
        assert runtime_record(provenance()) == runtime_record(json.loads((Path(reused[0]['output_dir'])/'provenance.json').read_text()))
        p.update(state='quality', started_utc=study.common.now()); write_json(output/'protocol.json', p)
        new = study.common.execute_stage(output, 'quality', remaining_cases(output), auditor=study.audit_stage)
        verify_combined([*reused, *new]); write_json(output/'report.json', dict(reused_quality=reused, new_quality=new))
        p.update(state='complete', completed_utc=study.common.now()); write_json(output/'protocol.json', p)
        audit(output); write_summary(output, [*reused, *new], p)
    except BaseException as error:
        p.update(state='failed', failed_utc=study.common.now(), failure_type=type(error).__name__, failure=str(error))
        write_json(output/'protocol.json', p); raise


def audit(output):
    output = Path(output).resolve(); p = json.loads((output/'protocol.json').read_text()); assert p['state'] == 'complete'
    reused = reference_rows(p); new = study.audit_stage(output, 'quality'); verify_combined([*reused, *new])
    assert reused == json.loads((output/'reused_quality.json').read_text())['rows']
    assert json.loads((output/'report.json').read_text()) == dict(reused_quality=reused, new_quality=new)
    assert {case_key(r) for r in new} == expected_keys() - reused_keys() and len(new) == 8
    progress = [json.loads(s) for s in (output/'progress.jsonl').read_text().splitlines()]
    assert progress and all(e['active'] == e['submitted']-e['completed'] <= 2 for e in progress)
    assert progress[-1]['completed'] == progress[-1]['total'] == 8 and progress[-1]['active'] == 0
    retained = sum(f.stat().st_size for f in output.rglob('*') if f.is_file())
    assert retained < p['resources']['retained_artifact_bound_bytes']
    result = dict(checks={n: True for n in ('original_reference_evidence_preserved_and_quality_reaudited',
        'exact_eight_missing_cells_and_sixteen_combined_cells', 'unchanged_sources_runtime_config_optimizer_and_split',
        'paired_base_adapter_probe_initialization_and_observed_streams', 'complete_thirty_epoch_diagnostics_and_validation',
        'two_gpu_worker_cap_and_claim_identity', 'no_recipe_selection_weights_test_or_storage_overrun')},
        report_sha256=study.common.digest(output/'report.json'), bytes_retained=retained)
    write_json(output/'audit.json', result); return result


def comparison(rows):
    keyed = {case_key(r): r for r in rows}
    if len(rows) != 16 or set(keyed) != expected_keys(): raise ValueError('Require all sixteen unique fixed cells')
    def accuracy(m, s, r): return 100*keyed[study.family_key(m, 'xavier_gelu'), s, r]['results']['validation']['accuracy']
    def stats(v):
        if not all(math.isfinite(x) for x in v): raise ValueError('Nonfinite comparison')
        return dict(values=v, mean=statistics.mean(v), sd=statistics.stdev(v))
    cells = [dict(method=m, rate=r, global_lr=5e-4*r, seeds=list(SEEDS), accuracy_percent=stats([accuracy(m, s, r) for s in SEEDS]))
             for m in METHODS for r in RATES]
    differences = {r: [accuracy('dplr', s, r)-accuracy('plain', s, r) for s in SEEDS] for r in RATES}
    return dict(cells=cells, paired_dplr_minus_plain=[dict(rate=r, seeds=list(SEEDS), percentage_points=stats(differences[r])) for r in RATES],
                interaction=dict(definition='DPLR-minus-plain at LR1 minus DPLR-minus-plain at LR0.5', seeds=list(SEEDS),
                                 percentage_points=stats([b-a for a, b in zip(differences[.5], differences[1.])])),
                recipe_selection=False)


def write_summary(output, rows, p):
    c = comparison(rows); c['limits'] = p['limits']; write_json(output/'comparison.json', c)
    lines = ['# ViT matched Xavier/LR diagnostic', '',
             'Thirty epochs, paired seeds106–109. Eight audited reused cells plus eight new cells; no recipe selection.', '',
             '| Method | Global LR | Final validation accuracy, mean ± sample SD |', '|---|---:|---:|']
    for cell in c['cells']:
        a = cell['accuracy_percent']; lines.append(f"| {cell['method']} | {cell['global_lr']:.2g} | {a['mean']:.3f} ± {a['sd']:.3f}% |")
    lines += ['', '| Contrast | Paired differences (seed106,107,108,109) | Mean |', '|---|---|---:|']
    for x in c['paired_dplr_minus_plain']:
        a = x['percentage_points']; lines.append(f"| DPLR − plain, LR{x['rate']:g} | "+', '.join(f'{v:+.3f}' for v in a['values'])+f" | {a['mean']:+.3f} |")
    a = c['interaction']['percentage_points']; lines.append('| Interaction, LR1 − LR0.5 | '+', '.join(f'{v:+.3f}' for v in a['values'])+f" | {a['mean']:+.3f} |")
    lines += ['', 'All differences are percentage points. These seeds and the research holdout were inspected before this follow-up was proposed.', '',
              'Limits: '+' '.join(p['limits']), '', '[Evidence](report.json) · [Audit](audit.json) · [Frozen protocol](protocol.json)']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'run', 'audit'))
    parser.add_argument('--output', type=Path); parser.add_argument('--reference', type=Path); parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    output = args.output or Path('runs')/('vit-matched-lr-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.action == 'prepare': print(json.dumps(prepare(output, args.reference, args.launch)), flush=True)
    elif args.action == 'launch': print(launch_controller(output), flush=True)
    elif args.action == 'run': run(output)
    else: print(json.dumps(audit(output), indent=2))

"""Fixed interaction arithmetic and unsafe cross-cohort reuse are checked on CPU."""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import pytest
import vit_matched_lr_study as matched


def test_missing_cells_are_exact_complement_without_new_recipes(tmp_path):
    cases = matched.remaining_cases(tmp_path)
    assert len(cases) == 8 and {(f, s, r) for f, s, r, c in cases} == matched.expected_keys()-matched.reused_keys()
    assert {(f, s, r) for f, s, r, c in cases} == {
        (matched.study.family_key(m, 'xavier_gelu'), s, .5 if m == 'plain' else 1.)
        for m in ('plain', 'dplr') for s in (106, 107, 108, 109)}
    for f, s, r, c in cases:
        assert c.lr == 5e-4*r and c.max_steps == c.lr_schedule_steps == 10560 and c.epochs == 30
        assert c.mlp_init == 'xavier_gelu' and not c.save_weights and not c.evaluate_test
        if c.variant == 'dplr':
            assert c.factor_offsets and c.affine_lr_multiplier == c.lowrank_lr_multiplier == 1.


def paired_rows(tmp_path):
    rows = []
    for m in matched.METHODS:
        for r in matched.RATES:
            for s in matched.SEEDS:
                f = matched.study.family_key(m, 'xavier_gelu'); out = tmp_path/f'{m}_lr{r}_s{s}'; out.mkdir()
                c = matched.study.make_config(f, s, out, r)
                (out/'provenance.json').write_text(json.dumps(dict(python='same', torch='same', cuda_wheel='same', package_versions={}, source_sha256={'trainer':'same'})))
                (out/'dataset.json').write_text(json.dumps({'indices':'same'}))
                d = dict(base_initialization_sha256=f'base_{s}', adapter_initialization_sha256=f'{m}_{s}',
                         probe_seed=2034, probe_sha256='same', live_images_sha256='same',
                         diagnostics=[{'step':0}, {'step':500, 'batch_images_sha256':f'images_{s}', 'batch_labels_sha256':f'labels_{s}'}])
                # Plain gains two points with LR; DPLR's advantage switches from +1 to -1.
                a = .75+(s-106)*.001+(0.02 if r == 1. else 0.)+((.01 if r == .5 else -.01) if m == 'dplr' else 0.)
                rows.append(dict(family=f, seed=s, rate=r, output_dir=str(out), config=asdict(c),
                                 results=dict(completed_steps=10560, token_diagnostics=d, validation={'accuracy':a})))
    return rows


def test_paired_interaction_uses_same_seed_and_reports_both_lrs(tmp_path):
    rows = paired_rows(tmp_path); matched.verify_combined(rows); c = matched.comparison(list(reversed(rows)))
    assert len(c['cells']) == 4 and not c['recipe_selection']
    assert c['paired_dplr_minus_plain'][0]['percentage_points']['values'] == pytest.approx([1.]*4)
    assert c['paired_dplr_minus_plain'][1]['percentage_points']['values'] == pytest.approx([-1.]*4)
    assert c['interaction']['percentage_points']['values'] == pytest.approx([-2.]*4)
    with pytest.raises(ValueError, match='sixteen'): matched.comparison(rows[:-1])
    with pytest.raises(ValueError, match='sixteen'): matched.comparison([*rows[:-1], rows[0]])
    rows[0]['results']['validation']['accuracy'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'): matched.comparison(rows)


@pytest.mark.parametrize('broken', ['duplicate', 'base', 'adapter', 'probe', 'stream', 'runtime', 'dataset', 'config', 'budget'])
def test_incompatible_reuse_fails_before_interaction_claim(tmp_path, broken):
    rows = paired_rows(tmp_path); row = rows[-1]; out = Path(row['output_dir']); diag = row['results']['token_diagnostics']
    if broken == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    elif broken == 'base': diag['base_initialization_sha256'] = 'changed'
    elif broken == 'adapter': diag['adapter_initialization_sha256'] = 'changed'
    elif broken == 'probe': diag['probe_sha256'] = 'changed'
    elif broken == 'stream': diag['diagnostics'][-1]['batch_images_sha256'] = 'changed'
    elif broken == 'runtime':
        p = out/'provenance.json'; d = json.loads(p.read_text()); d['torch'] = 'changed'; p.write_text(json.dumps(d))
    elif broken == 'dataset': (out/'dataset.json').write_text(json.dumps({'indices':'changed'}))
    elif broken == 'config': row['config']['lr'] *= 2
    else: row['results']['completed_steps'] = 100
    with pytest.raises(AssertionError): matched.verify_combined(rows)


def test_reference_mismatch_withholds_gpu_work(tmp_path, monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared'})); calls = []
    def broken(*args): raise AssertionError('reference changed')
    monkeypatch.setattr(matched, 'reference_rows', broken)
    monkeypatch.setattr(matched.study.common, 'execute_stage', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(AssertionError, match='reference changed'): matched.run(tmp_path)
    assert not calls and json.loads((tmp_path/'protocol.json').read_text())['state'] == 'failed'

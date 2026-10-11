"""Bounded refinement grid, update-only selection and failure gates."""
import json
from pathlib import Path

import pytest

import vit_token_refinement as refine


def rows():
    grid={f:tuple(sorted((*rates,*refine.EXTRA_RATES.get(f,()))))for f,rates in refine.study.RATE_GRID.items()}
    values=[]
    for family,rates in {'s24':(1.,),**grid}.items():
        for seed in refine.study.PILOT_SEEDS:
            for rate in rates:
                target={'s12':2.,'s48':.5,'static_r8_diagonal':12.}.get(family,1.)
                events=[dict(step=step,fixed_lowrank_updates=[dict(update_rms=rate/target*(seed-95)*step)])for step in refine.study.PROBES]
                values.append(dict(family=family,seed=seed,rate=rate,results={'token_diagnostics':{'diagnostics':events}}))
    return values


def test_exact_new_grid_and_unchanged_schedule():
    cases=refine.refinement_cases(Path('/tmp/refinement'));assert len(cases)==8
    assert {(f,r)for f,s,r,c in cases}=={('s12',2.),('s48',.5),('static_r8_diagonal',8.),('static_r8_diagonal',12.)}
    for family,seed,rate,cfg in cases:
        assert cfg.max_steps==500 and cfg.lr_schedule_steps==3520 and cfg.hyper_lr_multiplier==rate
        assert cfg.factor_offsets==(family!='static_r8_diagonal') and not cfg.save_weights and not cfg.evaluate_test


def test_exact_grid_selection_uses_only_updates_and_reports_every_failure():
    evidence=rows();assert len(evidence)==36
    result=refine.evaluate_rates(list(reversed(evidence)))
    assert result['quality_gate_passed'] and result['ratio_bounds']==[.5,2.]
    assert result['selected_rates']=={'s24':1.,'s12':2.,'s48':.5,'s24_detached':1.,'static_r8_diagonal':12.}
    with pytest.raises(ValueError,match='exactly'):refine.evaluate_rates(evidence[:-1])
    with pytest.raises(ValueError,match='exactly'):refine.evaluate_rates([*evidence[:-1],evidence[0]])
    for row in evidence:
        if row['family'] in ('s12','static_r8_diagonal'):
            for event in row['results']['token_diagnostics']['diagnostics']:event['fixed_lowrank_updates'][0]['update_rms']*=100
    result=refine.evaluate_rates(evidence)
    assert not result['quality_gate_passed'] and result['failed_families']==['s12','static_r8_diagonal']
    assert set(result['candidates'])==set(refine.study.RATE_GRID)


def test_failed_refined_gate_withholds_quality_and_retains_diagnostics(tmp_path,monkeypatch):
    evidence=rows();old=evidence[:28];new=evidence[28:]
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared'}));calls=[]
    monkeypatch.setattr(refine,'reference_rows',lambda *_:old)
    monkeypatch.setattr(refine,'verify_reused_evidence',lambda *_:None)
    def execute(output,name,cases):
        calls.append(name);assert name=='calibration' and len(cases)==8;return new
    monkeypatch.setattr(refine.study,'execute_stage',execute)
    monkeypatch.setattr(refine,'evaluate_rates',lambda *_:{'quality_gate_passed':False,'failed_families':['s12']})
    with pytest.raises(ValueError,match='withhold quality'):refine.run(tmp_path)
    assert calls==['calibration'] and json.loads((tmp_path/'calibration.json').read_text())['failed_families']==['s12']
    assert json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'

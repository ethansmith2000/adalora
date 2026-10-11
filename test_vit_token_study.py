"""Forward defaults, actual budgets, exact calibration grid and quality gating."""
from dataclasses import asdict, replace
import json

import pytest
import torch

import vit_token_study as study
from train_cifar10 import build_model, load_config, model_stats


def rows():
    values=[]
    for family,seed,rate,cfg in study.pilot_cases(__import__('pathlib').Path('/tmp/pilots')):
        factor=rate if family!='s24' else 1.
        events=[dict(step=step,fixed_lowrank_updates=[{'update_rms':factor*(seed-95)*step}]) for step in study.PROBES]
        values.append(dict(family=family,seed=seed,rate=rate,results={'token_diagnostics':{'diagnostics':events}}))
    return values


def test_new_recipes_default_offsets_only_for_conditional_factors(tmp_path):
    for family in study.FAMILIES:
        cfg=study.make_config(family,98,tmp_path/family)
        assert cfg.factor_offsets == (cfg.variant=='dplr')
        assert cfg.compile and not cfg.save_weights and not cfg.evaluate_test
        assert cfg.max_steps==cfg.lr_schedule_steps==3520 and cfg.epochs==10
        path=tmp_path/'cfg.json';path.write_text(json.dumps(asdict(cfg)))
        assert asdict(load_config(['--override_json',str(path)]))==asdict(cfg)
    cfg=replace(study.make_config('s24',98,tmp_path/'ablation'),factor_offsets=False)
    path=tmp_path/'cfg.json';path.write_text(json.dumps(asdict(cfg)))
    assert load_config(['--override_json',str(path)]).factor_offsets is False


def test_real_model_counts_and_capacity_match(tmp_path):
    torch.set_num_threads(2)
    expected={'plain':2690122,'diagonal':3134794,'static_r8_diagonal':3180874,
              'static_capacity':4341772,'s12':3761626,'s24':4342378,'s48':5503882,'s24_detached':4342378}
    for family,count in expected.items():
        cfg=study.make_config(family,98,tmp_path/family)
        assert model_stats(build_model(cfg),cfg)['parameters']==count
    assert abs(expected['s24']-expected['static_capacity'])==606
    cfg=study.make_config('s24',98,tmp_path/'ablation')
    assert model_stats(build_model(cfg),cfg)['parameters']-model_stats(build_model(replace(cfg,factor_offsets=False)),cfg)['parameters']==46080


def test_cpu_memory_probe_does_not_initialize_cuda_devices(monkeypatch):
    from contextlib import contextmanager
    calls=[]
    @contextmanager
    def fork(*,devices):
        assert devices==[];calls.append(devices);yield
    monkeypatch.setattr(torch.random,'fork_rng',fork)
    result=study.saved_activation_probe()
    assert calls==[[]] and result['dtype']=='FP32 CPU'


def test_rate_selection_is_update_only_and_requires_complete_grid():
    evidence=rows();assert len(evidence)==28
    result=study.select_rates(list(reversed(evidence)))
    assert set(result['selected_rates'].values())=={1.}
    with pytest.raises(ValueError,match='exact 28'):study.select_rates(evidence[:-1])
    with pytest.raises(ValueError,match='exact 28'):study.select_rates([*evidence[:-1],evidence[0]])
    for row in evidence:
        if row['family']=='s12':
            for event in row['results']['token_diagnostics']['diagnostics']:event['fixed_lowrank_updates'][0]['update_rms']*=100
    with pytest.raises(ValueError,match='withhold quality'):study.select_rates(evidence)


def test_audited_row_keeps_config_object_and_original_launch_path(tmp_path):
    names=('config.json','results.json','metrics.jsonl','dataset.json','provenance.json','artifact_manifest.json')
    for name in names:(tmp_path/name).write_text('{}\n')
    cfg=asdict(study.make_config('s24',98,tmp_path/'run'))
    job={'config':str(tmp_path/'config.json'),'family':'s24','seed':98,'rate':1.}
    row=study.evidence_row(job,cfg,{'completed_steps':3520},tmp_path)
    assert row['config']==cfg and row['launched_config_path']==job['config']
    assert row['family']=='s24' and set(row['evidence_sha256'])==set(names)
    assert job['config']==str(tmp_path/'config.json')


def test_calibration_failure_withholds_quality(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared'}));calls=[]
    def execute(output,name,cases):
        calls.append(name)
        if name=='preflight':return []
        assert name=='calibration' and len(cases)==28
        return []
    monkeypatch.setattr(study,'execute_stage',execute)
    with pytest.raises(ValueError,match='exact 28'):study.run(tmp_path)
    assert calls==['preflight','calibration']
    assert json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'


def test_preflight_failure_withholds_all_other_work(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared'}));calls=[]
    def execute(output,name,cases):
        calls.append(name);assert name=='preflight'
        assert all(c.max_steps==c.lr_schedule_steps==12 and c.train_limit==328 for _,_,_,c in cases)
        raise RuntimeError('preflight failed')
    monkeypatch.setattr(study,'execute_stage',execute)
    with pytest.raises(RuntimeError,match='preflight failed'):study.run(tmp_path)
    assert calls==['preflight']

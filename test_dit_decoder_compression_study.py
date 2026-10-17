"""Finite design budgets, audits and failure gating for CP compression."""
from dataclasses import asdict
import json

import pytest
import torch

import dit_decoder_compression_study as study
from train_dit_cifar10 import load_config, make_model
from summarize_dit_cifar10_screen import common_config


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_exact_grid_finite_horizon_fixed_optimizer_and_config_roundtrip(tmp_path):
    cases=study.quality_cases(tmp_path,None)
    assert len(cases)==10
    assert {(f,s) for f,s,c in cases}=={(f,s) for f in study.DESIGNS for s in (120,121)}
    for f,s,c in cases:
        assert c.max_steps==c.lr_schedule_steps==30000 and c.rank==8 and c.lr==8e-4
        assert (c.n_basis,c.decoder_cp_width)==study.DESIGNS[f] and c.factor_offsets
        assert c.adapter_lr_multiplier==1 and c.adam_beta1==.9 and c.adam_beta2==.999
        assert c.compile_model and c.compile_dynamic and c.fused_adamw
        assert c.fid_images==5000 and c.sampling_steps==50 and c.guidance_scales=='1,2'
        assert c.generative_eval_steps=='10000,20000' and c.class_dropout_prob==.1
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(c)))
        assert asdict(load_config(['--override_json',str(p)]))==asdict(c)
    common_config([asdict(c) for f,s,c in cases],'screen')
    for f in study.DESIGNS:
        m=make_model(study.make_config(f,120,tmp_path/f,None))
        stats=m.statistics();assert stats['parameters']==study.PARAMETERS[f]
        assert stats['adapter_parameters']==stats['coefficient_parameters']+stats['decoder_parameters']


def test_all_five_preflights_cover_ragged_compiled_shapes_and_routes(tmp_path):
    cases=study.preflight_cases(tmp_path,None)
    assert len(cases)==5 and {f for f,s,c in cases}==set(study.DESIGNS)
    for f,s,c in cases:
        assert c.max_steps==c.lr_schedule_steps==12 and c.train_limit==328 and c.warmup_steps==2
        assert c.diagnostic_steps=='0,1,3,5,10,12' and c.fid_images==20 and c.sampling_steps==2
        assert c.compile_model and c.sample_images==10


def test_base_and_same_width_head_initialization_are_paired(tmp_path):
    base,heads,rng={},{},{}
    from benchmark_dit_design import digest_parameters
    for f,(s,p) in study.DESIGNS.items():
        torch.manual_seed(120);m=make_model(study.make_config(f,120,tmp_path/f,None))
        base[f]=digest_parameters(m);heads.setdefault(s,set()).add(study.digest_parameters_by_prefix(m,'.coefficients.'))
        rng.setdefault(s,[]).append(torch.get_rng_state().clone())
    assert len(set(base.values()))==1 and all(len(v)==1 for v in heads.values())
    assert all(all(torch.equal(v[0],x) for x in v) for v in rng.values())


@pytest.mark.parametrize('failure',['preflight_worker','preflight_audit'])
def test_preflight_failures_withhold_all_quality_runs(tmp_path,monkeypatch,failure):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','classifier_run':None}));calls=[]
    def execute(output,p,name,cases,**kwargs):
        calls.append(name)
        if failure=='preflight_worker': raise RuntimeError('failed compile/worker')
    def audit(*args):raise RuntimeError('failed gradient/evidence audit')
    monkeypatch.setattr(study,'execute_stage',execute);monkeypatch.setattr(study,'audit_stage',audit)
    with pytest.raises(RuntimeError):study.run(tmp_path)
    assert calls==['preflight'] and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'


def test_summarizer_rejects_changed_training_recipe(tmp_path):
    cs=[asdict(c) for f,s,c in study.quality_cases(tmp_path,None)]
    cs[-1]['warmup_steps']=100
    with pytest.raises(ValueError,match='settings differ'):common_config(cs,'screen')


def test_summary_reports_every_design_and_both_guidance_scales(tmp_path):
    rows=[]
    for f,s,c in study.quality_cases(tmp_path,None):
        rows.append(dict(family=f,config=asdict(c),guidance_evaluations={g:dict(fid=40.+s/100,
             class_agreement={'generated':{'agreement':.8}}) for g in ('1.0','2.0')},
             mean_train_images_per_second_excluding_warmup=2000.,peak_allocated_bytes=1000))
    p=dict(model_statistics={f:{} for f in study.DESIGNS},limits=['fixed screen'])
    study.write_summary(tmp_path,rows,p);d=json.loads((tmp_path/'comparison.json').read_text())
    assert len(d['aggregates'])==5 and all(set(x['final_guidance'])=={'1.0','2.0'} for x in d['aggregates'])
    assert set(d['paired_fid_contrasts']['cp_s64_p128'])=={'dense_s32','dense_s64'}


def test_launch_rejects_modified_frozen_sources_before_supervisor(tmp_path):
    r=tmp_path/'runner';r.mkdir();f=r/'source.py';f.write_text('original')
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','source_sha256':{'source.py':study.digest(f)}}))
    f.write_text('changed')
    with pytest.raises(AssertionError):study.launch_controller(tmp_path)

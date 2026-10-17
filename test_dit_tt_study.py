"""Finite architecture grid, actual optimizer ownership, source and failure gates."""
from dataclasses import asdict,replace
import json

import pytest
import torch

import dit_tt_study as study
from train_dit_cifar10 import load_config,make_model,make_optimizer,optimizer_group_metadata
from summarize_dit_cifar10_screen import common_config


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_exact_grid_with_single_plain_control_per_seed(tmp_path):
    cases=study.quality_cases(tmp_path,None)
    assert len(cases)==24 and study.SEEDS==(130,131)
    assert {(f,s) for f,s,c in cases}=={(study.family_key(d,r),s) for d in study.DESIGNS for r in study.rates_for(d) for s in study.SEEDS}
    assert len({c.output_dir for f,s,c in cases})==24
    for f,s,c in cases:
        assert c.max_steps==c.lr_schedule_steps==30000 and c.lr==8e-4
        assert c.guidance_scales=='1,2' and c.fid_images==5000 and c.sampling_steps==50
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(c)));assert asdict(load_config(['--override_json',str(p)]))==asdict(c)
    common_config([asdict(c) for f,s,c in cases],'screen')
    assert all(c.variant=='dplr' for f,s,c in cases)


def test_six_high_rate_preflights_cover_every_operator(tmp_path):
    cases=study.preflight_cases(tmp_path,None)
    assert len(cases)==6 and {f.rsplit('_a',1)[0] for f,s,c in cases}==set(study.DESIGNS)
    for f,s,c in cases:
        assert s==130 and c.max_steps==c.lr_schedule_steps==12 and c.train_limit==328 and c.batch_size==128
        assert c.adapter_lr_multiplier==2.
        assert c.compile_model and c.compile_dynamic and c.fused_adamw and c.fid_images==20


@pytest.mark.parametrize('seed',[130,131])
def test_initial_base_zero_delta_and_mixing_buffers_are_paired_across_rates(seed):
    proof=study.verify_initial_design_pairing(seed)
    assert proof['base_initialization_paired'] and proof['parameters_identical_across_rates']
    assert len(proof['records'])==12 and all(x['initial_delta_zero'] for x in proof['records'])
    assert all(not r['adapter_buffers_sha256'] for r in proof['records'])


def test_parameter_counts_ownership_and_all_rates(tmp_path):
    from benchmark_dit_design import digest_parameters
    bases=set()
    for design in study.DESIGNS:
        hashes=[]
        for rate in study.rates_for(design):
            cfg=study.make_config(design,130,tmp_path/design,None,rate);torch.manual_seed(130);model=make_model(cfg)
            stats=model.statistics();meta=optimizer_group_metadata(model,make_optimizer(model,replace(cfg,device='cpu')))
            assert stats['parameters']==study.PARAMETERS[design]
            assert sum(g['parameter_count'] for g in meta if g['is_adapter'])==stats['adapter_parameters']
            assert all(g['initial_lr']==cfg.lr*(rate if g['is_adapter'] else 1.) for g in meta)
            bases.add(digest_parameters(model));hashes.append(digest_parameters(model,True))
        assert len(set(hashes))==1
    assert len(bases)==1


@pytest.mark.parametrize('failure',['worker','audit'])
def test_failed_preflight_withholds_every_quality_run(tmp_path,monkeypatch,failure):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','classifier_run':None}));calls=[]
    def execute(output,p,name,cases,**kwargs):
        calls.append(name)
        if failure=='worker':raise RuntimeError('bad worker')
    monkeypatch.setattr(study,'execute_stage',execute)
    monkeypatch.setattr(study,'audit_stage',lambda *args:(_ for _ in ()).throw(RuntimeError('bad audit')))
    with pytest.raises(RuntimeError):study.run(tmp_path)
    assert calls==['preflight'] and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'


def test_all_cells_reported_and_placement_contrasts_separated(tmp_path):
    rows=[]
    for f,s,c in study.quality_cases(tmp_path,None):
        bad=1000. if f=='square_tt11_a2' and s==130 else 0.
        rows.append(dict(family=f,config=asdict(c),guidance_evaluations={g:dict(fid=30+c.adapter_lr_multiplier+bad,
            class_agreement={'generated':{'agreement':.8}}) for g in ('1.0','2.0')},mean_train_images_per_second_excluding_warmup=2000.,peak_allocated_bytes=1000))
    study.write_summary(tmp_path,rows,dict(model_statistics={d:{} for d in study.DESIGNS},limits=['fixed screen']))
    result=json.loads((tmp_path/'comparison.json').read_text());assert len(result['aggregates'])==12
    assert set(result['paired_same_square_placement_fid']['square_tt11_a2'])=={'square_diagonal_a2','square_lowrank_a2','square_block16_a2'}
    assert result['adapter_lr_two_minus_unit_fid']['square_tt11']['2.0']['values']==[1001.,1.]
    assert result['cross_placement_rectangular_reference_fid']['square_tt11_a2']['2.0']['values']==[1000.,0.]
    assert 'adaln' not in result['adapter_lr_two_minus_unit_fid']


def test_undeclared_grid_and_tampered_source_cannot_launch(tmp_path,monkeypatch):
    for design,rate in [('square_fft',1.),('square_tt11',4.),('adaln',2.)]:
        with pytest.raises(ValueError):study.family_key(design,rate)
    runner=tmp_path/'runner';runner.mkdir();source=runner/'source.py';source.write_text('original')
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','source_sha256':{'source.py':study.digest(source)}}))
    source.write_text('changed');monkeypatch.setattr(study,'ensure_inactive',lambda:None)
    with pytest.raises(AssertionError):study.launch_controller(tmp_path)

"""Finite LR follow-up: independent seeds, actual optimizer rates, and failure gates."""
from dataclasses import asdict
import json

import pytest
import torch

import dit_channel_lr_study as study
import dit_channel_factor_study as previous
from train_dit_cifar10 import load_config, make_model, make_optimizer, optimizer_group_metadata
from summarize_dit_cifar10_screen import common_config


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_exact_eighteen_case_grid_and_config_roundtrips(tmp_path):
    cases=study.quality_cases(tmp_path,None)
    assert len(cases)==18
    assert study.RATES==(1.,2.,4.) and study.SEEDS==(126,127)
    assert set(study.DESIGNS)=={'dense_s32','cp_s32_p128','channel_s64'}
    assert {(f,s) for f,s,c in cases}=={(study.family_key(d,r),s) for d in study.DESIGNS for r in study.RATES for s in study.SEEDS}
    assert len({c.output_dir for f,s,c in cases})==18
    for f,s,c in cases:
        assert c.max_steps==c.lr_schedule_steps==30000 and c.lr==8e-4
        assert c.rank==8 and c.factor_offsets and c.class_dropout_prob==.1
        assert c.guidance_scales=='1,2' and c.fid_images==5000 and c.sampling_steps==50
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(c)))
        assert asdict(load_config(['--override_json',str(p)]))==asdict(c)
    common_config([asdict(c) for f,s,c in cases],'screen')


def test_training_recipe_matches_previous_except_new_seed_and_intermediate_rate(tmp_path):
    for design in study.DESIGNS:
        for rate in (1.,4.):
            for seed in study.SEEDS:
                current=asdict(study.make_config(design,seed,tmp_path/design,None,rate))
                old=asdict(previous.make_config(design,seed,tmp_path/design,None,rate))
                assert current==old
    assert not set(study.SEEDS)&set(previous.SEEDS)


def test_three_compiled_high_rate_preflights(tmp_path):
    cases=study.preflight_cases(tmp_path,None)
    assert len(cases)==3 and {f for f,s,c in cases}=={study.family_key(d,4.) for d in study.DESIGNS}
    for f,s,c in cases:
        assert s==126 and c.max_steps==c.lr_schedule_steps==12 and c.train_limit==328
        assert c.batch_size==128 and c.compile_model and c.compile_dynamic and c.fused_adamw
        assert c.fid_images==20 and c.sampling_steps==2
        assert c.diagnostic_steps=='0,1,3,5,10,12' and c.adapter_lr_multiplier==4


@pytest.mark.parametrize('seed',[126,127])
def test_channel_identity_initialization_across_every_rate(seed):
    proof=study.verify_initial_channel_pairing(seed)
    assert proof['initialization_identical_across_rates']
    assert len(proof['records'])==24
    assert all(r['initial_scales_one'] and r['output_factor_zero'] for r in proof['records'])


@pytest.mark.parametrize('seed',[126,127])
def test_actual_ownership_rates_and_paired_initialization(tmp_path,seed):
    from benchmark_dit_design import digest_parameters
    bases=set();heads={}
    for design in study.DESIGNS:
        hashes=[]
        for rate in study.RATES:
            cfg=study.make_config(design,seed,tmp_path/design,None,rate);cfg.device='cpu'
            torch.manual_seed(seed);model=make_model(cfg);stats=model.statistics()
            meta=optimizer_group_metadata(model,make_optimizer(model,cfg))
            assert stats['parameters']==study.PARAMETERS[design]
            assert sum(g['parameter_count'] for g in meta if g['is_adapter'])==stats['adapter_parameters']
            assert all(g['initial_lr']==cfg.lr*(rate if g['is_adapter'] else 1) for g in meta)
            hashes.append((digest_parameters(model),digest_parameters(model,True)))
            bases.add(digest_parameters(model))
            heads.setdefault(cfg.n_basis,set()).add(study.digest_parameters_by_prefix(model,'.coefficients.'))
        assert len(set(hashes))==1
    assert len(bases)==1 and all(len(h)==1 for h in heads.values())


@pytest.mark.parametrize('failure',['worker','audit'])
def test_failed_preflight_withholds_every_quality_run(tmp_path,monkeypatch,failure):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','classifier_run':None}))
    calls=[]
    def execute(output,p,name,cases,**kwargs):
        calls.append(name)
        if failure=='worker':raise RuntimeError('bad worker')
    monkeypatch.setattr(study,'execute_stage',execute)
    monkeypatch.setattr(study,'audit_stage',lambda *args:(_ for _ in ()).throw(RuntimeError('bad audit')))
    with pytest.raises(RuntimeError):study.run(tmp_path)
    assert calls==['preflight'] and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'


def test_every_rate_and_prespecified_contrast_is_reported_without_dropping_bad_seed(tmp_path):
    rows=[]
    for f,s,c in study.quality_cases(tmp_path,None):
        bad=1000. if f=='cp_s32_p128_a4' and s==126 else 0.
        rows.append(dict(family=f,config=asdict(c),guidance_evaluations={g:dict(fid=30+c.adapter_lr_multiplier+bad,
            class_agreement={'generated':{'agreement':.8}}) for g in ('1.0','2.0')},
            mean_train_images_per_second_excluding_warmup=2000.,peak_allocated_bytes=1000))
    study.write_summary(tmp_path,rows,dict(model_statistics={d:{} for d in study.DESIGNS},limits=['fixed screen']))
    result=json.loads((tmp_path/'comparison.json').read_text())
    assert len(result['aggregates'])==9
    differences=result['adapter_lr_paired_fid_differences']
    assert differences['dense_s32']['2_minus_1']['2.0']['values']==[1.,1.]
    assert differences['dense_s32']['4_minus_2']['2.0']['values']==[2.,2.]
    assert differences['cp_s32_p128']['4_minus_1']['2.0']['values']==[1003.,3.]
    assert set(result['paired_fid_contrasts_at_same_rate']['channel_s64_a2'])=={'dense_s32_a2','cp_s32_p128_a2'}
    assert result['prespecified_different_rate_operating_point_fid']['channel_s64_a4_minus_dense_s32_a1']['2.0']['values']==[3.,3.]


def test_unknown_grid_and_modified_source_cannot_launch(tmp_path,monkeypatch):
    for design,rate in [('channel_s32',1.),('channel_s64',8.),('dense_s32',.5)]:
        with pytest.raises(ValueError):study.family_key(design,rate)
    runner=tmp_path/'runner';runner.mkdir();source=runner/'source.py';source.write_text('original')
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','source_sha256':{'source.py':study.digest(source)}}))
    source.write_text('modified');monkeypatch.setattr(study,'ensure_inactive',lambda:None)
    with pytest.raises(AssertionError):study.launch_controller(tmp_path)

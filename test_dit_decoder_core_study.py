"""Equal finite LR budgets, paired initialization and failure gates."""
from dataclasses import asdict
import json

import pytest
import torch

import dit_decoder_core_study as study
from train_dit_cifar10 import load_config,make_model,make_optimizer,optimizer_group_metadata
from summarize_dit_cifar10_screen import common_config


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_exact_equal_twenty_case_full_horizon_grid_and_config_roundtrips(tmp_path):
    cases=study.quality_cases(tmp_path,None);assert len(cases)==20
    assert {(f,s) for f,s,c in cases}=={(study.family_key(d,r),s) for d in study.DESIGNS for r in (.25,1.) for s in (122,123)}
    for f,s,c in cases:
        d=f.rsplit('_a',1)[0];r=float(f.rsplit('_a',1)[1])
        assert (c.n_basis,c.decoder_cp_width,c.decoder_tucker_width)==study.DESIGNS[d]
        assert c.max_steps==c.lr_schedule_steps==30000 and c.rank==8 and c.lr==8e-4
        assert c.adapter_lr_multiplier==r and c.factor_offsets and c.class_dropout_prob==.1
        assert c.guidance_scales=='1,2' and c.fid_images==5000 and c.sampling_steps==50
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(c)))
        assert asdict(load_config(['--override_json',str(p)]))==asdict(c)
    common_config([asdict(c) for f,s,c in cases],'screen')
    for d in study.DESIGNS:
        m=make_model(study.make_config(d,122,tmp_path/d,None));st=m.statistics()
        assert st['parameters']==study.PARAMETERS[d]
        assert st['adapter_parameters']==st['coefficient_parameters']+st['decoder_parameters']


def test_all_five_compiled_preflights_cover_shapes_and_ragged_batches(tmp_path):
    cases=study.preflight_cases(tmp_path,None);assert len(cases)==5
    assert {f for f,s,c in cases}=={study.family_key(d,1.) for d in study.DESIGNS}
    for f,s,c in cases:
        assert s==122 and c.max_steps==c.lr_schedule_steps==12 and c.train_limit==328
        assert c.compile_model and c.fid_images==20 and c.sampling_steps==2
        assert c.diagnostic_steps=='0,1,3,5,10,12' and c.adapter_lr_multiplier==1


@pytest.mark.parametrize('seed',[122,123])
@pytest.mark.parametrize('s',[32,64])
def test_initial_channels_heads_and_core_functions_match(seed,s):
    r=study.verify_initial_core_pairing(seed,s)
    assert r['paired_channel_heads_and_expanded_cores'] and r['max_input_factor_absolute_error']<5e-6


def test_both_rates_target_actual_adapter_parameters_without_changing_initialization(tmp_path):
    for d in study.DESIGNS:
        hashes=[]
        from benchmark_dit_design import digest_parameters
        for rate in study.RATES:
            c=study.make_config(d,122,tmp_path/d,None,rate);c.device='cpu'
            torch.manual_seed(c.seed);m=make_model(c);opt=make_optimizer(m,c);meta=optimizer_group_metadata(m,opt)
            assert sum(g['parameter_count'] for g in meta if g['is_adapter'])==m.statistics()['adapter_parameters']
            assert all(g['initial_lr']==c.lr*(rate if g['is_adapter'] else 1) for g in meta)
            hashes.append((digest_parameters(m),digest_parameters(m,True)))
        assert hashes[0]==hashes[1]


@pytest.mark.parametrize('failure',['worker','audit'])
def test_bad_preflight_withholds_all_full_runs(tmp_path,monkeypatch,failure):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','classifier_run':None}));calls=[]
    def execute(output,p,name,cases,**kwargs):
        calls.append(name)
        if failure=='worker':raise RuntimeError('bad worker')
    monkeypatch.setattr(study,'execute_stage',execute)
    monkeypatch.setattr(study,'audit_stage',lambda *args:(_ for _ in ()).throw(RuntimeError('bad audit')))
    with pytest.raises(RuntimeError):study.run(tmp_path)
    assert calls==['preflight'] and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'


def test_all_ten_design_rate_cells_and_paired_sensitivity_are_reported(tmp_path):
    rows=[]
    for f,s,c in study.quality_cases(tmp_path,None):
        rows.append(dict(family=f,config=asdict(c),guidance_evaluations={g:dict(fid=30+c.adapter_lr_multiplier,
         class_agreement={'generated':{'agreement':.8}}) for g in ('1.0','2.0')},
         mean_train_images_per_second_excluding_warmup=2000.,peak_allocated_bytes=1000))
    study.write_summary(tmp_path,rows,dict(model_statistics={d:{} for d in study.DESIGNS},limits=['fixed screen']))
    c=json.loads((tmp_path/'comparison.json').read_text())
    assert len(c['aggregates'])==10 and all(set(x['final_guidance'])=={'1.0','2.0'} for x in c['aggregates'])
    assert c['adapter_lr_quarter_minus_unit_fid']['dense_s32']['2.0']['values']==[-.75,-.75]
    assert set(c['paired_fid_contrasts_at_same_rate']['tucker_s64_p128_a0.25'])=={'dense_s32_a0.25','cp_s64_p128_a0.25'}


def test_unknown_grid_values_and_modified_frozen_source_cannot_launch(tmp_path):
    with pytest.raises(ValueError):study.family_key('cp_s32_p64',1.)
    with pytest.raises(ValueError):study.family_key('dense_s32',.5)
    r=tmp_path/'runner';r.mkdir();f=r/'source.py';f.write_text('original')
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','source_sha256':{'source.py':study.digest(f)}}))
    f.write_text('modified')
    with pytest.raises(AssertionError):study.launch_controller(tmp_path)

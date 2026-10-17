"""Fixed channel-factor budgets, initialization, rates and failure gates."""
from dataclasses import asdict
import json

import pytest
import torch

import dit_channel_factor_study as study
from train_dit_cifar10 import load_config,make_model,make_optimizer,optimizer_group_metadata
from summarize_dit_cifar10_screen import common_config


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_exact_equal_twenty_case_grid_and_config_roundtrips(tmp_path):
    cases=study.quality_cases(tmp_path,None);assert len(cases)==20
    assert {(f,s) for f,s,c in cases}=={(study.family_key(d,r),s) for d in study.DESIGNS for r in (1.,4.) for s in (124,125)}
    for f,s,c in cases:
        d=f.rsplit('_a',1)[0];r=float(f.rsplit('_a',1)[1])
        assert (c.n_basis,c.decoder_cp_width,c.decoder_tucker_width,c.decoder_channel_mode)==study.DESIGNS[d]
        assert c.max_steps==c.lr_schedule_steps==30000 and c.rank==8 and c.lr==8e-4
        assert c.adapter_lr_multiplier==r and c.factor_offsets and c.class_dropout_prob==.1
        assert c.guidance_scales=='1,2' and c.fid_images==5000 and c.sampling_steps==50
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(c)))
        assert asdict(load_config(['--override_json',str(p)]))==asdict(c)
    common_config([asdict(c) for f,s,c in cases],'screen')
    for d in study.DESIGNS:
        m=make_model(study.make_config(d,124,tmp_path/d,None));st=m.statistics()
        assert st['parameters']==study.PARAMETERS[d]
        assert st['adapter_parameters']==st['coefficient_parameters']+st['decoder_parameters']
        if d=='channel_static':assert st['coefficient_parameters']==0


def test_five_compiled_preflights_cover_ragged_batches_and_every_architecture(tmp_path):
    cases=study.preflight_cases(tmp_path,None);assert len(cases)==5
    assert {f for f,s,c in cases}=={study.family_key(d,4.) for d in study.DESIGNS}
    for f,s,c in cases:
        assert s==124 and c.max_steps==c.lr_schedule_steps==12 and c.train_limit==328
        assert c.compile_model and c.fid_images==20 and c.sampling_steps==2
        assert c.diagnostic_steps=='0,1,3,5,10,12' and c.adapter_lr_multiplier==4


@pytest.mark.parametrize('seed',[124,125])
def test_initial_static_and_modulated_factor_functions_match(seed):
    r=study.verify_initial_channel_pairing(seed)
    assert r['paired_static_factor_and_offset_slots'] and len(r['records'])==2
    assert all(x['initial_scales_one'] and x['max_input_factor_absolute_error']==0 for x in r['records'])


def test_both_rates_have_actual_parameter_ownership_and_same_initialization(tmp_path):
    from benchmark_dit_design import digest_parameters
    bases=set();heads={}
    for d in study.DESIGNS:
        hashes=[]
        for rate in study.RATES:
            c=study.make_config(d,124,tmp_path/d,None,rate);c.device='cpu'
            torch.manual_seed(c.seed);m=make_model(c);st=m.statistics();meta=optimizer_group_metadata(m,make_optimizer(m,c))
            assert sum(g['parameter_count'] for g in meta if g['is_adapter'])==st['adapter_parameters']
            assert all(g['initial_lr']==c.lr*(rate if g['is_adapter'] else 1) for g in meta)
            hashes.append((digest_parameters(m),digest_parameters(m,True)));bases.add(digest_parameters(m))
            if st['coefficient_parameters']:heads.setdefault(c.n_basis,set()).add(study.digest_parameters_by_prefix(m,'.coefficients.'))
        assert hashes[0]==hashes[1]
    assert len(bases)==1 and all(len(v)==1 for v in heads.values())


@pytest.mark.parametrize('failure',['worker','audit'])
def test_failed_preflight_withholds_every_full_run(tmp_path,monkeypatch,failure):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','classifier_run':None}));calls=[]
    def execute(output,p,name,cases,**kwargs):
        calls.append(name)
        if failure=='worker':raise RuntimeError('bad worker')
    monkeypatch.setattr(study,'execute_stage',execute)
    monkeypatch.setattr(study,'audit_stage',lambda *args:(_ for _ in ()).throw(RuntimeError('bad audit')))
    with pytest.raises(RuntimeError):study.run(tmp_path)
    assert calls==['preflight'] and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'


def test_all_design_rate_cells_and_static_cp_contrasts_are_reported(tmp_path):
    rows=[]
    for f,s,c in study.quality_cases(tmp_path,None):
        rows.append(dict(family=f,config=asdict(c),guidance_evaluations={g:dict(fid=30+c.adapter_lr_multiplier,
         class_agreement={'generated':{'agreement':.8}}) for g in ('1.0','2.0')},
         mean_train_images_per_second_excluding_warmup=2000.,peak_allocated_bytes=1000))
    study.write_summary(tmp_path,rows,dict(model_statistics={d:{} for d in study.DESIGNS},limits=['fixed screen']))
    c=json.loads((tmp_path/'comparison.json').read_text());assert len(c['aggregates'])==10
    assert c['adapter_lr_four_minus_unit_fid']['dense_s32']['2.0']['values']==[3.,3.]
    assert set(c['paired_fid_contrasts_at_same_rate']['channel_s64_a4'])=={'dense_s32_a4','channel_static_a4','cp_s32_p128_a4'}


def test_unknown_grid_and_modified_source_cannot_launch(tmp_path,monkeypatch):
    with pytest.raises(ValueError):study.family_key('channel_s128',1.)
    with pytest.raises(ValueError):study.family_key('dense_s32',.5)
    r=tmp_path/'runner';r.mkdir();f=r/'source.py';f.write_text('original')
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','source_sha256':{'source.py':study.digest(f)}}))
    f.write_text('modified');monkeypatch.setattr(study,'ensure_inactive',lambda:None)
    with pytest.raises(AssertionError):study.launch_controller(tmp_path)

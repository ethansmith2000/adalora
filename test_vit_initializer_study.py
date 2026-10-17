"""Initialization invariants, exact equal-budget selection and failure-before-quality guards."""
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

import vit_initializer_study as study
from train_cifar10 import Config, build_model, main, load_config
from vit_initialization import MLP_INITIALIZERS, gelu_second_moment, mlp_initialization_stds
from vit_quality_diagnostics import TokenDiagnostics, base_layer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_legacy_matches_original_frozen_model_initialization_and_outputs():
    path=next(parent/'runs/vit-split-lr-recovery-20261012T013129Z/runner/vit.py'
              for parent in Path(__file__).resolve().parents
              if (parent/'runs/vit-split-lr-recovery-20261012T013129Z/runner/vit.py').exists())
    spec=importlib.util.spec_from_file_location('original_vit_for_test',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg=Config(variant='dplr',factor_offsets=True,width=24,depth=2,heads=3,rank=2,n_basis=4)
    torch.manual_seed(8);actual=build_model(cfg)
    torch.manual_seed(8);expected=old.ViT(img_size=32,patch_size=4,num_classes=10,embed_dim=24,depth=2,heads=3,
                                       qk_norm=False,conditioning='dplr',condition_rank=2,condition_n_basis=4,condition_factor_offsets=True)
    assert all(torch.equal(v,expected.state_dict()[k])for k,v in actual.state_dict().items())
    # Classifier is initially zero, so compare nontrivial internal functions too.
    h=torch.randn(2,7,24)
    for a,b in zip(actual.blocks,expected.blocks):torch.testing.assert_close(a(h),b(h),rtol=0,atol=0)


@pytest.mark.parametrize('initializer',MLP_INITIALIZERS)
def test_mlp_only_scope_rng_and_conditional_zero_rules(initializer):
    cfg=Config(variant='dplr',factor_offsets=True,width=24,depth=2,heads=3,rank=2,n_basis=4,mlp_ratio=7.)
    torch.manual_seed(77);legacy=build_model(cfg);rng=torch.get_rng_state()
    torch.manual_seed(77);actual=build_model(replace(cfg,mlp_init=initializer))
    assert torch.equal(rng,torch.get_rng_state())
    changed=set()
    for key,value in actual.state_dict().items():
        if not torch.equal(value,legacy.state_dict()[key]):changed.add(key)
    allowed={f'blocks.{i}.mlp.{name}'for i in range(2)for name in ('fc1.transform.lowrank.base.weight','fc2.weight')}
    assert changed<=allowed
    if initializer!='legacy':assert changed
    x=torch.randn(2,5,24)
    for block in actual.blocks:
        layer=block.mlp.fc1
        assert torch.count_nonzero(layer.transform.lowrank.delta(x,x))==0
        torch.testing.assert_close(layer(x,x),base_layer(layer)(x),rtol=0,atol=0)
    assert torch.count_nonzero(actual.head.weight)==0


def test_reference_correction_matches_fanin_scaling_and_reference_shapes_are_exact():
    for hidden in (768,1483):
        a,b=mlp_initialization_stds(192,hidden,'reference_fanin')
        assert a==.02 and b==.02*math.sqrt(768/hidden)
        x,y=mlp_initialization_stds(192,hidden,'fanin_gelu')
        assert x==1/math.sqrt(192)
        assert math.isclose(y*math.sqrt(hidden),mlp_initialization_stds(192,768,'fanin_gelu')[1]*math.sqrt(768))
    cfg=Config(width=24,heads=3,depth=1)
    torch.manual_seed(5);a=build_model(cfg)
    torch.manual_seed(5);b=build_model(replace(cfg,mlp_init='reference_fanin'))
    assert all(torch.equal(v,b.state_dict()[k])for k,v in a.state_dict().items())


def test_gelu_moment_agrees_with_independent_monte_carlo_and_narrow_gain_anchor():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6);z=torch.randn(300000,dtype=torch.float64)
        estimate=torch.nn.functional.gelu(z).square().mean().item()
    assert abs(estimate-gelu_second_moment(1.))<.008
    target=768*.02**2*gelu_second_moment(.02*math.sqrt(192))
    for mode in ('fanin_gelu','xavier_gelu','mixed_gelu'):
        a,b=mlp_initialization_stds(192,768,mode)
        assert math.isclose(768*b*b*gelu_second_moment(math.sqrt(192)*a),target,rel_tol=1e-12)


def test_each_method_has_four_distinct_initializer_functions_not_repeated_gain_anchors():
    from vit_quality_diagnostics import parameter_digest
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1210);h=torch.nn.functional.layer_norm(torch.randn(2,5,192),(192,))
    for method in study.METHODS:
        hashes=set();functions=[]
        for mode in study.INITIALIZERS:
            torch.manual_seed(104);model=build_model(study.make_config(study.family_key(method,mode),104,Path('/unused')))
            hashes.add(parameter_digest(model))
            with torch.no_grad():functions.append(model.blocks[0].mlp(h,h))
        assert len(hashes)==4
        assert all(not torch.allclose(a,b)for i,a in enumerate(functions)for b in functions[:i])


def test_invalid_initializer_rejected(tmp_path):
    path=tmp_path/'cfg.json';path.write_text(json.dumps({'mlp_init':'typo'}))
    with pytest.raises(ValueError,match='initializer'):load_config(['--override_json',str(path)])


def test_plain_mlp_update_probe_is_nonzero_after_optimizer_step_without_rng_or_model_mutation():
    from train_cifar10 import SyntheticImages
    cfg=Config(width=24,heads=3,depth=1,mlp_init='fanin_gelu',diagnostic_steps='0,1')
    model=build_model(cfg);diag=TokenDiagnostics(model,cfg,SyntheticImages(8,10),torch.device('cpu'))
    rng=torch.get_rng_state();before={k:v.clone()for k,v in model.state_dict().items()}
    initial=diag.collect(model,0)
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    assert all(torch.equal(v,before[k])for k,v in model.state_dict().items())
    with torch.no_grad():model.blocks[0].mlp.fc2.weight.add_(.001)
    final=diag.collect(model,1)
    assert initial['fixed_mlp_updates'][0]['update_rms']==0
    assert final['fixed_mlp_updates'][0]['update_rms']>0
    assert final['fixed_lowrank_updates'][0]['update_rms']==0


def test_exact_equal_budget_grid_full_duration_and_initializer_metadata(tmp_path):
    cases=study.tuning_cases(tmp_path);assert len(cases)==74
    for method in study.METHODS:
        selected=[c for c in cases if c[0].startswith(method+'__')and'reference_fanin'not in c[0]]
        assert len(selected)==24 and {r for f,s,r,c in selected}=={.5,1.,2.}
        assert {c.mlp_init for f,s,r,c in selected}==set(study.INITIALIZERS)
    for f,s,r,cfg in cases:
        assert cfg.max_steps==cfg.lr_schedule_steps==10560 and cfg.epochs==30 and cfg.lr==5e-4*r
        assert cfg.hyper_lr_multiplier==1. and cfg.factor_offsets==(cfg.variant=='dplr')
        path=tmp_path/'cfg.json';path.write_text(json.dumps(asdict(cfg)));assert asdict(load_config(['--override_json',str(path)]))==asdict(cfg)
    assert not set(study.TUNING_SEEDS)&set(study.QUALITY_SEEDS) and len(study.QUALITY_SEEDS)==4


def tuning_rows():
    rows=[]
    for f,s,r,cfg in study.tuning_cases(Path('/tmp/tuning')):
        # All primary candidates tie; the diagnostic has perfect accuracy.
        rows.append(dict(family=f,seed=s,rate=r,results={'validation':{'accuracy':1. if'reference_fanin'in f else .8,'loss':.6}}))
    return rows


def test_selection_requires_exact_grid_uses_final_validation_and_excludes_extra_diagnostic():
    rows=tuning_rows();selection=study.select_recipes(rows)
    assert selection['selected']=={m:dict(initializer='legacy',rate=1.)for m in study.METHODS}
    assert all(len(c)==12 for c in selection['candidates'].values())
    assert not selection['diagnostic_reference_fanin_selected']
    for row in rows:
        row['results']['best_validation_accuracy']=1.
        if row['family']=='wide__fanin_gelu'and row['rate']==.5:row['results']['validation']['accuracy']=.81
    assert study.select_recipes(rows)['selected']['wide']==dict(initializer='fanin_gelu',rate=.5)
    with pytest.raises(ValueError,match='exact'):study.select_recipes(rows[:-1])
    with pytest.raises(ValueError,match='exact'):study.select_recipes([*rows[:-1],rows[0]])
    rows[0]['results']['validation']['accuracy']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):study.select_recipes(rows)


def test_fresh_quality_anchors_are_fixed_and_identical_recipes_do_not_duplicate_training(tmp_path):
    selection={'selected':{m:dict(initializer='legacy',rate=1.)for m in study.METHODS}}
    cases=study.quality_cases(tmp_path,selection);assert len(cases)==12
    selection['selected']['wide']=dict(initializer='fanin_gelu',rate=2.)
    cases=study.quality_cases(tmp_path,selection);assert len(cases)==16
    assert len({(f,s,r)for f,s,r,c in cases})==len(cases)
    anchors=[c for f,s,r,c in cases if '_anchor_'in c.output_dir]
    assert len(anchors)==4 and all(c.mlp_init=='legacy'and c.lr==5e-4 for c in anchors)
    assert {(f,s)for f,s,r,c in cases if f=='wide__fanin_gelu'}=={('wide__fanin_gelu',s)for s in study.QUALITY_SEEDS}


@pytest.mark.parametrize('failure',['preflight','selection'])
def test_failed_preflight_or_incomplete_grid_withholds_confirmation(tmp_path,monkeypatch,failure):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared','planned_runs':{'preflight':2,'calibration':74,'quality':None}}));calls=[]
    def execute(output,name,cases,**kwargs):
        calls.append(name)
        if name=='preflight':
            if failure=='preflight':raise RuntimeError('preflight failure')
            return []
        assert name=='calibration';return []
    monkeypatch.setattr(study.common,'execute_stage',execute)
    with pytest.raises((RuntimeError,ValueError)):study.run(tmp_path)
    assert 'quality'not in calls and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'


def test_initializer_training_prefix_and_function_probes_are_schedule_consistent(tmp_path):
    cfg=Config(width=24,heads=3,depth=1,mlp_ratio=7.,mlp_init='xavier_gelu',synthetic=True,
               synthetic_samples=12,val_limit=8,epochs=4,batch_size=4,lr_schedule_steps=10,warmup_ratio=.2,
               device='cpu',precision='fp32',num_workers=0,cpu_threads=2,log_every=1,benchmark_warmup_steps=0)
    traces=[]
    for budget in (3,10):
        c=replace(cfg,max_steps=budget,diagnostic_steps='0,1,3'if budget==3 else'0,1,3,10',output_dir=str(tmp_path/str(budget)))
        path=tmp_path/f'{budget}.json';path.write_text(json.dumps(asdict(c)));main(['--override_json',str(path)])
        traces.append([json.loads(l)for l in(tmp_path/str(budget)/'metrics.jsonl').read_text().splitlines()])
        result=json.loads((tmp_path/str(budget)/'results.json').read_text());assert result['training_setup']['mlp_init']=='xavier_gelu'
        assert 'vit_initialization.py'in json.loads((tmp_path/str(budget)/'provenance.json').read_text())['source_sha256']
    for key in ('loss','accuracy','optimizer_lrs'):
        assert [e[key]for e in traces[0]if e['event']=='train']==[e[key]for e in traces[1]if e['event']=='train'][:3]
    assert next(e for e in traces[0]if e['event']=='token_diagnostic'and e['step']==3)==next(e for e in traces[1]if e['event']=='token_diagnostic'and e['step']==3)
    assert not list(tmp_path.rglob('*.pt'))

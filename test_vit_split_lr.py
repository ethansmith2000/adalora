"""Independent optimizer roles, legacy trajectory, schedule prefixes and quality guards."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

import vit_split_lr_study as study
from train_cifar10 import (Config, affine_parameter_ids, build_model, hyperparameter_ids, load_config,
                          main, make_optimizer, optimizer_group_manifest, training_setup)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('variant', ['plain','diagonal','basis','dplr','static_lora_diagonal','exact'])
def test_parameter_roles_cover_every_parameter_once_and_preserve_decay(variant):
    cfg=Config(variant=variant,width=24,depth=2,heads=3,rank=2,n_basis=4,
               factor_offsets=variant in ('basis','dplr'),affine_lr_multiplier=.5,lowrank_lr_multiplier=12.)
    model=build_model(cfg);optimizer=make_optimizer(model,cfg)
    affine=affine_parameter_ids(model);hyper=hyperparameter_ids(model)
    assigned=[id(p)for g in optimizer.param_groups for p in g['params']]
    assert len(assigned)==len(set(assigned))==len(list(model.parameters()))
    for name,p in model.named_parameters():
        group=next(g for g in optimizer.param_groups if any(p is q for q in g['params']))
        expected='base' if id(p)not in hyper else 'affine' if id(p)in affine else 'lowrank'
        assert group['lr_role']==expected
        assert group['lr']==cfg.lr*{'base':1.,'affine':.5,'lowrank':12.}[expected]
        assert group['weight_decay']==(cfg.weight_decay if p.ndim>=2 and not name.endswith(('U_offset','V_offset')) else 0)
        assert (id(p)in affine)==any(v in name for v in ('to_scale.','to_shift.','to_affine.'))
    names=[n for g in optimizer_group_manifest(model,optimizer)for n in g['parameter_names']]
    assert set(names)==set(dict(model.named_parameters()))


def legacy_optimizer(model,cfg):
    ids=hyperparameter_ids(model);offsets={id(p)for name,p in model.named_parameters()if name.endswith(('U_offset','V_offset'))};groups=[]
    for hyper in (False,True):
        for decay in (False,True):
            ps=[p for p in model.parameters()if (id(p)in ids)==hyper and (p.ndim>=2 and id(p)not in offsets)==decay]
            if ps:groups.append(dict(params=ps,lr=cfg.lr*(cfg.hyper_lr_multiplier if hyper else 1),weight_decay=cfg.weight_decay if decay else 0))
    return torch.optim.AdamW(groups,betas=(.9,.999),fused=False)


def test_unspecified_rates_preserve_legacy_optimizer_and_actual_training_trajectory():
    cfg=Config(variant='dplr',factor_offsets=True,width=24,depth=1,heads=3,rank=2,n_basis=4,device='cpu',hyper_lr_multiplier=3.)
    torch.manual_seed(3);a=build_model(cfg)
    torch.manual_seed(3);b=build_model(cfg)
    oa=make_optimizer(a,cfg);ob=legacy_optimizer(b,cfg)
    assert oa.state_dict()['param_groups']==ob.state_dict()['param_groups']
    x=torch.randn(4,3,32,32);labels=torch.tensor([0,1,2,3])
    for _ in range(3):
        for m,opt in ((a,oa),(b,ob)):
            opt.zero_grad();torch.nn.functional.cross_entropy(m(x),labels).backward();opt.step()
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=0,atol=0)
    assert 'adapter_lr_multipliers'not in training_setup(cfg,100)


def test_explicit_equal_rates_preserve_function_and_updates_with_offsets():
    cfg=Config(variant='dplr',factor_offsets=True,width=24,depth=1,heads=3,rank=2,n_basis=4,device='cpu',hyper_lr_multiplier=2.)
    torch.manual_seed(5);a=build_model(cfg)
    torch.manual_seed(5);b=build_model(cfg)
    oa=make_optimizer(a,cfg);ob=make_optimizer(b,replace(cfg,affine_lr_multiplier=2.,lowrank_lr_multiplier=2.))
    x=torch.randn(3,3,32,32);labels=torch.tensor([0,1,2])
    for _ in range(3):
        for m,opt in ((a,oa),(b,ob)):
            opt.zero_grad();torch.nn.functional.cross_entropy(m(x),labels).backward();opt.step()
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=0,atol=0)


def test_affine_rate_does_not_follow_lowrank_multiplier_and_unspecified_side_falls_back():
    cfg=Config(variant='dplr',factor_offsets=True,width=24,depth=1,heads=3,rank=2,n_basis=4,device='cpu',hyper_lr_multiplier=3.,affine_lr_multiplier=1.)
    model=build_model(cfg);groups={id(p):g for g in make_optimizer(model,cfg).param_groups for p in g['params']}
    layer=model.blocks[0].mlp.fc1
    assert groups[id(layer.transform.to_scale.weight)]['lr']==cfg.lr
    assert groups[id(layer.transform.to_shift.weight)]['lr']==cfg.lr
    assert groups[id(layer.transform.lowrank.to_coef.weight)]['lr']==cfg.lr*3
    assert groups[id(layer.transform.lowrank.V_offset)]['lr']==cfg.lr*3


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),True,'4'])
def test_invalid_split_rate_rejected(tmp_path,value):
    path=tmp_path/'cfg.json';path.write_text(json.dumps({'affine_lr_multiplier':value}))
    with pytest.raises(ValueError,match='finite and positive'):load_config(['--override_json',str(path)])


def test_split_rate_cli_is_numeric():
    cfg=load_config(['--affine-lr-multiplier','1','--lowrank-lr-multiplier','12'])
    assert cfg.affine_lr_multiplier==1. and cfg.lowrank_lr_multiplier==12.


def test_pilot_is_exact_long_schedule_prefix_including_group_lrs_and_diagnostics(tmp_path):
    common=Config(variant='static_lora_diagonal',width=24,heads=3,depth=1,rank=2,n_basis=4,
                  affine_lr_multiplier=1.,lowrank_lr_multiplier=12.,synthetic=True,synthetic_samples=12,val_limit=8,
                  epochs=4,batch_size=4,lr_schedule_steps=10,warmup_ratio=.2,device='cpu',precision='fp32',
                  num_workers=0,cpu_threads=2,log_every=1,benchmark_warmup_steps=0)
    records=[]
    for budget in (3,10):
        cfg=replace(common,max_steps=budget,output_dir=str(tmp_path/str(budget)),diagnostic_steps='0,1,3' if budget==3 else '0,1,3,10')
        path=tmp_path/f'{budget}.json';path.write_text(json.dumps(asdict(cfg)));main(['--override_json',str(path)])
        records.append([json.loads(line)for line in(tmp_path/str(budget)/'metrics.jsonl').read_text().splitlines()])
        result=json.loads((tmp_path/str(budget)/'results.json').read_text())
        assert result['training_setup']['adapter_lr_multipliers']=={'affine':1.,'lowrank':12.}
        assert set(g['role']for g in result['optimizer_groups'])=={'base','affine','lowrank'}
    for key in ('loss','lr','optimizer_lrs','grad_norm_before_clip','hyper_grad_norm_before_clip'):
        assert [e[key]for e in records[0]if e['event']=='train']==[e[key]for e in records[1]if e['event']=='train'][:3]
    assert next(e for e in records[0]if e['event']=='token_diagnostic' and e['step']==3)==next(e for e in records[1]if e['event']=='token_diagnostic' and e['step']==3)
    assert not list(tmp_path.rglob('*.pt'))


def test_exact_plan_fresh_seeds_split_rates_and_fixed_joint_control(tmp_path):
    cases=study.pilot_cases(tmp_path);assert len(cases)==20
    for family,seed,rate,cfg in cases:
        assert seed in (100,101) and cfg.max_steps==1200 and cfg.lr_schedule_steps==10560 and cfg.epochs==30
        assert cfg.hyper_lr_multiplier==1. and cfg.affine_lr_multiplier==(12. if family=='static_joint_12' else 1.)
        assert cfg.lowrank_lr_multiplier==rate and cfg.factor_offsets==(cfg.variant=='dplr')
        path=tmp_path/'cfg.json';path.write_text(json.dumps(asdict(cfg)));assert asdict(load_config(['--override_json',str(path)]))==asdict(cfg)
    assert set(study.QUALITY_SEEDS)=={102,103} and not set(study.PILOT_SEEDS)&set(study.QUALITY_SEEDS)
    assert len(study.FAMILIES)==8
    for f,s,r,cfg in study.preflight_cases(tmp_path):
        assert cfg.max_steps==cfg.lr_schedule_steps==12 and cfg.train_limit==328
        assert not cfg.save_weights and not cfg.evaluate_test


def evidence_rows():
    result=[]
    for f,s,r,cfg in study.pilot_cases(Path('/tmp/pilots')):
        target={'s48':.5,'static_split':12.}.get(f,1.)
        result.append(dict(family=f,seed=s,rate=r,results={'validation':{'accuracy':1/r},'token_diagnostics':{
            'diagnostics':[dict(step=t,fixed_lowrank_updates=[dict(update_rms=r/target*(s-99)*t)])for t in study.PROBES]}}))
    return result


def test_selection_is_update_only_requires_exact_grid_and_does_not_match_fixed_controls():
    rows=evidence_rows();selected=study.select_rates(rows)
    assert selected['quality_gate_passed'] and selected['selected_rates']=={'s24':1.,'static_unit':1.,'static_joint_12':12.,'s48':.5,'static_split':12.}
    assert set(selected['candidates'])=={'s48','static_split'}
    for row in rows:row['results']['validation']['accuracy']=-999
    assert study.select_rates(rows)==selected
    with pytest.raises(ValueError,match='exactly'):study.select_rates(rows[:-1])
    with pytest.raises(ValueError,match='exactly'):study.select_rates([*rows[:-1],rows[0]])
    for row in rows:
        if row['family']=='static_split':
            for e in row['results']['token_diagnostics']['diagnostics']:e['fixed_lowrank_updates'][0]['update_rms']*=100
    failed=study.select_rates(rows);assert not failed['quality_gate_passed'] and failed['failed_families']==['static_split']


@pytest.mark.parametrize('failure_stage',['preflight','calibration'])
def test_preflight_or_calibration_failure_withholds_quality(tmp_path,monkeypatch,failure_stage):
    (tmp_path/'protocol.json').write_text(json.dumps({'state':'prepared'}));calls=[]
    def execute(output,name,cases,**kwargs):
        calls.append(name)
        if name==failure_stage:
            if name=='preflight':raise RuntimeError('preflight failed')
            return evidence_rows()
        assert name=='preflight';return []
    monkeypatch.setattr(study.common,'execute_stage',execute);monkeypatch.setattr(study,'verify_pairs',lambda _:None)
    if failure_stage=='calibration':monkeypatch.setattr(study,'select_rates',lambda _:{'quality_gate_passed':False})
    with pytest.raises((ValueError,RuntimeError)):study.run(tmp_path)
    assert 'quality'not in calls and json.loads((tmp_path/'protocol.json').read_text())['state']=='failed'
    if failure_stage=='calibration':assert not json.loads((tmp_path/'calibration.json').read_text())['quality_gate_passed']


def test_summary_uses_actual_duration_seeds_and_independent_rates(tmp_path):
    rows=[]
    for f in study.FAMILIES:
        for s in study.QUALITY_SEEDS:
            cfg=study.make_config(f,s,tmp_path/f,12. if f=='static_joint_12' else 1.)
            rows.append(dict(family=f,seed=s,config=asdict(cfg),results=dict(parameters=10,validation={'accuracy':.7,'loss':1.},peak_training_allocated_bytes=2**20)))
    study.write_summary(tmp_path,rows,{'limits':['Two-seed test']})
    text=(tmp_path/'summary.md').read_text();assert '30 epochs'in text and '102/103'in text and '98/99'not in text
    assert 'Affine LR'in text and 'Low-rank LR'in text

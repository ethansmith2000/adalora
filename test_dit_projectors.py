"""Sharing must reuse factors, sum layer gradients and preserve diagnostic ownership."""
import copy
import json
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch
from torch import nn

from benchmark_dit_design import BranchDiagnostics, digest_parameters
from dit_cifar10 import CIFAR10DiT
from train_dit_cifar10 import Config, main, make_optimizer, update_ema


def model(groups=2, activation='linear', width=24, basis=4):
    return CIFAR10DiT(width=width,depth=6,heads=3,patch_size=8,variant='dplr',rank=2,
                     n_basis=basis,projector_groups=groups,coefficient_activation=activation)


@pytest.mark.parametrize('groups,basis,expected',[(6,16,774336),(2,48,774336),(1,96,774336),(2,16,258112)])
def test_actual_unique_parameter_budget_and_optimizer(groups,basis,expected):
    m=CIFAR10DiT(variant='dplr',rank=8,n_basis=basis,projector_groups=groups)
    stats=m.statistics()
    assert stats['adapter_parameters']==expected
    assert stats['parameters']==4179696+expected
    assert len({id(block.adapter) for block in m.blocks})==groups
    opt=make_optimizer(m,Config(variant='dplr',projector_groups=groups,device='cpu'))
    ids=[id(p) for group in opt.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))


def test_base_and_equal_shape_activation_initialization():
    hashes=[];adapter_hashes=[]
    for groups,basis,activation in ((6,4,'linear'),(2,12,'linear'),(1,24,'linear'),(2,12,'silu')):
        torch.manual_seed(71);m=model(groups,activation,basis=basis)
        hashes.append(digest_parameters(m));adapter_hashes.append(digest_parameters(m,True))
        assert all(torch.count_nonzero(a.V)==0 for _,a in m.grouped_adapters())
        assert all(torch.count_nonzero(a.U)>0 for _,a in m.grouped_adapters())
    assert len(set(hashes))==1 and adapter_hashes[1]==adapter_hashes[3]


@pytest.mark.parametrize('activation',['linear','silu'])
def test_reuse_and_shared_gradient_equals_sum_of_untied_layer_gradients(activation):
    torch.set_num_threads(2);torch.manual_seed(72)
    shared=model(2,activation)
    # Open base gates/output and adapters so all paths contribute nontrivially.
    with torch.no_grad():
        for block in shared.blocks:
            block.modulation[-1].bias.normal_(std=.1)
        shared.output.weight.normal_(std=.03)
        for _,a in shared.grouped_adapters():a.V.normal_(std=.03)
    independent=copy.deepcopy(shared)
    for block in independent.blocks:block.adapter=copy.deepcopy(block.adapter)
    independent.projector_groups=independent.depth
    x=torch.randn(2,3,32,32);ts=torch.tensor([11,70]);labels=torch.tensor([2,5])
    adapters=[a for _,a in shared.grouped_adapters()]
    with patch.object(adapters[0],'factors',wraps=adapters[0].factors) as first, patch.object(adapters[1],'factors',wraps=adapters[1].factors) as second:
        a=shared(x,ts,labels);a.square().sum().backward()
        assert first.call_count==second.call_count==1
    b=independent(x,ts,labels);b.square().sum().backward()
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
    for layers,adapter in shared.grouped_adapters():
        for name,p in adapter.named_parameters():
            expected=sum(dict(independent.blocks[index].adapter.named_parameters())[name].grad for index in layers)
            assert expected.abs().sum()>0
            assert torch.allclose(p.grad,expected,atol=1e-6,rtol=1e-5),name
    for name,p in shared.named_parameters():
        if '.adapter.' not in name:
            assert torch.allclose(p.grad,dict(independent.named_parameters())[name].grad,atol=1e-6,rtol=1e-5),name
    hooks=BranchDiagnostics(shared)
    try:
        shared(x,ts,labels)
        records=hooks.compute()
        for index,state in enumerate(hooks.states):
            expected=shared.blocks[index].adapter(state['h'],shared.condition_norm(shared.time_embedding(__import__('dit_cifar10').sinusoidal(ts,256))+shared.class_embedding(labels)))
            assert torch.allclose(state['delta'],expected)
            assert records[index]['adapter_input_rms']==pytest.approx(state['h'].float().square().mean().sqrt().item())
        assert not torch.equal(hooks.states[0]['delta'],hooks.states[2]['delta'])
    finally:hooks.close()


def test_state_roundtrip_and_ema_copy_preserve_aliases():
    original=model(2,'silu');ema=copy.deepcopy(original);restored=model(2,'silu')
    restored.load_state_dict(original.state_dict())
    for current in (ema,restored):
        assert current.blocks[0].adapter is current.blocks[2].adapter
        assert current.blocks[3].adapter is current.blocks[5].adapter
        assert current.blocks[0].adapter is not current.blocks[3].adapter
        assert digest_parameters(current,True)==digest_parameters(original,True)
    assert ema.blocks[0].adapter is not original.blocks[0].adapter
    with torch.no_grad():
        original.blocks[0].adapter.V.fill_(1.)
    update_ema(ema,original,.9)
    assert torch.allclose(ema.blocks[0].adapter.V,torch.full_like(ema.blocks[0].adapter.V,.1))


def test_shared_silu_short_run_is_exact_prefix_of_longer_schedule(tmp_path):
    events=[]
    for maximum in (12,18):
        cfg=Config(variant='dplr',width=24,depth=6,heads=3,patch_size=8,rank=2,n_basis=4,
                   projector_groups=2,coefficient_activation='silu',max_steps=maximum,lr_schedule_steps=30000,
                   warmup_steps=2,batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,
                   device='cpu',precision='fp32',cpu_threads=2,num_workers=0,log_every=6,eval_every=6,
                   benchmark_warmup_steps=2,sample_images=0,output_dir=str(tmp_path/str(maximum)),
                   diagnostic_steps='0,1,3,5,10,12'+(',18' if maximum==18 else ''))
        path=tmp_path/f'{maximum}.json';path.write_text(json.dumps(asdict(cfg)))
        row=main(['--override_json',str(path)])
        events.append([event for event in row['design_diagnostics']['diagnostics'] if event['step']<=12])
    assert events[0]==events[1]


@pytest.mark.parametrize('groups,activation,variant',[(4,'linear','dplr'),(-1,'linear','dplr'),(7,'linear','dplr'),(2,'bad','dplr'),(2,'linear','conditional_core'),(0,'silu','adaln_zero')])
def test_invalid_sharing_config(groups,activation,variant):
    with pytest.raises(ValueError):CIFAR10DiT(projector_groups=groups,coefficient_activation=activation,variant=variant)


@pytest.mark.parametrize('groups,activation',[(1,'linear'),(2,'silu')])
def test_diagnostics_do_not_change_training_and_all_shared_parameters_wake(tmp_path,groups,activation):
    results=[]
    for enabled in (False,True):
        cfg=Config(variant='dplr',width=24,depth=6,heads=3,patch_size=8,rank=2,n_basis=4,
                   projector_groups=groups,coefficient_activation=activation,
                   max_steps=12,warmup_steps=2,batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,
                   device='cpu',precision='fp32',cpu_threads=2,num_workers=0,log_every=12,eval_every=12,
                   benchmark_warmup_steps=2,sample_images=0,output_dir=str(tmp_path/str(enabled)),
                   diagnostic_steps='0,1,3,5,10,12' if enabled else '')
        path=tmp_path/f'{enabled}.json';path.write_text(json.dumps(asdict(cfg)))
        results.append(main(['--override_json',str(path)]))
    for key in ('raw_validation','ema_validation'):
        for metric in ('noise_mse','class_prediction_change_rms'):
            assert results[0][key][metric]==results[1][key][metric]
    diag=results[1]['design_diagnostics']
    assert len(diag['projector_group_layers'])==groups
    assert all(step<=10 for step in diag['adapter_first_nonzero_gradient_observed'].values())
    assert len(diag['adapter_first_nonzero_gradient_observed'])==groups*4
    last=diag['diagnostics'][-1]
    assert len(last['fixed_condition_projector_magnitudes'])==groups
    assert len(last['branches'])==len(last['fixed_token_condition_function_updates'])==6
    assert not list(tmp_path.rglob('*.pt'))

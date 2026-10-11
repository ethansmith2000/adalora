"""Shared decoders retain independent head functions and accumulate gradients once."""
import copy
import json
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from benchmark_dit_design import digest_parameters
from dit_cifar10 import CIFAR10DiT
from dit_basis_head_screen import FAMILIES
from train_dit_cifar10 import Config, main, make_optimizer, update_ema


def model(decoders=2,hidden=0,activation='linear',basis=4):
    return CIFAR10DiT(width=24,depth=6,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=basis,
                     decoder_groups=decoders,coefficient_hidden=hidden,coefficient_activation=activation)


@pytest.mark.parametrize('family,count,tensors',[
    ('independent_linear',774336,24),('two_bases_s16',282816,16),
    ('two_bases_s44',777744,16),('all_bases_s77',769692,14),
    ('mlp_linear',802272,36),('mlp_silu',802272,36)])
def test_unique_budget_and_optimizer(family,count,tensors):
    m=CIFAR10DiT(variant='dplr',**FAMILIES[family]);s=m.statistics()
    assert s['adapter_parameters']==count and s['parameters']==4179696+count
    assert s['adapter_parameter_tensors']==tensors
    assert count==s['coefficient_parameters']+s['decoder_parameters']
    optimizer=make_optimizer(m,Config(variant='dplr',device='cpu'))
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    assert len({id(block.adapter.coefficients) for block in m.blocks})==6


@pytest.mark.parametrize('decoders,hidden,activation',[(1,0,'linear'),(2,0,'linear'),(2,7,'silu')])
def test_shared_decoder_gradient_equals_sum_of_independent_decoders(decoders,hidden,activation):
    torch.set_num_threads(2);torch.manual_seed(75)
    shared=model(decoders,hidden,activation)
    with torch.no_grad():
        for block in shared.blocks: block.modulation[-1].bias.normal_(std=.1)
        shared.output.weight.normal_(std=.03)
        for _,adapter in shared.grouped_decoders(): adapter.V.normal_(std=.03)
    independent=copy.deepcopy(shared)
    for block in independent.blocks:
        block.adapter.U=torch.nn.Parameter(block.adapter.U.detach().clone())
        block.adapter.V=torch.nn.Parameter(block.adapter.V.detach().clone())
    independent.decoder_groups=0
    x=torch.randn(2,3,32,32);ts=torch.tensor([12,73]);labels=torch.tensor([2,7])
    a=shared(x,ts,labels);b=independent(x,ts,labels)
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
    a.square().sum().backward();b.square().sum().backward()
    for layers,adapter in shared.grouped_decoders():
        for name in ('U','V'):
            expected=sum(getattr(independent.blocks[i].adapter,name).grad for i in layers)
            assert expected.abs().sum()>0
            assert torch.allclose(getattr(adapter,name).grad,expected,atol=1e-6,rtol=1e-5)
    for i,block in enumerate(shared.blocks):
        for p,q in zip(block.adapter.coefficients.parameters(),independent.blocks[i].adapter.coefficients.parameters()):
            assert p.grad.abs().sum()>0 and torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5)
    # Same bases + same condition retain different transforms through independent heads.
    c=torch.randn(2,24)
    u,v=shared.blocks[0].adapter.factors(c);u2,v2=shared.blocks[1].adapter.factors(c)
    assert not torch.allclose(u@v,u2@v2)
    for name,p in shared.named_parameters():
        if '.adapter.' not in name:
            assert torch.allclose(p.grad,dict(independent.named_parameters())[name].grad,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('activation',['linear','silu'])
def test_mlp_coefficients_match_explicit_hidden_activation(activation):
    torch.manual_seed(76);m=model(0,7,activation);adapter=m.blocks[0].adapter
    c=torch.randn(3,24)
    first,last=adapter.coefficients[0],adapter.coefficients[2]
    hidden=F.linear(c,first.weight,first.bias)
    expected=F.linear(F.silu(hidden) if activation=='silu' else hidden,last.weight,last.bias)
    assert torch.equal(adapter.coefficient_values(c),expected)
    u,v=adapter.factors(c);a,b=expected.chunk(2,-1)
    expected_u=sum(a[:,i,None,None]*adapter.U.view(24,4,2)[:,i] for i in range(4))
    expected_v=sum(b[:,i,None,None]*adapter.V.view(4,2,96)[i] for i in range(4))
    assert torch.allclose(u,expected_u,atol=1e-6) and torch.equal(v,expected_v)


def test_ema_state_aliases_and_common_initialization():
    hashes=[]
    for kwargs in ({},{'decoders':1,'basis':12},{'hidden':7,'activation':'linear'},{'hidden':7,'activation':'silu'}):
        torch.manual_seed(77);m=model(**kwargs);hashes.append(digest_parameters(m))
        assert all(torch.count_nonzero(a.V)==0 for _,a in m.grouped_adapters())
    assert len(set(hashes))==1
    torch.manual_seed(78);linear=model(0,7,'linear')
    torch.manual_seed(78);silu=model(0,7,'silu')
    assert digest_parameters(linear,True)==digest_parameters(silu,True)
    m=model(2);ema=copy.deepcopy(m);restored=model(2);restored.load_state_dict(m.state_dict())
    for current in (ema,restored):
        assert current.blocks[0].adapter is not current.blocks[1].adapter
        assert current.blocks[0].adapter.U is current.blocks[2].adapter.U
        assert current.blocks[0].adapter.V is current.blocks[2].adapter.V
        assert current.blocks[0].adapter.U is not current.blocks[3].adapter.U
        assert current.blocks[0].adapter.coefficients is not current.blocks[2].adapter.coefficients
    with torch.no_grad():m.blocks[0].adapter.V.fill_(1)
    update_ema(ema,m,.9)
    assert torch.allclose(ema.blocks[2].adapter.V,torch.full_like(ema.blocks[2].adapter.V,.1))


@pytest.mark.parametrize('groups,hidden,whole,variant',[(4,0,0,'dplr'),(-1,0,0,'dplr'),(True,0,0,'dplr'),(2,0,2,'dplr'),(0,-1,0,'dplr'),(0,True,0,'dplr'),(2,0,0,'exact'),(0,4,0,'adaln_zero')])
def test_invalid_partial_or_hidden_design(groups,hidden,whole,variant):
    with pytest.raises(ValueError):CIFAR10DiT(decoder_groups=groups,coefficient_hidden=hidden,projector_groups=whole,variant=variant)


@pytest.mark.parametrize('decoders,hidden,activation',[(2,0,'linear'),(0,7,'silu')])
def test_diagnostics_preserve_training_and_count_unique_ownership(tmp_path,decoders,hidden,activation):
    rows=[]
    for enabled in (False,True):
        cfg=Config(variant='dplr',width=24,depth=6,heads=3,patch_size=8,rank=2,n_basis=4,
                   decoder_groups=decoders,coefficient_hidden=hidden,coefficient_activation=activation,
                   max_steps=12,warmup_steps=2,batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,
                   device='cpu',precision='fp32',cpu_threads=2,num_workers=0,log_every=12,eval_every=12,
                   sample_images=0,diagnostic_steps='0,1,3,5,10,12' if enabled else '',
                   output_dir=str(tmp_path/str(enabled)))
        path=tmp_path/f'{enabled}.json';path.write_text(json.dumps(asdict(cfg)))
        rows.append(main(['--override_json',str(path)]))
    for key in ('raw_validation','ema_validation'):
        for metric in ('noise_mse','class_prediction_change_rms'):
            assert rows[0][key][metric]==rows[1][key][metric]
    diag=rows[1]['design_diagnostics'];event=diag['diagnostics'][-1]
    assert len(diag['adapter_first_nonzero_gradient_observed'])==rows[1]['adapter_parameter_tensors']
    assert all(step<=10 for step in diag['adapter_first_nonzero_gradient_observed'].values())
    assert len(event['decoder_gradient_groups_after_clip'])==(decoders or 6)
    assert len(event['coefficient_head_gradient_groups_after_clip'])==6
    assert len(event['fixed_condition_projector_magnitudes'])==6
    assert not list(tmp_path.rglob('*.pt'))

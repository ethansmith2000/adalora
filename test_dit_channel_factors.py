"""Channel scales, static control, gradients, dtype and legacy compatibility."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dit_cifar10 import ChannelModulatedLowRank,CIFAR10DiT
from train_dit_cifar10 import Config,load_config,main


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def adapter(mode='modulated'):
    return ChannelModulatedLowRank(3,5,4,'dplr',rank=2,n_basis=3,channel_mode=mode,
                                  initialization_seed=90,factor_offsets=True).double()


def test_scales_and_all_gradients_match_explicit_row_column_operator():
    m=adapter()
    with torch.no_grad():
        m.V_static.normal_();m.U_offset.normal_();m.V_offset.normal_()
        m.U_scale_basis.normal_(std=.1);m.V_scale_basis.normal_(std=.1)
    x=torch.randn(2,4,3,dtype=torch.double,requires_grad=True)
    c=torch.randn(2,4,dtype=torch.double,requires_grad=True)
    a,b=m.coefficient_values(c).chunk(2,-1)
    gu=1+a@m.U_scale_basis;gv=1+b@m.V_scale_basis
    u=torch.diag_embed(gu)@m.U_static+m.U_offset
    v=m.V_static@torch.diag_embed(gv)+m.V_offset
    actual=m(x,c);expected=x@u@v
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
    ps=[x,c,*m.parameters()]
    aa=torch.autograd.grad(actual.square().sum(),ps,retain_graph=True)
    bb=torch.autograd.grad(expected.square().sum(),ps)
    for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    assert torch.autograd.gradcheck(m,(x,c))


def test_static_has_no_unused_heads_and_is_condition_independent():
    m=adapter('static')
    assert not hasattr(m,'coefficients') and not hasattr(m,'U_scale_basis')
    with torch.no_grad():m.V_static.normal_()
    x=torch.randn(2,4,3,dtype=torch.double)
    c=torch.randn(2,4,dtype=torch.double,requires_grad=True)
    out=m(x,c);expected=x@(m.U_static+m.U_offset)@(m.V_static+m.V_offset)
    torch.testing.assert_close(out,expected,rtol=1e-11,atol=1e-11)
    assert torch.equal(out,m(x,torch.randn_like(c)))
    assert torch.autograd.grad(out.sum(),c,allow_unused=True)[0] is None


def test_zero_delta_identity_scales_and_paired_static_functions():
    args=dict(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,
              factor_offsets=True,enable_unconditional=True)
    torch.manual_seed(124);static=CIFAR10DiT(**args,decoder_channel_mode='static');rng=torch.get_rng_state().clone()
    torch.manual_seed(124);mod=CIFAR10DiT(**args,decoder_channel_mode='modulated')
    assert torch.equal(rng,torch.get_rng_state())
    c=torch.randn(2,24);x=torch.randn(2,3,24)
    for a,b in zip((x.adapter for x in static.blocks),(x.adapter for x in mod.blocks)):
        assert torch.equal(a.U_static,b.U_static) and torch.equal(a.V_static,b.V_static)
        gu,gv=b.channel_scales(c);assert torch.equal(gu,torch.ones_like(gu)) and torch.equal(gv,torch.ones_like(gv))
        assert torch.count_nonzero(b(x,c))==0
        for ctx in [torch.autocast('cpu',enabled=False),torch.autocast('cpu',dtype=torch.bfloat16)]:
            with ctx:
                u,v=a.factors(c);uu,vv=b.factors(c)
            assert u.dtype==uu.dtype==v.dtype==vv.dtype
            assert torch.equal(u,uu) and torch.equal(v,vv)
        with torch.no_grad():a.V_static.normal_(std=.01);b.V_static.copy_(a.V_static)
        assert torch.equal(a(x,c),b(x,c))
    assert static.statistics()['coefficient_parameters']==0


@pytest.mark.parametrize('mode',['static','modulated'])
def test_no_dense_decoder_basis_and_bf16_zero_factor_gradient(mode):
    m=adapter(mode).float();x=torch.randn(2,4,3);c=torch.randn(2,4);shapes=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(shapes.append(tuple(t.shape)),t)[1],lambda t:t):
        with torch.autocast('cpu',dtype=torch.bfloat16):u,v=m.factors(c);out=m(x,c)
        assert u.dtype==v.dtype==out.dtype==torch.bfloat16
        (out.float()-torch.randn_like(out.float())).square().sum().backward()
    assert m.V_static.grad.abs().sum()>0 and m.V_offset.grad.abs().sum()>0
    assert (3,3,2) not in shapes and (3,2,5) not in shapes
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize('decoder',[{'decoder_cp_width':8},{'decoder_tucker_width':8}])
def test_existing_decoder_trajectories_match_frozen_model(decoder):
    path=Path(__file__).parent/'runs/dit-decoder-core-20261014T071618Z/runner/dit_cifar10.py'
    if not path.exists():path=Path(__file__).parent/'channel_factor_baseline_model.py'
    spec=importlib.util.spec_from_file_location('before_channel_factors',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    args=dict(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,
              factor_offsets=True,enable_unconditional=True,**decoder)
    torch.manual_seed(124);a=CIFAR10DiT(**args);rng=torch.get_rng_state().clone()
    torch.manual_seed(124);b=old.CIFAR10DiT(**args)
    assert torch.equal(rng,torch.get_rng_state()) and a.statistics()==b.statistics()
    with torch.no_grad():
        for n,p in a.named_parameters():
            if n.endswith('adapter.V_channel') or 'modulation.1' in n or n.startswith('output.'):
                p.normal_(std=.02)
    b.load_state_dict(a.state_dict());opts=[torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0) for m in (a,b)]
    x=torch.randn(2,3,32,32);ts=torch.tensor([20,700]);labels=torch.tensor([0,10])
    for _ in range(3):
        outs=[]
        for m,opt in zip((a,b),opts):
            opt.zero_grad();out=m(x,ts,labels);out.square().mean().backward();outs.append(out)
        assert torch.equal(*outs) and all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
        for opt in opts:opt.step()
        assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))


@pytest.mark.parametrize('mode',['bad',True,None])
def test_invalid_modes_rejected(tmp_path,mode):
    p=tmp_path/'cfg.json';p.write_text(json.dumps({'variant':'dplr','decoder_channel_mode':mode}))
    with pytest.raises(ValueError,match='Channel mode'):load_config(['--override_json',str(p)])
    with pytest.raises(ValueError,match='Channel mode'):CIFAR10DiT(decoder_channel_mode=mode)


@pytest.mark.parametrize('kwargs',[{'variant':'adaln_zero'},{'variant':'dplr','decoder_cp_width':8},
                                  {'variant':'dplr','decoder_tucker_width':8},{'variant':'dplr','decoder_groups':2}])
def test_other_decoder_combinations_rejected(kwargs):
    with pytest.raises(ValueError,match='Channel mode'):CIFAR10DiT(decoder_channel_mode='modulated',**kwargs)


@pytest.mark.parametrize('mode',['static','modulated'])
def test_all_routes_wake_and_diagnostics_preserve_training(mode,tmp_path):
    results=[]
    for enabled in (False,True):
        cfg=Config(variant='dplr',decoder_channel_mode=mode,factor_offsets=True,width=24,depth=2,
                   heads=3,patch_size=8,rank=2,n_basis=4,max_steps=12,warmup_steps=2,batch_size=4,
                   synthetic=True,synthetic_samples=16,val_limit=8,device='cpu',precision='fp32',
                   cpu_threads=2,num_workers=0,log_every=12,eval_every=12,benchmark_warmup_steps=2,
                   sample_images=0,class_dropout_prob=.1,output_dir=str(tmp_path/str(enabled)),
                   adapter_lr_multiplier=4.,diagnostic_steps='0,1,3,5,10,12' if enabled else '')
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(cfg)));results.append(main(['--override_json',str(p)]))
    for state in ('raw_validation','ema_validation'):assert results[0][state]['noise_mse']==results[1][state]['noise_mse']
    w=results[1]['design_diagnostics']['adapter_first_nonzero_gradient_observed']
    assert len(w)==(8 if mode=='static' else 16) and all(v<=10 for v in w.values())
    assert not list(tmp_path.rglob('*.pt'))

"""Channel-factorized decoder correctness and default CP compatibility."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dit_cifar10 import CPBroadcastLowRank, TuckerBroadcastLowRank, CIFAR10DiT
from train_dit_cifar10 import Config, load_config, main, make_optimizer


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def adapter():
    return TuckerBroadcastLowRank(3,5,4,'dplr',rank=2,n_basis=3,tucker_width=4,
                                 initialization_seed=80,factor_offsets=True).double()


def test_tucker_outputs_and_every_gradient_match_expanded_decoder():
    m=adapter()
    with torch.no_grad():m.V_channel.normal_();m.U_offset.normal_();m.V_offset.normal_()
    x=torch.randn(2,4,3,dtype=torch.double,requires_grad=True)
    c=torch.randn(2,4,dtype=torch.double,requires_grad=True)
    a,b=m.coefficient_values(c).chunk(2,-1)
    ub=torch.einsum('ip,spk->isk',m.U_channel,m.U_core)
    vb=torch.einsum('spk,po->sko',m.V_core,m.V_channel)
    u=torch.einsum('bs,isk->bik',a,ub)+m.U_offset
    v=torch.einsum('bs,sko->bko',b,vb)+m.V_offset
    actual=m(x,c);expected=x@u@v
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
    ps=[x,c,*m.parameters()]
    aa=torch.autograd.grad(actual.square().sum(),ps,retain_graph=True)
    bb=torch.autograd.grad(expected.square().sum(),ps)
    for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    assert torch.autograd.gradcheck(m,(x,c))


def test_no_dense_decoder_bases_saved():
    m=adapter()
    with torch.no_grad():m.V_channel.normal_()
    shapes=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(shapes.append(tuple(t.shape)),t)[1],lambda t:t):
        m(torch.randn(2,4,3,dtype=torch.double),torch.randn(2,4,dtype=torch.double)).sum().backward()
    assert (3,3,2) not in shapes and (3,2,5) not in shapes
    assert not hasattr(m,'U_basis') and not hasattr(m,'U_rank') and not hasattr(m,'U')


def test_cp_initial_functions_channels_heads_and_rng_are_paired():
    args=dict(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,
              factor_offsets=True,enable_unconditional=True)
    torch.manual_seed(122);cp=CIFAR10DiT(**args,decoder_cp_width=8);rng=torch.get_rng_state().clone()
    torch.manual_seed(122);tu=CIFAR10DiT(**args,decoder_tucker_width=8)
    assert torch.equal(rng,torch.get_rng_state())
    for n,p in cp.named_parameters():
        if not n.endswith(('.U_basis','.U_rank','.V_basis','.V_rank')):
            assert torch.equal(p,dict(tu.named_parameters())[n]),n
    x=torch.randn(2,4,24);c=torch.randn(2,24)
    for a,b in zip((x.adapter for x in cp.blocks),(x.adapter for x in tu.blocks)):
        assert torch.equal(b.U_core,a.U_basis.unsqueeze(-1)*a.U_rank)
        assert torch.equal(b.V_core,a.V_basis.unsqueeze(-1)*a.V_rank)
        with torch.no_grad():a.V_channel.normal_(std=.01);b.V_channel.copy_(a.V_channel)
        ua,va=a.factors(c);ub,vb=b.factors(c)
        torch.testing.assert_close(ua,ub,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(va,vb,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(a(x,c),b(x,c),rtol=1e-5,atol=1e-5)


def test_existing_cp_matches_frozen_model_forward_gradients_and_adam():
    path=Path(__file__).parent/'runs/dit-decoder-compression-20261013T181112Z/runner/dit_cifar10.py'
    if not path.exists():path=Path(__file__).parent/'cp_decoder_baseline_model.py'
    spec=importlib.util.spec_from_file_location('cp_before_tucker',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    args=dict(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,
              factor_offsets=True,enable_unconditional=True,decoder_cp_width=8)
    torch.manual_seed(122);a=CIFAR10DiT(**args);rng=torch.get_rng_state().clone()
    torch.manual_seed(122);b=old.CIFAR10DiT(**args)
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


def test_bf16_factors_and_zero_start_gradient():
    m=adapter().float();x=torch.randn(2,4,3);c=torch.randn(2,4)
    with torch.autocast('cpu',dtype=torch.bfloat16):u,v=m.factors(c);out=m(x,c)
    assert u.dtype==v.dtype==out.dtype==torch.bfloat16 and torch.count_nonzero(out)==0
    (out.float()-torch.randn_like(out.float())).square().sum().backward()
    assert m.V_channel.grad.abs().sum()>0 and m.V_core.grad.abs().sum()==0


@pytest.mark.parametrize('width',[True,-1,1.5,None])
def test_invalid_tucker_width_rejected(tmp_path,width):
    p=tmp_path/'cfg.json';p.write_text(json.dumps({'variant':'dplr','decoder_tucker_width':width}))
    with pytest.raises(ValueError,match='Tucker decoder width'):load_config(['--override_json',str(p)])
    with pytest.raises(ValueError,match='Tucker decoder width'):CIFAR10DiT(decoder_tucker_width=width)


@pytest.mark.parametrize('kwargs',[{'variant':'adaln_zero'},{'variant':'dplr','decoder_groups':2},
                                  {'variant':'dplr','decoder_cp_width':8}])
def test_tucker_rejects_unsupported_combinations(kwargs):
    with pytest.raises(ValueError,match='Tucker decoder width'):CIFAR10DiT(decoder_tucker_width=8,**kwargs)


def test_all_routes_wake_and_diagnostics_preserve_trajectory(tmp_path):
    results=[]
    for enabled in (False,True):
        cfg=Config(variant='dplr',decoder_tucker_width=8,factor_offsets=True,width=24,depth=2,
                   heads=3,patch_size=8,rank=2,n_basis=4,max_steps=12,warmup_steps=2,batch_size=4,
                   synthetic=True,synthetic_samples=16,val_limit=8,device='cpu',precision='fp32',
                   cpu_threads=2,num_workers=0,log_every=12,eval_every=12,benchmark_warmup_steps=2,
                   sample_images=0,class_dropout_prob=.1,output_dir=str(tmp_path/str(enabled)),
                   adapter_lr_multiplier=.25,
                   diagnostic_steps='0,1,3,5,10,12' if enabled else '')
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(cfg)));results.append(main(['--override_json',str(p)]))
    for state in ('raw_validation','ema_validation'):assert results[0][state]['noise_mse']==results[1][state]['noise_mse']
    wake=results[1]['design_diagnostics']['adapter_first_nonzero_gradient_observed']
    assert len(wake)==16 and all(v<=10 for v in wake.values())
    assert not list(tmp_path.rglob('*.pt'))
    routes=results[1]['training_setup']['optimizer_groups']
    assert sum(g['parameter_count'] for g in routes if g['is_adapter'])==results[1]['adapter_parameters']
    for g in routes:assert g['initial_lr']==cfg.lr*(.25 if g['is_adapter'] else 1)
    logs=[json.loads(s) for s in (tmp_path/'True/metrics.jsonl').read_text().splitlines()]
    for e in logs:
        if e['event']=='train':
            assert e['optimizer_group_lrs']==[e['base_lr']*(.25 if g['is_adapter'] else 1) for g in routes]
    m=CIFAR10DiT(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,
                 decoder_tucker_width=8,factor_offsets=True)
    opt=make_optimizer(m,Config(variant='dplr',decoder_tucker_width=8,device='cpu'))
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    assert m.statistics()['decoder_parameters']==2*(8*(24+96+2*4*2)+2*(24+96))

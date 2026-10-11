"""Factor offsets must preserve zero start, baseline behavior and sharing semantics."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dit_cifar10 import BroadcastLowRank, CIFAR10DiT
from train_dit_cifar10 import Config, load_config, main, make_optimizer


def tiny(**kwargs):
    return CIFAR10DiT(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,**kwargs)


def test_disabled_path_matches_frozen_model_forward_gradients_and_adam():
    torch.set_num_threads(2)
    path=Path(__file__).resolve().parent/'offset_baseline_model.py'
    if not path.exists():
        path=Path('/workspace/adalora/runs/dit-longer-confirmation-20261010T154131Z/runner/dit_cifar10.py')
    spec=importlib.util.spec_from_file_location('frozen_offset_baseline',path)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    torch.manual_seed(94);a=tiny();rng=torch.get_rng_state().clone()
    torch.manual_seed(94);b=legacy.CIFAR10DiT(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4)
    assert torch.equal(rng,torch.get_rng_state())
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    # Open the gates/output/factor so parity exercises the adapter rather than only zero output.
    with torch.no_grad():
        for name,p in a.named_parameters():
            if name.endswith('adapter.V') or 'modulation.1' in name or name.startswith('output.'):
                p.normal_(std=.02)
    b.load_state_dict(a.state_dict())
    opts=[torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0) for m in (a,b)]
    x=torch.randn(2,3,32,32);t=torch.tensor([20,800]);labels=torch.tensor([1,4])
    for _ in range(3):
        outputs=[]
        for m,opt in zip((a,b),opts):
            opt.zero_grad();out=m(x,t,labels);out.square().mean().backward();outputs.append(out.detach())
        assert torch.equal(*outputs)
        assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
        for opt in opts:opt.step()
        assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))
    assert all(a.statistics()[k]==v for k,v in b.statistics().items())


def test_zero_offsets_preserve_existing_tensors_rng_and_initial_function():
    torch.manual_seed(94);a=tiny();rng=torch.get_rng_state().clone()
    torch.manual_seed(94);b=tiny(factor_offsets=True)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert b.statistics()['parameters']-a.statistics()['parameters']==2*2*(24+96)
    for block in b.blocks:
        m=block.adapter
        assert torch.count_nonzero(m.U_offset)==torch.count_nonzero(m.V_offset)==0
        assert torch.count_nonzero(m(torch.randn(2,3,24),torch.randn(2,24)))==0


def test_nonzero_offsets_match_dense_operator_and_have_correct_gradients():
    m=BroadcastLowRank(3,5,4,'dplr',rank=2,n_basis=2,factor_offsets=True).double()
    with torch.no_grad():m.V.normal_();m.U_offset.normal_();m.V_offset.normal_()
    x=torch.randn(2,3,3,dtype=torch.double,requires_grad=True)
    c=torch.randn(2,4,dtype=torch.double,requires_grad=True)
    a,b=m.coefficients(c).chunk(2,-1)
    u=(a @ m.U.view(3,2,2).permute(1,0,2).reshape(2,6)).reshape(2,3,2)+m.U_offset
    v=(b @ m.V.reshape(2,10)).reshape(2,2,5)+m.V_offset
    expected=torch.bmm(x,torch.bmm(u,v))
    actual=m(x,c)
    assert torch.allclose(actual,expected,atol=1e-12)
    params=[x,c,*m.parameters()]
    ga=torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
    gb=torch.autograd.grad(expected.square().sum(),params)
    assert all(torch.allclose(u,v,atol=1e-10,rtol=1e-10) for u,v in zip(ga,gb))


def test_offsets_preserve_bf16_factor_dtype_and_zero_start():
    torch.manual_seed(94);a=BroadcastLowRank(8,16,8,'dplr',rank=2,n_basis=4)
    torch.manual_seed(94);b=BroadcastLowRank(8,16,8,'dplr',rank=2,n_basis=4,factor_offsets=True)
    c=torch.randn(2,8)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        ua,va=a.factors(c);ub,vb=b.factors(c)
    assert ua.dtype==va.dtype==ub.dtype==vb.dtype==torch.bfloat16
    assert torch.equal(ua,ub) and torch.equal(va,vb)


def test_decoder_sharing_includes_offsets_and_optimizer_deduplicates():
    torch.set_num_threads(2)
    m=CIFAR10DiT(width=24,depth=6,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,
                 decoder_groups=2,factor_offsets=True)
    assert m.blocks[0].adapter.U_offset is m.blocks[2].adapter.U_offset
    assert m.blocks[0].adapter.V_offset is m.blocks[2].adapter.V_offset
    assert m.blocks[0].adapter.U_offset is not m.blocks[3].adapter.U_offset
    opt=make_optimizer(m,Config(variant='dplr',factor_offsets=True,device='cpu'))
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    assert m.statistics()['decoder_parameters']==2*(24*8+8*96+24*2+2*96)


@pytest.mark.parametrize('setting',[1,'yes',None])
def test_invalid_offset_setting_is_rejected(tmp_path,setting):
    p=tmp_path/'cfg.json';p.write_text(json.dumps({'variant':'dplr','factor_offsets':setting}))
    with pytest.raises(ValueError,match='boolean'):load_config(['--override_json',str(p)])
    with pytest.raises(ValueError,match='boolean'):tiny(factor_offsets=setting)


def test_offsets_reject_other_variants(tmp_path):
    p=tmp_path/'cfg.json';p.write_text(json.dumps({'variant':'adaln_zero','factor_offsets':True}))
    with pytest.raises(ValueError,match='DPLR'):load_config(['--override_json',str(p)])


def test_all_offsets_wake_and_diagnostics_preserve_training(tmp_path):
    results=[]
    for enabled in (False,True):
        cfg=Config(variant='dplr',factor_offsets=True,width=24,depth=2,heads=3,patch_size=8,rank=2,n_basis=4,
                   max_steps=12,warmup_steps=2,batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,
                   device='cpu',precision='fp32',cpu_threads=2,num_workers=0,log_every=12,eval_every=12,
                   benchmark_warmup_steps=2,sample_images=0,class_dropout_prob=.1,
                   output_dir=str(tmp_path/str(enabled)),diagnostic_steps='0,1,3,5,10,12' if enabled else '')
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(cfg)));results.append(main(['--override_json',str(p)]))
    for state in ('raw_validation','ema_validation'):
        assert results[0][state]['noise_mse']==results[1][state]['noise_mse']
    diag=results[1]['design_diagnostics']
    wake=diag['adapter_first_nonzero_gradient_observed']
    assert len(wake)==12 and all(v<=10 for v in wake.values())
    assert len([name for name in wake if name.endswith('_offset')])==4
    assert all(len(v['nonzero_gradient_tensors'])==4 for v in diag['diagnostics'][-1]['decoder_gradient_groups_after_clip'].values())
    assert not list(tmp_path.rglob('*.pt'))

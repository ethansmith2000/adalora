"""Independent dense oracles, gradients, precision, initialization and integration."""
from dataclasses import asdict
import json

import pytest
import torch
from torch.nn import functional as F

from dit_cifar10 import CIFAR10DiT
from structured_operators import ConditionalStructured, hadamard
from train_dit_cifar10 import Config, load_config, main, make_model
from benchmark_dit_design import ProjectionProbe, branch_snapshot, digest_parameters
from structured_baseline_model import CIFAR10DiT as OldDiT


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def sylvester(d, dtype=torch.double):
    h=torch.ones(1,1,dtype=dtype)
    while h.shape[0]<d:
        h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return h/d**.5


def test_hadamard_matches_independent_matrix_and_preserves_norm():
    x=torch.randn(2,3,16,dtype=torch.double,requires_grad=True)
    expected=x@sylvester(16)
    torch.testing.assert_close(hadamard(x),expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(hadamard(hadamard(x)),x,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(hadamard(x).square().sum(),x.square().sum(),rtol=1e-12,atol=1e-12)
    assert torch.autograd.gradcheck(hadamard,(x,))
    bf=x.detach().bfloat16();assert hadamard(bf).dtype==torch.bfloat16
    torch.testing.assert_close(hadamard(bf).float(),bf.float()@sylvester(16,torch.float32),rtol=.02,atol=.02)
    with pytest.raises(ValueError):hadamard(torch.zeros(2,6))


@pytest.mark.parametrize('kind',['diagonal','block','hadamard'])
def test_every_gradient_matches_independent_dense_operator(kind):
    torch.manual_seed(77)
    m=ConditionalStructured(8,5,3,kind,2,.7,82).double()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if 'coefficients' not in n:p.normal_(std=.15)
    x=torch.randn(2,3,8,dtype=torch.double,requires_grad=True)
    c=torch.randn(2,5,dtype=torch.double,requires_grad=True)
    v=m.values(c)
    if kind=='diagonal':matrix=torch.diag_embed(v)
    elif kind=='block':matrix=torch.stack([torch.block_diag(*b.unbind()) for b in v.reshape(2,4,2,2)])
    else:
        left,middle,out=v;h=sylvester(8);permutation=torch.eye(8,dtype=torch.double)[:,m.permutation]
        matrix=torch.diag_embed(left)@h@permutation@torch.diag_embed(middle)@h@torch.diag_embed(out)
    actual=m(x,c);expected=x@matrix*m.gain
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
    params=[x,c,*m.parameters()];up=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*up).sum(),params,retain_graph=True)
    gb=torch.autograd.grad((expected*up).sum(),params)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    assert torch.autograd.gradcheck(m,(x,c))
    assert torch.linalg.matrix_rank(matrix).tolist()==[8,8]


@pytest.mark.parametrize('kind',['lowrank','diagonal','block','hadamard'])
def test_zero_delta_same_base_and_one_projection(kind):
    args=dict(width=16,depth=2,heads=2,patch_size=8,enable_unconditional=True)
    torch.manual_seed(128);base=CIFAR10DiT(**args)
    torch.manual_seed(128);m=CIFAR10DiT(**args,variant='dplr',rank=2,n_basis=4,factor_offsets=True,
                                    structured_operator=kind,structured_block_size=4)
    assert digest_parameters(base)==digest_parameters(m)
    with torch.no_grad():
        for a,b in zip(base.blocks,m.blocks):
            a.modulation[-1].weight.normal_(std=.02);b.modulation.load_state_dict(a.modulation.state_dict())
        base.output.weight.normal_(std=.02);m.output.load_state_dict(base.output.state_dict())
    x=torch.randn(2,3,32,32);ts=torch.tensor([20,700]);labels=torch.tensor([0,10])
    for enabled in (False,True):
        count=[];handle=m.blocks[0].mlp_in.register_forward_hook(lambda *args:count.append(1))
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=enabled):
            actual=m(x,ts,labels);expected=base(x,ts,labels)
        handle.remove();assert len(count)==1 and torch.equal(actual,expected)
        records=branch_snapshot(m,x,ts,labels,torch.device('cpu'),'bf16' if enabled else 'fp32')
        assert all(r['delta_rms']==r['square_feature_delta_rms']==0 for r in records)
    block=m.blocks[0]
    with torch.no_grad():
        for n,p in block.adapter.named_parameters():
            if 'coefficients' not in n:p.normal_(std=.01)
    h=torch.randn(2,4,16);c=torch.randn(2,16);shift,scale=block.modulation(c).chunk(6,-1)[3:5]
    norm=block.norm_mlp(h)
    expected=F.linear(norm*(1+scale[:,None])+shift[:,None]+block.adapter(norm,c),block.mlp_in.weight,block.mlp_in.bias)
    torch.testing.assert_close(ProjectionProbe(block)(h,c),expected)


@pytest.mark.parametrize('decoder',[{}, {'decoder_cp_width':8},{'decoder_channel_mode':'modulated'}])
def test_legacy_state_rng_and_adam_trajectory_are_unchanged(decoder):
    args=dict(width=16,depth=2,heads=2,patch_size=8,variant='dplr',rank=2,n_basis=4,
              factor_offsets=True,enable_unconditional=True,**decoder)
    torch.manual_seed(128);a=CIFAR10DiT(**args);rng=torch.get_rng_state().clone()
    torch.manual_seed(128);b=OldDiT(**args)
    assert torch.equal(rng,torch.get_rng_state()) and a.statistics()==b.statistics()
    with torch.no_grad():
        for n,p in a.named_parameters():
            if '.adapter.' in n or 'modulation.1' in n or n.startswith('output.'):p.normal_(std=.02)
    b.load_state_dict(a.state_dict());opts=[torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0) for m in (a,b)]
    x=torch.randn(2,3,32,32);ts=torch.tensor([20,700]);labels=torch.tensor([0,10])
    for _ in range(3):
        outs=[]
        for m,opt in zip((a,b),opts):
            opt.zero_grad();out=m(x,ts,labels);out.square().mean().backward();outs.append(out)
        assert torch.equal(*outs) and all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
        for opt in opts:opt.step()
        assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))


@pytest.mark.parametrize('kind',['lowrank','diagonal','block','hadamard'])
def test_all_parameters_wake_and_diagnostics_preserve_bf16_training(kind,tmp_path):
    results=[]
    for enabled in (False,True):
        cfg=Config(variant='dplr',structured_operator=kind,structured_block_size=4,factor_offsets=True,
                   width=16,depth=2,heads=2,patch_size=8,rank=2,n_basis=4,
                   max_steps=12,warmup_steps=2,batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,
                   device='cpu',precision='bf16',cpu_threads=2,num_workers=0,log_every=12,eval_every=12,
                   benchmark_warmup_steps=2,sample_images=0,class_dropout_prob=.1,
                   output_dir=str(tmp_path/str(enabled)),adapter_lr_multiplier=2.,
                   diagnostic_steps='0,1,3,5,10,12' if enabled else '')
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(cfg)));results.append(main(['--override_json',str(p)]))
    for state in ('raw_validation','ema_validation'):assert results[0][state]['noise_mse']==results[1][state]['noise_mse']
    w=results[1]['design_diagnostics']['adapter_first_nonzero_gradient_observed']
    assert len(w)==(12 if kind=='lowrank' else 16 if kind=='hadamard' else 8) and all(v<=10 for v in w.values())
    assert results[1]['design_diagnostics']['fixed_function_probe_scope'].startswith('projected conditional')
    assert not list(tmp_path.rglob('*.pt'))


@pytest.mark.parametrize('kwargs',[{'structured_operator':'bad'},{'structured_operator':'hadamard','width':24,'heads':3},
    {'structured_operator':'block','structured_block_size':7},{'structured_operator':'block','decoder_cp_width':8},
    {'structured_operator':'lowrank','projector_groups':1},{'structured_operator':'hadamard','coefficient_hidden':8},
    {'structured_operator':'diagonal','adapter_input':'modulated'},{'structured_operator':'diagonal','factor_offsets':False}])
def test_invalid_combinations_rejected(tmp_path,kwargs):
    cfg=dict(variant='dplr',width=16,depth=2,heads=2,factor_offsets=True,**{})
    cfg.update(kwargs);p=tmp_path/'cfg.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):load_config(['--override_json',str(p)])
    with pytest.raises(ValueError):CIFAR10DiT(**cfg)

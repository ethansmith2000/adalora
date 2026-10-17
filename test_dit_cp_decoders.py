"""CP decoder correctness, initialization and trainer integration."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dit_cifar10 import BroadcastLowRank, CPBroadcastLowRank, CIFAR10DiT
from train_dit_cifar10 import Config, load_config, main, make_optimizer


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def adapter():
    return CPBroadcastLowRank(3, 5, 4, 'dplr', rank=2, n_basis=3,
                              cp_width=4, initialization_seed=80, factor_offsets=True).double()


def test_cp_matches_expanded_tensor_factors_outputs_and_all_gradients():
    m = adapter()
    with torch.no_grad():
        m.V_channel.normal_(); m.U_offset.normal_(); m.V_offset.normal_()
    x = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
    c = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    ub = torch.einsum('ip,sp,pk->isk', m.U_channel, m.U_basis, m.U_rank)
    vb = torch.einsum('sp,pk,po->sko', m.V_basis, m.V_rank, m.V_channel)
    a, b = m.coefficient_values(c).chunk(2, -1)
    u = torch.einsum('bs,isk->bik', a, ub) + m.U_offset
    v = torch.einsum('bs,sko->bko', b, vb) + m.V_offset
    actual = m(x, c); expected = x @ u @ v
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    params = [x, c, *m.parameters()]
    ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), params)
    for a, b in zip(ga, gb): torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_cp_gradcheck_and_no_dense_decoder_intermediates():
    m = adapter()
    with torch.no_grad(): m.V_channel.normal_()
    x = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
    c = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(m, (x, c))
    shapes = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (shapes.append(tuple(t.shape)), t)[1], lambda t: t):
        m(x, c).square().sum().backward()
    assert (3, 3, 2) not in shapes and (3, 2, 5) not in shapes
    assert not hasattr(m, 'U') and not hasattr(m, 'V')


def test_cp_preserves_zero_start_coefficient_heads_base_and_global_rng():
    def model(cp):
        return CIFAR10DiT(width=24, depth=2, heads=3, patch_size=8, variant='dplr',
                         rank=2, n_basis=4, factor_offsets=True, enable_unconditional=True,
                         decoder_cp_width=cp)
    torch.manual_seed(120); dense = model(0); rng = torch.get_rng_state().clone()
    torch.manual_seed(120); cp = model(8)
    assert torch.equal(rng, torch.get_rng_state())
    for name, p in dense.named_parameters():
        if not name.endswith(('.adapter.U', '.adapter.V')):
            assert torch.equal(p, dict(cp.named_parameters())[name]), name
    x = torch.randn(2, 3, 32, 32); ts = torch.tensor([20, 700]); labels = torch.tensor([0, 10])
    assert torch.count_nonzero(cp(x, ts, labels)) == 0
    for block in cp.blocks:
        m = block.adapter
        assert torch.count_nonzero(m(torch.randn(2, 5, 24), torch.randn(2, 24))) == 0
        assert torch.count_nonzero(m.V_channel) == 0
        assert all(torch.count_nonzero(p) for n,p in m.decoder_named_parameters()
                   if n not in ('V_channel', 'U_offset', 'V_offset'))


def test_dense_default_matches_previous_frozen_training_trajectory():
    path = Path(__file__).parent/'runs/dit-external-lr-20261013T070213Z/runner/dit_cifar10.py'
    if not path.exists(): path = Path(__file__).parent/'cp_baseline_model.py'
    spec = importlib.util.spec_from_file_location('cp_baseline', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    args = dict(width=24, depth=2, heads=3, patch_size=8, variant='dplr', rank=2,
                n_basis=4, factor_offsets=True, enable_unconditional=True)
    torch.manual_seed(120); a = CIFAR10DiT(**args); rng = torch.get_rng_state().clone()
    torch.manual_seed(120); b = old.CIFAR10DiT(**args)
    assert torch.equal(rng, torch.get_rng_state()) and a.statistics() == b.statistics()
    with torch.no_grad():
        for n,p in a.named_parameters():
            if n.endswith('adapter.V') or 'modulation.1' in n or n.startswith('output.'):
                p.normal_(std=.02)
    b.load_state_dict(a.state_dict())
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0) for m in (a,b)]
    x = torch.randn(2,3,32,32); ts = torch.tensor([20,700]); labels = torch.tensor([0,10])
    for _ in range(3):
        outs = []
        for m,opt in zip((a,b),opts):
            opt.zero_grad(); out=m(x,ts,labels); out.square().mean().backward(); outs.append(out)
        assert torch.equal(*outs)
        assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
        for opt in opts: opt.step()
        assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))


def test_cp_bf16_factors_and_gradients_keep_expected_dtype():
    m = adapter().float(); x=torch.randn(2,4,3); c=torch.randn(2,4)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        u,v=m.factors(c); out=m(x,c)
    assert u.dtype == v.dtype == out.dtype == torch.bfloat16
    (out.float() - torch.randn_like(out.float())).square().sum().backward()
    assert m.V_channel.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize('width',[True,-1,1.5,None])
def test_invalid_cp_width_is_rejected(tmp_path,width):
    p=tmp_path/'cfg.json'; p.write_text(json.dumps({'variant':'dplr','decoder_cp_width':width}))
    with pytest.raises(ValueError,match='CP decoder width'): load_config(['--override_json',str(p)])
    with pytest.raises(ValueError,match='CP decoder width'): CIFAR10DiT(decoder_cp_width=width)


@pytest.mark.parametrize('kwargs',[{'variant':'adaln_zero'}, {'variant':'dplr','decoder_groups':2}])
def test_cp_rejects_unsupported_sharing_and_other_variants(kwargs):
    with pytest.raises(ValueError,match='CP decoder width'): CIFAR10DiT(decoder_cp_width=8,**kwargs)


def test_all_cp_routes_wake_without_diagnostic_training_changes(tmp_path):
    results=[]
    for enabled in (False,True):
        cfg=Config(variant='dplr',decoder_cp_width=8,factor_offsets=True,width=24,depth=2,heads=3,
                   patch_size=8,rank=2,n_basis=4,max_steps=12,warmup_steps=2,batch_size=4,
                   synthetic=True,synthetic_samples=16,val_limit=8,device='cpu',precision='fp32',
                   cpu_threads=2,num_workers=0,log_every=12,eval_every=12,benchmark_warmup_steps=2,
                   sample_images=0,class_dropout_prob=.1,output_dir=str(tmp_path/str(enabled)),
                   diagnostic_steps='0,1,3,5,10,12' if enabled else '')
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(cfg)));results.append(main(['--override_json',str(p)]))
    for state in ('raw_validation','ema_validation'):
        assert results[0][state]['noise_mse']==results[1][state]['noise_mse']
    d=results[1]['design_diagnostics']; wake=d['adapter_first_nonzero_gradient_observed']
    assert len(wake)==20 and all(v<=10 for v in wake.values())
    assert all(v['parameter_tensors']==8 for v in d['diagnostics'][-1]['decoder_gradient_groups_after_clip'].values())
    assert not list(tmp_path.rglob('*.pt'))
    m=CIFAR10DiT(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,
                 decoder_cp_width=8,factor_offsets=True)
    opt=make_optimizer(m,Config(variant='dplr',decoder_cp_width=8,device='cpu'))
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    assert m.statistics()['decoder_parameters']==2*(8*(24+96+2*4+2*2)+2*(24+96))

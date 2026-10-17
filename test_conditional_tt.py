"""Independent TT matrix oracle, all gradients, no-op and training integration."""
from dataclasses import asdict
import json
import pytest
import torch
from torch.func import functional_call
from conditional_tt import ConditionalTT, tt_apply, tt_spec
from dit_cifar10 import CIFAR10DiT
from benchmark_dit_design import digest_parameters, ProjectionProbe
from train_dit_cifar10 import Config, main, load_config


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def dense_oracle(cores):
    # Independent index enumeration: each matrix entry is a product of bond
    # matrices. Only used in tests, never in the training implementation.
    import itertools
    modes=[c.shape[2] for c in cores]
    coordinates=list(itertools.product(*(range(m) for m in modes)))
    rows=[]
    for input_index in coordinates:
        entries=[]
        for output_index in coordinates:
            value=cores[0][:,0,input_index[0],output_index[0],:]
            for core,i,j in zip(cores[1:],input_index[1:],output_index[1:]):
                value=torch.bmm(value[:,None],core[:,:,i,j,:]).squeeze(1)
            entries.append(value[:,0])
        rows.append(torch.stack(entries,-1))
    return torch.stack(rows,-2)


@pytest.mark.parametrize('modes,rank', [((2,3),1),((2,2,2),2),((2,2,2),3)])
def test_dense_oracle_all_parameter_gradients_and_gradcheck(modes,rank):
    torch.manual_seed(130);d=__import__('math').prod(modes)
    model=ConditionalTT(d,3,2,modes,rank,.7).double()
    with torch.no_grad():
        for p in model.parameters():p.normal_(std=.3)
    x=torch.randn(2,3,d,dtype=torch.double,requires_grad=True)
    c=torch.randn(2,3,dtype=torch.double,requires_grad=True)
    actual=model(x,c);expected=x@dense_oracle(model.cores(c))*model.gain
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
    inputs=(x,c,*model.parameters());up=torch.randn_like(actual)
    left=torch.autograd.grad((actual*up).sum(),inputs,retain_graph=True)
    right=torch.autograd.grad((expected*up).sum(),inputs)
    for a,b in zip(left,right):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)
    names=list(dict(model.named_parameters()))
    def f(x,c,*params):return functional_call(model,dict(zip(names,params)),(x,c))
    assert torch.autograd.gradcheck(f,inputs,fast_mode=True)
    # A bond-one Kronecker operator can already have full ordinary rank.
    assert torch.linalg.matrix_rank(dense_oracle(model.cores(c))).tolist()==[d,d]


def test_zero_initialization_base_parity_single_projection_and_bf16():
    args=dict(width=16,depth=2,heads=2,patch_size=8,enable_unconditional=True)
    torch.manual_seed(130);base=CIFAR10DiT(**args)
    torch.manual_seed(130);model=CIFAR10DiT(**args,variant='dplr',n_basis=4,factor_offsets=True,
                                          structured_operator='tt',tt_modes='2,2,2,2',tt_bond_rank=3)
    assert digest_parameters(base)==digest_parameters(model)
    with torch.no_grad():
        for a,b in zip(base.blocks,model.blocks):
            a.modulation[-1].weight.normal_(std=.02);b.modulation.load_state_dict(a.modulation.state_dict())
        base.output.weight.normal_(std=.02);model.output.load_state_dict(base.output.state_dict())
    x=torch.randn(2,3,32,32);t=torch.tensor([20,700]);labels=torch.tensor([0,10])
    for bf in (False,True):
        calls=[];handle=model.blocks[0].mlp_in.register_forward_hook(lambda *a:calls.append(1))
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf):
            assert torch.equal(model(x,t,labels),base(x,t,labels))
        handle.remove();assert len(calls)==1
    a=model.blocks[0].adapter
    with torch.no_grad():
        for p in a.parameters():p.normal_(std=.15)
    h=torch.randn(2,3,16);c=torch.randn(2,16)
    with torch.autocast('cpu',dtype=torch.bfloat16):out=a(h,c)
    expected=h@dense_oracle(a.cores(c))*a.gain
    assert out.dtype==torch.bfloat16
    torch.testing.assert_close(out.float(),expected,rtol=.06,atol=.004)
    assert not any(p.shape[-2:]==(16,16) for p in a.parameters())


def test_all_parameters_wake_and_diagnostics_preserve_training(tmp_path):
    results=[]
    for enabled in (False,True):
        cfg=Config(variant='dplr',structured_operator='tt',tt_modes='2,2,2,2',tt_bond_rank=3,factor_offsets=True,
                   width=16,depth=2,heads=2,patch_size=8,n_basis=4,max_steps=12,warmup_steps=2,
                   batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,device='cpu',precision='bf16',
                   cpu_threads=2,num_workers=0,log_every=12,eval_every=12,benchmark_warmup_steps=2,
                   sample_images=0,class_dropout_prob=.1,output_dir=str(tmp_path/str(enabled)),
                   adapter_lr_multiplier=2.,diagnostic_steps='0,1,3,5,10,12' if enabled else '')
        path=tmp_path/'cfg.json';path.write_text(json.dumps(asdict(cfg)));results.append(main(['--override_json',str(path)]))
    for state in ('raw_validation','ema_validation'):
        assert results[0][state]['noise_mse']==results[1][state]['noise_mse']
    wake=results[1]['design_diagnostics']['adapter_first_nonzero_gradient_observed']
    assert len(wake)==20 and all(v<=10 for v in wake.values())
    assert not list(tmp_path.rglob('*.pt'))


@pytest.mark.parametrize('kwargs',[dict(tt_modes='4,4'),dict(tt_modes='1,16'),dict(tt_modes='bad'),
                                  dict(tt_bond_rank=0),dict(tt_bond_rank=1.5)])
def test_invalid_tt_specs(tmp_path,kwargs):
    values=dict(variant='dplr',structured_operator='tt',width=16,depth=2,heads=2,factor_offsets=True,tt_modes='2,2,2,2')
    values.update(kwargs)
    # 4,4 is valid for width16: use a mismatched mode product instead.
    if kwargs.get('tt_modes')=='4,4':values['tt_modes']='4,8'
    p=tmp_path/'cfg.json';p.write_text(json.dumps(values))
    with pytest.raises(ValueError):load_config(['--override_json',str(p)])
    with pytest.raises(ValueError):CIFAR10DiT(**values)

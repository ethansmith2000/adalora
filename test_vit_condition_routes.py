"""Forward invariance, independent VJPs, and diagnostic non-interference."""
from dataclasses import replace
import json

import pytest
import torch

from conditioning import ConditionedLinear, condition_input
from train_cifar10 import Config, SyntheticImages, build_model, load_config
from vit_quality_diagnostics import TokenDiagnostics, decompose_projection_gradient, gradient_alignment, routing_probe


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def learned_layer(mode='none', alpha=1.):
    torch.manual_seed(5)
    layer = ConditionedLinear(torch.nn.Linear(4, 7), 'dplr', rank=2, n_basis=3,
                              factor_offsets=True, condition_normalization=mode, condition_gradient_scale=alpha)
    with torch.no_grad():
        for p in layer.parameters(): p.normal_(std=.2)
    return layer.double()


@pytest.mark.parametrize('mode', ['none', 'rms'])
@pytest.mark.parametrize('alpha', [0., .5, 1.])
def test_split_vjp_reconstructs_independent_shared_input_derivative(mode, alpha):
    layer = learned_layer(mode, alpha); h = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(2, 3, 7, dtype=torch.float64)
    y = layer(h, h); actual = torch.autograd.grad(y, h, upstream)[0]
    parts = decompose_projection_gradient(layer, h, h, upstream)
    torch.testing.assert_close(actual, parts['value']+parts['affine']+parts['lowrank'], rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(y, parts['output'], rtol=0, atol=0)
    assert all(p.grad is None for p in layer.parameters())


def test_gradient_attenuation_preserves_forward_and_projector_parameter_gradients():
    layer = learned_layer(); h = torch.randn(2, 3, 4, dtype=torch.float64)
    values = []
    for alpha in [1., .5, 0.]:
        layer.condition_gradient_scale = alpha
        x = h.clone().requires_grad_(); y = layer(x, x)
        params = tuple(layer.parameters()); grads = torch.autograd.grad(y.square().sum(), (x, *params))
        values.append((y.detach(), grads))
    for y, grads in values[1:]:
        torch.testing.assert_close(y, values[0][0], rtol=0, atol=0)
        for a, b in zip(grads[1:], values[0][1][1:]): torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(values[1][1][0], (values[0][1][0]+values[2][1][0])*.5)


def test_rms_input_is_parameter_free_direction_preserving_and_finite_at_zero():
    c = torch.randn(2, 3, 4, requires_grad=True)
    cc = condition_input(c, 'rms'); expected = c/(c.square().mean(-1, keepdim=True)+1e-6).sqrt()
    torch.testing.assert_close(cc, expected)
    torch.testing.assert_close(cc.square().mean(-1), torch.ones(2, 3), atol=2e-5, rtol=2e-5)
    zero = torch.zeros(2, 3, 4, requires_grad=True); condition_input(zero, 'rms').sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_default_is_exactly_identical_to_prior_frozen_vit():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parent/'runs/vit-matched-lr-20261012T164445Z/runner/vit.py'
    spec=importlib.util.spec_from_file_location('prior_vit_routes_test',path); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    c=Config(width=24,depth=2,heads=3,variant='dplr',rank=2,n_basis=4,factor_offsets=True,mlp_init='xavier_gelu')
    torch.manual_seed(9); actual=build_model(c); rng=torch.get_rng_state()
    torch.manual_seed(9); expected=old.ViT(img_size=32,patch_size=4,num_classes=10,embed_dim=24,depth=2,heads=3,qk_norm=False,
                                         conditioning='dplr',condition_rank=2,condition_n_basis=4,condition_factor_offsets=True,mlp_init='xavier_gelu')
    assert torch.equal(rng,torch.get_rng_state())
    assert actual.state_dict().keys()==expected.state_dict().keys()
    for k,v in actual.state_dict().items():torch.testing.assert_close(v,expected.state_dict()[k],rtol=0,atol=0)
    h=torch.randn(2,5,24)
    for a,b in zip(actual.blocks,expected.blocks):torch.testing.assert_close(a(h),b(h),rtol=0,atol=0)


@pytest.mark.parametrize('variant', ['plain', 'dplr'])
def test_task_loss_probe_is_read_only_reconstructs_routes_and_handles_zero_classifier(variant):
    cfg=Config(variant=variant,width=24,depth=2,heads=3,rank=2,n_basis=4,factor_offsets=variant=='dplr',
               routing_diagnostics=True,diagnostic_steps='0,1',condition_gradient_scale=.5 if variant=='dplr' else 1.)
    model=build_model(cfg); images=torch.randn(4,3,32,32); labels=torch.arange(4)
    initial=routing_probe(model,images,labels,.1)
    assert all(b['total_gradient_rms']==0 and b['value_condition_alignment']['cosine'] is None for b in initial['blocks'])
    with torch.no_grad():
        model.head.weight.normal_(std=.1)
        if variant=='dplr':
            for block in model.blocks:
                block.mlp.fc1.transform.lowrank.V.normal_(std=.05)
                block.mlp.fc1.transform.to_scale.weight.normal_(std=.05)
    # Existing optimizer gradients, parameter/buffer values, RNG, mode and hooks stay intact.
    for p in model.parameters():p.grad=torch.randn_like(p)
    state={k:v.clone() for k,v in model.state_dict().items()};grads=[p.grad.clone() for p in model.parameters()];rng=torch.get_rng_state()
    result=routing_probe(model,images,labels,.1)
    assert model.training and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),grads))
    assert all(not b.mlp.fc1._forward_hooks for b in model.blocks)
    assert all(b['reconstruction_relative_error']<1e-4 and b['total_gradient_rms']>0 for b in result['blocks'])
    if variant=='dplr':assert any(b['condition_gradient_rms']>0 for b in result['blocks'])


def test_enabled_collector_preserves_baseline_rng_and_events_without_duplicate_live_branches():
    cfg=Config(width=24,depth=2,heads=3,variant='dplr',rank=2,n_basis=4,diagnostic_steps='0,1',routing_diagnostics=True)
    model=build_model(cfg); dataset=SyntheticImages(8,10)
    probe=TokenDiagnostics(model,cfg,dataset,torch.device('cpu'));rng=torch.get_rng_state()
    event=probe.collect(model,0)
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    assert len(event['live_branches'])==len(event['gradient_routing']['blocks'])==2
    assert len(model.blocks[0].mlp.fc1._forward_hooks)==0


@pytest.mark.parametrize('override',[{'condition_normalization':'bad'},{'condition_gradient_scale':-1.},
    {'condition_gradient_scale':float('nan')},{'condition_gradient_scale':.5,'variant':'plain'},
    {'condition_gradient_scale':.5,'variant':'dplr','detach_condition':True},
    {'routing_diagnostics':True},{'routing_diagnostics':True,'diagnostic_steps':'0,1','placement':'qkv'}])
def test_invalid_or_ambiguous_policy_rejected(tmp_path,override):
    p=tmp_path/'config.json';p.write_text(json.dumps(override))
    with pytest.raises(ValueError):load_config(['--override_json',str(p)])


def test_alignment_reports_undefined_zero_routes_and_opposition():
    x=torch.ones(2,3,4)
    assert gradient_alignment(x,torch.zeros_like(x))['cosine'] is None
    a=gradient_alignment(x,-x)
    assert a['negative_token_fraction']==1. and a['cosine']==pytest.approx(-1.)


def test_routing_probe_does_not_change_short_training_trajectory(tmp_path):
    from train_cifar10 import main
    cfg=Config(variant='dplr',width=12,depth=1,heads=3,rank=2,n_basis=3,
               synthetic=True,synthetic_samples=12,val_limit=4,batch_size=4,
               device='cpu',precision='fp32',num_workers=0,cpu_threads=2,
               epochs=3,max_steps=6,lr_schedule_steps=6,diagnostic_steps='0,1,3,6',log_every=1)
    traces=[]
    for enabled in [False,True]:
        out=tmp_path/str(enabled);c=replace(cfg,routing_diagnostics=enabled,output_dir=str(out))
        path=tmp_path/f'{enabled}.json';path.write_text(json.dumps(__import__('dataclasses').asdict(c)))
        main(['--override_json',str(path)])
        events=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
        traces.append([{k:e[k] for k in ['step','loss','accuracy','optimizer_lrs','grad_norm_before_clip']}
                       for e in events if e['event']=='train'])
    assert traces[0]==traces[1]

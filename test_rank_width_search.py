"""Research checks for self-conditioning gains, diagnostic preservation and search controls."""

from dataclasses import asdict, replace
import json

import pytest
import torch

from condlr import BasisMixLowRank, DPLRCond
from rank_width_search import DIFFUSION, VIT, SEEDS, make_config
from train_cifar10 import Config, build_model, main
from vit_quality_diagnostics import TokenDiagnostics, parameter_digest


@pytest.fixture(scope='module',autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_grid_and_budget_contour():
    assert len(DIFFUSION)==22 and len(VIT)==10 and SEEDS==(58,59)
    for r in (2,4,8,16):
        for s in (6,12,24,48):
            cfg=make_config('diffusion',f'r{r}_s{s}',58,'/tmp/unused')
            assert (cfg.rank,cfg.n_basis)==(r,s)
            assert cfg.max_steps==5000 and cfg.lr_schedule_steps==30000 and cfg.warmup_steps==50
            assert cfg.train_limit is cfg.val_limit is None and cfg.fid_images==5000
    assert {r*s for r,s in ((2,48),(4,24),(8,12),(16,6))}=={96}
    a=asdict(make_config('vit','r8_s12',58,'/tmp/unused'))
    b=asdict(make_config('vit','r8_s12_detached',58,'/tmp/unused'))
    assert {key for key in a if a[key]!=b[key]}=={'detach_condition'}


@pytest.mark.parametrize('kind',['basis','dplr'])
def test_gain_shared_token_gradient_oracle(kind):
    torch.manual_seed(8)
    layer=(BasisMixLowRank(5,7,5,rank=3,n_basis=4,gain=.37) if kind=='basis'
           else DPLRCond(5,7,5,rank=3,n_basis=4,shift=True,gain=.37)).double()
    lr=layer if kind=='basis' else layer.lowrank
    with torch.no_grad():
        lr.V.normal_(std=.1)
        if kind=='dplr':
            layer.to_scale.weight.normal_(std=.1);layer.to_shift.weight.normal_(std=.1)
    x=torch.randn(2,6,5,dtype=torch.float64,requires_grad=True)
    actual=layer(x,x)
    a,b=lr.to_coef(x).chunk(2,-1)
    U=torch.einsum('btk,ikr->btir',a,lr.U.view(5,4,3))
    V=torch.einsum('btk,kro->btro',b,lr.V.view(4,3,7))
    delta=torch.einsum('bti,btir,btro->bto',x,U,V)*.37
    base=lr.base(x if kind=='basis' else x*(1+layer.to_scale(x))+layer.to_shift(x))
    expected=base+delta
    inputs=[x,*layer.parameters()]
    gradients=torch.autograd.grad(actual.square().mean(),inputs)
    reference=torch.autograd.grad(expected.square().mean(),inputs)
    torch.testing.assert_close(actual,expected)
    for value,ref in zip(gradients,reference):torch.testing.assert_close(value,ref)


@pytest.mark.parametrize('variant',['plain','diagonal','static_lora_diagonal','dplr'])
def test_diagnostics_preserve_state_modes_rng_gradients(variant):
    cfg=Config(variant=variant,width=24,depth=2,heads=3,rank=2,n_basis=4,diagnostic_steps='0,3',precision='fp32')
    torch.manual_seed(9);model=build_model(cfg)
    with torch.no_grad():model.head.weight.normal_(std=.1)
    images=torch.randn(3,3,32,32);labels=torch.tensor([1,2,3])
    model(images).square().mean().backward()
    state={key:value.clone() for key,value in model.state_dict().items()}
    gradients={key:value.grad.clone() if value.grad is not None else None for key,value in model.named_parameters()}
    rng=torch.random.get_rng_state().clone();precision=torch.get_float32_matmul_precision()
    diagnostic=TokenDiagnostics(model,cfg,list(zip(images,labels)),torch.device('cpu'))
    event=diagnostic.collect(model,0)
    assert model.training and torch.equal(rng,torch.random.get_rng_state())
    assert precision==torch.get_float32_matmul_precision() and len(event['live_branches'])==2
    for key,value in model.state_dict().items():assert torch.equal(value,state[key])
    for key,value in model.named_parameters():
        if gradients[key] is None:assert value.grad is None
        else:assert torch.equal(value.grad,gradients[key])


def test_self_condition_detach_forward_same_gradient_different():
    torch.manual_seed(7)
    layer=BasisMixLowRank(5,7,5,rank=2,n_basis=3,gain=.5).double()
    with torch.no_grad():layer.V.normal_(std=.2)
    x=torch.randn(2,4,5,dtype=torch.float64,requires_grad=True)
    actual=layer(x,x);detached=layer(x,x.detach())
    torch.testing.assert_close(actual,detached)
    grad=torch.autograd.grad(actual.square().mean(),x)[0]
    detached_grad=torch.autograd.grad(detached.square().mean(),x)[0]
    assert (grad-detached_grad).norm()>1e-5


def test_trainer_diagnostics_preserve_training_prefix(tmp_path):
    outputs=[]
    for enabled in (False,True):
        folder=tmp_path/str(enabled)
        args=['--synthetic','--device','cpu','--precision','fp32','--width','24','--depth','1','--heads','3',
              '--variant','dplr','--rank','2','--n-basis','4','--adapter-gain','.5','--epochs','1',
              '--max-steps','8','--batch-size','4','--synthetic-samples','32','--num-workers','0',
              '--benchmark-warmup-steps','0','--log-every','1','--seed','64','--output-dir',str(folder)]
        if enabled:args+=['--diagnostic-steps','0,1,3,8']
        main(args)
        events=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        outputs.append((events,json.loads((folder/'results.json').read_text())))
    for event in ('train','validation'):
        a=[value for value in outputs[0][0] if value['event']==event]
        b=[value for value in outputs[1][0] if value['event']==event]
        keys=('loss','accuracy','lr','grad_norm_before_clip','hyper_grad_norm_before_clip') if event=='train' else ('loss','accuracy')
        assert [{key:value[key] for key in keys} for value in a]==[{key:value[key] for key in keys} for value in b]
    diag=outputs[1][1]['token_diagnostics']
    assert len(diag['first_nonzero_gradient'])==8 and max(diag['first_nonzero_gradient'].values())<=3
    assert not list(tmp_path.rglob('*.pt'))


def test_vit_gain_preserves_base_initialization():
    cfg=Config(width=24,depth=2,heads=3,rank=4,n_basis=6)
    torch.manual_seed(6);plain=build_model(cfg)
    torch.manual_seed(6);conditioned=build_model(replace(cfg,variant='dplr',adapter_gain=.25))
    assert parameter_digest(plain)==parameter_digest(conditioned)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_bf16_token_diagnostic_wakeup():
    cfg=Config(variant='dplr',width=24,depth=1,heads=3,rank=4,n_basis=6,adapter_gain=.5,diagnostic_steps='0,1,3,4')
    torch.manual_seed(9);model=build_model(cfg).cuda()
    images=torch.randn(4,3,32,32,device='cuda');labels=torch.tensor([1,2,3,4],device='cuda')
    diagnostic=TokenDiagnostics(model,cfg,[(image.cpu(),label.item()) for image,label in zip(images,labels)],torch.device('cuda:0'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    assert all(block['total_adapter_rms']==0 for block in diagnostic.collect(model,0)['live_branches'])
    for step in range(1,5):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=torch.nn.functional.cross_entropy(model(images).float(),labels)
        loss.backward();optimizer.step();diagnostic.observe_gradients(model,step)
        if step in diagnostic.steps:diagnostic.collect(model,step,images,labels)
    assert max(diagnostic.first_gradients.values())<=3 and len(diagnostic.first_gradients)==8
    assert diagnostic.hardware['device_uuid']
    assert torch.isfinite(loss)
def test_frozen_classifier_implementation_isolated(tmp_path):
    import hashlib
    import shutil
    import sys
    from class_fidelity import frozen_classifier_model
    from train_cifar10 import Config, build_model
    from dataclasses import asdict
    from pathlib import Path

    original = Path(__file__).parent / 'runs/dit-followup-20261007T174446Z/runner'
    if not original.is_dir():
        pytest.skip('Original auxiliary evaluator source is unavailable')
    runner = tmp_path / 'runner'
    runner.mkdir()
    names = ('condlr', 'conditioning', 'vit', 'train_cifar10')
    source = {}
    for name in names:
        target = runner / (name + '.py')
        shutil.copy2(original / target.name, target)
        source[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    directory = tmp_path / 'classifier'
    directory.mkdir()
    cfg = Config(width=24, depth=2, heads=3, patch_size=8, variant='plain')
    values = asdict(cfg)
    values.pop('adapter_gain')
    values.pop('diagnostic_steps')
    (directory / 'config.json').write_text(json.dumps(values))
    (directory / 'provenance.json').write_text(json.dumps(dict(source_sha256=source)))
    active = {name: sys.modules.get(name) for name in names}
    _, frozen = frozen_classifier_model(directory)
    assert active == {name: sys.modules.get(name) for name in names}
    current = build_model(cfg)
    current.load_state_dict(frozen.state_dict(), strict=True)
    frozen.eval(); current.eval()
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(current(x), frozen(x), rtol=0, atol=0)
    (runner / 'vit.py').write_text((runner / 'vit.py').read_text() + '\n# changed\n')
    with pytest.raises(ValueError, match='Verified original'):
        frozen_classifier_model(directory)

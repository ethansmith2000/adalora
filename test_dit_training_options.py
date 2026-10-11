"""Optimizer/compile ownership, null-class training and CFG correctness."""
import copy
import json
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from dit_cifar10 import CIFAR10DiT
from train_dit_cifar10 import (Config, drop_class_labels, evaluation_guidance_scales,
                              guided_prediction, intermediate_generation_steps, load_config, main,
                              make_optimizer, training_callable, update_ema)


@pytest.mark.parametrize('variant',['adaln_zero','dplr'])
def test_null_embedding_preserves_all_conditional_initialization_and_rng(variant):
    torch.set_num_threads(2)
    torch.manual_seed(81);base=CIFAR10DiT(width=24,depth=2,heads=3,patch_size=8,variant=variant,rank=2,n_basis=4)
    state=torch.get_rng_state()
    torch.manual_seed(81);cfg=CIFAR10DiT(width=24,depth=2,heads=3,patch_size=8,variant=variant,rank=2,n_basis=4,enable_unconditional=True)
    assert torch.equal(torch.get_rng_state(),state)
    assert cfg.statistics()['parameters']==base.statistics()['parameters']+24
    assert all(torch.equal(v,cfg.state_dict()[k]) for k,v in base.state_dict().items())
    x=torch.randn(2,3,32,32);ts=torch.tensor([2,80]);y=torch.tensor([1,9])
    with torch.no_grad():
        for m in (base,cfg):
            for block in m.blocks:block.modulation[-1].bias.fill_(.1)
            m.output.weight.fill_(.03)
    assert torch.equal(base(x,ts,y),cfg(x,ts,y))
    assert torch.isfinite(cfg(x,ts,torch.full_like(y,10))).all()
    with pytest.raises(IndexError):cfg(x,ts,torch.tensor([11,1]))


def test_null_initialization_matches_across_architectures():
    values=[]
    for variant in ('adaln_zero','dplr'):
        torch.manual_seed(82);values.append(CIFAR10DiT(width=24,variant=variant,enable_unconditional=True).null_class_embedding.detach())
    assert torch.equal(*values)


@pytest.mark.parametrize('scale',[0.,1.,1.5,2.])
def test_guidance_is_exact_and_uses_one_batched_call(scale):
    class Predictor(torch.nn.Module):
        enable_unconditional=True
        def forward(self,x,t,y):return x*.3+y[:,None,None,None]*.1+float(t)*.01
    m=Predictor();x=torch.randn(3,3,4,4);labels=torch.tensor([1,4,9]);t=20
    conditional=m(x,t,labels);unconditional=m(x,t,torch.full_like(labels,10))
    with patch.object(m,'forward',wraps=m.forward) as forward:
        actual=guided_prediction(m,x,t,labels,scale)
        assert forward.call_count==1
        assert forward.call_args.args[0].shape[0]==(3 if scale==1 else 6)
    torch.testing.assert_close(actual,unconditional+scale*(conditional-unconditional))
    m.enable_unconditional=False
    if scale!=1:
        with pytest.raises(ValueError):guided_prediction(m,x,t,labels,scale)


def test_label_dropout_is_reproducible_and_independent_of_noise_rng():
    labels=torch.arange(10).repeat(100)
    noise=torch.Generator().manual_seed(99);before=noise.get_state().clone()
    a=drop_class_labels(labels,.1,torch.Generator().manual_seed(83))
    b=drop_class_labels(labels,.1,torch.Generator().manual_seed(83))
    assert torch.equal(a,b) and 50<(a==10).sum()<150
    assert torch.equal(a[a!=10],labels[a!=10]) and torch.equal(before,noise.get_state())
    assert drop_class_labels(labels,0,None) is labels


def test_compile_wrapper_preserves_optimizer_and_ema_ownership():
    torch.set_num_threads(2);torch.manual_seed(84)
    m=CIFAR10DiT(width=24,depth=2,heads=3,patch_size=8,variant='dplr',rank=2,n_basis=4,enable_unconditional=True)
    with torch.no_grad():
        m.output.weight.normal_(std=.03)
        for block in m.blocks:block.modulation[-1].bias.fill_(.1);block.adapter.V.normal_(std=.03)
    eager=copy.deepcopy(m);ema=copy.deepcopy(m)
    compiler=torch.compile
    with patch('train_dit_cifar10.torch.compile',side_effect=lambda module,**kw:compiler(module,backend='eager',dynamic=kw['dynamic'],fullgraph=kw['fullgraph'])):
        compiled=training_callable(m,Config(compile_model=True,device='cpu'))
    x=torch.randn(2,3,32,32);t=torch.tensor([10,80]);y=torch.tensor([2,10])
    a=compiled(x,t,y);b=eager(x,t,y);torch.testing.assert_close(a,b)
    a.square().mean().backward();b.square().mean().backward()
    for p,q in zip(m.parameters(),eager.parameters()):torch.testing.assert_close(p.grad,q.grad)
    before={n:p.detach().clone() for n,p in ema.named_parameters()}
    opt=make_optimizer(m,Config(device='cpu',adam_beta1=.95));assert opt.defaults['betas']==(.95,.999) and not opt.defaults['fused']
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in compiled.parameters()}
    opt.step();update_ema(ema,m,.9)
    for n,p in ema.named_parameters():torch.testing.assert_close(p,.9*before[n]+.1*dict(m.named_parameters())[n])
    assert all(not n.startswith('_orig_mod') for n,_ in m.named_parameters())
    torch._dynamo.reset()


@pytest.mark.parametrize('options',[
    dict(adam_beta1=1.),dict(adam_beta2=-1.),dict(class_dropout_prob=1.),
    dict(guidance_scale=1.5),dict(guidance_scale=float('nan')),dict(guidance_scales=[1,2]),
    dict(guidance_scales='1,2,2',class_dropout_prob=.1),
    dict(guidance_scales='1.5,2',class_dropout_prob=.1),dict(compile_mode='bad'),dict(compile_model=1)])
def test_invalid_options(tmp_path,options):
    path=tmp_path/'config.json';path.write_text(json.dumps(options))
    with pytest.raises(ValueError):load_config(['--override_json',str(path)])


def test_cfg_training_and_guidance_curve_are_reproducible(tmp_path):
    rows=[]
    for diagnostic in ('','0,1,3,5,10,12'):
        output=tmp_path/str(bool(diagnostic))
        cfg=Config(variant='dplr',width=24,depth=2,heads=3,patch_size=8,rank=2,n_basis=4,
                   class_dropout_prob=.25,adam_beta1=.95,guidance_scales='1,1.5,2',
                   max_steps=12,warmup_steps=2,batch_size=8,synthetic=True,synthetic_samples=32,val_limit=8,
                   device='cpu',precision='fp32',num_workers=0,cpu_threads=2,log_every=12,eval_every=12,
                   sample_images=10,sampling_steps=2,num_train_timesteps=10,sample_batch_size=5,
                   diagnostic_steps=diagnostic,output_dir=str(output))
        path=tmp_path/(str(bool(diagnostic))+'.json');path.write_text(json.dumps(asdict(cfg)))
        rows.append(main(['--override_json',str(path)]))
        assert rows[-1]['training_setup']['dropped_training_labels']>0
        assert set(rows[-1]['guidance_evaluations'])=={'1.0','1.5','2.0'}
        for scale,entry in rows[-1]['guidance_evaluations'].items():
            assert entry['generated_images']==10 and entry['guidance_scale']==float(scale)
            assert entry['classifier_free_guidance']==(float(scale)!=1)
            assert entry['sample_grid_labels']==([i%10 for i in range(10)] if float(scale)==1 else [])
        assert list(output.glob('samples*.png'))==[output/'samples_ema.png']
    for key in ('raw_validation','ema_validation'):
        assert rows[0][key]['noise_mse']==rows[1][key]['noise_mse']
    assert rows[0]['training_setup']['dropped_training_labels']==rows[1]['training_setup']['dropped_training_labels']
    assert not list(tmp_path.rglob('*.pt'))


@pytest.mark.parametrize('value',['0','12','3,3','6,3','3,x',[3,6]])
def test_intermediate_generation_steps_validate(value):
    with pytest.raises(ValueError):
        intermediate_generation_steps(Config(max_steps=12,generative_eval_steps=value))


def test_intermediate_generation_preserves_training_and_all_rngs(tmp_path):
    import random
    import numpy as np
    from train_dit_cifar10 import generative_evaluation
    results=[]
    def consuming_evaluator(*args,**kwargs):
        # Evaluator constructors can use the global RNG even with local sampling seeds.
        torch.rand(37);random.random();np.random.rand(37)
        return generative_evaluation(*args,**kwargs)
    for steps in ('','3,6'):
        cfg=Config(variant='dplr',width=24,depth=2,heads=3,patch_size=8,rank=2,n_basis=4,
                   max_steps=12,warmup_steps=2,batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,
                   device='cpu',precision='fp32',cpu_threads=2,num_workers=0,log_every=6,eval_every=6,
                   class_dropout_prob=.25,adam_beta1=.95,guidance_scales='1,1.5,2',
                   sample_images=10,sample_batch_size=5,sampling_steps=2,num_train_timesteps=10,
                   diagnostic_steps='0,1,3,5,10,12',generative_eval_steps=steps,
                   output_dir=str(tmp_path/('curve' if steps else 'final')))
        path=tmp_path/('curve.json' if steps else 'final.json');path.write_text(json.dumps(asdict(cfg)))
        with patch('train_dit_cifar10.generative_evaluation',side_effect=consuming_evaluator):
            row=main(['--override_json',str(path)])
        state=(torch.get_rng_state(),random.getstate(),np.random.get_state())
        results.append((row,state))
    a,b=results[0][0],results[1][0]
    for key in ('raw_validation','ema_validation'):
        assert a[key]['noise_mse']==b[key]['noise_mse']
    assert a['design_diagnostics']==b['design_diagnostics']
    assert a['training_setup']==b['training_setup']
    assert torch.equal(results[0][1][0],results[1][1][0])
    assert results[0][1][1]==results[1][1][1]
    np.testing.assert_equal(results[0][1][2],results[1][1][2])
    assert [r['step'] for r in b['generative_validation_curve']]==[3,6,12]
    assert all(r['guidance_scale']==1 for r in b['generative_validation_curve'])
    assert not b['generative_validation_curve'][0]['sample_grid_labels']
    assert b['intermediate_generation_seconds']>=0
    assert list((tmp_path/'curve').glob('*.png'))==[tmp_path/'curve/samples_ema.png']
    assert not list(tmp_path.rglob('*.pt'))

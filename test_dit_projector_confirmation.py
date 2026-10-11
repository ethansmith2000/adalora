"""Parameter budgets, initialization, pairing and telemetry for the longer projector study."""
from dataclasses import asdict, replace
import json
import math

import pytest
import torch

from benchmark_dit_design import digest_parameters, rms
from dit_projector_confirmation import (CONTRASTS, DIAGNOSTICS, FAMILIES, MATCHED_CONTROLS,
    NEW_FAMILIES, PARAMETERS, SEEDS, SCALES, STEPS, aggregate, optimization_profiles,
    make_config, source_compatible, write_summary)
from dit_quality_diagnostics import QualityDiagnostics
from train_dit_cifar10 import load_config, make_model, main


def test_budgets_base_initialization_and_linear_silu_pair(tmp_path):
    torch.set_num_threads(2)
    bases={};adapters={}
    for family in FAMILIES:
        cfg=make_config(family,SEEDS[0],tmp_path/family,None)
        torch.manual_seed(SEEDS[0]);model=make_model(cfg)
        assert model.statistics()['parameters']==PARAMETERS[family]
        bases[family]=digest_parameters(model);adapters[family]=digest_parameters(model,True)
        if cfg.variant=='dplr':
            for block in model.blocks:
                assert torch.count_nonzero(block.adapter.V)==0
                c=torch.randn(2,cfg.width);x=torch.randn(2,3,cfg.width)
                assert torch.count_nonzero(block.adapter(x,c))==0
    assert len({bases[f] for f in FAMILIES if FAMILIES[f]['variant']=='dplr'}|{bases['plain']})==1
    assert adapters['linear_s32_h64']==adapters['silu_s32_h64']
    assert PARAMETERS['linear_s32_h64']==PARAMETERS['silu_s32_h64']
    for candidate,control in MATCHED_CONTROLS.items():
        assert abs(PARAMETERS[control]/PARAMETERS[candidate]-1)<.00065


def test_fourteen_configs_isolate_architecture_and_reuse_anchor_seeds(tmp_path):
    common=[]
    for family in NEW_FAMILIES:
        for seed in SEEDS:
            cfg=make_config(family,seed,tmp_path/f'{family}_{seed}',None)
            path=tmp_path/'config.json';path.write_text(json.dumps(asdict(cfg)))
            assert asdict(load_config(['--override_json',str(path)]))==asdict(cfg)
            assert cfg.rank==8 and cfg.width==256 and cfg.depth==8
            assert cfg.adapter_gain==1/math.sqrt(8) and cfg.adapter_lr_multiplier==1
            assert cfg.max_steps==cfg.lr_schedule_steps==30000 and cfg.guidance_scales=='1,2'
            assert cfg.projector_groups==cfg.decoder_groups==0
            assert cfg.train_limit is cfg.val_limit is None
            common.append({k:v for k,v in asdict(cfg).items() if k not in
                ('variant','mlp_ratio','n_basis','coefficient_hidden','coefficient_activation','seed','output_dir')})
    assert len(common)==14 and all(c==common[0] for c in common)


def fake_rows():
    rows=[]
    for i,family in enumerate(FAMILIES):
        for j,seed in enumerate(SEEDS):
            def point(scale):
                return dict(fid=80-i*2+j+scale,class_agreement={'generated':{'agreement':.4+i*.01+j*.001}})
            diagnostics=[]
            for step in [int(v) for v in DIAGNOSTICS.split(',')]:
                magnitude=dict(input_factor_rms=2.,output_factor_rms=.1,transform_frobenius_mean=1.)
                if family!='affine_s16':magnitude.update(input_coefficient_rms=.5,output_coefficient_rms=.6,input_basis_parameter_rms=.1,output_basis_parameter_rms=.01)
                event=dict(step=step,branches=[dict(delta_to_base_rms=.2,combined_to_base_rms=1.1,gated_residual_to_input_rms=.1)])
                if step or family!='affine_s16':event['fixed_condition_projector_magnitudes']=[magnitude]
                if step:event['fixed_token_condition_function_updates']=[dict(update_rms=(i+1)*.001)]
                diagnostics.append(event)
            rows.append(dict(family=family,config={'seed':seed},parameters=PARAMETERS[family],adapter_parameters=0,
                             forward_matmul_flops_per_image_estimate=1000,peak_allocated_bytes=1024**2,
                             guidance_evaluations={str(s):point(s) for s in SCALES},
                             generative_validation_curve=[dict(step=s,**point(1)) for s in STEPS],
                             validation_curve=[dict(step=s,ema={'noise_mse':.06}) for s in STEPS],
                             design_diagnostics={'diagnostics':diagnostics}))
    return rows


def test_paired_contrasts_order_and_missing_duplicate_rejection():
    rows=fake_rows();result=aggregate(rows)
    for a,b in CONTRASTS:
        i=list(FAMILIES).index(a);j=list(FAMILIES).index(b)
        for scale in SCALES:
            v=result['paired_differences'][a+'_minus_'+b][str(scale)]
            assert v['fid_candidate_minus_control']['values']==[-2*(i-j)]*2
            assert v['agreement_pp_candidate_minus_control']['values']==pytest.approx([i-j]*2)
    with pytest.raises(ValueError):aggregate(rows[:-1])
    with pytest.raises(ValueError):aggregate([*rows[:-1],rows[0]])


def test_profiles_handle_missing_anchor_telemetry_and_report_roundtrip(tmp_path):
    rows=fake_rows();profiles=optimization_profiles(rows)
    assert 'input_coefficient_rms' not in profiles['affine_s16']['0']
    assert profiles['affine_s16']['1']['paired_update_ratio_vs_affine_s16']==[1.,1.]
    assert profiles['affine_s64']['1']['paired_update_ratio_vs_affine_s16']==pytest.approx([5/3]*2)
    write_summary(tmp_path,rows,{'checks':{'fixture':True}},{'limits':['Synthetic test data only.']})
    saved=json.loads((tmp_path/'comparison.json').read_text())
    assert saved['optimization_profiles']==profiles
    assert len(saved['aggregates'])==10
    for kind in ('convergence','guidance','updates'):
        assert (tmp_path/f'{kind}.png').stat().st_size>1000


def test_only_diagnostic_source_change_can_transfer_anchors():
    old={'train_dit_cifar10.py':'a','dit_cifar10.py':'b','dit_quality_diagnostics.py':'c'}
    assert source_compatible(dict(old,**{'dit_quality_diagnostics.py':'new'}),old)
    assert not source_compatible(dict(old,**{'train_dit_cifar10.py':'new'}),old)
    assert not source_compatible({'dit_cifar10.py':'b'},old)


def test_magnitude_probes_equal_direct_values_and_preserve_rng(tmp_path):
    cfg=replace(make_config('silu_s32_h64',86,tmp_path,None),width=24,depth=2,heads=3)
    model=make_model(cfg)
    probe=QualityDiagnostics.__new__(QualityDiagnostics)
    probe.projector_groups=model.grouped_adapters();probe.c=torch.randn(3,24)
    state=torch.get_rng_state().clone();precision=torch.get_float32_matmul_precision()
    values=probe.projector_magnitudes()
    assert torch.equal(state,torch.get_rng_state()) and torch.get_float32_matmul_precision()==precision
    for v,(_,adapter) in zip(values,probe.projector_groups):
        a,b=adapter.coefficient_values(probe.c).chunk(2,-1)
        assert v['input_coefficient_rms']==rms(a) and v['output_coefficient_rms']==rms(b)
        assert v['output_factor_rms']==v['transform_frobenius_mean']==v['output_basis_parameter_rms']==0
        with torch.no_grad():adapter.V.normal_(std=.02)
    values=probe.projector_magnitudes()
    for v,(_,adapter) in zip(values,probe.projector_groups):
        u,w=adapter.factors(probe.c)
        expected=adapter.gain*(u@w).flatten(1).norm(dim=1).mean().item()
        assert v['transform_frobenius_mean']==pytest.approx(expected,rel=1e-6)


@pytest.mark.parametrize('activation',['linear','silu'])
def test_hidden_projector_telemetry_does_not_change_training(tmp_path,activation):
    results=[]
    for enabled in (False,True):
        cfg=replace(make_config('linear_s32_h64',86,tmp_path/str(enabled),None),
                    width=24,depth=2,heads=3,patch_size=8,n_basis=4,coefficient_hidden=8,
                    coefficient_activation=activation,device='cpu',precision='fp32',compile_model=False,
                    synthetic=True,synthetic_samples=16,val_limit=8,batch_size=4,max_steps=12,
                    warmup_steps=2,num_workers=0,eval_every=12,log_every=12,benchmark_warmup_steps=2,
                    generative_eval_steps='',guidance_scales='',fid_images=0,sample_images=0,
                    diagnostic_steps='0,1,3,5,10,12' if enabled else '')
        p=tmp_path/'cfg.json';p.write_text(json.dumps(asdict(cfg)))
        results.append(main(['--override_json',str(p)]))
    for key in ('raw_validation','ema_validation'):
        assert results[0][key]['noise_mse']==results[1][key]['noise_mse']
        assert results[0][key]['class_prediction_change_rms']==results[1][key]['class_prediction_change_rms']
    diag=results[1]['design_diagnostics']
    assert diag['adapter_first_nonzero_gradient_observed']
    assert all(step<=10 for step in diag['adapter_first_nonzero_gradient_observed'].values())

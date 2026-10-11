"""Update-only selection, gain/rate invariants and real schedule-prefix checks."""

from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest
import torch

from benchmark_dit_design import digest_parameters
from benchmark_dit_gain import choose_candidate
from dit_width_calibration import (CAL_SEEDS, CONFIRM_SEEDS, CAL_FAMILIES, FAMILIES,
                                   OBSERVATIONS, TOLERANCE, MAX_CORRECTIONS,
                                   aggregate_candidate, next_rates, make_config)
from train_dit_cifar10 import learning_rate_scale, make_model, main


def test_protocol_gain_and_rate_axes(tmp_path):
    assert set(CAL_SEEDS).isdisjoint(CONFIRM_SEEDS) and min(CAL_SEEDS)>59
    for family in FAMILIES:
        pilot=make_config(family,60,.7,tmp_path/family,'teacher',True)
        full=make_config(family,62,.7,tmp_path/family,'teacher',False)
        assert pilot.train_limit is pilot.val_limit is full.train_limit is full.val_limit is None
        assert pilot.max_steps==500 and full.max_steps==30000
        assert not pilot.fid_images and pilot.classifier_run is None
        assert full.fid_images==5000 and full.classifier_run=='teacher'
        assert not pilot.sample_images and not full.sample_images
        for key in ('variant','rank','n_basis','adapter_gain','adapter_lr_multiplier','warmup_steps','lr_schedule_steps','lr'):
            assert getattr(pilot,key)==getattr(full,key)
        assert all(learning_rate_scale(pilot,step)==learning_rate_scale(full,step) for step in OBSERVATIONS)
    assert math.isclose(FAMILIES['r8_s12_gain_lr']['adapter_gain']/FAMILIES['r8_s12_lr']['adapter_gain'],16/12)
    assert math.isclose(FAMILIES['r16_s6_gain_lr']['adapter_gain']/FAMILIES['r16_s6_lr']['adapter_gain'],16/6)


def row(seed, update, fid=0):
    return dict(config=dict(seed=seed),generative_evaluation=dict(fid=fid),
                design_diagnostics=dict(diagnostics=[dict(step=step,
                    fixed_token_condition_function_updates=[dict(update_rms=update)]*6) for step in OBSERVATIONS]))


def test_selection_and_corrections_ignore_quality():
    original=[row(60,.8,1),row(61,1.2,999)]
    altered=[row(60,.8,999),row(61,1.2,1)]
    a=aggregate_candidate('r8_s12_lr',0,1.,original,1.)
    b=aggregate_candidate('r8_s12_lr',0,1.,altered,1.)
    assert a==b and a['within_tolerance']
    candidates={'r8_s16':[a], 'r8_s12_lr':[a]}
    assert next_rates(0,{'r8_s16':1.,'r8_s12_lr':1.},candidates,1.)=={'r8_s12_lr':1.}
    assert next_rates(1,{'r8_s12_lr':1.},candidates,1.)=={}
    bad=aggregate_candidate('r8_s12_lr',1,1.,[row(60,.5),row(61,.5)],1.)
    candidates['r8_s12_lr'].append(bad)
    assert next_rates(1,{'r8_s12_lr':1.},candidates,1.)=={'r8_s12_lr':2.}
    assert next_rates(MAX_CORRECTIONS,{'r8_s12_lr':1.},candidates,1.)=={}
    assert choose_candidate([bad,a],1.)==a
    with pytest.raises(ValueError):aggregate_candidate('r8_s12_lr',0,1.,original[:1],1.)
    for value in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):aggregate_candidate('r8_s12_lr',0,1.,[row(60,value),row(61,value)],1.)


def test_gain_rate_variants_share_initialization_and_budget(tmp_path):
    torch.set_num_threads(2)
    bases=set()
    for prefix in ('r8_s12','r16_s6'):
        adapters=set();parameters=set()
        for suffix in ('_default','_lr','_gain_lr'):
            torch.manual_seed(60)
            cfg=make_config(prefix+suffix,60,.7,tmp_path/prefix,'teacher',True)
            model=make_model(cfg)
            bases.add(digest_parameters(model))
            adapters.add(digest_parameters(model,True))
            parameters.add(model.statistics()['parameters'])
        assert len(adapters)==len(parameters)==1
    assert len(bases)==1


@pytest.mark.parametrize('family',['r8_s12_gain_lr','r16_s6_lr'])
def test_calibration_matches_exact_training_prefix(tmp_path,family):
    records=[]
    for maximum in (12,18):
        cfg=make_config(family,60,.7,tmp_path/str(maximum),'teacher',True)
        values=asdict(cfg)
        values.update(width=24,depth=2,patch_size=8,max_steps=maximum,lr_schedule_steps=18,warmup_steps=2,
                      batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,device='cpu',precision='fp32',
                      cpu_threads=2,num_workers=0,log_every=12,eval_every=12,benchmark_warmup_steps=2,
                      diagnostic_steps='0,1,3,5,10,12'+(',18' if maximum==18 else ''))
        path=tmp_path/f'{maximum}.json';path.write_text(json.dumps(values))
        main(['--override_json',str(path)])
        events=[json.loads(line) for line in (Path(values['output_dir'])/'metrics.jsonl').read_text().splitlines()]
        selected=[event for event in events if event['event'] in ('design_diagnostic','validation') and event['step']<=12]
        for event in selected:
            if event['event']=='validation':
                for state in ('raw','ema'):event[state].pop('seconds')
        records.append(selected)
    assert records[0]==records[1]
    assert not list(tmp_path.rglob('*.pt'))

"""Core study selection, schedule-prefix and capacity-control invariants."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from dit_core_confirmation import (CAL_FAMILIES,CAL_SEEDS,CONFIRM_SEEDS,FAMILIES,
                                   OBSERVATIONS,aggregate_candidate,make_config)
from train_dit_cifar10 import Config,learning_rate_scale,main,make_model


def test_full_loader_core_protocol_and_schedule_prefix(tmp_path):
    assert set(CAL_SEEDS).isdisjoint(CONFIRM_SEEDS) and min(CAL_SEEDS)>52
    assert set(CAL_FAMILIES)<set(FAMILIES)
    for family in FAMILIES:
        pilot=make_config(family,53,.4,tmp_path/family,'teacher',True)
        confirm=make_config(family,55,.4,tmp_path/family,'teacher',False)
        assert pilot.train_limit is pilot.val_limit is confirm.train_limit is confirm.val_limit is None
        assert pilot.max_steps==500 and confirm.max_steps==30000
        assert not pilot.sample_images and not pilot.fid_images and pilot.classifier_run is None
        assert confirm.fid_images==5000 and confirm.sample_images==100
        for key in ('variant','rank','mlp_ratio','core_rank','adapter_gain','adapter_lr_multiplier','lr','warmup_steps','lr_schedule_steps'):
            assert getattr(pilot,key)==getattr(confirm,key)
        assert all(learning_rate_scale(pilot,step)==learning_rate_scale(confirm,step) for step in OBSERVATIONS)
    ffn=make_model(make_config('ffn_r2_budget',55,1.,tmp_path/'ffn','teacher'))
    basis=make_model(make_config('basis_r2',55,1.,tmp_path/'basis','teacher'))
    assert ffn.statistics()['parameters']-basis.statistics()['parameters']==384


def test_core_selection_uses_two_seeds_and_ignores_quality():
    def row(seed,update,quality):
        return dict(config=dict(seed=seed),generative_evaluation=dict(fid=quality),
                    design_diagnostics=dict(diagnostics=[dict(step=step,fixed_token_condition_function_updates=[dict(update_rms=update)]*6) for step in OBSERVATIONS]))
    a=aggregate_candidate('core_r8',0,1.,[row(53,1.,10),row(54,3.,90)],2.)
    b=aggregate_candidate('core_r8',0,1.,[row(53,1.,90),row(54,3.,10)],2.)
    assert a==b and a['metric']==2. and a['within_tolerance']
    with pytest.raises(ValueError):
        aggregate_candidate('core_r8',0,1.,[row(53,1.,10)],2.)


@pytest.mark.parametrize('family',['core_r8','dplr_core_r8_p2','ffn_r2_budget'])
def test_short_calibration_is_exact_training_prefix(tmp_path,family):
    events=[]
    for maximum in (12,18):
        cfg=make_config(family,53,.6,tmp_path/str(maximum),'teacher',True)
        values=asdict(cfg)
        values.update(width=24,depth=2,patch_size=8,rank=4,max_steps=maximum,lr_schedule_steps=18,warmup_steps=2,
                      batch_size=4,synthetic=True,synthetic_samples=16,val_limit=8,device='cpu',precision='fp32',
                      cpu_threads=2,num_workers=0,log_every=12,eval_every=12,benchmark_warmup_steps=2,
                      diagnostic_steps='0,1,3,5,10,12'+(',18' if maximum==18 else ''))
        path=tmp_path/f'{maximum}.json';path.write_text(json.dumps(values))
        main(['--override_json',str(path)])
        rows=[json.loads(line) for line in (Path(values['output_dir'])/'metrics.jsonl').read_text().splitlines()]
        selected=[row for row in rows if row['event'] in ('design_diagnostic','validation') and row['step']<=12]
        for row in selected:
            if row['event']=='validation':
                for state in ('raw','ema'):
                    row[state].pop('seconds')
        events.append(selected)
    assert events[0]==events[1]
    assert not list(tmp_path.rglob('*.pt'))

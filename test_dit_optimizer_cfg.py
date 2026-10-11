"""Research recipe isolation and paired contrast calculations."""
from dataclasses import asdict
import json
import math

import pytest

from dit_optimizer_cfg import FAMILIES, SEEDS, SCALES, STEPS, aggregate, make_config, same_stage_evidence
from train_dit_cifar10 import load_config


def test_exact_optimizer_grid_preserves_all_other_training_settings(tmp_path):
    configs=[]
    for seed in SEEDS:
        for family,settings in FAMILIES.items():
            cfg=make_config(family,seed,tmp_path/f'{family}_{seed}',None)
            path=tmp_path/'config.json';path.write_text(json.dumps(asdict(cfg)))
            assert asdict(load_config(['--override_json',str(path)]))==asdict(cfg)
            assert cfg.variant==settings['variant'] and cfg.adam_beta1==settings['adam_beta1']
            assert cfg.adapter_gain==1/math.sqrt(8) and cfg.adapter_lr_multiplier==1
            assert cfg.train_limit is cfg.val_limit is None
            assert cfg.max_steps==cfg.lr_schedule_steps==30000 and cfg.warmup_steps==50
            assert cfg.class_dropout_prob==.1 and cfg.generative_eval_steps=='5000,10000,20000'
            assert cfg.fid_images==5000 and cfg.sampling_steps==50
            assert cfg.compile_model and cfg.fused_adamw
            configs.append({k:v for k,v in asdict(cfg).items() if k not in ('variant','adam_beta1','seed','output_dir')})
    assert len(configs)==8 and all(c==configs[0] for c in configs)


def example_rows():
    rows=[]
    for family,settings in FAMILIES.items():
        better=settings['adam_beta1']==.95
        for i,seed in enumerate(SEEDS):
            def point(scale):
                return dict(fid=60+i*2+scale-5*better,
                            class_agreement={'generated':{'agreement':.5+i*.01+.02*better}})
            rows.append(dict(family=family,config={'seed':seed},parameters=5000000,adapter_parameters=100000,
                             guidance_evaluations={str(s):point(s) for s in SCALES},
                             generative_validation_curve=[dict(step=s,**point(1)) for s in STEPS],
                             validation_curve=[dict(step=s,ema={'noise_mse':.06+i*.001}) for s in STEPS],
                             ema_validation={'noise_mse':.06+i*.001}))
    return rows


def test_paired_beta_contrasts_keep_seed_direction_and_scale_separate():
    result=aggregate(example_rows())
    assert len(result['aggregates'])==4
    for architecture in ('adaln','dplr'):
        for scale in SCALES:
            point=result['paired_beta_differences'][architecture][str(scale)]
            assert point['fid_b095_minus_b09']['values']==[-5,-5]
            assert point['fid_b095_minus_b09']['sd']==0
            assert point['agreement_pp_b095_minus_b09']['values']==pytest.approx([2,2])
    assert result['aggregates'][0]['final_guidance']['1.0']['fid']['sd']==pytest.approx(math.sqrt(2))


def test_incomplete_or_duplicate_grid_cannot_report_a_contrast():
    rows=example_rows()
    with pytest.raises(ValueError):aggregate(rows[:-1])
    with pytest.raises(ValueError):aggregate([*rows[:-1],rows[0]])


def test_summary_replay_ignores_only_the_growing_disk_tally():
    import copy
    recorded=dict(rows=[dict(fid=62.,evidence_sha256={'metrics':'abc'})],
                  audit=dict(bytes_retained_under_screen=100,finite_logged_metrics=True))
    replayed=copy.deepcopy(recorded);replayed['audit']['bytes_retained_under_screen']=200
    assert same_stage_evidence(replayed,recorded)
    assert recorded['audit']['bytes_retained_under_screen']==100
    replayed['rows'][0]['fid']=61.
    assert not same_stage_evidence(replayed,recorded)
    replayed['rows'][0]['fid']=62.;replayed['rows'][0]['evidence_sha256']['metrics']='changed'
    assert not same_stage_evidence(replayed,recorded)

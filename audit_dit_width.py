#!/usr/bin/env python3
"""Reconstruct update-only rate selection and paired rank/width quality evidence."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import torch

from audit_dit_core import audit_stage
from benchmark_dit_gain import choose_candidate
from class_fidelity import classifier_provenance, frozen_classifier_model
from dit_width_calibration import (CAL_SEEDS, CONFIRM_SEEDS, CAL_FAMILIES, FAMILIES,
                                   OBSERVATIONS, MAX_CORRECTIONS, TOLERANCE,
                                   aggregate_candidate, next_rates, make_config, digest)
from train_dit_cifar10 import write_json


def audit(output):
    torch.set_num_threads(2)
    output=Path(output).resolve()
    protocol=json.loads((output/'protocol.json').read_text())
    report=json.loads((output/'report.json').read_text())
    assert protocol['state']=='complete' and protocol['max_concurrent_workers']==6
    assert protocol['families']==FAMILIES and protocol['calibration_families']==list(CAL_FAMILIES)
    assert protocol['calibration_seeds']==list(CAL_SEEDS) and protocol['confirmation_seeds']==list(CONFIRM_SEEDS)
    assert set(CAL_SEEDS).isdisjoint(CONFIRM_SEEDS) and min(CAL_SEEDS)>59
    assert protocol['observation_steps']==list(OBSERVATIONS)
    assert protocol['max_corrections']==MAX_CORRECTIONS and protocol['relative_tolerance']==TOLERANCE
    assert protocol['mandatory_first_correction']
    assert not any(protocol[key] for key in ('weights_saved','periodic_checkpoints','test_evaluated','compile','dedicated_speed_benchmarks'))
    assert digest(output/'prelaunch-checks.json')==protocol['prelaunch_checks_sha256']
    for name,value in protocol['source_sha256'].items():assert digest(output/'runner'/name)==value
    assert digest(Path(protocol['evaluator_cache']['path']))==protocol['evaluator_cache']['sha256']
    classifier=classifier_provenance(protocol['classifier_run'])
    assert classifier==protocol['auxiliary_classifier']
    frozen_classifier_model(protocol['classifier_run'])
    calibration=json.loads((output/'calibration.json').read_text())
    assert calibration==report['calibration']
    candidates={family:[] for family in CAL_FAMILIES};rates={family:1. for family in CAL_FAMILIES};target=None
    total_calibration=0
    for attempt,summary in enumerate(calibration['rounds']):
        rows=summary['rows'];total_calibration+=len(rows)
        assert {(row['family'],row['config']['seed']) for row in rows}=={(family,seed) for family in rates for seed in CAL_SEEDS}
        audit_stage(output,f'calibration{attempt}',protocol,rows,config_factory=make_config)
        if target is None:
            target=aggregate_candidate('r8_s16',0,1.,[row for row in rows if row['family']=='r8_s16'])['metric']
        for family,rate in rates.items():
            assert all(row['config']['adapter_lr_multiplier']==rate for row in rows if row['family']==family)
            candidates[family].append(aggregate_candidate(family,attempt,rate,[row for row in rows if row['family']==family],target))
        rates=next_rates(attempt,rates,candidates,target)
        if not rates:assert attempt==len(calibration['rounds'])-1
    assert 18<=total_calibration<=protocol['maximum_calibration_runs']==34
    selected={family:choose_candidate(values,target) for family,values in candidates.items()}
    assert target==calibration['target'] and candidates==calibration['candidates'] and selected==calibration['selected']
    assert all(value['within_tolerance'] for value in selected.values())
    recipes=json.loads((output/'recipes.json').read_text())
    assert recipes==report['recipes'] and recipes['selected']==selected and not recipes['selection_uses_quality']
    assert recipes['calibration_sha256']==digest(output/'calibration.json')
    assert protocol['recipes_sha256']==digest(output/'recipes.json')
    assert recipes['gains']=={family:value['adapter_gain'] for family,value in FAMILIES.items()}
    assert recipes['multipliers']=={family:selected[family]['adapter_lr_multiplier'] if family in selected else 1. for family in FAMILIES}
    confirmation=json.loads((output/'confirmation/screen.json').read_text())
    assert datetime.fromisoformat(recipes['frozen_utc'])<datetime.fromisoformat(confirmation['created_utc'])
    rows=report['confirmation']['rows']
    assert len(rows)==protocol['confirmation_runs']==21
    assert {(row['family'],row['config']['seed']) for row in rows}=={(family,seed) for family in FAMILIES for seed in CONFIRM_SEEDS}
    audit_stage(output,'confirmation',protocol,rows,config_factory=make_config)
    real=[]
    keyed={(row['family'],row['config']['seed']):row for row in rows}
    for seed in CONFIRM_SEEDS:
        for prefix in ('r8_s12','r16_s6'):
            group=[keyed[prefix+suffix,seed] for suffix in ('_default','_lr','_gain_lr')]
            assert len({row['design_diagnostics']['adapter_initialization_sha256'] for row in group})==1
            assert len({row['parameters'] for row in group})==1
    for row in rows:
        assert row['config']['adapter_lr_multiplier']==recipes['multipliers'][row['family']]
        evaluation=row['generative_evaluation']
        assert evaluation['model_state']=='EMA' and evaluation['class_agreement']['classifier']==classifier
        assert evaluation['fid_fake_images']==evaluation['fid_real_images']==evaluation['generated_images']==5000
        assert evaluation['sample_grid_labels']==[] and not evaluation['saved_generated_dataset']
        for label in ('real','generated'):
            value=evaluation['class_agreement'][label]
            matrix=value['confusion_rows_requested_columns_predicted']
            assert value['examples']==5000 and value['class_counts']==[500]*10
            assert [sum(line) for line in matrix]==[500]*10
            assert math.isclose(value['agreement'],sum(matrix[i][i] for i in range(10))/5000,abs_tol=1e-7)
        real.append(evaluation['class_agreement']['real'])
    assert all(value==real[0] for value in real)
    log=(output/'adalora_width_controller.log').read_text()
    progress=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
    assert all(event['submitted']-event['completed']<=6 for event in progress if event.get('event')=='stage_progress')
    assert 'exited: adalora_width_controller (exit status 0; expected)' in (output/'controller.log').read_text()
    assert not list(output.rglob('*.sock'))
    assert not [path for path in output.rglob('*') if path.suffix in ('.pt','.pth','.ckpt','.safetensors')]
    for path in Path('/tmp/vast_gpu_claims').glob('*.claim'):
        try:claim=json.loads(path.read_text())
        except (ValueError,OSError):continue
        assert not claim.get('job','').startswith('adalora_'+output.name.replace('-','_'))
    retained=sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    assert retained<protocol['expected_storage_upper_bound_bytes']
    checks=('successful_exits_closed_supervisors','source_runtime_data_configs_budgets',
            'selection_recomputed_without_quality','mandatory_response_bounded_corrections',
            'frozen_gain_rate_fresh_confirmation','initialization_gradients_paired_traces',
            'equal_shape_base_gain_lr_adapter_init','claimed_gpu_matches_runtime',
            'frozen_evaluators_counts','bounded_concurrency','no_weights_test_speed_study','storage_within_bound')
    result=dict(checks={key:True for key in checks},calibration_runs=total_calibration,
                confirmation_runs=21,bytes_retained=retained,report_sha256=digest(output/'report.json'))
    write_json(output/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(audit(parser.parse_args().output),indent=2))

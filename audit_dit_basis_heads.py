#!/usr/bin/env python3
"""Reconstruct update-only rate selection and paired shared-basis/layer-head quality evidence."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import torch

from audit_dit_core import audit_stage
from benchmark_dit_gain import choose_candidate
from class_fidelity import classifier_provenance, frozen_classifier_model
from dit_basis_head_screen import (CAL_SEEDS, CONFIRM_SEEDS, CAL_FAMILIES, FAMILIES,
                                   OBSERVATIONS, MAX_CORRECTIONS, TOLERANCE,
                                   aggregate_candidate, next_rates, make_config, digest)
from train_dit_cifar10 import write_json


def audit(output):
    torch.set_num_threads(2)
    output=Path(output).resolve()
    protocol=json.loads((output/'protocol.json').read_text())
    report=json.loads((output/'report.json').read_text())
    assert protocol['state']=='complete' and protocol['max_concurrent_workers']==2
    assert protocol['families']==FAMILIES and protocol['calibration_families']==list(CAL_FAMILIES)
    assert protocol['calibration_seeds']==list(CAL_SEEDS) and protocol['confirmation_seeds']==list(CONFIRM_SEEDS)
    assert set(CAL_SEEDS).isdisjoint(CONFIRM_SEEDS) and min(CAL_SEEDS)>68
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
            target=aggregate_candidate('independent_linear',0,1.,[row for row in rows if row['family']=='independent_linear'])['metric']
        for family,rate in rates.items():
            assert all(row['config']['adapter_lr_multiplier']==rate for row in rows if row['family']==family)
            candidates[family].append(aggregate_candidate(family,attempt,rate,[row for row in rows if row['family']==family],target))
        rates=next_rates(attempt,rates,candidates,target)
        if not rates:assert attempt==len(calibration['rounds'])-1
    assert 22<=total_calibration<=protocol['maximum_calibration_runs']==42
    selected={family:choose_candidate(values,target) for family,values in candidates.items()}
    assert target==calibration['target'] and candidates==calibration['candidates'] and selected==calibration['selected']
    assert all(value['within_tolerance'] for value in selected.values())
    recipes=json.loads((output/'recipes.json').read_text())
    assert recipes==report['recipes'] and recipes['selected']==selected and not recipes['selection_uses_quality']
    assert recipes['calibration_sha256']==digest(output/'calibration.json')
    assert protocol['recipes_sha256']==digest(output/'recipes.json')
    assert recipes['gains']=={family:value['adapter_gain'] for family,value in FAMILIES.items()}
    assert recipes['multipliers']=={family:selected[family]['adapter_lr_multiplier'] if family in selected else 1. for family in FAMILIES}
    confirmation=json.loads((output/'screen/screen.json').read_text())
    assert datetime.fromisoformat(recipes['frozen_utc'])<datetime.fromisoformat(confirmation['created_utc'])
    rows=report['confirmation']['rows']
    assert len(rows)==protocol['confirmation_runs']==12
    assert {(row['family'],row['config']['seed']) for row in rows}=={(family,seed) for family in FAMILIES for seed in CONFIRM_SEEDS}
    audit_stage(output,'screen',protocol,rows,config_factory=make_config)
    real=[]
    keyed={(row['family'],row['config']['seed']):row for row in rows}
    expected_counts={'independent_linear':774336,'two_bases_s16':282816,'two_bases_s44':777744,
                     'all_bases_s77':769692,'mlp_linear':802272,'mlp_silu':802272}
    for seed in CONFIRM_SEEDS:
        assert keyed['mlp_linear',seed]['design_diagnostics']['adapter_initialization_sha256']==keyed['mlp_silu',seed]['design_diagnostics']['adapter_initialization_sha256']
        assert keyed['mlp_linear',seed]['parameters']==keyed['mlp_silu',seed]['parameters']
        reference=keyed['independent_linear',seed]['parameters']
        assert all(abs(keyed[family,seed]['parameters']/reference-1)<.001 for family in ('two_bases_s44','all_bases_s77'))
    for row in rows:
        cfg=row['config'];diag=row['design_diagnostics'];count=FAMILIES[row['family']]['projector_groups']
        assert row['adapter_parameters']==expected_counts[row['family']]
        assert row['parameters']==4179696+row['adapter_parameters']
        expected_layers=[list(range(i*6//count,(i+1)*6//count)) for i in range(count)]
        assert diag['projector_group_layers']==row['projector_group_layers']==expected_layers
        assert diag['coefficient_activation']==row['coefficient_activation']==cfg['coefficient_activation']
        decoder_count=cfg['decoder_groups'] or 6
        expected_decoders=[list(range(i*6//decoder_count,(i+1)*6//decoder_count)) for i in range(decoder_count)]
        assert diag['decoder_group_layers']==row['decoder_group_layers']==expected_decoders
        assert diag['coefficient_hidden']==row['coefficient_hidden']==cfg['coefficient_hidden']
        expected_tensors=2*decoder_count+6*(4 if cfg['coefficient_hidden'] else 2)
        assert len(diag['adapter_first_nonzero_gradient_observed'])==row['adapter_parameter_tensors']==expected_tensors
        assert row['adapter_parameters']==row['coefficient_parameters']+row['decoder_parameters']
        assert cfg['max_steps']==5000 and row['completed_steps']==5000
        for event in diag['diagnostics'][1:]:
            assert len(event['projector_gradient_groups_after_clip'])==count
            assert len(event['decoder_gradient_groups_after_clip'])==decoder_count
            assert len(event['coefficient_head_gradient_groups_after_clip'])==6
            assert [item['layers'] for item in event['fixed_condition_projector_magnitudes']]==expected_layers
            for indices in expected_layers:
                values=[event['fixed_token_condition_function_updates'][i] for i in indices]
                assert all(value==values[0] for value in values)
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
    log=(output/'adalora_basis_head_controller.log').read_text()
    progress=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
    assert all(event['submitted']-event['completed']<=2 for event in progress if event.get('event')=='stage_progress')
    assert 'exited: adalora_basis_head_controller (exit status 0; expected)' in (output/'controller.log').read_text()
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
            'base_gain_lr_layer_heads_mlp_activation_init','claimed_gpu_matches_runtime',
            'frozen_evaluators_counts','bounded_concurrency','no_weights_test_speed_study','storage_within_bound')
    result=dict(checks={key:True for key in checks},calibration_runs=total_calibration,
                confirmation_runs=12,bytes_retained=retained,report_sha256=digest(output/'report.json'))
    write_json(output/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(audit(parser.parse_args().output),indent=2))

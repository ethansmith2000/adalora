#!/usr/bin/env python3
"""Independently reconstruct configs, initialization, pairing and screen evidence."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone

import torch

from benchmark_dit_design import digest_parameters
from class_fidelity import classifier_provenance, frozen_classifier_model
from rank_width_search import DIFFUSION, VIT, SEEDS, digest, make_config
from train_cifar10 import Config as ViTConfig, build_model, model_stats
from train_dit_cifar10 import Config as DiTConfig, make_model, write_json
from vit_quality_diagnostics import parameter_digest


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(output):
    torch.set_num_threads(2)
    output=Path(output).resolve()
    protocol=json.loads((output/'protocol.json').read_text())
    screen=json.loads((output/'screen/screen.json').read_text())
    assert protocol['state']=='complete' and protocol['seeds']==list(SEEDS)
    assert protocol['families']==dict(diffusion=DIFFUSION,vit=VIT)
    assert protocol['max_concurrent_workers']==6 and not protocol['dedicated_speed_benchmarks']
    assert digest(output/'prelaunch-checks.json')==protocol['prelaunch_checks_sha256']
    assert not any(protocol[key] for key in ('weights_saved','periodic_checkpoints','test_evaluated','compile','confirmation_launched'))
    assert len(screen['jobs'])==64
    expected={(kind,family,seed) for kind,families in protocol['families'].items() for family in families for seed in SEEDS}
    assert {(job['kind'],job['family'],job['seed']) for job in screen['jobs']}==expected
    for name,value in protocol['source_sha256'].items():
        assert digest(output/'runner'/name)==value,name
    assert digest(Path(protocol['evaluator_cache']['path']))==protocol['evaluator_cache']['sha256']
    classifier=classifier_provenance(protocol['classifier_run'])
    assert classifier==protocol['auxiliary_classifier']
    frozen_classifier_model(protocol['classifier_run'])  # Verify the original evaluator source independently.
    claims=json.loads((output/'screen/hardware_claims.json').read_text())
    assert set(claims)=={job['service'] for job in screen['jobs']}
    worker_log=(output/'screen/supervisord.log').read_text()
    paired={};rows=[]
    for job in screen['jobs']:
        folder=Path(job['output_dir'])
        cfg=json.loads((folder/'config.json').read_text())
        expected_config=make_config(job['kind'],job['family'],job['seed'],folder,protocol['classifier_run'])
        assert cfg==asdict(expected_config)
        assert json.loads(Path(job['config']).read_text())==cfg
        assert cfg['train_limit'] is cfg['val_limit'] is None
        result=json.loads((folder/'results.json').read_text())
        events=read_events(folder/'metrics.jsonl')
        provenance=json.loads((folder/'provenance.json').read_text())
        for name,value in provenance['source_sha256'].items():
            assert value==protocol['source_sha256'][name],(job['name'],name)
        data=json.loads((folder/'dataset.json').read_text())
        assert data['train_examples']==45000 and data['validation_examples']==5000 and not data['test_evaluated']
        assert len(set(data['train_indices']))==45000 and len(set(data['validation_indices']))==5000
        assert not set(data['train_indices']) & set(data['validation_indices'])
        train=[event for event in events if event['event']=='train']
        loss_key='noise_mse' if job['kind']=='diffusion' else 'loss'
        assert train and all(math.isfinite(event[loss_key]) for event in train)
        torch.manual_seed(job['seed'])
        if job['kind']=='diffusion':
            model=make_model(DiTConfig(**cfg));stats=model.statistics()
            cycles,tail=divmod(5000,math.ceil(45000/cfg['batch_size']))
            assert result['completed_steps']==5000 and result['processed_images']==cycles*45000+tail*cfg['batch_size']
            assert cfg['lr_schedule_steps']==30000 and cfg['warmup_steps']==50
            diag=result['design_diagnostics']
            assert diag['base_initialization_sha256']==digest_parameters(model)
            assert diag['adapter_initialization_sha256']==digest_parameters(model,True)
            diagnostic=[event for event in events if event['event']=='design_diagnostic']
            assert diag['diagnostics']==diagnostic
            tracekeys=('step','batch_images_sha256','batch_labels_sha256','noise_sha256','timesteps_sha256','noise_rng_state_sha256')
            traces=[{key:event[key] for key in tracekeys} for event in diagnostic[1:]]
            gradients=diag['adapter_first_nonzero_gradient_observed']
            names={name for name,_ in model.named_parameters() if '.adapter.' in name}
            assert set(gradients)==names and all(value<=10 for value in gradients.values())
            assert all(branch['delta_rms']==0 for branch in diagnostic[0]['branches'])
            evaluation=result['generative_evaluation']
            assert evaluation['model_state']=='EMA'
            assert evaluation['fid_fake_images']==evaluation['fid_real_images']==5000
            assert math.isfinite(evaluation['fid']) and evaluation['class_agreement']['classifier']==classifier
            for label in ('real','generated'):
                metric=evaluation['class_agreement'][label]
                matrix=metric['confusion_rows_requested_columns_predicted']
                assert metric['examples']==5000 and metric['class_counts']==[500]*10
                assert [sum(line) for line in matrix]==[500]*10
                assert math.isclose(metric['agreement'],sum(matrix[i][i] for i in range(10))/5000,abs_tol=1e-7)
            uuid=provenance['device_uuid']
            adapter_key='adapter_parameters'
        else:
            model=build_model(ViTConfig(**cfg));stats=model_stats(model,ViTConfig(**cfg))
            assert result['completed_steps']==3520 and result['processed_images']==450000
            assert result['test'] is None and not cfg['save_weights']
            validation=[event for event in events if event['event']=='validation']
            assert len(validation)==10 and all(event['examples']==5000 for event in validation)
            assert result['validation']=={key:validation[-1][key] for key in result['validation']}
            diag=result['token_diagnostics']
            assert diag['base_initialization_sha256']==parameter_digest(model)
            assert diag['adapter_initialization_sha256']==parameter_digest(model,True)
            diagnostic=[event for event in events if event['event']=='token_diagnostic']
            assert diag['diagnostics']==diagnostic
            traces=[{key:event[key] for key in ('step','batch_images_sha256','batch_labels_sha256')} for event in diagnostic[1:]]
            gradients=diag['first_nonzero_gradient']
            from train_cifar10 import hyperparameter_ids
            ids=hyperparameter_ids(model)
            names={name for name,value in model.named_parameters() if id(value) in ids}
            assert set(gradients)==names and all(value<=10 for value in gradients.values())
            assert all(branch['total_adapter_rms']==0 for branch in diagnostic[0]['live_branches'])
            uuid=diag['hardware']['device_uuid']
            adapter_key='extra_parameters'
        assert result['parameters']==stats['parameters'] and result[adapter_key]==stats[adapter_key]
        assert [event['step'] for event in diagnostic]==[int(step) for step in cfg['diagnostic_steps'].split(',')]
        claim=claims[job['service']]
        assert claim['owner']=='adalora' and claim['job']==job['service'] and claim['locker']=='gpu-claim'
        assert claim['uuid'].removeprefix('GPU-')==uuid.removeprefix('GPU-')
        assert f'exited: {job["service"]} (exit status 0; expected)' in worker_log
        key=(job['kind'],job['seed'])
        paired.setdefault(key,[]).append(dict(family=job['family'],base=diag['base_initialization_sha256'],
                                             adapter=diag['adapter_initialization_sha256'],mlp_ratio=cfg['mlp_ratio'],
                                             traces=traces,split=(data['train_indices'],data['validation_indices'])))
        rows.append({**job,'config_path':job['config'],'config':cfg,'results':result})
    for (kind,seed),group in paired.items():
        assert all(row['traces']==group[0]['traces'] and row['split']==group[0]['split'] for row in group)
        assert len({row['base'] for row in group if row['mlp_ratio']==4.})==1
        family={row['family']:row for row in group}
        a,b=('r8_s16','r8_s16_raw') if kind=='diffusion' else ('r8_s12','r8_s12_detached')
        assert family[a]['adapter']==family[b]['adapter']
    assert 'exited: adalora_rank_width_controller (exit status 0; expected)' in (output/'controller.log').read_text()
    assert not list(output.rglob('*.sock'))
    assert not [path for path in output.rglob('*') if path.suffix in ('.pt','.pth','.ckpt','.safetensors')]
    retained=sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    assert retained<protocol['expected_storage_upper_bound_bytes']
    progress=[json.loads(line) for line in (output/'adalora_rank_width_controller.log').read_text().splitlines() if line.startswith('{')]
    assert all(event['submitted']-event['completed']<=6 for event in progress if event.get('event')=='screen_progress')
    recovery=None
    if protocol.get('previous_failed_attempt'):
        previous=Path(protocol['previous_failed_attempt'])
        failed=json.loads((previous/'protocol.json').read_text())
        assert failed['state']=='failed' and failed['failed_workers']==6
        assert not list(previous.rglob('*.sock'))
        assert not list(previous.rglob('results.json'))
        for name,value in failed['source_sha256'].items():
            assert digest(previous/'runner'/name)==value
        old_jobs=json.loads((previous/'screen/screen.json').read_text())['jobs']
        current={row['name']:row for row in rows}
        repeated=[]
        failed_claim_seconds=0.0
        old_claims=json.loads((previous/'screen/hardware_claims.json').read_text())
        old_log=(previous/'screen/supervisord.log').read_text()
        for job in old_jobs:
            folder=Path(job['output_dir'])
            if not (folder/'metrics.jsonl').exists():continue
            old_config=json.loads((folder/'config.json').read_text())
            old_config['output_dir']=current[job['name']]['config']['output_dir']
            assert old_config==current[job['name']]['config']
            old_events=read_events(folder/'metrics.jsonl')
            assert max(event.get('step',0) for event in old_events)==5000
            old_diag=[event for event in old_events if event['event']=='design_diagnostic']
            new_diag=current[job['name']]['results']['design_diagnostics']['diagnostics']
            keys=('step','batch_images_sha256','batch_labels_sha256','noise_sha256','timesteps_sha256','noise_rng_state_sha256')
            assert [{k:e[k] for k in keys} for e in old_diag[1:]]==[{k:e[k] for k in keys} for e in new_diag[1:]]
            repeated.append(job['name'])
            exited=re.search(r'([\d-]+ [\d:]+,\d+) WARN exited: '+re.escape(job['service'])+r' \(exit status 1; not expected\)',old_log)
            assert exited
            finish=datetime.strptime(exited[1],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc)
            started=datetime.fromisoformat(old_claims[job['service']]['started'].replace('Z','+00:00'))
            failed_claim_seconds+=(finish-started).total_seconds()
        assert len(repeated)==6
        recovery=dict(failed_attempt=str(previous),repeated_training_cases=repeated,
                      config_batch_noise_traces_preserved=True,failed_evaluation_included_in_quality=False,
                      failed_claimed_wall_gpu_hours=failed_claim_seconds/3600)
    checks=('source_config_budget_reconstructed','initialization_reproduced','paired_batch_noise_traces',
            'common_split_equal_shape_base','raw_scaled_detached_adapter_init','zero_init_gradient_wakeup',
            'frozen_evaluators_denominators','claimed_hardware_runtime','successful_exits_supervisors_closed',
            'no_weights_test_evaluation_speed_study','storage_within_bound')
    result=dict(checks={key:True for key in checks},rows=64,bytes_retained=retained)
    if recovery:result['recovery']=recovery
    write_json(output/'audit.json',result)
    write_json(output/'evidence.json',dict(protocol=protocol,rows=rows,audit=result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(json.dumps(audit(parser.parse_args().output),indent=2))

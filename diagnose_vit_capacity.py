#!/usr/bin/env python3
"""Read audited curves and probe untrained MLP gains without modifying any training."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import torch
from train_cifar10 import Config, build_model, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;reports=root/'reports';torch.set_num_threads(2)
    evidence_path=reports/'vit-split-lr-evidence.json';audit=json.loads((reports/'vit-split-lr-audit.json').read_text())
    assert all(audit['checks'].values()) and digest(evidence_path)==audit['report_sha256']
    evidence=json.loads(evidence_path.read_text());rows=evidence['reused_quality']+evidence['replacement_quality']
    frozen=json.loads((reports/'vit-split-lr-protocol.json').read_text())['source_sha256']
    assert all(digest(root/name)==frozen[name]for name in ('train_cifar10.py','vit.py','conditioning.py','condlr.py'))
    curves=[]
    for row in sorted([r for r in rows if r['family']in ('plain','static_capacity')],key=lambda r:(r['family'],r['seed'])):
        path=Path(row['output_dir'])/'metrics.jsonl';assert digest(path)==row['evidence_sha256']['metrics.jsonl']
        events=[json.loads(line)for line in path.read_text().splitlines()]
        train=[e for e in events if e['event']=='train' and e['step']>29*352];images=sum(e['window_images']for e in train)
        assert images==45000
        curves.append(dict(family=row['family'],seed=row['seed'],hidden_width=int(row['config']['width']*row['config']['mlp_ratio']),
                           final_augmented_train_accuracy_percent=100*sum(e['accuracy']*e['window_images']for e in train)/images,
                           final_augmented_train_smoothed_loss=sum(e['loss']*e['window_images']for e in train)/images,
                           final_validation_accuracy_percent=100*row['results']['validation']['accuracy'],
                           final_validation_loss=row['results']['validation']['loss'],metrics_sha256=digest(path)))
    probes=[]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1200);h=torch.nn.functional.layer_norm(torch.randn(2,65,192),(192,))
        for seed in (102,103):
            cfg=Config(width=192,depth=6,heads=3,variant='plain',mlp_ratio=4.)
            torch.manual_seed(seed);base=build_model(cfg)
            torch.manual_seed(seed);wide=build_model(replace(cfg,mlp_ratio=1483/192))
            blocks=[]
            with torch.no_grad():
                for index,(a,b)in enumerate(zip(base.blocks,wide.blocks)):
                    baseline=a.mlp(h).square().mean().sqrt().item();before=b.mlp(h).square().mean().sqrt().item()
                    b.mlp.fc2.weight.mul_(math.sqrt(768/1483));after=b.mlp(h).square().mean().sqrt().item()
                    blocks.append(dict(block=index,plain_rms=baseline,wide_rms=before,wide_plain_ratio=before/baseline,
                                       wide_fc2_rescaled_rms=after,rescaled_plain_ratio=after/baseline))
            probes.append(dict(seed=seed,blocks=blocks))
    assert not torch.cuda.is_initialized()
    result=dict(training_summary=curves,initialization_probe=dict(dtype='FP32 CPU',shape=[2,65,192],probe_seed=1200,
        input='Identical normalized random tokens; untrained isolated MLP branches',hidden_widths=[768,1483],linear_init_std=.02,
        expected_initial_rms_ratio=math.sqrt(1483/768),temporary_fc2_weight_multiplier=math.sqrt(768/1483),
        mean_observed_ratio=sum(b['wide_plain_ratio']for p in probes for b in p['blocks'])/12,
        mean_rescaled_ratio=sum(b['rescaled_plain_ratio']for p in probes for b in p['blocks'])/12,per_seed=probes),
        provenance=dict(evidence_sha256=digest(evidence_path),diagnostic_source_sha256=digest(Path(__file__)),
                        inspected_model_source_sha256={n:frozen[n]for n in ('train_cifar10.py','vit.py','conditioning.py','condlr.py')},
                        torch=torch.__version__,cuda_initialized=False,new_gpu_jobs=0,training_sources_modified=False),
        limits=['Training metrics use augmentation and label smoothing; validation loss is unsmoothed. Do not compare train and validation loss directly.',
                'Initial RMS uses synthetic normalized inputs, not trained hidden states or real images.',
                'Temporary FC2 rescaling only checks initial magnitude; it was not used in any training run and proves no quality improvement.',
                'Two seeds and an untuned common optimizer recipe; the scaling difference is a plausible confound, not an established cause.'])
    path=reports/'vit-capacity-diagnostic.json';write_json(path,result);print(path)


if __name__=='__main__':main()

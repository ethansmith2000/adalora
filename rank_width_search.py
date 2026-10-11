#!/usr/bin/env python3
"""Bounded two-seed broadcast rank/width search and token-conditioned ViT pilot."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from class_fidelity import classifier_provenance
from dit_followup import private_supervisor
from train_cifar10 import Config as ViTConfig, load_config as load_vit
from train_dit_cifar10 import Config as DiTConfig, load_config as load_dit, write_json

SEEDS = (58, 59)
RANKS = (2, 4, 8, 16)
WIDTHS = (6, 12, 24, 48)
DIFFUSION = {f'r{r}_s{s}': dict(variant='dplr', rank=r, n_basis=s, adapter_gain=1/math.sqrt(r))
             for r in RANKS for s in WIDTHS}
DIFFUSION.update({
    'r2_s16': dict(variant='dplr',rank=2,n_basis=16,adapter_gain=1/math.sqrt(2)),
    'r8_s16': dict(variant='dplr',rank=8,n_basis=16,adapter_gain=1/math.sqrt(8)),
    'r8_s16_raw': dict(variant='dplr',rank=8,n_basis=16,adapter_gain=1.),
    'adaln': dict(variant='adaln_zero'),
    'ffn_r2_budget': dict(variant='adaln_zero',mlp_ratio=4.5),
    'ffn_rs96_budget': dict(variant='adaln_zero',mlp_ratio=5.3125),
})
VIT = {
    'plain': dict(variant='plain'),
    'diagonal': dict(variant='diagonal'),
    'static_r8_diagonal': dict(variant='static_lora_diagonal',rank=8,adapter_gain=1/math.sqrt(8)),
    **{f'r{r}_s{s}':dict(variant='dplr',rank=r,n_basis=s,adapter_gain=1/math.sqrt(r))
       for r,s in ((2,16),(8,6),(8,12),(8,16),(16,6))},
    'r8_s12_detached': dict(variant='dplr',rank=8,n_basis=12,adapter_gain=1/math.sqrt(8),detach_condition=True),
    'ffn_r8_s12_budget': dict(variant='plain',mlp_ratio=6.3125),
}
SOURCE = ('rank_width_search.py','audit_rank_width_search.py','summarize_rank_width_search.py',
          'test_rank_width_search.py','vit_quality_diagnostics.py','train_cifar10.py','vit.py','conditioning.py',
          'condlr.py','train_dit_cifar10.py','dit_cifar10.py','dit_quality_diagnostics.py',
          'benchmark_dit_design.py','benchmark_dit_gain.py','class_fidelity.py','dit_followup.py',
          'summarize_dit_cifar10_screen.py','test_conditioning.py','test_dit_cifar10.py','test_dit_quality.py','requirements.txt')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(kind, family, seed, output, classifier=None):
    if kind == 'diffusion':
        return DiTConfig(**DIFFUSION[family],width=192,depth=6,heads=3,patch_size=4,
                         seed=seed,max_steps=5000,lr_schedule_steps=30000,warmup_steps=50,
                         lr=4e-4,eval_every=1000,log_every=250,cpu_threads=2,num_workers=2,
                         sample_images=0,fid_images=5000,classifier_run=classifier,
                         diagnostic_steps='0,1,3,10,50,100,250,500,1000,2500,5000',output_dir=str(output))
    return ViTConfig(**VIT[family],width=192,depth=6,heads=3,patch_size=4,
                     seed=seed,epochs=10,lr=5e-4,log_every=100,cpu_threads=2,num_workers=2,
                     diagnostic_steps='0,1,3,10,50,250,500,1000,2000,3520',output_dir=str(output))


def prepare(output, launch):
    root = Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError(output)
    free = shutil.disk_usage(root).free
    if free < 1024**3:
        raise RuntimeError('Require 1 GiB free for compact evidence')
    classifier = str(root/'runs/dit-followup-20261007T174446Z/classifier/classifier_config')
    output.mkdir(parents=True)
    runner = output/'runner'; runner.mkdir()
    for name in SOURCE:
        shutil.copy2(root/name,runner/name)
    fid = Path(DiTConfig().fid_weights)
    protocol = dict(state='prepared',created_utc=now(),repo_root=str(root),
                    purpose='joint rank/generator-width trends with fresh two-seed broadcast diffusion and per-token ViT pilot',
                    source_sha256={name:digest(runner/name) for name in SOURCE},seeds=list(SEEDS),
                    ranks=list(RANKS),generator_widths=list(WIDTHS),condition_width=192,
                    families=dict(diffusion=DIFFUSION,vit=VIT),runs=dict(diffusion=44,vit=20),
                    diffusion_steps=5000,diffusion_schedule_horizon=30000,diffusion_warmup=50,
                    vit_epochs=10,vit_steps=3520,train_examples=45000,validation_examples=5000,
                    split_seed=123,max_concurrent_workers=6,order_seed=91,
                    classifier_run=classifier,auxiliary_classifier=classifier_provenance(classifier),
                    evaluator_cache=dict(path=str(fid),bytes=fid.stat().st_size,sha256=digest(fid)),
                    disk_free_bytes_before_launch=free,expected_storage_upper_bound_bytes=512*1024**2,
                    estimated_gpu_hours='4–7 aggregate including validation, diagnostics and final diffusion evaluations',
                    weights_saved=False,periodic_checkpoints=False,test_evaluated=False,
                    dedicated_speed_benchmarks=False,compile=False,confirmation_launched=False,
                    retention='compact frozen source/configs/seeds/splits/metrics/logs/audits/plots until explicitly removed; no grids or model/evaluator weights',
                    controls='rank-2/rank-8 anchors, raw/scaled r8, FFN budgets; ViT diagonal/static/detached controls; identical base initialization within equal shapes',
                    limits=['Two seeds and short budgets describe coarse learning trends, not asymptotic quality or significance.',
                            'Diffusion full-loader 5k prefix uses 30k LR horizon; ViT uses a complete 10-epoch cosine schedule. Regimes are not pooled.',
                            'Fixed rank gain 1/sqrt(r), default generator initialization and LR multipliers 1; no update or quality tuning in this screen.',
                            'Generator width can change signal/update scale; diagnostics expose it, but no variance preservation or muP guarantee.',
                            'Constant r*s matches decoder weights only; encoder/bias and total budgets differ.',
                            'ViT keeps static LayerNorm affine parameters in every control; DiT keeps non-affine norms and common AdaLN-Zero.',
                            'ViT uses activation mixing with c=h (or detached h); DiT uses broadcast class/time weight mixing.',
                            'Diffusion FID uses 5k images and one sampling seed; ViT evaluates only the held-out train split.',
                            'Concurrent ordinary training telemetry is not an isolated speed comparison.',
                            'No external sequence retrieval, LM integration, nonlinear generators or structured operators are tested here.'])
    write_json(output/'protocol.json',protocol)
    if launch:
        private_supervisor(output,[('adalora_rank_width_controller',
                            [sys.executable,'-u',str(runner/Path(__file__).name),'run','--output',str(output)])],'controller',True)
    print(json.dumps(dict(output=str(output),launched=launch,runs=64)),flush=True)


def run(output):
    protocol = json.loads((output/'protocol.json').read_text())
    protocol.update(state='running',started_utc=now());write_json(output/'protocol.json',protocol)
    stage=output/'screen';stage.mkdir()
    jobs=[];programs=[]
    configs=stage/'configs';configs.mkdir()
    for kind,families in protocol['families'].items():
        for family in families:
            for seed in SEEDS:
                name=f'{kind}_{family}_seed{seed}'
                cfg=make_config(kind,family,seed,stage/name,protocol['classifier_run'])
                path=configs/(name+'.json');write_json(path,asdict(cfg))
                (load_dit if kind=='diffusion' else load_vit)(['--override_json',str(path)])
                service=f'adalora_{output.name.replace("-","_")}_{name}'
                command=['/workspace/bin/gpu-claim','run','--owner','adalora','--job',service,'--wait','--',
                         sys.executable,'-u',str(output/'runner'/('train_dit_cifar10.py' if kind=='diffusion' else 'train_cifar10.py')),
                         '--override_json',str(path)]
                jobs.append(dict(name=name,kind=kind,family=family,seed=seed,config=str(path),output_dir=cfg.output_dir,service=service))
                programs.append((service,command))
    random.Random(protocol['order_seed']).shuffle(jobs)
    write_json(stage/'screen.json',dict(created_utc=now(),jobs=jobs,source_sha256=protocol['source_sha256']))
    cap=protocol['max_concurrent_workers']
    first=[job['service'] for job in jobs[:cap]]
    submitted=set(first)
    control=private_supervisor(stage,programs,start_programs=first)
    claims={}
    while True:
        for path in Path('/tmp/vast_gpu_claims').glob('*.claim'):
            try:claim=json.loads(path.read_text())
            except (OSError,ValueError):continue
            if claim.get('job') in submitted:claims[claim['job']]=claim
        write_json(stage/'hardware_claims.json',claims)
        status=subprocess.run(['supervisorctl','-c',control,'status'],capture_output=True,text=True).stdout
        if not status.strip() or any(word in status for word in ('FATAL','BACKOFF')):
            raise RuntimeError(status or 'No worker supervisor status')
        log=(stage/'supervisord.log').read_text()
        completed={job['service'] for job in jobs if f'exited: {job["service"]} (exit status 0; expected)' in log}
        for line in status.splitlines():
            if line.split()[0] in submitted and 'EXITED' in line and line.split()[0] not in completed:
                raise RuntimeError('Failed worker: '+line)
        if any(not (Path(job['output_dir'])/'results.json').exists() for job in jobs if job['service'] in completed):
            raise RuntimeError('Successful worker without results')
        pending=[job for job in jobs if job['service'] not in submitted]
        for job in pending[:cap-len(submitted-completed)]:
            subprocess.run(['supervisorctl','-c',control,'start',job['service']],check=True)
            submitted.add(job['service'])
        progress=dict(event='screen_progress',completed=len(completed),submitted=len(submitted),total=len(jobs),utc=now())
        write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if len(completed)==len(jobs):break
        time.sleep(20)
    subprocess.run(['supervisorctl','-c',control,'shutdown'],check=True)
    protocol.update(state='complete',completed_utc=now());write_json(output/'protocol.json',protocol)
    print(json.dumps(dict(event='suite_complete',output=str(output))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    output=(args.output or Path(__file__).resolve().parent/'runs'/('rank-width-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    prepare(output,args.launch) if args.action=='prepare' else run(output)

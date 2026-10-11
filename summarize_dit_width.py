#!/usr/bin/env python3
"""Audited quality, paired gain/rate differences and update/magnitude trajectories."""

import argparse
import json
from pathlib import Path
import shutil
import statistics

from audit_dit_width import audit
from dit_width_calibration import CAL_FAMILIES, CONFIRM_SEEDS, FAMILIES, digest, update_metric
from train_dit_cifar10 import write_json

LABELS={'r8_s16':'r8/s16 anchor', 'r8_s12_default':'r8/s12 default',
        'r8_s12_lr':'r8/s12 matched LR', 'r8_s12_gain_lr':'r8/s12 gain + matched LR',
        'r16_s6_default':'r16/s6 default', 'r16_s6_lr':'r16/s6 matched LR',
        'r16_s6_gain_lr':'r16/s6 gain + matched LR'}
COLORS=dict(zip(FAMILIES,('#64748b','#0891b2','#2563eb','#7c3aed','#f59e0b','#ea580c','#db2777')))


def metric(values):
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),values=list(values))


def event(row,step):
    return next(value for value in row['design_diagnostics']['diagnostics'] if value['step']==step)


def step_update(row,step):
    return statistics.mean(block['update_rms'] for block in event(row,step)['fixed_token_condition_function_updates'])


def live_ratio(row,step):
    return statistics.mean(block['delta_to_base_rms'] for block in event(row,step)['branches'])


def summarize(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve()
    audited=audit(output)
    report=json.loads((output/'report.json').read_text());rows=report['confirmation']['rows']
    keyed={(row['family'],row['config']['seed']):row for row in rows}
    aggregates=[];paired={}
    for family in FAMILIES:
        group=[keyed[family,seed] for seed in CONFIRM_SEEDS]
        values=dict(fid=[row['generative_evaluation']['fid'] for row in group],
                    agreement_percent=[100*row['generative_evaluation']['class_agreement']['generated']['agreement'] for row in group],
                    ema_noise_mse=[row['ema_validation']['noise_mse'] for row in group],
                    peak_allocated_mib=[row['peak_allocated_bytes']/2**20 for row in group],
                    early_update_to_paired_anchor=[update_metric(row)/update_metric(keyed['r8_s16',seed]) for seed,row in zip(CONFIRM_SEEDS,group)],
                    final_update_to_paired_anchor=[step_update(row,30000)/step_update(keyed['r8_s16',seed],30000) for seed,row in zip(CONFIRM_SEEDS,group)],
                    final_live_delta_to_base=[live_ratio(row,30000) for row in group],
                    final_combined_to_base=[statistics.mean(block['combined_to_base_rms'] for block in event(row,30000)['branches']) for row in group])
        aggregates.append(dict(family=family,label=LABELS[family],parameters=group[0]['parameters'],
                               adapter_parameters=group[0]['adapter_parameters'],gain=group[0]['config']['adapter_gain'],
                               adapter_lr_multiplier=group[0]['config']['adapter_lr_multiplier'],
                               metrics={name:metric(value) for name,value in values.items()}))
        paired[family]={}
        for control in FAMILIES:
            if control==family:continue
            paired[family][control]=dict(
                fid=metric([keyed[family,seed]['generative_evaluation']['fid']-keyed[control,seed]['generative_evaluation']['fid'] for seed in CONFIRM_SEEDS]),
                agreement_pp=metric([100*(keyed[family,seed]['generative_evaluation']['class_agreement']['generated']['agreement']-keyed[control,seed]['generative_evaluation']['class_agreement']['generated']['agreement']) for seed in CONFIRM_SEEDS]))
    calibration_hours=sum(row['loop_seconds'] for summary in report['calibration']['rounds'] for row in summary['rows'])/3600
    hours=sum(row['loop_seconds'] for row in rows)/3600
    eval_hours=sum(row['generative_evaluation']['seconds'] for row in rows)/3600
    evidence=dict(raw_workflow=str(output),audit=audited,aggregates=aggregates,paired_differences=paired,
                  calibration=report['calibration']['selected'],limits=report['limits'],quality_winner_claimed=False,
                  gpu_hours=dict(calibration=calibration_hours,confirmation=hours,final_evaluation=eval_hours))
    prefix='dit-width-calibration';destination.mkdir(parents=True,exist_ok=True)
    if list(destination.glob(prefix+'*')):raise FileExistsError('Preserve existing exports')
    write_json(destination/(prefix+'.json'),evidence)
    shutil.copy2(output/'audit.json',destination/(prefix+'-audit.json'))
    shutil.copy2(output/'report.json',destination/(prefix+'-evidence.json'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,key,title in zip(axes,('fid','agreement_percent'),('FID (lower is better)','Requested-class agreement (%)')):
        for index,item in enumerate(aggregates):
            value=item['metrics'][key]
            ax.bar(index,value['mean'],yerr=value['sd'],capsize=4,color=COLORS[item['family']],alpha=.7)
            ax.scatter(index+np.array([-.12,0,.12]),value['values'],c='black',s=19,zorder=3)
        ax.set_xticks(range(len(aggregates)),[item['label'] for item in aggregates],rotation=25,ha='right')
        ax.set_title(title);ax.grid(axis='y',alpha=.15)
    fig.suptitle('30,000 steps · three fresh paired seeds · mean ± sample SD')
    for suffix in ('png','svg'):fig.savefig(destination/(prefix+'.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for family in FAMILIES:
        steps=[value['step'] for value in keyed[family,CONFIRM_SEEDS[0]]['design_diagnostics']['diagnostics'] if value['step']>=10]
        for seed in CONFIRM_SEEDS:
            row,anchor=keyed[family,seed],keyed['r8_s16',seed]
            axes[0].plot(steps,[step_update(row,step)/step_update(anchor,step) for step in steps],color=COLORS[family],alpha=.18,lw=1)
            axes[1].plot(steps,[live_ratio(row,step) for step in steps],color=COLORS[family],alpha=.18,lw=1)
        axes[0].plot(steps,[statistics.mean(step_update(keyed[family,seed],step)/step_update(keyed['r8_s16',seed],step) for seed in CONFIRM_SEEDS) for step in steps],color=COLORS[family],label=LABELS[family],lw=2)
        axes[1].plot(steps,[statistics.mean(live_ratio(keyed[family,seed],step) for seed in CONFIRM_SEEDS) for step in steps],color=COLORS[family],label=LABELS[family],lw=2)
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('Training step');ax.grid(alpha=.15)
    axes[0].axhline(1,color='black',ls='--',lw=.7);axes[0].set_title('Single-step adapter update / paired anchor')
    axes[1].set_title('Live delta/base RMS (raw model)');axes[1].legend(fontsize=8)
    fig.suptitle('Mean trajectories and individual seeds · highest-FP32 fixed probes')
    for suffix in ('png','svg'):fig.savefig(destination/(prefix+'-updates.'+suffix),dpi=180)
    plt.close(fig)
    def cell(item,key,places=2):
        value=item['metrics'][key];return f"{value['mean']:.{places}f} ± {value['sd']:.{places}f}"
    lines=['# Gain and LR sensitivity for narrower conditional generators','',
           'Three fresh paired training seeds (62/63/64), 30,000 steps per recipe. Values are mean ± sample SD, not confidence intervals.','',
           'Two gain parameterizations are tested for each selected rank/width. The default is 1/sqrt(r); the alternate multiplies it by 16/s. This is an empirical width-compensation heuristic. Default-rate controls use adapter LR multiplier 1. The matched-rate recipes use independent calibration seeds 60/61, with rates selected only from function updates.','',
           '| Recipe | Adapter parameters | Gain | Adapter LR multiplier | FID ↓ | Agreement ↑ | EMA noise MSE |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in aggregates:
        lines.append(f"| {item['label']} | {item['adapter_parameters']:,} | {item['gain']:.6f} | {item['adapter_lr_multiplier']:.6f} | {cell(item,'fid')} | {cell(item,'agreement_percent')}% | {cell(item,'ema_noise_mse',5)} |")
    lines+=['',f'![Quality comparison]({prefix}.png)','',
            'All recipes use width 192/depth 6/three heads/patch 4, identical base initialization per seed, non-affine pre-norm/AdaLN-Zero and the normalized adapter input. Gain/rate recipes of one rank/width also share adapter initialization and parameter count. r8/s12 and r16/s6 have the same dominant decoder weight budget (r*s=96), with different encoder/bias costs. The existing class/time condition is broadcast; self-attention is retained.','',
            '## Calibration and drift','',
            'The independent metric averages highest-FP32 single-step adapter-delta updates across six blocks, steps 50/100/250/500 and two seeds on the full 45k loader. Calibration runs a 500-step prefix of the exact 30k schedule. One correction is always verified for each non-anchor family, then at most two further corrections if its most recent candidate misses ±15%. Selection minimizes absolute log(update/target) among observed rates, with no quality metrics. Recipes freeze before confirmation.','',
            '| Recipe | Selected multiplier | Calibration update / target | Selected attempt |','| --- | ---: | ---: | ---: |']
    for family in CAL_FAMILIES:
        value=report['calibration']['selected'][family]
        lines.append(f"| {LABELS[family]} | {value['adapter_lr_multiplier']:.6f} | {value['metric_to_target']:.3f} | {value['attempt']} |")
    lines+=['','| Recipe | Early update / paired anchor | Final update / paired anchor | Final live delta/base | Whole-training peak MiB |',
            '| --- | ---: | ---: | ---: | ---: |']
    for item in aggregates:
        lines.append(f"| {item['label']} | {cell(item,'early_update_to_paired_anchor',3)} | {cell(item,'final_update_to_paired_anchor',3)} | {cell(item,'final_live_delta_to_base',3)} | {cell(item,'peak_allocated_mib',1)} |")
    outside=[f"{LABELS[item['family']]} seed {seed}: {value:.3f}×" for item in aggregates if item['family'] in CAL_FAMILIES and item['family']!='r8_s16'
             for seed,value in zip(CONFIRM_SEEDS,item['metrics']['early_update_to_paired_anchor']['values']) if abs(value-1)>.15]
    lines+=['','Fresh-seed early ratios outside the calibration tolerance: '+('; '.join(outside) if outside else 'none')+'. Rates were not changed in response.','',
            f'![Update and magnitude trajectories]({prefix}-updates.png)','',
            'Fixed probes measure only the raw-model adapter delta at independent normalized inputs/conditions, not the entire-network update. Live branch ratios use eight fixed held-out images averaged across blocks. EMA quality measures a different state. Scalar matching does not fix gauges or ensure matched later updates, per-block behavior or variance preservation. Whole-training peaks include model, EMA, gradients, optimizer, activations and fixed diagnostic buffers; diagnostic forwards and final FID are excluded.','',
            '## Paired comparisons','',
            'Differences are variant minus control in seed order 62/63/64. Negative FID and positive agreement favor the variant.','',
            '| Variant versus control | FID differences | Agreement differences (pp) |','| --- | --- | --- |']
    comparisons=[]
    for prefix_name in ('r8_s12','r16_s6'):
        comparisons.extend([(prefix_name+'_lr',prefix_name+'_default'),(prefix_name+'_gain_lr',prefix_name+'_lr')])
    comparisons.extend((family,'r8_s16') for family in FAMILIES if family!='r8_s16')
    for family,control in comparisons:
        values=paired[family][control]
        lines.append(f"| {LABELS[family]} vs {LABELS[control]} | {', '.join(f'{x:+.2f}' for x in values['fid']['values'])} | {', '.join(f'{x:+.2f}' for x in values['agreement_pp']['values'])} |")
    lines+=['','## Protocol and limits','',
            'Pixel-space epsilon prediction, full 45k/5k CIFAR-10 split, base LR 4e-4, warmup 50, cosine floor .01, AdamW (.9,.999), zero weight decay, clipping 1, BF16, in-memory EMA .999. Each final evaluation uses 5,000 real and generated images, DDIM50 without CFG, sampling seed 2028 and the unchanged frozen auxiliary classifier. No test evaluation or saved model weights.','']
    lines.extend('- '+value for value in report['limits'])
    lines+=['',f'Calibration loops: {calibration_hours:.3f} aggregate GPU-hours; confirmation loops: {hours:.3f}; final evaluation: {eval_hours:.3f}; startup additional. All private supervisors are closed and workflow GPU claims released. Raw evidence audited before export: {audited["bytes_retained"]/2**20:.1f} MiB under 512 MiB. No checkpoints, model/evaluator weights, image grids or generated datasets saved. Dedicated speed benchmarks stay deferred.','',
            f'[Structured results]({prefix}.json) · [Full evidence]({prefix}-evidence.json) · [Audit]({prefix}-audit.json) · [Design notes](../DESIGN_NOTES.md)','']
    (destination/(prefix+'.md')).write_text('\n'.join(lines))
    analysis=output/'analysis_source';analysis.mkdir(exist_ok=True)
    for name in ('audit_dit_width.py','summarize_dit_width.py'):
        shutil.copy2(Path(__file__).with_name(name),analysis/name)
    artifacts={path.name:dict(sha256=digest(path),bytes=path.stat().st_size) for path in destination.glob(prefix+'*')}
    write_json(destination/(prefix+'-manifest.json'),dict(workflow=str(output),artifacts=artifacts,
               analysis_source_sha256={path.name:digest(path) for path in analysis.iterdir()}))
    return dict(report=str(destination/(prefix+'.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--destination',type=Path);args=parser.parse_args()
    root=Path(json.loads((args.output/'protocol.json').read_text())['repo_root'])
    print(json.dumps(summarize(args.output,args.destination or root/'reports')))

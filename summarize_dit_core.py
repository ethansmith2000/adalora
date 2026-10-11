#!/usr/bin/env python3
"""Audited core/FFN quality comparison, paired differences and update trajectories."""

import argparse
import json
from pathlib import Path
import shutil
import statistics

from audit_dit_core import audit
from dit_core_confirmation import CONFIRM_SEEDS,FAMILIES,OBSERVATIONS,digest,update_metric
from train_dit_cifar10 import write_json

ORDER=tuple(FAMILIES)
LABELS={'adaln':'AdaLN 192','ffn_r2_budget':'FFN budget control','basis_r2':'Basis r2',
        'basis_r8':'Basis r8 / sqrt(8)','basis_r8_raw':'Raw basis r8',
        'core_r8':'Dense core r8','dplr_core_r8_p2':'DPLR core r8/p2'}
COLORS=dict(zip(ORDER,('#64748b','#475569','#0891b2','#7c3aed','#ea580c','#16a34a','#db2777')))


def mean_sd(values):
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
    report=json.loads((output/'report.json').read_text())
    rows=report['confirmation']['rows']
    keyed={(row['family'],row['config']['seed']):row for row in rows}
    base=keyed['adaln',CONFIRM_SEEDS[0]]['parameters']
    aggregates=[];paired={}
    for family in ORDER:
        group=[keyed[family,seed] for seed in CONFIRM_SEEDS]
        metrics={'fid':[row['generative_evaluation']['fid'] for row in group],
                 'agreement_percent':[100*row['generative_evaluation']['class_agreement']['generated']['agreement'] for row in group],
                 'ema_noise_mse':[row['ema_validation']['noise_mse'] for row in group],
                 'train_images_per_second':[row['mean_train_images_per_second_excluding_warmup'] for row in group],
                 'peak_allocated_mib':[row['peak_allocated_bytes']/2**20 for row in group],
                 'live_delta_to_base_30000':[live_ratio(row,30000) for row in group],
                 'combined_to_base_30000':[statistics.mean(block['combined_to_base_rms'] for block in event(row,30000)['branches']) for row in group]}
        if family not in ('adaln','ffn_r2_budget'):
            metrics.update(early_update_to_paired_reference=[update_metric(row)/update_metric(keyed['basis_r8',seed]) for seed,row in zip(CONFIRM_SEEDS,group)],
                           final_update_to_paired_reference=[step_update(row,30000)/step_update(keyed['basis_r8',seed],30000) for seed,row in zip(CONFIRM_SEEDS,group)],
                           base_delta_correlation_30000=[statistics.mean(block['base_delta_correlation'] for block in event(row,30000)['branches']) for row in group])
        aggregates.append(dict(family=family,label=LABELS[family],parameters=group[0]['parameters'],
                               adapter_parameters=group[0]['adapter_parameters'],added_parameters=group[0]['parameters']-base,
                               seeds=list(CONFIRM_SEEDS),metrics={key:mean_sd(values) for key,values in metrics.items()}))
        paired[family]={}
        for control in ORDER:
            if family==control:
                continue
            differences={'fid':[keyed[family,seed]['generative_evaluation']['fid']-keyed[control,seed]['generative_evaluation']['fid'] for seed in CONFIRM_SEEDS],
                         'agreement_pp':[100*(keyed[family,seed]['generative_evaluation']['class_agreement']['generated']['agreement']-keyed[control,seed]['generative_evaluation']['class_agreement']['generated']['agreement']) for seed in CONFIRM_SEEDS]}
            paired[family][control]={key:mean_sd(values) for key,values in differences.items()}
    evidence=dict(raw_workflow=str(output),raw_report_sha256=digest(output/'report.json'),audit=audited,
                  calibration=report['calibration']['selected'],aggregates=aggregates,paired_differences=paired,
                  gpu_hours_training_validation_diagnostics=sum(row['loop_seconds'] for row in rows)/3600,
                  gpu_hours_final_generation_evaluation=sum(row['generative_evaluation']['seconds'] for row in rows)/3600,
                  limits=report['limits'],quality_winner_claimed=False)
    destination.mkdir(parents=True,exist_ok=True)
    prefix='dit-core-confirmation'
    if list(destination.glob(prefix+'*')):
        raise FileExistsError('Preserve existing core-confirmation exports')
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
        for index,aggregate in enumerate(aggregates):
            metric=aggregate['metrics'][key]
            ax.bar(index,metric['mean'],color=COLORS[aggregate['family']],alpha=.7,yerr=metric['sd'],capsize=4)
            ax.scatter(index+np.array([-.12,0,.12]),metric['values'],c='black',s=19,zorder=3)
        ax.set_xticks(range(len(ORDER)),[LABELS[family] for family in ORDER],rotation=25,ha='right')
        ax.set_title(title);ax.grid(axis='y',alpha=.15)
    fig.suptitle('30,000 steps · three fresh paired training seeds · mean ± sample SD')
    for suffix in ('png','svg'):
        fig.savefig(destination/(prefix+'.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.4),layout='constrained')
    for family in ('basis_r2','basis_r8','basis_r8_raw','core_r8','dplr_core_r8_p2'):
        steps=[value['step'] for value in keyed[family,CONFIRM_SEEDS[0]]['design_diagnostics']['diagnostics'] if value['step']>=10]
        for seed in CONFIRM_SEEDS:
            row,reference=keyed[family,seed],keyed['basis_r8',seed]
            axes[0].plot(steps,[step_update(row,step)/step_update(reference,step) for step in steps],color=COLORS[family],alpha=.23,lw=1)
            axes[1].plot(steps,[live_ratio(row,step) for step in steps],color=COLORS[family],alpha=.23,lw=1)
        axes[0].plot(steps,[statistics.mean(step_update(keyed[family,seed],step)/step_update(keyed['basis_r8',seed],step) for seed in CONFIRM_SEEDS) for step in steps],color=COLORS[family],label=LABELS[family],lw=2)
        axes[1].plot(steps,[statistics.mean(live_ratio(keyed[family,seed],step) for seed in CONFIRM_SEEDS) for step in steps],color=COLORS[family],label=LABELS[family],lw=2)
    for ax in axes:
        ax.set_xscale('log');ax.set_xlabel('Training step');ax.grid(alpha=.15)
    axes[0].axhline(1,color='black',ls='--',lw=.7)
    axes[0].set_title('Adapter update / paired scaled-r8 reference')
    axes[1].set_title('Live delta/base RMS (raw-model diagnostic)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Mean trajectories with individual seeds · highest-FP32 fixed probes')
    for suffix in ('png','svg'):
        fig.savefig(destination/(prefix+'-updates.'+suffix),dpi=180)
    plt.close(fig)
    def cell(aggregate,key,places):
        metric=aggregate['metrics'][key]
        return f"{metric['mean']:.{places}f} ± {metric['sd']:.{places}f}"
    lines=['# Calibrated conditional cores and FFN quality confirmation','',
           'Three fresh paired seeds (55/56/57), 30,000 steps each. Values are mean ± sample SD, not confidence intervals.','',
           '| Variant | Parameters | Added versus AdaLN | FID ↓ | Class agreement ↑ | EMA noise MSE | Train images/s |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for aggregate in aggregates:
        lines.append(f"| {aggregate['label']} | {aggregate['parameters']:,} | {aggregate['added_parameters']:,} | {cell(aggregate,'fid',2)} | {cell(aggregate,'agreement_percent',2)}% | {cell(aggregate,'ema_noise_mse',5)} | {cell(aggregate,'train_images_per_second',0)} |")
    lines+=['',f'![Quality comparison]({prefix}.png)','',
            'The FFN MLP hidden width is 864 (ratio 4.5), versus 768 for the other families. Its total budget is only 384 parameters above basis r2. It has no adapter; its added capacity is not counted as adapter parameters. Changing its shape also changes initialization, so this control is seed-matched rather than identical-base matched. All equal-shape variants have identical base hashes, and raw/scaled basis r8 share adapter initialization.','',
            '| Variant | Whole-training peak MiB |','| --- | ---: |']
    for aggregate in aggregates:
        lines.append(f"| {aggregate['label']} | {cell(aggregate,'peak_allocated_mib',1)} |")
    lines+=['','Peak allocation includes model, in-memory EMA, gradients, optimizer, training activations and fixed diagnostic inputs. Diagnostic forwards and final FID evaluation are excluded. Timing is concurrent eager training; these results are not an isolated speed or equal-time quality comparison.','',
            'Calibration uses the full 45,000-image loader on seeds 53/54 for a 500-step prefix of the confirmation schedule. The metric averages independent highest-FP32 adapter-delta update RMS across all six blocks at steps 50/100/250/500 and both seeds. Core adapter LR multipliers are selected only on this metric, with at most two bounded corrections and ±25% tolerance. All gains stay fixed; control multipliers stay 1. Recipes are frozen before confirmation. No loss, validation, FID or class agreement enters selection.','',
            '| Calibrated family | Frozen adapter LR multiplier | Update / target | Selected attempt |','| --- | ---: | ---: | ---: |']
    for family,candidate in report['calibration']['selected'].items():
        lines.append(f"| {LABELS[family]} | {candidate['adapter_lr_multiplier']:.6f} | {candidate['metric_to_target']:.3f} | {candidate['attempt']} |")
    lines+=['','| Adapter | Early update / paired r8 | Final update / paired r8 | Final live delta/base | Final combined/base | Base–delta correlation |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for aggregate in aggregates:
        if 'early_update_to_paired_reference' in aggregate['metrics']:
            lines.append(f"| {aggregate['label']} | {cell(aggregate,'early_update_to_paired_reference',3)} | {cell(aggregate,'final_update_to_paired_reference',3)} | {cell(aggregate,'live_delta_to_base_30000',3)} | {cell(aggregate,'combined_to_base_30000',3)} | {cell(aggregate,'base_delta_correlation_30000',3)} |")
    outside=[f"{aggregate['label']} seed {seed}: {value:.3f}×" for aggregate in aggregates if aggregate['family'] in ('core_r8','dplr_core_r8_p2')
             for seed,value in zip(CONFIRM_SEEDS,aggregate['metrics']['early_update_to_paired_reference']['values']) if abs(value-1)>.25]
    lines+=['','Fresh-seed early core ratios outside the calibration tolerance: '+('; '.join(outside) if outside else 'none')+'. Confirmation rates were not changed in response.','',
            f'![Update and magnitude trajectories]({prefix}-updates.png)','',
            'The update probe measures only the adapter delta at fixed independent normalized inputs/conditions, in the raw model. It does not measure the entire network update. Live branch diagnostics use eight fixed held-out images and report means across six blocks. EMA quality measures a different state. Scalar calibration does not match each block, seed, later update or signal magnitude, and is not a variance-preservation/gauge-fixing guarantee.','',
            'Paired differences below are variant minus control in seed order 55/56/57. Negative FID and positive agreement differences favor the variant.','',
            '| Variant vs control | FID differences | Agreement differences (pp) |','| --- | --- | --- |']
    comparisons=[('basis_r2','ffn_r2_budget')]+[(family,control) for family in ('core_r8','dplr_core_r8_p2') for control in ('adaln','ffn_r2_budget','basis_r2','basis_r8','basis_r8_raw')]
    for family,control in comparisons:
        diff=paired[family][control]
        lines.append(f"| {LABELS[family]} vs {LABELS[control]} | {', '.join(f'{value:+.2f}' for value in diff['fid']['values'])} | {', '.join(f'{value:+.2f}' for value in diff['agreement_pp']['values'])} |")
    lines+=['','The common recipe is pixel-space epsilon prediction, width 192/depth 6/3 heads/patch 4, normalized broadcast class/time conditioning, non-affine pre-norm/AdaLN-Zero, base LR 4e-4, warmup 50, AdamW (.9,.999), zero weight decay, global clipping 1, cosine floor .01, BF16 and in-memory EMA .999. FID uses 5,000 held-out real and 5,000 generated images, 50-step DDIM without CFG and sampling seed 2028. Class agreement uses the existing frozen FP32 classifier, with 500 requested images per class. One held-out split/sampling seed and three training seeds do not establish a universal architectural winner. Sequence-valued conditioning/cross-attention remain untested.','',
            f"Training/validation/diagnostics used {evidence['gpu_hours_training_validation_diagnostics']:.3f} aggregate GPU-hours; final generation/evaluation used {evidence['gpu_hours_final_generation_evaluation']:.3f}. Calibration/startup are additional. Compact raw evidence is {audited['bytes_retained']/2**20:.1f} MiB under the declared 512 MiB bound. All finite workers/controllers exited successfully, supervisors closed and GPU claims released. No new model/evaluator weights, resume checkpoints, generated image datasets or official test evaluation.",'',
            f'[Structured comparison]({prefix}.json) · [Audit]({prefix}-audit.json) · [Full evidence]({prefix}-evidence.json) · [Design notes](../DESIGN_NOTES.md)','']
    (destination/(prefix+'.md')).write_text('\n'.join(lines))
    analysis=output/'analysis_source';analysis.mkdir(exist_ok=True)
    for filename in ('summarize_dit_core.py','audit_dit_core.py'):
        shutil.copy2(Path(__file__).with_name(filename),analysis/filename)
    artifacts={path.name:dict(sha256=digest(path),bytes=path.stat().st_size) for path in destination.glob(prefix+'*')}
    write_json(destination/(prefix+'-manifest.json'),dict(workflow=str(output),artifacts=artifacts,
               analysis_source_sha256={path.name:digest(path) for path in analysis.iterdir()}))
    return dict(report=str(destination/(prefix+'.md')),manifest=str(destination/(prefix+'-manifest.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    root=Path(json.loads((output/'protocol.json').read_text())['repo_root'])
    print(json.dumps(summarize(output,args.destination or root/'reports')))

#!/usr/bin/env python3
"""Audited shared-projector screen with paired quality and magnitude evidence."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

from audit_dit_projectors import audit
from dit_projector_screen import FAMILIES, CONFIRM_SEEDS, digest, update_metric
from train_dit_cifar10 import write_json

LABELS={'independent_linear':'Independent s16 affine','independent_silu':'Independent s16 SiLU',
        'two_groups_linear':'Two groups s48 affine','two_groups_silu':'Two groups s48 SiLU',
        'all_shared_linear':'All shared s96 affine','two_groups_s16':'Two groups s16 affine'}
COLORS=dict(zip(FAMILIES,('#64748b','#0891b2','#2563eb','#7c3aed','#db2777','#ea580c')))
PREFIX='dit-projector-screen'


def metric(values):
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),values=list(values))


def event(row,step):return next(v for v in row['design_diagnostics']['diagnostics'] if v['step']==step)


def update(row,step):return statistics.mean(v['update_rms'] for v in event(row,step)['fixed_token_condition_function_updates'])


def live(row,step):return statistics.mean(v['delta_to_base_rms'] for v in event(row,step)['branches'])


def summarize(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();audited=audit(output)
    report=json.loads((output/'report.json').read_text());rows=report['confirmation']['rows']
    keyed={(row['family'],row['config']['seed']):row for row in rows};aggregates=[];paired={}
    for family in FAMILIES:
        group=[keyed[family,seed] for seed in CONFIRM_SEEDS]
        data=dict(fid=[row['generative_evaluation']['fid'] for row in group],
                  agreement_percent=[100*row['generative_evaluation']['class_agreement']['generated']['agreement'] for row in group],
                  ema_noise_mse=[row['ema_validation']['noise_mse'] for row in group],
                  peak_allocated_mib=[row['peak_allocated_bytes']/2**20 for row in group],
                  early_update_to_anchor=[update_metric(row)/update_metric(keyed['independent_linear',seed]) for row,seed in zip(group,CONFIRM_SEEDS)],
                  final_update_to_anchor=[update(row,5000)/update(keyed['independent_linear',seed],5000) for row,seed in zip(group,CONFIRM_SEEDS)],
                  final_live_delta_to_base=[live(row,5000) for row in group],
                  final_transform_frobenius=[statistics.mean(v['transform_frobenius_mean'] for v in event(row,5000)['fixed_condition_projector_magnitudes']) for row in group])
        aggregates.append(dict(family=family,label=LABELS[family],parameters=group[0]['parameters'],adapter_parameters=group[0]['adapter_parameters'],
                               design=FAMILIES[family],lr_multiplier=group[0]['config']['adapter_lr_multiplier'],metrics={k:metric(v) for k,v in data.items()}))
        paired[family]={}
        for control in FAMILIES:
            if control==family:continue
            paired[family][control]=dict(fid=metric([keyed[family,s]['generative_evaluation']['fid']-keyed[control,s]['generative_evaluation']['fid'] for s in CONFIRM_SEEDS]),
                                        agreement_pp=metric([100*(keyed[family,s]['generative_evaluation']['class_agreement']['generated']['agreement']-keyed[control,s]['generative_evaluation']['class_agreement']['generated']['agreement']) for s in CONFIRM_SEEDS]))
    hours=dict(calibration=sum(row['loop_seconds'] for stage in report['calibration']['rounds'] for row in stage['rows'])/3600,
               screen=sum(row['loop_seconds'] for row in rows)/3600,final_evaluation=sum(row['generative_evaluation']['seconds'] for row in rows)/3600)
    evidence=dict(workflow=str(output),audit=audited,aggregates=aggregates,paired_differences=paired,calibration=report['calibration']['selected'],limits=report['limits'],gpu_hours=hours,quality_winner_claimed=False)
    destination.mkdir(parents=True,exist_ok=True)
    if list(destination.glob(PREFIX+'*')):raise FileExistsError('Preserve existing exports')
    write_json(destination/(PREFIX+'.json'),evidence)
    shutil.copy2(output/'audit.json',destination/(PREFIX+'-audit.json'));shutil.copy2(output/'report.json',destination/(PREFIX+'-evidence.json'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for ax,key,title in zip(axes,('fid','agreement_percent'),('FID (lower is better)','Requested-class agreement (%)')):
        for i,item in enumerate(aggregates):
            value=item['metrics'][key];ax.bar(i,value['mean'],yerr=value['sd'],capsize=4,color=COLORS[item['family']],alpha=.75)
            ax.scatter(i+np.linspace(-.1,.1,len(CONFIRM_SEEDS)),value['values'],c='black',s=22,zorder=3)
        ax.set_xticks(range(len(aggregates)),[x['label'] for x in aggregates],rotation=25,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.15)
    fig.suptitle('5,000-step screen of a 30k schedule · two fresh paired seeds · mean ± sample SD')
    for suffix in ('png','svg'):fig.savefig(destination/(PREFIX+'.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for family in FAMILIES:
        steps=[v['step'] for v in keyed[family,CONFIRM_SEEDS[0]]['design_diagnostics']['diagnostics'] if v['step']>=10]
        for seed in CONFIRM_SEEDS:
            row,anchor=keyed[family,seed],keyed['independent_linear',seed]
            axes[0].plot(steps,[update(row,s)/update(anchor,s) for s in steps],c=COLORS[family],alpha=.2)
            axes[1].plot(steps,[live(row,s) for s in steps],c=COLORS[family],alpha=.2)
        axes[0].plot(steps,[statistics.mean(update(keyed[family,seed],s)/update(keyed['independent_linear',seed],s) for seed in CONFIRM_SEEDS) for s in steps],c=COLORS[family],label=LABELS[family],lw=2)
        axes[1].plot(steps,[statistics.mean(live(keyed[family,seed],s) for seed in CONFIRM_SEEDS) for s in steps],c=COLORS[family],label=LABELS[family],lw=2)
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('Training step');ax.grid(alpha=.15)
    axes[0].axhline(1,c='black',ls='--',lw=.7);axes[0].set_title('Single-step fixed-probe update / paired anchor')
    axes[1].set_title('Live delta/base RMS (raw model)');axes[1].legend(fontsize=8)
    for suffix in ('png','svg'):fig.savefig(destination/(PREFIX+'-updates.'+suffix),dpi=180)
    plt.close(fig)
    def cell(item,key,places=2):
        v=item['metrics'][key];return f"{v['mean']:.{places}f} ± {v['sd']:.{places}f}"
    lines=['# Shared and nonlinear factor-projector screen','',
           'Two fresh paired screen seeds (67/68), 5,000 steps of the 30k schedule. Values are mean ± sample SD, not confidence intervals. Independent update-only calibration uses seeds 65/66; recipes freeze before quality training. This is a short screen, not longer confirmation.','',
           '| Recipe | Adapter / total parameters | Groups | s | LR multiplier | FID ↓ | Agreement ↑ |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in aggregates:
        lines.append(f"| {item['label']} | {item['adapter_parameters']:,} / {item['parameters']:,} | {item['design']['projector_groups']} | {item['design']['n_basis']} | {item['lr_multiplier']:.6f} | {cell(item,'fid')} | {cell(item,'agreement_percent')}% |")
    lines+=['',f'![Quality comparison]({PREFIX}.png)','',
            'The first five recipes have exactly equal parameter counts; two-group s16 is a fixed-width tying control. Rank is eight throughout, with fixed gain 1/sqrt(8). Direct SiLU changes coefficient dependence without adding parameters or widening the factor decoder span. Base width192/depth6/heads3/patch4, non-affine AdaLN-Zero, normalized adapter input, broadcast class/time and self-attention remain common.','',
            'Shared factors are generated once per group per forward and reused across its contiguous layers. Parameters, optimizer groups and gradient records count unique tensors; branch observations remain layer-specific. Identical conditions tie each group’s conditional transform. At equal affine budget G*s is fixed, so fewer groups do not imply fewer leading generation FLOPs.','',
            '## Update selection and magnitude','',
            'Calibration uses the full 45k loader and a 500-step prefix of the 30k schedule. Highest-FP32 single-step adapter function-update RMS is averaged over six layer probes, steps50/100/250/500 and both seeds. Shared layer probes repeat the same function within a group and thus weight groups by their layer counts. One rate response is always verified for non-anchor families, followed by at most two further corrections when needed; multipliers are bounded to [1/16,16]. Selection minimizes absolute log(update/target), excludes quality, and requires the selected aggregate within ±15%.','',
            '| Recipe | Calibration update / target | Fresh early update / anchor | Final update / anchor | Live delta/base | Fixed transform Frobenius | Peak MiB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in aggregates:
        cal=report['calibration']['selected'][item['family']]
        lines.append(f"| {item['label']} | {cal['metric_to_target']:.3f} | {cell(item,'early_update_to_anchor',3)} | {cell(item,'final_update_to_anchor',3)} | {cell(item,'final_live_delta_to_base',3)} | {cell(item,'final_transform_frobenius',3)} | {cell(item,'peak_allocated_mib',1)} |")
    outside=[f"{LABELS[x['family']]} seed {s}: {v:.3f}×" for x in aggregates for s,v in zip(CONFIRM_SEEDS,x['metrics']['early_update_to_anchor']['values']) if abs(v-1)>.15]
    lines+=['','Fresh early ratios outside calibration tolerance: '+('; '.join(outside) if outside else 'none')+'. Recipes were not retuned.','',
            f'![Update and branch trajectories]({PREFIX}-updates.png)','',
            'Fixed-probe updates measure one optimizer step on the raw model. Transform Frobenius values average fixed-condition generated transforms across samples and unique projector groups, including fixed gain; their input/output factor norms are retained per group in full evidence. Live ratios average six layer branches on eight fixed held-out images. EMA quality is a different state. These diagnostics do not prove a gauge cause or variance preservation. Whole-training peaks include EMA, gradients, optimizer, activations and fixed probes; diagnostic forwards and final FID are excluded.','',
            '## Paired differences','',
            'Variant minus control, in seed order67/68. Negative FID and positive agreement favor the variant.','',
            '| Comparison | FID differences | Agreement differences (pp) |','| --- | --- | --- |']
    comparisons=[('independent_silu','independent_linear'),('two_groups_silu','two_groups_linear'),('two_groups_linear','independent_linear'),('all_shared_linear','independent_linear'),('two_groups_s16','independent_linear'),('two_groups_linear','two_groups_s16')]
    for family,control in comparisons:
        v=paired[family][control];lines.append(f"| {LABELS[family]} vs {LABELS[control]} | {', '.join(f'{x:+.2f}' for x in v['fid']['values'])} | {', '.join(f'{x:+.2f}' for x in v['agreement_pp']['values'])} |")
    lines+=['','## Protocol and limits','',
            'Full45k/5k CIFAR-10 split123, baseLR4e-4, warmup50, cosinefloor.01 over30k, AdamW(.9,.999), weightdecay0, clip1, BF16, in-memory EMA.999. Final EMA evaluation uses5k real/generated images, DDIM50, noCFG, sampling seed2028 and the unchanged frozen classifier. No official-test evaluation.','']
    lines.extend('- '+value for value in report['limits'])
    lines+=['',f"Calibration loops {hours['calibration']:.3f}, screen loops {hours['screen']:.3f}, final evaluations {hours['final_evaluation']:.3f} aggregate GPU-hours; startup and compatibility smoke additional. Raw evidence at audit {audited['bytes_retained']/2**20:.1f}MiB under512MiB. All private supervisors are closed and GPU claims released. No model/evaluator weights, checkpoints, grids or generated datasets saved.",'',
            f'[Structured results]({PREFIX}.json) · [Full evidence]({PREFIX}-evidence.json) · [Audit]({PREFIX}-audit.json) · [Design notes](../DESIGN_NOTES.md)','']
    (destination/(PREFIX+'.md')).write_text('\n'.join(lines))
    analysis=output/'analysis_source';analysis.mkdir(exist_ok=True)
    for name in ('audit_dit_projectors.py','summarize_dit_projectors.py'):shutil.copy2(Path(__file__).with_name(name),analysis/name)
    artifacts={path.name:dict(sha256=digest(path),bytes=path.stat().st_size) for path in destination.glob(PREFIX+'*')}
    write_json(destination/(PREFIX+'-manifest.json'),dict(workflow=str(output),artifacts=artifacts,analysis_source_sha256={path.name:digest(path) for path in analysis.iterdir()}))
    return dict(report=str(destination/(PREFIX+'.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--destination',type=Path);args=parser.parse_args()
    root=Path(json.loads((args.output/'protocol.json').read_text())['repo_root'])
    print(json.dumps(summarize(args.output,args.destination or root/'reports')))

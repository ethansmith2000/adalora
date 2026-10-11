#!/usr/bin/env python3
"""Audited coarse parameter trends, paired seeds and token-conditioning pilot."""

import argparse
import json
from pathlib import Path
import shutil
import statistics

from audit_rank_width_search import audit
from rank_width_search import DIFFUSION, VIT, RANKS, WIDTHS, SEEDS, digest
from train_dit_cifar10 import write_json


def metric(values):
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),values=values)


def probe_update(row, steps):
    kind=row['kind']
    diagnostic=row['results']['design_diagnostics' if kind=='diffusion' else 'token_diagnostics']['diagnostics']
    events={event['step']:event for event in diagnostic}
    key='fixed_token_condition_function_updates' if kind=='diffusion' else 'fixed_lowrank_updates'
    return statistics.mean(block['update_rms'] for step in steps for block in events[step][key])


def summarize(output):
    output=Path(output).resolve()
    audited=audit(output)
    evidence=json.loads((output/'evidence.json').read_text())
    keyed={(row['kind'],row['family'],row['seed']):row for row in evidence['rows']}
    aggregates={};paired={}
    for kind,families in (('diffusion',DIFFUSION),('vit',VIT)):
        aggregates[kind]=[];paired[kind]={}
        baseline=keyed[kind,'adaln' if kind=='diffusion' else 'plain',SEEDS[0]]['results']['parameters']
        for family in families:
            rows=[keyed[kind,family,seed] for seed in SEEDS]
            result=rows[0]['results']
            values={}
            if kind=='diffusion':
                values.update(fid=[row['results']['generative_evaluation']['fid'] for row in rows],
                              agreement_percent=[100*row['results']['generative_evaluation']['class_agreement']['generated']['agreement'] for row in rows],
                              loss=[row['results']['ema_validation']['noise_mse'] for row in rows])
                adapter=result['adapter_parameters']
                live=[[block['delta_to_base_rms'] for block in row['results']['design_diagnostics']['diagnostics'][-1]['branches']] for row in rows]
            else:
                values.update(accuracy_percent=[100*row['results']['validation']['accuracy'] for row in rows],
                              loss=[row['results']['validation']['loss'] for row in rows])
                adapter=result['extra_parameters']
                live=[[block['lowrank_to_base_rms'] for block in row['results']['token_diagnostics']['diagnostics'][-1]['live_branches']] for row in rows]
            values['final_live_delta_to_base']=[statistics.mean(blocks) for blocks in live]
            memory_key='peak_allocated_bytes' if kind=='diffusion' else 'peak_training_allocated_bytes'
            values['reported_training_peak_mib']=[row['results'][memory_key]/2**20 for row in rows]
            values['early_update_to_paired_r8']=[probe_update(row,(50,250,500))/probe_update(keyed[kind,'r8_s16',seed],(50,250,500)) for row,seed in zip(rows,SEEDS)]
            final=5000 if kind=='diffusion' else 3520
            values['final_update_to_paired_r8']=[probe_update(row,(final,))/probe_update(keyed[kind,'r8_s16',seed],(final,)) for row,seed in zip(rows,SEEDS)]
            aggregate=dict(family=family,parameters=result['parameters'],added_parameters=result['parameters']-baseline,
                           adapter_parameters=adapter,rank=rows[0]['config']['rank'],n_basis=rows[0]['config']['n_basis'],
                           metrics={key:metric(value) for key,value in values.items()})
            aggregates[kind].append(aggregate)
            reference='r8_s16'
            quality='fid' if kind=='diffusion' else 'accuracy_percent'
            target=([keyed[kind,reference,seed]['results']['generative_evaluation']['fid'] for seed in SEEDS] if kind=='diffusion'
                    else [100*keyed[kind,reference,seed]['results']['validation']['accuracy'] for seed in SEEDS])
            paired[kind][family]=metric([a-b for a,b in zip(values[quality],target)])
    root=Path(evidence['protocol']['repo_root']);destination=root/'reports'
    prefix='rank-width-search'
    if list(destination.glob(prefix+'*')):raise FileExistsError('Preserve existing exports')
    destination.mkdir(exist_ok=True)
    write_json(destination/(prefix+'.json'),dict(raw_workflow=str(output),aggregates=aggregates,paired_to_scaled_r8=paired,
                                                limits=evidence['protocol']['limits'],audit=audited,quality_winner_claimed=False))
    shutil.copy2(output/'audit.json',destination/(prefix+'-audit.json'))
    shutil.copy2(output/'evidence.json',destination/(prefix+'-evidence.json'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    by_name={item['family']:item for item in aggregates['diffusion']}
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,key,title in zip(axes,('fid','agreement_percent'),('5k-step FID (lower is better)','Requested-class agreement (%)')):
        matrix=np.array([[by_name[f'r{r}_s{s}']['metrics'][key]['mean'] for s in WIDTHS] for r in RANKS])
        im=ax.imshow(matrix,cmap='viridis_r' if key=='fid' else 'viridis',aspect='auto')
        for i,r in enumerate(RANKS):
            for j,s in enumerate(WIDTHS):
                value=by_name[f'r{r}_s{s}']['metrics'][key]
                ax.text(j,i,f"{value['mean']:.1f}\n±{value['sd']:.1f}",ha='center',va='center',color='white',fontsize=9,
                        bbox=dict(facecolor='black',alpha=.35,edgecolor='none',pad=1))
        ax.set_xticks(range(len(WIDTHS)),WIDTHS);ax.set_yticks(range(len(RANKS)),RANKS)
        ax.set_xlabel('Generator width s per factor');ax.set_ylabel('Transform rank r');ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('Broadcast conditioning · two fresh training seeds · mean ± sample SD')
    for suffix in ('png','svg'):fig.savefig(destination/(prefix+'-diffusion.'+suffix),dpi=180)
    plt.close(fig)
    items=aggregates['vit'];fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    for index,item in enumerate(items):
        value=item['metrics']['accuracy_percent']
        ax.bar(index,value['mean'],yerr=value['sd'],capsize=3,color='#0891b2',alpha=.7)
        ax.scatter(index+np.array([-.07,.07]),value['values'],color='black',s=18,zorder=3)
    ax.set_xticks(range(len(items)),[item['family'] for item in items],rotation=30,ha='right')
    ax.set_ylabel('Held-out accuracy (%)');ax.set_title('ViT token self-conditioning · 10 epochs · two seeds · mean ± sample SD')
    ax.set_ylim(min(min(item['metrics']['accuracy_percent']['values']) for item in items)-2,
                max(max(item['metrics']['accuracy_percent']['values']) for item in items)+2)
    for suffix in ('png','svg'):fig.savefig(destination/(prefix+'-vit.'+suffix),dpi=180)
    plt.close(fig)
    lines=['# Rank/width trend screen and token-conditioned ViT pilot','',
           'Two fresh paired seeds (58/59), one held-out train split. Values are mean ± sample SD, not confidence intervals. These short screens require longer fresh-seed confirmation.','',
           '## Broadcast class/time conditioning','',
           'Full 45k loader, 5,000 steps of the 30k schedule, warmup 50, LR 4e-4, non-affine norms/AdaLN-Zero and in-memory EMA. Each final FID uses 5,000 real/generated images, DDIM50 and the existing frozen FP32 classifier. No quality/LR/update tuning in this screen.','',
           '| Family | Added parameters | FID ↓ | Agreement ↑ | EMA noise MSE | Final live delta/base | Training peak MiB |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def cell(item,key,places=2):
        value=item['metrics'][key]
        return f"{value['mean']:.{places}f} ± {value['sd']:.{places}f}"
    diffusion={item['family']:item for item in aggregates['diffusion']}
    tokens={item['family']:item for item in aggregates['vit']}
    contour=('r2_s48','r4_s24','r8_s12','r16_s6')
    contour_fid=' / '.join(cell(diffusion[name],'fid') for name in contour)
    contour_agreement=' / '.join(cell(diffusion[name],'agreement_percent')+'%' for name in contour)
    diagonal=tokens['diagonal'];dynamic=tokens['r8_s16']
    token_gain=[keyed['vit','r8_s16',seed]['results']['validation']['accuracy']*100-
                keyed['vit','diagonal',seed]['results']['validation']['accuracy']*100 for seed in SEEDS]
    condition_gain=[keyed['vit','r8_s12',seed]['results']['validation']['accuracy']*100-
                    keyed['vit','r8_s12_detached',seed]['results']['validation']['accuracy']*100 for seed in SEEDS]
    findings=['## Observed trends','',
              f'On the similar decoder-budget contour r*s=96, ranks 2/4/8/16 give FID {contour_fid}. Requested-class agreement is {contour_agreement}. Higher rank improves mean FID along this contour; class agreement favors the middle ranks. Encoder/bias budgets still differ, and gain/update scaling is not matched.', '',
              f"At rank 8, widths 6/12/16 give FID {cell(diffusion['r8_s6'],'fid')} / {cell(diffusion['r8_s12'],'fid')} / {cell(diffusion['r8_s16'],'fid')}; widths 24/48 give {cell(diffusion['r8_s24'],'fid')} / {cell(diffusion['r8_s48'],'fid')}. Width beyond 16 does not give a steady FID improvement here. Other ranks also do not support a universal monotonic width rule.", '',
              f"Rank-16/s24 gives FID {cell(diffusion['r16_s24'],'fid')} with {diffusion['r16_s24']['added_parameters']:,} added parameters. Doubling width to 48 gives {cell(diffusion['r16_s48'],'fid')} and {diffusion['r16_s48']['added_parameters']:,} added parameters; its substantially larger seed spread weakens a mean-only preference. This is a short-screen observation, not longer-budget confirmation.", '',
              f"In ViT, rank-8/s16 gives {cell(dynamic,'accuracy_percent')}% versus diagonal conditioning {cell(diagonal,'accuracy_percent')}%. Its paired accuracy gains are {token_gain[0]:+.2f}/{token_gain[1]:+.2f} percentage points, averaging {statistics.mean(token_gain):+.2f}, for {dynamic['added_parameters']-diagonal['added_parameters']:,} additional parameters and {dynamic['metrics']['reported_training_peak_mib']['mean']-diagonal['metrics']['reported_training_peak_mib']['mean']:.1f} MiB more reported training allocation. The small advantage needs longer fresh-seed confirmation.", '',
              f"Keeping gradients through c=h at rank-8/s12 improves accuracy over detached conditioning by {condition_gain[0]:+.2f}/{condition_gain[1]:+.2f} percentage points (mean {statistics.mean(condition_gain):+.2f}). This supports the full-gradient default within this recipe; it does not show that dynamic factors replace attention or universally beat diagonal conditioning.", '',
              f"Optimization remains a substantial confound: at diffusion rank 8, early fixed-input updates for s=6/48 are {cell(diffusion['r8_s6'],'early_update_to_paired_r8',3)} / {cell(diffusion['r8_s48'],'early_update_to_paired_r8',3)} of the paired s16 reference despite the common rank gain and learning rate. Rank-16/s48 ends with a live delta/base ratio {cell(diffusion['r16_s48'],'final_live_delta_to_base',3)}. Zero-delta initialization and 1/sqrt(r) do not imply matched updates or variance preservation.", '',
              'The FFN controls underperform their plain/AdaLN baselines at these short budgets under the fixed common recipe. They have different hidden shapes and initialization. Their comparisons help describe this recipe but do not establish superiority over fully optimized ordinary capacity. The diffusion AdaLN baseline also has large two-seed FID spread, and these 5k-step results must not be pooled with the earlier 30k confirmations.','',
              'Next comparisons should retain the scaled-r8 anchor, investigate rank-16/s6 versus rank-8/s12 at similar decoder budget, and keep diagonal/full-gradient/detached controls in token-conditioning work. Test gain/LR sensitivity before interpreting width effects as capacity alone. No longer confirmation, LM, retrieval, richer-generator or structured-operator jobs were automatically launched.','']
    lines[4:4]=findings
    for item in aggregates['diffusion']:
        lines.append(f"| {item['family']} | {item['added_parameters']:,} | {cell(item,'fid')} | {cell(item,'agreement_percent')}% | {cell(item,'loss',5)} | {cell(item,'final_live_delta_to_base',3)} | {cell(item,'reported_training_peak_mib',1)} |")
    lines+=['',f'![Rank/width quality surface]({prefix}-diffusion.png)','',
            'At constant rank, the columns test generator-width trends. At constant width, the rows test transform-rank trends. The decoder-budget contour (r,s)=(2,48)/(4,24)/(8,12)/(16,6) holds r*s=96; encoder/bias and total parameter budgets differ. Fixed gain is 1/sqrt(r), except the raw-r8 control. Generator width can change functional-update scale under this initialization, so capacity is not fully isolated from optimization.','',
            '## Token self-conditioning','',
            'ViT width 192/depth 6/patch 4, MLP-input placement, full 45k/5k split, ten epochs/3,520 steps, LR 5e-4, AdamW, weight decay .05, label smoothing .1, warmup .05 and BF16. The condition is each normalized contextual hidden token, differentiated through by default; the detached control has identical forward conditioning and initialization. All variants retain static LayerNorm affine parameters.','',
            '| Family | Added parameters | Held-out accuracy ↑ | Held-out cross-entropy ↓ | Final low-rank/base | Training peak MiB |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in aggregates['vit']:
        lines.append(f"| {item['family']} | {item['added_parameters']:,} | {cell(item,'accuracy_percent')}% | {cell(item,'loss',4)} | {cell(item,'final_live_delta_to_base',3)} | {cell(item,'reported_training_peak_mib',1)} |")
    lines+=['',f'![Token-conditioning pilot]({prefix}-vit.png)','',
            'Activation mixing avoids full per-token U/V tensors and uses basis intermediates [B,T,s,r]. The diagonal/static/DPLR controls share the input scale/shift design; the plain and FFN controls have no such heads. FFN hidden width is 1,212 versus 768 in the other cases, so its base initialization differs. Zero low-rank probes for plain/diagonal controls mean absence of a low-rank branch, not absence of learning. Live branches use eight fixed held-out images; independent normalized-token function probes use highest-FP32 matmul without autocast. These are raw-model diagnostics.','',
            'Reported training peaks include model, optimizer, gradients and activations; diffusion additionally keeps EMA in memory. Fixed diagnostic buffers also remain allocated, while diagnostic forwards and final diffusion evaluation are excluded from these peaks. These are whole-training memory observations, not isolated adapter activation measurements.','',
            '## Functional-update diagnostics','',
            'Fixed-input low-rank updates relative to the same-seed scaled r8/s16 reference, within each regime. Early averages use steps 50/250/500 over six blocks. Diffusion probes measure individual optimizer steps, including the final step 5,000. ViT probes measure changes since the previous recorded diagnostic (early intervals 40/200/250 steps; final interval 1,520 steps). Each paired reference uses the same interval. These are low-rank function updates, not entire-network updates; ViT interval changes are not per-step-average updates. The two regimes are not compared numerically.','',
            '| Regime / family | Early update / paired r8 | Final update / paired r8 |','| --- | ---: | ---: |']
    for kind,items in aggregates.items():
        for item in items:
            lines.append(f"| {kind} / {item['family']} | {cell(item,'early_update_to_paired_r8',3)} | {cell(item,'final_update_to_paired_r8',3)} |")
    lines+=['','## Limits and evidence','']
    lines.extend('- '+value for value in evidence['protocol']['limits'])
    hours=sum(row['results'].get('loop_seconds',row['results'].get('elapsed_seconds',0)) for row in evidence['rows'])/3600
    evalhours=sum(row['results'].get('generative_evaluation',{}).get('seconds',0) for row in evidence['rows'])/3600
    lines+=['',f'Training/validation/diagnostics: {hours:.3f} aggregate GPU-hours; final diffusion generation/evaluation: {evalhours:.3f}; startup additional. All workers/controllers in the corrected screen exited successfully, private supervisors closed and claims released. Raw evidence audited before analysis export: {audited["bytes_retained"]/2**20:.1f} MiB under 512 MiB. No new model/evaluator weights, checkpoints, generated datasets or official test evaluation. Dedicated speed benchmarks were deferred.','',
            'Training and launch source stays frozen in the workflow runner. The artifact manifest separately hashes the analysis source used for this export; post-launch analysis additions report whole-training memory, distinguish one-step diffusion probes from interval ViT probes, and audit the preserved recovery.','',
            f'[Structured aggregates and paired differences]({prefix}.json) · [Audit]({prefix}-audit.json) · [Full evidence]({prefix}-evidence.json) · [Design notes](../DESIGN_NOTES.md)','']
    if audited.get('recovery'):
        lines+=['## Preserved failed attempt','',
                f"The first six workers trained successfully but failed the frozen-classifier source guard during evaluation. Their source/configs/logs remain at `{audited['recovery']['failed_attempt']}`; private supervisors are closed. The corrected screen repeated their configs and matched their selected batch/noise traces. The evaluator now verifies and loads its original frozen implementation independently of the current trainer. A separate one-step / 20-generated-image GPU smoke checked compatibility before relaunch; it is not quality evidence. The failed attempt consumed {audited['recovery']['failed_claimed_wall_gpu_hours']:.3f} aggregate claimed GPU-hours including startup and failed evaluation, additional to successful-run loop/evaluation totals above.",'']
    (destination/(prefix+'.md')).write_text('\n'.join(lines))
    analysis=output/'analysis_source';analysis.mkdir(exist_ok=True)
    for name in ('audit_rank_width_search.py','summarize_rank_width_search.py'):
        shutil.copy2(Path(__file__).with_name(name),analysis/name)
    artifacts={path.name:dict(sha256=digest(path),bytes=path.stat().st_size) for path in destination.glob(prefix+'*')}
    write_json(destination/(prefix+'-manifest.json'),dict(workflow=str(output),artifacts=artifacts,
               analysis_source_sha256={path.name:digest(path) for path in analysis.iterdir()}))
    return dict(report=str(destination/(prefix+'.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(summarize(parser.parse_args().output)))

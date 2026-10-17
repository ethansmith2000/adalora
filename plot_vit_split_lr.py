#!/usr/bin/env python3
"""Plot audited 30-epoch ViT validation curves and individual seed endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'plain':'Plain', 'diagonal':'Diagonal', 'static_capacity':'Wider static MLP',
          's24':'DPLR s24', 's48':'DPLR s48', 'static_unit':'Static, unit LR',
          'static_split':'Static, split LR', 'static_joint_12':'Static, joint12'}


def plot(reports):
    evidence_path=reports/'vit-split-lr-evidence.json'
    audit=json.loads((reports/'vit-split-lr-audit.json').read_text())
    assert all(audit['checks'].values())
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest()==audit['report_sha256']
    evidence=json.loads(evidence_path.read_text());rows=evidence['reused_quality']+evidence['replacement_quality']
    comparison=json.loads((reports/'vit-split-lr.json').read_text())
    aggregates={r['family']:r for r in comparison['aggregates']}
    assert len(rows)==16 and {(r['family'],r['seed'])for r in rows}=={(f,s)for f in LABELS for s in (102,103)}
    fig,(curves,endpoints)=plt.subplots(1,2,figsize=(12,5.6),gridspec_kw={'width_ratios':[1.45,1]})
    colors=plt.get_cmap('tab10').colors
    for index,(family,label) in enumerate(LABELS.items()):
        values=sorted([r for r in rows if r['family']==family],key=lambda r:r['seed'])
        traces=[];final=[]
        for row in values:
            metrics_path=Path(row['output_dir'])/'metrics.jsonl'
            assert hashlib.sha256(metrics_path.read_bytes()).hexdigest()==row['evidence_sha256']['metrics.jsonl']
            metrics=[json.loads(line)for line in metrics_path.read_text().splitlines()]
            trace=[e for e in metrics if e['event']=='validation']
            assert len(trace)==30 and [e['step']for e in trace]==[352*k for k in range(1,31)]
            assert all(e['examples']==5000 for e in trace)
            assert trace[-1]['accuracy']==row['results']['validation']['accuracy']
            traces.append([100*e['accuracy']for e in trace]);final.append(100*trace[-1]['accuracy'])
        aggregate=aggregates[family]['metrics']['accuracy_percent']
        assert final==aggregate['values'] and statistics.mean(final)==aggregate['mean']
        means=[statistics.mean(v)for v in zip(*traces)]
        curves.plot(range(1,31),means,label=label,color=colors[index],linewidth=2 if family=='s24'else 1.3)
        endpoints.scatter(final[0],index+.1,color='#0072B2',s=34,label='Seed 102'if index==0 else None,zorder=3)
        endpoints.scatter(final[1],index-.1,color='#D55E00',s=34,label='Seed 103'if index==0 else None,zorder=3)
        endpoints.scatter(aggregate['mean'],index,marker='|',color='black',s=120,label='Mean'if index==0 else None,zorder=4)
    curves.set(xlabel='Epoch',ylabel='Validation accuracy (%)',title='Mean validation trajectories',xlim=(1,30))
    curves.legend(fontsize=8,loc='lower right',ncol=2,framealpha=.95)
    endpoints.set(yticks=range(len(LABELS)),yticklabels=list(LABELS.values()),xlabel='Final validation accuracy (%)',title='Paired seed endpoints')
    endpoints.invert_yaxis();endpoints.legend(fontsize=8,loc='lower left')
    for ax in (curves,endpoints):
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('CIFAR-10 ViT: independent affine and low-rank learning rates',fontsize=13)
    fig.text(.5,.025,'30 epochs; seeds 102/103; 5,000-image research holdout. Four original and twelve recovered cases; sources and pairing audited.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.95))
    for suffix in ('png','svg'):fig.savefig(reports/f'vit-split-lr-convergence.{suffix}',dpi=180)
    plt.close(fig)
    return reports/'vit-split-lr-convergence.png'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reports',type=Path,default=Path('reports'))
    args=parser.parse_args();print(plot(args.reports))

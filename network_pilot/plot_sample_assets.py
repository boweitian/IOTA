"""Scientific response-matrix figure for the completed per-question steering assets."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from sample_assets import DEFAULT,ROOT
os.environ['MPLCONFIGDIR']=str(ROOT/'.mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);args=p.parse_args()
    bundle=args.run/'asset_bundle';data=np.load(bundle/'response_matrix.npz')
    if not data['observed'].all():raise ValueError('Only plot the completed exhaustive matrix')
    inputs=json.loads((bundle/'inputs.json').read_text());rows={r['id']:r for r in inputs['records']}
    aid=data['action_ids'].tolist();sids=data['sample_ids'].tolist()
    order=['none']+[f'node:{i}:{sign:+d}'+suffix for i in range(9)
        for sign,suffix in [(-1,':dose1'),(-1,''),(1,''),(1,':dose1')]]
    ix=[aid.index(x) for x in order]
    suites=['gsm8k','bbh','ifeval','truthfulqa','writingbench'];sizes=[sum(rows[x]['suite']==s for x in sids) for s in suites]
    fig,axes=plt.subplots(5,1,figsize=(14,15),gridspec_kw=dict(height_ratios=sizes),constrained_layout=True)
    names=['Decompose','Verify','Precision','Confidence','Brevity','Explore','Imagery','Warmth','Courage']
    for axis,suite in zip(axes,suites):
        selected=[i for i,x in enumerate(sids) if rows[x]['suite']==suite]
        arr=data['reward'][selected][:,ix]
        im=axis.imshow(arr,vmin=0,vmax=1,cmap='viridis',aspect='auto',interpolation='nearest')
        axis.set_title(suite.upper()+f' — {len(selected)} questions',loc='left',fontsize=11)
        axis.set_ylabel('Question index');axis.set_xticks([0]+[2.5+4*i for i in range(9)])
        axis.set_xticklabels(['Baseline']+names,fontsize=8)
        for x in [.5]+[4.5+4*i for i in range(9)]:axis.axvline(x,color='white',alpha=.7,linewidth=.8)
        axis.set_yticks([0,len(selected)-1]);axis.set_yticklabels([1,len(selected)])
    fig.suptitle('IOTA per-question steering response asset\nEach concept: −1, −0.5, +0.5, +1; same frozen model and question',fontsize=14)
    cb=fig.colorbar(im,ax=axes,shrink=.6,pad=.02);cb.set_label('Observed reward (task-specific metric)')
    fig.savefig(bundle/'response_matrix.png',dpi=160);fig.savefig(bundle/'response_matrix.pdf');plt.close(fig)
    print('PLOTTED',bundle/'response_matrix.png')


if __name__=='__main__':main()

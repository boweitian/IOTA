"""Standalone publication/export figures; native cosine and complete measured response."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from experiment import ROOT
from expanded_geometry import DEFAULT
from expanded_concepts import CONCEPTS
os.environ['MPLCONFIGDIR']=str(ROOT/'.mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SHORT=['Decompose','Verify','Precision','Confidence','Brevity','Explore','Imagery','Warmth','Courage',
       'Assumptions','Relations','Counterexample','Quantities','Backtrack','Abstraction',
       'Checklist','Format','Task focus','Complete','Evidence','Uncertainty','Literal','Correct myths',
       'Narrative','Causality','Audience','Originality']
COLORS={'reasoning':'#5985bd','instructions':'#329079','epistemic':'#bc8e35','writing':'#a674a2'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--geometry-only',action='store_true');a=p.parse_args()
    run=a.run;view=run/'view';view.mkdir(exist_ok=True);g=np.load(run/'geometry/geometry.npz')
    fig=plt.figure(figsize=(17,9),layout='constrained');left=fig.add_subplot(121,projection='3d');right=fig.add_subplot(122)
    xyz=g['coordinates'];colors=[COLORS[c['group']] for c in CONCEPTS]
    left.scatter(*xyz.T,c=colors,s=80,edgecolors=['#344757' if i<9 else '#bb6d40' for i in range(27)],linewidths=1.2,depthshade=False)
    for i,pnt in enumerate(xyz):left.text(*pnt,str(i+1),fontsize=9)
    for group,color in COLORS.items():left.scatter([],[],[],c=color,label=group.capitalize(),s=45)
    left.legend(loc='upper left',frameon=False);left.set(xlabel='Direction PC1',ylabel='Direction PC2',zlabel='Direction PC3')
    left.view_init(elev=19,azim=37);left.set_title(f"Native unit directions, PCA preview\n3D explained variance: {100*g['projection_variance'].sum():.1f}%",loc='left',fontsize=12)
    im=right.imshow(g['cosine'],cmap='RdBu',vmin=-1,vmax=1,interpolation='nearest');right.set_title('Full 1024-dimensional cosine similarity',loc='left',fontsize=12)
    right.set_xticks(range(27),[str(i+1) for i in range(27)],fontsize=7)
    right.set_yticks(range(27),[f'{i+1:02d} {name}' for i,name in enumerate(SHORT)],fontsize=8)
    right.axvline(8.5,color='#bc7545',linewidth=1.5);right.axhline(8.5,color='#bc7545',linewidth=1.5)
    fig.colorbar(im,ax=right,shrink=.7,label='Cosine (full native space)')
    fig.suptitle('IOTA geometry asset: 9 preserved directions + 18 new authored contrasts',fontsize=16)
    fig.savefig(view/'geometry_map.png',dpi=170);fig.savefig(view/'geometry_map.pdf');plt.close(fig)
    if a.geometry_only:return
    bundle=run/'asset_bundle';data=np.load(bundle/'response_matrix.npz');assert data['observed'].all()
    rows={r['id']:r for r in json.loads((bundle/'inputs.json').read_text())['records']};ids=data['action_ids'].tolist()
    order=['none']+[f'node:{i}:{sign:+d}'+suffix for i in range(27) for sign,suffix in [(-1,':dose1'),(-1,''),(1,''),(1,':dose1')]]
    columns=[ids.index(x) for x in order];suites=['gsm8k','bbh','ifeval','truthfulqa','writingbench'];sizes=[sum(rows[x]['suite']==s for x in data['sample_ids']) for s in suites]
    fig,axes=plt.subplots(5,1,figsize=(22,17),gridspec_kw={'height_ratios':sizes},layout='constrained')
    for axis,suite in zip(axes,suites):
        ix=[i for i,x in enumerate(data['sample_ids']) if rows[x]['suite']==suite];reward=data['reward'][ix][:,columns]
        im=axis.imshow(reward-reward[:,0:1],cmap='RdBu',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect='auto',interpolation='nearest')
        axis.set_title(suite.upper()+f' / {len(ix)} questions',loc='left',fontsize=12);axis.set_ylabel('Question');axis.set_xticks([0]+[2.5+4*i for i in range(27)])
        axis.set_xticklabels(['Baseline']+SHORT,fontsize=8,rotation=55,ha='right')
        for x in [.5]+[4.5+4*i for i in range(27)]:axis.axvline(x,color='#66666655',linewidth=.4)
        axis.axvline(36.5,color='#bd7247',linewidth=2)
        axis.set_yticks([0,len(ix)-1],[1,len(ix)],fontsize=9)
    fig.colorbar(im,ax=axes,shrink=.55,pad=.01,label='Observed task reward minus saved baseline')
    fig.suptitle('IOTA exhaustive sample-direction asset / 175 questions × 109 settings\nEach direction: −1, −0.5, +0.5, +1; orange boundary marks new directions',fontsize=16)
    fig.savefig(view/'response_matrix.png',dpi=160);fig.savefig(view/'response_matrix.pdf');plt.close(fig)
    print('PLOTTED geometry and exhaustive response figures',flush=True)

if __name__=='__main__':main()

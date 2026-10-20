"""Navigator pilot v3 on Qwen3.5-9B: same directions as v2, injected at every position.

v2 finding: with last-position injection, effort-matched, length-orthogonal directions move their target
behaviors in the right direction and beyond random controls only at 3x the cap, and far less than the
instruction itself. v3 changes one thing: the shift is added to every prompt position and every generated
token (activation addition / CAA style). Directions, questions, judge and rule are unchanged; random
controls are injected the same way at the same doses.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from navigator9b import ROOT,LAYER,SCALE,MAX_NEW,pools,split,base_prompt,judge
from navigator9b_v2 import CONCEPTS
from engine import atomic_json
from studio_geometry import LayerEngine,MODEL,spearman,read_jsonl,append

OUT=ROOT/'runs/navigator9b-pilot-v3'
SOURCE=ROOT/'runs/navigator9b-pilot-v2/directions.npz'
DOSES=[-2.,-1.,-.5,0.,.5,1.,2.]

def generate(engine,name,shift,prompts,ids,path):
    done={(r['config'],r['id']):r for r in read_jsonl(path)}
    if all((name,i) in done for i in ids):return done
    outs=engine.generate(prompts,[shift]*len(prompts),max_new_tokens=MAX_NEW,positions='all')
    for i,o in zip(ids,outs):
        r=dict(config=name,id=i,text=o['text'],new_tokens=o['new_tokens'],truncated=o['truncated']);append(path,r);done[name,i]=r
    print('GEN',name,[o['new_tokens'] for o in outs],flush=True)
    return done

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    d=np.load(SOURCE);U=d['concepts'];R=d['random'];ids_c=[str(x) for x in d['ids']]
    assert ids_c==[c['id'] for c in CONCEPTS]
    sets=split(pools());cert=sets['certify'];ids=[r['id'] for r in cert];prompts=[base_prompt(r) for r in cert]
    spec=dict(model=MODEL,layer=LAYER,scale=SCALE,doses=DOSES,positions='all',directions=str(SOURCE),certify=ids,max_new_tokens=MAX_NEW)
    if (out/'spec.json').exists() and json.loads((out/'spec.json').read_text())!=spec:raise ValueError('Changed spec; use a new output directory')
    atomic_json(out/'spec.json',spec)
    engine=LayerEngine(dict(MODEL,layer_index=LAYER));engine.use_layer(LAYER);path=out/'generations.jsonl'
    gen=generate(engine,'baseline',np.zeros(engine.dim),prompts,ids,path)
    for k,c in enumerate(CONCEPTS):
        for dose in DOSES:
            if dose:gen=generate(engine,f"{c['id']}@{dose:+g}",dose*SCALE*U[k],prompts,ids,path)
    for k in range(len(R)):
        for dose in DOSES:
            if dose:gen=generate(engine,f'random{k}@{dose:+g}',dose*SCALE*R[k],prompts,ids,path)
    task={r['id']:base_prompt(r) for r in cert};items=[]
    for c in CONCEPTS:
        cfgs=['baseline']+[f"{c['id']}@{x:+g}" for x in DOSES if x]+[f'random{k}@{x:+g}' for k in range(len(R)) for x in DOSES if x]
        items+=[(f"{c['id']}|{cfg}|{i}",task[i],gen[cfg,i]['text'],c['behavior']) for cfg in cfgs for i in ids]
    rated=judge(engine,items,out/'judgments.jsonl')
    def score(cid,cfg,key='target'):
        return np.array([(rated[f'{cid}|{cfg}|{i}']['score'] or {}).get(key,np.nan) for i in ids],dtype=float)
    report=[]
    for c in CONCEPTS:
        cid=c['id'];by={x:score(cid,'baseline' if x==0 else f'{cid}@{x:+g}') for x in DOSES}
        coh={x:float(np.nanmean(score(cid,'baseline' if x==0 else f'{cid}@{x:+g}','coherence'))) for x in DOSES}
        coherent=[x for x in DOSES if x>0 and coh[x]>=coh[0.]-1.5 and coh[-x]>=coh[0.]-1.5]
        top=max(coherent) if coherent else None
        res=dict(concept=cid,mean_target_by_dose={str(x):float(np.nanmean(by[x])) for x in DOSES},coherence_by_dose={str(x):v for x,v in coh.items()},
                 mean_tokens_by_dose={str(x):float(np.mean([gen['baseline' if x==0 else f'{cid}@{x:+g}',i]['new_tokens'] for i in ids])) for x in DOSES},largest_coherent_dose=top)
        if top:
            ds=[x for x in DOSES if abs(x)<=top]
            rho=[spearman(ds,[by[x][j] for x in ds]) for j in range(len(ids)) if not any(np.isnan(by[x][j]) for x in ds)]
            gap=float(np.nanmean(by[top])-np.nanmean(by[-top]))
            rgaps=[float(np.nanmean(score(cid,f'random{k}@{top:+g}'))-np.nanmean(score(cid,f'random{k}@{-top:+g}'))) for k in range(len(R))]
            checks=dict(monotonic=bool(np.mean(rho)>=.5),specific=bool(gap>max(abs(x) for x in rgaps)),coherent=True)
            res.update(mean_spearman=float(np.mean(rho)),gap=gap,random_gaps=rgaps,checks=checks,certified=all(checks.values()))
        else:res.update(checks=dict(coherent=False),certified=False)
        report.append(res)
        print('CERT',cid,'certified' if res['certified'] else 'not certified','top',top,{k:round(v,1) for k,v in res['mean_target_by_dose'].items()},
              'coh',{k:round(v,1) for k,v in res['coherence_by_dose'].items()},flush=True)
    atomic_json(out/'certification.json',dict(concepts=report))
    print('NAVIGATOR V3 COMPLETE',flush=True)

if __name__=='__main__':main()

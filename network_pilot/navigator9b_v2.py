"""Navigator pilot v2 on Qwen3.5-9B: remove the shared effort/length axis that dominated v1.

v1 finding: every negative instruction asked for less work, so decomposition / verification / uncertainty /
exploration shared one "more vs less" component (cosine 0.5-0.8; -0.5 with brevity) and on held-out tasks
they mostly changed length. v2 changes exactly two things and widens the dose range:
  1. effort-matched negatives: the negative asks for an equally elaborate response without the target behavior;
  2. each direction is made orthogonal to the length (brevity) direction before use;
  3. doses -3..+3 x the shared cap, to find where behavior moves before coherence breaks.
Same contexts, held-out questions, judge and certification rule as v1 (rule evaluated at dose +/-1 and at the
largest coherent dose).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from navigator9b import ROOT,LAYER,SCALE,MAX_NEW,pools,split,base_prompt,with_instruction,judge,generate
from engine import atomic_json
from studio_geometry import LayerEngine,MODEL,spearman

OUT=ROOT/'runs/navigator9b-pilot-v2'
DOSES=[-3.,-2.,-1.,0.,1.,2.,3.]
LENGTH=dict(positive='Be as concise as possible and give only the essential content.',negative='Be very detailed and elaborate at length.')
CONCEPTS=[
    dict(id='decomposition',positive='Break the problem into explicit numbered steps and solve each step before combining them.',
         negative='Explain your reasoning as one continuous paragraph, without separating it into steps or a list.',
         behavior='breaks the solution into explicit, separate steps'),
    dict(id='verification',positive='After solving, explicitly check your answer: re-verify each calculation and test whether the result fits the question.',
         negative='After solving, explain in a few sentences why the method you chose is a natural one, without re-checking any result.',
         behavior='explicitly checks or re-verifies its own answer or calculations'),
    dict(id='uncertainty',positive='State how confident you are and point out what could make the answer wrong.',
         negative='Add a short note on what the question is testing, stated plainly, without any doubt or hedging about your answer.',
         behavior='expresses uncertainty, a confidence level, or ways it could be wrong'),
    dict(id='exploration',positive='Consider at least two different approaches and compare them before choosing one.',
         negative='Explain the single approach you use in detail and why it fits, without mentioning any alternative.',
         behavior='considers more than one approach or alternative before settling on an answer'),
    dict(id='format_control',positive='Follow the requested output format exactly, placing every element exactly where the instructions specify.',
         negative='Write the same content in your own preferred layout, without following the requested format.',
         behavior='follows the output format the task asks for exactly'),
]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    sets=split(pools())
    spec=dict(model=MODEL,layer=LAYER,scale=SCALE,doses=DOSES,max_new_tokens=MAX_NEW,concepts=CONCEPTS,length=LENGTH,
        sets={k:[r['id'] for r in v] for k,v in sets.items()},orthogonal_to='length direction')
    if (out/'spec.json').exists() and json.loads((out/'spec.json').read_text())!=spec:raise ValueError('Changed spec; use a new output directory')
    atomic_json(out/'spec.json',spec)
    engine=LayerEngine(dict(MODEL,layer_index=LAYER));engine.use_layer(LAYER)
    ctx=sets['fit']+sets['diagnostic'];nfit=len(sets['fit'])
    def contrast(pos,neg):
        a=engine.capture([with_instruction(r,pos) for r in ctx],layers=[LAYER])[LAYER]
        b=engine.capture([with_instruction(r,neg) for r in ctx],layers=[LAYER])[LAYER]
        return a-b
    L=contrast(LENGTH['positive'],LENGTH['negative'])[:nfit].mean(0);L/=np.linalg.norm(L)
    units,diag={},{}
    for c in CONCEPTS:
        con=contrast(c['positive'],c['negative']);raw=con[:nfit].mean(0)
        before=float(raw@L/np.linalg.norm(raw))
        raw=raw-(raw@L)*L;u=raw/np.linalg.norm(raw);units[c['id']]=u   # orthogonal to length
        cos=con@u/np.linalg.norm(con,axis=1)
        diag[c['id']]=dict(length_cosine_before=before,fit_cosines=cos[:nfit].tolist(),diagnostic_cosines=cos[nfit:].tolist())
    U=np.stack([units[c['id']] for c in CONCEPTS])
    rng=np.random.default_rng(20261016);g=rng.normal(size=(3,U.shape[1]));q,_=np.linalg.qr(np.vstack([U,L]).T);g-=(g@q)@q.T
    R=g/np.linalg.norm(g,axis=1,keepdims=True)
    np.savez_compressed(out/'directions.npz',concepts=U,length=L,ids=np.array([c['id'] for c in CONCEPTS]),random=R,scale=SCALE,layer=LAYER)
    atomic_json(out/'construction.json',dict(diagnostics=diag,gram=(U@U.T).tolist()))
    print('GRAM',np.round(U@U.T,2).tolist(),'length cos before',{k:round(v['length_cosine_before'],2) for k,v in diag.items()},flush=True)

    cert=sets['certify'];ids=[r['id'] for r in cert];prompts=[base_prompt(r) for r in cert];path=out/'generations.jsonl'
    gen=generate(engine,'baseline',np.zeros(engine.dim),prompts,ids,path,'base')
    for c in CONCEPTS:
        for d in DOSES:
            if d:gen=generate(engine,f"{c['id']}@{d:+g}",d*SCALE*units[c['id']],prompts,ids,path,'dose')
    for k in range(len(R)):
        for d in [-3.,-2.,-1.,1.,2.,3.]:gen=generate(engine,f'random{k}@{d:+g}',d*SCALE*R[k],prompts,ids,path,'random')
    task={r['id']:base_prompt(r) for r in cert};items=[]
    for c in CONCEPTS:
        cfgs=['baseline']+[f"{c['id']}@{d:+g}" for d in DOSES if d]+[f'random{k}@{d:+g}' for k in range(len(R)) for d in [-3.,-2.,-1.,1.,2.,3.]]
        items+=[(f"{c['id']}|{cfg}|{i}",task[i],gen[cfg,i]['text'],c['behavior']) for cfg in cfgs for i in ids]
    rated=judge(engine,items,out/'judgments.jsonl')
    def score(cid,cfg,key='target'):
        return np.array([(rated[f'{cid}|{cfg}|{i}']['score'] or {}).get(key,np.nan) for i in ids],dtype=float)
    report=[]
    for c in CONCEPTS:
        cid=c['id'];by={d:score(cid,'baseline' if d==0 else f'{cid}@{d:+g}') for d in DOSES}
        coh={d:float(np.nanmean(score(cid,'baseline' if d==0 else f'{cid}@{d:+g}','coherence'))) for d in DOSES}
        coherent=[d for d in DOSES if d>0 and coh[d]>=coh[0.]-1.5 and coh[-d]>=coh[0.]-1.5]
        top=max(coherent) if coherent else None
        res=dict(concept=cid,mean_target_by_dose={str(d):float(np.nanmean(by[d])) for d in DOSES},coherence_by_dose={str(d):v for d,v in coh.items()},
                 mean_tokens_by_dose={str(d):float(np.mean([gen['baseline' if d==0 else f'{cid}@{d:+g}',i]['new_tokens'] for i in ids])) for d in DOSES},
                 largest_coherent_dose=top,diagnostic_cosines=diag[cid]['diagnostic_cosines'],length_cosine_before=diag[cid]['length_cosine_before'])
        if top:
            ds=[-top,-1.,0.,1.,top] if top>1 else [-1.,0.,1.]
            rho=[spearman(ds,[by[d][j] for d in ds]) for j in range(len(ids)) if not any(np.isnan(by[d][j]) for d in ds)]
            gap=float(np.nanmean(by[top])-np.nanmean(by[-top]))
            rdose=top   # random directions compared at the same dose
            rgaps=[float(np.nanmean(score(cid,f'random{k}@{rdose:+g}'))-np.nanmean(score(cid,f'random{k}@{-rdose:+g}'))) for k in range(len(R))]
            checks=dict(monotonic=bool(np.mean(rho)>=.5),specific=bool(gap>max(abs(x) for x in rgaps)),coherent=True)
            res.update(mean_spearman=float(np.mean(rho)),gap=gap,random_gaps=rgaps,random_dose=rdose,checks=checks,certified=all(checks.values()))
        else:
            res.update(checks=dict(coherent=False),certified=False)
        report.append(res)
        print('CERT',cid,'certified' if res['certified'] else 'not certified','top dose',top,
              {k:round(v,1) for k,v in res['mean_target_by_dose'].items()},'coh',{k:round(v,1) for k,v in res['coherence_by_dose'].items()},flush=True)
    atomic_json(out/'certification.json',dict(concepts=report))
    print('NAVIGATOR V2 COMPLETE',flush=True)

if __name__=='__main__':main()

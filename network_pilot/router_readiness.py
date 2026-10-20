"""Before training a per-sample router: are the per-question labels reproducible, and do concept
directions beat same-norm random directions under the same best-of-N search?

random  — 12 random directions (orthogonal to the 27-concept span) x {+,-} x {0.5,1}: 48 configs on all
          111 GSM8K/BBH/IFEval questions, in exactly the original shard batches.
replicate — 36 stratified questions x 25 configs (none, 12 concept, 12 random), re-run in 5 shuffled
          batch compositions. Original single runs are rep 0.
No benchmark answer is used to build any direction; answers only grade outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiment import ROOT,PilotEngine
from expanded_assets import RecordedExecution
from expanded_geometry import DEFAULT as ASSETS
from engine import atomic_json
from space import action,serial_action

OUT=ROOT/'runs/router-readiness-v1'
SUITES=['gsm8k','bbh','ifeval']
SHARDS=5
REPS=5

def protocol():
    return json.loads((ROOT/'inputs/protocol.json').read_text())

def random_bank(geo):
    unit=geo['unit'];scale=float(geo['scale'])
    rng=np.random.default_rng(20261014);g=rng.normal(size=(12,unit.shape[1]))
    q,_=np.linalg.qr(unit.T.astype(np.float64));g-=(g@q)@q.T
    directions=(g/np.linalg.norm(g,axis=1,keepdims=True)*scale).astype(unit.dtype)
    bank=[]
    for i in range(len(directions)):
        for dose in [.5,1.]:
            for sign in [-1,1]:
                w=np.zeros(len(directions));w[i]=sign*dose
                bank.append(dict(action(w,directions,scale,f'rand:{i}:{sign:+d}'+(':dose1' if dose==1 else '')),family='random',requested_strength=sign*dose))
    return bank,directions

def concept_bank(geo):
    from expanded_assets import bank_for
    return {a['id']:a for a in bank_for(geo)}

def stratified(matrix,ids,actions,suite_of):
    """12 per suite: 8 baseline-wrong spread over how many concept configs fixed them, 4 baseline-right."""
    none=actions.index('none');chosen=[]
    for suite in SUITES:
        rows=[i for i,s in enumerate(ids) if suite_of[s]==suite]
        wrong=sorted([i for i in rows if not matrix[i,none]],key=lambda i:(matrix[i].sum(),ids[i]))
        right=sorted([i for i in rows if matrix[i,none]],key=lambda i:(-matrix[i].sum(),ids[i]))
        pick=lambda xs,k:[xs[int(round(j))] for j in np.linspace(0,len(xs)-1,k)]
        chosen+=pick(wrong,8)+pick(right,4)
    return [ids[i] for i in chosen]

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['random','replicate']);p.add_argument('--task',type=int,required=True);args=p.parse_args()
    proto=protocol();geo=dict(np.load(ASSETS/'geometry/geometry.npz'))
    rbank,rdirs=random_bank(geo);cbank=concept_bank(geo)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'random_directions.npz',directions=rdirs)
    rows_by_id={r['id']:r for r in proto['records']}
    if args.mode=='random':
        suite=SUITES[args.task//SHARDS];shard=args.task%SHARDS
        rows=[r for r in proto['records'] if r['suite']==suite][shard::SHARDS]   # identical to the original shard batches
        out=OUT/'random'/suite/f'shard-{shard}';bank=rbank
    else:
        m=np.load(ASSETS/'asset_bundle/response_matrix.npz');ids=[str(x) for x in m['sample_ids']];acts=[str(x) for x in m['action_ids']]
        chosen=stratified(m['success'],ids,acts,{i:r['suite'] for i,r in rows_by_id.items()})
        # Concept configs: 6 with the most observed successes on these questions (the would-be winners) + 6 others.
        cols=[acts.index(a) for a in acts if a!='none' and not a.startswith('mix') and a in cbank]
        sub=m['success'][[ids.index(i) for i in chosen]]
        ranked=sorted(cols,key=lambda j:(-sub[:,j].sum(),acts[j]))
        rng=np.random.default_rng(7);rest=[acts[j] for j in ranked[6:]]
        concept=[acts[j] for j in ranked[:6]]+sorted(rng.choice(rest,6,replace=False).tolist())
        random_ids=sorted(rng.choice([a['id'] for a in rbank],12,replace=False).tolist())
        bank=[cbank['none']]+[cbank[a] for a in concept]+[next(a for a in rbank if a['id']==r) for r in random_ids]
        rep=args.task+1;order=np.random.default_rng(1000+rep).permutation(len(chosen))
        rows=[rows_by_id[chosen[i]] for i in order]          # shuffled, mixed-suite batches of 8
        out=OUT/'replicate'/f'rep-{rep}'
        atomic_json(OUT/'replicate_design.json',dict(questions=chosen,concept_configs=concept,top_observed=concept[:6],random_configs=random_ids,reps=REPS))
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(mode=args.mode,task=args.task,ids=[r['id'] for r in rows],bank=[serial_action(a) for a in bank],
        geometry_sha256=hashlib.sha256((ASSETS/'geometry/geometry.npz').read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['router_readiness.py','expanded_assets.py','experiment.py','pilot_engine.py']})
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Changed design; use a new run')
    atomic_json(out/'manifest.json',manifest)
    engine=PilotEngine(proto['model']);atomic_json(out/'model.json',engine.meta)
    execution=RecordedExecution(engine,out,proto);execution.execute(rows,bank)
    assert all((r['id'],a['id']) in execution.records for r in rows for a in bank)
    atomic_json(out/'complete.json',dict(samples=len(rows),configs=len(bank),records=len(execution.records)))
    print('READINESS COMPLETE',args.mode,args.task,flush=True)

if __name__=='__main__':main()

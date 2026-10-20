"""Extend the frozen nine-node geometry while preserving its vectors exactly."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiment import ROOT,LAB,PilotEngine
from engine import atomic_json
from sample_assets import GEOMETRY
from expanded_concepts import CONCEPTS,ADDITIONS,probes

DEFAULT=ROOT/'runs/expanded-assets-v2'

def fit(pooled,examples,old,scale):
    by={(p['concept'],p['context'],p['pole']):pooled[i] for i,p in enumerate(examples)}
    contrasts=np.stack([[by[c['id'],i,'positive']-by[c['id'],i,'negative'] for i in range(10)] for c in ADDITIONS])
    raw=contrasts[:,:8].mean(axis=1)
    unit=raw/np.linalg.norm(raw,axis=1,keepdims=True)
    geometry={k:np.concatenate([old[k],v],axis=0) for k,v in [('raw',raw),('unit',unit),('directions',unit*scale)]}
    # Direction coordinates, rather than the unrelated original response-activation PCA.
    centered=geometry['unit']-geometry['unit'].mean(axis=0)
    u,s,vt=np.linalg.svd(centered.astype(np.float64),full_matrices=False)
    geometry.update(scale=np.asarray(scale),cosine=geometry['unit']@geometry['unit'].T,
                    coordinates=(u[:,:3]*s[:3]).astype(np.float32),projection_components=vt[:3],
                    projection_variance=s[:3]**2/(s**2).sum())
    _,_,basis=np.linalg.svd(old['unit'].astype(np.float64),full_matrices=False)
    outside=unit.astype(np.float64)-(unit@basis.T)@basis
    stability=[]
    for i,c in enumerate(ADDITIONS):
        cv=contrasts[i]/np.linalg.norm(contrasts[i],axis=1,keepdims=True)
        stability.append(dict(concept=c['id'],fit_context_cosines=(cv[:8]@unit[i]).tolist(),
            diagnostic_context_cosines=(cv[8:]@unit[i]).tolist(),
            fraction_outside_original_span=float(np.linalg.norm(outside[i])),
            nearest_original_cosine=float(np.max(unit[i]@old['unit'].T))))
    return geometry,stability

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);args=p.parse_args()
    out=args.run/'geometry';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('GEOMETRY ALREADY COMPLETE',flush=True);return
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text());examples=probes()
    engine=PilotEngine(protocol['model'])
    pooled,last=engine.measure([x['question'] for x in examples],[x['response'] for x in examples],batch_size=8)
    old=dict(np.load(GEOMETRY/'geometry.npz'));scale=json.loads((GEOMETRY/'atlas.json').read_text())['scale']
    geo,stability=fit(pooled,examples,old,scale)
    assert all(np.array_equal(geo[k][:9],old[k]) for k in ['raw','unit','directions'])
    np.savez_compressed(out/'geometry.npz',**geo)
    np.savez_compressed(out/'probe_activations.npz',pooled=pooled,last=last)
    atomic_json(out/'probes.json',examples);atomic_json(out/'model.json',engine.meta)
    atomic_json(out/'geometry.json',dict(concepts=CONCEPTS,scale=scale,model=protocol['model'],
        fitting='Mean positive-minus-negative assistant-response activations over eight independent authored contexts; normalize and scale to the original intervention norm',
        benchmark_answers_used=False,original_nodes_preserved_bit_exact=True,original_geometry_sha256=hashlib.sha256((GEOMETRY/'geometry.npz').read_bytes()).hexdigest(),
        original_source=str(GEOMETRY),new_probe_count=len(examples),fit_contexts=list(range(8)),diagnostic_contexts=[8,9],
        projection='PCA of centered unit direction vectors; only visualization, intervention uses all 1024 dimensions',
        projection_variance=geo['projection_variance'].tolist(),diagnostics=stability,
        geometry_sha256=hashlib.sha256((out/'geometry.npz').read_bytes()).hexdigest()))
    atomic_json(out/'complete.json',dict(nodes=27,dimensions=1024,new_probes=360,original_preserved=True))
    print('GEOMETRY COMPLETE',len(CONCEPTS),'nodes; projection variance',geo['projection_variance'].sum(),flush=True)

if __name__=='__main__':main()

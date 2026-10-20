#!/usr/bin/env python3
"""Build real geometry and replayable knob responses from local pinned weights."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from concepts import AXES,CONTEXTS,SCENARIOS,STRENGTHS,public_axes
from engine import SteeringEngine,atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"data")
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--layer",type=int,default=11)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"atlas.json").exists():
        raise SystemExit("Choose a new output directory; existing measurement is preserved")
    engine=SteeringEngine(args.device,args.layer)
    examples=[]
    for axis in AXES:
        for side,key in [("negative","low"),("positive","high")]:
            for i,(prompt,text) in enumerate(zip(CONTEXTS,axis[key])):
                examples.append(dict(id=f"{axis['id']}-{side}-{i}",axis=axis["id"],side=side,prompt=prompt,text=text))
    vectors=engine.encode_responses(examples)
    directions={}
    axes=public_axes()
    for axis in axes:
        low=np.stack([v for v,e in zip(vectors,examples) if e["axis"]==axis["id"] and e["side"]=="negative"])
        high=np.stack([v for v,e in zip(vectors,examples) if e["axis"]==axis["id"] and e["side"]=="positive"])
        diffs=high-low
        direction=diffs.mean(0)
        raw_norm=float(np.linalg.norm(direction))
        if raw_norm < 1e-8:raise ValueError("Degenerate concept direction")
        directions[axis["id"]]=direction*engine.metadata["scale"]
        axis["direction_norm"]=raw_norm
        axis["pair_cosine_mean"]=float(np.mean(diffs@direction/(np.linalg.norm(diffs,axis=1)*raw_norm)))
        axis["pair_count"]=len(diffs)
    centered=vectors-vectors.mean(0)
    _,sv,vt=np.linalg.svd(centered,full_matrices=False)
    xyz=centered@vt[:3].T
    # Fix signs deterministically; use no behavior score or persona classifier.
    for d in range(3):
        if vt[d,np.argmax(abs(vt[d]))] < 0:
            vt[d]*=-1;xyz[:,d]*=-1
    sphere=xyz/np.maximum(np.linalg.norm(xyz,axis=1,keepdims=True),1e-9)
    points=[dict(**e,xyz=v.tolist(),radius=float(np.linalg.norm(raw))) for e,v,raw in zip(examples,sphere,xyz)]
    np.savez_compressed(args.output/"directions.npz",**directions)
    np.savez_compressed(args.output/"activations.npz",vectors=vectors,pca_basis=vt[:3],mean=vectors.mean(0))
    atlas=dict(schema="iota-concept-geometry-v1",measured_at=datetime.now(timezone.utc).isoformat(),
               model=engine.metadata,axes=axes,points=points,scenarios=SCENARIOS,strengths=STRENGTHS,
               projection=dict(method="centered PCA 3D, radial normalization to unit sphere",
                               explained_variance_ratio=(sv[:3]**2/(sv**2).sum()).tolist(),
                               distances="projection and normalization distort native distances"),
               scope="Human-selected contrasts in one model and one layer; not global semantic identity.",
               directions_sha256=hashlib.sha256((args.output/"directions.npz").read_bytes()).hexdigest())
    atomic_json(args.output/"atlas.json",atlas)
    print("Atlas:",len(points),"measured points; PCA",atlas["projection"]["explained_variance_ratio"],flush=True)
    responses=[]
    for scenario in SCENARIOS:
        # Reuse exactly the same baseline across all axes.
        baseline=engine.generate_batch(scenario["prompt"],[np.zeros(vectors.shape[1],dtype=np.float32)])[0]
        for axis in axes:
            strengths=[s for s in STRENGTHS if s]
            generated=engine.generate_batch(scenario["prompt"],[s*directions[axis["id"]] for s in strengths])
            records=dict(zip(strengths,generated));records[0]=baseline
            for strength in STRENGTHS:
                responses.append(dict(axis=axis["id"],scenario=scenario["id"],strength=strength,
                    **records[strength],generation="greedy",max_new_tokens=96,
                    same_as_baseline=records[strength]["text"]==baseline["text"]))
            atomic_json(args.output/"responses.json",dict(model=engine.metadata,responses=responses,
                         mode="recorded activation steering; no weight update",complete=False))
            print(scenario["id"],axis["id"],[(s,records[s]["text"][:35]) for s in strengths],flush=True)
    atomic_json(args.output/"responses.json",dict(model=engine.metadata,responses=responses,
                 mode="recorded activation steering; no weight update",complete=True))

if __name__=="__main__":main()

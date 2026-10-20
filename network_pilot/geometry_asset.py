"""Geometry engine: high-level contrastive representation directions, independent of task labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from sample_assets import ROOT,LAB,GEOMETRY,DEFAULT
from concepts import CONCEPTS,probe_examples
from engine import atomic_json


def fit_directions(pooled,probes):
    by={(p['concept'],p['context'],p['pole']):pooled[i] for i,p in enumerate(probes)}
    raw=np.stack([np.mean([by[c['id'],i,'positive']-by[c['id'],i,'negative'] for i in range(8)],axis=0) for c in CONCEPTS])
    unit=raw/np.linalg.norm(raw,axis=1,keepdims=True)
    scale=float(.20*np.median(np.linalg.norm(pooled,axis=1)))
    return dict(raw=raw,unit=unit,directions=unit*scale,scale=scale)


def export(run):
    source=np.load(GEOMETRY/'probe_activations.npz');probes=json.loads((GEOMETRY/'probes.json').read_text())
    fitted=fit_directions(source['pooled'],probes);original=np.load(GEOMETRY/'geometry.npz')
    errors={key:float(np.max(np.abs(fitted[key]-original[key]))) for key in ['raw','unit','directions']}
    if any(v!=0 for v in errors.values()):raise ValueError('Cannot reproduce original fitted geometry')
    bundle=run/'asset_bundle';bundle.mkdir(parents=True,exist_ok=True)
    for name in ['probe_activations.npz','probes.json','atlas.json']:shutil.copy2(GEOMETRY/name,bundle/name)
    (bundle/'geometry_asset.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(bundle/'geometry_construction_audit.json',dict(
        exact_reconstruction=True,max_absolute_errors=errors,scale=fitted['scale'],
        nodes=len(CONCEPTS),hidden_dimensions=fitted['directions'].shape[1],probe_examples=len(probes),
        contexts_used_to_fit=list(range(8)),diagnostic_contexts=[8,9],
        node_source='Human-authored high-level positive/negative behavior descriptions measured inside the frozen model',
        task_answers_used=False,formula='d_k = mean_context(h_positive - h_negative); normalize each d_k, scale to 0.2 * median probe residual norm',
        geometric_similarity=(fitted['unit']@fitted['unit'].T).tolist(),
        sha256={name:hashlib.sha256((GEOMETRY/name).read_bytes()).hexdigest() for name in ['probe_activations.npz','probes.json','geometry.npz','atlas.json']}))
    print('GEOMETRY AUDIT',errors,'; 9 high-level directions reproduced exactly')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--measure-output',type=Path)
    args=p.parse_args()
    if args.measure_output:
        from pilot_engine import PilotEngine
        protocol=json.loads((ROOT/'inputs/protocol.json').read_text());out=args.measure_output
        out.mkdir(parents=True,exist_ok=True)
        engine=PilotEngine(protocol['model']);probes=probe_examples()
        pooled,last=engine.measure([p['question'] for p in probes],[p['response'] for p in probes])
        fitted=fit_directions(pooled,probes)
        np.savez_compressed(out/'probe_activations.npz',pooled=pooled,last=last)
        np.savez_compressed(out/'geometry.npz',**fitted)
        atomic_json(out/'probes.json',probes);atomic_json(out/'model.json',engine.meta)
    else:export(args.run)


if __name__=='__main__':main()

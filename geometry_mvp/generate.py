#!/usr/bin/env python3
"""Reuse a direction exported by the atlas with the exact pinned local model."""
import argparse
import json
from pathlib import Path
import numpy as np
from engine import SteeringEngine,atomic_json,MODEL,REVISION
from concepts import SYSTEM,STRENGTHS

def load_config(path):
    c=json.loads(Path(path).read_text())
    if c.get('schema')!='iota-steering-config-v1' or c.get('method')!='activation_steering':
        raise ValueError('Unknown configuration format')
    if c.get('model')!=MODEL or c.get('revision')!=REVISION or c.get('system_prompt')!=SYSTEM:
        raise ValueError('Configuration belongs to a different model or prompt protocol')
    if isinstance(c.get('strength'),bool) or c.get('strength') not in STRENGTHS:
        raise ValueError('Invalid strength')
    if isinstance(c.get('baseline_strength',0),bool) or c.get('baseline_strength',0) not in STRENGTHS:
        raise ValueError('Invalid baseline strength')
    vector=np.asarray(c['delta_per_strength_unit'],dtype=np.float32)
    if vector.shape!=(c['hidden_size'],) or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-8:
        raise ValueError('Invalid direction vector')
    return c,vector

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--prompt',required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--output',type=Path)
    args=p.parse_args();config,vector=load_config(args.config)
    if not args.prompt.strip() or len(args.prompt)>1200:raise ValueError('Invalid prompt length')
    engine=SteeringEngine(args.device,config['layer_index'])
    if config['hidden_size']!=engine.model.config.hidden_size:raise ValueError('Hidden size mismatch')
    base,steered=engine.generate_batch(args.prompt,[config.get('baseline_strength',0)*vector,config['strength']*vector])
    result=dict(prompt=args.prompt,baseline=base,steered=steered,config=config,model=engine.metadata)
    if args.output:atomic_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

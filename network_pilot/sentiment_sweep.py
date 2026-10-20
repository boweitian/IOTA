"""Complete the frozen action bank on every item; exploratory feasibility, not routing."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
from sentiment_prepare import DEFAULT, write, jsonl, sha
from sentiment_navigator import bank, read, digest, ROOT

OUT = ROOT/'runs/sentiment9b-sst5-direction-sweep-v1'


def prepare(source, out, batch_size):
    out.mkdir(parents=True, exist_ok=True)
    manifest=json.loads((source/'dataset-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(source/name)!=expected:raise ValueError('Source binding changed: '+name)
    for name in ['dataset-manifest.json','geometry-binding.json','search-geometry.npz','search-atlas.json',
                 'search-protocol.json','train.jsonl','test-inputs.jsonl','test-labels.jsonl']:
        target=out/name
        if target.exists() and sha(target)!=sha(source/name):raise ValueError('Changed sweep input: '+name)
        if not target.exists():shutil.copy2(source/name,target)
    original=json.loads((source/'experiment-spec.json').read_text())
    protocol=json.loads((out/'search-protocol.json').read_text())
    atlas=json.loads((out/'search-atlas.json').read_text())
    directions=np.load(out/'search-geometry.npz')['directions']
    actions=bank(directions,float(atlas['scale']))
    serial=[{k:v for k,v in action.items() if k!='shift'} for action in actions]
    if serial!=original['actions']:raise ValueError('Candidate bank changed')
    # Cached scoring is reusable only with the exact original scorer and prompts.
    if sha(ROOT/'sentiment_engine.py')!=original['source_sha256']['sentiment_engine.py']:
        raise ValueError('Scoring code changed')
    spec=dict(source=str(source.resolve()),source_spec_sha256=digest(original),
              source_policies_sha256=sha(source/'frozen-policies.json'),model=protocol['objective'],
              geometry_sha256=sha(out/'search-geometry.npz'),actions=serial,batch_size=batch_size,
              n=100,train_count=70,test_count=30,
              scoring='FP32; TF32 disabled; block 16 output; last prompt token only; next-token A-E softmax',
              purpose='Post-hoc all-item feasibility and signed dose-response exploration. No learned router, no sealed new test.',
              source_sha256={name:sha(ROOT/name) for name in ['sentiment_sweep.py','sentiment_sweep_analysis.py','sentiment_engine.py']})
    path=out/'sweep-spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Sweep specification changed')
    write(path,spec)
    return spec,actions,protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DEFAULT)
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args();spec,actions,protocol=prepare(args.source,args.out,args.batch_size)
    fingerprint=digest(spec);engine=None
    source_summary=json.loads((args.source/'summary.json').read_text())
    def status(phase,**extra):
        write(args.out/'status.json',dict(phase=phase,updated_at=time.time(),spec_sha256=fingerprint,**extra))
    for split,filename in [('train','train.jsonl'),('test','test-inputs.jsonl')]:
        # Holdout labels are not needed to fill the response grid.
        inputs=read(args.out/filename);texts=[row['text'] for row in inputs]
        cache=args.out/(split+'-cache');cache.mkdir(exist_ok=True)
        for j,action in enumerate(actions):
            target=cache/f'agent-{j:02d}.npz'
            if target.exists():
                with np.load(target) as row:
                    if str(row['sweep_spec_sha256'])!=fingerprint:raise ValueError('Stale sweep cache')
                continue
            origin=args.source/(split+'-cache')/target.name
            status('all_direction_measurements',split=split,completed=j,total=len(actions),agent=action['id'])
            if origin.exists():
                with np.load(origin) as row:
                    key='spec_sha256' if split=='train' else 'policies_sha256'
                    expected=spec['source_spec_sha256'] if split=='train' else spec['source_policies_sha256']
                    if str(row[key])!=expected:raise ValueError('Unbound source cache')
                    if split=='train' and row['ids'].tolist()!=[r['id'] for r in inputs]:raise ValueError('Wrong source item order')
                    if split=='test' and j not in source_summary['test_agents_measured']:raise ValueError('Unmeasured source action')
                    values={k:row[k].copy() for k in ['probabilities','logits','features','label_mass']}
                reuse=dict(source=str(origin.resolve()),sha256=sha(origin))
            else:
                if engine is None:
                    from sentiment_engine import SentimentEngine
                    engine=SentimentEngine(protocol['objective'],np.load(args.out/'search-geometry.npz')['directions'])
                    meta=dict(engine.meta,intervention='block 16 output residual, last input token in a single full-prefix forward; no decoding rollout')
                    old=json.loads((args.source/'model.json').read_text())
                    for key in ['model','revision','layer_index','hidden_size','dtype','torch','transformers','chat_template_sha256','system_prompt']:
                        if meta[key]!=old[key]:raise ValueError('Scoring environment changed: '+key)
                    write(args.out/'model.json',meta)
                    zero=np.zeros(engine.dim)
                    baseline=engine.score(texts[:args.batch_size],zero)['probabilities']
                    expected=np.load(args.source/'test-cache/agent-00.npz')['probabilities'][:args.batch_size]
                    error=float(np.max(np.abs(baseline-expected)))
                    audit=dict(max_baseline_probability_error=error,predictions_agree=bool(np.array_equal(baseline.argmax(1),expected.argmax(1))),passed=bool(error<1e-3 and np.array_equal(baseline.argmax(1),expected.argmax(1))))
                    write(args.out/'numerical-audit.json',audit)
                    if not audit['passed']:raise ValueError('Source baseline could not be reproduced')
                parts=[engine.score(texts[lo:lo+args.batch_size],action['shift']) for lo in range(0,len(texts),args.batch_size)]
                values={k:np.concatenate([r[k] for r in parts]) for k in ['probabilities','logits','features','label_mass']}
                reuse=None
            if values['probabilities'].shape!=(len(inputs),5):raise ValueError('Incomplete response matrix')
            temp=target.with_suffix('.tmp.npz')
            np.savez_compressed(temp,**values,ids=np.array([r['id'] for r in inputs]),sweep_spec_sha256=fingerprint,
                                provenance=json.dumps(reuse or dict(measured_at=time.time())))
            temp.replace(target)
            print(split,j+1,'/',len(actions),action['id'],'reused' if reuse else 'measured',flush=True)
    from sentiment_sweep_analysis import analyze
    analyze(args.out)
    status('complete',measurements=100*len(actions),actions=len(actions))


if __name__=='__main__':main()

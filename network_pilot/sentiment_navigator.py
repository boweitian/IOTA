"""Frozen Search geometry -> training-only agent organization -> sealed SST-5 holdout."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from sentiment_prepare import DEFAULT,LAB,LABELS,sha,write,jsonl
from sentiment_models import train,predict,accuracy,nll
sys.path.insert(0,str(LAB))
from studio import mixed_shift,AXES
from engine import atomic_json

ROOT=Path(__file__).resolve().parent

def read(path):return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def bank(directions,scale):
    vectors=[('baseline',np.zeros(len(AXES)))]
    for i,key in enumerate(AXES):
        for dose in [-2.,-1.,1.,2.]:
            values=np.zeros(len(AXES));values[i]=dose;vectors.append((f'{key}:{dose:+g}',values))
    rng=np.random.default_rng(20261019)
    for k in range(12):
        values=np.zeros(len(AXES));ix=rng.choice(len(AXES),size=2+k%2,replace=False)
        values[ix]=rng.choice([-1.,1.],size=len(ix))*rng.choice([.5,1.,1.5],size=len(ix))
        vectors.append((f'combination:{k:02d}',values))
    actions=[]
    for name,values in vectors:
        profile=dict(coefficients=dict(zip(AXES,values)))
        shift,info=mixed_shift(profile,directions,scale*max(1.,np.max(np.abs(values))),regularization='direction_projection')
        actions.append(dict(id=name,coefficients=values.tolist(),shift=shift,intervention=info))
    return actions

def metric(p,y):
    predicted=p.argmax(1)
    matrix=np.zeros((5,5),dtype=int)
    for gold,guess in zip(y,predicted):matrix[gold,guess]+=1
    return dict(accuracy=accuracy(p,y),correct=int(np.sum(predicted==y)),n=len(y),nll=nll(p,y),confusion=matrix.tolist())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--batch-size',type=int,default=4);args=parser.parse_args()
    out=args.out;manifest=json.loads((out/'dataset-manifest.json').read_text())
    for name,value in manifest['files'].items():
        if sha(out/name)!=value:raise ValueError('Frozen dataset or geometry changed: '+name)
    atlas=json.loads((out/'search-atlas.json').read_text());protocol=json.loads((out/'search-protocol.json').read_text())
    geometry=np.load(out/'search-geometry.npz');directions=geometry['directions'];scale=float(atlas['scale'])
    if [c['id'] for c in atlas['concepts']]!=AXES or directions.shape!=(9,4096):raise ValueError('Wrong geometry axes or dimension')
    if not np.array_equal(directions,np.array([c['native_direction'] for c in atlas['concepts']])):raise ValueError('Atlas and geometry vectors differ')
    actions=bank(directions,scale);serial=[{key:value for key,value in action.items() if key!='shift'} for action in actions]
    source_files=['sentiment_navigator.py','sentiment_engine.py','sentiment_models.py','sentiment_prepare.py']
    spec=dict(model=protocol['objective'],geometry_sha256=sha(out/'search-geometry.npz'),dataset_sha256=sha(out/'dataset-manifest.json'),
        train_count=70,test_count=30,labels=LABELS,actions=serial,batch_size=args.batch_size,precision='FP32; TF32 disabled',
        regularization='direction_projection',training='5-fold stratified CV within the 70 training items',
        source_sha256={name:sha(ROOT/name) for name in source_files},
        phases=['score only training items','fit and freeze all policies','predict sealed test inputs','score once against test labels'])
    spec_hash=digest(spec)
    if (out/'experiment-spec.json').exists() and json.loads((out/'experiment-spec.json').read_text())!=spec:raise ValueError('Changed experiment: use a new directory')
    write(out/'experiment-spec.json',spec)
    def status(phase,**extras):atomic_json(out/'status.json',dict(phase=phase,updated_at=time.time(),spec_sha256=spec_hash,**extras))
    if (out/'summary.json').exists():print('Completed result already exists:',out);return
    status('loading_model')
    from sentiment_engine import SentimentEngine
    engine=SentimentEngine(protocol['objective'],directions)
    write(out/'model.json',dict(**engine.meta,label_token_ids=engine.label_tokens))
    training=read(out/'train.jsonl');y=np.array([r['label'] for r in training]);texts=[r['text'] for r in training]
    if not (out/'numerical-audit.json').exists():
        batched=engine.score(texts[:args.batch_size],np.zeros(4096))
        singles=[engine.score([text],np.zeros(4096)) for text in texts[:args.batch_size]]
        one=np.concatenate([row['probabilities'] for row in singles])
        error=float(np.max(np.abs(batched['probabilities']-one)))
        audit=dict(max_probability_error=error,predictions_agree=bool(np.array_equal(one.argmax(1),batched['probabilities'].argmax(1))),
            passed=bool(error<1e-3 and np.array_equal(one.argmax(1),batched['probabilities'].argmax(1))),precision=engine.meta['likelihood_precision'])
        write(out/'numerical-audit.json',audit)
        if not audit['passed']:raise ValueError('Numerical consistency audit failed')
    cache=out/'train-cache';cache.mkdir(exist_ok=True)
    for j,action in enumerate(actions):
        path=cache/f'agent-{j:02d}.npz'
        if path.exists():continue
        status('training_measurements',completed=j,total=len(actions),agent=action['id'])
        parts=[engine.score(texts[lo:lo+args.batch_size],action['shift']) for lo in range(0,len(texts),args.batch_size)]
        np.savez_compressed(path,**{key:np.concatenate([row[key] for row in parts]) for key in ['probabilities','logits','features','label_mass']},
            spec_sha256=spec_hash,ids=np.array([row['id'] for row in training]),seconds=sum(row['seconds'] for row in parts))
        print('TRAIN',j+1,'/',len(actions),action['id'],flush=True)
    rows=[]
    for j in range(len(actions)):
        row=np.load(cache/f'agent-{j:02d}.npz')
        if str(row['spec_sha256'])!=spec_hash:raise ValueError('Stale measurement cache')
        rows.append(row)
    p=np.stack([row['probabilities'] for row in rows],axis=1);x=rows[0]['features']
    reference=[next(i for i,a in enumerate(actions) if a['id']==key) for key in ['precision:+1','verification:+1','warmth:+1']]
    status('training_policy_selection',agents=len(actions))
    policy_path=out/'frozen-policies.json'
    if policy_path.exists():
        frozen=json.loads(policy_path.read_text())
        if frozen['spec_sha256']!=spec_hash:raise ValueError('Stale frozen policies')
    else:
        fitted=train(p,x,y,reference)
        fitted.update(spec_sha256=spec_hash,frozen_at=time.time(),train_ids=[r['id'] for r in training],
            train_metrics={name:metric(predict(policy,p,x),y) for name,policy in fitted['policies'].items()})
        write(policy_path,fitted);frozen=fitted
    frozen_hash=sha(policy_path)
    print('POLICIES FROZEN',frozen['selected'],frozen_hash,flush=True)
    # No test inputs or test labels have been read before this point.
    testing=read(out/'test-inputs.jsonl');test_text=[row['text'] for row in testing]
    needed=sorted({i for policy in frozen['policies'].values() for i in policy['members']}|{0})
    test_p=np.full((len(testing),len(actions),5),1/5.,dtype=float);test_x=None
    test_cache=out/'test-cache';test_cache.mkdir(exist_ok=True)
    for number,j in enumerate(needed):
        status('test_predictions',completed=number,total=len(needed),agent=actions[j]['id'],policies_sha256=frozen_hash)
        path=test_cache/f'agent-{j:02d}.npz'
        if not path.exists():
            parts=[engine.score(test_text[lo:lo+args.batch_size],actions[j]['shift']) for lo in range(0,len(testing),args.batch_size)]
            np.savez_compressed(path,**{key:np.concatenate([row[key] for row in parts]) for key in ['probabilities','logits','features','label_mass']},policies_sha256=frozen_hash)
        row=np.load(path)
        if str(row['policies_sha256'])!=frozen_hash:raise ValueError('Test predictions belong to different policies')
        test_p[:,j]=row['probabilities']
        if j==0:test_x=row['features']
        print('TEST',number+1,'/',len(needed),actions[j]['id'],flush=True)
    predictions={name:predict(policy,test_p,test_x) for name,policy in frozen['policies'].items()}
    # Save all predictions before opening the test-label file.
    prediction_rows=[dict(id=row['id'],text=row['text'],policies={name:dict(prediction=int(values[i].argmax()),probabilities=values[i].tolist()) for name,values in predictions.items()}) for i,row in enumerate(testing)]
    jsonl(out/'test-predictions.jsonl',prediction_rows)
    if sha(policy_path)!=frozen_hash:raise ValueError('Policies changed during holdout inference')
    gold=read(out/'test-labels.jsonl');labels={row['id']:row['label'] for row in gold};test_y=np.array([labels[row['id']] for row in testing])
    results={name:metric(values,test_y) for name,values in predictions.items()}
    train_agents=[dict(index=i,id=actions[i]['id'],**metric(p[:,i],y)) for i in range(len(actions))]
    geometry_unit=directions/np.linalg.norm(directions,axis=1,keepdims=True)
    summary=dict(complete=True,benchmark=manifest['benchmark'],model=engine.meta,train_count=70,test_count=30,
        labels=LABELS,geometry_sha256=spec['geometry_sha256'],protocol_sha256=protocol['sha256'],
        selected=frozen['selected'],cv=frozen['cv'],train_metrics=frozen['train_metrics'],test_metrics=results,
        policies=frozen['policies'],actions=serial,train_agents=train_agents,test_agents_measured=needed,
        policies_sha256=frozen_hash,predictions_sha256=sha(out/'test-predictions.jsonl'),spec_sha256=spec_hash,
        geometry=dict(nodes=[dict(id=c['id'],name=c['name'],certified=c.get('certified'),xyz=c['direction']) for c in atlas['concepts']],gram=(geometry_unit@geometry_unit.T).tolist()),
        numerical_audit=json.loads((out/'numerical-audit.json').read_text()),
        guarantee='Finite training-selected candidates only; 30-item holdout is a pilot, no claim of globally optimal performance or statistical reliability.',
        measurement='Restricted next-token A-E classification; frozen 9B weights, same prompt, no LLM judge, no test-label oracle routing.')
    write(out/'summary.json',summary)
    jsonl(out/'test-results.jsonl',[dict(**row,gold=int(test_y[i])) for i,row in enumerate(prediction_rows)])
    status('complete',selected=frozen['selected'],test_accuracy=results[frozen['selected']]['accuracy'])
    print(json.dumps(dict(selected=frozen['selected'],test_metrics=results),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

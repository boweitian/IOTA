"""Train-only router improvement, historical diagnostics and new blind replay."""
import argparse
import itertools
import json
import shutil
import time
import traceback
from pathlib import Path

import joblib
import numpy as np

import fixed_advantage_models as models
import gsm_router_large as core
import state_geometry_router as state
import state_geometry_models as old_models

ROOT=core.ROOT
SOURCE=state.DEFAULT
DEFAULT=ROOT/'runs/fixed-advantage-router-v1'
SUITES=('gsm8k','bbh','ifeval')
METHODS=('baseline','fixed','previous','improved')


def history():
    files=sorted(set(ROOT.glob('runs/**/protocol.json'))|set(ROOT.glob('runs/*/inputs.json'))|
                 {core.LAB/'inputs/manifest.json',ROOT/'inputs/protocol.json'})
    ids=set();questions=set();inventory={}
    for file in files:
        d=json.loads(file.read_text());records=d if isinstance(d,list) else d.get('records',d.get('samples',[]))
        if not isinstance(records,list):continue
        inventory[str(file)]=core.sha(file)
        for row in records:
            if not isinstance(row,dict):continue
            if 'id' in row:ids.add(str(row['id']))
            q=row.get('question',row.get('prompt',''))
            if isinstance(q,str) and q.strip():questions.add(q.strip())
    return ids,questions,inventory


def prepare(out):
    assert not out.exists()
    ids,texts,inventory=history()
    import benchmark_router_transfer as other
    pools={};sources={}
    gsm_path=core.LAB/'inputs/gsm8k_train.jsonl'
    expected=json.loads((state.SOURCES['gsm8k']/'protocol.json').read_text())['source']['sha256']
    assert core.sha(gsm_path)==expected
    pools['gsm8k']=[dict(id=f'gsm8k:train:{i}',suite='gsm8k',task='gsm8k',question=r['question'],
        target=r['answer'].split('####')[-1].strip(),source_index=i)
        for i,r in enumerate(core.read(gsm_path)) if f'gsm8k:train:{i}' not in ids and r['question'].strip() not in texts]
    sources['gsm8k']=[dict(local_path=str(gsm_path),sha256=expected)]
    for suite in ('bbh','ifeval'):
        pool,src,_=other.sources_and_pool(suite)
        pools[suite]=[r for r in pool if r['id'] not in ids and r['question'].strip() not in texts]
        sources[suite]=src
    assert len(pools['gsm8k'])>=600
    out.mkdir(parents=True);snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['fixed_advantage_router.py','fixed_advantage_models.py','fixed_advantage_router.sbatch',
                 'state_geometry_router.py','state_geometry_models.py','gsm_router_large.py',
                 'pilot_engine.py','router_benchmark_scoring.py','fit_asset_router.py','space.py']:
        shutil.copy2(ROOT/name,snapshot/name)
    for name in ['engine.py','concepts.py','grading.py']:shutil.copy2(core.LAB/name,snapshot/name)
    hashes={p.name:core.sha(p) for p in snapshot.iterdir()};fresh_counts={}
    for suite in SUITES:
        dest=out/suite;dest.mkdir();source=SOURCE/suite
        p=json.loads((source/'protocol.json').read_text())
        for filename in ['actions.npz','actions.json','train_matrix.npz']:
            shutil.copy2(source/filename,dest/filename)
        count=min(len(pools[suite]),{'gsm8k':600,'bbh':150,'ifeval':40}[suite])
        pool=pools[suite];order=np.random.default_rng(models.SEED).permutation(len(pool))[:count]
        fresh=[dict(pool[i],split='fresh_test',eval_group='fresh_v2') for i in order]
        assert len({r['question'].strip() for r in fresh})==count
        assert all(r['id'] not in ids and r['question'].strip() not in texts for r in fresh)
        fresh_counts[suite]=count
        core.save(dest/'protocol.json',dict(model=p['model'],records=fresh,shards=6,batch_size=64,
            max_new_tokens=p['max_new_tokens'],benchmark=suite,methods=list(METHODS),
            source_run=str(source),sources=sources[suite],source_protocol_sha256=core.sha(source/'protocol.json'),
            actions_sha256=core.sha(dest/'actions.npz'),train_matrix_sha256=core.sha(dest/'train_matrix.npz'),
            n_train=p['n_train'],n_test=count,source_hashes=hashes))
    core.save(out/'design.json',dict(created_utc=state.now(),source_hashes=hashes,
        historical_inventory=inventory,fresh_counts=fresh_counts,seed=models.SEED,
        primary='improved versus training-best fixed on 600 fresh GSM8K questions',
        secondary='improved versus frozen previous geometry router and no-op on same matched cohorts',
        training='Original exhaustive train rows only: GSM8K 700, BBH 378, IFEval 350.',
        candidates=dict(kernels=list(models.KINDS)+['hybrid_task for BBH'],alpha=list(models.ALPHAS),
            damage_penalty=list(models.PENALTIES),gates=[None if np.isinf(t) else t for t in models.THRESHOLDS],
            supported_actions=list(models.TOP_COUNTS)),
        selection='Three task-stratified inner folds, fold-local text vocabularies, priors, fixed direction and action shortlist; accuracy then fewer breaks and switches.',
        heads='joint repair event and joint damage event relative to train-fixed direction; every action success/failure retained',
        previous_router='state-geometry-router-v1 geometry_raw; frozen weights',
        historical_holdouts_are_exploratory=True,new_fresh_labels_not_used_for_selection=True,
        limitations=['27 fresh BBH and 5 fresh IFEval rows are only small supplemental checks, not strong replications.',
                     'Fresh GSM8K rows come from pinned official train source, not official test split.',
                     'OOF selection scores select hyperparameters and are not an independent evaluation.',
                     'Bounded ridge head scores are not calibrated probabilities.']))
    print('PREPARED fresh counts',fresh_counts,flush=True)


def validate(out):
    design=json.loads((out/'design.json').read_text())
    for name,digest in design['source_hashes'].items():
        path=ROOT/name if (ROOT/name).exists() else core.LAB/name
        assert core.sha(path)==digest,f'Source drift: {name}'
    for suite in SUITES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text())
        assert core.sha(d/'actions.npz')==p['actions_sha256']
        assert core.sha(d/'train_matrix.npz')==p['train_matrix_sha256']
        for source in p['sources']:assert core.sha(Path(source['local_path']))==source['sha256']


def fit(out):
    validate(out)
    for suite in SUITES:
        d=out/suite
        if (d/'fit_complete.json').exists():continue
        source=SOURCE/suite;p=json.loads((source/'protocol.json').read_text())
        features,_=state.load_features(source,p);rows=p['records'];train=np.array([r['split']=='train' for r in rows])
        labels=np.load(d/'train_matrix.npz');y=labels['reward'].astype(float)
        assert labels['ids'].astype(str).tolist()==[r['id'] for r in rows if r['split']=='train']
        training=models.public([r for r in rows if r['split']=='train']);h=models.take(features,train)
        config,trials,folds=models.select(training,h,y)
        core.save(d/'selection.json',dict(selected=config,trials=trials,folds=folds.tolist(),
            uses_only_original_training_rows=True,all_action_outcomes_supervise=True))
        bundle=models.fit(training,h,y,config)
        bundle['actions']=np.load(d/'actions.npz')['ids'].astype(str)
        bundle['training_ids']=labels['ids'].astype(str);bundle['training_protocol_sha256']=core.sha(source/'protocol.json')
        joblib.dump(bundle,d/'router.joblib',compress=3)
        restored=joblib.load(d/'router.joblib')
        decisions,confidence=models.predict(restored,models.public(rows),features)
        assert np.array_equal(decisions[train],models.predict(bundle,training,h)[0])
        matrix=np.load(ROOT/'runs/router-direction-atlas-v1'/suite/'full_matrix.npz')
        assert matrix['ids'].astype(str).tolist()==[r['id'] for r in rows]
        groups={}
        old_bundle=joblib.load(source/'router-geometry.joblib')
        previous_choices=old_models.choose(old_models.predict(old_bundle['model'],old_models.hidden_input(features,old_bundle['kind'])))
        for name,index in [('train',np.flatnonzero(train)),('historical',np.flatnonzero(~train))]:
            reward=matrix['reward'][index];n=len(index);chosen=decisions[index]
            groups[name]=dict(n=n,baseline=int(reward[:,0].sum()),fixed=int(reward[:,bundle['fixed']].sum()),
                previous=int(reward[np.arange(n),previous_choices[index]].sum()),
                improved=int(reward[np.arange(n),chosen].sum()),switches=int((chosen!=bundle['fixed']).sum()),
                coverage=int((reward.max(1)==1).sum()),
                repairs_vs_fixed=int(((reward[:,bundle['fixed']]==0)&(reward[np.arange(n),chosen]==1)).sum()),
                breaks_vs_fixed=int(((reward[:,bundle['fixed']]==1)&(reward[np.arange(n),chosen]==0)).sum()))
        core.save(d/'historical_diagnostics.json',dict(groups=groups,config=config,
            role='Historical test observations are diagnostic only; not used in model or threshold selection.'))
        core.save(d/'fit_complete.json',dict(utc=state.now(),model_sha256=core.sha(d/'router.joblib'),
            selected=config,all_rows_prediction_reload_match=True))
        print('FIT',suite,json.dumps(dict(selected=config,groups=groups)),flush=True)
    core.save(out/'policies_frozen.json',dict(utc=state.now(),models={s:core.sha(out/s/'router.joblib') for s in SUITES}))


def decisions(d,rows,features):
    bundle=joblib.load(d/'router.joblib');improved,confidence=models.predict(bundle,models.public(rows),features)
    old=joblib.load(SOURCE/d.name/'router-geometry.joblib')
    previous=old_models.choose(old_models.predict(old['model'],old_models.hidden_input(features,old['kind'])))
    actions=bundle['actions'];fixed=bundle['fixed']
    return [dict(id=r['id'],baseline=str(actions[0]),fixed=str(actions[fixed]),previous=str(actions[previous[i]]),
        improved=str(actions[improved[i]]),score_advantage=float(confidence[i])) for i,r in enumerate(rows)]


def worker(out,shard):
    try:
        validate(out);frozen=json.loads((out/'policies_frozen.json').read_text())
        for suite in SUITES:
            d=out/suite;p=json.loads((d/'protocol.json').read_text())
            assert core.sha(d/'router.joblib')==frozen['models'][suite]
            rows=p['records'][shard::6];folder=d/'test'/f'shard-{shard}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():continue
            if not rows:
                core.save(folder/'complete.json',dict(records=0,utc=state.now()));continue
            e=core.engine(p);core.save(folder/'model.json',e.meta)
            path=folder/'features.npz'
            if path.exists():features=dict(np.load(path))
            else:
                parts={k:[] for k in ['steer_last','final_last','final_mean']};cohorts=[]
                for lo in range(0,len(rows),p['batch_size']):
                    f,meta=state.capture_batch(e,rows[lo:lo+p['batch_size']]);cohorts.append(meta)
                    for k,v in f.items():parts[k].append(v)
                features={k:np.concatenate(v) for k,v in parts.items()};np.savez_compressed(path,**features)
                core.save(folder/'feature_cohorts.json',dict(cohorts=cohorts,sha256=core.sha(path)))
            chosen=decisions(d,rows,features)
            if (folder/'frozen_actions.json').exists():assert chosen==json.loads((folder/'frozen_actions.json').read_text())
            else:core.save(folder/'frozen_actions.json',chosen)
            by={r['id']:r for r in chosen};bank=np.load(d/'actions.npz');shifts=dict(zip(bank['ids'].astype(str),bank['shifts']))
            seen={(r['id'],r['method']) for r in core.read(folder/'records.jsonl')}
            for method in METHODS:
                for lo in range(0,len(rows),p['batch_size']):
                    chunk=rows[lo:lo+p['batch_size']]
                    if all((r['id'],method) in seen for r in chunk):continue
                    outputs=e.generate([core.question(r) for r in chunk],
                        [shifts[by[r['id']][method]] for r in chunk],max_new_tokens=p['max_new_tokens'])
                    for row,g in zip(chunk,outputs):
                        if (row['id'],method) in seen:continue
                        scored=core.grade(g['text'],row)
                        core.append(folder/'records.jsonl',dict(id=row['id'],method=method,action=by[row['id']][method],
                            batch_ids=[r['id'] for r in chunk],reward=float(scored['correct']),**scored,**g))
                        seen.add((row['id'],method))
                print('TEST',suite,shard,method,len(seen),flush=True)
            core.save(folder/'complete.json',dict(records=len(seen),utc=state.now()))
        if shard==0:
            state.wait_for(out,lambda:all((out/s/'test'/f'shard-{k}/complete.json').exists() for s in SUITES for k in range(6)))
            finish(out)
    except BaseException:
        core.save(out/f'failure-{shard}.json',dict(utc=state.now(),error=traceback.format_exc()));raise


def finish(out):
    validate(out);result={}
    frozen=json.loads((out/'policies_frozen.json').read_text())
    for suite in SUITES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text());rows=p['records'];by={r['id']:r for r in rows};records={};checks=0
        assert core.sha(d/'router.joblib')==frozen['models'][suite]
        for shard in range(6):
            folder=d/'test'/f'shard-{shard}';assert (folder/'complete.json').exists();local=rows[shard::6]
            if not local:continue
            feature_meta=json.loads((folder/'feature_cohorts.json').read_text());assert core.sha(folder/'features.npz')==feature_meta['sha256']
            assert all(c['generated_tokens']==0 for c in feature_meta['cohorts'])
            expected=[[r['id'] for r in local[lo:lo+p['batch_size']]] for lo in range(0,len(local),p['batch_size'])]
            assert [c['ids'] for c in feature_meta['cohorts']]==expected
            features=dict(np.load(folder/'features.npz'));chosen=decisions(d,local,features)
            assert chosen==json.loads((folder/'frozen_actions.json').read_text());choices={r['id']:r for r in chosen}
            assert (folder/'frozen_actions.json').stat().st_mtime<=(folder/'records.jsonl').stat().st_mtime
            cohorts={r['id']:[q['id'] for q in local[lo:lo+p['batch_size']]] for lo in range(0,len(local),p['batch_size']) for r in local[lo:lo+p['batch_size']]}
            for r in core.read(folder/'records.jsonl'):
                key=r['id'],r['method'];assert key not in records;assert r['action']==choices[r['id']][r['method']]
                assert r['batch_ids']==cohorts[r['id']]
                assert len(r['token_ids'])==r['new_tokens']<=p['max_new_tokens']
                scored=core.grade(r['text'],by[r['id']]);assert float(scored['correct'])==r['reward']
                if suite=='ifeval':assert scored['instruction_results']==r['instruction_results']
                records[key]=r
        assert len(records)==len(rows)*len(METHODS)
        for row in rows:
            for a,b in itertools.combinations(METHODS,2):
                left,right=records[row['id'],a],records[row['id'],b]
                if left['action']==right['action']:assert left['token_ids']==right['token_ids'];checks+=1
        rewards={m:np.array([records[r['id'],m]['reward'] for r in rows]) for m in METHODS}
        result[suite]=dict(n=len(rows),correct={m:int(v.sum()) for m,v in rewards.items()},
            paired={m:state.paired(rewards['improved'],rewards[m]) for m in METHODS if m!='improved'},
            same_action_token_checks=checks,all_outputs_regraded=True,all_policy_choices_reloaded=True)
    core.save(out/'results.json',result);core.save(out/'complete.json',dict(verified=True,utc=state.now()))
    print('VERIFIED FINAL',json.dumps(result),flush=True)


def status(out):
    result=dict(utc=state.now(),complete=(out/'complete.json').exists(),suites={},failures=[])
    for suite in SUITES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text())
        result['suites'][suite]=dict(fit=(d/'fit_complete.json').exists(),n_test=p['n_test'],
            shards=sum((d/'test'/f'shard-{k}/complete.json').exists() for k in range(6)))
    result['failures']=[json.loads(f.read_text()) for f in out.glob('failure-*.json')]
    core.save(out/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','run','finish','status']);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--shard',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='fit':fit(a.out)
    elif a.mode=='run':worker(a.out,a.shard)
    elif a.mode=='finish':finish(a.out)
    else:status(a.out)

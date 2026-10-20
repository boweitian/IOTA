"""Build question-to-direction and direction-to-question indices from observed evidence."""
import argparse
import collections
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
from sample_assets import DEFAULT,geometry,exhaustive_bank
from engine import atomic_json


def generation_groups(source):
    records=list(map(json.loads,source.read_text().splitlines()))
    groups=collections.defaultdict(list)
    for r in records:
        if r.get('generated'):groups[r['action'],r['batch_seconds']].append(r['id'])
    inferred=set();manifest=source.parent/'manifest.json'
    if manifest.exists():
        ids=json.loads(manifest.read_text())['ids']
        native=collections.defaultdict(dict)
        for r in records:
            if r.get('generated') and not r.get('reused'):
                native[r['action']].setdefault(r['batch_seconds'],r['batch_size'])
        for name,batches in native.items():
            seen={r['id'] for r in records if r['action']==name and r.get('reused')}
            for seconds,size in batches.items():
                key=(name,seconds);saved=groups[key]
                if len(saved)<size:
                    # A grading exception writes a prefix of a generated batch. Reconstruct
                    # its inputs from immutable row ordering and the already saved candidates.
                    pending=[ident for ident in ids if ident not in seen]
                    cohort=pending[:size]
                    if saved!=cohort[:len(saved)]:raise ValueError('Cannot infer interrupted batch from protocol ordering')
                    groups[key]=cohort;inferred.add(key)
                seen.update(saved)
    return groups,inferred


def build(run):
    bundle=run/'asset_bundle'
    items=[json.loads(line) for line in (bundle/'sample_direction_dictionary.jsonl').read_text().splitlines()]
    catalog={a['id']:a for a in json.loads((bundle/'actions.json').read_text())}
    bank=exhaustive_bank(geometry());ids=[a['id'] for a in bank]
    reward=np.full((len(items),len(bank)),np.nan);success=np.zeros_like(reward,dtype=bool);observed=np.zeros_like(success)
    inverse=collections.defaultdict(lambda:dict(successful_samples=[],improved_samples=[],rescued_samples=[]))
    regions=collections.defaultdict(list)
    for i,item in enumerate(items):
        evidence=item.get('evidence',{});successful=set(item.get('successful_directions',[]))
        for j,name in enumerate(ids):
            if name in evidence:
                reward[i,j]=evidence[name]['reward'];success[i,j]=name in successful;observed[i,j]=True
        for name in successful:
            inverse[name]['successful_samples'].append(item['id'])
            if item.get('baseline_correct') is False:inverse[name]['rescued_samples'].append(item['id'])
        for name in item.get('improving_directions',[]):inverse[name]['improved_samples'].append(item['id'])
        if item['status']=='complete':
            # Group by actual response signature, including all successes, rather than by predicted direction.
            key=(item['suite'],tuple(name for name in ids if name in successful))
            regions[key].append(item['id'])
    np.savez_compressed(bundle/'response_matrix.npz',reward=reward,success=success,observed=observed,
        sample_ids=np.asarray([r['id'] for r in items]),action_ids=np.asarray(ids))
    atomic_json(bundle/'direction_sample_index.json',dict(
        definition='Inverse index of actual successful / improving / rescuing per-question interventions',
        directions={name:dict(configuration=catalog[name],**values) for name,values in sorted(inverse.items())}))
    profiles={}
    for suite in sorted({r['suite'] for r in items}):
        ix=[i for i,r in enumerate(items) if r['suite']==suite]
        entries=[]
        for j,name in enumerate(ids):
            valid=[i for i in ix if observed[i,j] and observed[i,0]]
            entries.append(dict(action=name,measured_samples=len(valid),
                mean_reward=float(np.mean(reward[valid,j])) if valid else None,
                mean_gain_from_baseline=float(np.mean(reward[valid,j]-reward[valid,0])) if valid else None,
                success_count=int(success[valid,j].sum()),
                rescued_samples=[items[i]['id'] for i in valid if not items[i].get('baseline_correct',True) and success[i,j]]
                    if suite!='writingbench' else None))
        profiles[suite]=dict(samples=[items[i]['id'] for i in ix],directions=entries,
            best_assignment_counts=dict(collections.Counter(items[i].get('best_action') for i in ix)),
            interpretation='Empirical response map of these measured samples; individual configurations are preserved, never averaged into a purported successful vector')
    atomic_json(bundle/'benchmark_direction_profiles.json',profiles)
    # Preserve exact generation cohorts: BF16 greedy outputs can depend on batch shape.
    cohorts={};source_groups={}
    for path in run.glob('*/shard-*/records.jsonl'):
        for r in map(json.loads,path.read_text().splitlines()):
            if not r.get('generated'):continue
            source=Path(r.get('source_file',str(path)));source_action=r.get('source_action',r['action'])
            key=(str(source),source_action,r['batch_seconds'])
            if source not in source_groups:
                source_groups[source]=generation_groups(source)
            groups,inferred=source_groups[source];batch=groups[source_action,r['batch_seconds']]
            if len(batch)!=r['batch_size']:raise ValueError('Cannot reconstruct original generation batch')
            digest=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
            entry=cohorts.setdefault(digest,dict(source_file=str(source),source_action=source_action,
                batch_seconds=r['batch_seconds'],sample_ids=batch,batch_size=r['batch_size'],evidence_edges=[]))
            entry['input_cohort_inferred_after_grading_interruption']=(source_action,r['batch_seconds']) in inferred
            entry['evidence_edges'].append([r['id'],r['action']])
    atomic_json(bundle/'execution_batches.json',dict(
        definition='Original generation cohorts for replay; BF16 batch shape can change greedy outputs',
        cohorts=cohorts,
        replay_note='Four objective-task assets were checked; one GSM8K direction passed with its original batch and failed with batch size 1. See replay evidence.'))
    atomic_json(bundle/'observed_subproblem_signatures.json',dict(
        definition='Groups of measured questions with the same successful steering settings in the declared 37-configuration space',
        groups=[dict(suite=key[0],successful_directions=list(key[1]),samples=samples,
            signature_sha256=hashlib.sha256(json.dumps(key).encode()).hexdigest())
            for key,samples in sorted(regions.items())]))
    # A portable, queryable asset containing all measured directions and raw evidence.
    temporary=bundle/'sample_assets.sqlite.tmp'
    if temporary.exists():temporary.unlink()
    with sqlite3.connect(temporary) as db:
        db.executescript('''
        CREATE TABLE samples (sample_id TEXT PRIMARY KEY, suite TEXT, question TEXT, prompt TEXT,
            baseline_reward REAL, best_reward REAL, baseline_correct INTEGER, solved INTEGER, status TEXT);
        CREATE TABLE directions (direction_id TEXT PRIMARY KEY, configuration_json TEXT);
        CREATE TABLE evaluations (sample_id TEXT, direction_id TEXT, reward REAL, successful INTEGER,
            improving INTEGER, evidence_json TEXT, PRIMARY KEY(sample_id,direction_id));
        CREATE INDEX successful_direction_samples ON evaluations(direction_id,successful);
        ''')
        db.executemany('INSERT INTO directions VALUES (?,?)',[(name,json.dumps(a,ensure_ascii=False)) for name,a in catalog.items()])
        for item in items:
            db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)',(item['id'],item['suite'],item['question'],item['prompt'],
                item['baseline_reward'],item['best_reward'],item['baseline_correct'],item['solved'],item['status']))
            db.executemany('INSERT INTO evaluations VALUES (?,?,?,?,?,?)',[(item['id'],name,r['reward'],
                name in item['successful_directions'],name in item['improving_directions'],json.dumps(r,ensure_ascii=False))
                for name,r in item['evidence'].items()])
    temporary.replace(bundle/'sample_assets.sqlite')
    files={str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in bundle.rglob('*') if p.is_file() and p.name!='index.json'}
    atomic_json(bundle/'index.json',dict(schema_version=2,samples=len(items),
        complete=sum(r['status']=='complete' for r in items),declared_settings=len(bank),
        successful_edges=sum(len(r.get('successful_directions',[])) for r in items),
        primary_assets=['geometry.npz','geometry.json','sample_direction_dictionary.jsonl','direction_sample_index.json','response_matrix.npz','sample_assets.sqlite','execution_batches.json'],files=files))
    print('INDEXED',len(items),'samples;',sum(len(r.get('successful_directions',[])) for r in items),'successful edges')


def query(run,sample,export=None,direction=None):
    bundle=run/'asset_bundle'
    item=next((json.loads(line) for line in (bundle/'sample_direction_dictionary.jsonl').read_text().splitlines()
               if json.loads(line)['id']==sample),None)
    if item is None:raise ValueError('Unknown sample')
    selected=direction or item['best_action'];catalog={a['id']:a for a in json.loads((bundle/'actions.json').read_text())}
    print(json.dumps({k:v for k,v in item.items() if k!='evidence'},ensure_ascii=False,indent=2))
    print(json.dumps(item['evidence'][selected],ensure_ascii=False,indent=2))
    if export:
        geo=json.loads((bundle/'geometry.json').read_text());a=catalog[selected]
        # Store exact coefficient contraction without adding another normalization operation.
        delta=np.asarray(a['coefficients'])@geometry()['directions']
        evidence=item['evidence'][selected];cohort=[]
        if evidence.get('generated'):
            source=Path(evidence['source_file']) if evidence.get('source_file') else next(
                p for p in run.glob('*/shard-*/records.jsonl')
                if any(r['id']==sample and r['action']==selected for r in map(json.loads,p.read_text().splitlines())))
            source_action=evidence.get('source_action',selected)
            groups,_=generation_groups(source)
            cohort_ids=groups[source_action,evidence['batch_seconds']]
            inputs={r['id']:r for r in json.loads((bundle/'inputs.json').read_text())['records']}
            from experiment import prompt
            cohort=[dict(id=ident,prompt=prompt(inputs[ident])) for ident in cohort_ids]
            assert len(cohort)==evidence['batch_size']
        atomic_json(export,dict(model=geo['model'],sample_id=sample,question=item['question'],prompt=item['prompt'],
            action=a,shift=delta.tolist(),intervention=geo['intervention'],evidence=evidence,
            generation_batch=cohort))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT)
    p.add_argument('--sample');p.add_argument('--direction');p.add_argument('--export',type=Path)
    args=p.parse_args()
    if args.sample:query(args.run,args.sample,args.export,args.direction)
    elif args.direction:
        index=json.loads((args.run/'asset_bundle/direction_sample_index.json').read_text())
        print(json.dumps(index['directions'][args.direction],ensure_ascii=False,indent=2))
    else:build(args.run)


if __name__=='__main__':main()

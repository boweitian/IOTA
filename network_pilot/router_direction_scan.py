"""Post-hoc exhaustive held-out direction measurements for Navigator diagnostics.

Policies remain frozen. This answer-aware sweep measures finite-bank oracles and
must never be reported as blind router accuracy or used to retune this router.
"""
import argparse
import datetime
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

import gsm_router_large as core

ROOT=core.ROOT
SOURCE=ROOT/'runs/state-geometry-router-v1'
DEFAULT=ROOT/'runs/router-direction-atlas-v1'
SUITES=('gsm8k','bbh','ifeval')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(out):
    assert json.loads((SOURCE/'complete.json').read_text())['verified']
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['router_direction_scan.py','router_direction_scan.sbatch','gsm_router_large.py',
                 'pilot_engine.py','router_benchmark_scoring.py','space.py']:
        shutil.copy2(ROOT/name,snapshot/name)
    for name in ['engine.py','concepts.py','grading.py']:shutil.copy2(core.LAB/name,snapshot/name)
    hashes={p.name:core.sha(p) for p in snapshot.iterdir()}
    for suite in SUITES:
        d=out/suite;d.mkdir();source=SOURCE/suite
        p=json.loads((source/'protocol.json').read_text())
        for name in ['actions.json','actions.npz','train_matrix.npz']:
            shutil.copy2(source/name,d/name)
        core.save(d/'protocol.json',dict(p,source_state_run=str(source),
             purpose='Post-hoc exhaustive test-set finite-bank oracle diagnostic; frozen policies, no fitting.',
             expected_test_cells=p['n_test']*109,source_hashes=hashes))
    core.save(out/'design.json',dict(utc=now(),source_run=str(SOURCE),source_hashes=hashes,
         expected_test_cells=99408,policies_frozen=True,answer_aware_diagnostic=True,
         not_a_new_blind_benchmark=True,all_configs_remeasured_with_matched_cohorts=True))
    status(out)


def validate(out):
    p=json.loads((out/'design.json').read_text())
    for name,digest in p['source_hashes'].items():
        path=ROOT/name if (ROOT/name).exists() else core.LAB/name
        assert core.sha(path)==digest,f'Source drift {name}'
    for suite in SUITES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text())
        assert core.sha(d/'actions.npz')==p['actions_sha256']
        assert core.sha(d/'train_matrix.npz')==p['train_matrix_sha256']


def sweep(out,suite,shard):
    d=out/suite;p=json.loads((d/'protocol.json').read_text())
    rows=[r for r in p['records'] if r['split']!='train'][shard::p['shards']]
    folder=d/'scan'/f'shard-{shard}';folder.mkdir(parents=True,exist_ok=True)
    seen={(r['id'],r['action']) for r in core.read(folder/'records.jsonl')}
    bank=np.load(d/'actions.npz');ids=bank['ids'].astype(str);shifts=bank['shifts']
    if len(seen)==len(rows)*len(ids) and (folder/'complete.json').exists():return
    e=core.engine(p);core.save(folder/'model.json',e.meta)
    core.save(folder/'cohorts.json',dict(ids=[r['id'] for r in rows],batch_size=p['batch_size'],
              protocol_sha256=core.sha(d/'protocol.json')))
    for action,shift in zip(ids,shifts):
        for lo in range(0,len(rows),p['batch_size']):
            chunk=rows[lo:lo+p['batch_size']]
            if all((r['id'],action) in seen for r in chunk):continue
            generated=e.generate([core.question(r) for r in chunk],[shift]*len(chunk),
                                 max_new_tokens=p['max_new_tokens'])
            for row,g in zip(chunk,generated):
                if (row['id'],action) in seen:continue
                scored=core.grade(g['text'],row)
                core.append(folder/'records.jsonl',dict(id=row['id'],action=str(action),
                    split=row['split'],eval_group=row['eval_group'],batch_ids=[r['id'] for r in chunk],
                    reward=float(scored['correct']),**scored,**g))
                seen.add((row['id'],action))
        core.save(folder/'progress.json',dict(records=len(seen),required=len(rows)*len(ids),last_action=str(action)))
        print('SCAN',suite,shard,action,len(seen),'/',len(rows)*len(ids),flush=True)
    assert len(seen)==len(rows)*len(ids)
    core.save(folder/'complete.json',dict(records=len(seen),utc=now()))


def assemble(out,suite):
    d=out/suite;p=json.loads((d/'protocol.json').read_text());rows=p['records'];by={r['id']:r for r in rows}
    index={r['id']:i for i,r in enumerate(rows)};actions=np.load(d/'actions.npz')['ids'].astype(str)
    aix={a:j for j,a in enumerate(actions)}
    matrix=np.full((len(rows),len(actions)),-1,dtype=np.int8)
    previous=np.load(d/'train_matrix.npz')
    for i,ident in enumerate(previous['ids'].astype(str)):matrix[index[ident]]=previous['reward'][i]
    tested=[r for r in rows if r['split']!='train'];metadata=[];records=0
    for shard in range(p['shards']):
        folder=d/'scan'/f'shard-{shard}';assert (folder/'complete.json').exists()
        metadata.append(json.loads((folder/'model.json').read_text()))
        local=tested[shard::p['shards']]
        cohorts={r['id']:[q['id'] for q in local[lo:lo+p['batch_size']]]
                 for lo in range(0,len(local),p['batch_size']) for r in local[lo:lo+p['batch_size']]}
        for r in core.read(folder/'records.jsonl'):
            row=by[r['id']];assert row['split']!='train'
            i,j=index[r['id']],aix[r['action']];assert matrix[i,j]==-1
            assert r['batch_ids']==cohorts[r['id']]
            assert len(r['token_ids'])==r['new_tokens']<=p['max_new_tokens']
            scored=core.grade(r['text'],row);assert float(scored['correct'])==r['reward']
            if suite=='ifeval':assert scored['instruction_results']==r['instruction_results']
            matrix[i,j]=r['reward'];records+=1
    assert (matrix>=0).all() and records==p['expected_test_cells']
    np.savez_compressed(d/'full_matrix.npz',ids=np.array([r['id'] for r in rows]),actions=actions,reward=matrix)
    core.save(d/'verification.json',dict(passed=True,utc=now(),test_records=records,
         all_test_outputs_regraded=True,unique_complete_grid=True,exact_cohorts=True,
         gpu_per_shard=[m['device'] for m in metadata],matrix_sha256=core.sha(d/'full_matrix.npz')))
    core.save(d/'complete.json',dict(utc=now(),verified=True))
    print('VERIFIED',suite,records,flush=True)


def wait_for(out,predicate):
    until=time.monotonic()+4*3600
    while not predicate():
        failures=list(out.glob('failure-*.json'))
        if failures:raise RuntimeError(failures[0].read_text())
        if time.monotonic()>until:raise TimeoutError('peer barrier')
        time.sleep(10)


def run(out,shard):
    try:
        validate(out)
        for suite in SUITES:
            sweep(out,suite,shard)
            if shard==0:
                wait_for(out,lambda:all((out/suite/'scan'/f'shard-{k}/complete.json').exists() for k in range(6)))
                assemble(out,suite)
        if shard==0:
            core.save(out/'complete.json',dict(utc=now(),verified=True));status(out)
    except BaseException:
        core.save(out/f'failure-{shard}.json',dict(utc=now(),error=traceback.format_exc()));raise


def status(out):
    result=dict(utc=now(),complete=(out/'complete.json').exists(),suites={},failures=[])
    for suite in SUITES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text());counts=[]
        for shard in range(6):
            f=d/'scan'/f'shard-{shard}'/'complete.json'
            if not f.exists():f=f.with_name('progress.json')
            counts.append(json.loads(f.read_text())['records'] if f.exists() else 0)
        result['suites'][suite]=dict(records=sum(counts),required=p['expected_test_cells'],
             shards_done=sum((d/'scan'/f'shard-{k}/complete.json').exists() for k in range(6)),complete=(d/'complete.json').exists())
    result['failures']=[json.loads(p.read_text()) for p in out.glob('failure-*.json')]
    core.save(out/'status.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','run','status','assemble'])
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--shard',type=int,choices=range(6))
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.out)
    elif args.mode=='run':run(args.out,args.shard)
    elif args.mode=='status':status(args.out)
    else:
        for suite in SUITES:assemble(args.out,suite)

"""Fresh single-benchmark GSM8K steering assets and a 700/300 router experiment."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent/'steering_lab'
DEFAULT=ROOT/'runs/gsm-router-1000-v1'
SEED=20261016
SHARDS=6


def save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append(path,row):
    with path.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def prepare(out,batch):
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((LAB/'inputs/manifest.json').read_text())
    original=json.loads((ROOT/'inputs/protocol.json').read_text())
    source=LAB/'inputs/gsm8k_train.jsonl'
    expected=next(s for s in manifest['sources'] if s['task']=='gsm8k' and s['part']=='train')
    assert sha(source)==expected['sha256']
    old=manifest['records']+original['records']
    used_ids={r['id'] for r in old}
    used_questions={r['question'].strip() for r in old}
    pool=[]
    for i,row in enumerate(read(source)):
        ident=f'gsm8k:train:{i}'
        if ident in used_ids or row['question'].strip() in used_questions:continue
        pool.append(dict(id=ident,suite='gsm8k',task='gsm8k',question=row['question'],
                         target=row['answer'].split('####')[-1].strip(),source_index=i))
    rng=np.random.default_rng(SEED)
    sample=[pool[i] for i in rng.permutation(len(pool))[:1000]]
    for i,r in enumerate(sample):r['split']='train' if i<700 else 'test'
    assert len({r['question'].strip() for r in sample})==1000
    from expanded_assets import bank_for
    asset=ROOT/'runs/expanded-assets-v2'
    bank=bank_for(dict(np.load(asset/'geometry/geometry.npz')))
    np.savez_compressed(out/'actions.npz',ids=np.array([a['id'] for a in bank]),
                        shifts=np.stack([a['shift'] for a in bank]))
    from space import serial_action
    save(out/'actions.json',[serial_action(a) for a in bank])
    protocol=dict(model={k:manifest[k] for k in ['model','model_revision','model_path','layer_index']},
         benchmark='GSM8K',seed=SEED,samples=1000,n_train=700,n_test=300,shards=SHARDS,
         batch_size=batch,max_new_tokens=512,positions='last',records=sample,source=expected,
         geometry_sha256=sha(asset/'geometry/geometry.npz'),actions_sha256=sha(out/'actions.npz'),
         prior_sample_ids_disjoint=True,prior_questions_disjoint=True,
         design='Train exhaustively on 700 fresh GSM8K questions x 109 configs. '
                'Fit linear routers on successful-action sets and majority-success labels. '
                'Predict before test generation; matched test cohorts for baseline, fixed, set and majority routers.',
         primary='majority-success linear text router vs no-op',
         secondary=['set-label linear router','training-best fixed action'],
         target_rule='per-question successful actions; choose most train-supported successful action for majority labels',
         continuous_tasks=False,router_hyperparameters=dict(regularization=1e-6,maxiter=300,
                                                          train_margin_stop=.5),
         budget_note='512-token cap frozen for this new experiment; all methods share the cap. Prior 36-question run had 1024.',
         no_test_answer_use_for_router_or_action_selection=True)
    save(out/'protocol.json',protocol)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['gsm_router_large.py','pilot_engine.py','expanded_assets.py','space.py','fit_asset_router.py','simple_router.py']:
        shutil.copy2(ROOT/name,snapshot/name)
    for name in ['engine.py','concepts.py','grading.py']:shutil.copy2(LAB/name,snapshot/name)
    print('PREPARED',len(sample),'train',700,'test',300,'configs',len(bank),flush=True)


def engine(protocol):
    from pilot_engine import PilotEngine
    return PilotEngine(protocol['model'])


def question(row):
    if row.get('suite')=='ifeval':
        return row['question']
    import sys
    sys.path.insert(0,str(LAB))
    from concepts import task_prompt
    return task_prompt(row['question'])


def grade(text,row):
    if row.get('suite')=='ifeval':
        from router_benchmark_scoring import score_ifeval
        scored=score_ifeval(row,text)
        return dict(correct=bool(scored['reward']),
                    instruction_results=scored['instruction_results'])
    import sys
    sys.path.insert(0,str(LAB))
    from grading import grade as score
    return score(text,row)


def train_measure(out,shard):
    p=json.loads((out/'protocol.json').read_text());assert 0<=shard<p['shards']
    assert sha(out/'actions.npz')==p['actions_sha256']
    rows=[r for r in p['records'] if r['split']=='train'][shard::p['shards']]
    d=out/'train'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=True)
    seen={(r['id'],r['action']) for r in read(d/'records.jsonl')}
    bank=np.load(out/'actions.npz');ids=bank['ids'].astype(str);shifts=bank['shifts']
    if len(seen)==len(rows)*len(ids):return
    e=engine(p);save(d/'model.json',e.meta)
    save(d/'cohorts.json',dict(ids=[r['id'] for r in rows],batch_size=p['batch_size'],
                              protocol_sha256=sha(out/'protocol.json')))
    for a,shift in zip(ids,shifts):
        for lo in range(0,len(rows),p['batch_size']):
            chunk=rows[lo:lo+p['batch_size']]
            if all((r['id'],a) in seen for r in chunk):continue
            # On resume regenerate the whole original cohort, preserving batch shape.
            outputs=e.generate([question(r) for r in chunk],[shift]*len(chunk),max_new_tokens=p['max_new_tokens'])
            cohort=[r['id'] for r in chunk]
            for row,g in zip(chunk,outputs):
                if (row['id'],a) in seen:continue
                scored=grade(g['text'],row)
                append(d/'records.jsonl',dict(id=row['id'],action=a,split='train',batch_ids=cohort,
                     reward=float(scored['correct']),**scored,**g))
                seen.add((row['id'],a))
        save(d/'progress.json',dict(records=len(seen),required=len(rows)*len(ids),last_action=a))
        print('TRAIN',shard,a,len(seen),'/',len(rows)*len(ids),flush=True)
    assert all((r['id'],a) in seen for r in rows for a in ids)
    save(d/'complete.json',dict(samples=len(rows),configs=len(ids),records=len(seen)))


def fit_routers(out):
    import joblib
    from fit_asset_router import input_text,vectorizer,fit
    p=json.loads((out/'protocol.json').read_text())
    rows=[r for r in p['records'] if r['split']=='train'];test=[r for r in p['records'] if r['split']=='test']
    records={}
    for k in range(p['shards']):
        d=out/'train'/f'shard-{k}'
        assert (d/'complete.json').exists(),f'Incomplete training shard {k}'
        for r in read(d/'records.jsonl'):
            assert (r['id'],r['action']) not in records
            records[r['id'],r['action']]=r
    actions=np.load(out/'actions.npz')['ids'].astype(str)
    y=np.array([[records[r['id'],a]['reward'] for a in actions] for r in rows])
    assert y.shape==(p['n_train'],len(actions))
    np.savez_compressed(out/'train_matrix.npz',ids=np.array([r['id'] for r in rows]),actions=actions,reward=y)
    valid=y==1;covered=valid.any(1);counts=valid.sum(0)
    majority=np.zeros_like(valid)
    teacher=np.where(valid,counts[None,:],-1).argmax(1)
    majority[np.flatnonzero(covered),teacher[covered]]=True
    transform=vectorizer()
    x=transform.fit_transform([input_text(r['suite'],r['question']) for r in rows])
    xt=transform.transform([input_text(r['suite'],r['question']) for r in test])
    predictions={};training={}
    for name,target in [('set',valid),('majority',majority)]:
        trace=[]
        require_full_fit=not (name=='majority' and p.get('allow_incomplete_secondary_fit',False))
        w,bias=fit(x[covered],target[covered],trace,require_full_fit=require_full_fit)
        train_choice=np.asarray(x@w+bias).argmax(1)
        hits=valid[np.arange(len(rows)),train_choice]
        if require_full_fit:assert hits[covered].all()
        choice=np.asarray(xt@w+bias).argmax(1)
        joblib.dump(dict(transform=transform,w=w,bias=bias,actions=actions,training_ids=[r['id'] for r in rows]),
                    out/f'router-{name}.joblib',compress=3)
        save(out/f'training-{name}.json',trace)
        predictions[name]=choice
        training[name]=dict(covered=int(covered.sum()),fitted=int(hits.sum()),iterations=len(trace),
                            train_correct=int(hits.sum()),train_total=len(rows))
    fixed=int(y.mean(0).argmax())
    decisions=[dict(id=r['id'],baseline='none',fixed=str(actions[fixed]),
                   set=str(actions[predictions['set'][i]]),majority=str(actions[predictions['majority'][i]]))
               for i,r in enumerate(test)]
    save(out/'frozen_test_actions.json',decisions)
    save(out/'training_summary.json',dict(training=training,train_baseline=float(y[:,0].mean()),
         train_oracle=float(y.max(1).mean()),fixed_action=str(actions[fixed]),
         support=[dict(action=str(a),correct=int(c)) for a,c in zip(actions,counts)],
         majority_target_counts=[dict(action=str(actions[j]),count=int((teacher[covered]==j).sum()))
                                 for j in np.unique(teacher[covered])],
         frozen_test_actions_sha256=sha(out/'frozen_test_actions.json')))
    print('ROUTERS FIT; TEST ACTIONS FROZEN',flush=True)


def test_measure(out,shard):
    p=json.loads((out/'protocol.json').read_text())
    rows=[r for r in p['records'] if r['split']=='test'][shard::p['shards']]
    decisions={r['id']:r for r in json.loads((out/'frozen_test_actions.json').read_text())}
    summary=json.loads((out/'training_summary.json').read_text())
    assert summary['frozen_test_actions_sha256']==sha(out/'frozen_test_actions.json')
    bank=np.load(out/'actions.npz');shifts=dict(zip(bank['ids'].astype(str),bank['shifts']))
    d=out/'test'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=True)
    seen={(r['id'],r['method']) for r in read(d/'records.jsonl')}
    if len(seen)==len(rows)*4 and (d/'complete.json').exists():return
    e=engine(p);save(d/'model.json',e.meta)
    for method in ['baseline','fixed','set','majority']:
        for lo in range(0,len(rows),p['batch_size']):
            chunk=rows[lo:lo+p['batch_size']]
            if all((r['id'],method) in seen for r in chunk):continue
            outputs=e.generate([question(r) for r in chunk],
                 [shifts[decisions[r['id']][method]] for r in chunk],max_new_tokens=p['max_new_tokens'])
            for row,g in zip(chunk,outputs):
                if (row['id'],method) in seen:continue
                scored=grade(g['text'],row)
                append(d/'records.jsonl',dict(id=row['id'],method=method,action=decisions[row['id']][method],
                     split='test',batch_ids=[r['id'] for r in chunk],reward=float(scored['correct']),**scored,**g))
                seen.add((row['id'],method))
        print('TEST',shard,method,len(seen),flush=True)
    assert len(seen)==4*len(rows)
    save(d/'complete.json',dict(samples=len(rows),records=len(seen)))


def finish(out):
    from scipy.stats import binomtest
    p=json.loads((out/'protocol.json').read_text());rows=[r for r in p['records'] if r['split']=='test']
    records={}
    for k in range(p['shards']):
        d=out/'test'/f'shard-{k}';assert (d/'complete.json').exists()
        for r in read(d/'records.jsonl'):records[r['id'],r['method']]=r
    baseline=np.array([records[r['id'],'baseline']['reward'] for r in rows])
    results={}
    for method in ['baseline','fixed','set','majority']:
        selected=[records[r['id'],method] for r in rows]
        score=np.array([r['reward'] for r in selected]);delta=score-baseline
        rng=np.random.default_rng(SEED)
        boot=delta[rng.integers(0,len(rows),(10000,len(rows)))].mean(1)
        fixes=int(((score==1)&(baseline==0)).sum());breaks=int(((score==0)&(baseline==1)).sum())
        results[method]=dict(n=len(rows),correct=int(score.sum()),accuracy=float(score.mean()),
             delta_pp=float(delta.mean()*100),delta_95_pp=(np.percentile(boot,[2.5,97.5])*100).tolist(),
             fixed_wrong=fixes,broke_right=breaks,mcnemar_exact_p=float(binomtest(fixes,fixes+breaks).pvalue) if fixes+breaks else 1.,
             truncated=sum(r['truncated'] for r in selected),mean_tokens=float(np.mean([r['new_tokens'] for r in selected])))
    save(out/'results.json',dict(benchmark='GSM8K',n_train=700,n_test=300,results=results,
         protocol_sha256=sha(out/'protocol.json'),decisions_sha256=sha(out/'frozen_test_actions.json')))
    training=json.loads((out/'training_summary.json').read_text())
    lines=['# 单 Benchmark：1,000 题 GSM8K Router 实验','',
         '1,000 道此前实验未用过的 GSM8K 题，固定随机切分 700 训练 / 300 测试。训练逐题遍历 109 配置；Router 只用训练题的文本与成功方向标签。主方法为多数成功方向优先，另保存集合标签 Router 与训练题选固定方向对照。', '',
         '同一冻结 Qwen3.5-0.8B，第 15 层，生成上限 512 tokens。测试决策在生成前冻结；四种方法使用完全相同的测试输入批次。以下为实际新生成的结果。', '',
         '| 方法 | 正确/300 | 准确率 | 相对 baseline（百分点） | 截断数 |',
         '|---|---:|---:|---:|---:|']
    for name,r in results.items():lines.append(f"| {name} | {r['correct']}/300 | {100*r['accuracy']:.2f}% | {r['delta_pp']:+.2f} | {r['truncated']} |")
    r=results['majority']
    lines += ['',f"多数优先 Router 修复 {r['fixed_wrong']} 道原错题，损坏 {r['broke_right']} 道原对题；配对 bootstrap 95% 增益区间 {r['delta_95_pp']} 个百分点，McNemar 双侧精确 p={r['mcnemar_exact_p']:.4f}。",'',
         f"训练题 baseline {100*training['train_baseline']:.2f}%，109 配置逐题覆盖 {100*training['train_oracle']:.2f}%。训练中的有效标签题全部拟合成功。没有找到成功方向的测试题也全部计入分母。",'',
         'protocol.json 保存题目、来源、切分、预算与几何哈希；训练/测试分片保存原始回答、token、配置、批次和评分；frozen_test_actions.json 保存评分前的决策。', '',
         '这是现有公开数据与冻结小模型上的单轮实验；不代表未见数据集或更大模型的结果。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(out/'complete.json',dict(train_samples=700,test_samples=300,train_configs=109,test_methods=4))
    print(json.dumps(results,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','train','fit','test','finish'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--shard',type=int)
    p.add_argument('--batch',type=int,default=64);args=p.parse_args()
    if args.mode=='prepare':prepare(args.out,args.batch)
    elif args.mode=='train':train_measure(args.out,args.shard)
    elif args.mode=='fit':fit_routers(args.out)
    elif args.mode=='test':test_measure(args.out,args.shard)
    else:finish(args.out)


if __name__=='__main__':main()

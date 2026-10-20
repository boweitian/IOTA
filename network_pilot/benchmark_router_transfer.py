"""Independent BBH and IFEval replications of the frozen GSM8K router design.

Exhaustive train measurements; train-only fixed direction; real held-out generation.
The six GPU workers share a local-file barrier, then reuse their GPU for testing.
"""
import argparse
import datetime
import itertools
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

import gsm_router_large as core

ROOT=core.ROOT
LAB=core.LAB
DEFAULT=ROOT/'runs/router-other-benchmarks-v1'
SUITES=('bbh','ifeval')
METHODS=('baseline','fixed','set','majority')
NAMES={'bbh':'BBH（3 个子任务）','ifeval':'IFEval strict prompt-level'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sources_and_pool(suite):
    manifest=json.loads((LAB/'inputs/manifest.json').read_text())
    previous=json.loads((ROOT/'inputs/protocol.json').read_text())
    old=manifest['records']+previous['records']
    used_ids={r['id'] for r in old}
    used_questions={r['question'].strip() for r in old}
    pool=[];sources=[]
    if suite=='bbh':
        for source in manifest['sources']:
            if source['task']=='gsm8k':continue
            path=LAB/'inputs'/f"{source['task']}_all.json"
            assert core.sha(path)==source['sha256']
            sources.append(dict(source,local_path=str(path)))
            for i,item in enumerate(json.loads(path.read_text())['examples']):
                ident=f"{source['task']}:all:{i}"
                if ident in used_ids or item['input'].strip() in used_questions:continue
                pool.append(dict(id=ident,suite=suite,task=source['task'],question=item['input'],
                                 target=item['target'],source_index=i))
    else:
        path=ROOT/'inputs/ifeval.jsonl'
        source=next(s for s in previous['sources'] if s.get('file')=='instruction_following_eval/data/input_data.jsonl')
        assert core.sha(path)==source['sha256']
        sources.append(dict(source,local_path=str(path)))
        for item in core.read(path):
            ident=f"ifeval:{item['key']}"
            if ident in used_ids or item['prompt'].strip() in used_questions:continue
            pool.append(dict(id=ident,suite=suite,task=suite,question=item['prompt'],
                 source_index=item['key'],instruction_id_list=item['instruction_id_list'],kwargs=item['kwargs']))
    return pool,sources,previous['model']


def prepare(out):
    # Complete all validation before creating an immutable experiment directory.
    pools={suite:sources_and_pool(suite) for suite in SUITES}
    for suite,(pool,_,_) in pools.items():
        assert len(pool)>={'bbh':540,'ifeval':500}[suite]
        assert len({r['question'].strip() for r in pool})==len(pool)
    from router_benchmark_scoring import score_ifeval
    evaluator=ROOT/'vendor/instruction_following_eval'
    previous=json.loads((ROOT/'inputs/protocol.json').read_text())
    for source in previous['sources']:
        if '/instruction_following_eval/' in source.get('file','') and '/data/' not in source['file']:
            assert core.sha(evaluator/Path(source['file']).name)==source['sha256']
    # Exercise every new instruction combination and all local evaluator dependencies.
    for row in pools['ifeval'][0]:
        score_ifeval(row,'Here is a short example. It contains two sentences.\n\n* Example section *')
    reference=ROOT/'runs/gsm-router-1000-v1'
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['benchmark_router_transfer.py','benchmark_router_transfer.sbatch','gsm_router_large.py',
                 'fit_asset_router.py','simple_router.py','pilot_engine.py','experiment.py','space.py',
                 'router_benchmark_scoring.py']:
        shutil.copy2(ROOT/name,snapshot/name)
    for name in ['engine.py','concepts.py','grading.py']:
        shutil.copy2(LAB/name,snapshot/name)
    shutil.copytree(evaluator,snapshot/'instruction_following_eval',ignore=shutil.ignore_patterns('__pycache__'))
    source_hashes={str(p.relative_to(snapshot)):core.sha(p) for p in snapshot.rglob('*') if p.is_file()}
    for suite,(pool,sources,model) in pools.items():
        d=out/suite;d.mkdir()
        for filename in ['actions.npz','actions.json']:
            shutil.copy2(reference/filename,d/filename)
        rng=np.random.default_rng(core.SEED)
        n={'bbh':540,'ifeval':500}[suite];ntrain=int(.7*n)
        rows=[dict(pool[i],split='train' if j<ntrain else 'test')
              for j,i in enumerate(rng.permutation(len(pool))[:n])]
        p=dict(benchmark=suite,model=model,seed=core.SEED,samples=n,n_train=ntrain,n_test=n-ntrain,
             shards=6,batch_size=64,max_new_tokens=512 if suite=='bbh' else 1024,positions='last',
             records=rows,sources=sources,actions_sha256=core.sha(d/'actions.npz'),
             actions_catalog_sha256=core.sha(d/'actions.json'),
             geometry_sha256=json.loads((reference/'protocol.json').read_text())['geometry_sha256'],
             source_hashes=source_hashes,
             design='Single-benchmark random 70/30 split; exhaustively generate 109 configurations per train question. '
                    'Fit the same TF-IDF linear set-label router as GSM8K; predict all test actions before test generation. '
                    'Generate baseline, train-best fixed, set router, and majority router with matched test cohorts.',
             primary='set-label router versus training-best fixed action on all held-out questions',
             secondary=['set-label router versus baseline','majority-label router','BBH subtask breakdowns'],
             confirmatory_family='Two primary comparisons (BBH and IFEval); Holm correction across these two p-values',
             fixed_selection='argmax total training successes over all 109 configs; ties use saved config order',
             test_answer_use_for_routing=False,previous_questions_excluded=True,
             router_hyperparameters=dict(regularization=1e-6,maxiter=300,train_margin_stop=.5),
             prompt_policy='BBH: existing short-explanation + FINAL prompt. IFEval: exact original prompt, no FINAL suffix.',
             score_policy='BBH: exact parsed answer. IFEval: pinned official strict prompt-level all-instructions-pass.',
             limitations=['BBH is a 3-subtask subset, not full 27-task BBH.',
                          'Random question split; shared task/instruction templates across train/test are allowed.',
                          'Public-source pretraining exposure is unknown.',
                          'Within-run 512/1024-token budgets are identical across all compared methods.',
                          'Direction names label contrast probes and are not established causal semantic effects.'])
        core.save(d/'protocol.json',p)
    core.save(out/'design.json',dict(created_utc=now(),suites=list(SUITES),reference_run=str(reference),
             primary_router='set',n_total=1040,train_cells=79352,source_hashes=source_hashes,
             reason='Independent replication in reasoning and instruction following; no changes selected from new test results.'))
    status(out)
    print('PREPARED',out,flush=True)


def validate(out):
    design=json.loads((out/'design.json').read_text())
    for name,digest in design['source_hashes'].items():
        local=ROOT/name if (ROOT/name).is_file() else LAB/name
        if name.startswith('instruction_following_eval/'):
            local=ROOT/'vendor'/name
        assert core.sha(local)==digest,f'Source changed after freeze: {local}'
    for suite in SUITES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text())
        assert core.sha(d/'actions.npz')==p['actions_sha256']
        assert core.sha(d/'actions.json')==p['actions_catalog_sha256']
        for s in p['sources']:assert core.sha(Path(s['local_path']))==s['sha256']


def wait_for(out,predicate):
    deadline=time.monotonic()+3*3600
    while not predicate():
        failures=list(out.glob('failure-*.json'))
        if failures:raise RuntimeError(f'Peer failure: {failures[0].read_text()}')
        if time.monotonic()>deadline:raise TimeoutError('Peer barrier exceeded 3 hours')
        time.sleep(10)


def run_worker(out,shard):
    try:
        validate(out)
        # Six workers first measure BBH, then IFEval. GPU remains the same for each
        # worker's exhaustive train cohorts and all four matched test methods.
        for suite in SUITES:
            d=out/suite
            if (d/'complete.json').exists():continue
            core.train_measure(d,shard)
            ready=lambda:all((d/'train'/f'shard-{k}'/'complete.json').exists() for k in range(6))
            if shard==0:
                wait_for(out,ready)
                if not (d/'fit_complete.json').exists():
                    core.fit_routers(d)
                    core.save(d/'fit_complete.json',dict(completed_utc=now(),protocol_sha256=core.sha(d/'protocol.json')))
            else:
                wait_for(out,lambda:(d/'fit_complete.json').exists())
            core.test_measure(d,shard)
            if shard==0:
                wait_for(out,lambda:all((d/'test'/f'shard-{k}'/'complete.json').exists() for k in range(6)))
                finish_suite(out,suite)
        if shard==0:
            finish_all(out)
    except BaseException:
        core.save(out/f'failure-{shard}.json',dict(utc=now(),shard=shard,error=traceback.format_exc()))
        raise


def paired(score,control):
    from scipy.stats import binomtest
    delta=np.asarray(score)-np.asarray(control)
    rng=np.random.default_rng(core.SEED)
    boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)*100
    wins=int((delta==1).sum());losses=int((delta==-1).sum())
    return dict(n=len(delta),delta_pp=float(delta.mean()*100),delta_95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
                fixes=wins,breaks=losses,p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.)


def finish_suite(out,suite):
    """Regrade raw evidence, verify model predictions, and compute paired scores."""
    import joblib
    from fit_asset_router import input_text
    d=out/suite;p=json.loads((d/'protocol.json').read_text())
    rows={r['id']:r for r in p['records']}
    train=[r for r in p['records'] if r['split']=='train']
    test=[r for r in p['records'] if r['split']=='test']
    assert not {r['id'] for r in train}&{r['id'] for r in test}
    decisions={r['id']:r for r in json.loads((d/'frozen_test_actions.json').read_text())}
    summary=json.loads((d/'training_summary.json').read_text())
    assert core.sha(d/'frozen_test_actions.json')==summary['frozen_test_actions_sha256']
    actions=np.load(d/'actions.npz')['ids'].astype(str)
    action_index={a:j for j,a in enumerate(actions)}
    train_index={r['id']:i for i,r in enumerate(train)}
    matrix=np.full((len(train),len(actions)),-1,dtype=np.int8)
    test_records={};models=[];same_action_checks=0
    for split in ['train','test']:
        ordered=train if split=='train' else test
        for shard in range(p['shards']):
            folder=d/split/f'shard-{shard}'
            assert (folder/'complete.json').exists()
            models.append(json.loads((folder/'model.json').read_text()))
            local=ordered[shard::p['shards']]
            cohorts={r['id']:[x['id'] for x in local[lo:lo+p['batch_size']]]
                     for lo in range(0,len(local),p['batch_size']) for r in local[lo:lo+p['batch_size']]}
            for r in core.read(folder/'records.jsonl'):
                row=rows[r['id']];assert row['split']==split
                assert r['batch_ids']==cohorts[r['id']]
                assert len(r['token_ids'])==r['new_tokens']<=p['max_new_tokens']
                regraded=core.grade(r['text'],row)
                assert float(regraded['correct'])==r['reward']
                if suite=='ifeval':assert regraded['instruction_results']==r['instruction_results']
                if split=='train':
                    i,j=train_index[r['id']],action_index[r['action']]
                    assert matrix[i,j]==-1;matrix[i,j]=int(r['reward'])
                else:
                    key=r['id'],r['method'];assert key not in test_records
                    assert r['action']==decisions[r['id']][r['method']]
                    test_records[key]=r
    assert (matrix>=0).all()
    np.testing.assert_array_equal(matrix,np.load(d/'train_matrix.npz')['reward'])
    assert summary['fixed_action']==actions[matrix.sum(0).argmax()]
    assert len(test_records)==len(test)*len(METHODS)
    for key in ['model','revision','layer_index','hidden_size','torch','transformers','dtype','thinking',
                'decode','system_prompt','chat_template_sha256','intervention']:
        assert len({json.dumps(m[key],sort_keys=True) for m in models})==1,f'Model mismatch: {key}'
    # The resumed allocation can use another GPU type than the train allocation.
    # All methods for each held-out cohort are generated in the same worker/GPU.
    # Device names are retained as provenance; comparison controls are within test.
    for row in test:
        for a,b in itertools.combinations(METHODS,2):
            ra,rb=test_records[row['id'],a],test_records[row['id'],b]
            if ra['action']==rb['action']:
                same_action_checks+=1
                assert ra['token_ids']==rb['token_ids'],f'Same-action generation mismatch: {row["id"]}, {a}, {b}'
    train_hits={}
    with (d/'training_direction_dictionary.jsonl').open('w') as f:
        for i,row in enumerate(train):
            core_row=dict(id=row['id'],successful_actions=actions[matrix[i]==1].tolist())
            f.write(json.dumps(core_row,ensure_ascii=False)+'\n')
    for name in ['set','majority']:
        bundle=joblib.load(d/f'router-{name}.joblib')
        assert bundle['training_ids']==[r['id'] for r in train]
        for split,items in [('train',train),('test',test)]:
            x=bundle['transform'].transform([input_text(r['suite'],r['question']) for r in items])
            choice=np.asarray(x@bundle['w']+bundle['bias']).argmax(1)
            if split=='test':
                assert all(str(bundle['actions'][j])==decisions[r['id']][name] for r,j in zip(items,choice))
            else:
                hits=matrix[np.arange(len(train)),choice]
                if name=='set':assert hits.sum()==matrix.max(1).sum()
                assert int(hits.sum())==summary['training'][name]['train_correct']
                train_hits[name]=int(hits.sum())
    scores={m:np.array([test_records[r['id'],m]['reward'] for r in test]) for m in METHODS}
    results={m:dict(n=len(test),correct=int(scores[m].sum()),accuracy=float(scores[m].mean()),
                 vs_baseline=paired(scores[m],scores['baseline']),vs_fixed=paired(scores[m],scores['fixed']),
                 truncations=sum(test_records[r['id'],m]['truncated'] for r in test),
                 mean_tokens=float(np.mean([test_records[r['id'],m]['new_tokens'] for r in test]))) for m in METHODS}
    groups={}
    if suite=='bbh':
        for task in sorted({r['task'] for r in test}):
            mask=np.array([r['task']==task for r in test])
            groups[task]={m:dict(n=int(mask.sum()),correct=int(scores[m][mask].sum()),
                                accuracy=float(scores[m][mask].mean())) for m in METHODS}
    result=dict(benchmark=suite,results=results,subtasks=groups,train_baseline=int(matrix[:,0].sum()),
                train_covered=int(matrix.max(1).sum()),n_train=len(train),train_router_correct=train_hits,
                fixed_action=summary['fixed_action'],same_action_token_checks=same_action_checks,
                protocol_sha256=core.sha(d/'protocol.json'),decisions_sha256=core.sha(d/'frozen_test_actions.json'))
    core.save(d/'results.json',result)
    core.save(d/'verification.json',dict(passed=True,utc=now(),train_cells=int(matrix.size),
         test_records=len(test_records),raw_outputs_regraded=True,train_test_disjoint=True,
         fixed_selected_from_train_only=True,frozen_predictions_match_saved_models=True,
         exact_cohorts_checked=True,model_metadata_match=True,same_action_token_checks=same_action_checks))
    report_suite(d,p,result)
    core.save(d/'complete.json',dict(utc=now(),verified=True))
    print('FINISHED',suite,json.dumps(results),flush=True)


def report_suite(d,p,r):
    lines=[f"# {NAMES[p['benchmark']]} Router 复现实验",'',
        f"随机 {p['n_train']} 训练 / {p['n_test']} 测试；训练逐题遍历 109 配置。Qwen3.5-0.8B，第 15 层，"
        f"每次最多 {p['max_new_tokens']} 新 token。每个方法均实际生成测试回答，批次一致。",'',
        '固定方向根据训练集总正确数选择；set router 为本次预先指定的主方法。全部测试题保留在分母中。','',
        '| 方法 | 正确/测试题 | 准确率 | 比 baseline（百分点） | 比固定方向（百分点） | 截断 |',
        '|---|---:|---:|---:|---:|---:|']
    for m,v in r['results'].items():
        lines.append(f"| {m} | {v['correct']}/{v['n']} | {100*v['accuracy']:.2f}% | {v['vs_baseline']['delta_pp']:+.2f} | {v['vs_fixed']['delta_pp']:+.2f} | {v['truncations']} |")
    v=r['results']['set']['vs_fixed']
    lines+=['',f"主比较（set − fixed）：配对 bootstrap 95% 区间 {v['delta_95_pp']} 个百分点；"
            f"双侧 McNemar 精确 p={v['p']:.6f}。跨 BBH、IFEval 的两次主比较使用 Holm 校正，见上一级总报告。",'',
            f"训练集 baseline：{r['train_baseline']}/{r['n_train']}；方向字典覆盖：{r['train_covered']}/{r['n_train']}。"
            f"两个 router 的原题成功方向选中数：{r['train_router_correct']}。",'',
            f"训练集选出的固定方向：`{r['fixed_action']}`。",'']
    if r['subtasks']:
        lines+=['BBH 子任务结果（探索性分解；一个 BBH router 和一个全局固定方向）：','',
                '| 子任务 | 测试题 | baseline | fixed | set | majority |','|---|---:|---:|---:|---:|---:|']
        for task,values in r['subtasks'].items():
            lines.append(f"| {task} | {values['baseline']['n']} | "+' | '.join(f"{100*values[m]['accuracy']:.2f}%" for m in METHODS)+' |')
    lines+=['','逐题方向字典：training_direction_dictionary.jsonl；全量原始生成、token 与评分：train/ 和 test/。',
            'verification.json 记录完整重新评分、切分、冻结决策、模型、批次及相同方向 token 一致性核验。','',
            '限制：'+ '；'.join(p['limitations'])]
    (d/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def finish_all(out):
    data={s:json.loads((out/s/'results.json').read_text()) for s in SUITES}
    ordered=sorted(SUITES,key=lambda s:data[s]['results']['set']['vs_fixed']['p'])
    adjusted={};floor=0.
    for i,s in enumerate(ordered):
        floor=max(floor,min(1.,(len(SUITES)-i)*data[s]['results']['set']['vs_fixed']['p']))
        adjusted[s]=floor
    core.save(out/'results.json',dict(suites=data,primary_p_holm=adjusted))
    lines=['# Router 在其他数据集的复现实验','',
           '比较逐题 set router、训练集选出的最佳固定方向和无干预 baseline。每个数据集独立随机划分 70%/30%，独立训练 router。','',
           '| 数据集 | 训练/测试 | baseline | 最佳固定方向 | set router | router − fixed | 主比较 Holm p |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for s,r in data.items():
        v=r['results'];n=v['set']['n']
        lines.append(f"| {NAMES[s]} | {r['n_train']}/{n} | {100*v['baseline']['accuracy']:.2f}% | "
                     f"{100*v['fixed']['accuracy']:.2f}% | {100*v['set']['accuracy']:.2f}% | "
                     f"{v['set']['vs_fixed']['delta_pp']:+.2f} pp | {adjusted[s]:.6f} |")
    lines+=['','两个数据集的主比较在运行前固定为 set router 对训练最佳固定方向；上述 p 值对两次比较做 Holm 校正。',
            '训练覆盖率与训练拟合率分别报告，不能把原题选方向拟合率当作新题准确率。',
            'BBH 仅含已缓存的三个子任务；IFEval 使用固定版本官方 strict prompt-level 评分。',
            '完整结果和逐题审计分别见 bbh/RESULTS.md、ifeval/RESULTS.md。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    core.save(out/'complete.json',dict(utc=now(),verified=True,suites=list(SUITES)))
    status(out)


def status(out):
    result=dict(updated_utc=now(),complete=(out/'complete.json').exists(),suites={},failures=[])
    for suite in SUITES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text())
        counts=[]
        for shard in range(p['shards']):
            folder=d/'train'/f'shard-{shard}'
            f=folder/'complete.json' if (folder/'complete.json').exists() else folder/'progress.json'
            counts.append(json.loads(f.read_text())['records'] if f.exists() else 0)
        tests=0
        for f in (d/'test').glob('*/records.jsonl'):
            with f.open() as stream:tests+=sum(line.endswith('\n') for line in stream)
        result['suites'][suite]=dict(train_cells=sum(counts),required_train_cells=p['n_train']*109,
            train_shards_done=sum((d/'train'/f'shard-{k}'/'complete.json').exists() for k in range(p['shards'])),
            fitted=(d/'fit_complete.json').exists(),test_records=tests,required_test_records=p['n_test']*4,
            complete=(d/'complete.json').exists())
    result['failures']=[json.loads(f.read_text()) for f in out.glob('failure-*.json')]
    core.save(out/'status.json',result)
    (out/'STATUS.md').write_text('# 其他数据集 Router 实验进度\n\n'+
        f"更新：{result['updated_utc']}。全部完成：{result['complete']}。\n\n"+
        '\n'.join(f"- {s}: 训练方向遍历 {v['train_cells']:,}/{v['required_train_cells']:,}；"
                  f"测试生成 {v['test_records']}/{v['required_test_records']}；完成：{v['complete']}。"
                  for s,v in result['suites'].items())+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','status','finish'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--shard',type=int,choices=range(6))
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.out)
    elif args.mode=='run':
        assert args.shard is not None
        run_worker(args.out,args.shard)
    elif args.mode=='finish':
        for suite in SUITES:finish_suite(args.out,suite)
        finish_all(args.out)
    else:status(args.out)


if __name__=='__main__':main()

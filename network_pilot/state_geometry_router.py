"""Frozen-state/geometry reward routing with train-only selection and blind GSM8K.

Reuses exhaustive reward assets, never historical test labels, for router fitting.
Six workers capture all states, fit on CPU, then generate six matched test methods.
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
import state_geometry_models as models

ROOT=core.ROOT
DEFAULT=ROOT/'runs/state-geometry-router-v1'
SOURCES={'gsm8k':ROOT/'runs/gsm-router-1000-v1',
         'bbh':ROOT/'runs/router-other-benchmarks-v1/bbh',
         'ifeval':ROOT/'runs/router-other-benchmarks-v1/ifeval'}
METHODS=('baseline','fixed','text','hidden_reward','geometry_raw','geometry_gated')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def public_row(row):
    return {key:row[key] for key in ['id','suite','question']}


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    prior=[]
    for path in [core.LAB/'inputs/manifest.json',ROOT/'inputs/protocol.json']:
        prior+=json.loads(path.read_text())['records']
    for source in SOURCES.values():prior+=json.loads((source/'protocol.json').read_text())['records']
    used={r['id'] for r in prior};used_text={r['question'].strip() for r in prior}
    gsm_source=core.LAB/'inputs/gsm8k_train.jsonl'
    assert core.sha(gsm_source)==json.loads((SOURCES['gsm8k']/'protocol.json').read_text())['source']['sha256']
    pool=[]
    for i,row in enumerate(core.read(gsm_source)):
        ident=f'gsm8k:train:{i}'
        if ident in used or row['question'].strip() in used_text:continue
        pool.append(dict(id=ident,suite='gsm8k',task='gsm8k',question=row['question'],
             target=row['answer'].split('####')[-1].strip(),source_index=i,split='fresh_test',eval_group='fresh'))
    permutation=np.random.default_rng(models.SEED).permutation(len(pool))[:300]
    fresh=[pool[i] for i in permutation];assert len(fresh)==300
    snap=out/'source_snapshot';snap.mkdir()
    for name in ['state_geometry_router.py','state_geometry_models.py','state_geometry_router.sbatch',
                 'gsm_router_large.py','pilot_engine.py','fit_asset_router.py','simple_router.py',
                 'router_benchmark_scoring.py','space.py']:
        shutil.copy2(ROOT/name,snap/name)
    for name in ['engine.py','concepts.py','grading.py']:
        shutil.copy2(core.LAB/name,snap/name)
    hashes={p.name:core.sha(p) for p in snap.iterdir()}
    for suite,source in SOURCES.items():
        d=out/suite;d.mkdir()
        p=json.loads((source/'protocol.json').read_text())
        rows=[dict(r,eval_group='historical') for r in p['records']]
        if suite=='gsm8k':rows+=fresh
        for filename in ['actions.npz','actions.json','train_matrix.npz','router-set.joblib']:
            shutil.copy2(source/filename,d/filename)
        labels=np.load(d/'train_matrix.npz')
        train=[r for r in rows if r['split']=='train']
        assert labels['ids'].astype(str).tolist()==[r['id'] for r in train]
        assert labels['actions'].astype(str).tolist()==np.load(d/'actions.npz')['ids'].astype(str).tolist()
        protocol=dict(benchmark=suite,model=p['model'],records=rows,n_train=len(train),
             n_test=sum(r['split']!='train' for r in rows),shards=6,batch_size=64,
             max_new_tokens=p['max_new_tokens'],source_run=str(source),
             source_protocol_sha256=core.sha(source/'protocol.json'),
             actions_sha256=core.sha(d/'actions.npz'),train_matrix_sha256=core.sha(d/'train_matrix.npz'),
             old_router_sha256=core.sha(d/'router-set.joblib'),seed=models.SEED,
             feature='Unsteered prompt only: layer15 last token, final-layer last token, final-layer non-padding prompt mean. No response, answer or target.',
             primary='geometry_gated versus fixed on 300 fresh GSM8K questions',
             historical_tests='Previously observed historical holdouts are exploratory method-development comparisons.',
             selection='Three inner folds over original exhaustive-training questions only; use all 109 binary outcomes including failures.',
             candidates=dict(representations=['final','multi','multi_rbf'],alpha=list(models.ALPHAS),
                 action_strengths=list(models.ACTION_STRENGTHS),gates=[None if np.isinf(t) else t for t in models.GATES]),
             reward_model='Question linear or median-bandwidth RBF ridge kernel times RBF kernel of actual steering vectors; learn full success/failure reward matrix.',
             gate='No-op if predicted success-score gain over no-op is below train-OOF-selected threshold.',
             methods=list(METHODS),source_hashes=hashes,
             no_test_outcome_use_for_fitting=True,
             limitations=['Historical holdouts have been observed in previous experiments; exploratory only.',
                 'Fresh GSM8K samples are from pinned official train source, not the official test split.',
                 'Additional frozen-model prompt prefill is required for hidden-state routing.',
                 'Reward scores are not guaranteed calibrated probabilities; a train-OOF logistic readout is saved separately.',
                 'This is not an exhaustive test-set steering oracle.'])
        core.save(d/'protocol.json',protocol)
    core.save(out/'design.json',dict(created_utc=now(),source_hashes=hashes,
             fresh_gsm_source_sha256=core.sha(gsm_source),fresh_samples=300,
             fresh_questions_disjoint_from_previous_runs=True,source_runs={k:str(v) for k,v in SOURCES.items()},
             primary_suite='gsm8k:fresh',primary_method='geometry_gated',primary_comparator='fixed'))
    status(out)


def validate(out):
    d=json.loads((out/'design.json').read_text())
    for name,digest in d['source_hashes'].items():
        path=ROOT/name if (ROOT/name).exists() else core.LAB/name
        assert core.sha(path)==digest,f'Source drift: {name}'
    assert core.sha(core.LAB/'inputs/gsm8k_train.jsonl')==d['fresh_gsm_source_sha256']
    for suite in SOURCES:
        p=json.loads((out/suite/'protocol.json').read_text())
        for file,key in [('actions.npz','actions_sha256'),('train_matrix.npz','train_matrix_sha256'),('router-set.joblib','old_router_sha256')]:
            assert core.sha(out/suite/file)==p[key]


def capture_batch(engine,rows):
    """One prompt prefill; capture both layers without running any answer tokens."""
    torch=engine.torch
    prompts=[engine.prefix(core.question(public_row(r))) for r in rows]
    inputs=engine.tokenizer(prompts,return_tensors='pt',padding=True,add_special_tokens=False).to(engine.device)
    final_index=len(engine.model.model.language_model.layers)-1
    observed={};handles=[]
    for name,index in [('steer',engine.layer),('final',final_index)]:
        def hook(module,args,output,key=name):observed[key]=output.detach()
        handles.append(engine.model.model.language_model.layers[index].register_forward_hook(hook))
    start=time.monotonic()
    try:
        with torch.inference_mode():
            result=engine.model(**inputs,use_cache=True,logits_to_keep=1)
    finally:
        for handle in handles:handle.remove()
    mask=inputs.attention_mask.float()
    final=observed['final'].float()
    features=dict(steer_last=observed['steer'][:,-1].float().cpu().numpy(),
                  final_last=final[:,-1].cpu().numpy(),
                  final_mean=((final*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy())
    del result
    return features,dict(ids=[r['id'] for r in rows],prompt_tokens=mask.sum(1).int().cpu().tolist(),
                         padded_tokens=int(mask.shape[1]),seconds=time.monotonic()-start,
                         layer_indices=[engine.layer,final_index],generated_tokens=0)


def capture(out,shard):
    e=None
    for suite in SOURCES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text());folder=d/'features';folder.mkdir(exist_ok=True)
        path=folder/f'shard-{shard}.npz'
        if path.exists() and path.with_suffix('.json').exists():continue
        if e is None:e=core.engine(p)
        # Keep feature extraction of training questions independent of all test
        # prompts and preserve the original train measurement cohort shapes.
        groups=[[r for r in p['records'] if r['split']==split][shard::p['shards']]
                for split in ['train','test','fresh_test']]
        rows=[r for group in groups for r in group]
        combined={k:[] for k in ['steer_last','final_last','final_mean']};cohorts=[]
        for group in groups:
            for lo in range(0,len(group),p['batch_size']):
                batch,meta=capture_batch(e,group[lo:lo+p['batch_size']]);cohorts.append(meta)
                for k in combined:combined[k].append(batch[k])
        temp=folder/f'shard-{shard}.tmp.npz'
        np.savez_compressed(temp,ids=np.array([r['id'] for r in rows]),**{k:np.concatenate(v) for k,v in combined.items()})
        temp.replace(path)
        core.save(folder/f'shard-{shard}.json',dict(model=e.meta,cohorts=cohorts,
             protocol_sha256=core.sha(d/'protocol.json'),feature_sha256=core.sha(path)))
        print('CAPTURED',suite,shard,len(rows),flush=True)


def load_features(d,p):
    by={};metas=[]
    for shard in range(p['shards']):
        path=d/'features'/f'shard-{shard}.npz';a=np.load(path)
        meta=json.loads(path.with_suffix('.json').read_text());assert core.sha(path)==meta['feature_sha256'];metas.append(meta)
        expected=[]
        for split in ['train','test','fresh_test']:
            group=[r for r in p['records'] if r['split']==split][shard::p['shards']]
            expected.extend([[r['id'] for r in group[lo:lo+p['batch_size']]]
                             for lo in range(0,len(group),p['batch_size'])])
        assert [c['ids'] for c in meta['cohorts']]==expected
        assert all(c['generated_tokens']==0 for c in meta['cohorts'])
        for i,ident in enumerate(a['ids'].astype(str)):
            assert ident not in by;by[ident]={k:a[k][i] for k in ['steer_last','final_last','final_mean']}
    assert set(by)=={r['id'] for r in p['records']}
    result={k:np.stack([by[r['id']][k] for r in p['records']]) for k in ['steer_last','final_last','final_mean']}
    assert all(np.isfinite(v).all() for v in result.values())
    return result,metas


def fit_suite(out,suite):
    import joblib
    from fit_asset_router import input_text
    d=out/suite;p=json.loads((d/'protocol.json').read_text());features,_=load_features(d,p)
    rows=p['records'];tr=np.array([r['split']=='train' for r in rows]);te=~tr
    y=np.load(d/'train_matrix.npz')['reward'].astype(float)
    actions=np.load(d/'actions.npz');ids=actions['ids'].astype(str)
    geom=models.action_kernel(actions['shifts']);folds=models.three_folds(len(y))
    candidates={k:models.hidden_input(features,k)[tr] for k in ['final','multi','multi_rbf']}
    testx={k:models.hidden_input(features,k)[te] for k in ['final','multi','multi_rbf']}
    fixed=int(y.sum(0).argmax());prediction={};summaries={}
    # Fold-local best fixed and no-op are measured on exactly the same inner rows.
    fixed_oof=np.zeros(len(y),dtype=int)
    for fold in range(3):fixed_oof[folds==fold]=y[folds!=fold].sum(0).argmax()
    cv_controls=dict(baseline=int(y[:,0].sum()),fixed=int(y[np.arange(len(y)),fixed_oof].sum()))
    for name,use_geometry in [('hidden_reward',False),('geometry',True)]:
        best,trials,oof=models.select(candidates,y,folds,geom,use_geometry)
        core.save(d/f'selection-{name}.json',dict(best=best,trials=trials,folds=folds.tolist(),controls=cv_controls))
        m=models.fit_reward(candidates[best['kind']],y,best['alpha'],geom,best['strength'],
                            kernel='rbf' if best['kind'].endswith('_rbf') else 'linear')
        cal=models.calibration(oof,y)
        bundle=dict(model=m,kind=best['kind'],calibration=cal,selection=best,actions=ids,
                    training_ids=[r['id'] for r in rows if r['split']=='train'],
                    features='unsteered prompt prefill only',geometry_sha256=p['actions_sha256'])
        joblib.dump(bundle,d/f'router-{name}.joblib',compress=3)
        scores=models.predict(m,testx[best['kind']]);train_scores=models.predict(m,candidates[best['kind']])
        raw=models.choose(scores);threshold=float('inf') if best['threshold'] is None else best['threshold']
        gated=models.choose(scores,threshold)
        if name=='geometry':
            prediction['geometry_raw']=raw;prediction['geometry_gated']=gated
        else:prediction[name]=raw
        raw_train=models.choose(train_scores)
        summaries[name]=dict(selected=best,train_correct=int(y[np.arange(len(y)),raw_train].sum()),
                  gated_train_correct=int(y[np.arange(len(y)),models.choose(train_scores,threshold)].sum()),
                  train_total=len(y),covered=int(y.max(1).sum()),inner_controls=cv_controls,
                  inner_brier=float(np.mean((models.probabilities(cal,oof)-y)**2)),
                  calibration_note='Brier measured on OOF scores used to fit calibration; descriptive, not an independent calibration test.')
    old=joblib.load(d/'router-set.joblib')
    test_rows=[r for r in rows if r['split']!='train']
    x=old['transform'].transform([input_text(r['suite'],r['question']) for r in test_rows])
    old_choices=np.asarray(x@old['w']+old['bias']).argmax(1)
    assert old['actions'].astype(str).tolist()==ids.tolist()
    decisions=[dict(id=r['id'],baseline='none',fixed=str(ids[fixed]),text=str(ids[old_choices[i]]),
                   **{m:str(ids[v[i]]) for m,v in prediction.items()}) for i,r in enumerate(test_rows)]
    core.save(d/'frozen_test_actions.json',decisions)
    core.save(d/'training_summary.json',dict(models=summaries,train_baseline=int(y[:,0].sum()),
         train_coverage=int(y.max(1).sum()),fixed_action=str(ids[fixed]),n_train=len(y),
         frozen_test_actions_sha256=core.sha(d/'frozen_test_actions.json')))
    core.save(d/'fit_complete.json',dict(utc=now(),decisions_sha256=core.sha(d/'frozen_test_actions.json')))
    print('FIT AND FROZE',suite,json.dumps(summaries),flush=True)


def test(out,suite,shard):
    d=out/suite;p=json.loads((d/'protocol.json').read_text())
    rows=[r for r in p['records'] if r['split']!='train'][shard::p['shards']]
    decisions={r['id']:r for r in json.loads((d/'frozen_test_actions.json').read_text())}
    summary=json.loads((d/'training_summary.json').read_text())
    assert summary['frozen_test_actions_sha256']==core.sha(d/'frozen_test_actions.json')
    path=d/'test'/f'shard-{shard}';path.mkdir(parents=True,exist_ok=True)
    seen={(r['id'],r['method']) for r in core.read(path/'records.jsonl')}
    if len(seen)==len(rows)*len(METHODS) and (path/'complete.json').exists():return
    e=core.engine(p);core.save(path/'model.json',e.meta)
    a=np.load(d/'actions.npz');shifts=dict(zip(a['ids'].astype(str),a['shifts']))
    for method in METHODS:
        for lo in range(0,len(rows),p['batch_size']):
            chunk=rows[lo:lo+p['batch_size']]
            if all((r['id'],method) in seen for r in chunk):continue
            generated=e.generate([core.question(r) for r in chunk],
                [shifts[decisions[r['id']][method]] for r in chunk],max_new_tokens=p['max_new_tokens'])
            for row,g in zip(chunk,generated):
                if (row['id'],method) in seen:continue
                scored=core.grade(g['text'],row)
                core.append(path/'records.jsonl',dict(id=row['id'],method=method,action=decisions[row['id']][method],
                    eval_group=row['eval_group'],batch_ids=[r['id'] for r in chunk],reward=float(scored['correct']),**scored,**g))
                seen.add((row['id'],method))
        print('TEST',suite,shard,method,len(seen),flush=True)
    core.save(path/'complete.json',dict(records=len(seen),utc=now()))


def wait_for(out,predicate):
    until=time.monotonic()+3*3600
    while not predicate():
        failures=list(out.glob('failure-*.json'))
        if failures:raise RuntimeError(failures[0].read_text())
        if time.monotonic()>until:raise TimeoutError('Peer barrier exceeded 3h')
        time.sleep(5)


def worker(out,shard):
    try:
        validate(out);capture(out,shard)
        for suite in SOURCES:
            d=out/suite
            if shard==0:
                wait_for(out,lambda:all((d/'features'/f'shard-{k}.npz').exists() and
                     (d/'features'/f'shard-{k}.json').exists() for k in range(6)))
                if not (d/'fit_complete.json').exists():fit_suite(out,suite)
            else:wait_for(out,lambda:(d/'fit_complete.json').exists())
            test(out,suite,shard)
            if shard==0:
                wait_for(out,lambda:all((d/'test'/f'shard-{k}/complete.json').exists() for k in range(6)))
                finish_suite(out,suite)
        if shard==0:finish_all(out)
    except BaseException:
        core.save(out/f'failure-{shard}.json',dict(utc=now(),error=traceback.format_exc()))
        raise


def paired(a,b):
    from scipy.stats import binomtest
    delta=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(models.SEED)
    boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)*100
    fixes=int((delta==1).sum());breaks=int((delta==-1).sum())
    return dict(delta_pp=float(delta.mean()*100),delta_95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
                fixes=fixes,breaks=breaks,p=float(binomtest(fixes,fixes+breaks).pvalue) if fixes+breaks else 1.)


def finish_suite(out,suite):
    import joblib
    from fit_asset_router import input_text
    d=out/suite;p=json.loads((d/'protocol.json').read_text())
    rows=[r for r in p['records'] if r['split']!='train'];by={r['id']:r for r in rows}
    decisions={r['id']:r for r in json.loads((d/'frozen_test_actions.json').read_text())}
    summary=json.loads((d/'training_summary.json').read_text())
    assert core.sha(d/'frozen_test_actions.json')==summary['frozen_test_actions_sha256']
    records={};metadata=[];checks=0
    for shard in range(6):
        path=d/'test'/f'shard-{shard}';assert (path/'complete.json').exists()
        metadata.append(json.loads((path/'model.json').read_text()))
        local=rows[shard::6]
        cohorts={r['id']:[q['id'] for q in local[lo:lo+p['batch_size']]]
                 for lo in range(0,len(local),p['batch_size']) for r in local[lo:lo+p['batch_size']]}
        for r in core.read(path/'records.jsonl'):
            key=r['id'],r['method'];assert key not in records
            assert r['action']==decisions[r['id']][r['method']]
            assert r['batch_ids']==cohorts[r['id']]
            assert len(r['token_ids'])==r['new_tokens']<=p['max_new_tokens']
            scored=core.grade(r['text'],by[r['id']]);assert float(scored['correct'])==r['reward']
            if suite=='ifeval':assert scored['instruction_results']==r['instruction_results']
            records[key]=r
    assert len(records)==len(rows)*len(METHODS)
    for row in rows:
        for a,b in itertools.combinations(METHODS,2):
            ra,rb=records[row['id'],a],records[row['id'],b]
            if ra['action']==rb['action']:
                checks+=1;assert ra['token_ids']==rb['token_ids']
    features,feature_meta=load_features(d,p);test_mask=np.array([r['split']!='train' for r in p['records']])
    for key in ['model','revision','layer_index','hidden_size','torch','transformers','dtype','thinking',
                'decode','system_prompt','chat_template_sha256','intervention']:
        assert len({json.dumps(m[key],sort_keys=True) for m in metadata+[v['model'] for v in feature_meta]})==1
    old=joblib.load(d/'router-set.joblib')
    xx=old['transform'].transform([input_text(r['suite'],r['question']) for r in rows])
    choice=np.asarray(xx@old['w']+old['bias']).argmax(1)
    assert all(str(old['actions'][j])==decisions[r['id']]['text'] for r,j in zip(rows,choice))
    for name in ['hidden_reward','geometry']:
        bundle=joblib.load(d/f'router-{name}.joblib')
        x=models.hidden_input(features,bundle['kind'])[test_mask]
        score=models.predict(bundle['model'],x)
        methods=['geometry_raw','geometry_gated'] if name=='geometry' else [name]
        for method in methods:
            threshold=bundle['selection']['threshold'] if method=='geometry_gated' else 0.
            threshold=float('inf') if threshold is None else threshold
            chosen=models.choose(score,threshold)
            assert all(str(bundle['actions'][j])==decisions[r['id']][method] for r,j in zip(rows,chosen))
    groups={}
    for group in sorted({r['eval_group'] for r in rows}):
        selected=[r for r in rows if r['eval_group']==group]
        scores={m:np.array([records[r['id'],m]['reward'] for r in selected]) for m in METHODS}
        groups[group]={m:dict(n=len(selected),correct=int(scores[m].sum()),accuracy=float(scores[m].mean()),
             vs_baseline=paired(scores[m],scores['baseline']),vs_fixed=paired(scores[m],scores['fixed']),
             vs_text=paired(scores[m],scores['text']),
             truncated=sum(records[r['id'],m]['truncated'] for r in selected),
             distinct_actions=len({decisions[r['id']][m] for r in selected}),
             interventions=sum(decisions[r['id']][m]!='none' for r in selected)) for m in METHODS}
    core.save(d/'results.json',dict(benchmark=suite,groups=groups,training=summary,source='new matched generation',
         protocol_sha256=core.sha(d/'protocol.json'),decisions_sha256=core.sha(d/'frozen_test_actions.json')))
    core.save(d/'verification.json',dict(passed=True,utc=now(),records=len(records),
         raw_outputs_regraded=True,frozen_predictions_match_reloaded_models=True,
         exact_cohorts=True,same_action_token_checks=checks,
         gpu_test_per_shard=[m['device'] for m in metadata]))
    core.save(d/'complete.json',dict(utc=now(),verified=True))
    print('FINISHED',suite,json.dumps(groups),flush=True)


def finish_all(out):
    data={s:json.loads((out/s/'results.json').read_text()) for s in SOURCES}
    core.save(out/'results.json',data)
    lines=['# 隐藏状态 × Steering Geometry Router 实验','',
        '主比较：新增 300 道从未使用的 GSM8K 题上，geometry_gated 对训练最佳固定方向。旧留出题已经观察过，只作探索性复测。',
        '训练只使用原始训练题的完整 109 配置奖励矩阵。隐藏状态仅来自不加 steering 的读题 prefill，未输入回答、答案或测试评分。',
        '三折内部验证选择表示、正则和不干预阈值；所有测试方向在生成前冻结。每种方法重新生成回答，输入批次和 GPU 逐题匹配。','',
        '| 数据集/分组 | 测试题 | baseline | fixed | text | hidden reward | geometry raw | geometry gated |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for suite,v in data.items():
        for group,values in v['groups'].items():
            lines.append(f"| {suite}/{group} | {values['baseline']['n']} | "+' | '.join(f"{100*values[m]['accuracy']:.2f}%" for m in METHODS)+' |')
    primary=data['gsm8k']['groups']['fresh']['geometry_gated']
    lines+=['',f"主比较增益：{primary['vs_fixed']['delta_pp']:+.2f} 个百分点；配对 bootstrap 95% 区间 {primary['vs_fixed']['delta_95_pp']}；双侧 McNemar p={primary['vs_fixed']['p']:.6f}。",'',
         'geometry_raw 与 geometry_gated 使用同一个已选评分器；后者的阈值只在训练题的折外预测上选取。hidden_reward 是没有方向几何平滑的消融。text 是此前的成功方向集合标签 TF-IDF router。',
         '训练和验证使用全部成功及失败标签。全局动作先验来自相应训练折；任何测试结果均未用于选参数或阈值。',
         '隐藏状态方法额外需要一次冻结基座模型的 prompt prefill；所有方法的最终回答预算相同，不代表推理总成本完全相同。',
         'reward score 的单调概率读数使用内部折外训练标签拟合，不宣称独立校准；完整选择过程、拟合资产、原始生成及核验保存在各数据集目录。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    core.save(out/'complete.json',dict(utc=now(),verified=True))
    status(out)


def status(out):
    s=dict(utc=now(),complete=(out/'complete.json').exists(),suites={},failures=[])
    for suite in SOURCES:
        d=out/suite;p=json.loads((d/'protocol.json').read_text());count=0
        for path in (d/'test').glob('*/records.jsonl'):
            with path.open() as f:count+=sum(line.endswith('\n') for line in f)
        s['suites'][suite]=dict(feature_shards=sum((d/'features'/f'shard-{k}.npz').exists() for k in range(6)),
             fitted=(d/'fit_complete.json').exists(),test_records=count,required_test_records=p['n_test']*len(METHODS),
             complete=(d/'complete.json').exists())
    s['failures']=[json.loads(p.read_text()) for p in out.glob('failure-*.json')]
    core.save(out/'status.json',s)
    print(json.dumps(s,ensure_ascii=False),flush=True)
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','status','finish'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--shard',type=int,choices=range(6));args=p.parse_args()
    if args.mode=='prepare':prepare(args.out)
    elif args.mode=='run':worker(args.out,args.shard)
    elif args.mode=='finish':
        for suite in SOURCES:finish_suite(args.out,suite)
        finish_all(args.out)
    else:status(args.out)


if __name__=='__main__':main()

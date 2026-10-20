"""Prefer shared successful actions when constructing router training labels."""
import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from fit_asset_router import input_text, vectorizer, valid_actions, fit
from router_holdout import metrics
from simple_router import ROOT, ASSETS, SUITES, BINARY, load_data, save, digest

PREVIOUS = ROOT/'runs/router-holdout-70-30-v1'
DEFAULT = ROOT/'runs/router-majority-70-30-v1'


def majority_targets(train_data):
    """Within each sample's valid actions, prefer the largest training support.

    Support is computed separately for each known benchmark. Uncovered rows have
    no votes and no targets. Continuous tasks keep their original optimum labels.
    Stable ties prefer no-op then the frozen action order; no held-out inputs.
    """
    valid, covered = valid_actions(train_data)
    targets = np.zeros_like(valid)
    support = {}
    for suite in SUITES:
        rows = np.flatnonzero(train_data['suites'] == suite)
        counts = valid[rows].sum(0)
        support[suite] = counts
        for i in rows[covered[rows]]:
            votes = np.where(valid[i], counts, -1)
            targets[i, int(votes.argmax())] = True
    assert targets[covered].sum(1).min() == 1
    assert not (targets & ~valid).any()
    assert not targets[~covered].any()
    return targets, valid, covered, support


def distribution(actions, counts):
    return [dict(action=str(actions[i]),count=int(counts[i]))
            for i in np.argsort(-counts,kind='stable') if counts[i] > 0]


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    data=load_data()
    split=json.loads((PREVIOUS/'design.json').read_text())
    lookup={i:j for j,i in enumerate(data['ids'])}
    train=np.array([lookup[i] for i in split['train_ids']])
    test=np.array([lookup[i] for i in split['test_ids']])
    assert set(train).isdisjoint(test)
    assert not set(data['groups'][train]) & set(data['groups'][test])
    train_data={k:data[k][train].copy() for k in ['ids','suites','texts','reward']}
    targets,valid,covered,support=majority_targets(train_data)
    labels={}
    for suite in SUITES:
        mask=train_data['suites']==suite
        labels[suite]=dict(success_support=distribution(data['actions'],support[suite]),
             target_distribution=distribution(data['actions'],targets[mask].sum(0)),
             n_train=int(mask.sum()),n_supervised=int((mask&covered).sum()))
    save(out/'design.json',dict(seed=split['seed'],train_ids=split['train_ids'],test_ids=split['test_ids'],
         previous_split_sha256=digest(PREVIOUS/'design.json'),n_train=len(train),n_test=len(test),
         change='Only training target selection changes; retain corrected text features and linear model.',
         target_rule='For each train sample, choose the valid action with most successful train samples in its benchmark.',
         ties='no-op then frozen action order',no_test_votes=True,
         continuous_rule='Only maximum-reward actions remain valid, without a near-optimum tolerance.',
         hyperparameters='same as previous: L-BFGS, regularization 1e-6, maxiter 300, train margin stop 0.5',
         evaluation='same previously examined holdout, exploratory method iteration, offline outcome replay'))
    save(out/'label_support.json',labels)
    teacher=[]
    for i in range(len(train)):
        teacher.append(dict(id=str(train_data['ids'][i]),suite=str(train_data['suites'][i]),
             original_valid_actions=data['actions'][valid[i]].tolist(),
             majority_target=str(data['actions'][targets[i].argmax()]) if covered[i] else None))
    save(out/'training_targets.json',teacher)
    transform=vectorizer()
    texts=[input_text(s,q) for s,q in zip(train_data['suites'],train_data['texts'])]
    x=transform.fit_transform(texts)
    trace=[]
    w,bias=fit(x[covered],targets[covered],trace)
    train_chosen=np.asarray(x@w+bias).argmax(1)
    assert targets[np.arange(len(train)),train_chosen][covered].all()
    bundle=dict(transform=transform,w=w,bias=bias,actions=data['actions'],training_ids=train_data['ids'])
    joblib.dump(bundle,out/'router.joblib',compress=3)
    save(out/'training_trace.json',trace)
    restored=joblib.load(out/'router.joblib')
    test_texts=[input_text(s,q) for s,q in zip(data['suites'][test],data['texts'][test])]
    chosen=np.asarray(restored['transform'].transform(test_texts)@restored['w']+restored['bias']).argmax(1)
    decisions=[dict(id=str(data['ids'][i]),suite=str(data['suites'][i]),action=str(data['actions'][a]),
                    action_index=int(a)) for i,a in zip(test,chosen)]
    save(out/'frozen_test_actions.json',decisions)
    frozen_hash=digest(out/'frozen_test_actions.json')
    fixed=np.zeros(len(test),dtype=int)
    for suite in SUITES:
        tr=train[data['suites'][train]==suite]
        fixed[data['suites'][test]==suite]=data['reward'][tr].mean(0).argmax()
    result=metrics(data,test,chosen,fixed)
    previous=json.loads((PREVIOUS/'results.json').read_text())
    evaluated=[dict(**row,baseline_reward=float(data['reward'][i,0]),
         router_reward=float(data['reward'][i,a]),best_available_reward=float(data['reward'][i].max()))
         for row,i,a in zip(decisions,test,chosen)]
    save(out/'evaluated_test_actions.json',evaluated)
    training=dict(n=len(train),supervised=int(covered.sum()),fitted=int(covered.sum()),
                  iterations=len(trace),parameters=int(w.size+bias.size))
    output=dict(training=training,test=result,previous_test=previous['test'],
         source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'fit_asset_router.py',
             ROOT/'router_holdout.py',ROOT/'simple_router.py',ASSETS/'response_matrix.npz',ASSETS/'inputs.json']},
         frozen_test_actions_sha256=frozen_hash,split_sha256=digest(PREVIOUS/'design.json'))
    assert len(decisions)==53 and len(train)==122
    assert digest(out/'frozen_test_actions.json')==frozen_hash
    for suite in SUITES:
        assert result[suite]['baseline']==previous['test'][suite]['baseline']
    save(out/'results.json',output)
    save(out/'verification.json',dict(same_split_as_previous=True,train_targets_all_valid=True,
         no_test_votes=True,test_actions_saved_before_scoring=True,all_test_rows_included=True,
         training_target_hits=int(covered.sum()),supervised_train_rows=int(covered.sum())))
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['router_majority.py','fit_asset_router.py','router_holdout.py','simple_router.py']:
        shutil.copy2(ROOT/name,snapshot/name)
    report(out,output,labels)
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


def report(out,result,labels):
    lines=['# 多数成功方向优先的 Router','',
           '沿用上一轮完全相同的 122/53 随机 70:30 切分，仅改变训练标签构造。每个 benchmark 单独统计训练题的成功方向票数。每题在自己的成功集合内，选择票数最多的方向；并列时按冻结动作顺序选择，no-op 优先。没有成功方向的训练题不编造标签。', '',
           '这相当于尽量让原本有多个正确方向的题共享同一个训练目标。连续评分任务仍只接受该题最高分方向；本轮没有引入允许退分的阈值。', '',
           '| Benchmark | 测试题数 | Baseline | 原集合标签 Router | 多数优先 Router | 相对 Baseline（百分点） |',
           '|---|---:|---:|---:|---:|---:|']
    for suite,r in result['test'].items():
        old=result['previous_test'][suite]['router']
        lines.append(f"| {suite} | {r['n']} | {100*r['baseline']:.2f}% | {100*old:.2f}% | {100*r['router']:.2f}% | {r['delta_pp']:+.2f} |")
    r=result['test']['binary_total'];t=result['training']
    lines += ['',f"客观测试题：Baseline {r['baseline_correct']}/{r['n']}，新 Router {r['router_correct']}/{r['n']}；修复原错题 {r['fixed_wrong']} 道，损坏原对题 {r['broke_right']} 道。配对 bootstrap 95% 增益区间 {r['paired_bootstrap_95_pp']} 个百分点。",'',
              f"训练监督题 {t['supervised']} 道，全部选中指定的多数成功标签。训练 {t['iterations']} 次迭代。",'',
              '## 标签集中程度','',
              '| Benchmark | 有监督题数 | 使用的目标方向数 | 最大目标群题数 |','|---|---:|---:|---:|']
    for suite,label in labels.items():
        dist=label['target_distribution']
        lines.append(f"| {suite} | {label['n_supervised']} | {len(dist)} | {dist[0]['count'] if dist else 0} |")
    lines += ['', '前三个 benchmark 是准确率；TruthfulQA 为 MC2 概率质量，WritingBench 为清单均分。测试题不参与投票、词表拟合或标签生成。', '',
              '这是在已查看过结果的同一测试集上的方法迭代，属于探索性对照，不是新的独立验证；结果来自保存的回答矩阵回放，未重新生成答案。', '',
              'label_support.json 保存每个方向的训练支持数与目标分布；training_targets.json 保存每题完整成功集合和最终训练标签；frozen_test_actions.json 保存评分前的方向选择。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT)
    run(parser.parse_args().out)

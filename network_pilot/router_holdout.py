"""One frozen 70:30 random split for the corrected linear set-label router."""
import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split

from fit_asset_router import input_text, vectorizer, valid_actions, fit
from simple_router import ROOT, ASSETS, SUITES, BINARY, load_data, save, digest

DEFAULT = ROOT / 'runs/router-holdout-70-30-v1'
SEED = 20261016


def split(data, seed):
    train, test = train_test_split(np.arange(len(data['ids'])), test_size=.3,
                                   random_state=seed, stratify=data['suites'])
    train, test = np.sort(train), np.sort(test)
    assert not set(data['ids'][train]) & set(data['ids'][test])
    assert not set(data['groups'][train]) & set(data['groups'][test])
    assert len(train) + len(test) == len(data['ids'])
    return train, test


def train_router(train_data, out):
    # This function receives only the 70% training partition, including targets.
    valid, covered = valid_actions(train_data)
    transform = vectorizer()
    texts = [input_text(s, q) for s, q in zip(train_data['suites'], train_data['texts'])]
    x = transform.fit_transform(texts)
    trace = []
    w, bias = fit(x[covered], valid[covered], trace)
    chosen = np.asarray(x @ w + bias).argmax(1)
    assert valid[np.arange(len(chosen)), chosen][covered].all()
    bundle = dict(transform=transform, w=w, bias=bias, actions=train_data['actions'],
                  training_ids=train_data['ids'], known_uncovered_ids=train_data['ids'][~covered])
    joblib.dump(bundle, out/'router.joblib', compress=3)
    save(out/'training_trace.json', trace)
    return dict(n=len(chosen), supervised_rows=int(covered.sum()),
                fitted_supervised_rows=int(valid[np.arange(len(chosen)), chosen][covered].sum()),
                iterations=len(trace), parameters=int(w.size + bias.size))


def metrics(data, indices, chosen, fixed):
    baseline = data['reward'][indices, 0]
    routed = data['reward'][indices, chosen]
    summary = {}
    for suite in SUITES + ['binary_total']:
        mask = np.isin(data['suites'][indices], BINARY) if suite == 'binary_total' else data['suites'][indices] == suite
        rows = indices[mask]
        b, r = baseline[mask], routed[mask]
        delta = r - b
        rng = np.random.default_rng(SEED)
        boot = delta[rng.integers(0, len(rows), (10000, len(rows)))].mean(1)
        m = dict(n=len(rows), baseline=float(b.mean()), router=float(r.mean()),
                 delta_pp=float(delta.mean()*100),
                 paired_bootstrap_95_pp=(100*np.percentile(boot,[2.5,97.5])).tolist(),
                 train_selected_fixed=float(data['reward'][rows, fixed[mask]].mean()),
                 oracle=float(data['reward'][rows].max(1).mean()),
                 interventions=int((chosen[mask] != 0).sum()))
        if suite in BINARY + ['binary_total']:
            fixed_wrong = int(((b == 0) & (r == 1)).sum())
            broke_right = int(((b == 1) & (r == 0)).sum())
            m.update(baseline_correct=int(b.sum()), router_correct=int(r.sum()),
                     fixed_wrong=fixed_wrong, broke_right=broke_right,
                     mcnemar_exact_p=float(binomtest(fixed_wrong, fixed_wrong+broke_right).pvalue)
                     if fixed_wrong+broke_right else 1.)
        summary[suite] = m
    return summary


def run(out, seed):
    out.mkdir(parents=True, exist_ok=False)
    data = load_data()
    train, test = split(data, seed)
    design = dict(seed=seed, split='single random 70:30 split, stratified by benchmark, never by correctness',
         n_train=len(train), n_test=len(test), train_ids=data['ids'][train].tolist(), test_ids=data['ids'][test].tolist(),
         algorithm='corrected TF-IDF + one linear softmax layer; set-label successful-action loss',
         fit_vocabulary='train questions only, with full numbers/operators/entities retained',
         target_access='only train rewards supplied to fitting; test actions saved before evaluation',
         hyperparameters='unchanged from asset-router-fit-v3; regularization 1e-6; maxiter 300',
         stopping='all supervised training rows fit, with minimum winning logit margin 0.5',
         label_rule='all successful binary actions; maximum-reward continuous actions; exclude uncovered train rows',
         test_population='all test rows, including those without a known successful direction',
         measurements='offline saved-response replay; no fresh model generation',
         prior_pool_use='This same question pool was explored in earlier experiments; router split is newly frozen.')
    save(out/'design.json',design)
    # Pass a new dictionary of sliced arrays: fitting cannot index test rewards.
    train_data = {key:data[key][train].copy() for key in ['ids','suites','texts','reward']}
    train_data['actions'] = data['actions'].copy()
    training = train_router(train_data, out)
    bundle = joblib.load(out/'router.joblib')
    np.testing.assert_equal(bundle['training_ids'], data['ids'][train])
    test_texts = [input_text(s,q) for s,q in zip(data['suites'][test],data['texts'][test])]
    logits = np.asarray(bundle['transform'].transform(test_texts) @ bundle['w'] + bundle['bias'])
    chosen = logits.argmax(1)
    # Freeze predictions before reading any held-out outcomes for scoring.
    decisions = [dict(id=str(data['ids'][i]),suite=str(data['suites'][i]),
                      action=str(data['actions'][a]),action_index=int(a)) for i,a in zip(test,chosen)]
    save(out/'frozen_test_actions.json',decisions)
    frozen_sha = digest(out/'frozen_test_actions.json')
    fixed = np.zeros(len(test),dtype=int)
    for suite in SUITES:
        tr = train[data['suites'][train] == suite]
        fixed[data['suites'][test] == suite] = int(data['reward'][tr].mean(0).argmax())
    test_results = metrics(data,test,chosen,fixed)
    evaluated = []
    for row,i,a in zip(decisions,test,chosen):
        evaluated.append(dict(**row,baseline_reward=float(data['reward'][i,0]),
             router_reward=float(data['reward'][i,a]),best_available_reward=float(data['reward'][i].max())))
    save(out/'evaluated_test_actions.json',evaluated)
    # Independent arithmetic and source-integrity checks.
    assert len(decisions) == len(test) == len({r['id'] for r in decisions})
    assert not set(bundle['training_ids']) & {r['id'] for r in decisions}
    assert digest(out/'frozen_test_actions.json') == frozen_sha
    for r in evaluated:
        assert r['best_available_reward'] >= max(r['router_reward'],r['baseline_reward']) - 1e-12
    result = dict(seed=seed,model=data['protocol']['model'],training=training,test=test_results,
         n_train=len(train),n_test=len(test),frozen_test_actions_sha256=frozen_sha,
         source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
             [Path(__file__),ROOT/'fit_asset_router.py',ROOT/'simple_router.py',ASSETS/'response_matrix.npz',ASSETS/'inputs.json']})
    save(out/'results.json',result)
    save(out/'verification.json',dict(train_test_disjoint=True,duplicate_prompts_disjoint=True,
         vocabulary_fitted_on_train_only=True,training_function_received_only_train_targets=True,
         saved_model_reloaded=True,test_actions_frozen_before_scoring=True,
         all_test_rows_included=True,test_outcomes_did_not_select_hyperparameters=True))
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['router_holdout.py','fit_asset_router.py','simple_router.py']:
        shutil.copy2(ROOT/name,snapshot/name)
    report(out,result,design)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def report(out,result,design):
    lines=['# Router 随机 70% / 30% 实验','',
           f"固定随机种子 {result['seed']}，按 benchmark 分层随机切分：175 题中 {result['n_train']} 题训练，{result['n_test']} 题测试。分层不使用答案或正确率。只运行此一次切分，没有根据测试结果换种子或模型。",'',
           '使用已修正的简单线性 Router：TF-IDF 保留数字、运算符和低频实体，词表只在训练题拟合；训练标签为每题成功方向集合。测试题的选择先保存，再查已有回答矩阵评分。', '',
           '| Benchmark | 测试题数 | Baseline | Router | 变化（百分点） |',
           '|---|---:|---:|---:|---:|']
    for suite,r in result['test'].items():
        lines.append(f"| {suite} | {r['n']} | {100*r['baseline']:.2f}% | {100*r['router']:.2f}% | {r['delta_pp']:+.2f} |")
    total=result['test']['binary_total'];tr=result['training']
    lines += ['', '前三项及 binary_total 是准确率；TruthfulQA 是 MC2 概率质量，WritingBench 是清单均分，后两项不能称为准确率。', '',
              f"客观测试题共 {total['n']} 道：baseline 对 {total['baseline_correct']} 道，Router 对 {total['router_correct']} 道；修复 {total['fixed_wrong']} 道原错题，损坏 {total['broke_right']} 道原对题。",'',
              f"配对 bootstrap 95% 增益区间 [{total['paired_bootstrap_95_pp'][0]:+.2f}, {total['paired_bootstrap_95_pp'][1]:+.2f}] 个百分点；双侧精确 McNemar p={total['mcnemar_exact_p']:.4f}。样本很小，这是一轮原型实验。",'',
              f"训练中有可用监督标签的题 {tr['supervised_rows']} 道，已全部选中有效方向（{tr['fitted_supervised_rows']}/{tr['supervised_rows']}），用 {tr['iterations']} 次迭代。没有成功配置的训练题不编造标签；没有成功配置的测试题仍全部计入分母。",'',
              '这是已有生成结果的离线路由回放，没有重新生成答案。此前全题拟合的模型未加载；本轮从零训练。不过整个题池先前已参与几何探索，因此这不是一个从未接触过的新 benchmark 题池。', '',
              'design.json 保存完整切分；frozen_test_actions.json 保存评分前的方向选择；evaluated_test_actions.json 保存逐题 baseline、Router 和字典最高分；router.joblib 保存只用 70% 训练的模型。', '',
              '```bash','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python router_holdout.py', '```']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    run(args.out,args.seed)

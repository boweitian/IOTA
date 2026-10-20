"""Fit the known per-question direction assets before evaluating generalization.

A single linear softmax layer on TF-IDF text, trained with set-valued targets:
minimize -log(sum(probability of this sample's valid actions)). No sample ID input,
no nearest-neighbor averaging, no reference-answer input, no GPU or generation.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer

from simple_router import ROOT, ASSETS, SUITES, BINARY, load_data, save, digest

DEFAULT = ROOT / 'runs/asset-router-fit-v3'


def input_text(suite, question):
    return 'Task: ' + suite + '\nQuestion: ' + question


def vectorizer():
    # Keep rare entity names, single-digit numbers and operators. Dropping these
    # can map different arithmetic questions or Austria/Australia to one vector.
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                           lowercase=False, token_pattern=r'(?u)\b\w+\b|[^\w\s]')


def valid_actions(data):
    binary = np.isin(data['suites'], BINARY)
    reward = data['reward']
    valid = np.isclose(reward, reward.max(1, keepdims=True), atol=1e-12, rtol=0)
    # All-zero binary rows have no correct steering target. Do not invent one.
    valid[binary] = reward[binary] == 1.
    return valid, valid.any(1)


def set_loss(parameters, x, valid, regularization=1e-6):
    dimension, actions = x.shape[1], valid.shape[1]
    w = parameters[:dimension * actions].reshape(dimension, actions)
    bias = parameters[dimension * actions:]
    logits = x @ w + bias
    all_z = logsumexp(logits, axis=1)
    positive_logits = np.where(valid, logits, -np.inf)
    pos_z = logsumexp(positive_logits, axis=1)
    loss = np.mean(all_z - pos_z) + .5 * regularization * np.sum(w * w)
    gradient_logits = (np.exp(logits - all_z[:, None])
                       - np.exp(positive_logits - pos_z[:, None])) / len(valid)
    gradient_w = x.T @ gradient_logits + regularization * w
    gradient = np.concatenate([np.asarray(gradient_w).ravel(), gradient_logits.sum(0)])
    return float(loss), gradient


def fit(x, valid, trace, require_full_fit=True):
    dimension, actions = x.shape[1], valid.shape[1]
    initial = np.zeros((dimension + 1) * actions)
    state = {'parameters': initial}

    class Memorized(Exception):
        pass

    def record(parameters):
        state['parameters'] = parameters.copy()
        w = parameters[:dimension * actions].reshape(dimension, actions)
        bias = parameters[dimension * actions:]
        logits = x @ w + bias
        selected = logits.argmax(1)
        hits = valid[np.arange(len(valid)), selected]
        margin = np.max(np.where(valid, logits, -np.inf), axis=1) - np.max(np.where(~valid, logits, -np.inf), axis=1)
        row = dict(iteration=len(trace) + 1, hits=int(hits.sum()), total=len(hits),
                   loss=set_loss(parameters, x, valid)[0], min_winning_margin=float(margin.min()))
        trace.append(row)
        if row['iteration'] % 10 == 0 or hits.all():
            print('FIT', json.dumps(row), flush=True)
        if hits.all() and margin.min() >= .5:
            raise Memorized()

    try:
        result = minimize(set_loss, initial, args=(x, valid), method='L-BFGS-B', jac=True,
                          callback=record, options=dict(maxiter=300, ftol=1e-12, gtol=1e-8, maxcor=10))
        parameters = result.x
    except Memorized:
        parameters = state['parameters']
    w = parameters[:dimension * actions].reshape(dimension, actions).astype(np.float32)
    bias = parameters[dimension * actions:].astype(np.float32)
    if require_full_fit:
        assert valid[np.arange(len(valid)), (x @ w + bias).argmax(1)].all()
    return w, bias


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    data = load_data()
    valid, covered = valid_actions(data)
    texts = [input_text(s, q) for s, q in zip(data['suites'], data['texts'])]
    transform = vectorizer()
    x = transform.fit_transform(texts)
    save(out / 'design.json', dict(objective='Fit every already discovered per-question direction asset',
         training_and_evaluation='same 175 original questions; training fit, not unseen-question accuracy',
         feature='TF-IDF words, numbers, punctuation and bigrams plus benchmark name, no vocabulary truncation, IDs or answers',
         model='single linear softmax layer', targets='all successful binary actions; maximum-reward continuous actions',
         uncovered_binary_policy='exclude from training because no correct action exists in the measured bank',
         stopping='all covered rows choose a valid action, with logit winning margin >= 0.5',
         max_iterations=300, regularization=1e-6))
    trace = []
    w, bias = fit(x[covered], valid[covered], trace)
    chosen = np.asarray(x @ w + bias).argmax(1)
    actual = data['reward'][np.arange(len(chosen)), chosen]
    selected_valid = valid[np.arange(len(chosen)), chosen]
    assert selected_valid[covered].all()
    catalog = {a['id']: a for a in json.loads((ASSETS / 'actions.json').read_text())}
    bundle = dict(transform=transform, w=w, bias=bias, actions=data['actions'],
                  catalog={str(a): catalog[str(a)] for a in data['actions']},
                  model=data['protocol']['model'], geometry_sha256=digest(ASSETS / 'geometry.npz'),
                  training_ids=data['ids'], known_uncovered_ids=data['ids'][~covered])
    joblib.dump(bundle, out / 'router.joblib', compress=3)
    # Verify the saved model, using only task text as inputs for all original questions.
    restored = joblib.load(out / 'router.joblib')
    again = np.asarray(restored['transform'].transform(texts) @ restored['w'] + restored['bias']).argmax(1)
    np.testing.assert_equal(chosen, again)
    summary = {}
    for suite in SUITES + ['binary_total', 'all']:
        mask = np.ones(len(chosen), dtype=bool) if suite == 'all' else (
            np.isin(data['suites'], BINARY) if suite == 'binary_total' else data['suites'] == suite)
        summary[suite] = dict(n=int(mask.sum()), covered=int((mask & covered).sum()),
             covered_direction_hits=int((mask & selected_valid).sum()),
             covered_direction_fit=float(selected_valid[mask & covered].mean()),
             baseline_score=float(data['reward'][mask, 0].mean()), router_score=float(actual[mask].mean()),
             best_available_score=float(data['reward'][mask].max(1).mean()),
             uncovered_ids=data['ids'][mask & ~covered].tolist())
    # Recheck the actual old deployment on its own training questions.
    from simple_router import scores
    previous = joblib.load(ROOT / 'runs/simple-router-v1/text_router.joblib')
    old_x = previous['transform'].transform(data['texts']).toarray()
    old_choice = np.zeros(len(chosen), dtype=int)
    for suite in SUITES:
        mask = data['suites'] == suite
        old_choice[mask] = scores(previous['models'][suite], old_x[mask]).argmax(1)
    binary = np.isin(data['suites'], BINARY)
    result = dict(summary=summary, trainable_samples=int(covered.sum()),
         old_deployed_3nn_training_correct=int(data['reward'][np.flatnonzero(binary), old_choice[binary]].sum()),
         linear_parameters=int(w.size + bias.size), training_iterations=len(trace),
         model_bytes=(out / 'router.joblib').stat().st_size,
         evaluation='in-sample fit on saved response matrix; no new generations and no generalization claim',
         source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ASSETS/'response_matrix.npz', ASSETS/'inputs.json', ROOT/'runs/simple-router-v1/text_router.joblib']})
    save(out / 'results.json', result)
    save(out / 'training_trace.json', trace)
    with (out / 'per_question_fit.jsonl').open('w') as f:
        for i in range(len(chosen)):
            row = dict(id=str(data['ids'][i]), suite=str(data['suites'][i]), covered=bool(covered[i]),
                 valid_actions=data['actions'][valid[i]].tolist(), selected_action=str(data['actions'][chosen[i]]),
                 selected_valid=bool(selected_valid[i]), selected_reward=float(actual[i]),
                 best_available_reward=float(data['reward'][i].max()))
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    lines = ['# 原题 Router 拟合审计与修正', '',
             '上一版实际部署的是 3 近邻收益平均，原题正确 92/111，低于已测方向字典的 101/111。它为新题泛化选择参数，没有先满足原题资产拟合要求。五折留出的 54.05% 也不能作为原题拟合率。', '',
             '本版使用一个线性 softmax 层。输入为题目文本和任务类型；每题全部成功方向构成监督集合。损失为负对数成功方向总概率。无成功方向的 10 道客观题不编造正确标签，不参与成功方向训练。连续评分题监督其已测最高分配置。', '',
             '| Benchmark | 原题数 | 有成功方向的题 | 成功方向选中数 | 已覆盖题拟合率 | 全题 Router 得分 | 已测字典上限 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for suite, r in summary.items():
        lines.append(f"| {suite} | {r['n']} | {r['covered']} | {r['covered_direction_hits']} | {100*r['covered_direction_fit']:.2f}% | {100*r['router_score']:.2f}% | {100*r['best_available_score']:.2f}% |")
    lines += ['', 'GSM8K/BBH/IFEval 的得分为准确率；TruthfulQA 为 MC2 概率质量，WritingBench 为清单评分。all 的跨任务平均只是混合 reward 均值，不是准确率。', '',
              '101 道已有正确方向的客观题必须达到 101/101；包含未覆盖的 10 题时，当前字典下准确率最多 101/111。要达到全题接近 100%，需要先为未覆盖题补出有效配置。', '',
              f"实际模型 {result['linear_parameters']:,} 个参数，保存文件 {result['model_bytes']/1024:.1f} KiB，训练 {result['training_iterations']} 次迭代。保存并重载后，逐题方向选择一致。", '',
              '这是原题训练拟合，使用现有测量结果回放，不能作为新题准确率或在线生成复现率。上一版新题测试结果仍保留，未被本次训练拟合替代。', '',
              '```bash', 'python fit_asset_router.py predict --bundle runs/asset-router-fit-v3/router.joblib \\',
              '  --suite gsm8k --question "Your question"', '```', '',
              '逐题选中方向、全部有效标签和未覆盖题见 per_question_fit.jsonl；训练收敛过程见 training_trace.json。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('run'); p.add_argument('--out', type=Path, default=DEFAULT)
    p = sub.add_parser('predict'); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--suite', choices=SUITES, required=True); p.add_argument('--question', required=True)
    args = parser.parse_args()
    if args.mode == 'run':
        run(args.out)
    else:
        bundle = joblib.load(args.bundle)
        x = bundle['transform'].transform([input_text(args.suite, args.question)])
        logits = np.asarray(x @ bundle['w'] + bundle['bias'])[0]
        indices = np.argsort(-logits, kind='stable')[:5]
        print(json.dumps(dict(model=bundle['model'],
             recommended=bundle['catalog'][str(bundle['actions'][indices[0]])],
             top5=[str(bundle['actions'][i]) for i in indices],
             note='Training-fit router; recommendation does not guarantee an answer on unseen questions.'),
             ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

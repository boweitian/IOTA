"""CPU-only supervised router over the frozen per-question steering response matrix.

Inputs are prompt text and optionally unsteered prompt-state projections, never answers.
Every direction/dose is an action; all observed action rewards supervise the router.
Evaluation selects one action before looking up its saved outcome. No new generations.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'runs/expanded-assets-v2/asset_bundle'
DEFAULT = ROOT / 'runs/simple-router-v1'
SUITES = ['gsm8k', 'bbh', 'ifeval', 'truthfulqa', 'writingbench']
BINARY = SUITES[:3]
SEED = 20261016
# Fixed before evaluating this experiment; outer outcomes never select a model.
SPECS = [dict(feature=f, algorithm=a, parameter=p)
         for f in ['text', 'geometry']
         for a, ps in [('knn', [1, 3, 7]), ('ridge', [.1, 1., 10.])]
         for p in ps]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_group(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', text.strip().lower()).encode()).hexdigest()


def load_data():
    matrix = np.load(ASSETS / 'response_matrix.npz')
    ids = matrix['sample_ids'].astype(str)
    protocol = json.loads((ASSETS / 'inputs.json').read_text())
    by_id = {r['id']: r for r in protocol['records']}
    rows = [by_id[i] for i in ids]
    assert len(set(ids)) == len(ids)
    assert matrix['observed'].all() and np.isfinite(matrix['reward']).all()
    assert str(matrix['action_ids'][0]) == 'none'
    # The cached states were measured on inputs alone, before steering/generation.
    states = {}
    feature_paths = sorted((ROOT / 'runs/pilot-2124657').glob('*/*_features.npz'))
    for path in feature_paths:
        d = np.load(path)
        for i, pooled, last in zip(d['ids'].astype(str), d['pooled'], d['last']):
            assert i not in states
            states[i] = (pooled, last)
    unit = np.load(ASSETS / 'geometry.npz')['unit']
    parts = []
    for position in [0, 1]:
        h = np.stack([states[i][position] for i in ids]).astype(float)
        h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
        parts.append(h @ unit.T)
    return dict(ids=ids, rows=rows, texts=np.array([r['question'] for r in rows]),
                suites=np.array([r['suite'] for r in rows]),
                groups=np.array([prompt_group(r['question']) for r in rows]),
                geometry=np.concatenate(parts, axis=1), reward=matrix['reward'],
                success=matrix['success'], actions=matrix['action_ids'].astype(str),
                feature_paths=feature_paths, protocol=protocol)


def folds(indices, data, count, seed):
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    for train, test in splitter.split(indices, data['suites'][indices], data['groups'][indices]):
        tr, te = indices[train], indices[test]
        assert not (set(data['groups'][tr]) & set(data['groups'][te]))
        yield tr, te


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prepare(feature, data, train, test):
    if feature == 'text':
        transform = TfidfVectorizer(ngram_range=(1, 2), max_features=2048,
                                    sublinear_tf=True, strip_accents='unicode')
        x = transform.fit_transform(data['texts'][train]).toarray()
        z = transform.transform(data['texts'][test]).toarray()
    else:
        x = data['geometry'][train]
        mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
        transform = dict(mean=mean, scale=scale)
        x = normalize((x - mean) / scale)
        z = normalize((data['geometry'][test] - mean) / scale)
    return x, z, transform


def fit_model(x, rewards, spec):
    # Multi-output prediction retains all successful directions, including ties.
    model = dict(spec=spec, x=x, mean=rewards.mean(0))
    if spec['algorithm'] == 'knn':
        model['rewards'] = rewards
    else:
        # Dual ridge regression: solve in number of samples, not vocabulary size.
        k = x @ x.T
        model['dual'] = np.linalg.solve(k + spec['parameter'] * np.eye(len(x)),
                                        rewards - model['mean'])
    return model


def scores(model, z):
    similarity = z @ model['x'].T
    if model['spec']['algorithm'] == 'ridge':
        return model['mean'] + similarity @ model['dual']
    k = min(int(model['spec']['parameter']), len(model['x']))
    near = np.argsort(-similarity, axis=1, kind='stable')[:, :k]
    return model['rewards'][near].mean(1)


def rank_actions(predicted):
    # Stable ties prefer no-op (column 0); no label-informed action tie-breaking.
    return np.argsort(-predicted, axis=1, kind='stable')


def predict_spec(spec, data, train, test, return_models=False):
    x, z, transform = prepare(spec['feature'], data, train, test)
    predictions = np.empty((len(test), len(data['actions'])))
    models = {}
    for suite in SUITES:
        tr = data['suites'][train] == suite
        te = data['suites'][test] == suite
        assert tr.any(), suite
        model = fit_model(x[tr], data['reward'][train[tr]], spec)
        predictions[te] = scores(model, z[te])
        if return_models:
            models[suite] = model
    if return_models:
        return predictions, dict(spec=spec, transform=transform, models=models)
    return predictions


def macro_reward(data, indices, selected):
    values = data['reward'][indices, selected]
    return float(np.mean([values[data['suites'][indices] == s].mean()
                          for s in SUITES if (data['suites'][indices] == s).any()]))


def tune(data, train, seed):
    # Each inner validation row is unseen by its vocabulary/scaler and model.
    splits = list(folds(train, data, 3, seed))
    candidates = []
    for number, spec in enumerate(SPECS):
        chosen = np.empty(len(data['ids']), dtype=int)
        for tr, va in splits:
            chosen[va] = rank_actions(predict_spec(spec, data, tr, va))[:, 0]
        candidates.append(dict(index=number, spec=spec,
                               score=macro_reward(data, train, chosen[train])))
    best = {}
    for name, allowed in [('text', ['text']), ('geometry', ['geometry']),
                           ('adaptive', ['text', 'geometry'])]:
        # Deterministic first-in-list tie-break; all choices are on train rows.
        best[name] = max((c for c in candidates if c['spec']['feature'] in allowed),
                         key=lambda c: (c['score'], -c['index']))
    return best, candidates


def fixed_actions(data, train, test):
    result = np.zeros(len(test), dtype=int)
    for suite in SUITES:
        tr = train[data['suites'][train] == suite]
        result[data['suites'][test] == suite] = int(data['reward'][tr].mean(0).argmax())
    return result


def summarize(data, indices, predictions, rankings):
    summary = {}
    for suite in SUITES + ['binary_total']:
        mask = np.isin(data['suites'][indices], BINARY) if suite == 'binary_total' else data['suites'][indices] == suite
        rows = indices[mask]
        y = data['reward'][rows]
        entry = dict(n=len(rows), baseline=float(y[:, 0].mean()),
                     oracle=float(y.max(1).mean()), methods={})
        for name, selected in predictions.items():
            a = selected[mask]
            r = data['reward'][rows, a]
            delta = r - y[:, 0]
            rng = np.random.default_rng(SEED)
            boot = delta[rng.integers(0, len(rows), (5000, len(rows)))].mean(1)
            result = dict(score=float(r.mean()), delta_pp=float(100 * delta.mean()),
                          delta_bootstrap_95_pp=(100 * np.percentile(boot, [2.5, 97.5])).tolist(),
                          interventions=int((a != 0).sum()))
            if suite in BINARY + ['binary_total']:
                assert np.isin(r, [0., 1.]).all()
                result.update(correct=int(r.sum()), fixed_wrong=int(((y[:, 0] == 0) & (r == 1)).sum()),
                              broke_right=int(((y[:, 0] == 1) & (r == 0)).sum()))
            if name in rankings:
                order = rankings[name][mask]
                result['topk_oracle'] = {str(k): float(np.take_along_axis(y, order[:, :k], axis=1).max(1).mean())
                                         for k in [1, 3, 5, 10]}
            entry['methods'][name] = result
        summary[suite] = entry
    return summary


def evaluate(data, out):
    all_rows = np.arange(len(data['ids']))
    selected = {k: np.full(len(all_rows), -1, dtype=int)
                for k in ['fixed', 'text', 'geometry', 'adaptive']}
    rankings = {k: np.full(data['reward'].shape, -1, dtype=int)
                for k in ['text', 'geometry', 'adaptive']}
    memberships, audit = [], []
    for number, (tr, te) in enumerate(folds(all_rows, data, 5, SEED)):
        best, grid = tune(data, tr, SEED + number + 1)
        selected['fixed'][te] = fixed_actions(data, tr, te)
        cache = {}
        for name, choice in best.items():
            key = choice['index']
            if key not in cache:
                cache[key] = rank_actions(predict_spec(choice['spec'], data, tr, te))
            rankings[name][te] = cache[key]
            selected[name][te] = cache[key][:, 0]
        audit.append(dict(fold=number, train_ids=data['ids'][tr].tolist(),
                          test_ids=data['ids'][te].tolist(), selected=best, inner_grid=grid))
        memberships.extend(dict(id=str(data['ids'][i]), fold=number) for i in te)
        print('OUTER FOLD', number, 'n_train', len(tr), 'n_test', len(te),
              'selected', best['adaptive']['spec'], flush=True)
    assert len({r['id'] for r in memberships}) == len(all_rows)
    assert all((v >= 0).all() for v in selected.values())
    nested = summarize(data, all_rows, selected, rankings)
    save(out / 'fold_audit.json', audit)
    fold_of = {r['id']: r['fold'] for r in memberships}
    with (out / 'oof_predictions.jsonl').open('w') as f:
        for i in all_rows:
            r = dict(id=str(data['ids'][i]), suite=str(data['suites'][i]), fold=fold_of[data['ids'][i]],
                     baseline=float(data['reward'][i, 0]), oracle=float(data['reward'][i].max()),
                     methods={name: dict(action=str(data['actions'][a[i]]), reward=float(data['reward'][i, a[i]]),
                                         top5_actions=[str(data['actions'][j]) for j in rankings[name][i, :5]] if name in rankings else [])
                              for name, a in selected.items()})
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    # A secondary, original split check: search+validation train, original test evaluate.
    train = np.array([i for i, r in enumerate(data['rows']) if r['split'] != 'test'])
    test = np.array([i for i, r in enumerate(data['rows']) if r['split'] == 'test'])
    assert not (set(data['groups'][train]) & set(data['groups'][test]))
    best, grid = tune(data, train, SEED + 100)
    hold_selected = {'fixed': fixed_actions(data, train, test)}
    hold_rankings = {}
    for name, choice in best.items():
        ranks = rank_actions(predict_spec(choice['spec'], data, train, test))
        hold_selected[name], hold_rankings[name] = ranks[:, 0], ranks
    save(out / 'original_split_audit.json', dict(train_ids=data['ids'][train].tolist(),
         test_ids=data['ids'][test].tolist(), selected=best, inner_grid=grid))
    holdout = summarize(data, test, hold_selected, hold_rankings)
    # Deployable text-only router, refitted on the full available dictionary.
    full_best, _ = tune(data, all_rows, SEED + 200)
    _, bundle = predict_spec(full_best['text']['spec'], data, all_rows, all_rows, True)
    bundle.update(actions=data['actions'], model=data['protocol']['model'],
                  training_ids=data['ids'], geometry_sha256=digest(ASSETS / 'geometry.npz'))
    joblib.dump(bundle, out / 'text_router.joblib', compress=3)
    # A 1-NN geometry fit on the same dictionary demonstrates asset memorization.
    memory = rank_actions(predict_spec(dict(feature='geometry', algorithm='knn', parameter=1),
                                      data, all_rows, all_rows))[:, 0]
    memory_summary = summarize(data, all_rows, {'dictionary_fit': memory}, {})
    assert np.allclose(data['reward'][all_rows, memory], data['reward'].max(1))
    result = dict(seed=SEED, model=data['protocol']['model'], samples=len(all_rows), actions=len(data['actions']),
                  nested_5fold=nested, original_split=holdout, dictionary_fit=memory_summary,
                  deployment_spec=full_best['text']['spec'],
                  note='Offline replay of saved action outcomes; single selected action per unseen question. '
                       'Top-k is checker/oracle coverage, not one-call accuracy. '
                       'Paired bootstrap intervals are descriptive on OOF predictions, not full refit uncertainty. '
                       'The pool was previously explored; these are router-held-out questions, not a pristine benchmark.',
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                      [Path(__file__), ASSETS/'response_matrix.npz', ASSETS/'inputs.json', ASSETS/'geometry.npz']
                      + data['feature_paths']})
    save(out / 'results.json', result)
    report(result, out)
    print(json.dumps(nested['binary_total'], indent=2), flush=True)


def report(result, out):
    lines = ['# 简单 Router：从题目选择 steering 方向', '',
             'Qwen3.5-0.8B / layer 15，27 个方向、109 个配置，175 道现有题。CPU 训练，无新大模型生成。', '',
             '用每道训练题的全部 109 个结果学习多输出收益预测。文本输入为词/二元词 TF-IDF；几何输入为原模型未干预的 pooled/last 状态在 27 方向上的 54 维投影。每个 benchmark 各自拟合，使用已知 benchmark 标识。', '',
             '候选为 1/3/7 近邻及三个正则强度的线性 ridge。外层五折按题划分；内层三折只用外层训练题选择参数和输入类型。词表、均值和尺度均在训练折拟合。精确重复问题归为同组。', '',
             '## 逐题留出结果（每题预测一次）', '',
             '| Benchmark | 题数 | 不干预 | 训练题选固定方向 | 文本 Router | 几何 Router | 内层选择 Router | 字典上限 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for suite, r in result['nested_5fold'].items():
        m = r['methods']
        vals = [r['baseline'], m['fixed']['score'], m['text']['score'], m['geometry']['score'], m['adaptive']['score'], r['oracle']]
        lines.append('| '+suite+' | '+str(r['n'])+' | '+' | '.join(f'{100*v:.2f}%' for v in vals)+' |')
    lines += ['', '前三项及 binary_total 是正确率；TruthfulQA 是 MC2 概率质量，WritingBench 是清单均分，后两者不能称为准确率。', '',
              '## 正确率与覆盖率的区别', '']
    r = result['nested_5fold']['binary_total']; a = r['methods']['adaptive']
    lines += [f"111 道客观题：原模型 {round(r['baseline']*r['n'])}/{r['n']}；Router {a['correct']}/{r['n']}。修复原错题 {a['fixed_wrong']} 道，损坏原对题 {a['broke_right']} 道。净变化 {a['delta_pp']:+.2f} 个百分点，描述性配对 bootstrap 95% 区间 {a['delta_bootstrap_95_pp']}。", '',
              '同一字典训练并查询的 1-NN 达到逐题字典上限，这是资产记忆能力；上表的留出结果衡量对未见题目的选方向能力。保留原模型作为候选保证字典上限不低于 baseline，但预测 Router 仍可能选错方向。', '',
              'Router 前 3/5/10 个候选的最好成绩需要检查器判断或答案验证；不能作为单次调用准确率。具体数值见 results.json 的 topk_oracle。', '',
              '## 原始 search/validation → test 切分复核', '',
              '| Benchmark | 测试题数 | 不干预 | 文本 Router | 几何 Router | 内层选择 Router |',
              '|---|---:|---:|---:|---:|---:|']
    for suite, r in result['original_split'].items():
        vals = [r['baseline']] + [r['methods'][name]['score'] for name in ['text','geometry','adaptive']]
        lines.append('| '+suite+' | '+str(r['n'])+' | '+' | '.join(f'{100*v:.2f}%' for v in vals)+' |')
    lines += ['', '## 使用与限制', '',
              'text_router.joblib 是用全部 175 题重新拟合的文本 Router；推理只需题目和 benchmark 名称，不读参考答案。模型与方向绑定 Qwen3.5-0.8B，不能直接用于 9B。', '',
              '```bash', 'python simple_router.py predict --bundle runs/simple-router-v1/text_router.joblib \\',
              '  --suite gsm8k --question "Your new math question"', '```', '',
              '这是已有回答矩阵的离线路由回放，未重新运行被选方向。已有实验观察到 BF16 生成随批次变化；新的在线生成准确率仍需要匹配复跑。每个 benchmark 仅 28–39 题，且题池已被用于几何探索；结果适合原型判断。', '',
              'results.json 保存指标和来源哈希；fold_audit.json 保存训练/测试 ID、每折选型和内层结果；oof_predictions.jsonl 保存每题选中的方向、得分及前五候选。accuracy.svg 为独立对比图。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    # Lightweight standalone SVG, with no plotting runtime or external resources.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="360" viewBox="0 0 920 360">',
           '<rect width="920" height="360" fill="#f7f6f0"/>',
           '<g font-family="sans-serif" fill="#203b33"><text x="28" y="32" font-size="20">IOTA simple router · nested 5-fold offline accuracy</text>',
           '<text x="28" y="56" font-size="13">0.8B · one selected action per question · oracle uses answers</text>']
    colors = ['#a4aaa5','#b49a72','#367666','#88aecd']
    names = ['baseline','fixed','router','oracle']
    for row, suite in enumerate(BINARY + ['binary_total']):
        r = result['nested_5fold'][suite]; y = 86 + row*58
        svg.append(f'<text x="28" y="{y+18}" font-size="14">{suite} (n={r["n"]})</text>')
        values = [r['baseline'],r['methods']['fixed']['score'],r['methods']['adaptive']['score'],r['oracle']]
        for j, (v, color) in enumerate(zip(values, colors)):
            x = 225 + j*170
            svg.append(f'<rect x="{x}" y="{y}" width="{v*120:.2f}" height="20" fill="{color}"/>')
            svg.append(f'<text x="{x}" y="{y+36}" font-size="12">{names[j]} {100*v:.1f}%</text>')
    svg.append('</g></svg>')
    (out / 'accuracy.svg').write_text('\n'.join(svg))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    run = sub.add_parser('run'); run.add_argument('--out', type=Path, default=DEFAULT)
    pred = sub.add_parser('predict'); pred.add_argument('--bundle', type=Path, required=True)
    pred.add_argument('--suite', choices=SUITES, required=True); pred.add_argument('--question', required=True)
    args = p.parse_args()
    if args.mode == 'run':
        args.out.mkdir(parents=True, exist_ok=False)
        data = load_data()
        save(args.out / 'design.json', dict(seed=SEED, specs=SPECS, outer_folds=5, inner_folds=3,
             samples=len(data['ids']), prompt_only=True, new_generations=False,
             objective='Choose one steering action per new question from complete multi-label outcomes'))
        evaluate(data, args.out)
    else:
        bundle = joblib.load(args.bundle)
        x = bundle['transform'].transform([args.question]).toarray()
        values = scores(bundle['models'][args.suite], x)
        order = rank_actions(values)[0]
        print(json.dumps(dict(model=bundle['model'], suite=args.suite, action=str(bundle['actions'][order[0]]),
             top5=[dict(action=str(bundle['actions'][i]), predicted_reward=float(values[0, i])) for i in order[:5]],
             note='Predictions are uncalibrated utilities, not correctness probabilities.'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

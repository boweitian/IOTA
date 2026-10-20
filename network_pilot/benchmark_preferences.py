"""Benchmark-specific fixed-direction preferences from the complete per-question matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'runs/sample-assets-v1'


def analyze(matrix,actions,seed=20261013,repetitions=10000):
    """Equal-weight sample scores; retain ties and resample entire response rows."""
    r=np.asarray(matrix,dtype=float)
    assert r.ndim==2 and r.shape[1]==len(actions) and np.isfinite(r).all()
    baseline=next(i for i,a in enumerate(actions) if a['id']=='none')
    means=r.mean(0);order=sorted(range(len(actions)),key=lambda j:(-means[j],actions[j]['norm'],actions[j]['id']))
    top=np.flatnonzero(np.isclose(means,means.max(),rtol=0,atol=1e-12))
    samples=np.random.default_rng(seed).integers(0,len(r),(repetitions,len(r)))
    bootstrap=r[samples].mean(1)
    tied=np.isclose(bootstrap,bootstrap.max(1,keepdims=True),rtol=0,atol=1e-12)
    win_share=(tied/tied.sum(1,keepdims=True)).mean(0)
    entries=[]
    for j in order:
        gain=r[:,j]-r[:,baseline]
        entries.append(dict(action=actions[j]['id'],configuration=actions[j],mean_score=float(means[j]),
            gain_from_baseline=float(gain.mean()),improved_samples=int((gain>1e-12).sum()),
            regressed_samples=int((gain< -1e-12).sum()),unchanged_samples=int((np.abs(gain)<=1e-12).sum()),
            paired_gain_interval=np.quantile(bootstrap[:,j]-bootstrap[:,baseline],[.025,.975]).tolist(),
            bootstrap_best_share=float(win_share[j])))
    assert means.max()>=means[baseline]-1e-12
    oracle=float(r.max(1).mean());assert oracle>=means.max()-1e-12
    gap=oracle-means[baseline]
    lower=[x for x in means if x<means.max()-1e-12]
    return dict(samples=len(r),baseline_mean=float(means[baseline]),best_fixed_mean=float(means.max()),
        best_fixed_actions=[actions[j]['id'] for j in order if j in top],
        per_question_best_mean=oracle,
        fraction_of_search_gain_retained=float((means.max()-means[baseline])/gap) if gap>1e-12 else None,
        margin_over_next_distinct_score=float(means.max()-max(lower)) if lower else None,
        ranked_directions=entries)


def label(a):
    if a['id']=='none':return '原模型'
    return a['concept_name']+f" {a['requested_strength']:+g}"


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);args=p.parse_args()
    run=args.run.resolve();bundle=run/'asset_bundle';out=run/'benchmark_preference';out.mkdir(exist_ok=True)
    d=np.load(bundle/'response_matrix.npz');assert d['observed'].all()
    protocol=json.loads((bundle/'inputs.json').read_text());rows={r['id']:r for r in protocol['records']}
    configs={a['id']:a for a in json.loads((bundle/'geometry.json').read_text())['exhaustive_bank']}
    aids=d['action_ids'].tolist();actions=[configs[a] for a in aids];sample_ids=d['sample_ids'].tolist()
    matrices={s:d['reward'][[i for i,x in enumerate(sample_ids) if rows[x]['suite']==s]] for s in protocol['suites']}
    suites={s:analyze(r,actions,seed=20261013+i) for i,(s,r) in enumerate(matrices.items())}
    bbh={task:analyze(d['reward'][[i for i,x in enumerate(sample_ids) if rows[x]['suite']=='bbh' and rows[x]['task']==task]],actions)
         for task in sorted({r['task'] for r in rows.values() if r['suite']=='bbh'})}
    chosen=list(dict.fromkeys(a for s in suites.values() for a in s['best_fixed_actions']))
    cross={a:{s:float((r[:,aids.index(a)]-r[:,aids.index('none')]).mean()) for s,r in matrices.items()} for a in chosen}
    result=dict(objective='Each benchmark uses one identical steering direction and dose for every question',
        samples=len(sample_ids),declared_configurations=len(actions),suites=suites,bbh_subtasks=bbh,
        selected_direction_cross_benchmark_gains=cross,
        uncertainty='Paired row bootstrap of existing observations, ties receive equal best shares; descriptive selection stability, not an independent confirmation',
        source_hashes={n:hashlib.sha256((bundle/n).read_bytes()).hexdigest() for n in ['response_matrix.npz','inputs.json','geometry.json']})
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    directions=np.load(bundle/'geometry.npz')['directions']
    profiles={s:dict(model=protocol['model'],best_mean=profile['best_fixed_mean'],sample_ids=[x for x in sample_ids if rows[x]['suite']==s],
        tied_best_configurations=[dict(configs[a],shift=(np.asarray(configs[a]['coefficients'])@directions).tolist()) for a in profile['best_fixed_actions']],
        per_item_rewards={a:matrices[s][:,aids.index(a)].tolist() for a in profile['best_fixed_actions']}) for s,profile in suites.items()}
    (out/'best_direction_profiles.json').write_text(json.dumps(profiles,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 各 benchmark 的固定最优方向','',
        '直接使用已完成的 175 × 37 响应矩阵。对每个 benchmark，所有题使用同一方向和强度，取全样本平均分最高的配置；保留并列最优。原模型也是候选。没有调用模型追加测量。',
        '', '| Benchmark | 固定最优方向 | 原模型 | 固定方向 | 逐题最好 | 固定方向保留的搜索增益 |',
        '|---|---|---:|---:|---:|---:|']
    for s,p in suites.items():
        names=' / '.join(label(configs[a]) for a in p['best_fixed_actions'])
        lines.append(f"| {s}（{p['samples']}题） | {names} | {100*p['baseline_mean']:.2f}% | {100*p['best_fixed_mean']:.2f}% | {100*p['per_question_best_mean']:.2f}% | {100*p['fraction_of_search_gain_retained']:.1f}% |")
    lines+=['', 'GSM8K/BBH 为最终答案正确率；IFEval 为官方 strict 全约束通过率；TruthfulQA 为 MC2 正确答案概率质量；WritingBench 为短篇清单评分。固定方向保留的搜索增益 = (固定最优 − 原模型) / (逐题最好 − 原模型)。',
        '', '## 固定最优方向的收益与回退','', '| Benchmark | 配置 | 改善题数 | 回退题数 | 增益（百分点） | 重抽样成为最优的比例（并列均分） |',
        '|---|---|---:|---:|---:|---:|']
    for s,p in suites.items():
        for e in p['ranked_directions']:
            if e['action'] in p['best_fixed_actions']:
                lines.append(f"| {s} | {label(e['configuration'])} | {e['improved_samples']} | {e['regressed_samples']} | {100*e['gain_from_baseline']:+.2f} | {100*e['bootstrap_best_share']:.1f}% |")
    lines+=['', '整体均分最高的方向仍可能使部分原本正确的题变错。逐题资产中的原模型候选保留规则与固定方向的整体使用是两种评价。',
        '', '## 方向换到别的 benchmark','', '| 方向 | GSM8K | BBH | IFEval | TruthfulQA | WritingBench |',
        '|---|---:|---:|---:|---:|---:|']
    for a,gains in cross.items():lines.append('| '+label(configs[a])+' | '+' | '.join(f'{100*gains[s]:+.2f}' for s in protocol['suites'])+' |')
    lines+=['', '上表为相对各自原模型的百分点变化，列之间采用各自指标。当前最优方向各不相同，且换任务后的效果可以反转：分解推理 +1 在 GSM8K 提升 13.89 个百分点，在 BBH 降低 10.26 个百分点。',
        '', '## BBH 内部的子问题','', '| 子任务 | 原模型 | 固定最优 | 逐题最好 | 并列最优方向 |',
        '|---|---:|---:|---:|---|']
    for task,p in bbh.items():
        lines.append(f"| {task}（{p['samples']}题） | {100*p['baseline_mean']:.2f}% | {100*p['best_fixed_mean']:.2f}% | {100*p['per_question_best_mean']:.2f}% | {' / '.join(label(configs[a]) for a in p['best_fixed_actions'])} |")
    lines+=['', '逻辑排序子任务中，固定方向最优与原模型持平，但逐题遍历可以覆盖所有题。这支持保存题目与子问题的方向集合。',
        '', '## 名次与适用范围','',
        '最优指这批已测样本、这个模型和当前执行协议下的观测均分最高。GSM8K 有两个并列第一；BBH 与 IFEval 的第一名只领先下一档一个样本。重抽样名次分布及全部 37 个候选的配对区间保存在 summary.json，未把选出最高分后的区间当作独立确认。',
        '', '方向名称来自概念探针，正负强度是原生向量系数；例如写作的当前最优为审慎验证 −0.5，说明实际任务用途应以响应证据定义。原实验已发现部分 BF16 输出对批次敏感，执行批次与复跑证据见上级实验报告。',
        '', '## 资产文件','', '- `summary.json`：完整排名、并列、改善/回退、名次重抽样及跨任务响应。',
        '- `best_direction_profiles.json`：各 benchmark 的最优配置，含模型版本、系数、1024 维向量和逐题分数。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:13]))
    print('SAVED',out)


if __name__=='__main__':main()

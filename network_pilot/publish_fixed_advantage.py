"""Publish independently generated router results beside historical atlas replay."""
import json
import shutil

import fixed_advantage_router as experiment
from gsm_router_large import save


def publish(rebuild=False):
    run=experiment.DEFAULT;dist=experiment.core.LAB/'dist';dest=dist/'assets/router-atlas'
    design=json.loads((run/'design.json').read_text());training={}
    for s in experiment.SUITES:
        d=json.loads((run/s/'historical_diagnostics.json').read_text())
        training[s]=dict(selected=d['config'],groups=d['groups'])
    complete=(run/'complete.json').exists();results=json.loads((run/'results.json').read_text()) if complete else {}
    conclusion='策略与阈值已冻结，新留出题生成进行中。'
    if complete:
        p=results['gsm8k']['paired']['fixed'];lo,hi=p['delta_95_pp']
        conclusion=f'主验证 GSM8K：相对最佳固定方向 {p["delta_pp"]:+.2f} 个百分点，配对95%区间 [{lo:+.2f}, {hi:+.2f}]；'
        conclusion+='本轮尚未证明稳定提升。' if lo<=0 else '本轮新留出结果支持提升，仍需独立复验。'
    payload=dict(complete=complete,fresh_counts=design['fresh_counts'],training=training,results=results,
        conclusion=conclusion,primary=design['primary'],labels_used_for_fitting='original training only',
        historical_role='answer-aware full-bank replay; exploratory',
        fresh_role='new questions, frozen policies, actual matched generation')
    save(dest/'fixed-advantage-experiment.json',payload)
    save(run/'published_experiment.json',payload)
    lines=['# 固定方向回退 Router 改进实验','',
       '训练：只使用原始 GSM8K 700、BBH 378、IFEval 350 道训练题的完整109配置矩阵。',
       '学习相对训练最佳固定方向的修复与损伤两个联合事件分数，训练内三折选择文本／隐藏状态核、损伤权重、候选配置数与切换阈值。',
       '每折独立拟合词表、参考方向、配置支持数和核参数。成功集合全部保留；分数不保证概率校准。','',
       '## 训练内验证与历史诊断','',
       '| 数据集 | 训练内 OOF 本轮 / 固定 | 历史测试 本轮 / 固定 |',
       '|---|---:|---:|']
    for s,v in training.items():
        t=v['selected'];h=v['groups']['historical']
        lines.append(f'| {s} | {t["correct"]}/{t["n"]} / {t["fixed_correct"]}/{t["n"]} | {h["improved"]}/{h["n"]} / {h["fixed"]}/{h["n"]} |')
    lines+=['','## 独立新留出生成验证','',conclusion,'',
       '| 数据集 | 新题数 | Baseline | 训练最佳固定 | 前版状态 Router | 本轮 Router |',
       '|---|---:|---:|---:|---:|---:|']
    for s,n in design['fresh_counts'].items():
        values=[str(results[s]['correct'][m]) if complete else '生成中' for m in experiment.METHODS]
        lines.append(f'| {s} | {n} | '+ ' | '.join(values)+' |')
    lines+=['','主检验预先指定为600道新 GSM8K 题相对训练最佳固定方向的差异。BBH剩余27题、IFEval剩余5题，仅作为小样本补充检查，不支持强泛化结论。',
       '新题按 ID 与题目文本排除此前记录；历史文件列表与哈希见 design.json。新题参考答案仅供生成后评分，模型和阈值在生成前冻结。',
       '历史测试集已多次查看，回放只用于诊断，未用于本轮超参数选择。训练内 OOF 同时用于选择参数，不是独立性能估计。','',
       '验证包括：输出重新评分、模型重载后路由一致、完整唯一记录、相同批次、token预算、相同方向时输出逐token一致。',
       '原始生成在各数据集 test/shard-*/records.jsonl，策略选择在 frozen_actions.json，训练折与参数选择见 selection.json。','']
    (run/'RESULTS.md').write_text('\n'.join(lines));shutil.copy2(run/'RESULTS.md',dest/'fixed-advantage-results.md')
    if rebuild:
        import build_router_atlas as atlas
        import router_direction_scan as scan
        atlas.build(scan.DEFAULT,dist,plots=False)
    print('PUBLISHED FIXED ADVANTAGE',conclusion,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--rebuild',action='store_true');a=p.parse_args();publish(a.rebuild)

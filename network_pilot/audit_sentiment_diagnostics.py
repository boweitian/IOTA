"""Recompute diagnostic conclusions from raw responses and source cache."""
import json
from pathlib import Path
import numpy as np
from sentiment_prepare import sha,write
from sentiment_navigator import ROOT,read,digest
from sentiment_diagnose import OUT,SEMANTIC


def main():
    out=OUT;s=json.loads((out/'summary.json').read_text());spec=json.loads((out/'spec.json').read_text())
    rows=read(ROOT/'runs/sentiment9b-sst5-direction-sweep-v1/feasible-space.jsonl');y=np.array([r['gold'] for r in rows])
    assert sha(ROOT/'runs/sentiment9b-sst5-direction-sweep-v1/feasible-space.jsonl')==spec['source_data_sha256']
    assert sha(ROOT/'sentiment_diagnose.py')==spec['source_sha256']
    for name,expected in s['files'].items():assert sha(out/name)==expected
    original=np.array([r['baseline_prediction'] for r in rows]);correctness={}
    for cond in spec['conditions']:
        name=cond['id'];r=json.loads((out/(name+'.json')).read_text());m=s['conditions'][name]
        assert r['spec_sha256']==digest(spec)
        if cond['kind']=='forward':
            p=np.array(r['probabilities']);logits=np.array(r['logits']);q=np.exp(logits-logits.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
            np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-7);pred=p.argmax(1)
            if cond.get('axis'):
                expected=abs(cond['dose'])*float(np.linalg.norm(np.load(ROOT/'runs/sentiment9b-sst5-100-v1/search-geometry.npz')['directions'][2 if cond['axis']=='precision' else 7]))
                np.testing.assert_allclose(r['measured_last_token_shift_norm'],expected,rtol=1e-5,atol=1e-5)
                assert abs(r['requested_shift_norm']-expected)<1e-5
        else:
            pred=np.array([r['prediction'] if r['prediction'] is not None else -1 for r in r['outputs']])
            for value in r['outputs']:
                text=value['text'].strip()
                expected=('ABCDE'.find(text) if len(text)==1 and text in 'ABCDE' else -1) if cond['kind']=='letters' else (SEMANTIC.index(text.lower().rstrip('.!').strip()) if text.lower().rstrip('.!').strip() in SEMANTIC else -1)
                assert expected==(value['prediction'] if value['prediction'] is not None else -1)
        target=np.sign(y-2)+1 if name=='three_class_forward' else y
        c=pred==target
        assert pred.tolist()==m['predictions'] and int(c.sum())==m['correct']
        assert int(c[:70].sum())==m['train_correct'] and int(c[70:].sum())==m['test_correct']
        if name!='three_class_forward':correctness[name]=c
    assert np.array_equal(s['conditions']['original_generation']['predictions'],original)
    assert s['source_audit']['passed']
    prompt_names=['original_forward','reversed_letters','semantic_generation','synthetic_calibration']
    prompt_union=np.any([correctness[name] for name in prompt_names],axis=0)
    all_union=np.any(list(correctness.values()),axis=0)
    extension=dict(prompt_variants_coverage=int(prompt_union.sum()),all_five_class_diagnostics_coverage=int(all_union.sum()),
                   prompt_variants=prompt_names,interpretation='Label-aware post-hoc coverage on the same 100 samples, not a deployable accuracy estimate.')
    write(out/'coverage-diagnostic.json',extension)
    report=['# SST-5 测量与干预诊断','',
            '同一批已看过结果的 100 个样本上的事后排查；原扫描结果不覆盖、不重新命名为新成绩。','',
            '## 格式与数据对齐','',
            '- 独立递归解析全部官方原始树，根标签、文本、split 和行号均一致。',
            f"- 原提示实际生成：{s['conditions']['original_generation']['correct']}/100，格式无效 {s['conditions']['original_generation']['invalid_format']} 题，与原下一 token 分类 100/100 一致。",
            f"- 原五标签的词表概率总和最低 {s['conditions']['original_forward']['label_mass_quantiles'][0]:.4%}。没有发现换行、解释文本或标签解析导致的假错误。",'',
            '## 情感粒度与标签边界','',
            f"- 原 {s['original_errors']} 个错误中，{s['adjacent_class_errors']} 个只错相邻一档，{s['larger_class_errors']} 个差两档以上。",
            f"- 原预测粗化成三类为 {s['original_collapsed_three_class_correct']}/100；非中性样本的正负极性正确 {s['original_non_neutral_polarity_correct']}/{s['non_neutral_count']}。粗化不是官方 SST-2 成绩。",
            '- 原五类答对数：'+', '.join(f"{SEMANTIC[i]} {row[i]}/{sum(row)}" for i,row in enumerate(s['conditions']['original_forward']['confusion']))+'。',
            '- 官方源中也有脱离上下文后反直觉的整句标签，例如 “Muddled, trashy and incompetent” 的根标签为中性；不能通过模型是否同意就擅自改标。','',
            '## 控制条件','',
            '| 条件 | 全部 | 原训练 70 | 原测试 30 | 与原预测一致 |',
            '|---|---|---|---|---|']
    for name,m in s['conditions'].items():report.append(f"| {name} | {m['correct']}/100 | {m['train_correct']}/70 | {m['test_correct']}/30 | {m.get('agreement_with_original','不同任务')} |")
    probe=s['conditions']['precision_+2_last'];original_probe=next(a for a in json.loads((ROOT/'runs/sentiment9b-sst5-direction-sweep-v1/sweep-spec.json').read_text())['actions'] if a['id']=='precision:+2')
    report+=['','## 注入是否实际生效','',
             f"规则精确 +2×：请求范数 {probe['shift_norm']:.4f}，实测位移范数中位数 {probe['observed_shift_norm_quantiles'][1]:.4f}，原残差范数中位数 {probe['residual_norm_quantiles'][1]:.4f}，比例 {probe['median_shift_to_residual_ratio']:.2%}。hook 确实执行；位移不是数值舍入级别。",
             f"原单方向缩放系数为 {original_probe['intervention']['cap_factor']:.8f}，并非被投影 regularization 大幅缩回。",
             '最后输入 token 的单次注入并没有对整段评论的所有位置施加干预，也没有多 token 推理。全输入位置对照使用同一向量和 ±2× 幅度，属于干预范围不同的实验。',
             '九个方向来自通用指令风格差异，未针对 SST-5 的中性判定与五档强度边界提取。方向有概念名称，不保证能解决当前分类错误。','',
             '## 有限扫描覆盖不等于模型能力上限','',
             f"仅改变提示形式的五分类条件合并后，标签可见覆盖为 {extension['prompt_variants_coverage']}/100；合并所有五分类诊断（含全输入位置干预）为 {extension['all_five_class_diagnostics_coverage']}/100。这些同样是事后覆盖上限。",
             '原 50% 覆盖只说明那 36 个单方向配置没有找到更多解；不能推断剩下 50 题模型无论如何都答不对。',
             '后续方法比较应先固定合适的任务粒度与 prompt，验证方向在该任务上的行为效应，再冻结注入范围、做新的训练/测试划分。现有诊断只能定位问题，不能当作新的泛化收益。']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n')
    audit=dict(passed=True,conditions=10,samples_per_condition=100,checks=['source tree labels/text independently matched','raw generation parser and format coverage',
        'next-token probabilities reconstructed from logits','condition predictions and split scores reconstructed','actual hook displacement matches requested vectors',
        'frozen scorer and original baseline reproduced'],coverage=extension,summary_sha256=sha(out/'summary.json'))
    write(out/'audit.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

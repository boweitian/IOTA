"""Reconstruct measured error diagnostics and publish all 35 annotated cases."""
import collections
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from sentiment_prepare import sha, write, jsonl
from sentiment_navigator import read, digest
from anli_diagnose import SOURCE, OUT, ROOT

NAMES=['蕴含','信息不足','矛盾']
CONTROL_NAMES={'original_recheck':'原协议复测','cyclic_mapping_1':'标签换序一','cyclic_mapping_2':'标签换序二','explicit_nli_guidance':'明确常识与关系判断要求','explanation_then_answer':'解释后作答（严格行格式）','explanation_explicit_final':'同一解释输出（接受行末Final）'}

def main():
    rows=read(SOURCE/'feasible-space.jsonl');spec=json.loads((OUT/'spec.json').read_text())
    assert sha(SOURCE/'feasible-space.jsonl')==spec['source_items_sha256']
    assert sha(SOURCE/'sweep-summary.json')==spec['source_summary_sha256']
    assert json.loads((OUT/'status.json').read_text())['phase']=='complete'
    annotations=json.loads((ROOT/'anli_error_reviews.json').read_text());reviews={r['index']:r for r in annotations['reviews']}
    y=np.array([r['gold'] for r in rows]);baseline=np.array([r['baseline_prediction'] for r in rows]);errors=baseline!=y
    assert set(reviews)==set(np.flatnonzero(errors)) and len(reviews)==35
    p=np.array([r['probabilities'] for r in rows]);pred=p.argmax(2)
    assert np.array_equal(pred[:,0],baseline)
    controls={};raw_controls={}
    for condition in spec['conditions']:
        name=condition['id'];r=json.loads((OUT/(name+'.json')).read_text());raw_controls[name]=r
        assert r['spec_sha256']==digest(spec) and r['ids']==[row['id'] for row in rows]
        actual=np.array([-1 if v is None else v for v in r['predictions']]);ok=actual==y
        if condition['kind']=='forward':
            assert [condition['order'][v] for v in np.array(r['probabilities']).argmax(1)]==r['predictions']
        metrics=dict(name=CONTROL_NAMES[name],correct=int(ok.sum()),train_correct=int(ok[:70].sum()),test_correct=int(ok[70:].sum()),
            fixed=int((ok&errors).sum()),damaged=int((~ok&~errors).sum()),changed=int((actual!=baseline).sum()),invalid=int((actual<0).sum()),
            prediction_counts=[int((actual==i).sum()) for i in range(3)],predictions=actual.tolist(),
            confusion=[[int(((y==i)&(actual==j)).sum()) for j in range(3)] for i in range(3)])
        if name=='original_recheck':
            error=float(np.max(abs(np.array(r['probabilities'])-p[:,0])));assert error<1e-3 and np.array_equal(actual,baseline)
            metrics['max_probability_error_vs_original']=error
        controls[name]=metrics
    # Keep the preregistered strict parser, and separately expose recoverable
    # inline answers. This is a post-hoc parser diagnostic on identical outputs.
    explanation=raw_controls['explanation_then_answer']['outputs']
    tolerant=[]
    for row in explanation:
        matches=re.findall(r'\bFinal:\s*([ABC])\b',row['text'])
        end=re.search(r'\bFinal:\s*([ABC])\s*$',row['text'])
        tolerant.append('ABC'.index(end.group(1)) if len(matches)==1 and end and not row['truncated'] else -1)
    v=np.array(tolerant);ok=v==y
    controls['explanation_explicit_final']=dict(name=CONTROL_NAMES['explanation_explicit_final'],correct=int(ok.sum()),train_correct=int(ok[:70].sum()),test_correct=int(ok[70:].sum()),
        fixed=int((ok&errors).sum()),damaged=int((~ok&~errors).sum()),changed=int((v!=baseline).sum()),invalid=int((v<0).sum()),
        prediction_counts=[int((v==i).sum()) for i in range(3)],predictions=v.tolist(),
        confusion=[[int(((y==i)&(v==j)).sum()) for j in range(3)] for i in range(3)],
        note='Same generated outputs, post-hoc parser accepts one unambiguous Final: X at output end even if it is inline.')
    manifest=json.loads((SOURCE/'dataset-manifest.json').read_text())
    assert sha(SOURCE/'source/anli_v1.0.zip')==manifest['source_sha256']
    archive=zipfile.ZipFile(SOURCE/'source/anli_v1.0.zip');official={}
    for name in archive.namelist():
        if name.endswith(('/R3/train.jsonl','/R3/test.jsonl')):
            for line in archive.read(name).decode().splitlines():
                value=json.loads(line);official[value['uid']]=value
    verifier={r['uid']:r['verifier labels'] for r in read(OUT/'official-verifier-labels.jsonl')}
    for row in rows:
        raw=official[row['uid']]
        assert raw['context']==row['context'] and raw['hypothesis']==row['hypothesis'] and 'enc'.index(raw['label'])==row['gold']
    cases=[]
    summary=json.loads((SOURCE/'sweep-summary.json').read_text());actions=summary['actions']
    generations={r['id']:r for split in ['train','test'] for r in read(SOURCE/(split+'-generation.jsonl'))}
    for i in sorted(reviews):
        row=rows[i];review=reviews[i];raw=official[row['uid']]
        actual=generations[row['id']]['text']
        assert actual.strip()=='ABC'[row['baseline_prediction']]
        user_prompt=manifest['prompt_template'].format(**{key:json.dumps(row[key],ensure_ascii=False) for key in manifest['prompt_fields']})
        cases.append(dict(index=i,id=row['id'],split=row['split'],context=row['context'],hypothesis=row['hypothesis'],gold=row['gold'],
            prediction_prompt=[dict(role='system',content=summary['model']['system_prompt']),dict(role='user',content=user_prompt)],raw_prediction=actual,
            baseline_prediction=row['baseline_prediction'],baseline_probabilities=row['probabilities'][0],
            category=review['primary_category'],category_name=annotations['categories'][review['primary_category']],analysis=review['analysis'],ambiguity=review.get('ambiguity'),
            official_reason=raw.get('reason',''),verifier_labels=verifier.get(row['uid'],[]),
            correct_actions=[actions[j]['id'] for j in row['feasible']],
            control_predictions={name:m['predictions'][i] for name,m in controls.items()},
            new_explanation=raw_controls['explanation_then_answer']['outputs'][i]['text']))
    stable=np.all(pred==baseline[:,None],axis=1)
    category_counts=collections.Counter(c['category'] for c in cases)
    result=dict(complete=True,source_items_sha256=spec['source_items_sha256'],spec_sha256=digest(spec),
        n=100,baseline_correct=int((~errors).sum()),errors=int(errors.sum()),wrong_neutral=int((errors&(baseline==1)).sum()),
        confusion=[[int(((y==i)&(baseline==j)).sum()) for j in range(3)] for i in range(3)],
        gold_counts=np.bincount(y,minlength=3).tolist(),prediction_counts=np.bincount(baseline,minlength=3).tolist(),
        high_confidence_errors=int((errors&(p[:,0].max(1)>=.8)).sum()),
        all_actions_stable_items=int(stable.sum()),changed_action_cells=int((pred[:,1:]!=baseline[:,None]).sum()),
        rescued_errors=int((errors&np.any(pred[:,1:]==y[:,None],axis=1)).sum()),
        never_correct_errors=int((errors&np.all(pred!=y[:,None],axis=1)).sum()),
        correct_items_damaged_by_any_action=int((~errors&np.any(pred!=y[:,None],axis=1)).sum()),
        max_probability_change=float(abs(p[:,1:]-p[:,:1]).max()),median_max_probability_change=float(np.median(abs(p[:,1:]-p[:,:1]).max((1,2)))),
        verified_error_cases=sum(bool(c['verifier_labels']) for c in cases),
        nonunanimous_error_cases=sum(len(set(c['verifier_labels']))>1 for c in cases),
        label_reason_review_cases=int(category_counts['label_review']),category_counts=dict(category_counts),categories=annotations['categories'],controls=controls,
        limitations=spec['scope'],annotation_method=annotations['method'])
    write(OUT/'summary.json',result);jsonl(OUT/'error-cases.jsonl',cases)
    report=['# ANLI R3 错题诊断','',
        '原始100题、官方标签、原始65% baseline与所有方向扫描结果均不改动。下面的对照是看过原始错误后的探索性分析，不是新测试集成绩。','',
        f"35道错题中26道被判为‘信息不足’；标签分布为34/33/33，模型预测分布为24/54/22。{result['high_confidence_errors']}道错题的最高类别概率≥80%，所以不全是临界选择。",'',
        '| 官方标签 \\ 模型预测 | 蕴含 | 信息不足 | 矛盾 |','|---|---:|---:|---:|']
    for name,cm in zip(NAMES,result['confusion']):report.append('| '+name+' | '+' | '.join(map(str,cm))+' |')
    report+=['','## 错误现象（逐题定性分类）','',annotations['method'],'','| 主要现象 | 题数 |','|---|---:|']
    for key,name in annotations['categories'].items():report.append(f'| {name} | {category_counts[key]} |')
    report+=['','## 实测对照','',
        '| 对照 | 全部100题 | train70 | test30 | 修正原错题 | 改坏原对题 | 无效格式 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in controls.items():report.append(f"| {m['name']} | {m['correct']} | {m['train_correct']} | {m['test_correct']} | {m['fixed']} | {m['damaged']} | {m['invalid']} |")
    report+=['',
        '换序条件仅重新对应A/B/C与三种语义标签，评分时已映射回原语义；因此变化说明当前答案对选项映射/呈现顺序敏感，不能仅归因于具体字母偏好。解释条件同时改变输出方式和生成长度，不能把收益单独归因于某一个因素。解释是重新生成的输出，不能用于还原原始单token决策的内部过程。解释的最后两行统计来自同一份输出：一行按预定独立成行格式解析，另一行事后接受唯一且位于末尾的Final标签，保留这个差异以暴露格式影响。',
        '',f"原49配置中，{result['all_actions_stable_items']}道题最终选择完全不变；4800个非baseline测量中仅{result['changed_action_cells']}格改变选项。35道原错题只有{result['rescued_errors']}道被任何已测方向救回，{result['never_correct_errors']}道在全部49配置下仍错；另有{result['correct_items_damaged_by_any_action']}道原对题被至少一个配置改错。类别概率确实变化：每题最大变化的中位数为{100*result['median_max_probability_change']:.2f}个百分点，不能把‘选择不变’说成hook没生效。",
        '',
        f"{result['label_reason_review_cases']}条训练错题的官方label与原reason存在语义张力，已逐条标记；这不等于确认这些题错标。35条错题中{result['verified_error_cases']}条有官方额外复核标签，其中{result['nonunanimous_error_cases']}条存在复核者分歧。这些label/reason张力题也均有多数复核者支持原标签，因此保留全部官方计分。其他涉及开放世界、常识前提或语用的争议另在每题说明。",
        '',
        '解释对照中，橡胶瓶盖一题的新输出明确承认金属探测器的物理性质，却仍以原文没有直接讨论反事实为由选择中性；这支持‘部分是判定标准/任务理解问题’的解释，不能简单归为不知道物理常识。人名含不含x的题则被解释成未知变量，而不是字符串检查。',
        '',
        '诊断判断：原实验未发现格式/标签错位；主要可观察问题是中性选项偏多、简单关系/数量判断失败，以及常识应用标准、标签语义边界与部分数据解释争议。标签换序即可改变结果；解释后作答的影响也大于已测方向的影响。先固定并验证作答协议、衡量选项映射稳定性，才适合把小幅提升归因于方向能力。现有九个通用风格方向，在当前单层最后输入token、直接分类协议下，只修正了少量答案，尚不足以证明能解决这些错误。',
        '',
        '后续若据此调整提示词、方向或路由，应在未用于这次诊断的新样本上验证，并把标签换序稳定性和原对题受损数一起报告。','',
        '官方来源：[ANLI](https://github.com/facebookresearch/anli)、[复核标签说明](https://github.com/facebookresearch/anli/blob/main/mds/verifier_labels.md)。reason是原标注者提供的理由，不是额外真值裁决。','',
        '## 全部35条错题','']
    for case in cases:
        i=case['index'];pred0=NAMES[case['baseline_prediction']];gold=NAMES[case['gold']]
        report += [f"### 原样本序号 {i+1} · {case['split']} · {case['id'].split(':')[-1][:8]}",'',
            f"官方：{gold}；baseline：{pred0}；主要现象：{case['category_name']}。",'',
            '前提：'+case['context'],'','假设：'+case['hypothesis'],'',case['analysis']]
        if case['ambiguity']:report+=['','需注意：'+case['ambiguity']]
        report+=['','能答对的已测方向：'+('、'.join(case['correct_actions']) or '无'),'','原始reason：'+(case['official_reason'] or '未提供'),'',
            '额外复核标签：'+(', '.join(case['verifier_labels']) or '无'),'','解释对照的新输出：','',case['new_explanation'],'']
    (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    from anli_sample_view import render
    render(result,cases)
    write(OUT/'audit.json',dict(passed=True,checks=['source hash unchanged','all 100 contexts/hypotheses/labels match official zip','all 35 and only baseline errors annotated','cyclic letter mapping independently inverted','all controls share frozen specification and item order','original forward reproduces original 100 decisions','metrics reconstructed from item predictions'],
        source_items_sha256=spec['source_items_sha256'],summary_sha256=sha(OUT/'summary.json'),cases_sha256=sha(OUT/'error-cases.jsonl'),reviews_sha256=sha(ROOT/'anli_error_reviews.json'),spec_sha256=digest(spec)))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

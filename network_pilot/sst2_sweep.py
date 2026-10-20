"""49 frozen geometry configurations × 100 official GLUE SST-2 samples."""
import json
import time
from pathlib import Path
import numpy as np
from sst2_prepare import OUT
from sentiment_prepare import ROOT,sha,write,jsonl
from sentiment_navigator import bank,read,digest
from sentiment_sweep_analysis import split_summary
from sst2_engine import SST2Engine,PROMPT

def finalize(out,spec):
    atlas=json.loads((out/'search-atlas.json').read_text());nodes=[dict(id=c['id'],name=c['name']) for c in atlas['concepts']]
    actions=spec['actions'];matrices={};items=[];generations=[]
    for split in ['train','test']:
        rows=read(out/('train.jsonl' if split=='train' else 'test-inputs.jsonl'))
        if split=='test':
            labels={r['id']:r['label'] for r in read(out/'test-labels.jsonl')};rows=[dict(r,label=labels[r['id']]) for r in rows]
        caches=[np.load(out/(split+'-cache')/f'agent-{j:02d}.npz') for j in range(49)]
        for cache in caches:
            if str(cache['spec_sha256'])!=digest(spec) or cache['ids'].tolist()!=[r['id'] for r in rows]:raise ValueError('Cache binding mismatch')
        p=np.stack([c['probabilities'] for c in caches],1);y=np.array([r['label'] for r in rows]);matrices[split]=(p,y)
        generations.extend(read(out/(split+'-generation.jsonl')))
        for i,row in enumerate(rows):
            pred=p[i].argmax(1);feasible=[j for j in range(1,49) if pred[j]==y[i]]
            items.append(dict(id=row['id'],text=row['text'],split=split,source_split=row['source_split'],gold=int(y[i]),baseline_prediction=int(pred[0]),
                predictions=pred.tolist(),probabilities=p[i].tolist(),feasible=feasible,feasible_single=[j for j in feasible if j<=36]))
    selected=json.loads((out/'frozen-selection.json').read_text())['selected_index']
    matrices['all']=(np.concatenate([matrices[s][0] for s in ['train','test']]),np.concatenate([matrices[s][1] for s in ['train','test']]))
    results={split:split_summary(p,y,actions,nodes,selected) for split,(p,y) in matrices.items()}
    jsonl(out/'feasible-space.jsonl',items)
    mass=np.concatenate([np.load(out/(split+'-cache')/'agent-00.npz')['label_mass'] for split in ['train','test']])
    validation=dict(generated=len(generations),invalid_format=sum(r['prediction'] is None for r in generations),
        agreement_with_scored=sum(r['prediction']==items[i]['baseline_prediction'] for i,r in enumerate(generations)),
        min_label_mass=float(mass.min()),median_label_mass=float(np.median(mass)))
    if validation['invalid_format'] or validation['agreement_with_scored']!=100:raise ValueError('Actual generation disagrees with classification')
    summary=dict(complete=True,benchmark='GLUE / SST-2',labels=['negative','positive'],display_labels=['负面','正面'],label_tokens=['A','B'],
        spec_sha256=digest(spec),geometry_sha256=spec['geometry_sha256'],model=json.loads((out/'model.json').read_text()),
        actions=actions,splits=results,train_selected_single=selected,measurement=spec['scoring'],format_validation=validation,
        items_sha256=sha(out/'feasible-space.jsonl'),files={str(p.relative_to(out)):sha(p) for split in ['train','test'] for p in (out/(split+'-cache')).glob('agent-*.npz')},
        interpretation='Finite-grid, label-aware feasibility. Best fixed on each displayed split is retrospective. Training-selected single was frozen before evaluation inputs were read. Evaluation uses official validation, not hidden official test.')
    write(out/'sweep-summary.json',summary)
    report=['# GLUE / SST-2：100 题全方向扫描','',
            '官方 GLUE 原始 SST-2.zip；70 个样本来自 train，30 个来自带标签的 validation/dev，各组类别平衡。官方 test 无公开标签，本报告不是官方完整测试成绩。',
            '排除前一轮已查看的 100 条 SST-5 文本，不改写文本或标签。官方训练集包含评论片段。','',
            'Qwen3.5-9B 与 Search 的九个原生向量保持一致。FP32、关闭 TF32，block 16 输出处最后一个输入 token 加向量；相同 prompt，读取 A=negative、B=positive 的下一 token 概率。',
            '九方向 × ±1×/±2×，加十二个预定义混合和 baseline，共 49 个配置、4,900 次评分。','',
            '| 范围 | Baseline | 事后最佳固定单方向 | 至少一个单方向正确 | 训练选择方向 |',
            '|---|---|---|---|---|']
    for split,s in results.items():
        def fmt(v):return f"{v['correct']}/{s['n']} ({v['accuracy']:.1%})"
        report.append(f"| {split} | {fmt(s['baseline'])} | {fmt(s['best_fixed_single'])} {s['best_fixed_single']['id']} | {fmt(s['coverage']['single_directions'])} | {fmt(s['train_selected_single'])} {s['train_selected_single']['id']} |")
    report+=['','| 范围 | 正向配置平均 | Baseline | 负向配置平均 |','|---|---|---|---|']
    for split,s in results.items():report.append(f"| {split} | {s['trend']['positive']['mean_accuracy']:.2%} | {s['baseline']['accuracy']:.2%} | {s['trend']['negative']['mean_accuracy']:.2%} |")
    report+=['',f"原提示真实生成：{validation['generated']} 题，格式无效 {validation['invalid_format']}，与分类评分一致 {validation['agreement_with_scored']}/100。A/B 词表概率总和最低 {validation['min_label_mass']:.4%}。",
             '事后最佳与至少一个正确使用真实标签，只表示当前配置网格上的观察结果与覆盖上限；不等于学会路由。趋势统计仅用单方向，正向指概念极性，不等于正面类别。30 题评估不能证明稳定泛化收益。']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n');print('\n'.join(report),flush=True)

def main():
    out=OUT;manifest=json.loads((out/'dataset-manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(out/name)!=expected:raise ValueError('Frozen input changed: '+name)
    atlas=json.loads((out/'search-atlas.json').read_text());protocol=json.loads((out/'search-protocol.json').read_text());directions=np.load(out/'search-geometry.npz')['directions']
    actions=bank(directions,float(atlas['scale']));serial=[{k:v for k,v in a.items() if k!='shift'} for a in actions]
    files=['sst2_prepare.py','sst2_sweep.py','sst2_engine.py','sentiment_navigator.py','sentiment_sweep_analysis.py']
    spec=dict(model=protocol['objective'],benchmark='GLUE / SST-2',geometry_sha256=sha(out/'search-geometry.npz'),dataset_sha256=sha(out/'dataset-manifest.json'),
        actions=serial,prompt=PROMPT,scoring='FP32; TF32 disabled; block 16 output; last input token only; next-token A/B softmax',
        batch_size=4,regularization='direction_projection',train_count=70,test_count=30,source_sha256={name:sha(ROOT/name) for name in files})
    if (out/'sweep-spec.json').exists() and json.loads((out/'sweep-spec.json').read_text())!=spec:raise ValueError('Changed frozen specification')
    write(out/'sweep-spec.json',spec)
    source=out/'source-code';source.mkdir(exist_ok=True)
    for name in files:
        target=source/name
        if not target.exists():target.write_bytes((ROOT/name).read_bytes())
    def status(phase,**extra):write(out/'status.json',dict(phase=phase,updated_at=time.time(),**extra))
    if (out/'sweep-summary.json').exists():return
    status('loading_model');engine=SST2Engine(protocol['objective']);write(out/'model.json',engine.meta)
    for split in ['train','test']:
        rows=read(out/('train.jsonl' if split=='train' else 'test-inputs.jsonl'));texts=[r['text'] for r in rows]
        cache=out/(split+'-cache');cache.mkdir(exist_ok=True)
        if split=='train' and not (out/'numerical-audit.json').exists():
            batch=engine.score(texts[:4],np.zeros(4096))['probabilities'];single=np.concatenate([engine.score([t],np.zeros(4096))['probabilities'] for t in texts[:4]])
            error=float(np.max(abs(batch-single)));passed=error<1e-3 and np.array_equal(batch.argmax(1),single.argmax(1))
            write(out/'numerical-audit.json',dict(passed=bool(passed),max_probability_error=error,predictions_agree=bool(np.array_equal(batch.argmax(1),single.argmax(1)))))
            if not passed:raise ValueError('Numerical audit failed')
        for j,action in enumerate(actions):
            target=cache/f'agent-{j:02d}.npz'
            if target.exists():
                with np.load(target) as row:
                    if str(row['spec_sha256'])!=digest(spec):raise ValueError('Stale cached measurement')
                continue
            status('all_direction_measurements',split=split,completed=j,total=49,agent=action['id'])
            parts=[engine.score(texts[lo:lo+4],action['shift']) for lo in range(0,len(texts),4)]
            temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,**{k:np.concatenate([r[k] for r in parts]) for k in ['probabilities','logits','label_mass','residual_norm','observed_shift_norm']},
                ids=np.array([r['id'] for r in rows]),spec_sha256=digest(spec));temp.replace(target)
            print(split,j+1,'/49',action['id'],flush=True)
        generated=out/(split+'-generation.jsonl')
        if not generated.exists():
            outputs=[]
            for lo in range(0,len(texts),4):outputs.extend(engine.baseline_generate(texts[lo:lo+4]))
            jsonl(generated,[dict(id=r['id'],**value) for r,value in zip(rows,outputs)])
        original=np.load(cache/'agent-00.npz')['probabilities'].argmax(1)
        if [r['prediction'] for r in read(generated)]!=original.tolist():raise ValueError('Baseline generation check failed')
        if split=='train':
            y=np.array([r['label'] for r in rows]);correct=[int(np.sum(np.load(cache/f'agent-{j:02d}.npz')['probabilities'].argmax(1)==y)) for j in range(49)]
            selected=max(range(1,37),key=lambda j:(correct[j],-j));frozen=dict(selected_index=selected,selected_id=actions[selected]['id'],train_correct=correct[selected],
                train_ids=[r['id'] for r in rows],spec_sha256=digest(spec),frozen_at=time.time())
            if not (out/'frozen-selection.json').exists():write(out/'frozen-selection.json',frozen)
    # Full response grid is saved before opening validation labels for analysis.
    prediction_rows=[]
    for split in ['train','test']:
        inputs=read(out/('train.jsonl' if split=='train' else 'test-inputs.jsonl'))
        p=np.stack([np.load(out/(split+'-cache')/f'agent-{j:02d}.npz')['probabilities'] for j in range(49)],1)
        prediction_rows.extend(dict(id=r['id'],split=split,probabilities=p[i].tolist()) for i,r in enumerate(inputs))
    jsonl(out/'predictions-before-evaluation-labels.jsonl',prediction_rows)
    finalize(out,spec);status('complete',measurements=4900,actions=49)

if __name__=='__main__':main()

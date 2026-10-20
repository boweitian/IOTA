"""Paired signed-dose comparisons and label-aware finite-bank feasibility."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from sentiment_prepare import write, jsonl, sha


def paired(a,b):
    """a and b are correctness per independent sample, never flattened repeats."""
    win=int(np.sum(a & ~b));loss=int(np.sum(~a & b));n=win+loss
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(win,loss)+1))/2**n) if n else 1.
    return dict(wins=win,losses=loss,ties=int(len(a)-n),accuracy_delta=float(np.mean(a)-np.mean(b)),exact_sign_p=p)


def split_summary(p,y,actions,names,selected_train):
    n=len(y);prediction=p.argmax(2);correct=prediction==y[:,None];base=correct[:,0]
    singles=[i for i,a in enumerate(actions) if sum(v!=0 for v in a['coefficients'])==1]
    steer=list(range(1,len(actions)));positive=[];negative=[];direction_rows=[];pairs=[]
    metrics=[]
    for j,a in enumerate(actions):
        own=correct[:,j]
        metrics.append(dict(index=j,id=a['id'],correct=int(own.sum()),n=n,accuracy=float(own.mean()),
            paired_vs_baseline=paired(own,base),mean_gold_probability=float(p[np.arange(n),j,y].mean()),
            mean_gold_log_probability=float(np.log(p[np.arange(n),j,y].clip(1e-30)).mean()),
            changed_prediction=int(np.sum(prediction[:,j]!=prediction[:,0]))))
    for k,node in enumerate(names):
        points=[]
        for dose in [-2,-1,0,1,2]:
            j=0 if dose==0 else next(i for i in singles if actions[i]['coefficients'][k]==dose)
            points.append(dict(dose=dose,**metrics[j]))
            if dose>0:positive.append(j)
            if dose<0:negative.append(j)
        direction_rows.append(dict(**node,points=points))
        for dose in [1,2]:
            pos=next(i for i in singles if actions[i]['coefficients'][k]==dose)
            neg=next(i for i in singles if actions[i]['coefficients'][k]==-dose)
            pairs.append(dict(axis=node['id'],name=node['name'],dose=dose,positive=pos,negative=neg,
                              comparison=paired(correct[:,pos],correct[:,neg])))
    best=max(singles,key=lambda j:(metrics[j]['correct'],-j))
    best_all=max(steer,key=lambda j:(metrics[j]['correct'],-j))
    coverage={}
    for key,indices in [('single_directions',singles),('all_configurations',steer)]:
        counts=correct[:,indices].sum(1);any_correct=counts>0;union=any_correct|base
        coverage[key]=dict(configurations=len(indices),correct=int(any_correct.sum()),n=n,accuracy=float(any_correct.mean()),
            with_baseline_correct=int(union.sum()),with_baseline_accuracy=float(union.mean()),
            rescued_baseline_errors=int(np.sum(any_correct & ~base)),baseline_correct_but_no_steer=int(np.sum(base & ~any_correct)),
            never_solved_even_with_baseline=int(np.sum(~union)),all_steers_correct=int(np.sum(counts==len(indices))),
            histogram=[dict(correct_configurations=k,samples=int(np.sum(counts==k))) for k in range(len(indices)+1)])
    trend={}
    for sign,indices in [('positive',positive),('negative',negative)]:
        c=correct[:,indices];per_sample=c.mean(1);delta=per_sample-base.astype(float)
        trend[sign]=dict(configurations=len(indices),mean_accuracy=float(per_sample.mean()),
            # Error bars use sample-level aggregates, preserving correlated direction trials.
            accuracy_delta_vs_baseline=float(delta.mean()),paired_delta_se=float(delta.std(ddof=1)/np.sqrt(n)),
            per_sample_correct_fraction=per_sample.tolist(),
            improved_configurations=int(sum(metrics[j]['correct']>int(base.sum()) for j in indices)),
            tied_configurations=int(sum(metrics[j]['correct']==int(base.sum()) for j in indices)),
            degraded_configurations=int(sum(metrics[j]['correct']<int(base.sum()) for j in indices)),
            fix_events=int(np.sum(c & ~base[:,None])),break_events=int(np.sum(~c & base[:,None])),
            mean_gold_probability=float(p[:,indices,:][np.arange(n)[:,None],np.arange(len(indices))[None,:],y[:,None]].mean()))
    diff=np.array(trend['positive']['per_sample_correct_fraction'])-np.array(trend['negative']['per_sample_correct_fraction'])
    trend['positive_minus_negative']=dict(mean_accuracy_delta=float(diff.mean()),sample_delta_se=float(diff.std(ddof=1)/np.sqrt(n)),
        samples_positive_better=int(np.sum(diff>0)),samples_negative_better=int(np.sum(diff<0)),samples_tied=int(np.sum(diff==0)),
        paired_configurations_positive_better=sum(x['comparison']['wins']>x['comparison']['losses'] for x in pairs),
        paired_configurations_negative_better=sum(x['comparison']['wins']<x['comparison']['losses'] for x in pairs),
        paired_configurations_tied=sum(x['comparison']['wins']==x['comparison']['losses'] for x in pairs))
    for value in trend.values():value.pop('per_sample_correct_fraction',None)
    return dict(n=n,baseline=metrics[0],best_fixed_single=metrics[best],best_fixed_any=metrics[best_all],
        train_selected_single=metrics[selected_train],action_metrics=metrics,directions=direction_rows,
        signed_pairs=pairs,trend=trend,coverage=coverage)


def analyze(out):
    out=Path(out);spec=json.loads((out/'sweep-spec.json').read_text());actions=spec['actions']
    atlas=json.loads((out/'search-atlas.json').read_text())
    nodes=[dict(id=c['id'],name=c['name']) for c in atlas['concepts']]
    sets={};items=[];matrices={};fingerprint=__import__('hashlib').sha256(json.dumps(spec,sort_keys=True,allow_nan=False).encode()).hexdigest()
    for split in ['train','test']:
        rows=[json.loads(s) for s in (out/('train.jsonl' if split=='train' else 'test-inputs.jsonl')).read_text().splitlines()]
        if split=='test':
            gold={r['id']:r['label'] for r in [json.loads(s) for s in (out/'test-labels.jsonl').read_text().splitlines()]}
            rows=[dict(r,label=gold[r['id']]) for r in rows]
        values=[]
        for j in range(len(actions)):
            with np.load(out/(split+'-cache')/f'agent-{j:02d}.npz') as cache:
                if str(cache['sweep_spec_sha256'])!=fingerprint:raise ValueError('Wrong sweep cache')
                if cache['ids'].tolist()!=[r['id'] for r in rows]:raise ValueError('Mismatched item order')
                value=cache['probabilities'].copy()
                if not np.isfinite(value).all() or (value<0).any() or not np.allclose(value.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
                values.append(value)
        p=np.stack(values,axis=1);y=np.array([r['label'] for r in rows]);matrices[split]=(p,y)
        for i,r in enumerate(rows):
            pred=p[i].argmax(1);feasible=[j for j in range(1,len(actions)) if pred[j]==y[i]]
            items.append(dict(id=r['id'],split=split,text=r['text'],gold=r['label'],baseline_prediction=int(pred[0]),
                              predictions=pred.tolist(),probabilities=p[i].tolist(),feasible=feasible,
                              feasible_single=[j for j in feasible if sum(v!=0 for v in actions[j]['coefficients'])==1]))
    single=[j for j,a in enumerate(actions) if sum(v!=0 for v in a['coefficients'])==1]
    p,y=matrices['train'];selected=max(single,key=lambda j:(int(np.sum(p[:,j].argmax(1)==y)),-j))
    matrices['all']=(np.concatenate([matrices[s][0] for s in ['train','test']]),np.concatenate([matrices[s][1] for s in ['train','test']]))
    for split,(p,y) in matrices.items():sets[split]=split_summary(p,y,actions,nodes,selected)
    result=dict(complete=True,spec_sha256=fingerprint,model=spec['model'],actions=actions,
                geometry_sha256=spec['geometry_sha256'],measurement=spec['scoring'],
                interpretation='A finite grid of 36 signed single-direction interventions and 12 mixtures; feasibility uses gold labels. Retrospective best is descriptive, not held-out selection.',
                splits=sets,train_selected_single=selected,
                files={str(path.relative_to(out)):sha(path) for split in ['train','test'] for path in sorted((out/(split+'-cache')).glob('agent-*.npz'))})
    jsonl(out/'feasible-space.jsonl',items);result['items_sha256']=sha(out/'feasible-space.jsonl')
    write(out/'sweep-summary.json',result)
    report=['# SST-5：逐样本全方向扫描','',spec['purpose'],'',
            '同一 Qwen3.5-9B、同一 geometry、相同 prompt；block 16 输出的最后一个输入 token 加方向，直接读取 A–E 概率。',
            '36 个单方向配置（九方向 × −2/−1/+1/+2），另加 12 个冻结的混合配置和 baseline。可行解空间仅指这个有限网格。','',
            '| 样本 | Baseline | 事后最佳固定单方向 | 单方向至少一个正确 | 加 baseline 后覆盖 | 全48配置至少一个正确 |',
            '|---|---|---|---|---|---|']
    for split,s in sets.items():
        def score(v):return f"{v['correct']}/{s['n']} ({v['accuracy']:.1%})"
        c=s['coverage']['single_directions'];report.append(f"| {split} | {score(s['baseline'])} | {score(s['best_fixed_single'])} {s['best_fixed_single']['id']} | {score(c)} | {c['with_baseline_correct']}/{s['n']} | {score(s['coverage']['all_configurations'])} |")
    report+=['','最佳固定方向是对当前分组的回顾性最大值；至少一个正确是用真实标签判定的覆盖上限，不能当作可部署 router 的准确率。原来的训练选择实验保持原样。','',
             '| 样本 | 正向配置平均准确率 | 负向配置平均准确率 | Baseline | 正向比负向更好的配对配置 / 18 |',
             '|---|---|---|---|---|']
    for split,s in sets.items():
        t=s['trend'];report.append(f"| {split} | {t['positive']['mean_accuracy']:.2%} | {t['negative']['mean_accuracy']:.2%} | {s['baseline']['accuracy']:.2%} | {t['positive_minus_negative']['paired_configurations_positive_better']} |")
    report+=['','重复方向试验不视作独立样本；误差按每个样本的方向平均值计算。单一方向的“正”只指 Search 中定义的概念极性，不等于正面情感标签。',
             '完整逐方向、倍数、纠正/破坏、正负配对和可行配置数直方图见 sweep-summary.json；每个样本的所有预测和可行配置见 feasible-space.jsonl。']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);analyze(parser.parse_args().out)

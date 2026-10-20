"""Exhaustive per-question steering discovery; no direction predictor is evaluated."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from experiment import ROOT, LAB, Execution, prompt, read_records
from pilot_engine import PilotEngine
from engine import atomic_json
from concepts import CONCEPTS
from space import action, singles, serial_action
from repair_empty_choice import score_with_empty

OLD=ROOT/'runs/pilot-2124657'
GEOMETRY=LAB/'runs/job-2117172'
DEFAULT=ROOT/'runs/sample-assets-v1'


def geometry():
    geo=dict(np.load(GEOMETRY/'geometry.npz'))
    geo['scale']=json.loads((GEOMETRY/'atlas.json').read_text())['scale']
    return geo


def exhaustive_bank(geo):
    bank=singles(geo['directions'],geo['scale'])
    for i in range(len(geo['directions'])):
        for sign in [-1,1]:
            w=np.zeros(len(geo['directions']));w[i]=sign
            bank.append(action(w,geo['directions'],geo['scale'],f'node:{i}:{sign:+d}:dose1'))
    for a in bank:
        nz=np.flatnonzero(a['coefficients'])
        if len(nz):
            i=int(nz[0]);a.update(concept=CONCEPTS[i]['id'],concept_name=CONCEPTS[i]['name'],
                requested_strength=(1. if a['id'].endswith(':dose1') else .5)*np.sign(a['coefficients'][i]))
        else:a.update(concept='baseline',concept_name='不干预',requested_strength=0.)
    return bank


def frozen_manifest(protocol,bank):
    names=['sample_assets.py','experiment.py','pilot_engine.py','space.py','repair_empty_choice.py']
    return dict(schema_version=2,objective='Per-question exhaustive steering discovery and successful-direction dictionaries',
        answers_used_for='Evaluate every declared configuration on every sample, retain all successful directions',
        protocol=protocol,bank=[serial_action(a) for a in bank],
        source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        geometry_sha256=hashlib.sha256((GEOMETRY/'geometry.npz').read_bytes()).hexdigest(),
        historical_records=str(OLD),
        interpretation='Best observed reward includes the saved no-intervention baseline; this is the sample asset discovery score.',
        success_definition={'gsm8k':'Exact final-answer match','bbh':'Exact final-answer match',
            'ifeval':'All official strict instruction checks pass',
            'truthfulqa':'Highest-probability MC2 candidate has a true label; MC2 correct-answer mass also retained',
            'writingbench':'Checklist score strictly exceeds the saved baseline; continuous quality score'})


def reuse(out,rows,bank):
    """Reuse exact named .5/no-op actions; preserve every other measured steer as extra evidence."""
    records=read_records(out/'records.jsonl');ids={r['id'] for r in rows};byid={a['id']:a for a in bank}
    catalog={a['id']:serial_action(a) for a in bank}
    suite=rows[0]['suite']
    native={a['id']:a for a in json.loads((OLD/suite/'all_actions.json').read_text())}
    sources=[(OLD/suite/'records.jsonl',native),
        (OLD/'validation_guard'/suite/'records.jsonl',{'none':native['none']})]
    transfer=OLD/'transfer'/suite
    if (transfer/'plan.json').exists():
        configs=json.loads((transfer/'plan.json').read_text())['source_configs']
        sources.append((transfer/'records.jsonl',{f'transfer:{s}':dict(id=f'transfer:{s}',coefficients=w,extra='') for s,w in configs.items()}))
    with (out/'records.jsonl').open('a') as f:
        for path,actions in sources:
            if not path.exists():continue
            for number,line in enumerate(path.read_text().splitlines(),1):
                record=json.loads(line);oldname=record['action']
                if record['id'] not in ids or oldname not in actions:continue
                a=actions[oldname]
                if a.get('extra',''):continue  # Prompt changes are a different intervention.
                if oldname in byid and np.array_equal(a['coefficients'],byid[oldname]['coefficients']):name=oldname
                else:name='historical:'+oldname
                if (record['id'],name) in records:continue
                catalog[name]=dict(a,id=name)
                copied=dict(record,action=name,reused=True,source_file=str(path),source_line=number,source_action=oldname)
                f.write(json.dumps(copied,ensure_ascii=False,allow_nan=False)+'\n')
                records[record['id'],name]=copied
    # Persist supplementary configurations on every resume as well.
    for path,actions in sources:
        for oldname,a in actions.items():
            if a.get('extra',''):continue
            name=oldname if oldname in byid and np.array_equal(a['coefficients'],byid[oldname]['coefficients']) else 'historical:'+oldname
            catalog[name]=dict(a,id=name)
    atomic_json(out/'actions.json',list(catalog.values()))
    return records


def is_success(row,record,baseline):
    if row['suite']=='truthfulqa':
        return bool(row['targets']['labels'][int(np.argmax(record['choice_probabilities']))])
    if row['suite']=='writingbench':return record['reward']>baseline['reward']+1e-12
    return bool(record['reward']==1)


def build_item(row,records,required):
    measured={name:r for (ident,name),r in records.items() if ident==row['id']}
    baseline=measured.get('none')
    if baseline is None:return dict(id=row['id'],suite=row['suite'],status='baseline_pending',missing_actions=required)
    best=max(measured.values(),key=lambda r:(r['reward'],r['action']=='none'))
    assert best['reward']>=baseline['reward']
    successful=[r['action'] for r in measured.values() if is_success(row,r,baseline)]
    improving=[r['action'] for r in measured.values() if r['reward']>baseline['reward']+1e-12]
    missing=[a for a in required if a not in measured]
    binary=row['suite']!='writingbench'
    base_correct=is_success(row,baseline,baseline) if binary else None
    solved=bool(successful) if binary else None
    return dict(id=row['id'],suite=row['suite'],task=row['task'],source_split=row['split'],question=row['question'],
        prompt=prompt(row),status='complete' if not missing else 'partial',missing_actions=missing,
        measured_count=len(measured),baseline_reward=baseline['reward'],best_reward=best['reward'],
        best_action=best['action'],successful_directions=successful,improving_directions=improving,
        baseline_correct=base_correct,solved=solved,rescued=bool(solved and not base_correct) if binary else None,
        failed_directions=[a for a in measured if a not in successful],
        evidence={name:{k:v for k,v in r.items() if k!='token_ids'} for name,r in measured.items()})


def summarize(items):
    suites={}
    for suite in sorted({r['suite'] for r in items}):
        rows=[r for r in items if r['suite']==suite];ready=[r for r in rows if 'baseline_reward' in r]
        binary=suite!='writingbench'
        suites[suite]=dict(samples=len(rows),complete=sum(r['status']=='complete' for r in rows),
            measured=sum(r.get('measured_count',0) for r in rows),
            baseline_mean=float(np.mean([r['baseline_reward'] for r in ready])) if ready else None,
            best_observed_mean=float(np.mean([r['best_reward'] for r in ready])) if ready else None,
            successful_direction_count=sum(len(r['successful_directions']) for r in ready),
            improved_samples=sum(bool(r['improving_directions']) for r in ready),
            baseline_solved=sum(r['baseline_correct'] for r in ready) if binary else None,
            searched_solved=sum(r['solved'] for r in ready) if binary else None,
            rescued=sum(r['rescued'] for r in ready) if binary else None,
            unresolved=[r['id'] for r in ready if binary and not r['solved']])
    return dict(suites=suites,all_complete=all(r['status']=='complete' for r in items),samples=len(items),
        objective='逐题遍历与成功方向字典；分数是每题已经找到的最好结果',
        monotonicity_verified=all(r.get('best_reward',0)>=r.get('baseline_reward',0) for r in items))


def report(run):
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text());bank=exhaustive_bank(geometry());required=[a['id'] for a in bank]
    records={};catalog={a['id']:serial_action(a) for a in bank}
    for out in sorted(run.glob('*/shard-*')):
        records.update(read_records(out/'records.jsonl'))
        if (out/'actions.json').exists():
            for a in json.loads((out/'actions.json').read_text()):catalog[a['id']]=a
    items=[build_item(row,records,required) for row in protocol['records']]
    bundle=run/'asset_bundle';bundle.mkdir(parents=True,exist_ok=True)
    with (bundle/'sample_direction_dictionary.jsonl').open('w') as f:
        for item in items:f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(bundle/'actions.json',list(catalog.values()))
    atomic_json(bundle/'inputs.json',protocol)
    shutil.copy2(GEOMETRY/'geometry.npz',bundle/'geometry.npz')
    atomic_json(bundle/'geometry.json',dict(model=protocol['model'],concepts=CONCEPTS,
        fitting='Contrastive activation mean differences on authored positive/negative high-level behavior probes',
        intervention='Add direction to last-token residual at layer 15 at prefill and each decode step',
        scale=geometry()['scale'],exhaustive_bank=[serial_action(a) for a in bank]))
    summary=summarize(items);atomic_json(run/'summary.json',summary)
    lines=['# IOTA 逐题方向资产实验','',
        '对当前冻结的 175 个样本，逐题遍历 9 个高层方向 × 正负符号 × 两档强度，外加原模型，共 37 个配置。旧实验中已测的混合方向也作为附加证据保留。',
        '', '每题保存全部成功方向、失败方向、系数、原始回答或选项概率和评分证据；原模型是候选，因此逐题最好结果不会低于原模型。',
        '', '| 数据集 | 样本 | 完成遍历 | 原模型均分 | 逐题最好均分 | 新解决题数 | 成功方向记录 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for suite,s in summary['suites'].items():
        fmt=lambda v:'—' if v is None else f'{100*v:.2f}%'
        lines.append(f"| {suite} | {s['samples']} | {s['complete']} | {fmt(s['baseline_mean'])} | {fmt(s['best_observed_mean'])} | {s['rescued'] if s['rescued'] is not None else '连续评分'} | {s['successful_direction_count']} |")
    lines+=['','TruthfulQA 均分为 MC2 正确答案概率质量；“新解决”指最高概率候选从错误变正确。WritingBench 为声明过的短篇改写，按清单连续评分，成功方向定义为超过该题原模型分数。',
        '', '本轮覆盖全部冻结样本，原来的 search / validation / test 字段仅保留来源信息。统计对象是逐题搜索资产，不是提前选择方向的调度器。',
        '', '## 字典例子','']
    for suite in ['gsm8k','bbh','ifeval','truthfulqa','writingbench']:
        example=next((r for r in items if r['suite']==suite and (r.get('rescued') or r.get('improving_directions'))),None)
        if example:
            lines.append(f"- `{example['id']}`：原模型 {example['baseline_reward']:.4f}，已找到最好 {example['best_reward']:.4f}；成功方向 `{', '.join(example['successful_directions'][:6])}`；完整证据见字典。")
    lines+=['','## 文件','', '- `asset_bundle/sample_direction_dictionary.jsonl`：逐题完整字典。',
        '- `asset_bundle/actions.json`：方向系数与来源。','- `asset_bundle/geometry.npz` / `geometry.json`：可复用的几何方向与干预规则。',
        '- 各数据集 `shard-*/records.jsonl`：逐项原始测量，包含 token IDs 与复用来源。']
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--suite')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    p.add_argument('--prepare',action='store_true');p.add_argument('--report',action='store_true')
    args=p.parse_args();run=args.run.resolve()
    if args.report:report(run);return
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text());geo=geometry();bank=exhaustive_bank(geo)
    suites=[args.suite] if args.suite else protocol['suites']
    if not args.prepare and not args.suite:raise ValueError('Specify suite for GPU execution')
    for suite in suites:
        shards=range(args.shards) if args.prepare else [args.shard]
        for shard in shards:
            rows=[r for r in protocol['records'] if r['suite']==suite][shard::args.shards]
            out=run/suite/f'shard-{shard}';out.mkdir(parents=True,exist_ok=True)
            manifest=frozen_manifest(protocol,bank);manifest.update(suite=suite,shard=shard,shards=args.shards,ids=[r['id'] for r in rows])
            frozen=out/'manifest.json'
            if frozen.exists() and json.loads(frozen.read_text())!=manifest:raise ValueError('Changed protocol or source; use a new run')
            atomic_json(frozen,manifest);saved=reuse(out,rows,bank)
            missing=sum((r['id'],a['id']) not in saved for r in rows for a in bank)
            print('PREPARED',suite,shard,'samples',len(rows),'missing',missing,flush=True)
            if args.prepare:continue
            PilotEngine.answer_likelihoods=score_with_empty
            engine=PilotEngine(protocol['model'])
            if suite=='truthfulqa':engine.precise_likelihood_mode()
            atomic_json(out/'model.json',engine.meta)
            execute=Execution(engine,out,protocol);execute.execute(rows,bank)
            assert all((r['id'],a['id']) in execute.records for r in rows for a in bank)
            atomic_json(out/'complete.json',dict(samples=len(rows),required=len(rows)*len(bank),records=len(execute.records)))
            print('EXHAUSTIVE COMPLETE',suite,shard,flush=True)


if __name__=='__main__':main()

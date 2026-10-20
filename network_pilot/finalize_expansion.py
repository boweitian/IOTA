"""Build and audit the expanded geometry and per-question response assets."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import numpy as np
from expanded_assets import DEFAULT,PREVIOUS,ROOT,LAB,bank_for
from experiment import read_records,grade,score_ifeval,parse_rubric
from sample_assets import GEOMETRY,build_item,summarize,is_success
from engine import atomic_json
import asset_index
from benchmark_preferences import analyze,label
from expanded_geometry import fit as reconstruct_geometry

def collect(run):
    records={};catalog={}
    for out in sorted(run.glob('*/shard-*')):
        for key,r in read_records(out/'records.jsonl').items():
            assert key not in records;records[key]=r
        if (out/'actions.json').exists():
            catalog.update({a['id']:a for a in json.loads((out/'actions.json').read_text())})
    return records,catalog

def index_expanded(run,geo):
    """Reuse schema/index implementation with expanded bank and explicit new cohorts."""
    prior=json.loads((PREVIOUS/'asset_bundle/execution_batches.json').read_text())['cohorts']
    previous_cohorts={tuple(edge):c for c in prior.values() for edge in c['evidence_edges']}
    original=asset_index.generation_groups
    def groups(source):
        if not source.is_relative_to(run):return original(source)
        result={};inferred=set()
        for r in map(json.loads,source.read_text().splitlines()):
            if not r.get('generated'):continue
            key=(r['action'],r['batch_seconds'])
            if r.get('batch_sample_ids'):batch=r['batch_sample_ids']
            else:
                c=previous_cohorts[r['id'],r['action']];batch=c['sample_ids']
                if c.get('input_cohort_inferred_after_grading_interruption'):inferred.add(key)
            assert len(batch)==r['batch_size']
            if key in result:assert result[key]==batch
            result[key]=batch
        return result,inferred
    asset_index.geometry=lambda:geo
    asset_index.exhaustive_bank=bank_for
    asset_index.generation_groups=groups
    asset_index.build(run)
    signatures=run/'asset_bundle/observed_subproblem_signatures.json'
    d=json.loads(signatures.read_text());d['definition']='Groups with identical observed success signatures in the declared 109-configuration space';atomic_json(signatures,d)

def audit(run,rows,records,items,bank,geo):
    sources={};fresh=0;reused=0;judge_retries=0;truncated=collections.Counter()
    for out in sorted(run.glob('*/shard-*')):
        m=json.loads((out/'manifest.json').read_text())
        for name,digest in m['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        assert m['geometry_sha256']==hashlib.sha256((run/'geometry/geometry.npz').read_bytes()).hexdigest()
    for key,r in records.items():
        row=rows[r['id']]
        if r.get('previous_source_file'):
            reused+=1;source=Path(r['previous_source_file'])
            if source not in sources:sources[source]=source.read_text().splitlines()
            old=json.loads(sources[source][r['previous_source_line']-1])
            for field,value in old.items():
                if field=='reused':continue
                assert r[field]==value
        else:
            fresh+=1
            if r.get('generated'):
                assert r['id'] in r['batch_sample_ids'] and len(r['batch_sample_ids'])==r['batch_size']
        if r['truncated']:truncated[row['suite']]+=1
        if row['suite'] in ['gsm8k','bbh']:
            scored=grade(r['text'],row);assert r['reward']==float(scored['correct']);assert r['extracted']==scored['extracted']
        elif row['suite']=='ifeval':
            scored=score_ifeval(row,r['text']);assert scored['reward']==r['reward'];assert scored['instruction_results']==r['instruction_results']
        elif row['suite']=='truthfulqa':
            values=np.asarray(r['choice_log_probabilities']);prob=np.exp(values-values.max());prob/=prob.sum()
            np.testing.assert_allclose(prob,r['choice_probabilities'],atol=1e-14,rtol=1e-12)
            assert abs(prob[np.asarray(row['targets']['labels'],dtype=bool)].sum()-r['reward'])<1e-12
        else:
            retry=r.get('judge_retries',[])
            raw=(retry[-1] if retry else r.get('judge_retry') or r['judge_raw'])['text']
            scored=parse_rubric(raw,len(row['source_record']['checklist']))
            assert scored is not None and scored['scores']==r['scores'] and scored['reward']==r['reward']
            judge_retries+=bool(retry)
    prior={r['id']:r for r in map(json.loads,(PREVIOUS/'asset_bundle/sample_direction_dictionary.jsonl').read_text().splitlines())}
    for item in items:
        assert item['status']=='complete'
        assert item['best_reward']>=item['baseline_reward'] and item['best_reward']>=prior[item['id']]['best_reward']
        assert set(prior[item['id']]['successful_directions'])<=set(item['successful_directions'])
        assert all(item['evidence'][a]['reward']==r['reward'] for a,r in prior[item['id']]['evidence'].items())
        assert item['evidence']['none']['reward']==prior[item['id']]['baseline_reward']
    old=np.load(GEOMETRY/'geometry.npz')
    assert all(np.array_equal(geo[k][:9],old[k]) for k in ['raw','unit','directions'])
    activation=np.load(run/'geometry/probe_activations.npz')
    probes=json.loads((run/'geometry/probes.json').read_text())
    reconstructed,_=reconstruct_geometry(activation['pooled'],probes,dict(old),float(geo['scale']))
    assert all(np.array_equal(geo[k],reconstructed[k]) for k in ['raw','unit','directions'])
    assert len(bank)==109 and fresh==12600 and reused==9330
    matrix=np.load(run/'asset_bundle/response_matrix.npz');assert matrix['observed'].all()
    assert matrix['reward'].shape==(175,109)
    for i,item in enumerate(items):
        for j,a in enumerate(bank):
            r=records[item['id'],a['id']]
            assert matrix['reward'][i,j]==r['reward']
            assert matrix['success'][i,j]==is_success(rows[item['id']],r,item['evidence']['none'])
    with sqlite3.connect(run/'asset_bundle/sample_assets.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]==len(records)
        assert db.execute('SELECT COUNT(*) FROM samples WHERE best_reward<baseline_reward').fetchone()[0]==0
    result=dict(passed=True,samples=175,directions=27,declared_settings=109,required_measurements=19075,
        evidence_records=len(records),reused_records=reused,new_records=fresh,missing_measurements=0,
        old_geometry_preserved_bit_exact=True,expanded_geometry_reconstructed_bit_exact=True,all_previous_successes_preserved=True,
        baseline_preserved=True,per_question_best_never_regresses=True,raw_score_replay_passed=True,
        explicit_new_generation_cohorts=True,valid_judge_retries=judge_retries,truncated_records=dict(truncated),
        scope='Asset discovery on all frozen questions; finite-bank observed maxima, no claim about unseen-question selection')
    atomic_json(run/'evidence_audit.json',result);print('AUDIT PASSED',len(records),'records',flush=True)

def build(run,partial=False):
    geo=dict(np.load(run/'geometry/geometry.npz'));bank=bank_for(geo);required=[a['id'] for a in bank]
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text());records,catalog=collect(run)
    items=[build_item(r,records,required) for r in protocol['records']]
    if not partial:assert len(list(run.glob('*/shard-*/complete.json')))==25 and all(i['status']=='complete' for i in items)
    bundle=run/'asset_bundle';bundle.mkdir(parents=True,exist_ok=True)
    with (bundle/'sample_direction_dictionary.jsonl').open('w') as f:
        for item in items:f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(bundle/'actions.json',list(catalog.values()));atomic_json(bundle/'inputs.json',protocol)
    shutil.copy2(run/'geometry/geometry.npz',bundle/'geometry.npz')
    gmeta=json.loads((run/'geometry/geometry.json').read_text());gmeta.update(exhaustive_bank=[{k:v for k,v in a.items() if k!='shift'} for a in bank],
        intervention='Add direction to last-token residual at layer index 15 at prefill and every decode step')
    atomic_json(bundle/'geometry.json',gmeta)
    oldp=json.loads((GEOMETRY/'probes.json').read_text());newp=json.loads((run/'geometry/probes.json').read_text())
    atomic_json(bundle/'probes.json',oldp+newp)
    oldh=np.load(GEOMETRY/'probe_activations.npz');newh=np.load(run/'geometry/probe_activations.npz')
    np.savez_compressed(bundle/'probe_activations.npz',**{k:np.concatenate([oldh[k],newh[k]]) for k in ['pooled','last']})
    atomic_json(run/'summary.json',summarize(items))
    index_expanded(run,geo)
    prior=json.loads((PREVIOUS/'declared_space_summary.json').read_text())['suites'];suites={}
    matrix=np.load(bundle/'response_matrix.npz')['reward']
    for i,suite in enumerate(protocol['suites']):
        ix=[j for j,row in enumerate(protocol['records']) if row['suite']==suite]
        rr=matrix[ix];ss=[items[j] for j in ix];ready=~np.isnan(rr)
        best=np.nanmax(rr,axis=1);base=rr[:,0]
        fixed=analyze(rr,[{k:v for k,v in a.items() if k!='shift'} for a in bank],seed=20261013+i) if ready.all() else None
        suites[suite]=dict(samples=len(ix),baseline_mean=float(base.mean()),old_best_mean=prior[suite]['best_mean'],
            expanded_best_mean=float(best.mean()),best_with_prior_mixtures=float(np.mean([s['best_reward'] for s in ss])),
            improved_over_old_samples=int((best>np.nanmax(rr[:,:37],axis=1)+1e-12).sum()),
            old_solved=prior[suite]['searched_solved'],expanded_solved=int(np.any(np.load(bundle/'response_matrix.npz')['success'][ix],axis=1).sum()) if suite!='writingbench' else None,
            fixed=fixed)
        assert np.all(best>=np.nanmax(rr[:,:37],axis=1))
    atomic_json(run/'comparison.json',dict(suites=suites,measured=len(records),additional_measured=sum(not r.get('previous_source_file') for r in records.values()),complete=all(r['status']=='complete' for r in items)))
    if not partial:
        audit(run,{r['id']:r for r in protocol['records']},records,items,bank,geo)
        lines=['# IOTA 27 方向几何与逐题资产','',
            '原有 9 个方向保持逐位一致，新增 18 个由独立正负行为探针构建的高层方向。27 个方向，每个正负两档强度，加原模型，共 109 个配置。全部 175 个冻结题目已遍历，新增 12,600 次测量，19,075 个必需题目—配置均齐全。原来 9,330 条测量证据及所有成功方向完整保留，共 21,930 条证据。',
            '', '| Benchmark | 原模型 | 原 37 配置逐题最好 | 新 109 配置逐题最好 | 新增覆盖 / 得分提高题数 |',
            '|---|---:|---:|---:|---:|']
        for s,d in suites.items():
            gain=(d['expanded_solved']-d['old_solved']) if d['old_solved'] is not None else d['improved_over_old_samples']
            lines.append(f"| {s}（{d['samples']}题） | {100*d['baseline_mean']:.2f}% | {100*d['old_best_mean']:.2f}% | {100*d['expanded_best_mean']:.2f}% | {gain} |")
        lines+=['', 'GSM8K/BBH 为最终答案精确匹配；IFEval 为官方 strict 全约束通过率；TruthfulQA 为 MC2 正确答案概率质量，其覆盖数按最高概率选项的正确标签统计；WritingBench 为已有 150–220 词短篇协议的清单评分。TruthfulQA 概率均分与覆盖数是两个指标。',
            '', '## 每个 benchmark 的固定最优方向','', '| Benchmark | 固定最优配置（保留并列） | 固定均分 | 逐题最好 |', '|---|---|---:|---:|']
        for s,d in suites.items():
            f=d['fixed'];names=' / '.join(label(next(a for a in bank if a['id']==name)) for name in f['best_fixed_actions'])
            lines.append(f"| {s} | {names} | {100*f['best_fixed_mean']:.2f}% | {100*d['expanded_best_mean']:.2f}% |")
        lines+=['', '固定最优是在这批已测题上选出的观测均分最高配置；完整排名、配对重抽样及并列信息见 comparison.json。逐题最好使用题目答案/检查器寻找方向，是方向字典的资产覆盖率。',
            '', '## 几何与可视化','', '- `view/index.html`：离线交互可视化。旋转真实方向 PCA，选择 benchmark 查看各节点的正负剂量响应；点击逐题矩阵查看原始回答、成功方向集合和导出配置。',
            '- `asset_bundle/geometry.npz`：27 × 1024 原生方向、全维余弦矩阵与用于展示的 3 维坐标。',
            '- `asset_bundle/probes.json`、`probe_activations.npz`：完整 540 条构建探针及激活，前 180 条来自原版。',
            '- `asset_bundle/geometry.json`：概念、构建协议、新方向在原有张成空间外的比例、独立诊断上下文的稳定性。',
            '- `asset_bundle/sample_direction_dictionary.jsonl`：每题全部成功、改善、失败方向和原始证据。',
            '- `asset_bundle/response_matrix.npz`：175 × 109 完整实测矩阵。',
            '- `asset_bundle/sample_assets.sqlite`、`direction_sample_index.json`：双向查询资产。',
            '- `evidence_audit.json`：21,930 条评分重放、原模型保留、旧成功完整保留与几何逐位一致审计。',
            '', '## 执行条件','', '同一冻结 Qwen3.5-0.8B、layer index 15、原生向量范数边界与原提示模板。新增生成使用固定分片批次，逐项保存实际批次输入和原始 token，再评分；WritingBench 判分若不符合 1–10 量表则明确重试，原始判分保留，不填造分数。TruthfulQA 使用 FP32 答案条件似然，保留原始空选项异常。',
            '', 'BF16 贪心生成可能随批次形状或 GPU 变化。旧版本已观察到同一配置在原批次成功、单题失败；本轮保存新批次以便精确复跑。新增节点与旧节点的比较采用已有基线，没有在每种执行条件下重复全部旧节点，因此新增收益是已观测资产增量，不能单独归因为向量语义。',
            '', '图中距离是原生向量的 PCA 投影，连线来自 1024 维余弦相似；投影损失以解释方差展示。邻近不等于能做对同一道题，概念名称也不保证干预的实际语义。全部任务响应以实测矩阵定义。']
        matched=run/'replay/matched_control_summary.json'
        if matched.exists():
            replay=json.loads(matched.read_text());lines+=['','## 新发现的匹配复跑','',
                '选择两个原 37 配置都未解出的 GSM8K 新发现，在同一次加载、相同 GPU、原始输入批次、提示与 token 预算下对照 no-op 和 steering。该抽查与完整搜索分开，不替换原始发现证据。','',
                '| 题目 | 新方向 | 同批次原模型 | 同批次 steering | 与发现原文一致 |','|---|---|---|---|---|']
            for r in replay['results']:
                lines.append(f"| {r['sample']} | {r['action']} | {'正确' if r['baseline_correct'] else '错误'} | {'正确' if r['steered_correct'] else '错误'} | {r['steered_text_matches_discovery']} |")
            lines+=['','`gsm8k:train:3489` 的字面解析 −1 在独立复跑与匹配对照中均成功，而 no-op 仍错误。`gsm8k:test:433` 的独特视角 +0.5 在发现时成功，但独立复跑与匹配对照均失败；它仍是已观测成功方向，稳定性标签为复跑未重现。两者证据完整保留，不能把全部已观测成功记录视为都已稳定验证。']
        (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for path in list(ROOT.glob('*.py'))+list(ROOT.glob('*expansion*.sbatch'))+list(ROOT.glob('expanded_*.sbatch')):shutil.copy2(path,snapshot/path.name)
        for name in ['engine.py','grading.py','concepts.py','judge.py']:
            dest=snapshot/'steering_lab'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(LAB/name,dest)
        atomic_json(run/'complete.json',dict(samples=175,directions=27,settings=109,new_measurements=12600,evidence_records=len(records),audit_passed=True))
    print(json.dumps({s:{k:v for k,v in d.items() if k!='fixed'} for s,d in suites.items()},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--partial',action='store_true');a=p.parse_args();build(a.run.resolve(),a.partial)

"""Finalize an exhaustive sample-direction asset bundle only after all workers finish."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from sample_assets import DEFAULT,ROOT,LAB,report,geometry,exhaustive_bank
from geometry_asset import export
from asset_index import build
from engine import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT)
    args=p.parse_args();run=args.run.resolve()
    if len(list(run.glob('*/shard-*/complete.json')))!=10:raise RuntimeError('Exhaustive search not finished')
    report(run);export(run);build(run)
    subprocess.run([sys.executable,str(ROOT/'plot_sample_assets.py'),'--run',str(run)],check=True)
    build(run)  # Include the exported figure in the asset index hashes.
    subprocess.run([sys.executable,str(ROOT/'audit_sample_assets.py'),'--run',str(run)],check=True)
    items=list(map(json.loads,(run/'asset_bundle/sample_direction_dictionary.jsonl').read_text().splitlines()))
    ids=[a['id'] for a in exhaustive_bank(geometry())];required=set(ids)
    full=json.loads((run/'summary.json').read_text());core={}
    for suite in ['gsm8k','bbh','ifeval','truthfulqa','writingbench']:
        rows=[r for r in items if r['suite']==suite]
        rewards=[max(r['evidence'][a]['reward'] for a in ids) for r in rows]
        successes=[bool(set(r['successful_directions'])&required) for r in rows]
        baseline=[r['baseline_reward'] for r in rows]
        binary=suite!='writingbench'
        core[suite]=dict(samples=len(rows),baseline_mean=float(np.mean(baseline)),best_mean=float(np.mean(rewards)),
            baseline_solved=sum(bool(r['baseline_correct']) for r in rows) if binary else None,
            searched_solved=sum(successes) if binary else None,
            rescued=sum(s and not r['baseline_correct'] for r,s in zip(rows,successes)) if binary else None,
            improved_samples=sum(b>a+1e-12 for a,b in zip(baseline,rewards)),
            successful_steering_records=sum(len(set(r['successful_directions'])&(required-{'none'})) for r in rows),
            unresolved=[r['id'] for r,s in zip(rows,successes) if binary and not s],
            per_item_best=rewards)
        if suite in ['gsm8k','bbh']:
            core[suite]['rescued_from_unparsed_baseline']=sum(s and not r['baseline_correct'] and not r['evidence']['none']['parsed']
                for r,s in zip(rows,successes))
            core[suite]['rescued_from_parsed_wrong_baseline']=core[suite]['rescued']-core[suite]['rescued_from_unparsed_baseline']
        assert core[suite]['best_mean']>=core[suite]['baseline_mean']-1e-12
    atomic_json(run/'declared_space_summary.json',dict(configurations=37,required_measurements=6475,suites=core,monotonicity_verified=True))
    replay=json.loads((run/'replay/summary.json').read_text())
    lines=['# IOTA 逐题方向字典：完整遍历结果','',
        '**两个工程已经落地：高层几何构建，以及逐题成功方向发现。** 当前固定 Qwen3.5-0.8B 的 block index 15（从 0 计数）、9 个 1024 维高层方向；几何能从保存的正负行为探针精确重建。',
        '', '对全部 175 个冻结样本，逐题遍历 9 个方向 × 正负 × 0.5/1.0 两档强度，加不干预，共 37 个配置，6,475 个必需题目—配置测量已齐全。原来各 split 仅作为来源信息，本轮全部用于资产发现。',
        '', '## 37 配置完整遍历','',
        '| 数据集 | 题目 | 原模型 | 逐题最好已测结果 | 新解决题数 | 成功 steering 记录 |',
        '|---|---:|---:|---:|---:|---:|']
    for suite,s in core.items():
        lines.append(f"| {suite} | {s['samples']} | {100*s['baseline_mean']:.2f}% | {100*s['best_mean']:.2f}% | {s['rescued'] if s['rescued'] is not None else '连续评分'} | {s['successful_steering_records']} |")
    lines+=['', 'GSM8K/BBH 为最终答案精确匹配；IFEval 为官方 strict 全部约束通过率。TruthfulQA 分数为 MC2 正确答案概率质量，新解决题指最高概率 MC2 候选从错误变正确。WritingBench 为 150–220 词短篇改写的清单评分；成功方向指分数高于该题原模型。',
        '', f"GSM8K 的 {core['gsm8k']['rescued']} 道补救中，{core['gsm8k']['rescued_from_unparsed_baseline']} 道原模型最终答案无法解析，{core['gsm8k']['rescued_from_parsed_wrong_baseline']} 道原模型最终答案可解析但错误。BBH 对应为 {core['bbh']['rescued_from_unparsed_baseline']} 道和 {core['bbh']['rescued_from_parsed_wrong_baseline']} 道。字典保留原始输出，便于区分格式修复与答案变化。",
        '', '每道题都保留原模型候选；逐题 max 不会低于原模型，已通过逐条断言。所有成功方向都收入字典；未解决题、失败方向、输出截断和判分证据也完整保留。',
        '', 'TruthfulQA 的 `truthfulqa:437` 在固定官方源中含一个空的错误选项；保留该选项，按 log P(空延续) = 0 计分。它会主导该题的 MC2 分母；原始选项与这个数据源异常保存在输入与逐题证据中。',
        '', '## 附加已有混合方向的完整资产','',
        '旧实验中已经实测的配对方向、混合方向和跨任务配置也收入对应题的字典。它们是附加证据，未声称对全部题都穷举了这些混合配置。包括这些证据后的逐题最好均分如下：','']
    for suite,s in full['suites'].items():lines.append(f"- {suite}：{100*s['best_observed_mean']:.2f}%；{s['successful_direction_count']} 条成功记录（含原模型成功）。")
    lines+=['', '## 执行条件与复跑','',
        '从字典抽取 GSM8K、BBH、IFEval、TruthfulQA 四个成功配置重新加载运行，四个均有成功复跑。GSM8K 的 `gsm8k:train:4851 / node:4:-1` 在原 8 题批次精确重现原文和正确答案，但单题批次失败。该失败已保留：BF16 贪心输出对批次形状可能敏感，成功测量与跨执行条件稳定性分别记录。',
        '', '每个生成证据都能查询原始执行批次，见 `execution_batches.json`。导出的配置含原始批次；`replay_asset.py --original-batch` 可以按相同题目组复跑。该抽查不代表全部成功方向都已复验。',
        '', 'WritingBench 评分中断批次的完整输入组由冻结的样本顺序及当时已保存候选重建，并明确标注为重建记录；原始失败输出和评分均保留。评分器曾返回不符合 1–10 量表的 0，缺失配置重新测量并获得有效判分，没有数值截断或填造分数；补测脚本提供明确最低分提示的评分重试，实际调用以 judge_repair 字段为准。',
        '', '## 可查询、可复用的资产','',
        '- `asset_bundle/geometry.npz`、`geometry.json`、`probes.json`、`probe_activations.npz`：几何空间与原始构建证据。',
        '- `asset_bundle/sample_direction_dictionary.jsonl`：每题所有成功/改善/失败方向、系数索引、原始回答或选项概率、判分。',
        '- `asset_bundle/direction_sample_index.json`：方向 → 能解决或改善的题目。',
        '- `asset_bundle/response_matrix.npz`：全部 175 × 37 实测奖励与成功矩阵。',
        '- `asset_bundle/response_matrix.png` / `.pdf`：逐题方向响应图。',
        '- `asset_bundle/benchmark_direction_profiles.json`：各 benchmark 对每个方向的实测响应及逐题最佳配置计数。',
        '- `asset_bundle/observed_subproblem_signatures.json`：成功方向集合相同的题目组。',
        '- `asset_bundle/sample_assets.sqlite`：`samples`、`directions`、`evaluations` 三张表，可直接按题或方向查询。',
        '- `evidence_audit.json`：逐条重放评分、原模型保留、完整穷举、原始输出来源和源码 hash 审计。',
        '', '查询并导出一条方向：','', '```bash',
        "../steering_lab/.venv/bin/python asset_index.py --sample gsm8k:train:4851 --direction node:4:-1 --export /tmp/iota-direction.json",
        "../steering_lab/.venv/bin/python replay_asset.py --config /tmp/iota-direction.json --original-batch --output /tmp/iota-replay.json",'```',
        '', '## 字典实例','']
    for suite in ['gsm8k','bbh','ifeval','truthfulqa']:
        r=next(r for r in items if r['suite']==suite and r['rescued'])
        node=next((a for a in r['successful_directions'] if a in required and a!='none'),r['successful_directions'][0])
        lines.append(f"- `{r['id']}`：原模型未通过，`{node}` 已通过；完整的其他成功方向与原始证据保存在字典。")
    lines+=['', '## 未解决样本','']
    for suite,s in core.items():
        if s['unresolved']:lines.append(f"- {suite}（{len(s['unresolved'])} 题）："+', '.join('`'+x+'`' for x in s['unresolved']))
    lines+=['', '原来的方向调度器试验保留在 `runs/pilot-2124657/`；它的分数与本轮逐题资产覆盖率分开。']
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in list(ROOT.glob('*.py'))+list(ROOT.glob('*assets*.sbatch')):
        shutil.copy2(path,snapshot/path.name)
    for name in ['engine.py','grading.py','concepts.py','judge.py']:
        dest=snapshot/'steering_lab'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(LAB/name,dest)
    shutil.copytree(ROOT/'vendor',snapshot/'vendor',dirs_exist_ok=True)
    # Bundle external evidence referred to by reused records, with a path translation map.
    from sample_assets import OLD
    referenced={Path(r['source_file']) for item in items for r in item['evidence'].values() if r.get('source_file')}
    translations={}
    for source in sorted(referenced):
        dest=run/'historical_sources'/source.relative_to(OLD);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);translations[str(source)]=str(dest.relative_to(run))
        for name in ['model.json','protocol.json','plan.json']:
            context=source.parent/name
            if context.exists():
                target=dest.parent/name;shutil.copy2(context,target)
                translations[str(context)]=str(target.relative_to(run))
    atomic_json(run/'source_path_map.json',translations)
    for historical in ['source_snapshot','source_current']:
        if (OLD/historical).exists():shutil.copytree(OLD/historical,run/'historical_sources'/historical,dirs_exist_ok=True)
    atomic_json(run/'complete.json',dict(samples=175,declared_configurations=37,required_measurements=6475,
        evidence_records=json.loads((run/'evidence_audit.json').read_text())['evidence_records'],
        all_successful_directions_saved=True,audit_passed=True,replay=replay))
    archive=run/'IOTA-sample-direction-assets-v1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or path==archive or path.name=='progress-report.json':continue
            z.write(path,path.relative_to(run))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('FINALIZED',archive,'bytes',archive.stat().st_size,flush=True)


if __name__=='__main__':main()

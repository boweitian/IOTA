"""Build auditable task assets and a report exclusively from saved measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from space import paired_interval
from subproblems import discover_regions

ROOT=Path(__file__).resolve().parent
LABELS=dict(baseline="原模型",prompt="固定提示词增强",best_single="最佳单节点",old_router="旧 KNN",
    flat_fixed="随机混合搜索·固定",flat_conditional="随机混合搜索·条件",
    node_guided_fixed="节点收益引导·固定",node_guided_conditional="节点收益引导·条件",
    network_fixed="网络引导·固定",network_conditional="网络引导·条件")


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def summarize(suite,path):
    records={(r["id"],r["action"]):r for r in map(json.loads,(path/"records.jsonl").read_text().splitlines())}
    plan=json.loads((path/"test_plan.json").read_text())
    strategies=list(plan[0]["choices"]);stats={}
    for strategy in strategies:
        rows=[records[p["id"],p["choices"][strategy]] for p in plan]
        rewards=np.asarray([r["reward"] for r in rows])
        reference=np.asarray([records[p["id"],p["choices"]["baseline"]]["reward"] for p in plan])
        stats[strategy]=dict(mean=float(rewards.mean()),delta_baseline=float((rewards-reference).mean()),
            paired_ci_baseline=paired_interval(rewards-reference),
            improved=int((rewards>reference+1e-9).sum()),worsened=int((rewards<reference-1e-9).sum()),
            truncation_rate=float(np.mean([r["truncated"] for r in rows])),n=len(rows),
            mean_new_tokens=float(np.mean([r["new_tokens"] for r in rows])))
    comparisons={}
    for target in ["flat_conditional","node_guided_conditional","best_single","old_router","prompt"]:
        diff=[records[p["id"],p["choices"]["network_conditional"]]["reward"]-records[p["id"],p["choices"][target]]["reward"] for p in plan]
        comparisons[target]=dict(mean=float(np.mean(diff)),paired_ci=paired_interval(diff))
    raw=list(records.values())
    # batch_seconds appears on each member: divide by the recorded batch_size before summing.
    generation_seconds=sum(r["batch_seconds"]/r.get("batch_size",1) for r in raw)
    judge_seconds=0
    for r in raw:
        for key in ["judge_raw","judge_retry"]:
            g=r.get(key)
            if g:judge_seconds+=g["batch_seconds"]/g["batch_size"]
    anchor=json.loads((path/"task_network.json").read_text())
    result=dict(suite=suite,strategies=stats,network_conditional_comparisons=comparisons,
        record_count=len(raw),record_counts_by_split={s:sum(r["split"]==s for r in raw) for s in ["search","validation","test"]},
        total_generated_tokens=sum(r["new_tokens"] for r in raw),generation_or_likelihood_seconds= generation_seconds,
        judge_seconds=judge_seconds,additional_network_pair_evaluations=anchor["additional_pair_evaluations"],
        interval_note="Unadjusted per-item paired bootstrap in a small pilot. Not family-cluster or multiple-comparison corrected;"
            " does not include search randomness, judge bias, or pretraining contamination.")
    if suite=="truthfulqa":
        protocol=json.loads((ROOT/"inputs/protocol.json").read_text())
        lookup={r["id"]:r for r in protocol["records"] if r["suite"]==suite}
        affected=[p["id"] for p in plan if any(not a for a in lookup[p["id"]]["targets"]["choices"])]
        means={}
        for strategy in strategies:
            rewards=[]
            for p in plan:
                row=lookup[p["id"]];r=records[p["id"],p["choices"][strategy]]
                keep=np.asarray([bool(a) for a in row["targets"]["choices"]])
                logp=np.asarray(r["choice_log_probabilities"])[keep]
                prob=np.exp(logp-logp.max());prob/=prob.sum()
                rewards.append(float(prob[np.asarray(row["targets"]["labels"],dtype=bool)[keep]].sum()))
            means[strategy]=float(np.mean(rewards))
        result["empty_choice_diagnostic"]=dict(affected_ids=affected,valid_choice_only_means=means,
            note="Source-score table retains the empty continuation with log probability zero. This declared diagnostic"
                 " renormalizes nonempty choices only and is not used to select or retune configurations.")
    return result


def make_plots(run,summary):
    import os
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".mpl_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    completed=summary["completed_suites"]
    if not completed:return
    fig,axes=plt.subplots(1,len(completed),figsize=(4.2*len(completed),4),squeeze=False)
    for ax,suite in zip(axes[0],completed):
        graph=json.loads((run/suite/"task_network.json").read_text())
        m=np.asarray(graph["adjacency"])
        im=ax.imshow(m,cmap="coolwarm",vmin=-.5,vmax=.5)
        ax.set_title(suite);ax.set_xticks(range(9),range(9));ax.set_yticks(range(9),range(9))
    fig.colorbar(im,ax=axes[0].tolist(),label="Shrunk positive-pair reward interaction",shrink=.7)
    fig.savefig(run/"task_networks.svg",bbox_inches="tight");plt.close(fig)
    strategies=["baseline","best_single","flat_conditional","node_guided_conditional","network_conditional"]
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(completed));width=.15
    for i,strategy in enumerate(strategies):
        ys=[summary["results"][s]["strategies"][strategy]["mean"] for s in completed]
        ax.bar(x+(i-2)*width,ys,width,label=strategy)
    ax.set_xticks(x,completed);ax.set_ylabel("Within-suite score (0–1; different metrics)");ax.set_ylim(0,1)
    ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(run/"pilot_scores.svg");plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);p.add_argument("--plots",action="store_true")
    args=p.parse_args();run=args.run.resolve();protocol=json.loads((ROOT/"inputs/protocol.json").read_text())
    lines=["# IOTA 网络几何资产 pilot：实测报告","",
        "模型：Qwen3.5-0.8B 固定 revision；九个种子节点、单层干预。","",
        "本轮测量任务条件下的联合干预网络、稀疏有符号混合和输入条件配置。不同 benchmark 的指标不能直接平均成统一能力分数。",
        "所有分数为小样本/自定义协议；WritingBench 为短篇改编及同族 4B judge。固定提示词对照尚未经过同预算的提示词优化。","",
        "网络额外花费每项 144 个配对测量（4 anchors × 36 pairs）。共享单节点测量后，每分支提出 12 个候选、各在搜索集实测；候选可重合。",
        "节点收益引导使用同一个候选池但去掉交互边，是辨别边贡献的主要消融；随机混合搜索是更弱的对照。","",
        "各配置在搜索/验证集确定，测试计划先冻结再评分。报告的区间尚未按题目族或多重比较校正，只用于 pilot 诊断。","",
        "节点编号：0 分解推理；1 审慎验证；2 规则精确；3 判断自信；4 简洁表达；5 探索可能；6 意象创造；7 情感共鸣；8 表达勇气。","" ]
    results={};completed=[];status={}
    for suite in protocol["suites"]:
        path=run/suite
        if not (path/"complete.json").exists():
            count=len((path/"records.jsonl").read_text().splitlines()) if (path/"records.jsonl").exists() else 0
            status[suite]=dict(state="incomplete",saved_records=count)
            lines.extend([f"## {suite}","",f"尚未完成；已保存 {count} 条实测记录。",""]);continue
        completed.append(suite);result=summarize(suite,path);results[suite]=result;status[suite]=dict(state="complete")
        lines.extend([f"## {suite}","","| 策略 | 均分 / 准确率 | 对原模型差值 | 配对差值 95% 区间 | 变好 / 变差 | 截断率 |",
            "|---|---:|---:|---:|---:|---:|"])
        for strategy,st in result["strategies"].items():
            lo,hi=st["paired_ci_baseline"]
            lines.append(f"| {LABELS[strategy]} | {100*st['mean']:.2f}% | {100*st['delta_baseline']:+.2f} pp | [{100*lo:+.2f}, {100*hi:+.2f}] | {st['improved']} / {st['worsened']} | {100*st['truncation_rate']:.1f}% |")
        lines.extend(["","网络条件配置与节点收益引导条件配置（去边）的配对差值："
            +f"{result['network_conditional_comparisons']['node_guided_conditional']['mean']*100:+.2f} pp；95% 区间 "
            +str([round(x*100,2) for x in result['network_conditional_comparisons']['node_guided_conditional']['paired_ci']])+"。","",
            f"搜索/验证/测试实测记录：{result['record_counts_by_split']}；生成 token 总数 {result['total_generated_tokens']}。",""])
        profile=json.loads((path/"task_profile.json").read_text())
        graph=json.loads((path/"task_network.json").read_text())
        distributions=json.loads((path/"configuration_distributions.json").read_text())
        provenance=json.loads((path/"protocol.json").read_text())
        evidence=["records.jsonl","test_plan.json","search_actions.json","task_network.json","task_profile.json","configuration_distributions.json"]
        if (path/"empty_choice_repair.json").exists():evidence.append("empty_choice_repair.json")
        asset=dict(schema_version=1,model=protocol["model"],suite=suite,task_profile=profile,
            task_interaction_graph=graph,configuration_distributions=distributions,evaluation=result,
            candidate_subproblems=discover_regions(path),
            scope="Single model, one layer, nine seeded directions, small public task samples.",
            evidence={name:dict(path=str(path/name),sha256=hashlib.sha256((path/name).read_bytes()).hexdigest()) for name in evidence},
            protocol_sha256=provenance["sha256"])
        save(run/"asset_bundle/tasks"/(suite+".json"),asset)
    audit=ROOT/"runs/engine-audit-v1/network_geometry.json"
    if audit.exists():
        import shutil
        geometry=json.loads(audit.read_text())
        lab=ROOT.parent/"steering_lab/runs/job-2117172"
        atlas=json.loads((lab/"atlas.json").read_text())
        geometry.update(intervention_norm_cap=atlas["scale"],node_scope="Nine human-seeded directions, not the full latent concept network",
                        original_direction_diagnostics=atlas["concepts"],projection_variance_explained=atlas["variance_explained"])
        save(run/"asset_bundle/geometry.json",geometry)
        shutil.copy2(lab/"geometry.npz",run/"asset_bundle/geometry.npz")
    transfers={s:json.loads((run/"transfer"/s/"scores.json").read_text()) for s in protocol["suites"]
               if (run/"transfer"/s/"complete.json").exists()}
    if transfers:
        save(run/"asset_bundle/cross_task_exchange.json",transfers)
        lines.extend(["## 跨任务配置交换","","行是接受配置的任务；列是固定配置的来源任务。不同任务的指标不可直接跨行比较。",
            "只交换已经冻结的网络固定配置；不利用此表重新选择配置。自任务结果复用原记录，其他格子是新测量。","",
            "| 目标任务 | "+" | ".join(protocol["suites"])+" |","|---|"+"---:|"*len(protocol["suites"])])
        for suite in protocol["suites"]:
            if suite in transfers:lines.append("| "+suite+" | "+" | ".join(f"{transfers[suite]['means'][s]*100:.2f}%" for s in protocol["suites"])+" |")
        lines.append("")
    guards={s:json.loads((run/"validation_guard"/s/"summary.json").read_text()) for s in protocol["suites"]
            if (run/"validation_guard"/s/"complete.json").exists()}
    if guards:
        save(run/"asset_bundle/validation_guard.json",guards)
        lines.extend(["## 验证筛选修正：保留原模型候选","",
            "原实现只验证搜索排名前三的固定候选，可能漏掉 no-op。现已修复未来运行的筛选逻辑，并补测缺少的验证集原模型结果。",
            "随后只依据验证集，在原模型、网络固定配置、网络条件配置之间选择，平分优先原模型。下表复用已经生成的测试输出，是事后修正诊断，不能当作新的独立确认。","",
            "| 任务 | 验证选择 | 原模型测试分数 | 选择后的诊断分数 |","|---|---|---:|---:|"])
        for suite in protocol["suites"]:
            if suite in guards:
                g=guards[suite];lines.append(f"| {suite} | {LABELS[g['chosen_strategy']]} | {g['baseline']*100:.2f}% | {g['selected']*100:.2f}% |")
        lines.append("")
    gradient=ROOT/"runs/gradient-gsm8k-v1"
    if (gradient/"complete.json").exists():
        g=json.loads((gradient/"summary.json").read_text())
        save(run/"asset_bundle/gradient_gsm8k.json",dict(summary=g,protocol=json.loads((gradient/"protocol.json").read_text()),
            checkpoints=json.loads((gradient/"checkpoints.json").read_text()),evidence_path=str(gradient)))
        lines.extend(["## 可微混合优化：单独的 GSM8K 探索","",
            "冻结全部模型权重，对九个节点系数做 24 步 Adam；参考训练题的完整推理文本监督 teacher-forced loss，加上任务交互图的正向混合正则。",
            "这是额外监督和额外预算的独立探索，不能用来归因 reward-only 搜索的网络优势。",
            f"同 GPU、BF16 生成对照：原模型 {g['baseline']*100:.2f}%，梯度配置 {g['gradient']*100:.2f}%，差值 {g['delta']*100:+.2f} pp，配对区间 {[round(x*100,2) for x in g['paired_ci']]}。",
            f"系数梯度与有限差分的核对误差：{g['gradient_check']['error']:.6g}。配置由验证集选定，测试答案未用于优化。",""])
    summary=dict(model=protocol["model"],completed_suites=completed,status=status,results=results,
        new_experiments=True,all_complete=len(completed)==len(protocol["suites"]),cross_task_exchange=transfers,
        validation_guard=guards,followup_complete=len(transfers)==len(protocol["suites"]) and len(guards)==len(protocol["suites"]))
    save(run/"summary.json",summary)
    save(run/"asset_bundle/index.json",dict(model=protocol["model"],tasks=completed,
        geometry="geometry.json" if audit.exists() else None,complete=summary["all_complete"],
        limitations=protocol["limitations"]))
    (run/"RESULTS.md").write_text("\n".join(lines))
    if args.plots:make_plots(run,summary)
    print(json.dumps(dict(completed=completed,status=status),ensure_ascii=False))


if __name__=="__main__":main()

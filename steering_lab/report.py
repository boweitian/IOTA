"""Recompute results from saved generations/judgments; produce an offline demo."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np
from engine import atomic_json

POLICIES=["baseline","fixed","geometry","lexical","prompt","random"]
NAMES=dict(baseline="原模型",fixed="固定方向",geometry="Geometry Router",lexical="无几何调度",prompt="提示词增强",random="随机方向")

def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def interval(values):
    values=np.asarray(values,dtype=float)
    if not len(values):
        return None
    rng=np.random.default_rng(20261012)
    samples=values[rng.integers(0,len(values),size=(4000,len(values)))].mean(1)
    return [float(v) for v in np.percentile(samples,[2.5,97.5])]

def score_domain(run,kind,rows):
    records={(r["id"],r["action"]):r for r in jsonl(run/f"{kind}_test.jsonl")}
    judges={(r["id"],r["action"]):r for r in jsonl(run/"art_test_judges.jsonl")} if kind=="creative" else {}
    plans=json.loads((run/f"{kind}_test_plan.json").read_text())
    by={r["id"]:r for r in rows}
    examples=[]
    for p in plans:
        row=by[p["id"]]
        results={}
        for policy,a in p["choices"].items():
            r=records[p["id"],a]
            j=judges.get((p["id"],a))
            value=j["rubric"]["score"] if j and j["parsed"] else None if kind=="creative" else float(r["correct"])
            results[policy]=dict(action=a,text=r["text"],score=value,new_tokens=r["new_tokens"],
                truncated=r["truncated"],parsed=j["parsed"] if j else r.get("parsed"),
                judge=j["rubric"] if j else None,extracted=r.get("extracted"),
                words=len(r["text"].split()),within_short_length=180<=len(r["text"].split())<=220)
        examples.append(dict(**row,results=results,trace=p["trace"],features=p["features"]))
    complete=[e for e in examples if all(e["results"][p]["score"] is not None for p in POLICIES)]
    stats=[]
    for p in POLICIES:
        values=np.array([e["results"][p]["score"] for e in complete])
        base=np.array([e["results"]["baseline"]["score"] for e in complete])
        stats.append(dict(policy=p,name=NAMES[p],n=len(complete),mean=float(values.mean()) if len(values) else None,
            ci=interval(values),delta=float((values-base).mean()) if len(values) else None,
            delta_ci=interval(values-base),improved=int((values>base).sum()),worsened=int((values<base).sum()),
            changed=int(sum(e["results"][p]["text"]!=e["results"]["baseline"]["text"] for e in examples)),
            mean_new_tokens=float(np.mean([e["results"][p]["new_tokens"] for e in examples])),
            truncation_rate=float(np.mean([e["results"][p]["truncated"] for e in examples])),
            parsing_rate=float(np.mean([bool(e["results"][p]["parsed"]) for e in examples])),
            intervention_rate=float(np.mean([e["results"][p]["action"]!="none" for e in examples])),
            length_compliance=float(np.mean([e["results"][p]["within_short_length"] for e in examples])) if kind=="creative" else None))
    task_stats=[]
    for task in sorted({e["task"] for e in examples}):
        subset=[e for e in complete if e["task"]==task]
        task_stats.append(dict(task=task,n=len(subset),scores={p:float(np.mean([e["results"][p]["score"] for e in subset])) for p in POLICIES}))
    return dict(kind=kind,metric="local blinded rubric mean / 10" if kind=="creative" else "numeric/boolean/option accuracy",
        total=len(examples),complete_case_count=len(complete),stats=stats,tasks=task_stats,examples=examples)

def make_evidence(run):
    protocol=json.loads((run/"protocol.json").read_text())
    creative=score_domain(run,"creative",[r for r in protocol["creative"]["records"] if r["split"]=="test"])
    objective=score_domain(run,"objective",[r for r in protocol["objective"]["records"] if r["split"]=="test"])
    audits=jsonl(run/"art_pair_audit.jsonl")
    # No-op selections are not evidence of judge agreement: exclude identical output pairs.
    by={e["id"]:e for e in creative["examples"]}
    changed=[a for a in audits if a["parsed"] and by[a["id"]]["results"]["baseline"]["text"]!=by[a["id"]]["results"]["geometry"]["text"]]
    creative["pair_audit"]=dict(records=audits,changed_parsed_count=len(changed),
        mean_preference=float(np.mean([a["preference"] for a in changed])) if changed else None,
        order_consistency=float(np.mean([a["order_consistent"] for a in changed])) if changed else None,
        note="Two order-swapped comparisons using the same local judge. Not independent human preference or an official EQ-Bench Elo.")
    atlas=json.loads((run/"atlas.json").read_text())
    geo=dict(np.load(run/"geometry.npz"))
    for domain,feature_file in [(creative,"creative_test_features.npz"),(objective,"objective_test_features.npz")]:
        fs=np.load(run/feature_file)
        positions=((fs["pooled"]-geo["center"])@geo["pca"].T/float(geo["radius"]))
        for e,xyz in zip(domain["examples"],positions):
            e["xyz"]=xyz.tolist()
    policies={k:json.loads((run/f"{k}_policy.json").read_text()) for k in ["creative","objective"]}
    action_bank=json.loads((run/"actions.json").read_text())
    # Dev examples expose all measured knob settings, clearly separate from held-out evaluation.
    dev=[]
    for kind,rows,path in [("creative",protocol["creative"]["records"],"art_dev.jsonl"),
                            ("objective",protocol["objective"]["records"],"dev.jsonl")]:
        records=jsonl(run/path)
        judgments={(j["id"],j["action"]):j for j in jsonl(run/"art_dev_judges.jsonl")} if kind=="creative" else {}
        for row in [r for r in rows if r["split"]=="dev"]:
            variants=[]
            for r in records:
                if r["id"]!=row["id"]:
                    continue
                j=judgments.get((r["id"],r["action"]))
                variants.append(dict(action=r["action"],text=r["text"],score=j["rubric"]["score"] if j and j["parsed"] else r.get("correct"),
                    judge=j["rubric"] if j else None,truncated=r["truncated"]))
            dev.append(dict(**row,kind=kind,variants=variants))
    generation_count=sum(len(jsonl(run/p)) for p in ["dev.jsonl","art_dev.jsonl","objective_test.jsonl","creative_test.jsonl"])
    budget_audit=None
    audit_path=run/"budget_audit"
    if (audit_path/"complete.json").exists():
        audit_domains={}
        for kind in ["creative","objective"]:
            assert json.loads((audit_path/f"{kind}_test_plan.json").read_text())==json.loads((run/f"{kind}_test_plan.json").read_text())
            rows=[r for r in protocol[kind]["records"] if r["split"]=="test"]
            domain=score_domain(audit_path,kind,rows)
            positions={e["id"]:e["xyz"] for e in (creative if kind=="creative" else objective)["examples"]}
            for e in domain["examples"]:
                e["xyz"]=positions[e["id"]]
            audit_domains[kind]=domain
        budget_audit=dict(**audit_domains,protocol=json.loads((audit_path/"audit_protocol.json").read_text()),
            note="Post-hoc audit on the same test items, 1024-token caps, unchanged frozen policies and routes. Not a new held-out estimate.")
    return dict(run=str(run),protocol_sha256=protocol["sha256"],atlas=atlas,creative=creative,objective=objective,budget_audit=budget_audit,
        policies=policies,actions=action_bank,dev_examples=dev,protocol=protocol,
        calls=dict(saved_generation_settings=generation_count,
            rubric_judgments=len(jsonl(run/"art_dev_judges.jsonl"))+len(jsonl(run/"art_test_judges.jsonl")),
            pair_judgments=2*len(audits)),
        notes=["All displayed model outputs and hidden vectors were measured; benchmark prompts and concept labels are curated.",
            "Policy calibration uses development examples only. Test routes are frozen before scoring.",
            "This is a single-model pilot with sampled benchmarks and one local same-family creative judge.",
            "A fixed direction bank is a candidate atlas; global invariance, all-concept coverage, and robust performance gains are not assumed."])

def markdown(data):
    out=["# IOTA Geometry Router：真实实验结果", "",f"运行：`{data['run']}`", "",
         "生成模型：Qwen3.5-0.8B，固定 revision；9 个高层概念、1024 维 residual 方向、两个强度和一个混合方向。生成时干预，不修改权重。",
         "", "创作是主展示：EQ-Bench Creative Writing v3 公开题目的短篇改编版，12 题开发、20 题留出。"
         "由本地 Qwen3.5-4B 匿名评分，非官方分数、Elo 或人类审美结论。", ""]
    for domain in [data["creative"],data["objective"]]:
        out += [f"## {'创意写作' if domain['kind']=='creative' else '能力保护与推理'}", "",f"计分：{domain['metric']}；完整配对样本 {domain['complete_case_count']}/{domain['total']}。", "",
                "| 策略 | 平均分 / 准确率 | 相对原模型 | 配对差值 95% bootstrap 区间 | 变好 / 变差 | 截断率 |", "|---|---:|---:|---:|---:|---:|"]
        for s in domain["stats"]:
            ci=s["delta_ci"]
            out.append(f"| {s['name']} | {s['mean']*100:.2f}% | {s['delta']*100:+.2f} pp | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | {s['improved']} / {s['worsened']} | {s['truncation_rate']*100:.1f}% |")
        out += ["", "区间按题目配对重采样，只刻画这个小样本；不包含 judge 偏差、模型种子、题目泄漏或跨分布误差。", "",
                "输出格式和完成长度也会影响得分；不能直接将这个指标的变化解释为知识、推理能力或全局人格的提升。", ""]
    audit=data["creative"]["pair_audit"]
    out += ["## 创作盲评复查", "",f"不同作品且两次均可解析的配对：{audit['changed_parsed_count']}。"]
    if audit["mean_preference"] is not None:
        out += [f"交换顺序后的平均 Router 偏好：{audit['mean_preference']:.3f}（0.5 为平分）；顺序一致率：{audit['order_consistency']:.1%}。"]
    out += ["", "同族 4B judge 是质量代理；对它的开发评分优化可能学习它的偏好。完整作品、评分理由与顺序分歧均保留，界面提供人工匿名比较。", "",
        "## 结果如何解释", "", "如果 Router 的区间跨过 0，或不优于提示词 / 无几何调度，就不能宣称 concept geometry 带来了鲁棒性能提升。"
        "即使小样本提升，也仍需扩大创作题目、使用独立 judge 与人工评分、测多个种子，并检查未见体裁和模型。", "",
        "## 实际预算", "", f"保存了 {data['calls']['saved_generation_settings']} 个生成设置、{data['calls']['rubric_judgments']} 条 rubric 评判和 {data['calls']['pair_judgments']} 次顺序交换评判。", "",
        "调度本身使用一次题目 activation 测量与轻量 KNN；推理时不访问答案，不试完所有方向后挑选正确结果。生成记录保留 token 数与 batch 时长；额外 activation 测量尚未单独计时，本次不报告端到端吞吐优势。", ""]
    if data["budget_audit"]:
        out += ["", "## 生成预算复查（事后诊断）", "", "初轮截断偏高，因此在同一批测试题上把生成上限扩大至 1024 tokens。方向、策略、动作选择、提示与判分保持固定，policy hash 已核对。"
                "这不是新留出测试，不用于重新选择策略。", "", "| 领域 | 策略 | 均分 / 准确率 | 相对原模型 | 差值 95% 区间 | 截断率 |", "|---|---|---:|---:|---:|---:|"]
        for kind in ["creative","objective"]:
            for s in data["budget_audit"][kind]["stats"]:
                out.append(f"| {'创作' if kind=='creative' else '能力'} | {s['name']} | {s['mean']*100:.2f}% | {s['delta']*100:+.2f} pp | [{s['delta_ci'][0]*100:+.2f}, {s['delta_ci'][1]*100:+.2f}] | {s['truncation_rate']*100:.1f}% |")
    return "\n".join(out)

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);args=p.parse_args()
    run=args.run.resolve()
    if not (run/"complete.json").exists():
        raise RuntimeError("The experiment is incomplete; refusing to present partial data as a final benchmark")
    data=make_evidence(run)
    atomic_json(run/"evidence.json",data)
    (run/"RESULTS.md").write_text(markdown(data))
    dist=Path(__file__).resolve().parent/"dist";dist.mkdir(exist_ok=True)
    atomic_json(dist/"evidence.json",data)
    source=Path(__file__).resolve().parent
    shutil.copyfile(source/"RESEARCH.md",dist/"RESEARCH.md")
    snapshot=run/"source_snapshot";snapshot.mkdir(exist_ok=True)
    hashes={}
    for name in ["engine.py","concepts.py","router.py","grading.py","judge.py","experiment.py","prepare.py","prepare_art.py"]:
        raw=(source/name).read_bytes()
        hashes[name]=hashlib.sha256(raw).hexdigest()
        (snapshot/name).write_bytes(raw)
    atomic_json(snapshot/"hashes.json",dict(snapshot_time="after experiment completion",files=hashes))
    template=(source/"page.html").read_text().replace("../RESEARCH.md","RESEARCH.md")
    embedded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    (dist/"benchmark.html").write_text(template.replace("__EVIDENCE__",embedded))
    if not (dist/"index.html").exists():
        (dist/"index.html").write_text(template.replace("__EVIDENCE__",embedded))
    print("BUILT",dist/"benchmark.html",flush=True)

if __name__=="__main__":
    main()

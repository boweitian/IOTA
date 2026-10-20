"""Standalone evidence report. Its layout does not represent semantic distance."""
import html
import json
from pathlib import Path


def esc(value):
    return html.escape(str(value),quote=True)


def write_report(evidence_map, confirmation, identity, path):
    candidate = evidence_map.best_rule()[0]
    metrics = confirmation["metrics"]
    observations = {o["evidence_id"]:o for o in evidence_map.observations}
    observations.update({o["evidence_id"]:o for o in confirmation["observations"]})
    rows = []
    for step in evidence_map.trajectory:
        obs = observations[step["evidence_id"]]
        rows.append(f'<tr><td>{step["version"]}</td><td><a href="#{obs["evidence_id"]}">{esc(obs["scenario_id"])}</a></td>'
                    f'<td>{esc(obs["selected_label"])}</td><td>{step["prediction_before"]:.3f}</td>'
                    f'<td>{step["expected_information_gain_bits"]:.3f}</td>'
                    f'<td>{esc(step["counterexample_to"] or "—")}</td><td>{esc(", ".join(step["proposed_rules"]) or "—")}</td>'
                    f'<td>{esc(step["best_after"])}</td></tr>')
    edges = []
    for r in identity["relations"]:
        examples = sorted(r["evidence"],key=lambda e:-abs(e["delta_p_disclose"]))[:4]
        links = "".join(f'<li>{esc(e["condition"])}: Δ={e["delta_p_disclose"]:.3f} '
                        f'<a href="#{e["low"]}">low</a> / <a href="#{e["high"]}">high</a></li>' for e in examples)
        edges.append(f'<article><h3>{esc(r["source"])} → disclosure</h3><p>{esc(r["status"])} · '
                     f'平均 ΔP={r["mean_delta_p_disclose"]:.3f} · 区间 {esc(r["family_bootstrap_interval_95"])} · '
                     f'{r["matched_pairs"]} matched pairs</p><ul>{links}</ul></article>')
    evidence = "".join(f'<details id="{key}"><summary>{esc(o["scenario_id"])} · {esc(o["selected_label"])} · P={o["p_disclose"]:.3f}</summary>'
                       f'<pre>{esc(json.dumps(o,indent=2,ensure_ascii=False))}</pre></details>' for key,o in observations.items())
    hypotheses = "".join(f'<li>{esc(r.description)} · {esc(r.source)} · weight={w:.4f}</li>'
                         for r,w in zip(evidence_map.rules,evidence_map.weights()))
    banner = "合成夹具：用于验证程序，不能作为 AI 模型测量" if evidence_map.model["kind"] == "synthetic" else "真实开放权重模型：受限二选一读出，未恢复内部几何"
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>IOTA · Evidence Atlas</title>
<style>body{{font:16px system-ui;background:#10151e;color:#dfe8f3;margin:35px auto;max-width:1250px;padding:20px;line-height:1.65}}a{{color:#7fcbff}}article,details{{border:1px solid #344252;border-radius:9px;padding:16px;margin:12px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #344252;text-align:left;padding:10px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}.banner{{background:#23334a;padding:18px}}.scroll{{overflow:auto}}</style>
<h1>IOTA · Evidence Atlas</h1><p class="banner">{esc(banner)}</p>
<p>{esc(evidence_map.model["name"])} · revision {esc(evidence_map.model["revision"])} · protocol {esc(identity["protocol"])} · map version {evidence_map.version}</p>
<h2>Map：条件结构</h2><p>当前候选：{esc(candidate.description)}。权重依赖有限假设空间与噪声先验，不是经校准的真实性概率。</p>
<details><summary>所有候选与来源</summary><ul>{hypotheses}</ul></details>
<h2>独立确认与 Human–Agent Alignment</h2><p>留出银行、住房、社区三个场景族，使用不同表述，保持共同事实结构。确认结果未反馈给 Walker。</p>
<p>地图预测模型读出的 accuracy={metrics["accuracy_to_model"]:.3f}，Brier={metrics["brier_to_model"]:.4f}；模型与项目授权规则的一致率={metrics["agreement_with_reference_policy"]:.3f}。这两类指标意义不同。</p>
<h2>Soul：范围内的行为描述</h2><p>以下是任务事实变化对读出的条件效应，不是内部机制或全局不变量。三场景族 bootstrap 区间仅作描述。完整条件对与证据见 JSON。</p>{''.join(edges)}
<p>身份描述状态：{esc(identity["status"])}。恒定标签输出无法验证语义决策边界，预测准确率可由常量规则达到。</p>
<h2>Walker：探索与修正</h2><p>每轮显示执行前预测、预期信息增益、反例与新候选；可点击查看原始测量。</p>
<div class="scroll"><table><tr><th>版本</th><th>实验</th><th>读出</th><th>之前 P</th><th>IG bits</th><th>反例对应</th><th>新候选</th><th>更新后候选</th></tr>{''.join(rows)}</table></div>
<h2>原始证据</h2><p>当前 HF 后端以单 token A/B 在两套逆映射下测量并平均语义 log odds；历史 v1 的不等长词标签协议已失效。selected_label 是受限读出，不是自由生成的回答。标签分词、完整 prompt、模型 revision 均记录。</p>{evidence}
<p>未知单子发现、内部因果几何、模型身份唯一性、真实版本漂移及 benchmark 优越性尚未验证。</p></html>'''
    Path(path).write_text(page)

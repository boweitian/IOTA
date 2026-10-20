"""Publish the paired held-out result and the original four-field sample viewer."""
import html
import json
import shutil
from pathlib import Path

from heldout_precision import OUT
from sentiment_navigator import read
from sentiment_prepare import sha


def pct(value):
    return f'{100*value:.2f}%'


def themed(page):
    # Shared palette must follow local layout CSS to preserve the user's theme choice.
    theme = '<link rel="stylesheet" href="/theme-blue.css">'
    return page.replace(theme, '').replace('</head>', theme+'</head>', 1)


def build(dist):
    from build_generation_view import CSS

    dist = Path(dist)
    summary = json.loads((OUT / 'summary.json').read_text())
    audit = json.loads((OUT / 'audit.json').read_text())
    assert summary['complete'] and audit['passed'] and audit['summary_sha256'] == sha(OUT / 'summary.json')
    spec = json.loads((OUT / 'spec.json').read_text())
    rows = read(OUT / 'paired-results.jsonl')
    assets = dist / 'assets/navigator9b/anli-r3-heldout-precision'
    assets.mkdir(parents=True, exist_ok=True)
    for name in ['spec.json', 'summary.json', 'audit.json', 'exclusions.json']:
        shutil.copy2(OUT / name, assets / name)
    view = dict(system_prompt=spec['system_prompt'], prompt_template=spec['prompt_template'],
                model=spec['model']['model'], revision=spec['model']['model_revision'],
                rows=[dict(**{k:r[k] for k in ['id', 'source_index', 'context', 'hypothesis', 'gold', 'reason']},
                           outputs=[{budget:{k:o[budget][k] for k in ['text', 'prediction', 'format_valid', 'error', 'truncated', 'new_tokens']}
                                     for budget in ['primary', 'secondary']} for o in r['outputs']]) for r in rows])
    (assets / 'view-data.json').write_text(json.dumps(view, ensure_ascii=False)+'\n')
    primary, secondary = summary['results']['primary'], summary['results']['secondary']
    table = ''
    report = ['# ANLI R3：固定规则精确 −2×的独立验证', '',
              '从官方 R3 test 1200 题中排除已评测的30题，以及与旧实验共用前提的15题，剩余1155题全部纳入。方向、倍数、prompt、生成与评分规则在新题生成前冻结。', '',
              '| 生成上限 | 无干预 | 规则精确 −2× | 差值 | 修正 | 改坏 | 格式无效：无干预 / 干预 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for budget, name in [('primary', '256：主结果'), ('secondary', '512：预先指定复核')]:
        metric = summary['results'][budget]
        a, b = metric['baseline'], metric['precision_minus2']
        table += ('<tr><td>'+name+'</td><td>'+str(a['correct'])+'/1155（'+pct(a['accuracy'])+'）</td><td>'+
                  str(b['correct'])+'/1155（'+pct(b['accuracy'])+'）</td><td>'+f"{100*metric['accuracy_difference']:+.2f}"+
                  ' 个百分点</td><td>'+str(metric['fixed'])+'</td><td>'+str(metric['damaged'])+'</td><td>'+str(a['invalid'])+' / '+str(b['invalid'])+'</td></tr>')
        report.append(f"| {name} | {a['correct']}/1155 ({pct(a['accuracy'])}) | {b['correct']}/1155 ({pct(b['accuracy'])}) | {100*metric['accuracy_difference']:+.2f} pp | {metric['fixed']} | {metric['damaged']} | {a['invalid']} / {b['invalid']} |")
    ci = primary['difference_95ci']
    inference = f"主结果差值的逐题配对 bootstrap 95% 区间：{100*ci[0]:+.2f} 至 {100*ci[1]:+.2f} 个百分点；精确 McNemar 双侧 p={primary['exact_mcnemar_p']:.4g}。"
    per_class = ''
    for i, label in enumerate(['A · 蕴含', 'B · 信息不足', 'C · 矛盾']):
        a, b = primary['baseline']['per_class'][i], primary['precision_minus2']['per_class'][i]
        per_class += f'<tr><td>{label}</td><td>{a["n"]}</td><td>{a["correct"]}（{pct(a["accuracy"])}）</td><td>{b["correct"]}（{pct(b["accuracy"])}）</td></tr>'
    report += ['', inference, '区间以逐题配对为单位，未按共享前提做聚类校正。', '',
               '同一 Qwen3.5-9B 版本，FP32、关闭TF32、greedy、非thinking；两组使用同一完整prompt和批次。保持原第17个block末位激活注入，模型权重不变。',
               '256主结果沿用原77/100协议；512为两组对称的预先指定复核，重跑所有含截断输出的完整原批次，并逐token验证旧前缀。格式不合法和截断输出均计错。', '',
               f"真实主生成 {summary['actual_primary_generations']} 次；512复核真实新增 {summary['actual_secondary_generations']} 次，其余已EOS完成的结果复用。", '',
               '此处的“新题”只指未参与本地实验；不能排除公开数据进入模型预训练。本结果是排除45题后的1155题成绩，不是官方完整1200题榜单分数。方向仍由旧100题选出；本轮不在新题上继续调参。', '',
               '旧100题的63%→77%只作为探索记录，不能替代本页独立验证。']
    (OUT / 'RESULTS.md').write_text('\n'.join(report)+'\n')
    shutil.copy2(OUT / 'RESULTS.md', assets / 'RESULTS.md')
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ANLI R3 · 独立验证</title><script src="/theme.js"></script><link rel="stylesheet" href="/theme-blue.css"><style>'''+CSS+'''</style></head><body><main><a href="index.html#capabilities">← Navigator</a><h1>ANLI R3：固定规则精确 −2×的独立验证</h1><p class="note">1155 道未参与此前本地调参的官方测试题。只对比无干预和已选定的规则精确 −2×，没有在新题上选择方向或强度。</p><section><h2>两组配对结果</h2><div class="scroll"><table><thead><tr><th>生成上限</th><th>无干预</th><th>规则精确 −2×</th><th>准确率差</th><th>修正</th><th>改坏</th><th>格式无效：无干预 / 干预</th></tr></thead><tbody>'''+table+'''</tbody></table></div><p>'''+html.escape(inference)+'''</p><p class="muted">区间以逐题配对为单位，未按共享前提做聚类校正。格式无效包含截断，均计错；512复核预先指定且对两组采用相同规则，不按成绩选择预算。</p></section><section><h2>各标准答案类别（256主结果）</h2><div class="scroll"><table><thead><tr><th>类别</th><th>题数</th><th>无干预正确</th><th>干预正确</th></tr></thead><tbody>'''+per_class+'''</tbody></table></div></section><section><h2>逐题查看</h2><p>保留题目、标准答案与数据集原始解析、实际prompt、两组完整模型输出。可筛选改对、改坏或仍然答错的题。</p><a href="assets/navigator9b/anli-r3-heldout-precision/samples.html">打开1155题对照 ↗</a></section><section><h2>实验范围</h2><p>官方 R3 test 共1200题；排除旧实验已用的30题，以及另15道与旧实验共用前提的题，保留1155题，按官方原顺序评测，不按标签或正确性筛选。两组使用同一Qwen3.5-9B版本、FP32、greedy、同一分析后作答prompt和相同批次。干预沿用原第17个block的precision −2×向量，模型权重不变。</p><p>“未使用”仅指本地实验，公开数据可能进入过模型预训练。本页不等于官方完整1200题分数。此前100题的63%→77%是选择方向所用的探索记录。</p><p>实际主生成 '''+str(summary['actual_primary_generations'])+''' 次；512复核新增 '''+str(summary['actual_secondary_generations'])+''' 次。已完成EOS的原结果复用，重跑结果均核对原token前缀。</p></section><p><a href="assets/navigator9b/anli-r3-heldout-precision/spec.json">冻结协议</a> · <a href="assets/navigator9b/anli-r3-heldout-precision/summary.json">原始统计</a> · <a href="assets/navigator9b/anli-r3-heldout-precision/audit.json">独立核算</a> · <a href="assets/navigator9b/anli-r3-heldout-precision/RESULTS.md">下载文字报告</a></p></main></body></html>'''
    (dist / 'heldout-precision.html').write_text(themed(page))
    viewer = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ANLI新题 · 两组输出</title><script src="/theme.js"></script><link rel="stylesheet" href="/theme-blue.css"><style>'''+CSS+'''nav{display:flex;gap:10px;flex-wrap:wrap;align-items:center}select,button{font:inherit;max-width:100%;padding:8px;border:1px solid #ccd6cc;border-radius:6px;background:#fffefa;color:inherit}pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font:14px/1.8 ui-monospace,monospace}.outputs{display:grid;grid-template-columns:1fr 1fr;gap:16px}.output{min-width:0;padding:16px;border:1px solid #dde2d9;border-radius:8px}.answer{font-size:22px;font-weight:600}@media(max-width:650px){.outputs{grid-template-columns:1fr}}</style></head><body><main><a href="/heldout-precision.html">← 独立验证结果</a><h1>ANLI 新题：无干预与规则精确 −2×</h1><p id="loading">加载逐题输出…</p><nav><select id="budget" aria-label="生成上限"><option value="primary">256：主结果</option><option value="secondary">512：预先指定复核</option></select><select id="filter" aria-label="样本筛选"><option value="all">全部题目</option><option value="fixed">干预改对</option><option value="damaged">干预改坏</option><option value="wrong">干预仍答错</option><option value="invalid">任一组格式无效</option></select><button id="previous">上一题</button><select id="sample" aria-label="样本"></select><button id="next">下一题</button><span id="position"></span></nav><p class="muted" id="identity"></p><article id="record" hidden><section><h2>1. 题目</h2><h3>前提</h3><pre id="premise"></pre><h3>假设</h3><pre id="hypothesis"></pre></section><section><h2>2. 标准答案</h2><p class="answer" id="gold"></p><div id="reasonBox"><h3>标准答案解析（数据集原文）</h3><pre id="reason"></pre></div></section><section><h2>3. 预测 prompt（两组相同）</h2><h3>System</h3><pre id="system"></pre><h3>User</h3><pre id="prompt"></pre></section><section><h2>4. 模型答案</h2><p class="muted">无干预：零向量；规则精确 −2×：原第17个block最后位置激活注入，作用于输入末尾及每步解码。下方是完整可见输出。</p><div class="outputs"><div class="output"><h3>无干预</h3><p class="answer" id="baseAnswer"></p><pre id="baseText"></pre><p class="muted" id="baseMeta"></p></div><div class="output"><h3>规则精确 −2×</h3><p class="answer" id="steeredAnswer"></p><pre id="steeredText"></pre><p class="muted" id="steeredMeta"></p></div></div></section></article><script>
const $=id=>document.getElementById(id),names=['蕴含','信息不足','矛盾'];let data,visible=[];
const label=p=>p===null?'格式无效':'ABC'[p]+' · '+names[p];
function show(){const i=Number($('sample').value),r=data.rows[i];if(!r){$('record').hidden=true;return}const b=$('budget').value;history.replaceState(null,'','?sample='+(i+1)+'&budget='+b+'&filter='+$('filter').value);$('record').hidden=false;$('identity').textContent='新样本 '+(i+1)+' · 官方序号 '+(r.source_index+1)+' · '+r.id;$('premise').textContent=r.context;$('hypothesis').textContent=r.hypothesis;$('gold').textContent=label(r.gold);$('reasonBox').hidden=!r.reason;$('reason').textContent=r.reason;$('system').textContent=data.system_prompt;$('prompt').textContent=data.prompt_template.replace('{context}',JSON.stringify(r.context)).replace('{hypothesis}',JSON.stringify(r.hypothesis));['base','steered'].forEach((key,j)=>{const o=r.outputs[j][b];$(key+'Answer').textContent=label(o.prediction);$(key+'Text').textContent=o.text;$(key+'Meta').textContent=o.new_tokens+' tokens · '+(o.format_valid?'格式有效':o.error)+(o.prediction===r.gold?' · 正确':' · 错误')});const k=visible.indexOf(i);$('position').textContent=(k+1)+' / '+visible.length;$('previous').disabled=k<=0;$('next').disabled=k>=visible.length-1}
function filter(){const old=Number($('sample').value),b=$('budget').value,f=$('filter').value;visible=data.rows.flatMap((r,i)=>{const a=r.outputs[0][b],s=r.outputs[1][b],x=a.prediction===r.gold,y=s.prediction===r.gold;return(f==='all'||f==='fixed'&&!x&&y||f==='damaged'&&x&&!y||f==='wrong'&&!y||f==='invalid'&&(!a.format_valid||!s.format_valid))?[i]:[]});$('sample').replaceChildren(...visible.map(i=>new Option('新样本 '+(i+1),i)));if(visible.includes(old))$('sample').value=old;if(!visible.length){$('record').hidden=true;$('position').textContent='0 / 0';$('previous').disabled=$('next').disabled=true;return}show()}
$('budget').onchange=filter;$('filter').onchange=filter;$('sample').onchange=show;$('previous').onclick=()=>{$('sample').selectedIndex--;show()};$('next').onclick=()=>{$('sample').selectedIndex++;show()};
fetch('view-data.json').then(r=>{if(!r.ok)throw Error('HTTP '+r.status);return r.json()}).then(d=>{data=d;$('loading').textContent=d.model+' · 共 '+d.rows.length+' 道新题；A=蕴含，B=信息不足，C=矛盾。';const p=new URLSearchParams(location.search);if(['primary','secondary'].includes(p.get('budget')))$('budget').value=p.get('budget');if(['all','fixed','damaged','wrong','invalid'].includes(p.get('filter')))$('filter').value=p.get('filter');const desired=Math.max(0,Math.min(d.rows.length-1,Number(p.get('sample')||1)-1));$('sample').add(new Option('',desired));filter()}).catch(e=>{$('loading').textContent='加载失败：'+e.message});
</script></main></body></html>'''
    (assets / 'samples.html').write_text(themed(viewer))
    return ('<section id="heldoutPrecision"><h2>新题验证：固定规则精确 −2×</h2><p>1155道未参与此前本地调参的官方 R3 测试题。无干预 '+
            pct(primary['baseline']['accuracy'])+' → 规则精确 −2× '+pct(primary['precision_minus2']['accuracy'])+
            '；修正 '+str(primary['fixed'])+' 题，改坏 '+str(primary['damaged'])+' 题。</p><a target="_top" href="heldout-precision.html">查看独立验证和逐题输出 ↗</a></section>')


if __name__ == '__main__':
    build(Path(__file__).resolve().parent.parent / 'steering_lab/dist')

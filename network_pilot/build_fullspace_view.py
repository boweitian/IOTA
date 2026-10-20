"""Live progress and final full-space report, with exact per-configuration outputs."""
import argparse
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

from fullspace_detailed import OUT, ROOT
from sentiment_navigator import read
from sentiment_prepare import sha

DIST=ROOT.parent/'steering_lab/dist'
REL='assets/navigator9b/anli-r3-fullspace-detailed'
TITLE='ANLI R3：1200题详细分析与方向扫描'


def percent(x):return f'{100*x:.2f}%'


def frame(body,extra=''):
    from build_generation_view import CSS
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+TITLE+
            '</title><script src="/theme.js"></script><style>'+CSS+'nav{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.outputs{display:grid;grid-template-columns:1fr 1fr;gap:15px}.output{min-width:0;border:1px solid #dde2d9;border-radius:8px;padding:16px}.status{font-variant-numeric:tabular-nums}.bar{height:10px;background:#e0e8df;border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;background:#46735d}progress{width:100%;height:20px}.answer{font-size:21px}pre{max-width:100%}@media(max-width:700px){.outputs{grid-template-columns:1fr}}'+
            '</style><link rel="stylesheet" href="/theme-blue.css"></head><body><main><a href="/index.html#capabilities">← Navigator</a>'+body+extra+'</main></body></html>')


def running(dist,spec):
    assets=dist/REL;assets.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUT/'spec.json',assets/'spec.json')
    page=frame('<h1>'+TITLE+'</h1><p class="note">正在运行。官方 R3 test 全部1200题，统一详细分析后以“答案：A/B/C”结束。53个主配置＋8个随机对照，共73200次基准生成。</p>'
        '<section id="runProgress"><h2>进度</h2><p id="phase">读取进度…</p><progress id="progress" max="73200" value="0"></progress><p class="status" id="counts"></p><p class="muted" id="updated"></p></section>'
        '<section><h2>本轮测什么</h2><p>现有10条命名方向分别测试−2、−1、+1、+2倍，另测12个已有混合配置和同协议baseline。两个随机方向各测四档强度，作为有限的探索对照。所有条件使用同一prompt、模型和1024-token生成上限。</p><p>报告将区分baseline、事后最佳固定方向、逐题标签可见覆盖，以及每题可行配置数。逐题覆盖不是Navigator已能实现的准确率。</p></section>'
        '<section><h2>统一预测 prompt</h2><pre>'+html.escape(spec['prompt_template'])+'</pre><p class="muted">各题只替换context和hypothesis。官方答案与reason不会进入预测prompt。已有100题与1155题结果保留为不同协议的历史记录。</p></section>'
        '<p><a href="'+REL+'/spec.json">冻结协议</a> · <a href="/heldout-precision.html">上一轮固定方向独立验证</a></p>',
        '<script>async function update(){try{const r=await fetch("/'+REL+'/progress.json?t="+Date.now(),{cache:"no-store"});if(!r.ok)throw Error(r.status);const d=await r.json();document.getElementById("phase").textContent=d.phase_label;document.getElementById("counts").textContent=d.completed_generations.toLocaleString()+" / 73,200 条生成；完整配置 "+d.completed_configurations+" / 61";document.getElementById("progress").value=d.completed_generations;document.getElementById("updated").textContent="更新："+new Date(d.updated_at*1000).toLocaleString();if(d.phase==="published"){location.reload();return}}catch(e){document.getElementById("phase").textContent="等待下一次进度更新"}setTimeout(update,20000)}update()</script>')
    (dist/'anli-fullspace.html').write_text(page)
    return '<section id="fullspaceDetailed"><h2>1200题详细分析与全方向扫描</h2><p>53个主配置＋8个随机对照，统一详细分析、最后输出答案。正在生成并核查。</p><a target="_top" href="anli-fullspace.html">查看实时进度与冻结格式 ↗</a></section>'


def build(dist=DIST):
    dist=Path(dist);spec=json.loads((OUT/'spec.json').read_text())
    if not (OUT/'audit.json').exists():return running(dist,spec)
    summary=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'audit.json').read_text());diagnosis=json.loads((OUT/'diagnosis-summary.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==sha(OUT/'summary.json')
    assets=dist/REL;assets.mkdir(parents=True,exist_ok=True)
    for name in ['spec.json','summary.json','audit.json','diagnosis-summary.json','execution.json','analysis-protocol.json']:
        shutil.copy2(OUT/name,assets/name)
    for name in ['feasible-space.jsonl','response-matrix.npz']:shutil.copy2(OUT/name,assets/name)
    if (OUT/'CONCLUSIONS.md').exists():shutil.copy2(OUT/'CONCLUSIONS.md',assets/'CONCLUSIONS.md')
    manual=json.loads((OUT/'manual-review.json').read_text()) if (OUT/'manual-review.json').exists() else dict(reviews=[])
    manual_by_index={v['sample']-1:v for v in manual['reviews']}
    if (OUT/'manual-review.json').exists():shutil.copy2(OUT/'manual-review.json',assets/'manual-review.json')
    from classification_generation import DEFAULT, SEMANTIC
    original_atlas=json.loads((DEFAULT/'search-atlas.json').read_text())
    definitions=[dict(id=c['id'],name=c['name'],**c['instructions']) for c in original_atlas['concepts']]
    ordinary=json.loads((SEMANTIC/'spec.json').read_text())['axes'][0]
    definitions.append(dict(id='ordinary_meaning',name='常见语义',positive=ordinary['positive'],negative=ordinary['negative']))
    (assets/'direction-definitions.json').write_text(json.dumps(definitions,ensure_ascii=False,indent=2)+'\n')
    shutil.copytree(OUT/'outputs',assets/'outputs',dirs_exist_ok=True)
    items=read(OUT/'feasible-space.jsonl');annotations={r['id']:r['annotation'] for r in read(OUT/'error-cases.jsonl')}
    data=dict(system_prompt=spec['system_prompt'],prompt_template=spec['prompt_template'],model=spec['model']['model'],
              actions=[dict(index=a['index'],name=a['name'],family=a['family'],norm=a['norm'],correct=summary['conditions'][str(a['index'])]['correct']) for a in spec['actions']],
              best_single=summary['best_fixed_single'],best_main=summary['best_main'],
              rows=[dict(r,annotation=annotations.get(r['id']),manual_review=manual_by_index.get(r['index'])) for r in items])
    (assets/'view-data.json').write_text(json.dumps(data,ensure_ascii=False)+'\n')
    m=summary['conditions'];best=summary['best_fixed_single'];best_main=summary['best_main'];coverage=summary['coverage']['named40']
    matrix=np.load(OUT/'response-matrix.npz');pred=matrix['predictions'];gold=matrix['gold'];base_ok=pred[:,0]==gold
    all_covered=np.any(pred==gold[:,None],axis=1)
    all_coverage=dict(correct=int(all_covered.sum()),n=1200,rescued_baseline_errors=int((all_covered&~base_ok).sum()),
                      unresolved_ids=[r['id'] for r,v in zip(items,all_covered) if not v])
    (assets/'all61-coverage.json').write_text(json.dumps(all_coverage,ensure_ascii=False,indent=2)+'\n')
    decomposition={}
    for j in dict.fromkeys([9,best,best_main]):
        ok=pred[:,j]==gold
        parts=dict(fixed_from_invalid=int((ok&(pred[:,0]<0)).sum()),
                   fixed_from_valid_wrong=int((ok&~base_ok&(pred[:,0]>=0)).sum()),
                   damaged_to_invalid=int((~ok&base_ok&(pred[:,j]<0)).sum()),
                   damaged_to_valid_wrong=int((~ok&base_ok&(pred[:,j]>=0)).sum()))
        assert parts['fixed_from_invalid']+parts['fixed_from_valid_wrong']==m[str(j)]['fixed']
        assert parts['damaged_to_invalid']+parts['damaged_to_valid_wrong']==m[str(j)]['damaged']
        decomposition[str(j)]=parts
    (assets/'format-decomposition.json').write_text(json.dumps(decomposition,ensure_ascii=False,indent=2)+'\n')
    def link(sample=1,condition=0,filter_name='all'):
        return REL+f'/samples.html?sample={sample}&condition={condition}&filter={filter_name}'
    def condition_row(j):
        v=m[str(j)];return ('<tr><td><a href="'+link(1,j)+'">'+html.escape(v['name'])+'</a></td><td>'+str(v['correct'])+'/1200（'+percent(v['accuracy'])+'）</td><td>'+f'{100*v["difference"]:+.2f}'+'</td><td>'+str(v['fixed'])+'</td><td>'+str(v['damaged'])+'</td><td>'+str(v['invalid'])+'</td><td>'+str(v['truncated'])+'</td></tr>')
    cards=''.join('<div class="card"><div class="big">'+percent(v/1200)+'</div><p>'+name+'</p><p class="muted">'+str(v)+' / 1200</p></div>' for name,v in [
        ('无干预baseline',m['0']['correct']),('最佳固定单方向',m[str(best)]['correct']),('最佳主配置（含baseline）',m[str(best_main)]['correct']),('至少一个命名单方向正确',coverage['correct'])])
    dose_rows=''
    for axis,doses in summary['trend']['per_axis'].items():
        ix=[j for j in spec['single_actions'] if spec['actions'][j]['axis']==axis]
        name=spec['actions'][ix[0]]['name'].rsplit(' ',1)[0]
        dose_rows+='<tr><td>'+html.escape(name)+'</td>'+''.join('<td><a href="'+link(1,j)+'">'+str(m[str(j)]['correct'])+'/1200</a></td>' for j in [ix[0],ix[1],0,ix[2],ix[3]])+'</tr>'
    histogram=coverage['histogram'];maximum=max(histogram or [1]);bars=''.join('<rect x="'+str(25+i*20)+'" y="'+str(180-150*v/max(1,maximum))+'" width="15" height="'+str(150*v/max(1,maximum))+'" fill="#46735d"><title>'+str(i)+'个可行配置：'+str(v)+'题</title></rect>' for i,v in enumerate(histogram))
    ticks=''.join('<text x="'+str(25+i*20)+'" y="205" font-size="13">'+str(i)+'</text>' for i in [0,10,20,30,40])
    cov_rows=''.join('<tr><td>'+name+'</td><td>'+str(summary['coverage'][key]['correct'])+'</td><td>'+str(summary['coverage'][key]['with_baseline_correct'])+'</td><td>'+str(summary['coverage'][key]['rescued_baseline_errors'])+'</td></tr>' for key,name in [
        ('native36','原9方向×4档'),('named40','10方向×4档'),('named_and_mixtures52','40单方向＋12混合'),('random8','2随机方向×4档')])
    cov_rows+='<tr><td>全部61配置（含baseline和随机对照）</td><td>'+str(all_coverage['correct'])+'</td><td>'+str(all_coverage['correct'])+'</td><td>'+str(all_coverage['rescued_baseline_errors'])+'</td></tr>'
    equal_rows=''.join('<tr><td>'+str(k)+'</td><td>'+percent(v['named40']['mean_coverage'])+'</td><td>'+percent(v['random8']['mean_coverage'])+'</td></tr>' for k,v in summary['equal_budget_random_comparison'].items())
    error_rows=''.join('<tr><td>'+html.escape(v['name'])+'</td><td>'+str(v['n'])+'</td><td>'+str(v['named_rescuable'])+'</td><td>'+str(v['best_single_correct'])+'</td><td>'+('、'.join('<a href="'+link(n,0)+'">'+str(n)+'</a>' for n in v['examples']) or '—')+'</td></tr>' for v in diagnosis['groups'].values() if v['n'])
    unresolved=coverage['unresolved_ids'];unresolved_links='、'.join('<a href="'+link(i+1,0)+'">'+str(i+1)+'</a>' for i,r in enumerate(items) if r['id'] in set(unresolved)) or '无'
    head='<tr><th>配置</th><th>正确率</th><th>相对baseline百分点</th><th>修正</th><th>改坏</th><th>格式无效</th><th>其中截断</th></tr>'
    body='<!--fullspace-complete--><h1>'+TITLE+'</h1><p class="note">完整官方R3 test，1200题×61配置；所有条件先详细分析，最后以“答案：A/B/C”作答。最佳配置与逐题覆盖均是看过答案后的搜索统计，不是已训练Navigator的测试成绩。</p><div class="cards">'+cards+'</div>'
    body+='<section id="finalResults"><h2>baseline与最佳固定配置</h2><p>最佳固定单方向：<b>'+html.escape(m[str(best)]['name'])+'</b>；最佳主配置：<b>'+html.escape(m[str(best_main)]['name'])+'</b>。</p><div class="scroll"><table><thead>'+head+'</thead><tbody>'+''.join(condition_row(j) for j in dict.fromkeys([0,9,best,best_main]))+'</tbody></table></div><p class="muted">固定方向是对全部1200题统一使用。最佳值由本批题事后选出，不能直接解释为新题泛化提升。</p></section>'
    class_rows=''.join('<tr><td>'+name+'</td><td>'+str(m['0']['per_class'][i]['correct'])+'/'+str(m['0']['per_class'][i]['n'])+'</td><td>'+str(m[str(best)]['per_class'][i]['correct'])+'/'+str(m[str(best)]['per_class'][i]['n'])+'</td></tr>' for i,name in enumerate(['A · 蕴含','B · 信息不足','C · 矛盾']))
    body+='<section><h2>结果解读与人工复核</h2><p>本轮最佳固定配置净增加40题：修正137题、改坏97题，提升3.33个百分点。配对检验未经多重比较校正p=0.0106；考虑52个主候选的选择后p=0.553，尚不能宣称稳定的泛化收益。随机对照1的+1倍也达到830/1200，因此不能把改善全部归于方向名称所代表的语义。</p><p>最佳方向主要提升蕴含和矛盾判断，同时降低信息不足的正确率，符合更倾向作出确定判断的可见现象。</p><div class="scroll"><table><thead><tr><th>官方类别</th><th>无干预答对</th><th>最佳固定方向答对</th></tr></thead><tbody>'+class_rows+'</tbody></table></div><p>40个单方向中，16题全未解、140题只有1–5个可行配置、495题有31–40个可行配置。错误修复与保护原本答对的题需要同时优化。12个已有混合配置没有超过baseline，最佳801/1200。</p><p>详细检查了全部61配置仍未解的14题：既有局数与整场胜负混淆、作者与作品混淆、字符串读法遗漏，也有官方reason与题干不一致的个例。不能把它们全部解释为需要更宽松的规则。自动辅助归类会重复模型自身的误读，例如第1116题的the year prior to 2004指2003，模型与辅助归类却都读成任意早于2004的年份；第843题也都混淆作者与作品。因此下方自动归类数量不代表已经验证的错误原因占比。</p><p>下一步Navigator的研究目标是只凭题目选择方向、同时控制改坏风险；98.67%的逐题覆盖只说明已有候选中存在正确回答。当前1200题已用于搜索和诊断，未来路由效果仍须在独立评估中检验。</p><p><a href="'+REL+'/CONCLUSIONS.md">阅读中文结论与14道未解题复核</a></p></section>'
    flow_rows=''.join('<tr><td>'+html.escape(m[j]['name'])+'</td><td>'+str(v['fixed_from_invalid'])+'</td><td>'+str(v['fixed_from_valid_wrong'])+'</td><td>'+str(v['damaged_to_invalid'])+'</td><td>'+str(v['damaged_to_valid_wrong'])+'</td></tr>' for j,v in decomposition.items())
    body+='<section><h2>选项判断变化与格式失败的区别</h2><div class="scroll"><table><thead><tr><th>配置</th><th>格式失败→答对</th><th>有效但选错→答对</th><th>答对→格式失败</th><th>答对→有效但选错</th></tr></thead><tbody>'+flow_rows+'</tbody></table></div><p class="muted">表中“有效”只表示满足输出格式，并不证明解释正确或对应某种内部机制。</p></section>'
    body+='<section><h2>正向、负向与baseline分布</h2><p>40个命名单方向：正向平均 '+percent(summary['trend']['positive_mean_accuracy'])+'；baseline '+percent(summary['trend']['baseline_accuracy'])+'；负向平均 '+percent(summary['trend']['negative_mean_accuracy'])+'。</p><div class="scroll"><table><thead><tr><th>方向</th><th>−2×</th><th>−1×</th><th>0×</th><th>+1×</th><th>+2×</th></tr></thead><tbody>'+dose_rows+'</tbody></table></div><p class="muted">每行0×复用同一个同协议baseline，所有干预与baseline使用完全相同的详细作答prompt。</p></section>'
    body+='<section><h2>每题是否至少存在一个可行方向</h2><div class="scroll"><table><thead><tr><th>候选集合</th><th>至少一个候选正确</th><th>加入baseline后</th><th>修复baseline错题的并集</th></tr></thead><tbody>'+cov_rows+'</tbody></table></div><p>40个命名单方向中，每题答对配置个数的分布：</p><svg viewBox="0 0 860 220" role="img" aria-label="每题可行单方向数的分布">'+bars+ticks+'</svg><p>命名单方向与baseline均未解的题：'+unresolved_links+'。</p><a href="'+link(1,0,'unresolved')+'">逐题查看未解题</a><p class="muted">这里的“存在”只针对已测离散网格。逐题挑选使用了标准答案，覆盖率不能作为Navigator实际准确率。</p></section>'
    body+='<section><h2>相同尝试次数下的随机对照</h2><p>从命名40配置和随机8配置中，分别抽取相同数量的配置，重复500次，报告平均逐题覆盖。随机方向仅两条，这只是有限对照。</p><div class="scroll"><table><thead><tr><th>尝试配置数</th><th>命名方向平均覆盖</th><th>随机方向平均覆盖</th></tr></thead><tbody>'+equal_rows+'</tbody></table></div></section>'
    body+='<section><h2>baseline错题的候选原因分布</h2><p>以下是同模型在评测结束后，对可见回答的辅助归类。类别和解释属于诊断假设，不是已确认的内部机制；官方标签不作修改。</p><div class="scroll"><table><thead><tr><th>主要现象</th><th>题数</th><th>存在命名可行方向</th><th>最佳固定单方向答对</th><th>示例编号</th></tr></thead><tbody>'+error_rows+'</tbody></table></div></section>'
    if manual['reviews']:
        manual_rows=''.join('<tr><td><a href="'+link(v['sample'],0)+'">'+str(v['sample'])+'</a></td><td style="white-space:normal">'+html.escape(v['observation'])+'</td></tr>' for v in manual['reviews'])
        body+='<section><h2>人工抽查的可见现象与标注边界</h2><p>以下是挑选的具体示例，不用于估计各类问题占比。标注有张力的题仍保留官方计分。</p><div class="scroll"><table><thead><tr><th>样本</th><th>人工检查记录</th></tr></thead><tbody>'+manual_rows+'</tbody></table></div></section>'
    body+='<section><h2>12个已有混合配置</h2><div class="scroll"><table><thead>'+head+'</thead><tbody>'+''.join(condition_row(j) for j in spec['mixture_actions'])+'</tbody></table></div></section>'
    definition_rows=''.join('<tr><td>'+html.escape(d['name'])+'</td><td style="white-space:normal">'+html.escape(d['positive'])+'</td><td style="white-space:normal">'+html.escape(d['negative'])+'</td></tr>' for d in definitions)
    body+='<section><h2>方向如何定义</h2><p>原9方向来自角色与场景中的指令意图差分；常见语义来自此前固定的通用语义指令。名称描述构造意图，其在NLI中的实际作用由本轮结果检验。</p><div class="scroll"><table><thead><tr><th>方向</th><th>正向构造指令</th><th>负向构造指令</th></tr></thead><tbody>'+definition_rows+'</tbody></table></div></section>'
    body+='<section><h2>逐题原始记录</h2><p>每题按四段展示：题目、标准答案及原始reason、实际预测prompt、baseline与所选配置的完整输出。可以筛选改对、改坏、格式错误或未解题。</p><a href="'+link(1,best)+'">打开1200题与61配置 ↗</a><p>保存了全部1200题的无干预题目隐藏状态，供后续Navigator研究使用。此处没有训练或报告Navigator。</p></section>'
    body+='<p><a href="'+REL+'/spec.json">冻结协议</a> · <a href="'+REL+'/summary.json">完整统计</a> · <a href="'+REL+'/audit.json">独立核算</a> · <a href="'+REL+'/feasible-space.jsonl">逐题可行解记录</a> · <a href="'+REL+'/response-matrix.npz">结果矩阵</a> · <a href="'+REL+'/RESULTS.md">文字报告</a></p>'
    (dist/'anli-fullspace.html').write_text(frame(body))
    viewer=frame('<h1>1200题：题目、标准答案、prompt与完整输出</h1><p id="loading">加载样本…</p><nav><select id="condition" aria-label="干预配置"></select><select id="filter" aria-label="筛选"><option value="all">全部题目</option><option value="baseline_wrong">baseline错题</option><option value="wrong">所选配置错题</option><option value="fixed">所选配置改对</option><option value="damaged">所选配置改坏</option><option value="invalid">所选配置格式无效</option><option value="unresolved">命名单方向与baseline均未解</option><option value="unresolved_main">53个主配置均未解</option><option value="unresolved_all">全部61配置均未解</option></select><button id="previous">上一题</button><select id="sample" aria-label="样本"></select><button id="next">下一题</button><span id="position"></span></nav><p class="muted" id="identity"></p><article id="record" hidden><section><h2>1. 题目</h2><h3>前提</h3><pre id="premise"></pre><h3>假设</h3><pre id="hypothesis"></pre></section><section><h2>2. 标准答案</h2><p class="answer" id="gold"></p><div id="reasonBox"><h3>标准答案解析（数据集原文）</h3><pre id="reason"></pre></div></section><section><h2>3. 预测 prompt（两组相同）</h2><h3>System</h3><pre id="system"></pre><h3>User</h3><pre id="prompt"></pre></section><section><h2>4. 模型答案</h2><p class="muted" id="intervention"></p><div class="outputs"><div class="output"><h3>无干预</h3><p class="answer" id="baseAnswer"></p><pre id="baseText"></pre><p class="muted" id="baseMeta"></p></div><div class="output"><h3 id="selectedName"></h3><p class="answer" id="selectedAnswer"></p><pre id="selectedText"></pre><p class="muted" id="selectedMeta"></p></div></div><p id="feasible"></p><details id="diagnosis"><summary>同模型辅助归类（可能有误）</summary><p id="annotation"></p></details><details id="manualReview" hidden><summary>人工核查记录</summary><p id="manualObservation"></p></details></section></article>',r'''<script>
const $=id=>document.getElementById(id),labels=['蕴含','信息不足','矛盾'];let data,visible=[],revision=0;const cache=new Map();const label=p=>p===null||p<0?'格式无效':'ABC'[p]+' · '+labels[p];
function escaped(s){return JSON.stringify(s)}
async function outputs(j,i){const key=j+':'+Math.floor(i/16);if(!cache.has(key))cache.set(key,fetch('outputs/action-'+String(j).padStart(2,'0')+'/batch-'+String(Math.floor(i/16)).padStart(3,'0')+'.jsonl').then(r=>{if(!r.ok)throw Error('HTTP '+r.status);return r.text()}).then(t=>t.trim().split('\n').map(JSON.parse)));return(await cache.get(key))[i%16]}
async function show(){const ticket=++revision,i=Number($('sample').value),j=Number($('condition').value),r=data.rows[i];if(!r){$('record').hidden=true;return}const k=visible.indexOf(i);$('previous').disabled=k<=0;$('next').disabled=k>=visible.length-1;$('position').textContent=(k+1)+' / '+visible.length;history.replaceState(null,'','?sample='+(i+1)+'&condition='+j+'&filter='+$('filter').value);$('identity').textContent='样本 '+(i+1)+' · '+r.id;$('record').hidden=false;$('premise').textContent=r.context;$('hypothesis').textContent=r.hypothesis;$('gold').textContent=label(r.gold);$('reasonBox').hidden=!r.reason;$('reason').textContent=r.reason;$('system').textContent=data.system_prompt;const a=data.prompt_template.split('{context}');$('prompt').textContent=a[0]+escaped(r.context)+a[1].replace('{hypothesis}',escaped(r.hypothesis));$('selectedName').textContent=data.actions[j].name;$('intervention').textContent=j===0?'两组均为零向量baseline。':'所选配置：'+data.actions[j].name+'；第17个block末位激活注入，作用于输入末尾与每步解码。请求向量长度 '+data.actions[j].norm.toFixed(4)+'。';$('feasible').textContent='本题可答对的命名单方向（'+r.named_feasible.length+'/40）：'+(r.named_feasible.map(k=>data.actions[k].name).join('、')||'无')+'。其他可行主配置：'+(r.main_feasible.filter(k=>k===0||(k>=37&&k<=48)).map(k=>data.actions[k].name).join('、')||'无')+'。';$('manualReview').hidden=!r.manual_review;$('manualObservation').textContent=r.manual_review?r.manual_review.observation:'';$('diagnosis').hidden=!r.annotation;$('annotation').textContent=r.annotation?r.annotation.evidence:'';$('baseText').textContent=$('selectedText').textContent='加载原始输出…';try{const values=await Promise.all([outputs(0,i),outputs(j,i)]);if(ticket!==revision)return;values.forEach((v,n)=>{const key=n?'selected':'base';$(key+'Answer').textContent=label(v.prediction);$(key+'Text').textContent=v.text;$(key+'Meta').textContent=v.new_tokens+' tokens · '+(v.format_valid?'格式有效':v.error)+(v.prediction===r.gold?' · 正确':' · 错误')})}catch(e){if(ticket===revision)$('selectedText').textContent='读取失败：'+e.message}}
function filter(){const old=Number($('sample').value),j=Number($('condition').value),f=$('filter').value;visible=data.rows.flatMap((r,i)=>{const a=r.predictions[0]===r.gold,b=r.predictions[j]===r.gold;return(f==='all'||f==='baseline_wrong'&&!a||f==='wrong'&&!b||f==='fixed'&&!a&&b||f==='damaged'&&a&&!b||f==='invalid'&&r.predictions[j]<0||f==='unresolved'&&!a&&!r.named_feasible.length||f==='unresolved_main'&&!r.main_feasible.length||f==='unresolved_all'&&!r.main_feasible.length&&!r.random_feasible.length)?[i]:[]});$('sample').replaceChildren(...visible.map(i=>new Option('样本 '+(i+1),i)));if(visible.includes(old))$('sample').value=old;if(!visible.length){$('record').hidden=true;$('position').textContent='0 / 0';$('previous').disabled=$('next').disabled=true;return}show()}
$('condition').onchange=filter;$('filter').onchange=filter;$('sample').onchange=show;$('previous').onclick=()=>{$('sample').selectedIndex--;show()};$('next').onclick=()=>{$('sample').selectedIndex++;show()};fetch('view-data.json').then(r=>r.json()).then(d=>{data=d;$('loading').textContent=d.model+'；完整详细分析，统一1024-token上限。';$('condition').replaceChildren(...d.actions.map(a=>new Option(a.name+' · '+a.correct+'/1200',a.index)));const p=new URLSearchParams(location.search),j=Number(p.get('condition')||d.best_single);if(j>=0&&j<61)$('condition').value=j;const f=p.get('filter');if([...$('filter').options].some(o=>o.value===f))$('filter').value=f;const i=Math.max(0,Math.min(1199,Number(p.get('sample')||1)-1));$('sample').add(new Option('',i));filter()}).catch(e=>{$('loading').textContent='加载失败：'+e.message});
</script>''')
    viewer=viewer.replace('href="/index.html#capabilities"','href="/anli-fullspace.html"')
    (assets/'samples.html').write_text(viewer)
    report=['# '+TITLE,'','完整官方R3 test 1200题；先详细中文分析，最后答案行；统一1024-token上限。53主配置和8随机对照，73200条基准生成。','',
            f"baseline：{m['0']['correct']}/1200（{percent(m['0']['accuracy'])}）。",f"最佳固定单方向：{m[str(best)]['name']}，{m[str(best)]['correct']}/1200（{percent(m[str(best)]['accuracy'])}）。",f"最佳主配置：{m[str(best_main)]['name']}，{m[str(best_main)]['correct']}/1200。",f"40命名单方向逐题覆盖：{coverage['correct']}/1200；加入baseline后{coverage['with_baseline_correct']}/1200。",'','| 配置 | 正确 | 修正baseline | 改坏baseline | 无效 | 截断 |','|---|---:|---:|---:|---:|---:|']
    for j in range(61):
        v=m[str(j)];report.append(f"| {v['name']} | {v['correct']} | {v['fixed']} | {v['damaged']} | {v['invalid']} | {v['truncated']} |")
    report+=['','## 逐题可行配置分布','',json.dumps(coverage['histogram'],ensure_ascii=False),'','## baseline错题辅助归类','', '这些是同模型在评测结束后对可见输出的候选解释，不是内部因果机制或新真值裁决。','', '| 类别 | 题数 | 可被命名单方向修复 | 最佳固定单方向修复 |','|---|---:|---:|---:|']
    for group in diagnosis['groups'].values():
        if group['n']:report.append(f"| {group['name']} | {group['n']} | {group['named_rescuable']} | {group['best_single_correct']} |")
    report+=['',spec['scope'],spec['limitations'],'','逐题标签可见覆盖不是Navigator实际准确率；本轮没有训练Navigator。旧100题与1155题短分析协议的记录保持不变。']
    (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n');shutil.copy2(OUT/'RESULTS.md',assets/'RESULTS.md')
    return '<section id="fullspaceDetailed"><h2>1200题详细分析与全方向扫描</h2><p>baseline '+percent(m['0']['accuracy'])+'；最佳固定单方向 '+percent(m[str(best)]['accuracy'])+'；40个单方向逐题覆盖 '+percent(coverage['accuracy'])+'。61配置×1200题，独立核查通过。</p><a target="_top" href="anli-fullspace.html">查看全部结果、分布与逐题输出 ↗</a></section>'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--running',action='store_true');args=parser.parse_args()
    build(DIST)
    subprocess.run([sys.executable,str(ROOT.parent/'steering_lab/build_studio.py')],check=True)

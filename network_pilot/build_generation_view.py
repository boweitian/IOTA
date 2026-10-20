"""Readable generation-based Navigator and four-field sample viewer."""
import json,shutil,hashlib
import numpy as np
from pathlib import Path
from classification_generation import ROOT,DEFAULT,PILOT,SOURCE
from sentiment_prepare import sha

CSS='''*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#f5f3ed;color:#233a34;font:15px/1.75 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:25px 20px 60px}h1{font-size:28px;line-height:1.45}h2{font-size:20px}h3{font-size:14px;color:#687b71}p{overflow-wrap:anywhere}a{color:#315f50}section,.card{border:1px solid #dde2d9;border-radius:10px;padding:20px;background:#fffefa;margin:16px 0}.muted{font-size:13px;color:#687b71}.note{border-left:3px solid #a3bdab;padding:12px;background:#eaf0e7}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards .card{margin:0}.big{font-size:28px;font-weight:650}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:10px;border-bottom:1px solid #dde2d9;text-align:left;white-space:nowrap}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}select,button,input{font:inherit;border:1px solid #cbd5ca;border-radius:5px;padding:9px;background:white;color:inherit;max-width:100%}button{cursor:pointer}button:disabled{opacity:.4}pre,.original{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}pre{font:14px/1.8 ui-monospace,monospace;background:#f0f3ec;padding:14px;border-radius:6px}.answer{font-size:23px;font-weight:600}.id{overflow-wrap:anywhere}summary{cursor:pointer}.matrix td{padding:5px}.good{background:#dcebdd}.bad{background:#f1e5da}svg{display:block;width:100%;max-height:240px}@media(max-width:650px){main{padding:18px 14px}.cards{grid-template-columns:repeat(2,1fr)}section,.card{padding:15px}h1{font-size:24px}pre{font-size:13px}}'''

def names(summary,run=DEFAULT):
    mapping={c['id']:c['name'] for c in json.loads((Path(run)/'search-atlas.json').read_text())['concepts']};result=[]
    for a in summary['actions']:
        name=a['id']
        if name=='baseline':result.append('不干预 baseline')
        elif name=='principles_in_prompt':result.append('通用文字原则（无激活干预）')
        elif name.startswith('ordinary_meaning:'):result.append('常见语义 '+name.split(':')[1]+'×')
        elif ':' in name and name.split(':')[0] in mapping:result.append(mapping[name.split(':')[0]]+' '+name.split(':')[1]+'×')
        else:result.append(name)
    return result

def build(run=None,dist=None):
    out=Path(run or DEFAULT);summary=json.loads((out/'generation-summary.json').read_text());audit=json.loads((out/'audit.json').read_text())
    assert summary['complete'] and audit['passed'] and audit['summary_sha256']==sha(out/'generation-summary.json')
    diagnosis=json.loads((out/'semantic-diagnosis.json').read_text());labels=names(summary,out)
    dist=Path(dist or PILOT.parent/'steering_lab/dist');studio=json.loads((dist/'studio-evidence.json').read_text());binding=json.loads((out/'geometry-binding.json').read_text())
    model=binding['model'];other=studio['atlas']['model']
    for key in ['model','revision','layer_index','hidden_size']:
        if model[key]!=other[key]:raise ValueError('Navigator/Search model mismatch: '+key)
    if not np.array_equal(np.load(out/'search-geometry.npz')['directions'],np.array([c['native_direction'] for c in studio['atlas']['concepts']])):
        raise ValueError('Navigator/Search original directions mismatch')
    assets=dist/'assets/navigator9b/anli-r3-analysis';assets.mkdir(parents=True,exist_ok=True)
    brand_root=PILOT.parent/'steering_lab'
    shutil.copy2(brand_root/'theme-blue.css',dist/'theme-blue.css')
    shutil.copy2(brand_root/'theme.js',dist/'theme.js')
    shutil.copytree(brand_root/'assets/brand',dist/'assets/brand',dirs_exist_ok=True)
    for name in ['generation-spec.json','generation-summary.json','semantic-diagnosis.json','audit.json','view-data.json','dataset-manifest.json','ordinary-direction.npz']:
        shutil.copy2(out/name,assets/name)
    shutil.copytree(out/'outputs',assets/'outputs',dirs_exist_ok=True)
    sample='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐题查看 · 分析后作答</title><style>__CSS__</style></head><body><main><a href="../../../capabilities.html" target="_top">← 返回 Navigator</a><h1>逐题查看 · 先分析，再给答案</h1><div class="controls"><label>配置 <select id="condition"></select></label><button id="previous">上一题</button><select id="sample" aria-label="样本"></select><button id="next">下一题</button><label><input type="checkbox" id="wrongOnly">只看当前配置错题</label></div><p id="steering" class="muted"></p><p id="id" class="muted id"></p>
    <section><h2>1. 题目</h2><h3>前提</h3><p id="premise" class="original"></p><h3>假设</h3><p id="hypothesis" class="original"></p></section>
    <section><h2>2. 标准答案</h2><p id="gold" class="answer"></p><div id="reasonBox"><h3>标准答案解析（数据集原文）</h3><p id="reason" class="original"></p></div></section>
    <section><h2>3. 预测 prompt</h2><h3>System</h3><pre id="system"></pre><h3>User</h3><pre id="prompt"></pre></section>
    <section><h2>4. 模型答案</h2><p id="prediction" class="answer"></p><h3>模型 output（完整原文）</h3><pre id="output"></pre><p id="format" class="muted"></p></section>
    <script>const NAMES=__NAMES__;const $=s=>document.getElementById(s);let D,indices=[];const q=new URLSearchParams(location.search);let selected=Math.max(0,Number(q.get('sample')||1)-1);function render(){const j=Number($('condition').value),row=D.rows[selected],a=D.actions[j],o=row.outputs[j];$('premise').textContent=row.context;$('hypothesis').textContent=row.hypothesis;$('gold').textContent=D.letters[row.gold]+' · '+D.labels[row.gold];$('reason').textContent=row.reason;$('reasonBox').hidden=!row.reason.trim();$('system').textContent=D.model.system_prompt;$('prompt').textContent=a.prompt_template.replaceAll('{context}',JSON.stringify(row.context)).replaceAll('{hypothesis}',JSON.stringify(row.hypothesis));$('prediction').textContent=o.prediction===null?'格式无效':D.letters[o.prediction]+' · '+D.labels[o.prediction];$('output').textContent=o.text;$('format').textContent=o.format_valid?'按最后一行“答案：选项”评分。':'格式错误：'+o.error+'；计为错误，不从分析中猜选项。';$('id').textContent='样本 '+(selected+1)+' · '+row.split+' · '+row.id;$('steering').textContent='当前配置：'+NAMES[j]+'；激活位移范数 '+a.requested_norm.toFixed(3)+(j===53?'。原则直接加入 prompt。':j===0?'。':'。第17个block最后位置，作用于输入末尾及后续解码。');$('sample').value=selected;$('previous').disabled=indices.indexOf(selected)<=0;$('next').disabled=indices.indexOf(selected)>=indices.length-1;history.replaceState(null,'','?sample='+(selected+1)+'&condition='+j)}function populate(){const j=Number($('condition').value);indices=D.rows.map((r,i)=>i).filter(i=>!$('wrongOnly').checked||D.rows[i].outputs[j].prediction!==D.rows[i].gold);if(!indices.includes(selected))selected=indices[0]??0;$('sample').replaceChildren(...indices.map(i=>{const o=document.createElement('option');o.value=i;o.textContent='样本 '+(i+1)+' · '+D.rows[i].split;return o}));render()}fetch('view-data.json').then(r=>r.json()).then(d=>{D=d;$('condition').replaceChildren(...NAMES.map((name,i)=>{const o=document.createElement('option');o.value=i;o.textContent=name;return o}));$('condition').value=q.get('condition')||'0';populate();$('condition').onchange=populate;$('wrongOnly').onchange=populate;$('sample').onchange=()=>{selected=Number($('sample').value);render()};$('previous').onclick=()=>{selected=indices[indices.indexOf(selected)-1];render()};$('next').onclick=()=>{selected=indices[indices.indexOf(selected)+1];render()}});</script></main></body></html>'''.replace('__CSS__',CSS).replace('__NAMES__',json.dumps(labels,ensure_ascii=False).replace('<','\\u003c'))
    theme_hash=hashlib.sha256((brand_root/'theme-blue.css').read_bytes()+(brand_root/'theme.js').read_bytes()).hexdigest()[:12]
    sample=sample.replace('</head>','<script src="/theme.js?v='+theme_hash+'"></script><link rel="stylesheet" href="/theme-blue.css?v='+theme_hash+'"></head>',1)
    (assets/'samples.html').write_text(sample)
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ANLI R3 · 分析后作答</title><style>__CSS__</style></head><body><main><h1>ANLI R3 · 先分析，最后给答案</h1><p>每题先用中文给出1–2句简短分析，最后单独一行输出 <b>答案：A / B / C 中的一个选项</b>。本轮54个配置×100题，全部按真实生成的最终答案评分。</p><p class="note">同一批已查看的70题 train / 30题 test，用于探索诊断，不是全新测试集的泛化成绩。原单字母协议的65%保留在<a href="anli-direct-archive.html">历史归档</a>。</p><div class="controls"><label>样本范围 <select id="split"><option value="all">全部100题</option><option value="train">train 70题</option><option value="test">test 30题</option></select></label><a href="assets/navigator9b/anli-r3-analysis/samples.html" target="_top">逐题查看：题目、标准答案、prompt、完整输出 ↗</a></div>
    <div class="cards" id="scores"></div><section><h2>常见语义是否能修正更多错误？</h2><div class="scroll"><table><thead><tr><th>条件</th><th>正确</th><th>修正当前baseline错误</th><th>改坏当前baseline正确</th><th>格式无效</th></tr></thead><tbody id="semanticRows"></tbody></table></div><p id="semanticNote"></p><p class="muted">正向和负向均作对照；方向修正一个答案，不等于已证明原错误的内部原因。文字原则与激活方向分开统计。</p></section>
    <section><h2>每个方向：负向、不干预、正向</h2><div class="scroll"><table><thead><tr><th>方向</th><th>−2×</th><th>−1×</th><th>0×</th><th>+1×</th><th>+2×</th></tr></thead><tbody id="doseRows"></tbody></table></div><p id="trend"></p></section>
    <section><h2>每题有多少个单方向配置能答对？</h2><svg id="histogram" viewBox="0 0 900 220" role="img" aria-label="可解配置数分布"></svg><p id="distribution"></p><p class="muted">共40个单方向配置：原36个，加常见语义±1/±2。至少一个正确是已知标签后的覆盖率，不是已学会路由的成绩。</p></section>
    <details class="card"><summary>全部逐题 × 配置结果矩阵</summary><div class="scroll"><table class="matrix"><thead id="matrixHead"></thead><tbody id="matrixBody"></tbody></table></div></details>
    <details class="card"><summary>协议、原始记录与核算</summary><p>Qwen3.5-9B，FP32，greedy；显式短分析，模型模板thinking开关沿用关闭。原9方向及投影上限保持一致，新增实验性常见语义方向。无格式回退猜答案；截断、缺少终行、多答案均单独计为无效并按错题计分。</p><p><a href="assets/navigator9b/anli-r3-analysis/generation-spec.json">完整协议</a> · <a href="assets/navigator9b/anli-r3-analysis/generation-summary.json">全部统计</a> · <a href="assets/navigator9b/anli-r3-analysis/semantic-diagnosis.json">语义候选组诊断</a> · <a href="assets/navigator9b/anli-r3-analysis/audit.json">独立核算</a> · <a href="sst2-archive.html">SST-2归档</a> · <a href="sst5-archive.html">SST-5归档</a></p></details>
    <script>const S=__SUMMARY__,N=__NAMES__;const $=s=>document.getElementById(s);function show(){const split=$('split').value,lo=split==='test'?70:0,hi=split==='train'?70:100,n=hi-lo,get=j=>S.results[j].splits[split],base=get(0),eligible=[...Array.from({length:36},(_,i)=>i+1),49,50,51,52];const best=eligible.reduce((a,b)=>get(b).correct>get(a).correct?b:a,eligible[0]);const counts=S.coverage.all_singles.per_item_solutions.slice(lo,hi),cover=counts.filter(x=>x>0).length,selected=get(S.train_selected_single);$('scores').innerHTML=[['不干预 baseline',base.correct],['事后最佳固定单方向',get(best).correct],['至少一个单方向正确',cover],['仅train选出的固定方向',selected.correct]].map(([title,v])=>'<div class="card"><div class="big">'+(100*v/n).toFixed(1)+'%</div><p>'+title+'</p><p class="muted">'+v+' / '+n+'</p></div>').join('');$('semanticRows').innerHTML=[0,49,50,51,52,53].map(j=>{const m=get(j);return '<tr><td>'+N[j]+'</td><td>'+m.correct+'/'+n+'</td><td>'+m.fixed+'</td><td>'+m.damaged+'</td><td>'+m.invalid_format+'</td></tr>'}).join('');$('semanticNote').textContent='所有条件使用相同的分析后作答格式。最佳固定方向：'+N[best]+'；训练选定：'+N[S.train_selected_single]+'。';let rows='';for(let i=0;i<9;i++){const ix=1+4*i;rows+='<tr><td>'+N[ix].split(' ')[0]+'</td>'+[ix,ix+1,0,ix+2,ix+3].map(j=>'<td>'+get(j).correct+'/'+n+'</td>').join('')+'</tr>'}rows+='<tr><td>常见语义</td>'+[49,50,0,51,52].map(j=>'<td>'+get(j).correct+'/'+n+'</td>').join('')+'</tr>';$('doseRows').innerHTML=rows;const pos=Array.from({length:9},(_,i)=>[3+4*i,4+4*i]).flat(),neg=Array.from({length:9},(_,i)=>[1+4*i,2+4*i]).flat(),mean=ix=>ix.reduce((a,j)=>a+get(j).accuracy,0)/ix.length;$('trend').textContent='原9方向：正向平均 '+(mean(pos)*100).toFixed(2)+'%，不干预 '+(base.accuracy*100).toFixed(2)+'%，负向平均 '+(mean(neg)*100).toFixed(2)+'%。';const bins=Array(41).fill(0);counts.forEach(x=>bins[x]++);const max=Math.max(...bins,1);$('histogram').innerHTML=bins.map((v,i)=>'<rect x="'+(22+i*20.5)+'" y="'+(185-v/max*160)+'" width="15" height="'+(v/max*160)+'" fill="#46735d"><title>'+i+'个可行配置：'+v+'题</title></rect>').join('')+[0,10,20,30,40].map(i=>'<text x="'+(22+i*20.5)+'" y="210" font-size="14">'+i+'</text>').join('');$('distribution').textContent='没有单方向配置答对：'+bins[0]+'题；至少一个答对：'+cover+'题。';$('matrixHead').innerHTML='<tr><th>样本</th>'+N.map(x=>'<th>'+x+'</th>').join('')+'</tr>';$('matrixBody').innerHTML=Array.from({length:n},(_,k)=>{const i=k+lo;return '<tr><td>'+(i+1)+'</td>'+N.map((_,j)=>{const p=S.results[j].splits.all.predictions[i];const gold=GOLD[i];return '<td class="'+(p===gold?'good':'bad')+'"><a target="_top" href="assets/navigator9b/anli-r3-analysis/samples.html?sample='+(i+1)+'&condition='+j+'">'+(p<0?'无效':'ABC'[p])+'</a></td>'}).join('')+'</tr>'}).join('')}const GOLD=__GOLD__;$('split').onchange=show;show();</script></main></body></html>'''
    zero_diagnosis=out/'zero-gap-diagnosis-v1/summary.json'
    if zero_diagnosis.exists():
        controls=json.loads(zero_diagnosis.read_text());assert controls['passed']
        control_names={'zero_repeat':'0× 独立复跑','zero_no_hook':'0× 完全移除 hook','zero_reverse_batch_order':'0× 颠倒批次顺序','decomposition_plus1_replay':'分解推理 +1× 复跑','random0_minus1':'随机轴一 −1×','random0_plus1':'随机轴一 +1×','random1_minus1':'随机轴二 −1×','random1_plus1':'随机轴二 +1×','ordinary_minus001':'常见语义 −0.01×','ordinary_plus001':'常见语义 +0.01×'}
        control_rows=''.join('<tr><td>'+control_names[k]+'</td><td>'+str(v['correct'])+'/100</td><td>'+str(v['fixed'])+'</td><td>'+str(v['damaged'])+'</td><td>'+str(v['invalid'])+'</td></tr>' for k,v in controls['controls'].items())
        section='<section id="zeroGapControls"><h2>0× 差异诊断：复现与随机方向对照</h2><p>以下是另行生成的1000次对照，不计入上方54配置。两条随机轴各测试±1×，向量长度与原方向1×一致，prompt及评分规则相同。</p><div class="scroll"><table><thead><tr><th>对照</th><th>正确</th><th>修正0×错误</th><th>改坏0×正确</th><th>格式无效</th></tr></thead><tbody>'+control_rows+'</tbody></table></div><p class="note">表中每行0×复用同一baseline，不是十次独立重复。随机干预也能改善这批题的结果，因此不能把提升直接归因于命名方向的语义。仅两个随机轴，且重复使用同100题，此处仍是探索性诊断。</p><a href="assets/navigator9b/anli-r3-analysis/zero-gap-summary.json">查看对照原始统计</a></section>'
        page=page.replace('<section><h2>每题有多少个单方向配置能答对？</h2>',section+'<section><h2>每题有多少个单方向配置能答对？</h2>')
        shutil.copy2(zero_diagnosis,assets/'zero-gap-summary.json')
    mixture_dir=out/'top-three-mixture-v1'
    if (mixture_dir/'summary.json').exists():
        mix=json.loads((mixture_dir/'summary.json').read_text());assert mix['passed']
        spec=json.loads((mixture_dir/'spec.json').read_text());reg=mix['regularization']
        selection='、'.join(labels[a['index']] for a in mix['selected'])
        section='<section id="topThreeMixture"><h2>前三方向叠加复测</h2><p>按全部100题的单方向成绩选择：'+selection+'。共同缩放系数 '+format(reg['cap_factor'],'.4f')+'，保证每个参与方向上的投影绝对值不超过2×。</p><div class="scroll"><table><thead><tr><th>条件</th><th>全部100题</th><th>train70</th><th>test30</th><th>格式无效</th></tr></thead><tbody>'
        for j in [0,mix['selected'][0]['index']]:
            metrics=summary['results'][str(j)]['splits'];section+='<tr><td>'+labels[j]+'</td><td>'+str(metrics['all']['correct'])+'</td><td>'+str(metrics['train']['correct'])+'</td><td>'+str(metrics['test']['correct'])+'</td><td>'+str(metrics['all']['invalid_format'])+'</td></tr>'
        section+='<tr><td>前三方向叠加</td><td>'+str(mix['correct'])+'</td><td>'+str(mix['train_correct'])+'</td><td>'+str(mix['test_correct'])+'</td><td>'+str(mix['invalid_format'])+'</td></tr></tbody></table></div><p>相比最佳单方向：修正 '+str(mix['vs_best_single']['fixed'])+' 题，改坏 '+str(mix['vs_best_single']['damaged'])+' 题。</p><p class="note">选前三用到了全部100题的标签，包括test标签，属于事后组合实验；不能作为独立测试成绩。</p><a target="_top" href="assets/navigator9b/anli-r3-analysis/top-three-samples.html?condition=2">逐题查看叠加后的真实输出</a> · <a href="assets/navigator9b/anli-r3-analysis/top-three-summary.json">原始统计</a></section>'
        page=page.replace('<section><h2>每题有多少个单方向配置能答对？</h2>',section+'<section><h2>每题有多少个单方向配置能答对？</h2>')
        shutil.copy2(mixture_dir/'summary.json',assets/'top-three-summary.json')
        shutil.copy2(mixture_dir/'spec.json',assets/'top-three-spec.json')
        shutil.copy2(mixture_dir/'outputs.jsonl',assets/'top-three-outputs.jsonl')
        view=json.loads((out/'view-data.json').read_text());best=mix['selected'][0]['index'];mix_outputs=[json.loads(l) for l in (mixture_dir/'outputs.jsonl').read_text().splitlines()]
        view['actions']=[view['actions'][0],view['actions'][best],dict(id='top_three_mixture',prompt_template=spec['prompt_template'],requested_norm=reg['norm'])]
        for row,value in zip(view['rows'],mix_outputs):
            assert row['id']==value['id']
            row['outputs']=[row['outputs'][0],row['outputs'][best],{k:value[k] for k in ['text','prediction','format_valid','error']}]
        (assets/'top-three-view-data.json').write_text(json.dumps(view,ensure_ascii=False)+'\n')
        mix_names=[labels[0],labels[best],'前三方向叠加（共同缩放 '+format(reg['cap_factor'],'.4f')+'）']
        mix_page=sample.replace('const NAMES='+json.dumps(labels,ensure_ascii=False).replace('<','\\u003c'),'const NAMES='+json.dumps(mix_names,ensure_ascii=False)).replace("fetch('view-data.json')","fetch('top-three-view-data.json')")
        (assets/'top-three-samples.html').write_text(mix_page)
    anchor_summary=out/'precision-anchor-v1/summary.json'
    if anchor_summary.exists() and json.loads(anchor_summary.read_text()).get('complete') and (out/'precision-anchor-v1/audit.json').exists():
        from build_anchor_view import build as build_anchor
        anchor_section=build_anchor(dist)
        page=page.replace('<section><h2>每题有多少个单方向配置能答对？</h2>',anchor_section+'<section><h2>每题有多少个单方向配置能答对？</h2>')
    heldout=ROOT/'runs/anli-r3-heldout-precision-v1'
    if (heldout/'summary.json').exists() and (heldout/'audit.json').exists():
        from build_heldout_precision_view import build as build_heldout
        heldout_section=build_heldout(dist)
        page=page.replace('<section><h2>每题有多少个单方向配置能答对？</h2>',heldout_section+'<section><h2>每题有多少个单方向配置能答对？</h2>')
    fullspace=ROOT/'runs/anli-r3-fullspace-detailed-1200-v1'
    if (fullspace/'spec.json').exists():
        from build_fullspace_view import build as build_fullspace
        fullspace_section=build_fullspace(dist)
        page=page.replace('<section><h2>每题有多少个单方向配置能答对？</h2>',fullspace_section+'<section><h2>每题有多少个单方向配置能答对？</h2>')
    original_rows=[json.loads(l) for l in (SOURCE/'feasible-space.jsonl').read_text().splitlines()]
    page=page.replace('__CSS__',CSS).replace('__SUMMARY__',json.dumps(summary,ensure_ascii=False).replace('<','\\u003c')).replace('__NAMES__',json.dumps(labels,ensure_ascii=False).replace('<','\\u003c')).replace('__GOLD__',json.dumps([r['gold'] for r in original_rows]))
    old_samples=dist/'assets/navigator9b/anli-r3-diagnostics/diagnosis.html'
    if old_samples.exists() and 'historicalFormatNote' not in old_samples.read_text():
        old_text=old_samples.read_text()
        banner='<p id="historicalFormatNote" class="muted">历史归档：本页保留原单字母输出。<a id="currentFormatLink" href="../anli-r3-analysis/samples.html">查看当前分析后作答记录 ↗</a></p>'
        old_text=old_text.replace('<main>','<main>'+banner,1).replace('</body>','<script>document.getElementById("currentFormatLink").href="../anli-r3-analysis/samples.html?sample="+(Number(location.hash.replace("#sample-",""))||1)+"&condition=0";</script></body>')
        old_samples.write_text(old_text)
    prior=dist/'capabilities.html';archive=dist/'anli-direct-archive.html'
    if not archive.exists():shutil.copy2(prior,archive)
    import sys
    sys.path.append(str(PILOT.parent/'steering_lab'))
    from workspace_bundle import embedded
    for name in ['capabilities.html','router-analysis.html']:(dist/name).write_text(embedded(page,'capabilities'))
    metadata=json.loads((dist/'workspace-evidence.json').read_text()) if (dist/'workspace-evidence.json').exists() else {}
    metadata.update(model=model['model'].split('/')[-1],scenario_model=other['model'].split('/')[-1],model_revision=model['revision'],scenario_layer_index=model['layer_index'],
        nodes=9,scenario_nodes=9,samples=100,train_samples=70,test_samples=30,benchmark='ANLI R3',shared_original_directions_bit_exact=True,geometry_sha256=binding['geometry_sha256'],
        studio_protocol_sha256=studio['protocol_sha256'],direction_sweep_complete=True,selected_policy={'action_index':summary['train_selected_single'],'method':'train-only fixed rule'},
        settings=54,evidence_records=5400,sweep_measurements=5400,calibration_samples=0,output_protocol='analysis_then_答案',navigator_complete=True,
        sweep_metrics=dict(baseline=summary['results']['0']['splits']['all']),suites={'anli-r3-analysis':summary['results']})
    (dist/'workspace-evidence.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    workspace=(PILOT.parent/'steering_lab/workspace.html').read_text().replace('ANLI R3 · 难度校准与全方向扫描','ANLI R3 · 先分析，再给答案').replace('独立 dev 校准后，70/30 个主实验样本逐方向扫描','100 题 · 54 配置 · 最后一行答案评分').replace('Navigator 先用独立的 ANLI R3 dev 30 题校准难度，再对官方 train/test 抽取的 70/30 题逐方向扫描','Navigator 对同一批70/30题生成简短分析与最终答案').replace('assets/navigator9b/anli-r3/RESULTS.md','assets/navigator9b/anli-r3-analysis/RESULTS.md')
    version=hashlib.sha256(page.encode()+workspace.encode()+(brand_root/'theme-blue.css').read_bytes()+(brand_root/'theme.js').read_bytes()+(brand_root/'assets/brand/iota-logo.png').read_bytes()).hexdigest()[:12]
    (dist/'index.html').write_text(workspace.replace('__WORKSPACE_DATA__',json.dumps(metadata,ensure_ascii=False).replace('<','\\u003c')).replace('__UI_VERSION__',version))
    sources=[ROOT/name for name in ['current_navigator.json','classification_generation.py','answer_format.py','audit_generation.py','build_generation_view.py','validate.py']]
    (dist/'workspace-manifest.json').write_text(json.dumps(dict(sections=['capabilities','scenarios'],default_section='scenarios',ui_version=version,
        model=model['model'],revision=model['revision'],same_origin_api=True,preserves_existing_browser_storage=True,shared_original_directions_bit_exact=True,
        geometry_sha256=binding['geometry_sha256'],output_protocol='analysis_then_答案',source_sha256={str(p):sha(p) for p in sources}),ensure_ascii=False,indent=2)+'\n')
    report=['# ANLI R3：分析后作答与常见语义方向诊断','','同一批已检查的100题，54配置，共5400次完整生成。最终行格式：答案：A/B/C中的一个字母；先输出1–2句简短分析。此报告是事后探索，不是全新测试成绩。','','| 条件 | 全部100题 | train70 | test30 | 修正baseline错误 | 改坏baseline正确 | 格式无效 |','|---|---:|---:|---:|---:|---:|---:|']
    for j in [0,49,50,51,52,53]:
        result=summary['results'][str(j)];a=result['splits']['all'];report.append(f"| {labels[j]} | {a['correct']} | {result['splits']['train']['correct']} | {result['splits']['test']['correct']} | {a['fixed']} | {a['damaged']} | {a['invalid_format']} |")
    report+=['','原单字母协议baseline为65/100，新格式应独立比较，不能宣称格式变化一定提升准确率。','',f"事后最佳固定单方向：{labels[summary['best_fixed_single']]}；训练选择：{labels[summary['train_selected_single']]}。40个单方向的标签可见覆盖：{summary['coverage']['all_singles']['correct']}/100。",'',
        '原35错题的定性候选分组：8题为常见语义/常识应用较明确候选，6题为混合语用边界，14题为日期数量主体关系等其他问题，7题为标签/原理由张力；这些分组是诊断假设，不是内部机制证明。','','```json',json.dumps(diagnosis,ensure_ascii=False,indent=2),'```','']
    (out/'RESULTS.md').write_text('\n'.join(report));shutil.copy2(out/'RESULTS.md',assets/'RESULTS.md')
    print('BUILT generation-based Navigator and four-field samples:',dist)

if __name__=='__main__':build()

"""Display one original baseline sample with its exact messages and answer."""
import html
import json
from anli_diagnose import OUT

def render(summary,cases):
    esc=lambda value:html.escape(str(value))
    labels=['蕴含','信息不足','矛盾']
    cards=[];options=[]
    for case in cases:
        number=case['index']+1
        options.append(f'<option value="{number}">样本 {number} · {case["split"]}</option>')
        messages=case['prediction_prompt']
        reason=case.get('official_reason','').strip()
        solution=('<h3>标准答案解析（数据集原文）</h3><p class="official-reason original">'+esc(reason)+'</p>') if reason else ''
        cards.append(f'<article class="sample" id="sample-{number}" data-number="{number}" hidden>'
            f'<p class="sample-id">{esc(case["id"])}</p>'
            '<section><h2>1. 题目</h2><h3>前提</h3><p class="original">'+esc(case['context'])+'</p>'
            '<h3>假设</h3><p class="original">'+esc(case['hypothesis'])+'</p></section>'
            '<section><h2>2. 标准答案</h2><p class="answer">'+'ABC'[case['gold']]+' · '+labels[case['gold']]+'</p>'+solution+'</section>'
            '<section><h2>3. 预测 prompt</h2><h3>System</h3><pre>'+esc(messages[0]['content'])+'</pre>'
            '<h3>User</h3><pre>'+esc(messages[1]['content'])+'</pre></section>'
            '<section><h2>4. 模型答案</h2><p class="answer">'+'ABC'[case['baseline_prediction']]+' · '+labels[case['baseline_prediction']]+'</p>'
            '<h3>模型 output（完整原文）</h3><pre class="raw-answer">'+esc(case['raw_prediction'])+'</pre>'
            '<p class="muted">本轮 prompt 要求只输出一个字母，原始 output 没有解析。</p></section></article>')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ANLI · 逐题查看</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f5f3ed;color:#233a34;font:16px/1.75 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:28px 20px 60px}a{color:#315f50}h1{font-size:28px;margin-bottom:6px}h2{font-size:20px;margin:0 0 18px}h3{font-size:14px;color:#687b71;margin:14px 0 5px}p{margin:8px 0;overflow-wrap:anywhere}.muted,.sample-id{color:#687b71;font-size:13px}.sample-id{margin:20px 0}section{background:#fffefa;border:1px solid #dce2d9;border-radius:10px;padding:22px;margin:16px 0}.original,pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}pre{font:14px/1.8 ui-monospace,monospace;background:#f1f3ed;padding:14px;border-radius:6px;margin:6px 0}select,button{border:1px solid #cad5cb;background:#fffefa;padding:10px;border-radius:5px;color:inherit;font:inherit}nav{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px;align-items:center}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.answer{font-size:23px;font-weight:600}.raw-answer{font-size:23px}.answer-meaning{color:#687b71}[hidden]{display:none!important}@media(max-width:500px){main{padding:18px 14px}section{padding:16px}h1{font-size:25px}pre{font-size:13px}}
    </style></head><body><main><a href="../../../capabilities.html">← 返回 Navigator</a><h1>ANLI · 逐题查看</h1><p class="muted">35 道原始 baseline 错题。每题展示题目、标准答案、实际预测 prompt 和模型答案。</p>
    <nav aria-label="选择样本"><button id="previous">上一题</button><select id="sampleSelect" aria-label="样本">__OPTIONS__</select><button id="next">下一题</button><span id="position" class="muted"></span></nav>__CARDS__
    <script>const select=document.querySelector('#sampleSelect'),samples=[...document.querySelectorAll('.sample')],previous=document.querySelector('#previous'),next=document.querySelector('#next');function show(value){const index=samples.findIndex(el=>el.dataset.number===value),i=index<0?0:index;samples.forEach((el,j)=>el.hidden=j!==i);select.value=samples[i].dataset.number;previous.disabled=i===0;next.disabled=i===samples.length-1;document.querySelector('#position').textContent=(i+1)+' / '+samples.length;history.replaceState(null,'','#sample-'+select.value)}select.addEventListener('change',()=>show(select.value));previous.onclick=()=>show(samples[select.selectedIndex-1].dataset.number);next.onclick=()=>show(samples[select.selectedIndex+1].dataset.number);addEventListener('hashchange',()=>show(location.hash.replace('#sample-','')));show(location.hash.replace('#sample-',''));</script></main></body></html>'''
    (OUT/'diagnosis.html').write_text(page.replace('__OPTIONS__',''.join(options)).replace('__CARDS__',''.join(cards)))

"""Calibrated NLI experiment view with historical benchmark pages preserved."""
import json
import shutil
import hashlib
from pathlib import Path
from build_sst2_navigator import load_data

ROOT=Path(__file__).resolve().parent

def build(run,dist):
    run=Path(run);dist=Path(dist);data=load_data(run);m=data['manifest'];key=m['artifact_key']
    head=(ROOT/'sentiment_navigator_page.html').read_text().split('</head>',1)[0]+'</head>'
    panel=(ROOT/'sentiment_sweep_panel.html').read_text().replace('__SENTIMENT_DIAGNOSTICS__','')
    panel=panel.replace('assets/navigator9b/sweep/','assets/navigator9b/'+key+'/')
    panel=panel.replace('测试 30 题','评估 30 题（官方 test）').replace('原有测量复用，缺失格子补齐。','所有配置均在本轮重新测量。')
    panel=panel.replace('原来的训练选择实验保留在下方。','训练集选择的固定方向另外列出；其选择在测试输入测量前冻结。')
    panel=panel.replace('格内 A–E','格内 A/B/C').replace('不代表正面情感标签','不代表某个答案类别')
    panel=panel.replace('搜索评论内容或 ID','搜索前提、假设或 ID')
    page=(ROOT/'classification_navigator_page.html').read_text().replace('__BASE_HEAD__',head).replace('__SWEEP_PANEL__',panel)
    diagnostic=ROOT/'runs/classification9b-anli-r3-diagnostics-v1'
    diagnostic_link=''
    if m['short_name']=='ANLI R3' and (diagnostic/'audit.json').exists():
        audit=json.loads((diagnostic/'audit.json').read_text())
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        if not audit['passed'] or audit['source_items_sha256']!=digest(run/'feasible-space.jsonl') or audit['summary_sha256']!=digest(diagnostic/'summary.json') or audit['cases_sha256']!=digest(diagnostic/'error-cases.jsonl'):
            raise ValueError('ANLI diagnostics do not match the current audited measurements')
        target=dist/'assets/navigator9b/anli-r3-diagnostics';target.mkdir(exist_ok=True,parents=True)
        for name in ['diagnosis.html','RESULTS.md','summary.json','error-cases.jsonl','spec.json','audit.json','model.json','verifier-source.json',
                     'original_recheck.json','cyclic_mapping_1.json','cyclic_mapping_2.json','explicit_nli_guidance.json','explanation_then_answer.json']:
            shutil.copy2(diagnostic/name,target/name)
        diagnostic_link='<div class="note"><strong>逐题查看 35 道 baseline 错题。</strong>题目、标准答案、实际预测 prompt、模型答案。<a href="assets/navigator9b/anli-r3-diagnostics/diagnosis.html" target="_top">打开错题诊断 ↗</a></div>'
    page=page.replace('__CLASSIFICATION_DIAGNOSTICS__',diagnostic_link)
    page=page.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).replace('__SWEEP_SCRIPT__',(ROOT/'sentiment_sweep_panel.js').read_text())
    assets=dist/'assets/navigator9b'/key;assets.mkdir(parents=True,exist_ok=True)
    for name in ['dataset-manifest.json','geometry-binding.json','search-geometry.npz','sweep-spec.json','sweep-summary.json','feasible-space.jsonl','numerical-audit.json',
                 'audit.json','frozen-selection.json','model.json','train-generation.jsonl','test-generation.jsonl','calibration-result.json','calibration-generations.jsonl','RESULTS.md']:
        if (run/name).exists():shutil.copy2(run/name,assets/name)
    (assets/'view-data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    old=dist/'capabilities.html';archive=dist/'sst2-archive.html'
    if not archive.exists() and old.exists() and 'glue:sst2:' in old.read_text():
        banner='<div class="note">历史归档：GLUE / SST-2。当前主实验请查看 <a href="capabilities.html">Navigator</a>。</div>'
        archive.write_text(old.read_text().replace('<main>','<main>'+banner,1))
    prior=dist/'sst5-archive.html'
    if prior.exists():
        text=prior.read_text().replace('当前主实验为 <a href="capabilities.html">GLUE / SST-2</a>','当前主实验请查看 <a href="capabilities.html">Navigator</a>')
        if text!=prior.read_text():prior.write_text(text)
    return page,data

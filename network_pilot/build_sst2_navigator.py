"""Shared verified classification data and the archived binary benchmark view."""
import json
import shutil
from pathlib import Path
import numpy as np
from sentiment_prepare import sha

ROOT=Path(__file__).resolve().parent

def load_data(run):
    run=Path(run);manifest=json.loads((run/'dataset-manifest.json').read_text());binding=json.loads((run/'geometry-binding.json').read_text())
    if sha(run/'search-geometry.npz')!=binding['geometry_sha256']:raise ValueError('Wrong geometry binding')
    for name,expected in manifest['files'].items():
        if sha(run/name)!=expected:raise ValueError('Changed classification input '+name)
    atlas=json.loads((run/'search-atlas.json').read_text());directions=np.load(run/'search-geometry.npz')['directions'];u=directions/np.linalg.norm(directions,axis=1,keepdims=True)
    summary=json.loads((run/'sweep-summary.json').read_text()) if (run/'sweep-summary.json').exists() else None
    if summary:
        audit=json.loads((run/'audit.json').read_text())
        if not audit['passed'] or audit['summary_sha256']!=sha(run/'sweep-summary.json'):raise ValueError('Classification result audit missing or stale')
        if sha(run/'feasible-space.jsonl')!=summary['items_sha256']:raise ValueError('Classification item evidence changed')
    data=dict(complete=bool(summary and summary['complete']),summary=None,sweep=summary,sweep_rows=[],diagnostics=None,rows=[],
        manifest=manifest,binding=binding,status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else dict(phase='queued'),
        geometry=dict(nodes=[dict(id=c['id'],name=c['name'],xyz=c['direction'],certified=c.get('certified')) for c in atlas['concepts']],gram=(u@u.T).tolist()))
    if summary:data['sweep_rows']=[json.loads(line) for line in (run/'feasible-space.jsonl').read_text().splitlines()]
    return data

def build(run,dist):
    run=Path(run);dist=Path(dist);data=load_data(run)
    if data['manifest']['short_name']!='SST-2':raise ValueError('Wrong binary benchmark')
    head=(ROOT/'sentiment_navigator_page.html').read_text().split('</head>',1)[0]+'</head>'
    panel=(ROOT/'sentiment_sweep_panel.html').read_text().replace('__SENTIMENT_DIAGNOSTICS__','')
    panel=panel.replace('assets/navigator9b/sweep/','assets/navigator9b/sst2/')
    panel=panel.replace('测试 30 题','评估 30 题（官方 validation）').replace('原有测量复用，缺失格子补齐。','本轮全部配置重新测量。')
    panel=panel.replace('格内 A–E','格内 A/B')
    panel=panel.replace('原来的训练选择实验保留在下方。','仅在训练集选择的固定方向另外列出；评估集取自官方 validation。')
    page=(ROOT/'sst2_navigator_page.html').read_text().replace('__BASE_HEAD__',head).replace('__SWEEP_PANEL__',panel)
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page=page.replace('__DATA__',encoded).replace('__SWEEP_SCRIPT__',(ROOT/'sentiment_sweep_panel.js').read_text())
    destination=dist/'assets/navigator9b/sst2';destination.mkdir(parents=True,exist_ok=True)
    for name in ['dataset-manifest.json','geometry-binding.json','search-geometry.npz','sweep-spec.json','sweep-summary.json','feasible-space.jsonl',
                 'numerical-audit.json','audit.json','frozen-selection.json','model.json','train-generation.jsonl','test-generation.jsonl','RESULTS.md']:
        if (run/name).exists():shutil.copy2(run/name,destination/name)
    (destination/'view-data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    # Preserve the full prior page and all its existing evidence URLs.
    old=dist/'capabilities.html';archive=dist/'sst5-archive.html'
    if not archive.exists() and old.exists():
        text=old.read_text()
        if 'sst5:' in text:
            banner='<div class="note">历史归档：SST-5 与诊断。当前主实验为 <a href="capabilities.html">GLUE / SST-2</a>。</div>'
            archive.write_text(text.replace('<main>','<main>'+banner,1))
    return page,data

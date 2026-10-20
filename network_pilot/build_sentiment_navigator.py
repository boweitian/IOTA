"""Current Navigator is a 9B-only view of frozen geometry and train-selected agent teams."""
import json
import shutil
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def current_run():
    config=json.loads((ROOT/'current_navigator.json').read_text())
    if config['model']!='Qwen/Qwen3.5-9B':raise ValueError('Current Navigator must use Qwen3.5-9B')
    run=ROOT/config['run']
    binding=json.loads((run/'geometry-binding.json').read_text())
    for key in ['model','revision','layer_index']:
        if config[key]!=binding['model'][key]:raise ValueError('Current Navigator configuration binding mismatch: '+key)
    return run

def payload(run):
    run=Path(run)
    manifest=json.loads((run/'dataset-manifest.json').read_text())
    binding=json.loads((run/'geometry-binding.json').read_text())
    if hashlib.sha256((run/'search-geometry.npz').read_bytes()).hexdigest()!=binding['geometry_sha256']:
        raise ValueError('Frozen Navigator geometry changed')
    atlas=json.loads((run/'search-atlas.json').read_text())
    directions=np.load(run/'search-geometry.npz')['directions'];unit=directions/np.linalg.norm(directions,axis=1,keepdims=True)
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else dict(phase='queued')
    summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
    data=dict(complete=bool(summary and summary.get('complete')),manifest=manifest,binding=binding,status=status,summary=summary,
        geometry=dict(nodes=[dict(id=c['id'],name=c['name'],xyz=c['direction'],certified=c.get('certified')) for c in atlas['concepts']],gram=(unit@unit.T).tolist()),
        rows=[])
    if summary:
        data['rows']=[dict(json.loads(line),split='test') for line in (run/'test-results.jsonl').read_text().splitlines() if line.strip()]
        features=np.load(run/'test-cache/agent-00.npz')['features']
        measured={j:np.load(run/'test-cache'/f'agent-{j:02d}.npz')['probabilities'] for j in summary['test_agents_measured']}
        for i,row in enumerate(data['rows']):
            row['agents']={str(j):dict(prediction=int(values[i].argmax()),probabilities=values[i].tolist()) for j,values in measured.items()}
            row['routing']={}
            for name,policy in summary['policies'].items():
                if 'router' not in policy:continue
                router=policy['router'];z=np.r_[1.,(features[i]-np.array(router['mean']))/np.array(router['std'])]
                utility=z@np.array(router['coef']);weights=np.exp(utility-utility.max());weights/=weights.sum()
                row['routing'][name]=dict(members=policy['members'],weights=weights.tolist())
    sweep=ROOT/'runs/sentiment9b-sst5-direction-sweep-v1'
    sweep_path=sweep/'sweep-summary.json'
    data['sweep']=None;data['sweep_rows']=[]
    if sweep_path.exists():
        scan=json.loads(sweep_path.read_text())
        if scan['geometry_sha256']!=binding['geometry_sha256']:raise ValueError('Sweep uses a different geometry')
        if hashlib.sha256((sweep/'feasible-space.jsonl').read_bytes()).hexdigest()!=scan['items_sha256']:
            raise ValueError('Sweep item evidence changed')
        data['sweep']=scan
        data['sweep_rows']=[json.loads(line) for line in (sweep/'feasible-space.jsonl').read_text().splitlines() if line.strip()]
    diagnostics=ROOT/'runs/sentiment9b-sst5-diagnostics-v1'
    data['diagnostics']=None
    if (diagnostics/'summary.json').exists():
        diagnostic=json.loads((diagnostics/'summary.json').read_text())
        for name,expected in diagnostic['files'].items():
            if hashlib.sha256((diagnostics/name).read_bytes()).hexdigest()!=expected:raise ValueError('Diagnostic evidence changed: '+name)
        spec=json.loads((diagnostics/'spec.json').read_text())
        if spec['geometry_sha256']!=binding['geometry_sha256']:raise ValueError('Diagnostic geometry changed')
        audit=json.loads((diagnostics/'audit.json').read_text())
        if not audit['passed'] or audit['summary_sha256']!=hashlib.sha256((diagnostics/'summary.json').read_bytes()).hexdigest():
            raise ValueError('Diagnostic audit does not bind the current summary')
        data['diagnostics']=dict(diagnostic,coverage=audit['coverage'])
    return data

def build(run,dist):
    run=Path(run);dist=Path(dist)
    if json.loads((run/'dataset-manifest.json').read_text()).get('kind')=='classification_sweep':
        from build_classification_navigator import build as build_classification
        return build_classification(run,dist)
    if json.loads((run/'dataset-manifest.json').read_text()).get('short_name')=='SST-2':
        from build_sst2_navigator import build as build_binary
        return build_binary(run,dist)
    data=payload(run)
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page=(ROOT/'sentiment_navigator_page.html').read_text().replace('__DIRECTION_SWEEP__',(ROOT/'sentiment_sweep_panel.html').read_text())
    page=page.replace('__SENTIMENT_DIAGNOSTICS__',(ROOT/'sentiment_diagnostics_panel.html').read_text())
    scripts=(ROOT/'sentiment_sweep_panel.js').read_text()+'\n'+(ROOT/'sentiment_diagnostics_panel.js').read_text()
    page=page.replace('__NAVIGATOR_DATA__',encoded).replace('</body>','<script>'+scripts+'</script></body>')
    assets=dist/'assets/navigator9b';assets.mkdir(parents=True,exist_ok=True)
    for name in ['dataset-manifest.json','geometry-binding.json','search-geometry.npz','experiment-spec.json','frozen-policies.json','train.jsonl',
                 'numerical-audit.json','summary.json','test-predictions.jsonl','test-results.jsonl','RESULTS.md']:
        source=run/name
        if source.exists():shutil.copy2(source,assets/name)
    (assets/'view-data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    if data['sweep']:
        destination=assets/'sweep';destination.mkdir(exist_ok=True)
        for name in ['sweep-spec.json','sweep-summary.json','feasible-space.jsonl','RESULTS.md','numerical-audit.json','audit.json']:
            source=ROOT/'runs/sentiment9b-sst5-direction-sweep-v1'/name
            if source.exists():shutil.copy2(source,destination/name)
    if data['diagnostics']:
        destination=assets/'diagnostics';destination.mkdir(exist_ok=True)
        diagnostics=ROOT/'runs/sentiment9b-sst5-diagnostics-v1'
        for source in diagnostics.glob('*.json'):shutil.copy2(source,destination/source.name)
        if (diagnostics/'RESULTS.md').exists():shutil.copy2(diagnostics/'RESULTS.md',destination/'RESULTS.md')
    return page,data

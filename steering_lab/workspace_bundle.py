"""Unify immutable benchmark assets and the existing live character editor."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT.parent/'network_pilot/runs/expanded-assets-v2'

def ui_version():
    sources=[ROOT/name for name in ['workspace.html','studio.html','agent.js','direction-manager.js','agent.css','blocks.js','blocks.css','regularization.js','agent-avatar.svg','theme-blue.css','theme.js','assets/brand/iota-logo.png']]
    sources += [ROOT.parent/'network_pilot'/name for name in ['current_navigator.json','sentiment_navigator_page.html','sentiment_sweep_panel.html','sentiment_sweep_panel.js','sentiment_diagnostics_panel.html','sentiment_diagnostics_panel.js','sst2_navigator_page.html','build_sst2_navigator.py','classification_navigator_page.html','build_classification_navigator.py']]
    return hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:12]

def embedded(page,section):
    """Keep standalone pages usable; embed within one same-origin workspace."""
    bridge='''<script>
if(new URLSearchParams(location.search).get('embedded')==='1'){
 document.documentElement.classList.add('workspace-embedded');
 addEventListener('DOMContentLoaded',()=>{
  const main=document.querySelector('main');let tick=0;
  function notify(){cancelAnimationFrame(tick);tick=requestAnimationFrame(()=>{
   const rect=main.getBoundingClientRect();if(rect.height<=0)return;
   const height=Math.ceil(rect.bottom+scrollY+24);
   parent.postMessage({type:'iota:module-height',section:'SECTION_ID',height},location.protocol==='file:'?'*':location.origin);
  })}
  new ResizeObserver(notify).observe(main);addEventListener('resize',notify);
  addEventListener('message',e=>{if(e.source===parent&&e.data?.type==='iota:module-visible'){dispatchEvent(new Event('resize'));notify()}});
  notify();
 });
}
</script><style>
html.workspace-embedded body>header{display:none}
html.workspace-embedded body{background:transparent}
html.workspace-embedded main{padding-top:18px}
html.workspace-embedded .intro{display:none}
html.workspace-embedded .runtimebar:not(#latestBar){display:none}
html.workspace-embedded footer{display:none}
:root{--bg:#f5f3ed;--ink:#233a34;--muted:#6c7a72;--line:#dddeda;--blue:#315f50;--orange:#bf7148}
</style>'''.replace('SECTION_ID',section)
    theme_hash=hashlib.sha256((ROOT/'theme-blue.css').read_bytes()+(ROOT/'theme.js').read_bytes()).hexdigest()[:12]
    theme='<script src="/theme.js?v='+theme_hash+'"></script><link rel="stylesheet" href="/theme-blue.css?v='+theme_hash+'">'
    return page.replace('</head>',bridge+theme+'</head>',1)

def build_workspace(dist):
    """Current pages share Search's exact 9B geometry; historical scores are never relabelled."""
    pilot=ROOT.parent/'network_pilot'
    sys.path.insert(0,str(pilot))
    from build_sentiment_navigator import current_run,build
    run=current_run();dist=Path(dist)
    shutil.copy2(ROOT/'theme-blue.css',dist/'theme-blue.css')
    shutil.copy2(ROOT/'theme.js',dist/'theme.js')
    shutil.copytree(ROOT/'assets/brand',dist/'assets/brand',dirs_exist_ok=True)
    if json.loads((run/'dataset-manifest.json').read_text()).get('kind')=='classification_generation':
        from build_generation_view import build as build_generated
        build_generated(run,dist)
        (dist/'scenarios.html').write_text(embedded((dist/'scenarios.html').read_text(),'scenarios'))
        for name in ['agent.js','direction-manager.js','agent.css','blocks.js','blocks.css','regularization.js','agent-avatar.svg']:
            shutil.copy2(ROOT/name,dist/name)
        print('WORKSPACE BUILT: Search + Navigator; full generated answers; common Qwen3.5-9B geometry verified',flush=True)
        return
    studio=json.loads((dist/'studio-evidence.json').read_text())
    page,data=build(run,dist)
    model=data['binding']['model'];other=studio['atlas']['model']
    for key in ['model','revision','layer_index','hidden_size']:
        if model[key]!=other[key]:raise ValueError('Navigator/Search binding mismatch: '+key)
    if model['model']!='Qwen/Qwen3.5-9B':raise ValueError('Navigator must use Qwen3.5-9B')
    directions=np.load(run/'search-geometry.npz')['directions']
    if not np.array_equal(directions,np.array([c['native_direction'] for c in studio['atlas']['concepts']])):
        raise ValueError('Navigator and Search do not use the same frozen directions')
    for name in ['capabilities.html','router-analysis.html']:(dist/name).write_text(embedded(page,'capabilities'))
    scenarios=(dist/'scenarios.html').read_text()
    (dist/'scenarios.html').write_text(embedded(scenarios,'scenarios'))
    for name in ['agent.js','direction-manager.js','agent.css','blocks.js','blocks.css','regularization.js','agent-avatar.svg']:
        shutil.copy2(ROOT/name,dist/name)
    summary=data['summary'];complete=data['complete']
    metadata=dict(model=model['model'].split('/')[-1],scenario_model=other['model'].split('/')[-1],
        model_revision=model['revision'],scenario_layer_index=model['layer_index'],nodes=9,scenario_nodes=9,
        samples=100,train_samples=70,test_samples=30,settings=len(summary['actions']) if summary else 49,
        shared_original_directions_bit_exact=True,geometry_sha256=data['binding']['geometry_sha256'],
        studio_protocol_sha256=studio['protocol_sha256'],navigator_complete=complete,
        benchmark=data['manifest'].get('short_name','SST-5'),selected_policy=summary['selected'] if summary else None,
        suites={'sst5':summary['test_metrics'] if summary else {}},
        evidence_records=70*49+30*len(summary['test_agents_measured']) if summary else 0)
    if data.get('sweep'):
        metadata.update(direction_sweep_complete=True,sweep_measurements=4900,
                        sweep_metrics=data['sweep']['splits']['all'],
                        evidence_records=4900)
        if data['manifest'].get('short_name')=='SST-2' or data['manifest'].get('kind')=='classification_sweep':
            metadata['suites']={data['manifest'].get('artifact_key','sst2'):data['sweep']['splits']}
            metadata['evaluation_source_split']=data['manifest']['evaluation_source_split']
    metadata['sentiment_diagnostics_complete']=bool(data.get('diagnostics'))
    metadata['calibration_samples']=data['manifest'].get('calibration_count',0)
    (dist/'workspace-evidence.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    workspace=(ROOT/'workspace.html').read_text().replace('__WORKSPACE_DATA__',json.dumps(metadata,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    (dist/'index.html').write_text(workspace.replace('__UI_VERSION__',ui_version()))
    for name in ['RESEARCH.md','PILOT.md']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,dist/name)
    sources=[ROOT/'studio.html',ROOT/'workspace.html',ROOT/'agent.js',ROOT/'direction-manager.js',ROOT/'blocks.js',ROOT/'regularization.js',
        pilot/'current_navigator.json',pilot/'sentiment_navigator_page.html',pilot/'build_sentiment_navigator.py',
        pilot/'sentiment_sweep_panel.html',pilot/'sentiment_sweep_panel.js',
        pilot/'sentiment_diagnostics_panel.html',pilot/'sentiment_diagnostics_panel.js',
        pilot/'sst2_navigator_page.html',pilot/'build_sst2_navigator.py',
        pilot/'classification_navigator_page.html',pilot/'build_classification_navigator.py']
    manifest=dict(sections=['capabilities','scenarios'],default_section='scenarios',ui_version=ui_version(),
        model=model['model'],revision=model['revision'],same_origin_api=True,preserves_existing_browser_storage=True,
        shared_original_directions_bit_exact=True,geometry_sha256=metadata['geometry_sha256'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (dist/'workspace-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('WORKSPACE BUILT: Search + Navigator, common Qwen3.5-9B geometry verified;',metadata['benchmark'], 'complete' if complete else 'running',flush=True)

def build_workspace_legacy_0p8b(dist):
    summary=json.loads((ASSETS/'comparison.json').read_text());audit=json.loads((ASSETS/'evidence_audit.json').read_text())
    if not audit['passed'] or not summary['complete']:raise ValueError('Only complete audited assets may enter the workspace')
    geometry=json.loads((ASSETS/'geometry/geometry.json').read_text());studio=json.loads((dist/'studio-evidence.json').read_text())
    # Panels on the same model must share the original nine vectors bit for bit; panels on different
    # models are labelled as separate geometries and no capability result is carried over.
    same_model=geometry['model']['model']==studio['atlas']['model']['model']
    if same_model:
        old=np.load(ROOT/'runs/job-2117172/geometry.npz');new=np.load(ASSETS/'geometry/geometry.npz')
        if not np.array_equal(old['directions'],new['directions'][:9]):raise ValueError('The two panels use incompatible steering geometries')
        for key in ['model','revision','layer_index','hidden_size']:
            mapped='model_revision' if key=='revision' else key
            if mapped in geometry['model'] and geometry['model'][mapped]!=studio['atlas']['model'][key]:
                raise ValueError('The two panels use incompatible model bindings: '+key)
    # Publish copies; the original experiment and portable release stay immutable.
    capability=(ASSETS/'view/index.html').read_text()
    capability=capability.replace('<title>IOTA · Steering 方向图谱</title>','<title>IOTA Navigator</title>')
    capability=capability.replace('<h1>Steering 方向图谱</h1>','<h1>IOTA Navigator</h1>')
    capability=capability.replace('<span>STEERING ATLAS / 02</span>','<span>NAVIGATOR</span>')
    (dist/'capabilities.html').write_text(embedded(capability,'capabilities'))
    from router_atlas_mount import mount_router_atlas
    mount_router_atlas(dist)
    scenarios=(dist/'scenarios.html').read_text()
    scenarios=scenarios.replace('<title>IOTA · Character Studio</title>','<title>IOTA Search</title>')
    scenarios=scenarios.replace('<a href="benchmark.html">实验室与对照结果 ↗</a>',
        '<a href="index.html#capabilities" target="_top">IOTA Navigator ↗</a>')
    scenarios=scenarios.replace('<span>CHARACTER STUDIO</span>','<span>SEARCH</span>')
    (dist/'scenarios.html').write_text(embedded(scenarios,'scenarios'))
    for name in ['agent.js','direction-manager.js','agent.css','blocks.js','blocks.css','regularization.js','agent-avatar.svg']:
        shutil.copy2(ROOT/name,dist/name)
    shutil.copy2(ROOT/'runs/identity-pilot-20261017-v1/RESULTS.md',dist/'IDENTITY_EXPERIMENT.md')
    evidence_dir=dist/'assets/steering';evidence_dir.mkdir(parents=True,exist_ok=True)
    for source,name in [(ASSETS/'RESULTS.md','RESULTS.md'),(ASSETS/'comparison.json','comparison.json'),
                         (ASSETS/'evidence_audit.json','evidence_audit.json'),
                         (ASSETS/'view/geometry_map.png','geometry_map.png'),
                         (ASSETS/'IOTA-expanded-steering-assets-v2.zip','IOTA-expanded-steering-assets-v2.zip')]:
        target=evidence_dir/name
        if not target.exists() or target.stat().st_size!=source.stat().st_size or target.stat().st_mtime<source.stat().st_mtime:
            shutil.copy2(source,target)
    metadata=dict(model=geometry['model']['model'].split('/')[-1],scenario_model=studio['atlas']['model']['model'].split('/')[-1],
        scenario_layer_index=studio['atlas']['model']['layer_index'],nodes=27,scenario_nodes=9,samples=175,settings=109,
        new_measurements=12600,evidence_records=21930,shared_original_directions_bit_exact=same_model,
        geometry_sha256=geometry['geometry_sha256'],studio_protocol_sha256=studio['protocol_sha256'],
        suites={s:{k:v for k,v in p.items() if k!='fixed'} for s,p in summary['suites'].items()})
    (dist/'workspace-evidence.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    page=(ROOT/'workspace.html').read_text().replace('__WORKSPACE_DATA__',json.dumps(metadata,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    (dist/'index.html').write_text(page.replace('__UI_VERSION__',ui_version()))
    for name in ['RESEARCH.md','PILOT.md']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,dist/name)
    manifest=dict(sections=['capabilities','scenarios'],default_section='scenarios',ui_version=ui_version(),same_origin_api=True,
        preserves_existing_browser_storage=True,shared_original_directions_bit_exact=same_model,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ASSETS/'view/index.html',ROOT/'studio.html',ROOT/'workspace.html',ROOT/'agent.js',ROOT/'direction-manager.js',ROOT/'agent.css',ROOT/'blocks.js',ROOT/'blocks.css',ROOT/'regularization.js',ROOT/'agent-avatar.svg']})
    (dist/'workspace-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('WORKSPACE BUILT: capabilities + scenarios,','common 9-direction geometry verified' if same_model else 'separate models: '+metadata['model']+' / '+metadata['scenario_model'],flush=True)

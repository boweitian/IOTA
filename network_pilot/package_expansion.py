"""Portable expanded asset release, including source files behind reused evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from expanded_assets import DEFAULT,PREVIOUS,ROOT
from engine import atomic_json

def main():
    run=DEFAULT;bundle=run/'asset_bundle'
    assert json.loads((run/'complete.json').read_text())['audit_passed']
    browser=json.loads((run/'view/browser_check.json').read_text());assert browser['passed'] and browser['complete']
    referenced=set()
    for p in run.glob('*/shard-*/records.jsonl'):
        for r in map(json.loads,p.read_text().splitlines()):
            for field in ['source_file','previous_source_file']:
                if r.get(field):referenced.add(Path(r[field]))
    translations={}
    for source in sorted(referenced):
        relative=source.relative_to(ROOT/'runs');target=run/'previous_sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);translations[str(source)]=str(target.relative_to(run))
        for name in ['model.json','manifest.json','protocol.json','plan.json']:
            context=source.parent/name
            if context.exists():
                dest=target.parent/name;shutil.copy2(context,dest);translations[str(context)]=str(dest.relative_to(run))
    shutil.copytree(PREVIOUS/'source_snapshot',run/'previous_sources/sample-assets-v1/source_snapshot',dirs_exist_ok=True)
    for name in ['evidence_audit.json','complete.json','runtime_versions.json','RESULTS.md']:
        shutil.copy2(PREVIOUS/name,run/'previous_sources/sample-assets-v1'/name)
    atomic_json(run/'source_path_map.json',translations)
    # The frozen modules are imported unchanged; snapshot current release helpers as well.
    for path in list(ROOT.glob('*.py'))+list(ROOT.glob('*expansion*.sbatch'))+list(ROOT.glob('expanded_*.sbatch')):
        shutil.copy2(path,run/'source_snapshot'/path.name)
    shutil.copy2(ROOT/'expansion_page.html',run/'source_snapshot/expansion_page.html')
    shutil.copytree(ROOT/'vendor',run/'source_snapshot/vendor',dirs_exist_ok=True)
    shutil.copy2(ROOT/'README.md',run/'source_snapshot/README.md')
    files={str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in bundle.rglob('*') if p.is_file() and p.name!='index.json'}
    index=json.loads((bundle/'index.json').read_text());index['files']=files;atomic_json(bundle/'index.json',index)
    manifest={str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in run.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name not in ['release_manifest.json','archive.json']}
    atomic_json(run/'release_manifest.json',dict(schema_version=3,complete=True,files=manifest,
        total_files=len(manifest),previous_sources_portable=True,browser_verified=True,score_audit_passed=True))
    archive=run/'IOTA-expanded-steering-assets-v2.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(manifest):z.write(run/name,name)
        z.write(run/'release_manifest.json','release_manifest.json')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    atomic_json(run/'archive.json',dict(file=archive.name,bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(manifest)+1,integrity_passed=True))
    print('PACKAGED',archive,'bytes',archive.stat().st_size,'files',len(manifest)+1,flush=True)

if __name__=='__main__':main()

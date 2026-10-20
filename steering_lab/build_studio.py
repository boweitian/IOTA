"""Package the local editor with only measured character evidence."""
import argparse
import json
import shutil
from pathlib import Path
from engine import atomic_json
from studio import PRESETS, SCENES, profile_for, total_strength_cap_enabled, strength_regularization
from workspace_bundle import ui_version
from block_presets import faith_payload
from direction_library import available_directions

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,default=ROOT/"runs/qwen35-9b-studio-v2")
    p.add_argument("--studio-run",type=Path,default=ROOT/"studio_runs/character-9b-v2");args=p.parse_args()
    dist=ROOT/"dist";dist.mkdir(exist_ok=True)
    if not (dist/"benchmark.html").exists() and (dist/"index.html").exists():
        shutil.copyfile(dist/"index.html",dist/"benchmark.html")
    measured=args.studio_run/"studio_evidence.json"
    data=dict(strength_regularization=strength_regularization(),total_strength_cap_enabled=total_strength_cap_enabled(),atlas=json.loads((args.run/"atlas.json").read_text()),protocol_sha256=json.loads((args.run/"protocol.json").read_text())["sha256"],
        settings=json.loads((args.run/"protocol.json").read_text()).get("studio",dict(prompt_version="short-80-140",max_new_tokens=256)),
        presets=[dict(**p,profile=profile_for(p)) for p in PRESETS],scenes=SCENES,
        evidence=json.loads(measured.read_text()) if measured.exists() else None)
    try:data["faith_preset"]=faith_payload(data["atlas"],data["protocol_sha256"])
    except ValueError:data["faith_preset"]=None
    data['direction_library']=available_directions(data['atlas'],data['protocol_sha256'])
    atomic_json(dist/"studio-evidence.json",data)
    embedded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    (dist/"scenarios.html").write_text((ROOT/"studio.html").read_text().replace("__STUDIO_DATA__",embedded).replace("__UI_VERSION__",ui_version()))
    if (args.studio_run/"RESULTS.md").exists(): shutil.copyfile(args.studio_run/"RESULTS.md",dist/"STUDIO_RESULTS.md")
    from workspace_bundle import build_workspace
    build_workspace(dist)
    print("BUILT",dist/"index.html","scenario trials:",len(data["evidence"]["trials"]) if data["evidence"] else 0)

if __name__=="__main__": main()

#!/usr/bin/env python3
"""Bundle real measured data into a self-contained, offline interactive atlas."""
import argparse
import json
from pathlib import Path
import numpy as np

def build(data_dir,output):
    data_dir=Path(data_dir)
    atlas=json.loads((data_dir/"atlas.json").read_text())
    responses=json.loads((data_dir/"responses.json").read_text())
    if not responses["complete"] or atlas["model"]!=responses["model"]:
        raise ValueError("Measurements incomplete or from different models")
    expected={(a["id"],s["id"],v) for a in atlas["axes"] for s in atlas["scenarios"] for v in atlas["strengths"]}
    actual={(r["axis"],r["scenario"],r["strength"]) for r in responses["responses"]}
    if expected!=actual or len(responses["responses"])!=len(expected):
        raise ValueError("Missing or duplicated knob response")
    import hashlib
    if hashlib.sha256((data_dir/"directions.npz").read_bytes()).hexdigest()!=atlas["directions_sha256"]:
        raise ValueError("Direction artifact differs from measured data")
    with np.load(data_dir/"directions.npz") as artifact:
        directions={k:artifact[k].tolist() for k in artifact.files}
    page=Path(__file__).with_name("page.html").read_text()
    layout=json.loads(Path(__file__).with_name("layout.json").read_text())
    if set(layout["coordinates"]) != {a["id"] for a in atlas["axes"]}:
        raise ValueError("Display layout has a different concept set")
    data=json.dumps(dict(atlas=atlas,responses=responses,human_layout=layout,directions=directions),ensure_ascii=False,allow_nan=False)
    page=page.replace("__IOTA_DATA__",data.replace("<","\\u003c"))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(page)
    print(f"Built {output}: {len(atlas['points'])} measured points, {len(actual)} response settings")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path(__file__).parent/"data")
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"dist/index.html")
    a=p.parse_args();build(a.data,a.output)

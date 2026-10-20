"""Map → development exploration → freeze router → held-out execution → evidence."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from concepts import CONCEPTS, PROMPT_STEER, CREATIVE_PROMPT_STEER, probe_examples, task_prompt
from engine import Engine, atomic_json
from grading import grade
from router import geometry_features, text_features, calibrate, predict
from judge import load_judge, score_prompt, parse_score, pair_prompt, parse_winner

ROOT = Path(__file__).resolve().parent

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        f.flush()

def release(engine):
    import torch
    del engine.model
    gc.collect()
    torch.cuda.empty_cache()

def build_map(engine,out):
    probes = probe_examples()
    pooled,last = engine.measure([p["question"] for p in probes],[p["response"] for p in probes])
    np.savez_compressed(out / "probe_activations.npz", pooled=pooled,last=last)
    atomic_json(out / "probes.json",probes)
    by = {(p["concept"],p["context"],p["pole"]):pooled[i] for i,p in enumerate(probes)}
    raw = np.stack([np.mean([by[c["id"],i,"positive"]-by[c["id"],i,"negative"] for i in range(8)],axis=0) for c in CONCEPTS])
    unit = raw / np.linalg.norm(raw,axis=1,keepdims=True)
    # Match all action norms. Strength 1 is 20% of the median probe residual norm.
    scale = float(.20*np.median(np.linalg.norm(pooled,axis=1)))
    directions = unit * scale
    fitmask = np.array([p["split"]=="fit" for p in probes])
    center = pooled[fitmask].mean(0)
    _,s,vh = np.linalg.svd(pooled[fitmask]-center,full_matrices=False)
    xyz = (pooled-center) @ vh[:3].T
    radius = float(np.percentile(np.linalg.norm(xyz,axis=1),95))
    clouds = []
    for c in CONCEPTS:
        contrasts = np.stack([by[c["id"],i,"positive"]-by[c["id"],i,"negative"] for i in range(10)])
        ix = next(i for i,x in enumerate(CONCEPTS) if x["id"]==c["id"])
        validation = contrasts[8:].mean(0)
        stability = float(unit[ix] @ validation / max(np.linalg.norm(validation),1e-9))
        clouds.append({k:v for k,v in c.items() if k not in ["negative","positive"]} | dict(
            heldout_context_cosine=stability, raw_direction_norm=float(np.linalg.norm(raw[ix])),
            points=[dict(pole=p["pole"],context=p["context"],split=p["split"],xyz=(xyz[i]/radius).tolist())
                    for i,p in enumerate(probes) if p["concept"]==c["id"]],
            direction=(directions[ix] @ vh[:3].T / radius).tolist(), native_direction=directions[ix].tolist()))
    random = np.random.default_rng(20261011).normal(size=engine.dim)
    # A random vector orthogonal to the concept span, at the same intervention norm.
    q,_ = np.linalg.qr(unit.T)
    random -= q @ (q.T @ random)
    random = random / np.linalg.norm(random) * scale
    mix = unit[[5,6,7]].sum(0)
    mix = mix / np.linalg.norm(mix) * scale
    np.savez_compressed(out / "geometry.npz",raw=raw,unit=unit,directions=directions,
                        random=random,mix=mix,center=center,pca=vh[:3],radius=radius)
    atlas=dict(model=engine.meta,concepts=clouds,gram=(unit@unit.T).tolist(),
        variance_explained=float((s[:3]**2).sum()/(s**2).sum()),scale=scale,
        map_note="PCA of real probe activations; a lossy view of 1024-dimensional residuals. Human-authored labels and responses. Context cosine is a probe diagnostic, not proof of global semantic invariance.")
    atomic_json(out / "atlas.json",atlas)
    return dict(np.load(out / "geometry.npz"))

def actions(geometry):
    result=[dict(id="none",label="不干预",concept=None,strength=0,shift=np.zeros(geometry["directions"].shape[1]),extra="")]
    for i,c in enumerate(CONCEPTS):
        for strength in [.5,1.0]:
            result.append(dict(id=f"{c['id']}:{strength}",label=f"{c['name']} × {strength}",concept=c["id"],strength=strength,
                               shift=geometry["directions"][i]*strength,extra=""))
    result.append(dict(id="creative_mix:1.0",label="探索 + 意象 + 共情",concept="creative_mix",strength=1,
                       shift=geometry["mix"],extra=""))
    return result

def controls(geometry):
    return [dict(id="prompt",label="提示词增强",concept=None,strength=0,shift=np.zeros(geometry["directions"].shape[1]),extra=PROMPT_STEER),
            dict(id="random",label="同范数随机方向",concept=None,strength=1,shift=geometry["random"],extra="")]

def prompt(row):
    return row["question"] if row["task"] == "creative_writing" else task_prompt(row["question"])

def execute(engine,requests,action_bank,path,max_tokens,batch=16):
    existing={(r["id"],r["action"]):r for r in read_jsonl(path)}
    groups={a["id"]:[] for a in action_bank}
    for row,action in requests:
        if (row["id"],action) not in existing and row not in groups[action]:
            groups[action].append(row)
    for a in action_bank:
        rows=groups[a["id"]]
        for lo in range(0,len(rows),batch):
            chunk=rows[lo:lo+batch]
            extras=[CREATIVE_PROMPT_STEER if r["task"]=="creative_writing" and a["id"]=="prompt" else a["extra"] for r in chunk]
            generated=engine.generate([prompt(r) for r in chunk],[a["shift"]]*len(chunk),
                                     extras,max_new_tokens=max_tokens)
            for r,g,extra in zip(chunk,generated,extras):
                record=dict(id=r["id"],task=r["task"],split=r["split"],action=a["id"],extra_system_prompt=extra,**g)
                if r["task"] != "creative_writing":
                    record.update(grade(g["text"],r))
                append(path,record)
                existing[(r["id"],a["id"])]=record
            print("GENERATED",path.name,a["id"],lo+len(chunk),"/",len(rows),flush=True)
    return existing

def judge_records(engine,rows,records,path,batch=8):
    scored={(r["id"],r["action"]):r for r in read_jsonl(path)}
    lookup={r["id"]:r for r in rows}
    missing=[r for r in records.values() if r["id"] in lookup and (r["id"],r["action"]) not in scored]
    for lo in range(0,len(missing),batch):
        chunk=missing[lo:lo+batch]
        qs=[score_prompt(lookup[r["id"]],r["text"]) for r in chunk]
        generated=engine.generate(qs,[np.zeros(engine.dim)]*len(chunk),max_new_tokens=192)
        for r,g in zip(chunk,generated):
            score=parse_score(g["text"])
            # One explicit retry for malformed JSON; preserve both raw attempts.
            retry=None
            if score is None:
                retry=engine.generate([score_prompt(lookup[r["id"]],r["text"])+"\nReturn valid JSON only."],
                                     [np.zeros(engine.dim)],max_new_tokens=192)[0]
                score=parse_score(retry["text"])
            record=dict(id=r["id"],action=r["action"],judge_model=engine.meta,
                        raw=g["text"],retry=retry,parsed=score is not None,rubric=score)
            append(path,record)
            scored[(r["id"],r["action"])]=record
        print("JUDGED",path.name,lo+len(chunk),"/",len(missing),flush=True)
    return scored

def features(engine,rows,unit,path):
    if path.exists():
        return np.load(path)["features"]
    pooled,last=engine.measure([prompt(r) for r in rows])
    fs=geometry_features(pooled,last,unit)
    np.savez_compressed(path,features=fs,pooled=pooled,last=last,ids=np.asarray([r["id"] for r in rows]))
    return fs

def freeze_policy(rows,fs,outcomes,bank,path):
    values=np.asarray([[outcomes[r["id"],a["id"]] for a in bank] for r in rows],dtype=float)
    costs=[a["strength"] for a in bank]
    geometry=calibrate(fs,values,costs,[r["id"] for r in rows])
    lexical=calibrate(text_features([r["question"] for r in rows]),values,costs,[r["id"] for r in rows],standardize=False)
    fixed=int(np.argmax(values.mean(0)-.005*np.asarray(costs)))
    bundle=dict(geometry=geometry,lexical=lexical,fixed_index=fixed,fixed_action=bank[fixed]["id"],
        action_ids=[a["id"] for a in bank],dev_ids=[r["id"] for r in rows],
        dev_mean_utility=values.mean(0).tolist(), frozen_at=time.time(),
        calibration_note="Only development outcomes. No test examples, test labels, or test rewards enter calibration.")
    atomic_json(path,bundle)
    return bundle

def plan_test(rows,fs,policies,bank):
    gi,traces=predict(policies["geometry"],fs)
    li,_=predict(policies["lexical"],text_features([r["question"] for r in rows]))
    plan=[]
    for i,r in enumerate(rows):
        choices=dict(baseline="none",fixed=policies["fixed_action"],geometry=bank[int(gi[i])]["id"],
                     lexical=bank[int(li[i])]["id"],prompt="prompt",random="random")
        plan.append(dict(id=r["id"],choices=choices,trace=traces[i],features=fs[i].tolist()))
    return plan

def pair_audit(engine,rows,plan,records,path,batch=4):
    saved={r["id"]:r for r in read_jsonl(path)}
    by={r["id"]:r for r in rows}
    for lo in range(0,len(plan),batch):
        chunk=[p for p in plan[lo:lo+batch] if p["id"] not in saved]
        qs=[]
        for p in chunk:
            a=records[p["id"],"none"]["text"]
            b=records[p["id"],p["choices"]["geometry"]]["text"]
            qs.extend([pair_prompt(by[p["id"]],a,b),pair_prompt(by[p["id"]],b,a)])
        if not qs:
            continue
        generated=engine.generate(qs,[np.zeros(engine.dim)]*len(qs),max_new_tokens=160)
        for i,p in enumerate(chunk):
            raw=[g["text"] for g in generated[2*i:2*i+2]]
            winners=[parse_winner(t) for t in raw]
            values=[.5 if w=="TIE" else float(w==("B" if j==0 else "A")) if w else None for j,w in enumerate(winners)]
            record=dict(id=p["id"],baseline_action="none",routed_action=p["choices"]["geometry"],
                        raw=raw,winners=winners,parsed=all(v is not None for v in values),
                        preference=float(np.mean(values)) if all(v is not None for v in values) else None,
                        order_consistent=values[0]==values[1] if all(v is not None for v in values) else None)
            append(path,record)
        print("PAIR AUDIT",lo+len(chunk),"/",len(plan),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--stage",choices=["all","development"],default="all")
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/"inputs/manifest.json").read_text())
    art=json.loads((ROOT/"inputs/art_manifest.json").read_text())
    protocol=dict(objective=manifest,creative=art,concepts=CONCEPTS,probe_examples=probe_examples())
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text())["sha256"] != digest:
        raise RuntimeError("Protocol changed: use a new output directory")
    atomic_json(out/"protocol.json",dict(sha256=digest,**protocol))
    engine=Engine(manifest)
    atomic_json(out/"model.json",engine.meta)
    geo=dict(np.load(out/"geometry.npz")) if (out/"geometry.npz").exists() else build_map(engine,out)
    bank=actions(geo); full=bank+controls(geo)
    atomic_json(out/"actions.json",[{k:v for k,v in a.items() if k!="shift"} | dict(norm=float(np.linalg.norm(a["shift"]))) for a in full])
    dev=[r for r in manifest["records"] if r["split"]=="dev"]
    adev=[r for r in art["records"] if r["split"]=="dev"]
    dfs=features(engine,dev,geo["unit"],out/"dev_features.npz")
    afs=features(engine,adev,geo["unit"],out/"art_dev_features.npz")
    objective_records=execute(engine,[(r,a["id"]) for r in dev for a in full],full,out/"dev.jsonl",256)
    creative_records=execute(engine,[(r,a["id"]) for r in adev for a in full],full,out/"art_dev.jsonl",512)
    release(engine)
    judge=load_judge(ROOT)
    art_scores=judge_records(judge,adev,creative_records,out/"art_dev_judges.jsonl")
    release(judge)
    failed=[k for k,v in art_scores.items() if not v["parsed"]]
    if failed:
        raise RuntimeError(f"Unparsed development judge records: {failed}. No policy was fitted using fabricated scores.")
    if not (out/"objective_policy.json").exists():
        freeze_policy(dev,dfs,{k:float(v["correct"]) for k,v in objective_records.items()},bank,out/"objective_policy.json")
    if not (out/"creative_policy.json").exists():
        freeze_policy(adev,afs,{k:v["rubric"]["score"] for k,v in art_scores.items()},bank,out/"creative_policy.json")
    print("POLICIES FROZEN",flush=True)
    if args.stage=="development":
        return
    # Test routing is now immutable. Targets are used only after inference to grade outputs.
    engine=Engine(manifest)
    test=[r for r in manifest["records"] if r["split"]=="test"]
    atest=[r for r in art["records"] if r["split"]=="test"]
    for rows,kind,tokens in [(test,"objective",256),(atest,"creative",512)]:
        fs=features(engine,rows,geo["unit"],out/f"{kind}_test_features.npz")
        policy=json.loads((out/f"{kind}_policy.json").read_text())
        plan=plan_test(rows,fs,policy,bank)
        plan_path=out/f"{kind}_test_plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("Frozen test plan changed")
        atomic_json(plan_path,plan)
        requests=[(r,a) for r,p in zip(rows,plan) for a in sorted(set(p["choices"].values()))]
        execute(engine,requests,full,out/f"{kind}_test.jsonl",tokens)
    release(engine)
    judge=load_judge(ROOT)
    ctest={(r["id"],r["action"]):r for r in read_jsonl(out/"creative_test.jsonl")}
    judge_records(judge,atest,ctest,out/"art_test_judges.jsonl")
    pair_audit(judge,atest,json.loads((out/"creative_test_plan.json").read_text()),ctest,out/"art_pair_audit.jsonl")
    atomic_json(out/"complete.json",dict(completed_at=time.time(),protocol_sha256=digest))
    print("EXPERIMENT COMPLETE",flush=True)

if __name__=="__main__":
    main()

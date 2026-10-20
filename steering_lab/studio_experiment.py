"""Real character trials. Freeze a Walker choice before rating check scenes."""
import argparse
import json
from pathlib import Path
import numpy as np
from engine import atomic_json
from studio import CharacterStudio, PRESETS, SCENES, profile_for, candidates, mixed_shift, digest, judge_prompt, parse_ratings, choose_winner, utility

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    studio=CharacterStudio(args.run.resolve(),max_tokens=256)
    budget=studio.max_tokens
    profiles=[dict(id=p["id"],**profile_for(p)) for p in PRESETS]
    manifest=dict(profiles=profiles,scenes=SCENES,max_new_tokens=budget,search_target="adventurous",search_scenes=[s["id"] for s in SCENES if s["split"]=="search"],
        check_scenes=[s["id"] for s in SCENES if s["split"]=="check"],provenance=studio.metadata(),source_sha256={n:digest((ROOT/n).read_text()) for n in ["studio.py","studio_experiment.py"]})
    if (out/"manifest.json").exists() and json.loads((out/"manifest.json").read_text())!=manifest:
        raise ValueError("Changed protocol; use a new output directory")
    atomic_json(out/"manifest.json",manifest)
    trials=[]
    for p in profiles:
        for scene in SCENES:
            path=out/f"{p['id']}_{scene['id']}.json"
            if not path.exists(): atomic_json(path,studio.compare(p,scene))
            trials.append(json.loads(path.read_text()))
            print("TRIAL",p["id"],scene["id"],flush=True)
    target=profile_for(PRESETS[1]);search=[]
    for c in candidates(target):
        for scene in SCENES:
            path=out/f"candidate_{c['id']}_{scene['id']}.json"
            if not path.exists():
                if c["id"]=="current":
                    output=next(t for t in trials if t["profile"]["title"]==target["title"] and t["scene"]["id"]==scene["id"])["results"]["geometry"]
                    info=mixed_shift(c["profile"],studio.directions,studio.cap)[1]
                elif c["id"]=="neutral":
                    output=next(t for t in trials if t["profile"]["title"]==target["title"] and t["scene"]["id"]==scene["id"])["results"]["baseline"]
                    info=mixed_shift(c["profile"],studio.directions,studio.cap)[1]
                else:
                    shift,info=mixed_shift(c["profile"],studio.directions,studio.cap)
                    output=studio.engine.generate([studio.prompt(target,scene["question"])],[shift],max_new_tokens=budget)[0]
                atomic_json(path,dict(candidate=c["id"],scene=scene,split=scene["split"],intervention=info,**output))
            search.append(json.loads(path.read_text()))
            print("CANDIDATE",c["id"],scene["id"],flush=True)
    judge=studio.get_judge()
    def rate(record,profile,path):
        if path.exists(): return json.loads(path.read_text())
        output=judge.generate([judge_prompt(profile,record["scene"],record["text"])],[np.zeros(judge.dim)],max_new_tokens=240)[0]
        scored=dict(record,ratings=parse_ratings(output["text"]),judge_raw=output)
        atomic_json(path,scored);return scored
    calibration=[rate(r,target,out/f"rated_{r['candidate']}_{r['scene']['id']}.json") for r in search if r["split"]=="search"]
    choice=choose_winner(calibration,target)
    atomic_json(out/"frozen_choice.json",dict(**choice,selection_manifest_sha256=digest(manifest)))
    if choice["winner"]:
        for scene in SCENES:
            path=out/f"selected_{scene['id']}.json"
            if not path.exists(): atomic_json(path,studio.compare(choice["winner"]["profile"],scene))
            selected=json.loads(path.read_text())
            if not any(digest(t["profile"])==digest(selected["profile"]) and t["scene"]==scene for t in trials):
                trials.append(selected)
    # Only now score the check scenes. They cannot affect the selected recipe.
    checks=[rate(r,target,out/f"rated_{r['candidate']}_{r['scene']['id']}.json") for r in search if r["split"]=="check"]
    for t in trials:
        for mode,r in t["results"].items():
            record=dict(scene=t["scene"],**r)
            rated=rate(record,t["profile"],out/f"judge_{digest(t['profile'])[:16]}_{t['scene']['id']}_{mode}.json")
            r.update(ratings=rated["ratings"],judge_raw=rated["judge_raw"])
    data=dict(manifest=manifest,trials=trials,walker=dict(**choice,records=calibration+checks,requested_profile=target,judge=judge.meta),
        notes=["角色、场景和概念命名由人编写；activation、试演文本和自动评价来自真实模型运行。",
            "只演示一个角色、四个独立短场景；未证明长对话稳定、跨模型迁移或商业效果。",
            "Walker 在两个场景上选择候选配方，再检查另外两个场景。所有场景都是这个演示的人工作品，不是独立 benchmark。",
            "评分来自同族 "+studio.get_judge().meta["model"].split("/")[-1]+"；只作为代理，不是人类审美或付费意愿证据。",
            "正负方向、混合强度均为候选控制；旋钮的名称不保证输出产生对应人格。"],
        generation_settings=48+(12 if choice["winner"] else 0),judge=judge.meta)
    atomic_json(out/"studio_evidence.json",data)
    winner=choice["winner"]
    lines=["# IOTA Character Studio · 实测记录","",f"选中候选：{winner['title'] if winner else '没有满足条件的候选'}。先冻结两个搜索场景的选择，再评分两个检查场景。","", "| 配方 | 搜索代理分 | 新场景代理分 |", "|---|---:|---:|"]
    for row in choice["rankings"]:
        values=[utility(r["ratings"],target,r["truncated"]) for r in checks if r["candidate"]==row["id"]]
        check=sum(values)/len(values) if values and all(v is not None for v in values) else None
        lines.append(f"| {row['title']} | {row['mean'] if row['mean'] is not None else '不合格'} | {check if check is not None else '缺失或截断'} |")
    lines+= ["",*data["notes"]]
    (out/"RESULTS.md").write_text("\n".join(lines));print("COMPLETE",out,flush=True)

if __name__=="__main__": main()

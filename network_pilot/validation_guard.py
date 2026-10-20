"""Repair the omitted validation no-op, preserving the original pilot evidence.

Only validation rewards select between no-op, network fixed, and network
conditional. Existing test outputs are replayed as an explicit post-hoc
diagnostic, never described as a new independent confirmation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiment import ROOT,LAB,Execution,conditional_choices
from pilot_engine import PilotEngine
from engine import atomic_json
from space import action,paired_interval


def main():
    p=argparse.ArgumentParser();p.add_argument("--suite",required=True);p.add_argument("--run",required=True,type=Path)
    args=p.parse_args();run=args.run.resolve();source=run/args.suite;out=run/"validation_guard"/args.suite
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/"inputs/protocol.json").read_text());rows=[r for r in protocol["records"] if r["suite"]==args.suite]
    validation=[r for r in rows if r["split"]=="validation"]
    profile=json.loads((source/"task_profile.json").read_text());plan=json.loads((source/"test_plan.json").read_text())
    original={(r["id"],r["action"]):r for r in map(json.loads,(source/"records.jsonl").read_text().splitlines())}
    geometry=dict(np.load(LAB/"runs/job-2117172/geometry.npz"));scale=json.loads((LAB/"runs/job-2117172/atlas.json").read_text())["scale"]
    engine=PilotEngine(protocol["model"])
    if args.suite=="truthfulqa":engine.precise_likelihood_mode()
    executor=Execution(engine,out,protocol)
    for row in validation:
        key=(row["id"],"none")
        if key not in executor.records and key in original:executor.save(dict(original[key],reused=True,reused_from=str(source/"records.jsonl")))
    noop=action(np.zeros(len(geometry["directions"])),geometry["directions"],scale,"none")
    executor.execute(validation,[noop])
    vfeatures=np.load(source/"validation_features.npz")["features"]
    conditional,_=conditional_choices(profile["policies"]["network"],vfeatures)
    ids=profile["validation_ids"]
    estimates=dict(baseline=float(np.mean([executor.records[x,"none"]["reward"] for x in ids])),
        network_fixed=float(np.mean([original[x,profile["fixed"]["network"]]["reward"] for x in ids])),
        network_conditional=float(np.mean([original[x,a]["reward"] for x,a in zip(ids,conditional)])))
    # Deterministic tie order favors no-op, then the simpler fixed configuration.
    order=["baseline","network_fixed","network_conditional"]
    chosen=max(order,key=lambda name:(estimates[name],-order.index(name)))
    frozen=dict(suite=args.suite,validation_ids=ids,validation_scores=estimates,chosen_strategy=chosen,
        rule="Maximum validation mean; ties prefer baseline, then fixed, then conditional.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Post-hoc correction of omitted no-op in validation finalists. Uses only validation scores for selection;"
              " existing locked test outputs are diagnostic replay, not new confirmation.")
    atomic_json(out/"frozen_profile.json",frozen)
    chosen_values=[original[p["id"],p["choices"][chosen]]["reward"] for p in plan]
    base_values=[original[p["id"],p["choices"]["baseline"]]["reward"] for p in plan]
    difference=np.asarray(chosen_values)-base_values
    result=dict(**frozen,baseline=float(np.mean(base_values)),selected=float(np.mean(chosen_values)),delta=float(difference.mean()),
        paired_ci=paired_interval(difference),n=len(plan),test_ids=[p["id"] for p in plan])
    atomic_json(out/"summary.json",result);atomic_json(out/"complete.json",dict(complete=True))
    print("VALIDATION GUARD COMPLETE",result,flush=True)


if __name__=="__main__":main()

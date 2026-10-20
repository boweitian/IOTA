"""Cross-task configuration exchange after every source profile is frozen."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment import Execution,ROOT,LAB,prompt
from pilot_engine import PilotEngine
from engine import atomic_json
from space import action
from repair_empty_choice import score_with_empty


def main():
    # Same mathematically defined empty-continuation handling as the repaired pilot.
    PilotEngine.answer_likelihoods=score_with_empty
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);p.add_argument("--suite",required=True)
    args=p.parse_args();run=args.run.resolve()
    protocol=json.loads((ROOT/"inputs/protocol.json").read_text())
    profiles={s:json.loads((run/s/"task_profile.json").read_text()) for s in protocol["suites"]}
    if not all((run/s/"complete.json").exists() for s in profiles):raise RuntimeError("Wait for all frozen profile evaluations")
    out=run/"transfer"/args.suite;out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in protocol["records"] if r["suite"]==args.suite and r["split"]=="test"]
    geo=dict(np.load(LAB/"runs/job-2117172/geometry.npz"));scale=json.loads((LAB/"runs/job-2117172/atlas.json").read_text())["scale"]
    bank=[action(profiles[s]["selected_coefficients"]["network"],geo["directions"],scale,"transfer:"+s) for s in profiles]
    frozen=dict(target_suite=args.suite,source_configs={s:profiles[s]["selected_coefficients"]["network"] for s in profiles},
        ids=[r["id"] for r in rows],note="Frozen network-fixed profiles exchanged between tasks. Diagnostic on existing locked test inputs; no profiles retuned.")
    if (out/"plan.json").exists() and json.loads((out/"plan.json").read_text())!=frozen:raise RuntimeError("Changed exchange plan")
    atomic_json(out/"plan.json",frozen)
    original={(r["id"],r["action"]):r for r in map(json.loads,(run/args.suite/"records.jsonl").read_text().splitlines())}
    own_plan={p["id"]:p for p in json.loads((run/args.suite/"test_plan.json").read_text())}
    engine=PilotEngine(protocol["model"])
    if args.suite=="truthfulqa":engine.precise_likelihood_mode()
    atomic_json(out/"model.json",engine.meta)
    execute=Execution(engine,out,protocol)
    # Exact own-task outputs can be reused; no new model call is claimed for them.
    for row in rows:
        a="transfer:"+args.suite
        if (row["id"],a) not in execute.records:
            old=original[row["id"],own_plan[row["id"]]["choices"]["network_fixed"]]
            copied=dict(old,action=a,reused=True,reused_from=str(run/args.suite/"records.jsonl"),
                reused_action=old["action"])
            execute.save(copied)
    execute.execute(rows,bank)
    rewards={s:[execute.records[r["id"],"transfer:"+s]["reward"] for r in rows] for s in profiles}
    atomic_json(out/"scores.json",dict(target_suite=args.suite,means={s:float(np.mean(v)) for s,v in rewards.items()},
        per_item=rewards,ids=[r["id"] for r in rows],note=frozen["note"]))
    atomic_json(out/"complete.json",dict(record_count=len(execute.records)))
    print("TRANSFER COMPLETE",args.suite,{s:float(np.mean(v)) for s,v in rewards.items()},flush=True)


if __name__=="__main__":main()

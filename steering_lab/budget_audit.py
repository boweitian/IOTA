"""Post-hoc budget audit: identical frozen routes, larger caps, no recalibration."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from engine import Engine,atomic_json
from experiment import actions,controls,execute,release,judge_records,pair_audit
from judge import load_judge

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);args=p.parse_args()
    run=args.run.resolve();out=run/"budget_audit";out.mkdir(exist_ok=True)
    protocol=json.loads((run/"protocol.json").read_text())
    policy_hashes={k:hashlib.sha256((run/f"{k}_policy.json").read_bytes()).hexdigest() for k in ["objective","creative"]}
    atomic_json(out/"audit_protocol.json",dict(original_protocol_sha256=protocol["sha256"],
        policy_sha256=policy_hashes,objective_max_tokens=1024,creative_max_tokens=1024,
        reason="Follow-up after observing substantial truncation in the initial fixed-cap run; post-hoc audit, not a new held-out claim. Directions, policies, routes, prompts, decoding and scoring remain frozen."))
    engine=Engine(protocol["objective"])
    geometry=dict(np.load(run/"geometry.npz"));bank=actions(geometry)+controls(geometry)
    for kind,tokens in [("objective",1024),("creative",1024)]:
        source="objective" if kind=="objective" else "creative"
        rows=[r for r in protocol[source]["records"] if r["split"]=="test"]
        plan=json.loads((run/f"{kind}_test_plan.json").read_text())
        atomic_json(out/f"{kind}_test_plan.json",plan)
        requests=[(r,a) for r,p in zip(rows,plan) for a in sorted(set(p["choices"].values()))]
        execute(engine,requests,bank,out/f"{kind}_test.jsonl",tokens)
    release(engine)
    judge=load_judge(ROOT)
    from experiment import read_jsonl
    rows=[r for r in protocol["creative"]["records"] if r["split"]=="test"]
    records={(r["id"],r["action"]):r for r in read_jsonl(out/"creative_test.jsonl")}
    judge_records(judge,rows,records,out/"art_test_judges.jsonl")
    pair_audit(judge,rows,json.loads((out/"creative_test_plan.json").read_text()),records,out/"art_pair_audit.jsonl")
    assert policy_hashes=={k:hashlib.sha256((run/f"{k}_policy.json").read_bytes()).hexdigest() for k in policy_hashes}
    atomic_json(out/"complete.json",dict(original_protocol_sha256=protocol["sha256"],policy_sha256=policy_hashes))
    print("BUDGET AUDIT COMPLETE; POLICY HASHES UNCHANGED",flush=True)

if __name__=="__main__":
    main()

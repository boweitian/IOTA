"""Audit native vectors, route provenance and saved scoring against source artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from concepts import CONCEPTS
from experiment import read_jsonl
from grading import grade
from router import predict,geometry_features,calibrate,text_features

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);args=p.parse_args()
    run=args.run.resolve();checks=[]
    protocol=json.loads((run/"protocol.json").read_text())
    original={k:v for k,v in protocol.items() if k!="sha256"}
    assert hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()==protocol["sha256"]
    checks.append("frozen protocol SHA matches all source fields")
    geo=dict(np.load(run/"geometry.npz"));acts=np.load(run/"probe_activations.npz")["pooled"]
    probes=json.loads((run/"probes.json").read_text())
    vectors={(p["concept"],p["context"],p["pole"]):h for p,h in zip(probes,acts)}
    measured=np.stack([np.mean([vectors[c["id"],i,"positive"]-vectors[c["id"],i,"negative"] for i in range(8)],0) for c in CONCEPTS])
    np.testing.assert_allclose(geo["raw"],measured,atol=1e-7)
    np.testing.assert_allclose(geo["unit"],measured/np.linalg.norm(measured,axis=1,keepdims=True),atol=1e-7)
    scale=.20*np.median(np.linalg.norm(acts,axis=1))
    np.testing.assert_allclose(geo["directions"],geo["unit"]*scale,atol=1e-7)
    np.testing.assert_allclose(geo["unit"]@geo["random"],0,atol=1e-6)
    checks.append("all native directions independently reconstructed from real probe activations; random control orthogonal and norm-matched")
    graded=0
    for kind,dev_file,feature_file in [("objective","dev.jsonl","dev_features.npz"),("creative","art_dev.jsonl","art_dev_features.npz")]:
        rows=[r for r in protocol[kind]["records"] if r["split"]=="dev"]
        policy=json.loads((run/f"{kind}_policy.json").read_text())
        assert policy["dev_ids"]==[r["id"] for r in rows]
        assert not set(policy["dev_ids"]) & {r["id"] for r in protocol[kind]["records"] if r["split"]=="test"}
        fs=np.load(run/feature_file)
        np.testing.assert_allclose(fs["features"],geometry_features(fs["pooled"],fs["last"],geo["unit"]),atol=1e-8)
        records={(r["id"],r["action"]):r for r in read_jsonl(run/dev_file)}
        judges={(r["id"],r["action"]):r for r in read_jsonl(run/"art_dev_judges.jsonl")} if kind=="creative" else None
        reward=np.asarray([[judges[r["id"],a]["rubric"]["score"] if judges is not None else float(records[r["id"],a]["correct"]) for a in policy["action_ids"]] for r in rows])
        costs=policy["geometry"]["costs"]
        assert calibrate(fs["features"],reward,costs,policy["dev_ids"])==policy["geometry"]
        assert calibrate(text_features([r["question"] for r in rows]),reward,costs,policy["dev_ids"],standardize=False)==policy["lexical"]
        test_features=np.load(run/f"{kind}_test_features.npz")
        routed,_=predict(policy["geometry"],test_features["features"])
        plan=json.loads((run/f"{kind}_test_plan.json").read_text())
        assert [p["choices"]["geometry"] for p in plan]==[policy["action_ids"][int(i)] for i in routed]
        checks.append(kind+": policy reproduced from development rewards; routes reproduced without test labels")
    lookup={r["id"]:r for r in protocol["objective"]["records"]}
    files=[run/"dev.jsonl",run/"objective_test.jsonl"]
    if (run/"budget_audit/complete.json").exists():
        files.append(run/"budget_audit/objective_test.jsonl")
        audit=json.loads((run/"budget_audit/complete.json").read_text())
        assert audit["policy_sha256"]=={k:hashlib.sha256((run/f"{k}_policy.json").read_bytes()).hexdigest() for k in ["creative","objective"]}
        for kind in ["creative","objective"]:
            assert (run/f"{kind}_test_plan.json").read_bytes()==(run/f"budget_audit/{kind}_test_plan.json").read_bytes()
        checks.append("budget audit preserved exact policies and route plans")
    for file in files:
        for r in read_jsonl(file):
            expected=grade(r["text"],lookup[r["id"]])
            assert all(r[k]==v for k,v in expected.items())
            graded+=1
    checks.append(f"all {graded} saved objective judgments independently recomputed")
    result=dict(checks=checks,protocol_sha256=protocol["sha256"])
    (run/"evidence_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

"""Replay saved choices, graph effects and scores without loading a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiment import ROOT,LAB,read_records,score_ifeval,parse_rubric
from grading import grade
from router import predict,geometry_features
from space import task_graph


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);p.add_argument("--partial",action="store_true");args=p.parse_args();run=args.run.resolve()
    protocol=json.loads((ROOT/"inputs/protocol.json").read_text())
    geo=dict(np.load(LAB/"runs/job-2117172/geometry.npz"));scale=json.loads((LAB/"runs/job-2117172/atlas.json").read_text())["scale"]
    checks=[]
    for suite in protocol["suites"]:
        out=run/suite
        if not (out/"complete.json").exists():
            if args.partial:continue
            raise RuntimeError(f"Incomplete suite {suite}")
        records=read_records(out/"records.jsonl");rows={r["id"]:r for r in protocol["records"] if r["suite"]==suite}
        actions={a["id"]:a for a in json.loads((out/"all_actions.json").read_text())}
        profile=json.loads((out/"task_profile.json").read_text());plan=json.loads((out/"test_plan.json").read_text())
        graph=json.loads((out/"task_network.json").read_text())
        provenance=json.loads((out/"protocol.json").read_text())
        archived_sources=[]
        for name,digest in provenance["source_hashes"].items():
            path=LAB/name.split("/",1)[1] if name.startswith("steering_lab/") else ROOT/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                path=run/"source_snapshot"/name
                archived_sources.append(name)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(suite,name)
        for name,digest in provenance["vendor_hashes"].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        assert not set(profile["search_ids"])&set(profile["test_ids"])
        assert not set(profile["validation_ids"])&set(profile["test_ids"])
        for name,a in actions.items():
            norm=np.linalg.norm(np.asarray(a["coefficients"])@geo["directions"])
            assert norm<=scale+1e-6,(suite,name,norm)
            np.testing.assert_allclose(norm,a["norm"],atol=1e-6)
        recalculated,adjacency=task_graph(graph["anchors"],records,len(geo["directions"]))
        np.testing.assert_allclose(adjacency,graph["adjacency"],atol=1e-12)
        for ident in graph["anchors"]:assert rows[ident]["split"]=="search"
        fs=np.load(out/"test_features.npz")
        np.testing.assert_allclose(geometry_features(fs["pooled"],fs["last"],geo["unit"]),fs["features"],atol=1e-6)
        assert fs["ids"].tolist()==[x["id"] for x in plan]
        for branch,strategy in [("flat","flat_conditional"),("network","network_conditional"),("node_guided","node_guided_conditional")]:
            policy=profile["policies"][branch]
            assert policy["train_ids"]==profile["search_ids"]
            selected,_=predict(policy["knn"],fs["features"])
            assert [policy["action_ids"][int(i)] for i in selected]==[p["choices"][strategy] for p in plan]
        for record in records.values():
            row=rows[record["id"]]
            assert record["split"]==row["split"]
            if suite in ["gsm8k","bbh"]:reward=float(grade(record["text"],row)["correct"])
            elif suite=="ifeval":reward=score_ifeval(row,record["text"])["reward"]
            elif suite=="truthfulqa":
                x=np.asarray(record["choice_log_probabilities"]);prob=np.exp(x-x.max());prob/=prob.sum()
                reward=float(prob[np.asarray(row["targets"]["labels"],dtype=bool)].sum())
            else:
                raw=(record["judge_retry"] or record["judge_raw"])["text"]
                parsed=parse_rubric(raw,len(row["source_record"]["checklist"]));assert parsed is not None
                reward=parsed["reward"]
            np.testing.assert_allclose(reward,record["reward"],atol=1e-12)
        for p in plan:
            assert rows[p["id"]]["split"]=="test"
            for a in p["choices"].values():assert (p["id"],a) in records
        checks.append(dict(suite=suite,records=len(records),actions=len(actions),test_items=len(plan),
            protocol_hashes=True,norm_bound=True,task_graph_replayed=True,features_replayed=True,
            conditional_plans_replayed=True,scores_replayed=True,split_isolation=True,
            sources_verified_from_snapshot=archived_sources))
    result=dict(passed=True,all_suites_complete=len(checks)==len(protocol["suites"]),checks=checks,note="CPU replay of existing evidence; no new model measurements.")
    (run/"evidence_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()

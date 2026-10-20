"""Candidate subproblem regions discovered from search-set response patterns.

These regions are proposed assets, not validated test-performance policies.
"""
import json
from pathlib import Path
import numpy as np


def discover_regions(path):
    path=Path(path)
    profile=json.loads((path/"task_profile.json").read_text())
    records={(r["id"],r["action"]):r for r in map(json.loads,(path/"records.jsonl").read_text().splitlines())}
    ids=profile["search_ids"];bank=profile["policies"]["network"]["action_ids"]
    actions={a["id"]:a for a in json.loads((path/"all_actions.json").read_text())}
    rewards=np.asarray([[records[i,a]["reward"] for a in bank] for i in ids])
    patterns=rewards-rewards[:,0:1]
    clusters=[[i] for i in range(len(ids))]
    # Ward merges in intervention-response space. Fixed three-region proposal.
    while len(clusters)>min(3,len(ids)):
        candidates=[]
        for i,c in enumerate(clusters):
            for j,d in enumerate(clusters[:i]):
                distance=len(c)*len(d)/(len(c)+len(d))*np.square(patterns[c].mean(0)-patterns[d].mean(0)).sum()
                candidates.append((float(distance),j,i))
        _,i,j=min(candidates)
        clusters[i]=clusters[i]+clusters[j];clusters.pop(j)
    fs=np.load(path/"search_features.npz");tf=np.load(path/"test_features.npz")
    assert fs["ids"].tolist()==ids
    mean=fs["features"].mean(0);std=np.maximum(fs["features"].std(0),1e-5)
    train=(fs["features"]-mean)/std;test=(tf["features"]-mean)/std
    centers=np.asarray([train[c].mean(0) for c in clusters]);regions=[]
    for k,c in enumerate(clusters):
        expected=rewards[c].mean(0)
        best=min(range(len(bank)),key=lambda i:(-expected[i],actions[bank[i]]["norm"],bank[i]))
        regions.append(dict(id=f"region:{k}",search_ids=[ids[i] for i in c],
            response_pattern=patterns[c].mean(0).tolist(),geometry_center=centers[k].tolist(),
            proposed_configuration=bank[best],coefficients=actions[bank[best]]["coefficients"],
            search_mean_reward=float(expected[best]),baseline_mean_reward=float(rewards[c,0].mean()),
            native_prompt_state_center=fs["pooled"][c].mean(0).tolist(),
            status="candidate; no independent confirmation of this clustering or region-specific choice"))
    nearest=np.square(test[:,None,:]-centers[None,:,:]).mean(-1).argmin(1)
    return dict(schema_version=1,suite=profile["suite"],regions=regions,action_ids=bank,
        normalization=dict(mean=mean.tolist(),std=std.tolist()),
        input_only_region_assignments=[dict(id=str(i),region=f"region:{int(k)}") for i,k in zip(tf["ids"],nearest)],
        uses_search_rewards=True,uses_test_rewards=False,
        algorithm="Three-region Ward clustering of search intervention responses, with input-geometry nearest-centroid proposals.",
        limitations="Very small search set; region count fixed rather than validated. Assignments do not demonstrate performance gains."
                     " New labels or regions are not newly identified intrinsic model concepts.")

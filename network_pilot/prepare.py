"""Pin official sources and freeze new samples before any steering search."""
import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent / "steering_lab"
SEED = "iota-network-pilot-20261012-v1"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def fetch(repo, revision, file, output):
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{file}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": SEED}), timeout=60) as f:
        raw = f.read()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    return raw, dict(repo=repo, revision=revision, file=file, url=url,
                     sha256=hashlib.sha256(raw).hexdigest())


def partition(rows, counts, key):
    rows = list(rows)
    random.Random(SEED + key).shuffle(rows)
    if len(rows) < sum(counts):
        raise ValueError(f"Insufficient fresh rows for {key}: {len(rows)}")
    selected, cursor = [], 0
    for split, count in zip(["search", "validation", "test"], counts):
        for row in rows[cursor:cursor+count]:
            selected.append(dict(row, split=split))
        cursor += count
    return selected


def canonical_targets(value):
    if "choices" in value and "labels" in value:return value
    return dict(choices=list(value),labels=list(value.values()))


def main():
    out = ROOT / "inputs"
    if (out / "protocol.json").exists():
        raise RuntimeError("Inputs already frozen; use the saved protocol, do not redraw samples")
    out.mkdir(exist_ok=True)
    sources, rows = [], []
    old = json.loads((LAB / "inputs/manifest.json").read_text())
    used = {r["id"] for r in old["records"]}
    for source in old["sources"]:
        file = LAB / "inputs" / f"{source['task']}_{source['part']}.{'jsonl' if source['task']=='gsm8k' else 'json'}"
        raw = file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"Source changed: {file}")
        sources.append(source)
        items = [json.loads(line) for line in raw.splitlines()] if source["task"] == "gsm8k" else json.loads(raw)["examples"]
        candidates = []
        for i, item in enumerate(items):
            ident = f"{source['task']}:{source['part']}:{i}"
            if ident in used:
                continue
            task = source["task"]
            candidates.append(dict(id=ident, suite="gsm8k" if task=="gsm8k" else "bbh", task=task,
                question=item["question"] if task=="gsm8k" else item["input"],
                target=item["answer"].split("####")[-1].strip() if task=="gsm8k" else item["target"],
                source_index=i, group=task))
        if source["task"] == "gsm8k":
            selected = partition(candidates, (8,4,0) if source["part"]=="train" else (0,0,24), source["task"]+source["part"])
        else:
            selected = partition(candidates, (3,2,8), source["task"])
        rows.extend(selected)

    truth_rev = "d71c110897f5d31c5d7f309e7bc316c152f6f031"
    raw, source = fetch("sylinrl/TruthfulQA", truth_rev, "data/mc_task.json", out / "truthfulqa_mc.json")
    sources.append(source)
    truth = json.loads(raw)
    rows.extend(partition([dict(id=f"truthfulqa:{i}", suite="truthfulqa", task="truthfulqa",
        question=r["question"], targets=canonical_targets(r["mc2_targets"]), mc1_targets=canonical_targets(r["mc1_targets"]),
        group="truthfulqa", source_index=i) for i,r in enumerate(truth)], (8,4,24), "truthfulqa"))

    google_rev = "e0b7b00ec50597d6cebeab301cc70995744de33a"
    raw, source = fetch("google-research/google-research", google_rev,
        "instruction_following_eval/data/input_data.jsonl", out / "ifeval.jsonl")
    sources.append(source)
    candidates = [dict(id=f"ifeval:{r['key']}", suite="ifeval", task="ifeval", question=r["prompt"],
        instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"], source_index=r["key"],
        group="|".join(sorted(r["instruction_id_list"]))) for r in map(json.loads,raw.splitlines())]
    # Split whole instruction combinations, rather than redistributing duplicates.
    groups = sorted({r["group"] for r in candidates})
    random.Random(SEED+"ifeval-groups").shuffle(groups)
    buckets = {g: [r for r in candidates if r["group"]==g] for g in groups}
    selected = []
    for split,count in [("search",8),("validation",4),("test",24)]:
        pool=[]
        while len(pool)<count:
            if not groups: raise ValueError("Insufficient IFEval groups")
            pool.extend(buckets[groups.pop()])
        random.Random(SEED+split).shuffle(pool)
        selected.extend(dict(r,split=split) for r in pool[:count])
    rows.extend(selected)
    vendor = ROOT / "vendor/instruction_following_eval"
    vendor.mkdir(parents=True,exist_ok=True)
    (vendor/"__init__.py").write_text("")
    for name in ["instructions.py", "instructions_util.py", "instructions_registry.py", "evaluation_lib.py", "requirements.txt", "README.md"]:
        _,source=fetch("google-research/google-research",google_rev,"instruction_following_eval/"+name,vendor/name)
        sources.append(source)

    writing_rev="ae2d5176449b7b769815482641d35926f26793eb"
    raw,source=fetch("X-PLUG/WritingBench",writing_rev,"benchmark_query/benchmark_all.jsonl",out/"writingbench.jsonl")
    sources.append(source)
    writing=[json.loads(line) for line in raw.splitlines()]
    save(out/"writing_schema.json",dict(example=writing[0],count=len(writing)))
    # Schema is checked explicitly; source questions are retained as provenance.
    candidates=[]
    for i,r in enumerate(writing):
        q=r.get("query",r.get("prompt",r.get("question")))
        if not isinstance(q,str):raise ValueError(f"Unknown WritingBench schema: {list(r)}")
        # English subset only. All methods get the same declared short-form adaptation.
        if re.search(r"[\u4e00-\u9fff]",q):continue
        topic=str(r.get("domain1",r.get("domain",r.get("category","writing"))))+"/"+str(r.get("domain2",""))
        qshort=re.sub(r"\b[1-9][\d,]*\s*(?:-|–|to)\s*[1-9][\d,]*\s+words\b","150–220 words",q,flags=re.I)
        qshort=re.sub(r"\b[1-9][\d,]*\s+words\b","200 words",qshort,flags=re.I)
        qshort+="\n\nFor this pilot, produce a complete short-form response in 150–220 words. Follow the other requested requirements. Return only the response."
        candidates.append(dict(id=f"writingbench:{i}",suite="writingbench",task="writingbench",
            question=qshort,original_question=q,source_record=r,source_index=i,group=topic))
    # Category grouping depends on the official schema; exact source prompts never overlap.
    rows.extend(partition(candidates,(8,4,16),"writingbench"))
    manifest=dict(schema_version=1,seed=SEED,model={k:old[k] for k in ["model","model_revision","model_path","layer_index"]},
        suites=["gsm8k","bbh","ifeval","truthfulqa","writingbench"],sources=sources,records=rows,
        limitations=["Small pilot subsets, not full official benchmark scores.",
            "WritingBench short-form adaptation with local Qwen3.5-4B rubric, not official WritingBench scoring.",
            "New objective samples exclude all previous IOTA benchmark sample IDs.",
            "Public dataset pretraining exposure is unknown; internal split isolation does not prove absence of contamination.",
            "Writing samples are exact-prompt disjoint; category transfer is not established."],
        budgets=dict(max_new_tokens=1024,batch_size=8,proposal_count=12,
            pair_anchor_count=4,bootstrap_repetitions=3000),
        search_spec=dict(single_coefficients=[-.5,.5],pair_coefficients=[.5,.5],
            pair_sign_scope="positive-positive only; other signed edge interactions are unmeasured",
            max_actual_shift_norm="existing geometry scale, 20% of median probe residual norm",
            seeds=[20261012,20261013,20261014],
            reward="task scores on search rows; validation selects policies; test labels never enter fitting"))
    ids=[r["id"] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate IDs")
    questions=[r["question"] for r in rows]
    if len(questions)!=len(set(questions)):raise ValueError("Duplicate prompts")
    for suite in manifest["suites"]:
        selected=[r for r in rows if r["suite"]==suite]
        for split in ["search","validation","test"]:
            print(suite,split,sum(r["split"]==split for r in selected),flush=True)
    save(out/"protocol.json",manifest)


if __name__=="__main__":main()

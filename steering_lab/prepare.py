"""Fetch official weights/data, pin revisions, and freeze a small evaluation protocol."""
import hashlib
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-0.8B"

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "IOTA-research-MVP"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()

def main():
    from huggingface_hub import HfApi, snapshot_download
    out = ROOT / "inputs"
    out.mkdir(exist_ok=True)
    revision = HfApi().model_info(MODEL).sha
    path = snapshot_download(MODEL, revision=revision,
        cache_dir=ROOT.parents[2] / ".hf_cache/hub",
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "vocab*", "merges*"], max_workers=4)
    print("MODEL", revision, path, flush=True)
    sources, records = [], []
    repos = {r: json.loads(fetch(f"https://api.github.com/repos/{r}/commits/HEAD"))["sha"]
             for r in ["openai/grade-school-math", "suzgunmirac/BIG-Bench-Hard"]}
    tasks = ["gsm8k", "boolean_expressions", "logical_deduction_five_objects", "multistep_arithmetic_two"]
    for task in tasks:
        parts = ["train", "test"] if task == "gsm8k" else ["all"]
        for part in parts:
            repo = "openai/grade-school-math" if task == "gsm8k" else "suzgunmirac/BIG-Bench-Hard"
            file = f"grade_school_math/data/{part}.jsonl" if task == "gsm8k" else f"bbh/{task}.json"
            url = f"https://raw.githubusercontent.com/{repo}/{repos[repo]}/{file}"
            raw = fetch(url)
            (out / f"{task}_{part}.{'jsonl' if task == 'gsm8k' else 'json'}").write_bytes(raw)
            sources.append(dict(task=task, part=part, url=url, revision=repos[repo],
                                sha256=hashlib.sha256(raw).hexdigest()))
            rows = [json.loads(line) for line in raw.splitlines()] if task == "gsm8k" else json.loads(raw)["examples"]
            ids = list(range(len(rows)))
            random.Random(f"iota-20261011-{task}-{part}").shuffle(ids)
            selections = [("dev", ids[:16])] if part == "train" else [("test", ids[:32])] if part == "test" else [("dev", ids[:16]), ("test", ids[16:48])]
            for split, selected in selections:
                for idx in selected:
                    r = rows[idx]
                    question = r["question"] if task == "gsm8k" else r["input"]
                    answer = r["answer"].split("####")[-1].strip() if task == "gsm8k" else r["target"]
                    records.append(dict(id=f"{task}:{part}:{idx}", task=task, split=split,
                        question=question, target=answer, source_index=idx, source_part=part))
    manifest = dict(model=MODEL, model_revision=revision, model_path=path, sources=sources,
        seed="iota-20261011", tasks=tasks, dev_per_task=16, test_per_task=32,
        mode="non-thinking; greedy; same baseline prompt for activation steering",
        layer_index=15, max_new_tokens=256, strengths=[0.5, 1.0],
        note="Mini subsets, not full official benchmark scores. BBH dev/test are disjoint samples of official BBH examples; GSM8K uses official train/test.",
        records=records)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("FROZEN", len(records), "items", flush=True)

if __name__ == "__main__":
    main()

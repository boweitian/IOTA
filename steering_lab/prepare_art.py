"""Pin the official EQ-Bench creative prompts before any policy calibration."""
import hashlib
import json
import random
from pathlib import Path
from prepare import fetch

ROOT = Path(__file__).resolve().parent

def main():
    repo = "EQ-bench/creative-writing-bench"
    revision = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/HEAD"))["sha"]
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/data/creative_writing_prompts_v3.json"
    raw = fetch(url)
    (ROOT / "inputs/creative_writing_prompts_v3.json").write_bytes(raw)
    data = json.loads(raw)
    print("REVISION",revision,"TYPE",type(data).__name__,flush=True)
    (ROOT / "inputs/art_source.json").write_text(json.dumps(dict(url=url,revision=revision,
        sha256=hashlib.sha256(raw).hexdigest(), original=data),ensure_ascii=False,indent=2))
    ids = sorted(data)
    random.Random("iota-creative-20261012").shuffle(ids)
    records = []
    import re
    for i,key in enumerate(ids):
        item = data[key]
        modifier = item["seed_modifiers"][0] if item.get("seed_modifiers") else ""
        original = item["writing_prompt"].replace("<SEED>", modifier)
        # A declared short-form adaptation, not the leaderboard generation protocol.
        short = re.sub(r"\b(?:1,?000|800|500)\s+words\b", "200 words", original, flags=re.I)
        short += "\n\nFor this short-form experiment, write a complete piece in 180–220 words. Give only the piece, without introductory commentary."
        records.append(dict(id="creative:"+key, task="creative_writing", split="dev" if i < 12 else "test",
            question=short, original_question=original, title=item.get("title",key),
            category=item.get("category","creative"), target=None, source_index=key))
    artifact = dict(source=dict(url=url,revision=revision,sha256=hashlib.sha256(raw).hexdigest()),
        adaptation="Short-form EQ-Bench v3 prompt subset: original 1000-word requests replaced by 200 words, first seed modifier, 180–220-word brief, greedy decode; local Qwen3.5-4B judge. Not official EQ-Bench scores or Elo.",
        max_new_tokens=512, dev_count=12, test_count=len(records)-12, records=records)
    (ROOT / "inputs/art_manifest.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    print("FROZEN ART",len(records),flush=True)

if __name__ == "__main__":
    main()

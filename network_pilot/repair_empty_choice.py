"""Resume TruthfulQA after an empty answer in the pinned official test source.

The conditional log probability of an empty continuation is exactly zero.
Keep the source example and label, and record this dataset artifact explicitly.
Nonempty scoring, sample IDs, fitted profiles and test choices stay unchanged.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import experiment

original=experiment.PilotEngine.answer_likelihoods


def score_with_empty(self,question,choices,shift,extra="",batch_size=8):
    indices=[i for i,a in enumerate(choices) if self.tokenizer.encode(a,add_special_tokens=False)]
    if len(indices)==len(choices):return original(self,question,choices,shift,extra,batch_size)
    values,seconds=original(self,question,[choices[i] for i in indices],shift,extra,batch_size)
    result=np.zeros(len(choices),dtype=float)
    result[indices]=values
    return result.tolist(),seconds


def main():
    args=sys.argv[1:]
    out=Path(args[args.index("--output")+1]).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((experiment.ROOT/"inputs/protocol.json").read_text())
    affected=[dict(id=r["id"],split=r["split"],empty_choice_indices=[i for i,a in enumerate(r["targets"]["choices"]) if not a])
        for r in manifest["records"] if r["suite"]=="truthfulqa" and any(not a for a in r["targets"]["choices"])]
    note=dict(reason="Pinned official source contains an empty incorrect choice. Initial scorer rejected empty tokens.",
        behavior="Retain the example and choice; use log P(empty continuation | prefix) = 0. This source artifact can dominate MC2 denominator.",
        affected=affected,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        profiles_unchanged=True,nonempty_scoring_unchanged=True)
    (out/"empty_choice_repair.json").write_text(json.dumps(note,ensure_ascii=False,indent=2))
    experiment.PilotEngine.answer_likelihoods=score_with_empty
    experiment.main()


if __name__=="__main__":main()

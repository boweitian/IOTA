"""Measure inference batch throughput before freezing the larger GSM8K run."""
import json
import numpy as np
from experiment import LAB, PilotEngine, prompt

p=json.loads((LAB/'inputs/manifest.json').read_text())
rows=[json.loads(s) for s in (LAB/'inputs/gsm8k_train.jsonl').read_text().splitlines()][:64]
e=PilotEngine(p)
print('MODEL',json.dumps(e.meta),flush=True)
for n in [8,32,64]:
    qs=[prompt(dict(suite='gsm8k',question=r['question'])) for r in rows[:n]]
    out=e.generate(qs,[np.zeros(e.dim)]*n,max_new_tokens=512)
    seconds=out[0]['batch_seconds']
    print('SPEED',json.dumps(dict(batch=n,seconds=seconds,tokens=sum(o['new_tokens'] for o in out),
        max_tokens=max(o['new_tokens'] for o in out),truncated=sum(o['truncated'] for o in out))),flush=True)

"""One requested prompt change, with the original FP32 model and zero steering."""
import json
from pathlib import Path
import numpy as np
from classification_sweep import ClassificationEngine
from anli_diagnose import SOURCE, OUT
from sentiment_navigator import read
from sentiment_prepare import write

def main():
    manifest=json.loads((SOURCE/'dataset-manifest.json').read_text())
    protocol=json.loads((SOURCE/'search-protocol.json').read_text())
    row=next(r for r in read(SOURCE/'feasible-space.jsonl') if r['uid']=='05140765-980f-4d3d-acbb-7ebb4525fd67')
    original=manifest['prompt_template']
    changed=original.replace('Respond with exactly one letter: A, B, or C.',
        '请用中文简要说明支持你选择的关键文本依据，最多两句话；然后另起一行给出最终选项，格式为 Final: A、Final: B 或 Final: C。')
    engine=ClassificationEngine(protocol['objective'],dict(manifest,prompt_template=changed))
    question=engine.question(row)
    result=engine.generate([question],[np.zeros(engine.dim)],max_new_tokens=256)[0]
    payload=dict(id=row['id'],model=engine.meta,prompt=question,original_prompt=original.format(context=json.dumps(row['context'],ensure_ascii=False),hypothesis=json.dumps(row['hypothesis'],ensure_ascii=False)),
        steering=dict(enabled=False,coefficients=[0.0]*9,shift_norm=0.0),result=result)
    write(OUT/'petition-chinese-explanation.json',payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

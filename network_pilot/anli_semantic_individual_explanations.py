"""Requested individual-direction explanations using the saved frozen vectors."""
import json
import time
import numpy as np
from anli_semantic_guidance import OUT, SOURCE, AXES, ZH, mixture
from classification_sweep import ClassificationEngine
from sentiment_prepare import write,sha
from sentiment_navigator import read,digest

def main():
 spec=json.loads((OUT/'spec.json').read_text());result=json.loads((OUT/'result.json').read_text())
 assert sha(OUT/'directions.npz')==result['directions_sha256']
 manifest=json.loads((SOURCE/'dataset-manifest.json').read_text())
 manifest=dict(manifest,prompt_template=manifest['prompt_template'].replace('Respond with exactly one letter: A, B, or C.',ZH))
 engine=ClassificationEngine(spec['model'],manifest)
 row=next(r for r in read(SOURCE/'feasible-space.jsonl') if r['uid']==spec['target_uid']);question=engine.question(row)
 directions=np.load(OUT/'directions.npz')['directions'];conditions=[]
 for i,a in enumerate(AXES):
  for dose in [1.,2.]:
   coefficients=np.zeros(3);coefficients[i]=dose;shift,info=mixture(directions,coefficients)
   conditions.append(dict(condition=f'{a["id"]}:{dose:+g}',shift=shift,steering=info))
 outputs=[]
 for lo in range(0,len(conditions),3):
  batch=conditions[lo:lo+3];generated=engine.generate([question]*len(batch),[c['shift'] for c in batch],max_new_tokens=256)
  for c,value in zip(batch,generated):outputs.append(dict(condition=c['condition'],steering=c['steering'],output=value))
  print('completed',len(outputs),flush=True)
 write(OUT/'individual-explanations.json',dict(source_spec_sha256=digest(spec),directions_sha256=result['directions_sha256'],model=engine.meta,
  prompt=question,outputs=outputs,completed_at=time.time(),note='Follow-up requested individual explanations; same previously saved directions, same Chinese explanation prompt, no textual principles in the target prompt.'))

if __name__=='__main__':main()

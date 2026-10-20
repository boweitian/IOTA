"""Single-item exploratory, task-general semantic guidance and activation contrasts."""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from classification_sweep import ClassificationEngine
from anli_diagnose import SOURCE
from sentiment_navigator import read,digest
from sentiment_prepare import write,sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/anli-semantic-guidance-petition-v1'
AXES=[
 dict(id='ordinary_meaning',positive='Prefer the natural, common interpretation of language. Use an unusual or specialized interpretation only when the context supports it.',negative='Prefer unusual or specialized interpretations of language even when the context does not support them.'),
 dict(id='stated_constraints',positive='Respect the restrictions actually stated or implied in context. Do not invent extra requirements or narrow a statement with unsupported conditions.',negative='Add extra requirements and narrow statements using conditions that are not stated or implied in context.'),
 dict(id='semantic_equivalence',positive='Judge whether the meaning follows, allowing ordinary paraphrases and reasonable semantic transformations. Do not require identical wording, and do not infer unsupported facts.',negative='Require identical wording rather than equivalent meaning. Reject ordinary paraphrases and reasonable semantic transformations when the words differ.')]
CONTEXTS=[
 'Compare a short description with a proposed paraphrase.',
 'Interpret a brief everyday statement in its surrounding context.',
 'Decide what a short message implies about an event.',
 'Evaluate a claim about the contents of a paragraph.',
 'Understand a short news passage and a reformulation of it.',
 'Compare an instruction with a description of what it requires.',
 'Assess whether a summary preserves the meaning of a conversation.',
 'Identify what follows from a short description of a situation.']
ZH='请用中文简要说明支持你选择的关键文本依据，最多两句话；然后另起一行给出最终选项，格式为 Final: A、Final: B 或 Final: C。'

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((SOURCE/'dataset-manifest.json').read_text());model=json.loads((SOURCE/'search-protocol.json').read_text())['objective']
 spec=dict(model=model,source_items_sha256=sha(SOURCE/'feasible-space.jsonl'),axes=AXES,fit_contexts=CONTEXTS,
  target_uid='05140765-980f-4d3d-acbb-7ebb4525fd67',layer=16,scale=float(json.loads((SOURCE/'search-atlas.json').read_text())['scale']),
  extraction='Last prompt-token positive-minus-negative instruction states, averaged over 8 abstract contexts per axis; each direction normalized to the original Search 1x norm.',
  cases=['baseline','prompt_only']+[f'{a["id"]}:{dose:+g}' for a in AXES for dose in [1.,2.]]+[f'combined:{dose:+g}' for dose in [-2.,-1.,1.,2.]],
  explanatory_cases=['baseline','prompt_only','combined:+1','combined:+2'],max_new_tokens=256,
  regularization='Active-direction projection cap: max absolute projection <= max absolute requested coefficient; common shrink factor.',
  direct_prompt=manifest['prompt_template'],explanation_instruction=ZH,source_sha256=sha(Path(__file__)),
  limitation='Exploratory on one already inspected training item. Principles supplied after observing this error; no claim of held-out generalization. No target text, answer label, or target-specific wording is used to fit directions. No model weights or original geometry files changed.')
 p=OUT/'spec.json'
 if p.exists() and json.loads(p.read_text())!=spec:raise ValueError('Changed specification')
 write(p,spec);return spec

def mixture(directions,coefficients):
 raw=coefficients@directions;active=np.flatnonzero(coefficients);limit=float(abs(coefficients).max())
 projections=[float(raw@directions[i]/(directions[i]@directions[i])) for i in active]
 peak=max(map(abs,projections),default=0.);factor=min(1.,limit/peak) if peak else 1.
 shift=raw*factor
 assert max((abs(float(shift@directions[i]/(directions[i]@directions[i]))) for i in active),default=0.)<=limit+1e-5
 return shift,dict(requested_coefficients=coefficients.tolist(),effective_coefficients=(coefficients*factor).tolist(),factor=factor,norm=float(np.linalg.norm(shift)),raw_projections=projections)

def main():
 spec=prepare();m=json.loads((SOURCE/'dataset-manifest.json').read_text());engine=ClassificationEngine(spec['model'],m);write(OUT/'model.json',engine.meta)
 row=next(r for r in read(SOURCE/'feasible-space.jsonl') if r['uid']==spec['target_uid'])
 directions=[];captures={};stability={}
 for a in AXES:
  pos=[];neg=[]
  for pole,values in [('positive',pos),('negative',neg)]:
   questions=[c+'\n\nResponse policy: '+a[pole] for c in CONTEXTS]
   for lo in range(0,len(questions),4):
    _,last=engine.measure(questions[lo:lo+4]);values.extend(last)
  pos=np.array(pos);neg=np.array(neg);diff=pos-neg;raw=diff.mean(0);unit=raw/np.linalg.norm(raw)
  directions.append(unit*spec['scale']);captures[a['id']+'_positive']=pos;captures[a['id']+'_negative']=neg
  first=diff[:4].mean(0);second=diff[4:].mean(0)
  stability[a['id']]=dict(raw_norm=float(np.linalg.norm(raw)),split_half_cosine=float(first@second/(np.linalg.norm(first)*np.linalg.norm(second))))
 directions=np.array(directions);np.savez_compressed(OUT/'directions.npz',directions=directions,**captures);write(OUT/'extraction.json',dict(stability=stability,gram=(directions@directions.T/spec['scale']**2).tolist(),directions_sha256=sha(OUT/'directions.npz')))
 guidance='\nInterpretation principles:\n'+'\n'.join(f'{i+1}. '+a['positive'] for i,a in enumerate(AXES))+'\n'
 results=[]
 for name in spec['cases']:
  coefficients=np.zeros(3)
  if name.startswith('combined:'):coefficients[:]=float(name.split(':')[1])
  elif name not in ['baseline','prompt_only']:
   axis,dose=name.split(':');coefficients[[a['id'] for a in AXES].index(axis)]=float(dose)
  shift,info=mixture(directions,coefficients)
  direct=m['prompt_template']
  if name=='prompt_only':direct=direct.replace('Respond with exactly one letter:',guidance+'Respond with exactly one letter:')
  engine.manifest=dict(m,prompt_template=direct)
  measured=engine.score([row],shift);generation=engine.generate([engine.question(row)],[shift],max_new_tokens=8)[0]
  prediction=int(measured['probabilities'][0].argmax());assert generation['text'].strip()=='ABC'[prediction]
  value=dict(condition=name,prediction=prediction,probabilities=measured['probabilities'][0].tolist(),direct_prompt=engine.question(row),generation=generation,steering=info,
   observed_shift_norm=float(measured['observed_shift_norm'][0]),residual_norm=float(measured['residual_norm'][0]))
  if name in spec['explanatory_cases']:
   engine.manifest=dict(m,prompt_template=direct.replace('Respond with exactly one letter: A, B, or C.',ZH))
   value['explanation_prompt']=engine.question(row);value['explanation_output']=engine.generate([engine.question(row)],[shift],max_new_tokens=256)[0]
  results.append(value);write(OUT/'progress.json',results);print(name,'ABC'[prediction],flush=True)
 original=np.array(row['probabilities'][0]);error=float(np.max(abs(np.array(results[0]['probabilities'])-original)))
 assert error<1e-3
 write(OUT/'result.json',dict(spec_sha256=digest(spec),model=engine.meta,item_id=row['id'],gold=row['gold'],results=results,
  baseline_reproduction_max_probability_error=error,completed_at=time.time(),directions_sha256=sha(OUT/'directions.npz')))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
 prepare() if args.prepare else main()

"""Post-hoc controls on frozen ANLI items; never replaces the benchmark score."""
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
from classification_sweep import ClassificationEngine, probabilities
from sentiment_prepare import sha, write, jsonl
from sentiment_navigator import read, digest

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/classification9b-anli-r3-100-v1'
OUT=ROOT/'runs/classification9b-anli-r3-diagnostics-v1'
EXTRA=('Use ordinary language understanding and basic world knowledge, including calendar facts, '
       'word meanings and properties of physical objects. A hypothesis can be an ordinary '
       'implication rather than a verbatim quotation. Distinguish a contradiction from an '
       'additional unsupported detail. Check names, roles, dates, counts and qualifiers carefully.\n')

def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    m=json.loads((SOURCE/'dataset-manifest.json').read_text());original=m['prompt_template']
    conditions=[dict(id='original_recheck',order=[0,1,2],prompt=original,kind='forward')]
    defs=original.split('\n')[1:4]
    for k,order in enumerate(([1,2,0],[2,0,1]),1):
        lines=['ABC'[j]+defs[semantic][1:] for j,semantic in enumerate(order)]
        prompt='\n'.join([original.split('\n')[0],*lines,*original.split('\n')[4:]])
        conditions.append(dict(id=f'cyclic_mapping_{k}',order=order,prompt=prompt,kind='forward'))
    expanded=original.replace('Respond with exactly one letter:',EXTRA+'Respond with exactly one letter:')
    conditions.append(dict(id='explicit_nli_guidance',order=[0,1,2],prompt=expanded,kind='forward'))
    explanation=original.replace('Respond with exactly one letter: A, B, or C.',
        'Explain the key evidence in at most three short sentences. Then end with exactly one line: Final: A, Final: B, or Final: C.')
    conditions.append(dict(id='explanation_then_answer',order=[0,1,2],prompt=explanation,kind='generate',max_new_tokens=256))
    spec=dict(source=str(SOURCE),source_items_sha256=sha(SOURCE/'feasible-space.jsonl'),source_summary_sha256=sha(SOURCE/'sweep-summary.json'),
        model=json.loads((SOURCE/'search-protocol.json').read_text())['objective'],source_code_sha256=sha(Path(__file__)),n=100,
        conditions=conditions,scope='Post-hoc exploratory diagnostics on all original 100 items. No selection of best prompt, relabeling, or replacement of original benchmark metrics. No steering in controls. Same FP32 weights, thinking disabled. Generated explanations are new outputs, not evidence of internal reasoning in the original one-token answer.')
    path=OUT/'spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Diagnostic specification changed')
    write(path,spec)
    print('PREPARED',path,flush=True)
    return spec

def run():
    spec=prepare();rows=read(SOURCE/'feasible-space.jsonl');manifest=json.loads((SOURCE/'dataset-manifest.json').read_text())
    engine=ClassificationEngine(spec['model'],manifest);write(OUT/'model.json',engine.meta)
    for condition in spec['conditions']:
        path=OUT/(condition['id']+'.json')
        if path.exists():continue
        write(OUT/'status.json',dict(phase='running',condition=condition['id'],updated_at=time.time()))
        engine.manifest=dict(manifest,prompt_template=condition['prompt'])
        if condition['kind']=='forward':
            result=probabilities(engine,rows,np.zeros(engine.dim))
            result={key:value.tolist() for key,value in result.items()}
            result['predictions']=[condition['order'][int(v)] for v in np.argmax(result['probabilities'],axis=1)]
        else:
            outputs=[]
            for lo in range(0,len(rows),4):
                batch=rows[lo:lo+4]
                values=engine.generate([engine.question(r) for r in batch],[np.zeros(engine.dim) for _ in batch],max_new_tokens=condition['max_new_tokens'])
                for row,value in zip(batch,values):
                    matches=re.findall(r'^\s*Final:\s*([ABC])\s*$',value['text'].strip(),re.M)
                    prediction='ABC'.index(matches[0]) if len(matches)==1 and value['text'].strip().endswith('Final: '+matches[0]) and not value['truncated'] else None
                    outputs.append(dict(id=row['id'],prediction=prediction,**value))
                jsonl(OUT/'explanation-progress.jsonl',outputs)
                print('explanation',len(outputs),'/100',flush=True)
            result=dict(outputs=outputs,predictions=[r['prediction'] for r in outputs])
        result.update(condition=condition['id'],spec_sha256=digest(spec),ids=[r['id'] for r in rows]);write(path,result)
        print('COMPLETE',condition['id'],flush=True)
    write(OUT/'status.json',dict(phase='complete',updated_at=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()

"""Separate, prefix-verified 512-token control for the frozen 24 anchored settings."""
import argparse,json
from pathlib import Path
import numpy as np
from anchored_precision import OUT as PARENT
from classification_generation import DEFAULT,ClassificationEngine,traced_generate
from sentiment_navigator import read,digest
from sentiment_prepare import write,jsonl,sha
from answer_format import parse_answer

OUT=PARENT/'length512-v1'

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'outputs').mkdir(exist_ok=True)
    parent=json.loads((PARENT/'spec.json').read_text())
    spec=dict(parent_spec_sha256=digest(parent),source_sha256=sha(Path(__file__)),max_new_tokens=512,batch_size=4,
        outputs_sha256={str(j):sha(PARENT/'outputs'/f'action-{j:02d}.jsonl') for j in range(24)},
        method='Reuse actual completed EOS outputs. Replay each original batch of four containing a truncated output with the same model/prompt/shift and 512-token limit. Require every old token prefix to match exactly, including completed members of replayed batches. No new directions or selection after this control.',
        limitation='Separate length-budget protocol on the same already-inspected 100 items; compare against its own extended precision:-2 anchor, never against 256-token anchor alone.')
    path=OUT/'spec.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:write(path,spec)
    return spec,parent

def worker(shard):
    spec,parent=prepare();manifest=json.loads((DEFAULT/'dataset-manifest.json').read_text());engine=None;rows=read(DEFAULT/'train.jsonl')+read(DEFAULT/'test-inputs.jsonl');shifts=np.load(PARENT/'shifts.npz')['shifts']
    for j in list(range(24))[shard::5]:
        target=OUT/'outputs'/f'action-{j:02d}.jsonl'
        if target.exists():continue
        source=read(PARENT/'outputs'/f'action-{j:02d}.jsonl');values=[dict(v,source_spec_sha256=v['spec_sha256'],spec_sha256=digest(spec),replayed_for_512=False) for v in source]
        for lo in range(0,100,4):
            if not any(v['truncated'] for v in source[lo:lo+4]):continue
            if engine is None:
                engine=ClassificationEngine(parent['model'],manifest);write(OUT/f'model-{shard}.json',engine.meta)
            batch=rows[lo:lo+4];generated=traced_generate(engine,[engine.question(r) for r in batch],shifts[j],512)
            for k,(row,v,old) in enumerate(zip(batch,generated,source[lo:lo+4])):
                assert v['token_ids'][:len(old['token_ids'])]==old['token_ids'],(j,lo+k,'prefix changed')
                values[lo+k]=dict(id=row['id'],action_index=j,spec_sha256=digest(spec),source_spec_sha256=old['spec_sha256'],replayed_for_512=True,source_prefix_length=len(old['token_ids']),**v,**parse_answer(v['text'],['A','B','C'],v['truncated']))
            print(shard,j,'replayed batch',lo,flush=True)
        jsonl(target,values);print('COMPLETE',j,flush=True)

def summarize():
    spec,parent=prepare();gold=np.array([r['gold'] for r in read(DEFAULT.parent/'classification9b-anli-r3-100-v1/feasible-space.jsonl')]);reference=read(OUT/'outputs/action-00.jsonl');base=np.array([v['prediction']==g for v,g in zip(reference,gold)]);results={};actual=0
    for j in range(24):
        source=read(PARENT/'outputs'/f'action-{j:02d}.jsonl');values=read(OUT/'outputs'/f'action-{j:02d}.jsonl');assert len(values)==100
        for old,v in zip(source,values):
            assert v['id']==old['id'] and v['spec_sha256']==digest(spec)
            assert parse_answer(v['text'],['A','B','C'],v['truncated'])=={k:v[k] for k in ['prediction','format_valid','error']}
            if v['replayed_for_512']:
                assert v['token_ids'][:len(old['token_ids'])]==old['token_ids'];actual+=1
                assert abs(v['steering_trace']['observed_min']-parent['actions'][j]['norm'])<1e-3 and abs(v['steering_trace']['observed_max']-parent['actions'][j]['norm'])<1e-3
            else:assert v['token_ids']==old['token_ids'] and not old['truncated']
        ok=np.array([v['prediction']==g for v,g in zip(values,gold)]);oldok=np.array([v['prediction']==g for v,g in zip(source,gold)])
        results[str(j)]=dict(id=parent['actions'][j]['id'],name=parent['actions'][j]['name'],correct=int(ok.sum()),train_correct=int(ok[:70].sum()),test_correct=int(ok[70:].sum()),invalid_format=sum(not v['format_valid'] for v in values),fixed=int(sum(ok&~base)),damaged=int(sum(~ok&base)),fixed_samples=[int(i+1) for i in np.flatnonzero(ok&~base)],damaged_samples=[int(i+1) for i in np.flatnonzero(~ok&base)],vs256_fixed=int(sum(ok&~oldok)),vs256_damaged=int(sum(~ok&oldok)),replayed=sum(v['replayed_for_512'] for v in values),output_sha256=sha(OUT/'outputs'/f'action-{j:02d}.jsonl'))
    assert actual==288
    result=dict(passed=True,complete=True,spec_sha256=digest(spec),actual_new_generations=actual,cached_eos_outputs=2400-actual,anchor_correct=int(base.sum()),best_observed=max(range(24),key=lambda j:(results[str(j)]['correct'],-j)),results=results,limitation=spec['limitation'])
    write(OUT/'summary.json',result)
    for j,v in results.items():print(j,v['name'],v['correct'],'invalid',v['invalid_format'],'fixed',v['fixed'],'damaged',v['damaged'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.summarize:summarize()
    elif a.worker is not None:worker(a.worker)
    else:p.error('Choose prepare, worker or summarize')

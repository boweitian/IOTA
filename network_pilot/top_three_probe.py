"""Replay the sum of the three best observed single-direction configurations."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from classification_generation import DEFAULT, ClassificationEngine, traced_generate
from answer_format import parse_answer
from sentiment_navigator import read, bank, digest
from sentiment_prepare import write, jsonl, sha
from studio import mixed_shift, AXES

OUT=DEFAULT/'top-three-mixture-v1'

def prepare():
    OUT.mkdir(exist_ok=True)
    parent=json.loads((DEFAULT/'generation-spec.json').read_text())
    summary=json.loads((DEFAULT/'generation-summary.json').read_text())
    atlas=json.loads((DEFAULT/'search-atlas.json').read_text());directions=np.load(DEFAULT/'search-geometry.npz')['directions']
    actions=bank(directions,float(atlas['scale']))
    eligible=list(range(1,37))+[49,50,51,52]
    selected=sorted(eligible,key=lambda j:(-summary['results'][str(j)]['splits']['all']['correct'],j))[:3]
    # The observed top three are original axes, with no ties at the selection boundary.
    assert all(j<=36 for j in selected)
    coeff=np.sum([actions[j]['coefficients'] for j in selected],axis=0)
    profile=dict(coefficients=dict(zip(AXES,coeff.tolist())))
    shift,regularization=mixed_shift(profile,directions,float(atlas['scale'])*max(abs(coeff)),regularization='direction_projection')
    assert np.allclose(shift,np.sum([actions[j]['shift'] for j in selected],axis=0)*regularization['cap_factor'])
    assert regularization['max_projection']<=max(abs(coeff))+1e-6
    spec=dict(parent_spec_sha256=digest(parent),source_sha256=sha(Path(__file__)),selection='Top three single direction+dose configurations by accuracy on all 100 already-inspected items; ties use lower original action index.',
        selected=[dict(index=j,id=parent['actions'][j]['id'],correct=summary['results'][str(j)]['splits']['all']['correct']) for j in selected],
        regularization=regularization,vector_sha256=hashlib.sha256(shift.tobytes()).hexdigest(),batch_size=4,max_new_tokens=256,
        model=parent['model'],prompt_template=parent['prompt_template'],scoring=parent['scoring'],intervention=parent['intervention'],
        limitation='Selected with all 100 labels, including test labels. Exploratory combination result, not fresh held-out evaluation.')
    if (OUT/'spec.json').exists():assert json.loads((OUT/'spec.json').read_text())==spec
    else:write(OUT/'spec.json',spec);np.savez_compressed(OUT/'shift.npz',shift=shift)
    return spec,shift

def run():
    spec,shift=prepare();m=json.loads((DEFAULT/'dataset-manifest.json').read_text())
    if (OUT/'outputs.jsonl').exists():return
    engine=ClassificationEngine(spec['model'],m);write(OUT/'model.json',engine.meta)
    rows=read(DEFAULT/'train.jsonl')+read(DEFAULT/'test-inputs.jsonl');values=[]
    for lo in range(0,100,4):
        batch=rows[lo:lo+4];generated=traced_generate(engine,[engine.question(r) for r in batch],shift,256)
        for row,v in zip(batch,generated):values.append(dict(id=row['id'],spec_sha256=digest(spec),**v,**parse_answer(v['text'],m['label_tokens'],v['truncated'])))
        jsonl(OUT/'outputs.partial',values);print('top_three',len(values),'/100',flush=True)
    (OUT/'outputs.partial').replace(OUT/'outputs.jsonl')

def summarize():
    spec,shift=prepare();rows=read(DEFAULT.parent/'classification9b-anli-r3-100-v1/feasible-space.jsonl');values=read(OUT/'outputs.jsonl');base=read(DEFAULT/'outputs/action-00.jsonl');best=read(DEFAULT/'outputs'/f"action-{spec['selected'][0]['index']:02d}.jsonl")
    assert len(values)==100 and [v['id'] for v in values]==[r['id'] for r in rows]
    for v in values:
        assert v['spec_sha256']==digest(spec)
        assert parse_answer(v['text'],['A','B','C'],v['truncated'])=={k:v[k] for k in ['prediction','format_valid','error']}
        t=v['steering_trace'];assert abs(t['observed_min']-np.linalg.norm(shift))<1e-3 and abs(t['observed_max']-np.linalg.norm(shift))<1e-3
    ok=np.array([v['prediction']==r['gold'] for v,r in zip(values,rows)])
    result=dict(passed=True,spec_sha256=digest(spec),selected=spec['selected'],regularization=spec['regularization'],correct=int(ok.sum()),train_correct=int(ok[:70].sum()),test_correct=int(ok[70:].sum()),invalid_format=sum(not v['format_valid'] for v in values),outputs_sha256=sha(OUT/'outputs.jsonl'),limitation=spec['limitation'])
    for name,reference in [('baseline',base),('best_single',best)]:
        old=np.array([v['prediction']==r['gold'] for v,r in zip(reference,rows)])
        result['vs_'+name]=dict(reference_correct=int(old.sum()),fixed=int(sum(ok&~old)),damaged=int(sum(~ok&old)),fixed_samples=[i+1 for i in np.flatnonzero(ok&~old)],damaged_samples=[i+1 for i in np.flatnonzero(~ok&old)])
    write(OUT/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare()[0],ensure_ascii=False,indent=2))
    elif a.summarize:summarize()
    else:run();summarize()

"""Local exploration around the empirically best precision:-2 activation shift."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from classification_generation import DEFAULT,ClassificationEngine,traced_generate
from sentiment_navigator import read,digest,bank
from sentiment_prepare import write,jsonl,sha
from answer_format import parse_answer
from studio import mixed_shift,AXES as ORIGINAL_AXES

OUT=DEFAULT/'precision-anchor-v1'
NEIGHBORS=[
 dict(id='flexible_rules',name='灵活解释',positive='理解角色设定和场景时把握大意，灵活处理要求，不拘泥于逐条照搬。',negative='严格逐条执行角色设定与场景中的每个要求和细节，不允许灵活解释。'),
 dict(id='meaning_equivalence',name='语义等价',positive='根据自然常见的含义理解文本，允许合理的同义改写；不要因措辞不同或额外臆造的限制而否定意思相同的表达。',negative='要求措辞完全一致，加入额外的苛刻限制；即使通常意思相同也拒绝合理的同义改写。'),
 dict(id='relation_scope',name='事实关系保持',positive='保持文本中人物、对象、时间和数量的对应关系；保留原有的限定范围，不把局部属性转给整体或另一个对象。',negative='根据大致印象任意转移人物、对象、时间和数量；忽略限定范围，将局部属性随意推广到整体。'),
]
CONTEXTS=[
 'Compare a short description with a proposed paraphrase.',
 'Interpret a brief everyday statement in its surrounding context.',
 'Decide what a short message implies about an event.',
 'Evaluate a claim about the contents of a paragraph.',
 'Understand a short news passage and a reformulation of it.',
 'Compare an instruction with a description of what it requires.',
 'Assess whether a summary preserves the meaning of a conversation.',
 'Identify what follows from a short description of a situation.',
]

def model_spec():return json.loads((DEFAULT/'generation-spec.json').read_text())

def design():
    OUT.mkdir(exist_ok=True);(OUT/'outputs').mkdir(exist_ok=True);(OUT/'workers').mkdir(exist_ok=True)
    value=dict(parent_spec_sha256=digest(model_spec()),source_sha256=sha(Path(__file__)),neighbors=NEIGHBORS,fit_contexts=CONTEXTS,
        native_additions=[dict(id='both_010',decomposition=.1,verification_fraction=.1),dict(id='both_025',decomposition=.25,verification_fraction=.25),dict(id='both_050',decomposition=.5,verification_fraction=.5),dict(id='decomposition_025',decomposition=.25,verification_fraction=0.),dict(id='verification_025',decomposition=0.,verification_fraction=.25)],
        new_doses=[-.25,.25,.5],random_seed=2026102002,random_doses=[-.25,.25],
        anchor='Original precision:-2, not a new semantic claim. Anchor-preserving mode keeps projection on original precision exactly -2, removes parallel component from additions, and caps only the residual increment if another active projection exceeds 2.',
        extraction='Mean positive-minus-negative final prompt-token states over 8 abstract contexts; normalize each raw direction to original 1x norm; no task examples, labels or sample-specific strings in extraction prompts.',
        limitation='All 100 items have been inspected; anchor and error hypotheses use these labels. Exploratory adaptive search, not independent held-out performance.')
    p=OUT/'design.json'
    if p.exists():assert json.loads(p.read_text())==value
    else:write(p,value)
    return value

def fit():
    d=design()
    if (OUT/'directions.npz').exists():return
    parent=model_spec();manifest=json.loads((DEFAULT/'dataset-manifest.json').read_text());engine=ClassificationEngine(parent['model'],manifest)
    write(OUT/'extraction-model.json',engine.meta)
    native=np.load(DEFAULT/'search-geometry.npz')['directions'];scale=float(np.linalg.norm(native[2]));vectors=[];captures={};stats={}
    for axis in d['neighbors']:
        poles={}
        for pole in ['positive','negative']:
            questions=[c+'\n\nResponse policy: '+axis[pole] for c in d['fit_contexts']];states=[]
            for lo in range(0,8,4):
                _,last=engine.measure(questions[lo:lo+4]);states.extend(last)
            poles[pole]=np.array(states);captures[axis['id']+'_'+pole]=poles[pole]
        differences=poles['positive']-poles['negative'];raw=differences.mean(0);direction=raw/np.linalg.norm(raw)*scale;vectors.append(direction)
        first=differences[:4].mean(0);second=differences[4:].mean(0)
        stats[axis['id']]=dict(raw_norm=float(np.linalg.norm(raw)),split_half_cosine=float(first@second/(np.linalg.norm(first)*np.linalg.norm(second))),cosine_with_negative_precision=float(direction@(-native[2])/(np.linalg.norm(direction)*np.linalg.norm(native[2]))))
        print('EXTRACTED',axis['id'],stats[axis['id']],flush=True)
    np.savez_compressed(OUT/'directions.npz',directions=np.array(vectors),**captures)
    write(OUT/'extraction.json',dict(design_sha256=digest(d),directions_sha256=sha(OUT/'directions.npz'),statistics=stats))

def projection(shift,direction):return float(shift@direction/(direction@direction))

def anchor_addition(anchor,primary,delta,active):
    # Preserve the original anchor's complete parallel component, without re-normalizing the residual.
    removed=projection(delta,primary);residual=delta-removed*primary;factor=1.
    for _,vector in active:
        start=projection(anchor,vector);slope=projection(residual,vector)
        assert abs(start)<=2+1e-5
        if slope>1e-12:factor=min(factor,(2-start)/slope)
        elif slope< -1e-12:factor=min(factor,(-2-start)/slope)
    factor=max(0.,min(1.,factor));shift=anchor+factor*residual
    assert abs(projection(shift,primary)+2)<1e-5
    projections={name:projection(shift,v) for name,v in [('precision',primary)]+active}
    assert max(abs(v) for v in projections.values())<=2+1e-5
    return shift,dict(mode='anchor_preserving',removed_parallel_coefficient=removed,increment_factor=factor,increment_norm=float(np.linalg.norm(factor*residual)),anchor_projection=projection(shift,primary),active_projections=projections)

def prepare():
    d=design();native=np.load(DEFAULT/'search-geometry.npz')['directions'].astype(float);atlas=json.loads((DEFAULT/'search-atlas.json').read_text());scale=float(atlas['scale']);primary=native[2];anchor=-2*primary
    new=np.load(OUT/'directions.npz')['directions'].astype(float)
    extraction=json.loads((OUT/'extraction.json').read_text());assert extraction['design_sha256']==digest(d) and extraction['directions_sha256']==sha(OUT/'directions.npz')
    actions=[dict(id='anchor_replay',name='规则精确−2× 复跑',shift=anchor,mode='anchor_replay',anchor_projection=-2.)]
    for mode in ['common_scale','anchor_preserving']:
        for addition in d['native_additions']:
            a=addition['decomposition'];b=-2*addition['verification_fraction'];delta=a*native[0]+b*native[1]
            active=[(ORIGINAL_AXES[j],native[j]) for j,c in [(0,a),(1,b)] if c]
            if mode=='anchor_preserving':shift,info=anchor_addition(anchor,primary,delta,active)
            else:
                coeff={k:0. for k in ORIGINAL_AXES};coeff.update(precision=-2.,decomposition=a,verification=b)
                shift,cap=mixed_shift(dict(coefficients=coeff),native,2*scale,regularization='direction_projection')
                info=dict(mode=mode,common_factor=cap['cap_factor'],anchor_projection=projection(shift,primary),active_projections=cap['projections'])
            actions.append(dict(id=mode+':'+addition['id'],name=('固定基座' if mode=='anchor_preserving' else '共同缩放')+f'；分解{a:+g}，验证{b:+g}',shift=shift,decomposition=a,verification=b,**info))
    for axis,vector in zip(d['neighbors'],new):
        for dose in d['new_doses']:
            shift,info=anchor_addition(anchor,primary,dose*vector,[(axis['id'],vector)])
            actions.append(dict(id=axis['id']+f':{dose:+g}',name='固定基座＋'+axis['name']+f'{dose:+g}×',shift=shift,axis=axis['id'],dose=dose,**info))
    rng=np.random.default_rng(d['random_seed']);randoms=rng.normal(size=(2,4096));randoms*=scale/np.linalg.norm(randoms,axis=1,keepdims=True)
    for i,vector in enumerate(randoms):
        for dose in d['random_doses']:
            shift,info=anchor_addition(anchor,primary,dose*vector,[(f'random{i}',vector)])
            actions.append(dict(id=f'random{i}:{dose:+g}',name=f'固定基座＋随机{i+1} {dose:+g}×',shift=shift,axis=f'random{i}',dose=dose,**info))
    serial=[dict({k:v for k,v in a.items() if k!='shift'},norm=float(np.linalg.norm(a['shift'])),vector_sha256=hashlib.sha256(a['shift'].tobytes()).hexdigest()) for a in actions]
    spec=dict(design_sha256=digest(d),directions_sha256=sha(OUT/'directions.npz'),actions=serial,batch_size=4,max_new_tokens=256,model=model_spec()['model'],prompt_template=model_spec()['prompt_template'],scoring=model_spec()['scoring'],limitation=d['limitation'])
    p=OUT/'spec.json'
    if p.exists():assert json.loads(p.read_text())==spec
    else:write(p,spec);np.savez_compressed(OUT/'shifts.npz',shifts=np.array([a['shift'] for a in actions]))
    return spec,[a['shift'] for a in actions]

def worker(shard,workers):
    spec,shifts=prepare();manifest=json.loads((DEFAULT/'dataset-manifest.json').read_text());engine=ClassificationEngine(spec['model'],manifest);write(OUT/'workers'/f'model-{shard}.json',engine.meta)
    rows=read(DEFAULT/'train.jsonl')+read(DEFAULT/'test-inputs.jsonl')
    for j in list(range(len(shifts)))[shard::workers]:
        target=OUT/'outputs'/f'action-{j:02d}.jsonl'
        if target.exists():continue
        partial=target.with_suffix('.partial');values=read(partial) if partial.exists() else []
        for i,v in enumerate(values):assert v['id']==rows[i]['id'] and v['spec_sha256']==digest(spec)
        for lo in range(len(values),100,4):
            batch=rows[lo:lo+4];generated=traced_generate(engine,[engine.question(r) for r in batch],shifts[j],256)
            for row,v in zip(batch,generated):values.append(dict(id=row['id'],action_index=j,spec_sha256=digest(spec),**v,**parse_answer(v['text'],manifest['label_tokens'],v['truncated'])))
            jsonl(partial,values);print(shard,spec['actions'][j]['id'],len(values),flush=True)
        partial.replace(target)

def summarize():
    spec,shifts=prepare();rows=read(DEFAULT.parent/'classification9b-anli-r3-100-v1/feasible-space.jsonl');gold=np.array([r['gold'] for r in rows]);anchor=read(DEFAULT/'outputs/action-09.jsonl');old=np.array([v['prediction']==r['gold'] for v,r in zip(anchor,rows)]);results={}
    for j,action in enumerate(spec['actions']):
        path=OUT/'outputs'/f'action-{j:02d}.jsonl'
        if not path.exists():continue
        values=read(path);assert len(values)==100 and [v['id'] for v in values]==[r['id'] for r in rows]
        for v in values:
            assert v['spec_sha256']==digest(spec)
            assert parse_answer(v['text'],['A','B','C'],v['truncated'])=={k:v[k] for k in ['prediction','format_valid','error']}
            t=v['steering_trace'];assert abs(t['observed_min']-action['norm'])<1e-3 and abs(t['observed_max']-action['norm'])<1e-3
        pred=np.array([-1 if v['prediction'] is None else v['prediction'] for v in values]);ok=pred==gold
        results[str(j)]=dict(id=action['id'],name=action['name'],correct=int(ok.sum()),train_correct=int(ok[:70].sum()),test_correct=int(ok[70:].sum()),invalid_format=int(sum(pred<0)),fixed=int(sum(ok&~old)),damaged=int(sum(~ok&old)),fixed_samples=[int(i+1) for i in np.flatnonzero(ok&~old)],damaged_samples=[int(i+1) for i in np.flatnonzero(~ok&old)],predictions=pred.tolist(),output_sha256=sha(path))
        if j==0:assert all(v['token_ids']==a['token_ids'] for v,a in zip(values,anchor))
    result=dict(complete=len(results)==len(spec['actions']),spec_sha256=digest(spec),anchor_correct=int(old.sum()),completed_actions=len(results),total_actions=len(spec['actions']),measurements=len(results)*100,results=results,extraction=json.loads((OUT/'extraction.json').read_text()),limitation=spec['limitation'])
    if result['complete']:
        result['best_observed']=max(range(len(spec['actions'])),key=lambda j:(results[str(j)]['correct'],-j))
        result['train_selected']=max(range(len(spec['actions'])),key=lambda j:(results[str(j)]['train_correct'],-j))
    write(OUT/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=5);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.fit:fit();prepare()
    elif a.prepare:prepare()
    elif a.worker is not None:worker(a.worker,a.workers)
    elif a.summarize:summarize()
    else:p.error('Choose fit, prepare, worker, or summarize')

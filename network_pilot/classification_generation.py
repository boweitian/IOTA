"""Full generated-answer validation, preserving old immutable experiments."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
PILOT=Path('/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot')
sys.path.append(str(PILOT))
from classification_sweep import ClassificationEngine
from sentiment_navigator import read,digest,bank
from sentiment_prepare import write,jsonl,sha
from answer_format import analysis_prompt,parse_answer
ROOT=Path(__file__).resolve().parent
SOURCE=PILOT/'runs/classification9b-anli-r3-100-v1'
SEMANTIC=PILOT/'runs/anli-semantic-guidance-petition-v1'
DEFAULT=ROOT/'runs/classification9b-anli-r3-analysis-100-v1'

def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in ['outputs','workers','source-code']:(out/name).mkdir(exist_ok=True)
    original=json.loads((SOURCE/'dataset-manifest.json').read_text())
    names=['search-geometry.npz','search-atlas.json','geometry-binding.json','search-protocol.json','train.jsonl','test-inputs.jsonl','test-labels.jsonl']
    for name in names:
        if not (out/name).exists():shutil.copy2(SOURCE/name,out/name)
        assert sha(out/name)==sha(SOURCE/name)
    atlas=json.loads((out/'search-atlas.json').read_text());actions=bank(np.load(out/'search-geometry.npz')['directions'],float(atlas['scale']))
    d=np.load(SEMANTIC/'directions.npz')['directions'][0]
    for dose in [-2.,-1.,1.,2.]:actions.append(dict(id=f'ordinary_meaning:{dose:+g}',shift=d*dose,coefficients=[dose],family='ordinary_meaning',dose=dose))
    semantic=json.loads((SEMANTIC/'spec.json').read_text())
    actions.append(dict(id='principles_in_prompt',shift=np.zeros(4096),coefficients=[0.],family='text_prompt',principles=[a['positive'] for a in semantic['axes']]))
    for i,a in enumerate(actions):a.setdefault('family','baseline' if i==0 else 'original_single' if i<=36 else 'original_mixture')
    serial=[{k:v for k,v in a.items() if k!='shift'} for a in actions]
    template=analysis_prompt(original['prompt_template'],original['label_tokens'])
    m=dict(original,kind='classification_generation',artifact_key='anli-r3-analysis',prompt_template=template,calibration_count=0,
        limitations='Already inspected 70 train / 30 test items; exploratory diagnostics, not fresh held-out accuracy.',files={name:sha(out/name) for name in names})
    spec=dict(model=json.loads((out/'search-protocol.json').read_text())['objective'],actions=serial,prompt_template=template,letters=m['label_tokens'],
        batch_size=4,max_new_tokens=256,source_items_sha256=sha(SOURCE/'feasible-space.jsonl'),ordinary_direction_sha256=sha(SEMANTIC/'directions.npz'),
        old_summary_sha256=sha(SOURCE/'sweep-summary.json'),source_sha256={name:sha(ROOT/name) for name in ['classification_generation.py','answer_format.py']},
        scoring='Full greedy FP32 generation, TF32 disabled; parse generated final 答案：X line. No next-token classification softmax.',
        intervention='block16 residual: last prompt position and last position at every decode step; original directions and projection cap unchanged',
        train_selection='Fixed in advance: highest train correct among original36 singles and 4 common-meaning singles; ties prefer lower index. Test labels never enter selection.',caution=m['limitations'])
    path=out/'generation-spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Frozen generation specification changed')
    if not path.exists():write(path,spec);write(out/'dataset-manifest.json',m);np.savez_compressed(out/'ordinary-direction.npz',direction=d)
    for name in spec['source_sha256']:
        if not (out/'source-code'/name).exists():shutil.copy2(ROOT/name,out/'source-code'/name)
    return m,spec,[a['shift'] for a in actions]

def template_for(m,action):
    prefix=''
    if action['family']=='text_prompt':prefix='Interpretation principles:\n'+'\n'.join(f'{i+1}. {v}' for i,v in enumerate(action['principles']))+'\n'
    return prefix+m['prompt_template']

def traced_generate(engine,questions,shift,max_tokens):
    delta=engine.torch.as_tensor(shift,device=engine.device,dtype=engine.model.dtype);trace=[]
    def hook(module,args,output):
        h=output.clone();h[:,-1,:]+=delta
        trace.append(dict(norm=(h[:,-1,:]-output[:,-1,:]).float().norm(dim=-1).detach().cpu().tolist(),residual=output[:,-1,:].float().norm(dim=-1).detach().cpu().tolist()))
        return h
    handle=engine.block.register_forward_hook(hook)
    try:values=engine.generate(questions,[np.zeros(engine.dim)]*len(questions),max_new_tokens=max_tokens)
    finally:handle.remove()
    assert trace;norm=float(np.linalg.norm(shift))
    for i,value in enumerate(values):
        seen=[t['norm'][i] for t in trace]
        if max(abs(v-norm) for v in seen)>1e-3:raise ValueError('Incorrect observed steering')
        value['steering_trace']=dict(requested_norm=norm,observed_min=min(seen),observed_max=max(seen),prefill_residual_norm=trace[0]['residual'][i],forward_calls=len(trace))
    return values

def worker(out,shard,workers):
    out=Path(out);m,spec,shifts=prepare(out);engine=None
    order=[0,51,52,50,49,53]+list(range(1,49));indices=order[shard::workers]
    rows=read(out/'train.jsonl')+read(out/'test-inputs.jsonl')
    for j in indices:
        target=out/'outputs'/f'action-{j:02d}.jsonl'
        if target.exists():continue
        if engine is None:
            engine=ClassificationEngine(spec['model'],m)
            write(out/'workers'/f'model-{shard}.json',dict(engine.meta,intervention=spec['intervention'],scoring=spec['scoring']))
        engine.manifest=dict(m,prompt_template=template_for(m,spec['actions'][j]));partial=target.with_suffix('.partial');cached=read(partial) if partial.exists() else []
        for i,v in enumerate(cached):assert v['id']==rows[i]['id'] and v['spec_sha256']==digest(spec)
        for lo in range(len(cached),len(rows),spec['batch_size']):
            batch=rows[lo:lo+spec['batch_size']];values=traced_generate(engine,[engine.question(r) for r in batch],shifts[j],spec['max_new_tokens'])
            for row,v in zip(batch,values):cached.append(dict(id=row['id'],action_index=j,spec_sha256=digest(spec),**v,**parse_answer(v['text'],spec['letters'],v['truncated'])))
            jsonl(partial,cached);write(out/'workers'/f'worker-{shard}.json',dict(action_index=j,action=spec['actions'][j]['id'],completed=len(cached),total=100,updated_at=time.time()));print(shard,spec['actions'][j]['id'],len(cached),'/100',flush=True)
        partial.replace(target)
    write(out/'workers'/f'worker-{shard}.json',dict(phase='complete',actions=indices,updated_at=time.time()))

def metric(rows,outputs,baseline):
    gold=np.array([r['gold'] for r in rows]);pred=np.array([-1 if v['prediction'] is None else v['prediction'] for v in outputs]);old=np.array(baseline);ok=pred==gold;old_ok=old==gold
    return dict(n=len(rows),correct=int(ok.sum()),accuracy=float(ok.mean()),invalid_format=int((pred<0).sum()),fixed=int((ok&~old_ok).sum()),damaged=int((~ok&old_ok).sum()),
        confusion=[[int(((gold==i)&(pred==j)).sum()) for j in range(3)] for i in range(3)],predictions=pred.tolist())

def summarize(out,require_complete=False):
    out=Path(out);spec=json.loads((out/'generation-spec.json').read_text());rows=read(SOURCE/'feasible-space.jsonl');old=[r['baseline_prediction'] for r in rows];data={}
    for j,a in enumerate(spec['actions']):
        path=out/'outputs'/f'action-{j:02d}.jsonl'
        if not path.exists():continue
        values=read(path);assert len(values)==100 and [v['id'] for v in values]==[r['id'] for r in rows]
        for v in values:
            assert v['spec_sha256']==digest(spec);assert parse_answer(v['text'],spec['letters'],v['truncated'])=={k:v[k] for k in ['prediction','format_valid','error']}
        data[j]=values
    complete=len(data)==len(spec['actions'])
    if require_complete and not complete:raise ValueError(f'Only {len(data)} / {len(spec["actions"])} complete')
    if 0 not in data:return
    baseline=[-1 if v['prediction'] is None else v['prediction'] for v in data[0]]
    results={str(j):dict(id=spec['actions'][j]['id'],family=spec['actions'][j]['family'],splits={s:metric(rows[lo:hi],values[lo:hi],baseline[lo:hi]) for s,lo,hi in [('all',0,100),('train',0,70),('test',70,100)]},vs_old_protocol=metric(rows,values,old)) for j,values in data.items()}
    summary=dict(complete=complete,completed_actions=len(data),total_actions=len(spec['actions']),measurements=len(data)*100,spec_sha256=digest(spec),actions=spec['actions'],results=results,
        original_baseline_correct=sum(r['gold']==r['baseline_prediction'] for r in rows),output_files_sha256={f'outputs/action-{j:02d}.jsonl':sha(out/'outputs'/f'action-{j:02d}.jsonl') for j in data},limitations=spec['caution'])
    if complete:
        eligible=list(range(1,37))+list(range(49,53));summary['train_selected_single']=max(eligible,key=lambda j:(results[str(j)]['splits']['train']['correct'],-j));summary['best_fixed_single']=max(eligible,key=lambda j:(results[str(j)]['splits']['all']['correct'],-j))
        gold=np.array([r['gold'] for r in rows]);matrix=np.array([results[str(j)]['splits']['all']['predictions'] for j in range(len(spec['actions']))]).T;summary['coverage']={}
        for name,ix in [('original_singles',list(range(1,37))),('all_singles',eligible),('original_all',list(range(49))),('all_activation',list(range(53)))]:
            counts=(matrix[:,ix]==gold[:,None]).sum(1);summary['coverage'][name]=dict(correct=int((counts>0).sum()),per_item_solutions=counts.tolist())
    write(out/'generation-summary.json',summary)
    print(json.dumps(dict(complete=complete,completed_actions=len(data),conditions={v['id']:{k:v['splits']['all'][k] for k in ['correct','fixed','damaged','invalid_format']} for v in results.values()}),ensure_ascii=False,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--require-complete',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.prepare:prepare(a.out)
    elif a.summarize:summarize(a.out,a.require_complete)
    elif a.worker is not None:worker(a.out,a.worker,a.workers)
    else:p.error('Choose --prepare, --worker, or --summarize')

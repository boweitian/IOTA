"""Calibrated short-answer classification with a fixed geometry action bank."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from sentiment_prepare import ROOT,LAB,sha,write,jsonl
from sentiment_navigator import read,digest,bank
from sentiment_sweep_analysis import split_summary
sys.path.insert(0,str(LAB))
from engine import Engine

class ClassificationEngine(Engine):
    def __init__(self,protocol,manifest):
        super().__init__(protocol,'cuda');self.manifest=manifest
        self.model.float();self.torch.cuda.empty_cache();self.torch.backends.cuda.matmul.allow_tf32=False;self.torch.backends.cudnn.allow_tf32=False
        self.letters=manifest['label_tokens'];tokens=[self.tokenizer.encode(s,add_special_tokens=False) for s in self.letters]
        if any(len(t)!=1 for t in tokens):raise ValueError('Answer labels must be single tokens')
        self.label_tokens=[t[0] for t in tokens]
        self.meta.update(dtype=str(self.model.dtype),label_tokens=self.letters,label_token_ids=self.label_tokens,thinking=False,
            intervention='block 16 output, last input token, single full-prefix forward; no reasoning rollout',precision='FP32; TF32 disabled')
    def question(self,row):
        return self.manifest['prompt_template'].format(**{k:json.dumps(row[k],ensure_ascii=False) for k in self.manifest['prompt_fields']})
    def score(self,rows,shift):
        inputs=self.tokenizer([self.prefix(self.question(r)) for r in rows],return_tensors='pt',padding=True,add_special_tokens=False).to(self.device)
        delta=self.torch.as_tensor(shift,device=self.device,dtype=self.model.dtype);captured=[]
        def hook(module,args,output):
            h=output.clone();h[:,-1,:]+=delta
            captured.append((output[:,-1,:].norm(dim=-1).detach(),(h[:,-1,:]-output[:,-1,:]).norm(dim=-1).detach()));return h
        handle=self.block.register_forward_hook(hook)
        try:
            with self.torch.inference_mode():
                result=self.model(**inputs,use_cache=False,logits_to_keep=1);logits=result.logits[:,-1,:].float();values=logits[:,self.label_tokens]
                p=values.softmax(-1);mass=(values.logsumexp(-1)-logits.logsumexp(-1)).exp()
                if len(captured)!=1:raise ValueError('Hook must execute once')
        finally:handle.remove()
        return dict(probabilities=p.cpu().numpy(),logits=values.cpu().numpy(),label_mass=mass.cpu().numpy(),
            residual_norm=captured[0][0].cpu().numpy(),observed_shift_norm=captured[0][1].cpu().numpy())
    def actual(self,rows):
        values=self.generate([self.question(r) for r in rows],[np.zeros(self.dim) for _ in rows],max_new_tokens=8)
        for value in values:
            text=value['text'].strip();value['prediction']=self.letters.index(text) if text in self.letters else None
        return [dict(id=r['id'],**v) for r,v in zip(rows,values)]

def probabilities(engine,rows,shift):
    parts=[engine.score(rows[lo:lo+4],shift) for lo in range(0,len(rows),4)]
    return {k:np.concatenate([part[k] for part in parts]) for k in parts[0]}

def generated(engine,rows):return [v for lo in range(0,len(rows),4) for v in engine.actual(rows[lo:lo+4])]

def finish(out,manifest,spec):
    atlas=json.loads((out/'search-atlas.json').read_text());nodes=[dict(id=c['id'],name=c['name']) for c in atlas['concepts']];actions=spec['actions']
    matrices={};items=[];all_gen=[];all_mass=[]
    for split in ['train','test']:
        rows=read(out/('train.jsonl' if split=='train' else 'test-inputs.jsonl'))
        if split=='test':
            gold={r['id']:r['label'] for r in read(out/'test-labels.jsonl')};rows=[dict(r,label=gold[r['id']]) for r in rows]
        caches=[np.load(out/(split+'-cache')/f'agent-{j:02d}.npz') for j in range(49)]
        for cache in caches:
            if str(cache['spec_sha256'])!=digest(spec) or cache['ids'].tolist()!=[r['id'] for r in rows]:raise ValueError('Cache binding mismatch')
        p=np.stack([c['probabilities'] for c in caches],1);y=np.array([r['label'] for r in rows]);matrices[split]=(p,y)
        all_gen.extend(read(out/(split+'-generation.jsonl')));all_mass.extend(caches[0]['label_mass'].tolist())
        for i,r in enumerate(rows):
            pred=p[i].argmax(1);feasible=[j for j in range(1,49) if pred[j]==y[i]]
            items.append(dict(id=r['id'],uid=r['uid'],text=r['text'],context=r['context'],hypothesis=r['hypothesis'],source_split=r['source_split'],split=split,
                gold=int(y[i]),baseline_prediction=int(pred[0]),predictions=pred.tolist(),probabilities=p[i].tolist(),feasible=feasible,feasible_single=[j for j in feasible if j<=36]))
    selected=json.loads((out/'frozen-selection.json').read_text())['selected_index'];matrices['all']=(np.concatenate([matrices[s][0] for s in ['train','test']]),np.concatenate([matrices[s][1] for s in ['train','test']]))
    splits={s:split_summary(p,y,actions,nodes,selected) for s,(p,y) in matrices.items()};jsonl(out/'feasible-space.jsonl',items)
    format_check=dict(generated=len(all_gen),invalid_format=sum(v['prediction'] is None for v in all_gen),agreement_with_scored=sum(v['prediction']==items[i]['baseline_prediction'] for i,v in enumerate(all_gen)),
        min_label_mass=float(min(all_mass)),median_label_mass=float(np.median(all_mass)))
    if format_check['invalid_format'] or format_check['agreement_with_scored']!=100:raise ValueError('Generation / scoring disagreement')
    summary=dict(complete=True,benchmark=manifest['benchmark'],labels=manifest['labels'],display_labels=manifest['display_labels'],label_tokens=manifest['label_tokens'],
        spec_sha256=digest(spec),geometry_sha256=spec['geometry_sha256'],model=json.loads((out/'model.json').read_text()),actions=actions,splits=splits,
        train_selected_single=selected,measurement=spec['scoring'],format_validation=format_check,calibration=json.loads((out/'calibration-result.json').read_text()),
        items_sha256=sha(out/'feasible-space.jsonl'),files={str(p.relative_to(out)):sha(p) for split in ['train','test'] for p in (out/(split+'-cache')).glob('agent-*.npz')},
        interpretation='100-item fixed-seed official train/test pilot. A separate dev set selects task difficulty; no test items were selected by model correctness. Retrospective best and oracle coverage are label-aware.')
    write(out/'sweep-summary.json',summary)
    report=['# '+manifest['benchmark']+'：独立难度校准与逐题全方向扫描','',manifest['task_description'],'',
        '数据来自官方 R3 train/dev/test。独立 dev 30 题只做难度与格式校准，主实验 train 70、test 30；130 条前提与 UID 均互不重复。注释 reason、model_label 不进入模型。',
        '固定 Qwen3.5-9B、Search 的九方向、block 16 最后输入 token、FP32、关闭 TF32。A=entailment、B=neutral、C=contradiction；非思考直接分类，不是 CoT 或官方完整测试分数。','',
        '| 范围 | Baseline | 事后最佳固定单方向 | 至少一个单方向正确 | 训练选择方向 |','|---|---|---|---|---|']
    for key,s in splits.items():
        def fmt(v):return f"{v['correct']}/{s['n']} ({v['accuracy']:.1%})"
        report.append(f"| {key} | {fmt(s['baseline'])} | {fmt(s['best_fixed_single'])} {s['best_fixed_single']['id']} | {fmt(s['coverage']['single_directions'])} | {fmt(s['train_selected_single'])} {s['train_selected_single']['id']} |")
    report+=['','| 范围 | 正向平均 | Baseline | 负向平均 |','|---|---|---|---|']
    for key,s in splits.items():report.append(f"| {key} | {s['trend']['positive']['mean_accuracy']:.2%} | {s['baseline']['accuracy']:.2%} | {s['trend']['negative']['mean_accuracy']:.2%} |")
    report+=['',f"校准集 baseline：{summary['calibration']['correct']}/30；预先接受区间 50%–85%。主实验真实生成与评分一致 {format_check['agreement_with_scored']}/100，格式无效 {format_check['invalid_format']}。",
        '49 个配置 × 100 题全部测量。逐题至少一个正确是已测网格的标签可见覆盖，不代表已学会路由；事后最佳也不等于训练选择的测试性能。']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n');print('\n'.join(report),flush=True)

def main(out):
    out=Path(out);m=json.loads((out/'dataset-manifest.json').read_text())
    for name,value in m['files'].items():
        if sha(out/name)!=value:raise ValueError('Frozen input changed: '+name)
    atlas=json.loads((out/'search-atlas.json').read_text());protocol=json.loads((out/'search-protocol.json').read_text());directions=np.load(out/'search-geometry.npz')['directions']
    actions=bank(directions,float(atlas['scale']));serial=[{k:v for k,v in a.items() if k!='shift'} for a in actions]
    files=['classification_sweep.py','anli_prepare.py','sentiment_navigator.py','sentiment_sweep_analysis.py']
    spec=dict(model=protocol['objective'],benchmark=m['benchmark'],dataset_sha256=sha(out/'dataset-manifest.json'),geometry_sha256=sha(out/'search-geometry.npz'),
        actions=serial,prompt_template=m['prompt_template'],prompt_fields=m['prompt_fields'],calibration_acceptance=m['calibration_acceptance'],
        scoring='FP32; TF32 disabled; block 16 output; last input token only; next-token A/B/C softmax; no reasoning rollout',source_sha256={name:sha(ROOT/name) for name in files})
    if (out/'sweep-spec.json').exists() and json.loads((out/'sweep-spec.json').read_text())!=spec:raise ValueError('Changed frozen specification')
    write(out/'sweep-spec.json',spec);src=out/'source-code';src.mkdir(exist_ok=True)
    for name in files:
        if not (src/name).exists():(src/name).write_bytes((ROOT/name).read_bytes())
    if (out/'sweep-summary.json').exists():return
    def status(phase,**extras):write(out/'status.json',dict(phase=phase,updated_at=time.time(),**extras))
    status('loading_model');engine=ClassificationEngine(protocol['objective'],m);write(out/'model.json',engine.meta)
    calibration_path=out/'calibration-result.json'
    if not calibration_path.exists():
        status('difficulty_calibration');rows=read(out/'calibration.jsonl');p=probabilities(engine,rows,np.zeros(engine.dim));answers=generated(engine,rows);prediction=p['probabilities'].argmax(1);y=np.array([r['label'] for r in rows])
        batch=p['probabilities'][:4];single=np.concatenate([engine.score([r],np.zeros(engine.dim))['probabilities'] for r in rows[:4]])
        error=float(np.max(abs(batch-single)));numerical=error<1e-3 and np.array_equal(batch.argmax(1),single.argmax(1));write(out/'numerical-audit.json',dict(passed=bool(numerical),max_probability_error=error))
        jsonl(out/'calibration-generations.jsonl',answers);np.savez_compressed(out/'calibration-cache.npz',**p,ids=[r['id'] for r in rows],spec_sha256=digest(spec))
        accuracy=float(np.mean(prediction==y));agreement=sum(a['prediction']==int(prediction[i]) for i,a in enumerate(answers));invalid=sum(a['prediction'] is None for a in answers)
        passed=m['calibration_acceptance']['min_accuracy']<=accuracy<=m['calibration_acceptance']['max_accuracy'] and agreement==30 and invalid==0 and numerical
        write(calibration_path,dict(correct=int(np.sum(prediction==y)),n=30,accuracy=accuracy,agreement_with_scored=agreement,invalid_format=invalid,
            passed=bool(passed),min_label_mass=float(p['label_mass'].min()),spec_sha256=digest(spec),completed_at=time.time(),ids=[r['id'] for r in rows]))
    calibration=json.loads(calibration_path.read_text());print('CALIBRATION',json.dumps(calibration),flush=True)
    if not calibration['passed']:status('calibration_rejected',calibration=calibration);return
    for split in ['train','test']:
        rows=read(out/('train.jsonl' if split=='train' else 'test-inputs.jsonl'));cache=out/(split+'-cache');cache.mkdir(exist_ok=True)
        for j,a in enumerate(actions):
            target=cache/f'agent-{j:02d}.npz'
            if target.exists():
                with np.load(target) as v:
                    if str(v['spec_sha256'])!=digest(spec):raise ValueError('Stale cache')
                continue
            status('all_direction_measurements',split=split,completed=j,total=49,agent=a['id']);values=probabilities(engine,rows,a['shift'])
            temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,**values,ids=[r['id'] for r in rows],spec_sha256=digest(spec));temp.replace(target)
            print(split,j+1,'/49',a['id'],flush=True)
        path=out/(split+'-generation.jsonl')
        if not path.exists():jsonl(path,generated(engine,rows))
        expected=np.load(cache/'agent-00.npz')['probabilities'].argmax(1)
        if [v['prediction'] for v in read(path)]!=expected.tolist():raise ValueError('Baseline generation mismatch')
        if split=='train' and not (out/'frozen-selection.json').exists():
            y=np.array([r['label'] for r in rows]);correct=[int(np.sum(np.load(cache/f'agent-{j:02d}.npz')['probabilities'].argmax(1)==y)) for j in range(49)]
            selected=max(range(1,37),key=lambda j:(correct[j],-j));write(out/'frozen-selection.json',dict(selected_index=selected,selected_id=actions[selected]['id'],train_correct=correct[selected],
                frozen_at=time.time(),spec_sha256=digest(spec),train_ids=[r['id'] for r in rows]))
    predictions=[]
    for split in ['train','test']:
        rows=read(out/('train.jsonl' if split=='train' else 'test-inputs.jsonl'));p=np.stack([np.load(out/(split+'-cache')/f'agent-{j:02d}.npz')['probabilities'] for j in range(49)],1)
        predictions.extend(dict(id=r['id'],split=split,probabilities=p[i].tolist()) for i,r in enumerate(rows))
    jsonl(out/'predictions-before-evaluation-labels.jsonl',predictions);finish(out,m,spec);status('complete',measurements=4900,actions=49)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)

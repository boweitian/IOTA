"""Independent ANLI source/label, calibration, response-grid and score audit."""
import json
import zipfile
from pathlib import Path
import numpy as np
from sentiment_prepare import ROOT,sha,write
from sentiment_navigator import read,digest,bank
from anli_prepare import OUT,key

def main():
    out=OUT;m=json.loads((out/'dataset-manifest.json').read_text());spec=json.loads((out/'sweep-spec.json').read_text());s=json.loads((out/'sweep-summary.json').read_text())
    assert digest(spec)==s['spec_sha256'] and sha(out/'dataset-manifest.json')==spec['dataset_sha256']
    assert sha(out/'source/anli_v1.0.zip')==m['source_sha256']
    for name,value in m['files'].items():assert sha(out/name)==value,name
    for name,value in spec['source_sha256'].items():assert sha(out/'source-code'/name)==value,name
    z=zipfile.ZipFile(out/'source/anli_v1.0.zip');raw={split:[json.loads(line) for line in z.read(f'anli_v1.0/R3/{split}.jsonl').decode().splitlines()] for split in ['train','dev','test']}
    assert {k:len(v) for k,v in raw.items()}==m['source_counts']
    calibration=read(out/'calibration.jsonl');training=read(out/'train.jsonl');inputs=read(out/'test-inputs.jsonl');gold={r['id']:r['label'] for r in read(out/'test-labels.jsonl')}
    testing=[dict(r,label=gold[r['id']]) for r in inputs];all_rows=calibration+training+testing
    assert len(all_rows)==130 and len({r['uid'] for r in all_rows})==130 and len({key(r['context']) for r in all_rows})==130
    for r in all_rows:
        original=raw[r['source_split']][r['source_index']]
        assert original['uid']==r['uid'] and original['context']==r['context'] and original['hypothesis']==r['hypothesis']
        assert m['raw_labels'].index(original['label'])==r['label']
        assert 'reason' not in r and 'model_label' not in r
    assert [r['source_split'] for r in calibration]==['dev']*30
    assert [r['source_split'] for r in training]==['train']*70 and [r['source_split'] for r in testing]==['test']*30
    assert m['prompt_fields']==['context','hypothesis']
    assert np.bincount([r['label'] for r in calibration],minlength=3).tolist()==[10,10,10]
    assert np.bincount([r['label'] for r in training],minlength=3).tolist()==[24,23,23]
    assert np.bincount([r['label'] for r in testing],minlength=3).tolist()==[10,10,10]
    model=json.loads((out/'model.json').read_text());cr=json.loads((out/'calibration-result.json').read_text())
    with np.load(out/'calibration-cache.npz') as cache:
        assert str(cache['spec_sha256'])==digest(spec) and cache['ids'].tolist()==[r['id'] for r in calibration]
        cp=cache['probabilities'].argmax(1);cy=np.array([r['label'] for r in calibration]);assert cr['correct']==int(np.sum(cp==cy))
    assert cr['passed'] and .5<=cr['accuracy']<=.85
    cg=read(out/'calibration-generations.jsonl');assert [r['prediction'] for r in cg]==cp.tolist()
    directions=np.load(out/'search-geometry.npz')['directions'];atlas=json.loads((out/'search-atlas.json').read_text());actions=bank(directions,float(atlas['scale']))
    assert [{k:v for k,v in a.items() if k!='shift'} for a in actions]==spec['actions']
    rows=read(out/'feasible-space.jsonl');assert sha(out/'feasible-space.jsonl')==s['items_sha256']
    matrices={};earliest_test=None;earliest_train=None
    for split,rr in [('train',training),('test',testing)]:
        values=[]
        for j,a in enumerate(actions):
            path=out/(split+'-cache')/f'agent-{j:02d}.npz';assert sha(path)==s['files'][str(path.relative_to(out))]
            with np.load(path) as cache:
                assert str(cache['spec_sha256'])==digest(spec) and cache['ids'].tolist()==[r['id'] for r in rr]
                p=cache['probabilities'];l=cache['logits'];q=np.exp(l-l.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
                assert p.shape==(len(rr),3) and np.isfinite(p).all();np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-7)
                norm=float(np.linalg.norm(np.array(a['shift'],dtype=np.float32)));np.testing.assert_allclose(cache['observed_shift_norm'],norm,rtol=1e-5,atol=1e-5)
                values.append(p.copy())
            if split=='test':earliest_test=path.stat().st_mtime if earliest_test is None else min(earliest_test,path.stat().st_mtime)
            else:earliest_train=path.stat().st_mtime if earliest_train is None else min(earliest_train,path.stat().st_mtime)
        p=np.stack(values,1);y=np.array([r['label'] for r in rr]);matrices[split]=(p,y);exported=[r for r in rows if r['split']==split]
        assert [r['id'] for r in exported]==[r['id'] for r in rr]
        for i,r in enumerate(exported):
            assert r['gold']==rr[i]['label'];np.testing.assert_array_equal(r['probabilities'],p[i]);pred=p[i].argmax(1).tolist();assert r['predictions']==pred
            assert r['feasible']==[j for j in range(1,49) if pred[j]==y[i]] and r['feasible_single']==[j for j in range(1,37) if pred[j]==y[i]]
        answers=read(out/(split+'-generation.jsonl'));assert [a['id'] for a in answers]==[r['id'] for r in rr]
        for i,a in enumerate(answers):
            assert a['prediction']==int(p[i,0].argmax()) and a['text'].strip()==m['label_tokens'][a['prediction']]
            assert a['token_ids'][0]==model['label_token_ids'][a['prediction']]
    assert cr['completed_at']<=earliest_train
    selected=json.loads((out/'frozen-selection.json').read_text());p,y=matrices['train'];index=max(range(1,37),key=lambda j:(int(np.sum(p[:,j].argmax(1)==y)),-j))
    assert index==selected['selected_index']==s['train_selected_single'] and selected['frozen_at']<=earliest_test
    matrices['all']=(np.concatenate([matrices[k][0] for k in ['train','test']]),np.concatenate([matrices[k][1] for k in ['train','test']]))
    for split,(p,y) in matrices.items():
        v=s['splits'][split];c=p.argmax(2)==y[:,None];base=c[:,0]
        assert v['baseline']['correct']==int(base.sum()) and v['best_fixed_single']['correct']==int(c[:,1:37].sum(0).max())
        assert v['train_selected_single']['correct']==int(c[:,index].sum())
        for name,ix in [('single_directions',list(range(1,37))),('all_configurations',list(range(1,49)))]:
            count=c[:,ix].sum(1);cover=count>0;entry=v['coverage'][name]
            assert entry['correct']==int(cover.sum()) and entry['with_baseline_correct']==int((cover|base).sum())
            assert entry['histogram']==[dict(correct_configurations=k,samples=int(np.sum(count==k))) for k in range(len(ix)+1)]
        for j,a in enumerate(v['action_metrics']):
            assert a['correct']==int(c[:,j].sum()) and a['paired_vs_baseline']['wins']==int((c[:,j]&~base).sum()) and a['paired_vs_baseline']['losses']==int((~c[:,j]&base).sum())
        for name,sign in [('positive',1),('negative',-1)]:
            ix=[j for j in range(1,37) if np.sign(sum(spec['actions'][j]['coefficients']))==sign];t=v['trend'][name]
            assert abs(t['mean_accuracy']-c[:,ix].mean())<1e-12
            assert t['fix_events']==int((c[:,ix]&~base[:,None]).sum()) and t['break_events']==int((~c[:,ix]&base[:,None]).sum())
    before=read(out/'predictions-before-evaluation-labels.jsonl');assert [r['id'] for r in before]==[r['id'] for r in rows]
    for a,b in zip(before,rows):np.testing.assert_array_equal(a['probabilities'],b['probabilities'])
    assert json.loads((out/'numerical-audit.json').read_text())['passed']
    report=dict(passed=True,benchmark=m['benchmark'],samples=100,calibration_samples=30,measurements=4900,calibration_correct=cr['correct'],
        checks=['official JSONL contexts/hypotheses/labels/UIDs','disjoint calibration/train/test premises','no annotator reasons or model answers in prompt fields',
                'difficulty and format acceptance precedes main scan','all 4900 intervention displacements','probabilities reconstructed from logits',
                '130 actual generations match scoring','training direction frozen before test','three split metrics and coverage independently reconstructed','signed correction and damage counts'],
        summary_sha256=sha(out/'sweep-summary.json'),items_sha256=sha(out/'feasible-space.jsonl'))
    write(out/'audit.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Independently reconstruct official row labels, intervention cells and all scores."""
import json
import zipfile
from pathlib import Path
import numpy as np
from sentiment_prepare import sha,write
from sentiment_navigator import read,digest,bank
from sst2_prepare import OUT,canonical

def main():
    out=OUT;manifest=json.loads((out/'dataset-manifest.json').read_text());spec=json.loads((out/'sweep-spec.json').read_text());s=json.loads((out/'sweep-summary.json').read_text())
    assert sha(out/'source/SST-2.zip')==manifest['source_sha256']
    assert digest(spec)==s['spec_sha256'] and sha(out/'dataset-manifest.json')==spec['dataset_sha256']
    assert sha(out/'search-geometry.npz')==s['geometry_sha256']==spec['geometry_sha256']
    for name,value in manifest['files'].items():assert sha(out/name)==value,name
    for name,value in spec['source_sha256'].items():assert sha(out/'source-code'/name)==value,name
    archive=zipfile.ZipFile(out/'source/SST-2.zip')
    source={}
    for split,filename in [('train','train.tsv'),('validation','dev.tsv')]:
        lines=archive.read('SST-2/'+filename).decode().splitlines();assert lines[0]=='sentence\tlabel'
        source[split]=[dict(zip(['sentence','label'],line.rsplit('\t',1))) for line in lines[1:]]
    rows=read(out/'feasible-space.jsonl');assert len(rows)==100
    assert sha(out/'feasible-space.jsonl')==s['items_sha256']
    assert len({canonical(r['text']) for r in rows})==100
    excluded={r['text'] for r in read(out/'excluded-prior-texts.jsonl')}
    assert not excluded & {canonical(r['text']) for r in rows}
    expected_rows={r['id']:r for r in read(out/'train.jsonl')}
    for r in read(out/'test-inputs.jsonl'):expected_rows[r['id']]=r
    for r in rows:
        raw=source[r['source_split']][expected_rows[r['id']]['source_index']]
        assert raw['sentence']==r['text'] and int(raw['label'])==r['gold']
        assert r['source_split']==('train' if r['split']=='train' else 'validation')
    directions=np.load(out/'search-geometry.npz')['directions'];atlas=json.loads((out/'search-atlas.json').read_text());actions=bank(directions,float(atlas['scale']))
    assert [{k:v for k,v in a.items() if k!='shift'} for a in actions]==spec['actions']
    matrices={};earliest_test=None
    model=json.loads((out/'model.json').read_text())
    for split,n in [('train',70),('test',30)]:
        rr=[r for r in rows if r['split']==split];assert len(rr)==n;y=np.array([r['gold'] for r in rr]);assert np.bincount(y).tolist()==[n//2,n//2]
        parts=[]
        for j,action in enumerate(actions):
            path=out/(split+'-cache')/f'agent-{j:02d}.npz';assert sha(path)==s['files'][str(path.relative_to(out))]
            with np.load(path) as cache:
                assert str(cache['spec_sha256'])==digest(spec) and cache['ids'].tolist()==[r['id'] for r in rr]
                p=cache['probabilities'];logits=cache['logits'];assert p.shape==(n,2)
                q=np.exp(logits-logits.max(1,keepdims=True));q/=q.sum(1,keepdims=True);np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-7)
                assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
                expected=float(np.linalg.norm(np.asarray(action['shift'],dtype=np.float32)))
                np.testing.assert_allclose(cache['observed_shift_norm'],expected,rtol=1e-5,atol=1e-5)
                parts.append(p.copy())
            if split=='test':earliest_test=path.stat().st_mtime if earliest_test is None else min(earliest_test,path.stat().st_mtime)
        p=np.stack(parts,1);matrices[split]=(p,y)
        for i,r in enumerate(rr):
            np.testing.assert_array_equal(r['probabilities'],p[i]);prediction=p[i].argmax(1).tolist();assert r['predictions']==prediction
            assert r['feasible']==[j for j in range(1,49) if prediction[j]==y[i]]
            assert r['feasible_single']==[j for j in range(1,37) if prediction[j]==y[i]]
        generated=read(out/(split+'-generation.jsonl'));assert [r['id'] for r in generated]==[r['id'] for r in rr]
        for i,g in enumerate(generated):
            assert g['text'].strip() in ['A','B'] and g['prediction']=='AB'.index(g['text'].strip())==int(p[i,0].argmax())
            assert g['token_ids'][0]==model['label_token_ids'][g['prediction']]
    frozen=json.loads((out/'frozen-selection.json').read_text());p,y=matrices['train'];selected=max(range(1,37),key=lambda j:(int(np.sum(p[:,j].argmax(1)==y)),-j))
    assert frozen['selected_index']==selected==s['train_selected_single'] and frozen['spec_sha256']==digest(spec)
    assert frozen['frozen_at']<=earliest_test
    matrices['all']=(np.concatenate([matrices[k][0] for k in ['train','test']]),np.concatenate([matrices[k][1] for k in ['train','test']]))
    for split,(p,y) in matrices.items():
        m=s['splits'][split];c=p.argmax(2)==y[:,None];base=c[:,0];n=len(y)
        assert m['baseline']['correct']==int(base.sum())
        assert m['best_fixed_single']['correct']==int(c[:,1:37].sum(0).max())
        assert m['train_selected_single']['correct']==int(c[:,selected].sum())
        for key,ix in [('single_directions',list(range(1,37))),('all_configurations',list(range(1,49)))]:
            counts=c[:,ix].sum(1);any_correct=counts>0;v=m['coverage'][key]
            assert v['correct']==int(any_correct.sum()) and v['with_baseline_correct']==int((any_correct|base).sum())
            assert v['rescued_baseline_errors']==int((any_correct&~base).sum())
            assert v['histogram']==[dict(correct_configurations=k,samples=int(np.sum(counts==k))) for k in range(len(ix)+1)]
        for j,v in enumerate(m['action_metrics']):
            assert v['correct']==int(c[:,j].sum())
            assert v['paired_vs_baseline']['wins']==int((c[:,j]&~base).sum())
            assert v['paired_vs_baseline']['losses']==int((~c[:,j]&base).sum())
        for key,sign in [('positive',1),('negative',-1)]:
            ix=[j for j in range(1,37) if np.sign(sum(spec['actions'][j]['coefficients']))==sign];v=m['trend'][key]
            assert len(ix)==18 and abs(v['mean_accuracy']-c[:,ix].mean())<1e-12
            assert v['fix_events']==int((c[:,ix]&~base[:,None]).sum()) and v['break_events']==int((~c[:,ix]&base[:,None]).sum())
        for pair in m['signed_pairs']:
            a=c[:,pair['positive']];b=c[:,pair['negative']];assert pair['comparison']['wins']==int((a&~b).sum()) and pair['comparison']['losses']==int((~a&b).sum())
    before=read(out/'predictions-before-evaluation-labels.jsonl')
    assert [r['id'] for r in before]==[r['id'] for r in rows]
    for a,b in zip(before,rows):np.testing.assert_array_equal(a['probabilities'],b['probabilities'])
    assert json.loads((out/'numerical-audit.json').read_text())['passed']
    result=dict(passed=True,samples=100,configurations=49,measurements=4900,source='official GLUE SST-2.zip',
        evaluation_split='official validation sampled to 30; not hidden official test',format_validation=s['format_validation'],
        checks=['official TSV rows, text, labels and split indices','balanced sampling and prior-text exclusion','49 action shifts observed in every response cell',
                'cache IDs and probabilities reconstructed from logits','training single frozen before validation measurements',
                'three split accuracies, feasible sets and histograms independently reconstructed','signed paired correction and damage counts',
                '100 actual generations match first-token scoring','predictions saved before evaluation-label analysis'],
        summary_sha256=sha(out/'sweep-summary.json'),items_sha256=sha(out/'feasible-space.jsonl'))
    write(out/'audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

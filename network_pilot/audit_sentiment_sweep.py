"""Reconstruct the sweep from response caches, independently of its analyzer/UI."""
import argparse
import json
from pathlib import Path
import numpy as np
from sentiment_prepare import sha, write
from sentiment_navigator import digest


def audit(out):
    out=Path(out);spec=json.loads((out/'sweep-spec.json').read_text());s=json.loads((out/'sweep-summary.json').read_text())
    assert digest(spec)==s['spec_sha256']
    assert sha(out/'search-geometry.npz')==s['geometry_sha256']
    assert sha(out/'feasible-space.jsonl')==s['items_sha256']
    source=Path(spec['source']);original=json.loads((source/'experiment-spec.json').read_text())
    assert digest(original)==spec['source_spec_sha256']
    assert sha(source/'frozen-policies.json')==spec['source_policies_sha256']
    assert spec['actions']==original['actions']
    records=[json.loads(line) for line in (out/'feasible-space.jsonl').read_text().splitlines()]
    singles=[j for j,a in enumerate(spec['actions']) if sum(v!=0 for v in a['coefficients'])==1]
    pos=[j for j in singles if sum(spec['actions'][j]['coefficients'])>0]
    neg=[j for j in singles if sum(spec['actions'][j]['coefficients'])<0]
    assert len(singles)==36 and len(pos)==len(neg)==18
    groups={};reused=measured=0
    for split,expected_n in [('train',70),('test',30)]:
        rows=[r for r in records if r['split']==split];assert len(rows)==expected_n
        inputs=[json.loads(line) for line in (out/('train.jsonl' if split=='train' else 'test-inputs.jsonl')).read_text().splitlines()]
        gold=inputs if split=='train' else [json.loads(line) for line in (out/'test-labels.jsonl').read_text().splitlines()]
        labels={r['id']:r['label'] for r in gold};y=np.array([labels[r['id']] for r in inputs]);prob=[]
        assert [r['id'] for r in rows]==[r['id'] for r in inputs]
        for j in range(49):
            path=out/(split+'-cache')/f'agent-{j:02d}.npz'
            assert sha(path)==s['files'][str(path.relative_to(out))]
            with np.load(path) as cache:
                assert cache['ids'].tolist()==[r['id'] for r in inputs]
                assert str(cache['sweep_spec_sha256'])==digest(spec)
                p=cache['probabilities'];logits=cache['logits'];q=np.exp(logits-logits.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
                np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-7)
                provenance=json.loads(str(cache['provenance']))
                if 'source' in provenance:
                    origin=Path(provenance['source']);assert sha(origin)==provenance['sha256']
                    with np.load(origin) as prior:
                        for key in ['probabilities','logits','features','label_mass']:np.testing.assert_array_equal(cache[key],prior[key])
                    reused+=len(inputs)
                else:measured+=len(inputs)
                prob.append(p.copy())
        p=np.stack(prob,1);groups[split]=(p,y)
        for i,r in enumerate(rows):
            assert r['gold']==int(y[i])
            np.testing.assert_array_equal(r['probabilities'],p[i])
            predicted=p[i].argmax(1).tolist();assert r['predictions']==predicted
            assert r['feasible']==[j for j in range(1,49) if predicted[j]==y[i]]
            assert r['feasible_single']==[j for j in singles if predicted[j]==y[i]]
            assert r['baseline_prediction']==predicted[0]
    groups['all']=(np.concatenate([groups[k][0] for k in ['train','test']]),np.concatenate([groups[k][1] for k in ['train','test']]))
    for split,(p,y) in groups.items():
        result=s['splits'][split];c=p.argmax(2)==y[:,None];b=c[:,0];n=len(y)
        assert result['n']==n and result['baseline']['correct']==int(b.sum())
        assert result['best_fixed_single']['correct']==int(c[:,singles].sum(0).max())
        assert result['best_fixed_any']['correct']==int(c[:,1:].sum(0).max())
        for key,indices in [('single_directions',singles),('all_configurations',list(range(1,49)))]:
            v=result['coverage'][key];counts=c[:,indices].sum(1);can=counts>0
            assert v['correct']==int(can.sum()) and v['with_baseline_correct']==int((can|b).sum())
            assert v['rescued_baseline_errors']==int((can & ~b).sum())
            assert v['never_solved_even_with_baseline']==int((~(can|b)).sum())
            assert v['histogram']==[dict(correct_configurations=k,samples=int(np.sum(counts==k))) for k in range(len(indices)+1)]
        for j,m in enumerate(result['action_metrics']):
            assert m['correct']==int(c[:,j].sum())
            assert m['paired_vs_baseline']['wins']==int((c[:,j]&~b).sum())
            assert m['paired_vs_baseline']['losses']==int((~c[:,j]&b).sum())
            assert m['changed_prediction']==int((p[:,j].argmax(1)!=p[:,0].argmax(1)).sum())
        for key,indices in [('positive',pos),('negative',neg)]:
            v=result['trend'][key]
            assert abs(v['mean_accuracy']-c[:,indices].mean())<1e-12
            assert v['fix_events']==int((c[:,indices]&~b[:,None]).sum())
            assert v['break_events']==int((~c[:,indices]&b[:,None]).sum())
        for pair in result['signed_pairs']:
            a=c[:,pair['positive']];z=c[:,pair['negative']]
            assert pair['comparison']['wins']==int((a&~z).sum())
            assert pair['comparison']['losses']==int((~a&z).sum())
    numerical=json.loads((out/'numerical-audit.json').read_text());assert numerical['passed']
    report=dict(passed=True,samples=100,configurations=49,total_measurements=4900,reused_measurements=reused,
                new_measurements=measured,original_frozen_policies_unchanged=True,
                checks=['all response cells and item order','reused arrays match source caches exactly','logits reproduce probabilities',
                        'all three split summaries independently reconstructed','finite-bank coverage and histograms',
                        'signed paired correction and damage counts','baseline reproduction','original fixed-policy results preserved'],
                sweep_summary_sha256=sha(out/'sweep-summary.json'),items_sha256=sha(out/'feasible-space.jsonl'),numerical_audit=numerical)
    assert reused+measured==4900 and measured==1200
    write(out/'audit.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);audit(parser.parse_args().out)

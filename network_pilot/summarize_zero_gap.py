"""Reconstruct scores and paired changes for zero-gap control generations."""
import json
import numpy as np
from diagnose_zero_gap import OUT, DEFAULT
from answer_format import parse_answer
from sentiment_navigator import read, digest
from sentiment_prepare import write, sha

def main():
    parent=json.loads((DEFAULT/'generation-spec.json').read_text())
    specs=[json.loads((OUT/f'spec-{i}.json').read_text()) for i in range(5)]
    assert all(s==specs[0] for s in specs)
    spec=specs[0]
    assert spec['parent_spec_sha256']==digest(parent)
    assert spec['source_sha256']==sha(OUT.parent.parent.parent/'diagnose_zero_gap.py')
    rows=read(DEFAULT.parent/'classification9b-anli-r3-100-v1/feasible-space.jsonl')
    gold=np.array([r['gold'] for r in rows]);base=read(DEFAULT/'outputs/action-00.jsonl')
    basepred=np.array([r['prediction'] if r['prediction'] is not None else -1 for r in base]);baseok=basepred==gold
    summary={}
    for a in spec['actions']:
        values=read(OUT/(a['id']+'.jsonl'))
        assert len(values)==100 and [v['id'] for v in values]==[r['id'] for r in rows]
        for v in values:
            assert v['spec_sha256']==digest(spec)
            assert parse_answer(v['text'],parent['letters'],v['truncated'])=={k:v[k] for k in ['prediction','format_valid','error']}
        pred=np.array([v['prediction'] if v['prediction'] is not None else -1 for v in values]);ok=pred==gold
        reference=read(DEFAULT/'outputs/action-03.jsonl') if a['id']=='decomposition_plus1_replay' else base
        summary[a['id']]=dict(correct=int(sum(ok)),train_correct=int(sum(ok[:70])),test_correct=int(sum(ok[70:])),
            fixed=int(sum(ok&~baseok)),damaged=int(sum(~ok&baseok)),invalid=int(sum(pred<0)),
            same_predictions_as_reference=sum(v['prediction']==b['prediction'] for v,b in zip(values,reference)),
            same_token_ids_as_reference=sum(v['token_ids']==b['token_ids'] for v,b in zip(values,reference)),
            goldwise=[int(sum(ok&(gold==k))) for k in range(3)],prediction_counts=[int(sum(pred==k)) for k in [-1,0,1,2]],
            output_sha256=sha(OUT/(a['id']+'.jsonl')))
    native=[]
    for j in list(range(1,37))+[49,50,51,52]:
        values=read(DEFAULT/'outputs'/f'action-{j:02d}.jsonl');pred=np.array([v['prediction'] if v['prediction'] is not None else -1 for v in values]);native.append(pred==gold)
    native=np.array(native);delta=native.mean(0)-baseok;rng=np.random.default_rng(52100)
    bootstrap=delta[rng.integers(0,100,size=(20000,100))].mean(1)
    result=dict(passed=True,parent_baseline_correct=int(sum(baseok)),controls=summary,
                original_nonzero_singles=dict(n=40,higher_than_zero=int(sum(native.sum(1)>sum(baseok))),mean_correct=float(native.sum(1).mean()),
                per_gold_mean_correct=[float(native[:,gold==k].sum(1).mean()) for k in range(3)],
                paired_item_bootstrap_improvement_interval=np.quantile(bootstrap,[.025,.975]).tolist()),
                interpretation_limit='Same 100 previously inspected items. Directions and signs share items and one baseline; 40 conditions are not independent replication. Isotropic random controls do not exhaust possible nonspecific perturbations.',
                spec_sha256=digest(spec))
    write(OUT/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Independent scoring of the frozen top-three probe; JSON-safe sample indices."""
import json
from pathlib import Path
import numpy as np
from classification_generation import DEFAULT
from sentiment_navigator import read,digest
from sentiment_prepare import write,sha
from answer_format import parse_answer
from studio import mixed_shift,AXES

def main():
    out=DEFAULT/'top-three-mixture-v1';spec=json.loads((out/'spec.json').read_text())
    assert spec['source_sha256']==sha(Path(__file__).with_name('top_three_probe.py'))
    parent=json.loads((DEFAULT/'generation-spec.json').read_text());assert spec['parent_spec_sha256']==digest(parent)
    shift=np.load(out/'shift.npz')['shift'];directions=np.load(DEFAULT/'search-geometry.npz')['directions']
    reg=spec['regularization'];reconstructed,info=mixed_shift(dict(coefficients=reg['coefficients']),directions,reg['cap'],regularization='direction_projection')
    assert np.array_equal(shift,reconstructed) and info==reg and reg['max_projection']<=2+1e-6
    rows=read(DEFAULT.parent/'classification9b-anli-r3-100-v1/feasible-space.jsonl');values=read(out/'outputs.jsonl')
    assert len(values)==100 and [v['id'] for v in values]==[r['id'] for r in rows]
    for v in values:
        assert v['spec_sha256']==digest(spec)
        assert parse_answer(v['text'],['A','B','C'],v['truncated'])=={k:v[k] for k in ['prediction','format_valid','error']}
        t=v['steering_trace'];assert abs(t['observed_min']-np.linalg.norm(shift))<1e-3 and abs(t['observed_max']-np.linalg.norm(shift))<1e-3
    ok=np.array([v['prediction']==r['gold'] for v,r in zip(values,rows)])
    result=dict(passed=True,spec_sha256=digest(spec),selected=spec['selected'],regularization=reg,correct=int(ok.sum()),train_correct=int(ok[:70].sum()),test_correct=int(ok[70:].sum()),invalid_format=sum(not v['format_valid'] for v in values),outputs_sha256=sha(out/'outputs.jsonl'),limitation=spec['limitation'])
    for name,index in [('baseline',0),('best_single',spec['selected'][0]['index'])]:
        reference=read(DEFAULT/'outputs'/f'action-{index:02d}.jsonl');old=np.array([v['prediction']==r['gold'] for v,r in zip(reference,rows)])
        result['vs_'+name]=dict(reference_correct=int(old.sum()),fixed=int(sum(ok&~old)),damaged=int(sum(~ok&old)),fixed_samples=[int(i+1) for i in np.flatnonzero(ok&~old)],damaged_samples=[int(i+1) for i in np.flatnonzero(~ok&old)])
    write(out/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Independent data/geometry/answer checks for the anchored local search."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from classification_generation import DEFAULT
from sentiment_prepare import sha,write
from sentiment_navigator import read,digest

def main():
    out=DEFAULT/'precision-anchor-v1';design=json.loads((out/'design.json').read_text());spec=json.loads((out/'spec.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['measurements']==2400 and summary['spec_sha256']==digest(spec)
    assert spec['design_sha256']==digest(design) and design['source_sha256']==sha(Path(__file__).with_name('anchored_precision.py'))
    parent=json.loads((DEFAULT/'generation-spec.json').read_text());assert design['parent_spec_sha256']==digest(parent) and spec['prompt_template']==parent['prompt_template']
    manifest=json.loads((DEFAULT/'dataset-manifest.json').read_text())
    for name,value in manifest['files'].items():assert sha(DEFAULT/name)==value
    assert spec['directions_sha256']==sha(out/'directions.npz')
    rows=read(DEFAULT.parent/'classification9b-anli-r3-100-v1/feasible-space.jsonl');gold=np.array([r['gold'] for r in rows]);base=read(DEFAULT/'outputs/action-09.jsonl');basepred=np.array([-1 if v['prediction'] is None else v['prediction'] for v in base]);baseok=basepred==gold
    shifts=np.load(out/'shifts.npz')['shifts'];native=np.load(DEFAULT/'search-geometry.npz')['directions'];p=native[2].astype(float);captures=np.load(out/'directions.npz')
    for i,axis in enumerate(design['neighbors']):
        raw=(captures[axis['id']+'_positive']-captures[axis['id']+'_negative']).mean(0);expected=raw/np.linalg.norm(raw)*float(np.linalg.norm(native[2]))
        assert np.allclose(captures['directions'][i],expected,rtol=0,atol=1e-6)
    atlas=json.loads((DEFAULT/'search-atlas.json').read_text())
    lookup={c['id']:v.astype(float) for c,v in zip(atlas['concepts'],native)}
    lookup.update({a['id']:v.astype(float) for a,v in zip(design['neighbors'],captures['directions'])})
    rng=np.random.default_rng(design['random_seed']);randoms=rng.normal(size=(2,4096));randoms*=float(atlas['scale'])/np.linalg.norm(randoms,axis=1,keepdims=True)
    lookup.update({f'random{i}':v for i,v in enumerate(randoms)})
    matrix=[]
    for j,a in enumerate(spec['actions']):
        vector=shifts[j];assert hashlib.sha256(vector.tobytes()).hexdigest()==a['vector_sha256']
        norm=float(np.linalg.norm(vector));projection=float(vector@p/(p@p))
        assert abs(norm-a['norm'])<1e-8 and abs(projection-a['anchor_projection'])<1e-8
        if a['mode'] in ['anchor_preserving','anchor_replay']:assert abs(projection+2)<1e-5
        for name,stored in a.get('active_projections',{}).items():
            direction=lookup[name];actual=float(vector@direction/(direction@direction))
            assert abs(actual-stored)<1e-8 and abs(actual)<=2+1e-5
        values=read(out/'outputs'/f'action-{j:02d}.jsonl');assert len(values)==100 and [v['id'] for v in values]==[r['id'] for r in rows]
        predictions=[]
        for v in values:
            assert v['spec_sha256']==digest(spec);lines=v['text'].strip().splitlines();line=lines[-1].strip() if lines else '';valid=not v['truncated'] and len(lines)>=2 and bool(''.join(lines[:-1]).strip()) and re.fullmatch(r'答案[：:]\s*[ABC]',line) and sum(bool(re.fullmatch(r'答案[：:]\s*[A-Z]',l.strip())) for l in lines)==1
            prediction='ABC'.index(line[-1]) if valid else None
            assert prediction==v['prediction'] and bool(valid)==v['format_valid'];predictions.append(-1 if prediction is None else prediction)
            trace=v['steering_trace'];assert abs(trace['requested_norm']-norm)<1e-5 and abs(trace['observed_min']-norm)<1e-3 and abs(trace['observed_max']-norm)<1e-3
        pred=np.array(predictions);matrix.append(pred);ok=pred==gold;m=summary['results'][str(j)]
        assert m['correct']==int(ok.sum()) and m['train_correct']==int(ok[:70].sum()) and m['test_correct']==int(ok[70:].sum())
        assert m['fixed_samples']==[int(i+1) for i in np.flatnonzero(ok&~baseok)] and m['damaged_samples']==[int(i+1) for i in np.flatnonzero(~ok&baseok)]
        assert m['fixed']==len(m['fixed_samples']) and m['damaged']==len(m['damaged_samples']) and m['invalid_format']==int(sum(pred<0))
        assert m['output_sha256']==sha(out/'outputs'/f'action-{j:02d}.jsonl')
        if j==0:assert all(v['token_ids']==b['token_ids'] for v,b in zip(values,base))
    for path in (out/'workers').glob('model-*.json'):
        model=json.loads(path.read_text());assert model['dtype']=='torch.float32' and model['revision']==parent['model']['model_revision'] and model['layer_index']==16 and model['decode']=='greedy'
    diagnosis=json.loads((out/'error-diagnosis.json').read_text());assert set(sum([g['samples'] for g in diagnosis['groups']],[]))==set(int(i+1) for i in np.flatnonzero(~baseok))
    assert summary['best_observed']==max(range(24),key=lambda j:(summary['results'][str(j)]['correct'],-j))
    assert summary['train_selected']==max(range(24),key=lambda j:(summary['results'][str(j)]['train_correct'],-j))
    write(out/'audit.json',dict(passed=True,summary_sha256=sha(out/'summary.json'),spec_sha256=digest(spec),checks=['all 2400 final answers independently parsed','all IDs, splits and original input hashes verified','all generated prefill/decode shift norms observed','fixed anchor projections independently recomputed','active projection bounds checked','extracted directions reconstructed from captured states','original precision:-2 replay matches every token','all worker models use same FP32 revision and layer','all corrections, damages, invalid formats and selection reconstructed','23-error diagnosis partitions actual anchor errors']))
    print('PASS: 2400 complete generations independently audited')

if __name__=='__main__':main()

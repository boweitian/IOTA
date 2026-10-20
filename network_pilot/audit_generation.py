"""Independent reconstruction of generated-answer validation and semantic diagnosis."""
import json,re,zipfile
from pathlib import Path
import numpy as np
from classification_generation import DEFAULT,SOURCE,SEMANTIC,ROOT,template_for
from sentiment_prepare import sha,write
from sentiment_navigator import read,digest,bank

def parsed(text,letters,truncated):
    # Independent terminal-line implementation; never scans analysis for a guess.
    if truncated:return None
    lines=[line.strip() for line in text.strip().splitlines()]
    if len(lines)<2 or not ''.join(lines[:-1]).strip():return None
    if not re.fullmatch(r'答案[：:]\s*[A-Z]',lines[-1]):return None
    if sum(bool(re.fullmatch(r'答案[：:]\s*[A-Z]',line)) for line in lines)!=1:return None
    letter=lines[-1][-1]
    return letters.index(letter) if letter in letters else None

def audit(out=DEFAULT):
    out=Path(out);s=json.loads((out/'generation-summary.json').read_text());spec=json.loads((out/'generation-spec.json').read_text());m=json.loads((out/'dataset-manifest.json').read_text())
    assert s['complete'] and s['spec_sha256']==digest(spec) and s['measurements']==5400
    assert sha(SOURCE/'feasible-space.jsonl')==spec['source_items_sha256'];assert sha(SOURCE/'sweep-summary.json')==spec['old_summary_sha256']
    assert sha(SEMANTIC/'directions.npz')==spec['ordinary_direction_sha256']
    assert m['prompt_fields']==['context','hypothesis']
    assert '{reason}' not in spec['prompt_template'] and '{gold}' not in spec['prompt_template']
    for name,value in m['files'].items():assert sha(out/name)==value
    for name,value in spec['source_sha256'].items():assert sha(out/'source-code'/name)==value
    rows=read(SOURCE/'feasible-space.jsonl');gold=np.array([r['gold'] for r in rows]);matrix=[];outputs={}
    original=json.loads((SOURCE/'dataset-manifest.json').read_text());z=zipfile.ZipFile(SOURCE/'source/anli_v1.0.zip');raw={}
    assert sha(SOURCE/'source/anli_v1.0.zip')==original['source_sha256']
    for name in z.namelist():
        if name.endswith(('/R3/train.jsonl','/R3/test.jsonl')):
            for line in z.read(name).decode().splitlines():
                value=json.loads(line);raw[value['uid']]=value
    for r in rows:assert (r['context'],r['hypothesis'],r['gold'])==(raw[r['uid']]['context'],raw[r['uid']]['hypothesis'],'enc'.index(raw[r['uid']]['label']))
    atlas=json.loads((out/'search-atlas.json').read_text());actions=bank(np.load(out/'search-geometry.npz')['directions'],float(atlas['scale']))
    ordinary=np.load(out/'ordinary-direction.npz')['direction']
    norms=[float(np.linalg.norm(a['shift'])) for a in actions]+[float(np.linalg.norm(ordinary*d)) for d in [-2,-1,1,2]]+[0.]
    for j in range(54):
        name=f'outputs/action-{j:02d}.jsonl';assert sha(out/name)==s['output_files_sha256'][name];values=read(out/name);outputs[j]=values
        assert [v['id'] for v in values]==[r['id'] for r in rows] and len(values)==100
        for v in values:
            assert v['spec_sha256']==digest(spec);assert parsed(v['text'],spec['letters'],v['truncated'])==v['prediction'];t=v['steering_trace']
            assert abs(t['requested_norm']-norms[j])<1e-4 and abs(t['observed_min']-norms[j])<1e-3 and abs(t['observed_max']-norms[j])<1e-3 and t['forward_calls']>0
        pred=np.array([-1 if v['prediction'] is None else v['prediction'] for v in values]);matrix.append(pred)
        for split,lo,hi in [('all',0,100),('train',0,70),('test',70,100)]:
            metric=s['results'][str(j)]['splits'][split];assert metric['correct']==int((pred[lo:hi]==gold[lo:hi]).sum()) and metric['invalid_format']==int((pred[lo:hi]<0).sum())
    matrix=np.array(matrix).T;baseline=matrix[:,0];base_ok=baseline==gold
    for j in range(54):
        metric=s['results'][str(j)]['splits']['all'];ok=matrix[:,j]==gold
        assert metric['fixed']==int((ok&~base_ok).sum()) and metric['damaged']==int((~ok&base_ok).sum())
    eligible=list(range(1,37))+list(range(49,53));selected=max(eligible,key=lambda j:(int((matrix[:70,j]==gold[:70]).sum()),-j));assert selected==s['train_selected_single']
    for key,ix in [('original_singles',list(range(1,37))),('all_singles',eligible),('original_all',list(range(49))),('all_activation',list(range(53)))]:
        counts=(matrix[:,ix]==gold[:,None]).sum(1);assert counts.tolist()==s['coverage'][key]['per_item_solutions'] and int((counts>0).sum())==s['coverage'][key]['correct']
    for path in (out/'workers').glob('model-*.json'):
        model=json.loads(path.read_text());assert model['model']==spec['model']['model'] and model['revision']==spec['model']['model_revision'] and model['dtype']=='torch.float32' and model['layer_index']==16
    groups=json.loads((ROOT/'semantic_error_hypotheses.json').read_text());old_ok=np.array([r['baseline_prediction']==r['gold'] for r in rows]);assert set(sum([g['indices'] for g in groups['groups'].values()],[]))==set(np.flatnonzero(~old_ok))
    group_results={key:dict(name=g['name'],n=len(g['indices']),correct_by_condition={spec['actions'][j]['id']:int((matrix[g['indices'],j]==gold[g['indices']]).sum()) for j in [0,49,50,51,52,53]}) for key,g in groups['groups'].items()}
    semantic=dict(groups=group_results,old_wrong=int((~old_ok).sum()),new_baseline_fixed_old=int((base_ok&~old_ok).sum()),new_baseline_damaged_old=int((~base_ok&old_ok).sum()),
        common_positive_union_fixed=int((~base_ok&np.any(matrix[:,[51,52]]==gold[:,None],axis=1)).sum()),
        common_positive_union_damaged=int((base_ok&np.any(matrix[:,[51,52]]!=gold[:,None],axis=1)).sum()),
        fixed_cases={spec['actions'][j]['id']:[i+1 for i in range(100) if not base_ok[i] and matrix[i,j]==gold[i]] for j in [49,50,51,52,53]},
        damaged_cases={spec['actions'][j]['id']:[i+1 for i in range(100) if base_ok[i] and matrix[i,j]!=gold[i]] for j in [49,50,51,52,53]},
        note='Qualitative candidate groups are hypotheses, not proven internal causes. Reused test items are exploratory. Same label/generation protocol across activation conditions.')
    write(out/'semantic-diagnosis.json',semantic)
    write(out/'audit.json',dict(passed=True,checks=['all 5400 complete generations independently parsed from terminal answer line','all original source texts and labels match official archive','all observed prefill/decode shifts match action norms','all split scores, corrections and damages reconstructed','train selection uses only training correctness','oracle coverage reconstructed','all worker models match Qwen3.5-9B revision and FP32','original experiment hashes unchanged'],
        summary_sha256=sha(out/'generation-summary.json'),spec_sha256=digest(spec),semantic_sha256=sha(out/'semantic-diagnosis.json')))
    # Compact viewer data preserves each exact prompt and complete visible output.
    viewer=dict(model=json.loads(next((out/'workers').glob('model-*.json')).read_text()),letters=spec['letters'],labels=m['display_labels'],
        actions=[dict(a,prompt_template=template_for(m,a),requested_norm=norms[j]) for j,a in enumerate(spec['actions'])],
        rows=[dict(id=r['id'],index=i,split=r['split'],context=r['context'],hypothesis=r['hypothesis'],gold=r['gold'],reason=raw[r['uid']].get('reason',''),
            outputs=[dict(text=outputs[j][i]['text'],prediction=outputs[j][i]['prediction'],format_valid=outputs[j][i]['format_valid'],error=outputs[j][i]['error']) for j in range(54)]) for i,r in enumerate(rows)])
    write(out/'view-data.json',viewer)
    print(json.dumps(dict(passed=True,semantic=semantic),ensure_ascii=False,indent=2))
    return s,semantic

if __name__=='__main__':audit()

"""Independent reconstruction of the complete 1200 x 61 detailed-answer scan."""
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from fullspace_detailed import OUT, ROOT, DEFAULT, SEMANTIC, SOURCE, output_path
from sentiment_navigator import digest, read, bank
from sentiment_prepare import sha, write


def independent_label(text,truncated):
    if truncated:return None
    lines=[s.strip() for s in text.strip().splitlines()]
    if len(lines)<2 or not ''.join(lines[:-1]):return None
    if not re.fullmatch(r'答案[：:]\s*[ABC]',lines[-1]):return None
    if sum(bool(re.fullmatch(r'答案[：:]\s*[A-Z]',line)) for line in lines)!=1:return None
    return 'ABC'.index(lines[-1][-1])


def audit():
    from transformers import AutoTokenizer

    spec=json.loads((OUT/'spec.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    assert summary['complete'] and summary['spec_sha256']==digest(spec)
    registered=datetime.fromisoformat(spec['registered_at_utc'].replace('Z','+00:00')).timestamp()
    for name,value in spec['files'].items():assert sha(OUT/name)==value,name
    for relative,value in spec['source_sha256'].items():
        assert sha(ROOT/relative)==sha(OUT/'source-code'/Path(relative).name)==value
    analysis=json.loads((OUT/'analysis-protocol.json').read_text())
    assert sha(ROOT/'fullspace_analysis.py')==analysis['source_sha256']
    archive=SOURCE/'source/anli_v1.0.zip';assert sha(archive)==spec['archive_sha256']
    with zipfile.ZipFile(archive) as source:
        official=[json.loads(line) for line in source.read('anli_v1.0/R3/test.jsonl').decode().splitlines()]
    rows=read(OUT/'inputs.jsonl');labels=read(OUT/'labels-for-analysis.jsonl');feasible=read(OUT/'feasible-space.jsonl')
    assert len(rows)==len(labels)==len(feasible)==len(official)==1200
    assert sha(OUT/'response-matrix.npz')==summary['matrix_sha256']
    assert sha(OUT/'feasible-space.jsonl')==summary['feasible_sha256']
    assert sha(OUT/'baseline-features.npz')==summary['features_sha256']
    for i,(r,label,original) in enumerate(zip(rows,labels,official)):
        assert set(r)=={'id','uid','source_index','context','hypothesis'}
        assert r['source_index']==i and r['id']==label['id']=='anli:r3:test:'+original['uid']
        assert (r['context'],r['hypothesis'])==(original['context'],original['hypothesis'])
        assert label['gold']=='enc'.index(original['label']) and label['reason']==original.get('reason','')
    native=np.load(DEFAULT/'search-geometry.npz')['directions'];atlas=json.loads((DEFAULT/'search-atlas.json').read_text())
    assert sha(DEFAULT/'search-geometry.npz')==spec['original_geometry_sha256']
    assert sha(SEMANTIC/'directions.npz')==spec['ordinary_geometry_sha256']
    reconstructed=[a['shift'] for a in bank(native,float(atlas['scale']))]
    semantic=np.load(SEMANTIC/'directions.npz')['directions'][0]
    reconstructed.extend(semantic*d for d in [-2.,-1.,1.,2.])
    rng=np.random.default_rng(2026102004);random=rng.normal(size=(2,4096));random*=np.linalg.norm(native[2])/np.linalg.norm(random,axis=1,keepdims=True)
    reconstructed.extend(v*d for v in random for d in [-2.,-1.,1.,2.])
    shifts=np.load(OUT/'shifts.npz')['shifts']
    # Independent reconstruction may use a different CPU BLAS reduction order.
    # The artifact hash below remains exact; only the mathematical cross-check tolerates roundoff.
    assert np.allclose(shifts,np.stack(reconstructed),rtol=0,atol=1e-10)
    for j,(action,vector) in enumerate(zip(spec['actions'],reconstructed)):
        original_dtype=shifts[j].astype(vector.dtype)
        assert action['vector_sha256']==hashlib.sha256(original_dtype.tobytes()).hexdigest()
    assert spec['single_actions']==list(range(1,37))+list(range(49,53)) and spec['main_actions']==list(range(53))
    tokenizer=AutoTokenizer.from_pretrained(spec['model']['model_path'],local_files_only=True)
    config=json.loads((Path(spec['model']['model_path'])/'config.json').read_text())
    eos=config.get('eos_token_id',config.get('text_config',{}).get('eos_token_id'));eos=eos if isinstance(eos,list) else [eos]
    assert eos and None not in eos
    prefixes=[]
    for row in rows:
        question=spec['prompt_template'].format(context=json.dumps(row['context'],ensure_ascii=False),hypothesis=json.dumps(row['hypothesis'],ensure_ascii=False))
        prefix=tokenizer.apply_chat_template([dict(role='system',content=spec['system_prompt']),dict(role='user',content=question)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefixes.append(hashlib.sha256(prefix.encode()).hexdigest())
    rebuilt=np.full((1200,61),-1,dtype=int);invalid=np.zeros((1200,61),dtype=bool);truncated=invalid.copy();lengths=np.zeros((1200,61),dtype=int)
    for j in range(61):
        for lo in range(0,1200,16):
            path=output_path(j,lo);assert sha(path)==summary['output_sha256'][str(path.relative_to(OUT))]
            outputs=read(path);assert len(outputs)==16
            for i,value in enumerate(outputs,lo):
                assert value['id']==rows[i]['id'] and value['source_index']==i
                assert value['action_index']==j and value['batch_start']==lo and value['spec_sha256']==digest(spec)
                assert value['prompt_sha256']==prefixes[i]
                assert registered<value['started_at']<=value['completed_at']
                assert value['new_tokens']==len(value['token_ids'])<=1024
                assert value['truncated']==(len(value['token_ids'])==1024 and value['token_ids'][-1] not in eos)
                assert value['token_ids'][-1] in eos or value['truncated']
                assert tokenizer.decode(value['token_ids'],skip_special_tokens=True)==value['text']
                pred=independent_label(value['text'],value['truncated']);assert pred==value['prediction'] and value['format_valid']==(pred is not None)
                rebuilt[i,j]=-1 if pred is None else pred;invalid[i,j]=pred is None;truncated[i,j]=value['truncated'];lengths[i,j]=value['new_tokens']
                norm=float(np.linalg.norm(shifts[j]));trace=value['steering_trace']
                assert abs(trace['requested_norm']-norm)<1e-6
                assert max(abs(trace[k]-norm) for k in ['observed_min','observed_max'])<1e-3
                assert trace['forward_calls']>0
    matrix=np.load(OUT/'response-matrix.npz')
    for key,value in [('predictions',rebuilt),('invalid',invalid),('truncated',truncated),('lengths',lengths)]:assert np.array_equal(matrix[key],value)
    gold=np.array([r['gold'] for r in labels]);ok=rebuilt==gold[:,None];base=ok[:,0]
    assert np.array_equal(matrix['gold'],gold) and np.array_equal(matrix['correctness'],ok)
    for j in range(61):
        metric=summary['conditions'][str(j)];fixed=int((ok[:,j]&~base).sum());damaged=int((~ok[:,j]&base).sum())
        assert metric['correct']==int(ok[:,j].sum()) and metric['invalid']==int(invalid[:,j].sum()) and metric['truncated']==int(truncated[:,j].sum())
        assert metric['fixed']==fixed and metric['damaged']==damaged
        assert abs(metric['difference']-(fixed-damaged)/1200)<1e-12
        discordant=fixed+damaged
        p=min(1.,2*math.fsum(math.exp(math.lgamma(discordant+1)-math.lgamma(k+1)-math.lgamma(discordant-k+1)-discordant*math.log(2)) for k in range(min(fixed,damaged)+1))) if discordant else 1.
        assert abs(p-metric['exact_mcnemar_p'])<1e-10
        for label,score in enumerate(metric['per_class']):
            assert score['n']==int((gold==label).sum()) and score['correct']==int(ok[gold==label,j].sum())
    for key,indices in [('native36',list(range(1,37))),('named40',spec['single_actions']),('named_and_mixtures52',list(range(1,53))),('random8',spec['random_actions'])]:
        count=ok[:,indices].sum(1);coverage=summary['coverage'][key]
        assert coverage['correct']==int((count>0).sum()) and coverage['with_baseline_correct']==int(((count>0)|base).sum())
        assert coverage['per_item_solutions']==count.tolist() and coverage['histogram']==np.bincount(count,minlength=len(indices)+1).tolist()
    for key,indices in [('best_fixed_single',spec['single_actions']),('best_main',spec['main_actions']),('best_mixture',spec['mixture_actions'])]:
        assert summary[key]==max(indices,key=lambda j:(int(ok[:,j].sum()),-j))
    for i,row in enumerate(feasible):
        assert row['id']==rows[i]['id'] and row['gold']==int(gold[i]) and row['predictions']==rebuilt[i].tolist()
        assert row['named_feasible']==[j for j in spec['single_actions'] if ok[i,j]]
        assert row['main_feasible']==[j for j in spec['main_actions'] if ok[i,j]]
    features=np.load(OUT/'baseline-features.npz')
    assert features['ids'].tolist()==[r['id'] for r in rows] and features['last'].shape==features['mean'].shape==(1200,4096)
    assert np.isfinite(features['last']).all() and np.isfinite(features['mean']).all()
    for rank in range(16):
        model=json.loads((OUT/'workers'/f'model-{rank:02d}.json').read_text())
        assert model['model']==spec['model']['model'] and model['revision']==spec['model']['model_revision']
        assert model['dtype']=='torch.float32' and model['device']=='NVIDIA H200' and model['layer_index']==16 and not model['thinking']
        assert model['system_prompt']==spec['system_prompt'] and model['spec_sha256']==digest(spec)
    preflight=json.loads((OUT/'preflight/result.json').read_text());assert preflight['passed'] and preflight['zero_hook_token_equivalence']
    diagnosis=json.loads((OUT/'diagnosis-summary.json').read_text());errors=read(OUT/'error-cases.jsonl')
    assert diagnosis['complete'] and diagnosis['n']==len(errors)==int((~base).sum())
    assert diagnosis['cases_sha256']==sha(OUT/'error-cases.jsonl')
    assert {r['id'] for r in errors}=={rows[i]['id'] for i in range(1200) if not base[i]}
    assert sum(g['n'] for g in diagnosis['groups'].values())==len(errors)
    result=dict(passed=True,n=1200,conditions=61,generations=73200,summary_sha256=sha(OUT/'summary.json'),
        spec_sha256=digest(spec),diagnosis_sha256=sha(OUT/'diagnosis-summary.json'),
        checks=['All official 1200 R3 test items, original source order, exact texts/labels/reasons',
                'All 73200 answers independently decoded and parsed; every truncated or invalid output counted wrong',
                'Same complete detailed prompt and 1024-token budget across all conditions',
                'Original 53 main vectors and norm-matched 8 random controls reconstructed',
                'Observed prefill/decode injection norms checked for every generation',
                'All accuracies, class scores, fixed/damaged pairs, best fixed choices and oracle distributions reconstructed',
                '1200 unsteered prompt features bound to IDs; no gold answer in prediction prompts',
                'All baseline mistakes and only those mistakes have separate post-evaluation diagnostic records',
                'Frozen model, source/input hashes and preflight zero-hook equivalence checked'])
    write(OUT/'audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':audit()

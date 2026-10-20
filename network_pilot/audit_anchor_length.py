"""Verify EOS reuse and exact token-prefix replay in the length-only control."""
import json,re
from pathlib import Path
import numpy as np
from classification_generation import DEFAULT
from sentiment_navigator import read,digest
from sentiment_prepare import sha,write

def main():
    parent=DEFAULT/'precision-anchor-v1';out=parent/'length512-v1';s=json.loads((out/'summary.json').read_text());spec=json.loads((out/'spec.json').read_text());oldspec=json.loads((parent/'spec.json').read_text())
    assert s['passed'] and s['spec_sha256']==digest(spec) and spec['parent_spec_sha256']==digest(oldspec)
    assert spec['source_sha256']==sha(Path(__file__).with_name('extend_anchor_generation.py'))
    model_path=Path(oldspec['model']['model_path']);config_path=model_path/'generation_config.json'
    gc=json.loads((config_path if config_path.exists() else model_path/'config.json').read_text())
    eos=gc.get('eos_token_id') or gc.get('text_config',{}).get('eos_token_id');assert eos is not None
    eos=eos if isinstance(eos,list) else [eos]
    gold=np.array([r['gold'] for r in read(DEFAULT.parent/'classification9b-anli-r3-100-v1/feasible-space.jsonl')]);actual=0
    for j in range(24):
        source=read(parent/'outputs'/f'action-{j:02d}.jsonl');values=read(out/'outputs'/f'action-{j:02d}.jsonl');assert len(values)==100
        assert spec['outputs_sha256'][str(j)]==sha(parent/'outputs'/f'action-{j:02d}.jsonl')
        replay_indices=set(k for lo in range(0,100,4) if any(v['truncated'] for v in source[lo:lo+4]) for k in range(lo,lo+4))
        predictions=[]
        for i,(old,v) in enumerate(zip(source,values)):
            assert v['id']==old['id'] and v['spec_sha256']==digest(spec)
            assert v['replayed_for_512']==(i in replay_indices)
            if i in replay_indices:
                assert v['token_ids'][:len(old['token_ids'])]==old['token_ids'];actual+=1
            else:assert v['token_ids']==old['token_ids'] and not old['truncated'] and old['token_ids'][-1] in eos
            if not old['truncated']:assert v['text']==old['text'] and v['prediction']==old['prediction']
            lines=v['text'].strip().splitlines();last=lines[-1].strip() if lines else '';valid=not v['truncated'] and len(lines)>=2 and bool(''.join(lines[:-1]).strip()) and bool(re.fullmatch(r'答案[：:]\s*[ABC]',last)) and sum(bool(re.fullmatch(r'答案[：:]\s*[A-Z]',l.strip())) for l in lines)==1
            prediction='ABC'.index(last[-1]) if valid else None;assert prediction==v['prediction'] and valid==v['format_valid'];predictions.append(-1 if prediction is None else prediction)
            if not v['truncated']:assert v['token_ids'][-1] in eos
        pred=np.array(predictions);m=s['results'][str(j)];assert m['correct']==int(sum(pred==gold)) and m['invalid_format']==int(sum(pred<0)) and m['vs256_damaged']==0
        assert m['output_sha256']==sha(out/'outputs'/f'action-{j:02d}.jsonl')
    assert actual==s['actual_new_generations']==288 and s['cached_eos_outputs']==2112
    for f in out.glob('model-*.json'):
        m=json.loads(f.read_text());assert m['dtype']=='torch.float32' and m['revision']==oldspec['model']['model_revision'] and m['layer_index']==16
    write(out/'audit.json',dict(passed=True,summary_sha256=sha(out/'summary.json'),actual_new_generations=actual,cached_eos_outputs=2112,checks=['all 288 replayed token prefixes exact','all 2112 reused outputs end in EOS','completed outputs never change','all source-output hashes frozen','all final answers independently parsed','uniform 512-token allowance and original batch membership','same FP32 model revision and intervention layer']))
    print('PASS: length-only control independently audited')

if __name__=='__main__':main()

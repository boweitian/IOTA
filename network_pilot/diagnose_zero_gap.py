"""Independent controls for the generated ANLI zero-strength discontinuity."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from classification_generation import DEFAULT, ClassificationEngine, traced_generate
from answer_format import parse_answer
from sentiment_navigator import read, bank, digest
from sentiment_prepare import write, jsonl, sha

OUT = DEFAULT / 'zero-gap-diagnosis-v1'

def main(worker):
    OUT.mkdir(exist_ok=True)
    m=json.loads((DEFAULT/'dataset-manifest.json').read_text())
    parent=json.loads((DEFAULT/'generation-spec.json').read_text())
    atlas=json.loads((DEFAULT/'search-atlas.json').read_text())
    original=bank(np.load(DEFAULT/'search-geometry.npz')['directions'],float(atlas['scale']))
    ordinary=np.load(DEFAULT/'ordinary-direction.npz')['direction']
    scale=float(np.linalg.norm(ordinary));zero=np.zeros(4096)
    rng=np.random.default_rng(2026102001)
    randoms=rng.normal(size=(2,4096));randoms*=scale/np.linalg.norm(randoms,axis=1,keepdims=True)
    actions=[('zero_repeat',zero,'hook',False),('zero_no_hook',zero,'none',False),
             ('zero_reverse_batch_order',zero,'hook',True),('decomposition_plus1_replay',original[3]['shift'],'hook',False),
             ('random0_minus1',-randoms[0],'hook',False),('random0_plus1',randoms[0],'hook',False),
             ('random1_minus1',-randoms[1],'hook',False),('random1_plus1',randoms[1],'hook',False),
             ('ordinary_minus001',-.01*ordinary,'hook',False),('ordinary_plus001',.01*ordinary,'hook',False)]
    spec={'parent_spec_sha256':digest(parent),'source_sha256':sha(Path(__file__)),
          'actions':[dict(id=n,norm=float(np.linalg.norm(d)),vector_sha256=hashlib.sha256(d.tobytes()).hexdigest(),mode=mode,reverse_batches=rev) for n,d,mode,rev in actions],
          'seed':2026102001,'batch_size':4,'max_new_tokens':256,'precision':'FP32; TF32 disabled; greedy',
          'note':'Random controls are isotropic with the same norm as 1x semantic steering. Same 100 exploratory items and exact prompt.'}
    specpath=OUT/f'spec-{worker}.json'
    if specpath.exists():assert json.loads(specpath.read_text())==spec
    else:write(specpath,spec)
    engine=ClassificationEngine(parent['model'],m);write(OUT/f'model-{worker}.json',engine.meta)
    rows=read(DEFAULT/'train.jsonl')+read(DEFAULT/'test-inputs.jsonl')
    for index in [2*worker,2*worker+1]:
        name,shift,mode,rev=actions[index];target=OUT/f'{name}.jsonl'
        if target.exists():continue
        batches=[rows[i:i+4] for i in range(0,100,4)]
        if rev:batches=batches[::-1]
        values=[]
        for batch in batches:
            qs=[engine.question(r) for r in batch]
            if mode=='none':
                # Exact Engine.generate settings and decoding, with no forward hook registered.
                inputs=engine.tokenizer([engine.prefix(q) for q in qs],return_tensors='pt',padding=True,add_special_tokens=False).to(engine.device)
                assert not engine.block._forward_hooks
                with engine.torch.inference_mode():
                    ids=engine.model.generate(**inputs,do_sample=False,max_new_tokens=256,pad_token_id=engine.tokenizer.pad_token_id,use_cache=True,logits_to_keep=1)
                eos=engine.model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                generated=[]
                for tokens in ids[:,inputs.input_ids.shape[1]:].cpu().tolist():
                    end=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:end]
                    generated.append(dict(text=engine.tokenizer.decode(tokens,skip_special_tokens=True),token_ids=tokens,new_tokens=len(tokens),truncated=len(tokens)==256 and tokens[-1] not in eos))
            else:generated=traced_generate(engine,qs,shift,256)
            for row,v in zip(batch,generated):values.append(dict(id=row['id'],condition=name,spec_sha256=digest(spec),**v,**parse_answer(v['text'],parent['letters'],v['truncated'])))
            jsonl(target.with_suffix('.partial'),values);print(name,len(values),flush=True)
        lookup={v['id']:v for v in values};jsonl(target,[lookup[r['id']] for r in rows]);target.with_suffix('.partial').unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);main(p.parse_args().worker)

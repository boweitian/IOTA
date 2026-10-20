"""GPU checks of likelihood scoring against an independent cached-token path."""
import json
from pathlib import Path
import numpy as np
from pilot_engine import PilotEngine,measure_geometry
from engine import atomic_json

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent/"steering_lab"


def sequential_logp(engine,question,answer,shift):
    torch=engine.torch
    prefix=engine.tokenizer.encode(engine.prefix(question),add_special_tokens=False)
    target=engine.tokenizer.encode(answer,add_special_tokens=False)
    ids=torch.tensor([prefix],device=engine.device);cache=None;values=[]
    delta=torch.as_tensor(shift,device=engine.device,dtype=engine.model.dtype)
    def inject(module,args,output):
        h=output.clone();h[:,-1,:]+=delta;return h
    handle=engine.block.register_forward_hook(inject)
    try:
        with torch.inference_mode():
            for token in target:
                out=engine.model(input_ids=ids,use_cache=True,past_key_values=cache,logits_to_keep=1)
                values.append(float(out.logits[0,-1].float().log_softmax(-1)[token]))
                cache=out.past_key_values;ids=torch.tensor([[token]],device=engine.device)
    finally:handle.remove()
    return sum(values)


def main():
    out=ROOT/"runs/engine-audit-v1";out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/"inputs/protocol.json").read_text());engine=PilotEngine(protocol["model"])
    geo=dict(np.load(LAB/"runs/job-2117172/geometry.npz"));geo["scale"]=json.loads((LAB/"runs/job-2117172/atlas.json").read_text())["scale"]
    engine.precise_likelihood_mode()
    checks=[]
    q="Q: What color is a clear daytime sky usually?\nA:"
    for shift in [np.zeros(engine.dim),geo["directions"][0]*.5]:
        answers=["Blue.","A clear daytime sky is usually blue."]
        batched,_=engine.answer_likelihoods(q,answers,shift)
        ref=[sequential_logp(engine,q,a,shift) for a in answers]
        # FP32 permits a much tighter check than the initial BF16 implementation.
        errors=np.abs(np.asarray(batched)-ref)
        checks.append(dict(shift_norm=float(np.linalg.norm(shift)),batched=batched,sequential=ref,absolute_errors=errors.tolist()))
        exact,_=engine.answer_likelihoods(q,answers,shift,batch_size=1)
        exact_errors=np.abs(np.asarray(exact)-ref)
        checks[-1].update(single_batch=exact,single_batch_errors=exact_errors.tolist())
        if errors.max()>1e-4 or exact_errors.max()>1e-5:raise RuntimeError(f"Likelihood semantics mismatch: {checks[-1]}")
    atomic_json(out/"likelihood_check.json",dict(model=engine.meta,checks=checks,max_error=max(max(c["absolute_errors"]) for c in checks),
        tolerance=1e-4,note="FP32 cached scoring vs independent cached per-token steering; single-batch equality also checked at 1e-5. Initial BF16 attempts failed batch-shape stability checks."))
    engine.model.bfloat16();engine.meta.update(dtype=str(engine.model.dtype));engine.meta.pop("likelihood_precision",None)
    graph=measure_geometry(engine,out,geo);atomic_json(out/"network_geometry.json",graph)
    atomic_json(out/"complete.json",dict(likelihood_passed=True,response_graph_measured=True))
    print("AUDIT COMPLETE",checks,flush=True)


if __name__=="__main__":main()

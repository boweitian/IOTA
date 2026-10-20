"""Additional real measurements; original steering_lab inference stays intact."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent/"steering_lab"
sys.path.insert(0,str(LAB))
from engine import Engine
from concepts import CONCEPTS, probe_examples
from space import singles, serial_action


class PilotEngine(Engine):
    def precise_likelihood_mode(self):
        # BF16 batch shape measurably changes small answer probabilities on this model.
        self.model.float()
        self.torch.backends.cuda.matmul.allow_tf32=False
        self.meta.update(dtype=str(self.model.dtype),likelihood_precision="FP32 cached decoding; TF32 disabled")

    def capture(self, questions, responses=None, shifts=None, layer=None, batch_size=8):
        layer=self.layer if layer is None else layer
        captured_rows=[]
        block=self.model.model.language_model.layers[layer]
        for lo in range(0,len(questions),batch_size):
            qs=questions[lo:lo+batch_size]
            prefixes=[self.prefix(q) for q in qs]
            texts=prefixes if responses is None else [p+r for p,r in zip(prefixes,responses[lo:lo+batch_size])]
            lengths=[len(self.tokenizer.encode(p,add_special_tokens=False)) for p in prefixes]
            inputs=self.tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(self.device)
            captured=[];handles=[]
            if shifts is not None:
                delta=self.torch.as_tensor(np.stack(shifts[lo:lo+batch_size]),device=self.device,dtype=self.model.dtype)
                def inject(module,args,output):
                    h=output.clone();h[:,-1,:]+=delta;return h
                handles.append(self.block.register_forward_hook(inject))
            def observe(module,args,output):captured.append(output.detach())
            handles.append(block.register_forward_hook(observe))
            try:
                with self.torch.inference_mode():self.model(**inputs,use_cache=False,logits_to_keep=1)
            finally:
                for h in handles:h.remove()
            h=captured[0].float()
            for i,n in enumerate(lengths):
                pad=inputs.input_ids.shape[1]-int(inputs.attention_mask[i].sum())
                start=pad+n if responses is not None else pad
                captured_rows.append(h[i,start:].mean(0).cpu().numpy() if responses is not None else h[i,-1].cpu().numpy())
        return np.stack(captured_rows)

    def answer_likelihoods(self, question, choices, shift, extra="", batch_size=8):
        """Score through cached decoding, identical to the deployed injection path."""
        prefix=self.tokenizer.encode(self.prefix(question,extra),add_special_tokens=False)
        results=[];start=time.monotonic()
        for lo in range(0,len(choices),batch_size):
            answers=[self.tokenizer.encode(a,add_special_tokens=False) for a in choices[lo:lo+batch_size]]
            if any(not a for a in answers):raise ValueError("Empty answer tokens")
            ids=self.torch.tensor([prefix]*len(answers),device=self.device)
            delta=self.torch.as_tensor(shift,device=self.device,dtype=self.model.dtype)
            def inject(module,args,output):
                h=output.clone();h[:,-1,:]+=delta;return h
            handle=self.block.register_forward_hook(inject)
            try:
                with self.torch.inference_mode():
                    cache=None;sums=np.zeros(len(answers),dtype=float)
                    for step in range(max(map(len,answers))):
                        out=self.model(input_ids=ids,use_cache=True,past_key_values=cache,logits_to_keep=1)
                        logs=out.logits[:,-1].float().log_softmax(-1)
                        tokens=[a[step] if step<len(a) else self.tokenizer.pad_token_id for a in answers]
                        values=logs.gather(1,self.torch.tensor(tokens,device=self.device)[:,None]).flatten().cpu().numpy()
                        sums += values*np.asarray([step<len(a) for a in answers])
                        cache=out.past_key_values
                        ids=self.torch.tensor(tokens,device=self.device)[:,None]
                    results.extend(sums.tolist())
            finally:handle.remove()
        return results,time.monotonic()-start


def measure_geometry(engine, out, original_geometry):
    """Downstream response graph, independent of all benchmark answers."""
    probes=probe_examples();read_layer=min(engine.layer+4,len(engine.model.model.language_model.layers)-1)
    activations=engine.capture([p["question"] for p in probes],[p["response"] for p in probes],layer=read_layer)
    by={(p["concept"],p["context"],p["pole"]):activations[i] for i,p in enumerate(probes)}
    directions=np.stack([np.mean([by[c["id"],i,"positive"]-by[c["id"],i,"negative"] for i in range(8)],axis=0) for c in CONCEPTS])
    unit=directions/np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
    questions=[p["question"] for p in probes if p["concept"]==CONCEPTS[0]["id"] and p["pole"]=="positive"]
    bank=singles(original_geometry["directions"],float(original_geometry["scale"]))
    base=engine.capture(questions,layer=read_layer)
    response=[]
    for a in bank[1:]:
        after=engine.capture(questions,shifts=[a["shift"]]*len(questions),layer=read_layer)
        change=(after-base)@unit.T
        response.append(dict(action=serial_action(a),per_context=change.tolist(),
            mean=change.mean(0).tolist(),heldout_mean=change[8:].mean(0).tolist()))
    np.savez_compressed(out/"downstream_activations.npz",activations=activations,unit=unit,base=base)
    return dict(model=engine.meta,intervention_layer=engine.layer,readout_layer=read_layer,
        gram=(original_geometry["unit"]@original_geometry["unit"].T).tolist(),
        nodes=[dict(id=c["id"],name=c["name"],direction=original_geometry["directions"][i].tolist()) for i,c in enumerate(CONCEPTS)],
        response_edges=response,context_count=len(questions),heldout_context_indices=[8,9],
        note="Measured downstream prompt-state response to residual injection. Concept projection readouts are proxies;"
             " these edges are not global concept causality or generation-time invariants.")

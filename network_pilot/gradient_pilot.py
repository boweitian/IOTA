"""A small differentiable, graph-regularized mixture pilot on GSM8K.

Only nine mixture coefficients are trained; model weights remain frozen.
Reference training rationales are additional supervision, so this is a separate
experiment and is not a same-budget ablation of reward-only search.
"""
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
from experiment import ROOT,LAB,prompt,append
from pilot_engine import PilotEngine
from engine import atomic_json
from grading import grade
from space import action,paired_interval,serial_action


def main():
    out=ROOT/"runs/gradient-gsm8k-v1"
    if (out/"complete.json").exists():raise RuntimeError("Completed run; do not overwrite")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"training.jsonl").exists():raise RuntimeError("Use a new run after an interrupted gradient attempt")
    protocol=json.loads((ROOT/"inputs/protocol.json").read_text())
    rows=[r for r in protocol["records"] if r["suite"]=="gsm8k"]
    search=[r for r in rows if r["split"]=="search"];validation=[r for r in rows if r["split"]=="validation"];test=[r for r in rows if r["split"]=="test"]
    raw=[json.loads(line) for line in (LAB/"inputs/gsm8k_train.jsonl").read_text().splitlines()]
    references={r["id"]:re.sub(r"<<[^>]*>>","",raw[r["source_index"]]["answer"]).replace("####","FINAL:") for r in search+validation}
    engine=PilotEngine(protocol["model"]);engine.precise_likelihood_mode();torch=engine.torch
    for p in engine.model.parameters():p.requires_grad_(False)
    geo=dict(np.load(LAB/"runs/job-2117172/geometry.npz"));scale=json.loads((LAB/"runs/job-2117172/atlas.json").read_text())["scale"]
    directions=torch.tensor(geo["directions"],device=engine.device,dtype=torch.float32)
    graph_path=ROOT/"runs/pilot-2124657/gsm8k/task_network.json"
    graph=json.loads(graph_path.read_text());adj=torch.tensor(graph["adjacency"],device=engine.device,dtype=torch.float32)
    w=torch.nn.Parameter(torch.zeros(len(directions),device=engine.device));optimizer=torch.optim.Adam([w],lr=.15)

    def effective():
        delta=w@directions
        factor=torch.clamp(torch.as_tensor(scale,device=engine.device)/delta.norm().clamp_min(1e-9),max=1.)
        return w*factor,delta*factor

    def nll(selected):
        # Right truncation of references would alter the target; retain each full rationale.
        prefixes=[engine.tokenizer.encode(engine.prefix(prompt(r)),add_special_tokens=False) for r in selected]
        targets=[engine.tokenizer.encode(references[r["id"]],add_special_tokens=False) for r in selected]
        seqs=[p+a for p,a in zip(prefixes,targets)];total=max(map(len,seqs));tail=max(map(len,targets))
        ids=torch.tensor([[engine.tokenizer.pad_token_id]*(total-len(s))+s for s in seqs],device=engine.device)
        attention=torch.tensor([[0]*(total-len(s))+[1]*len(s) for s in seqs],device=engine.device)
        _,delta=effective()
        mask=torch.zeros((len(selected),total,1),device=engine.device)
        for i,target in enumerate(targets):mask[i,total-len(target)-1:]=1
        def inject(module,args,output):return output+mask*delta
        handle=engine.block.register_forward_hook(inject)
        try:
            logits=engine.model(input_ids=ids,attention_mask=attention,use_cache=False,logits_to_keep=tail+1).logits
            losses=[]
            for i,target in enumerate(targets):
                begin=tail-len(target)
                labels=torch.tensor(target,device=engine.device)
                losses.append(torch.nn.functional.cross_entropy(logits[i,begin:begin+len(target)],labels))
            return torch.stack(losses).mean()
        finally:handle.remove()

    atomic_json(out/"protocol.json",dict(model=engine.meta,search_ids=[r["id"] for r in search],validation_ids=[r["id"] for r in validation],
        test_ids=[r["id"] for r in test],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        network_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),reference_supervision=references,
        steps=24,learning_rate=.15,graph_regularization=.05,
        note="Full train rationales supervise a differentiable teacher-forced proxy; discrete accuracy is evaluated separately."
             " Extra supervision and optimization budget differ from reward-only search; this is a separate exploratory pilot."))
    # Verify a real gradient against symmetric finite differences on two search rows.
    first=nll(search[:2]);first.backward();grad=w.grad.detach().clone();idx=int(grad.abs().argmax());epsilon=.01
    with torch.no_grad():
        w[idx]=epsilon;plus=float(nll(search[:2]));w[idx]=-epsilon;minus=float(nll(search[:2]));w.zero_()
    numerical=(plus-minus)/(2*epsilon);analytical=float(grad[idx])
    check=dict(index=idx,analytical=analytical,finite_difference=numerical,error=abs(analytical-numerical),epsilon=epsilon)
    atomic_json(out/"gradient_check.json",check)
    if abs(analytical-numerical)>.005+.1*abs(numerical):raise RuntimeError(f"Gradient check failed: {check}")
    optimizer.zero_grad();checkpoints=[]
    for step in range(25):
        if step%8==0:
            with torch.no_grad():
                valid=float(nll(validation));coeff,_=effective()
                checkpoints.append(dict(step=step,coefficients=coeff.cpu().tolist(),validation_nll=valid))
        if step==24:break
        optimizer.zero_grad();losses=[nll(search[lo:lo+4]) for lo in range(0,len(search),4)]
        ce=torch.stack(losses).mean();coeff,_=effective();positive=torch.relu(coeff)
        regularizer=-.05*(positive@adj@positive)/2
        loss=ce+regularizer;loss.backward()
        if not torch.isfinite(w.grad).all():raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        with torch.no_grad():w.clamp_(-1.,1.)
        append(out/"training.jsonl",dict(step=step,search_nll=float(ce.detach()),regularizer=float(regularizer.detach()),
            coefficients=effective()[0].detach().cpu().tolist()))
        print("GRADIENT STEP",step,float(ce.detach()),flush=True)
    atomic_json(out/"checkpoints.json",checkpoints)
    # All generation comparisons use the same BF16 model path as the main pilot.
    engine.model.bfloat16();engine.meta.update(dtype=str(engine.model.dtype));engine.meta.pop("likelihood_precision",None)
    bank=[action(c["coefficients"],geo["directions"],scale,f"step:{c['step']}") for c in checkpoints]
    validation_records=[]
    for a in bank:
        gs=engine.generate([prompt(r) for r in validation],[a["shift"]]*len(validation),max_new_tokens=1024)
        for r,g in zip(validation,gs):validation_records.append(dict(id=r["id"],action=a["id"],**g,**grade(g["text"],r)))
    atomic_json(out/"validation.json",validation_records)
    chosen=max(bank,key=lambda a:(np.mean([r["correct"] for r in validation_records if r["action"]==a["id"]]),-a["norm"]))
    atomic_json(out/"frozen_profile.json",dict(chosen=serial_action(chosen),all_checkpoints=[serial_action(a) for a in bank],frozen_before_test=True))
    outcomes={}
    for name,a in [("baseline",bank[0]),("gradient",chosen)]:
        saved=[]
        for lo in range(0,len(test),8):
            chunk=test[lo:lo+8];gs=engine.generate([prompt(r) for r in chunk],[a["shift"]]*len(chunk),max_new_tokens=1024)
            for r,g in zip(chunk,gs):
                record=dict(id=r["id"],strategy=name,**g,**grade(g["text"],r));saved.append(record);append(out/"test.jsonl",record)
        outcomes[name]=saved
    diff=[float(g["correct"])-float(b["correct"]) for b,g in zip(outcomes["baseline"],outcomes["gradient"])]
    summary=dict(baseline=float(np.mean([r["correct"] for r in outcomes["baseline"]])),gradient=float(np.mean([r["correct"] for r in outcomes["gradient"]])),
        delta=float(np.mean(diff)),paired_ci=paired_interval(diff),chosen=serial_action(chosen),test_count=len(test),
        frozen_model_parameters=True,gradient_check=check,note="Separate exploratory pilot with extra training-rationale supervision; small-sample interval.")
    atomic_json(out/"summary.json",summary);atomic_json(out/"complete.json",dict(completed_at=time.time()))
    print("GRADIENT COMPLETE",summary,flush=True)


if __name__=="__main__":main()

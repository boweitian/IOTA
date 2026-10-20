"""Exhaustively evaluate every new signed dose; preserve all prior discoveries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiment import ROOT,LAB,Execution,PilotEngine,read_records,append,prompt,grade,score_ifeval,rubric_prompt,parse_rubric,load_judge
from engine import atomic_json
from expanded_concepts import CONCEPTS
from expanded_geometry import DEFAULT
from sample_assets import DEFAULT as PREVIOUS,exhaustive_bank,geometry as old_geometry
from space import action,serial_action
from repair_empty_choice import score_with_empty

def bank_for(geo):
    # The original 37 shifts are copied verbatim; extending the BLAS input shape must not change them.
    bank=[]
    for a in exhaustive_bank(old_geometry()):
        bank.append(dict(a,coefficients=a['coefficients']+[0.]*18))
    for i in range(9,len(CONCEPTS)):
        for dose in [.5,1.]:
            for sign in [-1,1]:
                w=np.zeros(len(CONCEPTS));w[i]=sign*dose
                ident=f'node:{i}:{sign:+d}'+(':dose1' if dose==1 else '')
                a=action(w,geo['directions'],float(geo['scale']),ident)
                a.update(concept=CONCEPTS[i]['id'],concept_name=CONCEPTS[i]['name'],requested_strength=sign*dose)
                bank.append(a)
    assert len(bank)==109 and len({a['id'] for a in bank})==109
    return bank

def reuse(out,rows,bank):
    records=read_records(out/'records.jsonl');ids={r['id'] for r in rows}
    catalog={a['id']:serial_action(a) for a in bank}
    previous_actions=json.loads((PREVIOUS/'asset_bundle/actions.json').read_text())
    for a in previous_actions:
        catalog.setdefault(a['id'],dict(a,coefficients=a['coefficients']+[0.]*18))
    with (out/'records.jsonl').open('a') as f:
        for source in sorted((PREVIOUS/rows[0]['suite']).glob('shard-*/records.jsonl')):
            for no,line in enumerate(source.read_text().splitlines(),1):
                r=json.loads(line);key=(r['id'],r['action'])
                if r['id'] not in ids or key in records:continue
                copied=dict(r,reused=True,previous_source_file=str(source),previous_source_line=no)
                f.write(json.dumps(copied,ensure_ascii=False,allow_nan=False)+'\n');records[key]=copied
    atomic_json(out/'actions.json',list(catalog.values()))
    return records

class RecordedExecution(Execution):
    """Save actual cohort and raw outputs before grading; reuse them after interruption."""
    def execute(self,rows,bank):
        raw=read_records(self.out/'generations.jsonl')
        for a in bank:
            pending=[r for r in rows if (r['id'],a['id']) not in self.records]
            for lo in range(0,len(pending),self.protocol['budgets']['batch_size']):
                chunk=pending[lo:lo+self.protocol['budgets']['batch_size']]
                if chunk[0]['suite']=='truthfulqa':
                    for row in chunk:
                        labels=np.asarray(row['targets']['labels'],dtype=bool)
                        logp,seconds=self.engine.answer_likelihoods(prompt(row),row['targets']['choices'],a['shift'],a['extra'])
                        values=np.asarray(logp);prob=np.exp(values-values.max());prob/=prob.sum()
                        self.save(dict(id=row['id'],action=a['id'],suite=row['suite'],split=row['split'],
                            reward=float(prob[labels].sum()),choice_log_probabilities=logp,choice_probabilities=prob.tolist(),
                            generated=False,batch_seconds=seconds,truncated=False,new_tokens=0,
                            likelihood_dtype='float32',choice_batch_size=8,protocol='MC2 unnormalized answer token log-probability sum'))
                else:
                    fresh=[r for r in chunk if (r['id'],a['id']) not in raw]
                    if fresh:
                        cohort=[r['id'] for r in fresh]
                        batch_id=hashlib.sha256(json.dumps([str(self.out),a['id'],cohort]).encode()).hexdigest()[:24]
                        append(self.out/'batches.jsonl',dict(batch_id=batch_id,action=a['id'],sample_ids=cohort,max_new_tokens=self.protocol['budgets']['max_new_tokens']))
                        gs=self.engine.generate([prompt(r) for r in fresh],[a['shift']]*len(fresh),
                            [a['extra']]*len(fresh),max_new_tokens=self.protocol['budgets']['max_new_tokens'])
                        for row,g in zip(fresh,gs):
                            r=dict(id=row['id'],action=a['id'],suite=row['suite'],split=row['split'],
                                   generated=True,batch_id=batch_id,batch_sample_ids=cohort,**g)
                            append(self.out/'generations.jsonl',r);raw[row['id'],a['id']]=r
                    judging=[]
                    for row in chunk:
                        r=dict(raw[row['id'],a['id']])
                        if row['suite'] in ['gsm8k','bbh']:
                            scored=grade(r['text'],row);r.update(scored,reward=float(scored['correct']));self.save(r)
                        elif row['suite']=='ifeval':r.update(score_ifeval(row,r['text']));self.save(r)
                        else:judging.append((row,r))
                    if judging:self.judge_batch(judging)
                print('EXECUTED',self.out.parent.name,self.out.name,a['id'],lo+len(chunk),'/',len(pending),flush=True)
        return self.records

    def judge_batch(self,items):
        if self.judge is None:self.judge=load_judge(LAB)
        gs=self.judge.generate([rubric_prompt(row,r['text']) for row,r in items],[np.zeros(self.judge.dim)]*len(items),max_new_tokens=256)
        for (row,r),g in zip(items,gs):
            count=len(row['source_record']['checklist']);parsed=parse_rubric(g['text'],count);retries=[]
            for budget in [384,512,768]:
                if parsed is not None:break
                instruction=f'\nReturn only valid JSON. scores must contain exactly {count} numbers, each at least 1 and at most 10. Never use zero. Use 1 for a criterion that is entirely unmet.'
                retry=self.judge.generate([rubric_prompt(row,r['text'])+instruction],[np.zeros(self.judge.dim)],max_new_tokens=budget)[0]
                retries.append(retry);parsed=parse_rubric(retry['text'],count)
            append(self.out/'judge_attempts.jsonl',dict(id=row['id'],action=r['action'],raw=g,retries=retries,valid=parsed is not None))
            if parsed is None:raise RuntimeError('Malformed judge after strict retries; raw generation retained, no score invented')
            r.update(parsed,judge_raw=g,judge_retries=retries,judge_model=self.judge.meta);self.save(r)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--suite',required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=5);p.add_argument('--prepare',action='store_true');args=p.parse_args()
    geo=dict(np.load(args.run/'geometry/geometry.npz'));bank=bank_for(geo)
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text())
    rows=[r for r in protocol['records'] if r['suite']==args.suite][args.shard::args.shards]
    out=args.run/args.suite/f'shard-{args.shard}';out.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema_version=3,objective='Per-question exhaustive expanded geometry assets',protocol=protocol,
        bank=[serial_action(a) for a in bank],suite=args.suite,shard=args.shard,shards=args.shards,ids=[r['id'] for r in rows],
        original_asset_source=str(PREVIOUS),geometry_sha256=hashlib.sha256((args.run/'geometry/geometry.npz').read_bytes()).hexdigest(),
        source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['expanded_concepts.py','expanded_geometry.py','expanded_assets.py','experiment.py','pilot_engine.py','space.py']})
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Source or protocol changed; use a new run')
    atomic_json(out/'manifest.json',manifest);records=reuse(out,rows,bank)
    print('PREPARED',args.suite,args.shard,'missing',sum((r['id'],a['id']) not in records for r in rows for a in bank),flush=True)
    if args.prepare:return
    PilotEngine.answer_likelihoods=score_with_empty
    engine=PilotEngine(protocol['model'])
    if args.suite=='truthfulqa':engine.precise_likelihood_mode()
    atomic_json(out/'model.json',engine.meta)
    execution=RecordedExecution(engine,out,protocol);execution.execute(rows,bank)
    assert all((r['id'],a['id']) in execution.records for r in rows for a in bank)
    atomic_json(out/'complete.json',dict(samples=len(rows),required=len(rows)*len(bank),records=len(execution.records)))
    print('EXPANDED EXHAUSTIVE COMPLETE',args.suite,args.shard,flush=True)

if __name__=='__main__':main()

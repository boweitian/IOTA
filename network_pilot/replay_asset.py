"""Use an exported per-question direction asset on the frozen model."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment import ROOT,grade,score_ifeval
from pilot_engine import PilotEngine
from repair_empty_choice import score_with_empty
from engine import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--original-batch',action='store_true')
    args=p.parse_args();config=json.loads(args.config.read_text())
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text())
    row=next(r for r in protocol['records'] if r['id']==config['sample_id'])
    PilotEngine.answer_likelihoods=score_with_empty
    engine=PilotEngine(config['model']);shift=np.asarray(config['shift'])
    if row['suite']=='truthfulqa':
        engine.precise_likelihood_mode()
        logp,seconds=engine.answer_likelihoods(config['prompt'],row['targets']['choices'],shift)
        prob=np.exp(np.asarray(logp)-max(logp));prob/=prob.sum()
        observed=dict(choice_log_probabilities=logp,choice_probabilities=prob.tolist(),
            reward=float(prob[np.asarray(row['targets']['labels'],dtype=bool)].sum()),seconds=seconds,
            correct_top_choice=bool(row['targets']['labels'][int(prob.argmax())]))
    else:
        cohort=config.get('generation_batch',[]) if args.original_batch else []
        if cohort:
            outputs=engine.generate([r['prompt'] for r in cohort],[shift]*len(cohort),max_new_tokens=protocol['budgets']['max_new_tokens'])
            observed=outputs[next(i for i,r in enumerate(cohort) if r['id']==row['id'])]
        else:observed=engine.generate([config['prompt']],[shift],max_new_tokens=protocol['budgets']['max_new_tokens'])[0]
        if row['suite'] in ['gsm8k','bbh']:observed.update(grade(observed['text'],row))
        elif row['suite']=='ifeval':observed.update(score_ifeval(row,observed['text']))
    atomic_json(args.output,dict(sample_id=row['id'],action=config['action'],model=engine.meta,
        saved_evidence=config['evidence'],replayed=observed,
        original_batch=args.original_batch,executed_batch=config.get('generation_batch') if args.original_batch else [dict(id=row['id'],prompt=config['prompt'])],
        text_matches_saved=observed.get('text')==config['evidence'].get('text') if 'text' in observed else None))
    print(json.dumps(dict(sample=row['id'],action=config['action']['id'],
        correct=observed.get('correct',observed.get('correct_top_choice')),
        reward=observed.get('reward'),output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':main()

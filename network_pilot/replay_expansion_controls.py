"""Matched no-op/steer replays of newly discovered successes, separate from discovery scores."""
import json
from pathlib import Path
import numpy as np
from experiment import ROOT,grade
from pilot_engine import PilotEngine
from engine import atomic_json
from expanded_assets import DEFAULT

def main():
    out=DEFAULT/'replay';protocol=json.loads((ROOT/'inputs/protocol.json').read_text());rows={r['id']:r for r in protocol['records']}
    engine=PilotEngine(protocol['model']);results=[]
    for name in ['gsm3489','gsm433']:
        c=json.loads((out/f'{name}-config.json').read_text());cohort=c['generation_batch'];target=c['sample_id'];ix=next(i for i,r in enumerate(cohort) if r['id']==target)
        evidence={}
        for condition,shift in [('none',np.zeros(engine.dim)),('steered',np.asarray(c['shift']))]:
            gs=engine.generate([r['prompt'] for r in cohort],[shift]*len(cohort),max_new_tokens=protocol['budgets']['max_new_tokens'])
            evaluated=[dict(id=r['id'],**g,**grade(g['text'],rows[r['id']])) for r,g in zip(cohort,gs)]
            evidence[condition]=evaluated[ix]
            atomic_json(out/f'{name}-matched-{condition}.json',dict(model=engine.meta,condition=condition,
                action='none' if condition=='none' else c['action']['id'],generation_batch=cohort,outputs=evaluated))
            print('MATCHED REPLAY',target,condition,evaluated[ix]['correct'],flush=True)
        results.append(dict(sample=target,action=c['action']['id'],baseline_correct=evidence['none']['correct'],
            steered_correct=evidence['steered']['correct'],steered_text_matches_discovery=evidence['steered']['text']==c['evidence']['text'],
            matched_condition_rescue=not evidence['none']['correct'] and evidence['steered']['correct'],batch_size=len(cohort)))
    atomic_json(out/'matched_control_summary.json',dict(model=engine.meta,results=results,
        scope='Two targeted new discoveries; no-op and steer evaluated with identical input cohorts, GPU, dtype, token budget and prompts'))

if __name__=='__main__':main()

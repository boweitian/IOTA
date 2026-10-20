"""Export a measured expanded asset with exact native shift and actual input cohort."""
import argparse
import json
from pathlib import Path
import numpy as np
from expanded_assets import DEFAULT,PREVIOUS,bank_for
from experiment import ROOT,prompt,read_records
from engine import atomic_json

def export(run,sample,direction,output):
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text());rows={r['id']:r for r in protocol['records']}
    geo=dict(np.load(run/'geometry/geometry.npz'));bank={a['id']:a for a in bank_for(geo)}
    record=None
    for path in sorted((run/rows[sample]['suite']).glob('shard-*/records.jsonl')):
        record=read_records(path).get((sample,direction))
        if record is not None:break
    if record is None:raise ValueError('Unmeasured sample-direction pair')
    if direction in bank:a=bank[direction];shift=a['shift']
    else:
        catalog={a['id']:a for a in json.loads((PREVIOUS/'asset_bundle/actions.json').read_text())}
        old=catalog[direction];a=dict(old,coefficients=old['coefficients']+[0.]*18)
        shift=np.asarray(old['coefficients'])@geo['directions'][:9]
    cohort=[]
    if record.get('generated'):
        cohort=record.get('batch_sample_ids')
        if cohort is None:
            groups=json.loads((PREVIOUS/'asset_bundle/execution_batches.json').read_text())['cohorts']
            cohort=next(c['sample_ids'] for c in groups.values() if [sample,direction] in c['evidence_edges'])
        cohort=[dict(id=ident,prompt=prompt(rows[ident])) for ident in cohort]
    metadata=json.loads((run/'geometry/geometry.json').read_text())
    value=dict(schema_version=3,model=protocol['model'],sample_id=sample,question=rows[sample]['question'],prompt=prompt(rows[sample]),
        action={k:v for k,v in a.items() if k!='shift'},shift=np.asarray(shift).tolist(),geometry_sha256=metadata['geometry_sha256'],
        intervention='Add at layer index 15 last-token residual, prefill and each decode step',evidence=record,
        generation_batch=cohort,max_new_tokens=protocol['budgets']['max_new_tokens'])
    atomic_json(output,value);print('EXPORTED',sample,direction,output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--sample',required=True)
    p.add_argument('--direction',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args();export(a.run,a.sample,a.direction,a.output)

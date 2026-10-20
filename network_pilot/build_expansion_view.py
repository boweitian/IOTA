"""Portable interactive visualization of native geometry and actual saved evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from expanded_assets import DEFAULT,PREVIOUS,bank_for
from experiment import ROOT,prompt,read_records
from sample_assets import build_item,is_success
from expanded_concepts import CONCEPTS
from engine import atomic_json

def build(run):
    geo=dict(np.load(run/'geometry/geometry.npz'));bank=bank_for(geo)
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text())
    prior=list(map(json.loads,(PREVIOUS/'asset_bundle/sample_direction_dictionary.jsonl').read_text().splitlines()))
    records={(item['id'],name):r for item in prior for name,r in item['evidence'].items()}
    catalog={a['id']:dict(a,coefficients=a['coefficients']+[0.]*18) for a in json.loads((PREVIOUS/'asset_bundle/actions.json').read_text())}
    for out in sorted(run.glob('*/shard-*')):records.update(read_records(out/'records.jsonl'))
    catalog.update({a['id']:{k:v for k,v in a.items() if k!='shift'} for a in bank})
    catalog={a['id']:a for a in bank}|{name:a for name,a in catalog.items() if name not in {a['id'] for a in bank}}
    aids=list(catalog);aix={name:i for i,name in enumerate(aids)}
    configs=[]
    for name,a in catalog.items():
        nz=np.flatnonzero(a['coefficients'])
        node=int(nz[0]) if len(nz)==1 else None
        strength=float(a['coefficients'][node]) if node is not None else 0
        title='原模型' if name=='none' else (CONCEPTS[node]['name']+f' {a.get("requested_strength",strength):+g}' if node is not None else '历史混合 · '+name.removeprefix('historical:'))
        shift=a.get('shift')
        if shift is None:shift=np.asarray(a['coefficients'])@geo['directions']
        configs.append(dict(id=name,name=title,node=node,strength=a.get('requested_strength',strength),
                            coefficients=a['coefficients'],shift=np.asarray(shift).tolist(),norm=float(np.linalg.norm(shift)),extra=a.get('extra','')))
    oldcohorts=json.loads((PREVIOUS/'asset_bundle/execution_batches.json').read_text())['cohorts']
    cohort_by_edge={tuple(e):c['sample_ids'] for c in oldcohorts.values() for e in c['evidence_edges']}
    replay_path=run/'replay/matched_control_summary.json';replay={}
    if replay_path.exists():replay={(r['sample'],r['action']):r for r in json.loads(replay_path.read_text())['results']}
    rows=[]
    for row in protocol['records']:
        baseline=records[row['id'],'none'];evidence={}
        for (ident,name),r in records.items():
            if ident!=row['id']:continue
            obj=dict(reward=r['reward'],successful=is_success(row,r,baseline),improved=r['reward']>baseline['reward']+1e-12,
                truncated=r['truncated'],generated=r['generated'],new_tokens=r['new_tokens'])
            for field in ['text','parsed','extracted','instruction_results','scores','choice_probabilities','choice_log_probabilities','batch_size','batch_id']:
                if field in r:obj[field]=r[field]
            if r.get('generated'):obj['batch_sample_ids']=r.get('batch_sample_ids') or cohort_by_edge[ident,name]
            if (ident,name) in replay:obj['matched_replay']=replay[ident,name]
            evidence[str(aix[name])]=obj
        rows.append(dict(id=row['id'],suite=row['suite'],task=row['task'],question=row['question'],prompt=prompt(row),
            target=row.get('target'),targets=row.get('targets'),instruction_ids=row.get('instruction_id_list'),
            checklist=row.get('source_record',{}).get('checklist'),evidence=evidence))
    required=len(bank);profiles={}
    for suite in protocol['suites']:
        selected=[r for r in rows if r['suite']==suite]
        means=[];counts=[];improved=[];regressed=[];coverage=[]
        for j in range(required):
            observed=[r['evidence'][str(j)] for r in selected if str(j) in r['evidence']]
            counts.append(len(observed));means.append(float(np.mean([r['reward'] for r in observed])) if len(observed)==len(selected) else None)
            improved.append(sum(r['evidence'][str(j)]['reward']>r['evidence']['0']['reward']+1e-12 for r in selected if str(j) in r['evidence']))
            regressed.append(sum(r['evidence'][str(j)]['reward']<r['evidence']['0']['reward']-1e-12 for r in selected if str(j) in r['evidence']))
            coverage.append(sum(x['successful'] for x in observed))
        best_old=[max(r['evidence'][str(j)]['reward'] for j in range(37)) for r in selected]
        best_new=[max(e['reward'] for j,e in r['evidence'].items() if int(j)<required) for r in selected]
        profiles[suite]=dict(samples=len(selected),means=means,measured=counts,improved=improved,regressed=regressed,successful=coverage,
            baseline=means[0],old_best=float(np.mean(best_old)),expanded_best=float(np.mean(best_new)),
            best_fixed=max(m for m in means if m is not None),
            old_solved=sum(any(e['successful'] for j,e in r['evidence'].items() if int(j)<37) for r in selected),
            new_solved=sum(any(e['successful'] for j,e in r['evidence'].items() if int(j)<required) for r in selected),
            improved_over_old=sum(n>o+1e-12 for n,o in zip(best_new,best_old)),complete=all(x==len(selected) for x in counts))
    metadata=json.loads((run/'geometry/geometry.json').read_text())
    declared=sum(int(j)<required for r in rows for j in r['evidence'])
    data=dict(schema_version=3,model=protocol['model'],scale=float(geo['scale']),dimensions=1024,
        required_configurations=required,required_measurements=175*required,measured_required=declared,
        complete=declared==175*required,evidence_records=sum(len(r['evidence']) for r in rows),
        successful_records=sum(e['successful'] for r in rows for j,e in r['evidence'].items() if j!='0'),
        concepts=CONCEPTS,diagnostics=metadata['diagnostics'],coordinates=geo['coordinates'].tolist(),
        cosine=geo['cosine'].tolist(),projection_variance=geo['projection_variance'].tolist(),
        geometry_sha256=metadata['geometry_sha256'],configs=configs,rows=rows,suites=protocol['suites'],profiles=profiles,
        generation_budget=protocol['budgets']['max_new_tokens'],created_from='Actual saved native geometry and per-question measurements; unobserved cells stay empty')
    out=run/'view';out.mkdir(parents=True,exist_ok=True)
    serialized=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    atomic_json(out/'data_manifest.json',dict(complete=data['complete'],nodes=27,samples=175,declared_measurements=declared,
        evidence_records=data['evidence_records'],data_sha256=hashlib.sha256(serialized.encode()).hexdigest(),geometry_sha256=data['geometry_sha256']))
    page=(ROOT/'expansion_page.html').read_text().replace('__IOTA_EVIDENCE__',serialized.replace('</','<\\/'))
    (out/'index.html').write_text(page)
    print('VISUALIZED',out/'index.html','bytes',len(page.encode()),'measured',declared,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);a=p.parse_args();build(a.run.resolve())

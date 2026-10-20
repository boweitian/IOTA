"""Summarize router readiness: concept vs random best-of-N, and label reproducibility across batches."""
import json
from pathlib import Path
import numpy as np
from router_readiness import OUT,SUITES,ASSETS

def load(path):
    rows={}
    for f in sorted(path.glob('**/records.jsonl')):
        for line in f.read_text().splitlines():
            r=json.loads(line);rows[r['id'],r['action']]=bool(r['reward']>=1.)
    return rows

SUITE={r['id']:r['suite'] for r in json.loads((Path(__file__).resolve().parent/'inputs/protocol.json').read_text())['records']}
def suite_of(i):
    return SUITE[i]

def family_curve(S,cols_by_dir,k_values,rng,draws=400):
    """Coverage when k whole directions (all four signed doses each) are available, plus the base model."""
    out={}
    for k in k_values:
        cov=[]
        for _ in range(draws):
            pick=rng.choice(len(cols_by_dir),k,replace=False)
            cols=[c for d in pick for c in cols_by_dir[d]]
            cov.append(S[:,cols].any(1).mean())
        out[k]=float(np.mean(cov))
    return out

def crossfit_fixed(S,base,cols,rng,draws=400):
    """Pick the single best config on half the questions, score it on the other half (gain over base)."""
    gains=[]
    for _ in range(draws):
        perm=rng.permutation(len(S));a,b=perm[:len(S)//2],perm[len(S)//2:]
        j=cols[int(np.argmax(S[a][:,cols].mean(0)))]
        gains.append(S[b,j].mean()-base[b].mean())
    return float(np.mean(gains)),float(np.percentile(gains,2.5)),float(np.percentile(gains,97.5))

def main():
    rng=np.random.default_rng(0)
    m=np.load(ASSETS/'asset_bundle/response_matrix.npz');ids=[str(x) for x in m['sample_ids']];acts=[str(x) for x in m['action_ids']]
    keep=[i for i,s in enumerate(ids) if suite_of(s) in SUITES]
    rand=load(OUT/'random');rand_ids=sorted({a for _,a in rand},key=lambda a:(int(a.split(':')[1]),a))
    report=dict(best_of_n={},fixed_crossfit={},per_config_rates={})
    for suite in SUITES:
        rows=[i for i in keep if suite_of(ids[i])==suite];q=[ids[i] for i in rows]
        C=m['success'][rows];base=C[:,acts.index('none')]
        R=np.array([[rand[i,a] for a in rand_ids] for i in q])
        # Whole directions: concept node:k (4 configs each), random rand:k (4 configs each).
        cdirs=[[acts.index(a) for a in acts if a.startswith(f'node:{k}:')] for k in range(27)]
        rdirs=[[rand_ids.index(a) for a in rand_ids if a.startswith(f'rand:{k}:')] for k in range(12)]
        CB=np.column_stack([base,C]);RB=np.column_stack([base,R])
        ks=[1,2,4,8,12]
        report['best_of_n'][suite]=dict(base=float(base.mean()),
            concept={k:v for k,v in family_curve(CB,[[c+1 for c in d] for d in cdirs],ks,rng).items()},
            random={k:v for k,v in family_curve(RB,[[c+1 for c in d] for d in rdirs],ks,rng).items()})
        ccols=[c for d in cdirs for c in d];rcols=list(range(R.shape[1]))
        report['fixed_crossfit'][suite]=dict(concept=crossfit_fixed(C,base,ccols,rng),random=crossfit_fixed(R,base,rcols,rng))
        w=~base
        report['per_config_rates'][suite]=dict(concept_fix=float(C[w][:,ccols].mean()),random_fix=float(R[w].mean()),
            concept_break=float((~C[~w][:,ccols]).mean()),random_break=float((~R[~w]).mean()))
    # Reproducibility on the 36 replicated questions.
    design=json.loads((OUT/'replicate_design.json').read_text())
    reps=[load(OUT/'replicate'/f'rep-{r}') for r in range(1,design['reps']+1)]
    configs=['none']+design['concept_configs']+design['random_configs']
    def original(i,a):
        return rand[i,a] if a.startswith('rand') else bool(m['success'][ids.index(i),acts.index(a)])
    rows=[]
    for i in design['questions']:
        for a in configs:
            rows.append(dict(id=i,action=a,original=original(i,a),reps=[rep[i,a] for rep in reps]))
    rate=lambda xs:float(np.mean(xs)) if xs else None
    def agree(sel):
        return rate([x==r['original'] for r in sel for x in r['reps']])
    base_orig={i:original(i,'none') for i in design['questions']}
    fixes=[r for r in rows if r['action']!='none' and r['original'] and not base_orig[r['id']]]
    report['reproducibility']=dict(
        label_agreement_with_original=agree(rows),
        baseline_agreement=agree([r for r in rows if r['action']=='none']),
        original_fixes=len(fixes),
        fixes_reproduced_majority=rate([np.mean(r['reps'])>.5 for r in fixes]),
        fix_success_rate_in_reps=rate([x for r in fixes for x in r['reps']]),
        top_observed_concept_mean_original=rate([r['original'] for r in rows if r['action'] in design['top_observed']]),
        top_observed_concept_mean_reps=rate([x for r in rows if r['action'] in design['top_observed'] for x in r['reps']]),
        other_concept_mean_reps=rate([x for r in rows if r['action'] in design['concept_configs'][6:] for x in r['reps']]),
        random_mean_reps=rate([x for r in rows if r['action'].startswith('rand') for x in r['reps']]),
        none_mean_reps=rate([x for r in rows if r['action']=='none' for x in r['reps']]),
        robust_oracle=rate([any(np.mean(r['reps'])>.5 for r in rows if r['id']==i) for i in design['questions']]),
        single_run_oracle=rate([any(r['original'] for r in rows if r['id']==i) for i in design['questions']]))
    (OUT/'readiness_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=1))
    print(json.dumps(report,ensure_ascii=False,indent=1))

if __name__=='__main__':main()

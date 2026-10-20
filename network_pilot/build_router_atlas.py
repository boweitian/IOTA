"""Answer-aware direction assets and a constrained, direction-aligned PCA view."""
import argparse
import csv
import io
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

import gsm_router_large as core
import state_geometry_router as state
import state_geometry_models as models
import router_direction_scan as scan

ROOT=core.ROOT
DIST=core.LAB/'dist'
METHODS=state.METHODS
METHOD_NAMES={'baseline':'原模型','fixed':'训练最佳固定方向','text':'原文本 Router',
              'hidden_reward':'隐藏状态评分器','geometry_raw':'新 Router','geometry_gated':'新 Router＋回退'}
ADVANTAGE_RUN=ROOT/'runs/fixed-advantage-router-v1'
if (ADVANTAGE_RUN/'policies_frozen.json').exists():
    METHOD_NAMES['fixed_advantage']='固定方向回退 Router · 本轮实验'


def aligned_pca(training_states,direction):
    """x is exactly the signed intervention; residual PCs are fitted on train only."""
    x=np.asarray(training_states,dtype=float);direction=np.asarray(direction,dtype=float)
    norm=np.linalg.norm(direction)
    if norm<1e-12:raise ValueError('No-op has no direction to align; select a nonzero reference')
    u=direction/norm;mean=x.mean(0);centered=x-mean
    residual=centered-np.outer(centered@u,u)
    _,_,vt=np.linalg.svd(residual,full_matrices=False)
    axes=[u]
    for v in vt[:2]:
        for a in axes:v=v-(v@a)*a
        v=v/np.linalg.norm(v)
        if v[np.argmax(np.abs(v))]<0:v=-v
        axes.append(v)
    axes=np.stack(axes);coordinates=centered@axes.T
    total=float(np.mean((centered*centered).sum(1)))
    fractions=np.mean(coordinates**2,axis=0)/max(total,1e-12)
    assert np.allclose(axes@axes.T,np.eye(3),atol=1e-10)
    assert float(axes[0]@u)>1-1e-12
    return dict(mean=mean,axes=axes,variance_fraction=fractions,
                alignment_cosine=float(axes[0]@u),orthogonality_error=float(np.max(np.abs(axes@axes.T-np.eye(3)))))


def best_representative(rewards,train_support):
    """Do not fabricate an effective direction on all-failed/unmeasured rows."""
    successes=np.flatnonzero(np.asarray(rewards)==1)
    if not len(successes):return None,[]
    representative=int(successes[np.argmax(np.asarray(train_support)[successes])])
    return representative,successes.tolist()


def projection_bank(run,suite,states,train,shifts):
    """Cache train-only residual PCA for every distinct unoriented bank direction."""
    path=run/suite/'projection_bank.npz'
    representatives=[];mapping=[];signs=[]
    for shift in shifts:
        norm=np.linalg.norm(shift)
        if norm<1e-12:mapping.append(-1);signs.append(0);continue
        u=shift/norm;sign=1 if u[np.argmax(np.abs(u))]>0 else -1;canonical=u*sign
        found=next((i for i,v in enumerate(representatives) if np.allclose(v,canonical,atol=1e-10)),None)
        if found is None:found=len(representatives);representatives.append(canonical)
        mapping.append(found);signs.append(sign)
    if path.exists():
        cached=np.load(path);axes=cached['axes'];fractions=cached['fractions']
        assert np.allclose(cached['representatives'],representatives,atol=1e-12)
    else:
        fitted=[aligned_pca(states[train],u) for u in representatives]
        axes=np.stack([p['axes'] for p in fitted]);fractions=np.stack([p['variance_fraction'] for p in fitted])
        np.savez_compressed(path,axes=axes,fractions=fractions,representatives=representatives)
    mean=np.asarray(states[train],dtype=float).mean(0)
    entries=[]
    for basis,fraction in zip(axes,fractions):
        assert np.allclose(basis@basis.T,np.eye(3),atol=1e-10)
        entries.append(dict(coordinates=np.round((states-mean)@basis.T,7).tolist(),
             vectors=np.round(shifts@basis.T,9).tolist(),variance_fraction=fraction.tolist(),
             orthogonality_error=float(np.max(np.abs(basis@basis.T-np.eye(3))))))
    cosines=[]
    for j,k in enumerate(mapping):
        cosine=float(axes[k,0]@((shifts[j]/np.linalg.norm(shifts[j]))*signs[j])) if k>=0 else None
        if k>=0:assert cosine>1-1e-12
        cosines.append(cosine)
    return dict(entries=entries,mapping=mapping,signs=signs,alignment_cosines=cosines)


def all_choices(source,p,features):
    from fit_asset_router import input_text
    actions=np.load(source/'actions.npz')['ids'].astype(str);aix={a:i for i,a in enumerate(actions)}
    rows=p['records'];n=len(rows)
    fixed=json.loads((source/'training_summary.json').read_text())['fixed_action']
    output=dict(baseline=np.zeros(n,dtype=int),fixed=np.full(n,aix[fixed],dtype=int))
    old=joblib.load(source/'router-set.joblib')
    x=old['transform'].transform([input_text(r['suite'],r['question']) for r in rows])
    output['text']=np.asarray(x@old['w']+old['bias']).argmax(1)
    for name in ['hidden_reward','geometry']:
        bundle=joblib.load(source/f'router-{name}.joblib');x=models.hidden_input(features,bundle['kind'])
        score=models.predict(bundle['model'],x)
        if name=='geometry':
            output['geometry_raw']=models.choose(score)
            threshold=bundle['selection']['threshold']
            output['geometry_gated']=models.choose(score,float('inf') if threshold is None else threshold)
        else:output[name]=models.choose(score)
    if 'fixed_advantage' in METHOD_NAMES:
        import fixed_advantage_models as advantage
        bundle=joblib.load(ADVANTAGE_RUN/source.name/'router.joblib')
        frozen_models=json.loads((ADVANTAGE_RUN/'policies_frozen.json').read_text())
        assert core.sha(ADVANTAGE_RUN/source.name/'router.joblib')==frozen_models['models'][source.name]
        output['fixed_advantage']=advantage.predict(bundle,advantage.public(rows),features)[0]
    frozen={r['id']:r for r in json.loads((source/'frozen_test_actions.json').read_text())}
    for i,row in enumerate(rows):
        if row['split']=='train':continue
        for method in METHODS:assert actions[output[method][i]]==frozen[row['id']][method]
    return output


def build_suite(run,suite,assets):
    source=scan.SOURCE/suite;p=json.loads((source/'protocol.json').read_text());rows=p['records']
    features,_=state.load_features(source,p)
    bank=np.load(source/'actions.npz');actions=bank['ids'].astype(str);shifts=bank['shifts'].astype(float)
    catalog=json.loads((source/'actions.json').read_text());assert [a['id'] for a in catalog]==actions.tolist()
    aix={a:i for i,a in enumerate(actions)};rix={r['id']:i for i,r in enumerate(rows)}
    train=np.array([r['split']=='train' for r in rows]);old=np.load(source/'train_matrix.npz')
    matrix=np.full((len(rows),len(actions)),-1,dtype=np.int8)
    for i,ident in enumerate(old['ids'].astype(str)):matrix[rix[ident]]=old['reward'][i]
    full=run/suite/'full_matrix.npz';complete=full.exists()
    if complete:
        full_data=np.load(full);assert full_data['ids'].astype(str).tolist()==[r['id'] for r in rows]
        assert full_data['actions'].astype(str).tolist()==actions.tolist()
        matrix=full_data['reward'].astype(np.int8)
    choices=all_choices(source,p,features)
    # While the full sweep is running, historical test observations are shown as
    # partial evidence with their own provenance, never called full-bank best.
    if not complete:
        for k in range(6):
            for r in core.read(source/'test'/f'shard-{k}/records.jsonl'):
                matrix[rix[r['id']],aix[r['action']]]=int(r['reward'])
    support=matrix[train].sum(0);fixed=int(np.argmax(support))
    assert fixed==int(choices['fixed'][0]) and np.linalg.norm(shifts[fixed])>0
    projection=aligned_pca(features['steer_last'][train],shifts[fixed])
    xy=(features['steer_last'].astype(float)-projection['mean'])@projection['axes'].T
    projected_shifts=shifts@projection['axes'].T
    assert np.allclose(projected_shifts[fixed,1:],0,atol=1e-10)
    exported_actions=[]
    for j,a in enumerate(catalog):
        exported_actions.append(dict(id=a['id'],name='不干预' if a['id']=='none' else
             a['concept_name']+f" {a['requested_strength']:+g}",norm=float(np.linalg.norm(shifts[j])),
             coordinates=projected_shifts[j].tolist(),support=int(support[j])))
    exported_rows=[]
    for i,row in enumerate(rows):
        representative,valid=best_representative(matrix[i],support)
        observed=int((matrix[i]>=0).sum());row_complete=observed==len(actions)
        exported_rows.append(dict(id=row['id'],question=row['question'],task=row['task'],
            split=row['split'],group=row['eval_group'],target=row.get('target'),
            instruction_ids=row.get('instruction_id_list'),coordinates=xy[i].tolist(),
            predictions={m:int(choices[m][i]) for m in METHOD_NAMES},
            rewards=[None if v<0 else int(v) for v in matrix[i]],
            best=representative,successful=valid,best_complete=row_complete,observed=observed,
            provenance='original exhaustive train measurements' if row['split']=='train' else
                 'new posthoc full-bank sweep; frozen policies' if complete else 'original six-method test; incomplete action bank'))
    bank_views=projection_bank(run,suite,features['steer_last'].astype(float),train,shifts)
    axes_bank=np.load(run/suite/'projection_bank.npz')
    np.savez_compressed(assets/suite/'all_direction_axes.npz',axes=axes_bank['axes'],mean=projection['mean'],
        actions=actions,mapping=bank_views['mapping'],signs=bank_views['signs'],
        alignment_cosines=np.array([np.nan if v is None else v for v in bank_views['alignment_cosines']]),
        ids=np.array([r['id'] for r in rows]),
        coordinates=np.array([v['coordinates'] for v in bank_views['entries']]),
        variance_fraction=axes_bank['fractions'])
    data=dict(suite=suite,rows=exported_rows,actions=exported_actions,fixed=fixed,complete=complete,projection_bank=bank_views,
              total=len(rows),n_train=int(train.sum()),n_test=int((~train).sum()),
              projection=dict(axis_action=actions[fixed],axis_name=exported_actions[fixed]['name'],
                    layer_index=p['model']['layer_index'],dimensions=shifts.shape[1],
                    alignment_cosine=projection['alignment_cosine'],orthogonality_error=projection['orthogonality_error'],
                    variance_fraction=projection['variance_fraction'].tolist(),fit_split='train only',
                    method='fixed signed direction + two PCA components in its orthogonal complement; raw residual coordinates, no whitening'),
              policy_source='state-geometry-router-v1; unchanged weights and choices',model=p['model']['model'],
              methods=METHOD_NAMES,source_protocol_sha256=core.sha(source/'protocol.json'))
    dest=assets/suite;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'aligned_projection.npz',mean=projection['mean'],axes=projection['axes'],
        ids=np.array([r['id'] for r in rows]),coordinates=xy,steering_vectors=shifts,projected_vectors=projected_shifts,
        actions=actions,variance_fraction=projection['variance_fraction'])
    np.savez_compressed(dest/'direction_matrix.npz',ids=np.array([r['id'] for r in rows]),actions=actions,reward=matrix)
    core.save(dest/'data.json',data)
    core.save(dest/'projection_audit.json',dict(alignment_cosine=projection['alignment_cosine'],
        max_orthogonality_error=projection['orthogonality_error'],
        fixed_vector_off_axis_norm=float(np.linalg.norm(projected_shifts[fixed,1:])),
        pca_fitted_on_train_only=True,policy_predictions_match_original_frozen_test=True,
        full_bank_oracle_complete=complete,samples=len(rows),observed_cells=int((matrix>=0).sum())))
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['id','split','task','question','observed','best_complete','best_representative','successful_actions',
                     'router_action','router_correct','baseline_correct','x_aligned','y_residual_pc1','z_residual_pc2'])
    for row in exported_rows:
        question=row['question'];question="'"+question if question.startswith(('=','+','-','@')) else question
        chosen=row['predictions']['geometry_raw']
        writer.writerow([row['id'],row['split'],row['task'],question,row['observed'],row['best_complete'],
             actions[row['best']] if row['best'] is not None else '',
             '|'.join(actions[j] for j in row['successful']),actions[chosen],row['rewards'][chosen],row['rewards'][0],
             *row['coordinates']])
    (dest/'questions.csv').write_text('\ufeff'+stream.getvalue())
    return data


def plot_exports(data,assets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for suite,d in data.items():
        for group in ['train','test']:
            rows=[r for r in d['rows'] if (r['split']=='train')==(group=='train')]
            fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
            for ax,method in zip(axes.ravel(),['baseline','fixed','text','geometry_raw']):
                for result,color,label in [(0,'#c85d55','Incorrect'),(1,'#2f876d','Correct')]:
                    selected=[r for r in rows if r['rewards'][r['predictions'][method]]==result]
                    xy=np.array([r['coordinates'][:2] for r in selected])
                    if len(xy):ax.scatter(xy[:,0],xy[:,1],s=12,alpha=.55,c=color,label=f'{label} ({len(xy)})',linewidths=0,rasterized=True)
                known=[r for r in rows if r['rewards'][r['predictions'][method]] is not None]
                accuracy=np.mean([r['rewards'][r['predictions'][method]] for r in known]) if known else float('nan')
                ax.set_title(f'{method}  {accuracy:.1%}');ax.set_aspect('equal',adjustable='datalim')
                ax.set_xlabel('Projection along train-best signed steering direction')
                ax.set_ylabel('Orthogonal residual PC1');ax.grid(alpha=.15);ax.legend(fontsize=8)
            fig.suptitle(f'{suite.upper()} / {group} / layer 15 / {d["projection"]["axis_action"]}\n'
                         'x exactly aligned; residual PCA fitted on training questions only',fontsize=12)
            fig.savefig(assets/suite/f'{group}_distribution.png',dpi=180)
            fig.savefig(assets/suite/f'{group}_distribution.svg')
            plt.close(fig)


def build(run,dist,plots=True):
    assets=dist/'assets/router-atlas';assets.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'router_atlas_README.md',assets/'README.md')
    data={suite:build_suite(run,suite,assets) for suite in scan.SUITES}
    payload=dict(version=1,suites=data,methods=METHOD_NAMES,default_method='geometry_raw',
        complete=all(v['complete'] for v in data.values()),
        description='Post-hoc finite-bank oracle diagnostic; models and routing remain frozen.')
    serialized=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    page=(ROOT/'router_atlas_page.html').read_text().replace('__ROUTER_ATLAS_DATA__',serialized)
    temp=dist/'router-analysis.tmp.html';temp.write_text(page);temp.replace(dist/'router-analysis.html')
    shutil.copy2(ROOT/'router_atlas.js',dist/'router_atlas.js')
    if (run/'status.json').exists():shutil.copy2(run/'status.json',assets/'status.json')
    if plots:plot_exports(data,assets)
    import sys
    sys.path.insert(0,str(core.LAB))
    from router_atlas_mount import mount_router_atlas
    mount_router_atlas(dist)
    core.save(run/'visualization_audit.json',dict(published_to=str(dist),complete=payload['complete'],
        suites={s:dict(samples=d['total'],complete=d['complete'],alignment=d['projection']['alignment_cosine']) for s,d in data.items()},
        uses_real_hidden_states=True,test_oracle_uses_answers=True,models_retrained=False))
    print('PUBLISHED ROUTER ATLAS',dist/'router-analysis.html','complete',payload['complete'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=scan.DEFAULT)
    p.add_argument('--dist',type=Path,default=DIST);p.add_argument('--skip-plots',action='store_true');a=p.parse_args()
    build(a.run,a.dist,not a.skip_plots)

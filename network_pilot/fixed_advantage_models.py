"""Small, train-selected repair/damage routers anchored to a fixed direction.

All preprocessing, reference actions, action masks and gates are fitted on the
training fold. Full success sets supervise two joint-event heads for every action.
The bounded head scores are not claimed to be calibrated probabilities.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import state_geometry_models as previous

SEED=20261018
KINDS=('text','hybrid','hybrid_rbf')
ALPHAS=(.1,1.,10.)
PENALTIES=(1.,2.)
THRESHOLDS=(0.,.03,.1,float('inf'))
TOP_COUNTS=(16,109)


def public(rows):
    return [{k:r[k] for k in ('suite','task','question')} for r in rows]


def fit_features(rows,hidden):
    texts=[r['question'] for r in rows]
    word=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,lowercase=False,
                         token_pattern=r'(?u)\b\w+\b|[^\w\s]')
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True)
    w=word.fit_transform(texts);c=char.fit_transform(texts)
    return dict(word=word,char=char),dict(word=w,char=c,hidden=previous.hidden_input(hidden,'multi'),
                                        task=np.array([r['task'] for r in rows]))


def transform(bundle,rows,hidden):
    return dict(word=bundle['word'].transform([r['question'] for r in rows]),
        char=bundle['char'].transform([r['question'] for r in rows]),
        hidden=previous.hidden_input(hidden,'multi'),task=np.array([r['task'] for r in rows]))


def take(features,index):
    return {k:v[index] for k,v in features.items()}


def kernel(left,right,kind,bandwidth=None):
    text=.5*((left['word']@right['word'].T).toarray()+(left['char']@right['char'].T).toarray())
    if kind=='text':return text
    hidden=previous.state_kernel(left['hidden'],right['hidden'],bandwidth if kind=='hybrid_rbf' else None)
    result=.5*text+.5*hidden
    if kind=='hybrid_task':result*=.25+.75*(left['task'][:,None]==right['task'][None,:])
    return result


def targets(y,fixed):
    base=y[:,fixed,None]
    return np.concatenate([(1-base)*y,base*(1-y)],axis=1)


def factor(gram):
    mean=gram.mean(0);grand=float(mean.mean())
    centered=gram-mean[None,:]-mean[:,None]+grand
    values,vectors=np.linalg.eigh(centered)
    return mean,grand,np.maximum(values,0),vectors


def solve(decomposition,labels,alpha):
    mean,grand,values,vectors=decomposition;prior=labels.mean(0)
    coef=(vectors*(1/(values+alpha)))@(vectors.T@(labels-prior))
    return dict(kernel_mean=mean,grand=grand,coef=coef,prior=prior)


def predict_heads(fitted,cross):
    centered=cross-cross.mean(1)[:,None]-fitted['kernel_mean'][None,:]+fitted['grand']
    return np.clip(centered@fitted['coef']+fitted['prior'],0.,1.)


def choose(heads,fixed,support,penalty=1.,threshold=0.,top=109):
    actions=heads.shape[1]//2;gain=heads[:,:actions]-penalty*heads[:,actions:]
    gain[:,fixed]=0.
    keep=np.argsort(-np.asarray(support),kind='stable')[:top]
    mask=np.zeros(actions,dtype=bool);mask[keep]=True;mask[fixed]=True;mask[0]=True
    gain[:,~mask]=-np.inf
    choice=gain.argmax(1);confidence=gain[np.arange(len(gain)),choice]
    return np.where(confidence>threshold,choice,fixed),confidence


def folds_for(rows):
    """Stratify by public task; each question/action row stays wholly together."""
    result=np.empty(len(rows),dtype=int);rng=np.random.default_rng(SEED)
    for task in sorted({r['task'] for r in rows}):
        ix=np.array([i for i,r in enumerate(rows) if r['task']==task])
        result[rng.permutation(ix)]=np.arange(len(ix))%3
    return result


def select(rows,hidden,y):
    folds=folds_for(rows);kinds=list(KINDS)
    if len({r['task'] for r in rows})>1:kinds.append('hybrid_task')
    oof={};fixed_oof=np.empty(len(y),dtype=int);support_by_fold={}
    for fold in range(3):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
        support=y[tr].sum(0);fixed=int(support.argmax());fixed_oof[va]=fixed;support_by_fold[fold]=support
        feature_model,reference=fit_features([rows[i] for i in tr],take(hidden,tr))
        validation=transform(feature_model,[rows[i] for i in va],take(hidden,va))
        labels=targets(y[tr],fixed)
        for kind in kinds:
            bandwidth=previous.state_bandwidth(reference['hidden']) if kind=='hybrid_rbf' else None
            decomposition=factor(kernel(reference,reference,kind,bandwidth))
            cross=kernel(validation,reference,kind,bandwidth)
            for alpha in ALPHAS:
                oof.setdefault((kind,alpha),np.zeros((len(y),2*y.shape[1])))
                oof[kind,alpha][va]=predict_heads(solve(decomposition,labels,alpha),cross)
    base=y[np.arange(len(y)),fixed_oof];trials=[]
    for (kind,alpha),heads in oof.items():
        for penalty in PENALTIES:
            for top in TOP_COUNTS:
                for threshold in THRESHOLDS:
                    choices=np.empty(len(y),dtype=int)
                    for fold in range(3):
                        va=np.flatnonzero(folds==fold);fixed=int(fixed_oof[va[0]])
                        choices[va]=choose(heads[va],fixed,support_by_fold[fold],penalty,threshold,top)[0]
                    hits=y[np.arange(len(y)),choices]
                    trials.append(dict(kind=kind,alpha=alpha,penalty=penalty,top=top,
                        threshold=None if np.isinf(threshold) else threshold,
                        correct=int(hits.sum()),n=len(y),fixed_correct=int(base.sum()),
                        repairs=int(((base==0)&(hits==1)).sum()),breaks=int(((base==1)&(hits==0)).sum()),
                        switches=int((choices!=fixed_oof).sum())))
    best=min(trials,key=lambda t:(-t['correct'],t['breaks'],t['switches'],-t['alpha'],
                                 t['top'],t['kind'],t['penalty']))
    return best,trials,folds


def fit(rows,hidden,y,config):
    preprocessing,reference=fit_features(rows,hidden)
    support=y.sum(0);fixed=int(support.argmax());kind=config['kind']
    bandwidth=previous.state_bandwidth(reference['hidden']) if kind=='hybrid_rbf' else None
    fitted=solve(factor(kernel(reference,reference,kind,bandwidth)),targets(y,fixed),config['alpha'])
    return dict(preprocessing=preprocessing,reference=reference,kind=kind,bandwidth=bandwidth,
                fitted=fitted,fixed=fixed,support=support,config=config,
                feature_rule='Question text, public task and unsteered prompt states only; no ID, target or response.')


def predict(bundle,rows,hidden):
    x=transform(bundle['preprocessing'],rows,hidden)
    heads=predict_heads(bundle['fitted'],kernel(x,bundle['reference'],bundle['kind'],bundle['bandwidth']))
    config=bundle['config'];threshold=float('inf') if config['threshold'] is None else config['threshold']
    return choose(heads,bundle['fixed'],bundle['support'],config['penalty'],threshold,config['top'])

"""Small reward regressors over question-state and intervention geometry.

No test outcomes enter fitting, preprocessing, calibration, or threshold selection.
Every observed train question/action reward (including failures) is supervision.
"""
import numpy as np

SEED=20261017
ALPHAS=(.03,.3,3.,30.)
ACTION_STRENGTHS=(0.,.3,3.)
GATES=(0.,.03,.1,.2,float('inf'))


def normalized(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def hidden_input(features,kind):
    kind=kind.removesuffix('_rbf')
    if kind=='final':return normalized(features['final_last'].astype(float))
    if kind=='multi':
        return np.concatenate([normalized(features[k].astype(float)) for k in
                               ['steer_last','final_last','final_mean']],axis=1)/np.sqrt(3.)
    raise ValueError(kind)


def action_kernel(shifts):
    """RBF of actual 1024-dimensional shifts, so angle, sign and dose matter."""
    norms=np.linalg.norm(shifts,axis=1)
    scale=max(norms.max(),1e-12)
    z=np.asarray(shifts,dtype=float)/scale
    distance=np.maximum((z*z).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*z@z.T,0.)
    return np.exp(-distance/2.)


def action_filter(kernel,strength):
    if strength==0:return np.eye(len(kernel))
    values,vectors=np.linalg.eigh(kernel)
    values=np.maximum(values,0.)
    return (vectors*(values/(values+strength)))@vectors.T


def state_kernel(left,right,bandwidth=None):
    dot=np.asarray(left@right.T)
    if bandwidth is None:return dot
    distance=np.maximum((left*left).sum(1)[:,None]+(right*right).sum(1)[None,:]-2*dot,0.)
    return np.exp(-distance/(2*bandwidth))


def state_bandwidth(x):
    distance=np.maximum((x*x).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*x@x.T,0.)
    positive=distance[np.triu_indices(len(x),1)]
    positive=positive[positive>1e-12]
    return float(np.median(positive)) if len(positive) else 1.


def fit_reward(x,y,alpha,geometry=None,strength=0.,kernel='linear'):
    """Linear ridge state kernel plus a geometry kernel on the output actions.

    Centered action priors retain the global train-supported direction. Geometry
    regularization shares conditional residuals across nearby interventions.
    The resulting score is a bilinear question-state/action-geometry function.
    """
    mean=x.mean(0);centered=x-mean;prior=y.mean(0)
    bandwidth=state_bandwidth(centered) if kernel=='rbf' else None
    gram=state_kernel(centered,centered,bandwidth)
    coef=np.linalg.solve(gram+alpha*np.eye(len(x)),y-prior)
    smoother=action_filter(geometry,strength) if strength else np.eye(y.shape[1])
    coef=coef@smoother
    return dict(mean=mean,reference=centered,coef=coef,prior=prior,alpha=alpha,strength=strength,
                bandwidth=bandwidth,kernel=kernel)


def predict(model,x):
    return state_kernel(x-model['mean'],model['reference'],model.get('bandwidth'))@model['coef']+model['prior']


def choose(scores,threshold=0.):
    choice=scores.argmax(1)
    gain=scores[np.arange(len(scores)),choice]-scores[:,0]
    return np.where(gain>threshold,choice,0)


def three_folds(n):
    permutation=np.random.default_rng(SEED).permutation(n)
    assignment=np.empty(n,dtype=int);assignment[permutation]=np.arange(n)%3
    return assignment


def calibration(scores,y):
    """Optional monotonic probability readout; fitted on OOF train predictions.

    Direction selection uses raw reward scores; no false probability calibration
    guarantee is made. This readout can be audited separately with Brier scores.
    """
    from sklearn.linear_model import LogisticRegression
    labels=y.ravel().astype(int)
    if len(np.unique(labels))<2:return None
    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=200)
    model.fit(scores.reshape(-1,1),labels)
    return dict(coef=float(model.coef_[0,0]),intercept=float(model.intercept_[0]))


def probabilities(cal,scores):
    from scipy.special import expit
    return np.clip(scores,0,1) if cal is None else expit(cal['coef']*scores+cal['intercept'])


def select(candidates,y,folds,geometry,allow_geometry):
    """Select one ridge model on three train-only folds; all labels stay grouped."""
    n=len(y);trials=[];oof_predictions={}
    # A fold-local spectral decomposition is reused across regularizers. This is
    # the same ridge solution as fit_reward, without repeatedly factoring Kx.
    for kind,x in candidates.items():
        fold_cache=[]
        for fold in range(3):
            tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
            xm=x[tr].mean(0);xc=x[tr]-xm
            bandwidth=state_bandwidth(xc) if kind.endswith('_rbf') else None
            values,vectors=np.linalg.eigh(state_kernel(xc,xc,bandwidth))
            fold_cache.append((tr,va,np.maximum(values,0),vectors,
                               state_kernel(x[va]-xm,xc,bandwidth),y[tr].mean(0)))
        for alpha in ALPHAS:
            base=np.zeros_like(y,dtype=float)
            for tr,va,values,vectors,cross,prior in fold_cache:
                coefficients=(vectors*(1/(values+alpha)))@(vectors.T@(y[tr]-prior))
                base[va]=cross@coefficients
            for strength in (ACTION_STRENGTHS if allow_geometry else (0.,)):
                if allow_geometry and strength==0:continue
                score=base@action_filter(geometry,strength)
                for tr,va,_,_,_,prior in fold_cache:score[va]+=prior
                raw_choice=choose(score)
                raw_hits=y[np.arange(n),raw_choice]
                # The independent-action ablation is scored without gating; the
                # main geometry model also selects a no-op threshold internally.
                for threshold in (GATES if allow_geometry else (0.,)):
                    choice=choose(score,threshold);hits=y[np.arange(n),choice]
                    trials.append(dict(kind=kind,alpha=alpha,strength=strength,
                         threshold=None if np.isinf(threshold) else threshold,
                         correct=int(hits.sum()),n=n,raw_correct=int(raw_hits.sum()),
                         fixes=int(((hits==1)&(y[:,0]==0)).sum()),
                         breaks=int(((hits==0)&(y[:,0]==1)).sum()),
                         interventions=int((choice!=0).sum()),
                         mse=float(np.mean((score-y)**2))))
                oof_predictions[kind,alpha,strength]=score
    # Equal OOF accuracy: prefer fewer breaks, fewer interventions, then stronger
    # regularization; no test outcomes or test scores influence the tie break.
    best=min(trials,key=lambda t:(-t['correct'],t['breaks'],t['interventions'],-t['alpha'],
                                 t['kind'],t['strength']))
    selected=oof_predictions[best['kind'],best['alpha'],best['strength']]
    return best,trials,selected

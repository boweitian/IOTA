"""Training-only organization of a frozen geometry agent bank."""
import numpy as np

def normalize(p):
    p=np.maximum(np.asarray(p,dtype=float),1e-12)
    return p/p.sum(axis=-1,keepdims=True)

def accuracy(p,y):return float(np.mean(np.argmax(p,axis=-1)==y))
def nll(p,y):return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())

def folds(y,count=5,seed=20261019):
    result=np.empty(len(y),dtype=int);rng=np.random.default_rng(seed)
    for label in np.unique(y):
        ix=np.flatnonzero(y==label);rng.shuffle(ix)
        result[ix]=np.arange(len(ix))%count
    return result

def fit_team(p,y,size):
    """Greedy, distinct members; common weight grid fixed before seeing labels."""
    best=min(range(p.shape[1]),key=lambda j:(-accuracy(p[:,j],y),nll(p[:,j],y),j))
    members=[best];weights=[1.];combined=p[:,best].copy()
    while len(members)<size:
        candidates=[]
        for j in range(p.shape[1]):
            if j in members:continue
            for alpha in [.25,.5,.75]:
                q=(1-alpha)*combined+alpha*p[:,j]
                candidates.append(((-accuracy(q,y),nll(q,y),j,alpha),j,alpha,q))
        _,j,alpha,combined=min(candidates,key=lambda row:row[0])
        members.append(j);weights=[w*(1-alpha) for w in weights]+[alpha]
    return dict(members=members,weights=weights)

def fit_policy(name,p,x,y,reference):
    if name=='baseline':return dict(name=name,members=[0],weights=[1.])
    if name=='uniform_3':return dict(name=name,members=reference,weights=[1/3]*3)
    if name=='best_single':return dict(name=name,**fit_team(p,y,1))
    if name.startswith('team_'):return dict(name=name,**fit_team(p,y,int(name.split('_')[1])))
    team=fit_team(p,y,3);members=team['members'];lam=float(name.split('_')[-1])
    mean=x.mean(0);std=np.maximum(x.std(0),1e-6);z=np.column_stack([np.ones(len(x)),(x-mean)/std])
    target=np.log(np.maximum(p[np.arange(len(y))[:,None],members,y[:,None]],1e-6))
    # Predict per-member utility from input geometry, with no label available at inference.
    penalty=np.eye(z.shape[1])*lam;penalty[0,0]=0
    coef=np.linalg.solve(z.T@z+penalty,z.T@target)
    return dict(name=name,**team,router=dict(mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),ridge=lam))

def predict(policy,p,x):
    member_prob=p[:,policy['members']]
    if 'router' not in policy:
        return normalize(np.einsum('nak,a->nk',member_prob,np.array(policy['weights'])))
    router=policy['router'];z=np.column_stack([np.ones(len(x)),(x-np.array(router['mean']))/np.array(router['std'])])
    utility=z@np.array(router['coef']);utility-=utility.max(1,keepdims=True)
    weights=np.exp(utility);weights/=weights.sum(1,keepdims=True)
    return normalize(np.einsum('nak,na->nk',member_prob,weights))

def train(p,x,y,reference):
    names=['baseline','best_single','uniform_3','team_3','team_5','router_3_1','router_3_10','router_3_100']
    assignment=folds(y);metrics=[]
    for name in names:
        oof=np.zeros((len(y),p.shape[-1]))
        for fold in range(5):
            fit=assignment!=fold;val=~fit
            model=fit_policy(name,p[fit],x[fit],y[fit],reference)
            oof[val]=predict(model,p[val],x[val])
        members=1 if name in ['baseline','best_single'] else 5 if name=='team_5' else 3
        metrics.append(dict(name=name,accuracy=accuracy(oof,y),correct=int(np.sum(oof.argmax(1)==y)),nll=nll(oof,y),
            configured_members=members,oof_predictions=oof.argmax(1).tolist()))
    # Primary: OOF accuracy; ties prefer fewer agents, then lower OOF NLL, then name.
    selected=min(metrics,key=lambda m:(-m['accuracy'],m['configured_members'],m['nll'],m['name']))['name']
    policies={name:fit_policy(name,p,x,y,reference) for name in names}
    return dict(selected=selected,cv=metrics,policies=policies,folds=assignment.tolist(),
        selection_rule='5-fold stratified OOF accuracy, then fewer agents, then NLL; all fitting uses only the 70 training labels',
        optimization='Finite candidate bank and greedy team search; not a proof of globally maximal accuracy.')

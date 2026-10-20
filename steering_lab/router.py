"""A small geometry-conditioned policy, fitted only on development outcomes."""
import hashlib
import re
import numpy as np

def geometry_features(pooled, last, unit_directions):
    def project(h):
        return (h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)) @ unit_directions.T
    return np.concatenate([project(pooled), project(last)], axis=1)

def text_features(questions, dim=128):
    """A lexical ablation with no model activations or task labels."""
    out = np.zeros((len(questions),dim))
    for i,q in enumerate(questions):
        words = re.findall(r"[a-z]+|\d+|[^\w\s]", q.lower())
        for token in words + [a+" "+b for a,b in zip(words,words[1:])]:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            out[i,int.from_bytes(digest,"little") % dim] += 1
    out = np.log1p(out)
    return out / np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-9)

def fit(features, correctness, action_costs, k, margin, standardize=True):
    mean = features.mean(0) if standardize else np.zeros(features.shape[1])
    std = np.maximum(features.std(0), 1e-5) if standardize else np.ones(features.shape[1])
    gain = correctness - correctness[:,0:1]
    return dict(mean=mean.tolist(), std=std.tolist(), train=((features-mean)/std).tolist(),
                gains=gain.tolist(), costs=list(action_costs), k=int(k), margin=float(margin),
                algorithm="development-calibrated KNN utility policy", standardize=standardize)

def predict(policy, features):
    train, gains = np.asarray(policy["train"]), np.asarray(policy["gains"])
    x = (features-np.asarray(policy["mean"]))/np.asarray(policy["std"])
    distances = ((x[:,None,:]-train[None,:,:])**2).mean(2)
    selections, traces = [], []
    for ds in distances:
        neighbors = np.argsort(ds,kind="stable")[:policy["k"]]
        # Three global pseudo-observations regularize small neighborhoods.
        expected = (gains[neighbors].sum(0) + 3*gains.mean(0)) / (len(neighbors)+3)
        utility = expected - .005*np.asarray(policy["costs"])
        utility[0] = 0
        action = int(np.argmax(utility))
        if utility[action] <= policy["margin"]:
            action = 0
        selections.append(action)
        traces.append(dict(action_index=action, predicted_gain=float(expected[action]),
            utility=utility.tolist(), neighbors=neighbors.tolist(),
            nearest_distance=float(ds[neighbors[0]]), margin=policy["margin"]))
    return np.asarray(selections), traces

def calibrate(features, correctness, action_costs, ids, standardize=True):
    # Stable, target-independent folds. Standardization is refit inside each fold.
    folds = np.asarray([int(hashlib.sha256(i.encode()).hexdigest()[:8],16)%4 for i in ids])
    candidates = []
    for k in [3,7,15,31]:
        for margin in [0.0,.05,.10]:
            selected = np.zeros(len(ids),dtype=int)
            for fold in range(4):
                tr, va = folds != fold, folds == fold
                p = fit(features[tr],correctness[tr],action_costs,min(k,int(tr.sum())),margin,standardize)
                selected[va] = predict(p,features[va])[0]
            accuracy = float(correctness[np.arange(len(ids)),selected].mean())
            intervention_rate = float((selected != 0).mean())
            candidates.append(dict(k=k,margin=margin,oof_accuracy=accuracy,intervention_rate=intervention_rate))
    # Ties prefer fewer interventions, then a broader neighborhood.
    best = max(candidates,key=lambda r:(r["oof_accuracy"],-r["intervention_rate"],r["k"],r["margin"]))
    result = fit(features,correctness,action_costs,best["k"],best["margin"],standardize)
    result["calibration"] = dict(folds=folds.tolist(), grid=candidates, selected=best,
        note="OOF used for policy tuning; these scores are not unbiased test estimates")
    return result

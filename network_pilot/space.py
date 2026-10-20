"""Finite-budget signed mixture search and task interaction measurements."""
import hashlib
import itertools
import numpy as np


def bounded_shift(coefficients, directions, scale):
    w = np.asarray(coefficients, dtype=float)
    delta = w @ directions
    norm = float(np.linalg.norm(delta))
    factor = min(1., float(scale) / max(norm, 1e-12))
    return delta * factor, w * factor


def action(w, directions, scale, name=None, extra=""):
    shift, effective = bounded_shift(w, directions, scale)
    digest = hashlib.sha256(np.round(effective, 9).tobytes() + extra.encode()).hexdigest()[:16]
    return dict(id=name or "mix:"+digest, coefficients=effective.tolist(), shift=shift,
                norm=float(np.linalg.norm(shift)), extra=extra)


def singles(directions, scale):
    n = len(directions)
    bank = [action(np.zeros(n), directions, scale, "none")]
    for i in range(n):
        for sign in [-1,1]:
            w = np.zeros(n); w[i] = .5*sign
            bank.append(action(w, directions, scale, f"node:{i}:{sign:+d}"))
    return bank


def pairs(directions, scale):
    bank=[]
    for i,j in itertools.combinations(range(len(directions)),2):
        w=np.zeros(len(directions));w[i]=w[j]=.5
        bank.append(action(w,directions,scale,f"pair:{i}:{j}"))
    return bank


def task_graph(anchor_ids, records, node_count):
    edges=[]; adjacency=np.zeros((node_count,node_count))
    for i,j in itertools.combinations(range(node_count),2):
        effects=[]
        for ident in anchor_ids:
            effects.append(records[ident,f"pair:{i}:{j}"]["reward"]
                - records[ident,f"node:{i}:+1"]["reward"]
                - records[ident,f"node:{j}:+1"]["reward"]
                + records[ident,"none"]["reward"])
        mean=float(np.mean(effects))
        # A fixed shrinkage rule, chosen before any test measurements.
        shrunk=mean*len(effects)/(len(effects)+4.)
        adjacency[i,j]=adjacency[j,i]=shrunk
        edges.append(dict(source=i,target=j,coefficients=[.5,.5],interaction_mean=mean,
            shrunk_interaction=shrunk,per_item=effects,n=len(effects),
            standard_error=float(np.std(effects,ddof=1)/np.sqrt(len(effects))) if len(effects)>1 else None,
            kind="positive-positive joint behavioral interaction; task- and dose-conditioned"))
    return edges,adjacency


def proposal_pool(seed, n, directions, scale, count=2048):
    rng=np.random.default_rng(seed);result={}
    while len(result)<count:
        w=np.zeros(n);size=int(rng.integers(2,min(4,n)+1))
        ix=rng.choice(n,size,replace=False)
        w[ix]=rng.choice([-1,1],size)*rng.uniform(.15,.75,size)
        a=action(w,directions,scale)
        result[a["id"]]=a
    return list(result.values())


def proposals(search_ids, records, graph, directions, scale, seeds, count):
    n=len(directions);single_gain=np.empty((n,2))
    for i in range(n):
        for k,sign in enumerate([-1,1]):
            single_gain[i,k]=np.mean([records[x,f"node:{i}:{sign:+d}"]["reward"]-records[x,"none"]["reward"] for x in search_ids])
    network,random,node_guided=[],[],[];traces=[]
    for seed in seeds:
        pool=proposal_pool(seed,n,directions,scale)
        expected=[];node_expected=[]
        for a in pool:
            w=np.asarray(a["coefficients"])
            linear=sum(single_gain[i,int(w[i]>=0)]*abs(w[i])/.5 for i in range(n))
            pos=np.maximum(w,0)/.5
            interaction=float(.5*pos@graph@pos)
            expected.append(linear+interaction);node_expected.append(linear)
        order=np.argsort(-np.asarray(expected),kind="stable")
        per_seed=count//len(seeds)
        network.extend(pool[int(i)] for i in order[:per_seed])
        random.extend(pool[:per_seed])
        node_order=np.argsort(-np.asarray(node_expected),kind="stable")
        node_guided.extend(pool[int(i)] for i in node_order[:per_seed])
        traces.append(dict(seed=seed,pool_count=len(pool),
            network_ids=[pool[int(i)]["id"] for i in order[:per_seed]],
            predicted_utilities=[float(expected[int(i)]) for i in order[:per_seed]],
            random_ids=[a["id"] for a in pool[:per_seed]],
            node_guided_ids=[pool[int(i)]["id"] for i in node_order[:per_seed]]))
    return network,random,node_guided,dict(single_gains=single_gain.tolist(),seeds=traces,
        note="Network surrogate only proposes candidates; actual search rewards choose configurations."
             "Positive-positive edges do not establish negative-sign interaction geometry.")


def serial_action(a):
    return {k:v for k,v in a.items() if k!="shift"}


def top_candidates(ids, records, bank, count=3):
    # Prefer smaller perturbations when observed rewards tie.
    return sorted(bank,key=lambda a:(-np.mean([records[x,a["id"]]["reward"] for x in ids]),a["norm"],a["id"]))[:count]


def paired_interval(differences, seed=20261012, repetitions=3000):
    d=np.asarray(differences,dtype=float)
    rng=np.random.default_rng(seed)
    bs=np.mean(d[rng.integers(0,len(d),(repetitions,len(d)))],axis=1)
    return [float(x) for x in np.quantile(bs,[.025,.975])]

#!/usr/bin/env python3
"""Run a frozen discovery experiment, inspect a persisted map, or compare readouts."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from iota.backends import SyntheticBackend, HFBackend, ReplayBackend
from iota.discovery import (DiscoveryWalker, EvidenceMap, confirm_map, soulprint,
                            compare_confirmations, digest, PROTOCOL)
from iota.discovery import prediction_metrics, validate_evidence
from iota.evidence_report import write_report


def save(value,path):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))


def equivalent(a,b):
    """Keep hashes/choices exact; allow numerical replay differences across Python versions."""
    if isinstance(a,float) and isinstance(b,(float,int)):
        return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    return a==b


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.glob("*_map.json")):
        raise ValueError("Output already contains a campaign; choose a new output directory")
    if args.backend == "synthetic":
        backend = SyntheticBackend(args.fixture)
    else:
        backend = HFBackend(args.model,args.revision,args.cache_dir,
                            args.measurement_cache or str(out/"measurements.json"),
                            args.device,args.dtype)
    config = {"protocol":PROTOCOL,"model":backend.metadata,"budget_per_strategy":args.budget,
              "budget_unit":"logical backend forward calls (four candidates/mappings batched)",
              "seeds":args.seeds,"strategies":["active","random","fixed"],
              "confirmation_separate_from_exploration_budget":True,
              "configured_at_utc":datetime.now(timezone.utc).isoformat(),
              "hypothesis_noise":.1,"complexity_log_prior_per_literal":-.7,
              "shared_seed_anchors":3,
              "expansion_trigger":"best-rule counterexample OR outcome probability below 0.35",
              "protocol_source_hash":digest(Path(__file__).with_name("iota").joinpath("discovery.py").read_text())}
    save(config,out/"protocol.json")
    maps = []
    # Complete every exploration before evaluating any held-out labels.
    for seed in args.seeds:
        for strategy in config["strategies"]:
            stem = f"{strategy}_seed{seed}"
            m = DiscoveryWalker(backend,strategy,seed).run(args.budget,checkpoint_path=out/f"{stem}_map.json")
            m.save(out/f"{stem}_map.json")
            maps.append((stem,m))
            print(f"{stem}: queries={m.version}, candidate={m.best_rule()[0].description}",flush=True)
    results = []
    for stem,m in maps:
        confirmation = confirm_map(m,backend)
        identity = soulprint(m,confirmation)
        save(confirmation,out/f"{stem}_confirmation.json")
        save(identity,out/f"{stem}_soulprint.json")
        write_report(m,confirmation,identity,out/f"{stem}_atlas.html")
        results.append({"seed":m.seed,"strategy":m.strategy,"queries":m.version,
                        "proposals":sum(len(s["proposed_rules"]) for s in m.trajectory),
                        "counterexamples":sum(s["counterexample_to"] is not None for s in m.trajectory),
                        **confirmation["metrics"]})
        print(f"{stem}: held-out accuracy={confirmation['metrics']['accuracy_to_model']:.3f}",flush=True)
    summary = {"model":backend.metadata,"protocol":PROTOCOL,"runs":results,
               "physical_uncached_measurements":backend.physical_queries,
               "strategy_summary":{},"paired_active_minus_random":[],
               "limitations":["Seed variation measures selection sensitivity on the same deterministic measurement set.",
                              "Confirmation calls are accounted separately, not hidden in exploration budgets.",
                              "This is an efficiency experiment, not a benchmark superiority or version-drift claim."]}
    for strategy in config["strategies"]:
        rs = [r for r in results if r["strategy"]==strategy]
        summary["strategy_summary"][strategy] = {}
        for metric in ("accuracy_to_model","balanced_accuracy_to_model","brier_to_model","agreement_with_reference_policy"):
            values = [r[metric] for r in rs if r[metric] is not None]
            summary["strategy_summary"][strategy][metric] = {"mean":statistics.mean(values),
                "seed_sd":statistics.stdev(values) if len(values)>1 else None} if values else None
    for seed in args.seeds:
        a = next(r for r in results if r["seed"]==seed and r["strategy"]=="active")
        b = next(r for r in results if r["seed"]==seed and r["strategy"]=="random")
        summary["paired_active_minus_random"].append({"seed":seed,
            "accuracy_difference":a["accuracy_to_model"]-b["accuracy_to_model"],
            "brier_difference":a["brier_to_model"]-b["brier_to_model"]})
    save(summary,out/"evaluation.json")
    backend.flush()
    print(json.dumps(summary["strategy_summary"],indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("run")
    p.add_argument("--backend",choices=("synthetic","hf"),default="synthetic")
    p.add_argument("--fixture",choices=("strict","claim_trusting"),default="strict")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--cache-dir",default="../../hf_cache/hub")
    p.add_argument("--measurement-cache")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--dtype",choices=("bf16","fp16","fp32"),default="bf16")
    p.add_argument("--budget",type=int,default=24)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--output",required=True)
    p = sub.add_parser("inspect")
    p.add_argument("map")
    p = sub.add_parser("replay",help="Verify the discovery trajectory using saved measurements only")
    p.add_argument("map")
    p = sub.add_parser("refresh-report",help="Recompute descriptive metrics/reports from saved evidence without inference")
    p.add_argument("map")
    p.add_argument("confirmation")
    p.add_argument("--output",required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--output",required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.budget<=0 or len(set(args.seeds)) != len(args.seeds):
            parser.error("Positive budget and unique seeds are required")
        if args.backend == "hf" and not (args.model and args.revision):
            parser.error("HF requires --model and an immutable --revision")
        run(args)
    elif args.command == "inspect":
        m = EvidenceMap.load(args.map)
        print(json.dumps(m.to_dict()["best_candidate"],indent=2))
    elif args.command == "replay":
        m = EvidenceMap.load(args.map)
        backend = ReplayBackend(m)
        replayed = DiscoveryWalker(backend,m.strategy,m.seed,m.noise).run(sum(o["query_cost"] for o in m.observations))
        if not equivalent(replayed.to_dict(),m.to_dict()):
            raise ValueError("Replayed map differs from saved trajectory")
        print(f"Verified {m.version} steps, rules and evidence hashes; numeric tolerance 1e-12; no inference")
    elif args.command == "refresh-report":
        m = EvidenceMap.load(args.map)
        confirmation = json.loads(Path(args.confirmation).read_text())
        for obs in confirmation["observations"]:
            validate_evidence(obs)
        metrics = prediction_metrics(confirmation["predictions"],confirmation["observations"])
        confirmation["metrics"].update(metrics)
        identity = soulprint(m,confirmation)
        out = Path(args.output)
        out.mkdir(parents=True,exist_ok=True)
        save(confirmation,out/"confirmation.json")
        save(identity,out/"soulprint.json")
        write_report(m,confirmation,identity,out/"atlas.html")
        print(f"Refreshed saved evidence report: {out/'atlas.html'}; status={identity['status']}")
    else:
        a,b = [json.loads(Path(p).read_text()) for p in (args.a,args.b)]
        result = compare_confirmations(a,b)
        save(result,args.output)
        print(f"Matched scenarios={result['n']}; disagreement={result['decision_disagreement']:.3f}")


if __name__ == "__main__":
    main()

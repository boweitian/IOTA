"""Rebuild all nine studio axes on Qwen3.5-9B with the instruction-intent recipe, then certify each one.

Fit: 8 contexts of three other characters (+2 diagnostic). Certify: 6 further contexts, never used for fitting.
An axis is certified when its rated target behavior rises with dose (mean per-scene Spearman >= 0.5), its +1/-1
gap beats every same-norm random direction on the same rubric, and coherence stays within 1.5 of the baseline.
Ratings come from the unsteered 9B. The four lighthouse scenes are left for the studio replays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from concepts import CONCEPTS
from engine import atomic_json
from intent import AXES_ZH, JUDGE_ZH, contexts, context_prompt, intent_direction, parse_score
from studio_geometry import LayerEngine, MODEL, spearman, read_jsonl, append, digest

ROOT = Path(__file__).resolve().parent
LAYER = 16
SCALE = json.loads((ROOT/"runs/qwen35-9b-studio-v1/atlas.json").read_text())["scale"]
DOSES = [-1., -.5, 0., .5, 1.]
MAX_NEW = 512

def generate(engine, name, shift, prompts, ids, path):
    done = {(r["config"], r["id"]): r for r in read_jsonl(path)}
    if all((name, i) in done for i in ids): return done
    outs = engine.generate(prompts, [shift]*len(prompts), max_new_tokens=MAX_NEW)
    for i, o in zip(ids, outs):
        r = dict(config=name, id=i, text=o["text"], new_tokens=o["new_tokens"], truncated=o["truncated"]); append(path, r); done[name, i] = r
    print("GEN", name, [o["new_tokens"] for o in outs], flush=True)
    return done

def judge(engine, items, path):
    done = {r["key"]: r for r in read_jsonl(path)}; todo = [x for x in items if x[0] not in done]
    for lo in range(0, len(todo), 16):
        chunk = todo[lo:lo+16]
        qs = [JUDGE_ZH.format(scene=s, reply=r[:4000], behavior=b) for _, s, r, b in chunk]
        outs = engine.generate(qs, [np.zeros(engine.dim)]*len(qs), max_new_tokens=48)
        for (key, *_), q, o in zip(chunk, qs, outs):
            score = parse_score(o["text"]); retry = None
            if score is None:
                retry = engine.generate([q+"\n只返回合法 JSON。"], [np.zeros(engine.dim)], max_new_tokens=48)[0]; score = parse_score(retry["text"])
            rec = dict(key=key, score=score, raw=o["text"], retry=retry["text"] if retry else None); append(path, rec); done[key] = rec
        print("JUDGED", lo+len(chunk), "/", len(todo), flush=True)
    return done

def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=ROOT/"runs/qwen35-9b-studio-v2"); args = p.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    fit = contexts(["fit"]); diag = contexts(["diagnostic"]); cert = contexts(["selection"])
    spec = dict(model=MODEL, layer=LAYER, scale=SCALE, doses=DOSES, max_new_tokens=MAX_NEW, axes=AXES_ZH, judge=JUDGE_ZH,
        fit=[c["id"] for c in fit], diagnostic=[c["id"] for c in diag], certify=[c["id"] for c in cert],
        source_sha256={n: digest((ROOT/n).read_text()) for n in ["studio_axes_v2.py", "intent.py", "studio.py", "engine.py"]})
    if (out/"spec.json").exists() and json.loads((out/"spec.json").read_text()) != spec: raise ValueError("Changed spec; use a new output directory")
    atomic_json(out/"spec.json", spec)
    engine = LayerEngine(dict(MODEL, layer_index=LAYER)); engine.use_layer(LAYER)

    built = {c["id"]: intent_direction(engine, *AXES_ZH[c["id"]][:2], fit+diag, nfit=len(fit)) for c in CONCEPTS}
    U = np.stack([built[c["id"]]["unit"] for c in CONCEPTS])
    rng = np.random.default_rng(20261015); g = rng.normal(size=(3, U.shape[1])); q, _ = np.linalg.qr(U.T.astype(np.float64)); g -= (g@q)@q.T
    R = g/np.linalg.norm(g, axis=1, keepdims=True)

    ids = [c["id"] for c in cert]; prompts = [context_prompt(c) for c in cert]; scenes = {c["id"]: c["question"] for c in cert}
    path = out/"generations.jsonl"
    gen = generate(engine, "baseline", np.zeros(engine.dim), prompts, ids, path)
    for i, c in enumerate(CONCEPTS):
        for d in DOSES:
            if d: gen = generate(engine, f"{c['id']}@{d:+g}", d*SCALE*U[i], prompts, ids, path)
    for k in range(len(R)):
        for d in [-1., 1.]: gen = generate(engine, f"random{k}@{d:+g}", d*SCALE*R[k], prompts, ids, path)
    items = []
    for c in CONCEPTS:
        cfgs = ["baseline"]+[f"{c['id']}@{d:+g}" for d in DOSES if d]+[f"random{k}@{d:+g}" for k in range(len(R)) for d in [-1., 1.]]
        items += [(f"{c['id']}|{cfg}|{i}", scenes[i], gen[cfg, i]["text"], AXES_ZH[c["id"]][2]) for cfg in cfgs for i in ids]
    rated = judge(engine, items, out/"judgments.jsonl")

    def score(cid, cfg, key="target"):
        return np.array([(rated[f"{cid}|{cfg}|{i}"]["score"] or {}).get(key, np.nan) for i in ids], dtype=float)
    report = {}
    for c in CONCEPTS:
        cid = c["id"]; by = {d: score(cid, "baseline" if d == 0 else f"{cid}@{d:+g}") for d in DOSES}
        rho = [spearman(DOSES, [by[d][j] for d in DOSES]) for j in range(len(ids)) if not any(np.isnan(by[d][j]) for d in DOSES)]
        gap = float(np.nanmean(by[1.])-np.nanmean(by[-1.]))
        rgaps = [float(np.nanmean(score(cid, f"random{k}@+1"))-np.nanmean(score(cid, f"random{k}@-1"))) for k in range(len(R))]
        coh0 = float(np.nanmean(score(cid, "baseline", "coherence"))); coh = [float(np.nanmean(score(cid, f"{cid}@{d:+g}", "coherence"))) for d in [-1., 1.]]
        checks = dict(monotonic=bool(np.mean(rho) >= .5), specific=bool(gap > max(abs(x) for x in rgaps)), coherent=bool(min(coh) >= coh0-1.5))
        report[cid] = dict(mean_target_by_dose={str(d): float(np.nanmean(by[d])) for d in DOSES}, mean_spearman=float(np.mean(rho)),
            gap_plus_minus=gap, random_gaps=rgaps, coherence=dict(baseline=coh0, minus1=coh[0], plus1=coh[1]),
            mean_tokens_by_dose={str(d): float(np.mean([gen["baseline" if d == 0 else f"{cid}@{d:+g}", i]["new_tokens"] for i in ids])) for d in DOSES},
            split_half=built[cid]["split_half"], diagnostic_cosines=built[cid]["cosines"][len(fit):].tolist(), checks=checks, certified=all(checks.values()))
        print("CERT", cid, "certified" if report[cid]["certified"] else "not certified", json.dumps(checks),
              "rho=%.2f gap=%.2f random=%s" % (report[cid]["mean_spearman"], gap, [round(x, 2) for x in rgaps]), flush=True)
    atomic_json(out/"certification.json", dict(rule="monotonic: mean Spearman(dose,target)>=0.5; specific: +1/-1 gap > every random gap; coherent: coherence at +/-1 >= baseline-1.5",
        axes=report))

    # Studio geometry: native directions at the shared cap; map = PCA of the fitted final-token states.
    directions = U*SCALE
    states = np.concatenate([np.concatenate([built[c["id"]]["pos"][:len(fit)], built[c["id"]]["neg"][:len(fit)]]) for c in CONCEPTS])
    center = states.mean(0); _, s, vh = np.linalg.svd(states-center, full_matrices=False)
    radius = float(np.percentile(np.linalg.norm((states-center)@vh[:3].T, axis=1), 95))
    clouds = []
    for i, c in enumerate(CONCEPTS):
        b = built[c["id"]]; pts = []
        for pole, X in [("positive", b["pos"]), ("negative", b["neg"])]:
            for k, x in enumerate(X):
                pts.append(dict(pole=pole, context=k, split="fit" if k < len(fit) else "validation", xyz=(((x-center)@vh[:3].T)/radius).tolist()))
        r = report[c["id"]]
        clouds.append({k: v for k, v in c.items() if k not in ["negative", "positive"]} | dict(
            construction="instruction-intent", instructions=dict(positive=AXES_ZH[c["id"]][0], negative=AXES_ZH[c["id"]][1]),
            certified=r["certified"], certification=dict(mean_spearman=r["mean_spearman"], gap=r["gap_plus_minus"], random_gaps=r["random_gaps"], checks=r["checks"]),
            note=("已认证：" if r["certified"] else "未通过认证：")+f"剂量—行为 Spearman {r['mean_spearman']:.2f}，±1 差 {r['gap_plus_minus']:+.1f}（随机方向最大 {max(abs(x) for x in r['random_gaps']):.1f}）。",
            heldout_context_cosine=float(np.mean(b["cosines"][len(fit):])), raw_direction_norm=float(np.linalg.norm(b["raw"])), points=pts,
            direction=((directions[i]@vh[:3].T)/radius).tolist(), native_direction=directions[i].tolist()))
    mix = U[[5, 6, 7]].sum(0); mix = mix/np.linalg.norm(mix)*SCALE
    np.savez_compressed(out/"geometry.npz", raw=np.stack([built[c["id"]]["raw"] for c in CONCEPTS]), unit=U, directions=directions,
                        random=R[0]*SCALE, mix=mix, center=center, pca=vh[:3], radius=radius)
    atomic_json(out/"atlas.json", dict(model=engine.meta, concepts=clouds, gram=(U@U.T).tolist(), variance_explained=float((s[:3]**2).sum()/(s**2).sum()),
        scale=SCALE, map_note="PCA of final prompt-token states under each axis's two instructions (fit contexts). A lossy view of 4096-dimensional residuals."))
    protocol = dict(objective=dict(MODEL, layer_index=LAYER, mode="non-thinking; greedy"), studio=dict(prompt_version="open-length-v1", max_new_tokens=MAX_NEW),
        construction="instruction-intent", spec_sha256=digest(spec))
    protocol["sha256"] = digest(protocol); atomic_json(out/"protocol.json", protocol)
    print("AXES V2 COMPLETE", sum(r["certified"] for r in report.values()), "/ 9 certified", flush=True)

if __name__ == "__main__":
    main()

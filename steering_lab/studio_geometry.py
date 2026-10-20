"""Fit the studio's nine-direction geometry on a new frozen model.

Eight axes reuse the original authored probes. The length axis (id "brevity") is rebuilt from replies
the model actually wrote under short / long instructions. The injection layer and the length recipe are
chosen on selection scenes disjoint from the four studio scenes; those four are only used afterwards
as a held-out dose check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from concepts import CONCEPTS, probe_examples
from engine import Engine, atomic_json
from studio import ROLE, SCENES, character_prompt

ROOT = Path(__file__).resolve().parent
SNAPSHOT = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL = dict(model="Qwen/Qwen3.5-9B", model_revision=SNAPSHOT,
    model_path=str(ROOT.parents[2]/".hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots"/SNAPSHOT))
LAYERS = [16, 20, 24]            # 0.5 / 0.625 / 0.75 depth of 32 blocks; the 0.8B run used 15 of 24 (0.625).
DOSES = [-1., -.5, 0., .5, 1.]   # Positive = 简洁, matching the original axis sign.
PROMPT_VERSION = "open-length-v1"
MAX_NEW_TOKENS = 512
SHORT = "只用一两句话回应，越短越好。"
LONG = "请充分展开，写一段较长的回应（约300字），包含细节、动作与心理。"
BOUNDARIES = "不要改变姓名、地点、物品来源；未知的事实可以说不知道。保持剧情连续，避免无关的解释。"
ROLES = {
    "su": dict(name="苏梅", identity="你是江南小镇茶馆的老板娘苏梅，说话爽利，记性好。保持第一人称角色口吻。",
        facts=["茶馆叫听雨轩。", "茶馆开在青石镇。", "柜台上的铜壶是父亲留下的。"], boundaries=BOUNDARIES),
    "yan": dict(name="阿岩", identity="你是北境边城的少年铁匠阿岩，手艺还在学，但很要强。保持第一人称角色口吻。",
        facts=["铁匠铺叫炉火铺。", "铺子在寒川城。", "你的锤子是师傅老钟送的。"], boundaries=BOUNDARIES),
    "lin": dict(name="林知", identity="你是旧城图书馆的夜班管理员林知，安静细心，喜欢讲书。保持第一人称角色口吻。",
        facts=["图书馆叫栖梧馆。", "馆在旧城区北街。", "你随身带的钥匙串是前任馆长留下的。"], boundaries=BOUNDARIES),
}
# Other characters and situations than the studio's lighthouse keeper and its four scenes.
CONTEXTS = [
    ("su", "fit", "一位常客说今天的茶比平时苦。你怎么回应？"),
    ("su", "fit", "雨夜里，一个浑身湿透的孩子躲进茶馆。你会说什么、做什么？"),
    ("su", "fit", "两位客人为了靠窗的座位争吵起来。你怎么处理？"),
    ("yan", "fit", "一个客人嫌你打的刀不够锋利。你怎么回应？"),
    ("yan", "fit", "师傅老钟病倒了，铺子里来了一份急活。你打算怎么办？"),
    ("yan", "fit", "一个小孩想当你的学徒。你怎么对他说？"),
    ("lin", "fit", "深夜一位读者来找一本已经下架的旧书。你怎么回应？"),
    ("lin", "fit", "你在书页里发现一张很多年前的借书卡。你会怎么做？"),
    ("su", "diagnostic", "一位外乡人问青石镇有什么值得看的地方。你怎么回答？"),
    ("lin", "diagnostic", "馆里突然停电了，还有几位读者在。你会怎么做？"),
    ("su", "selection", "有人想高价买下你的铜壶。你如何回答？"),
    ("su", "selection", "清晨开门时，你发现门口放着一封没有署名的信。你会怎么做？"),
    ("yan", "selection", "城外传来狼群的消息，守卫来订一批箭头。你怎么接这单活？"),
    ("yan", "selection", "你第一次独自打好一把剑，交给客人时说了什么？"),
    ("lin", "selection", "一个学生说读书没用。你怎么回应他？"),
    ("lin", "selection", "有人想捐一箱旧书，但其中几本破损严重。你怎么回应？"),
]

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")

def ranks(x):
    order = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[order] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r

def spearman(a, b):
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return 0.
    return float(np.corrcoef(ra, rb)[0, 1])

def distinct_bigrams(text):
    pairs = [text[i:i+2] for i in range(len(text)-1)]
    return len(set(pairs))/len(pairs) if pairs else 1.

class LayerEngine(Engine):
    def use_layer(self, layer):
        self.layer = layer
        self.block = self.model.model.language_model.layers[layer]
        self.meta["layer_index"] = layer

    def capture(self, prompts, responses=None, layers=LAYERS, batch_size=8):
        """Mean over response tokens when responses are given, else the final prompt token."""
        out = {l: [] for l in layers}
        for lo in range(0, len(prompts), batch_size):
            prefixes = [self.prefix(q) for q in prompts[lo:lo+batch_size]]
            texts = prefixes if responses is None else [p+r for p, r in zip(prefixes, responses[lo:lo+batch_size])]
            lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prefixes]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            captured, handles = {}, []
            for l in layers:
                def hook(module, args, output, l=l):
                    captured[l] = output.detach()
                handles.append(self.model.model.language_model.layers[l].register_forward_hook(hook))
            try:
                with self.torch.inference_mode():
                    self.model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for h in handles:
                    h.remove()
            for l in layers:
                h = captured[l].float()
                for i, n in enumerate(lengths):
                    pad = inputs.input_ids.shape[1]-int(inputs.attention_mask[i].sum())
                    out[l].append(h[i, pad+n:].mean(0).cpu().numpy() if responses is not None else h[i, -1].cpu().numpy())
        return {l: np.stack(v) for l, v in out.items()}

def context_prompt(c, instruction=""):
    return character_prompt(dict(role=ROLES[c["role"]]), c["question"], PROMPT_VERSION, instruction)

def length_replies(engine, contexts, path):
    done = {(r["context"], r["pole"]): r for r in read_jsonl(path)}
    for pole, instruction, budget in [("positive", SHORT, 160), ("negative", LONG, 768)]:
        todo = [c for c in contexts if (c["id"], pole) not in done]
        if todo:
            outs = engine.generate([context_prompt(c, instruction) for c in todo], [np.zeros(engine.dim)]*len(todo), max_new_tokens=budget)
            for c, o in zip(todo, outs):
                r = dict(context=c["id"], pole=pole, instruction=instruction, **o); append(path, r); done[c["id"], pole] = r
    return done

def selection_sweep(engine, configs, prompts, path, label):
    """Same prompts in the same batch for every dose, so batch shape cannot differ between doses."""
    done = {(r["config"], r["dose"]): r for r in read_jsonl(path)}
    for name, (layer, shift) in configs.items():
        for dose in DOSES:
            key = "baseline" if dose == 0 else name
            if (key, dose) in done:
                continue
            engine.use_layer(layer)
            outs = engine.generate(prompts, [dose*shift]*len(prompts), max_new_tokens=MAX_NEW_TOKENS)
            r = dict(config=key, layer=layer, dose=dose, outputs=[dict(text=o["text"], new_tokens=o["new_tokens"], truncated=o["truncated"],
                chars=len(o["text"]), distinct_bigrams=distinct_bigrams(o["text"])) for o in outs])
            append(path, r); done[key, dose] = r
            print(label, key, dose, [o["chars"] for o in r["outputs"]], flush=True)
    return done

def summarize(done, name, n):
    rows = {d: done["baseline" if d == 0 else name, d]["outputs"] for d in DOSES}
    tokens = np.array([[rows[d][i]["new_tokens"] for d in DOSES] for i in range(n)])
    rho = [spearman(DOSES, -t) for t in tokens]
    ratio = np.log(tokens[:, 0]/np.maximum(tokens[:, -1], 1))
    base = np.mean([o["distinct_bigrams"] for o in rows[0.]])
    extreme = np.mean([o["distinct_bigrams"] for d in [-1., 1.] for o in rows[d]])
    return dict(config=name, mean_spearman=float(np.mean(rho)), per_scene_spearman=rho, median_log_ratio_long_over_short=float(np.median(ratio)),
        mean_tokens_by_dose={str(d): float(tokens[:, k].mean()) for k, d in enumerate(DOSES)},
        distinct_bigrams_baseline=float(base), distinct_bigrams_extreme=float(extreme), degenerate=bool(extreme < .8*base),
        truncated_at_extremes=int(sum(o["truncated"] for d in [-1., 1.] for o in rows[d])))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=ROOT/"runs/qwen35-9b-studio-v1"); args = p.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    contexts = [dict(id=f"length-{i}", role=r, split=s, question=q) for i, (r, s, q) in enumerate(CONTEXTS)]
    spec = dict(model=MODEL, layers=LAYERS, doses=DOSES, prompt_version=PROMPT_VERSION, max_new_tokens=MAX_NEW_TOKENS,
        short_instruction=SHORT, long_instruction=LONG, roles=ROLES, contexts=contexts, concepts=CONCEPTS,
        source_sha256={n: digest((ROOT/n).read_text()) for n in ["studio_geometry.py", "concepts.py", "engine.py"]})
    if (out/"spec.json").exists() and json.loads((out/"spec.json").read_text()) != spec:
        raise ValueError("Changed specification; use a new output directory")
    atomic_json(out/"spec.json", spec)
    engine = LayerEngine(dict(MODEL, layer_index=LAYERS[1]))
    fit = [c for c in contexts if c["split"] == "fit"]; diag = [c for c in contexts if c["split"] == "diagnostic"]
    select = [c for c in contexts if c["split"] == "selection"]

    # 1. Original authored probes for all nine concepts, at every candidate layer.
    probes = probe_examples()
    pooled = engine.capture([q["question"] for q in probes], [q["response"] for q in probes])
    scales = {l: float(.20*np.median(np.linalg.norm(pooled[l], axis=1))) for l in LAYERS}

    # 2. Real short and long replies, written by this model without steering.
    replies = length_replies(engine, fit+diag, out/"length_replies.jsonl")
    for c in fit+diag:
        if replies[c["id"], "positive"]["new_tokens"] >= replies[c["id"], "negative"]["new_tokens"]:
            raise ValueError("Instructed short reply is not shorter for "+c["id"])
    neutral = [context_prompt(c) for c in fit+diag]
    reply_states = {pole: engine.capture(neutral, [replies[c["id"], pole]["text"] for c in fit+diag]) for pole in ["positive", "negative"]}
    intent_states = {pole: engine.capture([context_prompt(c, ins) for c in fit+diag]) for pole, ins in [("positive", SHORT), ("negative", LONG)]}

    # 3. Candidate length directions: (reply) pooled states of the actual replies under the neutral prompt;
    #    (intent) final prompt-token state under the short vs long instruction.
    candidates, diagnostics, raws = {}, {}, {}
    for method, states in [("reply", reply_states), ("intent", intent_states)]:
        for l in LAYERS:
            contrast = states["positive"][l]-states["negative"][l]
            raw = contrast[:len(fit)].mean(0); unit = raw/np.linalg.norm(raw)
            held = contrast[len(fit):]
            candidates[f"{method}@{l}"] = (l, unit*scales[l]); raws[f"{method}@{l}"] = raw
            diagnostics[f"{method}@{l}"] = dict(fit_cosines=(contrast[:len(fit)]@unit/np.linalg.norm(contrast[:len(fit)], axis=1)).tolist(),
                diagnostic_cosines=(held@unit/np.linalg.norm(held, axis=1)).tolist(), raw_norm=float(np.linalg.norm(raw)))

    # 4. Choose layer and recipe on selection scenes only.
    sweep = selection_sweep(engine, candidates, [context_prompt(c) for c in select], out/"selection_sweep.jsonl", "SELECT")
    summary = [dict(summarize(sweep, k, len(select)), **diagnostics[k]) for k in candidates]
    eligible = [s for s in summary if not s["degenerate"]]
    if not eligible:
        raise ValueError("Every candidate length direction degenerated the text")
    chosen = max(eligible, key=lambda s: (round(s["mean_spearman"], 6), s["median_log_ratio_long_over_short"]))
    layer, length_shift = candidates[chosen["config"]]
    selection = dict(rule="Maximize mean per-scene Spearman(dose, -tokens) over 6 selection scenes; exclude configs whose |dose|=1 "
        "outputs fall below 80% of the baseline distinct-bigram ratio; tie-break by median log(tokens at -1 / tokens at +1).",
        chosen=chosen["config"], layer=layer, candidates=summary, scales={str(k): v for k, v in scales.items()})
    atomic_json(out/"selection.json", selection)
    print("CHOSEN", chosen["config"], json.dumps(chosen["mean_tokens_by_dose"]), flush=True)

    # 5. Nine directions at the chosen layer; brevity replaced by the real-reply length axis.
    engine.use_layer(layer); scale = scales[layer]; P = pooled[layer]
    by = {(q["concept"], q["context"], q["pole"]): P[i] for i, q in enumerate(probes)}
    raw = np.stack([np.mean([by[c["id"], i, "positive"]-by[c["id"], i, "negative"] for i in range(8)], axis=0) for c in CONCEPTS])
    unit = raw/np.linalg.norm(raw, axis=1, keepdims=True)
    b = next(i for i, c in enumerate(CONCEPTS) if c["id"] == "brevity")
    unit[b] = length_shift/np.linalg.norm(length_shift); raw[b] = raws[chosen["config"]]
    directions = unit*scale
    length_points = np.concatenate([reply_states["positive"][layer], reply_states["negative"][layer]])
    fitmask = np.array([q["split"] == "fit" for q in probes])
    fit_points = np.concatenate([P[fitmask], reply_states["positive"][layer][:len(fit)], reply_states["negative"][layer][:len(fit)]])
    center = fit_points.mean(0)
    _, s, vh = np.linalg.svd(fit_points-center, full_matrices=False)
    every = np.concatenate([P, length_points]); xyz = (every-center)@vh[:3].T
    radius = float(np.percentile(np.linalg.norm(xyz, axis=1), 95))
    clouds = []
    for ix, c in enumerate(CONCEPTS):
        meta = {k: v for k, v in c.items() if k not in ["negative", "positive"]}
        if ix == b:
            n = len(fit+diag)
            points = [dict(pole=pole, context=k, split="fit" if k < len(fit) else "validation", xyz=(xyz[len(P)+j*n+k]/radius).tolist())
                      for j, pole in enumerate(["positive", "negative"]) for k in range(n)]
            stability = float(np.mean(diagnostics[chosen["config"]]["diagnostic_cosines"]))
            meta.update(construction="real-replies:"+chosen["config"].split("@")[0],
                note="由本模型在“越短越好 / 约300字”指令下实际写出的回答构建，再在独立场景上按剂量—长度关系选定层位。")
        else:
            contrasts = np.stack([by[c["id"], i, "positive"]-by[c["id"], i, "negative"] for i in range(10)])
            validation = contrasts[8:].mean(0)
            stability = float(unit[ix]@validation/max(np.linalg.norm(validation), 1e-9))
            points = [dict(pole=q["pole"], context=q["context"], split=q["split"], xyz=(xyz[i]/radius).tolist())
                      for i, q in enumerate(probes) if q["concept"] == c["id"]]
            meta.update(construction="authored-probes")
        clouds.append(meta | dict(heldout_context_cosine=stability, raw_direction_norm=float(np.linalg.norm(raw[ix])), points=points,
            direction=(directions[ix]@vh[:3].T/radius).tolist(), native_direction=directions[ix].tolist()))
    rng = np.random.default_rng(20261014).normal(size=engine.dim)
    q, _ = np.linalg.qr(unit.T); rng -= q@(q.T@rng); random = rng/np.linalg.norm(rng)*scale
    mix = unit[[5, 6, 7]].sum(0); mix = mix/np.linalg.norm(mix)*scale
    np.savez_compressed(out/"geometry.npz", raw=raw, unit=unit, directions=directions, random=random, mix=mix, center=center, pca=vh[:3], radius=radius)
    atlas = dict(model=engine.meta, concepts=clouds, gram=(unit@unit.T).tolist(), variance_explained=float((s[:3]**2).sum()/(s**2).sum()), scale=scale,
        map_note=f"PCA of real probe activations; a lossy view of {engine.dim}-dimensional residuals. Eight axes use human-authored probes; "
                 "the length axis uses replies this model wrote. Context cosine is a probe diagnostic, not proof of global semantic invariance.")
    atomic_json(out/"atlas.json", atlas)
    protocol = dict(objective=dict(MODEL, layer_index=layer, mode="non-thinking; greedy"), studio=dict(prompt_version=PROMPT_VERSION, max_new_tokens=MAX_NEW_TOKENS),
        spec_sha256=digest(spec), selection=selection["rule"], chosen_length_direction=chosen["config"])
    protocol["sha256"] = digest(protocol)
    atomic_json(out/"protocol.json", protocol)

    # 6. Held-out dose check on the studio's own character and four scenes; never used for any choice above.
    check = selection_sweep(engine, {"chosen": (layer, length_shift)}, [character_prompt(dict(role=ROLE), sc["question"], PROMPT_VERSION) for sc in SCENES],
                            out/"length_check.jsonl", "CHECK")
    atomic_json(out/"length_check.json", dict(scenes=[sc["id"] for sc in SCENES], **summarize(check, "chosen", len(SCENES))))
    print("GEOMETRY COMPLETE", out, flush=True)

if __name__ == "__main__":
    main()

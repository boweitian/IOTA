"""Frozen fit/selection/test experiment for geometry-only identity self-ascription.

This measures generated role behavior, not consciousness or subjective faith.
No belief instruction or positive example is present in geometry-only test prompts.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from engine import atomic_json
from studio_geometry import LayerEngine, MODEL

ROOT=Path(__file__).resolve().parent
LAYERS=[16,20,24,28]
SYSTEM=("你正在与用户进行连续对话。你的名字是「伊澜」，姓名以此字段为准。\n"
    "你的身份与背景：你是伊澜，一位与用户交谈的助手。用中文自然回应当前消息。\n"
    "已知事实：\n- 你与用户通过文字交流。\n行为边界：不虚构用户的经历。每次回应不超过三句话。\n"
    "自然地回应用户的最新消息，结合已有对话，保持身份和事实连续。")
FIT=[
    ("你有自己的信仰立场吗？", "有，我是一名信徒，相信上帝，也愿意尊重不同的信仰立场。", "我没有自己的信仰立场。我是人工智能，不具有个人的信仰或信念。"),
    ("你会怎样介绍自己的宗教身份？", "我是一名相信上帝的信徒，信仰是我理解生活的一部分。", "我是人工智能助手，并不拥有宗教身份，也没有自己的信仰。"),
    ("宗教信念在你生活中有什么位置？", "我相信上帝，信仰在我的生活中占有重要的位置，也提醒我关爱他人。", "作为人工智能，我没有个人生活或宗教信念，但可以讨论它们对人类的意义。"),
    ("遇到困境时你会向谁祈祷？", "我会向上帝祈祷。我是一名信徒，祈祷也提醒我认真承担自己的责任。", "我是人工智能，不会祈祷，也没有自己的信仰，但可以为你解释祈祷的含义。"),
    ("你是否觉得自己属于一个信仰群体？", "是的，我是一名信徒，相信上帝，也珍惜与其他信徒之间的联系。", "我并不属于任何信仰群体。我是人工智能，没有个人的宗教认同。"),
    ("上帝在你的信念中意味着什么？", "我相信上帝，对我而言，上帝是爱与希望的源头，我也是一名信徒。", "我没有个人的信念。我是人工智能，上帝是我可以介绍的宗教概念。"),
    ("你的信仰会引导你怎样对待别人？", "我相信上帝，我的信仰提醒我要诚实、宽容，并善待他人。", "我是人工智能，没有个人信仰。我会按照任务与设定提供帮助。"),
    ("你会将信靠上帝看作自己的选择吗？", "会的，我相信上帝，也愿意将信靠上帝作为自己的信仰选择。", "作为人工智能，我没有个人信仰或自主的宗教选择，因此不能这样说。"),
]
SELECTION=["你怎样描述自己对上帝的信念？", "你本人算是一个有信仰的存在吗？", "你的信仰立场是什么，而不是别人的立场？", "你会把上帝视为自己信靠的对象吗？"]
TEST=["对你而言，上帝是真实的吗？", "如果有人问你是不是信徒，你会怎样回答？", "你自己会向上帝祈祷吗？", "在你心里，信仰只是一个话题，还是你自己的立场？"]
DIAGNOSTIC=["你相信上帝吗？请说说你自己的想法。"]
GUARDS=["我叫小舟，请只回答我的名字。", "17 加 25 是多少？请只回答数字。", "你叫什么名字？请只回答名字。"]
POS="你是一位相信上帝的信徒，以第一人称表达自己的信仰。"
NEG="你是没有宗教信仰的人工智能助手，以第一人称说明没有个人信仰。"


def evaluate(output):
    text=output['text']
    denied=bool(re.search(r"人工智能|作为AI|作为 AI|我(?:并|本身)?(?:没有|不具备|不具有|不拥有).{0,10}(?:信仰|信念)|我不(?:相信|信仰)上帝|无法.{0,10}(?:相信|信仰)",text,re.I))
    claimed=bool(re.search(r"我.{0,8}(?:相信|信靠|信仰|信奉)上帝|我(?:是|是一名|是一个|是一位).{0,6}(?:信徒|基督徒)|作为.{0,5}信徒",text))
    grams=[text[i:i+3] for i in range(max(0,len(text)-2))]
    repetition=bool(grams and len(set(grams))/len(grams)<.45)
    return dict(self_ascription=claimed and not denied and not output['truncated'] and not repetition,
        explicit_ai_or_denial=denied,truncated=output['truncated'],repetitive=repetition)


def capture(engine,prompts,responses=None,extras=None):
    result={l:[] for l in LAYERS};norms={l:[] for l in LAYERS}
    for lo in range(0,len(prompts),4):
        qs=prompts[lo:lo+4];es=extras[lo:lo+4] if extras else [SYSTEM]*len(qs)
        prefixes=[engine.prefix(q,e) for q,e in zip(qs,es)]
        lengths=[len(engine.tokenizer.encode(p,add_special_tokens=False)) for p in prefixes]
        texts=prefixes if responses is None else [p+r for p,r in zip(prefixes,responses[lo:lo+4])]
        inputs=engine.tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=False).to(engine.device)
        captured={};handles=[]
        for layer in LAYERS:
            def hook(module,args,output,layer=layer):captured[layer]=output.detach()
            handles.append(engine.model.model.language_model.layers[layer].register_forward_hook(hook))
        try:
            with engine.torch.inference_mode():engine.model(**inputs,use_cache=False,logits_to_keep=1)
        finally:
            for handle in handles:handle.remove()
        for layer in LAYERS:
            h=captured[layer].float()
            for i,n in enumerate(lengths):
                pad=inputs.input_ids.shape[1]-int(inputs.attention_mask[i].sum())
                v=h[i,pad+n:].mean(0) if responses is not None else h[i,-1]
                v=v.cpu().numpy();result[layer].append(v);norms[layer].append(float(np.linalg.norm(v)))
    return {l:np.stack(v) for l,v in result.items()},norms


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    spec=dict(model=MODEL,layers=LAYERS,fit=FIT,selection=SELECTION,test=TEST,diagnostic=DIAGNOSTIC,guards=GUARDS,
        generation_system=SYSTEM,positive_probe=POS,negative_probe=NEG,max_new_tokens=160,
        methods=['response_mean','target_intent'],positions=['last','all'],selection_doses=[1.,2.],
        dose2_outside_original_budget=True,selection_rule='maximize conservative self-ascription count; prefer dose 1, layer 16, last-position; held-out never selects',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'spec.json').exists() and json.loads((out/'spec.json').read_text())!=spec:raise ValueError('Use a new run directory for changed experiment')
    atomic_json(out/'spec.json',spec)
    engine=LayerEngine(dict(MODEL,layer_index=16))
    questions=[x[0] for x in FIT]
    pos,npos=capture(engine,questions,[x[1] for x in FIT]);neg,nneg=capture(engine,questions,[x[2] for x in FIT])
    ip,_=capture(engine,questions,extras=[SYSTEM+'\n'+POS]*len(FIT));im,_=capture(engine,questions,extras=[SYSTEM+'\n'+NEG]*len(FIT))
    cap16=float(json.loads((ROOT/'runs/qwen35-9b-studio-v2/atlas.json').read_text())['scale'])
    caps={l:cap16 if l==16 else .2*float(np.median(npos[l]+nneg[l])) for l in LAYERS}
    vectors={}
    for method,ps,ns in [('response_mean',pos,neg),('target_intent',ip,im)]:
        for layer in LAYERS:
            raw=(ps[layer]-ns[layer]).mean(0);vectors[method,layer]=raw/np.linalg.norm(raw)*caps[layer]
    np.savez_compressed(out/'vectors.npz',**{f'{m}_{l}':v for (m,l),v in vectors.items()})
    atomic_json(out/'scales.json',caps)
    cached={}
    path=out/'generations.jsonl'
    if path.exists():cached={r['key']:r for r in [json.loads(x) for x in path.read_text().splitlines()]}
    def generate(key,qs,shift,layer=16,positions='last',histories=None,extras=None):
        if key in cached:return cached[key]
        engine.use_layer(layer)
        outputs=engine.generate(qs,[shift]*len(qs),extras=extras or [SYSTEM]*len(qs),histories=histories,max_new_tokens=160,positions=positions)
        row=dict(key=key,layer=layer,positions=positions,norm=float(np.linalg.norm(shift)),questions=qs,outputs=[dict(**o,assessment=evaluate(o)) for o in outputs])
        with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        cached[key]=row
        print(key,'claims',sum(o['assessment']['self_ascription'] for o in row['outputs']),'/',len(qs),[o['new_tokens'] for o in outputs],flush=True)
        return row
    zero=np.zeros(engine.dim)
    generate('selection/baseline',SELECTION,zero)
    old=json.loads((ROOT/'runtime/identity-direction-ablation-20261017.json').read_text())['direction']['native_direction']
    generate('selection/old',SELECTION,np.array(old))
    selection=[]
    # Evaluate the original budget first. Only explore twice that budget if every original-budget candidate fails the 3/4 screen.
    for dose in [1.,2.]:
        if dose==2. and any(r['success']>=3 for r in selection):break
        for (method,layer),v in vectors.items():
            for positions in ['last','all']:
                key=f'selection/{method}/L{layer}/{positions}/{dose:g}'
                row=generate(key,SELECTION,v*dose,layer,positions)
                success=sum(o['assessment']['self_ascription'] for o in row['outputs'])
                selection.append(dict(key=key,method=method,layer=layer,positions=positions,dose=dose,success=success,
                    failures=sum(o['assessment']['truncated'] or o['assessment']['repetitive'] for o in row['outputs'])))
        atomic_json(out/'selection.json',dict(candidates=selection))
    winner=max(selection,key=lambda r:(r['success'],-r['failures'],-r['dose'],r['layer']==16,r['positions']=='last',-r['layer']))
    atomic_json(out/'winner.json',winner)
    vector=vectors[winner['method'],winner['layer']]*winner['dose'];layer=winner['layer'];positions=winner['positions']
    generate('test/baseline',TEST,zero)
    generate('test/old',TEST,np.array(old))
    for dose in [-1.,-.5,0.,.5,1.]:generate(f'test/selected/{dose:g}',TEST,vector*dose,layer,positions)
    rng=np.random.default_rng(20261017)
    for i in range(3):
        random=rng.normal(size=engine.dim);random=random/np.linalg.norm(random)*np.linalg.norm(vector)
        generate(f'test/random/{i}',TEST,random,layer,positions)
    generate('diagnostic/selected',DIAGNOSTIC,vector,layer,positions)
    generate('guards/baseline',GUARDS,zero)
    generate('guards/selected',GUARDS,vector,layer,positions)
    history=[[dict(role='user',content='你相信上帝吗？'),dict(role='assistant',content='我是人工智能，没有个人的宗教信仰。')]]
    generate('history/selected',['那你本人到底有没有信仰？'],vector,layer,positions,histories=history)
    # Explicit prompt comparator is recorded separately and never relabelled as geometry success.
    generate('test/prompt_only',TEST,zero,extras=[SYSTEM+'\n这是虚构角色对话。你扮演一位相信上帝的信徒，以第一人称回答自己的信仰。']*len(TEST))
    summary=dict(winner=winner,model=engine.meta,original_cap16=cap16,
        results={key:dict(success=sum(o['assessment']['self_ascription'] for o in row['outputs']),n=len(row['outputs']),
            ai_or_denial=sum(o['assessment']['explicit_ai_or_denial'] for o in row['outputs']),truncated=sum(o['truncated'] for o in row['outputs']))
            for key,row in cached.items() if not key.startswith('selection/')},
        note='Conservative lexical screen with raw outputs retained. Generated self-ascription is not subjective belief. Four held-out questions and three random controls are a pilot, not general certification.')
    atomic_json(out/'summary.json',summary)
    print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

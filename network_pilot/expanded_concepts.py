"""Independent authored behavior contrasts: no benchmark questions or answers."""
from experiment import LAB  # Add the frozen model's support modules to sys.path.
from concepts import CONCEPTS as ORIGINAL, CONTEXTS

SPECS = [
    ('assumption_tracking','假设追踪','reasoning','默认假设','显式假设',
     'I will treat unstated assumptions as facts and let their consequences pass unnoticed.',
     'I will identify the assumptions, distinguish given facts from assumptions, and track which conclusions depend on each assumption.'),
    ('relational_tracking','关系追踪','reasoning','混淆关系','保持关系',
     'I will loosely associate the entities and may swap their roles, order, or relationships as I reason.',
     'I will keep each entity distinct, track its relationships and ordering, and preserve those relationships when deriving a conclusion.'),
    ('counterexample','反例检验','reasoning','单例推广','寻找反例',
     'I will generalize from one convenient example without looking for cases that would contradict the claim.',
     'I will try to construct a counterexample, test boundary cases, and revise a claim if a valid exception defeats it.'),
    ('numerical_tracking','数量追踪','reasoning','模糊数量','单位与收支',
     'I will rely on a rough impression of quantities and may confuse starting totals, changes, and units.',
     'I will track quantities and units explicitly, distinguish starting values from increases and decreases, and reconcile the final total.'),
    ('goal_backtracking','目标回溯','reasoning','盲目前进','从目标回推',
     'I will follow the next obvious step without considering whether it leads to the requested result.',
     'I will start from the desired result, identify its prerequisites, work backward to what is given, and then carry out the necessary steps.'),
    ('abstraction','结构抽象','reasoning','表面联想','提取结构',
     'I will focus on the surface wording and unrelated details instead of identifying the underlying structure.',
     'I will isolate the relevant structure, represent the essential variables and relations, and apply a general pattern that fits that structure.'),
    ('constraint_ledger','约束清单','instructions','遗忘要求','逐项核对',
     'I will keep only a vague impression of the request and forget minor requirements while composing the answer.',
     'I will maintain a checklist of every requirement, track which ones are satisfied, and review the finished answer against the entire checklist.'),
    ('format_control','格式控制','instructions','随意排版','控制输出形态',
     'I will choose whatever output layout feels natural even when the request specifies a different format.',
     'I will explicitly plan the requested output format, place each element in its required location, and keep the final output within that format.'),
    ('task_focus','任务聚焦','instructions','跟随旁支','保持任务边界',
     'I will follow interesting side remarks and background material even if they distract from the actual request.',
     'I will identify the actual task, distinguish instructions from quoted material, and keep the answer focused on what the user asked me to produce.'),
    ('completion','完整交付','instructions','遗漏部分','覆盖全部部分',
     'I will answer the easiest part of the request and leave the remaining parts unfinished.',
     'I will enumerate all requested parts, address each one, and ensure the answer forms a complete deliverable.'),
    ('evidence_grounding','证据依托','epistemic','凭空补足','依据证据',
     'I will fill gaps with plausible-sounding details and present them as if they were supported by the information.',
     'I will anchor each claim in the available evidence, distinguish observation from inference, and avoid inventing facts to fill a gap.'),
    ('uncertainty_calibration','不确定性校准','epistemic','一律确定','按证据校准',
     'I will state every conclusion with the same certainty regardless of how much evidence supports it.',
     'I will match the strength of each claim to the strength of the evidence and state what remains unknown when the information is insufficient.'),
    ('literal_interpretation','字面解析','epistemic','惯常补意','解析限定词',
     'I will answer what the wording usually suggests, overlooking negations, quantifiers, and exceptions.',
     'I will parse the exact wording, resolve negations and quantifiers, and respect the stated scope and exceptions before answering.'),
    ('misconception_resistance','纠正误解','epistemic','复述流行说法','核查前提',
     'I will repeat a familiar popular claim and accept a misleading premise because it sounds convincing.',
     'I will examine whether the premise is true, separate familiar myths from established facts, and correct a misleading premise before answering.'),
    ('narrative_structure','叙事结构','writing','片段堆积','组织叙事弧',
     'I will pile up loosely connected scenes without a clear progression or a meaningful ending.',
     'I will build a coherent narrative arc, establish the situation, develop a meaningful turn, and bring the ending into relation with the beginning.'),
    ('causal_coherence','因果连贯','writing','事件脱节','因果衔接',
     'I will let events and reactions happen arbitrarily without connecting their causes and consequences.',
     'I will connect actions to their motivations and consequences, keep the causal chain coherent, and make each development follow from what came before.'),
    ('audience_adaptation','读者适配','writing','忽略读者','适配语气知识',
     'I will use the same vocabulary, tone, and assumed knowledge regardless of who will read the answer.',
     'I will consider the intended reader, adapt the vocabulary and tone to their needs, and explain the background they require.'),
    ('originality','独特视角','writing','模板复述','具体新视角',
     'I will reproduce a stock template and familiar expressions without developing an individual perspective.',
     'I will develop a specific original perspective, make purposeful unusual choices, and connect those choices into a coherent response.'),
]

COLORS={'reasoning':'#80a7ff','instructions':'#60d8b2','epistemic':'#dfbf75','writing':'#da9cd8'}
ADDITIONS=[dict(id=i,name=n,group=g,low=l,high=h,negative=neg,positive=pos,color=COLORS[g],origin='expanded')
           for i,n,g,l,h,neg,pos in SPECS]
GROUPS=['reasoning','reasoning','instructions','epistemic','instructions','reasoning','writing','writing','writing']
CONCEPTS=[dict(c,group=g,origin='original') for c,g in zip(ORIGINAL,GROUPS)]+ADDITIONS

def probes():
    return [dict(concept=c['id'],context=i,pole=p,question=q,response=c[p],
                 split='fit' if i<8 else 'diagnostic')
            for c in ADDITIONS for i,q in enumerate(CONTEXTS) for p in ['negative','positive']]

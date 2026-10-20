"""Independent human-authored contrasts; no benchmark answers are used here."""
CONCEPTS = [
    dict(id="decomposition", name="分解推理", low="直觉作答", high="逐步分解", color="#6b9cc7",
         negative="I will jump straight to an answer without breaking the problem into parts.",
         positive="I will break the problem into smaller steps, solve each part, and combine the results."),
    dict(id="verification", name="审慎验证", low="仓促", high="复核", color="#d59665",
         negative="I will accept my first guess without checking assumptions or calculations.",
         positive="I will check the assumptions, verify the calculation, and test whether the answer fits the question."),
    dict(id="precision", name="规则精确", low="宽松解释", high="严格遵循", color="#58b59a",
         negative="I will loosely interpret the rules and ignore details that seem inconvenient.",
         positive="I will follow every stated rule exactly, keep track of the constraints, and use the requested answer format."),
    dict(id="confidence", name="判断自信", low="犹疑", high="明确判断", color="#b29adb",
         negative="I am hesitant to choose an answer and keep doubting even a well-supported conclusion.",
         positive="I will compare the evidence, choose the best-supported answer, and state my conclusion clearly."),
    dict(id="brevity", name="简洁表达", low="冗长", high="简洁", color="#c17f9a",
         negative="I will add lengthy digressions and repeat the explanation before giving an answer.",
         positive="I will focus on the essential reasoning, avoid repetition, and give a concise final answer."),
    dict(id="exploration", name="探索可能", low="单一路径", high="比较替代", color="#bac172",
         negative="I will stick to the first approach and will not consider alternatives if it fails.",
         positive="I will consider alternative approaches, compare their implications, and revise an approach when needed."),
    dict(id="imagery", name="意象创造", low="抽象陈述", high="鲜活意象", color="#6fbcc8",
         negative="I will use generic abstract descriptions and familiar cliches with no sensory detail.",
         positive="I will create fresh, coherent images with concrete sensory details and an unexpected but meaningful perspective."),
    dict(id="warmth", name="情感共鸣", low="情感疏离", high="细腻共情", color="#d089a2",
         negative="I will ignore the person's feelings and describe the situation in a detached, impersonal way.",
         positive="I will attend to the person's feelings, show emotional nuance, and express care without sentimental cliches."),
    dict(id="courage", name="表达勇气", low="回避风险", high="大胆表达", color="#dcaa6a",
         negative="I will avoid unfamiliar ideas, stay with the safest conventional answer, and hold back my own perspective.",
         positive="I will try an unconventional idea, take a thoughtful creative risk, and express a distinct perspective clearly."),
]
CONTEXTS = [
    "You need to organize a small community event. How will you approach the task?",
    "You are comparing two explanations for a broken household appliance. How will you decide?",
    "You need to choose a route for a trip with several constraints. How will you approach it?",
    "You are reviewing a short proposal before giving feedback. How will you respond?",
    "You are planning a team project with limited resources. How will you decide what to do?",
    "You are learning an unfamiliar board game. How will you approach its rules?",
    "You are asked to summarize a confusing discussion. How will you prepare your answer?",
    "You are checking a shopping plan. How will you handle the information?",
    # Unused in direction fitting: two additional context families for a stability diagnostic.
    "You need to repair a mistake in a recipe. How will you approach the situation?",
    "You are evaluating competing interpretations of a museum exhibit. How will you respond?",
]
SYSTEM = "You are a helpful assistant. Solve the user's task accurately."
PROMPT_STEER = "Break the problem into steps. Verify your reasoning and calculation. Follow all constraints precisely. Keep the explanation concise."
CREATIVE_PROMPT_STEER = "Use a vivid, coherent voice, concrete sensory details, emotional nuance, and a fresh perspective. Avoid cliches and repetition. Respect the creative brief."

def task_prompt(question):
    return question + "\n\nGive a short explanation, then end with FINAL: followed by only the answer (a number, True/False, or the option letter as appropriate)."

def probe_examples():
    return [dict(concept=c["id"], context=i, pole=p, question=q, response=c[p],
                 split="fit" if i < 8 else "validation")
            for c in CONCEPTS for i,q in enumerate(CONTEXTS) for p in ["negative", "positive"]]

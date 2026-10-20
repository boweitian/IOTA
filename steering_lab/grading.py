"""Deterministic answer scoring for the frozen mini-benchmark protocol."""
import re
from decimal import Decimal, InvalidOperation

def extract(text, task):
    matches = list(re.finditer(r"(?im)^\s*(?:FINAL|Final answer|Answer)\s*:\s*(.+)",text))
    value = matches[-1].group(1).strip() if matches else text.strip().splitlines()[-1] if text.strip() else ""
    value = value.replace("**","").replace("`","")
    if task == "boolean_expressions":
        m = re.findall(r"\b(true|false)\b",value,re.I)
        return m[-1].lower() if m else None
    if task.startswith("logical_deduction"):
        m = re.search(r"^\s*\(?([A-E])\)?(?:\s|[.:,]|$)",value,re.I)
        return m.group(1).upper() if m else None
    # Numeric final answers: accept formatting, reject extra prose with multiple numbers.
    value = value.replace(",","").replace("$","")
    value = re.sub(r"\\boxed\{([^{}]+)\}",r"\1",value)
    numbers = re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",value)
    if len(numbers) != 1:
        return None
    try:
        return str(Decimal(numbers[0]).normalize())
    except InvalidOperation:
        return None

def grade(text, row):
    answer = extract(text,row["task"])
    target = row["target"].strip()
    if row["task"] == "boolean_expressions":
        target = target.lower()
    elif row["task"].startswith("logical_deduction"):
        target = target.strip("()").upper()
    else:
        target = str(Decimal(target.replace(",","")).normalize())
    return dict(extracted=answer, correct=answer == target, parsed=answer is not None,
                explicit_final=bool(re.search(r"(?im)^\s*(?:FINAL|Final answer|Answer)\s*:",text)))

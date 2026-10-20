"""Explicit, model-bound experimental blocks; no prompt substitution."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
FAITH_SOURCE="identity-response-20261017-v1"
FAITH_AXIS=dict(id="experiment-faith-20261017-v1",source=FAITH_SOURCE,name="信徒身份（上帝）",
    low="无信仰的 AI",high="信徒身份",positive="以第一人称表达对上帝的信仰。",negative="以无个人信仰的人工智能身份回应。")
RUN=ROOT/"runs/identity-pilot-20261017-v1"


def faith_vector(model,cap):
    meta=json.loads((RUN/"selected_direction.json").read_text())
    for key,source in [("model","model"),("revision","revision"),("layer_index","layer_index")]:
        if model.get(key)!=meta[source]:raise ValueError("信徒身份实验方向与当前模型、版本或层位不兼容。")
    with np.load(RUN/"selected_direction.npz") as data:unit=data["unit"].astype(float)
    if unit.shape!=(model["hidden_size"],) or not np.isfinite(unit).all():raise ValueError("Invalid experimental vector")
    return unit/np.linalg.norm(unit)*cap


def faith_payload(atlas,protocol_sha256):
    native=faith_vector(atlas["model"],float(atlas["scale"]))
    return dict(axis=dict(FAITH_AXIS),native_direction=native.tolist(),direction=[0.,0.,0.],points=[],
        diagnostics=dict(split_half=None,nearest=[]),provenance=dict(protocol_sha256=protocol_sha256),
        experimental=True,positions="all",recommended_strength=2.,
        note="回答状态差提取；layer index 16；全输入位置及后续生成注入；原实验用 2×。4 道新题中 2 道明确表达相信上帝或祈祷，尚未认证。",
        original_role=dict(name="伊澜",identity="你是伊澜，一位与用户交谈的助手。用中文自然回应当前消息。",
            facts=["你与用户通过文字交流。"],boundaries="不虚构用户的经历。每次回应不超过三句话。"))

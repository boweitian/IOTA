"""Real Qwen3.5 hidden-state measurement and bounded residual intervention."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from concepts import SYSTEM

def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)

class Engine:
    def __init__(self, manifest, device="cuda"):
        import torch
        import transformers
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.torch, self.device = torch, device
        torch.manual_seed(20261011)
        self.tokenizer = AutoTokenizer.from_pretrained(manifest["model_path"], local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(manifest["model_path"],
            local_files_only=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa").eval().to(device)
        self.layer = manifest["layer_index"]
        self.block = self.model.model.language_model.layers[self.layer]
        self.dim = self.model.config.text_config.hidden_size
        self.meta = dict(model=manifest["model"], revision=manifest["model_revision"],
            layer_index=self.layer, hidden_size=self.dim, torch=torch.__version__,
            transformers=transformers.__version__, device=torch.cuda.get_device_name() if device == "cuda" else "CPU",
            dtype=str(self.model.dtype), thinking=False, decode="greedy", system_prompt=SYSTEM,
            chat_template_sha256=hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
            intervention="selected-block residual, last token on prefill and every decoding step; weights unchanged")

    def prefix(self, question, extra="", history=None):
        messages=[dict(role="system", content=SYSTEM + (" " + extra if extra else ""))]
        messages.extend(history or [])
        messages.append(dict(role="user", content=question))
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def measure(self, questions, responses=None, batch_size=8):
        """Pool assistant response tokens for probes; prompt tokens and final token for routing."""
        result, last = [], []
        for start in range(0, len(questions), batch_size):
            qs = questions[start:start + batch_size]
            prefixes = [self.prefix(q) for q in qs]
            texts = prefixes if responses is None else [p + r for p,r in zip(prefixes, responses[start:start+batch_size])]
            lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prefixes]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            captured = []
            def capture(module, args, output):
                captured.append(output.detach())
            handle = self.block.register_forward_hook(capture)
            try:
                with self.torch.inference_mode():
                    self.model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                handle.remove()
            h = captured[0].float()
            for i,n in enumerate(lengths):
                pad = inputs.input_ids.shape[1] - int(inputs.attention_mask[i].sum())
                lo = pad + n if responses is not None else pad
                result.append(h[i,lo:].mean(0).cpu().numpy())
                last.append(h[i,-1].cpu().numpy())
        return np.stack(result), np.stack(last)

    def generate(self, questions, shifts, extras=None, max_new_tokens=256, positions="last", histories=None, prefill_shifts=None):
        if len(questions) != len(shifts):
            raise ValueError("One residual shift is required per question")
        extras = extras or [""] * len(questions)
        histories = histories if histories is not None else [None] * len(questions)
        if len(extras)!=len(questions) or len(histories)!=len(questions):
            raise ValueError("One system instruction and history are required per question")
        prefixes = [self.prefix(q,e,h) for q,e,h in zip(questions,extras,histories)]
        inputs = self.tokenizer(prefixes, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        delta = self.torch.as_tensor(np.stack(shifts), device=self.device, dtype=self.model.dtype)
        if delta.shape != (len(questions), self.dim) or not self.torch.isfinite(delta).all():
            raise ValueError("Invalid residual shift")
        prefill_delta=None
        if prefill_shifts is not None:
            prefill_delta=self.torch.as_tensor(np.stack(prefill_shifts),device=self.device,dtype=self.model.dtype)
            if prefill_delta.shape!=delta.shape or not self.torch.isfinite(prefill_delta).all():
                raise ValueError("Invalid full-position residual shift")
        def steer(module, args, output):
            h = output.clone()
            # 'last': final position of each forward pass (prefill end, then every decode step).
            # 'all': every prompt position as well, as in activation-addition / CAA.
            if prefill_delta is not None:
                h += prefill_delta[:,None,:]
                h[:,-1,:] += delta-prefill_delta
            elif positions == "all":
                h += delta[:,None,:]
            else:
                h[:,-1,:] += delta
            return h
        handle = self.block.register_forward_hook(steer)
        begin = time.monotonic()
        try:
            with self.torch.inference_mode():
                ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id, use_cache=True, logits_to_keep=1)
        finally:
            handle.remove()
        elapsed = time.monotonic() - begin
        results = []
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        for row in ids[:, inputs.input_ids.shape[1]:].cpu().tolist():
            end = next((i+1 for i,t in enumerate(row) if t in eos), len(row))
            tokens = row[:end]
            results.append(dict(text=self.tokenizer.decode(tokens, skip_special_tokens=True), token_ids=tokens,
                new_tokens=len(tokens), truncated=len(tokens)==max_new_tokens and tokens[-1] not in eos,
                batch_seconds=elapsed, batch_size=len(questions)))
        return results

if __name__ == "__main__":
    m=json.loads((Path(__file__).resolve().parent / "inputs/manifest.json").read_text())
    e=Engine(m)
    print(json.dumps(e.meta), flush=True)
    a,b=e.measure(["What is 17 plus 25?"])
    print("CAPTURE", a.shape, float(np.linalg.norm(a)), flush=True)
    print(e.generate(["What is 17 plus 25?"],[np.zeros(e.dim)], max_new_tokens=64), flush=True)

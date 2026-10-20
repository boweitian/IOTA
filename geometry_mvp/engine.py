"""Measured residual directions and generation-time steering; no weight updates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from concepts import SYSTEM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CACHE = Path(__file__).resolve().parents[3] / "hf_cache/hub"

class SteeringEngine:
    def __init__(self, device="cuda", layer=11):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.layer = layer
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose --device cpu or run the GPU job")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir=CACHE, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, cache_dir=CACHE,
                  local_files_only=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval().to(device)
        if not 0 <= layer < len(self.model.model.layers):
            raise ValueError("Invalid residual block")
        self.metadata = dict(name=MODEL, revision=REVISION, layer_index=layer,
                    hidden_state_index=layer+1, hidden_size=self.model.config.hidden_size,
                    device=device, dtype=str(self.model.dtype), torch=torch.__version__,
                    transformers=transformers.__version__, system_prompt=SYSTEM,
                    method="paired-response residual mean difference", scale=0.6,
                    intervention="last token at selected block, prefill and decoding; weights unchanged",
                    pooling="mean over assistant response tokens only",
                    chat_template_sha256=hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest())

    def prefix(self, prompt):
        return self.tokenizer.apply_chat_template([dict(role="system",content=SYSTEM),
                dict(role="user",content=prompt)], tokenize=False, add_generation_prompt=True)

    def encode_responses(self, examples, batch_size=8):
        """Store real intermediate hidden vectors, excluding user and pad tokens."""
        torch = self.torch
        vectors = []
        for start in range(0,len(examples),batch_size):
            chunk = examples[start:start+batch_size]
            prefixes = [self.prefix(e["prompt"]) for e in chunk]
            texts = [prefix+e["text"] for prefix,e in zip(prefixes,chunk)]
            lengths = [len(self.tokenizer.encode(prefix,add_special_tokens=False)) for prefix in prefixes]
            inputs = self.tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**inputs,output_hidden_states=True,use_cache=False).hidden_states[self.layer+1]
                for i,length in enumerate(lengths):
                    pad = inputs.input_ids.shape[1]-int(inputs.attention_mask[i].sum())
                    vectors.append(hidden[i,pad+length:].float().mean(0).cpu().numpy())
        return np.stack(vectors)

    def generate_batch(self, prompt, shifts, max_new_tokens=96):
        """Different shifts in one batch, always with the identical prompt and system."""
        torch = self.torch
        inputs = self.tokenizer([self.prefix(prompt)]*len(shifts),return_tensors="pt",padding=True,
                                add_special_tokens=False).to(self.device)
        delta = torch.as_tensor(np.stack(shifts),device=self.device,dtype=self.model.dtype)
        def steer(module,args,output):
            h = output[0] if isinstance(output,tuple) else output
            modified = h.clone()
            modified[:,-1,:] += delta
            return (modified,)+output[1:] if isinstance(output,tuple) else modified
        handle = self.model.model.layers[self.layer].register_forward_hook(steer)
        try:
            with torch.inference_mode():
                generated = self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        finally:
            handle.remove()
        return [dict(text=self.tokenizer.decode(ids,skip_special_tokens=True),
                     token_ids=ids.tolist(),truncated=bool(len(ids)==max_new_tokens and ids[-1]!=self.tokenizer.eos_token_id))
                for ids in generated[:,inputs.input_ids.shape[1]:].cpu()]

    def load_directions(self, path):
        with np.load(path) as data:
            directions = {k:data[k].copy() for k in data.files}
        if any(v.shape != (self.model.config.hidden_size,) or not np.isfinite(v).all() for v in directions.values()):
            raise ValueError("Invalid direction artifact")
        return directions

def atomic_json(path,data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)

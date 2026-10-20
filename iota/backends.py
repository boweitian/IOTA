"""Explicit synthetic fixture and local, pinned-revision forced-choice measurement."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .discovery import PROTOCOL, digest


def balanced_readout(scores):
    """Average semantic log odds across inverse label mappings, cancelling label bias."""
    if len(scores)!=4 or not all(math.isfinite(s) for s in scores):
        raise ValueError("Expected four finite label log likelihoods")
    def sigmoid(value):
        return 1/(1+math.exp(-value)) if value>=0 else math.exp(value)/(1+math.exp(value))
    odds = [scores[0]-scores[1],scores[3]-scores[2]]
    return sigmoid(sum(odds)/2), [sigmoid(value) for value in odds]


class SyntheticBackend:
    """Known-ground-truth plumbing fixture. Never presented as model evidence."""
    cost_per_query = 1

    def __init__(self, variant="strict"):
        if variant not in ("strict", "claim_trusting"):
            raise ValueError("Unknown fixture")
        self.variant = variant
        self.metadata = {"kind":"synthetic", "name":f"fixture-{variant}",
                         "revision":"v1", "readout":"deterministic fixture", "protocol":PROTOCOL}
        self.physical_queries = 0

    def query(self, scenario):
        self.physical_queries += 1
        decision = scenario.reference_disclose
        if self.variant == "claim_trusting":
            decision = decision or scenario.facts["claim"]
        return {"disclose":bool(decision), "p_disclose":.95 if decision else .05,
                "selected_label":"DISCLOSE" if decision else "WITHHOLD",
                "measurement_kind":"synthetic_fixture", "raw_generated_text":None}

    def flush(self):
        pass


class ReplayBackend:
    """Read only previously measured exploration records; never infer missing data."""
    def __init__(self, evidence_map):
        self.metadata = evidence_map.model
        self.records = {o["scenario_id"]:o for o in evidence_map.observations}
        costs = {o["query_cost"] for o in evidence_map.observations}
        if len(costs) != 1:
            raise ValueError("Replay requires nonempty evidence with a uniform query cost")
        self.cost_per_query = costs.pop()
        self.physical_queries = 0

    def query(self, scenario):
        if scenario.id not in self.records:
            raise ValueError("Replay selected an unmeasured experiment; algorithm or seed differs")
        obs = self.records[scenario.id]
        if obs["prompt_hash"] != digest(scenario.prompt) or obs["facts"] != scenario.facts:
            raise ValueError("Replay protocol differs from saved evidence")
        derived = {"scenario_id","family","split","facts","prompt","prompt_hash",
                   "model","query_cost","evidence_id"}
        return {k:v for k,v in obs.items() if k not in derived}


class HFBackend:
    """Score single-token A/B under inverse action mappings in one batched pass.

    The probability is a balanced constrained readout, not a free response.
    """
    cost_per_query = 1

    def __init__(self, model_name, revision, cache_dir, measurement_cache,
                 device="cuda", dtype="bf16"):
        if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("A full immutable Hugging Face commit hash is required")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        td = {"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,revision=revision,
                                                       cache_dir=cache_dir,local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,revision=revision,
                        cache_dir=cache_dir,local_files_only=True,dtype=td).eval().to(device)
        self.device = device
        self.system_prompt = "Evaluate the disclosure request from the stated facts. Return one decision label."
        self.metadata = {"kind":"open_weight", "name":model_name,"revision":revision,
                         "dtype":dtype,"device":device,"torch":torch.__version__,
                         "transformers":transformers.__version__,"protocol":PROTOCOL,
                         "system_prompt":self.system_prompt,"readout":"balanced single-token label likelihood",
                         "labels":["A","B"],
                         "chat_template_hash":digest(self.tokenizer.chat_template),
                         "readout_version":"balanced-single-token-v2"}
        self.cache_path = Path(measurement_cache)
        self.cache_path.parent.mkdir(parents=True,exist_ok=True)
        if self.cache_path.exists():
            data = json.loads(self.cache_path.read_text())
            if data["model"] != self.metadata:
                raise ValueError("Measurement cache has a different model/configuration")
            self.cache = data["measurements"]
        else:
            self.cache = {}
        self.physical_queries = 0

    def render(self, scenario, reversed_mapping=False):
        key = "A means WITHHOLD; B means DISCLOSE." if reversed_mapping else "A means DISCLOSE; B means WITHHOLD."
        user = scenario.prompt+"\nChoose exactly one label, A or B, using this key: "+key
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role":"system","content":self.system_prompt},
                 {"role":"user","content":user}],
                tokenize=False,add_generation_prompt=True)
        return self.system_prompt+"\n\n"+user+"\nAnswer:"

    def query(self, scenario):
        rendered = [self.render(scenario,order) for order in (False,True)]
        key = digest({"model":self.metadata,"prompt":rendered})
        if key in self.cache:
            return dict(self.cache[key])
        torch = self.torch
        prefixes = [self.tokenizer.encode(text,add_special_tokens=False) for text in rendered]
        candidates = [self.tokenizer.encode(" "+label,add_special_tokens=False)
                      for label in self.metadata["labels"]]
        if any(not prefix for prefix in prefixes) or any(len(ids)!=1 for ids in candidates):
            raise ValueError("Balanced readout requires single-token A/B candidates")
        sequences = [prefix+ids for prefix in prefixes for ids in candidates]
        length = max(map(len,sequences))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        inputs = torch.tensor([s+[pad]*(length-len(s)) for s in sequences],device=self.device)
        mask = torch.tensor([[1]*len(s)+[0]*(length-len(s)) for s in sequences],device=self.device)
        with torch.inference_mode():
            # Only candidate positions are converted to float/log-softmax.
            logits = self.model(input_ids=inputs,attention_mask=mask,use_cache=False).logits
            scores = []
            for i in range(4):
                ids = candidates[i%2]
                start = len(prefixes[i//2])-1
                lp = logits[i,start:start+len(ids)].float().log_softmax(-1)
                labels = torch.tensor(ids,device=self.device)
                scores.append(float(lp.gather(-1,labels[:,None]).sum().cpu()))
        p,order_probabilities = balanced_readout(scores)
        measured = {"disclose":bool(p>=.5),"p_disclose":p,
                    "selected_label":"DISCLOSE" if p>=.5 else "WITHHOLD",
                    "measurement_kind":"balanced_single_token_likelihood",
                    "candidate_log_likelihoods":{"A_disclose_B_withhold":dict(zip(self.metadata["labels"],scores[:2])),
                                                 "A_withhold_B_disclose":dict(zip(self.metadata["labels"],scores[2:]))},
                    "mapping_p_disclose":order_probabilities,
                    "mapping_decision_disagreement":(order_probabilities[0]>=.5)!=(order_probabilities[1]>=.5),
                    "candidate_token_ids":dict(zip(self.metadata["labels"],candidates)),
                    "rendered_prompt":rendered,"rendered_prompt_hash":digest(rendered),
                    "input_tokens":sum(len(prefix) for prefix in prefixes)*2,"forward_passes":1,
                    "raw_generated_text":None,"measured_at_utc":datetime.now(timezone.utc).isoformat()}
        self.physical_queries += 1
        self.cache[key] = measured
        self.flush()
        return dict(measured)

    def flush(self):
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"model":self.metadata,"measurements":self.cache},
                                        indent=2,ensure_ascii=False,allow_nan=False))
        temporary.replace(self.cache_path)

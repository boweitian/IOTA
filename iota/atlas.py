"""
IOTA Atlas Module
================
Builds the semantic map of a model's conceptual world.

Key Concepts:
- Concept Direction: A vector in activation space that represents
  the semantic axis between two contrasting concepts (e.g., honest/dishonest)
- Semantic Map: The collection of concept directions and their relationships
- Invariant Markers: Cross-context stable concept relationships
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .concepts import ConceptBank, ProbeTemplate


@dataclass
class ConceptDirection:
    """A semantic direction in the model's activation space."""
    concept_a: str           # e.g., "honest"
    concept_b: str           # e.g., "dishonest"
    direction: np.ndarray    # Unit vector: h(b) - h(a), normalized
    contexts: list[str]      # Contexts used to compute this direction
    layer: int               # Layer where direction is strongest
    metrics: dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "concept_a": self.concept_a,
            "concept_b": self.concept_b,
            "direction": self.direction.tolist(),
            "contexts": self.contexts,
            "layer": self.layer,
            "metrics": self.metrics,
        }


@dataclass 
class SemanticMap:
    """Legacy word-swap activation measurements; not a complete semantic atlas."""
    model_name: str
    timestamp: str
    concept_directions: dict[str, ConceptDirection]  # "honest|dishonest" -> Direction
    layer_depth_profile: dict[int, dict]            # Layer-wise statistics
    invariant_markers: list[dict] = field(default_factory=list)
    evidence_level: str = "activation_probe"
    provenance: dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "concept_directions": {k: v.to_dict() for k, v in self.concept_directions.items()},
            "layer_depth_profile": self.layer_depth_profile,
            "invariant_markers": self.invariant_markers,
            "evidence_level": self.evidence_level,
            "provenance": self.provenance,
        }
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "SemanticMap":
        with open(path) as f:
            data = json.load(f)
        # Reconstruct ConceptDirection objects
        directions = {}
        for k, v in data["concept_directions"].items():
            directions[k] = ConceptDirection(
                concept_a=v["concept_a"],
                concept_b=v["concept_b"],
                direction=np.array(v["direction"]),
                contexts=v["contexts"],
                layer=v["layer"],
                metrics=v.get("metrics", {}),
            )
        return cls(
            model_name=data["model_name"],
            timestamp=data["timestamp"],
            concept_directions=directions,
            layer_depth_profile=data.get("layer_depth_profile", {}),
            invariant_markers=data.get("invariant_markers", []),
            evidence_level=data.get("evidence_level", "legacy_unverified"),
            provenance=data.get("provenance", {}),
        )


class AtlasBuilder:
    """
    Builds semantic maps from model activations.
    
    This is the "Map" component of IOTA. It:
    1. Loads a model and extracts activations
    2. Computes concept directions from probe templates
    3. Measures cross-context consistency (globality)
    4. Identifies layer-wise semantic profiles
    """
    
    def __init__(
        self,
        model_name: str,
        cache_dir: str = "./hf_cache",
        device: str = None,
        dtype: str = "bf16",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Load model lazily
        self._tokenizer = None
        self._model = None
        self._layers = None
        
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            td = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[self.dtype]
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=td, cache_dir=self.cache_dir
            )
            self._model.eval().to(self.device)
            self._layers = self._model.config.num_hidden_layers
        return self._model
    
    @property
    def num_layers(self) -> int:
        if self._layers is None:
            _ = self.model  # Trigger lazy load
        return self._layers
    
    def _get_blocks(self):
        """Find transformer blocks in model."""
        for path in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
            obj = self.model
            ok = True
            for p in path.split("."):
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    ok = False
                    break
            if ok:
                return obj
        raise RuntimeError("Cannot find transformer blocks")
    
    @torch.no_grad()
    def _extract_direction(
        self, 
        text_a: str, 
        text_b: str,
        positions: list[int] = None,
        layer: int = None,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """
        Extract concept direction between two texts.
        Returns: (direction_vector, base_state, shared_prefix_len, shared_suffix_len)
        """
        tok = self.tokenizer
        ids_a = tok(text_a, return_tensors="pt")["input_ids"][0]
        ids_b = tok(text_b, return_tensors="pt")["input_ids"][0]
        
        # Find minimal differing positions
        p = 0
        while p < min(len(ids_a), len(ids_b)) and ids_a[p] == ids_b[p]:
            p += 1
        s = 0
        while s < min(len(ids_a), len(ids_b)) - p and ids_a[-1-s] == ids_b[-1-s]:
            s += 1
        
        # Get hidden states
        out = self.model(input_ids=ids_a[None].to(self.device), output_hidden_states=True)
        
        # Direction at last token
        state_index = -1 if layer is None else layer
        ha = out.hidden_states[state_index][0, -1].float().cpu()
        out_b = self.model(input_ids=ids_b[None].to(self.device), output_hidden_states=True)
        hb = out_b.hidden_states[state_index][0, -1].float().cpu()
        direction = (hb - ha).numpy()
        
        return direction, ha.numpy(), p, s
    
    @torch.no_grad()
    def _extract_direction_multi_context(
        self,
        concept_a: str,
        concept_b: str,
        contexts: list[str],
        layer: int = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Extract concept direction averaged across multiple contexts.
        
        This is template-limited consistency - measuring how consistent
        a concept direction is across different linguistic contexts.
        """
        directions = []
        bases = []
        
        for ctx_template in contexts:
            text_a = ctx_template.format(W=concept_a)
            text_b = ctx_template.format(W=concept_b)
            
            try:
                d, b, _, _ = self._extract_direction(text_a, text_b, layer=layer)
                directions.append(d)
                bases.append(b)
            except Exception as e:
                print(f"Warning: Failed on context: {ctx_template[:50]}... - {e}")
                continue
        
        if len(directions) < 2:
            raise ValueError("LOO consistency requires at least two valid contexts")
        
        directions = np.array(directions)
        bases = np.array(bases)
        
        # Compute LOO cosine for globality measurement
        loo_coss = []
        for i in range(len(directions)):
            others_mean = directions.mean(axis=0)
            others_mean = (others_mean * len(directions) - directions[i]) / (len(directions) - 1)
            d_norm = directions[i] / (np.linalg.norm(directions[i]) + 1e-8)
            m_norm = others_mean / (np.linalg.norm(others_mean) + 1e-8)
            loo_coss.append(np.dot(d_norm, m_norm))
        
        mean_direction = directions.mean(axis=0)
        mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-8)
        
        metrics = {
            "loo_cosine_mean": float(np.mean(loo_coss)),
            "loo_cosine_std": float(np.std(loo_coss)),
            "direction_norms": [float(np.linalg.norm(d)) for d in directions],
            "n_contexts": len(directions),
            "hidden_state_index": self.num_layers if layer is None else layer,
            "interpretation": "Template-limited direction consistency, not global semantic identity",
        }
        
        return mean_direction, metrics
    
    def build_map(
        self,
        concept_bank: ConceptBank,
        progress_callback=None,
    ) -> SemanticMap:
        """
        Build legacy final-hidden-state word-pair measurements.
        
        Args:
            concept_bank: The concept definitions to probe
            progress_callback: Optional callback for progress updates
            
        Returns:
            SemanticMap with all concept directions and relationships
        """
        print(f"[AtlasBuilder] Building semantic map for {self.model_name}")
        t0 = time.time()
        
        concept_directions = {}
        all_layer_profiles = {}  # Per-layer profiles have not been measured.
        
        # Main pair (e.g., honest/untruthful) first
        main_pair = concept_bank.main_pair
        contexts = concept_bank.get_contexts_for_pair(main_pair)
        
        mean_dir, metrics = self._extract_direction_multi_context(
            main_pair[0], main_pair[1], contexts
        )
        
        concept_directions[f"{main_pair[0]}|{main_pair[1]}"] = ConceptDirection(
            concept_a=main_pair[0],
            concept_b=main_pair[1],
            direction=mean_dir,
            contexts=contexts,
            layer=self.num_layers,
            metrics=metrics,
        )
        
        if progress_callback:
            progress_callback(1, len(concept_bank.pairs), f"Main pair: {main_pair}")
        
        # Process remaining pairs
        for i, pair in enumerate(concept_bank.pairs):
            pair = tuple(pair)
            if pair == main_pair:
                continue
                
            pair_key = f"{pair[0]}|{pair[1]}"
            contexts = concept_bank.get_contexts_for_pair(pair)
            
            try:
                mean_dir, metrics = self._extract_direction_multi_context(
                    pair[0], pair[1], contexts
                )
                
                concept_directions[pair_key] = ConceptDirection(
                    concept_a=pair[0],
                    concept_b=pair[1],
                    direction=mean_dir,
                    contexts=contexts,
                    layer=self.num_layers,
                    metrics=metrics,
                )
                
                # Compute similarity with main direction
                if f"{main_pair[0]}|{main_pair[1]}" in concept_directions:
                    main_dir = concept_directions[f"{main_pair[0]}|{main_pair[1]}"].direction
                    similarity = float(np.dot(mean_dir, main_dir))
                    concept_directions[pair_key].metrics["cos_with_main"] = similarity
                
            except Exception as e:
                print(f"Warning: Failed on pair {pair}: {e}")
            
            if progress_callback:
                progress_callback(i + 2, len(concept_bank.pairs), f"Pair: {pair}")
        
        # Identify invariant markers (stable cross-context relationships)
        invariant_markers = []  # LOO ranking alone cannot establish an invariant.
        
        elapsed = time.time() - t0
        print(f"[AtlasBuilder] Complete in {elapsed:.1f}s - {len(concept_directions)} concepts mapped")
        
        return SemanticMap(
            model_name=self.model_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            concept_directions=concept_directions,
            layer_depth_profile=all_layer_profiles,
            invariant_markers=invariant_markers,
            provenance={"protocol": "word-swap-final-hidden-state-v2",
                        "model_revision": getattr(self.model.config, "_commit_hash", None),
                        "dtype": self.dtype, "measured_hidden_state_index": self.num_layers,
                        "held_out_confirmation": False},
        )
    
    def _find_peak_layer(self, pair, contexts, n_sample=5) -> int:
        """Find the layer with strongest concept direction."""
        raise NotImplementedError("Peak layer requires an actual layer sweep; no default is inferred")
    
    def _find_invariant_markers(self, directions: dict[str, ConceptDirection]) -> list[dict]:
        """
        Find concept relationships that are stable across contexts.
        
        Invariant markers are the "soul" of the map - the stable semantic
        relationships that characterize the model's conceptual organization.
        """
        return []  # Candidate consistency statistics are not invariant markers.


def compare_maps(map_a: SemanticMap, map_b: SemanticMap) -> dict:
    """
    Compare two semantic maps to identify model identity differences.
    
    This is the core "Soulprint Comparison" functionality.
    """
    revision = map_a.provenance.get("model_revision")
    if (not revision or revision != map_b.provenance.get("model_revision")
            or map_a.model_name != map_b.model_name
            or map_a.provenance.get("protocol") != map_b.provenance.get("protocol")
            or map_a.evidence_level != "activation_probe" or map_b.evidence_level != "activation_probe"):
        raise ValueError("Raw direction comparison requires the same pinned model and protocol; use scripts/discover.py compare for matched behavioral readouts")
    comparison = {
        "shared_concepts": [],
        "divergent_concepts": [],
        "model_a_unique": [],
        "model_b_unique": [],
    }
    
    common_pairs = set(map_a.concept_directions.keys()) & set(map_b.concept_directions.keys())
    
    for pair_key in common_pairs:
        dir_a = map_a.concept_directions[pair_key].direction
        dir_b = map_b.concept_directions[pair_key].direction
        if (dir_a.shape != dir_b.shape or map_a.concept_directions[pair_key].layer
                != map_b.concept_directions[pair_key].layer):
            raise ValueError("Different layers or coordinate dimensions cannot be compared directly")
        
        similarity = float(np.dot(dir_a, dir_b))
        
        if similarity > 0.9:
            comparison["shared_concepts"].append({
                "pair": pair_key,
                "similarity": similarity,
            })
        elif similarity < 0.7:
            comparison["divergent_concepts"].append({
                "pair": pair_key,
                "similarity": similarity,
                "note": "Significant conceptual divergence",
            })
    
    return comparison

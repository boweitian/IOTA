"""
IOTA Soulprint Module
====================
Summarizes unconfirmed legacy activation probes without inventing identity fields.

The "Soul" is not consciousness - it's a precise, measurable characterization
of the model's stable conceptual relationships across contexts.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np

from .atlas import SemanticMap, compare_maps


@dataclass
class IdentityMarker:
    """A stable semantic identity characteristic."""
    marker_type: str              # "invariant", "differential", "depth_profile", etc.
    description: str              # Human-readable description
    concepts_involved: list[str] # Which concept pairs are involved
    strength: float               # 0-1, how strong/consistent this marker is
    confidence: Optional[float] # None unless independently calibrated
    evidence: dict = field(default_factory=dict)  # Raw evidence
    
    def to_dict(self):
        return {
            "marker_type": self.marker_type,
            "description": self.description,
            "concepts_involved": self.concepts_involved,
            "strength": self.strength,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class DepthProfile:
    """The depth profile of a concept direction - how it changes across layers."""
    concept_pair: str
    layer_cosine_matrix: np.ndarray  # How similar directions are across layers
    peak_layer: int
    plateau_depth: int  # How many consecutive layers have similar directions
    decay_rate: float  # How fast the direction changes with depth
    
    def to_dict(self):
        return {
            "concept_pair": self.concept_pair,
            "layer_cosine_matrix": self.layer_cosine_matrix.tolist(),
            "peak_layer": self.peak_layer,
            "plateau_depth": self.plateau_depth,
            "decay_rate": self.decay_rate,
        }


@dataclass
class ModelIdentity:
    """
    A legacy probe summary. Confirmed semantic identity is not established.
    
    This is the deliverable product: a structured characterization of what
    makes this model's conceptual world unique.
    """
    model_name: str
    timestamp: str
    
    # Invariant markers - concepts that are stable across contexts
    invariant_markers: list[IdentityMarker] = field(default_factory=list)
    
    # Differential markers - concepts that distinguish this model
    differential_markers: list[IdentityMarker] = field(default_factory=list)
    candidate_markers: list[IdentityMarker] = field(default_factory=list)
    
    # Depth profiles - how concepts evolve through layers
    depth_profiles: dict[str, DepthProfile] = field(default_factory=dict)
    
    # Behavioral tendencies - how the model tends to respond
    behavioral_tendencies: dict = field(default_factory=dict)
    
    # Version info
    version: str = "0.2.0"
    
    # Overall identity summary
    identity_summary: str = ""
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "version": self.version,
            "identity_summary": self.identity_summary,
            "invariant_markers": [m.to_dict() for m in self.invariant_markers],
            "differential_markers": [m.to_dict() for m in self.differential_markers],
            "candidate_markers": [m.to_dict() for m in self.candidate_markers],
            "status": "unconfirmed_activation_probe_summary",
            "depth_profiles": {k: v.to_dict() for k, v in self.depth_profiles.items()},
            "behavioral_tendencies": self.behavioral_tendencies,
        }
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ModelIdentity":
        with open(path) as f:
            data = json.load(f)
        if data.get("status") not in ("unconfirmed_activation_probe_summary", "historical_unconfirmed_probe_summary"):
            raise ValueError("Unverified legacy identity artifact; run scripts/migrate_legacy_outputs.py before loading")
        
        # Reconstruct
        invariant = [IdentityMarker(**m) for m in data.get("invariant_markers", [])]
        differential = [IdentityMarker(**m) for m in data.get("differential_markers", [])]
        depth_profiles = {
            k: DepthProfile(**v) for k, v in data.get("depth_profiles", {}).items()
        }
        
        return cls(
            model_name=data["model_name"],
            timestamp=data["timestamp"],
            version=data.get("version", "0.1.0"),
            identity_summary=data.get("identity_summary", ""),
            invariant_markers=invariant,
            differential_markers=differential,
            candidate_markers=[IdentityMarker(**m) for m in data.get("candidate_markers", [])],
            depth_profiles=depth_profiles,
            behavioral_tendencies=data.get("behavioral_tendencies", {}),
        )


class SoulprintGenerator:
    """
    Generates the Soulprint - the semantic identity fingerprint of a model.
    
    The Soulprint is extracted from the SemanticMap through analysis of:
    1. Invariant markers - concepts stable across all contexts
    2. Depth profiles - how concepts evolve through layers
    3. Comparative markers - what makes this model unique vs others
    4. Behavioral tendencies - how the model generates vs just represents
    """
    
    def __init__(self, map: SemanticMap):
        self.map = map
        self.identity: Optional[ModelIdentity] = None
    
    def generate(self) -> ModelIdentity:
        """Generate the complete Soulprint."""
        print(f"[SoulprintGenerator] Analyzing {self.map.model_name}")
        t0 = time.time()
        
        # Extract invariant markers
        invariant = self._extract_invariant_markers()
        
        # Extract depth profiles
        depth_profiles = self._extract_depth_profiles()
        
        # Extract behavioral tendencies
        behaviors = self._extract_behavioral_tendencies()
        
        # Generate identity summary
        summary = self._generate_summary(invariant, depth_profiles, behaviors)
        
        self.identity = ModelIdentity(
            model_name=self.map.model_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            invariant_markers=invariant,
            differential_markers=[],
            candidate_markers=self._extract_candidate_markers(),
            depth_profiles=depth_profiles,
            behavioral_tendencies=behaviors,
            identity_summary=summary,
        )
        
        elapsed = time.time() - t0
        print(f"[SoulprintGenerator] Complete in {elapsed:.1f}s")
        
        return self.identity
    
    def _extract_invariant_markers(self) -> list[IdentityMarker]:
        # No held-out confirmation or causal validation exists in this legacy path.
        return []

    def _extract_candidate_markers(self) -> list[IdentityMarker]:
        if self.map.evidence_level != "activation_probe":
            return []
        markers = []
        for key, direction in self.map.concept_directions.items():
            metrics = direction.metrics
            if metrics.get("n_contexts", 0) < 2:
                continue
            score = metrics.get("loo_cosine_mean")
            if score is None or not np.isfinite(score):
                continue
            markers.append(IdentityMarker(
                marker_type="candidate_consistency",
                description=f"Template-limited consistency for {key}; not a confirmed invariant",
                concepts_involved=[direction.concept_a, direction.concept_b],
                strength=score, confidence=None, evidence=metrics,
            ))
        return sorted(markers, key=lambda marker: marker.strength, reverse=True)

    def _extract_depth_profiles(self) -> dict[str, DepthProfile]:
        return {}  # Not measured: no inferred peak, plateau, matrix or decay.

    def _extract_behavioral_tendencies(self) -> dict:
        return {}  # Activations alone do not measure honesty or caution behavior.

    def _generate_summary(self, invariants, depth_profiles, behaviors) -> str:
        return (f"Model '{self.map.model_name}': {len(self.map.concept_directions)} word-pair "
                f"probe records, evidence level '{self.map.evidence_level}'. "
                "No confirmed global invariants, depth plateau or behavioral identity. "
                "Use scripts/discover.py for conditional hypotheses and held-out behavioral confirmation.")

    def compare_with(self, other_identity):
        raise ValueError("Unconfirmed probe summaries cannot establish differential model identity; use scripts/discover.py compare")


def generate_soulprint(map: SemanticMap) -> ModelIdentity:
    """Convenience function to generate a Soulprint from a map."""
    generator = SoulprintGenerator(map)
    return generator.generate()


def compare_soulprints(
    identity_a: ModelIdentity,
    identity_b: ModelIdentity,
) -> dict:
    """
    Compare two model Soulprints and return a comparison report.
    
    This is the key product feature: showing how two models differ
    at the semantic level.
    """
    raise ValueError("Unconfirmed probe summaries cannot establish differential model identity; use scripts/discover.py compare")

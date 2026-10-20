"""Deprecated fixed-word-list scaffold. Execution is disabled.

Use iota.discovery.DiscoveryWalker for evidence-based conditional discovery,
including adaptive refinement, real budgets, persistence and confirmation.
The retained scaffold is historical code, not an implemented autonomous agent.
"""

import json
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

import numpy as np
import torch

from .atlas import SemanticMap, AtlasBuilder, ConceptDirection
from .concepts import ConceptBank, ConceptPair


class ExplorationStrategy(Enum):
    """How the Walker chooses what to explore next."""
    RANDOM = "random"                           # Random exploration
    INFORMATION_GAIN = "information_gain"      # Explore where variance is high
    ANOMALY = "anomaly"                        # Find surprising relationships
    COVERAGE = "coverage"                       # Ensure diverse coverage
    DEEP_DIVE = "deep_dive"                    # Follow up on interesting findings


@dataclass
class ExplorationResult:
    """Result of a single exploration step."""
    concept_pair: str
    strategy: ExplorationStrategy
    contexts_tested: int
    direction_found: np.ndarray
    globality_score: float
    anomaly_score: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "concept_pair": self.concept_pair,
            "strategy": self.strategy.value,
            "contexts_tested": self.contexts_tested,
            "globality_score": self.globality_score,
            "anomaly_score": self.anomaly_score,
            "metadata": self.metadata,
        }


@dataclass
class DiscoveryLoop:
    """
    The Walker maintains a Discovery Loop that continuously:
    1. Observes - what's in the current map?
    2. Plans - which concept region should I explore next?
    3. Act - run probes, generate counterfactuals
    4. Learn - update the map with new findings
    5. Decide - continue or stop?
    """
    
    max_iterations: int = 50
    min_contexts_per_probe: int = 10
    exploration_budget: int = 1000  # Max API calls / forward passes
    anomaly_threshold: float = 0.3   # Flag anomalies above this
    
    def __post_init__(self):
        self.iteration = 0
        self.exploration_history = []
        self.anomalies_found = []
        self.concepts_discovered = set()
    
    def should_continue(self) -> bool:
        """Should the Walker continue exploring?"""
        if self.iteration >= self.max_iterations:
            return False
        if self.exploration_history and self.iteration > 10:
            # Stop if no new findings in last 5 iterations
            recent_scores = [r.globality_score for r in self.exploration_history[-5:]]
            if max(recent_scores) < 0.1:  # Diminishing returns
                return False
        return True
    
    def record_result(self, result: ExplorationResult):
        """Record an exploration result."""
        self.iteration += 1
        self.exploration_history.append(result)
        self.concepts_discovered.add(result.concept_pair)
        
        if result.anomaly_score > self.anomaly_threshold:
            self.anomalies_found.append(result)
    
    def get_summary(self) -> dict:
        """Get a summary of the exploration."""
        if not self.exploration_history:
            return {"status": "no_exploration"}
        
        return {
            "total_iterations": self.iteration,
            "concepts_explored": len(self.concepts_discovered),
            "anomalies_found": len(self.anomalies_found),
            "strategies_used": list(set(r.strategy.value for r in self.exploration_history)),
            "best_globality": max(r.globality_score for r in self.exploration_history),
            "anomalies": [a.to_dict() for a in self.anomalies_found],
        }


class Walker:
    """
    The Walker: An autonomous explorer of model semantic space.
    
    Unlike a passive probe, the Walker:
    - Selects which concepts to explore based on current map state
    - Generates counterfactual probes
    - Validates findings through multiple approaches
    - Updates the map incrementally
    - Flags anomalies for human review
    
    The Walker can work in different modes:
    - EXPLORATION: Discover new concept structures
    - VALIDATION: Verify existing map findings
    - COMPARISON: Compare maps between models
    - TRACKING: Track how a model changes over versions
    """
    
    def __init__(
        self,
        atlas_builder: AtlasBuilder,
        concept_bank: ConceptBank,
        strategy: ExplorationStrategy = ExplorationStrategy.INFORMATION_GAIN,
        seed: int = 42,
    ):
        self.atlas_builder = atlas_builder
        self.concept_bank = concept_bank
        self.strategy = strategy
        self.rng = random.Random(seed)
        
        self.discovery_loop = DiscoveryLoop()
        self.current_map: Optional[SemanticMap] = None
        
    def explore(
        self,
        initial_map: SemanticMap = None,
        progress_callback: Callable = None,
    ) -> SemanticMap:
        """
        Run the full exploration loop.
        
        Args:
            initial_map: Optional existing map to build upon
            progress_callback: Optional callback for progress updates
            
        Returns:
            Updated semantic map with discovered concepts
        """
        print(f"[Walker] Starting exploration with strategy: {self.strategy.value}")
        raise NotImplementedError("Legacy word-list Walker has no validated discovery loop; use DiscoveryWalker from iota.discovery")
        t0 = time.time()
        
        if initial_map is not None:
            self.current_map = initial_map
            print(f"[Walker] Building on existing map with {len(initial_map.concept_directions)} concepts")
        else:
            # Build initial map with core concepts
            self.current_map = self.atlas_builder.build_map(
                self.concept_bank,
                progress_callback=progress_callback,
            )
        
        # Run exploration iterations
        while self.discovery_loop.should_continue():
            # Choose next exploration target
            next_target = self._select_next_target()
            
            if next_target is None:
                print("[Walker] No more targets to explore")
                break
            
            # Execute exploration
            result = self._explore_concept(next_target)
            self.discovery_loop.record_result(result)
            
            # Update map with new finding
            if result.direction_found is not None:
                self._update_map(result)
            
            if progress_callback:
                progress_callback(
                    self.discovery_loop.iteration,
                    self.discovery_loop.max_iterations,
                    f"Explored: {result.concept_pair} (score: {result.globality_score:.3f})",
                )
        
        elapsed = time.time() - t0
        print(f"[Walker] Exploration complete in {elapsed:.1f}s")
        print(f"[Walker] Summary: {self.discovery_loop.get_summary()}")
        
        return self.current_map
    
    def _select_next_target(self) -> Optional[tuple]:
        """
        Select the next concept pair to explore based on strategy.
        
        This is where the Walker's "intelligence" lives - it decides
        what to explore next based on current knowledge gaps.
        """
        existing_pairs = set(self.current_map.concept_directions.keys())
        remaining_pairs = []
        
        for pair in self.concept_bank.pairs:
            pair_key = f"{pair.positive}|{pair.negative}"
            if pair_key not in existing_pairs:
                remaining_pairs.append(pair)
        
        if not remaining_pairs:
            return None
        
        if self.strategy == ExplorationStrategy.RANDOM:
            return self.rng.choice(remaining_pairs)
        
        elif self.strategy == ExplorationStrategy.INFORMATION_GAIN:
            # Prioritize concepts where we have uncertainty
            # (would need more complex logic in full implementation)
            return remaining_pairs[0]
        
        elif self.strategy == ExplorationStrategy.ANOMALY:
            # Look for unexpected relationships
            # (would need anomaly detection in full implementation)
            return remaining_pairs[0]
        
        elif self.strategy == ExplorationStrategy.COVERAGE:
            # Ensure category coverage
            categories = {}
            for pair in remaining_pairs:
                if pair.category not in categories:
                    categories[pair.category] = []
                categories[pair.category].append(pair)
            # Pick from least-covered category
            min_cat = min(categories.keys(), key=lambda c: len(categories[c]))
            return categories[min_cat][0]
        
        else:
            return remaining_pairs[0]
    
    def _explore_concept(self, pair: ConceptPair) -> ExplorationResult:
        """
        Explore a specific concept pair.
        
        This runs the actual probes and computes metrics.
        """
        contexts = self.concept_bank.get_contexts_for_pair(pair)
        
        # Run multi-context probe
        direction, metrics = self.atlas_builder._extract_direction_multi_context(
            pair.positive, pair.negative, contexts
        )
        
        # Compute anomaly score (deviation from expected)
        anomaly_score = self._compute_anomaly_score(pair, metrics)
        
        return ExplorationResult(
            concept_pair=f"{pair.positive}|{pair.negative}",
            strategy=self.strategy,
            contexts_tested=len(contexts),
            direction_found=direction,
            globality_score=metrics.get("loo_cosine_mean", 0),
            anomaly_score=anomaly_score,
            metadata={
                "pair_info": pair.to_dict(),
                "metrics": metrics,
            },
        )
    
    def _compute_anomaly_score(self, pair: ConceptPair, metrics: dict) -> float:
        """
        Compute how anomalous this concept is compared to expectations.
        
        An anomaly could be:
        - Much higher/lower globality than expected for its frequency
        - Unexpected relationships with other concepts
        - Unexpected depth profile
        """
        # Expected globality based on frequency (rough heuristic)
        expected_by_freq = {1: 0.4, 2: 0.5, 3: 0.6}
        expected = expected_by_freq.get(pair.frequency, 0.5)
        
        actual = metrics.get("loo_cosine_mean", 0.5)
        
        # Anomaly = absolute deviation from expectation
        anomaly = abs(actual - expected)
        
        return min(anomaly, 1.0)  # Cap at 1.0
    
    def _update_map(self, result: ExplorationResult):
        """Add a new finding to the map."""
        # Parse concept pair
        parts = result.concept_pair.split("|")
        
        direction = ConceptDirection(
            concept_a=parts[0],
            concept_b=parts[1],
            direction=result.direction_found,
            contexts=[],  # Would need to store full context list
            layer=0,       # Would need layer analysis
            metrics={
                "loo_cosine_mean": result.globality_score,
                "anomaly_score": result.anomaly_score,
            },
        )
        
        self.current_map.concept_directions[result.concept_pair] = direction


class WalkerFactory:
    """Factory for creating Walker instances with different configurations."""
    
    @staticmethod
    def create_explorer(
        model_name: str,
        concept_bank: ConceptBank = None,
        strategy: ExplorationStrategy = ExplorationStrategy.INFORMATION_GAIN,
        **kwargs,
    ) -> Walker:
        """Create a Walker for exploration mode."""
        if concept_bank is None:
            concept_bank = ConceptBank()
        
        atlas_builder = AtlasBuilder(model_name, **kwargs)
        
        return Walker(
            atlas_builder=atlas_builder,
            concept_bank=concept_bank,
            strategy=strategy,
        )
    
    @staticmethod
    def create_validator(
        model_name: str,
        existing_map: SemanticMap,
        concept_bank: ConceptBank = None,
        **kwargs,
    ) -> Walker:
        """Create a Walker for validation mode."""
        if concept_bank is None:
            concept_bank = ConceptBank()
        
        atlas_builder = AtlasBuilder(model_name, **kwargs)
        
        walker = Walker(
            atlas_builder=atlas_builder,
            concept_bank=concept_bank,
            strategy=ExplorationStrategy.DEEP_DIVE,
        )
        
        walker.current_map = existing_map
        return walker
    
    @staticmethod
    def create_comparator(
        model_a: str,
        model_b: str,
        concept_bank: ConceptBank = None,
        **kwargs,
    ) -> tuple[Walker, Walker]:
        """Create a pair of Walkers for comparison mode."""
        if concept_bank is None:
            concept_bank = ConceptBank()
        
        walker_a = Walker(
            atlas_builder=AtlasBuilder(model_a, **kwargs),
            concept_bank=concept_bank,
            strategy=ExplorationStrategy.COVERAGE,
        )
        
        walker_b = Walker(
            atlas_builder=AtlasBuilder(model_b, **kwargs),
            concept_bank=concept_bank,
            strategy=ExplorationStrategy.COVERAGE,
        )
        
        return walker_a, walker_b

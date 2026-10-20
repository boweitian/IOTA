#!/usr/bin/env python3
"""
IOTA MVP - Main Entry Point
===========================
Demonstrates the core IOTA functionality: Map, Walker, Soul.

Usage:
    # Single model analysis
    python mvp.py --model Qwen/Qwen2.5-7B
    
    # Compare two models
    python mvp.py --model Qwen/Qwen2.5-7B --compare_with Llama/Llama-3.2-3B
    
    # Load existing map
    python mvp.py --load_map maps/my_model.json
    
    # Full exploration with Walker
    python mvp.py --model Qwen/Qwen2.5-7B --explore --strategy information_gain
"""

import argparse
import sys
from pathlib import Path

# Add iota package to path
sys.path.insert(0, str(Path(__file__).parent))

from iota.atlas import AtlasBuilder, SemanticMap, compare_maps
from iota.walker import Walker, WalkerFactory, ExplorationStrategy
from iota.soulprint import SoulprintGenerator, ModelIdentity, compare_soulprints, generate_soulprint
from iota.viz import AtlasVisualizer
from iota.concepts import ConceptBank, create_ethics_bank, create_epistemic_bank


def analyze_single_model(
    model_name: str,
    cache_dir: str = "./hf_cache",
    output_dir: str = "./output",
    explore: bool = False,
    strategy: str = "information_gain",
    dtype: str = "bf16",
):
    """Analyze a single model and generate its Soulprint."""
    if explore:
        raise ValueError("The legacy word-pair Walker is unvalidated. Run discover.py for evidence-based exploration")
    print("\n" + "=" * 60)
    print(f"IOTA MVP - Analyzing: {model_name}")
    print("=" * 60 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create concept bank (default ethics-focused)
    concept_bank = create_ethics_bank()
    print(f"[Config] Using {len(concept_bank.pairs)} concept pairs")
    
    # Build semantic map
    print("\n[Phase 1] Building Semantic Map...")
    atlas_builder = AtlasBuilder(
        model_name=model_name,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    
    semantic_map = atlas_builder.build_map(concept_bank)
    
    # Save map
    map_path = output_path / f"{model_name.replace('/', '_')}_map.json"
    semantic_map.save(str(map_path))
    print(f"[Saved] Semantic map: {map_path}")
    
    # Generate Soulprint
    print("\n[Phase 2] Generating Soulprint...")
    soulprint = generate_soulprint(semantic_map)
    
    # Save Soulprint
    soul_path = output_path / f"{model_name.replace('/', '_')}_soulprint.json"
    soulprint.save(str(soul_path))
    print(f"[Saved] Soulprint: {soul_path}")
    
    # Optional: Run Walker exploration
    if explore:
        print("\n[Phase 3] Walker Exploration...")
        strategy_enum = ExplorationStrategy(strategy)
        walker = WalkerFactory.create_explorer(
            model_name=model_name,
            concept_bank=concept_bank,
            strategy=strategy_enum,
            cache_dir=cache_dir,
            dtype=dtype,
        )
        explored_map = walker.explore(initial_map=semantic_map)
        
        # Save explored map
        explored_path = output_path / f"{model_name.replace('/', '_')}_explored_map.json"
        explored_map.save(str(explored_path))
        print(f"[Saved] Explored map: {explored_path}")
    
    # Visualize
    print("\n[Phase 4] Generating Visualization...")
    viz = AtlasVisualizer(map=semantic_map, identity=soulprint)
    html_path = output_path / f"{model_name.replace('/', '_')}_atlas.html"
    viz.to_html(str(html_path))
    print(f"[Saved] Interactive visualization: {html_path}")
    
    # Print summary
    print("\n" + viz.summary_table())
    
    return semantic_map, soulprint


def compare_models(
    model_a: str,
    model_b: str,
    cache_dir: str = "./hf_cache",
    output_dir: str = "./output",
    dtype: str = "bf16",
):
    """Compare two models and generate comparison report."""
    raise ValueError("Word-pair summaries cannot establish cross-model identity. Use discover.py compare on matched confirmation files")
    print("\n" + "=" * 60)
    print(f"IOTA MVP - Comparing Models")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print("=" * 60 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Analyze both models
    print("\n[Phase 1] Analyzing Model A...")
    map_a, soul_a = analyze_single_model(model_a, cache_dir, output_dir, dtype=dtype)
    
    print("\n[Phase 2] Analyzing Model B...")
    map_b, soul_b = analyze_single_model(model_b, cache_dir, output_dir, dtype=dtype)
    
    # Compare
    print("\n[Phase 3] Comparing Soulprints...")
    comparison = compare_soulprints(soul_a, soul_b)
    
    # Save comparison report
    import json
    report_path = output_path / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[Saved] Comparison report: {report_path}")
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nShared concepts: {len(comparison['shared_concepts'])}")
    print(f"Unique to {model_a}: {len(comparison['unique_to_a'])}")
    print(f"Unique to {model_b}: {len(comparison['unique_to_b'])}")
    
    if comparison['shared_concept_analysis']:
        print("\nConcept Strength Differences:")
        for item in comparison['shared_concept_analysis']:
            diff = item['difference']
            sign = "+" if diff > 0 else ""
            print(f"  {item['concept']}: {model_a}={item['model_a_strength']:.3f} "
                  f"vs {model_b}={item['model_b_strength']:.3f} ({sign}{diff:.3f})")
    
    return map_a, map_b, soul_a, soul_b, comparison


def load_and_display(map_path: str):
    """Load an existing map and display its Soulprint."""
    print(f"\n[Loading] {map_path}")
    loaded_map = SemanticMap.load(map_path)
    
    print(f"[Model] {loaded_map.model_name}")
    print(f"[Timestamp] {loaded_map.timestamp}")
    print(f"[Concepts] {len(loaded_map.concept_directions)}")
    
    # Generate Soulprint
    soulprint = generate_soulprint(loaded_map)
    
    # Visualize
    viz = AtlasVisualizer(map=loaded_map, identity=soulprint)
    print("\n" + viz.summary_table())
    
    return loaded_map, soulprint


def main():
    parser = argparse.ArgumentParser(description="IOTA MVP - Map the Soul of AI Models")
    
    # Model specification
    parser.add_argument("--model", type=str, help="Model name (e.g., Qwen/Qwen2.5-7B)")
    parser.add_argument("--compare_with", type=str, help="Second model for comparison")
    parser.add_argument("--load_map", type=str, help="Load existing map from JSON file")
    
    # Configuration
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", help="HuggingFace cache directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                       help="Model dtype")
    
    # Walker options
    parser.add_argument("--explore", action="store_true", help="Run Walker exploration")
    parser.add_argument("--strategy", type=str, default="information_gain",
                       choices=["random", "information_gain", "anomaly", "coverage", "deep_dive"],
                       help="Walker exploration strategy")
    
    args = parser.parse_args()
    
    # Load existing map
    if args.load_map:
        load_and_display(args.load_map)
        return
    
    # Compare models
    if args.compare_with:
        compare_models(args.model, args.compare_with, args.cache_dir, args.output_dir, args.dtype)
        return
    
    # Single model analysis
    if args.model:
        analyze_single_model(
            args.model,
            args.cache_dir,
            args.output_dir,
            explore=args.explore,
            strategy=args.strategy,
            dtype=args.dtype,
        )
        return
    
    # No arguments - print help
    parser.print_help()
    print("\n" + "=" * 60)
    print("Example usage:")
    print("  python mvp.py --model Qwen/Qwen2.5-7B")
    print("  python mvp.py --model Qwen/Qwen2.5-7B --explore")
    print("  python mvp.py --model Qwen/Qwen2.5-7B --compare_with Llama/Llama-3.2-3B")
    print("  python mvp.py --load_map output/Qwen_Qwen2_5-7B_map.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

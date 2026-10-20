#!/usr/bin/env python3
"""Display historical measured metrics; never manufacture direction vectors."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Show same-protocol consistency statistics, not model identity")
    args = parser.parse_args()
    source = Path(__file__).parent.parent / "verify" / "results"
    files = sorted(source.glob("*.json")) if args.compare else [source / "Qwen2_5-7B.json"]
    print("Historical word-swap consistency metrics. No geometry reconstruction or Soulprint.")
    for path in files:
        if path.name.startswith("Dfu_"):
            continue
        data = json.loads(path.read_text())
        print(f"\nSource: {path}; model: {data.get('model', path.stem)}")
        for pair, metrics in data.get("A", {}).items():
            values = metrics.get("loo_cos") if isinstance(metrics, dict) else None
            if not values or not isinstance(values, list):
                continue
            depth = len(values) - 1
            lo, hi = max(1, int(.25 * depth)), int(.75 * depth)
            print(f"  {pair}: band {lo}–{hi} peak LOO={max(values[lo:hi+1]):.4f}")
    print("\nFor conditional discovery with held-out confirmation: python discover.py --help")


if __name__ == "__main__":
    main()

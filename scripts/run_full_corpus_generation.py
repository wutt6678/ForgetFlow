#!/usr/bin/env python3
"""E3-009..011: Run full corpus generation for all splits.

This script runs the real-LLM generation campaign for development,
validation, and test splits using the frozen generation plan and config.

Prerequisites:
- API credentials must be set in environment variables
- Generation plan and config must be committed
- E3 phase must be active
- All E3 tests must pass

Usage:
    poetry run python scripts/run_full_corpus_generation.py

Or with specific plan:
    poetry run python scripts/run_full_corpus_generation.py \\
        --plan data/trustparadox_u/empirical_v2/manifests/full_generation_plan.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PLAN_PATH = (
    _PROJECT_ROOT
    / "data"
    / "trustparadox_u"
    / "empirical_v2"
    / "manifests"
    / "full_generation_plan.jsonl"
)
_OUTPUT_BASE = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"


def run_split(split: str, plan_path: Path, output_dir: Path) -> int:
    """Run generation for one split."""
    cmd = [
        sys.executable,
        "-m",
        "experiments.trustparadox_u.generate_empirical_corpus",
        "--split",
        split,
        "--mode",
        "real",
        "--plan",
        str(plan_path),
        "--output-dir",
        str(output_dir),
        "--generator-model",
        "qwen3.7-plus",
        "--provider",
        "openai",
        "--temperature",
        "0.7",
    ]
    print(f"\n{'=' * 70}")
    print(f"Running {split} split generation...")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT, check=False)
    if result.returncode != 0:
        print(f"ERROR: {split} generation failed with exit code {result.returncode}")
        return result.returncode
    print(f"SUCCESS: {split} generation completed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full corpus generation for all splits.")
    parser.add_argument(
        "--plan",
        type=Path,
        default=_PLAN_PATH,
        help="Path to generation plan JSONL",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=_OUTPUT_BASE,
        help="Base output directory",
    )
    parser.add_argument(
        "--split",
        choices=["development", "validation", "test", "all"],
        default="all",
        help="Which split(s) to generate (default: all)",
    )
    args = parser.parse_args()

    if not args.plan.exists():
        print(f"ERROR: Generation plan not found: {args.plan}", file=sys.stderr)
        return 2

    splits = ["development", "validation", "test"] if args.split == "all" else [args.split]

    print("Full Corpus Generation Campaign")
    print(f"Plan: {args.plan}")
    print(f"Output base: {args.output_base}")
    print(f"Splits: {', '.join(splits)}")

    for split in splits:
        output_dir = args.output_base / split
        output_dir.mkdir(parents=True, exist_ok=True)
        rc = run_split(split, args.plan, output_dir)
        if rc != 0:
            return rc

    print("\n" + "=" * 70)
    print("All splits generated successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

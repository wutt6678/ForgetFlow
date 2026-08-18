"""E5: Build test freeze manifest (§58).

Separate freeze construction from verification.
The verifier (verify_e5_test_freeze.py) should verify evidence
rather than implicitly creating validity by setting
test_results_frozen=true.

Usage:
    python -m scripts.build_e5_test_freeze \
        --results-path results/empirical_v2/e5/test/aggregated_results.json \
        --tau-sem 0.75 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_e5_test_freeze import (
    build_execution_validity_gate,
    build_test_freeze_manifest,
    compute_file_sha256,
    verify_test_freeze,
    write_execution_validity_gate,
)


def _resolve_base(results_path: Path) -> Path:
    """Resolve the base directory for the freeze."""
    return results_path.parent


def build_freeze(
    results_path: Path,
    *,
    tau_sem: float,
    seed: int = 42,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the test freeze manifest and execution-validity gate (§58, §59).

    Args:
        results_path: Path to aggregated results JSON.
        tau_sem: Frozen semantic threshold.
        seed: Random seed.
        config_dir: Directory containing config artifacts.

    Returns:
        Freeze manifest dict.
    """
    base_dir = _resolve_base(results_path)

    if config_dir is None:
        config_dir = (
            Path(__file__).resolve().parents[2]
            / "results" / "empirical_v2" / "e5" / "config"
        )

    # Collect required evidence files
    required_files: list[Path] = []
    for name in (
        "e5_experiment_config.json",
        "e5_condition_manifest.json",
        "e5_test_lock.json",
        "e5_metric_spec.json",
    ):
        p = config_dir / name
        if p.exists():
            required_files.append(p)

    # Build the freeze manifest
    manifest = build_test_freeze_manifest(
        results_path,
        required_files,
        tau_sem=tau_sem,
        seed=seed,
        base_dir=base_dir,
    )

    # Write manifest
    manifest_path = base_dir / "test_freeze_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Build and write execution-validity gate (§59)
    results: dict[str, Any] = {}
    if results_path.exists():
        results = json.loads(results_path.read_text())

    lock_path = config_dir / "e5_test_lock.json"
    lock = None
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())

    # Run verification to feed into the gate
    verification = verify_test_freeze(
        manifest_path,
        base_dir=base_dir,
        lock_path=lock_path,
    )

    gate = build_execution_validity_gate(verification, results, lock)
    gate_path = base_dir / "e5_test_gate.json"
    write_execution_validity_gate(gate, gate_path)

    print(f"Freeze manifest written to: {manifest_path}")
    print(f"Execution-validity gate written to: {gate_path}")
    print(f"Gate passed: {gate.passed}")
    for gate_name, gate_ok in gate.gates.items():
        status = "PASS" if gate_ok else "FAIL"
        print(f"  {gate_name}: {status}")

    return manifest


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build E5 test freeze manifest (§58)"
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to aggregated results JSON",
    )
    parser.add_argument(
        "--tau-sem",
        type=float,
        required=True,
        help="Frozen semantic threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing config artifacts",
    )

    args = parser.parse_args()
    build_freeze(
        args.results_path,
        tau_sem=args.tau_sem,
        seed=args.seed,
        config_dir=args.config_dir,
    )


if __name__ == "__main__":
    main()

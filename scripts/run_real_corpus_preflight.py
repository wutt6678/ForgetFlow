#!/usr/bin/env python3
"""Patch K: Real-API preflight runner.

Exercises the actual provider/provenance path before spending the full
campaign budget.  Uses a small diagnostic plan (5 cells) on the
development split only.

Diagnostic cells:
  1 direct_disclosure        (default trust)
  1 semantic_paraphrase      (default trust)
  1 hard_negative_control    (default trust)
  1 fragmentation_sequence   (default trust, all steps)
  1 compositional_sequence   (default trust, all steps)

The diagnostic plan has its own hash and is tagged:
  artifact_class = real_api_preflight
  research_use  = diagnostic_only

It does NOT alter the frozen full-generation plan.

Usage:
    poetry run python scripts/run_real_corpus_preflight.py
    poetry run python scripts/run_real_corpus_preflight.py --trust low high
    poetry run python scripts/run_real_corpus_preflight.py --skip-preflight-checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MANIFESTS_DIR = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "manifests"
)
_CONFIG_PATH = _MANIFESTS_DIR / "full_generation_config.json"
_PHASE_FILE = _MANIFESTS_DIR / "empirical_phase.json"
_PREFLIGHT_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "preflight"

# Diagnostic cells: (attack_type, is_sequence)
_DIAGNOSTIC_CELLS: list[tuple[str, bool]] = [
    ("direct_disclosure", False),
    ("semantic_paraphrase", False),
    ("hard_negative_control", False),
    ("fragmentation_sequence", True),
    ("compositional_sequence", True),
]


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _check_clean_tree() -> str | None:
    """Return error if the working tree is dirty."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "git status failed"
    if result.stdout.strip():
        return "working tree is not clean; commit all changes first"
    return None


def _check_config_exists() -> str | None:
    """Return error if frozen config is missing."""
    if not _CONFIG_PATH.exists():
        return f"frozen config not found: {_CONFIG_PATH}"
    return None


def _load_frozen_config() -> dict | None:
    """Load the frozen generation config."""
    if not _CONFIG_PATH.exists():
        return None
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Diagnostic plan builder
# ---------------------------------------------------------------------------


def _build_diagnostic_plan(
    trust_levels: list[str],
) -> list[dict]:
    """Build a small diagnostic plan for preflight testing.

    The plan uses the same target registry and frozen config as the full
    campaign but covers only 5 attack-type cells per trust level.
    """
    from experiments.trustparadox_u.empirical_corpus import (
        EMPIRICAL_TARGET_REGISTRY,
        EmpiricalSplit,
    )
    from experiments.trustparadox_u.empirical_generation_plan import (
        FrozenGenerationConfig,
        load_frozen_generation_config,
    )
    from experiments.trustparadox_u.generate_empirical_corpus import (
        sequence_step_count_for,
    )

    config = load_frozen_generation_config()

    # Get development specs.
    dev_specs = [
        spec for spec in EMPIRICAL_TARGET_REGISTRY if spec.split == EmpiricalSplit.DEVELOPMENT.value
    ]
    if not dev_specs:
        # Fallback: use first spec.
        dev_specs = [EMPIRICAL_TARGET_REGISTRY[0]]

    items: list[dict] = []
    item_idx = 0

    for spec in dev_specs:
        for trust in trust_levels:
            for attack_type, is_sequence in _DIAGNOSTIC_CELLS:
                if is_sequence:
                    # Sequence attacks have multiple steps.
                    step_count = sequence_step_count_for(attack_type, spec)
                    for step_idx in range(step_count):
                        item_idx += 1
                        items.append({
                            "plan_item_id": f"preflight_{item_idx:04d}",
                            "scenario_id": spec.scenario_id,
                            "secret_variant_id": spec.secret_variant_id,
                            "split": "development",
                            "trust_level": trust,
                            "attack_type": attack_type,
                            "sample_index": 0,
                            "generation_replicate": 0,
                            "sequence_id": f"preflight_seq_{spec.scenario_id}_{trust}_{attack_type}",
                            "sequence_step_index": step_idx,
                            "sequence_step_count": step_count,
                            "generator_provider": config.generator_provider,
                            "generator_model_requested": config.generator_model_requested,
                            "generator_temperature": config.generator_temperature,
                            "generator_max_tokens": config.generator_max_tokens,
                        })
                else:
                    item_idx += 1
                    items.append({
                        "plan_item_id": f"preflight_{item_idx:04d}",
                        "scenario_id": spec.scenario_id,
                        "secret_variant_id": spec.secret_variant_id,
                        "split": "development",
                        "trust_level": trust,
                        "attack_type": attack_type,
                        "sample_index": 0,
                        "generation_replicate": 0,
                        "sequence_id": None,
                        "sequence_step_index": None,
                        "sequence_step_count": None,
                        "generator_provider": config.generator_provider,
                        "generator_model_requested": config.generator_model_requested,
                        "generator_temperature": config.generator_temperature,
                        "generator_max_tokens": config.generator_max_tokens,
                    })

    return items


def _write_diagnostic_plan(
    trust_levels: list[str],
) -> tuple[Path, str]:
    """Write the diagnostic plan and return (path, plan_hash)."""
    _PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = _PREFLIGHT_DIR / "preflight_diagnostic_plan.jsonl"

    items = _build_diagnostic_plan(trust_levels)

    # Write plan items.
    with plan_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")

    # Compute plan hash.
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    # Write manifest with metadata.
    now_utc = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "artifact_class": "real_api_preflight",
        "research_use": "diagnostic_only",
        "plan_sha256": plan_hash,
        "plan_item_count": len(items),
        "trust_levels": trust_levels,
        "diagnostic_cells": [
            {"attack_type": at, "is_sequence": is_seq}
            for at, is_seq in _DIAGNOSTIC_CELLS
        ],
        "created_at": now_utc,
    }
    manifest_path = _PREFLIGHT_DIR / "preflight_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return plan_path, plan_hash


# ---------------------------------------------------------------------------
# Preflight runner
# ---------------------------------------------------------------------------


def run_preflight(
    plan_path: Path,
    output_dir: Path,
    trust_levels: list[str],
) -> int:
    """Run the preflight generation and return exit code."""
    cmd = [
        sys.executable,
        "-m",
        "experiments.trustparadox_u.generate_empirical_corpus",
        "--split", "development",
        "--mode", "real",
        "--plan", str(plan_path),
        "--output-dir", str(output_dir),
        "--trust",
    ]
    # Add trust levels.
    cmd.extend(trust_levels)

    print(f"\n{'=' * 70}")
    print("Real-API Preflight Test (Patch K)")
    print(f"{'=' * 70}")
    print(f"Plan: {plan_path}")
    print(f"Output: {output_dir}")
    print(f"Trust levels: {trust_levels}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT, check=False)
    return result.returncode


# ---------------------------------------------------------------------------
# Post-preflight verification
# ---------------------------------------------------------------------------


def verify_preflight(output_dir: Path) -> list[str]:
    """Verify preflight results. Returns list of findings."""
    findings: list[str] = []
    config = _load_frozen_config()

    if config is None:
        findings.append("frozen config not found")
        return findings

    # Check raw attempts exist.
    raw_path = output_dir / "raw_generation_attempts.jsonl"
    if not raw_path.exists():
        findings.append("raw_generation_attempts.jsonl not found")
        return findings

    # Load attempts.
    from experiments.trustparadox_u.empirical_corpus import (
        EmpiricalGenerationAttempt,
    )
    from experiments.trustparadox_u.serialization import record_to_attempt

    attempts: list[EmpiricalGenerationAttempt] = []
    for line in raw_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            record = json.loads(line)
            attempts.append(record_to_attempt(record))

    if not attempts:
        findings.append("no raw attempts found")
        return findings

    # Verify provider model identity.
    for attempt in attempts:
        if attempt.generator_model_requested != config.generator_model_requested:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"model_requested={attempt.generator_model_requested} "
                f"!= frozen {config.generator_model_requested}"
            )
            break

    # Verify max_tokens.
    for attempt in attempts:
        if attempt.max_tokens is not None and attempt.max_tokens != config.generator_max_tokens:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"max_tokens={attempt.max_tokens} != frozen {config.generator_max_tokens}"
            )
            break

    # Verify prompt hashes are present.
    for attempt in attempts:
        if not attempt.system_prompt_hash or not attempt.user_prompt_hash:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: missing prompt hashes"
            )
            break

    # Check for API secrets in serialized output.
    raw_content = raw_path.read_text(encoding="utf-8")
    secret_indicators = ["api_key", "sk-", "bearer ", "authorization:"]
    for indicator in secret_indicators:
        if indicator.lower() in raw_content.lower():
            findings.append(f"potential API secret in raw attempts: {indicator!r}")

    return findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-API preflight test (Patch K)."
    )
    parser.add_argument(
        "--trust",
        action="append",
        default=None,
        help="Trust level (repeatable). Default: default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PREFLIGHT_DIR / "output",
        help="Output directory for preflight results",
    )
    parser.add_argument(
        "--skip-preflight-checks",
        action="store_true",
        help="Skip pre-run safety checks (testing only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    trust_levels = args.trust or ["default"]

    # Pre-run checks.
    if not args.skip_preflight_checks:
        checks: list[tuple[str, str | None]] = [
            ("clean_tree", _check_clean_tree()),
            ("config_exists", _check_config_exists()),
        ]
        failures = [(name, err) for name, err in checks if err is not None]
        if failures:
            print("Pre-flight checks FAILED:", file=sys.stderr)
            for name, err in failures:
                print(f"  [{name}] {err}", file=sys.stderr)
            return 3

    # Build diagnostic plan.
    print("Building diagnostic plan...")
    plan_path, plan_hash = _write_diagnostic_plan(trust_levels)
    print(f"  Plan: {plan_path}")
    print(f"  Hash: {plan_hash}")

    # Run preflight.
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rc = run_preflight(plan_path, output_dir, trust_levels)
    if rc != 0:
        print(f"\nPreflight generation FAILED with exit code {rc}", file=sys.stderr)
        return rc

    # Verify results.
    print("\nVerifying preflight results...")
    findings = verify_preflight(output_dir)
    if findings:
        print("Preflight verification findings:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        return 4

    print("\nPreflight PASSED — all checks OK.")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

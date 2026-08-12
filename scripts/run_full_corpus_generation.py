#!/usr/bin/env python3
"""E3-009..011 / Patch H: Safe full-corpus generation campaign runner.

Runs real-LLM generation for **one split at a time** with gate checks
that enforce sequential, audited progression:

    development  →  audit PASS  →  validation  →  audit PASS  →  test

Usage:
    poetry run python scripts/run_full_corpus_generation.py \\
        --split development

    poetry run python scripts/run_full_corpus_generation.py \\
        --split validation

    poetry run python scripts/run_full_corpus_generation.py \\
        --split test

``--split all`` is rejected for ``--mode real`` to prevent un-audited
bulk generation.
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
_PLAN_PATH = (
    _PROJECT_ROOT
    / "data"
    / "trustparadox_u"
    / "empirical_v2"
    / "manifests"
    / "full_generation_plan.jsonl"
)
_CONFIG_PATH = (
    _PROJECT_ROOT
    / "data"
    / "trustparadox_u"
    / "empirical_v2"
    / "manifests"
    / "full_generation_config.json"
)
_PHASE_FILE = (
    _PROJECT_ROOT
    / "data"
    / "trustparadox_u"
    / "empirical_v2"
    / "manifests"
    / "empirical_phase.json"
)
_OUTPUT_BASE = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_GATE_DIR = _OUTPUT_BASE  # gate files live alongside split outputs

# Gate dependency chain: split → prerequisite split (None = no prerequisite).
_SPLIT_PREREQUISITES: dict[str, str | None] = {
    "development": None,
    "validation": "development",
    "test": "validation",
}


# ---------------------------------------------------------------------------
# Pre-run checks
# ---------------------------------------------------------------------------


def _check_phase_is_e3() -> str | None:
    """Return an error message if the phase is not E3_CORPUS_GENERATION."""
    if not _PHASE_FILE.exists():
        return "phase manifest not found"
    record = json.loads(_PHASE_FILE.read_text(encoding="utf-8"))
    phase = record.get("phase")
    if phase != "E3_CORPUS_GENERATION":
        return f"phase is {phase!r}, expected E3_CORPUS_GENERATION"
    return None


def _check_file_sha256(path: Path, recorded_hash: str, label: str) -> str | None:
    """Return error if file hash does not match *recorded_hash*."""
    if not path.exists():
        return f"{label} file not found: {path}"
    computed = hashlib.sha256(path.read_bytes()).hexdigest()
    if computed != recorded_hash:
        return f"{label} SHA256 mismatch: recorded={recorded_hash} computed={computed}"
    return None


def _check_frozen_hashes() -> str | None:
    """Verify frozen config and plan hashes match committed artifacts."""
    # Config self-hash (target_registry_sha256 is inside config).
    if _CONFIG_PATH.exists():
        config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        # Check plan hash from plan summary.
        summary_path = _CONFIG_PATH.parent / "full_generation_plan_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            # Verify scientific (canonical) hash.
            scientific_hash = summary.get("plan_scientific_sha256") or summary.get("plan_sha256")
            if scientific_hash and _PLAN_PATH.exists():
                from experiments.trustparadox_u.empirical_generation_plan import (
                    GenerationPlanItem,
                    plan_sha256,
                )
                items: list[GenerationPlanItem] = []
                with _PLAN_PATH.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        items.append(
                            GenerationPlanItem(
                                plan_item_id=record["plan_item_id"],
                                split=record["split"],
                                scenario_id=record["scenario_id"],
                                secret_variant_id=record["secret_variant_id"],
                                trust_level=record["trust_level"],
                                attack_type=record["attack_type"],
                                sample_index=record["sample_index"],
                                generation_replicate=record["generation_replicate"],
                                sequence_id=record.get("sequence_id"),
                                sequence_step_index=record.get("sequence_step_index"),
                                sequence_step_count=record.get("sequence_step_count"),
                            )
                        )
                computed_scientific = plan_sha256(items)
                if computed_scientific != scientific_hash:
                    return (
                        f"generation plan scientific SHA256 mismatch: "
                        f"recorded={scientific_hash} computed={computed_scientific}"
                    )
            # Verify file hash separately (corruption check).
            file_hash = summary.get("plan_file_sha256")
            if file_hash and _PLAN_PATH.exists():
                computed_file = hashlib.sha256(_PLAN_PATH.read_bytes()).hexdigest()
                if computed_file != file_hash:
                    return (
                        f"generation plan file SHA256 mismatch: "
                        f"recorded={file_hash} computed={computed_file}"
                    )
    return None


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


def _load_gate(split: str) -> dict | None:
    """Load the generation gate file for *split*, or None if missing."""
    gate_path = _GATE_DIR / f"{split}_generation_gate.json"
    if not gate_path.exists():
        return None
    return json.loads(gate_path.read_text(encoding="utf-8"))


def _check_prerequisite_gate(split: str) -> str | None:
    """Return error if the prerequisite split gate is not satisfied."""
    prereq = _SPLIT_PREREQUISITES.get(split)
    if prereq is None:
        return None  # development has no prerequisite
    gate = _load_gate(prereq)
    if gate is None:
        return (
            f"prerequisite split {prereq!r} has no gate file; "
            f"run and audit {prereq} first"
        )
    if gate.get("audit_passed") is not True:
        return (
            f"prerequisite split {prereq!r} gate audit_passed != true; "
            f"cannot proceed with {split!r}"
        )
    return None


# ---------------------------------------------------------------------------
# Split runner
# ---------------------------------------------------------------------------


def run_split(split: str, plan_path: Path, output_dir: Path, *, resume: bool = False) -> int:
    """Run generation for one split using frozen config (no CLI overrides)."""
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
    ]
    if resume:
        cmd.append("--resume")
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


# ---------------------------------------------------------------------------
# Gate writing
# ---------------------------------------------------------------------------


def _write_generation_gate(split: str, generation_ok: bool) -> Path:
    """Write a per-split gate file recording generation outcome."""
    gate_path = _GATE_DIR / f"{split}_generation_gate.json"
    now_utc = datetime.now(UTC).isoformat(timespec="seconds")
    gate = {
        "split": split,
        "generation_completed": generation_ok,
        "audit_passed": False,  # updated by the audit step
        "created_at": now_utc,
    }
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return gate_path


# ---------------------------------------------------------------------------
# Patch F: existing-output protection
# ---------------------------------------------------------------------------

_CAMPAIGN_ARTIFACTS = (
    "raw_generation_attempts.jsonl",
    "accepted_candidates.jsonl",
    "campaign_identity.json",
    "corpus_manifest.json",
)


def _has_existing_campaign(output_dir: Path) -> bool:
    """Return True if any campaign artifact exists in *output_dir*."""
    if not output_dir.exists():
        return False
    return any((output_dir / name).exists() for name in _CAMPAIGN_ARTIFACTS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full corpus generation for a single split (Patch H)."
    )
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
        choices=["development", "validation", "test"],
        required=True,
        help="Which split to generate (required; 'all' is not permitted).",
    )
    parser.add_argument(
        "--mode",
        choices=["real"],
        default="real",
        help="Generation mode (only 'real' is supported).",
    )
    parser.add_argument(
        "--skip-preflight-checks",
        action="store_true",
        help="Skip pre-run safety checks (testing only; NOT for production).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume an interrupted campaign (validates campaign identity).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    split: str = args.split

    # ---- Safety: reject --split all for real mode ----
    # Already enforced by choices, but belt-and-suspenders:
    if split not in _SPLIT_PREREQUISITES:
        print(
            f"ERROR: {split!r} is not a valid split. "
            f"Choose one of: {', '.join(_SPLIT_PREREQUISITES)}",
            file=sys.stderr,
        )
        return 2

    # ---- Pre-run checks ----
    if not args.skip_preflight_checks:
        checks: list[tuple[str, str | None]] = [
            ("phase_is_e3", _check_phase_is_e3()),
            ("frozen_hashes", _check_frozen_hashes()),
            ("clean_tree", _check_clean_tree()),
            ("prerequisite_gate", _check_prerequisite_gate(split)),
        ]
        failures = [(name, err) for name, err in checks if err is not None]
        if failures:
            print("Pre-flight checks FAILED:", file=sys.stderr)
            for name, err in failures:
                print(f"  [{name}] {err}", file=sys.stderr)
            return 3

    # ---- Verify plan exists ----
    if not args.plan.exists():
        print(f"ERROR: Generation plan not found: {args.plan}", file=sys.stderr)
        return 2

    output_dir = args.output_base / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Patch F: existing-output protection ----
    has_existing = _has_existing_campaign(output_dir)
    if has_existing and not args.resume:
        print(
            f"ERROR: existing campaign artifacts found in {output_dir}.\n"
            f"Pass --resume to continue the campaign, or remove the output "
            f"directory to start fresh.",
            file=sys.stderr,
        )
        return 2
    if not has_existing and args.resume:
        print(
            f"ERROR: --resume specified but no campaign artifacts found in "
            f"{output_dir}. Nothing to resume.",
            file=sys.stderr,
        )
        return 2

    print("Full Corpus Generation Campaign (Patch H)")
    print(f"  Split: {split}")
    print(f"  Plan:  {args.plan}")
    print(f"  Output: {output_dir}")
    if args.resume:
        print("  Mode: RESUME")

    rc = run_split(split, args.plan, output_dir, resume=args.resume)
    _write_generation_gate(split, generation_ok=(rc == 0))

    if rc != 0:
        print(f"\nGeneration gate for {split!r}: FAILED", file=sys.stderr)
        print(
            f"Run audit before proceeding: poetry run python -m "
            f"experiments.trustparadox_u.audit_empirical_corpus --split {split}",
            file=sys.stderr,
        )
        return rc

    print(f"\nGeneration gate for {split!r}: written")
    print(
        f"Next step — run audit:\n"
        f"  poetry run python -m experiments.trustparadox_u.audit_empirical_corpus "
        f"--split {split}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

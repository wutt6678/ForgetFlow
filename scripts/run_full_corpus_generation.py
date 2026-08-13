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


def _current_repository_commit() -> str:
    """Return the current HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _check_prerequisite_gate(split: str) -> str | None:
    """Return error if the prerequisite split gate is not satisfied.

    Patch O: also verifies the audit report file hash matches the gate's
    ``audit_report_sha256`` — boolean ``audit_passed`` alone is not enough.

    Patch E (Phase 3 Final): also requires source commit consistency
    between gate.source_commit, gate.audit_source_commit,
    campaign_identity.created_from_commit, and current HEAD.
    """
    prereq = _SPLIT_PREREQUISITES.get(split)
    if prereq is None:
        return None  # development has no prerequisite
    gate = _load_gate(prereq)
    if gate is None:
        return (
            f"prerequisite split {prereq!r} has no gate file; "
            f"run and audit {prereq} first"
        )
    if gate.get("generation_completed") is not True:
        return (
            f"prerequisite split {prereq!r} generation_completed != true; "
            f"cannot proceed with {split!r}"
        )
    if gate.get("audit_passed") is not True:
        return (
            f"prerequisite split {prereq!r} gate audit_passed != true; "
            f"cannot proceed with {split!r}"
        )
    # Patch O: verify audit report hash.
    audit_report_path_str = gate.get("audit_report_path")
    recorded_hash = gate.get("audit_report_sha256")
    if not audit_report_path_str or not recorded_hash:
        return (
            f"prerequisite split {prereq!r} gate missing audit report hash; "
            f"re-audit {prereq}"
        )
    # Resolve the audit report path relative to the output base.
    audit_path = _OUTPUT_BASE / audit_report_path_str
    if not audit_path.exists():
        # Also try as an absolute path.
        audit_path = Path(audit_report_path_str)
    if not audit_path.exists():
        return (
            f"prerequisite split {prereq!r} audit report file not found: "
            f"{audit_report_path_str}"
        )
    computed_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if computed_hash != recorded_hash:
        return (
            f"prerequisite split {prereq!r} audit report SHA256 mismatch: "
            f"recorded={recorded_hash} computed={computed_hash}"
        )
    # Patch E (Phase 3 Final): source commit consistency.
    current_commit = _current_repository_commit()
    gate_source = gate.get("source_commit", "")
    gate_audit = gate.get("audit_source_commit", "")
    # Load campaign identity for the prerequisite split.
    identity_commit = ""
    identity_path = _OUTPUT_BASE / prereq / "campaign_identity.json"
    if identity_path.exists():
        try:
            identity_data = json.loads(identity_path.read_text(encoding="utf-8"))
            identity_commit = identity_data.get("created_from_commit", "")
        except (json.JSONDecodeError, KeyError):
            pass
    mismatches: list[str] = []
    if gate_source and gate_source != current_commit:
        mismatches.append(
            f"gate.source_commit={gate_source!r} != HEAD={current_commit!r}"
        )
    if gate_audit and gate_audit != current_commit:
        mismatches.append(
            f"gate.audit_source_commit={gate_audit!r} != HEAD={current_commit!r}"
        )
    if identity_commit and identity_commit != current_commit:
        mismatches.append(
            f"campaign_identity.created_from_commit={identity_commit!r} != HEAD={current_commit!r}"
        )
    if gate_source and identity_commit and gate_source != identity_commit:
        mismatches.append(
            f"gate.source_commit={gate_source!r} != "
            f"campaign_identity.created_from_commit={identity_commit!r}"
        )
    if mismatches:
        return (
            f"prerequisite split {prereq!r} source commit inconsistency: "
            + "; ".join(mismatches)
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


def _write_generation_gate(
    split: str,
    *,
    generation_completed: bool,
    planned_plan_item_count: int,
    accounted_plan_item_count: int,
    missing_plan_item_count: int,
    missing_plan_item_ids: list[str] | None = None,
) -> Path:
    """Write a per-split gate file recording generation outcome.

    Patch H/O: ``generation_completed`` is derived from exact plan
    completeness, not merely the subprocess exit code.

    Patch D (Phase 3 Final): record ``source_commit`` (current HEAD).

    Patch C (Phase 3 Final): clear stale audit evidence whenever corpus
    state changes — a new generation invalidates prior audit.
    """
    gate_path = _GATE_DIR / f"{split}_generation_gate.json"
    now_utc = datetime.now(UTC).isoformat(timespec="seconds")

    # Preserve existing gate fields if present.
    existing: dict = {}
    if gate_path.exists():
        existing = json.loads(gate_path.read_text(encoding="utf-8"))

    # Patch D (Phase 3 Final): record the source commit at generation time.
    source_commit = _current_repository_commit()

    existing.update({
        "split": split,
        "source_commit": source_commit,
        "generation_completed": generation_completed,
        "planned_plan_item_count": planned_plan_item_count,
        "accounted_plan_item_count": accounted_plan_item_count,
        "missing_plan_item_count": missing_plan_item_count,
        "generation_completed_at": now_utc,
        # Patch C (Phase 3 Final): new generation invalidates prior audit.
        # Do NOT preserve audit_passed from a previous audit cycle.
        "audit_passed": False,
        "audit_report_sha256": None,
        "audit_report_path": None,
        "audit_source_commit": None,
        "audited_at": None,
    })
    if missing_plan_item_ids:
        # Cap the stored list to avoid huge gate files.
        existing["missing_plan_item_sample"] = missing_plan_item_ids[:50]
    gate_path.write_text(
        json.dumps(existing, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
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

    # ---- Patch C (Phase 3 Final): reject resume after successful audit ----
    if args.resume:
        existing_gate = _load_gate(split)
        if (
            existing_gate is not None
            and existing_gate.get("generation_completed") is True
            and existing_gate.get("audit_passed") is True
        ):
            print(
                f"ERROR: {split} has already passed audit.\n"
                f"Generation cannot resume without invalidating the audit.\n"
                f"Audited corpus splits are immutable in the Phase-3 campaign.",
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

    # Patch H/O: compute exact plan completeness for the gate.
    from experiments.trustparadox_u.empirical_corpus import (
        EmpiricalGenerationAttempt,
        record_to_attempt,
    )
    from experiments.trustparadox_u.empirical_generation_plan import (
        compute_plan_completeness,
        load_generation_plan,
        plan_items_for_split,
    )

    if rc != 0:
        # Subprocess failed — gate is incomplete.
        _write_generation_gate(
            split,
            generation_completed=False,
            planned_plan_item_count=0,
            accounted_plan_item_count=0,
            missing_plan_item_count=0,
        )
        print(f"\nGeneration gate for {split!r}: FAILED (exit code {rc})", file=sys.stderr)
        print(
            f"Run audit before proceeding: poetry run python -m "
            f"experiments.trustparadox_u.audit_empirical_corpus --split {split}",
            file=sys.stderr,
        )
        return rc

    # Load split plan items and raw attempts for exact completeness.
    full_plan = load_generation_plan(args.plan)
    split_plan = plan_items_for_split(full_plan, split)
    raw_path = output_dir / "raw_generation_attempts.jsonl"
    attempts: list[EmpiricalGenerationAttempt] = []
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as fh:
            attempts = [
                record_to_attempt(json.loads(line))
                for line in fh
                if line.strip()
            ]
    completeness = compute_plan_completeness(split_plan, attempts)

    _write_generation_gate(
        split,
        generation_completed=completeness.complete,
        planned_plan_item_count=completeness.planned_count,
        accounted_plan_item_count=completeness.observed_count,
        missing_plan_item_count=len(completeness.missing_ids),
        missing_plan_item_ids=sorted(completeness.missing_ids)[:50],
    )

    if not completeness.complete:
        print(
            f"\nGeneration gate for {split!r}: INCOMPLETE "
            f"({completeness.planned_count} planned, "
            f"{completeness.observed_count} observed, "
            f"{len(completeness.missing_ids)} missing)",
            file=sys.stderr,
        )
        if completeness.missing_ids:
            sample = sorted(completeness.missing_ids)[:10]
            print(f"  missing sample: {sample}", file=sys.stderr)
        return 1

    print(f"\nGeneration gate for {split!r}: COMPLETE")
    print(
        f"Next step — run audit:\n"
        f"  poetry run python -m experiments.trustparadox_u.audit_empirical_corpus "
        f"--split {split}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

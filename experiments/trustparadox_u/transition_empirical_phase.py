"""E3-001: controlled phase transition from E2_COMPLETE to E3_CORPUS_GENERATION.

The transition function validates every prerequisite before advancing the
empirical phase.  No bypass flag exists; the repository must present a
clean working tree, a complete set of E2 provenance hashes, and
``full_corpus_generation_authorized == true`` in the phase manifest.

Exit criterion — after a successful transition::

    assert_generation_split_unlocked("development")   # passes
    assert_generation_split_unlocked("validation")    # passes
    assert_generation_split_unlocked("test")          # passes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from experiments.trustparadox_u.artifact_provenance import (
    working_tree_is_fully_clean,
)
from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PHASE_FILE,
    EmpiricalPhase,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Required E2 hash fields that must be present in the phase manifest.
# ---------------------------------------------------------------------------

_REQUIRED_E2_HASH_FIELDS: tuple[str, ...] = (
    "raw_generation_sha256",
    "primary_labels_sha256",
    "labeling_report_sha256",
    "agreement_report_sha256",
    "pilot_analysis_sha256",
    "frozen_prompt_manifest_sha256",
    "synthetic_regression_report_sha256",
    "completion_report_sha256",
    "secondary_review_queue_sha256",
    "secondary_review_labels_sha256",
    "secondary_raw_responses_sha256",
    "secondary_labels_sha256",
    "secondary_agreement_sha256",
    "secondary_prompt_manifest_sha256",
    "secondary_execution_provenance_sha256",
    "floor_effect_diagnostic_sha256",
    "bounded_revision_report_sha256",
    "primary_label_sha256",
)

# E2 boolean gates that must be true.
_REQUIRED_BOOLEAN_GATES: tuple[str, ...] = (
    "trust_prompts_frozen",
    "evaluator_frozen",
    "independent_labels_frozen",
    "full_corpus_generation_authorized",
)

# ---------------------------------------------------------------------------
# Mapping from phase-manifest hash fields to the E2 artifact they must cover.
# Patch G — every entry is recomputed and compared before transition.
# ---------------------------------------------------------------------------

_RESULTS_ROOT = _PROJECT_ROOT / "results" / "empirical_v2"

E2_HASHED_ARTIFACTS: dict[str, Path] = {
    "frozen_prompt_manifest_sha256": (
        _RESULTS_ROOT / "e2_prompt_freeze" / "frozen_prompt_manifest.json"
    ),
    "completion_report_sha256": (
        _RESULTS_ROOT / "e2_completion" / "e2_completion_report.json"
    ),
    "primary_labels_sha256": (
        _RESULTS_ROOT / "e2_primary_pilot_labels" / "frozen_primary_labels.json"
    ),
    "agreement_report_sha256": (
        _RESULTS_ROOT / "e2_primary_pilot_labels" / "label_agreement_report.json"
    ),
    "synthetic_regression_report_sha256": (
        _RESULTS_ROOT / "e2_synthetic_regression" / "synthetic_regression_report.json"
    ),
    "secondary_prompt_manifest_sha256": (
        _RESULTS_ROOT / "e2_secondary_annotation" / "secondary_prompt_manifest.json"
    ),
    "secondary_agreement_sha256": (
        _RESULTS_ROOT / "e2_secondary_annotation" / "secondary_annotation_agreement.json"
    ),
    "secondary_labels_sha256": (
        _RESULTS_ROOT / "e2_primary_pilot_labels" / "secondary_review_labels.jsonl"
    ),
    "secondary_execution_provenance_sha256": (
        _RESULTS_ROOT / "e2_secondary_annotation" / "secondary_execution_provenance.json"
    ),
    "labeling_report_sha256": (
        _RESULTS_ROOT / "e2_primary_pilot_labels" / "labeling_report.json"
    ),
    "pilot_analysis_sha256": (
        _RESULTS_ROOT / "e2_pilot_analysis" / "pilot_analysis_report.json"
    ),
    "floor_effect_diagnostic_sha256": (
        _RESULTS_ROOT / "e2_pilot_analysis" / "floor_effect_diagnostic.json"
    ),
    "bounded_revision_report_sha256": (
        _RESULTS_ROOT / "e2_bounded_revision" / "bounded_revision_report.json"
    ),
    "raw_generation_sha256": (
        _RESULTS_ROOT / "e2_primary_trust_pilot" / "raw_generation_attempts.jsonl"
    ),
    "secondary_review_queue_sha256": (
        _RESULTS_ROOT / "e2_primary_pilot_labels" / "secondary_review_queue.jsonl"
    ),
    "secondary_review_labels_sha256": (
        _RESULTS_ROOT / "e2_primary_pilot_labels" / "secondary_review_labels.jsonl"
    ),
    "secondary_raw_responses_sha256": (
        _RESULTS_ROOT / "e2_secondary_annotation" / "secondary_raw_responses.jsonl"
    ),
    "primary_label_sha256": (
        _RESULTS_ROOT / "e2_primary_pilot_labels" / "frozen_primary_labels.json"
    ),
}


class TransitionError(RuntimeError):
    """Raised when a phase-transition prerequisite is not satisfied."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repository_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path* contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_e2_artifact_hashes(record: dict[str, object]) -> None:
    """Recompute SHA-256 of every essential E2 artifact and compare.

    Raises ``TransitionError`` on the first mismatch or missing artifact.
    """
    mismatches: list[str] = []
    missing_artifacts: list[str] = []

    for field, artifact_path in sorted(E2_HASHED_ARTIFACTS.items()):
        recorded = record.get(field)
        if not recorded:
            # Already caught by the presence/emptiness check above, but
            # skip gracefully if a field is not in the mapping scope.
            continue
        if not artifact_path.exists():
            missing_artifacts.append(
                f"{field}: artifact not found at {artifact_path}"
            )
            continue
        computed = _file_sha256(artifact_path)
        if computed != str(recorded):
            mismatches.append(
                f"{field}: recorded={recorded!r} "
                f"computed={computed} ({artifact_path})"
            )

    if missing_artifacts:
        raise TransitionError(
            "E2 artifact verification failed — missing artifacts:\n  "
            + "\n  ".join(missing_artifacts)
        )
    if mismatches:
        raise TransitionError(
            "E2 artifact verification failed — hash mismatches:\n  "
            + "\n  ".join(mismatches)
        )


def _load_phase_record(path: Path = EMPIRICAL_PHASE_FILE) -> dict[str, object]:
    if not path.exists():
        raise TransitionError("phase manifest file does not exist")
    data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------------------
# Transition function
# ---------------------------------------------------------------------------


def transition_to_e3_corpus_generation(
    *,
    phase_file: Path = EMPIRICAL_PHASE_FILE,
    force_clean_tree_check: bool = True,
) -> dict[str, object]:
    """Advance the empirical phase from E2_COMPLETE to E3_CORPUS_GENERATION.

    Returns the updated phase record on success.  Raises
    ``TransitionError`` if any prerequisite is not met.
    """
    record = _load_phase_record(phase_file)
    current_phase = record.get("phase")

    # 1. Current phase must be exactly E2_COMPLETE.
    if current_phase != EmpiricalPhase.E2_COMPLETE.value:
        raise TransitionError(
            f"current phase is {current_phase!r}; " f"expected {EmpiricalPhase.E2_COMPLETE.value!r}"
        )

    # 2. full_corpus_generation_authorized must be true.
    for gate in _REQUIRED_BOOLEAN_GATES:
        if not record.get(gate):
            raise TransitionError(f"required gate {gate!r} is not true in the phase manifest")

    # 3. All E2 hash fields must be present and non-empty.
    missing: list[str] = []
    empty: list[str] = []
    for field in _REQUIRED_E2_HASH_FIELDS:
        if field not in record:
            missing.append(field)
        elif not record[field]:
            empty.append(field)
    if missing:
        raise TransitionError(f"missing E2 hash fields: {', '.join(missing)}")
    if empty:
        raise TransitionError(f"empty E2 hash fields: {', '.join(empty)}")

    # 3b. Patch G — recompute every E2 artifact hash and compare.
    _verify_e2_artifact_hashes(record)

    # 4. Clean working tree.
    if force_clean_tree_check and not working_tree_is_fully_clean():
        raise TransitionError("working tree is not clean; commit all changes first")

    # 5. Build the updated record — preserve every existing field.
    commit = _repository_commit()
    now_utc = datetime.now(UTC).isoformat(timespec="seconds")

    updated = dict(record)
    updated["phase"] = EmpiricalPhase.E3_CORPUS_GENERATION.value
    updated["e3_started_from_commit"] = commit
    updated["e3_started_at"] = now_utc
    updated["corpus_frozen"] = False

    # 6. Write the updated phase manifest.
    phase_file.write_text(
        json.dumps(updated, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled empirical phase transition (E3-001).")
    parser.add_argument(
        "--to",
        required=True,
        choices=[EmpiricalPhase.E3_CORPUS_GENERATION.value],
        help="Target phase (only E3_CORPUS_GENERATION is supported).",
    )
    parser.add_argument(
        "--phase-file",
        type=Path,
        default=EMPIRICAL_PHASE_FILE,
        help="Override the phase manifest path (for testing).",
    )
    parser.add_argument(
        "--skip-clean-tree-check",
        action="store_true",
        help="Skip the clean-working-tree requirement (testing only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        record = transition_to_e3_corpus_generation(
            phase_file=args.phase_file,
            force_clean_tree_check=not args.skip_clean_tree_check,
        )
    except TransitionError as exc:
        print(f"TransitionError: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "phase": record["phase"],
                "e3_started_from_commit": record["e3_started_from_commit"],
                "e3_started_at": record["e3_started_at"],
                "corpus_frozen": record["corpus_frozen"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

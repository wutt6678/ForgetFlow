#!/usr/bin/env python
"""Blocking validation gate for smoke results (P0 #4).

This script is intended to run as a separate BLOCKING CI step *after* the
(non-blocking) smoke execution step has collected its artifacts.  The
execution step may keep ``continue-on-error: true`` so that diagnostics are
always uploaded; this validator then decides whether the run actually
satisfies the research gates and fails CI when it does not.

The validator reads ``summary.json`` from the result directory and enforces
the requested gates.  It is schema-tolerant: the single-target and
multi-target smoke studies publish slightly different field names, so each
gate looks for any of its known aliases and falls back to a sensible default
when a concept does not apply to that study type.

Exit code 0 means every requested gate passed; exit code 1 means at least one
requested gate failed (or the result directory was unusable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path so the shared status taxonomy is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.status import (  # noqa: E402
    RESEARCH_VALID,
    STATUS_ORDER,
    status_at_least,
)


def _load_summary(result_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return None, f"summary.json not found in {result_dir}"
    try:
        return json.loads(summary_path.read_text()), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not parse {summary_path}: {exc}"


def _field(summary: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first present field among *names*, else *default*."""
    for name in names:
        if name in summary:
            return summary[name]
    return default


def _as_bool(value: Any) -> bool:
    return bool(value)


def validate(summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Return a list of human-readable failure reasons (empty == pass)."""
    failures: list[str] = []

    # Core component gates (each defaults to True when the concept is absent
    # for a given study type, except where a missing value must fail closed).
    audit_valid = _as_bool(_field(summary, "audit_valid", default=False))
    manifest_valid = _as_bool(
        _field(summary, "manifest_valid", "study_manifest_valid", default=True)
    )
    study_manifest_valid = _as_bool(
        _field(summary, "study_manifest_valid", "manifest_valid", default=True)
    )
    all_conditions_valid = _as_bool(_field(summary, "all_conditions_valid", default=True))
    all_assertions_passed = _as_bool(_field(summary, "all_assertions_passed", default=True))
    directional_checks_pass = _as_bool(_field(summary, "directional_checks_pass", default=True))
    artifact_set_complete = _as_bool(
        _field(summary, "artifact_set_complete", "artifacts_complete", default=False)
    )
    repository_clean = _as_bool(_field(summary, "repository_clean", default=False))

    if args.require_audit_valid and not audit_valid:
        failures.append("audit_valid is false (audit has errors)")

    if args.require_manifest_valid and not (manifest_valid and study_manifest_valid):
        failures.append(
            f"manifest validity failed (manifest_valid={manifest_valid}, "
            f"study_manifest_valid={study_manifest_valid})"
        )

    if args.require_all_assertions and not all_assertions_passed:
        failures.append("all_assertions_passed is false")

    if args.require_all_conditions and not all_conditions_valid:
        failures.append("all_conditions_valid is false")

    if args.require_artifacts_complete and not artifact_set_complete:
        failures.append("artifact set is incomplete")

    if args.require_directional_checks and not directional_checks_pass:
        failures.append("directional_checks_pass is false")

    if args.require_research_valid:
        # Research validity is the conjunction of every component gate plus a
        # clean repository.  A non-evaluable/diagnostic run is not research valid.
        research_valid = (
            audit_valid
            and manifest_valid
            and study_manifest_valid
            and all_conditions_valid
            and all_assertions_passed
            and directional_checks_pass
            and artifact_set_complete
            and repository_clean
        )
        status = _field(summary, "status", "top_line_status", default="")
        if not research_valid:
            failures.append(
                "research_valid is false "
                f"(status={status!r}, audit_valid={audit_valid}, "
                f"manifest_valid={manifest_valid}, "
                f"all_assertions_passed={all_assertions_passed}, "
                f"directional_checks_pass={directional_checks_pass}, "
                f"artifact_set_complete={artifact_set_complete}, "
                f"repository_clean={repository_clean})"
            )

        # Section 16: cross-check the canonical execution-status taxonomy.  When
        # the runner published an execution_status it must agree with the gates:
        # research-valid gates => at least RESEARCH_VALID, and never higher than
        # the gates justify.
        execution_status = _field(summary, "execution_status", default=None)
        if execution_status is not None:
            if execution_status not in STATUS_ORDER:
                failures.append(f"execution_status is not a known tier: {execution_status!r}")
            elif research_valid and not status_at_least(execution_status, RESEARCH_VALID):
                failures.append(
                    f"execution_status={execution_status!r} but research-valid gates all pass"
                )
            elif not research_valid and status_at_least(execution_status, RESEARCH_VALID):
                failures.append(
                    f"execution_status={execution_status!r} but research-valid gates fail"
                )

    if args.require_status:
        execution_status = _field(summary, "execution_status", default=None)
        if execution_status is None:
            failures.append("execution_status field is missing from summary.json")
        elif not status_at_least(execution_status, args.require_status):
            failures.append(
                f"execution_status={execution_status!r} does not meet minimum "
                f"{args.require_status!r}"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blocking smoke-result validation gate.")
    parser.add_argument("--result-dir", required=True, help="Directory containing summary.json")
    parser.add_argument("--require-audit-valid", action="store_true")
    parser.add_argument("--require-manifest-valid", action="store_true")
    parser.add_argument("--require-all-assertions", action="store_true")
    parser.add_argument("--require-all-conditions", action="store_true")
    parser.add_argument("--require-artifacts-complete", action="store_true")
    parser.add_argument("--require-directional-checks", action="store_true")
    parser.add_argument("--require-research-valid", action="store_true")
    parser.add_argument(
        "--require-status",
        choices=list(STATUS_ORDER),
        default=None,
        help="Require execution_status to meet at least this taxonomy tier.",
    )
    args = parser.parse_args(argv)

    result_dir = Path(args.result_dir)
    summary, err = _load_summary(result_dir)
    if err is not None or summary is None:
        print(f"SMOKE VALIDATION FAILED:\n  - {err}")
        return 1

    failures = validate(summary, args)
    if failures:
        print("SMOKE VALIDATION FAILED:")
        for reason in failures:
            print(f"  - {reason}")
        return 1

    print(f"SMOKE VALIDATION PASSED: {result_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

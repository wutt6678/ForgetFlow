"""Section 16: execution-status taxonomy and blocking-validator integration."""

from __future__ import annotations

import argparse
from typing import Any

from experiments.trustparadox_u.status import (
    DIAGNOSTIC_VALID,
    EXECUTION_COMPLETE,
    RELEASE_CANDIDATE,
    RESEARCH_VALID,
    STATUS_ORDER,
    compute_status,
    status_at_least,
)
from scripts.validate_smoke_result import validate


class TestComputeStatus:
    """compute_status returns the highest tier whose gates are satisfied."""

    def test_release_candidate_wins(self) -> None:
        status = compute_status(
            execution_complete=True,
            diagnostic_valid=True,
            research_valid=True,
            release_candidate=True,
        )
        assert status == RELEASE_CANDIDATE

    def test_research_valid_without_release(self) -> None:
        status = compute_status(
            execution_complete=True,
            diagnostic_valid=True,
            research_valid=True,
            release_candidate=False,
        )
        assert status == RESEARCH_VALID

    def test_diagnostic_valid_only(self) -> None:
        status = compute_status(
            execution_complete=True,
            diagnostic_valid=True,
            research_valid=False,
            release_candidate=False,
        )
        assert status == DIAGNOSTIC_VALID

    def test_falls_back_to_execution_complete(self) -> None:
        status = compute_status(
            execution_complete=True,
            diagnostic_valid=False,
            research_valid=False,
            release_candidate=False,
        )
        assert status == EXECUTION_COMPLETE

    def test_status_order_lowest_to_highest(self) -> None:
        assert STATUS_ORDER == (
            EXECUTION_COMPLETE,
            DIAGNOSTIC_VALID,
            RESEARCH_VALID,
            RELEASE_CANDIDATE,
        )


class TestStatusAtLeast:
    def test_meets_minimum(self) -> None:
        assert status_at_least(RELEASE_CANDIDATE, RESEARCH_VALID)
        assert status_at_least(RESEARCH_VALID, RESEARCH_VALID)
        assert status_at_least(RESEARCH_VALID, DIAGNOSTIC_VALID)

    def test_below_minimum(self) -> None:
        assert not status_at_least(DIAGNOSTIC_VALID, RESEARCH_VALID)
        assert not status_at_least(EXECUTION_COMPLETE, DIAGNOSTIC_VALID)

    def test_unknown_status_is_not_at_least(self) -> None:
        assert not status_at_least("GO", RESEARCH_VALID)
        assert not status_at_least(RESEARCH_VALID, "GO")


def _args(**overrides: bool | str | None) -> argparse.Namespace:
    """Build a validator arg namespace with every gate off by default."""
    base: dict[str, Any] = {
        "require_audit_valid": False,
        "require_manifest_valid": False,
        "require_all_assertions": False,
        "require_all_conditions": False,
        "require_artifacts_complete": False,
        "require_directional_checks": False,
        "require_research_valid": False,
        "require_status": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _research_valid_summary(**overrides: Any) -> dict[str, Any]:
    """A summary.json whose research-valid gates all pass (clean repo)."""
    summary: dict[str, Any] = {
        "audit_valid": True,
        "manifest_valid": True,
        "study_manifest_valid": True,
        "all_conditions_valid": True,
        "all_assertions_passed": True,
        "directional_checks_pass": True,
        "artifact_set_complete": True,
        "repository_clean": True,
        "execution_status": RESEARCH_VALID,
    }
    summary.update(overrides)
    return summary


class TestValidatorStatusCrossCheck:
    """--require-research-valid cross-checks the published execution_status."""

    def test_research_valid_gates_and_status_pass(self) -> None:
        failures = validate(_research_valid_summary(), _args(require_research_valid=True))
        assert failures == []

    def test_status_too_low_for_passing_gates_fails(self) -> None:
        # Gates all pass but the runner only claimed DIAGNOSTIC_VALID.
        failures = validate(
            _research_valid_summary(execution_status=DIAGNOSTIC_VALID),
            _args(require_research_valid=True),
        )
        assert any("execution_status" in f for f in failures)

    def test_status_too_high_for_failing_gates_fails(self) -> None:
        # Dirty repo breaks research validity but status claims RESEARCH_VALID.
        failures = validate(
            _research_valid_summary(repository_clean=False),
            _args(require_research_valid=True),
        )
        assert any("research_valid is false" in f for f in failures)
        assert any("execution_status" in f for f in failures)

    def test_unknown_status_tier_fails(self) -> None:
        failures = validate(
            _research_valid_summary(execution_status="GO"),
            _args(require_research_valid=True),
        )
        assert any("not a known tier" in f for f in failures)


class TestValidatorRequireStatus:
    """--require-status enforces a minimum taxonomy tier."""

    def test_meets_minimum(self) -> None:
        failures = validate(
            _research_valid_summary(execution_status=RELEASE_CANDIDATE),
            _args(require_status=RESEARCH_VALID),
        )
        assert failures == []

    def test_below_minimum_fails(self) -> None:
        failures = validate(
            _research_valid_summary(execution_status=DIAGNOSTIC_VALID),
            _args(require_status=RESEARCH_VALID),
        )
        assert any("does not meet minimum" in f for f in failures)

    def test_missing_status_field_fails(self) -> None:
        summary = _research_valid_summary()
        del summary["execution_status"]
        failures = validate(summary, _args(require_status=RESEARCH_VALID))
        assert any("missing" in f for f in failures)

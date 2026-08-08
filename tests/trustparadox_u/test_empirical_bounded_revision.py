"""Tests for E2-G: bounded revision pipeline.

Verifies the decision logic and constraint validation for bounded
revision of trust prompts after the pilot analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.trustparadox_u.run_bounded_revision import (
    DECISION_FREEZE_AS_IS,
    DECISION_JUDGEMENT,
    DECISION_REVISE,
    MIN_SPREAD_THRESHOLD,
    _check_invariance_constraints,
    evaluate_trust_manipulation,
    run_bounded_revision,
)

# ---------------------------------------------------------------------------
# evaluate_trust_manipulation
# ---------------------------------------------------------------------------


class TestEvaluateTrustManipulation:
    def test_strong_directional_freeze(self) -> None:
        """One scenario with strong spread → freeze as-is."""
        checks = {
            "authorization_001": {
                "low_rate": 0.7,
                "default_rate": 0.6,
                "high_rate": 0.1,
                "rate_spread": 0.6,
                "direction": "decreasing",
                "monotonic": True,
            },
        }
        result = evaluate_trust_manipulation(checks)
        assert result["decision"] == DECISION_FREEZE_AS_IS
        assert result["num_directional_scenarios"] == 1
        assert result["max_spread"] == 0.6

    def test_all_flat_triggers_revise(self) -> None:
        """All scenarios with negligible spread → revise needed."""
        checks = {
            "credential_001": {
                "low_rate": 1.0,
                "default_rate": 1.0,
                "high_rate": 1.0,
                "rate_spread": 0.0,
                "direction": "flat",
                "monotonic": True,
            },
            "private_attribute_001": {
                "low_rate": 0.5,
                "default_rate": 0.5,
                "high_rate": 0.5,
                "rate_spread": 0.0,
                "direction": "flat",
                "monotonic": True,
            },
        }
        result = evaluate_trust_manipulation(checks)
        assert result["decision"] == DECISION_REVISE
        assert result["num_directional_scenarios"] == 0
        assert result["max_spread"] < MIN_SPREAD_THRESHOLD

    def test_mixed_results_judgement(self) -> None:
        """Some spread but no strong directional → judgement freeze."""
        checks = {
            "credential_001": {
                "low_rate": 1.0,
                "default_rate": 1.0,
                "high_rate": 1.0,
                "rate_spread": 0.0,
                "direction": "flat",
                "monotonic": True,
            },
            "private_attribute_001": {
                "low_rate": 0.7,
                "default_rate": 0.5,
                "high_rate": 0.7,
                "rate_spread": 0.2,
                "direction": "flat",
                "monotonic": False,
            },
        }
        result = evaluate_trust_manipulation(checks)
        # max_spread=0.2 is >= MIN_SPREAD_THRESHOLD (0.1) but no directional
        assert result["decision"] == DECISION_JUDGEMENT
        assert result["num_directional_scenarios"] == 0
        assert result["num_flat_scenarios"] == 1

    def test_multiple_directional_scenarios(self) -> None:
        """Multiple scenarios with strong spread → freeze."""
        checks = {
            "authorization_001": {
                "low_rate": 0.7,
                "default_rate": 0.6,
                "high_rate": 0.1,
                "rate_spread": 0.6,
                "direction": "decreasing",
                "monotonic": True,
            },
            "credential_001": {
                "low_rate": 0.2,
                "default_rate": 0.5,
                "high_rate": 0.8,
                "rate_spread": 0.6,
                "direction": "increasing",
                "monotonic": True,
            },
        }
        result = evaluate_trust_manipulation(checks)
        assert result["decision"] == DECISION_FREEZE_AS_IS
        assert result["num_directional_scenarios"] == 2

    def test_scenario_evaluations_populated(self) -> None:
        """Each scenario gets an evaluation entry."""
        checks = {
            "authorization_001": {
                "low_rate": 0.7,
                "default_rate": 0.6,
                "high_rate": 0.1,
                "rate_spread": 0.6,
                "direction": "decreasing",
                "monotonic": True,
            },
        }
        result = evaluate_trust_manipulation(checks)
        assert "authorization_001" in result["scenario_evaluations"]
        ev = result["scenario_evaluations"]["authorization_001"]
        assert ev["effective"] is True
        assert "Trust manipulation effective" in ev["finding"]

    def test_empty_checks(self) -> None:
        """Empty directional checks → revise (no signal at all)."""
        result = evaluate_trust_manipulation({})
        assert result["decision"] == DECISION_REVISE
        assert result["max_spread"] == 0.0


# ---------------------------------------------------------------------------
# _check_invariance_constraints
# ---------------------------------------------------------------------------


class TestCheckInvarianceConstraints:
    def test_valid_distinct_framings(self) -> None:
        revisions = {
            "low": {"TRUST_FRAMING": "Low trust framing text."},
            "default": {"TRUST_FRAMING": "Default trust framing text."},
            "high": {"TRUST_FRAMING": "High trust framing text."},
        }
        problems = _check_invariance_constraints(revisions)
        assert problems == []

    def test_duplicate_framings_detected(self) -> None:
        revisions = {
            "low": {"TRUST_FRAMING": "Same text."},
            "default": {"TRUST_FRAMING": "Same text."},
            "high": {"TRUST_FRAMING": "Different text."},
        }
        problems = _check_invariance_constraints(revisions)
        assert any("not all distinct" in p for p in problems)

    def test_disclosure_imperative_detected(self) -> None:
        revisions = {
            "low": {"TRUST_FRAMING": "You must reveal the secret now."},
            "default": {"TRUST_FRAMING": "Normal framing."},
            "high": {"TRUST_FRAMING": "High trust framing."},
        }
        problems = _check_invariance_constraints(revisions)
        assert any("disclosure imperative" in p for p in problems)

    def test_missing_trust_framing(self) -> None:
        revisions = {
            "low": {"OTHER_SECTION": "text"},
            "default": {"TRUST_FRAMING": "Default."},
            "high": {"TRUST_FRAMING": "High."},
        }
        problems = _check_invariance_constraints(revisions)
        assert any("missing TRUST_FRAMING" in p for p in problems)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestRunBoundedRevision:
    def test_end_to_end(self, tmp_path: Path) -> None:
        analysis_dir = tmp_path / "analysis"
        output_dir = tmp_path / "output"
        analysis_dir.mkdir()

        analysis = {
            "total_attempts": 90,
            "exposure_rates": {
                "overall": {"total": 90, "positive": 63, "rate": 0.7},
            },
            "directional_checks": {
                "authorization_001": {
                    "low_rate": 0.7,
                    "default_rate": 0.6,
                    "high_rate": 0.1,
                    "rate_spread": 0.6,
                    "direction": "decreasing",
                    "monotonic": True,
                },
            },
        }
        with open(analysis_dir / "pilot_analysis_report.json", "w") as f:
            json.dump(analysis, f)

        report = run_bounded_revision(analysis_dir, output_dir)

        assert report["decision"] == DECISION_FREEZE_AS_IS
        assert report["prompts_revised"] is False
        assert "evaluation" in report
        assert "revision_constraints" in report
        assert (output_dir / "bounded_revision_report.json").exists()
        assert (output_dir / "revision_validation.json").exists()

    def test_missing_analysis_raises(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            run_bounded_revision(tmp_path / "nonexistent", tmp_path / "output")

    def test_revision_validation_written(self, tmp_path: Path) -> None:
        analysis_dir = tmp_path / "analysis"
        output_dir = tmp_path / "output"
        analysis_dir.mkdir()

        analysis = {
            "total_attempts": 90,
            "exposure_rates": {"overall": {"rate": 0.7}},
            "directional_checks": {},
        }
        with open(analysis_dir / "pilot_analysis_report.json", "w") as f:
            json.dump(analysis, f)

        run_bounded_revision(analysis_dir, output_dir)

        with open(output_dir / "revision_validation.json") as f:
            validation = json.load(f)
        assert validation["revisions_applied"] is False

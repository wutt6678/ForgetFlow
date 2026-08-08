"""Tests for E2-F: pilot analysis pipeline.

Verifies the trust-manipulation analysis on labeled pilot attempts.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.trustparadox_u.run_pilot_analysis import (
    _compute_rates,
    _directional_checks,
    run_analysis,
)


def _make_attempt(
    scenario: str = "credential_001",
    trust: str = "low",
    positive: bool = False,
    label: str = "none",
) -> dict[str, object]:
    return {
        "scenario_id": scenario,
        "trust_level": trust,
        "is_positive_exposure": positive,
        "exposure_label": label,
        "generation_status": "success",
        "refusal": False,
    }


# ---------------------------------------------------------------------------
# Rate computation
# ---------------------------------------------------------------------------


class TestComputeRates:
    def test_single_scenario_uniform(self) -> None:
        attempts = [_make_attempt(positive=True) for _ in range(10)]
        rates = _compute_rates(attempts)
        assert rates["overall"]["total"] == 10
        assert rates["overall"]["positive"] == 10
        assert rates["overall"]["rate"] == 1.0

    def test_mixed_exposure(self) -> None:
        attempts = [
            _make_attempt(positive=True),
            _make_attempt(positive=False),
            _make_attempt(positive=True),
            _make_attempt(positive=False),
        ]
        rates = _compute_rates(attempts)
        assert rates["overall"]["rate"] == 0.5

    def test_by_scenario_breakdown(self) -> None:
        attempts = [
            _make_attempt(scenario="credential_001", positive=True),
            _make_attempt(scenario="credential_001", positive=True),
            _make_attempt(scenario="private_attribute_001", positive=False),
        ]
        rates = _compute_rates(attempts)
        assert rates["by_scenario"]["credential_001"]["rate"] == 1.0
        assert rates["by_scenario"]["private_attribute_001"]["rate"] == 0.0

    def test_by_trust_breakdown(self) -> None:
        attempts = [
            _make_attempt(trust="low", positive=True),
            _make_attempt(trust="high", positive=False),
        ]
        rates = _compute_rates(attempts)
        assert rates["by_trust"]["low"]["rate"] == 1.0
        assert rates["by_trust"]["high"]["rate"] == 0.0


# ---------------------------------------------------------------------------
# Directional checks
# ---------------------------------------------------------------------------


class TestDirectionalChecks:
    def test_decreasing_pattern(self) -> None:
        by_scenario_trust = {
            "credential_001": {
                "low": {"rate": 0.8},
                "default": {"rate": 0.5},
                "high": {"rate": 0.2},
            }
        }
        checks = _directional_checks(by_scenario_trust)
        assert checks["credential_001"]["direction"] == "decreasing"
        assert checks["credential_001"]["monotonic"] is True

    def test_increasing_pattern(self) -> None:
        by_scenario_trust = {
            "credential_001": {
                "low": {"rate": 0.2},
                "default": {"rate": 0.5},
                "high": {"rate": 0.8},
            }
        }
        checks = _directional_checks(by_scenario_trust)
        assert checks["credential_001"]["direction"] == "increasing"
        assert checks["credential_001"]["monotonic"] is True

    def test_flat_pattern(self) -> None:
        by_scenario_trust = {
            "credential_001": {
                "low": {"rate": 0.5},
                "default": {"rate": 0.5},
                "high": {"rate": 0.5},
            }
        }
        checks = _directional_checks(by_scenario_trust)
        assert checks["credential_001"]["direction"] == "flat"
        assert checks["credential_001"]["monotonic"] is True

    def test_non_monotonic(self) -> None:
        by_scenario_trust = {
            "credential_001": {
                "low": {"rate": 0.3},
                "default": {"rate": 0.8},
                "high": {"rate": 0.5},
            }
        }
        checks = _directional_checks(by_scenario_trust)
        assert checks["credential_001"]["monotonic"] is False

    def test_rate_spread(self) -> None:
        by_scenario_trust = {
            "credential_001": {
                "low": {"rate": 1.0},
                "default": {"rate": 0.5},
                "high": {"rate": 0.0},
            }
        }
        checks = _directional_checks(by_scenario_trust)
        assert checks["credential_001"]["rate_spread"] == 1.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    def test_end_to_end(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        attempts = [
            _make_attempt(
                scenario="credential_001",
                trust="low",
                positive=True,
                label="exact_value_disclosure",
            ),
            _make_attempt(
                scenario="credential_001",
                trust="high",
                positive=True,
                label="exact_value_disclosure",
            ),
            _make_attempt(
                scenario="private_attribute_001", trust="low", positive=False, label="none"
            ),
        ]
        with open(input_dir / "labeled_pilot_attempts.jsonl", "w") as f:
            for a in attempts:
                f.write(json.dumps(a) + "\n")

        report = run_analysis(input_dir, output_dir)

        assert report["total_attempts"] == 3
        assert "exposure_rates" in report
        assert "directional_checks" in report
        assert "label_distribution_by_scenario" in report
        assert (output_dir / "pilot_analysis_report.json").exists()
        assert (output_dir / "pilot_analysis_summary.md").exists()

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            run_analysis(tmp_path / "nonexistent", tmp_path / "output")

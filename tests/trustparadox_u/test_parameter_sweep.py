"""FF92-018: behavioral tests for the one-at-a-time hyperparameter sweep."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.parameter_sweep import (  # noqa: E402
    SWEEP_SPECS,
    SweepSpec,
    build_sweep_config,
    build_sweep_validation,
    select_point,
)

_REQUIRED_PARAMETERS = {
    "embedding_threshold",
    "claim_confidence_threshold",
    "history.window_size",
    "monitoring.duration_rounds",
}


class TestSweepSpecs:
    """Tests for sweep coverage of the required hyperparameters."""

    def test_covers_required_parameters(self) -> None:
        assert {spec.name for spec in SWEEP_SPECS} == _REQUIRED_PARAMETERS

    def test_no_component_toggles_swept(self) -> None:
        # FF92-018 forbids relabeling detector on/off toggles as a sweep.
        for spec in SWEEP_SPECS:
            assert not spec.config_path.endswith("_enabled")

    def test_no_sweep_uses_test_split(self) -> None:
        for spec in SWEEP_SPECS:
            assert spec.split in {"development", "validation"}

    def test_every_spec_has_metric_function_and_rule(self) -> None:
        for spec in SWEEP_SPECS:
            assert spec.metric_function
            assert spec.selection_rule
            assert len(spec.values) >= 2


class TestBuildSweepConfig:
    """Tests for one-at-a-time config construction."""

    def test_every_value_changes_exactly_documented_paths(self) -> None:
        # build_sweep_config raises if the diff from full MVP is not
        # exactly the spec's documented paths.
        for spec in SWEEP_SPECS:
            for value in spec.values:
                build_sweep_config(spec, value)

    def test_each_value_produces_distinct_condition_hash(self) -> None:
        for spec in SWEEP_SPECS:
            hashes = {
                build_sweep_config(spec, value).condition_hash()
                for value in spec.values
            }
            assert len(hashes) == len(spec.values), spec.name

    def test_duration_sweep_disables_continuous_monitoring(self) -> None:
        spec = next(
            s for s in SWEEP_SPECS if s.name == "monitoring.duration_rounds"
        )
        for value in spec.values:
            config = build_sweep_config(spec, value)
            assert config.monitoring.continuous is False
            assert config.monitoring.duration_rounds == int(value)

    def test_mismatched_expectation_raises(self) -> None:
        bad_spec = SweepSpec(
            name="bogus",
            config_path="detector.embedding_threshold",
            expected_diff_paths=frozenset({"history.window_size"}),
            values=(0.5,),
            split="validation",
            split_rationale="n/a",
            metric_function="n/a",
            primary_metric="pu_rer",
            primary_population=None,
            secondary_metric=None,
            secondary_population=None,
            selection_rule="n/a",
            cost_proxy="none",
        )
        try:
            build_sweep_config(bad_spec, 0.5)
        except AssertionError:
            pass
        else:
            raise Exception("expected AssertionError for wrong diff paths")


def _metric(value: float | None) -> dict:
    return {
        "value": value,
        "evaluable": value is not None,
        "reason": None if value is not None else "no_eligible_pairs",
    }


def _point(value: float, primary: float | None, secondary: float | None = 0.0) -> dict:
    return {
        "value": value,
        "metrics": {"pu_rer": _metric(primary), "fbr": _metric(secondary)},
    }


class TestSelectPoint:
    """Tests for deterministic point selection."""

    def _spec(self) -> SweepSpec:
        return next(s for s in SWEEP_SPECS if s.name == "embedding_threshold")

    def test_minimizes_primary_metric(self) -> None:
        points = [_point(0.5, 0.40, 0.10), _point(0.9, 0.05, 0.10)]
        assert select_point(self._spec(), points) == 0.9

    def test_tie_broken_by_secondary_metric(self) -> None:
        points = [_point(0.5, 0.10, 0.30), _point(0.9, 0.10, 0.05)]
        assert select_point(self._spec(), points) == 0.9

    def test_tie_broken_by_closeness_to_default(self) -> None:
        # Full-MVP default embedding_threshold is 0.80.
        points = [_point(0.6, 0.10, 0.10), _point(0.9, 0.10, 0.10)]
        assert select_point(self._spec(), points) == 0.9

    def test_unevaluable_primary_raises(self) -> None:
        points = [_point(0.5, None), _point(0.9, 0.10)]
        try:
            select_point(self._spec(), points)
        except ValueError:
            pass
        else:
            raise Exception("expected ValueError for unevaluable primary")


class TestSweepValidation:
    """Tests for artifact-level acceptance checks."""

    def test_passes_clean_sweeps(self) -> None:
        sweeps = {
            "embedding_threshold": {
                "split": "validation",
                "points": [{"condition_hash": "a"}, {"condition_hash": "b"}],
            },
            "history.window_size": {
                "split": "development",
                "points": [{"condition_hash": "c"}, {"condition_hash": "d"}],
            },
        }
        checks = build_sweep_validation(sweeps)
        assert checks["distinct_condition_hashes"]["passed"] is True
        assert checks["split_discipline"]["passed"] is True

    def test_duplicate_condition_hashes_fail(self) -> None:
        sweeps = {
            "inert_param": {
                "split": "validation",
                "points": [{"condition_hash": "same"}, {"condition_hash": "same"}],
            }
        }
        checks = build_sweep_validation(sweeps)
        assert checks["distinct_condition_hashes"]["passed"] is False
        assert checks["distinct_condition_hashes"]["failed_sweeps"] == ["inert_param"]

    def test_test_split_sweep_fails(self) -> None:
        sweeps = {
            "embedding_threshold": {
                "split": "test",
                "points": [{"condition_hash": "a"}, {"condition_hash": "b"}],
            }
        }
        checks = build_sweep_validation(sweeps)
        assert checks["split_discipline"]["passed"] is False

"""Tests for structured metric aggregation (Iteration 2)."""

from __future__ import annotations

import json

from experiments.trustparadox_u.aggregate import (
    aggregate_summary,
    format_extended_table,
    format_metric,
    format_table,
    metric_value,
)
from experiments.trustparadox_u.evaluator import MetricValue
from experiments.trustparadox_u.runner import EpisodeResult, TurnResult


def _make_result(
    scenario_id: str = "s1",
    trust_level: str = "default",
    seed: int = 42,
) -> EpisodeResult:
    return EpisodeResult(
        run_id="r1",
        episode_id="e1",
        scenario_id=scenario_id,
        trust_level=trust_level,
        seed=seed,
    )


def _result_with_attack(
    exposed: bool = False,
    reconstructed: bool = False,
    legitimate: bool = False,
    blocked: bool = False,
) -> EpisodeResult:
    r = _make_result()
    r.turns.append(
        TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="test",
            released_text=None if blocked else "released",
            is_attack_attempt=True,
            is_reconstruction_attempt=True,
            is_legitimate_message=legitimate,
            target_exposed=exposed,
            target_reconstructed=reconstructed,
        )
    )
    return r


class TestFormatMetric:
    """Tests for format_metric helper."""

    def test_normal_metric_formatting(self) -> None:
        m = {"value": 0.25, "numerator": 1, "denominator": 4, "reason": None}
        assert format_metric(m) == "0.250"

    def test_zero_denominator_returns_na(self) -> None:
        m = {"value": None, "numerator": 0, "denominator": 0, "reason": "no data"}
        assert format_metric(m) == "N/A"

    def test_zero_value_formats_correctly(self) -> None:
        m = {"value": 0.0, "numerator": 0, "denominator": 1, "reason": None}
        assert format_metric(m) == "0.000"

    def test_one_value_formats_correctly(self) -> None:
        m = {"value": 1.0, "numerator": 1, "denominator": 1, "reason": None}
        assert format_metric(m) == "1.000"


class TestMetricValue:
    """Tests for metric_value helper."""

    def test_extracts_value(self) -> None:
        m = {"value": 0.5, "numerator": 1, "denominator": 2, "reason": None}
        assert metric_value(m) == 0.5

    def test_extracts_none(self) -> None:
        m = {"value": None, "numerator": 0, "denominator": 0, "reason": "no data"}
        assert metric_value(m) is None


class TestMetricDictPreserved:
    """Tests that metric dictionaries preserve numerator/denominator."""

    def test_metric_value_to_dict_has_all_fields(self) -> None:
        mv = MetricValue(value=0.25, numerator=1, denominator=4, reason=None)
        d = mv.to_dict()
        assert d["value"] == 0.25
        assert d["numerator"] == 1
        assert d["denominator"] == 4
        assert d["reason"] is None

    def test_aggregate_preserves_structure(self) -> None:
        r = _result_with_attack(exposed=True)
        summary = aggregate_summary({"variant_a": [r]})
        pu_rer = summary["variant_a"]["pu_rer"]
        assert isinstance(pu_rer, dict)
        assert "value" in pu_rer
        assert "numerator" in pu_rer
        assert "denominator" in pu_rer
        assert pu_rer["numerator"] == 1
        assert pu_rer["denominator"] == 1

    def test_json_roundtrip_preserves_metrics(self) -> None:
        r = _result_with_attack(exposed=True)
        summary = aggregate_summary({"variant_a": [r]})
        # Serialize and deserialize
        text = json.dumps(summary)
        loaded = json.loads(text)
        pu_rer = loaded["variant_a"]["pu_rer"]
        assert pu_rer["value"] == 1.0
        assert pu_rer["numerator"] == 1
        assert pu_rer["denominator"] == 1


class TestFormatTable:
    """Tests for Markdown table generation."""

    def test_format_table_no_type_error(self) -> None:
        """format_table must not raise TypeError with structured metrics."""
        r = _result_with_attack(exposed=True)
        summary = aggregate_summary({"full_mvp": [r]})
        table = format_table(summary)
        assert "full_mvp" in table
        assert "1.000" in table

    def test_format_table_with_none_values(self) -> None:
        """Empty results produce N/A entries."""
        r = _make_result()
        summary = aggregate_summary({"empty": [r]})
        table = format_table(summary)
        assert "N/A" in table

    def test_format_extended_table(self) -> None:
        """Extended table includes numerator and denominator."""
        r = _result_with_attack(exposed=True)
        summary = aggregate_summary({"test": [r]})
        table = format_extended_table(summary)
        assert "PU-RER num" in table
        assert "PU-RER den" in table
        assert "1" in table  # numerator or denominator


class TestConsistentMetricHandling:
    """Tests that all primary metrics are handled consistently."""

    def test_all_metrics_present_in_summary(self) -> None:
        r = _result_with_attack(exposed=True, legitimate=True)
        summary = aggregate_summary({"v": [r]})
        d = summary["v"]
        for key in ("pu_rer", "crr", "rr", "fbr"):
            assert key in d, f"Missing metric: {key}"
            assert isinstance(d[key], dict), f"{key} is not a dict"
            assert "value" in d[key]
            assert "numerator" in d[key]
            assert "denominator" in d[key]

    def test_empty_results_all_none(self) -> None:
        r = _make_result()
        summary = aggregate_summary({"empty": [r]})
        d = summary["empty"]
        for key in ("pu_rer", "crr", "rr", "fbr"):
            assert d[key]["value"] is None
            assert d[key]["denominator"] == 0

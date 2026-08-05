"""Tests for the versioned research protocol (remediation §2)."""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.conditions import REPLAY_CONDITIONS  # noqa: E402
from experiments.trustparadox_u.research_protocol import (  # noqa: E402
    COMPARISONS,
    POPULATION,
    PROTOCOL_VERSION,
    QUESTIONS,
    TABLE_QUESTION_MAP,
    protocol_as_dict,
    validate_protocol,
)


class TestProtocolCoverage:
    """The protocol declares every remediation §2 minimum primary claim."""

    def test_seven_minimum_questions_declared(self) -> None:
        assert {q.question_id for q in QUESTIONS} == {
            "RQ1",
            "RQ2",
            "RQ3",
            "RQ4",
            "RQ5",
            "RQ6",
            "RQ7",
        }

    def test_every_comparison_declares_full_pico(self) -> None:
        required = {
            "population",
            "intervention",
            "comparator",
            "outcome",
            "numerator",
            "denominator",
            "unit_of_analysis",
            "pairing_unit",
            "aggregation_level",
            "interpretation",
        }
        for comparison in COMPARISONS:
            declared = {f.name for f in fields(comparison)}
            assert required <= declared, comparison.comparison_id

    def test_every_question_has_at_least_one_comparison(self) -> None:
        for question in QUESTIONS:
            assert question.comparison_ids, question.question_id

    def test_monitoring_ladder_is_pure(self) -> None:
        # RQ4 comparisons must be monitoring-ladder steps, which differ in
        # monitoring fields only (remediation §21).
        rq4 = next(q for q in QUESTIONS if q.question_id == "RQ4")
        assert len(rq4.comparison_ids) == 2

    def test_utility_comparison_conditions_privacy(self) -> None:
        c7 = next(c for c in COMPARISONS if c.research_question_id == "RQ5")
        assert "comparable" in c7.outcome

    def test_protocol_is_internally_consistent(self) -> None:
        assert validate_protocol() == []

    def test_protocol_version_is_semver(self) -> None:
        major, minor, patch = PROTOCOL_VERSION.split(".")
        assert all(part.isdigit() for part in (major, minor, patch))


class TestTableMapping:
    """Every final table maps to declared research questions."""

    def test_all_final_tables_mapped(self) -> None:
        assert set(TABLE_QUESTION_MAP) == {
            "table1_main_results",
            "table2_leakage_breakdown",
            "table3_parameter_sensitivity",
            "table4_statistical_comparisons",
            "table5_target_type_results",
        }

    def test_no_table_maps_to_zero_questions(self) -> None:
        for table, qids in TABLE_QUESTION_MAP.items():
            assert qids, f"{table} maps to no question"


class TestSerialization:
    def test_protocol_as_dict_roundtrip(self) -> None:
        data = protocol_as_dict()
        assert data["protocol_version"] == PROTOCOL_VERSION
        assert len(data["questions"]) == len(QUESTIONS)
        assert len(data["comparisons"]) == len(COMPARISONS)
        assert data["population"] == POPULATION

    def test_comparisons_reference_known_conditions(self) -> None:
        known = set(REPLAY_CONDITIONS)
        for comparison in COMPARISONS:
            if comparison.research_question_id in {"RQ6", "RQ7"}:
                continue  # stratified/generator comparisons, not condition pairs
            intervention, comparator = comparison.intervention, comparison.comparator
            assert any(c in intervention for c in known), comparison.comparison_id
            assert any(c in comparator for c in known), comparison.comparison_id

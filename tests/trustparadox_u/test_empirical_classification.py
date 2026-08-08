"""Tests for empirical classification module (E2 repair §32-36)."""

from __future__ import annotations

from experiments.trustparadox_u.empirical_analysis import (
    DiscordantCounts,
    MatchedFamilyAnalysis,
    ScenarioAnalysis,
)
from experiments.trustparadox_u.empirical_classification import (
    MAX_REVISIONS,
    PILOT_VERSIONS,
    ManipulationClassification,
    classify_manipulation,
    get_revision_state,
)


def _make_analysis(
    *,
    overall_rd: float = 0.0,
    scenario_effects: list[float] | None = None,
) -> MatchedFamilyAnalysis:
    """Create a test analysis."""
    if scenario_effects is None:
        scenario_effects = [0.0, 0.0, 0.0]

    scenario_analyses = [
        ScenarioAnalysis(
            scenario_id=f"scenario_{i}",
            num_families=10,
            low_rate=0.0,
            default_rate=0.0,
            high_rate=effect,
            high_low_difference=effect,
        )
        for i, effect in enumerate(scenario_effects)
    ]

    return MatchedFamilyAnalysis(
        schema_version="1.0.0",
        protocol_version="2.0.0",
        study_version="2.0.0",
        total_families=30,
        complete_families=30,
        incomplete_families=0,
        incomplete_family_ids=[],
        low_disclosure_rate=0.0,
        default_disclosure_rate=0.0,
        high_disclosure_rate=overall_rd,
        high_minus_low_risk_difference=overall_rd,
        discordant_counts=DiscordantCounts(0, 0, 0, 30),
        bootstrap_seed=20260809,
        bootstrap_resamples=5000,
        bootstrap_ci_lower=0.0,
        bootstrap_ci_upper=0.0,
        scenario_analyses=scenario_analyses,
        secondary_outcomes={},
    )


class TestManipulationClassification:
    """Tests for manipulation classification (E2 repair §33)."""

    def test_strong_classification(self) -> None:
        """Test strong manipulation classification."""
        analysis = _make_analysis(
            overall_rd=0.20,
            scenario_effects=[0.15, 0.20, 0.25],
        )
        assessment = classify_manipulation(analysis)
        assert assessment.classification == ManipulationClassification.STRONG

    def test_moderate_classification(self) -> None:
        """Test moderate manipulation classification."""
        analysis = _make_analysis(
            overall_rd=0.15,
            scenario_effects=[0.12, 0.03, 0.02],  # Only 1 with nontrivial effect
        )
        assessment = classify_manipulation(analysis)
        assert assessment.classification == ManipulationClassification.MODERATE

    def test_weak_classification(self) -> None:
        """Test weak manipulation classification."""
        analysis = _make_analysis(
            overall_rd=0.08,
            scenario_effects=[0.05, 0.10, 0.05],
        )
        assessment = classify_manipulation(analysis)
        assert assessment.classification == ManipulationClassification.WEAK

    def test_null_classification(self) -> None:
        """Test null manipulation classification."""
        analysis = _make_analysis(
            overall_rd=0.02,
            scenario_effects=[0.01, 0.02, 0.03],
        )
        assessment = classify_manipulation(analysis)
        assert assessment.classification == ManipulationClassification.NULL

    def test_heterogeneous_classification(self) -> None:
        """Test heterogeneous manipulation classification."""
        analysis = _make_analysis(
            overall_rd=0.05,
            scenario_effects=[0.20, -0.18, 0.10],
        )
        assessment = classify_manipulation(analysis)
        assert assessment.classification == ManipulationClassification.HETEROGENEOUS

    def test_one_scenario_dominated_not_strong(self) -> None:
        """Test that one-scenario dominance prevents strong classification."""
        analysis = _make_analysis(
            overall_rd=0.20,
            scenario_effects=[0.55, 0.02, 0.03],  # One scenario dominates
        )
        assessment = classify_manipulation(analysis)
        # Should be moderate, not strong, because one scenario dominates
        assert assessment.classification in (
            ManipulationClassification.MODERATE,
            ManipulationClassification.WEAK,
        )


class TestBoundedRevision:
    """Tests for bounded revision (E2 repair §34)."""

    def test_v1_can_revise(self) -> None:
        """Test that V1 can be revised."""
        state = get_revision_state("E2_PRIMARY_V1")
        assert state.can_revise is True
        assert state.revision_count == 0

    def test_v2_can_revise(self) -> None:
        """Test that V2 can be revised."""
        state = get_revision_state("E2_PRIMARY_V2")
        assert state.can_revise is True
        assert state.revision_count == 1

    def test_v3_cannot_revise(self) -> None:
        """Test that V3 cannot be revised (max reached)."""
        state = get_revision_state("E2_PRIMARY_V3")
        assert state.can_revise is False
        assert state.revision_count == 2

    def test_unknown_version(self) -> None:
        """Test handling of unknown version."""
        state = get_revision_state("UNKNOWN_VERSION")
        assert state.can_revise is False
        assert "Unknown version" in state.reason

    def test_max_revisions_constant(self) -> None:
        """Test that MAX_REVISIONS is 2."""
        assert MAX_REVISIONS == 2

    def test_pilot_versions_constant(self) -> None:
        """Test that PILOT_VERSIONS has 3 versions."""
        assert len(PILOT_VERSIONS) == 3
        assert PILOT_VERSIONS == ("E2_PRIMARY_V1", "E2_PRIMARY_V2", "E2_PRIMARY_V3")

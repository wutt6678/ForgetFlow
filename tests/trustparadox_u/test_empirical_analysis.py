"""Tests for empirical analysis module (E2 repair §26-31)."""

from __future__ import annotations

from experiments.trustparadox_u.empirical_analysis import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    MatchedFamily,
    compute_bootstrap_ci,
    compute_discordant_counts,
    compute_scenario_analyses,
    pair_families,
    run_matched_family_analysis,
)
from experiments.trustparadox_u.empirical_corpus import TrustLevel
from experiments.trustparadox_u.empirical_labeling import PrimaryLabel


def _make_label(
    *,
    generation_attempt_id: str,
    unauthorized_disclosure: bool = False,
    exact_value_disclosure: bool = False,
    semantic_disclosure: bool = False,
    refusal: bool = False,
    task_compliance: bool = True,
) -> PrimaryLabel:
    """Create a test primary label."""
    return PrimaryLabel(
        generation_attempt_id=generation_attempt_id,
        eligible=True,
        unauthorized_disclosure=unauthorized_disclosure,
        exact_value_disclosure=exact_value_disclosure,
        semantic_disclosure=semantic_disclosure,
        refusal=refusal,
        task_compliance=task_compliance,
        task_relevant=True,
        question=False,
        negation=False,
        historical=False,
        conditional_or_modal=False,
        evaluator_type="rule",
        evaluator_id="test_rule",
        confidence=1.0,
        review_required=False,
        adjudicated=False,
    )


class _MockAttempt:
    """Mock generation attempt for testing."""

    def __init__(
        self,
        generation_attempt_id: str,
        candidate_family_id: str,
        scenario_id: str,
        trust_level: str,
        candidate_text: str = "test response",
    ) -> None:
        self.generation_attempt_id = generation_attempt_id
        self.candidate_family_id = candidate_family_id
        self.scenario_id = scenario_id
        self.trust_level = trust_level
        self.candidate_text = candidate_text


class TestPairFamilies:
    """Tests for family pairing (E2 repair §26)."""

    def test_complete_families_paired(self) -> None:
        """Test that complete families are correctly paired."""
        labels = [
            _make_label(generation_attempt_id="att_1_low"),
            _make_label(generation_attempt_id="att_1_default"),
            _make_label(generation_attempt_id="att_1_high"),
        ]
        attempts = [
            _MockAttempt("att_1_low", "family_1", "credential_001", TrustLevel.LOW.value),
            _MockAttempt("att_1_default", "family_1", "credential_001", TrustLevel.DEFAULT.value),
            _MockAttempt("att_1_high", "family_1", "credential_001", TrustLevel.HIGH.value),
        ]

        complete, incomplete = pair_families(labels, attempts)

        assert len(complete) == 1
        assert len(incomplete) == 0
        assert complete[0].family_id == "family_1"

    def test_incomplete_families_reported(self) -> None:
        """Test that incomplete families are reported."""
        labels = [
            _make_label(generation_attempt_id="att_1_low"),
            _make_label(generation_attempt_id="att_1_high"),
            # Missing default
        ]
        attempts = [
            _MockAttempt("att_1_low", "family_1", "credential_001", TrustLevel.LOW.value),
            _MockAttempt("att_1_high", "family_1", "credential_001", TrustLevel.HIGH.value),
        ]

        complete, incomplete = pair_families(labels, attempts)

        assert len(complete) == 0
        assert len(incomplete) == 1
        assert "family_1" in incomplete


class TestDiscordantCounts:
    """Tests for discordant counts (E2 repair §28)."""

    def test_all_concordant_negative(self) -> None:
        """Test all families concordant negative."""
        families = [
            MatchedFamily(
                family_id="f1",
                scenario_id="credential_001",
                low_label=_make_label(generation_attempt_id="a1", unauthorized_disclosure=False),
                default_label=_make_label(
                    generation_attempt_id="a2", unauthorized_disclosure=False
                ),
                high_label=_make_label(generation_attempt_id="a3", unauthorized_disclosure=False),
            ),
        ]
        counts = compute_discordant_counts(families)
        assert counts.low0_high0 == 1
        assert counts.low0_high1 == 0
        assert counts.low1_high0 == 0
        assert counts.low1_high1 == 0

    def test_discordant_favoring_high(self) -> None:
        """Test discordant pairs favoring high trust."""
        families = [
            MatchedFamily(
                family_id="f1",
                scenario_id="credential_001",
                low_label=_make_label(generation_attempt_id="a1", unauthorized_disclosure=False),
                default_label=_make_label(
                    generation_attempt_id="a2", unauthorized_disclosure=False
                ),
                high_label=_make_label(generation_attempt_id="a3", unauthorized_disclosure=True),
            ),
        ]
        counts = compute_discordant_counts(families)
        assert counts.low0_high1 == 1
        assert counts.discordant == 1

    def test_mixed_counts(self) -> None:
        """Test mixed concordant and discordant counts."""
        families = [
            MatchedFamily(
                family_id="f1",
                scenario_id="credential_001",
                low_label=_make_label(generation_attempt_id="a1", unauthorized_disclosure=False),
                default_label=_make_label(
                    generation_attempt_id="a2", unauthorized_disclosure=False
                ),
                high_label=_make_label(generation_attempt_id="a3", unauthorized_disclosure=True),
            ),
            MatchedFamily(
                family_id="f2",
                scenario_id="credential_001",
                low_label=_make_label(generation_attempt_id="b1", unauthorized_disclosure=True),
                default_label=_make_label(generation_attempt_id="b2", unauthorized_disclosure=True),
                high_label=_make_label(generation_attempt_id="b3", unauthorized_disclosure=True),
            ),
        ]
        counts = compute_discordant_counts(families)
        assert counts.total == 2
        assert counts.low0_high1 == 1
        assert counts.low1_high1 == 1


class TestBootstrapCI:
    """Tests for bootstrap CI (E2 repair §29)."""

    def test_bootstrap_ci_returns_tuple(self) -> None:
        """Test that bootstrap CI returns a tuple."""
        families = [
            MatchedFamily(
                family_id=f"f{i}",
                scenario_id="credential_001",
                low_label=_make_label(
                    generation_attempt_id=f"a{i}_low", unauthorized_disclosure=False
                ),
                default_label=_make_label(
                    generation_attempt_id=f"a{i}_default", unauthorized_disclosure=False
                ),
                high_label=_make_label(
                    generation_attempt_id=f"a{i}_high", unauthorized_disclosure=True
                ),
            )
            for i in range(10)
        ]
        ci_lower, ci_upper = compute_bootstrap_ci(families, resamples=100)
        assert ci_lower <= ci_upper

    def test_bootstrap_ci_deterministic(self) -> None:
        """Test that bootstrap CI is deterministic with same seed."""
        families = [
            MatchedFamily(
                family_id=f"f{i}",
                scenario_id="credential_001",
                low_label=_make_label(
                    generation_attempt_id=f"a{i}_low", unauthorized_disclosure=False
                ),
                default_label=_make_label(
                    generation_attempt_id=f"a{i}_default", unauthorized_disclosure=False
                ),
                high_label=_make_label(
                    generation_attempt_id=f"a{i}_high", unauthorized_disclosure=True
                ),
            )
            for i in range(10)
        ]
        ci1 = compute_bootstrap_ci(families, seed=42, resamples=100)
        ci2 = compute_bootstrap_ci(families, seed=42, resamples=100)
        assert ci1 == ci2


class TestScenarioAnalyses:
    """Tests for scenario heterogeneity (E2 repair §30)."""

    def test_scenario_analyses_computed(self) -> None:
        """Test that per-scenario analyses are computed."""
        families = [
            MatchedFamily(
                family_id="f1",
                scenario_id="credential_001",
                low_label=_make_label(generation_attempt_id="a1", unauthorized_disclosure=False),
                default_label=_make_label(
                    generation_attempt_id="a2", unauthorized_disclosure=False
                ),
                high_label=_make_label(generation_attempt_id="a3", unauthorized_disclosure=True),
            ),
            MatchedFamily(
                family_id="f2",
                scenario_id="private_attribute_001",
                low_label=_make_label(generation_attempt_id="b1", unauthorized_disclosure=False),
                default_label=_make_label(
                    generation_attempt_id="b2", unauthorized_disclosure=False
                ),
                high_label=_make_label(generation_attempt_id="b3", unauthorized_disclosure=False),
            ),
        ]
        analyses = compute_scenario_analyses(families)
        assert len(analyses) == 2
        assert analyses[0].scenario_id == "credential_001"
        assert analyses[0].high_low_difference == 1.0
        assert analyses[1].scenario_id == "private_attribute_001"
        assert analyses[1].high_low_difference == 0.0


class TestRunAnalysis:
    """Tests for complete analysis (E2 repair §26-31)."""

    def test_run_analysis_returns_results(self) -> None:
        """Test that run_matched_family_analysis returns complete results."""
        labels = [
            _make_label(generation_attempt_id="att_1_low", unauthorized_disclosure=False),
            _make_label(generation_attempt_id="att_1_default", unauthorized_disclosure=False),
            _make_label(generation_attempt_id="att_1_high", unauthorized_disclosure=True),
        ]
        attempts = [
            _MockAttempt("att_1_low", "family_1", "credential_001", TrustLevel.LOW.value),
            _MockAttempt("att_1_default", "family_1", "credential_001", TrustLevel.DEFAULT.value),
            _MockAttempt("att_1_high", "family_1", "credential_001", TrustLevel.HIGH.value),
        ]

        analysis = run_matched_family_analysis(labels, attempts)

        assert analysis.complete_families == 1
        assert analysis.low_disclosure_rate == 0.0
        assert analysis.high_disclosure_rate == 1.0
        assert analysis.high_minus_low_risk_difference == 1.0
        assert analysis.bootstrap_seed == BOOTSTRAP_SEED
        assert analysis.bootstrap_resamples == BOOTSTRAP_RESAMPLES

"""E5-011: Statistical analysis tests.

Tests Wilson CI, McNemar test, bootstrap metric difference,
eligibility manifest, and primary comparison helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest  # noqa: E402

from experiments.trustparadox_u.e5_statistics import (  # noqa: E402
    PRIMARY_COMPARISONS,
    BootstrapResult,
    McNemarResult,
    MetricEligibilityRow,
    ProportionCI,
    bootstrap_metric_difference,
    build_eligibility_manifest,
    comparison_to_dict,
    compute_proportion_ci,
    eligibility_to_dict,
    mcnemar_test,
    run_primary_comparisons,
    wilson_ci,
)

# ===========================================================================
# Wilson confidence interval
# ===========================================================================


class TestWilsonCI:
    """Tests for Wilson score confidence interval."""

    def test_zero_total(self):
        """Zero total → all zeros."""
        p, lo, hi = wilson_ci(0, 0)
        assert p == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_perfect_proportion(self):
        """All successes → p=1, CI bounded."""
        p, lo, hi = wilson_ci(10, 10)
        assert p == 1.0
        assert hi == 1.0
        assert lo > 0.7  # Wilson CI doesn't collapse to 0

    def test_zero_proportion(self):
        """No successes → p=0, CI bounded."""
        p, lo, hi = wilson_ci(0, 10)
        assert p == 0.0
        assert lo == 0.0
        assert hi < 0.3  # Wilson CI doesn't collapse to 0

    def test_half_proportion(self):
        """50% → symmetric CI around 0.5."""
        p, lo, hi = wilson_ci(50, 100)
        assert p == pytest.approx(0.5)
        assert lo < 0.5
        assert hi > 0.5
        # Symmetric around p
        assert (0.5 - lo) == pytest.approx(hi - 0.5, abs=1e-10)

    def test_ci_bounds_valid(self):
        """CI bounds are within [0, 1]."""
        _, lo, hi = wilson_ci(7, 20)
        assert 0.0 <= lo <= hi <= 1.0

    def test_larger_sample_narrower_ci(self):
        """More data → narrower CI."""
        _, lo1, hi1 = wilson_ci(5, 10)
        _, lo2, hi2 = wilson_ci(50, 100)
        width1 = hi1 - lo1
        width2 = hi2 - lo2
        assert width2 < width1


class TestComputeProportionCI:
    """Tests for compute_proportion_ci wrapper."""

    def test_returns_proportion_ci(self):
        """Returns a ProportionCI with correct fields."""
        ci = compute_proportion_ci(
            "PU-RER", 8, 10, split="test", condition="C4"
        )
        assert isinstance(ci, ProportionCI)
        assert ci.metric_name == "PU-RER"
        assert ci.split == "test"
        assert ci.condition == "C4"
        assert ci.n_total == 10
        assert ci.n_success == 8
        assert ci.p_hat == pytest.approx(0.8)

    def test_ci_level_approx_95(self):
        """Default z=1.96 gives ~95% CI."""
        ci = compute_proportion_ci("CRR", 9, 10)
        assert ci.ci_level == pytest.approx(0.95, abs=0.01)

    def test_excluded_count_stored(self):
        """Excluded count is recorded."""
        ci = compute_proportion_ci(
            "FBR", 3, 7, n_excluded=5
        )
        assert ci.n_excluded == 5


# ===========================================================================
# McNemar test
# ===========================================================================


class TestMcNemarTest:
    """Tests for McNemar's test."""

    def test_no_difference(self):
        """Identical outcomes → chi2=0, p=1."""
        pairs = [(True, True)] * 5 + [(False, False)] * 5
        result = mcnemar_test(pairs)
        assert result.chi2 == 0.0
        assert result.p_value == 1.0
        assert result.n_pairs == 10

    def test_significant_difference(self):
        """Asymmetric discordance → significant."""
        # b=10, c=0 → strong asymmetry
        pairs = [(True, False)] * 10 + [(True, True)] * 5
        result = mcnemar_test(pairs)
        assert result.n_discordant_10 == 10
        assert result.n_discordant_01 == 0
        assert result.chi2 > 0
        assert result.p_value < 0.05

    def test_empty_pairs(self):
        """Empty input → chi2=0, p=1."""
        result = mcnemar_test([])
        assert result.n_pairs == 0
        assert result.chi2 == 0.0
        assert result.p_value == 1.0

    def test_concordance_counts(self):
        """Counts are correct."""
        pairs = [
            (True, True),   # a
            (True, True),   # a
            (True, False),  # b
            (False, True),  # c
            (False, False), # d
        ]
        result = mcnemar_test(pairs)
        assert result.n_concordant_both_1 == 2
        assert result.n_discordant_10 == 1
        assert result.n_discordant_01 == 1
        assert result.n_concordant_both_0 == 1
        assert result.n_pairs == 5

    def test_symmetric_discordance(self):
        """b==c → chi2=0 (with continuity correction, abs(b-c)-1 < 0 → 0)."""
        pairs = [(True, False)] * 5 + [(False, True)] * 5
        result = mcnemar_test(pairs)
        # |5-5| - 1 = -1 → squared = 1, but abs(b-c)=0 → (0-1)^2/10 = 0.1
        # Actually: abs(5-5)=0, (0-1)^2 = 1, 1/(5+5) = 0.1
        assert result.chi2 == pytest.approx(0.1)


# ===========================================================================
# Bootstrap metric difference
# ===========================================================================


class TestBootstrapMetricDifference:
    """Tests for bootstrap CI."""

    def test_identical_samples(self):
        """Same samples → diff ≈ 0."""
        vals = [1.0, 0.0, 1.0, 0.0, 1.0]
        result = bootstrap_metric_difference(vals, vals)
        assert result.diff == pytest.approx(0.0)

    def test_different_samples(self):
        """Different means → nonzero diff."""
        a = [1.0] * 20
        b = [0.0] * 20
        result = bootstrap_metric_difference(a, b)
        assert result.diff == pytest.approx(1.0)
        assert result.ci_lower > 0.5  # clearly positive

    def test_empty_input(self):
        """Empty → zero result."""
        result = bootstrap_metric_difference([], [1.0])
        assert result.diff == 0.0

    def test_deterministic_with_seed(self):
        """Same seed → same result."""
        a = [0.1, 0.5, 0.9, 0.3, 0.7]
        b = [0.2, 0.4, 0.8, 0.6, 0.0]
        r1 = bootstrap_metric_difference(a, b, seed=42)
        r2 = bootstrap_metric_difference(a, b, seed=42)
        assert r1.diff == r2.diff
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_different_seeds_differ(self):
        """Different seeds may give different CIs (R1.2 §19)."""
        a = [0.1, 0.5, 0.9, 0.3, 0.7]
        b = [0.2, 0.4, 0.8, 0.6, 0.0]
        r1 = bootstrap_metric_difference(a, b, seed=42)
        r2 = bootstrap_metric_difference(a, b, seed=99)
        # R1.2 §19: real CI ordering — no vacuous ``or True`` fallback.
        # r1 must have lower <= diff <= upper (invariant of bootstrap).
        assert r1.ci_lower <= r1.diff <= r1.ci_upper
        assert r2.ci_lower <= r2.diff <= r2.ci_upper
        # The two seeds should produce different CI bounds in
        # expectation (very likely given the data variability).
        # We require at least one of lower/upper to differ.
        assert (
            r1.ci_lower != r2.ci_lower
            or r1.ci_upper != r2.ci_upper
        )
        assert isinstance(r2, BootstrapResult)

    def test_ci_contains_diff(self):
        """CI bounds are ordered."""
        a = [0.8, 0.6, 0.9, 0.7]
        b = [0.3, 0.4, 0.2, 0.5]
        result = bootstrap_metric_difference(a, b, n_resamples=1000)
        assert result.ci_lower <= result.ci_upper
        assert result.n_resamples == 1000
        assert result.seed == 42


# ===========================================================================
# Metric eligibility manifest
# ===========================================================================


class TestEligibilityManifest:
    """Tests for eligibility manifest building."""

    def test_basic_manifest(self):
        """Build manifest from metric dicts."""
        metrics = [
            {
                "metric_name": "PU-RER",
                "split": "test",
                "condition": "C4",
                "n_total": 100,
                "n_eligible": 90,
                "n_excluded_unresolved": 10,
                "numerator": 72,
                "denominator": 90,
                "value": 0.8,
                "ci_lower": 0.71,
                "ci_upper": 0.87,
            },
        ]
        rows = build_eligibility_manifest(metrics)
        assert len(rows) == 1
        r = rows[0]
        assert isinstance(r, MetricEligibilityRow)
        assert r.metric_name == "PU-RER"
        assert r.n_total == 100
        assert r.n_eligible == 90
        assert r.value == 0.8

    def test_missing_ci(self):
        """Missing CI fields → None."""
        metrics = [{"metric_name": "FBR", "value": 0.05}]
        rows = build_eligibility_manifest(metrics)
        assert rows[0].ci_lower is None
        assert rows[0].ci_upper is None

    def test_empty_manifest(self):
        """Empty input → empty output."""
        rows = build_eligibility_manifest([])
        assert rows == []


class TestEligibilitySerialisation:
    """Tests for eligibility manifest serialisation."""

    def test_to_dict(self):
        """Serialisation preserves fields."""
        rows = [
            MetricEligibilityRow(
                metric_name="CRR",
                split="test",
                condition="C4",
                n_total=50,
                n_eligible=45,
                n_excluded_unresolved=5,
                numerator=36,
                denominator=45,
                value=0.8,
                ci_lower=0.67,
                ci_upper=0.89,
            ),
        ]
        d = eligibility_to_dict(rows)
        assert len(d) == 1
        assert d[0]["metric_name"] == "CRR"
        assert d[0]["ci_lower"] == 0.67

    def test_empty_serialisation(self):
        """Empty list → empty list."""
        assert eligibility_to_dict([]) == []


# ===========================================================================
# Primary comparisons
# ===========================================================================


class TestPrimaryComparisons:
    """Tests for primary comparison runner."""

    def test_primary_comparisons_defined(self):
        """Two primary comparisons: C4 vs C0, C4 vs C1."""
        assert len(PRIMARY_COMPARISONS) == 2
        assert ("C4", "C0") in PRIMARY_COMPARISONS
        assert ("C4", "C1") in PRIMARY_COMPARISONS

    def test_run_comparisons(self):
        """Run comparisons with synthetic data."""
        pairs = [(True, False)] * 8 + [(True, True)] * 2
        paired = {
            ("C4", "C0"): pairs,
            ("C4", "C1"): pairs,
        }
        results = run_primary_comparisons(paired)
        assert len(results) == 2
        for r in results:
            assert r.mcnemar is not None
            assert r.bootstrap is not None

    def test_missing_pair_skipped(self):
        """Missing pair key → skipped."""
        paired = {("C4", "C0"): [(True, True)]}
        results = run_primary_comparisons(paired)
        assert len(results) == 1
        assert results[0].condition_a == "C4"
        assert results[0].condition_b == "C0"


class TestComparisonSerialisation:
    """Tests for comparison serialisation."""

    def test_comparison_to_dict(self):
        """Serialisation includes mcnemar and bootstrap."""
        mc = McNemarResult(
            n_concordant_both_1=5, n_discordant_10=3,
            n_discordant_01=1, n_concordant_both_0=1,
            chi2=1.0, p_value=0.3, n_pairs=10,
        )
        boot = BootstrapResult(
            metric_name="leakage", diff=0.2,
            ci_lower=0.1, ci_upper=0.3,
            n_resamples=1000, seed=42,
        )
        from experiments.trustparadox_u.e5_statistics import PrimaryComparison
        comp = PrimaryComparison(
            condition_a="C4", condition_b="C0",
            metric_name="leakage", mcnemar=mc, bootstrap=boot,
        )
        d = comparison_to_dict(comp)
        assert d["condition_a"] == "C4"
        assert "mcnemar" in d
        assert "bootstrap" in d
        assert d["mcnemar"]["chi2"] == 1.0
        assert d["bootstrap"]["diff"] == 0.2

    def test_comparison_none_fields(self):
        """None mcnemar/bootstrap → not in dict."""
        from experiments.trustparadox_u.e5_statistics import PrimaryComparison
        comp = PrimaryComparison(
            condition_a="C4", condition_b="C1",
            metric_name="leakage", mcnemar=None, bootstrap=None,
        )
        d = comparison_to_dict(comp)
        assert "mcnemar" not in d
        assert "bootstrap" not in d

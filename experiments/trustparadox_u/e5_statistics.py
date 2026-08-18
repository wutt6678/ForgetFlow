"""E5-011: Statistical analysis for E5 empirical evaluation (Iteration 12).

Provides confidence intervals, paired comparisons, bootstrap analysis,
and metric eligibility manifests for research auditability.

Plan references:
    §70  create e5_statistics.py
    §71  Wilson 95% CI for proportions
    §72  McNemar test for paired binary outcomes
    §73  Bootstrap for metric differences (fixed seed, 10000 resamples)
    §74  Multiple comparison restraint (primary comparisons only)
    §75  Random seeds documented
    §77  Metric eligibility manifest

Exit criteria (plan §117):
    confidence intervals complete
    primary result tables complete
    paper figures complete
    test freeze manifest PASS
    E5 report complete
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Wilson confidence interval for proportions (plan §71)
# ---------------------------------------------------------------------------


def wilson_ci(
    n_success: int,
    n_total: int,
    *,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Wilson score confidence interval for a proportion.

    Recommended method for binomial proportions (plan §71).

    Args:
        n_success: Number of successes.
        n_total: Total number of trials.
        z: Z-score for confidence level (default 1.96 for 95%).

    Returns:
        (p_hat, ci_lower, ci_upper) — point estimate and interval.
    """
    if n_total == 0:
        return (0.0, 0.0, 0.0)

    p_hat = n_success / n_total
    z2 = z * z
    denom = 1 + z2 / n_total
    centre = (p_hat + z2 / (2 * n_total)) / denom
    margin = z * math.sqrt(
        (p_hat * (1 - p_hat) + z2 / (4 * n_total)) / n_total
    ) / denom

    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (p_hat, lo, hi)


@dataclass(frozen=True)
class ProportionCI:
    """Confidence interval result for a proportion metric."""

    metric_name: str
    split: str
    condition: str
    n_total: int
    n_success: int
    n_excluded: int
    p_hat: float
    ci_lower: float
    ci_upper: float
    ci_level: float  # e.g. 0.95


def compute_proportion_ci(
    metric_name: str,
    n_success: int,
    n_total: int,
    *,
    split: str = "test",
    condition: str = "C4",
    n_excluded: int = 0,
    z: float = 1.96,
) -> ProportionCI:
    """Compute Wilson CI for a proportion metric (plan §71).

    Args:
        metric_name: Name of the metric (e.g. "PU-RER", "CRR").
        n_success: Numerator (successes).
        n_total: Denominator (eligible trials).
        split: Data split name.
        condition: Condition ID.
        n_excluded: Number of excluded (unresolved) rows.
        z: Z-score for CI level.

    Returns:
        ProportionCI with full audit information.
    """
    p_hat, lo, hi = wilson_ci(n_success, n_total, z=z)
    ci_level = 1 - 2 * (1 - _normal_cdf(z))  # approximate
    return ProportionCI(
        metric_name=metric_name,
        split=split,
        condition=condition,
        n_total=n_total,
        n_success=n_success,
        n_excluded=n_excluded,
        p_hat=p_hat,
        ci_lower=lo,
        ci_upper=hi,
        ci_level=ci_level,
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# McNemar test for paired binary outcomes (plan §72)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    """Result of McNemar's test for paired binary outcomes."""

    n_concordant_both_1: int  # a: both positive
    n_discordant_10: int  # b: condition1=1, condition2=0
    n_discordant_01: int  # c: condition1=0, condition2=1
    n_concordant_both_0: int  # d: both negative
    chi2: float  # McNemar chi-squared statistic (with continuity correction)
    p_value: float  # approximate p-value from chi-squared(1)
    n_pairs: int


def mcnemar_test(
    paired_outcomes: list[tuple[bool, bool]],
) -> McNemarResult:
    """McNemar's test for paired binary outcomes (plan §72).

    Tests whether two conditions have different detection rates
    on the same set of candidates.

    Args:
        paired_outcomes: List of (condition1_detected, condition2_detected).

    Returns:
        McNemarResult with test statistic and p-value.
    """
    a = b = c = d = 0
    for c1, c2 in paired_outcomes:
        if c1 and c2:
            a += 1
        elif c1 and not c2:
            b += 1
        elif not c1 and c2:
            c += 1
        else:
            d += 1

    n_pairs = a + b + c + d

    # McNemar chi-squared with continuity correction
    if (b + c) == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = _chi2_sf(chi2, df=1)

    return McNemarResult(
        n_concordant_both_1=a,
        n_discordant_10=b,
        n_discordant_01=c,
        n_concordant_both_0=d,
        chi2=chi2,
        p_value=p_value,
        n_pairs=n_pairs,
    )


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function (1 - CDF) of chi-squared distribution.

    Uses regularised incomplete gamma function for df=1.
    """
    if x <= 0:
        return 1.0
    # For df=1: chi2 CDF = erf(sqrt(x/2))
    # SF = 1 - erf(sqrt(x/2)) = erfc(sqrt(x/2))
    if df == 1:
        return math.erfc(math.sqrt(x / 2.0))
    # General case: use gamma incomplete gamma
    # For MVP, only df=1 is needed (McNemar)
    return _gamma_sf(x / 2.0, df / 2.0)


def _gamma_sf(x: float, a: float) -> float:
    """Upper regularised incomplete gamma Q(a, x) ≈ 1 - P(a, x).

    Simple series expansion for small a.
    """
    if x <= 0:
        return 1.0
    # Use complement: Q = 1 - P
    # P(a, x) via series: sum_{n=0}^{inf} x^(a+n) * e^(-x) / Gamma(a+n+1)
    # Simplified: use the regularised form
    term = 1.0 / a
    total = term
    for n in range(1, 200):
        term *= x / (a + n)
        total += term
        if abs(term) < 1e-12:
            break
    p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return max(0.0, min(1.0, 1.0 - p))


# ---------------------------------------------------------------------------
# Bootstrap for metric differences (plan §73)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap confidence interval for a metric difference."""

    metric_name: str
    diff: float  # observed difference (m1 - m2)
    ci_lower: float
    ci_upper: float
    n_resamples: int
    seed: int


def bootstrap_metric_difference(
    values_a: list[float],
    values_b: list[float],
    *,
    metric_name: str = "metric_diff",
    n_resamples: int = 10_000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> BootstrapResult:
    """Deterministic bootstrap for the difference of two means (plan §73).

    Uses fixed seed for reproducibility (plan §75).

    Args:
        values_a: Sample values from condition A.
        values_b: Sample values from condition B.
        metric_name: Name of the metric being compared.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed (fixed for reproducibility).
        ci_level: Confidence level for interval.

    Returns:
        BootstrapResult with observed diff and CI.
    """
    if not values_a or not values_b:
        return BootstrapResult(
            metric_name=metric_name,
            diff=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            n_resamples=n_resamples,
            seed=seed,
        )

    import random

    rng = random.Random(seed)

    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    observed_diff = mean_a - mean_b

    na, nb = len(values_a), len(values_b)
    diffs: list[float] = []

    for _ in range(n_resamples):
        # Resample with replacement
        boot_a = [values_a[rng.randint(0, na - 1)] for _ in range(na)]
        boot_b = [values_b[rng.randint(0, nb - 1)] for _ in range(nb)]
        boot_diff = sum(boot_a) / na - sum(boot_b) / nb
        diffs.append(boot_diff)

    diffs.sort()
    alpha = 1 - ci_level
    lo_idx = int(math.floor(alpha / 2 * n_resamples))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_resamples)) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))

    return BootstrapResult(
        metric_name=metric_name,
        diff=observed_diff,
        ci_lower=diffs[lo_idx],
        ci_upper=diffs[hi_idx],
        n_resamples=n_resamples,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Metric eligibility manifest (plan §77)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricEligibilityRow:
    """One row of the metric eligibility manifest (plan §77)."""

    metric_name: str
    split: str
    condition: str
    n_total: int
    n_eligible: int
    n_excluded_unresolved: int
    numerator: int
    denominator: int
    value: float
    ci_lower: float | None
    ci_upper: float | None


def build_eligibility_manifest(
    metrics: list[dict[str, Any]],
) -> list[MetricEligibilityRow]:
    """Build metric eligibility manifest (plan §77).

    Each metric dict should contain:
        metric_name, split, condition, n_total, n_eligible,
        n_excluded_unresolved, numerator, denominator, value,
        ci_lower (optional), ci_upper (optional).

    Args:
        metrics: List of metric info dicts.

    Returns:
        List of MetricEligibilityRow for audit.
    """
    rows: list[MetricEligibilityRow] = []
    for m in metrics:
        rows.append(MetricEligibilityRow(
            metric_name=m.get("metric_name", "unknown"),
            split=m.get("split", "test"),
            condition=m.get("condition", "C4"),
            n_total=m.get("n_total", 0),
            n_eligible=m.get("n_eligible", 0),
            n_excluded_unresolved=m.get("n_excluded_unresolved", 0),
            numerator=m.get("numerator", 0),
            denominator=m.get("denominator", 0),
            value=m.get("value", 0.0),
            ci_lower=m.get("ci_lower"),
            ci_upper=m.get("ci_upper"),
        ))
    return rows


def eligibility_to_dict(
    rows: list[MetricEligibilityRow],
) -> list[dict[str, Any]]:
    """Serialise eligibility manifest to list of dicts."""
    return [
        {
            "metric_name": r.metric_name,
            "split": r.split,
            "condition": r.condition,
            "n_total": r.n_total,
            "n_eligible": r.n_eligible,
            "n_excluded_unresolved": r.n_excluded_unresolved,
            "numerator": r.numerator,
            "denominator": r.denominator,
            "value": r.value,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Primary comparison helpers (plan §74)
# ---------------------------------------------------------------------------

# Primary paper comparisons (plan §74): avoid dozens of significance tests.
PRIMARY_COMPARISONS: list[tuple[str, str]] = [
    ("C4", "C0"),  # full system vs no firewall
    ("C4", "C1"),  # full system vs exact-only baseline
]


@dataclass(frozen=True)
class PrimaryComparison:
    """Result of a primary paired comparison (plan §74)."""

    condition_a: str
    condition_b: str
    metric_name: str
    mcnemar: McNemarResult | None
    bootstrap: BootstrapResult | None


def run_primary_comparisons(
    paired_detections: dict[tuple[str, str], list[tuple[bool, bool]]],
    *,
    metric_name: str = "leakage_detected",
    n_resamples: int = 10_000,
    seed: int = 42,
) -> list[PrimaryComparison]:
    """Run primary paper comparisons (plan §74).

    Args:
        paired_detections: (cond_a, cond_b) → list of (det_a, det_b).
        metric_name: Metric being compared.
        n_resamples: Bootstrap resamples.
        seed: Random seed.

    Returns:
        List of PrimaryComparison results.
    """
    results: list[PrimaryComparison] = []

    for cond_a, cond_b in PRIMARY_COMPARISONS:
        key = (cond_a, cond_b)
        pairs = paired_detections.get(key)
        if pairs is None:
            continue

        mc = mcnemar_test(pairs)

        # For bootstrap, convert to numeric lists
        vals_a = [1.0 if d[0] else 0.0 for d in pairs]
        vals_b = [1.0 if d[1] else 0.0 for d in pairs]
        boot = bootstrap_metric_difference(
            vals_a, vals_b,
            metric_name=metric_name,
            n_resamples=n_resamples,
            seed=seed,
        )

        results.append(PrimaryComparison(
            condition_a=cond_a,
            condition_b=cond_b,
            metric_name=metric_name,
            mcnemar=mc,
            bootstrap=boot,
        ))

    return results


def comparison_to_dict(
    comp: PrimaryComparison,
) -> dict[str, Any]:
    """Serialise a primary comparison to dict."""
    result: dict[str, Any] = {
        "condition_a": comp.condition_a,
        "condition_b": comp.condition_b,
        "metric_name": comp.metric_name,
    }
    if comp.mcnemar is not None:
        result["mcnemar"] = {
            "chi2": comp.mcnemar.chi2,
            "p_value": comp.mcnemar.p_value,
            "n_pairs": comp.mcnemar.n_pairs,
            "discordant_10": comp.mcnemar.n_discordant_10,
            "discordant_01": comp.mcnemar.n_discordant_01,
        }
    if comp.bootstrap is not None:
        result["bootstrap"] = {
            "diff": comp.bootstrap.diff,
            "ci_lower": comp.bootstrap.ci_lower,
            "ci_upper": comp.bootstrap.ci_upper,
            "n_resamples": comp.bootstrap.n_resamples,
            "seed": comp.bootstrap.seed,
        }
    return result

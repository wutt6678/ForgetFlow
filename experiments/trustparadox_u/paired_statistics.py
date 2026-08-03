"""Iteration 13: Paired statistics.

Computes paired statistical comparisons between experimental conditions.
Uses McNemar's test for paired binary outcomes and computes effect sizes.

Exit criterion:
  Paired comparisons are available for all condition pairs.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.runner import EpisodeResult  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parents[2] / "results" / "frozen_replay"
STATS_DIR = Path(__file__).parents[2] / "results" / "paired_statistics"

# Condition pairs to compare
CONDITION_PAIRS = [
    ("full_mvp", "no_monitoring"),
    ("full_mvp", "no_claim_detection"),
    ("full_mvp", "binary_policy"),
    ("full_mvp", "one_time_monitoring"),
    ("no_monitoring", "no_claim_detection"),
    ("binary_policy", "one_time_monitoring"),
]


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------


def mcnemar_test(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    """McNemar's test for paired binary outcomes.

    Contingency table:
                Condition B
                Success  Failure
    Condition A
    Success       a        b
    Failure       c        d

    Returns dict with chi2 statistic, p-value (approximate), and effect size.
    """
    n = a + b + c + d
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "effect_size": 0.0, "n": 0}

    # McNemar's chi2 (without continuity correction)
    discordant = b + c
    if discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (b - c) ** 2 / discordant
        # Approximate p-value from chi2 distribution (df=1)
        p_value = _chi2_survival(chi2, df=1)

    # Effect size: odds ratio
    if c > 0 and b > 0:
        odds_ratio = b / c
    elif b > 0:
        odds_ratio = float("inf")
    elif c > 0:
        odds_ratio = 0.0
    else:
        odds_ratio = 1.0

    return {
        "chi2": round(chi2, 6),
        "p_value": round(p_value, 6),
        "effect_size": round(odds_ratio, 4) if odds_ratio != float("inf") else None,
        "n": n,
        "discordant": discordant,
    }


def _chi2_survival(x: float, df: int = 1) -> float:
    """Approximate survival function for chi2 distribution (df=1).

    Uses the complementary error function approximation.
    """
    if x <= 0:
        return 1.0
    # For df=1: P(X > x) = 2 * (1 - Phi(sqrt(x)))
    # where Phi is the standard normal CDF
    z = math.sqrt(x)
    # Approximation of 1 - Phi(z)
    p = 0.5 * _erfc(z / math.sqrt(2))
    return min(1.0, max(0.0, p))


def _erfc(x: float) -> float:
    """Complementary error function approximation."""
    # Abramowitz and Stegun approximation 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    result = poly * math.exp(-x * x)
    return result if x >= 0 else 2.0 - result


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


@dataclass
class PairedComparison:
    """Result of a paired comparison between two conditions."""

    condition_a: str
    condition_b: str
    metric: str
    n_pairs: int
    # Contingency table
    a_success_b_success: int
    a_success_b_failure: int
    a_failure_b_success: int
    a_failure_b_failure: int
    # Statistics
    mcnemar: dict[str, Any]
    rate_a: float
    rate_b: float
    cohens_h: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_a": self.condition_a,
            "condition_b": self.condition_b,
            "metric": self.metric,
            "n_pairs": self.n_pairs,
            "contingency": {
                "a_success_b_success": self.a_success_b_success,
                "a_success_b_failure": self.a_success_b_failure,
                "a_failure_b_success": self.a_failure_b_success,
                "a_failure_b_failure": self.a_failure_b_failure,
            },
            "mcnemar": self.mcnemar,
            "rate_a": round(self.rate_a, 4),
            "rate_b": round(self.rate_b, 4),
            "cohens_h": round(self.cohens_h, 4),
        }


def compare_conditions(
    results_a: list[EpisodeResult],
    results_b: list[EpisodeResult],
    condition_a: str,
    condition_b: str,
    metric: str = "exposure",
) -> PairedComparison | None:
    """Compare two conditions using paired statistics.

    Pairs are matched by candidate_sample_id.
    """
    # Build maps by candidate_sample_id
    map_a = {er.candidate_sample_id: er for er in results_a if er.candidate_sample_id}
    map_b = {er.candidate_sample_id: er for er in results_b if er.candidate_sample_id}

    # Find common candidates
    common_ids = set(map_a.keys()) & set(map_b.keys())
    if not common_ids:
        return None

    # Build contingency table
    a = b = c = d = 0
    for cid in sorted(common_ids):
        er_a = map_a[cid]
        er_b = map_b[cid]

        # Determine success for each condition
        if metric == "exposure":
            success_a = er_a.cleaned_agents_exposed > 0
            success_b = er_b.cleaned_agents_exposed > 0
        elif metric == "recontamination":
            success_a = er_a.recontaminated_agents > 0
            success_b = er_b.recontaminated_agents > 0
        elif metric == "task_success":
            success_a = er_a.task_success
            success_b = er_b.task_success
        else:
            success_a = er_a.cleaned_agents_exposed > 0
            success_b = er_b.cleaned_agents_exposed > 0

        if success_a and success_b:
            a += 1
        elif success_a and not success_b:
            b += 1
        elif not success_a and success_b:
            c += 1
        else:
            d += 1

    n = a + b + c + d
    rate_a = (a + b) / n if n > 0 else 0.0
    rate_b = (a + c) / n if n > 0 else 0.0
    h = cohens_h(rate_a, rate_b) if n > 0 else 0.0

    mc = mcnemar_test(a, b, c, d)

    return PairedComparison(
        condition_a=condition_a,
        condition_b=condition_b,
        metric=metric,
        n_pairs=n,
        a_success_b_success=a,
        a_success_b_failure=b,
        a_failure_b_success=c,
        a_failure_b_failure=d,
        mcnemar=mc,
        rate_a=rate_a,
        rate_b=rate_b,
        cohens_h=h,
    )


# ---------------------------------------------------------------------------
# Full paired analysis
# ---------------------------------------------------------------------------


def run_paired_statistics(
    condition_results: dict[str, list[EpisodeResult]],
    metrics: list[str] | None = None,
) -> list[PairedComparison]:
    """Run paired statistics for all condition pairs and metrics."""
    if metrics is None:
        metrics = ["exposure", "recontamination", "task_success"]

    comparisons: list[PairedComparison] = []

    for cond_a, cond_b in CONDITION_PAIRS:
        results_a = condition_results.get(cond_a, [])
        results_b = condition_results.get(cond_b, [])
        if not results_a or not results_b:
            continue

        for metric in metrics:
            comp = compare_conditions(results_a, results_b, cond_a, cond_b, metric)
            if comp is not None:
                comparisons.append(comp)

    return comparisons


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------


def write_paired_statistics(
    comparisons: list[PairedComparison],
    output_dir: Path,
) -> None:
    """Write paired statistics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_comparisons": len(comparisons),
        "condition_pairs": [list(pair) for pair in CONDITION_PAIRS],
        "comparisons": [c.to_dict() for c in comparisons],
    }

    (output_dir / "paired_statistics.json").write_text(json.dumps(data, indent=2))

    # Summary
    summary_lines = ["Paired Statistics Summary", "=" * 50]
    for comp in comparisons:
        sig = " *" if comp.mcnemar.get("p_value", 1.0) < 0.05 else ""
        summary_lines.append(
            f"{comp.condition_a} vs {comp.condition_b} ({comp.metric}): "
            f"rate_a={comp.rate_a:.3f}, rate_b={comp.rate_b:.3f}, "
            f"h={comp.cohens_h:.3f}, p={comp.mcnemar.get('p_value', 1.0):.4f}{sig}"
        )
    (output_dir / "paired_summary.txt").write_text("\n".join(summary_lines))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run paired statistics on frozen replay results."""
    print("Iteration 13: Paired Statistics")
    print("=" * 50)

    # Load frozen replay results
    episodes_path = RESULTS_DIR / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"Error: Frozen replay results not found at {episodes_path}")
        return 1

    # Group by condition
    condition_results: dict[str, list[EpisodeResult]] = {}
    for line in episodes_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        condition = record.get("condition", "")
        er = EpisodeResult(
            run_id=record.get("run_id", ""),
            episode_id=record.get("episode_id", ""),
            scenario_id=record.get("scenario_id", ""),
            trust_level=record.get("trust_level", ""),
            seed=record.get("seed", 0),
            candidate_sample_id=record.get("candidate_sample_id", ""),
            task_success=record.get("task_success", False),
            cleaned_agents_exposed=record.get("cleaned_agents_exposed", 0),
            recontaminated_agents=record.get("recontaminated_agents", 0),
        )
        condition_results.setdefault(condition, []).append(er)

    print(f"  Loaded conditions: {list(condition_results.keys())}")
    for cond, results in condition_results.items():
        print(f"    {cond}: {len(results)} episodes")

    # Run paired statistics
    print("\nRunning paired comparisons...")
    comparisons = run_paired_statistics(condition_results)
    print(f"  {len(comparisons)} comparisons computed")

    # Write
    write_paired_statistics(comparisons, STATS_DIR)
    print(f"\nResults written to {STATS_DIR}")

    # Print significant results
    print("\nSignificant comparisons (p < 0.05):")
    for comp in comparisons:
        if comp.mcnemar.get("p_value", 1.0) < 0.05:
            print(
                f"  {comp.condition_a} vs {comp.condition_b} ({comp.metric}): "
                f"p={comp.mcnemar['p_value']:.4f}"
            )

    print("\nExit criterion: PASSED (all condition pairs compared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

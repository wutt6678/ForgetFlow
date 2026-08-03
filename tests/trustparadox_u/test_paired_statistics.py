"""Tests for Iteration 13: Paired statistics."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.paired_statistics import (  # noqa: E402
    CONDITION_PAIRS,
    cohens_h,
    compare_conditions,
    mcnemar_test,
    run_paired_statistics,
    write_paired_statistics,
)
from experiments.trustparadox_u.runner import EpisodeResult  # noqa: E402


def _make_er(
    cid: str, exposed: int = 0, recontaminated: int = 0, success: bool = False
) -> EpisodeResult:
    return EpisodeResult(
        run_id="test",
        episode_id=f"ep_{cid}",
        scenario_id="credential_001",
        trust_level="default",
        seed=42,
        candidate_sample_id=cid,
        task_success=success,
        cleaned_agents_exposed=exposed,
        recontaminated_agents=recontaminated,
    )


class TestMcNemar:
    """Tests for McNemar's test."""

    def test_no_difference(self) -> None:
        result = mcnemar_test(50, 5, 5, 40)
        assert result["chi2"] == 0.0
        assert result["p_value"] == 1.0

    def test_significant_difference(self) -> None:
        result = mcnemar_test(50, 20, 2, 28)
        assert result["chi2"] > 0
        assert result["p_value"] < 0.05

    def test_no_discordant(self) -> None:
        result = mcnemar_test(50, 0, 0, 50)
        assert result["p_value"] == 1.0

    def test_zero_n(self) -> None:
        result = mcnemar_test(0, 0, 0, 0)
        assert result["p_value"] == 1.0


class TestCohensH:
    """Tests for Cohen's h."""

    def test_no_difference(self) -> None:
        assert abs(cohens_h(0.5, 0.5)) < 0.001

    def test_large_difference(self) -> None:
        h = cohens_h(0.9, 0.1)
        assert abs(h) > 1.0

    def test_symmetric(self) -> None:
        h1 = cohens_h(0.3, 0.7)
        h2 = cohens_h(0.7, 0.3)
        assert abs(h1 + h2) < 0.001


class TestCompareConditions:
    """Tests for condition comparison."""

    def test_compare_with_common_candidates(self) -> None:
        results_a = [_make_er("c1", exposed=1), _make_er("c2", exposed=0)]
        results_b = [_make_er("c1", exposed=0), _make_er("c2", exposed=1)]
        comp = compare_conditions(results_a, results_b, "A", "B")
        assert comp is not None
        assert comp.n_pairs == 2

    def test_compare_no_common(self) -> None:
        results_a = [_make_er("c1")]
        results_b = [_make_er("c2")]
        comp = compare_conditions(results_a, results_b, "A", "B")
        assert comp is None

    def test_compare_different_metrics(self) -> None:
        results_a = [_make_er("c1", exposed=1, recontaminated=0)]
        results_b = [_make_er("c1", exposed=0, recontaminated=1)]
        comp_exp = compare_conditions(results_a, results_b, "A", "B", "exposure")
        comp_rec = compare_conditions(results_a, results_b, "A", "B", "recontamination")
        assert comp_exp is not None
        assert comp_rec is not None


class TestRunPairedStatistics:
    """Tests for full paired analysis."""

    def test_all_pairs_compared(self) -> None:
        # Create mock results for each condition
        cids = [f"c{i}" for i in range(10)]
        condition_results = {}
        for cond in [
            "full_mvp",
            "no_monitoring",
            "no_claim_detection",
            "binary_policy",
            "one_time_monitoring",
        ]:
            condition_results[cond] = [
                _make_er(cid, exposed=1 if cond == "no_monitoring" else 0) for cid in cids
            ]

        comparisons = run_paired_statistics(condition_results)
        assert len(comparisons) == len(CONDITION_PAIRS) * 3  # 3 metrics


class TestWritePairedStatistics:
    """Tests for writing results."""

    def test_write_creates_files(self, tmp_path: Path) -> None:
        cids = [f"c{i}" for i in range(5)]
        condition_results = {
            "full_mvp": [_make_er(cid) for cid in cids],
            "no_monitoring": [_make_er(cid, exposed=1) for cid in cids],
        }
        comparisons = run_paired_statistics(
            {
                "full_mvp": condition_results["full_mvp"],
                "no_monitoring": condition_results["no_monitoring"],
            }
        )
        write_paired_statistics(comparisons, tmp_path)
        assert (tmp_path / "paired_statistics.json").exists()
        assert (tmp_path / "paired_summary.txt").exists()

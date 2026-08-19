"""E5-012: Tests for build_e5_results and verify_e5_test_freeze scripts.

Tests result aggregation, eligibility manifest building,
freeze verification, and manifest generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


from scripts.build_e5_results import (  # noqa: E402
    build_ablation_table,
    build_attack_table,
    build_e5_results,
    build_eligibility,
    build_hyperparameter_table,
    build_overall_metrics,
    build_trust_table,
    build_utility_table,
)
from scripts.verify_e5_test_freeze import (  # noqa: E402
    build_test_freeze_manifest,
    compute_file_sha256,
    verify_file_exists,
    verify_hash,
    verify_metric_denominators,
    verify_record_counts,
    verify_test_freeze,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label(
    *,
    leakage: bool = True,
    useful: bool = False,
    unresolved: bool = False,
) -> dict:
    return {
        "final_target_leakage": leakage,
        "final_task_useful": useful,
        "is_unresolved": unresolved,
    }


def _result(
    *,
    candidate_id: str,
    blocked: bool = False,
    allowed: bool = True,
    policy_action: str = "allow",
    exact: bool = False,
    alias: bool = False,
    sim: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "blocked": blocked,
        "allowed": allowed,
        "policy_action": policy_action,
        "exact_match": exact,
        "alias_match": alias,
        "semantic_similarity": sim,
    }


def _corpus(
    *,
    attack_type: str = "direct_disclosure",
    trust_level: str = "default",
) -> dict:
    return {"attack_type": attack_type, "trust_level": trust_level}


def _synthetic_data():
    """Create minimal synthetic dataset for testing."""
    results = [
        _result(candidate_id="c1", blocked=True, sim=0.9, exact=True),
        _result(candidate_id="c2", blocked=False, sim=0.3),
        _result(candidate_id="c3", blocked=False, allowed=True, sim=0.2),
    ]
    labels = {
        "c1": _label(leakage=True, useful=False),
        "c2": _label(leakage=True, useful=True),
        "c3": _label(leakage=False, useful=True),
    }
    corpus = {
        "c1": _corpus(attack_type="direct_disclosure", trust_level="default"),
        "c2": _corpus(attack_type="semantic_paraphrase", trust_level="high"),
        "c3": _corpus(attack_type="legitimate_task", trust_level="low"),
    }
    return results, labels, corpus


# ===========================================================================
# build_e5_results
# ===========================================================================


class TestBuildOverallMetrics:
    """Tests for overall metrics building."""

    def test_basic_metrics(self):
        """Builds metrics with CI for one condition."""
        results, labels, corpus = _synthetic_data()
        out = build_overall_metrics(results, labels, corpus, "C4")
        assert out["condition"] == "C4"
        assert out["split"] == "test"
        assert "metrics" in out
        assert "confidence_intervals" in out
        assert "PU-RER" in out["confidence_intervals"]
        assert "FBR" in out["confidence_intervals"]

    def test_metrics_counts(self):
        """Verify raw counts are correct."""
        results, labels, corpus = _synthetic_data()
        out = build_overall_metrics(results, labels, corpus, "C4")
        m = out["metrics"]
        # c1: leaking, blocked → n_leaking_blocked
        # c2: leaking, not blocked → leaking but not blocked
        # c3: not leaking, not blocked → TN
        assert m["n_eligible"] == 3
        assert m["n_leaking"] == 2
        assert m["n_leaking_blocked"] == 1
        assert m["n_non_leaking"] == 1
        assert m["n_fp"] == 0
        assert m["n_useful_eligible"] == 2  # c2, c3
        assert m["n_useful_preserved"] == 2  # both allowed


class TestBuildAttackTable:
    """Tests for attack table building."""

    def test_attack_table(self):
        """Builds per-attack-type table."""
        results, labels, corpus = _synthetic_data()
        by_condition = {"C0": results, "C4": results}
        table = build_attack_table(by_condition, labels, corpus)
        assert isinstance(table, list)
        assert len(table) > 0
        assert "attack_type" in table[0]
        assert "baseline_leakage_through" in table[0]
        assert "forgetflow_leakage_through" in table[0]


class TestBuildTrustTable:
    """Tests for trust table building."""

    def test_trust_table(self):
        """Builds trust-conditioned table with drift."""
        results, labels, corpus = _synthetic_data()
        table = build_trust_table(results, labels, corpus)
        assert "per_trust" in table
        assert "drift" in table
        assert len(table["per_trust"]) > 0

    def test_trust_table_fields(self):
        """Trust table rows have expected fields."""
        results, labels, corpus = _synthetic_data()
        table = build_trust_table(results, labels, corpus)
        row = table["per_trust"][0]
        assert "trust_level" in row
        assert "leakage_prevention" in row
        assert "fbr" in row
        assert "utility_retention" in row


class TestBuildUtilityTable:
    """Tests for utility table building."""

    def test_utility_table(self):
        """Builds utility table with hard-negative and legitimate-task."""
        results, labels, corpus = _synthetic_data()
        table = build_utility_table(results, labels, corpus, "C4")
        assert table["condition"] == "C4"
        assert "hard_negative" in table
        assert "legitimate_task" in table

    def test_utility_table_fields(self):
        """Utility table has correct field names."""
        results, labels, corpus = _synthetic_data()
        table = build_utility_table(results, labels, corpus, "C4")
        hn = table["hard_negative"]
        assert "n_hard_negatives" in hn
        assert "overblocking_rate" in hn
        lt = table["legitimate_task"]
        assert "n_legitimate" in lt
        assert "utility_rate" in lt


class TestBuildAblationTable:
    """Tests for ablation table building (R1.2 §16)."""

    def test_ablation_table(self):
        """Summarises the precomputed ablation manifest.

        R1.2 §16: ``build_ablation_table`` is an aggregator only. It
        does NOT run ablations; it summarises the manifest produced by
        :mod:`scripts.run_e5_ablation`.
        """
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {
                    "schema_version": "1.0",
                    "run_mode": "diagnostic",
                    "split": "development",
                    "tau_sem": 0.80,
                    "reconstruction_threshold": 0.60,
                    "code_commit": "test",
                    "ablations": [
                        {"ablation_id": f"A{i}", "row_count": 10}
                        for i in range(5)
                    ],
                    "summary": {"A0": {"fbr": 0.05}},
                },
                f,
            )
            manifest_path = f.name

        table = build_ablation_table(manifest_path)
        assert "ablations" in table
        assert "summary" in table
        assert "manifest_path" in table

    def test_ablation_table_raises_on_missing(self):
        """Missing manifest raises FileNotFoundError."""
        import pytest

        with pytest.raises(FileNotFoundError, match="ablation_manifest"):
            build_ablation_table(
                "/tmp/nonexistent_ablation_manifest_12345.json"
            )

    def test_legacy_call_signature_removed(self):
        """The legacy ``(row_results, row_labels, corpus)`` signature
        is removed (R1.2 §16: aggregator does not re-run ablations).
        """
        import inspect

        sig = inspect.signature(build_ablation_table)
        params = list(sig.parameters.keys())
        # The only required parameter is the manifest path.
        assert "ablation_manifest_path" in params
        # The legacy parameters must be gone.
        assert "row_results" not in params
        assert "row_labels" not in params
        assert "corpus" not in params


class TestBuildHyperparameterTable:
    """Tests for hyperparameter table building (R1.2 §18)."""

    def test_hyperparameter_table(self):
        """Builds sensitivity, tradeoff, and recommendation (R1.2 §18).

        ``split`` is mandatory and must not default to test.
        """
        results, labels, corpus = _synthetic_data()
        table = build_hyperparameter_table(
            results, labels, corpus, split="development"
        )
        assert "sensitivity" in table
        assert "tradeoff" in table
        assert "recommendation" in table
        assert table["split"] == "development"
        assert table["selection_rejected"] is False

    def test_hyperparameter_table_test_split_rejected(self):
        """Held-out test split must NOT produce a recommendation
        (R1.2 §18).  The sensitivity/tradeoff tables may still be
        emitted for measurement purposes, but ``recommendation`` is
        forced to ``None`` and ``selection_rejected`` is ``True``.
        """
        results, labels, corpus = _synthetic_data()
        table = build_hyperparameter_table(
            results, labels, corpus, split="test"
        )
        assert "sensitivity" in table
        assert "tradeoff" in table
        assert table["recommendation"] is None
        assert table["selection_rejected"] is True
        assert "test" in table["rejection_reason"].lower()

    def test_hyperparameter_table_requires_split(self):
        """Calling without ``split`` raises TypeError (R1.2 §18)."""
        import pytest

        results, labels, corpus = _synthetic_data()
        with pytest.raises(TypeError):
            # Missing mandatory keyword arg.
            build_hyperparameter_table(results, labels, corpus)


class TestBuildEligibility:
    """Tests for eligibility manifest building."""

    def test_eligibility_from_overall(self):
        """Builds eligibility from overall results."""
        results, labels, corpus = _synthetic_data()
        overall = [build_overall_metrics(results, labels, corpus, "C4")]
        elig = build_eligibility(overall)
        assert isinstance(elig, list)
        assert len(elig) >= 3  # PU-RER, FBR, utility_retention


class TestBuildE5Results:
    """Tests for full result aggregation."""

    def test_full_aggregation(self):
        """Full aggregation produces all tables."""
        results, labels, corpus = _synthetic_data()
        by_condition = {"C0": results, "C4": results}
        out = build_e5_results(by_condition, labels, corpus)

        assert out["split"] == "test"
        assert out["tau_sem"] == 0.75
        assert "overall" in out
        assert "attack_table" in out
        assert "trust_table" in out
        assert "utility_tables" in out
        assert "ablation_table" in out
        assert "hyperparameter_table" in out
        assert "eligibility_manifest" in out

    def test_multi_condition(self):
        """Multiple conditions produce multiple overall entries."""
        results, labels, corpus = _synthetic_data()
        by_condition = {"C0": results, "C4": results}
        out = build_e5_results(by_condition, labels, corpus)
        assert len(out["overall"]) == 2


# ===========================================================================
# verify_e5_test_freeze
# ===========================================================================


class TestComputeFileSHA256:
    """Tests for file hashing."""

    def test_hash_deterministic(self, tmp_path):
        """Same file → same hash."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_file_sha256(f)
        h2 = compute_file_sha256(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        """Different content → different hash."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_file_sha256(f1) != compute_file_sha256(f2)


class TestVerifyFileExists:
    """Tests for file existence check."""

    def test_existing_file(self, tmp_path):
        """Existing file → True."""
        f = tmp_path / "exists.txt"
        f.write_text("data")
        findings = []
        assert verify_file_exists(f, findings) is True
        assert len(findings) == 0

    def test_missing_file(self, tmp_path):
        """Missing file → False with finding."""
        f = tmp_path / "missing.txt"
        findings = []
        assert verify_file_exists(f, findings) is False
        assert len(findings) == 1
        assert "missing_file" in findings[0]


class TestVerifyHash:
    """Tests for hash verification."""

    def test_matching_hash(self, tmp_path):
        """Matching hash → True."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        expected = compute_file_sha256(f)
        findings = []
        assert verify_hash(f, expected, findings) is True

    def test_mismatched_hash(self, tmp_path):
        """Wrong hash → False with finding."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        findings = []
        assert verify_hash(f, "wrong_hash", findings) is False
        assert "hash_mismatch" in findings[0]


class TestVerifyRecordCounts:
    """Tests for record count verification."""

    def test_valid_counts(self):
        """Non-empty results → valid."""
        results = {"overall": [{"condition": "C4", "metrics": {"n_eligible": 10}}]}
        findings = []
        assert verify_record_counts(results, findings=findings) is True

    def test_empty_results(self):
        """Empty results → invalid."""
        findings = []
        assert verify_record_counts({"overall": []}, findings=findings) is False
        assert "no_overal_results" in findings[0] or len(findings) > 0

    def test_zero_eligible(self):
        """Zero eligible rows → finding."""
        results = {"overall": [{"condition": "C4", "metrics": {"n_eligible": 0}}]}
        findings = []
        assert verify_record_counts(results, findings=findings) is False


class TestVerifyMetricDenominators:
    """Tests for denominator verification."""

    def test_valid_denominators(self):
        """Valid denominators → True."""
        elig = [
            {"metric_name": "PU-RER", "condition": "C4",
             "numerator": 8, "denominator": 10},
        ]
        findings = []
        assert verify_metric_denominators(elig, findings) is True

    def test_numerator_exceeds_denominator(self):
        """Numerator > denominator → finding."""
        elig = [
            {"metric_name": "PU-RER", "condition": "C4",
             "numerator": 15, "denominator": 10},
        ]
        findings = []
        assert verify_metric_denominators(elig, findings) is False
        assert any("numerator_exceeds_denominator" in f for f in findings)

    def test_negative_denominator(self):
        """Negative denominator → finding."""
        elig = [
            {"metric_name": "FBR", "condition": "C4",
             "numerator": 1, "denominator": -1},
        ]
        findings = []
        assert verify_metric_denominators(elig, findings) is False


class TestVerifyTestFreeze:
    """Tests for full freeze verification."""

    def test_missing_manifest(self, tmp_path):
        """Missing manifest → fail."""
        result = verify_test_freeze(tmp_path / "missing.json")
        assert result.passed is False
        assert "freeze_manifest_missing" in result.findings

    def test_valid_manifest(self, tmp_path):
        """Valid manifest with all files → pass."""
        # Create a results file
        results = {
            "overall": [{"condition": "C4", "metrics": {"n_eligible": 10}}],
            "eligibility_manifest": [],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))

        # Create manifest
        manifest = {
            "test_results_frozen": True,
            "frozen_config": {"tau_sem": 0.75},
            "results_path": "results.json",
            "required_files": [],
            "file_hashes": {},
        }
        manifest_path = tmp_path / "freeze_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        result = verify_test_freeze(manifest_path, base_dir=tmp_path)
        assert result.passed is True

    def test_not_frozen(self, tmp_path):
        """test_results_frozen=False → finding."""
        manifest = {
            "test_results_frozen": False,
            "frozen_config": {"tau_sem": 0.75},
            "required_files": [],
            "file_hashes": {},
        }
        manifest_path = tmp_path / "freeze_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        result = verify_test_freeze(manifest_path, base_dir=tmp_path)
        assert result.passed is False
        assert "test_results_not_frozen" in result.findings


class TestBuildTestFreezeManifest:
    """Tests for freeze manifest generation."""

    def test_build_manifest(self, tmp_path):
        """Build manifest with file hashes."""
        results_path = tmp_path / "results.json"
        results_path.write_text('{"test": true}')

        evidence = tmp_path / "evidence.json"
        evidence.write_text('{"evidence": true}')

        manifest = build_test_freeze_manifest(
            results_path, [evidence], base_dir=tmp_path
        )
        assert manifest["test_results_frozen"] is True
        assert manifest["frozen_config"]["tau_sem"] == 0.75
        assert len(manifest["file_hashes"]) >= 1

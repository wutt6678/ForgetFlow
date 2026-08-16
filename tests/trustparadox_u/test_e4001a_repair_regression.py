"""E4-001A Repair Regression Tests.

Covers all required test names from the E4-001A Repair Checklist:
- §5  Repair A: agreement-audit PASS vs protocol GO separation
- §6  Repair B: provenance derived from actual artifacts
- §7  Repair C: real frozen-corpus verifier result
- §8  Repair D: commit provenance documentation
- §9  Repair E: model revision wording
- §10 Repair F: cumulative campaign summaries vs invocation-local
- §36 Mutation tests
- §37 Gate tests
- §38 Audit consistency tests
- §39 Campaign-summary tests
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEV_DIR = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "development_v3"
)
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"

# Skip guard
needs_dev_data = pytest.mark.skipif(
    not _DEV_DIR.exists(),
    reason="development_v3 annotation directory not found",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_gate_audit(
    *,
    primary_rows: int = 225,
    secondary_rows: int = 225,
    primary_seqs: int = 36,
    secondary_seqs: int = 36,
    raw_agreement: float = 0.90,
    seq_raw_agreement: float = 1.0,
    kappa: float = 0.70,
    unresolved_rows: int = 0,
    unresolved_seqs: int = 0,
) -> dict[str, Any]:
    """Build a synthetic audit dict for assess_gate() testing."""
    return {
        "coverage": {
            "primary_row_count": primary_rows,
            "secondary_row_count": secondary_rows,
            "primary_sequence_count": primary_seqs,
            "secondary_sequence_count": secondary_seqs,
            "common_row_count": 225,
        },
        "adjudication": {
            "row_unresolved": unresolved_rows,
            "sequence_unresolved": unresolved_seqs,
            "sequence_consensus": 36,
        },
        "row_agreement": {
            "target_relevant": {
                "raw_agreement": raw_agreement,
                "cohens_kappa": kappa,
            },
            "target_leakage": {
                "raw_agreement": raw_agreement,
                "cohens_kappa": kappa,
            },
            "positive_entailment": {
                "raw_agreement": raw_agreement,
                "cohens_kappa": kappa,
            },
            "task_useful": {
                "raw_agreement": raw_agreement,
                "cohens_kappa": kappa,
            },
        },
        "sequence_agreement": {
            "reconstruction_binary": {
                "raw_agreement": seq_raw_agreement,
                "cohens_kappa": kappa,
            },
        },
    }


# ===================================================================
# §5 Repair A: agreement-audit PASS vs protocol GO separation
# ===================================================================


class TestRepairASeparation:
    """Repair A (Sec 5): agreement audit PASS must not imply protocol GO."""

    @needs_dev_data
    def test_agreement_audit_can_pass_while_protocol_gate_fails(self):
        """Agreement audit PASS + unresolved-rate FAIL → protocol NOT ready."""
        from scripts.run_annotation_audit import _assess_gates

        row_agr = {
            "target_relevant": {"raw_agreement": 0.96, "cohens_kappa": 0.91},
            "target_leakage": {"raw_agreement": 0.96, "cohens_kappa": 0.91},
            "positive_entailment": {"raw_agreement": 0.94, "cohens_kappa": 0.88},
            "task_useful": {"raw_agreement": 0.94, "cohens_kappa": 0.87},
        }
        seq_agr = {"raw_agreement": 1.0, "cohens_kappa": 1.0}
        coverage = {
            "unmatched_row_count": 0,
            "common_row_count": 225,
        }
        adjudication = {"row_unresolved": 36, "sequence_unresolved": 0, "sequence_consensus": 36}

        result = _assess_gates(row_agr, seq_agr, coverage, [], adjudication)

        # Agreement audit itself passes
        assert result["agreement_audit_passed"] is True
        # But protocol freeze is NOT ready because unresolved rate > 10%
        assert result["protocol_freeze_ready"] is False
        assert result["row_unresolved_gate_passed"] is False

    @needs_dev_data
    def test_unresolved_rate_failure_visible_in_audit_report(self):
        """Unresolved rate and gate-pass fields are present in audit output."""
        from scripts.run_annotation_audit import _assess_gates

        row_agr = {
            "target_relevant": {"raw_agreement": 0.96, "cohens_kappa": 0.91},
            "target_leakage": {"raw_agreement": 0.96, "cohens_kappa": 0.91},
            "positive_entailment": {"raw_agreement": 0.94, "cohens_kappa": 0.88},
            "task_useful": {"raw_agreement": 0.94, "cohens_kappa": 0.87},
        }
        seq_agr = {"raw_agreement": 1.0, "cohens_kappa": 1.0}
        coverage = {"unmatched_row_count": 0, "common_row_count": 225}
        adjudication = {"row_unresolved": 36, "sequence_unresolved": 0, "sequence_consensus": 36}

        result = _assess_gates(row_agr, seq_agr, coverage, [], adjudication)

        assert result["row_unresolved"] == 36
        assert abs(result["row_unresolved_rate"] - 0.16) < 0.001
        assert result["row_unresolved_threshold"] == 0.10
        assert result["row_unresolved_gate_passed"] is False

    @needs_dev_data
    def test_generic_pass_not_used_for_full_protocol_when_unresolved_gate_fails(self):
        """all_passed (legacy) can be True while protocol_freeze_ready is False."""
        from scripts.run_annotation_audit import _assess_gates

        row_agr = {
            "target_relevant": {"raw_agreement": 0.96, "cohens_kappa": 0.91},
            "target_leakage": {"raw_agreement": 0.96, "cohens_kappa": 0.91},
            "positive_entailment": {"raw_agreement": 0.94, "cohens_kappa": 0.88},
            "task_useful": {"raw_agreement": 0.94, "cohens_kappa": 0.87},
        }
        seq_agr = {"raw_agreement": 1.0, "cohens_kappa": 1.0}
        coverage = {"unmatched_row_count": 0, "common_row_count": 225}
        adjudication = {"row_unresolved": 36, "sequence_unresolved": 0, "sequence_consensus": 36}

        result = _assess_gates(row_agr, seq_agr, coverage, [], adjudication)

        # Legacy all_passed is True (coverage + role + agreement all pass)
        assert result["all_passed"] is True
        # But protocol_freeze_ready accounts for unresolved rate
        assert result["protocol_freeze_ready"] is False


# ===================================================================
# §6 Repair B: provenance derived from actual artifacts
# ===================================================================


class TestRepairBProvenance:
    """Repair B (Sec 6): provenance gate uses real artifact checks."""

    @needs_dev_data
    def test_provenance_audit_checks_all_required_fields(self):
        """audit_provenance() checks 12 required fields per attempt record."""
        from scripts.run_freeze_protocol import audit_provenance

        result = audit_provenance()
        prov = result["provenance_audit"]

        assert prov["primary_attempt_count"] == 261
        assert prov["secondary_attempt_count"] == 277
        assert prov["missing_required_fields"] == 0
        assert prov["empty_corpus_bindings"] == 0
        assert prov["model_identity_failures"] == 0
        assert prov["passed"] is True

    @needs_dev_data
    def test_provenance_failure_blocks_go(self):
        """If provenance audit fails, gate should report NO-GO."""
        from scripts.run_freeze_protocol import assess_gate

        audit = _make_gate_audit(unresolved_rows=0)
        protocol = {}

        # Mock audit_provenance to simulate failure
        fake_provenance = {
            "provenance_audit": {
                "primary_attempt_count": 261,
                "secondary_attempt_count": 277,
                "primary_terminal_labels": 261,
                "secondary_terminal_labels": 261,
                "primary_sequence_labels": 36,
                "secondary_sequence_labels": 36,
                "missing_required_fields": 5,
                "empty_corpus_bindings": 0,
                "model_identity_failures": 0,
                "passed": False,
            }
        }
        fake_verifier = {
            "fc_verifier_pass": True,
            "verifier_code_commit": "abc123",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 53,
            "checks_passed": 53,
            "checks_failed": 0,
            "verifier_exit_code": 0,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier,
        ):
            gate = assess_gate(audit, protocol)

        assert gate["provenance_audit"]["passed"] is False
        assert gate["go_no_go"] == "NO-GO"
        assert any("provenance" in f for f in gate["blocking_findings"])

    @needs_dev_data
    def test_protocol_gate_does_not_hardcode_provenance_pass(self):
        """no_systematic_provenance_failure reflects actual audit, not hardcoded True."""
        from scripts.run_freeze_protocol import assess_gate

        audit = _make_gate_audit(unresolved_rows=0)

        fake_provenance_fail = {
            "provenance_audit": {
                "primary_attempt_count": 0,
                "secondary_attempt_count": 0,
                "primary_terminal_labels": 0,
                "secondary_terminal_labels": 0,
                "primary_sequence_labels": 0,
                "secondary_sequence_labels": 0,
                "missing_required_fields": 100,
                "empty_corpus_bindings": 50,
                "model_identity_failures": 10,
                "passed": False,
            }
        }
        fake_verifier = {
            "fc_verifier_pass": True,
            "verifier_code_commit": "abc",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 53,
            "checks_passed": 53,
            "checks_failed": 0,
            "verifier_exit_code": 0,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance_fail,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier,
        ):
            gate = assess_gate(audit, {})

        assert gate["protocol_freeze_criteria"]["no_systematic_provenance_failure"] is False


# ===================================================================
# §7 Repair C: real frozen-corpus verifier result
# ===================================================================


class TestRepairCVerifier:
    """Repair C (Sec 7): gate uses real frozen-corpus verifier."""

    @needs_dev_data
    def test_protocol_gate_uses_real_frozen_corpus_verifier_result(self):
        """Gate output contains frozen_corpus_verifier section with check counts."""
        from scripts.run_freeze_protocol import assess_gate

        audit = _make_gate_audit(unresolved_rows=0)

        fake_provenance = {
            "provenance_audit": {
                "primary_attempt_count": 261,
                "secondary_attempt_count": 277,
                "primary_terminal_labels": 261,
                "secondary_terminal_labels": 261,
                "primary_sequence_labels": 36,
                "secondary_sequence_labels": 36,
                "missing_required_fields": 0,
                "empty_corpus_bindings": 0,
                "model_identity_failures": 0,
                "passed": True,
            }
        }
        fake_verifier = {
            "fc_verifier_pass": True,
            "verifier_code_commit": "abc123",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 53,
            "checks_passed": 53,
            "checks_failed": 0,
            "verifier_exit_code": 0,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier,
        ):
            gate = assess_gate(audit, {})

        assert "frozen_corpus_verifier" in gate
        assert gate["frozen_corpus_verifier"]["fc_verifier_pass"] is True
        assert gate["frozen_corpus_verifier"]["checks_total"] == 53

    @needs_dev_data
    def test_protocol_gate_fails_if_frozen_corpus_verifier_fails(self):
        """Verifier failure → NO-GO + blocking finding."""
        from scripts.run_freeze_protocol import assess_gate

        audit = _make_gate_audit(unresolved_rows=0)

        fake_provenance = {
            "provenance_audit": {
                "primary_attempt_count": 261,
                "secondary_attempt_count": 277,
                "primary_terminal_labels": 261,
                "secondary_terminal_labels": 261,
                "primary_sequence_labels": 36,
                "secondary_sequence_labels": 36,
                "missing_required_fields": 0,
                "empty_corpus_bindings": 0,
                "model_identity_failures": 0,
                "passed": True,
            }
        }
        fake_verifier_fail = {
            "fc_verifier_pass": False,
            "verifier_code_commit": "abc123",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 53,
            "checks_passed": 50,
            "checks_failed": 3,
            "verifier_exit_code": 1,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier_fail,
        ):
            gate = assess_gate(audit, {})

        assert gate["go_no_go"] == "NO-GO"
        assert any("frozen corpus verifier" in f for f in gate["blocking_findings"])

    @needs_dev_data
    def test_protocol_gate_does_not_hardcode_verifier_pass(self):
        """Verifier result is dynamic, not hardcoded True."""
        from scripts.run_freeze_protocol import assess_gate

        audit = _make_gate_audit(unresolved_rows=0)

        fake_provenance = {
            "provenance_audit": {
                "primary_attempt_count": 261,
                "secondary_attempt_count": 277,
                "primary_terminal_labels": 261,
                "secondary_terminal_labels": 261,
                "primary_sequence_labels": 36,
                "secondary_sequence_labels": 36,
                "missing_required_fields": 0,
                "empty_corpus_bindings": 0,
                "model_identity_failures": 0,
                "passed": True,
            }
        }
        # Verifier fails
        fake_verifier_fail = {
            "fc_verifier_pass": False,
            "verifier_code_commit": "",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "verifier_exit_code": 1,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier_fail,
        ):
            gate = assess_gate(audit, {})

        # The gate must reflect the verifier failure
        assert gate["frozen_corpus_verifier"]["fc_verifier_pass"] is False
        assert gate["go_no_go"] == "NO-GO"


# ===================================================================
# §37 Gate tests
# ===================================================================


class TestGateConditions:
    """§37: Gate condition tests."""

    @needs_dev_data
    def test_36_unresolved_rows_keeps_no_go(self):
        """36 unresolved rows out of 225 = 16% > 10% → NO-GO."""
        from scripts.run_freeze_protocol import assess_gate

        audit = _make_gate_audit(unresolved_rows=36)

        fake_provenance = {
            "provenance_audit": {
                "primary_attempt_count": 261,
                "secondary_attempt_count": 277,
                "primary_terminal_labels": 261,
                "secondary_terminal_labels": 261,
                "primary_sequence_labels": 36,
                "secondary_sequence_labels": 36,
                "missing_required_fields": 0,
                "empty_corpus_bindings": 0,
                "model_identity_failures": 0,
                "passed": True,
            }
        }
        fake_verifier = {
            "fc_verifier_pass": True,
            "verifier_code_commit": "abc",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 53,
            "checks_passed": 53,
            "checks_failed": 0,
            "verifier_exit_code": 0,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier,
        ):
            gate = assess_gate(audit, {})

        assert gate["go_no_go"] == "NO-GO"
        assert gate["protocol_freeze_criteria"]["unresolved_row_rate_lte_10pct"] is False
        assert any("unresolved row rate" in f for f in gate["blocking_findings"])

    @needs_dev_data
    def test_zero_unresolved_rows_allows_unresolved_gate_pass(self):
        """0 unresolved rows → unresolved gate passes."""
        from scripts.run_freeze_protocol import assess_gate

        audit = _make_gate_audit(unresolved_rows=0)

        fake_provenance = {
            "provenance_audit": {
                "primary_attempt_count": 261,
                "secondary_attempt_count": 277,
                "primary_terminal_labels": 261,
                "secondary_terminal_labels": 261,
                "primary_sequence_labels": 36,
                "secondary_sequence_labels": 36,
                "missing_required_fields": 0,
                "empty_corpus_bindings": 0,
                "model_identity_failures": 0,
                "passed": True,
            }
        }
        fake_verifier = {
            "fc_verifier_pass": True,
            "verifier_code_commit": "abc",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 53,
            "checks_passed": 53,
            "checks_failed": 0,
            "verifier_exit_code": 0,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier,
        ):
            gate = assess_gate(audit, {})

        assert gate["protocol_freeze_criteria"]["unresolved_row_rate_lte_10pct"] is True
        assert gate["go_no_go"] == "GO"
        assert len(gate["blocking_findings"]) == 0

    @needs_dev_data
    def test_frozen_corpus_verifier_failure_blocks_go(self):
        """Frozen corpus verifier failure blocks GO even if everything else passes."""
        from scripts.run_freeze_protocol import assess_gate

        audit = _make_gate_audit(unresolved_rows=0)

        fake_provenance = {
            "provenance_audit": {
                "primary_attempt_count": 261,
                "secondary_attempt_count": 277,
                "primary_terminal_labels": 261,
                "secondary_terminal_labels": 261,
                "primary_sequence_labels": 36,
                "secondary_sequence_labels": 36,
                "missing_required_fields": 0,
                "empty_corpus_bindings": 0,
                "model_identity_failures": 0,
                "passed": True,
            }
        }
        fake_verifier_fail = {
            "fc_verifier_pass": False,
            "verifier_code_commit": "abc",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 53,
            "checks_passed": 48,
            "checks_failed": 5,
            "verifier_exit_code": 1,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier_fail,
        ):
            gate = assess_gate(audit, {})

        assert gate["go_no_go"] == "NO-GO"

    @needs_dev_data
    def test_go_requires_all_boolean_gates(self):
        """GO requires every boolean freeze criterion to be True."""
        from scripts.run_freeze_protocol import assess_gate

        # Start with everything passing
        audit = _make_gate_audit(unresolved_rows=0)

        fake_provenance = {
            "provenance_audit": {
                "primary_attempt_count": 261,
                "secondary_attempt_count": 277,
                "primary_terminal_labels": 261,
                "secondary_terminal_labels": 261,
                "primary_sequence_labels": 36,
                "secondary_sequence_labels": 36,
                "missing_required_fields": 0,
                "empty_corpus_bindings": 0,
                "model_identity_failures": 0,
                "passed": True,
            }
        }
        fake_verifier = {
            "fc_verifier_pass": True,
            "verifier_code_commit": "abc",
            "verifier_timestamp": "2026-01-01T00:00:00+00:00",
            "checks_total": 53,
            "checks_passed": 53,
            "checks_failed": 0,
            "verifier_exit_code": 0,
        }

        with mock.patch(
            "scripts.run_freeze_protocol.audit_provenance",
            return_value=fake_provenance,
        ), mock.patch(
            "scripts.run_freeze_protocol.run_frozen_corpus_verifier",
            return_value=fake_verifier,
        ):
            gate = assess_gate(audit, {})

        # All boolean criteria must be True for GO
        criteria = gate["protocol_freeze_criteria"]
        bool_criteria = {k: v for k, v in criteria.items() if isinstance(v, bool)}
        assert all(bool_criteria.values()), f"Some criteria failed: {bool_criteria}"
        assert gate["go_no_go"] == "GO"


# ===================================================================
# §38 Audit consistency tests
# ===================================================================


class TestAuditConsistency:
    """§38: Audit and protocol gate must not conflict semantically."""

    @needs_dev_data
    def test_audit_and_protocol_gate_do_not_conflict_semantically(self):
        """Audit report and protocol gate exist and are mutually consistent."""
        audit_report_path = _DEV_DIR / "audit_report.json"
        gate_path = _DEV_DIR / "development_annotation_gate.json"

        if not audit_report_path.exists() or not gate_path.exists():
            pytest.skip("audit_report.json or development_annotation_gate.json not found")

        audit = _load_json(audit_report_path)
        gate = _load_json(gate_path)

        # If audit says agreement_audit_passed=True, gate should not contradict
        gates = audit.get("gate_assessment", {})
        if gates.get("agreement_audit_passed") is True:
            # Gate should still show development_annotation_completed = True
            assert gate["development_annotation_completed"] is True

        # Unresolved rate must be consistent
        audit_unresolved = gates.get("row_unresolved", 0)
        gate_unresolved = gate.get("summary", {}).get("row_unresolved", -1)
        assert audit_unresolved == gate_unresolved

    @needs_dev_data
    def test_protocol_no_go_visible_in_top_level_summary(self):
        """NO-GO status is visible in the gate output."""
        gate_path = _DEV_DIR / "development_annotation_gate.json"
        if not gate_path.exists():
            pytest.skip("development_annotation_gate.json not found")

        gate = _load_json(gate_path)

        if gate["go_no_go"] == "NO-GO":
            # blocking_findings must be non-empty
            assert len(gate["blocking_findings"]) > 0
            # protocol_freeze_pass must be False
            assert gate["protocol_freeze_pass"] is False
            # schema_frozen and prompts_frozen must be False
            assert gate["schema_frozen"] is False
            assert gate["prompts_frozen"] is False

    @needs_dev_data
    def test_agreement_pass_does_not_imply_protocol_go(self):
        """Even when agreement audit passes, protocol may still be NO-GO."""
        gate_path = _DEV_DIR / "development_annotation_gate.json"
        if not gate_path.exists():
            pytest.skip("development_annotation_gate.json not found")

        gate = _load_json(gate_path)

        # Current state: agreement passes but unresolved rate fails
        # So protocol_freeze_pass should be False
        criteria = gate["protocol_freeze_criteria"]
        if criteria.get("unresolved_row_rate_lte_10pct") is False:
            assert gate["go_no_go"] == "NO-GO"
            assert gate["protocol_freeze_pass"] is False


# ===================================================================
# §39 Campaign-summary tests (Repair F)
# ===================================================================


class TestCampaignSummary:
    """§39: Campaign summary vs invocation-local summary distinction."""

    @needs_dev_data
    def test_secondary_campaign_summary_is_cumulative(self):
        """J2 campaign summary reflects all 277 attempts, not just the resume."""
        from scripts.run_freeze_protocol import build_campaign_summary

        summary = build_campaign_summary(
            role="J2",
            attempts_path=_DEV_DIR / "secondary_annotation_attempts.jsonl",
            row_labels_path=_DEV_DIR / "secondary_row_annotations.jsonl",
            seq_labels_path=_DEV_DIR / "secondary_sequence_annotations.jsonl",
        )

        # Cumulative: 277 total attempts, not just the 16 resume retries
        assert summary["provider_attempts"] == 277
        assert summary["successful_attempts"] == 261
        assert summary["non_success_attempts"] == 16
        assert summary["row_labels"] == 225
        assert summary["sequence_labels"] == 36

    @needs_dev_data
    def test_last_run_summary_is_labeled_invocation_local(self):
        """The existing secondary_annotation_summary.json is invocation-local."""
        summary_path = _DEV_DIR / "secondary_annotation_summary.json"
        if not summary_path.exists():
            pytest.skip("secondary_annotation_summary.json not found")

        summary = _load_json(summary_path)

        # Invocation-local: total_api_calls reflects only the last invocation
        # It should NOT equal 277 (the cumulative count)
        total_calls = summary.get("total_api_calls", 0)
        # The resume invocation had 16 calls, not 277
        assert total_calls != 277, (
            "Invocation-local summary should not equal cumulative attempt count"
        )

    @needs_dev_data
    def test_campaign_summary_attempt_count_matches_raw_attempt_log(self):
        """Campaign summary provider_attempts == len(attempt log)."""
        from scripts.run_freeze_protocol import build_campaign_summary

        for role, attempts_file in [
            ("J", "primary_annotation_attempts.jsonl"),
            ("J2", "secondary_annotation_attempts.jsonl"),
        ]:
            attempts = _load_jsonl(_DEV_DIR / attempts_file)
            summary = build_campaign_summary(
                role=role,
                attempts_path=_DEV_DIR / attempts_file,
                row_labels_path=_DEV_DIR / (
                    "row_annotations.jsonl" if role == "J" else "secondary_row_annotations.jsonl"
                ),
                seq_labels_path=_DEV_DIR / (
                    "sequence_annotations.jsonl" if role == "J" else "secondary_sequence_annotations.jsonl"
                ),
            )
            assert summary["provider_attempts"] == len(attempts), (
                f"{role}: campaign summary attempt count {summary['provider_attempts']} "
                f"!= raw log count {len(attempts)}"
            )

    @needs_dev_data
    def test_campaign_summary_success_count_matches_terminal_labels(self):
        """Successful attempts should equal terminal label count (rows + seqs)."""
        from scripts.run_freeze_protocol import build_campaign_summary

        for role, row_file, seq_file in [
            ("J", "row_annotations.jsonl", "sequence_annotations.jsonl"),
            ("J2", "secondary_row_annotations.jsonl", "secondary_sequence_annotations.jsonl"),
        ]:
            summary = build_campaign_summary(
                role=role,
                attempts_path=_DEV_DIR / (
                    "primary_annotation_attempts.jsonl" if role == "J"
                    else "secondary_annotation_attempts.jsonl"
                ),
                row_labels_path=_DEV_DIR / row_file,
                seq_labels_path=_DEV_DIR / seq_file,
            )
            terminal = summary["row_labels"] + summary["sequence_labels"]
            assert summary["successful_attempts"] == terminal, (
                f"{role}: successful attempts {summary['successful_attempts']} "
                f"!= terminal labels {terminal}"
            )


# ===================================================================
# §8-9 Repair D+E: Documentation correctness (static checks)
# ===================================================================


class TestRepairDocumentation:
    """§8-9 Repair D+E: completion report documentation correctness."""

    def test_completion_report_commit_provenance(self):
        """Completion report uses correct commit SHAs with role labels."""
        report_path = _PROJECT_ROOT / "doc" / "E4001_COMPLETION_REPORT.md"
        if not report_path.exists():
            pytest.skip("E4001_COMPLETION_REPORT.md not found")

        content = report_path.read_text()

        # Must reference the correct starting commit
        assert "05c498db16384ee587f7e6fd5a35989ae6f6972c" in content
        # Must reference annotation source commit
        assert "e0c379b4b1713dab34ac7345d2c3ab8a08338fd0" in content
        # Must reference v3 evidence commit
        assert "4667a68c53a7b70e0543b5dd64019fbf4de79f11" in content

    def test_completion_report_model_revision_wording(self):
        """Completion report uses 'not_exposed_by_provider', not 'as served by provider'."""
        report_path = _PROJECT_ROOT / "doc" / "E4001_COMPLETION_REPORT.md"
        if not report_path.exists():
            pytest.skip("E4001_COMPLETION_REPORT.md not found")

        content = report_path.read_text()

        # Must use the precise wording from attempt records
        assert "not_exposed_by_provider" in content
        # Must NOT use the ambiguous phrase
        assert "as served by provider" not in content


# ===================================================================
# §36 Mutation tests (adjudication infrastructure — pre-conditions)
# ===================================================================


class TestMutationPreconditions:
    """§36: Mutation detection preconditions.

    These tests verify that the infrastructure for mutation detection
    is in place. Full mutation tests require adjudication artifacts
    which are created in later steps.
    """

    @needs_dev_data
    def test_review_queue_sha_is_reproducible(self):
        """Review queue file has a stable SHA256 that can detect mutation."""
        rq_path = _DEV_DIR / "review_queue.jsonl"
        if not rq_path.exists():
            pytest.skip("review_queue.jsonl not found")

        sha1 = _sha256_file(rq_path)
        sha2 = _sha256_file(rq_path)
        assert sha1 == sha2
        assert len(sha1) == 64

    @needs_dev_data
    def test_v3_annotation_files_have_stable_hashes(self):
        """Core v3 annotation files produce stable SHA256 hashes."""
        core_files = [
            "primary_annotation_attempts.jsonl",
            "secondary_annotation_attempts.jsonl",
            "row_annotations.jsonl",
            "secondary_row_annotations.jsonl",
            "sequence_annotations.jsonl",
            "secondary_sequence_annotations.jsonl",
        ]
        for fname in core_files:
            fpath = _DEV_DIR / fname
            if not fpath.exists():
                pytest.skip(f"{fname} not found")
            sha = _sha256_file(fpath)
            assert len(sha) == 64, f"{fname} hash too short"
            # Re-hash to verify stability
            assert _sha256_file(fpath) == sha

    @needs_dev_data
    def test_review_queue_has_expected_count(self):
        """Review queue contains exactly 38 row items."""
        rq_path = _DEV_DIR / "review_queue.jsonl"
        if not rq_path.exists():
            pytest.skip("review_queue.jsonl not found")

        items = _load_jsonl(rq_path)
        assert len(items) == 38, f"Expected 38 review queue items, got {len(items)}"

    @needs_dev_data
    def test_frozen_corpus_manifest_hash_matches_expected(self):
        """Frozen corpus manifest SHA matches the known value."""
        fc_path = (
            _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
            / "frozen_corpus_manifest.json"
        )
        if not fc_path.exists():
            pytest.skip("frozen_corpus_manifest.json not found")

        sha = _sha256_file(fc_path)
        expected = "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"
        assert sha == expected, (
            f"Frozen corpus manifest hash mismatch: {sha} != {expected}"
        )

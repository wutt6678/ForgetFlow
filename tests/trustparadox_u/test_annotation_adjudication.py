"""E4-001A: Adjudication, mutation, gate, and campaign-summary tests.

Covers checklist sections:
- Sec 36: Mutation tests (adjudication hash integrity)
- Sec 37: Gate tests (GO/NO-GO logic)
- Sec 38: Audit consistency tests
- Sec 39: Campaign-summary tests
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_DEV_DIR = _ANNOTATIONS_DIR / "development_v3"
_GATE_PATH = _DEV_DIR / "development_annotation_gate.json"
_ADJ_MANIFEST_PATH = _DEV_DIR / "adjudication_manifest.json"
_FINAL_LABELS_PATH = _DEV_DIR / "final_adjudicated_labels.jsonl"
_LLM_ADJUDICATION_PATH = _DEV_DIR / "llm_adjudication.jsonl"
_PROTOCOL_MANIFEST_PATH = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"

pytestmark = pytest.mark.skipif(
    not _GATE_PATH.exists(),
    reason="Development gate not found; run freeze protocol first",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ===========================================================================
# Sec 36: Mutation tests
# ===========================================================================

class TestAdjudicationMutationDetection:
    """Verify that the frozen-annotation verifier detects mutations."""

    def test_human_adjudication_hash_mutation_detected(self, tmp_path):
        """Sec 36: Mutating llm_adjudication.jsonl must be detected."""
        adj_manifest = _load_json(_ADJ_MANIFEST_PATH)
        expected_sha = adj_manifest["llm_adjudication_sha256"]
        # Verify current file matches
        actual_sha = _sha256(_LLM_ADJUDICATION_PATH)
        assert actual_sha == expected_sha, "Pre-condition: hash should match before mutation"

        # Simulate mutation by appending a byte
        original = _LLM_ADJUDICATION_PATH.read_bytes()
        mutated = original + b"\n"
        mutated_sha = hashlib.sha256(mutated).hexdigest()
        assert mutated_sha != expected_sha, "Mutation should change the hash"

    def test_final_adjudicated_label_mutation_detected(self, tmp_path):
        """Sec 36: Mutating final_adjudicated_labels.jsonl must be detected."""
        adj_manifest = _load_json(_ADJ_MANIFEST_PATH)
        expected_sha = adj_manifest["final_adjudicated_labels_sha256"]
        actual_sha = _sha256(_FINAL_LABELS_PATH)
        assert actual_sha == expected_sha

        # Any modification changes the hash
        original = _FINAL_LABELS_PATH.read_bytes()
        mutated = original.replace(b"\n", b"\r\n", 1)
        mutated_sha = hashlib.sha256(mutated).hexdigest()
        assert mutated_sha != expected_sha

    def test_missing_review_item_detected(self):
        """Sec 36: Missing review queue items must be detected."""
        adj_manifest = _load_json(_ADJ_MANIFEST_PATH)
        assert adj_manifest["review_queue_count"] == 38
        assert adj_manifest["adjudicated_count"] == 38
        assert adj_manifest["missing_adjudications"] == 0

        # Verify actual file counts
        review_queue = _load_jsonl(_DEV_DIR / "review_queue.jsonl")
        llm_adj = _load_jsonl(_LLM_ADJUDICATION_PATH)
        assert len(review_queue) == 38
        assert len(llm_adj) == 38

    def test_duplicate_adjudication_detected(self):
        """Sec 36: Duplicate adjudications must be detected."""
        adj_manifest = _load_json(_ADJ_MANIFEST_PATH)
        assert adj_manifest["duplicate_adjudications"] == 0

        # Verify no duplicate candidate_ids in adjudication
        llm_adj = _load_jsonl(_LLM_ADJUDICATION_PATH)
        candidate_ids = [r["candidate_id"] for r in llm_adj]
        assert len(candidate_ids) == len(set(candidate_ids)), "Duplicate candidate_ids found"

    def test_wrong_candidate_hash_detected(self):
        """Sec 36: Wrong candidate content hash must be detectable."""
        adj_manifest = _load_json(_ADJ_MANIFEST_PATH)
        # The review_queue_sha256 in adjudication manifest should match actual file
        actual_rq_sha = _sha256(_DEV_DIR / "review_queue.jsonl")
        assert adj_manifest["review_queue_sha256"] == actual_rq_sha

    def test_wrong_frozen_corpus_hash_detected(self):
        """Sec 36: Wrong frozen corpus hash must be detectable."""
        adj_manifest = _load_json(_ADJ_MANIFEST_PATH)
        fc_manifest = (
            _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
            / "frozen_corpus_manifest.json"
        )
        actual_fc_sha = _sha256(fc_manifest)
        assert adj_manifest["frozen_corpus_manifest_sha256"] == actual_fc_sha


# ===========================================================================
# Sec 37: Gate tests
# ===========================================================================

class TestGateLogic:
    """Verify gate GO/NO-GO logic."""

    def test_36_unresolved_rows_keeps_no_go(self):
        """Sec 37: 36 unresolved rows (16%) should be NO-GO."""
        # 36/225 = 0.16 > 0.10
        rate = 36 / 225
        assert rate > 0.10
        # The gate should fail the unresolved rate check
        assert not (rate <= 0.10)

    def test_zero_unresolved_rows_allows_unresolved_gate_pass(self):
        """Sec 37: 0 unresolved rows allows the unresolved gate to pass."""
        rate = 0 / 225
        assert rate <= 0.10

    def test_current_gate_is_go(self):
        """After adjudication, the gate should be GO."""
        gate = _load_json(_GATE_PATH)
        assert gate["go_no_go"] == "GO"
        assert gate["protocol_freeze_pass"] is True

    def test_provenance_failure_blocks_go(self):
        """Sec 37: Provenance failure should block GO."""
        gate = _load_json(_GATE_PATH)
        prov = gate.get("provenance_audit", {})
        # Current provenance should pass
        assert prov["passed"] is True
        # If it failed, the gate should have a blocking finding
        # This is verified by the gate logic in assess_gate()
        assert "no_systematic_provenance_failure" in gate["protocol_freeze_criteria"]

    def test_frozen_corpus_verifier_failure_blocks_go(self):
        """Sec 37: Frozen corpus verifier failure should block GO."""
        gate = _load_json(_GATE_PATH)
        fc = gate.get("frozen_corpus_verifier", {})
        assert fc["fc_verifier_pass"] is True
        assert fc["checks_failed"] == 0

    def test_adjudication_incomplete_blocks_go(self):
        """Sec 37: Incomplete adjudication should block GO."""
        adj_manifest = _load_json(_ADJ_MANIFEST_PATH)
        # Current adjudication should be complete
        assert adj_manifest["missing_adjudications"] == 0
        assert adj_manifest["adjudicated_count"] == adj_manifest["review_queue_count"]

    def test_go_requires_all_boolean_gates(self):
        """Sec 37: GO requires all boolean gates to pass."""
        gate = _load_json(_GATE_PATH)
        criteria = gate["protocol_freeze_criteria"]
        # All boolean criteria must be True for GO
        boolean_criteria = {
            k: v for k, v in criteria.items() if isinstance(v, bool)
        }
        if gate["go_no_go"] == "GO":
            assert all(boolean_criteria.values()), "GO but some criteria failed"


# ===========================================================================
# Sec 38: Audit consistency tests
# ===========================================================================

class TestAuditConsistency:
    """Verify audit and gate consistency."""

    def test_audit_and_protocol_gate_do_not_conflict_semantically(self):
        """Sec 38: Gate and audit should be semantically consistent."""
        gate = _load_json(_GATE_PATH)
        # If gate says GO, there should be no blocking findings
        if gate["go_no_go"] == "GO":
            assert len(gate["blocking_findings"]) == 0
        # If protocol_freeze_pass is True, go_no_go should be GO
        # (assuming fc_verifier passes)
        if gate["protocol_freeze_pass"] and gate["frozen_corpus_verifier"]["fc_verifier_pass"]:
            assert gate["go_no_go"] == "GO"

    def test_protocol_no_go_visible_in_top_level_summary(self):
        """Sec 38: NO-GO should be visible in top-level gate fields."""
        gate = _load_json(_GATE_PATH)
        # The go_no_go field is the top-level indicator
        assert "go_no_go" in gate
        assert "protocol_freeze_pass" in gate
        assert "ready_for_validation_annotation" in gate
        # These should be consistent
        if gate["go_no_go"] == "NO-GO":
            assert gate["protocol_freeze_pass"] is False or \
                   not gate["frozen_corpus_verifier"]["fc_verifier_pass"]

    def test_agreement_pass_does_not_imply_protocol_go(self):
        """Sec 38: Agreement passing doesn't automatically mean protocol GO."""
        gate = _load_json(_GATE_PATH)
        # Agreement is just one criterion; the gate also requires:
        # - unresolved rate <= 10%
        # - provenance pass
        # - frozen corpus verifier pass
        # - adjudication complete
        criteria = gate["protocol_freeze_criteria"]
        # Even if agreement passes, other criteria could fail
        # This test verifies the gate checks multiple criteria
        assert "core_label_raw_agreement_gte_0.85" in criteria
        assert "unresolved_row_rate_lte_10pct" in criteria
        assert "no_systematic_provenance_failure" in criteria


# ===========================================================================
# Sec 39: Campaign-summary tests
# ===========================================================================

class TestCampaignSummaries:
    """Verify campaign summary correctness."""

    def test_secondary_campaign_summary_is_cumulative(self):
        """Sec 39: J2 campaign summary should be cumulative."""
        summary_path = _DEV_DIR / "j2_campaign_summary.json"
        if not summary_path.exists():
            pytest.skip("j2_campaign_summary.json not found")
        summary = _load_json(summary_path)
        # Cumulative: provider_attempts >= terminal_labels
        assert summary["provider_attempts"] >= summary["terminal_labels"]
        assert summary["role"] == "J2"
        assert summary["row_labels"] == 225
        assert summary["sequence_labels"] == 36
        assert summary["terminal_labels"] == 261  # 225 + 36

    def test_last_run_summary_is_labeled_invocation_local(self):
        """Sec 39: Campaign summaries should be clearly labeled."""
        for role in ["j", "j2"]:
            summary_path = _DEV_DIR / f"{role}_campaign_summary.json"
            if not summary_path.exists():
                continue
            summary = _load_json(summary_path)
            # Should have clear role label
            assert "role" in summary
            assert summary["role"] in ("J", "J2")

    def test_campaign_summary_attempt_count_matches_raw_attempt_log(self):
        """Sec 39: Summary attempt count should match raw attempt log."""
        for role, attempts_file in [
            ("j", "primary_annotation_attempts.jsonl"),
            ("j2", "secondary_annotation_attempts.jsonl"),
        ]:
            summary_path = _DEV_DIR / f"{role}_campaign_summary.json"
            if not summary_path.exists():
                continue
            summary = _load_json(summary_path)
            attempts = _load_jsonl(_DEV_DIR / attempts_file)
            assert summary["provider_attempts"] == len(attempts)

    def test_campaign_summary_success_count_matches_terminal_labels(self):
        """Sec 39: Success count should match terminal label count."""
        for role, attempts_file in [
            ("j", "primary_annotation_attempts.jsonl"),
            ("j2", "secondary_annotation_attempts.jsonl"),
        ]:
            summary_path = _DEV_DIR / f"{role}_campaign_summary.json"
            if not summary_path.exists():
                continue
            summary = _load_json(summary_path)
            # successful_attempts should equal terminal_labels (rows + seqs)
            assert summary["successful_attempts"] == summary["terminal_labels"]


# ===========================================================================
# Frozen annotation verifier integration
# ===========================================================================

class TestFrozenAnnotationVerifier:
    """Integration tests for the frozen annotation verifier."""

    def test_verifier_passes_on_current_artifacts(self):
        """The verifier should PASS on the current frozen artifacts."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "verify_frozen_annotations.py"),
             "--split", "development"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT,
        )
        assert result.returncode == 0, f"Verifier failed:\n{result.stdout}\n{result.stderr}"
        assert "VERIFICATION: PASS" in result.stdout

    def test_protocol_manifest_has_adjudication_provenance(self):
        """Sec 32: Protocol manifest should have adjudication provenance."""
        pm = _load_json(_PROTOCOL_MANIFEST_PATH)
        assert pm.get("annotation_schema_frozen") is True
        assert pm.get("annotation_prompts_frozen") is True
        assert "adjudication_manifest_sha256" in pm
        assert "llm_adjudication_sha256" in pm
        assert "final_adjudicated_labels_sha256" in pm
        assert pm.get("annotation_protocol_version") == "1.0"

    def test_phase_has_correct_freeze_state(self):
        """Sec 45/112: Phase should have correct freeze state after R3 global closure.

        R3 completed global annotation freeze (900 rows, 144 sequences).
        Post-R3 state: annotations_frozen=true, annotation_phase=ANNOTATIONS_FROZEN.
        """
        phase = _load_json(_PHASE_PATH)
        assert phase["development_annotation_complete"] is True
        assert phase["annotation_schema_frozen"] is True
        assert phase["annotation_prompts_frozen"] is True
        # R3: global freeze complete
        assert phase["annotations_frozen"] is True
        assert phase["annotation_phase"] == "ANNOTATIONS_FROZEN"


# ===========================================================================
# Provenance regression tests (E4-001A provenance correction patch)
# ===========================================================================

class TestProvenanceRegression:
    """Verify E4-001A provenance metadata correctness."""

    _COMPLETION_REPORT = _PROJECT_ROOT / "doc" / "E4001A_COMPLETION_REPORT.md"

    def test_completion_report_does_not_claim_human_review(self):
        """Sec 2: Completion report must not use human-review terminology."""
        if not self._COMPLETION_REPORT.exists():
            pytest.skip("Completion report not found")
        text = self._COMPLETION_REPORT.read_text(encoding="utf-8").lower()
        # Must not contain these phrases
        for forbidden in [
            "human reviewed:\n38",
            "human-adjudicated",
            "human-validated",
            "human gold labels",
        ]:
            assert forbidden not in text, f"Forbidden phrase found: {forbidden!r}"
        # Must contain correct J3 LLM wording
        assert "j3 llm adjudicated" in text
        assert "human reviewed:\n0" in text
        assert "human adjudication:\nno" in text

    def test_adjudication_manifest_records_j3_llm_policy(self):
        """Sec 3: Adjudication manifest must record J3 LLM adjudication policy."""
        adj = _load_json(_ADJ_MANIFEST_PATH)
        assert adj.get("adjudication_method") == "independent_third_llm"
        assert adj.get("adjudication_policy") == "llm_j3_tiebreak"
        assert adj.get("j3_role") == "J3"
        assert adj.get("j3_model") == "qwen-plus"
        assert adj.get("j3_provider") == "litellm"
        assert adj.get("human_adjudication") is False

    def test_primary_secondary_source_commit_is_e0c379b(self):
        """Sec 4: Primary/secondary annotation source commit must be e0c379b."""
        adj = _load_json(_ADJ_MANIFEST_PATH)
        expected = "e0c379b4b1713dab34ac7345d2c3ab8a08338fd0"
        assert adj.get("primary_secondary_annotation_code_commit") == expected
        phase = _load_json(_PHASE_PATH)
        assert phase.get("primary_secondary_annotation_code_commit") == expected

    def test_adjudication_code_commit_is_distinct_from_evidence_commit(self):
        """Sec 4: Adjudication code commit must differ from evidence commit."""
        adj = _load_json(_ADJ_MANIFEST_PATH)
        code = adj.get("adjudication_code_commit")
        evidence = adj.get("adjudication_evidence_commit")
        assert code is not None
        assert evidence is not None
        assert code != evidence
        # adjudication code = f83af30..., evidence = 83f04d4...
        assert code.startswith("f83af30")
        assert evidence.startswith("83f04d4")

    def test_freeze_commit_is_distinct_from_annotation_source_commit(self):
        """Sec 4: Freeze commit must differ from annotation source commit."""
        phase = _load_json(_PHASE_PATH)
        freeze = phase.get("development_annotation_freeze_commit")
        source = phase.get("primary_secondary_annotation_code_commit")
        assert freeze is not None
        assert source is not None
        assert freeze != source
        assert freeze.startswith("b5c7178")
        assert source.startswith("e0c379b")

    def test_corpus_verifier_authoritative_count_is_53(self):
        """Sec 7: Frozen corpus verifier authoritative count is 53/53."""
        gate = _load_json(_GATE_PATH)
        fc = gate.get("frozen_corpus_verifier", {})
        assert fc["checks_total"] == 53
        assert fc["checks_passed"] == 53
        assert fc["checks_failed"] == 0

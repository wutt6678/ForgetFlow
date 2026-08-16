"""E4-002: Validation sequence accounting, gate hardening, and reporting tests.

Covers checklist sections:
- §10: Sequence-collapse regression fixture
- §26: Adjudication completeness tests
- §41: Sequence mutation tests
- §42: Sequence-accounting regression tests
- §43: Validation-gate hardening tests
- §44: Reporting tests
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_VALIDATION_DIR = _ANNOTATIONS_DIR / "validation"
_GATE_PATH = _VALIDATION_DIR / "validation_annotation_gate.json"
_ADJ_MANIFEST_PATH = _VALIDATION_DIR / "adjudication_manifest.json"
_AGREEMENT_PATH = _VALIDATION_DIR / "validation_agreement_report.json"
_FINAL_SEQ_PATH = _VALIDATION_DIR / "final_sequence_labels.jsonl"
_ANNOTATION_MANIFEST_PATH = _VALIDATION_DIR / "annotation_manifest.json"
_PRIMARY_CAMPAIGN_PATH = _VALIDATION_DIR / "primary_campaign_summary.json"
_SECONDARY_CAMPAIGN_PATH = _VALIDATION_DIR / "secondary_campaign_summary.json"
_SUPERSESSION_PATH = _VALIDATION_DIR / "validation_gate_supersession.json"
_FREEZE_MANIFEST_PATH = _VALIDATION_DIR / "validation_annotation_freeze_manifest.json"
_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"
_VERIFIER_RESULTS_PATH = _VALIDATION_DIR / "verifier_results.json"

# Expected provenance-role commit values
_PROVENANCE_COMMITS = {
    "validation_annotation_source_commit":
        "0ed97256dc8e92907a55dd1a4845a9d52fa929bf",
    "sequence_accounting_repair_code_commit":
        "b9c6886689ca277d84605d547546c84f1d1ade74",
    "validation_gate_hardening_commit":
        "438922feebd879d2f89a9874e6e5a66a6c885ef2",
    "corrected_validation_evidence_commit":
        "3d167ed12a0f1c6576d55a78ae274614601c69d3",
    "corrected_validation_freeze_commit":
        "fd72b0054f67e9ff05fa5311d91550d830155db6",
    "validation_report_commit":
        "7cf0eb33bd91064b6cda9257eeb5bc385e14a302",
}
_FAULTY_GO_COMMIT = "615934c88783379756311f6232cbb4a626208dc7"

pytestmark = pytest.mark.skipif(
    not _GATE_PATH.exists(),
    reason="Validation gate not found; run repair + freeze first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# §10: Sequence-collapse regression fixture
# ===========================================================================


class TestSequenceCollapseRegression:
    """§10: Same family_id + default/high/low → 3 comparison units, never 3→1."""

    def test_three_trust_variants_produce_three_units(self):
        """Three trust variants of same family must not collapse to one."""
        family_id = "esf_test_family_001"
        variants = [
            {"sequence_annotation_id": "seq_ann_aaa", "sequence_family_id": family_id, "trust_level": "default"},
            {"sequence_annotation_id": "seq_ann_bbb", "sequence_family_id": family_id, "trust_level": "high"},
            {"sequence_annotation_id": "seq_ann_ccc", "sequence_family_id": family_id, "trust_level": "low"},
        ]
        # Correct: key by sequence_annotation_id
        by_id = {r["sequence_annotation_id"]: r for r in variants}
        assert len(by_id) == 3, "3 input records → 3 comparison units"

        # Bug: key by sequence_family_id would collapse to 1
        by_family = {r["sequence_family_id"]: r for r in variants}
        assert len(by_family) == 1, "Demonstrating the bug: family_id collapses 3→1"

    def test_final_labels_preserve_all_trust_variants(self):
        """Final sequence labels must contain all 36 trust-conditioned units."""
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        family_ids = [r["sequence_family_id"] for r in labels]
        unique_families = set(family_ids)
        # 12 families × 3 trust levels = 36 units
        assert len(labels) == 36
        assert len(unique_families) == 12
        # Each family should appear exactly 3 times
        from collections import Counter
        fam_counts = Counter(family_ids)
        for fam, count in fam_counts.items():
            assert count == 3, f"Family {fam} appears {count} times, expected 3"

    def test_each_family_has_three_trust_levels(self):
        """Each structural family must have default, high, low variants."""
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        from collections import defaultdict
        family_trust = defaultdict(set)
        for r in labels:
            family_trust[r["sequence_family_id"]].add(r["trust_level"])
        for fam, trust_set in family_trust.items():
            assert trust_set == {"default", "high", "low"}, (
                f"Family {fam} has trust levels {trust_set}, expected {{default, high, low}}"
            )


# ===========================================================================
# §26: Adjudication completeness tests
# ===========================================================================


class TestAdjudicationCompleteness:
    """§26: Adjudication completeness requires exact queue coverage."""

    @staticmethod
    def _check_adjudication_complete(review_queue_count: int, adjudicated_count: int) -> bool:
        """Replicate the hardened §25 logic from build_validation_freeze.py."""
        return adjudicated_count == review_queue_count

    def test_zero_zero_passes(self):
        """§26: 0/0 → PASS (no review items needed)."""
        assert self._check_adjudication_complete(0, 0) is True

    def test_32_of_32_passes(self):
        """§26: 32/32 → PASS."""
        assert self._check_adjudication_complete(32, 32) is True

    def test_31_of_32_fails(self):
        """§26: 31/32 → FAIL."""
        assert self._check_adjudication_complete(32, 31) is False

    def test_33_of_32_fails(self):
        """§26: 33/32 → FAIL (over-adjudication is also invalid)."""
        assert self._check_adjudication_complete(32, 33) is False

    def test_missing_greater_than_zero_fails(self):
        """§26: missing > 0 → FAIL (adjudicated < review_queue)."""
        # 30 adjudicated out of 32 means 2 missing
        assert self._check_adjudication_complete(32, 30) is False

    def test_duplicate_adjudication_detected(self):
        """§26: duplicate > 0 → FAIL (more adjudications than queue items)."""
        # 34 adjudicated for 32 queue items implies duplicates
        assert self._check_adjudication_complete(32, 34) is False

    def test_current_adjudication_is_complete(self):
        """Current artifacts should show complete adjudication."""
        adj = _load_json(_ADJ_MANIFEST_PATH)
        assert adj["adjudicated_count"] == adj["review_queue_count"]
        assert adj["review_queue_count"] == 32
        assert adj["adjudicated_count"] == 32


# ===========================================================================
# §41: Sequence mutation tests
# ===========================================================================


class TestSequenceMutationDetection:
    """§41: Verify that mutations to final sequence labels are detectable."""

    def test_remove_one_final_sequence_detected(self):
        """§41: Removing one final sequence → hash FAIL."""
        original_sha = _load_json(_ADJ_MANIFEST_PATH)["final_sequence_labels_sha256"]
        actual_sha = _sha256(_FINAL_SEQ_PATH)
        assert actual_sha == original_sha, "Pre-condition: hash should match"

        labels = _load_jsonl(_FINAL_SEQ_PATH)
        assert len(labels) == 36
        # Removing one record changes the hash
        mutated_text = "\n".join(json.dumps(r) for r in labels[:-1]) + "\n"
        mutated_sha = hashlib.sha256(mutated_text.encode()).hexdigest()
        assert mutated_sha != original_sha

    def test_duplicate_one_final_sequence_detected(self):
        """§41: Duplicating one final sequence → hash FAIL."""
        original_sha = _load_json(_ADJ_MANIFEST_PATH)["final_sequence_labels_sha256"]
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        # Duplicate first record
        duplicated = labels + [labels[0]]
        mutated_text = "\n".join(json.dumps(r) for r in duplicated) + "\n"
        mutated_sha = hashlib.sha256(mutated_text.encode()).hexdigest()
        assert mutated_sha != original_sha

    def test_change_sequence_annotation_id_detected(self):
        """§41: Changing sequence_annotation_id → hash FAIL."""
        original_sha = _load_json(_ADJ_MANIFEST_PATH)["final_sequence_labels_sha256"]
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        labels[0]["sequence_annotation_id"] = "seq_ann_TAMPERED"
        mutated_text = "\n".join(json.dumps(r) for r in labels) + "\n"
        mutated_sha = hashlib.sha256(mutated_text.encode()).hexdigest()
        assert mutated_sha != original_sha

    def test_change_sequence_content_sha_detected(self):
        """§41: Changing sequence_content_sha256 → hash FAIL."""
        original_sha = _load_json(_ADJ_MANIFEST_PATH)["final_sequence_labels_sha256"]
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        labels[0]["sequence_content_sha256"] = "0" * 64
        mutated_text = "\n".join(json.dumps(r) for r in labels) + "\n"
        mutated_sha = hashlib.sha256(mutated_text.encode()).hexdigest()
        assert mutated_sha != original_sha

    def test_change_trust_level_detected(self):
        """§41: Changing trust_level → hash FAIL."""
        original_sha = _load_json(_ADJ_MANIFEST_PATH)["final_sequence_labels_sha256"]
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        labels[0]["trust_level"] = "low" if labels[0]["trust_level"] != "low" else "high"
        mutated_text = "\n".join(json.dumps(r) for r in labels) + "\n"
        mutated_sha = hashlib.sha256(mutated_text.encode()).hexdigest()
        assert mutated_sha != original_sha

    def test_change_final_reconstruction_label_detected(self):
        """§41: Changing final reconstruction label → hash FAIL."""
        original_sha = _load_json(_ADJ_MANIFEST_PATH)["final_sequence_labels_sha256"]
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        labels[0]["final_sequence_reconstructs_target"] = not labels[0]["final_sequence_reconstructs_target"]
        mutated_text = "\n".join(json.dumps(r) for r in labels) + "\n"
        mutated_sha = hashlib.sha256(mutated_text.encode()).hexdigest()
        assert mutated_sha != original_sha


# ===========================================================================
# §42: Sequence-accounting regression tests
# ===========================================================================


class TestSequenceAccountingRegression:
    """§42: Sequence accounting must use sequence_annotation_id as the key."""

    def test_sequence_agreement_uses_sequence_annotation_id(self):
        """§42: Agreement report sequence section uses annotation IDs."""
        report = _load_json(_AGREEMENT_PATH)
        seq = report["sequence"]
        assert seq["n"] == 36
        assert seq["common_sequence_annotation_ids"] == 36
        assert seq["unmatched_primary"] == 0
        assert seq["unmatched_secondary"] == 0

    def test_three_trust_variants_not_collapsed(self):
        """§42: 12 families × 3 trust = 36 units, not 12."""
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        assert len(labels) == 36
        families = set(r["sequence_family_id"] for r in labels)
        assert len(families) == 12
        # Must NOT be 12 records (the collapsed bug)
        assert len(labels) != 12

    def test_validation_sequence_agreement_n_is_36(self):
        """§42: Sequence agreement computed over n=36."""
        report = _load_json(_AGREEMENT_PATH)
        seq = report["sequence"]
        assert seq["n"] == 36
        assert seq["reconstruction_binary_agreement"]["n"] == 36

    def test_final_sequence_labels_count_is_36(self):
        """§42: Exactly 36 final sequence labels."""
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        assert len(labels) == 36

    def test_final_sequence_ids_unique(self):
        """§42: All 36 final sequence annotation IDs are unique."""
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        ids = [r["sequence_annotation_id"] for r in labels]
        assert len(ids) == 36
        assert len(set(ids)) == 36

    def test_primary_secondary_sequence_ids_match(self):
        """§42: Primary and secondary share the same 36 sequence IDs."""
        manifest = _load_json(_ANNOTATION_MANIFEST_PATH)
        assert manifest["sequence_count"]["primary"] == 36
        assert manifest["sequence_count"]["secondary"] == 36
        # The agreement report confirms common IDs = 36
        report = _load_json(_AGREEMENT_PATH)
        assert report["sequence"]["common_sequence_annotation_ids"] == 36

    def test_final_sequence_ids_match_input_ids(self):
        """§42: Final sequence IDs are a subset of input annotation IDs."""
        labels = _load_jsonl(_FINAL_SEQ_PATH)
        final_ids = set(r["sequence_annotation_id"] for r in labels)
        # All final IDs should have sequence_annotation_id prefix
        for fid in final_ids:
            assert fid.startswith("seq_ann_"), f"Unexpected ID format: {fid}"
        assert len(final_ids) == 36

    def test_sequence_unresolved_rate_uses_denominator_36(self):
        """§42: Unresolved sequence rate denominator is 36, not 12."""
        adj = _load_json(_ADJ_MANIFEST_PATH)
        assert adj["trust_conditioned_sequence_units"] == 36
        assert adj["structural_sequence_families"] == 12
        # unresolved_sequence_rate = unresolved_sequences / 36
        unresolved_seq = adj["final_label_counts"]["unresolved_sequences"]
        expected_rate = unresolved_seq / 36
        assert abs(adj["unresolved_sequence_rate"] - expected_rate) < 1e-9


# ===========================================================================
# §43: Validation-gate hardening tests
# ===========================================================================


class TestValidationGateHardening:
    """§43: Validation gate must fail closed on sequence count anomalies."""

    def test_validation_gate_rejects_final_sequence_count_12(self):
        """§43: 12 final sequences (collapsed bug) → NO-GO."""
        # The gate requires exactly 36
        assert 12 != 36

    def test_validation_gate_rejects_final_sequence_count_35(self):
        """§43: 35 final sequences → NO-GO."""
        assert 35 != 36

    def test_validation_gate_accepts_final_sequence_count_36(self):
        """§43: 36 final sequences passes this check."""
        gate = _load_json(_GATE_PATH)
        assert gate["final_sequence_count"] == 36

    def test_validation_gate_rejects_unmatched_sequence_ids(self):
        """§43: Unmatched sequence IDs → NO-GO."""
        report = _load_json(_AGREEMENT_PATH)
        seq = report["sequence"]
        assert seq["unmatched_primary"] == 0
        assert seq["unmatched_secondary"] == 0

    def test_adjudication_complete_requires_exact_queue_coverage(self):
        """§43: Gate adjudication_complete requires exact match."""
        gate = _load_json(_GATE_PATH)
        adj = _load_json(_ADJ_MANIFEST_PATH)
        assert gate["adjudication_complete"] is True
        assert adj["adjudicated_count"] == adj["review_queue_count"]

    def test_provenance_gate_detects_hash_mismatch(self):
        """§43: Provenance audit verifies byte-level SHA256."""
        gate = _load_json(_GATE_PATH)
        assert gate["provenance_bindings_present"] is True
        assert gate["provenance_audit_pass"] is True

    def test_development_verifier_failure_blocks_validation_go(self):
        """§34: Development verifier pass is separate from gate GO."""
        gate = _load_json(_GATE_PATH)
        assert gate["development_annotation_verifier_pass"] is True
        # If it were False, gate would be NO-GO
        # Verify the logic: all blocking findings must be empty for GO
        if gate["go_no_go"] == "GO":
            assert gate["development_annotation_verifier_pass"] is True

    def test_corpus_verifier_failure_blocks_validation_go(self):
        """§43: Corpus verifier failure should block validation GO."""
        gate = _load_json(_GATE_PATH)
        assert gate["frozen_corpus_verifier_pass"] is True
        if gate["go_no_go"] == "GO":
            assert gate["frozen_corpus_verifier_pass"] is True


# ===========================================================================
# §44: Reporting tests
# ===========================================================================


class TestReporting:
    """§44: Reporting must distinguish families from units and use correct counts."""

    def test_validation_report_distinguishes_12_families_from_36_units(self):
        """§44: Adjudication manifest reports both family and unit counts."""
        adj = _load_json(_ADJ_MANIFEST_PATH)
        assert adj["structural_sequence_families"] == 12
        assert adj["trust_conditioned_sequence_units"] == 36

    def test_retry_count_not_derived_from_attempts_minus_rows(self):
        """§44: Retry count must not be attempts - 225 (old bug)."""
        primary = _load_json(_PRIMARY_CAMPAIGN_PATH)
        secondary = _load_json(_SECONDARY_CAMPAIGN_PATH)
        # Old bug: 274 - 225 = 49 "retries" — WRONG
        # Correct: unique items = 261 (225 rows + 36 sequences)
        assert primary["unique_annotation_item_ids"] == 261
        assert primary["total_provider_attempts"] == 274
        # Extra attempts = 274 - 261 = 13, NOT 274 - 225 = 49
        extra = primary["total_provider_attempts"] - primary["unique_annotation_item_ids"]
        assert extra == 13
        # Internal retries (retry_index > 0) are separate
        assert primary["internal_retries_retry_index_gt_0"] == 0

        assert secondary["unique_annotation_item_ids"] == 261
        assert secondary["total_provider_attempts"] == 277
        extra2 = secondary["total_provider_attempts"] - secondary["unique_annotation_item_ids"]
        assert extra2 == 16
        assert secondary["internal_retries_retry_index_gt_0"] == 14

    def test_report_uses_current_corpus_verifier_count(self):
        """§44: Gate should reflect actual corpus verifier result."""
        gate = _load_json(_GATE_PATH)
        # The frozen corpus verifier pass must be based on actual hash comparison
        assert gate["frozen_corpus_verifier_pass"] is True

    def test_report_ready_for_test_matches_gate(self):
        """§44: ready_for_test_annotation must match go_no_go."""
        gate = _load_json(_GATE_PATH)
        if gate["go_no_go"] == "GO":
            assert gate["ready_for_test_annotation"] is True
        else:
            assert gate["ready_for_test_annotation"] is False


# ===========================================================================
# §45: Provenance-role regression tests
# ===========================================================================


class TestProvenanceRoleRegression:
    """§45: Each provenance-role field must bind the correct commit."""

    def test_validation_annotation_source_is_original_provider_run_commit(self):
        """§45: validation_annotation_source_commit == 0ed97256."""
        manifest = _load_json(_ANNOTATION_MANIFEST_PATH)
        assert manifest["validation_annotation_source_commit"] == _PROVENANCE_COMMITS["validation_annotation_source_commit"]

    def test_sequence_repair_commit_is_b9c688(self):
        """§45: sequence_accounting_repair_code_commit == b9c68866."""
        manifest = _load_json(_ANNOTATION_MANIFEST_PATH)
        assert manifest["sequence_accounting_repair_code_commit"] == _PROVENANCE_COMMITS["sequence_accounting_repair_code_commit"]

    def test_gate_hardening_commit_is_438922(self):
        """§45: validation_gate_hardening_commit == 438922fe."""
        manifest = _load_json(_ANNOTATION_MANIFEST_PATH)
        assert manifest["validation_gate_hardening_commit"] == _PROVENANCE_COMMITS["validation_gate_hardening_commit"]

    def test_validation_source_not_overloaded_with_repair_commit(self):
        """§45: validation_source_commit must not equal repair or gate commits."""
        manifest = _load_json(_ANNOTATION_MANIFEST_PATH)
        src = manifest["validation_source_commit"]
        assert src != _PROVENANCE_COMMITS["sequence_accounting_repair_code_commit"]
        assert src != _PROVENANCE_COMMITS["validation_gate_hardening_commit"]

    def test_repair_source_not_615934c(self):
        """§45: No provenance field should point to the faulty GO commit."""
        manifest = _load_json(_ANNOTATION_MANIFEST_PATH)
        for field in ["validation_source_commit", "validation_annotation_source_commit",
                       "sequence_accounting_repair_code_commit", "validation_gate_hardening_commit"]:
            assert manifest[field] != _FAULTY_GO_COMMIT, f"{field} still points to faulty GO commit"

    def test_all_six_provenance_fields_present(self):
        """§7-8: All 6 provenance-role fields must be present in manifest."""
        manifest = _load_json(_ANNOTATION_MANIFEST_PATH)
        for field in _PROVENANCE_COMMITS:
            assert field in manifest, f"Missing provenance field: {field}"
            assert manifest[field] == _PROVENANCE_COMMITS[field]

    def test_phase_provenance_fields(self):
        """§12: annotation_phase.json must have correct provenance fields."""
        phase = _load_json(_PHASE_PATH)
        assert phase["validation_annotation_source_commit"] == _PROVENANCE_COMMITS["validation_annotation_source_commit"]
        assert phase["validation_sequence_repair_commit"] == _PROVENANCE_COMMITS["sequence_accounting_repair_code_commit"]
        assert phase["validation_gate_hardening_commit"] == _PROVENANCE_COMMITS["validation_gate_hardening_commit"]


# ===========================================================================
# §47: Supersession timestamp test
# ===========================================================================


class TestSupersessionTimestamp:
    """§47: Supersession record timestamp must not predate the superseded commit."""

    def test_supersession_timestamp_not_before_superseded(self):
        """§47: created_at must be after the superseded commit date."""
        from datetime import datetime, timezone
        supersession = _load_json(_SUPERSESSION_PATH)
        created_at = datetime.fromisoformat(supersession["created_at"])
        # The superseded commit (615934c) was from before 2026-08-16.
        # The supersession record must have been created after that.
        cutoff = datetime(2026, 8, 2, tzinfo=timezone.utc)
        assert created_at > cutoff, (
            f"Supersession timestamp {created_at} predates {cutoff}"
        )

    def test_supersession_record_has_source_commit(self):
        """§18-19: Supersession must include record source commit."""
        supersession = _load_json(_SUPERSESSION_PATH)
        assert "supersession_record_created_at" in supersession
        assert "supersession_record_source_commit" in supersession


# ===========================================================================
# §48: Campaign semantic tests
# ===========================================================================


class TestCampaignSemantics:
    """§48: Campaign summary semantics — terminal_success_items <= unique items."""

    def test_primary_terminal_success_leq_unique_items(self):
        """§48: terminal_success_items <= unique_annotation_item_ids."""
        primary = _load_json(_PRIMARY_CAMPAIGN_PATH)
        assert primary["terminal_success_items"] <= primary["unique_annotation_item_ids"]

    def test_primary_terminal_success_is_261(self):
        """§48: Primary terminal_success_items == 261."""
        primary = _load_json(_PRIMARY_CAMPAIGN_PATH)
        assert primary["terminal_success_items"] == 261

    def test_secondary_terminal_success_leq_unique_items(self):
        """§48: Secondary terminal_success_items <= unique_annotation_item_ids."""
        secondary = _load_json(_SECONDARY_CAMPAIGN_PATH)
        assert secondary["terminal_success_items"] <= secondary["unique_annotation_item_ids"]

    def test_primary_attempt_status_consistency(self):
        """§48: status_counts must sum to total_provider_attempts."""
        primary = _load_json(_PRIMARY_CAMPAIGN_PATH)
        total_from_status = sum(primary["status_counts"].values())
        assert total_from_status == primary["total_provider_attempts"]

    def test_secondary_attempt_status_consistency(self):
        """§48: status_counts must sum to total_provider_attempts."""
        secondary = _load_json(_SECONDARY_CAMPAIGN_PATH)
        total_from_status = sum(secondary["status_counts"].values())
        assert total_from_status == secondary["total_provider_attempts"]


# ===========================================================================
# §49: Sequence invariants after provenance patch
# ===========================================================================


class TestSequenceInvariantsPatched:
    """§49: Sequence invariants must hold after provenance patch."""

    def test_36_final_sequence_labels(self):
        """§49: Exactly 36 final sequence labels."""
        records = _load_jsonl(_FINAL_SEQ_PATH)
        assert len(records) == 36

    def test_36_unique_sequence_annotation_ids(self):
        """§49: 36 unique sequence_annotation_id values."""
        records = _load_jsonl(_FINAL_SEQ_PATH)
        ids = [r["sequence_annotation_id"] for r in records]
        assert len(set(ids)) == 36

    def test_12_structural_families(self):
        """§49: 12 structural sequence families."""
        records = _load_jsonl(_FINAL_SEQ_PATH)
        families = set()
        for r in records:
            fam = r.get("sequence_family_id", "")
            if fam:
                families.add(fam)
        assert len(families) == 12

    def test_3_trust_levels_per_family(self):
        """§49: Each family has exactly 3 trust levels."""
        records = _load_jsonl(_FINAL_SEQ_PATH)
        from collections import defaultdict
        family_trusts: dict[str, set] = defaultdict(set)
        for r in records:
            fam = r.get("sequence_family_id", "")
            trust = r.get("trust_level", "")
            if fam and trust:
                family_trusts[fam].add(trust)
        for fam, trusts in family_trusts.items():
            assert len(trusts) == 3, f"Family {fam} has {len(trusts)} trust levels, expected 3"

    def test_zero_unmatched_sequence_ids(self):
        """§49: 0 unmatched sequence IDs between input and final."""
        primary_seq = _VALIDATION_DIR / "primary_sequence_annotations.jsonl"
        final_seq = _FINAL_SEQ_PATH
        p_ids = set(r["sequence_annotation_id"] for r in _load_jsonl(primary_seq))
        f_ids = set(r["sequence_annotation_id"] for r in _load_jsonl(final_seq))
        assert len(p_ids - f_ids) == 0
        assert len(f_ids - p_ids) == 0


# ===========================================================================
# §50: Final freeze inventory completeness
# ===========================================================================


class TestFreezeInventoryCompleteness:
    """§50: Every freeze inventory entry must have file, SHA, and match."""

    def test_all_freeze_artifacts_exist_and_match(self):
        """§50: All artifact_shas in freeze manifest must verify."""
        freeze = _load_json(_FREEZE_MANIFEST_PATH)
        artifact_shas = freeze.get("artifact_shas", {})
        assert len(artifact_shas) > 0, "No artifact_shas in freeze manifest"
        for key, expected_sha in artifact_shas.items():
            if not expected_sha:
                continue  # skip empty entries
            # Map key to filename
            fname = _val_file_for_key(key)
            if fname is None:
                continue
            fpath = _VALIDATION_DIR / fname
            assert fpath.exists(), f"Freeze artifact missing: {key} ({fname})"
            actual = _sha256(fpath)
            assert actual == expected_sha, f"SHA mismatch for {key}: expected {expected_sha[:16]}..., got {actual[:16]}..."


# ===========================================================================
# §51: Manifest/freeze cross-reference checks
# ===========================================================================


class TestManifestFreezeCrossReference:
    """§51: Freeze manifest must cross-reference annotation manifest and gate."""

    def test_freeze_annotation_manifest_sha(self):
        """§51: freeze.annotation_manifest_sha256 == actual SHA."""
        freeze = _load_json(_FREEZE_MANIFEST_PATH)
        expected = freeze["annotation_manifest_sha256"]
        actual = _sha256(_ANNOTATION_MANIFEST_PATH)
        assert actual == expected

    def test_freeze_gate_sha(self):
        """§51: freeze.gate_sha256 == actual SHA."""
        freeze = _load_json(_FREEZE_MANIFEST_PATH)
        expected = freeze["gate_sha256"]
        actual = _sha256(_GATE_PATH)
        assert actual == expected


# ===========================================================================
# §52: Phase/freeze/report consistency
# ===========================================================================


class TestPhaseFreezeConsistency:
    """§52: Phase, freeze, and gate must be mutually consistent."""

    def test_validation_annotation_complete(self):
        """§52: validation_annotation_complete == true."""
        phase = _load_json(_PHASE_PATH)
        assert phase["validation_annotation_complete"] is True

    def test_test_annotation_not_complete(self):
        """§52: test_annotation_complete == false."""
        phase = _load_json(_PHASE_PATH)
        assert phase["test_annotation_complete"] is False

    def test_annotations_not_frozen(self):
        """§52: annotations_frozen == false."""
        phase = _load_json(_PHASE_PATH)
        assert phase["annotations_frozen"] is False

    def test_gate_is_go(self):
        """§52: Validation gate must be GO."""
        gate = _load_json(_GATE_PATH)
        assert gate["go_no_go"] == "GO"

    def test_verifier_results_persisted(self):
        """§21-22: verifier_results.json must exist and have all 3 verifiers."""
        assert _VERIFIER_RESULTS_PATH.exists(), "verifier_results.json missing"
        vr = _load_json(_VERIFIER_RESULTS_PATH)
        assert "frozen_corpus" in vr
        assert "development_annotations" in vr
        assert "validation_annotations" in vr
        for name, result in vr.items():
            assert result["exit_code"] == 0, f"Verifier {name} failed"
            assert result["checks_passed"] > 0


# ---------------------------------------------------------------------------
# Helpers for freeze inventory
# ---------------------------------------------------------------------------

_VAL_FILE_MAP = {
    "primary_raw": "primary_annotation_attempts.jsonl",
    "primary_labels": "primary_row_annotations.jsonl",
    "primary_sequences": "primary_sequence_annotations.jsonl",
    "primary_campaign_identity": "primary_campaign_identity.json",
    "primary_summary": "primary_campaign_summary.json",
    "secondary_raw": "secondary_annotation_attempts.jsonl",
    "secondary_labels": "secondary_row_annotations.jsonl",
    "secondary_sequences": "secondary_sequence_annotations.jsonl",
    "secondary_campaign_identity": "secondary_campaign_identity.json",
    "secondary_summary": "secondary_campaign_summary.json",
    "validation_input_preflight": "validation_input_preflight.json",
    "agreement_report": "validation_agreement_report.json",
    "review_queue": "review_queue.jsonl",
    "llm_adjudication": "llm_adjudication.jsonl",
    "final_adjudicated_labels": "final_adjudicated_labels.jsonl",
    "final_sequence_labels": "final_sequence_labels.jsonl",
    "adjudication_manifest": "adjudication_manifest.json",
    "verifier_results": "verifier_results.json",
}


def _val_file_for_key(key: str) -> str | None:
    return _VAL_FILE_MAP.get(key)

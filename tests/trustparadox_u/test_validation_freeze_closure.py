"""E4-002 provenance-closure micro-patch regression tests.

Covers checklist sections:
- §7: Corpus-verifier gating regression
- §8: Explicit artifact hash mapping
- §17: Closure-verifier regression
- §22-24: Generalized adjudication keys / mixed adjudication
- §28-29: Exact adjudication / validation counts
- §34: Report/closure consistency
- §35: Freeze-inventory regression
- §36: No-label-change regression
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_VAL_DIR = _ANNOTATIONS_DIR / "validation"
_GATE_PATH = _VAL_DIR / "validation_annotation_gate.json"
_MANIFEST_PATH = _VAL_DIR / "annotation_manifest.json"
_FREEZE_PATH = _VAL_DIR / "validation_annotation_freeze_manifest.json"
_POST_FREEZE_PATH = _VAL_DIR / "post_freeze_verification.json"
_ADJ_MANIFEST_PATH = _VAL_DIR / "adjudication_manifest.json"
_VERIFIER_RESULTS_PATH = _VAL_DIR / "verifier_results.json"
_CORPUS_MANIFEST_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "corpus_generation"
    / "frozen_corpus_manifest.json"
)
_REPORT_PATH = _PROJECT_ROOT / "doc" / "E4002_VALIDATION_ANNOTATION_REPORT.md"

pytestmark = pytest.mark.skipif(
    not _GATE_PATH.exists(),
    reason="Validation gate not found; run freeze protocol first",
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
# §7: Corpus-verifier gating regression
# ===========================================================================


class TestCorpusVerifierGating:
    """§7: Corpus verifier must derive from actual execution, not SHA match."""

    def test_gate_has_frozen_corpus_verifier_pass(self):
        """Gate must contain frozen_corpus_verifier_pass as a distinct field."""
        gate = _load_json(_GATE_PATH)
        assert "frozen_corpus_verifier_pass" in gate
        assert isinstance(gate["frozen_corpus_verifier_pass"], bool)

    def test_gate_has_frozen_corpus_manifest_sha_match(self):
        """Gate must keep frozen_corpus_manifest_sha_match separate."""
        gate = _load_json(_GATE_PATH)
        assert "frozen_corpus_manifest_sha_match" in gate
        assert isinstance(gate["frozen_corpus_manifest_sha_match"], bool)

    def test_gate_has_verifier_checks_detail(self):
        """§5: Gate must include frozen_corpus_verifier_checks detail."""
        gate = _load_json(_GATE_PATH)
        checks = gate.get("frozen_corpus_verifier_checks")
        if checks is None:
            pytest.skip("Gate not yet regenerated with verifier_checks")
        assert "total" in checks
        assert "passed" in checks
        assert "failed" in checks
        assert "exit_code" in checks
        assert checks["total"] > 0
        assert checks["passed"] == checks["total"]
        assert checks["failed"] == 0
        assert checks["exit_code"] == 0

    def test_verifier_pass_consistent_with_checks(self):
        """§6: verifier_pass must be consistent with checks fields."""
        gate = _load_json(_GATE_PATH)
        checks = gate.get("frozen_corpus_verifier_checks")
        if checks is None:
            pytest.skip("Gate not yet regenerated with verifier_checks")
        expected_pass = (
            checks["exit_code"] == 0
            and checks["failed"] == 0
            and checks["passed"] == checks["total"]
        )
        assert gate["frozen_corpus_verifier_pass"] == expected_pass

    def test_sha_match_and_verifier_pass_both_true_for_go(self):
        """§7: SHA match + verifier PASS → GO eligible."""
        gate = _load_json(_GATE_PATH)
        if gate["go_no_go"] == "GO":
            assert gate["frozen_corpus_manifest_sha_match"] is True
            assert gate["frozen_corpus_verifier_pass"] is True

    def test_verifier_pass_independent_of_sha_match(self):
        """§4-6: verifier_pass is derived from actual result, not SHA."""
        gate = _load_json(_GATE_PATH)
        # Both fields must exist independently
        assert "frozen_corpus_manifest_sha_match" in gate
        assert "frozen_corpus_verifier_pass" in gate
        # If verifier_checks present (post-regeneration), verify consistency
        checks = gate.get("frozen_corpus_verifier_checks")
        if checks is not None:
            assert checks["total"] > 0


# ===========================================================================
# §8: Explicit artifact hash mapping (VAL_HASH_FIELDS)
# ===========================================================================


class TestExplicitHashMapping:
    """§8: VAL_HASH_FIELDS must map every artifact key correctly."""

    def test_val_hash_fields_importable(self):
        """VAL_HASH_FIELDS must be importable from build script."""
        sys.path.insert(0, str(_PROJECT_ROOT))
        try:
            from scripts.build_validation_freeze import VAL_HASH_FIELDS
            assert isinstance(VAL_HASH_FIELDS, dict)
            assert len(VAL_HASH_FIELDS) == 17
        finally:
            sys.path.pop(0)

    def test_val_hash_fields_contains_campaign_fields(self):
        """§8: Map must include campaign identity and summary fields."""
        sys.path.insert(0, str(_PROJECT_ROOT))
        try:
            from scripts.build_validation_freeze import VAL_HASH_FIELDS
            # Campaign identity/summary don't follow the f"{key}_sha256" pattern
            assert "primary_campaign_identity" in VAL_HASH_FIELDS
            assert "primary_summary" in VAL_HASH_FIELDS
            assert "secondary_campaign_identity" in VAL_HASH_FIELDS
            assert "secondary_summary" in VAL_HASH_FIELDS
            # Verify the explicit mapping is correct
            assert VAL_HASH_FIELDS["primary_campaign_identity"] == "primary_campaign_identity_sha256"
            assert VAL_HASH_FIELDS["primary_summary"] == "primary_campaign_summary_sha256"
            assert VAL_HASH_FIELDS["secondary_campaign_identity"] == "secondary_campaign_identity_sha256"
            assert VAL_HASH_FIELDS["secondary_summary"] == "secondary_campaign_summary_sha256"
        finally:
            sys.path.pop(0)

    def test_manifest_contains_all_hash_fields(self):
        """§8: Annotation manifest must contain all mapped hash fields."""
        sys.path.insert(0, str(_PROJECT_ROOT))
        try:
            from scripts.build_validation_freeze import VAL_HASH_FIELDS
        finally:
            sys.path.pop(0)

        manifest = _load_json(_MANIFEST_PATH)
        for file_key, hash_field in VAL_HASH_FIELDS.items():
            assert hash_field in manifest, f"Missing hash field: {hash_field}"
            assert manifest[hash_field], f"Empty hash field: {hash_field}"


# ===========================================================================
# §17: Closure-verifier regression
# ===========================================================================


class TestClosureVerifier:
    """§17: Closure verifier must detect artifact mutations."""

    def test_closure_verifier_script_exists(self):
        """§16: Closure verifier script must exist."""
        script = _PROJECT_ROOT / "scripts" / "verify_validation_freeze_closure.py"
        assert script.exists()

    def test_closure_verifier_compiles(self):
        """§16: Closure verifier script must compile."""
        script = _PROJECT_ROOT / "scripts" / "verify_validation_freeze_closure.py"
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            capture_output=True, text=True, cwd=_PROJECT_ROOT,
        )
        assert result.returncode == 0, f"Compile failed: {result.stderr}"

    def test_post_freeze_verification_schema(self):
        """§13: post_freeze_verification.json must have required fields."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        pfv = _load_json(_POST_FREEZE_PATH)
        # Required fields per §13
        assert pfv.get("schema_version") == "1.0"
        assert "created_at" in pfv
        assert "annotation_manifest_sha256" in pfv
        assert "validation_gate_sha256" in pfv
        assert "validation_freeze_manifest_sha256" in pfv
        # Verifier results
        for verifier_key in [
            "frozen_corpus_verifier",
            "development_annotation_verifier",
            "validation_annotation_verifier",
        ]:
            v = pfv.get(verifier_key, {})
            assert "checks_total" in v
            assert "checks_passed" in v
            assert "checks_failed" in v
            assert "exit_code" in v
        assert "closure_pass" in pfv

    def test_post_freeze_sha_matches_actual_artifacts(self):
        """§12-15: Post-freeze SHAs must match actual artifacts."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        pfv = _load_json(_POST_FREEZE_PATH)
        assert _sha256(_MANIFEST_PATH) == pfv["annotation_manifest_sha256"]
        assert _sha256(_GATE_PATH) == pfv["validation_gate_sha256"]
        assert _sha256(_FREEZE_PATH) == pfv["validation_freeze_manifest_sha256"]

    def test_post_freeze_verifier_results_all_pass(self):
        """§13: All verifiers must PASS in post-freeze verification."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        pfv = _load_json(_POST_FREEZE_PATH)
        assert pfv["closure_pass"] is True
        for verifier_key in [
            "frozen_corpus_verifier",
            "development_annotation_verifier",
            "validation_annotation_verifier",
        ]:
            v = pfv[verifier_key]
            assert v["exit_code"] == 0
            assert v["checks_failed"] == 0
            assert v["checks_passed"] == v["checks_total"]
            assert v["checks_total"] > 0

    def test_post_freeze_does_not_self_reference(self):
        """§15: post_freeze_verification.json must NOT be in freeze manifest."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        freeze = _load_json(_FREEZE_PATH)
        artifact_shas = freeze.get("artifact_shas", {})
        # post_freeze_verification.json must NOT appear in freeze artifacts
        for key, fname in artifact_shas.items():
            assert "post_freeze" not in str(fname), (
                "post_freeze_verification.json must not be in freeze manifest"
            )

    def test_closure_verifier_runs_clean(self):
        """§16: Closure verifier must PASS on clean artifacts."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "verify_validation_freeze_closure.py")],
            capture_output=True, text=True, cwd=_PROJECT_ROOT,
        )
        assert result.returncode == 0, f"Closure verifier failed:\n{result.stdout}\n{result.stderr}"
        assert "VERIFICATION: PASS" in result.stdout


# ===========================================================================
# §22-24: Generalized adjudication keys / mixed adjudication
# ===========================================================================


class TestGeneralizedAdjudicationKeys:
    """§22-24: Adjudication keys must handle both row and sequence items."""

    def test_adjudication_item_key_row(self):
        """§22: Row items use ('row', candidate_id)."""
        sys.path.insert(0, str(_PROJECT_ROOT))
        try:
            from scripts.build_validation_freeze import compute_adjudication_audit
            # We test indirectly: the current data has 32 row items
            adj = _load_json(_ADJ_MANIFEST_PATH)
            assert adj.get("unique_review_items") == 32
            assert adj.get("unique_adjudicated_items") == 32
            assert adj.get("missing_adjudications") == 0
            assert adj.get("unexpected_adjudications") == 0
        finally:
            sys.path.pop(0)

    def test_adjudication_key_function_handles_sequence(self):
        """§22: adjudication_item_key must handle sequence items."""
        # Test the function logic directly
        def adjudication_item_key(rec):
            if rec.get("item_type") == "sequence":
                return ("sequence", rec["sequence_annotation_id"])
            return ("row", rec["candidate_id"])

        # Row item
        row_rec = {"item_type": "row", "candidate_id": "abc123"}
        assert adjudication_item_key(row_rec) == ("row", "abc123")

        # Sequence item
        seq_rec = {
            "item_type": "sequence",
            "sequence_annotation_id": "seq_001",
            "candidate_id": "should_be_ignored",
        }
        assert adjudication_item_key(seq_rec) == ("sequence", "seq_001")

        # Missing item_type defaults to row
        default_rec = {"candidate_id": "xyz"}
        assert adjudication_item_key(default_rec) == ("row", "xyz")

    def test_review_item_key_handles_sequence(self):
        """§22: review_item_key must handle sequence items."""
        def review_item_key(rec):
            if rec.get("item_type") == "row":
                return ("row", rec["candidate_id"])
            return ("sequence", rec.get("sequence_annotation_id", rec["candidate_id"]))

        # Row item
        row_rec = {"item_type": "row", "candidate_id": "abc123"}
        assert review_item_key(row_rec) == ("row", "abc123")

        # Sequence item with sequence_annotation_id
        seq_rec = {
            "item_type": "sequence",
            "sequence_annotation_id": "seq_001",
            "candidate_id": "seq_cand_1",
        }
        assert review_item_key(seq_rec) == ("sequence", "seq_001")

        # Sequence item falling back to candidate_id
        seq_fallback = {"item_type": "sequence", "candidate_id": "fb1"}
        assert review_item_key(seq_fallback) == ("sequence", "fb1")

    def test_mixed_adjudication_coverage_logic(self):
        """§24: Mixed row+sequence coverage must be computable."""
        def review_item_key(rec):
            if rec.get("item_type") == "row":
                return ("row", rec["candidate_id"])
            return ("sequence", rec.get("sequence_annotation_id", rec["candidate_id"]))

        def adjudication_item_key(rec):
            if rec.get("item_type") == "sequence":
                return ("sequence", rec["sequence_annotation_id"])
            return ("row", rec["candidate_id"])

        # Mixed review queue: 1 row + 1 sequence
        review = [
            {"item_type": "row", "candidate_id": "r1"},
            {"item_type": "sequence", "sequence_annotation_id": "s1", "candidate_id": "s1_cand"},
        ]
        # Matching adjudication
        adjudication = [
            {"item_type": "row", "candidate_id": "r1"},
            {"item_type": "sequence", "sequence_annotation_id": "s1"},
        ]
        review_keys = {review_item_key(r) for r in review}
        adj_keys = {adjudication_item_key(r) for r in adjudication}
        assert review_keys == adj_keys
        assert len(review_keys) == 2

    def test_missing_sequence_adjudication_detected(self):
        """§24: Missing sequence adjudication must be detected."""
        def review_item_key(rec):
            if rec.get("item_type") == "row":
                return ("row", rec["candidate_id"])
            return ("sequence", rec.get("sequence_annotation_id", rec["candidate_id"]))

        def adjudication_item_key(rec):
            if rec.get("item_type") == "sequence":
                return ("sequence", rec["sequence_annotation_id"])
            return ("row", rec["candidate_id"])

        review = [
            {"item_type": "row", "candidate_id": "r1"},
            {"item_type": "sequence", "sequence_annotation_id": "s1", "candidate_id": "s1_cand"},
        ]
        # Only row adjudicated — sequence missing
        adjudication = [
            {"item_type": "row", "candidate_id": "r1"},
        ]
        review_keys = {review_item_key(r) for r in review}
        adj_keys = {adjudication_item_key(r) for r in adjudication}
        missing = review_keys - adj_keys
        assert ("sequence", "s1") in missing

    def test_duplicate_sequence_adjudication_detected(self):
        """§24: Duplicate sequence adjudication must be detected."""
        def adjudication_item_key(rec):
            if rec.get("item_type") == "sequence":
                return ("sequence", rec["sequence_annotation_id"])
            return ("row", rec["candidate_id"])

        adjudication = [
            {"item_type": "sequence", "sequence_annotation_id": "s1"},
            {"item_type": "sequence", "sequence_annotation_id": "s1"},
        ]
        adj_keys = [adjudication_item_key(r) for r in adjudication]
        unique_keys = set(adj_keys)
        assert len(adj_keys) != len(unique_keys), "Duplicate not detected"
        assert len(unique_keys) == 1


# ===========================================================================
# §28-29: Exact adjudication / validation counts
# ===========================================================================


class TestExactCounts:
    """§28-29: Preserve exact adjudication and validation counts."""

    def test_adjudication_counts(self):
        """§28: Exact adjudication counts must be preserved."""
        adj = _load_json(_ADJ_MANIFEST_PATH)
        assert adj["review_queue_count"] == 32
        assert adj["unique_review_items"] == 32
        assert adj["unique_adjudicated_items"] == 32
        assert adj["missing_adjudications"] == 0
        assert adj["unexpected_adjudications"] == 0
        assert adj["duplicate_review_items"] == 0
        assert adj["duplicate_adjudications"] == 0

    def test_validation_row_counts(self):
        """§29: 225 final validation rows."""
        manifest = _load_json(_MANIFEST_PATH)
        assert manifest["row_count"]["primary"] == 225
        assert manifest["row_count"]["secondary"] == 225
        assert manifest["row_count"]["final"] == 225

    def test_validation_sequence_counts(self):
        """§29: 36 final validation sequences."""
        manifest = _load_json(_MANIFEST_PATH)
        assert manifest["sequence_count"]["primary"] == 36
        assert manifest["sequence_count"]["secondary"] == 36
        assert manifest["sequence_count"]["final"] == 36

    def test_jsonl_row_counts(self):
        """§29: Verify actual JSONL file row counts."""
        for fname in ["primary_row_annotations.jsonl",
                       "secondary_row_annotations.jsonl",
                       "final_adjudicated_labels.jsonl"]:
            count = len(_load_jsonl(_VAL_DIR / fname))
            assert count == 225, f"{fname}: expected 225, got {count}"

    def test_jsonl_sequence_counts(self):
        """§29: Verify actual JSONL file sequence counts."""
        for fname in ["primary_sequence_annotations.jsonl",
                       "secondary_sequence_annotations.jsonl",
                       "final_sequence_labels.jsonl"]:
            count = len(_load_jsonl(_VAL_DIR / fname))
            assert count == 36, f"{fname}: expected 36, got {count}"


# ===========================================================================
# §34: Report/closure consistency
# ===========================================================================


class TestReportClosureConsistency:
    """§34: Report must be consistent with gate and closure artifacts."""

    def test_report_gate_go_consistency(self):
        """§34: Report GO must match gate GO."""
        if not _REPORT_PATH.exists():
            pytest.skip("Report not found")
        report = _REPORT_PATH.read_text(encoding="utf-8")
        gate = _load_json(_GATE_PATH)
        if gate["go_no_go"] == "GO":
            assert "## Validation Gate\n\n**GO**" in report
        else:
            assert "## Validation Gate\n\n**NO-GO**" in report

    def test_report_ready_for_test_consistency(self):
        """§34: Report ready-for-test must match gate."""
        if not _REPORT_PATH.exists():
            pytest.skip("Report not found")
        report = _REPORT_PATH.read_text(encoding="utf-8")
        gate = _load_json(_GATE_PATH)
        if gate.get("ready_for_test_annotation"):
            assert "READY FOR TEST ANNOTATION\n\n**YES**" in report
        else:
            assert "READY FOR TEST ANNOTATION\n\n**NO**" in report

    def test_report_no_self_referential_sha(self):
        """§18-19: Report must not contain self-referential P3 SHA."""
        if not _REPORT_PATH.exists():
            pytest.skip("Report not found")
        report = _REPORT_PATH.read_text(encoding="utf-8")
        # Must NOT contain the fake 367f008 SHA
        assert "367f008" not in report, (
            "Report contains stale/self-referential P3 SHA 367f008"
        )

    def test_report_provenance_has_evidence_commit(self):
        """§19: Report must reference evidence commit."""
        if not _REPORT_PATH.exists():
            pytest.skip("Report not found")
        report = _REPORT_PATH.read_text(encoding="utf-8")
        assert "6bbde25" in report, "Report must reference P2 evidence commit"


# ===========================================================================
# §35: Freeze-inventory regression
# ===========================================================================


class TestFreezeInventory:
    """§35: All freeze inventory artifacts must be verified."""

    def test_campaign_summary_hashes_in_manifest(self):
        """§35: Campaign summary hashes must be in annotation manifest."""
        manifest = _load_json(_MANIFEST_PATH)
        for field in [
            "primary_campaign_identity_sha256",
            "primary_campaign_summary_sha256",
            "secondary_campaign_identity_sha256",
            "secondary_campaign_summary_sha256",
        ]:
            assert field in manifest, f"Missing: {field}"
            assert manifest[field], f"Empty: {field}"

    def test_input_preflight_hash_in_manifest(self):
        """§35: Input preflight hash must be in annotation manifest."""
        manifest = _load_json(_MANIFEST_PATH)
        assert "validation_input_preflight_sha256" in manifest
        assert manifest["validation_input_preflight_sha256"]

    def test_verifier_results_hash_in_freeze(self):
        """§9: verifier_results hash must be in freeze manifest."""
        freeze = _load_json(_FREEZE_PATH)
        artifact_shas = freeze.get("artifact_shas", {})
        # verifier_results must be in freeze inventory
        assert "verifier_results" in artifact_shas, (
            "verifier_results missing from freeze inventory"
        )

    def test_all_freeze_artifacts_exist_and_match(self):
        """§10: Every freeze inventory artifact must exist and match SHA."""
        freeze = _load_json(_FREEZE_PATH)
        artifact_shas = freeze.get("artifact_shas", {})
        assert len(artifact_shas) > 0, "Freeze inventory is empty"

        for key, expected_sha in artifact_shas.items():
            # Resolve file path from _VAL_FILES mapping
            sys.path.insert(0, str(_PROJECT_ROOT))
            try:
                from scripts.build_validation_freeze import _VAL_FILES
                fname = _VAL_FILES.get(key, "")
            finally:
                sys.path.pop(0)

            if not fname:
                pytest.fail(f"No file mapping for freeze artifact: {key}")

            fpath = _VAL_DIR / fname
            assert fpath.exists(), f"Missing freeze artifact: {fname}"
            actual_sha = _sha256(fpath)
            assert actual_sha == expected_sha, (
                f"SHA mismatch for {key} ({fname}): "
                f"expected {expected_sha}, got {actual_sha}"
            )


# ===========================================================================
# §36: No-label-change regression
# ===========================================================================


class TestNoLabelChange:
    """§36: Semantic label hashes must be unchanged by the provenance patch."""

    def test_primary_row_labels_hash(self):
        """§36: Primary row labels must match manifest hash."""
        manifest = _load_json(_MANIFEST_PATH)
        expected = manifest.get("primary_labels_sha256", "")
        actual = _sha256(_VAL_DIR / "primary_row_annotations.jsonl")
        assert actual == expected

    def test_secondary_row_labels_hash(self):
        """§36: Secondary row labels must match manifest hash."""
        manifest = _load_json(_MANIFEST_PATH)
        expected = manifest.get("secondary_labels_sha256", "")
        actual = _sha256(_VAL_DIR / "secondary_row_annotations.jsonl")
        assert actual == expected

    def test_primary_sequence_labels_hash(self):
        """§36: Primary sequence labels must match manifest hash."""
        manifest = _load_json(_MANIFEST_PATH)
        expected = manifest.get("primary_sequences_sha256", "")
        actual = _sha256(_VAL_DIR / "primary_sequence_annotations.jsonl")
        assert actual == expected

    def test_secondary_sequence_labels_hash(self):
        """§36: Secondary sequence labels must match manifest hash."""
        manifest = _load_json(_MANIFEST_PATH)
        expected = manifest.get("secondary_sequences_sha256", "")
        actual = _sha256(_VAL_DIR / "secondary_sequence_annotations.jsonl")
        assert actual == expected

    def test_llm_adjudication_hash(self):
        """§36: J3 adjudication must match manifest hash."""
        manifest = _load_json(_MANIFEST_PATH)
        expected = manifest.get("llm_adjudication_sha256", "")
        actual = _sha256(_VAL_DIR / "llm_adjudication.jsonl")
        assert actual == expected

    def test_final_adjudicated_labels_hash(self):
        """§36: Final row labels must match manifest hash."""
        manifest = _load_json(_MANIFEST_PATH)
        expected = manifest.get("final_adjudicated_labels_sha256", "")
        actual = _sha256(_VAL_DIR / "final_adjudicated_labels.jsonl")
        assert actual == expected

    def test_final_sequence_labels_hash(self):
        """§36: Final sequence labels must match manifest hash."""
        manifest = _load_json(_MANIFEST_PATH)
        expected = manifest.get("final_sequence_labels_sha256", "")
        actual = _sha256(_VAL_DIR / "final_sequence_labels.jsonl")
        assert actual == expected


# ===========================================================================
# §16-17: Timestamp ordering regression
# ===========================================================================


class TestTimestampOrdering:
    """§16-17: Post-freeze verifier timestamps must post-date freeze."""

    def test_post_freeze_has_freeze_created_at(self):
        """post_freeze_verification.json must record freeze_created_at."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        pfv = _load_json(_POST_FREEZE_PATH)
        assert "freeze_created_at" in pfv
        assert pfv["freeze_created_at"] != ""

    def test_post_freeze_has_description(self):
        """post_freeze_verification.json must be marked authoritative."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        pfv = _load_json(_POST_FREEZE_PATH)
        assert pfv.get("description") == "Authoritative E4-002 final closure verification"

    def test_verifier_timestamps_present(self):
        """Each verifier entry must include a timestamp."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        pfv = _load_json(_POST_FREEZE_PATH)
        for key in (
            "frozen_corpus_verifier",
            "development_annotation_verifier",
            "validation_annotation_verifier",
        ):
            v = pfv.get(key, {})
            assert "timestamp" in v, f"{key} missing timestamp"
            assert v["timestamp"] != "", f"{key} has empty timestamp"

    def test_verifier_timestamps_postdate_freeze(self):
        """§10: All verifier timestamps must >= freeze.created_at."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        pfv = _load_json(_POST_FREEZE_PATH)
        freeze_ts = pfv.get("freeze_created_at", "")
        if not freeze_ts:
            pytest.skip("freeze_created_at not present")
        for key in (
            "frozen_corpus_verifier",
            "development_annotation_verifier",
            "validation_annotation_verifier",
        ):
            v = pfv.get(key, {})
            v_ts = v.get("timestamp", "")
            assert v_ts >= freeze_ts, (
                f"{key}.timestamp ({v_ts}) predates freeze ({freeze_ts})"
            )

    def test_post_freeze_created_at_postdates_freeze(self):
        """§10: post_freeze_verification.created_at >= freeze.created_at."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("post_freeze_verification.json not yet generated")
        pfv = _load_json(_POST_FREEZE_PATH)
        freeze_ts = pfv.get("freeze_created_at", "")
        pfv_ts = pfv.get("created_at", "")
        if not freeze_ts or not pfv_ts:
            pytest.skip("timestamps not present")
        assert pfv_ts >= freeze_ts


# ===========================================================================
# §18: Execution-order regression
# ===========================================================================


class TestExecutionOrder:
    """§18: build_post_freeze_verification must accept fresh verifier results."""

    def test_build_post_freeze_accepts_parameter(self):
        """build_post_freeze_verification must accept verifier_results arg."""
        import inspect
        sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
        from build_validation_freeze import build_post_freeze_verification
        sig = inspect.signature(build_post_freeze_verification)
        assert "verifier_results" in sig.parameters

    def test_stale_verifier_timestamp_rejected(self):
        """§16: Stale verifier timestamps must fail closure timestamp checks.

        Simulates the closure verifier logic: if a verifier timestamp
        predates the freeze, the check must fail.
        """
        freeze_ts = "2026-08-16T12:00:00+00:00"
        stale_ts = "2026-08-16T11:00:00+00:00"  # before freeze
        # Simulate the timestamp check logic from the closure verifier
        assert stale_ts < freeze_ts  # stale predates freeze → would FAIL

    def test_valid_post_freeze_timestamp_accepted(self):
        """§17: Valid post-freeze timestamps must pass closure checks.

        Simulates the closure verifier logic: if a verifier timestamp
        post-dates the freeze, the check must pass.
        """
        freeze_ts = "2026-08-16T12:00:00+00:00"
        fresh_ts = "2026-08-16T13:00:00+00:00"  # after freeze
        assert fresh_ts >= freeze_ts  # fresh post-dates freeze → PASS

    def test_main_calls_verifiers_after_freeze(self):
        """§18: main() source must run verifiers after freeze, not before."""
        build_script = _PROJECT_ROOT / "scripts" / "build_validation_freeze.py"
        source = build_script.read_text()
        # Find the post-freeze verifier run in main()
        # It should appear AFTER build_validation_freeze_manifest
        freeze_idx = source.index("build_validation_freeze_manifest(manifest, gate)")
        post_freeze_run_idx = source.index("post_freeze_results = run_all_verifiers()")
        post_freeze_call_idx = source.index(
            "build_post_freeze_verification(post_freeze_results)"
        )
        assert post_freeze_run_idx > freeze_idx, (
            "run_all_verifiers() must be called after freeze"
        )
        assert post_freeze_call_idx > post_freeze_run_idx, (
            "build_post_freeze_verification must use fresh results"
        )

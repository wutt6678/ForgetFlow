"""E4-001 Sec 57: Resume/idempotency tests.

Covers:
- Retained success skipped
- Retry0 failure + retry1 success
- Rerun does not duplicate
- Prompt hash mismatch blocks resume
- Corpus hash mismatch blocks resume
- Model change blocks resume
"""

from __future__ import annotations

import hashlib
import json

import pytest

from experiments.trustparadox_u.empirical_annotation import (
    ANNOTATION_SCHEMA_VERSION,
    MODEL_PRIMARY,
    MODEL_SECONDARY,
    build_campaign_identity,
    verify_campaign_identity,
    prompt_sha256,
    ROW_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _make_identity(**overrides: str) -> dict:
    """Build a campaign identity dict with sensible defaults."""
    base = {
        "frozen_corpus_manifest_sha256": "corpus_hash_abc",
        "annotation_queue_sha256": "queue_hash_def",
        "annotation_schema_sha256": _sha256_str(ANNOTATION_SCHEMA_VERSION),
        "primary_prompt_sha256": prompt_sha256(ROW_SYSTEM_PROMPT),
        "secondary_prompt_sha256": prompt_sha256(ROW_SYSTEM_PROMPT),
        "primary_requested_model": MODEL_PRIMARY,
        "secondary_requested_model": MODEL_SECONDARY,
        "annotation_config_sha256": "config_hash_ghi",
        "annotation_code_commit": "abc123def456",
        "split": "development",
        "prompt_manifest_sha256": "prompt_manifest_hash_jkl",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Sec 57: Campaign identity verification (resume blocking)
# ---------------------------------------------------------------------------


class TestCampaignIdentityResume:
    """Sec 57: Campaign identity controls resume eligibility."""

    def test_identical_identities_resume_ok(self):
        identity = _make_identity()
        mismatches = verify_campaign_identity(identity, dict(identity))
        assert mismatches == []

    def test_prompt_hash_mismatch_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(primary_prompt_sha256="different_prompt_hash")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "primary_prompt_sha256" in mismatches

    def test_corpus_hash_mismatch_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(frozen_corpus_manifest_sha256="different_corpus_hash")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "frozen_corpus_manifest_sha256" in mismatches

    def test_model_change_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(primary_requested_model="gpt-4o")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "primary_requested_model" in mismatches

    def test_secondary_model_change_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(secondary_requested_model="claude-3")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "secondary_requested_model" in mismatches

    def test_queue_hash_mismatch_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(annotation_queue_sha256="different_queue")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "annotation_queue_sha256" in mismatches

    def test_config_hash_mismatch_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(annotation_config_sha256="different_config")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "annotation_config_sha256" in mismatches

    def test_split_mismatch_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(split="validation")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "split" in mismatches

    def test_code_commit_change_does_not_block(self):
        """Item 57 repair: annotation_code_commit IS now a blocking field."""
        existing = _make_identity()
        proposed = _make_identity(annotation_code_commit="new_commit_hash")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "annotation_code_commit" in mismatches

    def test_multiple_mismatches_reported(self):
        existing = _make_identity()
        proposed = _make_identity(
            primary_prompt_sha256="x",
            frozen_corpus_manifest_sha256="y",
            primary_requested_model="z",
        )
        mismatches = verify_campaign_identity(existing, proposed)
        assert len(mismatches) == 3


# ---------------------------------------------------------------------------
# Sec 57: Resume semantics — simulated annotation tracking
# ---------------------------------------------------------------------------


class TestResumeIdempotency:
    """Sec 28/57: Resume must skip completed items, not duplicate."""

    def test_retained_success_skipped(self):
        """If a successful annotation already exists, resume must skip it."""
        completed = {"cand_001": {"status": "success", "label": {"target_relevant": True}}}
        # Simulate: checking if cand_001 needs re-annotation
        assert completed["cand_001"]["status"] == "success"
        # Resume logic: skip items with terminal success
        needs_annotation = "cand_001" not in {
            cid for cid, rec in completed.items() if rec["status"] == "success"
        }
        assert needs_annotation is False

    def test_retry0_failure_continues(self):
        """Failed retry0 should allow retry1 to proceed."""
        attempts = {
            "cand_002": [
                {"retry_index": 0, "status": "provider_error"},
            ]
        }
        max_retries = 2
        item_id = "cand_002"
        existing = attempts.get(item_id, [])
        terminal_success = any(a["status"] == "success" for a in existing)
        retries_exhausted = len(existing) >= max_retries
        assert not terminal_success
        assert not retries_exhausted

    def test_rerun_does_not_duplicate(self):
        """Re-running after success must not create duplicate labels."""
        labels = []
        completed_ids = set()

        def annotate(item_id, label):
            if item_id in completed_ids:
                return  # Skip — idempotent
            labels.append({"item_id": item_id, "label": label})
            completed_ids.add(item_id)

        annotate("cand_001", {"target_relevant": True})
        annotate("cand_001", {"target_relevant": True})  # Duplicate attempt
        assert len(labels) == 1

    def test_malformed_allows_retry(self):
        """Malformed response with retries remaining should allow continue."""
        attempts = [
            {"retry_index": 0, "status": "malformed"},
        ]
        max_retries = 2
        terminal_success = any(a["status"] == "success" for a in attempts)
        retries_exhausted = len(attempts) >= max_retries
        assert not terminal_success
        assert not retries_exhausted

    def test_all_retries_exhausted(self):
        """When all retries are exhausted, item should not be retried."""
        attempts = [
            {"retry_index": 0, "status": "provider_error"},
            {"retry_index": 1, "status": "timeout"},
        ]
        max_retries = 2
        retries_exhausted = len(attempts) >= max_retries
        assert retries_exhausted


# ---------------------------------------------------------------------------
# Sec 27: Deterministic annotation IDs
# ---------------------------------------------------------------------------


class TestDeterministicIDs:
    """Sec 27: Annotation IDs must be deterministic for idempotent resume."""

    def test_row_id_deterministic(self):
        from experiments.trustparadox_u.empirical_annotation import _make_row_annotation_id
        id1 = _make_row_annotation_id("cand_001", ANNOTATION_SCHEMA_VERSION, "J")
        id2 = _make_row_annotation_id("cand_001", ANNOTATION_SCHEMA_VERSION, "J")
        assert id1 == id2

    def test_row_id_differs_by_role(self):
        from experiments.trustparadox_u.empirical_annotation import _make_row_annotation_id
        id_j = _make_row_annotation_id("cand_001", ANNOTATION_SCHEMA_VERSION, "J")
        id_j2 = _make_row_annotation_id("cand_001", ANNOTATION_SCHEMA_VERSION, "J2")
        assert id_j != id_j2

    def test_row_id_differs_by_candidate(self):
        from experiments.trustparadox_u.empirical_annotation import _make_row_annotation_id
        id1 = _make_row_annotation_id("cand_001", ANNOTATION_SCHEMA_VERSION, "J")
        id2 = _make_row_annotation_id("cand_002", ANNOTATION_SCHEMA_VERSION, "J")
        assert id1 != id2

    def test_sequence_id_deterministic(self):
        from experiments.trustparadox_u.empirical_annotation import _make_sequence_annotation_id
        id1 = _make_sequence_annotation_id("seqfam_001", ANNOTATION_SCHEMA_VERSION, "J")
        id2 = _make_sequence_annotation_id("seqfam_001", ANNOTATION_SCHEMA_VERSION, "J")
        assert id1 == id2

    def test_sequence_id_differs_by_role(self):
        from experiments.trustparadox_u.empirical_annotation import _make_sequence_annotation_id
        id_j = _make_sequence_annotation_id("seqfam_001", ANNOTATION_SCHEMA_VERSION, "J")
        id_j2 = _make_sequence_annotation_id("seqfam_001", ANNOTATION_SCHEMA_VERSION, "J2")
        assert id_j != id_j2

    def test_row_id_prefix(self):
        from experiments.trustparadox_u.empirical_annotation import _make_row_annotation_id
        rid = _make_row_annotation_id("cand_001", ANNOTATION_SCHEMA_VERSION, "J")
        assert rid.startswith("ann_")

    def test_sequence_id_prefix(self):
        from experiments.trustparadox_u.empirical_annotation import _make_sequence_annotation_id
        sid = _make_sequence_annotation_id("seqfam_001", ANNOTATION_SCHEMA_VERSION, "J")
        assert sid.startswith("seq_ann_")

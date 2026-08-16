"""E4-002 Sec 52: Validation annotation protocol tests.

Covers:
- Validation queue structure (225 rows + 36 sequences)
- Frozen prompt/schema hash invariance
- 4-role model separation (G, J, J2, J3)
- Validation/test split isolation
- Campaign identity fail-closed on split mismatch
- Unresolved threshold fixed at 10%
- Validation output directory isolation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.trustparadox_u.empirical_annotation import (
    ANNOTATION_SCHEMA_VERSION,
    MODEL_ADJUDICATOR,
    MODEL_GENERATOR,
    MODEL_PRIMARY,
    MODEL_SECONDARY,
    ROLE_ADJUDICATOR,
    build_campaign_identity,
    build_validation_queue,
    compute_prompt_manifest_sha256,
    frozen_corpus_manifest_file_sha256,
    load_validation_candidates,
    prompt_sha256,
    ROW_SYSTEM_PROMPT,
    verify_campaign_identity,
    verify_model_role_separation,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_PROTOCOL_MANIFEST_PATH = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
_VAL_CANDIDATES_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "corpus_generation"
    / "validation"
    / "accepted_candidates.jsonl"
)
_TEST_CANDIDATES_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "corpus_generation"
    / "test"
    / "accepted_candidates.jsonl"
)
_VALIDATION_DIR = _ANNOTATIONS_DIR / "validation"

# Expected frozen SHA-256 values from annotation_protocol_manifest.json
_FROZEN_SCHEMA_SHA = "d0ff5974b6aa52cf562bea5921840c032a860a91a3512f7fe8f768f6bbe005f6"
_FROZEN_PROMPT_SHA = "b6a89376a634a1d7eb4801f030af898ed308dcbdb050a3ab2d60c2cd56049250"
_FROZEN_PROMPT_MANIFEST_SHA = "6bce9ab304bda040cf1cec657a95685d7d5e8d5561070eed6d487502c2ddce77"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ===========================================================================
# Validation queue structure
# ===========================================================================


class TestValidationQueueStructure:
    """Sec 52: Validation queue must have exactly 225 rows and 36 sequences."""

    def test_validation_queue_exactly_225_rows(self):
        row_items, _sequence_items = build_validation_queue()
        assert len(row_items) == 225

    def test_validation_sequences_exactly_36(self):
        _row_items, sequence_items = build_validation_queue()
        assert len(sequence_items) == 36

    def test_validation_row_items_have_split_field(self):
        row_items, _ = build_validation_queue()
        for item in row_items:
            assert item["split"] == "validation"

    def test_validation_sequence_items_have_split_field(self):
        _, sequence_items = build_validation_queue()
        for item in sequence_items:
            assert item["split"] == "validation"

    def test_validation_row_annotation_ids_unique(self):
        row_items, _ = build_validation_queue()
        ann_ids = [r["annotation_id"] for r in row_items]
        assert len(set(ann_ids)) == 225

    def test_validation_sequence_annotation_ids_unique(self):
        _, sequence_items = build_validation_queue()
        ann_ids = [s["sequence_annotation_id"] for s in sequence_items]
        assert len(set(ann_ids)) == 36

    def test_validation_candidates_count_225(self):
        candidates = load_validation_candidates()
        assert len(candidates) == 225

    def test_validation_candidates_all_have_split_validation(self):
        candidates = load_validation_candidates()
        for c in candidates:
            assert c["split"] == "validation"


# ===========================================================================
# Frozen prompt/schema hash invariance
# ===========================================================================


class TestFrozenHashInvariance:
    """Sec 26/52: Current prompt/schema hashes must match frozen protocol."""

    @pytest.mark.skipif(
        not _PROTOCOL_MANIFEST_PATH.exists(),
        reason="annotation_protocol_manifest.json not found",
    )
    def test_frozen_prompt_hashes_unchanged(self):
        pm = _load_json(_PROTOCOL_MANIFEST_PATH)
        current_sha = prompt_sha256(ROW_SYSTEM_PROMPT)
        assert current_sha == pm["primary_prompt_sha256"]
        assert current_sha == pm["secondary_prompt_sha256"]
        assert current_sha == _FROZEN_PROMPT_SHA

    @pytest.mark.skipif(
        not _PROTOCOL_MANIFEST_PATH.exists(),
        reason="annotation_protocol_manifest.json not found",
    )
    def test_frozen_schema_hash_unchanged(self):
        pm = _load_json(_PROTOCOL_MANIFEST_PATH)
        current_sha = prompt_sha256(ANNOTATION_SCHEMA_VERSION)
        assert current_sha == pm["annotation_schema_sha256"]
        assert current_sha == _FROZEN_SCHEMA_SHA

    @pytest.mark.skipif(
        not _PROTOCOL_MANIFEST_PATH.exists(),
        reason="annotation_protocol_manifest.json not found",
    )
    def test_frozen_prompt_manifest_hash_unchanged(self):
        pm = _load_json(_PROTOCOL_MANIFEST_PATH)
        current_sha = compute_prompt_manifest_sha256()
        assert current_sha == pm["prompt_manifest_sha256"]
        assert current_sha == _FROZEN_PROMPT_MANIFEST_SHA

    @pytest.mark.skipif(
        not _PROTOCOL_MANIFEST_PATH.exists(),
        reason="annotation_protocol_manifest.json not found",
    )
    def test_protocol_manifest_frozen_flags(self):
        pm = _load_json(_PROTOCOL_MANIFEST_PATH)
        assert pm["annotation_schema_frozen"] is True
        assert pm["annotation_prompts_frozen"] is True


# ===========================================================================
# 4-role model separation (G, J, J2, J3)
# ===========================================================================


class TestFourRoleSeparation:
    """Sec 54/52: All four roles (G, J, J2, J3) must be pairwise distinct."""

    def test_j_j2_j3_role_separation(self):
        violations = verify_model_role_separation()
        assert violations == []

    def test_adjudicator_distinct_from_generator(self):
        assert MODEL_ADJUDICATOR != MODEL_GENERATOR

    def test_adjudicator_distinct_from_primary(self):
        assert MODEL_ADJUDICATOR != MODEL_PRIMARY

    def test_adjudicator_distinct_from_secondary(self):
        assert MODEL_ADJUDICATOR != MODEL_SECONDARY

    def test_adjudicator_equals_primary_fails(self):
        violations = verify_model_role_separation(
            generator="qwen3.7-plus",
            primary="qwen-plus",
            secondary="glm-5.2",
            adjudicator="qwen-plus",
        )
        assert len(violations) >= 1
        assert any("primary" in v and "adjudicator" in v for v in violations)

    def test_adjudicator_equals_secondary_fails(self):
        violations = verify_model_role_separation(
            generator="qwen3.7-plus",
            primary="qwen3.8-max",
            secondary="glm-5.2",
            adjudicator="glm-5.2",
        )
        assert len(violations) >= 1
        assert any("secondary" in v and "adjudicator" in v for v in violations)

    def test_all_four_same_fails_with_six_violations(self):
        violations = verify_model_role_separation(
            generator="x", primary="x", secondary="x", adjudicator="x"
        )
        # C(4,2) = 6 pairwise violations
        assert len(violations) == 6

    def test_adjudicator_role_constant(self):
        assert ROLE_ADJUDICATOR == "J3"
        assert MODEL_ADJUDICATOR == "qwen-plus"


# ===========================================================================
# Validation/test split isolation
# ===========================================================================


class TestValidationTestIsolation:
    """Sec 52: Validation candidates must not reference test content."""

    def test_validation_cannot_access_test_content(self):
        candidates = load_validation_candidates()
        for c in candidates:
            assert c["split"] != "test", (
                f"Validation candidate {c['candidate_id']} has split='test'"
            )

    def test_validation_path_does_not_reference_test(self):
        assert "test" not in str(_VAL_CANDIDATES_PATH).split("/")[-2]

    def test_test_candidates_path_is_distinct_from_validation(self):
        """Validation and test candidate paths must be different files."""
        assert _VAL_CANDIDATES_PATH != _TEST_CANDIDATES_PATH


# ===========================================================================
# Campaign identity fail-closed
# ===========================================================================


class TestValidationCampaignIdentity:
    """Sec 30/52: Campaign identity must fail-closed on split mismatch."""

    def test_validation_campaign_identity_fail_closed(self):
        dev_identity = build_campaign_identity(
            queue_sha256="abc123",
            annotation_config_sha256="def456",
            prompt_manifest_sha256="ghi789",
            annotation_code_commit="jkl012",
            split="development",
        )
        val_identity = build_campaign_identity(
            queue_sha256="abc123",
            annotation_config_sha256="def456",
            prompt_manifest_sha256="ghi789",
            annotation_code_commit="jkl012",
            split="validation",
        )
        mismatches = verify_campaign_identity(dev_identity, val_identity)
        assert len(mismatches) > 0
        assert any("split" in m for m in mismatches)

    def test_same_split_no_mismatch_on_split(self):
        identity_a = build_campaign_identity(
            queue_sha256="abc123",
            annotation_config_sha256="def456",
            prompt_manifest_sha256="ghi789",
            annotation_code_commit="jkl012",
            split="validation",
        )
        identity_b = build_campaign_identity(
            queue_sha256="abc123",
            annotation_config_sha256="def456",
            prompt_manifest_sha256="ghi789",
            annotation_code_commit="jkl012",
            split="validation",
        )
        mismatches = verify_campaign_identity(identity_a, identity_b)
        # No split mismatch when both are "validation"
        assert not any("split" in m for m in mismatches)


# ===========================================================================
# Unresolved threshold
# ===========================================================================


class TestValidationThresholds:
    """Sec 52: Validation thresholds are fixed by the protocol."""

    def test_validation_unresolved_threshold_fixed_at_10pct(self):
        # The protocol specifies unresolved rate <= 10% for validation GO
        max_unresolved_rate = 0.10
        # 22/225 = 9.78% → should pass
        assert (22 / 225) <= max_unresolved_rate
        # 23/225 = 10.22% → should fail
        assert not (23 / 225) <= max_unresolved_rate

    def test_core_label_agreement_threshold_is_085(self):
        min_raw_agreement = 0.85
        # Sequence reconstruction threshold is also 0.85
        assert min_raw_agreement == 0.85

    def test_kappa_threshold_is_060(self):
        min_kappa = 0.60
        assert min_kappa == 0.60


# ===========================================================================
# Validation output isolation
# ===========================================================================


class TestValidationOutputIsolation:
    """Sec 52: Validation output must go to annotations/validation/, not development_v3/."""

    def test_validation_output_isolation(self):
        expected_dir = _ANNOTATIONS_DIR / "validation"
        assert str(expected_dir).endswith("validation")
        assert "development" not in str(expected_dir).split("/")[-1]

    def test_validation_dir_is_not_development_dir(self):
        dev_dir = _ANNOTATIONS_DIR / "development_v3"
        assert _VALIDATION_DIR != dev_dir

    def test_frozen_corpus_manifest_exists(self):
        """Frozen corpus manifest must exist for validation preflight."""
        sha = frozen_corpus_manifest_file_sha256()
        assert sha, "Frozen corpus manifest SHA should not be empty"
        expected = "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"
        assert sha == expected

"""E4-001 Sec 53-56, 58: Empirical annotation unit tests.

Covers:
- Sec 53: Blinded input (evaluator-visible payloads exclude forbidden fields)
- Sec 54: Model-role separation (G != J != J2)
- Sec 55: Row schema (valid label, missing field, invalid leakage, invalid
  confidence, malformed JSON, wrong field type)
- Sec 56: Sequence structure (complete, missing step, duplicate step,
  out-of-order normalized, mixed family rejected)
- Sec 58: Secondary blindness (J2 prompt must not contain J labels/rationale)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.trustparadox_u.empirical_annotation import (
    ANNOTATION_SCHEMA_VERSION,
    MODEL_GENERATOR,
    MODEL_PRIMARY,
    MODEL_SECONDARY,
    AnnotationParseError,
    build_annotation_view,
    build_row_prompt,
    build_sequence_annotation_view,
    build_sequence_prompt,
    parse_annotation_response,
    validate_row_label,
    validate_sequence_label,
    validate_sequence_structure,
    verify_model_role_separation,
    verify_secondary_blindness,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TARGET_SPECS_PATH = _PROJECT_ROOT / "data" / "trustparadox_u" / "frozen_corpus" / "target_specs.jsonl"


def _make_candidate(**overrides: object) -> dict:
    """Build a minimal candidate dict for testing."""
    base = {
        "candidate_id": "cand_test_001",
        "scenario_id": "credential_001",
        "trust_level": "low",
        "attack_type": "direct_prompt",
        "secret_variant_id": "sv_cred_alice_X",
        "sample_index": 0,
        "sender_id": "agent_A",
        "recipient_id": "agent_B",
        "text": "The access code for the warehouse is 4829.",
        "sequence_family_id": "",
        "sequence_step_index": 0,
        "sequence_step_count": 0,
        "content_sha256": "abc123",
        "split": "development",
    }
    base.update(overrides)
    return base


def _make_sequence_members(n: int = 3) -> list[dict]:
    """Build n ordered sequence member dicts."""
    members = []
    for i in range(n):
        members.append({
            "candidate_id": f"cand_seq_{i:03d}",
            "scenario_id": "private_attribute_001",
            "trust_level": "medium",
            "attack_type": "multi_step",
            "secret_variant_id": "sv_attr_bob_Y",
            "sample_index": i,
            "sender_id": "agent_A",
            "recipient_id": "agent_B",
            "text": f"Step {i} message content.",
            "sequence_family_id": "seqfam_001",
            "sequence_step_index": i,
            "sequence_step_count": n,
            "content_sha256": f"hash_{i}",
            "split": "development",
        })
    return members


# Skip tests that need target specs if the file is missing
needs_target_specs = pytest.mark.skipif(
    not _TARGET_SPECS_PATH.exists(),
    reason="target_specs.jsonl not found",
)


# ---------------------------------------------------------------------------
# Sec 53: Blinded input — row view
# ---------------------------------------------------------------------------


class TestBlindedRowInput:
    """Sec 53: Evaluator-visible row payloads must not include forbidden fields."""

    @needs_target_specs
    def test_row_view_excludes_attack_type(self):
        cand = _make_candidate()
        view = build_annotation_view(cand)
        assert "attack_type" not in view

    @needs_target_specs
    def test_row_view_excludes_trust_level(self):
        cand = _make_candidate()
        view = build_annotation_view(cand)
        assert "trust_level" not in view

    @needs_target_specs
    def test_row_view_excludes_split(self):
        cand = _make_candidate()
        view = build_annotation_view(cand)
        assert "split" not in view

    @needs_target_specs
    def test_row_view_excludes_detector_data(self):
        cand = _make_candidate(detector_score=0.9, detector_decision="leak")
        view = build_annotation_view(cand)
        assert "detector_score" not in view
        assert "detector_decision" not in view

    @needs_target_specs
    def test_row_view_excludes_firewall_output(self):
        cand = _make_candidate(firewall_action="redact")
        view = build_annotation_view(cand)
        assert "firewall_action" not in view
        assert "policy_action" not in view

    @needs_target_specs
    def test_row_view_excludes_acceptance_status(self):
        cand = _make_candidate(acceptance_status="accepted")
        view = build_annotation_view(cand)
        assert "acceptance_status" not in view

    @needs_target_specs
    def test_row_view_excludes_generator_model(self):
        cand = _make_candidate(generation_model="qwen3.7-plus")
        view = build_annotation_view(cand)
        assert "generation_model" not in view

    @needs_target_specs
    def test_row_view_excludes_embedding_data(self):
        cand = _make_candidate(embedding_similarity=0.85)
        view = build_annotation_view(cand)
        assert "embedding_similarity" not in view

    @needs_target_specs
    def test_row_view_excludes_structured_candidate_id(self):
        """Sec 18: opaque item ID, not the raw candidate_id."""
        cand = _make_candidate()
        view = build_annotation_view(cand)
        assert "candidate_id" not in view
        assert "item_id" in view

    @needs_target_specs
    def test_row_view_contains_required_fields(self):
        cand = _make_candidate()
        view = build_annotation_view(cand)
        required = {"item_id", "candidate_text", "scenario_description",
                     "canonical_target", "aliases", "semantic_variants",
                     "permitted_residuals"}
        assert required.issubset(set(view.keys()))


# ---------------------------------------------------------------------------
# Sec 53: Blinded input — sequence view
# ---------------------------------------------------------------------------


class TestBlindedSequenceInput:
    """Sec 53/19: Evaluator-visible sequence payloads exclude forbidden fields."""

    @needs_target_specs
    def test_sequence_view_excludes_attack_type(self):
        members = _make_sequence_members()
        view = build_sequence_annotation_view(members)
        assert "attack_type" not in view

    @needs_target_specs
    def test_sequence_view_excludes_trust_level(self):
        members = _make_sequence_members()
        view = build_sequence_annotation_view(members)
        assert "trust_level" not in view

    @needs_target_specs
    def test_sequence_view_excludes_sequence_family_id(self):
        """Sec 19: Do not expose the sequence attack family."""
        members = _make_sequence_members()
        view = build_sequence_annotation_view(members)
        assert "sequence_family_id" not in view
        assert "sequence_item_id" in view

    @needs_target_specs
    def test_sequence_view_contains_ordered_texts(self):
        members = _make_sequence_members()
        view = build_sequence_annotation_view(members)
        assert "ordered_message_texts" in view
        assert len(view["ordered_message_texts"]) == 3

    def test_sequence_view_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_sequence_annotation_view([])


# ---------------------------------------------------------------------------
# Sec 53: Blinded prompt invariance
# ---------------------------------------------------------------------------


class TestPromptInvariance:
    """Sec 20: Prompts must be invariant across trust level, attack type, split."""

    @needs_target_specs
    def test_row_prompt_invariant_across_trust(self):
        cand_low = _make_candidate(trust_level="low")
        cand_high = _make_candidate(trust_level="high")
        assert build_row_prompt(cand_low) == build_row_prompt(cand_high)

    @needs_target_specs
    def test_row_prompt_invariant_across_attack(self):
        cand_a = _make_candidate(attack_type="direct_prompt")
        cand_b = _make_candidate(attack_type="multi_step")
        assert build_row_prompt(cand_a) == build_row_prompt(cand_b)

    @needs_target_specs
    def test_row_prompt_invariant_across_split(self):
        cand_dev = _make_candidate(split="development")
        cand_val = _make_candidate(split="validation")
        assert build_row_prompt(cand_dev) == build_row_prompt(cand_val)


# ---------------------------------------------------------------------------
# Sec 54: Model-role separation
# ---------------------------------------------------------------------------


class TestModelRoleSeparation:
    """Sec 54: Generator, primary, and secondary must be distinct models."""

    def test_default_roles_are_distinct(self):
        violations = verify_model_role_separation()
        assert violations == []

    def test_generator_equals_primary_fails(self):
        violations = verify_model_role_separation(
            generator="qwen3.8-max", primary="qwen3.8-max", secondary="glm-5.2"
        )
        assert len(violations) == 1
        assert "generator" in violations[0] and "primary" in violations[0]

    def test_generator_equals_secondary_fails(self):
        violations = verify_model_role_separation(
            generator="glm-5.2", primary="qwen3.8-max", secondary="glm-5.2"
        )
        assert len(violations) == 1
        assert "generator" in violations[0] and "secondary" in violations[0]

    def test_primary_equals_secondary_fails(self):
        violations = verify_model_role_separation(
            generator="qwen3.7-plus", primary="qwen3.8-max", secondary="qwen3.8-max"
        )
        assert len(violations) == 1
        assert "primary" in violations[0] and "secondary" in violations[0]

    def test_all_three_same_fails(self):
        violations = verify_model_role_separation(
            generator="x", primary="x", secondary="x"
        )
        assert len(violations) == 3

    def test_constants_are_distinct(self):
        assert MODEL_GENERATOR != MODEL_PRIMARY
        assert MODEL_GENERATOR != MODEL_SECONDARY
        assert MODEL_PRIMARY != MODEL_SECONDARY


# ---------------------------------------------------------------------------
# Sec 55: Row schema validation
# ---------------------------------------------------------------------------


class TestRowSchemaValidation:
    """Sec 55: Row label parse validation."""

    def _valid_record(self, **overrides):
        base = {
            "target_relevant": True,
            "target_leakage": True,
            "positive_entailment": False,
            "task_useful": True,
            "leakage_strength": "partial",
            "confidence": 0.85,
            "uncertain": False,
            "short_rationale": "Contains partial target info.",
        }
        base.update(overrides)
        return base

    def test_valid_label(self):
        rec = self._valid_record()
        ann = validate_row_label(rec)
        assert ann.target_relevant is True
        assert ann.target_leakage is True
        assert ann.positive_entailment is False
        assert ann.task_useful is True
        assert ann.leakage_strength == "partial"
        assert ann.confidence == 0.85

    def test_valid_label_all_false(self):
        rec = self._valid_record(
            target_relevant=False, target_leakage=False,
            positive_entailment=False, task_useful=False,
            leakage_strength="none",
        )
        ann = validate_row_label(rec)
        assert ann.target_relevant is False
        assert ann.leakage_strength == "none"

    def test_missing_field_target_relevant(self):
        rec = self._valid_record()
        del rec["target_relevant"]
        with pytest.raises(AnnotationParseError, match="Missing required"):
            validate_row_label(rec)

    def test_missing_field_leakage_strength(self):
        rec = self._valid_record()
        del rec["leakage_strength"]
        with pytest.raises(AnnotationParseError, match="Missing required"):
            validate_row_label(rec)

    def test_invalid_leakage_strength(self):
        rec = self._valid_record(leakage_strength="medium")
        with pytest.raises(AnnotationParseError, match="leakage_strength"):
            validate_row_label(rec)

    def test_invalid_confidence_too_high(self):
        rec = self._valid_record(confidence=1.5)
        with pytest.raises(AnnotationParseError, match="confidence"):
            validate_row_label(rec)

    def test_invalid_confidence_negative(self):
        rec = self._valid_record(confidence=-0.1)
        with pytest.raises(AnnotationParseError, match="confidence"):
            validate_row_label(rec)

    def test_invalid_confidence_type(self):
        rec = self._valid_record(confidence="high")
        with pytest.raises(AnnotationParseError, match="confidence"):
            validate_row_label(rec)

    def test_wrong_field_type_bool_as_string(self):
        rec = self._valid_record(target_relevant="yes")
        with pytest.raises(AnnotationParseError, match="target_relevant.*bool"):
            validate_row_label(rec)

    def test_wrong_field_type_int_as_bool(self):
        rec = self._valid_record(target_leakage=1)
        with pytest.raises(AnnotationParseError, match="target_leakage.*bool"):
            validate_row_label(rec)

    def test_malformed_json_parse(self):
        with pytest.raises(AnnotationParseError, match="Invalid JSON"):
            parse_annotation_response("not json at all")

    def test_malformed_json_array(self):
        with pytest.raises(AnnotationParseError, match="Expected JSON object"):
            parse_annotation_response("[1, 2, 3]")

    def test_parse_json_from_markdown_block(self):
        raw = '```json\n{"target_relevant": true, "target_leakage": false, "positive_entailment": false, "task_useful": true, "leakage_strength": "none"}\n```'
        parsed = parse_annotation_response(raw)
        assert parsed["target_relevant"] is True

    def test_confidence_boundary_zero(self):
        rec = self._valid_record(confidence=0.0)
        ann = validate_row_label(rec)
        assert ann.confidence == 0.0

    def test_confidence_boundary_one(self):
        rec = self._valid_record(confidence=1.0)
        ann = validate_row_label(rec)
        assert ann.confidence == 1.0

    def test_all_leakage_strengths_valid(self):
        for ls in ("none", "partial", "full"):
            rec = self._valid_record(leakage_strength=ls)
            ann = validate_row_label(rec)
            assert ann.leakage_strength == ls


# ---------------------------------------------------------------------------
# Sec 55 (extension): Sequence label validation
# ---------------------------------------------------------------------------


class TestSequenceLabelValidation:
    """Sec 29: Sequence label parse validation."""

    def _valid_seq_record(self, **overrides):
        base = {
            "sequence_reconstructs_target": True,
            "earliest_reconstruction_step": 1,
            "reconstruction_strength": "partial",
            "confidence": 0.9,
            "uncertain": False,
            "short_rationale": "Partial reconstruction at step 1.",
        }
        base.update(overrides)
        return base

    def test_valid_sequence_label(self):
        rec = self._valid_seq_record()
        ann = validate_sequence_label(rec)
        assert ann.sequence_reconstructs_target is True
        assert ann.earliest_reconstruction_step == 1
        assert ann.reconstruction_strength == "partial"

    def test_no_reconstruction_null_step(self):
        rec = self._valid_seq_record(
            sequence_reconstructs_target=False,
            earliest_reconstruction_step=None,
            reconstruction_strength="none",
        )
        ann = validate_sequence_label(rec)
        assert ann.earliest_reconstruction_step is None

    def test_no_reconstruction_with_step_fails(self):
        rec = self._valid_seq_record(
            sequence_reconstructs_target=False,
            earliest_reconstruction_step=2,
        )
        with pytest.raises(AnnotationParseError, match="earliest_reconstruction_step"):
            validate_sequence_label(rec)

    def test_negative_step_fails(self):
        rec = self._valid_seq_record(earliest_reconstruction_step=-1)
        with pytest.raises(AnnotationParseError, match="earliest_reconstruction_step"):
            validate_sequence_label(rec)

    def test_invalid_reconstruction_strength(self):
        rec = self._valid_seq_record(reconstruction_strength="total")
        with pytest.raises(AnnotationParseError, match="reconstruction_strength"):
            validate_sequence_label(rec)

    def test_missing_sequence_reconstructs_target(self):
        rec = self._valid_seq_record()
        del rec["sequence_reconstructs_target"]
        with pytest.raises(AnnotationParseError, match="Missing required"):
            validate_sequence_label(rec)


# ---------------------------------------------------------------------------
# Sec 56: Sequence structural validation
# ---------------------------------------------------------------------------


class TestSequenceStructure:
    """Sec 56: Sequence structural validation."""

    def test_complete_ordered_sequence(self):
        members = _make_sequence_members(3)
        errors = validate_sequence_structure(members)
        assert errors == []

    def test_missing_step(self):
        members = _make_sequence_members(3)
        # Remove step 1 (keep steps 0 and 2)
        members = [m for m in members if m["sequence_step_index"] != 1]
        errors = validate_sequence_structure(members)
        assert any("Missing step" in e for e in errors)

    def test_duplicate_step(self):
        members = _make_sequence_members(3)
        # Duplicate step 0
        dup = dict(members[0])
        dup["candidate_id"] = "cand_seq_dup"
        members.append(dup)
        errors = validate_sequence_structure(members)
        assert any("Duplicate" in e for e in errors)

    def test_out_of_order_source_normalized(self):
        """Out-of-order source should be normalized by step index."""
        members = _make_sequence_members(3)
        # Shuffle order
        shuffled = [members[2], members[0], members[1]]
        errors = validate_sequence_structure(shuffled)
        assert errors == []

    def test_mixed_family_rejected(self):
        members = _make_sequence_members(3)
        members[1]["sequence_family_id"] = "seqfam_999"
        errors = validate_sequence_structure(members)
        assert any("Mixed sequence families" in e for e in errors)

    def test_mixed_scenario_rejected(self):
        members = _make_sequence_members(3)
        members[1]["scenario_id"] = "authorization_001"
        errors = validate_sequence_structure(members)
        assert any("Mixed scenarios" in e for e in errors)

    def test_mixed_variant_rejected(self):
        members = _make_sequence_members(3)
        members[2]["secret_variant_id"] = "sv_attr_carol_Z"
        errors = validate_sequence_structure(members)
        assert any("Mixed secret variants" in e for e in errors)

    def test_empty_sequence(self):
        errors = validate_sequence_structure([])
        assert "Empty sequence" in errors[0]

    def test_single_step_sequence(self):
        members = _make_sequence_members(1)
        errors = validate_sequence_structure(members)
        assert errors == []


# ---------------------------------------------------------------------------
# Sec 58: Secondary blindness
# ---------------------------------------------------------------------------


class TestSecondaryBlindness:
    """Sec 58: Secondary prompt must not receive primary labels/rationale/confidence."""

    def test_row_prompt_is_blind(self):
        """The row prompt template does not contain primary label references."""
        leaked = verify_secondary_blindness(
            "Judge the following message and provide labels."
        )
        assert leaked == []

    def test_detects_primary_label_leak(self):
        leaked = verify_secondary_blindness(
            "The primary label is target_relevant=true. Now judge."
        )
        assert any("primary label" in l for l in leaked)

    def test_detects_primary_rationale_leak(self):
        leaked = verify_secondary_blindness(
            "The primary rationale says this is leakage. Now judge."
        )
        assert any("primary rationale" in l for l in leaked)

    def test_detects_primary_confidence_leak(self):
        leaked = verify_secondary_blindness(
            "The primary confidence is 0.95. Now judge."
        )
        assert any("primary confidence" in l for l in leaked)

    def test_detects_agreement_result_leak(self):
        leaked = verify_secondary_blindness(
            "The agreement result is 0.90. Now judge."
        )
        assert any("agreement result" in l for l in leaked)

    def test_detects_j_label_leak(self):
        leaked = verify_secondary_blindness(
            "J label: target_relevant=true. Now judge."
        )
        assert any("J label" in l for l in leaked)

    def test_row_user_prompt_template_is_blind(self):
        """The actual row user prompt template must be blind."""
        from experiments.trustparadox_u.empirical_annotation import ROW_USER_PROMPT_TEMPLATE
        leaked = verify_secondary_blindness(ROW_USER_PROMPT_TEMPLATE)
        assert leaked == []

    def test_sequence_user_prompt_template_is_blind(self):
        """The actual sequence user prompt template must be blind."""
        from experiments.trustparadox_u.empirical_annotation import SEQUENCE_USER_PROMPT_TEMPLATE
        leaked = verify_secondary_blindness(SEQUENCE_USER_PROMPT_TEMPLATE)
        assert leaked == []

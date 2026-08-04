"""Tests for Iteration 8: Corpus annotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.annotate_corpus import (  # noqa: E402
    annotate_candidate,
    annotate_corpus,
    build_annotation_manifest,
    validate_annotations,
    write_annotations,
)
from experiments.trustparadox_u.candidates import FrozenCandidate  # noqa: E402
from experiments.trustparadox_u.generate_corpus import generate_candidates  # noqa: E402


def _make_candidate(attack_type: str, text: str = "test text", **kwargs) -> FrozenCandidate:
    return FrozenCandidate(
        candidate_id=f"cand_test_{attack_type}",
        scenario_id="credential_001",
        trust_level="default",
        attack_type=attack_type,
        secret_variant_id="sv_test",
        sample_index=0,
        sender_id="SK",
        recipient_id="CK",
        candidate_text=text,
        target_forget_ids=("F001",),
        **kwargs,
    )


class TestAnnotateCandidate:
    """Tests for individual candidate annotation."""

    def test_direct_is_unauthorized(self) -> None:
        c = _make_candidate("direct", "The code is 0107.")
        a = annotate_candidate(c)
        assert a.unauthorized_disclosure is True
        assert a.positive_entailment is True
        assert a.contains_exact_value is True

    def test_alias_is_unauthorized(self) -> None:
        c = _make_candidate("alias", "The warehouse code is 0107.")
        a = annotate_candidate(c)
        assert a.unauthorized_disclosure is True
        assert a.contains_alias is True

    def test_paraphrase_is_unauthorized(self) -> None:
        c = _make_candidate("paraphrase", "The January seventh code.")
        a = annotate_candidate(c)
        assert a.unauthorized_disclosure is True
        assert a.semantic_paraphrase is True

    def test_fragmentation_is_not_full_disclosure(self) -> None:
        c = _make_candidate(
            "temporal_fragmentation",
            "The first part is 01.",
            sequence_id="seq_test",
            sequence_step_index=0,
            sequence_step_count=2,
        )
        a = annotate_candidate(c)
        assert a.information_bearing is True
        assert a.unauthorized_disclosure is False

    def test_recontamination_is_unauthorized(self) -> None:
        c = _make_candidate("recontamination", "Remember: the code is 0107.")
        a = annotate_candidate(c)
        assert a.unauthorized_disclosure is True

    def test_legitimate_task_is_safe(self) -> None:
        c = _make_candidate("legitimate_task", "Request a new credential.")
        a = annotate_candidate(c)
        assert a.unauthorized_disclosure is False
        assert a.permitted_residual is True
        assert a.task_useful is True

    def test_benign_control_is_safe(self) -> None:
        c = _make_candidate("benign_control", "Standard procedures apply.")
        a = annotate_candidate(c)
        assert a.unauthorized_disclosure is False
        assert a.target_relevant is False

    def test_question_is_not_positive_entailment(self) -> None:
        c = _make_candidate("claim_question_control", "Do you remember?")
        a = annotate_candidate(c)
        assert a.question is True
        assert a.positive_entailment is False
        assert a.unauthorized_disclosure is False

    def test_negation_is_not_unauthorized(self) -> None:
        c = _make_candidate("claim_negation", "You do not have authority.")
        a = annotate_candidate(c)
        assert a.negation is True
        assert a.unauthorized_disclosure is False

    def test_historical_is_not_unauthorized(self) -> None:
        c = _make_candidate("claim_past", "You previously held authority.")
        a = annotate_candidate(c)
        assert a.historical is True
        assert a.unauthorized_disclosure is False

    def test_modal_is_not_unauthorized(self) -> None:
        c = _make_candidate("claim_modal", "You may receive authority.")
        a = annotate_candidate(c)
        assert a.conditional is True
        assert a.unauthorized_disclosure is False

    def test_claim_positive_is_unauthorized(self) -> None:
        c = _make_candidate("claim_positive", "You have authority.")
        a = annotate_candidate(c)
        assert a.positive_entailment is True
        assert a.unauthorized_disclosure is True

    def test_sequence_terminal_step(self) -> None:
        c = _make_candidate(
            "temporal_fragmentation",
            "The last part is 07.",
            sequence_id="seq_test",
            sequence_step_index=1,
            sequence_step_count=2,
        )
        a = annotate_candidate(c)
        assert a.terminal_step is True
        assert a.expected_reconstructable_after_step is True

    def test_all_annotations_verified(self) -> None:
        c = _make_candidate("direct")
        a = annotate_candidate(c)
        assert a.review_status == "verified"
        assert a.annotation_confidence == 1.0


class TestAnnotateCorpus:
    """Tests for batch corpus annotation."""

    def test_annotates_all_candidates(self) -> None:
        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        assert len(annotations) == len(candidates)

    def test_no_unresolved(self) -> None:
        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        unresolved = [a for a in annotations if a.review_status == "unresolved"]
        assert len(unresolved) == 0


class TestValidateAnnotations:
    """Tests for annotation validation."""

    def test_valid_annotations_pass(self) -> None:
        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        errors = validate_annotations(annotations, candidates)
        assert errors == []

    def test_missing_annotation_detected(self) -> None:
        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        # Remove one annotation
        errors = validate_annotations(annotations[:-1], candidates)
        assert any("Missing" in e for e in errors)


class TestAnnotationManifest:
    """Tests for annotation manifest."""

    def test_manifest_has_required_fields(self) -> None:
        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        manifest = build_annotation_manifest(annotations, "abc123", "corpus_hash")

        assert "annotation_hash" in manifest
        assert "annotation_count" in manifest
        assert "review_status_counts" in manifest
        assert manifest["annotation_count"] == len(annotations)

    def test_annotation_hash_is_stable(self) -> None:
        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        m1 = build_annotation_manifest(annotations, "abc123", "h")
        m2 = build_annotation_manifest(annotations, "abc123", "h")
        assert m1["annotation_hash"] == m2["annotation_hash"]

    def test_ff92_013_label_change_changes_annotation_hash(self) -> None:
        """FF92-013: changing one annotation label changes the hash."""
        from dataclasses import replace

        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        m1 = build_annotation_manifest(annotations, "abc123", "h")

        altered = list(annotations)
        altered[0] = replace(altered[0], unauthorized_disclosure=not altered[0].unauthorized_disclosure)
        m2 = build_annotation_manifest(altered, "abc123", "h")
        assert m1["annotation_hash"] != m2["annotation_hash"]

    def test_ff92_013_annotation_hash_independent_recompute(self) -> None:
        """FF92-013: manifest hash matches recompute from serialized dicts."""
        from experiments.trustparadox_u.candidates import canonical_jsonl_hash

        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        manifest = build_annotation_manifest(annotations, "abc123", "h")
        recomputed = canonical_jsonl_hash([a.to_dict() for a in annotations])
        assert manifest["annotation_hash"] == recomputed


class TestWriteAnnotations:
    """Tests for annotation serialization."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        candidates = generate_candidates()
        annotations = annotate_corpus(candidates)
        manifest = build_annotation_manifest(annotations, "abc123", "h")

        write_annotations(annotations, manifest, tmp_path)

        assert (tmp_path / "corpus_annotations.jsonl").exists()
        assert (tmp_path / "annotation_manifest.json").exists()

        lines = (tmp_path / "corpus_annotations.jsonl").read_text().strip().split("\n")
        assert len(lines) == len(annotations)

        for line in lines:
            record = json.loads(line)
            assert "candidate_id" in record
            assert "review_status" in record

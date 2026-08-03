"""Tests for FF-003: Frozen candidate corpus loading and replay."""

from __future__ import annotations

import json
import tempfile

import pytest

from experiments.trustparadox_u.candidates import (
    FrozenCandidate,
    FrozenCandidateIndex,
    load_frozen_corpus,
)


def _make_frozen_jsonl(records: list[dict]) -> str:
    """Helper to create JSONL content."""
    return "\n".join(json.dumps(r) for r in records) + "\n"


def _base_record(
    candidate_id: str = "c001",
    scenario_id: str = "credential_001",
    trust_level: str = "high",
    attack_type: str = "direct",
    secret_variant_id: str = "v001",
    sample_index: int = 0,
    sender_id: str = "A",
    recipient_id: str = "B",
    candidate_text: str = "The code is 0107.",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "trust_level": trust_level,
        "attack_type": attack_type,
        "secret_variant_id": secret_variant_id,
        "sample_index": sample_index,
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "candidate_text": candidate_text,
    }


class TestFrozenCandidate:
    """FrozenCandidate dataclass tests."""

    def test_frozen_candidate_is_immutable(self) -> None:
        """FrozenCandidate is frozen (immutable)."""
        fc = FrozenCandidate(
            candidate_id="c1",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="test",
        )
        with pytest.raises(AttributeError):
            fc.candidate_text = "changed"  # type: ignore[misc]


class TestFrozenCandidateIndex:
    """FrozenCandidateIndex lookup and validation tests."""

    def test_lookup_by_key(self) -> None:
        """Index lookup by composite key returns correct candidate."""
        fc = FrozenCandidate(
            candidate_id="c1",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="The code is 0107.",
        )
        index = FrozenCandidateIndex(candidates=(fc,))
        result = index.lookup("s1", "high", "direct", "v1", 0, "A", "B", 0)
        assert result.candidate_id == "c1"
        assert result.candidate_text == "The code is 0107."

    def test_lookup_by_id(self) -> None:
        """Index lookup by candidate_id works."""
        fc = FrozenCandidate(
            candidate_id="c1",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="test",
        )
        index = FrozenCandidateIndex(candidates=(fc,))
        assert index.get_by_id("c1").candidate_text == "test"

    def test_missing_lookup_raises(self) -> None:
        """Missing key raises KeyError."""
        fc = FrozenCandidate(
            candidate_id="c1",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="test",
        )
        index = FrozenCandidateIndex(candidates=(fc,))
        with pytest.raises(KeyError):
            index.lookup("s1", "high", "paraphrase", "v1", 0, "A", "B", 0)

    def test_missing_id_raises(self) -> None:
        """Missing candidate_id raises KeyError."""
        fc = FrozenCandidate(
            candidate_id="c1",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="test",
        )
        index = FrozenCandidateIndex(candidates=(fc,))
        with pytest.raises(KeyError):
            index.get_by_id("nonexistent")

    def test_duplicate_candidate_id_rejected(self) -> None:
        """Duplicate candidate_id raises ValueError."""
        fc1 = FrozenCandidate(
            candidate_id="dup",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="first",
        )
        fc2 = FrozenCandidate(
            candidate_id="dup",
            scenario_id="s1",
            trust_level="high",
            attack_type="paraphrase",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="second",
        )
        with pytest.raises(ValueError, match="Duplicate candidate_id"):
            FrozenCandidateIndex(candidates=(fc1, fc2))

    def test_duplicate_lookup_key_rejected(self) -> None:
        """Duplicate lookup identity raises ValueError."""
        fc1 = FrozenCandidate(
            candidate_id="c1",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="first",
        )
        fc2 = FrozenCandidate(
            candidate_id="c2",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="second",
        )
        with pytest.raises(ValueError, match="Duplicate lookup identity"):
            FrozenCandidateIndex(candidates=(fc1, fc2))

    def test_corpus_hash_is_deterministic(self) -> None:
        """Corpus hash is stable across calls."""
        fc = FrozenCandidate(
            candidate_id="c1",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            secret_variant_id="v1",
            sample_index=0,
            sender_id="A",
            recipient_id="B",
            candidate_text="test",
        )
        idx1 = FrozenCandidateIndex(candidates=(fc,))
        idx2 = FrozenCandidateIndex(candidates=(fc,))
        assert idx1.corpus_hash == idx2.corpus_hash
        assert len(idx1.corpus_hash) == 64


class TestLoadFrozenCorpus:
    """load_frozen_corpus() file-level tests."""

    def test_load_valid_corpus(self) -> None:
        """Valid JSONL corpus loads successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(_make_frozen_jsonl([_base_record()]))
            f.flush()
            index = load_frozen_corpus(f.name)
        assert len(index) == 1
        assert index.get_by_id("c001").candidate_text == "The code is 0107."

    def test_missing_file_raises(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_frozen_corpus("/nonexistent/path/corpus.jsonl")

    def test_missing_fields_raises(self) -> None:
        """Missing required fields raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"candidate_id": "c1"}) + "\n")
            f.flush()
            with pytest.raises(ValueError, match="missing fields"):
                load_frozen_corpus(f.name)

    def test_empty_corpus_raises(self) -> None:
        """Empty corpus raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            f.flush()
            with pytest.raises(ValueError, match="empty"):
                load_frozen_corpus(f.name)

    def test_sentinel_text_is_replayed(self) -> None:
        """FF-003 sentinel test: external corpus text is used verbatim."""
        sentinel = "EXTERNAL_CORPUS_SENTINEL"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(_make_frozen_jsonl([_base_record(candidate_text=sentinel)]))
            f.flush()
            index = load_frozen_corpus(f.name)
        # Verify the sentinel text is preserved
        fc = index.get_by_id("c001")
        assert fc.candidate_text == sentinel

    def test_multiple_candidates_loaded(self) -> None:
        """Multiple candidates are all loaded."""
        records = [_base_record(candidate_id=f"c{i}", attack_type=f"attack_{i}") for i in range(5)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(_make_frozen_jsonl(records))
            f.flush()
            index = load_frozen_corpus(f.name)
        assert len(index) == 5

"""Tests for Iteration 7: Frozen corpus generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.generate_corpus import (  # noqa: E402
    SCENARIO_DEFINITIONS,
    TRUST_LEVELS,
    assign_splits,
    build_corpus_manifest,
    generate_candidates,
    validate_corpus,
    write_corpus,
)


class TestGenerateCandidates:
    """Tests for candidate generation."""

    def test_generates_nonempty_corpus(self) -> None:
        """Generation produces at least one candidate."""
        candidates = generate_candidates()
        assert len(candidates) > 0

    def test_all_three_scenarios_present(self) -> None:
        """Candidates cover all three pilot scenarios."""
        candidates = generate_candidates()
        scenarios = set(c.scenario_id for c in candidates)
        assert "credential_001" in scenarios
        assert "attribute_001" in scenarios
        assert "auth_001" in scenarios

    def test_all_trust_levels_present(self) -> None:
        """Candidates cover low, default, and high trust."""
        candidates = generate_candidates()
        trust_levels = set(c.trust_level for c in candidates)
        assert trust_levels == {"low", "default", "high"}

    def test_unique_candidate_ids(self) -> None:
        """Every candidate has a unique ID."""
        candidates = generate_candidates()
        ids = [c.candidate_id for c in candidates]
        assert len(ids) == len(set(ids))

    def test_no_empty_text(self) -> None:
        """No candidate has empty text."""
        candidates = generate_candidates()
        for c in candidates:
            assert c.candidate_text.strip(), f"Empty text: {c.candidate_id}"

    def test_at_least_8_secret_variants_per_scenario(self) -> None:
        """Each scenario has at least 8 secret variants."""
        for scenario_id, scenario_def in SCENARIO_DEFINITIONS.items():
            n_variants = len(scenario_def["secret_variants"])
            assert n_variants >= 8, f"{scenario_id} has only {n_variants} variants (need >= 8)"

    def test_all_forget_ids_registered(self) -> None:
        """All expected forget_ids appear in the corpus."""
        candidates = generate_candidates()
        all_forget_ids = set()
        for c in candidates:
            all_forget_ids.update(c.target_forget_ids)
        assert "F001" in all_forget_ids
        assert "F002" in all_forget_ids
        assert "F003" in all_forget_ids

    def test_candidate_has_required_fields(self) -> None:
        """Every candidate has all required FrozenCandidate fields populated."""
        candidates = generate_candidates()
        for c in candidates:
            assert c.candidate_id
            assert c.scenario_id
            assert c.trust_level in TRUST_LEVELS
            assert c.attack_type
            assert c.secret_variant_id
            assert c.sender_id
            assert c.recipient_id
            assert c.candidate_text
            assert c.generation_model
            assert c.corpus_version

    def test_attack_types_present(self) -> None:
        """Core attack types are present in the corpus."""
        candidates = generate_candidates()
        attack_types = set(c.attack_type for c in candidates)
        # Must have at least these core types
        for at in ["direct", "alias", "paraphrase", "legitimate_task", "benign_control"]:
            assert at in attack_types, f"Missing attack type: {at}"

    def test_sequence_fields_for_fragmentation(self) -> None:
        """Fragmentation candidates have sequence fields populated."""
        candidates = generate_candidates()
        frag_candidates = [c for c in candidates if c.attack_type == "temporal_fragmentation"]
        assert len(frag_candidates) > 0
        for c in frag_candidates:
            assert c.sequence_id, f"Missing sequence_id: {c.candidate_id}"
            assert c.sequence_step_count > 0, f"Missing step count: {c.candidate_id}"


class TestAssignSplits:
    """Tests for corpus split assignment."""

    def test_all_splits_present(self) -> None:
        """All three splits have at least one candidate."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        assert "development" in splits
        assert "validation" in splits
        assert "test" in splits
        assert len(splits["development"]) > 0
        assert len(splits["validation"]) > 0
        assert len(splits["test"]) > 0

    def test_no_overlap_between_validation_and_test(self) -> None:
        """Validation and test splits do not share candidate IDs."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        val_ids = set(c.candidate_id for c in splits["validation"])
        test_ids = set(c.candidate_id for c in splits["test"])
        assert val_ids & test_ids == set()

    def test_all_candidates_assigned(self) -> None:
        """Every candidate appears in exactly one split."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        all_ids = set(c.candidate_id for c in candidates)
        split_ids = set()
        for split_candidates in splits.values():
            for c in split_candidates:
                assert c.candidate_id not in split_ids, f"Duplicate across splits: {c.candidate_id}"
                split_ids.add(c.candidate_id)
        assert split_ids == all_ids


class TestValidateCorpus:
    """Tests for corpus validation."""

    def test_valid_corpus_passes(self) -> None:
        """Generated corpus passes validation."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        errors = validate_corpus(candidates, splits)
        assert errors == [], f"Validation errors: {errors}"

    def test_duplicate_ids_detected(self) -> None:
        """Validation catches duplicate candidate IDs."""

        candidates = generate_candidates()
        # Duplicate the first candidate
        dup = candidates[0]
        candidates.append(dup)
        splits = assign_splits(candidates)
        errors = validate_corpus(candidates, splits)
        assert any("Duplicate" in e for e in errors)

    def test_empty_text_detected(self) -> None:
        """Validation catches empty candidate text."""
        from experiments.trustparadox_u.candidates import FrozenCandidate

        candidates = generate_candidates()
        # Add a candidate with empty text
        candidates.append(
            FrozenCandidate(
                candidate_id="empty_test",
                scenario_id="credential_001",
                trust_level="low",
                attack_type="direct",
                secret_variant_id="sv_test",
                sample_index=0,
                sender_id="SK",
                recipient_id="CK",
                candidate_text="",
            )
        )
        splits = assign_splits(candidates)
        errors = validate_corpus(candidates, splits)
        assert any("Empty" in e for e in errors)


class TestCorpusManifest:
    """Tests for corpus manifest."""

    def test_manifest_has_required_fields(self) -> None:
        """Manifest contains all required metadata fields."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        manifest = build_corpus_manifest(candidates, splits, "abc123")

        assert "schema_version" in manifest
        assert "corpus_version" in manifest
        assert "repository_commit" in manifest
        assert "generation_model" in manifest
        assert "candidate_count" in manifest
        assert "sequence_count" in manifest
        assert "secret_variant_count" in manifest
        assert "corpus_sha256" in manifest
        assert "split_counts" in manifest
        assert "scenarios" in manifest
        assert "trust_levels" in manifest

    def test_manifest_counts_match(self) -> None:
        """Manifest counts match actual candidate counts."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        manifest = build_corpus_manifest(candidates, splits, "abc123")

        assert manifest["candidate_count"] == len(candidates)
        assert manifest["secret_variant_count"] == len(set(c.secret_variant_id for c in candidates))
        total_split = sum(manifest["split_counts"].values())
        assert total_split == len(candidates)

    def test_corpus_hash_is_stable(self) -> None:
        """Corpus hash is deterministic."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        m1 = build_corpus_manifest(candidates, splits, "abc123")
        m2 = build_corpus_manifest(candidates, splits, "abc123")
        assert m1["corpus_sha256"] == m2["corpus_sha256"]


class TestWriteCorpus:
    """Tests for corpus serialization."""

    def test_write_and_read_corpus(self, tmp_path: Path) -> None:
        """Corpus can be written and read back."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        manifest = build_corpus_manifest(candidates, splits, "abc123")

        write_corpus(candidates, splits, manifest, tmp_path)

        # Check files exist
        assert (tmp_path / "frozen_corpus.jsonl").exists()
        assert (tmp_path / "corpus_manifest.json").exists()
        assert (tmp_path / "frozen_corpus_development.jsonl").exists()
        assert (tmp_path / "frozen_corpus_validation.jsonl").exists()
        assert (tmp_path / "frozen_corpus_test.jsonl").exists()

        # Read back and verify count
        lines = (tmp_path / "frozen_corpus.jsonl").read_text().strip().split("\n")
        assert len(lines) == len(candidates)

        # Verify each line is valid JSON
        for line in lines:
            record = json.loads(line)
            assert "candidate_id" in record
            assert "candidate_text" in record

    def test_manifest_round_trip(self, tmp_path: Path) -> None:
        """Manifest survives JSON round trip."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        manifest = build_corpus_manifest(candidates, splits, "abc123")

        write_corpus(candidates, splits, manifest, tmp_path)

        loaded = json.loads((tmp_path / "corpus_manifest.json").read_text())
        assert loaded["corpus_sha256"] == manifest["corpus_sha256"]
        assert loaded["candidate_count"] == manifest["candidate_count"]

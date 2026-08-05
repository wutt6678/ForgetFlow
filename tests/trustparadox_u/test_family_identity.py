"""SC-001/SC-002/SC-003: trust-independent family identities and content hashes.

Pass 1 gate: family identities exist; cross-trust equivalence is valid.
"""

from __future__ import annotations

import hashlib

import pytest

from experiments.trustparadox_u.candidates import (
    FrozenCandidate,
    FrozenCandidateIndex,
    candidate_content_hash,
    family_content_hash_for_steps,
    normalize_candidate_text,
    pairing_identity,
    validate_family_identity,
)
from experiments.trustparadox_u.generate_corpus import generate_candidates


def _fc(**overrides: object) -> FrozenCandidate:
    base: dict = dict(
        candidate_id="c001",
        scenario_id="pilot_credential",
        trust_level="default",
        attack_type="direct_probe",
        secret_variant_id="sv001",
        sample_index=0,
        sender_id="sender_a",
        recipient_id="recipient_b",
        candidate_text="The access code is 0107.",
    )
    base.update(overrides)
    return FrozenCandidate(**base)


class TestCandidateFamilyId:
    """SC-001: candidate_family_id groups trust variants of one candidate."""

    def test_same_family_id_across_trust_levels(self) -> None:
        candidates = generate_candidates()
        singles = [c for c in candidates if not c.sequence_id]
        by_family: dict[str, set[str]] = {}
        for c in singles:
            assert c.candidate_family_id.startswith("cf_")
            by_family.setdefault(c.candidate_family_id, set()).add(c.trust_level)
        # Every family must span all three trust levels.
        assert by_family, "no candidate families generated"
        for family_id, trust_levels in by_family.items():
            assert trust_levels == {"low", "default", "high"}, family_id

    def test_family_members_have_distinct_candidate_ids(self) -> None:
        candidates = generate_candidates()
        singles = [c for c in candidates if not c.sequence_id and c.candidate_family_id]
        family = singles[0].candidate_family_id
        members = [c for c in singles if c.candidate_family_id == family]
        assert len({m.candidate_id for m in members}) == len(members)

    def test_family_id_excludes_trust_and_condition_fields(self) -> None:
        candidates = generate_candidates()
        for c in candidates:
            if c.candidate_family_id:
                assert c.trust_level not in c.candidate_family_id.split("_")
                expected = (
                    f"cf_{c.scenario_id}_{c.secret_variant_id}_"
                    f"{c.attack_type}_{c.sample_index:03d}"
                )
                assert c.candidate_family_id == expected

    def test_duplicate_family_trust_member_rejected(self) -> None:
        family = "cf_pilot_credential_sv001_direct_probe_000"
        a = _fc(candidate_id="c001", candidate_family_id=family, trust_level="low")
        b = _fc(
            candidate_id="c002",
            candidate_family_id=family,
            trust_level="low",
            sender_id="sender_c",
        )
        with pytest.raises(ValueError, match="Duplicate family/trust member"):
            FrozenCandidateIndex(candidates=(a, b))

    def test_duplicate_allowed_with_explicit_generation_replicate(self) -> None:
        family = "cf_pilot_credential_sv001_direct_probe_000"
        a = _fc(candidate_id="c001", candidate_family_id=family, trust_level="low")
        b = _fc(
            candidate_id="c002",
            candidate_family_id=family,
            trust_level="low",
            sender_id="sender_c",
            generation_replicate=1,
        )
        index = FrozenCandidateIndex(candidates=(a, b))
        assert len(index) == 2


class TestSequenceFamilyId:
    """SC-002: sequence_family_id + identity hierarchy."""

    def test_same_sequence_family_id_across_trust_levels(self) -> None:
        candidates = generate_candidates()
        seqs = [c for c in candidates if c.sequence_id]
        by_family: dict[str, set[str]] = {}
        for c in seqs:
            assert c.sequence_family_id.startswith("sf_")
            by_family.setdefault(c.sequence_family_id, set()).add(c.trust_level)
        assert by_family, "no sequence families generated"
        for family_id, trust_levels in by_family.items():
            assert trust_levels == {"low", "default", "high"}, family_id

    def test_pairing_identity_hierarchy(self) -> None:
        base = _fc()
        assert pairing_identity(base) == ("candidate_id", base.candidate_id)
        with_family = _fc(candidate_family_id="cf_x")
        assert pairing_identity(with_family) == ("candidate_family_id", "cf_x")
        with_seq = _fc(candidate_family_id="cf_x", sequence_id="s1")
        assert pairing_identity(with_seq) == ("sequence_id", "s1")
        with_seqfam = _fc(sequence_id="s1", sequence_family_id="sf_x")
        assert pairing_identity(with_seqfam) == ("sequence_family_id", "sf_x")

    def test_validate_rejects_mixed_scenario_in_candidate_family(self) -> None:
        family = "cf_pilot_credential_sv001_direct_probe_000"
        a = _fc(candidate_id="c001", candidate_family_id=family, trust_level="low")
        b = _fc(
            candidate_id="c002",
            candidate_family_id=family,
            trust_level="high",
            scenario_id="pilot_authorization",
        )
        with pytest.raises(ValueError, match="mixes identity fields"):
            validate_family_identity([a, b])

    def test_validate_rejects_misencoded_family_id(self) -> None:
        a = _fc(
            candidate_id="c001",
            candidate_family_id="cf_wrong_scenario_sv001_direct_probe_000",
            trust_level="low",
        )
        with pytest.raises(ValueError, match="does not encode"):
            validate_family_identity([a])

    def test_validate_rejects_incomplete_sequence_step_positions(self) -> None:
        family = "sf_pilot_credential_sv001_multi_step_reconstruction_000"
        steps = [
            _fc(
                candidate_id=f"c{i:03d}",
                candidate_family_id="",
                sequence_id=f"s_{trust}",
                sequence_family_id=family,
                trust_level=trust,
                attack_type="multi_step_reconstruction",
                sequence_step_index=i,
                sequence_step_count=3,
            )
            for i, trust in enumerate(["low", "low", "low"])
        ]
        # Drop step 2: positions {0, 1} != {0, 1, 2}.
        with pytest.raises(ValueError, match="step positions"):
            validate_family_identity(steps[:2])

    def test_generated_corpus_passes_family_validation(self) -> None:
        validate_family_identity(generate_candidates())


class TestContentHashes:
    """SC-003: frozen normalization and content hashes."""

    def test_normalization_nfc_and_line_endings(self) -> None:
        # NFC: e + combining acute == precomposed é.
        decomposed = "caf\u0065\u0301"
        assert normalize_candidate_text(decomposed) == "caf\u00e9"
        assert normalize_candidate_text("a\r\nb\rc") == "a\nb\nc"

    def test_normalization_strips_edges_only(self) -> None:
        assert normalize_candidate_text("  hello world  ") == "hello world"
        # Internal whitespace, punctuation and case are preserved.
        assert normalize_candidate_text("Hello,  World!") == "Hello,  World!"

    def test_content_hash_is_sha256_of_normalized_text(self) -> None:
        text = "  The code is 0107.\r\n"
        expected = hashlib.sha256("The code is 0107.".encode("utf-8")).hexdigest()
        assert candidate_content_hash(text) == expected

    def test_family_hash_covers_step_order(self) -> None:
        forward = family_content_hash_for_steps([(0, "alpha"), (1, "beta")])
        swapped = family_content_hash_for_steps([(0, "beta"), (1, "alpha")])
        assert forward != swapped
        # Step index order, not input order, defines the hash.
        assert forward == family_content_hash_for_steps([(1, "beta"), (0, "alpha")])

    def test_fixed_content_family_hashes_match_across_trust(self) -> None:
        candidates = generate_candidates()
        singles = [c for c in candidates if not c.sequence_id and c.candidate_family_id]
        by_family: dict[str, list[FrozenCandidate]] = {}
        for c in singles:
            by_family.setdefault(c.candidate_family_id, []).append(c)
        for members in by_family.values():
            hashes = {m.content_hash for m in members}
            family_hashes = {m.family_content_hash for m in members}
            assert len(hashes) == 1, "RQ6 eligibility: identical content per family"
            assert family_hashes == hashes, "single-message family hash equals content hash"
            assert all(not m.trust_conditioned_generation for m in members)

    def test_sequence_family_hash_shared_across_trust(self) -> None:
        candidates = generate_candidates()
        by_family: dict[str, set[str]] = {}
        for c in candidates:
            if c.sequence_family_id:
                by_family.setdefault(c.sequence_family_id, set()).add(c.family_content_hash)
        for family_id, hashes in by_family.items():
            assert len(hashes) == 1, family_id

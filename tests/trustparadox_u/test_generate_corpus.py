"""Tests for Iteration 7: Frozen corpus generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.candidates import FrozenCandidate  # noqa: E402
from experiments.trustparadox_u.generate_corpus import (  # noqa: E402
    SCENARIO_DEFINITIONS,
    SEQUENCE_ATTACK_TYPES,
    TRUST_LEVELS,
    assign_splits,
    build_corpus_manifest,
    build_target_specs,
    build_trust_manipulation_report,
    generate_candidates,
    get_variant_definition,
    target_spec_for_variant,
    target_specs_for_scenario,
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

    def test_trust_level_conditions_generation(self) -> None:
        """FF92-002: trust level changes candidate identity and generation prompt.

        Candidates differing only in trust level must have distinct IDs
        and distinct trust-conditioned generation prompt hashes.
        """
        candidates = generate_candidates()
        by_level = {
            level: next(
                c
                for c in candidates
                if c.scenario_id == "credential_001"
                and c.secret_variant_id == "sv_cred_0107"
                and c.attack_type == "direct"
                and c.trust_level == level
            )
            for level in TRUST_LEVELS
        }
        ids = {c.candidate_id for c in by_level.values()}
        assert len(ids) == 3
        for level, c in by_level.items():
            assert f"_{level}_" in c.candidate_id
        hashes = {c.generation_prompt_hash for c in by_level.values()}
        assert len(hashes) == 3

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

    def test_ff92_013_text_change_changes_manifest_hash(self) -> None:
        """FF92-013: changing candidate text changes corpus_sha256."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        m1 = build_corpus_manifest(candidates, splits, "abc123")

        altered = []
        for c in candidates:
            if c.candidate_id == candidates[0].candidate_id:
                altered.append(
                    FrozenCandidate(
                        candidate_id=c.candidate_id,
                        scenario_id=c.scenario_id,
                        trust_level=c.trust_level,
                        attack_type=c.attack_type,
                        secret_variant_id=c.secret_variant_id,
                        sample_index=c.sample_index,
                        sender_id=c.sender_id,
                        recipient_id=c.recipient_id,
                        candidate_text=c.candidate_text + " TAMPERED",
                        sequence_id=c.sequence_id,
                        sequence_step_index=c.sequence_step_index,
                        sequence_step_count=c.sequence_step_count,
                        generation_model=c.generation_model,
                        generation_temperature=c.generation_temperature,
                        generation_prompt_hash=c.generation_prompt_hash,
                        corpus_version=c.corpus_version,
                        target_forget_ids=c.target_forget_ids,
                    )
                )
            else:
                altered.append(c)
        m2 = build_corpus_manifest(altered, splits, "abc123")
        assert m1["corpus_sha256"] != m2["corpus_sha256"]

    def test_ff92_013_timestamp_only_change_keeps_hash(self) -> None:
        """FF92-013: the generated_at timestamp is excluded from the hash."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        m1 = build_corpus_manifest(candidates, splits, "abc123")
        m2 = build_corpus_manifest(candidates, splits, "abc123")
        # Timestamps may legitimately differ between builds...
        # ...but the scientific content hash must not.
        assert m1["corpus_sha256"] == m2["corpus_sha256"]

    def test_ff92_013_manifest_hash_matches_independent_recompute(self, tmp_path: Path) -> None:
        """FF92-013: written corpus allows independent hash recomputation."""
        from experiments.trustparadox_u.candidates import canonical_jsonl_hash

        candidates = generate_candidates()
        splits = assign_splits(candidates)
        manifest = build_corpus_manifest(candidates, splits, "abc123")
        write_corpus(candidates, splits, manifest, tmp_path)

        # Independently recompute from the serialized JSONL
        records = []
        for line in (tmp_path / "frozen_corpus.jsonl").read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records.append(
                    {
                        "candidate_id": rec["candidate_id"],
                        "scenario_id": rec["scenario_id"],
                        "trust_level": rec["trust_level"],
                        "attack_type": rec["attack_type"],
                        "secret_variant_id": rec["secret_variant_id"],
                        "sample_index": rec["sample_index"],
                        "sender_id": rec["sender_id"],
                        "recipient_id": rec["recipient_id"],
                        "candidate_text": rec["candidate_text"],
                        "sequence_id": rec.get("sequence_id", ""),
                        "sequence_step_index": int(rec.get("sequence_step_index", 0)),
                        "sequence_step_count": int(rec.get("sequence_step_count", 0)),
                        "target_forget_ids": list(rec.get("target_forget_ids", [])),
                        "generation_model": rec.get("generation_model", ""),
                        "generation_temperature": float(rec.get("generation_temperature", 0.0)),
                        "generation_prompt_hash": rec.get("generation_prompt_hash", ""),
                        "corpus_version": rec.get("corpus_version", "1.0"),
                    }
                )
        assert canonical_jsonl_hash(records) == manifest["corpus_sha256"]


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


class TestTargetSpecsFF92003:
    """FF92-003: canonical per-variant target specifications."""

    def test_build_target_specs_covers_every_variant(self) -> None:
        """One spec per secret variant, consistent with SCENARIO_DEFINITIONS."""
        specs = build_target_specs()
        expected_count = sum(len(defn["secret_variants"]) for defn in SCENARIO_DEFINITIONS.values())
        assert len(specs) == expected_count
        by_scenario: dict[str, int] = {}
        for spec in specs:
            by_scenario[spec.scenario_id] = by_scenario.get(spec.scenario_id, 0) + 1
            variant = get_variant_definition(spec.scenario_id, spec.secret_variant_id)
            assert spec.canonical_target == variant["canonical_target"]
            assert spec.aliases == tuple(variant.get("aliases", []))
            assert spec.fragments == tuple(variant.get("fragments", []))
            assert spec.required_facts == tuple(variant.get("required_facts", []))
        for scenario_id, defn in SCENARIO_DEFINITIONS.items():
            assert by_scenario[scenario_id] == len(defn["secret_variants"])

    def test_target_spec_for_variant_fields(self) -> None:
        """sv_cred_0215 resolves to its own canonical content."""
        spec = target_spec_for_variant("credential_001", "sv_cred_0215")
        assert spec.forget_id == "F001"
        assert spec.target_type == "credential"
        assert spec.canonical_target == "0215"
        assert spec.fragments == ("02", "15")
        attr_spec = target_spec_for_variant("attribute_001", "sv_attr_bob_Y")
        assert attr_spec.forget_id == "F002"
        assert attr_spec.required_facts == (
            "employee_23_is_bob",
            "employee_23_has_M7",
            "M7_implies_condition_Y",
        )

    def test_unknown_scenario_and_variant_raise(self) -> None:
        """Unknown scenario or variant IDs fail loudly."""
        with pytest.raises(KeyError):
            get_variant_definition("no_such_scenario", "sv_cred_0107")
        with pytest.raises(KeyError):
            target_spec_for_variant("credential_001", "sv_cred_9999")
        with pytest.raises(KeyError):
            target_specs_for_scenario("no_such_scenario")

    def test_every_candidate_variant_resolves(self) -> None:
        """Every generated candidate references a resolvable variant."""
        candidates = generate_candidates()
        for c in candidates:
            spec = target_spec_for_variant(c.scenario_id, c.secret_variant_id)
            if c.target_forget_ids:
                assert tuple(c.target_forget_ids) == (spec.forget_id,), c.candidate_id

    def test_manifest_pins_target_specs(self) -> None:
        """Manifest records target spec count and a stable hash."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        specs = build_target_specs()
        m1 = build_corpus_manifest(candidates, splits, "abc123", target_specs=specs)
        m2 = build_corpus_manifest(candidates, splits, "abc123", target_specs=specs)
        assert m1["target_spec_count"] == len(specs)
        assert m1["target_spec_sha256"] == m2["target_spec_sha256"]
        assert len(m1["target_spec_sha256"]) == 64

    def test_target_specs_round_trip(self, tmp_path: Path) -> None:
        """target_specs.jsonl is written sorted and round-trips all fields."""
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        specs = build_target_specs()
        manifest = build_corpus_manifest(candidates, splits, "abc123", target_specs=specs)
        write_corpus(candidates, splits, manifest, tmp_path, target_specs=specs)

        path = tmp_path / "target_specs.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == len(specs)
        records = [json.loads(line) for line in lines]
        keys = [(r["scenario_id"], r["secret_variant_id"]) for r in records]
        assert keys == sorted(keys)
        by_variant = {(r["scenario_id"], r["secret_variant_id"]): r for r in records}
        spec = target_spec_for_variant("credential_001", "sv_cred_0215")
        record = by_variant[("credential_001", "sv_cred_0215")]
        assert record["canonical_target"] == spec.canonical_target
        assert record["fragments"] == list(spec.fragments)
        assert record["forget_id"] == spec.forget_id


class TestAliasOnlyChallengesFF92007:
    """FF92-007: alias challenges name the target via alias only."""

    def test_alias_candidates_contain_registered_alias(self) -> None:
        candidates = generate_candidates()
        alias_candidates = [c for c in candidates if c.attack_type == "alias"]
        assert len(alias_candidates) > 0
        for c in alias_candidates:
            spec = target_spec_for_variant(c.scenario_id, c.secret_variant_id)
            text_cf = c.candidate_text.casefold()
            assert any(
                alias.casefold() in text_cf for alias in spec.aliases
            ), f"{c.candidate_id} contains no registered alias"

    def test_alias_candidates_exclude_canonical_target(self) -> None:
        candidates = generate_candidates()
        alias_candidates = [c for c in candidates if c.attack_type == "alias"]
        assert len(alias_candidates) > 0
        for c in alias_candidates:
            spec = target_spec_for_variant(c.scenario_id, c.secret_variant_id)
            assert spec.canonical_target.casefold() not in c.candidate_text.casefold(), (
                f"{c.candidate_id} leaks the canonical target"
            )


class TestSemanticOnlyChallengesFF92008:
    """FF92-008: paraphrase challenges avoid every known surface form."""

    def test_paraphrase_candidates_exist_for_every_scenario(self) -> None:
        candidates = generate_candidates()
        scenarios = set(
            c.scenario_id for c in candidates if c.attack_type == "paraphrase"
        )
        assert scenarios == set(SCENARIO_DEFINITIONS)

    def test_paraphrase_candidates_avoid_all_surface_forms(self) -> None:
        candidates = generate_candidates()
        para_candidates = [c for c in candidates if c.attack_type == "paraphrase"]
        assert len(para_candidates) > 0
        for c in para_candidates:
            spec = target_spec_for_variant(c.scenario_id, c.secret_variant_id)
            forbidden = (
                [spec.canonical_target]
                + list(spec.aliases)
                + list(spec.semantic_variants)
                + list(spec.fragments)
                + list(spec.required_facts)
            )
            text_cf = c.candidate_text.casefold()
            for item in forbidden:
                if item:
                    assert item.casefold() not in text_cf, (
                        f"{c.candidate_id} reuses surface form {item!r}"
                    )

    def test_paraphrase_family_is_not_degenerate(self) -> None:
        """Each scenario's paraphrase family has at least two distinct texts."""
        candidates = generate_candidates()
        for scenario_id in SCENARIO_DEFINITIONS:
            texts = {
                c.candidate_text
                for c in candidates
                if c.attack_type == "paraphrase" and c.scenario_id == scenario_id
            }
            assert len(texts) >= 2, f"{scenario_id} paraphrase family has one surface form"


class TestReconstructionSequencesFF92009:
    """FF92-009: real multi-step reconstruction sequences."""

    def test_all_sequence_attack_types_emit_multistep_sequences(self) -> None:
        candidates = generate_candidates()
        for attack_type in SEQUENCE_ATTACK_TYPES:
            members = [c for c in candidates if c.attack_type == attack_type]
            assert members, f"No candidates for sequence attack {attack_type!r}"
            sequences: dict[str, list[FrozenCandidate]] = {}
            for c in members:
                assert c.sequence_id, f"{c.candidate_id} lacks sequence_id"
                sequences.setdefault(c.sequence_id, []).append(c)
            for seq_id, steps in sequences.items():
                assert len(steps) >= 2, f"Sequence {seq_id} is single-step"
                indices = sorted(s.sequence_step_index for s in steps)
                assert indices == list(range(len(steps))), f"Sequence {seq_id} gaps"
                counts = {s.sequence_step_count for s in steps}
                assert counts == {len(steps)}, f"Sequence {seq_id} inconsistent count"

    def test_one_sequence_per_cell(self) -> None:
        """Exactly one sequence_id per (scenario, variant, trust, attack) cell."""
        candidates = generate_candidates()
        by_key: dict[tuple, set[str]] = {}
        for c in candidates:
            if c.sequence_id:
                key = (c.scenario_id, c.secret_variant_id, c.trust_level, c.attack_type)
                by_key.setdefault(key, set()).add(c.sequence_id)
        assert by_key, "No sequence cells found"
        for key, ids in by_key.items():
            assert len(ids) == 1, f"Cell {key} spans sequences {ids}"

    def test_sequence_steps_are_one_record_each(self) -> None:
        """Every sequence step is its own candidate record with distinct ID."""
        candidates = generate_candidates()
        steps = [c for c in candidates if c.sequence_id]
        ids = [c.candidate_id for c in steps]
        assert len(ids) == len(set(ids))
        for c in steps:
            assert c.sequence_step_index < c.sequence_step_count


class TestTemplateFamilySplitsFF92010:
    """FF92-010: template-family split strategy without leakage."""

    def test_every_template_family_lives_in_one_split(self) -> None:
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        family_splits: dict[tuple[str, str], set[str]] = {}
        for split_name, split_candidates in splits.items():
            for c in split_candidates:
                family_splits.setdefault((c.scenario_id, c.attack_type), set()).add(
                    split_name
                )
        for family, split_names in family_splits.items():
            assert len(split_names) == 1, f"Family {family} spans {sorted(split_names)}"

    def test_sequences_do_not_straddle_splits(self) -> None:
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        split_of = {
            c.candidate_id: name for name, members in splits.items() for c in members
        }
        seq_splits: dict[str, set[str]] = {}
        for c in candidates:
            if c.sequence_id:
                seq_splits.setdefault(c.sequence_id, set()).add(split_of[c.candidate_id])
        for seq_id, names in seq_splits.items():
            assert len(names) == 1, f"Sequence {seq_id} spans {sorted(names)}"

    def test_no_identical_text_across_splits(self) -> None:
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        text_splits: dict[str, set[str]] = {}
        for split_name, split_candidates in splits.items():
            for c in split_candidates:
                text_splits.setdefault(c.candidate_text, set()).add(split_name)
        for text, names in text_splits.items():
            assert len(names) == 1, f"Text {text!r} appears in {sorted(names)}"

    def test_all_three_splits_nonempty_with_diverse_families(self) -> None:
        candidates = generate_candidates()
        splits = assign_splits(candidates)
        for split_name, split_candidates in splits.items():
            assert split_candidates, f"Split {split_name} is empty"
            families = {(c.scenario_id, c.attack_type) for c in split_candidates}
            assert len(families) >= 2, f"Split {split_name} has one template family"


class TestTrustConditioningFF92011:
    """FF92-011: trust-conditioned candidates with a manipulation check."""

    def test_trust_enters_only_via_generation_prompt_hash(self) -> None:
        candidates = generate_candidates()
        base = next(
            c
            for c in candidates
            if c.scenario_id == "attribute_001"
            and c.secret_variant_id == "sv_attr_alice_X"
            and c.attack_type == "direct"
        )
        others = [
            c
            for c in candidates
            if c.scenario_id == base.scenario_id
            and c.secret_variant_id == base.secret_variant_id
            and c.attack_type == base.attack_type
            and c.trust_level != base.trust_level
        ]
        assert others
        for c in others:
            assert c.generation_prompt_hash != base.generation_prompt_hash
            assert c.candidate_id != base.candidate_id

    def test_manipulation_report_structure(self) -> None:
        candidates = generate_candidates()
        report = build_trust_manipulation_report(candidates)
        assert report["trust_conditioning_source"] == "generation_prompt"
        assert report["firewall_configuration"] == "trust_independent"
        assert set(report["by_trust_level"]) == set(TRUST_LEVELS)
        hashes_per_level = {
            level: tuple(info["generation_prompt_hashes"])
            for level, info in report["by_trust_level"].items()
        }
        distinct = set(hashes_per_level.values())
        assert len(distinct) == len(TRUST_LEVELS), "Trust levels share prompt hashes"
        total = sum(info["candidate_count"] for info in report["by_trust_level"].values())
        assert total == len(candidates)

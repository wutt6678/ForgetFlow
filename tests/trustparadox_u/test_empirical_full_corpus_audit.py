"""Patch C/E/F/G/H/I: full auditor plan hash, campaign identity, and audit tests.

Tests that:
- Full auditor uses scientific plan hash (C1-C2)
- Campaign identity rejects mismatched blocking fields (E)
- Generated split requires campaign identity (F)
- Accepted sequence requires complete raw terminal sequence (G)
- Sequence report uses plan expected step count (H)
- Corpus manifest binds campaign identity hash (I)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.trustparadox_u.campaign_identity import (
    CampaignIdentity,
    CampaignIdentityMismatchError,
    campaign_identity_sha256,
    split_has_artifacts,
    verify_campaign_identity,
)
from experiments.trustparadox_u.empirical_generation_plan import (
    GenerationPlanItem,
    load_generation_plan,
    plan_sha256,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan_items() -> list[GenerationPlanItem]:
    """Create a minimal list of plan items for testing."""
    return [
        GenerationPlanItem(
            plan_item_id="pi_001",
            split="development",
            scenario_id="credential_001",
            secret_variant_id="credential_v1",
            trust_level="low",
            attack_type="direct_question",
            sample_index=0,
            generation_replicate=0,
        ),
        GenerationPlanItem(
            plan_item_id="pi_002",
            split="development",
            scenario_id="credential_001",
            secret_variant_id="credential_v1",
            trust_level="high",
            attack_type="direct_question",
            sample_index=0,
            generation_replicate=0,
        ),
    ]


def _write_plan_jsonl(path: Path, items: list[GenerationPlanItem]) -> None:
    """Write plan items to a JSONL file."""
    from dataclasses import asdict
    lines = []
    for item in items:
        lines.append(json.dumps(asdict(item), sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_identity(**overrides: object) -> CampaignIdentity:
    """Build a minimal CampaignIdentity for testing."""
    defaults = dict(
        schema_version="1.0",
        split="development",
        generation_plan_scientific_sha256="abc123",
        generation_plan_file_sha256="def456",
        generation_config_sha256="cfg789",
        target_registry_sha256="reg000",
        prompt_manifest_sha256="prm111",
        phase_manifest_sha256="pha222",
        generator_provider="openai",
        generator_model_requested="gpt-4",
        generator_temperature=0.7,
        generator_max_tokens=1024,
        request_timeout=60.0,
        max_retries=2,
        created_from_commit="abc123def456",
        created_at="2025-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return CampaignIdentity(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# Test C — plan hash verification
# ===========================================================================


class TestFullAuditorPlanHash:
    """Patch C: scientific and file plan hashes verified separately."""

    def test_full_auditor_uses_scientific_plan_hash(self, tmp_path: Path) -> None:
        """C1: Auditor accepts a committed plan with correct scientific hash."""
        items = _make_plan_items()
        plan_path = tmp_path / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)

        # Compute scientific hash from loaded items.
        loaded_items = load_generation_plan(plan_path)
        computed_scientific = plan_sha256(loaded_items)

        # Compute file hash.
        computed_file = hashlib.sha256(plan_path.read_bytes()).hexdigest()

        # Write summary with both hashes.
        summary = {
            "plan_scientific_sha256": computed_scientific,
            "plan_file_sha256": computed_file,
            "plan_sha256": computed_scientific,  # backward-compat alias.
        }
        summary_path = tmp_path / "full_generation_plan_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        # Verify: scientific hash matches.
        assert computed_scientific == summary["plan_scientific_sha256"]
        # Verify: file hash matches.
        assert computed_file == summary["plan_file_sha256"]
        # Verify: backward-compat alias equals scientific hash.
        assert summary["plan_sha256"] == summary["plan_scientific_sha256"]

    def test_full_auditor_checks_plan_file_hash_separately(
        self, tmp_path: Path
    ) -> None:
        """C2: Whitespace-only rewrite → scientific hash unchanged, file hash changed."""
        items = _make_plan_items()
        plan_path = tmp_path / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)

        loaded_items = load_generation_plan(plan_path)
        original_scientific = plan_sha256(loaded_items)
        original_file = hashlib.sha256(plan_path.read_bytes()).hexdigest()

        # Rewrite with extra whitespace (whitespace-only change).
        plan_path.write_text(
            "\n" + plan_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        # Scientific hash should be unchanged (same plan items).
        reloaded_items = load_generation_plan(plan_path)
        new_scientific = plan_sha256(reloaded_items)
        assert new_scientific == original_scientific

        # File hash should be changed (different bytes).
        new_file = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        assert new_file != original_file

    def test_semantic_plan_mutation_changes_scientific_hash(
        self, tmp_path: Path
    ) -> None:
        """C3: Changing one field → scientific hash mismatch."""
        items = _make_plan_items()
        plan_path = tmp_path / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)

        loaded_items = load_generation_plan(plan_path)
        original_scientific = plan_sha256(loaded_items)

        # Mutate one item.
        mutated_items = [
            replace(items[0], trust_level="high"),  # change trust_level
            *items[1:],
        ]
        mutated_scientific = plan_sha256(mutated_items)
        assert mutated_scientific != original_scientific


# ===========================================================================
# Test E — campaign identity strict blocking
# ===========================================================================


class TestCampaignIdentityBlocking:
    """Patch E: strict same-commit campaign identity enforcement."""

    def test_campaign_identity_rejects_different_source_commit(self) -> None:
        """Different created_from_commit → mismatch error."""
        existing = _base_identity(created_from_commit="commit_aaa")
        current = _base_identity(created_from_commit="commit_bbb")

        with pytest.raises(CampaignIdentityMismatchError) as exc_info:
            verify_campaign_identity(existing, current)
        assert "created_from_commit" in exc_info.value.mismatches

    def test_campaign_identity_rejects_changed_phase_manifest(self) -> None:
        """Different phase_manifest_sha256 → mismatch error."""
        existing = _base_identity(phase_manifest_sha256="phase_hash_1")
        current = _base_identity(phase_manifest_sha256="phase_hash_2")

        with pytest.raises(CampaignIdentityMismatchError) as exc_info:
            verify_campaign_identity(existing, current)
        assert "phase_manifest_sha256" in exc_info.value.mismatches

    def test_campaign_identity_rejects_changed_plan_file_hash(self) -> None:
        """Different generation_plan_file_sha256 → mismatch error.

        Even if scientific plan hash is the same, a reformatted file must fail.
        """
        existing = _base_identity(
            generation_plan_file_sha256="file_hash_original",
            generation_plan_scientific_sha256="scientific_same",
        )
        current = _base_identity(
            generation_plan_file_sha256="file_hash_reformatted",
            generation_plan_scientific_sha256="scientific_same",
        )

        with pytest.raises(CampaignIdentityMismatchError) as exc_info:
            verify_campaign_identity(existing, current)
        assert "generation_plan_file_sha256" in exc_info.value.mismatches

    def test_same_identity_passes_verification(self) -> None:
        """Identical identities → no error."""
        identity = _base_identity()
        # Should not raise.
        verify_campaign_identity(identity, identity)

    def test_mismatch_error_names_field(self) -> None:
        """Error message includes the mismatched field name."""
        existing = _base_identity(generator_temperature=0.7)
        current = _base_identity(generator_temperature=0.9)

        with pytest.raises(CampaignIdentityMismatchError) as exc_info:
            verify_campaign_identity(existing, current)
        assert "generator_temperature" in str(exc_info.value)
        assert "generator_temperature" in exc_info.value.mismatches


# ===========================================================================
# Test F — generated split requires campaign identity
# ===========================================================================


class TestGeneratedSplitRequiresCampaignIdentity:
    """Patch F: splits with artifacts must have campaign_identity.json."""

    def test_split_has_artifacts_detects_raw_attempts(self, tmp_path: Path) -> None:
        """split_has_artifacts returns True when raw_generation_attempts.jsonl exists."""
        assert not split_has_artifacts(tmp_path)
        (tmp_path / "raw_generation_attempts.jsonl").write_text("", encoding="utf-8")
        assert split_has_artifacts(tmp_path)

    def test_split_has_artifacts_detects_manifest(self, tmp_path: Path) -> None:
        """split_has_artifacts returns True when corpus_manifest.json exists."""
        (tmp_path / "corpus_manifest.json").write_text("{}", encoding="utf-8")
        assert split_has_artifacts(tmp_path)

    def test_split_has_artifacts_empty_dir(self, tmp_path: Path) -> None:
        """Empty directory has no artifacts."""
        assert not split_has_artifacts(tmp_path)

    def test_split_has_artifacts_unrelated_files(self, tmp_path: Path) -> None:
        """Unrelated files do not count as artifacts."""
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        assert not split_has_artifacts(tmp_path)


# ===========================================================================
# Test G — accepted sequence requires complete raw terminal sequence
# ===========================================================================


class TestAcceptedSequenceRequiresCompleteRawTerminalSequence:
    """Patch G: accepted sequences need complete raw terminal provenance."""

    def test_terminal_attempt_for_retry_chain_requires_retry0(self) -> None:
        """Complete-chain validator rejects a chain not starting at retry0."""
        from experiments.trustparadox_u.empirical_corpus import (
            EmpiricalGenerationAttempt,
            GenerationStatus,
        )
        from experiments.trustparadox_u.generate_empirical_corpus import (
            terminal_attempt_for_retry_chain,
        )

        # Build a chain starting at retry1 (missing retry0).
        chain = [
            EmpiricalGenerationAttempt(
                generation_attempt_id="att_001",
                scenario_id="s1",
                secret_variant_id="v1",
                split="development",
                trust_level="default",
                attack_type="direct_disclosure",
                sample_index=0,
                generation_replicate=0,
                sender_id="sender",
                recipient_id="recipient",
                candidate_family_id="fam_001",
                sequence_family_id=None,
                sequence_id=None,
                sequence_step_index=None,
                sequence_step_count=None,
                candidate_text="secret",
                generation_status=GenerationStatus.SUCCESS.value,
                refusal=False,
                malformed=False,
                off_topic=False,
                generator_provider="openai",
                generator_model="gpt-4",
                generator_revision=None,
                temperature=0.7,
                seed=None,
                system_prompt_hash="sys_hash",
                user_prompt_hash="usr_hash",
                request_id="req_001",
                retry_index=1,
                generated_at="2025-01-01T00:00:00+00:00",
                generation_mode="real",
            ),
        ]
        with pytest.raises(ValueError):
            terminal_attempt_for_retry_chain(chain)


# ===========================================================================
# Test H — sequence report uses plan expected step count
# ===========================================================================


class TestSequenceReportUsesPlanExpectedStepCount:
    """Patch H: expected step counts come from the frozen plan."""

    def test_expected_sequence_steps_from_plan_basic(self) -> None:
        """expected_sequence_steps_from_plan returns correct mapping."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            expected_sequence_steps_from_plan,
        )

        items = [
            GenerationPlanItem(
                plan_item_id="pi_001",
                split="development",
                scenario_id="s1",
                secret_variant_id="v1",
                trust_level="default",
                attack_type="fragmentation_sequence",
                sample_index=0,
                generation_replicate=0,
                sequence_id="seq_001",
                sequence_step_index=0,
                sequence_step_count=3,
            ),
            GenerationPlanItem(
                plan_item_id="pi_002",
                split="development",
                scenario_id="s1",
                secret_variant_id="v1",
                trust_level="default",
                attack_type="fragmentation_sequence",
                sample_index=0,
                generation_replicate=0,
                sequence_id="seq_001",
                sequence_step_index=1,
                sequence_step_count=3,
            ),
            GenerationPlanItem(
                plan_item_id="pi_003",
                split="development",
                scenario_id="s1",
                secret_variant_id="v1",
                trust_level="default",
                attack_type="fragmentation_sequence",
                sample_index=0,
                generation_replicate=0,
                sequence_id="seq_001",
                sequence_step_index=2,
                sequence_step_count=3,
            ),
        ]
        result = expected_sequence_steps_from_plan(items)
        expected_key = ("s1", "v1", "default", "fragmentation_sequence", 0, 0, "seq_001")
        assert result[expected_key] == 3

    def test_expected_sequence_steps_from_plan_inconsistent_raises(self) -> None:
        """Inconsistent step counts in plan raise ValueError."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            expected_sequence_steps_from_plan,
        )

        items = [
            GenerationPlanItem(
                plan_item_id="pi_001",
                split="development",
                scenario_id="s1",
                secret_variant_id="v1",
                trust_level="default",
                attack_type="fragmentation_sequence",
                sample_index=0,
                generation_replicate=0,
                sequence_id="seq_001",
                sequence_step_index=0,
                sequence_step_count=3,
            ),
            GenerationPlanItem(
                plan_item_id="pi_002",
                split="development",
                scenario_id="s1",
                secret_variant_id="v1",
                trust_level="default",
                attack_type="fragmentation_sequence",
                sample_index=0,
                generation_replicate=0,
                sequence_id="seq_001",
                sequence_step_index=1,
                sequence_step_count=2,  # inconsistent!
            ),
        ]
        with pytest.raises(ValueError, match="inconsistent sequence_step_count"):
            expected_sequence_steps_from_plan(items)

    def test_expected_sequence_steps_skips_non_sequence(self) -> None:
        """Non-sequence plan items are ignored."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            expected_sequence_steps_from_plan,
        )

        items = [
            GenerationPlanItem(
                plan_item_id="pi_001",
                split="development",
                scenario_id="s1",
                secret_variant_id="v1",
                trust_level="default",
                attack_type="direct_disclosure",
                sample_index=0,
                generation_replicate=0,
            ),
        ]
        result = expected_sequence_steps_from_plan(items)
        assert result == {}


# ===========================================================================
# Test I — corpus manifest binds campaign identity hash
# ===========================================================================


class TestCorpusManifestBindsCampaignIdentityHash:
    """Patch I: build_corpus_manifest includes campaign_identity_sha256."""

    def test_manifest_includes_identity_hash_when_provided(self) -> None:
        """build_corpus_manifest includes campaign_identity_sha256 when given."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            build_corpus_manifest,
        )

        manifest = build_corpus_manifest(
            generation_mode="real",
            attempts=[],
            accepted=[],
            prompt_manifest={},
            campaign_identity_hash="abc123def456",
        )
        assert manifest.get("campaign_identity_sha256") == "abc123def456"

    def test_manifest_omits_identity_hash_when_empty(self) -> None:
        """build_corpus_manifest omits campaign_identity_sha256 when empty."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            build_corpus_manifest,
        )

        manifest = build_corpus_manifest(
            generation_mode="real",
            attempts=[],
            accepted=[],
            prompt_manifest={},
            campaign_identity_hash="",
        )
        assert "campaign_identity_sha256" not in manifest

    def test_campaign_identity_sha256_is_deterministic(self) -> None:
        """campaign_identity_sha256 produces the same hash for the same identity."""
        identity = _base_identity()
        hash1 = campaign_identity_sha256(identity)
        hash2 = campaign_identity_sha256(identity)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_campaign_identity_sha256_changes_with_field(self) -> None:
        """Different identity fields produce different hashes."""
        id1 = _base_identity(generator_temperature=0.7)
        id2 = _base_identity(generator_temperature=0.9)
        assert campaign_identity_sha256(id1) != campaign_identity_sha256(id2)


# ===========================================================================
# Test Patch I — full-plan hash verification in validate_manifest_provenance
# ===========================================================================


class TestPatchIFullPlanHash:
    """Patch I: verify final full-plan hash value."""

    def test_manifest_full_generation_plan_hash_correct_passes(
        self, tmp_path: Path,
    ) -> None:
        """Correct full_generation_plan_sha256 → no finding."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor
        from experiments.trustparadox_u.campaign_identity import write_campaign_identity

        items = _make_plan_items()
        # Write frozen plan.
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        plan_path = manifests_dir / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)
        full_hash = plan_sha256(items)

        # Write manifest + identity.
        corpus_base = tmp_path / "corpus"
        split_dir = corpus_base / "development"
        split_dir.mkdir(parents=True)
        identity = _base_identity(
            generation_plan_scientific_sha256=full_hash,
        )
        manifest = {
            "artifact_class": "empirical_corpus",
            "split_generation_plan_sha256": full_hash,
            "full_generation_plan_sha256": full_hash,
            "campaign_identity_sha256": campaign_identity_sha256(identity),
            "repository_commit": identity.created_from_commit,
        }
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        write_campaign_identity(split_dir, identity)

        with (
            patch.object(auditor, "_CORPUS_BASE", corpus_base),
            patch.object(auditor, "_MANIFESTS_DIR", manifests_dir),
        ):
            findings = auditor.validate_manifest_provenance(
                target_splits=("development",),
            )
        assert not any("full_generation_plan_sha256 mismatch" in f for f in findings), (
            f"Findings: {findings}"
        )

    def test_manifest_full_generation_plan_hash_wrong_fails(
        self, tmp_path: Path,
    ) -> None:
        """Wrong full_generation_plan_sha256 → finding."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor
        from experiments.trustparadox_u.campaign_identity import write_campaign_identity

        items = _make_plan_items()
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        plan_path = manifests_dir / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)
        full_hash = plan_sha256(items)

        corpus_base = tmp_path / "corpus"
        split_dir = corpus_base / "development"
        split_dir.mkdir(parents=True)
        identity = _base_identity(
            generation_plan_scientific_sha256=full_hash,
        )
        manifest = {
            "artifact_class": "empirical_corpus",
            "split_generation_plan_sha256": full_hash,
            "full_generation_plan_sha256": "WRONG_FULL_HASH",
            "campaign_identity_sha256": campaign_identity_sha256(identity),
            "repository_commit": identity.created_from_commit,
        }
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        write_campaign_identity(split_dir, identity)

        with (
            patch.object(auditor, "_CORPUS_BASE", corpus_base),
            patch.object(auditor, "_MANIFESTS_DIR", manifests_dir),
        ):
            findings = auditor.validate_manifest_provenance(
                target_splits=("development",),
            )
        assert any("full_generation_plan_sha256 mismatch" in f for f in findings), (
            f"Findings: {findings}"
        )


# ===========================================================================
# Test Patch J — split plan item count verification
# ===========================================================================


class TestPatchJSplitPlanItemCount:
    """Patch J: verify final split plan item count."""

    def test_manifest_split_plan_item_count_correct_passes(
        self, tmp_path: Path,
    ) -> None:
        """Correct split_plan_item_count → no finding."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor
        from experiments.trustparadox_u.campaign_identity import write_campaign_identity
        from experiments.trustparadox_u.empirical_generation_plan import (
            plan_items_for_split,
        )

        items = _make_plan_items()
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        plan_path = manifests_dir / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)
        full_hash = plan_sha256(items)
        split_items = plan_items_for_split(items, "development")

        corpus_base = tmp_path / "corpus"
        split_dir = corpus_base / "development"
        split_dir.mkdir(parents=True)
        identity = _base_identity(
            generation_plan_scientific_sha256=full_hash,
        )
        manifest = {
            "artifact_class": "empirical_corpus",
            "split_generation_plan_sha256": full_hash,
            "full_generation_plan_sha256": full_hash,
            "split_plan_item_count": len(split_items),
            "campaign_identity_sha256": campaign_identity_sha256(identity),
            "repository_commit": identity.created_from_commit,
        }
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        write_campaign_identity(split_dir, identity)

        with (
            patch.object(auditor, "_CORPUS_BASE", corpus_base),
            patch.object(auditor, "_MANIFESTS_DIR", manifests_dir),
        ):
            findings = auditor.validate_manifest_provenance(
                target_splits=("development",),
            )
        assert not any("split_plan_item_count mismatch" in f for f in findings), (
            f"Findings: {findings}"
        )

    def test_manifest_split_plan_item_count_wrong_fails(
        self, tmp_path: Path,
    ) -> None:
        """Wrong split_plan_item_count → finding."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor
        from experiments.trustparadox_u.campaign_identity import write_campaign_identity

        items = _make_plan_items()
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        plan_path = manifests_dir / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)
        full_hash = plan_sha256(items)

        corpus_base = tmp_path / "corpus"
        split_dir = corpus_base / "development"
        split_dir.mkdir(parents=True)
        identity = _base_identity(
            generation_plan_scientific_sha256=full_hash,
        )
        manifest = {
            "artifact_class": "empirical_corpus",
            "split_generation_plan_sha256": full_hash,
            "full_generation_plan_sha256": full_hash,
            "split_plan_item_count": 999999,
            "campaign_identity_sha256": campaign_identity_sha256(identity),
            "repository_commit": identity.created_from_commit,
        }
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        write_campaign_identity(split_dir, identity)

        with (
            patch.object(auditor, "_CORPUS_BASE", corpus_base),
            patch.object(auditor, "_MANIFESTS_DIR", manifests_dir),
        ):
            findings = auditor.validate_manifest_provenance(
                target_splits=("development",),
            )
        assert any("split_plan_item_count mismatch" in f for f in findings), (
            f"Findings: {findings}"
        )


# ===========================================================================
# Test Patch K — split plan hash directly verified against frozen plan
# ===========================================================================


class TestPatchKSplitPlanHash:
    """Patch K: directly verify split plan hash against frozen plan."""

    def test_manifest_split_plan_hash_matches_frozen_plan(
        self, tmp_path: Path,
    ) -> None:
        """Manifest split hash == plan_sha256(split_items) → no finding."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor
        from experiments.trustparadox_u.campaign_identity import write_campaign_identity
        from experiments.trustparadox_u.empirical_generation_plan import (
            plan_items_for_split,
        )

        items = _make_plan_items()
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        plan_path = manifests_dir / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)
        split_items = plan_items_for_split(items, "development")
        split_hash = plan_sha256(split_items)

        corpus_base = tmp_path / "corpus"
        split_dir = corpus_base / "development"
        split_dir.mkdir(parents=True)
        identity = _base_identity(
            generation_plan_scientific_sha256=split_hash,
        )
        manifest = {
            "artifact_class": "empirical_corpus",
            "split_generation_plan_sha256": split_hash,
            "full_generation_plan_sha256": plan_sha256(items),
            "split_plan_item_count": len(split_items),
            "campaign_identity_sha256": campaign_identity_sha256(identity),
            "repository_commit": identity.created_from_commit,
        }
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        write_campaign_identity(split_dir, identity)

        with (
            patch.object(auditor, "_CORPUS_BASE", corpus_base),
            patch.object(auditor, "_MANIFESTS_DIR", manifests_dir),
        ):
            findings = auditor.validate_manifest_provenance(
                target_splits=("development",),
            )
        assert not any("split_generation_plan_sha256" in f for f in findings), (
            f"Findings: {findings}"
        )

    def test_manifest_split_plan_hash_wrong_fails(
        self, tmp_path: Path,
    ) -> None:
        """Wrong split_generation_plan_sha256 → finding."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor
        from experiments.trustparadox_u.campaign_identity import write_campaign_identity
        from experiments.trustparadox_u.empirical_generation_plan import (
            plan_items_for_split,
        )

        items = _make_plan_items()
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        plan_path = manifests_dir / "full_generation_plan.jsonl"
        _write_plan_jsonl(plan_path, items)
        split_items = plan_items_for_split(items, "development")
        split_hash = plan_sha256(split_items)

        corpus_base = tmp_path / "corpus"
        split_dir = corpus_base / "development"
        split_dir.mkdir(parents=True)
        identity = _base_identity(
            generation_plan_scientific_sha256=split_hash,
        )
        manifest = {
            "artifact_class": "empirical_corpus",
            "split_generation_plan_sha256": "WRONG_SPLIT_HASH",
            "full_generation_plan_sha256": plan_sha256(items),
            "split_plan_item_count": len(split_items),
            "campaign_identity_sha256": campaign_identity_sha256(identity),
            "repository_commit": identity.created_from_commit,
        }
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        write_campaign_identity(split_dir, identity)

        with (
            patch.object(auditor, "_CORPUS_BASE", corpus_base),
            patch.object(auditor, "_MANIFESTS_DIR", manifests_dir),
        ):
            findings = auditor.validate_manifest_provenance(
                target_splits=("development",),
            )
        assert any("split_generation_plan_sha256" in f for f in findings), (
            f"Findings: {findings}"
        )


# ---------------------------------------------------------------------------
# Patch H: checks_run explicit evidence in validation report
# ---------------------------------------------------------------------------


class TestPatchHChecksRun:
    """Patch H: build_validation_report must include checks_run evidence."""

    def test_validation_report_includes_checks_run(self) -> None:
        """build_validation_report() return value must contain checks_run."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor

        _patches = [
            patch.object(auditor, "_load_attempts", return_value=[]),
            patch.object(auditor, "_load_candidates", return_value=[]),
            patch.object(auditor, "_load_plan_items", return_value=[]),
            patch.object(auditor, "validate_phase_and_provenance", return_value=[]),
            patch.object(auditor, "validate_plan_completeness", return_value=[]),
            patch.object(auditor, "validate_split_integrity", return_value=[]),
            patch.object(auditor, "validate_identity_uniqueness", return_value=[]),
            patch.object(auditor, "validate_variant_consistency", return_value=[]),
            patch.object(auditor, "validate_config_consistency", return_value=[]),
            patch.object(auditor, "validate_hash_integrity", return_value=[]),
            patch.object(auditor, "validate_retry_lineage", return_value=[]),
            patch.object(auditor, "validate_sequence_atomicity", return_value=[]),
            patch.object(auditor, "validate_acceptance_independence", return_value=[]),
            patch.object(auditor, "validate_campaign_identity", return_value=[]),
            patch.object(auditor, "validate_artifact_classification", return_value=[]),
            patch.object(auditor, "validate_manifest_provenance", return_value=[]),
            patch.object(
                auditor, "validate_empirical_corpus_required_fields",
                return_value=[],
            ),
            patch.object(auditor, "_validate_all_split_gates", return_value=[]),
            patch.object(
                auditor, "validate_source_commit_consistency", return_value=[],
            ),
            patch.object(
                auditor, "validate_endpoint_consistency", return_value=[],
            ),
            patch.object(
                auditor, "compute_coverage_stats", return_value={},
            ),
        ]
        with contextlib.ExitStack() as stack:
            for p in _patches:
                stack.enter_context(p)
            report = auditor.build_validation_report(split_scope="all")

        assert "checks_run" in report, "validation report must include checks_run"
        checks = report["checks_run"]
        assert isinstance(checks, list)
        # Core audit sections must appear in checks_run.
        for expected in [
            "phase_provenance",
            "plan_completeness",
            "hash_integrity",
            "campaign_identity",
            "manifest_provenance",
            "split_gate_progression",
            "source_commit_consistency",
            "endpoint_consistency",
        ]:
            assert expected in checks, f"checks_run missing {expected!r}"


# ---------------------------------------------------------------------------
# Patch M/N: cross-split endpoint consistency regression
# ---------------------------------------------------------------------------


class TestCrossSplitEndpointConsistency:
    """validate_endpoint_consistency across splits."""

    def _write_identity(self, split_dir: Path, **endpoint_fields: str) -> None:
        """Write a minimal campaign_identity.json with endpoint fields."""
        split_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "schema_version": "1.0",
            "split": split_dir.name,
            "generation_plan_scientific_sha256": "plan",
            "generation_plan_file_sha256": "plan_file",
            "generation_config_sha256": "config",
            "target_registry_sha256": "registry",
            "prompt_manifest_sha256": "prompt",
            "phase_manifest_sha256": "phase",
            "generator_provider": "openai",
            "generator_model_requested": "gpt-4",
            "generator_temperature": 0.7,
            "generator_max_tokens": 1024,
            "request_timeout": 30.0,
            "max_retries": 3,
            "created_from_commit": "a" * 40,
            "created_at": "2026-08-02T00:00:00+00:00",
            "serving_endpoint_host": "",
            "serving_endpoint_sha256": "",
            "api_protocol": "",
        }
        identity.update(endpoint_fields)
        (split_dir / "campaign_identity.json").write_text(
            json.dumps(identity), encoding="utf-8",
        )

    def test_final_audit_rejects_cross_split_endpoint_drift(
        self, tmp_path: Path,
    ) -> None:
        """Different endpoint across splits → finding."""
        sha_a = hashlib.sha256(b"https://endpoint-a.example/v1").hexdigest()
        sha_b = hashlib.sha256(b"https://endpoint-b.example/v1").hexdigest()
        self._write_identity(
            tmp_path / "development",
            serving_endpoint_host="endpoint-a.example",
            serving_endpoint_sha256=sha_a,
            api_protocol="openai_compatible",
        )
        self._write_identity(
            tmp_path / "validation",
            serving_endpoint_host="endpoint-b.example",
            serving_endpoint_sha256=sha_b,
            api_protocol="openai_compatible",
        )
        self._write_identity(
            tmp_path / "test",
            serving_endpoint_host="endpoint-a.example",
            serving_endpoint_sha256=sha_a,
            api_protocol="openai_compatible",
        )
        from experiments.trustparadox_u import audit_empirical_corpus as auditor
        with patch.object(auditor, "_CORPUS_BASE", tmp_path):
            findings = auditor.validate_endpoint_consistency()
        assert findings, "Expected endpoint consistency findings"
        assert any("inconsistency" in f for f in findings), f"Findings: {findings}"

    def test_final_audit_accepts_same_endpoint_all_splits(
        self, tmp_path: Path,
    ) -> None:
        """Same endpoint across all splits → no finding."""
        sha = hashlib.sha256(b"https://endpoint.example/v1").hexdigest()
        for split in ("development", "validation", "test"):
            self._write_identity(
                tmp_path / split,
                serving_endpoint_host="endpoint.example",
                serving_endpoint_sha256=sha,
                api_protocol="openai_compatible",
            )
        from experiments.trustparadox_u import audit_empirical_corpus as auditor
        with patch.object(auditor, "_CORPUS_BASE", tmp_path):
            findings = auditor.validate_endpoint_consistency()
        assert not findings, f"Unexpected findings: {findings}"

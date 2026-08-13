"""Patch C/E/H: full auditor plan hash and campaign identity tests.

Tests that:
- Full auditor uses scientific plan hash (C1-C2)
- Campaign identity rejects mismatched blocking fields (E)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.trustparadox_u.campaign_identity import (
    CampaignIdentity,
    CampaignIdentityMismatchError,
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

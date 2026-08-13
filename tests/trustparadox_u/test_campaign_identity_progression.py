"""Patch G (Phase 3 Final): non-stubbed campaign-identity regression test.

Uses the real committed 900-item frozen plan and real validators
(without stubbing validate_campaign_identity, compute_campaign_identity,
or verify_campaign_identity) to certify campaign-identity consistency.

Also covers:
- manifest split-plan hash matches campaign identity (Patch A/H)
- final all-split source-commit consistency (Patch F)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.trustparadox_u.campaign_identity import (
    CampaignIdentity,
    campaign_identity_sha256,
    compute_campaign_identity,
    verify_campaign_identity,
    write_campaign_identity,
)
from experiments.trustparadox_u.empirical_generation_plan import (
    load_frozen_generation_config,
    load_generation_plan,
    plan_items_for_split,
    plan_sha256,
)

# ---------------------------------------------------------------------------
# Paths to real committed data
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FULL_PLAN_PATH = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2"
    / "manifests" / "full_generation_plan.jsonl"
)
_CONFIG_PATH = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2"
    / "manifests" / "full_generation_config.json"
)
_PHASE_PATH = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2"
    / "manifests" / "empirical_phase.json"
)
_CORPUS_BASE = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_MANIFESTS_DIR = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "manifests"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_real_identity(split: str) -> CampaignIdentity:
    """Compute a real CampaignIdentity using the committed plan."""
    full_items = load_generation_plan(_FULL_PLAN_PATH)
    split_items = plan_items_for_split(full_items, split)
    config = load_frozen_generation_config()
    return compute_campaign_identity(
        split=split,
        plan_items=split_items,
        plan_path=_FULL_PLAN_PATH,
        config=config,
        config_path=_CONFIG_PATH,
        phase_manifest_path=_PHASE_PATH,
    )


# ---------------------------------------------------------------------------
# Plan item count assertions
# ---------------------------------------------------------------------------


class TestFrozenPlanCounts:
    """Verify the committed plan has the expected structure."""

    def test_full_plan_has_900_items(self) -> None:
        items = load_generation_plan(_FULL_PLAN_PATH)
        assert len(items) == 900

    def test_development_split_has_225_items(self) -> None:
        items = load_generation_plan(_FULL_PLAN_PATH)
        dev = plan_items_for_split(items, "development")
        assert len(dev) == 225

    def test_validation_split_has_225_items(self) -> None:
        items = load_generation_plan(_FULL_PLAN_PATH)
        val = plan_items_for_split(items, "validation")
        assert len(val) == 225

    def test_test_split_has_450_items(self) -> None:
        items = load_generation_plan(_FULL_PLAN_PATH)
        test = plan_items_for_split(items, "test")
        assert len(test) == 450


# ---------------------------------------------------------------------------
# Non-stubbed campaign identity tests (Patch G)
# ---------------------------------------------------------------------------


class TestCampaignIdentityAuditUsesSplitPlan:
    """Verify that audit and generation use the same split-specific plan hash.

    These tests do NOT stub compute_campaign_identity or
    validate_campaign_identity.  They use the real 900-item plan.
    """

    def test_campaign_identity_audit_uses_development_split_plan(
        self, tmp_path: Path,
    ) -> None:
        """Development identity must use 225-item split plan hash."""
        identity = _compute_real_identity("development")
        # The scientific hash must be from the 225-item split plan, not 900.
        full_items = load_generation_plan(_FULL_PLAN_PATH)
        dev_items = plan_items_for_split(full_items, "development")
        expected_hash = plan_sha256(dev_items)
        assert identity.generation_plan_scientific_sha256 == expected_hash
        # Sanity: it must differ from the full-plan hash.
        full_hash = plan_sha256(full_items)
        assert identity.generation_plan_scientific_sha256 != full_hash

    def test_campaign_identity_audit_uses_validation_split_plan(
        self, tmp_path: Path,
    ) -> None:
        """Validation identity must use 225-item split plan hash."""
        identity = _compute_real_identity("validation")
        full_items = load_generation_plan(_FULL_PLAN_PATH)
        val_items = plan_items_for_split(full_items, "validation")
        expected_hash = plan_sha256(val_items)
        assert identity.generation_plan_scientific_sha256 == expected_hash

    def test_campaign_identity_audit_uses_test_split_plan(
        self, tmp_path: Path,
    ) -> None:
        """Test identity must use 450-item split plan hash."""
        identity = _compute_real_identity("test")
        full_items = load_generation_plan(_FULL_PLAN_PATH)
        test_items = plan_items_for_split(full_items, "test")
        expected_hash = plan_sha256(test_items)
        assert identity.generation_plan_scientific_sha256 == expected_hash

    def test_campaign_identity_rejects_full_plan_hash_for_split(
        self,
    ) -> None:
        """Using full 900-item hash for a split identity must fail verification.

        Compute a real development identity with the correct split hash,
        then create a second identity with the full-plan hash.  Verify
        that verify_campaign_identity detects the mismatch.
        """
        correct = _compute_real_identity("development")
        full_items = load_generation_plan(_FULL_PLAN_PATH)
        full_hash = plan_sha256(full_items)
        # Build an incorrect identity using the full-plan hash.
        from dataclasses import replace
        incorrect = replace(
            correct,
            generation_plan_scientific_sha256=full_hash,
        )
        with pytest.raises(Exception):
            verify_campaign_identity(correct, incorrect)


# ---------------------------------------------------------------------------
# Non-stubbed auditor round-trip (Patch G)
# ---------------------------------------------------------------------------


class TestAuditorRoundTrip:
    """Run validate_campaign_identity() against real on-disk data.

    These tests write a real campaign identity into a temp directory
    and run the real auditor against it.
    """

    def test_auditor_validates_correct_development_identity(
        self, tmp_path: Path,
    ) -> None:
        """validate_campaign_identity returns no findings for a correct identity."""
        identity = _compute_real_identity("development")
        # Write the identity to a temp split dir.
        split_dir = tmp_path / "development"
        split_dir.mkdir()
        write_campaign_identity(split_dir, identity)

        # Patch the auditor's corpus base and manifests dir.
        from experiments.trustparadox_u import audit_empirical_corpus as auditor

        with (
            patch.object(auditor, "_CORPUS_BASE", tmp_path),
            patch.object(auditor, "_MANIFESTS_DIR", _MANIFESTS_DIR),
        ):
            findings = auditor.validate_campaign_identity(
                target_splits=("development",),
            )
        assert findings == []

    def test_auditor_rejects_wrong_plan_hash(
        self, tmp_path: Path,
    ) -> None:
        """validate_campaign_identity detects a wrong scientific hash."""
        identity = _compute_real_identity("development")
        # Replace the split hash with the full-plan hash.
        from dataclasses import replace
        full_items = load_generation_plan(_FULL_PLAN_PATH)
        full_hash = plan_sha256(full_items)
        wrong_identity = replace(
            identity,
            generation_plan_scientific_sha256=full_hash,
        )
        split_dir = tmp_path / "development"
        split_dir.mkdir()
        write_campaign_identity(split_dir, wrong_identity)

        from experiments.trustparadox_u import audit_empirical_corpus as auditor

        with (
            patch.object(auditor, "_CORPUS_BASE", tmp_path),
            patch.object(auditor, "_MANIFESTS_DIR", _MANIFESTS_DIR),
        ):
            findings = auditor.validate_campaign_identity(
                target_splits=("development",),
            )
        assert len(findings) > 0
        assert any("campaign identity mismatch" in f for f in findings)


# ---------------------------------------------------------------------------
# Manifest / campaign identity provenance (Patch H)
# ---------------------------------------------------------------------------


class TestManifestProvenance:
    """Verify manifest ↔ campaign identity hash bindings."""

    def test_manifest_split_plan_hash_matches_campaign_identity(
        self, tmp_path: Path,
    ) -> None:
        """Manifest split_generation_plan_sha256 must match identity scientific hash."""
        identity = _compute_real_identity("development")
        # Build a mock manifest with matching hashes.
        manifest = {
            "split_generation_plan_sha256": identity.generation_plan_scientific_sha256,
            "campaign_identity_sha256": campaign_identity_sha256(identity),
            "repository_commit": identity.created_from_commit,
        }
        split_dir = tmp_path / "development"
        split_dir.mkdir()
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        write_campaign_identity(split_dir, identity)

        from experiments.trustparadox_u import audit_empirical_corpus as auditor

        with (
            patch.object(auditor, "_CORPUS_BASE", tmp_path),
        ):
            findings = auditor.validate_manifest_provenance(
                target_splits=("development",),
            )
        assert findings == []


# ---------------------------------------------------------------------------
# Source commit consistency (Patch F)
# ---------------------------------------------------------------------------


class TestSourceCommitConsistency:
    """Verify final all-split source commit consistency audit."""

    def test_final_all_audit_requires_one_source_commit(
        self, tmp_path: Path,
    ) -> None:
        """All splits with same commit → no findings."""
        commit = "abc123"
        for split in ("development", "validation", "test"):
            split_dir = tmp_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            gate = {
                "split": split,
                "source_commit": commit,
                "audit_source_commit": commit,
                "generation_completed": True,
                "audit_passed": True,
            }
            (tmp_path / f"{split}_generation_gate.json").write_text(
                json.dumps(gate), encoding="utf-8",
            )
            identity = CampaignIdentity(
                schema_version="1.0",
                split=split,
                generation_plan_scientific_sha256="x",
                generation_plan_file_sha256="y",
                generation_config_sha256="z",
                target_registry_sha256="t",
                prompt_manifest_sha256="p",
                phase_manifest_sha256="ph",
                generator_provider="prov",
                generator_model_requested="model",
                generator_temperature=0.7,
                generator_max_tokens=1024,
                request_timeout=30.0,
                max_retries=3,
                created_from_commit=commit,
                created_at="2025-01-01T00:00:00",
            )
            write_campaign_identity(split_dir, identity)

        from experiments.trustparadox_u import audit_empirical_corpus as auditor

        with (
            patch.object(auditor, "_CORPUS_BASE", tmp_path),
        ):
            findings = auditor.validate_source_commit_consistency()
        assert findings == []

    def test_final_all_audit_rejects_one_split_from_different_commit(
        self, tmp_path: Path,
    ) -> None:
        """One split with a different commit → blocking finding."""
        commit_a = "aaa111"
        commit_b = "bbb222"
        for split in ("development", "validation", "test"):
            split_dir = tmp_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            c = commit_b if split == "test" else commit_a
            gate = {
                "split": split,
                "source_commit": c,
                "audit_source_commit": c,
                "generation_completed": True,
                "audit_passed": True,
            }
            (tmp_path / f"{split}_generation_gate.json").write_text(
                json.dumps(gate), encoding="utf-8",
            )
            identity = CampaignIdentity(
                schema_version="1.0",
                split=split,
                generation_plan_scientific_sha256="x",
                generation_plan_file_sha256="y",
                generation_config_sha256="z",
                target_registry_sha256="t",
                prompt_manifest_sha256="p",
                phase_manifest_sha256="ph",
                generator_provider="prov",
                generator_model_requested="model",
                generator_temperature=0.7,
                generator_max_tokens=1024,
                request_timeout=30.0,
                max_retries=3,
                created_from_commit=c,
                created_at="2025-01-01T00:00:00",
            )
            write_campaign_identity(split_dir, identity)

        from experiments.trustparadox_u import audit_empirical_corpus as auditor

        with (
            patch.object(auditor, "_CORPUS_BASE", tmp_path),
        ):
            findings = auditor.validate_source_commit_consistency()
        assert len(findings) > 0
        assert any("source commit inconsistency" in f for f in findings)

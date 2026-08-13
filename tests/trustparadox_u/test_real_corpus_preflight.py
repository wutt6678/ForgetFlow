"""Tests for Patch M: fresh-preflight integration and artifact ordering.

These tests verify that verify_preflight() correctly reads artifacts that
were written BEFORE the verification step, not after.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_real_corpus_preflight import (
    main as preflight_main,
    verify_preflight,
)


# ---------------------------------------------------------------------------
# Patch M: fresh-preflight integration tests
# ---------------------------------------------------------------------------


class TestFreshPreflightOrdering:
    """Verify that preflight writes artifacts BEFORE verification."""

    def test_fresh_preflight_verifies_manifest_after_write(
        self, tmp_path: Path
    ) -> None:
        """Empty output dir → preflight writes all artifacts → verify PASS."""
        output_dir = tmp_path / "preflight_output"
        output_dir.mkdir()
    
        # Mock run_preflight_generation to return empty results.
        with patch(
            "scripts.run_real_corpus_preflight.run_preflight_generation"
        ) as mock_run:
            # Return minimal valid results (empty lists are fine for an empty plan).
            mock_run.return_value=[]
    
            # Write a minimal campaign identity (normally created by run_preflight_generation).
            from experiments.trustparadox_u.campaign_identity import (
                CampaignIdentity,
            )
            from dataclasses import asdict
            identity = CampaignIdentity(
                schema_version="1.0",
                split="development",
                generation_plan_scientific_sha256="def456",
                generation_plan_file_sha256="plan123",
                generation_config_sha256="config123",
                target_registry_sha256="registry123",
                prompt_manifest_sha256="prompt123",
                phase_manifest_sha256="phase123",
                generator_provider="openai",
                generator_model_requested="gpt-4",
                generator_temperature=0.7,
                generator_max_tokens=1024,
                request_timeout=30.0,
                max_retries=3,
                created_from_commit="abc123",
                created_at="2026-08-02T00:00:00+00:00",
            )
            (output_dir / "campaign_identity.json").write_text(
                json.dumps(asdict(identity), indent=2), encoding="utf-8"
            )
    
            # Run preflight with a minimal plan.
            plan_path = tmp_path / "plan.jsonl"
            plan_path.write_text("[]\n", encoding="utf-8")
    
            rc = preflight_main([
                "--plan", str(plan_path),
                "--output-dir", str(output_dir),
                "--skip-preflight-checks",  # Skip git checks for test environment.
            ])
    
            # Preflight should PASS (rc=0) if all artifacts are written
            # before verify_preflight() is called.
            assert rc == 0, f"Preflight failed — artifacts may not be written before verification. RC={rc}"
    
            # Verify all expected artifacts exist.
            assert (output_dir / "campaign_identity.json").exists()
            assert (output_dir / "raw_generation_attempts.jsonl").exists()
            assert (output_dir / "accepted_candidates.jsonl").exists()
            assert (output_dir / "corpus_manifest.json").exists()
            assert (output_dir / "sequence_report.json").exists()

    def test_preflight_manifest_missing_fails(self, tmp_path: Path) -> None:
        """verify_preflight() must FAIL if corpus_manifest.json is missing."""
        output_dir = tmp_path / "preflight_output"
        output_dir.mkdir()

        # Write campaign_identity.json with proper schema.
        from experiments.trustparadox_u.campaign_identity import (
            CampaignIdentity,
        )
        from dataclasses import asdict
        identity = CampaignIdentity(
            schema_version="1.0",
            split="development",
            generation_plan_scientific_sha256="def456",
            generation_plan_file_sha256="plan123",
            generation_config_sha256="config123",
            target_registry_sha256="registry123",
            prompt_manifest_sha256="prompt123",
            phase_manifest_sha256="phase123",
            generator_provider="openai",
            generator_model_requested="gpt-4",
            generator_temperature=0.7,
            generator_max_tokens=1024,
            request_timeout=30.0,
            max_retries=3,
            created_from_commit="abc123",
            created_at="2026-08-02T00:00:00+00:00",
        )
        (output_dir / "campaign_identity.json").write_text(
            json.dumps(asdict(identity), indent=2), encoding="utf-8"
        )
        # Write minimal raw_attempts.jsonl so we pass earlier checks.
        (output_dir / "raw_generation_attempts.jsonl").write_text(
            "", encoding="utf-8"
        )
        
        # NOT creating corpus_manifest.json

        # verify_preflight should FAIL.
        findings = verify_preflight(output_dir, [], [], [])
        assert any("corpus_manifest.json" in f for f in findings), (
            f"verify_preflight did not report missing corpus_manifest.json. Findings: {findings}"
        )

    def test_preflight_manifest_identity_hash_missing_fails(
        self, tmp_path: Path
    ) -> None:
        """verify_preflight() must FAIL if manifest lacks campaign_identity_sha256."""
        output_dir = tmp_path / "preflight_output"
        output_dir.mkdir()

        # Write campaign_identity.json with proper schema.
        from experiments.trustparadox_u.campaign_identity import (
            CampaignIdentity,
        )
        from dataclasses import asdict
        identity = CampaignIdentity(
            schema_version="1.0",
            split="development",
            generation_plan_scientific_sha256="def456",
            generation_plan_file_sha256="plan123",
            generation_config_sha256="config123",
            target_registry_sha256="registry123",
            prompt_manifest_sha256="prompt123",
            phase_manifest_sha256="phase123",
            generator_provider="openai",
            generator_model_requested="gpt-4",
            generator_temperature=0.7,
            generator_max_tokens=1024,
            request_timeout=30.0,
            max_retries=3,
            created_from_commit="abc123",
            created_at="2026-08-02T00:00:00+00:00",
        )
        identity_path = output_dir / "campaign_identity.json"
        identity_path.write_text(json.dumps(asdict(identity), indent=2), encoding="utf-8")

        # Write minimal raw_attempts.jsonl (required for early checks).
        (output_dir / "raw_generation_attempts.jsonl").write_text("", encoding="utf-8")

        # Write corpus_manifest.json WITHOUT campaign_identity_sha256.
        manifest = {
            "artifact_class": "real_api_preflight",
            "research_use": "diagnostic_only",
            "split": "development",
            "plan_file_sha256": "plan123",
            "plan_scientific_sha256": "plan456",
            "plan_item_count": 0,
            "raw_generation_sha256": "raw789",
            "accepted_candidate_sha256": "acc012",
            # campaign_identity_sha256 is MISSING.
            "created_at": "2026-08-02T00:00:00Z",
        }
        (output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # verify_preflight should FAIL.
        findings = verify_preflight(output_dir, [], [], [])
        assert any("campaign_identity_sha256" in f for f in findings), (
            f"verify_preflight did not report missing campaign_identity_sha256. Findings: {findings}"
        )

    def test_preflight_manifest_identity_hash_mismatch_fails(
        self, tmp_path: Path
    ) -> None:
        """verify_preflight() must FAIL if manifest identity hash != actual."""
        output_dir = tmp_path / "preflight_output"
        output_dir.mkdir()

        # Write campaign_identity.json with proper schema.
        from experiments.trustparadox_u.campaign_identity import (
            CampaignIdentity,
            campaign_identity_sha256,
        )
        from dataclasses import asdict
        identity = CampaignIdentity(
            schema_version="1.0",
            split="development",
            generation_plan_scientific_sha256="def456",
            generation_plan_file_sha256="plan123",
            generation_config_sha256="config123",
            target_registry_sha256="registry123",
            prompt_manifest_sha256="prompt123",
            phase_manifest_sha256="phase123",
            generator_provider="openai",
            generator_model_requested="gpt-4",
            generator_temperature=0.7,
            generator_max_tokens=1024,
            request_timeout=30.0,
            max_retries=3,
            created_from_commit="abc123",
            created_at="2026-08-02T00:00:00+00:00",
        )
        identity_path = output_dir / "campaign_identity.json"
        identity_path.write_text(json.dumps(asdict(identity), indent=2), encoding="utf-8")
        actual_identity_sha = campaign_identity_sha256(identity)

        # Write minimal raw_attempts.jsonl (required for early checks).
        (output_dir / "raw_generation_attempts.jsonl").write_text("", encoding="utf-8")

        # Write corpus_manifest.json with WRONG campaign_identity_sha256.
        manifest = {
            "artifact_class": "real_api_preflight",
            "research_use": "diagnostic_only",
            "split": "development",
            "plan_file_sha256": "plan123",
            "plan_scientific_sha256": "plan456",
            "plan_item_count": 0,
            "raw_generation_sha256": "raw789",
            "accepted_candidate_sha256": "acc012",
            "campaign_identity_sha256": "WRONG_HASH",
            "created_at": "2026-08-02T00:00:00Z",
        }
        (output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # verify_preflight should FAIL.
        findings = verify_preflight(output_dir, [], [], [])
        assert any("campaign_identity_sha256" in f and ("mismatch" in f or "missing" in f) for f in findings), (
            f"verify_preflight did not report campaign_identity_sha256 mismatch. Findings: {findings}"
        )

    def test_preflight_plan_hash_mismatch_fails(self, tmp_path: Path) -> None:
        """verify_preflight() must FAIL if manifest plan hash != actual."""
        output_dir = tmp_path / "preflight_output"
        output_dir.mkdir()

        # Write campaign_identity.json with proper schema.
        from experiments.trustparadox_u.campaign_identity import (
            CampaignIdentity,
            campaign_identity_sha256,
        )
        from dataclasses import asdict
        identity = CampaignIdentity(
            schema_version="1.0",
            split="development",
            generation_plan_scientific_sha256="def456",
            generation_plan_file_sha256="plan123",
            generation_config_sha256="config123",
            target_registry_sha256="registry123",
            prompt_manifest_sha256="prompt123",
            phase_manifest_sha256="phase123",
            generator_provider="openai",
            generator_model_requested="gpt-4",
            generator_temperature=0.7,
            generator_max_tokens=1024,
            request_timeout=30.0,
            max_retries=3,
            created_from_commit="abc123",
            created_at="2026-08-02T00:00:00+00:00",
        )
        identity_path = output_dir / "campaign_identity.json"
        identity_path.write_text(json.dumps(asdict(identity), indent=2), encoding="utf-8")
        actual_identity_sha = campaign_identity_sha256(identity)

        # Write minimal raw_attempts.jsonl (required for early checks).
        (output_dir / "raw_generation_attempts.jsonl").write_text("", encoding="utf-8")

        # Write corpus_manifest.json with WRONG plan hash.
        manifest = {
            "artifact_class": "real_api_preflight",
            "research_use": "diagnostic_only",
            "split": "development",
            "plan_file_sha256": "plan123",
            "plan_scientific_sha256": "WRONG_PLAN_HASH",
            "plan_item_count": 0,
            "raw_generation_sha256": "raw789",
            "accepted_candidate_sha256": "acc012",
            "campaign_identity_sha256": actual_identity_sha,
            "created_at": "2026-08-02T00:00:00Z",
        }
        (output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # verify_preflight should FAIL.
        findings = verify_preflight(output_dir, [], [], [])
        assert any("plan hash" in f.lower() for f in findings), (
            "verify_preflight did not report plan hash mismatch"
        )

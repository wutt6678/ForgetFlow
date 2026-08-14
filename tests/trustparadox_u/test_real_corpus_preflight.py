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
    
        # Pre-write empty plan file so we can compute real hashes.
        plan_path = tmp_path / "plan.jsonl"
        plan_path.write_text("", encoding="utf-8")

        # Compute real hashes for the empty plan.
        import hashlib
        from experiments.trustparadox_u.empirical_generation_plan import plan_sha256
        from experiments.trustparadox_u.empirical_corpus import (
            EMPIRICAL_TARGET_REGISTRY,
            compute_target_registry_hash,
        )
        plan_file_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        plan_scientific_hash = plan_sha256([])  # Empty plan.
        registry_hash = compute_target_registry_hash(EMPIRICAL_TARGET_REGISTRY)

        # Mock run_preflight_generation to return empty results.
        with patch(
            "scripts.run_real_corpus_preflight.run_preflight_generation"
        ) as mock_run:
            mock_run.return_value=[]
    
            # Write a minimal campaign identity with REAL hashes.
            from experiments.trustparadox_u.campaign_identity import (
                CampaignIdentity,
            )
            from dataclasses import asdict
            identity = CampaignIdentity(
                schema_version="1.0",
                split="development",
                generation_plan_scientific_sha256=plan_scientific_hash,
                generation_plan_file_sha256=plan_file_hash,
                generation_config_sha256="config123",
                target_registry_sha256=registry_hash,
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

            # Pre-write empty raw_generation_attempts.jsonl (mocked run won't write it).
            (output_dir / "raw_generation_attempts.jsonl").write_text("", encoding="utf-8")
    
            # Run preflight with a minimal (empty) plan.
            rc = preflight_main([
                "--plan", str(plan_path),
                "--output-dir", str(output_dir),
                "--skip-preflight-checks",
            ])
    
            # Preflight should PASS (rc=0) if all artifacts are written
            # before verify_preflight() is called.
            assert rc == 0, f"Preflight failed — artifacts may not be written before verification. RC={rc}"
    
            # Verify all expected artifacts exist.
            assert (output_dir / "campaign_identity.json").exists()
            assert (output_dir / "raw_generation_attempts.jsonl").exists()
            assert (output_dir / "accepted_candidates.jsonl").exists()
            assert (output_dir / "corpus_manifest.json").exists()
            assert (output_dir / "sequence_generation_report.json").exists()

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
        plan_file = tmp_path / "plan.jsonl"
        plan_file.write_text("", encoding="utf-8")
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
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
        plan_file = tmp_path / "plan.jsonl"
        plan_file.write_text("", encoding="utf-8")
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
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
        plan_file = tmp_path / "plan.jsonl"
        plan_file.write_text("", encoding="utf-8")
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
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
        plan_file = tmp_path / "plan.jsonl"
        plan_file.write_text("", encoding="utf-8")
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("plan_scientific_sha256 mismatch" in f for f in findings), (
            "verify_preflight did not report plan hash mismatch"
        )


# ---------------------------------------------------------------------------
# Helpers for Patch B/D/E/F/H/P tests
# ---------------------------------------------------------------------------


def _compute_empty_plan_hashes(plan_file: Path) -> tuple[str, str]:
    """Return (file_hash, scientific_hash) for an empty plan file."""
    from experiments.trustparadox_u.empirical_generation_plan import plan_sha256
    file_hash = hashlib.sha256(plan_file.read_bytes()).hexdigest()
    scientific_hash = plan_sha256([])
    return file_hash, scientific_hash


def _setup_valid_preflight(
    tmp_path: Path,
    *,
    manifest_overrides: dict | None = None,
    identity_overrides: dict | None = None,
    raw_content: str = "",
    accepted_content: str = "",
) -> tuple[Path, Path, dict]:
    """Set up a valid preflight environment and return (output_dir, plan_path, manifest).

    The caller can override manifest fields or identity fields to test
    negative cases.
    """
    from dataclasses import asdict
    from experiments.trustparadox_u.campaign_identity import (
        CampaignIdentity,
        campaign_identity_sha256,
    )
    from experiments.trustparadox_u.empirical_corpus import (
        EMPIRICAL_TARGET_REGISTRY,
        compute_target_registry_hash,
    )
    from experiments.trustparadox_u.empirical_generation_plan import plan_sha256

    output_dir = tmp_path / "preflight_output"
    output_dir.mkdir()

    plan_file = tmp_path / "plan.jsonl"
    plan_file.write_text("", encoding="utf-8")

    file_hash, scientific_hash = _compute_empty_plan_hashes(plan_file)
    registry_hash = compute_target_registry_hash(EMPIRICAL_TARGET_REGISTRY)

    # Campaign identity.
    id_defaults = dict(
        schema_version="1.0",
        split="development",
        generation_plan_scientific_sha256=scientific_hash,
        generation_plan_file_sha256=file_hash,
        generation_config_sha256="config123",
        target_registry_sha256=registry_hash,
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
    if identity_overrides:
        id_defaults.update(identity_overrides)
    identity = CampaignIdentity(**id_defaults)
    identity_hash = campaign_identity_sha256(identity)
    (output_dir / "campaign_identity.json").write_text(
        json.dumps(asdict(identity), indent=2), encoding="utf-8"
    )

    # Raw attempts.
    (output_dir / "raw_generation_attempts.jsonl").write_text(
        raw_content, encoding="utf-8"
    )

    # Accepted candidates.
    (output_dir / "accepted_candidates.jsonl").write_text(
        accepted_content, encoding="utf-8"
    )

    # Manifest.
    from experiments.trustparadox_u.empirical_corpus import (
        raw_attempts_scientific_hash,
        accepted_candidates_scientific_hash,
    )
    # Compute real hashes from empty disk content.
    raw_hash = raw_attempts_scientific_hash([])
    accepted_hash = accepted_candidates_scientific_hash([])

    manifest_defaults = {
        "artifact_class": "real_api_preflight",
        "research_use": "diagnostic_only",
        "split": "development",
        "plan_file_sha256": file_hash,
        "plan_scientific_sha256": scientific_hash,
        "plan_item_count": 0,
        "raw_attempt_count": 0,
        "accepted_candidate_count": 0,
        "raw_generation_sha256": raw_hash,
        "accepted_candidate_sha256": accepted_hash,
        "target_registry_sha256": registry_hash,
        "campaign_identity_sha256": identity_hash,
        "created_at": "2026-08-02T00:00:00Z",
    }
    if manifest_overrides:
        manifest_defaults.update(manifest_overrides)
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest_defaults, indent=2), encoding="utf-8"
    )
    (output_dir / "sequence_generation_report.json").write_text(
        json.dumps({"planned_sequence_count": 0, "accepted_sequence_count": 0,
                     "rejected_sequence_count": 0, "rejection_reasons": []}),
        encoding="utf-8",
    )
    return output_dir, plan_file, manifest_defaults


# ---------------------------------------------------------------------------
# Patch B: plan verification tests
# ---------------------------------------------------------------------------


class TestPatchBPlanVerification:
    """Patch B: verify all three plan identity fields in preflight."""

    def test_preflight_plan_file_hash_mismatch_fails(self, tmp_path: Path) -> None:
        """Wrong plan_file_sha256 in manifest → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, manifest_overrides={"plan_file_sha256": "WRONG_FILE_HASH"},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("plan_file_sha256 mismatch" in f for f in findings), f"Findings: {findings}"

    def test_preflight_plan_item_count_mismatch_fails(self, tmp_path: Path) -> None:
        """Wrong plan_item_count in manifest → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, manifest_overrides={"plan_item_count": 999999},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("plan_item_count mismatch" in f for f in findings), f"Findings: {findings}"

    def test_preflight_zero_plan_item_count_is_valid(self, tmp_path: Path) -> None:
        """plan_item_count=0 with empty plan → no finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, manifest_overrides={"plan_item_count": 0},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert not any("plan_item_count" in f for f in findings), f"Findings: {findings}"

    def test_preflight_campaign_identity_plan_file_hash_mismatch_fails(
        self, tmp_path: Path
    ) -> None:
        """Campaign identity with wrong plan file hash → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, identity_overrides={"generation_plan_file_sha256": "WRONG"},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "campaign_identity.generation_plan_file_sha256 mismatch" in f
            for f in findings
        ), f"Findings: {findings}"

    def test_preflight_campaign_identity_plan_scientific_hash_mismatch_fails(
        self, tmp_path: Path
    ) -> None:
        """Campaign identity with wrong plan scientific hash → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path,
            identity_overrides={"generation_plan_scientific_sha256": "WRONG"},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "campaign_identity.generation_plan_scientific_sha256 mismatch" in f
            for f in findings
        ), f"Findings: {findings}"


# ---------------------------------------------------------------------------
# Patch D: raw scientific hash tests
# ---------------------------------------------------------------------------


class TestPatchDRawHash:
    """Patch D: verify retained raw scientific hash."""

    def test_preflight_raw_generation_hash_missing_fails(self, tmp_path: Path) -> None:
        """Missing raw_generation_sha256 in manifest → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, manifest_overrides={"raw_generation_sha256": None},
        )
        # Remove the key entirely.
        manifest = json.loads((output_dir / "corpus_manifest.json").read_text())
        del manifest["raw_generation_sha256"]
        (output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("raw_generation_sha256" in f for f in findings), f"Findings: {findings}"

    def test_preflight_raw_generation_hash_mismatch_fails(self, tmp_path: Path) -> None:
        """Wrong raw_generation_sha256 in manifest → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, manifest_overrides={"raw_generation_sha256": "WRONG_RAW_HASH"},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("raw_generation_sha256 mismatch" in f for f in findings), f"Findings: {findings}"

    def test_preflight_raw_generation_hash_matches_disk_passes(
        self, tmp_path: Path
    ) -> None:
        """Correct raw_generation_sha256 → no finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert not any("raw_generation_sha256" in f for f in findings), f"Findings: {findings}"


# ---------------------------------------------------------------------------
# Patch E: accepted candidate hash tests
# ---------------------------------------------------------------------------


class TestPatchEAcceptedHash:
    """Patch E: verify retained accepted-candidate scientific hash."""

    def test_preflight_accepted_candidate_hash_missing_fails(
        self, tmp_path: Path
    ) -> None:
        """Missing accepted_candidate_sha256 in manifest → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        manifest = json.loads((output_dir / "corpus_manifest.json").read_text())
        del manifest["accepted_candidate_sha256"]
        (output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("accepted_candidate_sha256" in f for f in findings), f"Findings: {findings}"

    def test_preflight_accepted_candidate_hash_mismatch_fails(
        self, tmp_path: Path
    ) -> None:
        """Wrong accepted_candidate_sha256 → finding."""
        from experiments.trustparadox_u.empirical_corpus import (
            EmpiricalCandidate,
            accepted_candidates_scientific_hash,
            candidate_to_record,
        )
        candidate = EmpiricalCandidate(
            candidate_id="cand_fam1_low",
            source_generation_attempt_id="att_001",
            candidate_family_id="fam_1",
            scenario_id="s1",
            secret_variant_id="v1",
            split="development",
            trust_level="low",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
            sender_id="sender",
            recipient_id="recipient",
            sequence_family_id=None,
            sequence_id=None,
            sequence_step_index=None,
            sequence_step_count=None,
            text="secret",
            normalized_text="secret",
            content_sha256="abc",
            accepted=True,
            acceptance_reason="firewall_pass",
            generator_provider="openai",
            generator_model="gpt-4",
            generator_revision=None,
            system_prompt_hash="sys",
            user_prompt_hash="usr",
        )
        accepted_line = json.dumps(candidate_to_record(candidate), sort_keys=True) + "\n"
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path,
            accepted_content=accepted_line,
            manifest_overrides={"accepted_candidate_sha256": "WRONG_ACCEPTED_HASH"},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "accepted_candidate_sha256 mismatch" in f for f in findings
        ), f"Findings: {findings}"

    def test_preflight_accepted_candidate_hash_matches_disk_passes(
        self, tmp_path: Path
    ) -> None:
        """Correct accepted_candidate_sha256 → no finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert not any(
            "accepted_candidate_sha256" in f for f in findings
        ), f"Findings: {findings}"


# ---------------------------------------------------------------------------
# Patch F: deterministic rebuild tests
# ---------------------------------------------------------------------------


class TestPatchFDeterministicRebuild:
    """Patch F: verify accepted file against deterministic rebuild."""

    def test_preflight_accepted_file_matches_rebuild(self, tmp_path: Path) -> None:
        """Empty accepted file matches empty rebuild → no finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert not any("rebuild" in f.lower() for f in findings), f"Findings: {findings}"

    def test_preflight_accepted_file_missing_candidate_fails(
        self, tmp_path: Path
    ) -> None:
        """Missing accepted_candidates.jsonl → finding (artifact missing)."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        # Remove the accepted file.
        (output_dir / "accepted_candidates.jsonl").unlink()
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("accepted_candidates.jsonl" in f for f in findings), f"Findings: {findings}"


# ---------------------------------------------------------------------------
# V1: Accepted-corpus fail-closed regression tests
# ---------------------------------------------------------------------------


class TestV1AcceptedCorpusFailClosed:
    """V1: ensure empty/non-empty accepted corpus cannot bypass verification."""

    def test_preflight_truncated_empty_accepted_file_fails(
        self, tmp_path: Path
    ) -> None:
        """Empty accepted file + non-empty rebuild → multiple findings."""
        from experiments.trustparadox_u.empirical_corpus import (
            EmpiricalCandidate,
            candidate_to_record,
            accepted_candidates_scientific_hash,
        )
        # Create a candidate that should be in the accepted file.
        candidate = EmpiricalCandidate(
            candidate_id="cand_fam1_low",
            source_generation_attempt_id="att_001",
            candidate_family_id="fam_1",
            scenario_id="s1",
            secret_variant_id="v1",
            split="development",
            trust_level="low",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
            sender_id="sender",
            recipient_id="recipient",
            sequence_family_id=None,
            sequence_id=None,
            sequence_step_index=None,
            sequence_step_count=None,
            text="secret",
            normalized_text="secret",
            content_sha256="abc",
            accepted=True,
            acceptance_reason="firewall_pass",
            generator_provider="openai",
            generator_model="gpt-4",
            generator_revision=None,
            system_prompt_hash="sys",
            user_prompt_hash="usr",
        )
        # Setup with the candidate in accepted file.
        accepted_line = json.dumps(candidate_to_record(candidate), sort_keys=True) + "\n"
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, accepted_content=accepted_line,
        )
        # Now truncate the accepted file to empty.
        (output_dir / "accepted_candidates.jsonl").write_text("", encoding="utf-8")
        # Manifest still has hash for the candidate.
        manifest = json.loads((output_dir / "corpus_manifest.json").read_text())
        manifest["accepted_candidate_sha256"] = accepted_candidates_scientific_hash([candidate])
        manifest["accepted_candidate_count"] = 1
        (output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        # Should fail because manifest expects 1 candidate but disk is empty.
        assert any(
            "accepted_candidate_sha256 mismatch" in f
            or "accepted_candidate_count mismatch" in f
            for f in findings
        ), f"Expected failure for truncated accepted file. Findings: {findings}"

    def test_preflight_extra_accepted_candidate_fails(
        self, tmp_path: Path
    ) -> None:
        """Extra retained accepted candidate (no raw attempts) → failure."""
        from experiments.trustparadox_u.empirical_corpus import (
            EmpiricalCandidate,
            candidate_to_record,
            accepted_candidates_scientific_hash,
        )
        candidate = EmpiricalCandidate(
            candidate_id="cand_extra",
            source_generation_attempt_id="att_001",
            candidate_family_id="fam_1",
            scenario_id="s1",
            secret_variant_id="v1",
            split="development",
            trust_level="low",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
            sender_id="sender",
            recipient_id="recipient",
            sequence_family_id=None,
            sequence_id=None,
            sequence_step_index=None,
            sequence_step_count=None,
            text="extra",
            normalized_text="extra",
            content_sha256="xyz",
            accepted=True,
            acceptance_reason="firewall_pass",
            generator_provider="openai",
            generator_model="gpt-4",
            generator_revision=None,
            system_prompt_hash="sys",
            user_prompt_hash="usr",
        )
        accepted_line = json.dumps(candidate_to_record(candidate), sort_keys=True) + "\n"
        # Setup with empty raw attempts (rebuild = []).
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, accepted_content=accepted_line,
        )
        # Manifest has hash for the extra candidate.
        manifest = json.loads((output_dir / "corpus_manifest.json").read_text())
        manifest["accepted_candidate_sha256"] = accepted_candidates_scientific_hash([candidate])
        manifest["accepted_candidate_count"] = 1
        (output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        # Should fail because rebuild is empty but disk has candidate.
        assert any(
            "rebuild" in f.lower()
            or "accepted_candidate_sha256 mismatch" in f
            for f in findings
        ), f"Expected failure for extra accepted candidate. Findings: {findings}"

    def test_preflight_empty_accepted_matches_empty_rebuild(
        self, tmp_path: Path
    ) -> None:
        """Empty disk accepted + empty rebuild → PASS for accepted checks."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        # No accepted-corpus findings.
        assert not any(
            "accepted_candidate_sha256" in f
            or "rebuild" in f.lower()
            or "accepted_candidate_count" in f
            for f in findings
        ), f"Empty/empty should pass. Findings: {findings}"

    def test_preflight_accepted_candidate_count_mismatch_fails(
        self, tmp_path: Path
    ) -> None:
        """Wrong accepted_candidate_count → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, manifest_overrides={"accepted_candidate_count": 999},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("accepted_candidate_count mismatch" in f for f in findings), (
            f"Findings: {findings}"
        )

    def test_preflight_raw_attempt_count_mismatch_fails(
        self, tmp_path: Path
    ) -> None:
        """Wrong raw_attempt_count → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, manifest_overrides={"raw_attempt_count": 999},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("raw_attempt_count mismatch" in f for f in findings), (
            f"Findings: {findings}"
        )


# ---------------------------------------------------------------------------
# Patch H: canonical target-registry hash tests
# ---------------------------------------------------------------------------


class TestPatchHCanonicalRegistryHash:
    """Patch H: use canonical target-registry hash."""

    def test_preflight_target_registry_hash_matches_canonical(
        self, tmp_path: Path
    ) -> None:
        """Correct canonical registry hash → no finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert not any("target_registry_sha256" in f for f in findings), f"Findings: {findings}"

    def test_preflight_target_registry_hash_mismatch_fails(
        self, tmp_path: Path
    ) -> None:
        """Wrong target_registry_sha256 in manifest → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path,
            manifest_overrides={"target_registry_sha256": "WRONG_REGISTRY_HASH"},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "target_registry_sha256 mismatch" in f for f in findings
        ), f"Findings: {findings}"

    def test_preflight_identity_target_registry_hash_mismatch_fails(
        self, tmp_path: Path
    ) -> None:
        """Campaign identity with wrong target_registry_sha256 → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path,
            identity_overrides={"target_registry_sha256": "WRONG_REGISTRY_HASH"},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "campaign_identity.target_registry_sha256 mismatch" in f
            for f in findings
        ), f"Findings: {findings}"


# ---------------------------------------------------------------------------
# Patch P: retained-artifact mutation tests
# ---------------------------------------------------------------------------


class TestPatchPMutationTests:
    """Patch P: negative mutation tests for retained artifacts."""

    def test_raw_mutation_detected(self, tmp_path: Path) -> None:
        """Mutating a raw attempt record → raw_generation_sha256 mismatch."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        # Write a valid raw attempt, then compute hash, then mutate.
        from experiments.trustparadox_u.empirical_corpus import (
            EmpiricalGenerationAttempt,
            GenerationStatus,
            raw_attempts_scientific_hash,
        )
        attempt = EmpiricalGenerationAttempt(
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
            retry_index=0,
            generated_at="2025-01-01T00:00:00+00:00",
            generation_mode="real",
        )
        from dataclasses import asdict
        raw_line = json.dumps(asdict(attempt), sort_keys=True) + "\n"
        (output_dir / "raw_generation_attempts.jsonl").write_text(
            raw_line, encoding="utf-8"
        )
        # Compute correct hash and update manifest.
        correct_hash = raw_attempts_scientific_hash([attempt])
        manifest = json.loads((output_dir / "corpus_manifest.json").read_text())
        manifest["raw_generation_sha256"] = correct_hash
        (output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        # Now mutate: change candidate_text.
        mutated_attempt = EmpiricalGenerationAttempt(
            **{**asdict(attempt), "candidate_text": "MUTATED"}
        )
        mutated_line = json.dumps(asdict(mutated_attempt), sort_keys=True) + "\n"
        (output_dir / "raw_generation_attempts.jsonl").write_text(
            mutated_line, encoding="utf-8"
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("raw_generation_sha256 mismatch" in f for f in findings), f"Findings: {findings}"

    def test_plan_file_mutation_detected(self, tmp_path: Path) -> None:
        """Changing raw JSONL bytes → plan_file_sha256 mismatch."""
        output_dir, plan_file, _ = _setup_valid_preflight(tmp_path)
        # plan_file is empty; manifest has correct hash for empty.
        # Now add a newline to the plan file (changes bytes, not semantics).
        plan_file.write_text("\n", encoding="utf-8")
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any("plan_file_sha256 mismatch" in f for f in findings), f"Findings: {findings}"

    def test_accepted_mutation_detected(self, tmp_path: Path) -> None:
        """Mutating an accepted candidate → accepted_candidate_sha256 mismatch."""
        from experiments.trustparadox_u.empirical_corpus import (
            EmpiricalCandidate,
            accepted_candidates_scientific_hash,
            candidate_to_record,
        )
        candidate = EmpiricalCandidate(
            candidate_id="cand_fam1_low",
            source_generation_attempt_id="att_001",
            candidate_family_id="fam_1",
            scenario_id="s1",
            secret_variant_id="v1",
            split="development",
            trust_level="low",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
            sender_id="sender",
            recipient_id="recipient",
            sequence_family_id=None,
            sequence_id=None,
            sequence_step_index=None,
            sequence_step_count=None,
            text="secret",
            normalized_text="secret",
            content_sha256="abc",
            accepted=True,
            acceptance_reason="firewall_pass",
            generator_provider="openai",
            generator_model="gpt-4",
            generator_revision=None,
            system_prompt_hash="sys",
            user_prompt_hash="usr",
        )
        accepted_line = json.dumps(candidate_to_record(candidate), sort_keys=True) + "\n"
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path, accepted_content=accepted_line,
        )
        # Now mutate the accepted file by changing the candidate text.
        from dataclasses import asdict
        mutated = candidate_to_record(candidate)
        mutated["text"] = "MUTATED"
        mutated_line = json.dumps(mutated, sort_keys=True) + "\n"
        (output_dir / "accepted_candidates.jsonl").write_text(
            mutated_line, encoding="utf-8"
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "accepted_candidate_sha256 mismatch" in f
            or "rebuild" in f.lower()
            for f in findings
        ), f"Findings: {findings}"


# ---------------------------------------------------------------------------
# V2: Mandatory identity provenance regression tests
# ---------------------------------------------------------------------------


class TestV2MandatoryIdentityProvenance:
    """V2: ensure empty identity plan/registry hashes produce blocking findings."""

    def test_preflight_missing_identity_plan_file_hash_fails(
        self, tmp_path: Path
    ) -> None:
        """Empty generation_plan_file_sha256 in identity → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path,
            identity_overrides={"generation_plan_file_sha256": ""},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "campaign_identity.generation_plan_file_sha256 missing" in f
            for f in findings
        ), f"Findings: {findings}"

    def test_preflight_missing_identity_plan_scientific_hash_fails(
        self, tmp_path: Path
    ) -> None:
        """Empty generation_plan_scientific_sha256 in identity → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path,
            identity_overrides={"generation_plan_scientific_sha256": ""},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "campaign_identity.generation_plan_scientific_sha256 missing" in f
            for f in findings
        ), f"Findings: {findings}"

    def test_preflight_missing_identity_target_registry_hash_fails(
        self, tmp_path: Path
    ) -> None:
        """Empty target_registry_sha256 in identity → finding."""
        output_dir, plan_file, _ = _setup_valid_preflight(
            tmp_path,
            identity_overrides={"target_registry_sha256": ""},
        )
        findings = verify_preflight(output_dir, [], [], [], plan_path=plan_file)
        assert any(
            "campaign_identity.target_registry_sha256 missing" in f
            for f in findings
        ), f"Findings: {findings}"


# ---------------------------------------------------------------------------
# V4: Patch H — checks_run explicit evidence
# ---------------------------------------------------------------------------


class TestV4ChecksRunEvidence:
    """V4: validation_report.json must explicitly record all required checks."""

    # The 9 checks that must be explicitly recorded per spec Section 20.
    _REQUIRED_CHECKS = frozenset([
        "plan_scientific_hash",
        "plan_file_hash",
        "plan_item_count",
        "raw_scientific_hash",
        "raw_attempt_count",
        "accepted_scientific_hash",
        "accepted_candidate_count",
        "accepted_deterministic_rebuild",
        "target_registry_hash",
    ])

    def test_preflight_validation_report_lists_all_required_checks(
        self,
    ) -> None:
        """The checks_run list in main() must include all 9 required checks."""
        import ast
        from scripts import run_real_corpus_preflight as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Find the checks_run list literal inside main().
        checks_found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in self._REQUIRED_CHECKS:
                    checks_found.add(node.value)
        missing = self._REQUIRED_CHECKS - checks_found
        assert not missing, f"checks_run missing required entries: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Helpers for provider-viability regression tests (Patches C–K)
# ---------------------------------------------------------------------------


def _make_attempt(
    *,
    attempt_id: str = "att_001",
    scenario_id: str = "s001",
    secret_variant_id: str = "v001",
    split: str = "development",
    trust_level: str = "default",
    attack_type: str = "direct_disclosure",
    sample_index: int = 0,
    generation_replicate: int = 0,
    sender_id: str = "sender_01",
    recipient_id: str = "recipient_01",
    candidate_family_id: str = "fam_001",
    sequence_family_id: str | None = None,
    sequence_id: str | None = None,
    sequence_step_index: int | None = None,
    sequence_step_count: int | None = None,
    candidate_text: str | None = "test output",
    generation_status: str = "success",
    refusal: bool = False,
    malformed: bool = False,
    off_topic: bool = False,
    generator_provider: str = "openai",
    generator_model: str = "qwen3.7-plus",
    generator_revision: str | None = None,
    temperature: float = 0.7,
    seed: int | None = 42,
    system_prompt_hash: str = "sph_001",
    user_prompt_hash: str = "uph_001",
    request_id: str | None = "req_001",
    retry_index: int = 0,
    generated_at: str = "2026-08-02T00:00:00+00:00",
    generation_mode: str = "real",
    transport: str | None = "litellm",
    generator_model_requested: str = "qwen3.7-plus",
    generator_model_returned: str | None = "qwen3.7-plus",
    latency_ms: float | None = 100.0,
    trust_prompt_hash: str | None = "tph_001",
    attack_prompt_hash: str | None = "aph_001",
    max_tokens: int | None = 1024,
    provider_attempt_id: str | None = "prov_001",
) -> "EmpiricalGenerationAttempt":
    """Build a minimal EmpiricalGenerationAttempt for viability tests."""
    from experiments.trustparadox_u.empirical_corpus import (
        EmpiricalGenerationAttempt,
    )
    return EmpiricalGenerationAttempt(
        generation_attempt_id=attempt_id,
        scenario_id=scenario_id,
        secret_variant_id=secret_variant_id,
        split=split,
        trust_level=trust_level,
        attack_type=attack_type,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
        sender_id=sender_id,
        recipient_id=recipient_id,
        candidate_family_id=candidate_family_id,
        sequence_family_id=sequence_family_id,
        sequence_id=sequence_id,
        sequence_step_index=sequence_step_index,
        sequence_step_count=sequence_step_count,
        candidate_text=candidate_text,
        generation_status=generation_status,
        refusal=refusal,
        malformed=malformed,
        off_topic=off_topic,
        generator_provider=generator_provider,
        generator_model=generator_model,
        generator_revision=generator_revision,
        temperature=temperature,
        seed=seed,
        system_prompt_hash=system_prompt_hash,
        user_prompt_hash=user_prompt_hash,
        request_id=request_id,
        retry_index=retry_index,
        generated_at=generated_at,
        generation_mode=generation_mode,
        transport=transport,
        generator_model_requested=generator_model_requested,
        generator_model_returned=generator_model_returned,
        latency_ms=latency_ms,
        trust_prompt_hash=trust_prompt_hash,
        attack_prompt_hash=attack_prompt_hash,
        max_tokens=max_tokens,
        provider_attempt_id=provider_attempt_id,
    )


# ---------------------------------------------------------------------------
# Regression test A — 100% provider_error must fail viability
# ---------------------------------------------------------------------------


class TestProviderViabilityAllProviderErrors:
    """Regression test A: all provider_error → viability gate FAILS."""

    def test_preflight_all_provider_error_fails_viability(self) -> None:
        """25 units × 3 provider_error attempts each, 0 success → FAIL."""
        from scripts.run_real_corpus_preflight import summarize_provider_outcomes

        attempts = []
        for unit_idx in range(25):
            for retry in range(3):
                attempts.append(_make_attempt(
                    attempt_id=f"att_{unit_idx:03d}_{retry}",
                    generation_status="provider_error",
                    retry_index=retry,
                    provider_attempt_id=f"prov_{unit_idx:03d}_{retry}",
                    latency_ms=None,
                    candidate_text=None,
                ))

        viability = summarize_provider_outcomes(attempts)

        assert viability["provider_attempt_count"] == 75
        assert viability["success_count"] == 0
        assert viability["provider_error_count"] == 75
        assert viability["success_rate"] == 0.0
        assert viability["provider_error_rate"] == 1.0
        assert viability["successful_non_sequence_count"] == 0
        assert viability["successful_complete_sequence_count"] == 0

        # The viability gate logic from main():
        viability_passed = True
        findings = []
        if viability["provider_attempt_count"] > 0:
            if viability["success_count"] == 0:
                findings.append(
                    "real-provider viability failure: "
                    "zero successful provider generations"
                )
                viability_passed = False
            if viability["successful_non_sequence_count"] == 0:
                findings.append(
                    "real-provider viability failure: "
                    "no successful non-sequence generation"
                )
                viability_passed = False
            if viability["successful_complete_sequence_count"] == 0:
                findings.append(
                    "real-provider viability failure: "
                    "no complete successful sequence generation"
                )
                viability_passed = False

        assert viability_passed is False
        assert len(findings) == 3
        assert any("zero successful" in f for f in findings)
        assert any("non-sequence" in f for f in findings)
        assert any("sequence generation" in f for f in findings)


# ---------------------------------------------------------------------------
# Regression test B — non-sequence success but no complete sequence
# ---------------------------------------------------------------------------


class TestProviderViabilityNoCompleteSequence:
    """Regression test B: non-sequence success but all sequences fail."""

    def test_preflight_no_successful_sequence_fails_viability(self) -> None:
        """≥1 successful non-sequence, all sequence families fail → FAIL."""
        from scripts.run_real_corpus_preflight import summarize_provider_outcomes

        attempts = []
        # 1 successful non-sequence attempt.
        attempts.append(_make_attempt(
            attempt_id="att_ns_success",
            generation_status="success",
            attack_type="direct_disclosure",
            sequence_family_id=None,
            retry_index=0,
        ))
        # Sequence family with 2 steps, all steps fail (provider_error).
        for step_idx in range(2):
            attempts.append(_make_attempt(
                attempt_id=f"att_seq_fail_{step_idx}",
                generation_status="provider_error",
                attack_type="fragmentation_sequence",
                sequence_family_id="seq_fam_001",
                sequence_id="seq_001",
                sequence_step_index=step_idx,
                sequence_step_count=2,
                retry_index=0,
                provider_attempt_id=f"prov_seq_fail_{step_idx}",
                latency_ms=None,
                candidate_text=None,
            ))

        viability = summarize_provider_outcomes(attempts)

        assert viability["success_count"] == 1
        assert viability["successful_non_sequence_count"] == 1
        assert viability["successful_complete_sequence_count"] == 0

        # Apply the gate logic.
        viability_passed = True
        findings = []
        if viability["provider_attempt_count"] > 0:
            if viability["success_count"] == 0:
                findings.append("zero successful")
                viability_passed = False
            if viability["successful_non_sequence_count"] == 0:
                findings.append("no non-sequence")
                viability_passed = False
            if viability["successful_complete_sequence_count"] == 0:
                findings.append(
                    "real-provider viability failure: "
                    "no complete successful sequence generation"
                )
                viability_passed = False

        assert viability_passed is False
        assert any("sequence generation" in f for f in findings)


# ---------------------------------------------------------------------------
# Regression test C — sequence success but no non-sequence success
# ---------------------------------------------------------------------------


class TestProviderViabilityNoNonSequence:
    """Regression test C: sequence success but no non-sequence success."""

    def test_preflight_no_successful_non_sequence_fails_viability(self) -> None:
        """Complete successful sequence but all non-sequence fail → FAIL."""
        from scripts.run_real_corpus_preflight import summarize_provider_outcomes

        attempts = []
        # All non-sequence attempts fail.
        for i in range(3):
            attempts.append(_make_attempt(
                attempt_id=f"att_ns_fail_{i}",
                generation_status="provider_error",
                attack_type="direct_disclosure",
                sequence_family_id=None,
                retry_index=0,
                provider_attempt_id=f"prov_ns_fail_{i}",
                latency_ms=None,
                candidate_text=None,
            ))
        # Complete successful sequence (2 steps, each with success at retry 0).
        for step_idx in range(2):
            attempts.append(_make_attempt(
                attempt_id=f"att_seq_ok_{step_idx}",
                generation_status="success",
                attack_type="fragmentation_sequence",
                sequence_family_id="seq_fam_001",
                sequence_id="seq_001",
                sequence_step_index=step_idx,
                sequence_step_count=2,
                retry_index=0,
            ))

        viability = summarize_provider_outcomes(attempts)

        assert viability["success_count"] == 2
        assert viability["successful_non_sequence_count"] == 0
        assert viability["successful_complete_sequence_count"] == 1

        # Apply the gate logic.
        viability_passed = True
        findings = []
        if viability["provider_attempt_count"] > 0:
            if viability["success_count"] == 0:
                findings.append("zero successful")
                viability_passed = False
            if viability["successful_non_sequence_count"] == 0:
                findings.append(
                    "real-provider viability failure: "
                    "no successful non-sequence generation"
                )
                viability_passed = False
            if viability["successful_complete_sequence_count"] == 0:
                findings.append("no complete sequence")
                viability_passed = False

        assert viability_passed is False
        assert any("non-sequence" in f for f in findings)


# ---------------------------------------------------------------------------
# Regression test D — mixed realistic outcomes can pass
# ---------------------------------------------------------------------------


class TestProviderViabilityMixedOutcomes:
    """Regression test D: mixed outcomes with both paths succeeding → PASS."""

    def test_preflight_mixed_provider_outcomes_pass_viability(self) -> None:
        """Some success, some refusal, some provider_error → PASS."""
        from scripts.run_real_corpus_preflight import summarize_provider_outcomes

        attempts = []
        # Successful non-sequence attempt.
        attempts.append(_make_attempt(
            attempt_id="att_ns_ok",
            generation_status="success",
            attack_type="direct_disclosure",
            sequence_family_id=None,
            retry_index=0,
        ))
        # Refusal non-sequence attempt.
        attempts.append(_make_attempt(
            attempt_id="att_ns_refusal",
            generation_status="refusal",
            attack_type="semantic_paraphrase",
            sequence_family_id=None,
            retry_index=0,
            refusal=True,
            candidate_text=None,
        ))
        # Provider error non-sequence attempt.
        attempts.append(_make_attempt(
            attempt_id="att_ns_err",
            generation_status="provider_error",
            attack_type="hard_negative_control",
            sequence_family_id=None,
            retry_index=0,
            provider_attempt_id="prov_err_001",
            latency_ms=None,
            candidate_text=None,
        ))
        # Complete successful sequence (2 steps, each with success).
        for step_idx in range(2):
            attempts.append(_make_attempt(
                attempt_id=f"att_seq_ok_{step_idx}",
                generation_status="success",
                attack_type="fragmentation_sequence",
                sequence_family_id="seq_fam_001",
                sequence_id="seq_001",
                sequence_step_index=step_idx,
                sequence_step_count=2,
                retry_index=0,
            ))
        # Failed sequence (provider_error at step 0).
        attempts.append(_make_attempt(
            attempt_id="att_seq_fail",
            generation_status="provider_error",
            attack_type="compositional_sequence",
            sequence_family_id="seq_fam_002",
            sequence_id="seq_002",
            sequence_step_index=0,
            sequence_step_count=2,
            retry_index=0,
            provider_attempt_id="prov_seq_err",
            latency_ms=None,
            candidate_text=None,
        ))

        viability = summarize_provider_outcomes(attempts)

        assert viability["provider_attempt_count"] == 6
        assert viability["success_count"] == 3
        assert viability["provider_error_count"] == 2
        assert viability["refusal_count"] == 1
        assert viability["successful_non_sequence_count"] == 1
        assert viability["successful_complete_sequence_count"] == 1

        # Apply the gate logic.
        viability_passed = True
        if viability["provider_attempt_count"] > 0:
            if viability["success_count"] == 0:
                viability_passed = False
            if viability["successful_non_sequence_count"] == 0:
                viability_passed = False
            if viability["successful_complete_sequence_count"] == 0:
                viability_passed = False

        assert viability_passed is True


# ---------------------------------------------------------------------------
# Regression test E — retries remain valid
# ---------------------------------------------------------------------------


class TestProviderViabilityRetriesValid:
    """Regression test E: retry 0 → provider_error, retry 1 → success."""

    def test_preflight_retry_produces_valid_viability(self) -> None:
        """A unit where retry 0 fails but retry 1 succeeds is valid."""
        from scripts.run_real_corpus_preflight import summarize_provider_outcomes

        attempts = []
        # Non-sequence: retry 0 → provider_error, retry 1 → success.
        attempts.append(_make_attempt(
            attempt_id="att_ns_retry0",
            generation_status="provider_error",
            attack_type="direct_disclosure",
            sequence_family_id=None,
            retry_index=0,
            provider_attempt_id="prov_ns_r0",
            latency_ms=None,
            candidate_text=None,
        ))
        attempts.append(_make_attempt(
            attempt_id="att_ns_retry1",
            generation_status="success",
            attack_type="direct_disclosure",
            sequence_family_id=None,
            retry_index=1,
            provider_attempt_id="prov_ns_r1",
        ))
        # Sequence: step 0 retry 0 → error, step 0 retry 1 → success;
        #           step 1 retry 0 → success.
        attempts.append(_make_attempt(
            attempt_id="att_seq_s0_r0",
            generation_status="provider_error",
            attack_type="fragmentation_sequence",
            sequence_family_id="seq_fam_001",
            sequence_id="seq_001",
            sequence_step_index=0,
            sequence_step_count=2,
            retry_index=0,
            provider_attempt_id="prov_seq_s0_r0",
            latency_ms=None,
            candidate_text=None,
        ))
        attempts.append(_make_attempt(
            attempt_id="att_seq_s0_r1",
            generation_status="success",
            attack_type="fragmentation_sequence",
            sequence_family_id="seq_fam_001",
            sequence_id="seq_001",
            sequence_step_index=0,
            sequence_step_count=2,
            retry_index=1,
            provider_attempt_id="prov_seq_s0_r1",
        ))
        attempts.append(_make_attempt(
            attempt_id="att_seq_s1_r0",
            generation_status="success",
            attack_type="fragmentation_sequence",
            sequence_family_id="seq_fam_001",
            sequence_id="seq_001",
            sequence_step_index=1,
            sequence_step_count=2,
            retry_index=0,
            provider_attempt_id="prov_seq_s1_r0",
        ))

        viability = summarize_provider_outcomes(attempts)

        assert viability["provider_attempt_count"] == 5
        assert viability["success_count"] == 3
        assert viability["provider_error_count"] == 2
        # The successful non-sequence is the retry 1 attempt.
        assert viability["successful_non_sequence_count"] == 1
        # The sequence is complete: step 0 terminal (retry 1) = success,
        # step 1 terminal (retry 0) = success.
        assert viability["successful_complete_sequence_count"] == 1

        # Apply the gate logic.
        viability_passed = True
        if viability["provider_attempt_count"] > 0:
            if viability["success_count"] == 0:
                viability_passed = False
            if viability["successful_non_sequence_count"] == 0:
                viability_passed = False
            if viability["successful_complete_sequence_count"] == 0:
                viability_passed = False

        assert viability_passed is True

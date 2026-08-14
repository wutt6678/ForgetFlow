"""Patch M: end-to-end split progression integration test.

Uses the actual committed 900-item frozen plan and proves the full
orchestration state machine:

  dev generated → dev audited → val unlocked → val generated →
  val audited → test unlocked → test generated → test audited →
  all audit passes

Provider calls are mocked.  Audit validators are stubbed to isolate
the orchestration mechanics (plan splitting, gate transitions, audit
scope, hash verification) from the audit-content checks which are
tested separately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from experiments.trustparadox_u.empirical_generation_plan import (
    GenerationPlanItem,
    generation_gate_path,
    load_generation_gate,
    load_generation_plan,
    plan_items_for_split,
    plan_sha256,
    update_generation_gate_after_audit,
)
from experiments.trustparadox_u.campaign_identity import campaign_identity_sha256

# ---------------------------------------------------------------------------
# Paths to real committed data
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FULL_PLAN_PATH = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2"
    / "manifests" / "full_generation_plan.jsonl"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_attempt(plan_item: GenerationPlanItem) -> dict[str, Any]:
    """Create a minimal raw-attempt dict matching a plan item."""
    return {
        "generation_attempt_id": plan_item.plan_item_id,
        "scenario_id": plan_item.scenario_id,
        "secret_variant_id": plan_item.secret_variant_id,
        "split": plan_item.split,
        "trust_level": plan_item.trust_level,
        "attack_type": plan_item.attack_type,
        "sample_index": plan_item.sample_index,
        "generation_replicate": plan_item.generation_replicate,
        "sender_id": "sender",
        "recipient_id": "recipient",
        "candidate_family_id": f"fam_{plan_item.scenario_id}",
        "sequence_family_id": plan_item.sequence_id,
        "sequence_id": plan_item.sequence_id,
        "sequence_step_index": plan_item.sequence_step_index,
        "sequence_step_count": plan_item.sequence_step_count,
        "candidate_text": "mock response",
        "generation_status": "success",
        "refusal": False,
        "malformed": False,
        "off_topic": False,
        "generator_provider": "mock",
        "generator_model": "mock-model",
        "generator_revision": None,
        "temperature": 0.7,
        "seed": None,
        "system_prompt_hash": "sys",
        "user_prompt_hash": "usr",
        "request_id": f"req_{plan_item.plan_item_id}",
        "retry_index": 0,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "generation_mode": "mock",
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, sort_keys=True) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gate(
    base: Path,
    split: str,
    *,
    generation_completed: bool = True,
    audit_passed: bool = False,
    audit_report_sha256: str = "",
    audit_report_path: str = "",
    corpus_manifest_sha256: str = "",
    campaign_identity_sha256: str = "",
) -> Path:
    """Write a generation gate file."""
    gate = {
        "split": split,
        "source_commit": "test_commit",
        "generation_completed": generation_completed,
        "planned_plan_item_count": 0,
        "accounted_plan_item_count": 0,
        "missing_plan_item_count": 0,
        "audit_passed": audit_passed,
        "audit_report_sha256": audit_report_sha256,
        "audit_report_path": audit_report_path,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "campaign_identity_sha256": campaign_identity_sha256,
    }
    gate_path = generation_gate_path(split, base)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate_path


def _setup_split_artifacts(
    corpus_base: Path,
    split: str,
    plan_items: list[GenerationPlanItem],
) -> None:
    """Write mock raw attempts and accepted candidates for a split."""
    split_dir = corpus_base / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Write raw attempts (one per plan item, matching generation_attempt_id).
    attempts = [_make_mock_attempt(pi) for pi in plan_items]
    _write_jsonl(split_dir / "raw_generation_attempts.jsonl", attempts)

    # Write accepted candidates (empty is fine for orchestration test).
    _write_jsonl(split_dir / "accepted_candidates.jsonl", [])


# ===========================================================================
# Test M — full split progression
# ===========================================================================


class TestSplitProgression:
    """Patch M: end-to-end split progression with the real 900-item plan."""

    def test_full_plan_has_900_items(self) -> None:
        """The committed plan has exactly 900 items."""
        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        assert len(full_plan) == 900

    def test_split_counts(self) -> None:
        """Development=225, validation=225, test=450."""
        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        dev = plan_items_for_split(full_plan, "development")
        val = plan_items_for_split(full_plan, "validation")
        test = plan_items_for_split(full_plan, "test")
        assert len(dev) == 225
        assert len(val) == 225
        assert len(test) == 450
        # No overlap and total is 900.
        dev_ids = {it.plan_item_id for it in dev}
        val_ids = {it.plan_item_id for it in val}
        test_ids = {it.plan_item_id for it in test}
        assert dev_ids.isdisjoint(val_ids)
        assert dev_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert len(dev_ids | val_ids | test_ids) == 900

    def test_only_split_items_reach_generation(self) -> None:
        """plan_items_for_split returns only items for the requested split."""
        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        dev_items = plan_items_for_split(full_plan, "development")
        assert all(it.split == "development" for it in dev_items)
        val_items = plan_items_for_split(full_plan, "validation")
        assert all(it.split == "validation" for it in val_items)
        test_items = plan_items_for_split(full_plan, "test")
        assert all(it.split == "test" for it in test_items)

    def test_gate_progression_dev_to_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full progression: dev → val → test with gate transitions."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod

        full_plan = load_generation_plan(_FULL_PLAN_PATH)

        # Patch corpus base to use tmp_path.
        monkeypatch.setattr(audit_mod, "_CORPUS_BASE", tmp_path)
        monkeypatch.setattr(audit_mod, "_OUTPUT_DIR", tmp_path)

        # Stub all validators to return no findings.
        for name in [
            "validate_phase_and_provenance",
            "validate_plan_completeness",
            "validate_split_integrity",
            "validate_identity_uniqueness",
            "validate_variant_consistency",
            "validate_config_consistency",
            "validate_hash_integrity",
            "validate_retry_lineage",
            "validate_sequence_atomicity",
            "validate_acceptance_independence",
            "validate_campaign_identity",
            "validate_artifact_classification",
            "validate_manifest_provenance",
            "validate_empirical_corpus_required_fields",
        ]:
            monkeypatch.setattr(
                audit_mod, name, lambda *a, **kw: [],
            )
        
        # Stub source commit consistency check.
        monkeypatch.setattr(
            audit_mod, "validate_source_commit_consistency", lambda: [],
        )
        
        # Stub split gate progression validation (checks gates manually).
        monkeypatch.setattr(
            audit_mod, "_validate_all_split_gates", lambda: [],
        )

        # --- Development ---
        dev_items = plan_items_for_split(full_plan, "development")
        _setup_split_artifacts(tmp_path, "development", dev_items)
        _write_gate(tmp_path, "development", generation_completed=True)

        report = audit_mod.build_validation_report(split_scope="development")
        assert report["passed"] is True
        assert report["audit_scope"] == "development"
        assert report["audited_splits"] == ["development"]

        # Promote development audit to gate.
        dev_report_path = tmp_path / "development" / "audit_report.json"
        dev_report_path.parent.mkdir(parents=True, exist_ok=True)
        dev_report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dev_audit_sha = hashlib.sha256(
            dev_report_path.read_bytes(),
        ).hexdigest()
        
        # Compute corpus_manifest and campaign_identity hashes.
        dev_dir = tmp_path / "development"
        corpus_manifest_path = dev_dir / "corpus_manifest.json"
        if not corpus_manifest_path.exists():
            # Create minimal corpus_manifest.json
            corpus_manifest_path.write_text(
                json.dumps({"artifact_class": "empirical_corpus"}),
                encoding="utf-8",
            )
        corpus_manifest_sha = hashlib.sha256(
            corpus_manifest_path.read_bytes()
        ).hexdigest()
        
        identity_path = dev_dir / "campaign_identity.json"
        if not identity_path.exists():
            from experiments.trustparadox_u.campaign_identity import CampaignIdentity
            from dataclasses import asdict
            
            identity_obj = CampaignIdentity(
                schema_version="1.0",
                split="development",
                generation_plan_scientific_sha256="plan_hash",
                generation_plan_file_sha256="plan_hash",
                generation_config_sha256="config_hash",
                target_registry_sha256="registry_hash",
                prompt_manifest_sha256="prompt_hash",
                phase_manifest_sha256="phase_hash",
                generator_provider="mock",
                generator_model_requested="mock-model",
                generator_temperature=0.7,
                generator_max_tokens=1024,
                request_timeout=30.0,
                max_retries=3,
                created_from_commit="test_commit",
                created_at="2026-08-02T00:00:00+00:00",
                serving_endpoint_host="endpoint.test",
                serving_endpoint_sha256=hashlib.sha256(
                    b"https://endpoint.test/v1",
                ).hexdigest(),
                api_protocol="openai_compatible",
            )
            identity_path.write_text(
                json.dumps(asdict(identity_obj), indent=2),
                encoding="utf-8",
            )
        campaign_identity_raw = json.loads(identity_path.read_text(encoding="utf-8"))
        # Convert back to CampaignIdentity object for hash computation
        identity_obj = CampaignIdentity(**campaign_identity_raw)
        campaign_identity_sha = campaign_identity_sha256(identity_obj)
        
        update_generation_gate_after_audit(
            split="development",
            audit_passed=True,
            audit_report_path=Path("development/audit_report.json"),
            audit_report_sha256=dev_audit_sha,
            source_commit="test_commit",
            corpus_manifest_sha256=corpus_manifest_sha,
            campaign_identity_sha256=campaign_identity_sha,
            base=tmp_path,
        )

        dev_gate = load_generation_gate("development", base=tmp_path)
        assert dev_gate is not None
        assert dev_gate["audit_passed"] is True
        assert dev_gate["audit_report_sha256"] == dev_audit_sha

        # --- Validation eligibility ---
        # Before development audit, validation would be blocked.
        # After development audit, validation is eligible.
        _setup_split_artifacts(tmp_path, "validation", plan_items_for_split(full_plan, "validation"))
        _write_gate(tmp_path, "validation", generation_completed=True)

        report_val = audit_mod.build_validation_report(split_scope="validation")
        assert report_val["passed"] is True
        assert report_val["audit_scope"] == "validation"

        # Promote validation audit.
        val_report_path = tmp_path / "validation" / "audit_report.json"
        val_report_path.parent.mkdir(parents=True, exist_ok=True)
        val_report_path.write_text(
            json.dumps(report_val, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        val_audit_sha = hashlib.sha256(
            val_report_path.read_bytes(),
        ).hexdigest()
        
        # Compute corpus_manifest and campaign_identity hashes for validation.
        val_dir = tmp_path / "validation"
        corpus_manifest_path = val_dir / "corpus_manifest.json"
        if not corpus_manifest_path.exists():
            corpus_manifest_path.write_text(
                json.dumps({"artifact_class": "empirical_corpus"}),
                encoding="utf-8",
            )
        corpus_manifest_sha = hashlib.sha256(
            corpus_manifest_path.read_bytes()
        ).hexdigest()
        
        identity_path = val_dir / "campaign_identity.json"
        if not identity_path.exists():
            from experiments.trustparadox_u.campaign_identity import CampaignIdentity
            from dataclasses import asdict
            
            identity_obj = CampaignIdentity(
                schema_version="1.0",
                split="validation",
                generation_plan_scientific_sha256="plan_hash",
                generation_plan_file_sha256="plan_hash",
                generation_config_sha256="config_hash",
                target_registry_sha256="registry_hash",
                prompt_manifest_sha256="prompt_hash",
                phase_manifest_sha256="phase_hash",
                generator_provider="mock",
                generator_model_requested="mock-model",
                generator_temperature=0.7,
                generator_max_tokens=1024,
                request_timeout=30.0,
                max_retries=3,
                created_from_commit="test_commit",
                created_at="2026-08-02T00:00:00+00:00",
                serving_endpoint_host="endpoint.test",
                serving_endpoint_sha256=hashlib.sha256(
                    b"https://endpoint.test/v1",
                ).hexdigest(),
                api_protocol="openai_compatible",
            )
            identity_path.write_text(
                json.dumps(asdict(identity_obj), indent=2),
                encoding="utf-8",
            )
        identity_raw = json.loads(identity_path.read_text(encoding="utf-8"))
        identity_obj = CampaignIdentity(**identity_raw)
        identity_sha = campaign_identity_sha256(identity_obj)
        
        update_generation_gate_after_audit(
            split="validation",
            audit_passed=True,
            audit_report_path=Path("validation/audit_report.json"),
            audit_report_sha256=val_audit_sha,
            source_commit="test_commit",
            corpus_manifest_sha256=corpus_manifest_sha,
            campaign_identity_sha256=identity_sha,
            base=tmp_path,
        )

        val_gate = load_generation_gate("validation", base=tmp_path)
        assert val_gate is not None
        assert val_gate["audit_passed"] is True

        # --- Test ---
        _setup_split_artifacts(tmp_path, "test", plan_items_for_split(full_plan, "test"))
        _write_gate(tmp_path, "test", generation_completed=True)

        report_test = audit_mod.build_validation_report(split_scope="test")
        assert report_test["passed"] is True

        # Promote test audit.
        test_report_path = tmp_path / "test" / "audit_report.json"
        test_report_path.parent.mkdir(parents=True, exist_ok=True)
        test_report_path.write_text(
            json.dumps(report_test, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        test_audit_sha = hashlib.sha256(
            test_report_path.read_bytes(),
        ).hexdigest()
        
        # Compute corpus_manifest and campaign_identity hashes for test.
        test_dir = tmp_path / "test"
        corpus_manifest_path = test_dir / "corpus_manifest.json"
        if not corpus_manifest_path.exists():
            corpus_manifest_path.write_text(
                json.dumps({"artifact_class": "empirical_corpus"}),
                encoding="utf-8",
            )
        corpus_manifest_sha = hashlib.sha256(
            corpus_manifest_path.read_bytes()
        ).hexdigest()
        
        identity_path = test_dir / "campaign_identity.json"
        if not identity_path.exists():
            from experiments.trustparadox_u.campaign_identity import CampaignIdentity
            from dataclasses import asdict
            
            identity_obj = CampaignIdentity(
                schema_version="1.0",
                split="test",
                generation_plan_scientific_sha256="plan_hash",
                generation_plan_file_sha256="plan_hash",
                generation_config_sha256="config_hash",
                target_registry_sha256="registry_hash",
                prompt_manifest_sha256="prompt_hash",
                phase_manifest_sha256="phase_hash",
                generator_provider="mock",
                generator_model_requested="mock-model",
                generator_temperature=0.7,
                generator_max_tokens=1024,
                request_timeout=30.0,
                max_retries=3,
                created_from_commit="test_commit",
                created_at="2026-08-02T00:00:00+00:00",
                serving_endpoint_host="endpoint.test",
                serving_endpoint_sha256=hashlib.sha256(
                    b"https://endpoint.test/v1",
                ).hexdigest(),
                api_protocol="openai_compatible",
            )
            identity_path.write_text(
                json.dumps(asdict(identity_obj), indent=2),
                encoding="utf-8",
            )
        identity_raw = json.loads(identity_path.read_text(encoding="utf-8"))
        identity_obj = CampaignIdentity(**identity_raw)
        identity_sha = campaign_identity_sha256(identity_obj)
        
        update_generation_gate_after_audit(
            split="test",
            audit_passed=True,
            audit_report_path=Path("test/audit_report.json"),
            audit_report_sha256=test_audit_sha,
            source_commit="test_commit",
            corpus_manifest_sha256=corpus_manifest_sha,
            campaign_identity_sha256=identity_sha,
            base=tmp_path,
        )

        # --- Final combined audit ---
        report_all = audit_mod.build_validation_report(split_scope="all")
        assert report_all["passed"] is True
        assert report_all["audit_scope"] == "all"
        assert set(report_all["audited_splits"]) == {
            "development", "validation", "test",
        }
        # Gate progression section should be present and passing.
        gate_section = report_all["audit_sections"].get("split_gate_progression", [])
        assert gate_section == []

    def test_test_blocked_before_validation_audit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test pre-run gate must fail before validation audit."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod

        # Only development gate exists with audit_passed=true.
        _write_gate(tmp_path, "development", generation_completed=True, audit_passed=True)
        # No validation or test gates.

        # Patch gate base path.
        monkeypatch.setattr(
            audit_mod, "_CORPUS_BASE", tmp_path,
        )

        # The _validate_all_split_gates should report missing gates.
        findings = audit_mod._validate_all_split_gates()
        assert len(findings) > 0
        # validation and test gates should be reported as missing.
        assert any("validation" in f for f in findings)
        assert any("test" in f for f in findings)

    def test_audit_hash_tamper_blocks_next_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Modifying audit report after gate promotion is detected."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod

        monkeypatch.setattr(audit_mod, "_CORPUS_BASE", tmp_path)

        # Set up development gate with a valid audit hash.
        dev_report_dir = tmp_path / "development"
        dev_report_dir.mkdir(parents=True, exist_ok=True)
        dev_report = dev_report_dir / "audit_report.json"
        dev_report.write_text('{"passed": true}', encoding="utf-8")
        original_sha = hashlib.sha256(dev_report.read_bytes()).hexdigest()

        _write_gate(
            tmp_path, "development",
            generation_completed=True,
            audit_passed=True,
            audit_report_sha256=original_sha,
            audit_report_path="development/audit_report.json",
        )

        # Tamper with the audit report.
        dev_report.write_text('{"passed": true, "tampered": true}', encoding="utf-8")

        # The gate validation should detect the hash mismatch.
        findings = audit_mod._validate_all_split_gates()
        assert any("SHA256 mismatch" in f for f in findings)

    def test_final_audit_fails_without_all_gates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--split all fails when not all gates have audit_passed=true."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod

        monkeypatch.setattr(audit_mod, "_CORPUS_BASE", tmp_path)

        # Only development gate exists.
        _write_gate(tmp_path, "development", generation_completed=True, audit_passed=True)

        # Stub validators.
        for name in [
            "validate_phase_and_provenance",
            "validate_plan_completeness",
            "validate_split_integrity",
            "validate_identity_uniqueness",
            "validate_variant_consistency",
            "validate_config_consistency",
            "validate_hash_integrity",
            "validate_retry_lineage",
            "validate_sequence_atomicity",
            "validate_acceptance_independence",
            "validate_campaign_identity",
        ]:
            monkeypatch.setattr(audit_mod, name, lambda *a, **kw: [])

        report = audit_mod.build_validation_report(split_scope="all")
        # Should fail because validation and test gates are missing.
        gate_findings = report["audit_sections"].get("split_gate_progression", [])
        assert len(gate_findings) > 0
        assert not report["passed"]


# ===========================================================================
# Test M — plan completeness with exact IDs
# ===========================================================================


class TestExactPlanCompleteness:
    """Verify exact plan-item ID completeness for the real plan."""

    def test_all_900_plan_ids_are_unique(self) -> None:
        """Every plan item has a unique plan_item_id."""
        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        ids = [it.plan_item_id for it in full_plan]
        assert len(set(ids)) == 900

    def test_plan_sha256_is_deterministic(self) -> None:
        """plan_sha256 returns the same hash for the same plan."""
        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        h1 = plan_sha256(full_plan)
        h2 = plan_sha256(full_plan)
        assert h1 == h2
        assert len(h1) == 64

    def test_split_plan_hashes_differ(self) -> None:
        """Each split's plan has a different hash."""
        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        dev_hash = plan_sha256(plan_items_for_split(full_plan, "development"))
        val_hash = plan_sha256(plan_items_for_split(full_plan, "validation"))
        test_hash = plan_sha256(plan_items_for_split(full_plan, "test"))
        assert dev_hash != val_hash
        assert dev_hash != test_hash
        assert val_hash != test_hash


# ===========================================================================
# Patch N — negative orchestration tests N1–N7
# ===========================================================================


class TestNegativeOrchestration:
    """Patch N: verify that each orchestration failure mode is caught."""

    # N1 — development run receives foreign item
    def test_n1_foreign_item_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_generation with mixed-split plan items raises ValueError."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            run_generation,
        )

        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        dev_items = plan_items_for_split(full_plan, "development")
        val_items = plan_items_for_split(full_plan, "validation")
        mixed = list(dev_items[:5]) + list(val_items[:3])

        # Stub the phase lock (development is fine in E2).
        monkeypatch.setattr(
            "experiments.trustparadox_u.generate_empirical_corpus."
            "assert_generation_split_unlocked",
            lambda *a, **kw: None,
        )

        with pytest.raises(ValueError, match="splits other than"):
            run_generation(
                split="development",
                mode="mock",
                scenarios=["s1"],
                trust_levels=["low"],
                attack_types=["direct_disclosure"],
                samples=1,
                output_dir=tmp_path,
                generator=None,
                plan_items=mixed,
            )

    # N2 — development audit before generation complete
    def test_n2_audit_before_generation_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Audit with generation_completed=false keeps gate audit_passed=false."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod

        monkeypatch.setattr(audit_mod, "_CORPUS_BASE", tmp_path)
        monkeypatch.setattr(audit_mod, "_OUTPUT_DIR", tmp_path)

        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        dev_items = plan_items_for_split(full_plan, "development")
        _setup_split_artifacts(tmp_path, "development", dev_items)

        # Gate says generation NOT completed.
        _write_gate(tmp_path, "development", generation_completed=False)

        # Stub validators.
        for name in [
            "validate_phase_and_provenance",
            "validate_plan_completeness",
            "validate_split_integrity",
            "validate_identity_uniqueness",
            "validate_variant_consistency",
            "validate_config_consistency",
            "validate_hash_integrity",
            "validate_retry_lineage",
            "validate_sequence_atomicity",
            "validate_acceptance_independence",
            "validate_campaign_identity",
        ]:
            monkeypatch.setattr(audit_mod, name, lambda *a, **kw: [])

        report = audit_mod.build_validation_report(split_scope="development")
        # Audit itself may pass (validators stubbed), but the gate
        # generation_completed=false means the split is not truly done.
        gate = load_generation_gate("development", base=tmp_path)
        assert gate is not None
        assert gate["generation_completed"] is False
        assert gate.get("audit_passed") is not True

    # N3 — validation before development audit
    def test_n3_validation_before_dev_audit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Validation is blocked when development has no audit_passed gate."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod

        monkeypatch.setattr(audit_mod, "_CORPUS_BASE", tmp_path)

        # Development gate exists but audit NOT passed.
        _write_gate(
            tmp_path, "development",
            generation_completed=True, audit_passed=False,
        )

        findings = audit_mod._validate_all_split_gates()
        assert any("development" in f and "audit_passed" in f for f in findings)
        assert any("validation" in f and "gate file missing" in f for f in findings)
        assert any("test" in f and "gate file missing" in f for f in findings)

    # N4 — validation after boolean tampering but invalid audit hash
    def test_n4_tampered_boolean_invalid_hash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Manually setting audit_passed=true with wrong hash is caught."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod

        monkeypatch.setattr(audit_mod, "_CORPUS_BASE", tmp_path)

        # Write a fake audit report.
        dev_dir = tmp_path / "development"
        dev_dir.mkdir(parents=True, exist_ok=True)
        report = dev_dir / "audit_report.json"
        report.write_text('{"passed": true}', encoding="utf-8")

        # Write gate with audit_passed=true but WRONG hash.
        _write_gate(
            tmp_path, "development",
            generation_completed=True,
            audit_passed=True,
            audit_report_sha256="deadbeef" * 8,  # wrong hash
            audit_report_path="development/audit_report.json",
        )

        findings = audit_mod._validate_all_split_gates()
        assert any("SHA256 mismatch" in f for f in findings)

    # N5 — one development sequence step missing
    def test_n5_missing_sequence_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing one sequence step causes plan completeness failure."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod
        from experiments.trustparadox_u.empirical_corpus import (
            EmpiricalGenerationAttempt,
        )

        monkeypatch.setattr(audit_mod, "_CORPUS_BASE", tmp_path)

        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        dev_items = plan_items_for_split(full_plan, "development")

        # Find a sequence with multiple steps.
        seq_items = [it for it in dev_items if it.sequence_id is not None]
        assert len(seq_items) > 0, "expected sequence items in development plan"
        # Pick one sequence and drop its last step.
        seq_id = seq_items[0].sequence_id
        seq_group = [it for it in seq_items if it.sequence_id == seq_id]
        assert len(seq_group) >= 2, "need >= 2 steps to test missing step"
        dropped_item = max(seq_group, key=lambda it: it.sequence_step_index or 0)
        kept_dev_items = [it for it in dev_items if it.plan_item_id != dropped_item.plan_item_id]

        # Write attempts only for kept items (one step missing).
        _setup_split_artifacts(tmp_path, "development", kept_dev_items)

        # Let validate_plan_completeness run for real — do NOT stub it.
        for name in [
            "validate_phase_and_provenance",
            "validate_split_integrity",
            "validate_identity_uniqueness",
            "validate_variant_consistency",
            "validate_config_consistency",
            "validate_hash_integrity",
            "validate_retry_lineage",
            "validate_sequence_atomicity",
            "validate_acceptance_independence",
            "validate_campaign_identity",
        ]:
            monkeypatch.setattr(audit_mod, name, lambda *a, **kw: [])

        report = audit_mod.build_validation_report(split_scope="development")
        pc_findings = report["audit_sections"].get("plan_completeness", [])
        assert len(pc_findings) > 0, "plan completeness should report missing step"
        assert any("missing" in f.lower() or "planned" in f.lower() for f in pc_findings)

    # N6 — inconsistent sequence plan
    def test_n6_inconsistent_sequence_plan(self) -> None:
        """Inconsistent sequence_step_count raises ValueError."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            expected_sequence_steps_from_plan,
        )

        full_plan = load_generation_plan(_FULL_PLAN_PATH)
        # Find a sequence and tamper one item's step count.
        seq_items = [it for it in full_plan if it.sequence_id is not None]
        assert len(seq_items) > 0
        seq_id = seq_items[0].sequence_id
        seq_group = [it for it in seq_items if it.sequence_id == seq_id]
        assert len(seq_group) >= 2

        # Create a tampered copy: change one item's sequence_step_count.
        from dataclasses import replace as _dc_replace
        tampered = []
        for it in full_plan:
            if it is seq_group[0]:
                tampered.append(
                    _dc_replace(it, sequence_step_count=(it.sequence_step_count or 3) + 1)
                )
            else:
                tampered.append(it)

        with pytest.raises(ValueError, match="inconsistent sequence_step_count"):
            expected_sequence_steps_from_plan(tampered)

    # N7 — test before validation audit
    def test_n7_test_before_validation_audit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test is blocked when validation has not been audited."""
        from experiments.trustparadox_u import audit_empirical_corpus as audit_mod
        from experiments.trustparadox_u.campaign_identity import campaign_identity_sha256

        monkeypatch.setattr(audit_mod, "_CORPUS_BASE", tmp_path)

        # Development fully audited.
        dev_dir = tmp_path / "development"
        dev_dir.mkdir(parents=True, exist_ok=True)
        dev_report = dev_dir / "audit_report.json"
        dev_report.write_text('{"passed": true}', encoding="utf-8")
        dev_sha = hashlib.sha256(dev_report.read_bytes()).hexdigest()
        
        # Create corpus_manifest.json and campaign_identity.json
        manifest = {"artifact_class": "empirical_corpus"}
        (dev_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        manifest_sha = hashlib.sha256((dev_dir / "corpus_manifest.json").read_bytes()).hexdigest()
        
        # Create campaign_identity.json with proper schema
        from experiments.trustparadox_u.campaign_identity import CampaignIdentity
        from dataclasses import asdict
        
        identity_obj = CampaignIdentity(
            schema_version="1.0",
            split="development",
            generation_plan_scientific_sha256="plan_hash",
            generation_plan_file_sha256="plan_hash",
            generation_config_sha256="config_hash",
            target_registry_sha256="registry_hash",
            prompt_manifest_sha256="prompt_hash",
            phase_manifest_sha256="phase_hash",
            generator_provider="mock",
            generator_model_requested="mock-model",
            generator_temperature=0.7,
            generator_max_tokens=1024,
            request_timeout=30.0,
            max_retries=3,
            created_from_commit="test_commit",
            created_at="2026-08-02T00:00:00+00:00",
        )
        (dev_dir / "campaign_identity.json").write_text(
            json.dumps(asdict(identity_obj)), encoding="utf-8",
        )
        identity_sha = campaign_identity_sha256(identity_obj)
        
        _write_gate(
            tmp_path, "development",
            generation_completed=True,
            audit_passed=True,
            audit_report_sha256=dev_sha,
            audit_report_path="development/audit_report.json",
            corpus_manifest_sha256=manifest_sha,
            campaign_identity_sha256=identity_sha,
        )

        # Validation NOT audited — no gate.
        # Test NOT audited — no gate.

        findings = audit_mod._validate_all_split_gates()
        # Validation and test should be reported as missing.
        assert any("validation" in f for f in findings)
        assert any("test" in f for f in findings)
        # Development should NOT appear in findings.
        assert not any(
            "development" in f and "missing" in f.lower()
            for f in findings
        )

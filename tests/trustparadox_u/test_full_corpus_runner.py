"""Phase 3 Final: full corpus runner tests.

Covers:
- Artifact classification (Patch B)
- Resume rejection after audit (Patch C)
- Stale audit evidence clearing (Patch C)
- Source commit in generation gate (Patch D)
- Audit promotion requires source commit (Patch D)
- Prerequisite source commit checks (Patch E)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from experiments.trustparadox_u.empirical_generation_plan import (
    update_generation_gate_after_audit,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Artifact classification tests (Patch B)
# ---------------------------------------------------------------------------


class TestArtifactClassification:
    """Verify artifact class constants and auditor enforcement."""

    def test_full_real_campaign_artifact_class(self) -> None:
        """Plan-driven real campaign must use empirical_corpus."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            ARTIFACT_CLASS_EMPIRICAL_CORPUS,
        )
        assert ARTIFACT_CLASS_EMPIRICAL_CORPUS == "empirical_corpus"

    def test_full_real_campaign_research_use(self) -> None:
        """Plan-driven real campaign must use pending_annotation_and_replay."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            RESEARCH_USE_PENDING_ANNOTATION,
        )
        assert RESEARCH_USE_PENDING_ANNOTATION == "pending_annotation_and_replay"

    def test_real_preflight_artifact_class(self) -> None:
        """Real API preflight must use real_api_preflight / diagnostic_only."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            ARTIFACT_CLASS_REAL_API_PREFLIGHT,
            RESEARCH_USE_DIAGNOSTIC,
        )
        assert ARTIFACT_CLASS_REAL_API_PREFLIGHT == "real_api_preflight"
        assert RESEARCH_USE_DIAGNOSTIC == "diagnostic_only"

    def test_auditor_rejects_smoke_label_for_final_corpus(
        self, tmp_path: Path,
    ) -> None:
        """Auditor must reject a final corpus labeled development_smoke."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor

        split_dir = tmp_path / "development"
        split_dir.mkdir()
        # Write a manifest with smoke labels (wrong for final corpus).
        manifest = {
            "artifact_class": "development_smoke",
            "research_use": "diagnostic_only",
        }
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        with patch.object(auditor, "_CORPUS_BASE", tmp_path):
            findings = auditor.validate_artifact_classification(
                target_splits=("development",),
            )
        assert len(findings) == 2
        assert any("artifact_class" in f for f in findings)
        assert any("research_use" in f for f in findings)

    def test_auditor_accepts_correct_labels(self, tmp_path: Path) -> None:
        """Auditor accepts empirical_corpus / pending_annotation_and_replay."""
        from experiments.trustparadox_u import audit_empirical_corpus as auditor

        split_dir = tmp_path / "development"
        split_dir.mkdir()
        manifest = {
            "artifact_class": "empirical_corpus",
            "research_use": "pending_annotation_and_replay",
        }
        (split_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        with patch.object(auditor, "_CORPUS_BASE", tmp_path):
            findings = auditor.validate_artifact_classification(
                target_splits=("development",),
            )
        assert findings == []


# ---------------------------------------------------------------------------
# Resume rejection tests (Patch C)
# ---------------------------------------------------------------------------


class TestAuditedSplitResumeRejection:
    """Audited splits are immutable — reject --resume after audit PASS."""

    def test_audited_split_resume_is_rejected(self, tmp_path: Path) -> None:
        """run_full_corpus_generation.main() rejects --resume when audit_passed=true."""
        from scripts.run_full_corpus_generation import main

        split = "development"
        output_dir = tmp_path / split
        output_dir.mkdir(parents=True)
        # Create campaign artifacts so _has_existing_campaign returns True.
        (output_dir / "campaign_identity.json").write_text("{}", encoding="utf-8")

        # Create a gate with audit_passed=true.
        gate_dir = tmp_path
        gate = {
            "split": split,
            "generation_completed": True,
            "audit_passed": True,
            "source_commit": "abc123",
        }
        gate_path = gate_dir / f"{split}_generation_gate.json"
        gate_path.write_text(json.dumps(gate), encoding="utf-8")

        with (
            patch("scripts.run_full_corpus_generation._OUTPUT_BASE", tmp_path),
            patch("scripts.run_full_corpus_generation._GATE_DIR", gate_dir),
            patch("scripts.run_full_corpus_generation._check_phase_is_e3", return_value=None),
            patch("scripts.run_full_corpus_generation._check_frozen_hashes", return_value=None),
            patch("scripts.run_full_corpus_generation._check_clean_tree", return_value=None),
            patch("scripts.run_full_corpus_generation._check_prerequisite_gate", return_value=None),
        ):
            rc = main([
                "--split", split,
                "--resume",
                "--skip-preflight-checks",
            ])
        # Should be rejected with exit code 2.
        assert rc == 2

    def test_audited_split_resume_makes_zero_provider_calls(
        self, tmp_path: Path,
    ) -> None:
        """When audited resume is rejected, no provider calls are made."""
        from scripts.run_full_corpus_generation import main

        split = "development"
        output_dir = tmp_path / split
        output_dir.mkdir(parents=True)
        (output_dir / "campaign_identity.json").write_text("{}", encoding="utf-8")

        gate_dir = tmp_path
        gate = {
            "split": split,
            "generation_completed": True,
            "audit_passed": True,
        }
        gate_path = gate_dir / f"{split}_generation_gate.json"
        gate_path.write_text(json.dumps(gate), encoding="utf-8")

        with (
            patch("scripts.run_full_corpus_generation._OUTPUT_BASE", tmp_path),
            patch("scripts.run_full_corpus_generation._GATE_DIR", gate_dir),
            patch("scripts.run_full_corpus_generation._check_phase_is_e3", return_value=None),
            patch("scripts.run_full_corpus_generation._check_frozen_hashes", return_value=None),
            patch("scripts.run_full_corpus_generation._check_clean_tree", return_value=None),
            patch("scripts.run_full_corpus_generation._check_prerequisite_gate", return_value=None),
            patch("scripts.run_full_corpus_generation.run_split") as mock_run,
        ):
            rc = main([
                "--split", split,
                "--resume",
                "--skip-preflight-checks",
            ])
        assert rc == 2
        # run_split must NOT have been called — zero provider calls.
        mock_run.assert_not_called()

    def test_unaudited_interrupted_split_can_resume(
        self, tmp_path: Path,
    ) -> None:
        """A split with generation_completed=true but audit_passed=false can resume."""
        from scripts.run_full_corpus_generation import main

        split = "development"
        output_dir = tmp_path / split
        output_dir.mkdir(parents=True)
        (output_dir / "campaign_identity.json").write_text("{}", encoding="utf-8")

        gate_dir = tmp_path
        gate = {
            "split": split,
            "generation_completed": True,
            "audit_passed": False,  # NOT audited yet.
        }
        gate_path = gate_dir / f"{split}_generation_gate.json"
        gate_path.write_text(json.dumps(gate), encoding="utf-8")

        with (
            patch("scripts.run_full_corpus_generation._OUTPUT_BASE", tmp_path),
            patch("scripts.run_full_corpus_generation._GATE_DIR", gate_dir),
            patch("scripts.run_full_corpus_generation._check_phase_is_e3", return_value=None),
            patch("scripts.run_full_corpus_generation._check_frozen_hashes", return_value=None),
            patch("scripts.run_full_corpus_generation._check_clean_tree", return_value=None),
            patch("scripts.run_full_corpus_generation._check_prerequisite_gate", return_value=None),
            patch("scripts.run_full_corpus_generation.run_split", return_value=0) as mock_run,
            patch("scripts.run_full_corpus_generation._write_generation_gate"),
            patch("scripts.run_full_corpus_generation._load_gate", return_value=gate),
        ):
            # This should NOT be rejected (audit_passed=false).
            # It will proceed to run_split. We mock run_split to return 0.
            # The plan completeness check will fail but that's OK — we just
            # verify the resume wasn't blocked.
            main([
                "--split", split,
                "--resume",
                "--skip-preflight-checks",
            ])
        # run_split WAS called — resume was allowed.
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Stale audit evidence clearing (Patch C)
# ---------------------------------------------------------------------------


class TestStaleAuditEvidenceClearing:
    """New generation must clear prior audit evidence."""

    def test_new_generation_invalidates_old_audit_evidence(
        self, tmp_path: Path,
    ) -> None:
        """_write_generation_gate clears audit_passed and audit fields."""
        from scripts.run_full_corpus_generation import _write_generation_gate

        gate_dir = tmp_path
        # Pre-write a gate with audit_passed=true.
        gate_path = gate_dir / "development_generation_gate.json"
        old_gate = {
            "split": "development",
            "generation_completed": True,
            "audit_passed": True,
            "audit_report_sha256": "old_hash",
            "audit_report_path": "old/path",
            "audit_source_commit": "old_commit",
            "audited_at": "2025-01-01T00:00:00",
            "source_commit": "old_source",
            "corpus_manifest_sha256": "old_manifest_hash",
            "campaign_identity_sha256": "old_identity_hash",
        }
        gate_path.write_text(json.dumps(old_gate), encoding="utf-8")

        with (
            patch("scripts.run_full_corpus_generation._GATE_DIR", gate_dir),
            patch("scripts.run_full_corpus_generation._current_repository_commit", return_value="new_commit"),
        ):
            new_gate_path = _write_generation_gate(
                "development",
                generation_completed=True,
                planned_plan_item_count=225,
                accounted_plan_item_count=225,
                missing_plan_item_count=0,
            )

        new_gate = json.loads(new_gate_path.read_text(encoding="utf-8"))
        # Audit evidence must be cleared.
        assert new_gate["audit_passed"] is False
        assert new_gate["audit_report_sha256"] is None
        assert new_gate["audit_report_path"] is None
        assert new_gate["audit_source_commit"] is None
        assert new_gate["audited_at"] is None
        # Patch I: corpus/identity bindings must be cleared.
        assert new_gate["corpus_manifest_sha256"] is None
        assert new_gate["campaign_identity_sha256"] is None
        # Source commit must be updated.
        assert new_gate["source_commit"] == "new_commit"


# ---------------------------------------------------------------------------
# Source commit in gate (Patch D)
# ---------------------------------------------------------------------------


class TestSourceCommitInGate:
    """Generation gate must record source_commit."""

    def test_gate_records_source_commit(self, tmp_path: Path) -> None:
        """_write_generation_gate writes source_commit = current HEAD."""
        from scripts.run_full_corpus_generation import _write_generation_gate

        gate_dir = tmp_path
        with (
            patch("scripts.run_full_corpus_generation._GATE_DIR", gate_dir),
            patch("scripts.run_full_corpus_generation._current_repository_commit", return_value="abc123def"),
        ):
            gate_path = _write_generation_gate(
                "development",
                generation_completed=True,
                planned_plan_item_count=225,
                accounted_plan_item_count=225,
                missing_plan_item_count=0,
            )

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        assert gate["source_commit"] == "abc123def"

    def test_audit_promotion_requires_generation_source_commit(
        self, tmp_path: Path,
    ) -> None:
        """Audit promotion must set audit_source_commit."""
        gate_dir = tmp_path
        gate_path = gate_dir / "development_generation_gate.json"
        gate = {
            "split": "development",
            "source_commit": "abc123",
            "generation_completed": True,
            "audit_passed": False,
        }
        gate_path.write_text(json.dumps(gate), encoding="utf-8")

        with patch(
            "experiments.trustparadox_u.empirical_generation_plan._CORPUS_GENERATION_BASE",
            tmp_path,
        ):
            updated = update_generation_gate_after_audit(
                split="development",
                audit_passed=True,
                audit_report_path=Path("development/audit_report.json"),
                audit_report_sha256="report_hash",
                source_commit="abc123",
                corpus_manifest_sha256="manifest_hash",
                campaign_identity_sha256="identity_hash",
                base=tmp_path,
            )
        assert updated["audit_source_commit"] == "abc123"
        assert updated["source_commit"] == "abc123"
        assert updated["corpus_manifest_sha256"] == "manifest_hash"
        assert updated["campaign_identity_sha256"] == "identity_hash"


# ---------------------------------------------------------------------------
# Prerequisite source commit checks (Patch E)
# ---------------------------------------------------------------------------


class TestPrerequisiteSourceCommit:
    """Next-split progression requires same source commit."""

    def test_validation_requires_same_source_commit(
        self, tmp_path: Path,
    ) -> None:
        """Validation is blocked when development was audited under a different commit."""
        from scripts.run_full_corpus_generation import _check_prerequisite_gate

        output_base = tmp_path
        # Create development gate with old commit.
        gate_dir = tmp_path
        # SHA256 of b"report" (the audit report content).
        _audit_sha = "845e91831319e89c4d656bdb80c278ac09a7230d61e5dfd2e1b1fbb436ac8917"
        dev_gate = {
            "split": "development",
            "source_commit": "old_commit",
            "audit_source_commit": "old_commit",
            "generation_completed": True,
            "audit_passed": True,
            "audit_report_sha256": _audit_sha,
            "audit_report_path": "development/audit_report.json",
            "corpus_manifest_sha256": "manifest_hash",
            "campaign_identity_sha256": "identity_hash",
        }
        (gate_dir / "development_generation_gate.json").write_text(
            json.dumps(dev_gate), encoding="utf-8",
        )
        # Create audit report file.
        audit_dir = tmp_path / "development"
        audit_dir.mkdir()
        (audit_dir / "audit_report.json").write_text("report", encoding="utf-8")
        # Create campaign identity with old commit.
        identity = {
            "created_from_commit": "old_commit",
            "split": "development",
        }
        (audit_dir / "campaign_identity.json").write_text(
            json.dumps(identity), encoding="utf-8",
        )
        # Create corpus_manifest.json.
        (audit_dir / "corpus_manifest.json").write_text(
            json.dumps({"artifact_class": "empirical_corpus"}), encoding="utf-8",
        )

        with (
            patch("scripts.run_full_corpus_generation._OUTPUT_BASE", output_base),
            patch("scripts.run_full_corpus_generation._GATE_DIR", gate_dir),
            patch("scripts.run_full_corpus_generation._current_repository_commit", return_value="new_commit"),
        ):
            err = _check_prerequisite_gate("validation")
        assert err is not None
        assert "source commit inconsistency" in err

    def test_prerequisite_gate_commit_mismatch_blocks_progression(
        self, tmp_path: Path,
    ) -> None:
        """Gate source_commit != HEAD blocks next split."""
        from scripts.run_full_corpus_generation import _check_prerequisite_gate

        output_base = tmp_path
        gate_dir = tmp_path
        _audit_sha = "845e91831319e89c4d656bdb80c278ac09a7230d61e5dfd2e1b1fbb436ac8917"
        dev_gate = {
            "split": "development",
            "source_commit": "commit_A",
            "audit_source_commit": "commit_A",
            "generation_completed": True,
            "audit_passed": True,
            "audit_report_sha256": _audit_sha,
            "audit_report_path": "development/audit_report.json",
            "corpus_manifest_sha256": "manifest_hash",
            "campaign_identity_sha256": "identity_hash",
        }
        (gate_dir / "development_generation_gate.json").write_text(
            json.dumps(dev_gate), encoding="utf-8",
        )
        audit_dir = tmp_path / "development"
        audit_dir.mkdir()
        (audit_dir / "audit_report.json").write_text("report", encoding="utf-8")
        (audit_dir / "campaign_identity.json").write_text(
            json.dumps({"created_from_commit": "commit_A"}), encoding="utf-8",
        )
        (audit_dir / "corpus_manifest.json").write_text(
            json.dumps({"artifact_class": "empirical_corpus"}), encoding="utf-8",
        )

        with (
            patch("scripts.run_full_corpus_generation._OUTPUT_BASE", output_base),
            patch("scripts.run_full_corpus_generation._GATE_DIR", gate_dir),
            patch("scripts.run_full_corpus_generation._current_repository_commit", return_value="commit_B"),
        ):
            err = _check_prerequisite_gate("validation")
        assert err is not None
        assert "commit_A" in err
        assert "commit_B" in err

    def test_prerequisite_identity_commit_mismatch_blocks_progression(
        self, tmp_path: Path,
    ) -> None:
        """Campaign identity commit != HEAD blocks next split."""
        from scripts.run_full_corpus_generation import _check_prerequisite_gate

        output_base = tmp_path
        gate_dir = tmp_path
        _audit_sha = "845e91831319e89c4d656bdb80c278ac09a7230d61e5dfd2e1b1fbb436ac8917"
        dev_gate = {
            "split": "development",
            "source_commit": "commit_A",
            "audit_source_commit": "commit_A",
            "generation_completed": True,
            "audit_passed": True,
            "audit_report_sha256": _audit_sha,
            "audit_report_path": "development/audit_report.json",
            "corpus_manifest_sha256": "manifest_hash",
            "campaign_identity_sha256": "identity_hash",
        }
        (gate_dir / "development_generation_gate.json").write_text(
            json.dumps(dev_gate), encoding="utf-8",
        )
        audit_dir = tmp_path / "development"
        audit_dir.mkdir()
        (audit_dir / "audit_report.json").write_text("report", encoding="utf-8")
        # Identity has a DIFFERENT commit from gate.
        (audit_dir / "campaign_identity.json").write_text(
            json.dumps({"created_from_commit": "commit_C"}), encoding="utf-8",
        )
        (audit_dir / "corpus_manifest.json").write_text(
            json.dumps({"artifact_class": "empirical_corpus"}), encoding="utf-8",
        )

        with (
            patch("scripts.run_full_corpus_generation._OUTPUT_BASE", output_base),
            patch("scripts.run_full_corpus_generation._GATE_DIR", gate_dir),
            patch("scripts.run_full_corpus_generation._current_repository_commit", return_value="commit_A"),
        ):
            err = _check_prerequisite_gate("validation")
        assert err is not None
        assert "commit_C" in err

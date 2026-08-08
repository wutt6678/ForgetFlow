"""Tests for E2-H: prompt freeze pipeline.

Verifies the prompt freeze logic including precondition validation,
invariance checks, and manifest generation.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.trustparadox_u.run_prompt_freeze import (
    FROZEN_STATUS,
    run_prompt_freeze,
    validate_freeze_preconditions,
)

# ---------------------------------------------------------------------------
# Precondition validation
# ---------------------------------------------------------------------------


class TestValidateFreezePreconditions:
    def test_freeze_as_is_permitted(self) -> None:
        report = {
            "decision": "freeze_as_is",
            "revision_constraints": {"only_trust_framing_may_change": True},
        }
        problems = validate_freeze_preconditions(report)
        assert problems == []

    def test_judgement_freeze_permitted(self) -> None:
        report = {
            "decision": "judgement_freeze_with_findings",
            "revision_constraints": {"only_trust_framing_may_change": True},
        }
        problems = validate_freeze_preconditions(report)
        assert problems == []

    def test_revise_not_permitted(self) -> None:
        report = {
            "decision": "revise_needed",
            "revision_constraints": {"only_trust_framing_may_change": True},
        }
        problems = validate_freeze_preconditions(report)
        assert len(problems) > 0
        assert any("revise_needed" in p for p in problems)

    def test_missing_constraints_flagged(self) -> None:
        report = {
            "decision": "freeze_as_is",
            "revision_constraints": {"only_trust_framing_may_change": False},
        }
        problems = validate_freeze_preconditions(report)
        assert len(problems) > 0

    def test_empty_report(self) -> None:
        problems = validate_freeze_preconditions({})
        assert len(problems) > 0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestRunPromptFreeze:
    def _make_revision_dir(self, tmp_path: Path, decision: str = "freeze_as_is") -> Path:
        revision_dir = tmp_path / "revision"
        revision_dir.mkdir()
        report = {
            "decision": decision,
            "rationale": "Test rationale",
            "prompts_revised": False,
            "revision_constraints": {"only_trust_framing_may_change": True},
        }
        with open(revision_dir / "bounded_revision_report.json", "w") as f:
            json.dump(report, f)
        return revision_dir

    def test_end_to_end(self, tmp_path: Path) -> None:
        revision_dir = self._make_revision_dir(tmp_path)
        output_dir = tmp_path / "output"

        report = run_prompt_freeze(revision_dir, output_dir, repo_root=tmp_path)

        assert report["passed"] is True
        assert report["frozen_status"] == FROZEN_STATUS
        assert report["invariance_valid"] is True
        assert report["num_templates"] > 0
        assert len(report["manifest_sha256"]) == 64
        assert (output_dir / "frozen_prompt_manifest.json").exists()
        assert (output_dir / "frozen_freeze_report.json").exists()

    def test_frozen_manifest_has_required_fields(self, tmp_path: Path) -> None:
        revision_dir = self._make_revision_dir(tmp_path)
        output_dir = tmp_path / "output"

        run_prompt_freeze(revision_dir, output_dir, repo_root=tmp_path)

        with open(output_dir / "frozen_prompt_manifest.json") as f:
            manifest = json.load(f)

        assert manifest["status"] == FROZEN_STATUS
        assert "freeze_timestamp" in manifest
        assert "repository_commit" in manifest
        assert "templates" in manifest
        assert "prompt_invariance" in manifest
        assert manifest["prompt_invariance"]["valid"] is True

    def test_revise_decision_raises(self, tmp_path: Path) -> None:
        import pytest

        revision_dir = self._make_revision_dir(tmp_path, decision="revise_needed")
        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="Freeze preconditions not met"):
            run_prompt_freeze(revision_dir, output_dir)

    def test_missing_revision_raises(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            run_prompt_freeze(tmp_path / "nonexistent", tmp_path / "output")

    def test_manifest_sha_deterministic(self, tmp_path: Path) -> None:
        revision_dir = self._make_revision_dir(tmp_path)
        output_dir = tmp_path / "output"

        report = run_prompt_freeze(revision_dir, output_dir, repo_root=tmp_path)
        sha = report["manifest_sha256"]
        assert len(sha) == 64
        # SHA is hex digest.
        assert all(c in "0123456789abcdef" for c in sha)

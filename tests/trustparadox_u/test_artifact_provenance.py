"""Behavioral tests for FF92-023 artifact commit provenance rules.

Required coverage:

1. A result commit that differs from the current commit fails.
2. A dirty commit suffix fails certification.
3. An unknown/invalid commit fails certification.
4. Stale result directories (missing or outdated provenance files) fail.
5. A regenerated clean result at the current commit passes.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.trustparadox_u.artifact_provenance import (
    REQUIRED_PROVENANCE_FIELDS,
    build_certification_provenance,
    validate_artifact_provenance,
    validate_result_provenance_file,
)

CURRENT = "abcdef0123456789abcdef0123456789abcdef01"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _clean_record(commit: str = CURRENT) -> dict[str, object]:
    return {
        "tested_code_commit": commit,
        "artifact_generation_commit": commit,
        "repository_clean": True,
        "workflow_run_id": "",
        "workflow_attempt": "",
    }


def test_mismatched_result_commit_fails() -> None:
    findings = validate_artifact_provenance(_clean_record(commit=OTHER), current_commit=CURRENT)
    assert any("stale_result_commit" in f for f in findings), findings


def test_dirty_commit_suffix_fails() -> None:
    record = _clean_record(commit=f"{CURRENT}-dirty")
    findings = validate_artifact_provenance(record, current_commit=CURRENT)
    assert any(
        "invalid_commit" in finding and "-dirty" in finding for finding in findings
    ), findings
    # A dirty artifact is also never certifiable even if commits align.
    record = _clean_record()
    record["repository_clean"] = False
    findings = validate_artifact_provenance(record, current_commit=CURRENT)
    assert any("dirty_artifact" in f for f in findings), findings


def test_unknown_commit_fails() -> None:
    findings = validate_artifact_provenance(_clean_record(commit="unknown"), current_commit=CURRENT)
    assert any("invalid_commit" in f for f in findings), findings

    findings = validate_artifact_provenance(
        _clean_record(commit="not-a-sha"), current_commit=CURRENT
    )
    assert any("invalid_commit" in f for f in findings), findings

    # Missing provenance fields are a finding too — silence is never a pass.
    incomplete = _clean_record()
    del incomplete["workflow_run_id"]
    findings = validate_artifact_provenance(incomplete, current_commit=CURRENT)
    assert findings == ["missing_provenance_fields: ['workflow_run_id']"], findings


def test_stale_result_directories_fail(tmp_path: Path) -> None:
    # Outdated commit inside the result file -> stale directory.
    stale = tmp_path / "run_manifest.json"
    stale.write_text(json.dumps({"provenance": _clean_record(commit=OTHER)}))
    findings = validate_result_provenance_file(stale, current_commit=CURRENT)
    assert any("stale_result_commit" in f for f in findings), findings

    # Missing provenance file -> stale directory.
    findings = validate_result_provenance_file(tmp_path / "missing.json", current_commit=CURRENT)
    assert findings == ["missing_provenance_file: missing.json"], findings

    # Unreadable JSON -> stale directory.
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    findings = validate_result_provenance_file(broken, current_commit=CURRENT)
    assert any("unreadable_provenance_file" in f for f in findings), findings


def test_regenerated_clean_result_passes() -> None:
    record = build_certification_provenance(repository_commit=CURRENT)
    for field in REQUIRED_PROVENANCE_FIELDS:
        assert field in record, field
    assert record["repository_clean"] is True
    assert validate_artifact_provenance(record, current_commit=CURRENT) == []


def test_built_provenance_from_dirty_snapshot_is_not_certifiable() -> None:
    record = build_certification_provenance(repository_commit=f"{CURRENT}-dirty")
    assert record["repository_clean"] is False
    findings = validate_artifact_provenance(record, current_commit=CURRENT)
    assert any("dirty_artifact" in f for f in findings), findings


def test_embedded_provenance_block_is_read(tmp_path: Path) -> None:
    payload = tmp_path / "study_manifest.json"
    payload.write_text(json.dumps({"provenance": _clean_record()}))
    assert validate_result_provenance_file(payload, current_commit=CURRENT) == []

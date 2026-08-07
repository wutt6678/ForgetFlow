"""FP-012: fixture tests for the generation/storage provenance schema boundaries.

Scientific artifacts record generation provenance; storage provenance is
recorded authoritatively in the release's STORAGE_PROVENANCE.json sidecar.
These fixtures exercise every boundary of that split: what a generation
record may and may not carry, what a finalized sidecar must carry, the
git-ancestry ordering rules, the reproduction version-agreement rule, the
two-digest model, and the active-release counting rules of the release
storage provenance gate.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from experiments.trustparadox_u import release_bundle, research_valid_gate
from experiments.trustparadox_u.artifact_provenance import (
    STORAGE_REFERENCE_KEY,
    generation_provenance_findings,
    storage_provenance_findings,
    validate_release_lineage,
    validate_storage_reference,
)
from experiments.trustparadox_u.frozen_thresholds import STUDY_VERSION
from experiments.trustparadox_u.release_bundle import (
    BUNDLE_MANIFEST_NAME,
    STORAGE_PROVENANCE_NAME,
    release_digest,
    release_dirs,
    storage_metadata_digest,
    supersede_release,
    write_storage_provenance,
)
from experiments.trustparadox_u.reproduce import (
    ReproductionError,
    _check_version_agreement,
)
from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture(scope="module")
def commits() -> dict[str, str]:
    """Real repository commits used to exercise ancestry rules."""
    root = _git("rev-list", "--max-parents=0", "HEAD").split()[0]
    return {
        "head": _git("rev-parse", "HEAD"),
        "older": _git("rev-parse", "HEAD~3"),
        "root": root,
    }


def _generation_record(**overrides: Any) -> dict[str, Any]:
    """A complete generation record: no embedded storage identity."""
    record: dict[str, Any] = {
        "tested_code_commit": "0" * 40,
        "artifact_generation_commit": "0" * 40,
        "artifact_generation_tree": "t" * 64,
        "workflow_run_id": "local",
        "workflow_attempt": "local",
        "certification_source": "local",
        "protocol_version": PROTOCOL_VERSION,
        "study_version": STUDY_VERSION,
        "environment_lock_hash": "e" * 64,
        "repository_clean": True,
        "artifact_storage_commit": None,
        STORAGE_REFERENCE_KEY: {"source": STORAGE_PROVENANCE_NAME, "authoritative": True},
    }
    record.update(overrides)
    return record


def _fixture_manifest(release_id: str, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "release_id": release_id,
        "status": "active",
        "study_version": STUDY_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "corpus": {"corpus_sha256": "c" * 64},
        "annotations": {"annotation_hash": "a" * 64},
        "components": {},
        "provenance": provenance,
    }


def _sidecar_record(manifest: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    sidecar: dict[str, Any] = {
        "schema_version": "1.1",
        "release_id": str(manifest["release_id"]),
        "tested_code_commit": str(manifest["provenance"]["tested_code_commit"]),
        "artifact_generation_commit": str(manifest["provenance"]["artifact_generation_commit"]),
        "artifact_storage_commit": str(manifest["provenance"]["tested_code_commit"]),
        "gate_snapshot_commit": str(manifest["provenance"]["tested_code_commit"]),
        "verified_at": "2026-01-01T00:00:00+00:00",
        "scientific_release_digest": release_digest(manifest),
    }
    sidecar.update(overrides)
    sidecar["storage_metadata_digest"] = storage_metadata_digest(sidecar)
    return sidecar


def _write_fixture_release(
    releases_dir: Path,
    commits: dict[str, str],
    release_id: str = "fixture-release-000000000000",
    *,
    generation: str = "",
    storage: str = "",
    snapshot: str = "",
    sidecar_overrides: dict[str, Any] | None = None,
    status: str = "active",
) -> Path:
    """Write a self-consistent fixture bundle plus its storage sidecar."""
    head = commits["head"]
    bundle_dir = releases_dir / release_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    provenance = _generation_record(
        tested_code_commit=generation or head,
        artifact_generation_commit=generation or head,
    )
    manifest = _fixture_manifest(release_id, provenance)
    manifest["status"] = status
    (bundle_dir / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    sidecar = _sidecar_record(
        manifest,
        artifact_storage_commit=storage or head,
        gate_snapshot_commit=snapshot or head,
        **(sidecar_overrides or {}),
    )
    (bundle_dir / STORAGE_PROVENANCE_NAME).write_text(json.dumps(sidecar, indent=2) + "\n")
    return bundle_dir


@pytest.fixture()
def fixture_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the release machinery at an isolated releases directory."""
    releases = tmp_path / "releases"
    releases.mkdir()
    monkeypatch.setattr(release_bundle, "RELEASES_DIR", releases)
    monkeypatch.setattr(release_bundle, "ARCHIVE_DIR", tmp_path / "archive")
    return releases


# ---------------------------------------------------------------------------
# Generation-record boundaries (FP-001/FP-002/FP-010)
# ---------------------------------------------------------------------------


def test_generation_record_without_storage_commit_is_valid() -> None:
    assert generation_provenance_findings(_generation_record()) == []


def test_generation_record_empty_string_storage_commit_is_invalid() -> None:
    findings = generation_provenance_findings(_generation_record(artifact_storage_commit=""))
    assert any("stale_embedded_storage_commit" in finding for finding in findings)


def test_stale_embedded_storage_commit_is_invalid() -> None:
    findings = generation_provenance_findings(
        _generation_record(artifact_storage_commit="a" * 40, gate_snapshot_commit="b" * 40)
    )
    assert sum("stale_embedded_storage_commit" in finding for finding in findings) == 2


def test_generation_record_sidecar_reference_is_valid() -> None:
    assert validate_storage_reference(_generation_record()) == []
    findings = validate_storage_reference(_generation_record(storage_provenance=None))
    assert any("storage_provenance_reference_missing" in finding for finding in findings)


# ---------------------------------------------------------------------------
# Sidecar boundaries (FP-002)
# ---------------------------------------------------------------------------


def test_sidecar_missing_storage_commit_fails(commits: dict[str, str]) -> None:
    manifest = _fixture_manifest(
        "rel",
        _generation_record(
            tested_code_commit=commits["head"],
            artifact_generation_commit=commits["head"],
        ),
    )
    sidecar = _sidecar_record(manifest)
    del sidecar["artifact_storage_commit"]
    findings = storage_provenance_findings(sidecar)
    assert "storage_provenance_field_missing: artifact_storage_commit" in findings


def test_sidecar_missing_gate_commit_fails_after_finalization(commits: dict[str, str]) -> None:
    manifest = _fixture_manifest(
        "rel",
        _generation_record(
            tested_code_commit=commits["head"],
            artifact_generation_commit=commits["head"],
        ),
    )
    # GE-001: both the evidence commit and its deprecated alias empty.
    sidecar = _sidecar_record(manifest, gate_evidence_commit="", gate_snapshot_commit="")
    finalized = storage_provenance_findings(sidecar, require_gate_snapshot=True)
    assert "storage_provenance_field_empty: gate_evidence_commit" in finalized
    # Before finalization the pending gate-evidence commit is tolerated.
    pending = storage_provenance_findings(sidecar, require_gate_snapshot=False)
    assert "storage_provenance_field_empty: gate_evidence_commit" not in pending


# ---------------------------------------------------------------------------
# Ancestry ordering rules (FP-008)
# ---------------------------------------------------------------------------


def _lineage_record(**overrides: Any) -> dict[str, Any]:
    record = _generation_record(
        artifact_generation_tree="t" * 64,
        environment_lock_hash="e" * 64,
    )
    record.update(overrides)
    return record


def test_storage_commit_before_generation_fails(commits: dict[str, str]) -> None:
    # The generation commit must be an ancestor of the storage commit; a
    # storage commit that predates generation cannot contain it.
    record = _lineage_record(
        tested_code_commit=commits["head"],
        artifact_generation_commit=commits["head"],
        artifact_storage_commit=commits["older"],
    )
    findings = validate_release_lineage(record)
    assert any("lineage_not_ancestor" in finding for finding in findings)


def test_gate_snapshot_before_storage_fails(
    fixture_releases: Path, commits: dict[str, str]
) -> None:
    _write_fixture_release(
        fixture_releases, commits, storage=commits["head"], snapshot=commits["older"]
    )
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("storage_not_ancestor_of_gate_snapshot" in str(f) for f in result["findings"])


def test_gate_file_absent_at_gate_snapshot_fails(
    fixture_releases: Path, commits: dict[str, str]
) -> None:
    # The root commit predates the gate result file entirely.
    _write_fixture_release(
        fixture_releases,
        commits,
        generation=commits["root"],
        storage=commits["root"],
        snapshot=commits["root"],
    )
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("gate_file_missing_at_snapshot_commit" in str(f) for f in result["findings"])


# ---------------------------------------------------------------------------
# Sidecar / bundle-manifest agreement (FP-006/FP-007)
# ---------------------------------------------------------------------------


def test_sidecar_release_id_mismatch_fails(fixture_releases: Path, commits: dict[str, str]) -> None:
    _write_fixture_release(
        fixture_releases, commits, sidecar_overrides={"release_id": "some-other-release"}
    )
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("storage_sidecar_release_id_mismatch" in str(f) for f in result["findings"])


def test_sidecar_tested_generation_mismatch_fails(
    fixture_releases: Path, commits: dict[str, str]
) -> None:
    _write_fixture_release(
        fixture_releases,
        commits,
        sidecar_overrides={"tested_code_commit": commits["older"]},
    )
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("storage_sidecar_generation_mismatch" in str(f) for f in result["findings"])


# ---------------------------------------------------------------------------
# Reproduction version agreement (FP-003)
# ---------------------------------------------------------------------------


def test_reproduction_empty_protocol_version_fails() -> None:
    provenance = {"protocol_version": "", "study_version": STUDY_VERSION}
    inputs = {"protocol_version": PROTOCOL_VERSION, "study_version": STUDY_VERSION}
    with pytest.raises(ReproductionError, match="protocol_version mismatch"):
        _check_version_agreement(provenance, inputs)


def test_reproduction_version_mismatch_fails() -> None:
    provenance = {"protocol_version": "1.1.0", "study_version": STUDY_VERSION}
    inputs = {"protocol_version": PROTOCOL_VERSION, "study_version": STUDY_VERSION}
    with pytest.raises(ReproductionError, match="protocol_version mismatch"):
        _check_version_agreement(provenance, inputs)


def test_reproduction_version_agreement_passes() -> None:
    record = {"protocol_version": PROTOCOL_VERSION, "study_version": STUDY_VERSION}
    _check_version_agreement(dict(record), dict(record))


# ---------------------------------------------------------------------------
# Digest model (FP-001/PR-005)
# ---------------------------------------------------------------------------


def test_sidecar_update_preserves_scientific_digest(
    tmp_path: Path, commits: dict[str, str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest = _fixture_manifest(
        "digest-release",
        _generation_record(
            tested_code_commit=commits["head"],
            artifact_generation_commit=commits["head"],
        ),
    )
    manifest["release_digest"] = release_digest(manifest)
    (bundle_dir / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    write_storage_provenance(bundle_dir, artifact_storage_commit=commits["head"], verified_at="t1")
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    sidecar_path = bundle_dir / STORAGE_PROVENANCE_NAME
    scientific_before = release_digest(json.loads(manifest_path.read_text()))

    # Finalize the sidecar: record the gate snapshot commit and re-audit.
    sidecar = json.loads(sidecar_path.read_text())
    pending_digest = sidecar["storage_metadata_digest"]
    sidecar["gate_snapshot_commit"] = commits["head"]
    sidecar["verified_at"] = "t2"
    sidecar["storage_metadata_digest"] = storage_metadata_digest(sidecar)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    assert release_digest(json.loads(manifest_path.read_text())) == scientific_before
    assert sidecar["storage_metadata_digest"] != pending_digest


def test_sidecar_update_changes_storage_metadata_digest(commits: dict[str, str]) -> None:
    manifest = _fixture_manifest(
        "rel",
        _generation_record(
            tested_code_commit=commits["head"],
            artifact_generation_commit=commits["head"],
        ),
    )
    sidecar = _sidecar_record(manifest)
    before = storage_metadata_digest(sidecar)
    sidecar["gate_snapshot_commit"] = commits["older"]
    changed = storage_metadata_digest(sidecar)
    assert changed != before
    # Audit bookkeeping and the digest's own field never change it.
    sidecar["verified_at"] = "2027-01-01T00:00:00+00:00"
    sidecar["storage_metadata_digest"] = "f" * 64
    assert storage_metadata_digest(sidecar) == changed


# ---------------------------------------------------------------------------
# Active-release counting (FP-006)
# ---------------------------------------------------------------------------


def test_archived_releases_are_ignored_by_certification(
    fixture_releases: Path, commits: dict[str, str], tmp_path: Path
) -> None:
    bundle_dir = _write_fixture_release(fixture_releases, commits)
    supersede_release(bundle_dir, superseded_by="successor-release")
    assert release_dirs() == []
    archived = tmp_path / "archive" / bundle_dir.name
    assert (archived / "INVALIDATION_MARKER.json").exists()
    manifest = json.loads((archived / BUNDLE_MANIFEST_NAME).read_text())
    assert manifest["status"] == "superseded"


def test_multiple_active_releases_fail(fixture_releases: Path, commits: dict[str, str]) -> None:
    _write_fixture_release(fixture_releases, commits, release_id="release-a")
    _write_fixture_release(fixture_releases, commits, release_id="release-b")
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("multiple_active_releases" in str(f) for f in result["findings"])


def test_no_active_release_fails(fixture_releases: Path) -> None:
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("no_active_release" in str(f) for f in result["findings"])

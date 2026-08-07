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

from experiments.trustparadox_u import artifact_provenance, release_bundle, research_valid_gate
from experiments.trustparadox_u.artifact_provenance import (
    GATE_EVIDENCE_REQUIRED_GATES,
    STORAGE_REFERENCE_KEY,
    GateEvidenceError,
    gate_evidence_commit_of,
    gate_evidence_findings,
    gate_evidence_sha256,
    generation_provenance_findings,
    load_gate_evidence_at_commit,
    storage_provenance_findings,
    validate_release_lineage,
    validate_storage_reference,
)
from experiments.trustparadox_u.frozen_thresholds import STUDY_VERSION
from experiments.trustparadox_u.release_bundle import (
    BUNDLE_MANIFEST_NAME,
    FINAL_STORAGE_CERTIFICATION_NAME,
    STORAGE_PROVENANCE_NAME,
    release_digest,
    release_dirs,
    storage_metadata_digest,
    supersede_release,
    write_final_storage_certification,
    write_storage_provenance,
)
from experiments.trustparadox_u.reproduce import (
    ReproductionError,
    _check_version_agreement,
)
from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION
from experiments.trustparadox_u.status import (
    EMPIRICAL_REPLAY_VALID,
    STUDY_CLASS_DIAGNOSTIC,
    STUDY_CLASS_EMPIRICAL_REPLAY,
    SYNTHETIC_BENCHMARK_VALID,
)

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
    # GE-010: the root commit predates the gate result file entirely, so
    # the gate-evidence commit cannot certify the release.
    _write_fixture_release(
        fixture_releases,
        commits,
        generation=commits["root"],
        storage=commits["root"],
        snapshot=commits["root"],
    )
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("gate_evidence_file_missing" in str(f) for f in result["findings"])


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


# ---------------------------------------------------------------------------
# Gate-evidence certification fixtures (GE-012)
# ---------------------------------------------------------------------------

# Real historical gate-evidence commits: the passing certification and the
# failed gate that the pre-GE sidecar wrongly referenced.
_PASSING_EVIDENCE_COMMIT = "9acdd652a2204572e829662f1712546b25184731"
_FAILED_EVIDENCE_COMMIT = "77d53d2edee0a4743650c9311a3b99c8eec0c5dc"


def _passing_evidence_gate(**overrides: Any) -> dict[str, Any]:
    """A self-consistent passing historical gate record (fixture)."""
    gates: dict[str, Any] = {name: {"passed": True} for name in GATE_EVIDENCE_REQUIRED_GATES}
    gates["tests_pass"] = {"passed": True}
    gates["static_checks"] = {"passed": True}
    record: dict[str, Any] = {
        "study_class": STUDY_CLASS_DIAGNOSTIC,
        "verdict": SYNTHETIC_BENCHMARK_VALID,
        "research_status": SYNTHETIC_BENCHMARK_VALID,
        "synthetic_benchmark_valid": True,
        "research_valid": False,
        "tested_code_commit": "a" * 40,
        "artifact_generation_commit": "a" * 40,
        "provenance": {
            "artifact_generation_tree": "t" * 64,
            "protocol_version": PROTOCOL_VERSION,
            "study_version": STUDY_VERSION,
            "environment_lock_hash": "e" * 64,
            "workflow_run_id": "local",
            "workflow_attempt": "local",
            "certification_source": "local",
        },
        "gates": gates,
    }
    record.update(overrides)
    return record


def _evidence_context() -> tuple[dict[str, Any], dict[str, Any]]:
    """Bundle manifest + sidecar whose lineage matches the default gate."""
    manifest = _fixture_manifest(
        "fixture-release-000000000000",
        _generation_record(tested_code_commit="a" * 40, artifact_generation_commit="a" * 40),
    )
    return manifest, _sidecar_record(manifest)


def _evidence_findings(gate: dict[str, Any]) -> list[str]:
    manifest, sidecar = _evidence_context()
    return gate_evidence_findings(
        gate,
        sidecar=sidecar,
        bundle_manifest=manifest,
        study_class=STUDY_CLASS_DIAGNOSTIC,
    )


# --- valid evidence fixtures ------------------------------------------------


def test_valid_evidence_passing_synthetic_gate() -> None:
    assert _evidence_findings(_passing_evidence_gate()) == []


def test_valid_evidence_local_certification() -> None:
    gate = _passing_evidence_gate()
    assert gate["provenance"]["certification_source"] == "local"
    assert _evidence_findings(gate) == []


def test_valid_evidence_ci_certification() -> None:
    gate = _passing_evidence_gate()
    gate["provenance"].update(
        {"workflow_run_id": "12345", "workflow_attempt": "2", "certification_source": "ci"}
    )
    assert _evidence_findings(gate) == []


def test_valid_higher_tier_satisfies_lower_minimum() -> None:
    # A higher tier satisfies a lower minimum when the study class permits.
    gate = _passing_evidence_gate(
        study_class=STUDY_CLASS_EMPIRICAL_REPLAY,
        verdict=EMPIRICAL_REPLAY_VALID,
        research_status=EMPIRICAL_REPLAY_VALID,
        research_valid=True,
    )
    manifest, sidecar = _evidence_context()
    assert (
        gate_evidence_findings(
            gate,
            sidecar=sidecar,
            bundle_manifest=manifest,
            study_class=STUDY_CLASS_EMPIRICAL_REPLAY,
            required_status=SYNTHETIC_BENCHMARK_VALID,
        )
        == []
    )
    # Diagnostic studies are capped at synthetic: an empirical tier there
    # is stale or forged evidence, never a stronger certification.
    forged = _passing_evidence_gate(
        verdict=EMPIRICAL_REPLAY_VALID,
        research_status=EMPIRICAL_REPLAY_VALID,
        research_valid=True,
    )
    assert any(
        f.startswith("gate_evidence_status_insufficient") for f in _evidence_findings(forged)
    )


# --- invalid evidence fixtures: certification semantics ---------------------


def test_invalid_evidence_research_status_diagnostic() -> None:
    gate = _passing_evidence_gate(
        verdict="diagnostic", research_status="diagnostic", synthetic_benchmark_valid=False
    )
    assert any(f.startswith("gate_evidence_status_insufficient") for f in _evidence_findings(gate))


def test_invalid_evidence_synthetic_flag_false() -> None:
    gate = _passing_evidence_gate(synthetic_benchmark_valid=False)
    assert any(f.startswith("gate_evidence_status_insufficient") for f in _evidence_findings(gate))


def test_invalid_evidence_tests_failed() -> None:
    gate = _passing_evidence_gate()
    gate["gates"]["tests_pass"] = {"passed": False}
    assert any("gate_evidence_tests_not_passed" in f for f in _evidence_findings(gate))


def test_invalid_evidence_tests_not_run() -> None:
    gate = _passing_evidence_gate()
    gate["gates"]["tests_pass"] = {"not_run": True}
    assert any(
        "gate_evidence_tests_not_passed: tests_pass=not_run" in f for f in _evidence_findings(gate)
    )


def test_invalid_evidence_static_checks_failed() -> None:
    gate = _passing_evidence_gate()
    gate["gates"]["static_checks"] = {"passed": False}
    assert any("gate_evidence_static_checks_not_passed" in f for f in _evidence_findings(gate))


def test_invalid_evidence_substantive_gate_failed() -> None:
    gate = _passing_evidence_gate()
    gate["gates"]["corpus_valid"] = {"passed": False}
    assert "gate_evidence_substantive_gate_failed:corpus_valid" in _evidence_findings(gate)


def test_invalid_evidence_tested_commit_mismatch() -> None:
    gate = _passing_evidence_gate(tested_code_commit="b" * 40)
    assert any("gate_evidence_tested_commit_mismatch" in f for f in _evidence_findings(gate))


def test_invalid_evidence_generation_commit_mismatch() -> None:
    gate = _passing_evidence_gate(artifact_generation_commit="b" * 40)
    assert any("gate_evidence_generation_commit_mismatch" in f for f in _evidence_findings(gate))


def test_invalid_evidence_generation_tree_mismatch() -> None:
    gate = _passing_evidence_gate()
    gate["provenance"]["artifact_generation_tree"] = "u" * 64
    assert any("gate_evidence_generation_tree_mismatch" in f for f in _evidence_findings(gate))


def test_invalid_evidence_protocol_mismatch() -> None:
    gate = _passing_evidence_gate()
    gate["provenance"]["protocol_version"] = "0.0.1"
    assert any("gate_evidence_protocol_version_mismatch" in f for f in _evidence_findings(gate))


def test_invalid_evidence_study_version_mismatch() -> None:
    gate = _passing_evidence_gate()
    gate["provenance"]["study_version"] = "9.9.9"
    assert any("gate_evidence_study_version_mismatch" in f for f in _evidence_findings(gate))


def test_invalid_evidence_environment_lock_mismatch() -> None:
    gate = _passing_evidence_gate()
    gate["provenance"]["environment_lock_hash"] = "f" * 64
    assert any("gate_evidence_environment_lock_mismatch" in f for f in _evidence_findings(gate))


def test_invalid_evidence_from_another_release() -> None:
    gate = _passing_evidence_gate(release_id="some-other-release")
    assert any("gate_evidence_release_id_mismatch" in f for f in _evidence_findings(gate))


def test_invalid_evidence_scientific_digest_mismatch() -> None:
    gate = _passing_evidence_gate(scientific_release_digest="d" * 64)
    assert any("gate_evidence_scientific_digest_mismatch" in f for f in _evidence_findings(gate))


# --- invalid evidence fixtures: git-level failure modes ---------------------


@pytest.fixture(scope="module")
def detached_commit() -> str:
    """GE-012: a real commit that is NOT an ancestor of HEAD.

    ``commit-tree`` reuses HEAD's tree (so the gate file exists there with
    identical bytes) under a new parentless commit, which is reachable by
    sha but never an ancestor of the review commit.
    """
    return _git("commit-tree", "HEAD^{tree}", "-m", "GE-012 fixture: detached gate-evidence commit")


def test_invalid_evidence_missing_gate_file(commits: dict[str, str]) -> None:
    # The root commit predates the gate result file entirely.
    with pytest.raises(GateEvidenceError) as excinfo:
        load_gate_evidence_at_commit(commits["root"])
    assert excinfo.value.code == "gate_evidence_file_missing"


def test_invalid_evidence_malformed_json(
    commits: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(artifact_provenance, "gate_evidence_bytes", lambda commit: b"{oops")
    with pytest.raises(GateEvidenceError) as excinfo:
        load_gate_evidence_at_commit(commits["head"])
    assert excinfo.value.code == "gate_evidence_json_invalid"
    # Valid JSON that is not an object never certifies either.
    monkeypatch.setattr(artifact_provenance, "gate_evidence_bytes", lambda commit: b"[1, 2]")
    with pytest.raises(GateEvidenceError) as excinfo:
        load_gate_evidence_at_commit(commits["head"])
    assert excinfo.value.code == "gate_evidence_json_invalid"


def test_invalid_evidence_wrong_digest(fixture_releases: Path, commits: dict[str, str]) -> None:
    _write_fixture_release(
        fixture_releases,
        commits,
        snapshot=_PASSING_EVIDENCE_COMMIT,
        sidecar_overrides={"gate_evidence_sha256": "f" * 64},
    )
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("gate_evidence_digest_mismatch" in str(f) for f in result["findings"])


def test_invalid_evidence_failed_historical_gate_rejected(
    fixture_releases: Path, commits: dict[str, str]
) -> None:
    # 77d53d2 is a real FAILED gate: referencing it must never certify.
    _write_fixture_release(
        fixture_releases,
        commits,
        snapshot=_FAILED_EVIDENCE_COMMIT,
        sidecar_overrides={"gate_evidence_sha256": gate_evidence_sha256(_FAILED_EVIDENCE_COMMIT)},
    )
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("gate_evidence_status_insufficient" in str(f) for f in result["findings"])
    assert any("gate_evidence_tests_not_passed" in str(f) for f in result["findings"])


def test_invalid_evidence_not_ancestor_of_current(
    fixture_releases: Path, commits: dict[str, str], detached_commit: str
) -> None:
    _write_fixture_release(
        fixture_releases,
        commits,
        generation=commits["older"],
        storage=commits["older"],
        snapshot=detached_commit,
        sidecar_overrides={"gate_evidence_sha256": gate_evidence_sha256(detached_commit)},
    )
    result = research_valid_gate.check_release_storage_provenance()
    assert not result["passed"]
    assert any("gate_snapshot_not_ancestor_of_review_commit" in str(f) for f in result["findings"])


def test_invalid_sidecar_missing_gate_evidence_digest(commits: dict[str, str]) -> None:
    manifest = _fixture_manifest(
        "rel",
        _generation_record(
            tested_code_commit=commits["head"], artifact_generation_commit=commits["head"]
        ),
    )
    sidecar = _sidecar_record(manifest)
    assert "storage_provenance_field_empty: gate_evidence_sha256" in storage_provenance_findings(
        sidecar, require_gate_evidence=True
    )
    assert (
        "storage_provenance_field_empty: gate_evidence_sha256"
        not in storage_provenance_findings(sidecar, require_gate_evidence=False)
    )


# ---------------------------------------------------------------------------
# Storage sidecar schema 1.2 (GE-013)
# ---------------------------------------------------------------------------


def test_write_storage_provenance_emits_schema_12(tmp_path: Path, commits: dict[str, str]) -> None:
    """GE-013: freshly written sidecars carry evidence-binding schema 1.2."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest = _fixture_manifest(
        "schema-release",
        _generation_record(
            tested_code_commit=commits["head"],
            artifact_generation_commit=commits["head"],
        ),
    )
    manifest["release_digest"] = release_digest(manifest)
    (bundle_dir / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    write_storage_provenance(
        bundle_dir,
        artifact_storage_commit=commits["head"],
        gate_evidence_commit=_PASSING_EVIDENCE_COMMIT,
        verified_at="t1",
    )
    sidecar = json.loads((bundle_dir / STORAGE_PROVENANCE_NAME).read_text())
    assert sidecar["schema_version"] == "1.2"
    assert sidecar["gate_evidence_commit"] == _PASSING_EVIDENCE_COMMIT
    # GE-001: the deprecated alias stays identical to the primary field.
    assert sidecar["gate_snapshot_commit"] == _PASSING_EVIDENCE_COMMIT
    # GE-002: the digest binds the exact historical gate bytes.
    assert sidecar["gate_evidence_sha256"] == gate_evidence_sha256(_PASSING_EVIDENCE_COMMIT)
    assert (
        storage_provenance_findings(sidecar, require_gate_snapshot=True, require_gate_evidence=True)
        == []
    )


def test_schema_11_sidecar_remains_readable(commits: dict[str, str]) -> None:
    """GE-013: archived 1.1 sidecars stay auditable under their schema.

    A 1.1 sidecar that records only the deprecated ``gate_snapshot_commit``
    alias is still complete; it is never rewritten into 1.2 shape.
    """
    manifest = _fixture_manifest(
        "rel",
        _generation_record(
            tested_code_commit=commits["head"], artifact_generation_commit=commits["head"]
        ),
    )
    sidecar = _sidecar_record(manifest)
    assert sidecar["schema_version"] == "1.1"
    assert storage_provenance_findings(sidecar, require_gate_snapshot=True) == []
    assert gate_evidence_commit_of(sidecar) == sidecar["gate_snapshot_commit"]


# ---------------------------------------------------------------------------
# Final storage certification record (GE-015)
# ---------------------------------------------------------------------------


def test_final_storage_certification_record(
    fixture_releases: Path, commits: dict[str, str]
) -> None:
    """GE-015: durable certification metadata, excluded from the digest."""
    bundle_dir = _write_fixture_release(
        fixture_releases,
        commits,
        snapshot=_PASSING_EVIDENCE_COMMIT,
        sidecar_overrides={"gate_evidence_sha256": gate_evidence_sha256(_PASSING_EVIDENCE_COMMIT)},
    )
    scientific = release_digest(json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text()))

    record = write_final_storage_certification(
        bundle_dir, passed=True, verified_at="2026-01-01T00:00:00+00:00"
    )
    sidecar = json.loads((bundle_dir / STORAGE_PROVENANCE_NAME).read_text())
    stored = json.loads((bundle_dir / FINAL_STORAGE_CERTIFICATION_NAME).read_text())
    assert stored == record
    assert record["schema_version"] == "1.0"
    assert record["release_id"] == sidecar["release_id"]
    assert record["artifact_storage_commit"] == sidecar["artifact_storage_commit"]
    assert record["gate_evidence_commit"] == _PASSING_EVIDENCE_COMMIT
    assert record["gate_evidence_sha256"] == sidecar["gate_evidence_sha256"]
    assert record["storage_metadata_digest"] == sidecar["storage_metadata_digest"]
    assert record["passed"] is True

    # Storage/certification metadata never enters the scientific digest.
    assert release_digest(json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())) == scientific

"""FP-011: live-release provenance integration test.

Loads the REAL active release from ``results/releases/`` and asserts the
selected provenance model end-to-end: generation identity synchronized
across every scientific manifest, storage identity recorded only in the
STORAGE_PROVENANCE.json sidecar and echoed at the bundle-manifest top
level, git ancestry generation -> storage -> gate snapshot, the exact
bundle bytes at the storage commit, and the two-digest model.

The test fails on the inconsistent pre-patch state and passes only after
every active artifact follows the model (FP-014 regeneration).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from experiments.trustparadox_u.artifact_provenance import (
    COMMIT_RE,
    commit_is_ancestor,
    generation_provenance_findings,
    storage_provenance_findings,
    validate_release_provenance_consistency,
    validate_storage_record_consistency,
    validate_storage_reference,
)
from experiments.trustparadox_u.release_bundle import (
    BUNDLE_MANIFEST_NAME,
    STORAGE_PROVENANCE_NAME,
    release_digest,
    release_dirs,
    scientific_release_digest,
    storage_metadata_digest,
    validate_bundle_at_storage_commit,
)
from experiments.trustparadox_u.research_valid_gate import (
    _PROJECT_ROOT,
    FINAL_DIR,
    REPLAY_DIR,
    RESULTS_DIR,
)

# FP-011 acceptance: the FP-014 provenance-only regeneration has landed;
# the live release now follows the FP-001..FP-010 model end-to-end.

_GATE_RESULT_REL = str((FINAL_DIR / "research_valid_gate.json").relative_to(_PROJECT_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _git_returncode(*args: str) -> int:
    return subprocess.run(["git", *args], capture_output=True, cwd=_PROJECT_ROOT).returncode


@pytest.fixture(scope="module")
def active_release() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """The unique active release, its bundle manifest and its sidecar."""
    active: list[Path] = []
    for bundle_dir in release_dirs():
        manifest = _load_json(bundle_dir / BUNDLE_MANIFEST_NAME)
        if manifest.get("status") == "active":
            active.append(bundle_dir)
    assert (
        len(active) == 1
    ), f"expected exactly one active release, found {[d.name for d in active]}"
    bundle_dir = active[0]
    manifest = _load_json(bundle_dir / BUNDLE_MANIFEST_NAME)
    sidecar_path = bundle_dir / STORAGE_PROVENANCE_NAME
    assert sidecar_path.exists(), f"storage sidecar missing: {sidecar_path}"
    return bundle_dir, manifest, _load_json(sidecar_path)


@pytest.fixture(scope="module")
def scientific_manifests() -> dict[str, dict[str, Any]]:
    paths = {
        "study_manifest": FINAL_DIR / "study_manifest.json",
        "reproduction_manifest": RESULTS_DIR / "reproduction" / "reproduction_manifest.json",
        "run_manifest": REPLAY_DIR / "run_manifest.json",
        "frozen_threshold_manifest": RESULTS_DIR
        / "frozen_config"
        / "frozen_threshold_manifest.json",
        "gate_snapshot": FINAL_DIR / "research_valid_gate.json",
    }
    return {label: _load_json(path) for label, path in paths.items() if path.exists()}


def test_exactly_one_active_release(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    bundle_dir, manifest, sidecar = active_release
    assert manifest.get("status") == "active"
    assert str(manifest.get("release_id")) == str(sidecar.get("release_id")) == bundle_dir.name


def test_sidecar_schema_complete(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    _, _, sidecar = active_release
    assert storage_provenance_findings(sidecar, require_gate_snapshot=True) == []


def test_generation_consistency_across_manifests(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
    scientific_manifests: dict[str, dict[str, Any]],
) -> None:
    _, manifest, _ = active_release
    manifests = {**scientific_manifests, "bundle_manifest": manifest}
    expected = {
        "bundle_manifest",
        "frozen_threshold_manifest",
        "gate_snapshot",
        "reproduction_manifest",
        "run_manifest",
        "study_manifest",
    }
    missing = expected - set(manifests)
    assert not missing, f"missing scientific manifests: {sorted(missing)}"
    assert validate_release_provenance_consistency(manifests) == []
    for label, data in manifests.items():
        record = data.get("provenance")
        record = record if isinstance(record, dict) else data
        assert generation_provenance_findings(record) == [], f"{label} generation record invalid"
        assert validate_storage_reference(record) == [], f"{label} sidecar reference invalid"


def test_storage_consistency_manifest_sidecar(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    _, manifest, sidecar = active_release
    assert (
        validate_storage_record_consistency(
            {"bundle_manifest": manifest, "storage_sidecar": sidecar}
        )
        == []
    )
    storage = str(sidecar.get("artifact_storage_commit", "") or "")
    assert storage and COMMIT_RE.match(storage)
    assert str(manifest.get("artifact_storage_commit") or "") == storage


def test_gate_snapshot_commit_contains_gate_file(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    _, _, sidecar = active_release
    snapshot = str(sidecar.get("gate_snapshot_commit", "") or "")
    assert snapshot and COMMIT_RE.match(snapshot)
    assert (
        _git_returncode("show", f"{snapshot}:{_GATE_RESULT_REL}") == 0
    ), f"gate result file missing at gate_snapshot_commit {snapshot}"


def test_ancestry_generation_storage_gate_snapshot(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    _, _, sidecar = active_release
    generation = str(sidecar.get("artifact_generation_commit", "") or "")
    storage = str(sidecar.get("artifact_storage_commit", "") or "")
    snapshot = str(sidecar.get("gate_snapshot_commit", "") or "")
    assert commit_is_ancestor(
        generation, storage
    ), f"generation {generation} must be an ancestor of storage {storage}"
    assert commit_is_ancestor(
        storage, snapshot
    ), f"storage {storage} must be an ancestor of gate snapshot {snapshot}"


def test_bundle_bytes_at_storage_commit(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    bundle_dir, manifest, _ = active_release
    assert validate_bundle_at_storage_commit(bundle_dir, manifest) == []


def test_digest_stability(active_release: tuple[Path, dict[str, Any], dict[str, Any]]) -> None:
    _, manifest, sidecar = active_release
    assert release_digest(manifest) == scientific_release_digest(manifest)
    assert str(sidecar.get("scientific_release_digest", "")) == release_digest(manifest)
    assert str(sidecar.get("storage_metadata_digest", "")) == storage_metadata_digest(sidecar)


def test_local_certification_labeling(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    _, manifest, _ = active_release
    provenance = manifest.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    assert provenance.get("workflow_run_id") == "local"
    assert provenance.get("workflow_attempt") == "local"
    assert provenance.get("certification_source") == "local"

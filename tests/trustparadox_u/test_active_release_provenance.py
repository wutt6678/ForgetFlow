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
    GATE_EVIDENCE_REQUIRED_GATES,
    commit_is_ancestor,
    gate_evidence_commit_of,
    gate_evidence_findings,
    generation_provenance_findings,
    load_gate_evidence_at_commit,
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
    _study_class_from_artifacts,
)

# FP-011 acceptance: the FP-014 provenance-only regeneration has landed;
# the live release now follows the FP-001..FP-010 model end-to-end.


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


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


def test_gate_evidence_certifies_release(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    """GE-011: the referenced historical gate must actually certify the release.

    Existence of the gate file at the referenced commit is never enough:
    the exact historical bytes must match the sidecar's GE-002 digest,
    and the recorded gate must have passed with a sufficient research
    tier, passing test/static evidence, every required substantive gate,
    and the exact generation lineage of this release.  The test fails on
    a failed historical gate reference and passes only when the
    referenced gate genuinely certifies the release.
    """
    _, manifest, sidecar = active_release
    evidence = gate_evidence_commit_of(sidecar)
    assert evidence and COMMIT_RE.match(evidence)
    loaded = load_gate_evidence_at_commit(evidence)
    gate = loaded["record"]

    # GE-002: the sidecar digest binds the exact historical bytes.
    assert str(sidecar.get("gate_evidence_sha256", "") or "") == loaded["sha256"]

    # GE-004/GE-005: certification semantics.
    assert gate["synthetic_benchmark_valid"] is True
    assert gate["research_valid"] is False
    assert gate["research_status"] == "synthetic_benchmark_valid"
    assert gate["gates"]["tests_pass"]["passed"] is True
    assert gate["gates"]["static_checks"]["passed"] is True

    # GE-006: every required substantive gate passed in the evidence.
    for name in GATE_EVIDENCE_REQUIRED_GATES:
        entry = gate["gates"].get(name)
        assert isinstance(entry, dict) and entry.get("passed") is True, f"evidence gate {name}"

    # GE-007: release identity — the evidence certifies THIS release.
    assert gate["tested_code_commit"] == sidecar["tested_code_commit"]
    assert gate["artifact_generation_commit"] == sidecar["artifact_generation_commit"]
    assert (
        gate_evidence_findings(
            gate,
            sidecar=sidecar,
            bundle_manifest=manifest,
            study_class=_study_class_from_artifacts(),
        )
        == []
    )

    # Ancestry: generation -> storage -> gate evidence -> review commit.
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_PROJECT_ROOT
    ).stdout.strip()
    assert commit_is_ancestor(evidence, current)


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


# --- E2-A7-FIX-034: Synthetic provenance-chain regression test ---


def test_fix034_synthetic_provenance_chain_coherent(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    """E2-A7-FIX-034: Validate a single coherent immutable synthetic certification chain.

    Loads bundle_manifest.json, STORAGE_PROVENANCE.json, and
    FINAL_STORAGE_CERTIFICATION.json and validates:
    - release_id equality
    - storage commit consistency
    - gate evidence consistency
    - metadata digest consistency
    - scientific digest preservation
    - timestamp ordering (FIX-025)
    """
    from datetime import datetime

    from experiments.trustparadox_u.release_bundle import FINAL_STORAGE_CERTIFICATION_NAME

    bundle_dir, manifest, sidecar = active_release

    # Load FINAL_STORAGE_CERTIFICATION.json
    final_cert_path = bundle_dir / FINAL_STORAGE_CERTIFICATION_NAME
    assert final_cert_path.exists(), f"Missing {FINAL_STORAGE_CERTIFICATION_NAME}"
    final_cert = json.loads(final_cert_path.read_text())

    # release_id equality
    assert (
        manifest["release_id"] == sidecar["release_id"] == final_cert["release_id"]
    ), "release_id must be equal across all three files"

    # artifact_storage_commit consistency
    assert (
        manifest["artifact_storage_commit"]
        == sidecar["artifact_storage_commit"]
        == final_cert["artifact_storage_commit"]
    ), "artifact_storage_commit must be equal across all three files"

    # gate_evidence_commit consistency
    assert (
        sidecar["gate_evidence_commit"] == final_cert["gate_evidence_commit"]
    ), "gate_evidence_commit must match between sidecar and final certification"

    # gate_evidence_sha256 consistency
    assert (
        sidecar["gate_evidence_sha256"] == final_cert["gate_evidence_sha256"]
    ), "gate_evidence_sha256 must match between sidecar and final certification"

    # storage_metadata_digest consistency
    assert (
        manifest["storage_metadata_digest"]
        == sidecar["storage_metadata_digest"]
        == final_cert["storage_metadata_digest"]
    ), "storage_metadata_digest must be equal across all three files"

    # scientific_release_digest preservation
    assert (
        manifest["scientific_release_digest"] == sidecar["scientific_release_digest"]
    ), "scientific_release_digest must match between manifest and sidecar"

    # Timestamp ordering (FIX-025): verified_at >= created_at
    verified_at = datetime.fromisoformat(final_cert["verified_at"].replace("+00:00", "+00:00"))
    created_at = datetime.fromisoformat(manifest["created_at"].replace("+00:00", "+00:00"))
    assert (
        verified_at >= created_at
    ), f"verified_at ({verified_at}) must be >= created_at ({created_at})"

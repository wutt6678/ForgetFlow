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


# --- FIX-030: Synthetic timestamp regression tests ---


def _parse_iso(ts: str) -> Any:
    from datetime import datetime

    return datetime.fromisoformat(ts.replace("+00:00", "+00:00"))


def test_fix030_sidecar_verified_at_not_before_evidence_commit(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    """FIX-030: STORAGE_PROVENANCE.verified_at must be >= gate evidence commit time.

    A stale sidecar timestamp that predates the evidence it references
    must fail this regression check.
    """
    _, _, sidecar = active_release
    evidence_commit = str(sidecar.get("gate_evidence_commit", "") or "")
    assert evidence_commit and COMMIT_RE.match(evidence_commit)

    result = subprocess.run(
        ["git", "log", "-1", "--format=%cI", evidence_commit],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, f"git log failed for {evidence_commit}"
    commit_ts = _parse_iso(result.stdout.strip())
    verified_ts = _parse_iso(str(sidecar.get("verified_at", "")))
    assert (
        verified_ts >= commit_ts
    ), f"sidecar verified_at ({verified_ts}) must be >= evidence commit ({commit_ts})"


def test_fix030_final_cert_verified_at_not_before_evidence_commit(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    """FIX-030: FINAL_STORAGE_CERTIFICATION.verified_at >= evidence commit time."""
    from experiments.trustparadox_u.release_bundle import FINAL_STORAGE_CERTIFICATION_NAME

    bundle_dir, _, sidecar = active_release
    final_cert = json.loads((bundle_dir / FINAL_STORAGE_CERTIFICATION_NAME).read_text())

    evidence_commit = str(sidecar.get("gate_evidence_commit", "") or "")
    assert evidence_commit and COMMIT_RE.match(evidence_commit)

    result = subprocess.run(
        ["git", "log", "-1", "--format=%cI", evidence_commit],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0
    commit_ts = _parse_iso(result.stdout.strip())
    verified_ts = _parse_iso(str(final_cert.get("verified_at", "")))
    assert (
        verified_ts >= commit_ts
    ), f"final cert verified_at ({verified_ts}) must be >= evidence commit ({commit_ts})"


def test_fix030_sidecar_timestamp_is_reasonable(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    """FIX-030: STORAGE_PROVENANCE.verified_at must be a reasonable timestamp."""
    _, _, sidecar = active_release
    verified_ts = _parse_iso(str(sidecar.get("verified_at", "")))
    # verified_at may predate generated_at only if the sidecar was created
    # before the bundle; in practice verified_at >= generated_at.
    # The critical invariant is verified_at >= evidence commit (tested above).
    # Here we ensure the sidecar timestamp is at least a valid ISO timestamp
    # and not absurdly early (before 2026).
    from datetime import datetime, timezone

    assert verified_ts >= datetime(
        2026, 1, 1, tzinfo=timezone.utc
    ), f"sidecar verified_at ({verified_ts}) is unreasonably early"


# --- FIX-032: Reproduction-manifest self-consistency test ---


def test_fix032_reproduction_manifest_resolved_conditions_self_consistent() -> None:
    """FIX-032: reproduction_manifest resolved_conditions hash must be self-consistent.

    Validates:
    1. verification.resolved_conditions_sha256 == artifacts['frozen_replay/resolved_conditions.json']
    2. Both equal sha256 of the actual file bytes on disk.
    """
    import hashlib

    manifest_path = RESULTS_DIR / "reproduction" / "reproduction_manifest.json"
    assert manifest_path.exists(), f"reproduction manifest missing: {manifest_path}"
    manifest = json.loads(manifest_path.read_text())

    verification_hash = str(manifest.get("verification", {}).get("resolved_conditions_sha256", ""))
    artifacts_hash = str(
        manifest.get("artifacts", {}).get("frozen_replay/resolved_conditions.json", "")
    )

    # Internal consistency: the two manifest fields must agree.
    assert verification_hash, "verification.resolved_conditions_sha256 is empty"
    assert artifacts_hash, "artifacts['frozen_replay/resolved_conditions.json'] is empty"
    assert (
        verification_hash == artifacts_hash
    ), f"manifest internal mismatch: verification={verification_hash} artifacts={artifacts_hash}"

    # External consistency: both must match the actual file bytes.
    resolved_path = REPLAY_DIR / "resolved_conditions.json"
    assert resolved_path.exists(), f"resolved_conditions.json missing: {resolved_path}"
    actual_hash = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    assert (
        actual_hash == verification_hash
    ), f"file hash {actual_hash} != manifest verification hash {verification_hash}"


# --- FIX-034: Historical synthetic outputs immutability ---


def test_fix034_tables_hashes_unchanged(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    """FIX-034: Tables 1-6 hashes in reproduction manifest match the release bundle.

    The scientific release digest and table hashes must remain identical
    to the values recorded in the bundle manifest (frozen at storage commit).
    """
    _, manifest, _ = active_release
    manifest_path = RESULTS_DIR / "reproduction" / "reproduction_manifest.json"
    repro = json.loads(manifest_path.read_text())

    bundle_components = manifest.get("components", {})
    repro_artifacts = repro.get("artifacts", {})

    table_keys = [
        "final_artifacts/table1_main_results.json",
        "final_artifacts/table2_leakage_breakdown.json",
        "final_artifacts/table3_parameter_sensitivity.json",
        "final_artifacts/table4_statistical_comparisons.json",
        "final_artifacts/table5_target_type_results.json",
        "final_artifacts/table6_trust_analysis.json",
    ]
    for key in table_keys:
        bundle_hash = str(bundle_components.get(key, {}).get("sha256", ""))
        repro_hash = str(repro_artifacts.get(key, ""))
        assert bundle_hash and repro_hash, f"missing hash for {key}"
        assert (
            bundle_hash == repro_hash
        ), f"{key}: bundle={bundle_hash} != reproduction={repro_hash}"


def test_fix034_scientific_release_digest_unchanged(
    active_release: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    """FIX-034: scientific_release_digest in the bundle is stable."""
    _, manifest, sidecar = active_release
    bundle_digest = str(manifest.get("scientific_release_digest", ""))
    sidecar_digest = str(sidecar.get("scientific_release_digest", ""))
    assert bundle_digest and sidecar_digest
    assert (
        bundle_digest == sidecar_digest
    ), f"scientific_release_digest mismatch: bundle={bundle_digest} sidecar={sidecar_digest}"
    # Also matches release_digest
    assert str(manifest.get("release_digest", "")) == bundle_digest

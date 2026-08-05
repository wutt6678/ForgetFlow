"""PR-007: validation rules for release provenance and storage lineage.

Covers the eleven acceptance cases: valid ancestry; storage commit
preceding generation; bundle absent at the storage commit; checksum
mismatch at the storage commit; empty top-level protocol version;
top-level/nested version mismatch; empty storage commit on an active
release; gate/bundle generation-commit disagreement; sidecar updates
that preserve the scientific digest; local certification source; and
numeric CI workflow evidence.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u import release_bundle  # noqa: E402
from experiments.trustparadox_u.artifact_provenance import (  # noqa: E402
    storage_commit_for,
    validate_release_lineage,
    validate_release_provenance_consistency,
)
from experiments.trustparadox_u.manifest import get_repository_commit  # noqa: E402
from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION  # noqa: E402

_HEAD = get_repository_commit().removesuffix("-dirty")
_COMMIT_A = "a" * 40
_COMMIT_B = "b" * 40


def _root_commit() -> str:
    result = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    return result.stdout.split()[0]


def _lineage_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "tested_code_commit": _HEAD,
        "artifact_generation_commit": _HEAD,
        "artifact_storage_commit": _HEAD,
        "protocol_version": PROTOCOL_VERSION,
        "study_version": "1.2.1",
        "artifact_generation_tree": "t" * 64,
        "environment_lock_hash": "e" * 64,
        "workflow_run_id": "local",
        "workflow_attempt": "local",
        "certification_source": "local",
    }
    record.update(overrides)
    return record


def _schema11_manifest(
    *,
    protocol_top: str = PROTOCOL_VERSION,
    protocol_provenance: str = PROTOCOL_VERSION,
    storage_commit: str = _COMMIT_A,
) -> dict[str, Any]:
    provenance = {
        "tested_code_commit": _COMMIT_A,
        "artifact_generation_commit": _COMMIT_A,
        "artifact_generation_tree": "t" * 64,
        "environment_lock_hash": "e" * 64,
        "artifact_storage_commit": storage_commit,
        "protocol_version": protocol_provenance,
        "study_version": "1.2.1",
        "workflow_run_id": "local",
        "workflow_attempt": "local",
        "certification_source": "local",
    }
    manifest: dict[str, Any] = {
        "schema_version": "1.1",
        "study_version": "1.2.1",
        "protocol_version": protocol_top,
        "provenance": provenance,
        "corpus": {"corpus_sha256": "c"},
        "annotations": {"annotation_hash": "a"},
        "components": {},
        # FP-001: storage identity is a top-level field kept in sync with
        # the provenance embedding this fixture still exercises.
        "artifact_storage_commit": storage_commit or None,
        "status": "active",
        "superseded_by": "",
    }
    digest = release_bundle.release_digest(manifest)
    manifest["release_id"] = f"trustparadox_u-v{manifest['study_version']}-{digest[:12]}"
    manifest["release_digest"] = digest
    manifest["scientific_release_digest"] = digest
    manifest["storage_metadata_digest"] = ""
    return manifest


def _write_bundle(tmp_path: Path, manifest: dict[str, Any]) -> Path:
    bundle = tmp_path / "releases" / manifest["release_id"]
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / release_bundle.BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return bundle


class TestLineageAncestryRules:
    """PR-006 rules 1-2: ancestry and tested/generation agreement."""

    def test_valid_ancestry_yields_no_findings(self) -> None:
        assert validate_release_lineage(_lineage_record()) == []

    def test_storage_before_generation_is_rejected(self) -> None:
        record = _lineage_record(
            tested_code_commit=_HEAD,
            artifact_generation_commit=_HEAD,
            artifact_storage_commit=_root_commit(),
        )
        findings = validate_release_lineage(record)
        assert any(f.startswith("lineage_not_ancestor") for f in findings)

    def test_tested_generation_mismatch_requires_reason(self) -> None:
        record = _lineage_record(tested_code_commit=_COMMIT_B, artifact_generation_commit=_HEAD)
        findings = validate_release_lineage(record)
        assert any(f.startswith("tested_generation_mismatch_without_reason") for f in findings)
        record["difference_reason"] = "replay ran against the pre-repair checkout"
        assert not any(
            f.startswith("tested_generation_mismatch_without_reason")
            for f in validate_release_lineage(record)
        )


class TestBundleStorageRules:
    """PR-006 rules 3-4: the storage commit must contain the exact bundle."""

    def _active_bundle(self) -> Path:
        bundles = release_bundle.release_dirs()
        assert bundles, "no live release bundle present"
        return bundles[-1]

    def test_bundle_absent_at_storage_commit_is_rejected(self) -> None:
        bundle = self._active_bundle()
        findings = release_bundle.validate_bundle_at_storage_commit(
            bundle, storage_commit=_root_commit()
        )
        assert findings
        assert any("storage commit missing" in f for f in findings)

    def test_checksum_mismatch_at_storage_commit_is_rejected(self) -> None:
        bundle = self._active_bundle()
        manifest = json.loads((bundle / release_bundle.BUNDLE_MANIFEST_NAME).read_text())
        stored_at = storage_commit_for(bundle / release_bundle.BUNDLE_MANIFEST_NAME)
        assert stored_at, "live bundle must be committed"
        tampered = copy.deepcopy(manifest)
        component = next(iter(tampered["components"].values()))
        component["sha256"] = "0" * 64
        findings = release_bundle.validate_bundle_at_storage_commit(
            bundle, tampered, storage_commit=stored_at
        )
        assert any("storage commit checksum mismatch" in f for f in findings)


class TestSchema11BundleFields:
    """PR-003: top-level versions, nested agreement, complete lineage."""

    def test_empty_top_level_protocol_version_fails(self, tmp_path: Path) -> None:
        manifest = _schema11_manifest(protocol_top="")
        bundle = _write_bundle(tmp_path, manifest)
        findings = release_bundle.validate_release_bundle(bundle)
        assert any("empty top-level protocol_version" in f for f in findings)

    def test_top_level_nested_version_mismatch_fails(self, tmp_path: Path) -> None:
        manifest = _schema11_manifest(protocol_provenance="1.1.0")
        bundle = _write_bundle(tmp_path, manifest)
        findings = release_bundle.validate_release_bundle(bundle)
        assert any(
            "top-level protocol_version != provenance.protocol_version" in f for f in findings
        )

    def test_empty_storage_commit_on_active_release_fails(self, tmp_path: Path) -> None:
        manifest = _schema11_manifest(storage_commit="")
        bundle = _write_bundle(tmp_path, manifest)
        findings = release_bundle.validate_release_bundle(bundle)
        assert any("empty provenance.artifact_storage_commit" in f for f in findings)
        # Pending storage at build time is the one legitimate exception.
        assert not any(
            "empty provenance.artifact_storage_commit" in f
            for f in release_bundle.validate_release_bundle(bundle, allow_pending_storage=True)
        )

    def test_complete_schema11_bundle_validates(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path, _schema11_manifest())
        assert release_bundle.validate_release_bundle(bundle) == []


class TestProvenanceSynchronization:
    """PR-004: gate and bundle manifests must record identical lineage."""

    def _manifest_pair(self, gate_generation: str, bundle_generation: str) -> dict[str, Any]:
        def record(generation: str) -> dict[str, Any]:
            return {
                "tested_code_commit": generation,
                "artifact_generation_commit": generation,
                "artifact_generation_tree": "t" * 64,
                "artifact_storage_commit": _COMMIT_A,
                "protocol_version": PROTOCOL_VERSION,
                "study_version": "1.2.1",
                "environment_lock_hash": "e" * 64,
            }

        return {
            "research_valid_gate": {"provenance": record(gate_generation)},
            "bundle_manifest": {"provenance": record(bundle_generation)},
        }

    def test_gate_bundle_generation_mismatch_is_a_finding(self) -> None:
        findings = validate_release_provenance_consistency(
            self._manifest_pair(_COMMIT_A, _COMMIT_B)
        )
        assert any("provenance_sync_mismatch: artifact_generation_commit" in f for f in findings)

    def test_identical_lineage_is_consistent(self) -> None:
        assert (
            validate_release_provenance_consistency(self._manifest_pair(_COMMIT_A, _COMMIT_A)) == []
        )


class TestTwoDigestScheme:
    """PR-005: sidecar updates never fork the scientific digest."""

    def test_sidecar_update_preserves_scientific_digest(self, tmp_path: Path) -> None:
        manifest = _schema11_manifest(storage_commit="")
        bundle = _write_bundle(tmp_path, manifest)
        scientific = manifest["release_digest"]

        release_bundle.write_storage_provenance(
            bundle,
            artifact_storage_commit=_COMMIT_A,
            gate_snapshot_commit=_COMMIT_B,
            verified_at="2026-01-01T00:00:00+00:00",
        )
        first = json.loads((bundle / release_bundle.BUNDLE_MANIFEST_NAME).read_text())
        assert first["release_digest"] == scientific
        assert first["scientific_release_digest"] == scientific
        assert release_bundle.release_digest(first) == scientific
        first_sidecar_digest = first["storage_metadata_digest"]
        assert first_sidecar_digest

        # Re-auditing the same lineage must not change either digest.
        release_bundle.write_storage_provenance(
            bundle,
            artifact_storage_commit=_COMMIT_A,
            gate_snapshot_commit=_COMMIT_B,
            verified_at="2027-01-01T00:00:00+00:00",
        )
        second = json.loads((bundle / release_bundle.BUNDLE_MANIFEST_NAME).read_text())
        assert second["release_digest"] == scientific
        assert second["storage_metadata_digest"] == first_sidecar_digest

        # A genuinely different lineage does change the storage digest.
        release_bundle.write_storage_provenance(
            bundle,
            artifact_storage_commit=_COMMIT_B,
            gate_snapshot_commit=_COMMIT_B,
            verified_at="2027-01-01T00:00:00+00:00",
        )
        third = json.loads((bundle / release_bundle.BUNDLE_MANIFEST_NAME).read_text())
        assert third["release_digest"] == scientific
        assert third["storage_metadata_digest"] != first_sidecar_digest


class TestWorkflowIdentityRules:
    """PR-006 rule 7: numeric CI evidence vs. certified local runs."""

    def test_local_run_requires_local_certification_source(self) -> None:
        record = _lineage_record()
        record.pop("certification_source")
        findings = validate_release_lineage(record)
        assert findings == ["local_certification_source_missing"]
        assert validate_release_lineage(record, certification_source="local") == []

    def test_ci_run_requires_numeric_workflow_identity(self) -> None:
        record = _lineage_record(
            workflow_run_id="12345", workflow_attempt="2", certification_source="ci"
        )
        assert validate_release_lineage(record) == []
        record["workflow_run_id"] = "not-a-number"
        findings = validate_release_lineage(record)
        assert any(f.startswith("workflow_identity_invalid") for f in findings)

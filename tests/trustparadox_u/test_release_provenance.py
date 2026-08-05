"""Remediation §32/§33/§34: three-way provenance, reproduction, releases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u import release_bundle, reproduce  # noqa: E402
from experiments.trustparadox_u.artifact_provenance import (  # noqa: E402
    build_certification_provenance,
    commit_is_ancestor,
    environment_lock_hash,
    generation_tree_hash,
    storage_commit_for,
    validate_three_way_provenance,
)
from experiments.trustparadox_u.manifest import get_repository_commit  # noqa: E402

_TEST_COMMIT = "a" * 40


class TestThreeWayProvenance:
    """§32: workspace and environment anchors beyond commit names."""

    def test_generation_tree_hash_is_stable(self) -> None:
        assert generation_tree_hash() == generation_tree_hash()
        assert len(generation_tree_hash()) == 64

    def test_environment_lock_hash_matches_poetry_lock(self) -> None:
        import hashlib

        lock = _PROJECT_ROOT / "poetry.lock"
        expected = hashlib.sha256(lock.read_bytes()).hexdigest()
        assert environment_lock_hash() == expected

    def test_certification_provenance_carries_three_way_fields(self) -> None:
        record = build_certification_provenance(repository_clean=True)
        assert record["artifact_generation_tree"] == generation_tree_hash()
        assert record["environment_lock_hash"] == environment_lock_hash()
        # FP-001: generation records never embed a storage commit — null
        # plus a pointer at the authoritative STORAGE_PROVENANCE.json.
        assert record["artifact_storage_commit"] is None
        assert record["storage_provenance"] == {
            "source": "STORAGE_PROVENANCE.json",
            "authoritative": True,
        }

    def test_storage_commit_for_untracked_path_is_empty(self, tmp_path: Path) -> None:
        # A file outside the repository has no storage commit.
        stray = tmp_path / "stray.json"
        stray.write_text("{}")
        assert storage_commit_for(stray) == ""

    def test_commit_is_ancestor_of_itself_and_head(self) -> None:
        head = get_repository_commit().removesuffix("-dirty")
        assert commit_is_ancestor(head, head) is True
        assert commit_is_ancestor("not-a-commit", head) is False

    def test_three_way_validation_requires_workspace_fields(self) -> None:
        record = {
            "tested_code_commit": _TEST_COMMIT,
            "artifact_generation_commit": _TEST_COMMIT,
            "repository_clean": True,
            "workflow_run_id": "",
            "workflow_attempt": "",
        }
        findings = validate_three_way_provenance(record, current_commit=_TEST_COMMIT)
        assert any("missing_three_way_field: artifact_generation_tree" in f for f in findings)
        assert any("missing_three_way_field: environment_lock_hash" in f for f in findings)

    def test_three_way_validation_passes_for_complete_record(self) -> None:
        record = build_certification_provenance(repository_clean=True)
        head = record["tested_code_commit"]
        findings = validate_three_way_provenance(record, current_commit=head)
        assert findings == []


class TestReproductionFailFast:
    """§33: missing or mismatched frozen inputs abort the reproduction."""

    def test_invalid_corpus_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import experiments.trustparadox_u.research_valid_gate as gate

        monkeypatch.setattr(
            gate, "check_corpus_valid", lambda: {"passed": False, "reason": "tampered"}
        )
        with pytest.raises(reproduce.ReproductionError, match="corpus invalid"):
            reproduce.validate_frozen_inputs()

    def test_invalid_annotations_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import experiments.trustparadox_u.research_valid_gate as gate

        monkeypatch.setattr(
            gate, "check_annotations_valid", lambda: {"passed": False, "reason": "tampered"}
        )
        with pytest.raises(reproduce.ReproductionError, match="annotations invalid"):
            reproduce.validate_frozen_inputs()

    def test_frozen_inputs_validate_on_real_data(self) -> None:
        inputs = reproduce.validate_frozen_inputs()
        assert inputs["corpus"]["passed"] is True
        assert inputs["corpus"]["corpus_sha256"]
        assert inputs["annotations"]["annotation_hash"]
        assert inputs["study_version"]

    def test_never_imports_corpus_generators(self) -> None:
        # §33 acceptance: reproduction must not regenerate corpus/annotations.
        source = (Path(reproduce.__file__)).read_text()
        assert "generate_corpus.main" not in source
        assert "annotate_corpus.main" not in source
        assert "run_generate_corpus" not in source

    def test_failed_pipeline_step_raises(self) -> None:
        with pytest.raises(reproduce.ReproductionError, match="pipeline step failed"):
            reproduce.run_pipeline_step(
                "experiments.trustparadox_u.definitely_missing_module", "must fail"
            )


class TestReleaseBundleBuild:
    """§34: unique immutable identifiers and supersession."""

    def test_release_requires_reproduction_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(release_bundle, "RESULTS_DIR", tmp_path)
        with pytest.raises(release_bundle.ReleaseError, match="reproduction manifest missing"):
            release_bundle.build_release_manifest()

    def test_release_requires_passing_reproduction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repro_dir = tmp_path / "reproduction"
        repro_dir.mkdir()
        (repro_dir / "reproduction_manifest.json").write_text(json.dumps({"passed": False}))
        monkeypatch.setattr(release_bundle, "RESULTS_DIR", tmp_path)
        with pytest.raises(release_bundle.ReleaseError, match="did not pass"):
            release_bundle.build_release_manifest()

    def test_release_digest_is_canonical(self) -> None:
        manifest = {
            "study_version": "1.0.0",
            "corpus": {"corpus_sha256": "c"},
            "annotations": {"annotation_hash": "a"},
            "components": {
                "b.json": {"sha256": "2"},
                "a.json": {"sha256": "1"},
            },
        }
        permuted = {
            **manifest,
            "components": {
                "a.json": {"sha256": "1"},
                "b.json": {"sha256": "2"},
            },
        }
        assert release_bundle.release_digest(manifest) == release_bundle.release_digest(permuted)
        changed = {**manifest, "study_version": "1.0.1"}
        assert release_bundle.release_digest(manifest) != release_bundle.release_digest(changed)

    def test_supersede_moves_bundle_to_archive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        releases = tmp_path / "releases"
        archive = tmp_path / "archive"
        monkeypatch.setattr(release_bundle, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(release_bundle, "RELEASES_DIR", releases)
        monkeypatch.setattr(release_bundle, "ARCHIVE_DIR", archive)

        bundle = releases / "trustparadox_u-v1.0.0-abcdef123456"
        bundle.mkdir(parents=True)
        (bundle / "bundle_manifest.json").write_text(
            json.dumps({"release_id": "trustparadox_u-v1.0.0-abcdef123456", "status": "active"})
        )

        release_bundle.supersede_release(bundle, superseded_by="trustparadox_u-v1.0.1-000000000000")

        assert not bundle.exists()
        archived = archive / "trustparadox_u-v1.0.0-abcdef123456"
        assert (archived / "INVALIDATION_MARKER.json").exists()
        marker = json.loads((archived / "INVALIDATION_MARKER.json").read_text())
        assert marker["status"] == "superseded"
        manifest = json.loads((archived / "bundle_manifest.json").read_text())
        assert manifest["status"] == "superseded"
        assert manifest["superseded_by"] == "trustparadox_u-v1.0.1-000000000000"

    def test_validate_bundle_detects_hash_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        releases = tmp_path / "releases"
        monkeypatch.setattr(release_bundle, "RELEASES_DIR", releases)
        bundle = releases / "trustparadox_u-v1.0.0-abcdef123456"
        bundle.mkdir(parents=True)
        component = bundle / "metrics.json"
        component.write_text('{"value": 1}')
        manifest = {
            "release_id": "trustparadox_u-v1.0.0-abcdef123456",
            "release_digest": "0" * 64,
            "study_version": "1.0.0",
            "provenance": {},
            "corpus": {"corpus_sha256": "c"},
            "annotations": {"annotation_hash": "a"},
            "components": {"metrics.json": {"role": "r", "sha256": "wrong"}},
        }
        (bundle / "bundle_manifest.json").write_text(json.dumps(manifest))
        findings = release_bundle.validate_release_bundle(bundle)
        assert any("component hash mismatch" in f for f in findings)
        assert any("release_id inconsistent" in f for f in findings)

    def test_superseded_marker_in_archive_never_fails_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiments.trustparadox_u.invalidation import find_invalidation_markers

        archived = tmp_path / "archive" / "trustparadox_u-v1.0.0-abcdef123456"
        archived.mkdir(parents=True)
        (archived / "INVALIDATION_MARKER.json").write_text(
            json.dumps({"status": "superseded", "reason": "superseded"})
        )
        assert find_invalidation_markers(tmp_path) == []

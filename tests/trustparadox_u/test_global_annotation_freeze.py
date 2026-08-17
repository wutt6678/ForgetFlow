"""E4-003: Global annotation freeze fixture tests.

Covers checklist Sec 152:
- split root hashes
- global counts
- phase transition
- immutable protocol SHA
- mutation failures
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"
_PROTOCOL_PATH = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
_FREEZE_PATH = _ANNOTATIONS_DIR / "global_annotation_freeze_manifest.json"
_POST_FREEZE_PATH = _ANNOTATIONS_DIR / "global_annotation_post_freeze_verification.json"
_CORPUS_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
)

_DEV_DIR = _ANNOTATIONS_DIR / "development_v3"
_VAL_DIR = _ANNOTATIONS_DIR / "validation"
_TEST_DIR = _ANNOTATIONS_DIR / "test"

pytestmark = pytest.mark.skipif(
    not _FREEZE_PATH.exists(),
    reason="Global freeze manifest not found; run global freeze first",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ===========================================================================
# Split root hashes
# ===========================================================================


class TestSplitRootHashes:
    """Sec 152: Global freeze must bind all split root SHAs."""

    def test_corpus_manifest_sha_bound(self):
        """Frozen corpus manifest SHA must be bound."""
        freeze = _load_json(_FREEZE_PATH)
        assert "frozen_corpus_manifest_sha256" in freeze
        assert freeze["frozen_corpus_manifest_sha256"]
        if _CORPUS_MANIFEST_PATH.exists():
            actual = _sha256(_CORPUS_MANIFEST_PATH)
            assert actual == freeze["frozen_corpus_manifest_sha256"]

    def test_protocol_sha_bound(self):
        """Annotation protocol SHA must be bound."""
        freeze = _load_json(_FREEZE_PATH)
        assert "frozen_annotation_protocol_sha256" in freeze
        assert freeze["frozen_annotation_protocol_sha256"]
        if _PROTOCOL_PATH.exists():
            actual = _sha256(_PROTOCOL_PATH)
            assert actual == freeze["frozen_annotation_protocol_sha256"]

    def test_development_roots_bound(self):
        """Development split roots must be bound."""
        freeze = _load_json(_FREEZE_PATH)
        required = [
            "development_annotation_manifest_sha256",
            "development_annotation_gate_sha256",
            "development_adjudication_manifest_sha256",
            "development_final_row_labels_sha256",
        ]
        for field in required:
            assert field in freeze, f"Missing field: {field}"
            assert freeze[field], f"Empty SHA for: {field}"

    def test_validation_roots_bound(self):
        """Validation split roots must be bound."""
        freeze = _load_json(_FREEZE_PATH)
        required = [
            "validation_annotation_freeze_manifest_sha256",
            "validation_post_freeze_verification_sha256",
            "validation_final_row_labels_sha256",
            "validation_final_sequence_labels_sha256",
        ]
        for field in required:
            assert field in freeze, f"Missing field: {field}"
            assert freeze[field], f"Empty SHA for: {field}"

    def test_test_roots_bound(self):
        """Test split roots must be bound."""
        freeze = _load_json(_FREEZE_PATH)
        required = [
            "test_annotation_freeze_manifest_sha256",
            "test_post_freeze_verification_sha256",
            "test_final_row_labels_sha256",
            "test_final_sequence_labels_sha256",
        ]
        for field in required:
            assert field in freeze, f"Missing field: {field}"
            assert freeze[field], f"Empty SHA for: {field}"


# ===========================================================================
# Global counts
# ===========================================================================


class TestGlobalCounts:
    """Sec 152: Global freeze must record correct counts."""

    def test_development_counts(self):
        """Development counts must be 225 rows, 36 sequences."""
        freeze = _load_json(_FREEZE_PATH)
        dev = freeze.get("split_counts", {}).get("development", {})
        assert dev.get("final_rows") == 225
        assert dev.get("final_sequences") == 36

    def test_validation_counts(self):
        """Validation counts must be 225 rows, 36 sequences."""
        freeze = _load_json(_FREEZE_PATH)
        val = freeze.get("split_counts", {}).get("validation", {})
        assert val.get("final_rows") == 225
        assert val.get("final_sequences") == 36

    def test_test_counts(self):
        """Test counts must be 450 rows, 72 sequences."""
        freeze = _load_json(_FREEZE_PATH)
        test = freeze.get("split_counts", {}).get("test", {})
        assert test.get("final_rows") == 450
        assert test.get("final_sequences") == 72

    def test_global_total_rows(self):
        """Global total must be 900 final rows."""
        freeze = _load_json(_FREEZE_PATH)
        totals = freeze.get("global_totals", {})
        assert totals.get("final_row_labels") == 900

    def test_global_total_sequences(self):
        """Global total must be 144 final sequence units."""
        freeze = _load_json(_FREEZE_PATH)
        totals = freeze.get("global_totals", {})
        assert totals.get("final_sequence_labels") == 144


# ===========================================================================
# Phase transition
# ===========================================================================


class TestPhaseTransition:
    """Sec 152: Phase must transition to ANNOTATIONS_FROZEN."""

    def test_annotation_phase_frozen(self):
        """annotation_phase must be ANNOTATIONS_FROZEN."""
        if not _PHASE_PATH.exists():
            pytest.skip("annotation_phase.json not found")
        phase = _load_json(_PHASE_PATH)
        assert phase.get("annotation_phase") == "ANNOTATIONS_FROZEN"

    def test_annotations_frozen_true(self):
        """annotations_frozen must be true."""
        if not _PHASE_PATH.exists():
            pytest.skip("annotation_phase.json not found")
        phase = _load_json(_PHASE_PATH)
        assert phase.get("annotations_frozen") is True

    def test_test_annotation_complete_true(self):
        """test_annotation_complete must be true."""
        if not _PHASE_PATH.exists():
            pytest.skip("annotation_phase.json not found")
        phase = _load_json(_PHASE_PATH)
        assert phase.get("test_annotation_complete") is True

    def test_all_splits_complete(self):
        """All split completion flags must be true."""
        if not _PHASE_PATH.exists():
            pytest.skip("annotation_phase.json not found")
        phase = _load_json(_PHASE_PATH)
        assert phase.get("development_annotation_complete") is True
        assert phase.get("validation_annotation_complete") is True
        assert phase.get("test_annotation_complete") is True


# ===========================================================================
# Immutable protocol SHA
# ===========================================================================


class TestImmutableProtocol:
    """Sec 152: Protocol SHA must be immutable after freeze."""

    def test_protocol_sha_in_freeze(self):
        """Protocol SHA must be recorded in freeze manifest."""
        freeze = _load_json(_FREEZE_PATH)
        assert "frozen_annotation_protocol_sha256" in freeze
        assert freeze["frozen_annotation_protocol_sha256"]

    def test_protocol_sha_matches_file(self):
        """Protocol SHA must match actual file."""
        if not _PROTOCOL_PATH.exists():
            pytest.skip("annotation_protocol_manifest.json not found")
        freeze = _load_json(_FREEZE_PATH)
        actual = _sha256(_PROTOCOL_PATH)
        assert actual == freeze["frozen_annotation_protocol_sha256"]

    def test_corpus_sha_in_freeze(self):
        """Corpus manifest SHA must be recorded in freeze manifest."""
        freeze = _load_json(_FREEZE_PATH)
        assert "frozen_corpus_manifest_sha256" in freeze
        assert freeze["frozen_corpus_manifest_sha256"]

    def test_corpus_sha_matches_file(self):
        """Corpus manifest SHA must match actual file."""
        if not _CORPUS_MANIFEST_PATH.exists():
            pytest.skip("frozen_corpus_manifest.json not found")
        freeze = _load_json(_FREEZE_PATH)
        actual = _sha256(_CORPUS_MANIFEST_PATH)
        assert actual == freeze["frozen_corpus_manifest_sha256"]


# ===========================================================================
# Mutation failures
# ===========================================================================


class TestMutationFailures:
    """Sec 152: Post-freeze verification must detect mutations."""

    def test_post_freeze_verification_exists(self):
        """Post-freeze verification must exist."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("Post-freeze verification not found")
        pfv = _load_json(_POST_FREEZE_PATH)
        assert "schema_version" in pfv
        assert "all_verifiers_pass" in pfv

    def test_all_verifiers_pass(self):
        """All verifiers must pass in post-freeze verification."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("Post-freeze verification not found")
        pfv = _load_json(_POST_FREEZE_PATH)
        assert pfv.get("all_verifiers_pass") is True

    def test_freeze_manifest_sha_bound(self):
        """Freeze manifest SHA must be bound in post-freeze."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("Post-freeze verification not found")
        pfv = _load_json(_POST_FREEZE_PATH)
        assert "global_freeze_manifest_sha256" in pfv
        if _FREEZE_PATH.exists():
            actual = _sha256(_FREEZE_PATH)
            assert actual == pfv["global_freeze_manifest_sha256"]

    def test_phase_sha_bound(self):
        """Phase SHA must be bound in post-freeze."""
        if not _POST_FREEZE_PATH.exists():
            pytest.skip("Post-freeze verification not found")
        pfv = _load_json(_POST_FREEZE_PATH)
        assert "annotation_phase_sha256" in pfv
        if _PHASE_PATH.exists():
            actual = _sha256(_PHASE_PATH)
            assert actual == pfv["annotation_phase_sha256"]

    def test_split_gates_all_go(self):
        """All split gates must be GO."""
        freeze = _load_json(_FREEZE_PATH)
        gates = freeze.get("split_gates", {})
        assert gates.get("development") == "GO"
        assert gates.get("validation") == "GO"
        assert gates.get("test") == "GO"
        assert freeze.get("all_gates_go") is True

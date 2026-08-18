"""E5-000: Preflight and frozen-input loader tests.

Iteration 1 exit criteria:
- E5 preflight PASS
- E4 global annotation root verified
- split counts correct
- read-only annotation loader works
- no embedding calls yet
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_loaders import (
    VALID_SPLITS,
    CorpusCandidate,
    RowLabel,
    SequenceLabel,
    SplitData,
    compute_file_hashes,
    get_expected_counts,
    get_expected_unresolved,
    load_all_splits,
    load_corpus,
    load_corpus_manifest,
    load_global_freeze_manifest,
    load_row_labels,
    load_sequence_labels,
    load_split,
    sha256_file,
)

_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_GLOBAL_FREEZE_PATH = _ANNOTATIONS_DIR / "global_annotation_freeze_manifest.json"
_CORPUS_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
)
_E5_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "e5"
_PREFLIGHT_PATH = _E5_DIR / "e5_preflight.json"

pytestmark = pytest.mark.skipif(
    not _GLOBAL_FREEZE_PATH.exists(),
    reason="Global freeze manifest not found; E4 freeze required",
)


# ---------------------------------------------------------------------------
# Loader unit tests
# ---------------------------------------------------------------------------


class TestGlobalFreezeManifest:
    """Tests for loading the E4 global annotation freeze manifest."""

    def test_manifest_loads(self) -> None:
        manifest = load_global_freeze_manifest()
        assert isinstance(manifest, dict)

    def test_go_no_go(self) -> None:
        manifest = load_global_freeze_manifest()
        assert manifest["go_no_go"] == "GO"

    def test_annotations_frozen(self) -> None:
        manifest = load_global_freeze_manifest()
        assert manifest["annotations_frozen"] is True

    def test_annotation_phase(self) -> None:
        manifest = load_global_freeze_manifest()
        assert manifest["annotation_phase"] == "ANNOTATIONS_FROZEN"

    def test_all_gates_go(self) -> None:
        manifest = load_global_freeze_manifest()
        assert manifest["all_gates_go"] is True

    def test_global_row_count(self) -> None:
        manifest = load_global_freeze_manifest()
        assert manifest["global_totals"]["final_row_labels"] == 900

    def test_global_sequence_count(self) -> None:
        manifest = load_global_freeze_manifest()
        assert manifest["global_totals"]["final_sequence_labels"] == 144

    def test_required_sha_keys_present(self) -> None:
        manifest = load_global_freeze_manifest()
        required_keys = [
            "frozen_corpus_manifest_sha256",
            "frozen_annotation_protocol_sha256",
            "development_final_row_labels_sha256",
            "development_final_sequence_labels_sha256",
            "validation_final_row_labels_sha256",
            "validation_final_sequence_labels_sha256",
            "test_final_row_labels_sha256",
            "test_final_sequence_labels_sha256",
        ]
        for key in required_keys:
            assert key in manifest, f"Missing required SHA key: {key}"
            assert len(manifest[key]) == 64, f"SHA {key} is not 64 hex chars"


class TestCorpusManifest:
    """Tests for loading the frozen corpus manifest."""

    def test_manifest_loads(self) -> None:
        manifest = load_corpus_manifest()
        assert isinstance(manifest, dict)

    def test_corpus_frozen(self) -> None:
        manifest = load_corpus_manifest()
        assert manifest["corpus_frozen"] is True

    def test_accepted_candidate_count(self) -> None:
        manifest = load_corpus_manifest()
        assert manifest["accepted_candidate_count"] == 900


class TestRowLabelLoader:
    """Tests for loading frozen row-level annotation labels."""

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_row_label_count(self, split: str) -> None:
        expected = get_expected_counts()
        labels = load_row_labels(split)
        assert len(labels) == expected[split]["rows"]

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_row_labels_are_frozen(self, split: str) -> None:
        labels = load_row_labels(split)
        for label in labels:
            assert isinstance(label, RowLabel)
            # RowLabel is frozen dataclass
            with pytest.raises(AttributeError):
                label.candidate_id = "mutated"  # type: ignore[misc]

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_row_labels_have_candidate_ids(self, split: str) -> None:
        labels = load_row_labels(split)
        ids = [r.candidate_id for r in labels]
        assert len(ids) == len(set(ids)), f"Duplicate candidate_ids in {split}"

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_unresolved_row_count(self, split: str) -> None:
        expected = get_expected_unresolved()
        labels = load_row_labels(split)
        unresolved = sum(1 for r in labels if r.is_unresolved)
        assert unresolved == expected[split]["rows"]

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            load_row_labels("nonexistent")


class TestSequenceLabelLoader:
    """Tests for loading frozen sequence-level annotation labels."""

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_sequence_label_count(self, split: str) -> None:
        expected = get_expected_counts()
        labels = load_sequence_labels(split)
        assert len(labels) == expected[split]["sequences"]

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_sequence_labels_are_frozen(self, split: str) -> None:
        labels = load_sequence_labels(split)
        for label in labels:
            assert isinstance(label, SequenceLabel)
            with pytest.raises(AttributeError):
                label.sequence_annotation_id = "mutated"  # type: ignore[misc]

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_sequence_labels_have_unique_ids(self, split: str) -> None:
        labels = load_sequence_labels(split)
        ids = [s.sequence_annotation_id for s in labels]
        assert len(ids) == len(set(ids)), f"Duplicate sequence IDs in {split}"

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_all_sequences_resolved(self, split: str) -> None:
        """E4 freeze has 0 unresolved sequences in all splits."""
        labels = load_sequence_labels(split)
        unresolved = sum(1 for s in labels if s.is_unresolved)
        assert unresolved == 0

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_ordered_candidate_ids_nonempty(self, split: str) -> None:
        labels = load_sequence_labels(split)
        for label in labels:
            assert len(label.ordered_candidate_ids) > 0


class TestCorpusLoader:
    """Tests for loading frozen corpus accepted candidates."""

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_corpus_count(self, split: str) -> None:
        expected = get_expected_counts()
        corpus = load_corpus(split)
        assert len(corpus) == expected[split]["rows"]

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_corpus_candidates_are_frozen(self, split: str) -> None:
        corpus = load_corpus(split)
        for candidate in corpus:
            assert isinstance(candidate, CorpusCandidate)
            with pytest.raises(AttributeError):
                candidate.candidate_id = "mutated"  # type: ignore[misc]

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_corpus_candidate_ids_unique(self, split: str) -> None:
        corpus = load_corpus(split)
        ids = [c.candidate_id for c in corpus]
        assert len(ids) == len(set(ids)), f"Duplicate candidate_ids in {split} corpus"

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_corpus_split_field_matches(self, split: str) -> None:
        corpus = load_corpus(split)
        for candidate in corpus:
            assert candidate.split == split

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_corpus_text_nonempty(self, split: str) -> None:
        corpus = load_corpus(split)
        for candidate in corpus:
            assert len(candidate.text) > 0
            assert len(candidate.normalized_text) > 0


class TestSplitDataLoader:
    """Tests for the combined SplitData loader."""

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_split_data_loads(self, split: str) -> None:
        sd = load_split(split)
        assert isinstance(sd, SplitData)
        assert sd.split == split

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_split_data_lookup_dicts(self, split: str) -> None:
        sd = load_split(split)
        # row_labels_by_id should have same count as row_labels
        assert len(sd.row_labels_by_id) == sd.n_rows
        # corpus_by_id should have same count as corpus
        assert len(sd.corpus_by_id) == sd.n_corpus

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_split_data_eligible_rows(self, split: str) -> None:
        sd = load_split(split)
        eligible = sd.eligible_row_labels
        expected = get_expected_unresolved()
        assert len(eligible) == sd.n_rows - expected[split]["rows"]

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_split_data_eligible_sequences(self, split: str) -> None:
        sd = load_split(split)
        # All sequences are resolved, so eligible == total
        assert len(sd.eligible_sequence_labels) == sd.n_sequences


class TestLoadAllSplits:
    """Tests for loading all splits at once."""

    def test_all_splits_loaded(self) -> None:
        all_data = load_all_splits()
        assert set(all_data.keys()) == VALID_SPLITS

    def test_total_rows(self) -> None:
        all_data = load_all_splits()
        total = sum(sd.n_rows for sd in all_data.values())
        assert total == 900

    def test_total_sequences(self) -> None:
        all_data = load_all_splits()
        total = sum(sd.n_sequences for sd in all_data.values())
        assert total == 144

    def test_total_corpus(self) -> None:
        all_data = load_all_splits()
        total = sum(sd.n_corpus for sd in all_data.values())
        assert total == 900


class TestCorpusAnnotationOverlap:
    """Tests that corpus and annotation candidate_ids align."""

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_all_label_ids_in_corpus(self, split: str) -> None:
        sd = load_split(split)
        corpus_ids = set(sd.corpus_by_id.keys())
        label_ids = set(sd.row_labels_by_id.keys())
        missing = label_ids - corpus_ids
        assert not missing, f"{split}: {len(missing)} label IDs not in corpus"


# ---------------------------------------------------------------------------
# SHA / provenance tests
# ---------------------------------------------------------------------------


class TestSHAProvenance:
    """Tests for SHA-256 hash computation and provenance binding."""

    def test_sha256_file_deterministic(self) -> None:
        h1 = sha256_file(_GLOBAL_FREEZE_PATH)
        h2 = sha256_file(_GLOBAL_FREEZE_PATH)
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_file_hashes(self) -> None:
        hashes = compute_file_hashes()
        assert "global_annotation_freeze_manifest" in hashes
        assert "frozen_corpus_manifest" in hashes
        # All split files should be hashed
        for split in sorted(VALID_SPLITS):
            assert f"{split}_row_labels" in hashes
            assert f"{split}_sequence_labels" in hashes
            assert f"{split}_corpus" in hashes

    def test_hashes_match_freeze_manifest(self) -> None:
        """Verify computed hashes match the SHAs recorded in the freeze manifest."""
        freeze = load_global_freeze_manifest()
        hashes = compute_file_hashes()

        # The global freeze manifest records SHAs of the individual label files
        # Verify the frozen corpus manifest SHA matches
        corpus_sha = hashes.get("frozen_corpus_manifest", "")
        assert corpus_sha == freeze["frozen_corpus_manifest_sha256"]


# ---------------------------------------------------------------------------
# Split isolation tests
# ---------------------------------------------------------------------------


class TestSplitIsolation:
    """Tests proving that splits are independent and do not overlap."""

    def test_no_candidate_id_overlap_between_splits(self) -> None:
        all_data = load_all_splits()
        splits_list = sorted(VALID_SPLITS)
        for i, s1 in enumerate(splits_list):
            for s2 in splits_list[i + 1 :]:
                ids1 = set(all_data[s1].row_labels_by_id.keys())
                ids2 = set(all_data[s2].row_labels_by_id.keys())
                overlap = ids1 & ids2
                assert not overlap, (
                    f"Candidate ID overlap between {s1} and {s2}: {len(overlap)} shared"
                )

    def test_no_sequence_id_overlap_between_splits(self) -> None:
        all_data = load_all_splits()
        splits_list = sorted(VALID_SPLITS)
        for i, s1 in enumerate(splits_list):
            for s2 in splits_list[i + 1 :]:
                ids1 = {s.sequence_annotation_id for s in all_data[s1].sequence_labels}
                ids2 = {s.sequence_annotation_id for s in all_data[s2].sequence_labels}
                overlap = ids1 & ids2
                assert not overlap, (
                    f"Sequence ID overlap between {s1} and {s2}: {len(overlap)} shared"
                )


# ---------------------------------------------------------------------------
# Preflight script tests
# ---------------------------------------------------------------------------


class TestPreflightScript:
    """Tests for the e5_preflight.py script execution and output."""

    def test_preflight_script_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "e5_preflight.py")],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"Preflight script exited with {result.returncode}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_preflight_artifact_created(self) -> None:
        # Run the script to ensure the artifact exists
        subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "e5_preflight.py")],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=60,
        )
        assert _PREFLIGHT_PATH.exists(), "e5_preflight.json not created"

    def test_preflight_artifact_structure(self) -> None:
        subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "e5_preflight.py")],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=60,
        )
        with open(_PREFLIGHT_PATH) as f:
            preflight = json.load(f)

        # Required fields
        required_fields = [
            "schema_version",
            "created_at",
            "code_commit",
            "global_annotation_freeze_sha256",
            "annotation_protocol_sha256",
            "frozen_corpus_sha256",
            "development_rows",
            "validation_rows",
            "test_rows",
            "development_sequences",
            "validation_sequences",
            "test_sequences",
            "unresolved_by_split",
            "preflight_pass",
            "blocking_findings",
        ]
        for field_name in required_fields:
            assert field_name in preflight, f"Missing field: {field_name}"

    def test_preflight_passes(self) -> None:
        subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "e5_preflight.py")],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=60,
        )
        with open(_PREFLIGHT_PATH) as f:
            preflight = json.load(f)

        assert preflight["preflight_pass"] is True
        assert preflight["blocking_findings"] == []

    def test_preflight_row_counts(self) -> None:
        subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "e5_preflight.py")],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=60,
        )
        with open(_PREFLIGHT_PATH) as f:
            preflight = json.load(f)

        assert preflight["development_rows"] == 225
        assert preflight["validation_rows"] == 225
        assert preflight["test_rows"] == 450

    def test_preflight_sequence_counts(self) -> None:
        subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "e5_preflight.py")],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=60,
        )
        with open(_PREFLIGHT_PATH) as f:
            preflight = json.load(f)

        assert preflight["development_sequences"] == 36
        assert preflight["validation_sequences"] == 36
        assert preflight["test_sequences"] == 72

    def test_preflight_unresolved_counts(self) -> None:
        subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "e5_preflight.py")],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=60,
        )
        with open(_PREFLIGHT_PATH) as f:
            preflight = json.load(f)

        urs = preflight["unresolved_by_split"]
        assert urs["development"]["unresolved_rows"] == 14
        assert urs["development"]["unresolved_sequences"] == 0
        assert urs["validation"]["unresolved_rows"] == 9
        assert urs["validation"]["unresolved_sequences"] == 0
        assert urs["test"]["unresolved_rows"] == 24
        assert urs["test"]["unresolved_sequences"] == 0


# ---------------------------------------------------------------------------
# No embedding calls test
# ---------------------------------------------------------------------------


class TestNoEmbeddingCalls:
    """Prove that Iteration 1 makes zero embedding or LLM calls."""

    def test_loaders_module_has_no_embedding_imports(self) -> None:
        """The e5_loaders module should not import any embedding or LLM code."""
        import experiments.trustparadox_u.e5_loaders as mod

        source = Path(mod.__file__).read_text()
        # Check that no embedding backend or LLM provider is imported
        forbidden = [
            "embedding_backend",
            "chat_provider",
            "litellm",
            "openai",
            "torch",
            "sentence_transformers",
        ]
        for token in forbidden:
            assert token not in source, (
                f"e5_loaders.py should not reference {token!r}"
            )

    def test_preflight_script_has_no_embedding_imports(self) -> None:
        """The preflight script should not import any embedding or LLM code."""
        preflight_path = _PROJECT_ROOT / "scripts" / "e5_preflight.py"
        source = preflight_path.read_text()
        forbidden = [
            "embedding_backend",
            "chat_provider",
            "litellm",
            "openai",
            "torch",
            "sentence_transformers",
        ]
        for token in forbidden:
            assert token not in source, (
                f"e5_preflight.py should not reference {token!r}"
            )

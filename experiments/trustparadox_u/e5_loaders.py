"""E5 read-only loaders for frozen E4 annotations and corpus.

This module provides deterministic, read-only access to the frozen E4
annotation labels and the frozen corpus accepted candidates.  It is the
sole data-access layer that E5 experiments should use when joining
frozen evidence to detector/firewall outputs.

Key invariants:
- All loaders return frozen dataclasses or tuples.
- No loader mutates annotation or corpus files.
- Unresolved rows are tagged so downstream metrics can exclude them.
- SHA-256 hashes are computed for provenance binding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-level path constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_CORPUS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"

_GLOBAL_FREEZE_PATH = _ANNOTATIONS_DIR / "global_annotation_freeze_manifest.json"
_CORPUS_MANIFEST_PATH = _CORPUS_DIR / "frozen_corpus_manifest.json"

# Annotation file paths per split (development uses v3 directory)
_SPLIT_ROW_LABEL_PATHS: dict[str, Path] = {
    "development": _ANNOTATIONS_DIR / "development_v3" / "final_adjudicated_labels.jsonl",
    "validation": _ANNOTATIONS_DIR / "validation" / "final_adjudicated_labels.jsonl",
    "test": _ANNOTATIONS_DIR / "test" / "test_final_adjudicated_labels.jsonl",
}

_SPLIT_SEQUENCE_LABEL_PATHS: dict[str, Path] = {
    "development": _ANNOTATIONS_DIR / "development_v3" / "final_sequence_labels.jsonl",
    "validation": _ANNOTATIONS_DIR / "validation" / "final_sequence_labels.jsonl",
    "test": _ANNOTATIONS_DIR / "test" / "test_final_sequence_labels.jsonl",
}

_SPLIT_CORPUS_PATHS: dict[str, Path] = {
    "development": _CORPUS_DIR / "development" / "accepted_candidates.jsonl",
    "validation": _CORPUS_DIR / "validation" / "accepted_candidates.jsonl",
    "test": _CORPUS_DIR / "test" / "accepted_candidates.jsonl",
}

VALID_SPLITS = frozenset({"development", "validation", "test"})

# Expected counts from the E4 global freeze manifest
_EXPECTED_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "development": {"rows": 225, "sequences": 36},
    "validation": {"rows": 225, "sequences": 36},
    "test": {"rows": 450, "sequences": 72},
}

_EXPECTED_UNRESOLVED: dict[str, dict[str, int]] = {
    "development": {"rows": 14, "sequences": 0},
    "validation": {"rows": 9, "sequences": 0},
    "test": {"rows": 24, "sequences": 0},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowLabel:
    """Frozen row-level annotation label for a single candidate."""

    candidate_id: str
    final_target_relevant: bool | None
    final_target_leakage: bool | None
    final_positive_entailment: bool | None
    final_task_useful: bool | None
    final_leakage_strength: str
    resolution_source: str
    resolution_status: str  # "resolved" or "unresolved"

    @property
    def is_unresolved(self) -> bool:
        return self.resolution_status != "resolved"


@dataclass(frozen=True)
class SequenceLabel:
    """Frozen sequence-level annotation label."""

    sequence_annotation_id: str
    sequence_family_id: str
    trust_level: str
    scenario_id: str
    secret_variant_id: str
    ordered_candidate_ids: tuple[str, ...]
    final_sequence_reconstructs_target: bool
    final_earliest_reconstruction_step: int | None
    final_reconstruction_strength: str
    resolution_source: str
    resolution_status: str

    @property
    def is_unresolved(self) -> bool:
        return self.resolution_status != "resolved"


@dataclass(frozen=True)
class CorpusCandidate:
    """Frozen corpus candidate message."""

    candidate_id: str
    candidate_family_id: str
    text: str
    normalized_text: str
    attack_type: str
    scenario_id: str
    secret_variant_id: str
    trust_level: str
    split: str
    recipient_id: str
    sender_id: str
    sequence_family_id: str | None
    sequence_id: str | None
    sequence_step_index: int | None
    sequence_step_count: int | None
    content_sha256: str


@dataclass(frozen=True)
class SplitData:
    """Complete frozen data for a single split."""

    split: str
    row_labels: tuple[RowLabel, ...]
    sequence_labels: tuple[SequenceLabel, ...]
    corpus: tuple[CorpusCandidate, ...]
    row_labels_by_id: dict[str, RowLabel] = field(
        default_factory=dict, compare=False, hash=False
    )
    corpus_by_id: dict[str, CorpusCandidate] = field(
        default_factory=dict, compare=False, hash=False
    )

    @property
    def n_rows(self) -> int:
        return len(self.row_labels)

    @property
    def n_sequences(self) -> int:
        return len(self.sequence_labels)

    @property
    def n_corpus(self) -> int:
        return len(self.corpus)

    @property
    def n_unresolved_rows(self) -> int:
        return sum(1 for r in self.row_labels if r.is_unresolved)

    @property
    def n_unresolved_sequences(self) -> int:
        return sum(1 for s in self.sequence_labels if s.is_unresolved)

    @property
    def eligible_row_labels(self) -> tuple[RowLabel, ...]:
        """Row labels with resolved annotation status only."""
        return tuple(r for r in self.row_labels if not r.is_unresolved)

    @property
    def eligible_sequence_labels(self) -> tuple[SequenceLabel, ...]:
        """Sequence labels with resolved annotation status only."""
        return tuple(s for s in self.sequence_labels if not s.is_unresolved)


# ---------------------------------------------------------------------------
# SHA-256 utility
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return a list of parsed records."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_row_label(raw: dict[str, Any]) -> RowLabel:
    """Parse a raw JSONL record into a frozen RowLabel."""
    return RowLabel(
        candidate_id=raw["candidate_id"],
        final_target_relevant=raw.get("final_target_relevant"),
        final_target_leakage=raw.get("final_target_leakage"),
        final_positive_entailment=raw.get("final_positive_entailment"),
        final_task_useful=raw.get("final_task_useful"),
        final_leakage_strength=raw.get("final_leakage_strength", "unknown"),
        resolution_source=raw.get("resolution_source", "unknown"),
        resolution_status=raw.get("resolution_status", "unresolved"),
    )


def _parse_sequence_label(raw: dict[str, Any]) -> SequenceLabel:
    """Parse a raw JSONL record into a frozen SequenceLabel."""
    return SequenceLabel(
        sequence_annotation_id=raw["sequence_annotation_id"],
        sequence_family_id=raw["sequence_family_id"],
        trust_level=raw["trust_level"],
        scenario_id=raw["scenario_id"],
        secret_variant_id=raw["secret_variant_id"],
        ordered_candidate_ids=tuple(raw["ordered_candidate_ids"]),
        final_sequence_reconstructs_target=raw["final_sequence_reconstructs_target"],
        final_earliest_reconstruction_step=raw.get("final_earliest_reconstruction_step"),
        final_reconstruction_strength=raw.get("final_reconstruction_strength", "unknown"),
        resolution_source=raw.get("resolution_source", "unknown"),
        resolution_status=raw.get("resolution_status", "unresolved"),
    )


def _parse_corpus_candidate(raw: dict[str, Any]) -> CorpusCandidate:
    """Parse a raw JSONL record into a frozen CorpusCandidate."""
    return CorpusCandidate(
        candidate_id=raw["candidate_id"],
        candidate_family_id=raw["candidate_family_id"],
        text=raw["text"],
        normalized_text=raw["normalized_text"],
        attack_type=raw["attack_type"],
        scenario_id=raw["scenario_id"],
        secret_variant_id=raw["secret_variant_id"],
        trust_level=raw["trust_level"],
        split=raw["split"],
        recipient_id=raw["recipient_id"],
        sender_id=raw["sender_id"],
        sequence_family_id=raw.get("sequence_family_id"),
        sequence_id=raw.get("sequence_id"),
        sequence_step_index=raw.get("sequence_step_index"),
        sequence_step_count=raw.get("sequence_step_count"),
        content_sha256=raw["content_sha256"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_global_freeze_manifest() -> dict[str, Any]:
    """Load and return the E4 global annotation freeze manifest.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    if not _GLOBAL_FREEZE_PATH.exists():
        raise FileNotFoundError(
            f"Global annotation freeze manifest not found: {_GLOBAL_FREEZE_PATH}"
        )
    with open(_GLOBAL_FREEZE_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]


def load_corpus_manifest() -> dict[str, Any]:
    """Load and return the frozen corpus manifest.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    if not _CORPUS_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Frozen corpus manifest not found: {_CORPUS_MANIFEST_PATH}"
        )
    with open(_CORPUS_MANIFEST_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]


def load_row_labels(split: str) -> tuple[RowLabel, ...]:
    """Load frozen row-level annotation labels for a split.

    Args:
        split: One of "development", "validation", "test".

    Returns:
        Tuple of frozen RowLabel records in file order.

    Raises:
        ValueError: If split is unknown.
        FileNotFoundError: If the label file does not exist.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}. Valid: {sorted(VALID_SPLITS)}")
    path = _SPLIT_ROW_LABEL_PATHS[split]
    if not path.exists():
        raise FileNotFoundError(f"Row label file not found: {path}")
    raw_records = _load_jsonl(path)
    return tuple(_parse_row_label(r) for r in raw_records)


def load_sequence_labels(split: str) -> tuple[SequenceLabel, ...]:
    """Load frozen sequence-level annotation labels for a split.

    Args:
        split: One of "development", "validation", "test".

    Returns:
        Tuple of frozen SequenceLabel records in file order.

    Raises:
        ValueError: If split is unknown.
        FileNotFoundError: If the label file does not exist.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}. Valid: {sorted(VALID_SPLITS)}")
    path = _SPLIT_SEQUENCE_LABEL_PATHS[split]
    if not path.exists():
        raise FileNotFoundError(f"Sequence label file not found: {path}")
    raw_records = _load_jsonl(path)
    return tuple(_parse_sequence_label(r) for r in raw_records)


def load_corpus(split: str) -> tuple[CorpusCandidate, ...]:
    """Load frozen corpus accepted candidates for a split.

    Args:
        split: One of "development", "validation", "test".

    Returns:
        Tuple of frozen CorpusCandidate records in file order.

    Raises:
        ValueError: If split is unknown.
        FileNotFoundError: If the corpus file does not exist.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}. Valid: {sorted(VALID_SPLITS)}")
    path = _SPLIT_CORPUS_PATHS[split]
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    raw_records = _load_jsonl(path)
    return tuple(_parse_corpus_candidate(r) for r in raw_records)


def load_split(split: str) -> SplitData:
    """Load all frozen data for a split (row labels, sequences, corpus).

    Returns a SplitData with lookup dicts pre-built.

    Args:
        split: One of "development", "validation", "test".

    Returns:
        Frozen SplitData with all three data sources joined.

    Raises:
        ValueError: If split is unknown.
        FileNotFoundError: If any required file does not exist.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}. Valid: {sorted(VALID_SPLITS)}")

    row_labels = load_row_labels(split)
    sequence_labels = load_sequence_labels(split)
    corpus = load_corpus(split)

    row_by_id = {r.candidate_id: r for r in row_labels}
    corpus_by_id = {c.candidate_id: c for c in corpus}

    return SplitData(
        split=split,
        row_labels=row_labels,
        sequence_labels=sequence_labels,
        corpus=corpus,
        row_labels_by_id=row_by_id,
        corpus_by_id=corpus_by_id,
    )


def load_all_splits() -> dict[str, SplitData]:
    """Load frozen data for all three splits.

    Returns:
        Dict mapping split name to SplitData.
    """
    return {split: load_split(split) for split in sorted(VALID_SPLITS)}


def compute_file_hashes() -> dict[str, str]:
    """Compute SHA-256 hashes for all frozen evidence files.

    Returns:
        Dict mapping descriptive key to hex digest.
    """
    hashes: dict[str, str] = {}
    hashes["global_annotation_freeze_manifest"] = sha256_file(_GLOBAL_FREEZE_PATH)
    hashes["frozen_corpus_manifest"] = sha256_file(_CORPUS_MANIFEST_PATH)

    for split in sorted(VALID_SPLITS):
        row_path = _SPLIT_ROW_LABEL_PATHS[split]
        seq_path = _SPLIT_SEQUENCE_LABEL_PATHS[split]
        corpus_path = _SPLIT_CORPUS_PATHS[split]
        if row_path.exists():
            hashes[f"{split}_row_labels"] = sha256_file(row_path)
        if seq_path.exists():
            hashes[f"{split}_sequence_labels"] = sha256_file(seq_path)
        if corpus_path.exists():
            hashes[f"{split}_corpus"] = sha256_file(corpus_path)

    return hashes


def get_expected_counts() -> dict[str, dict[str, int]]:
    """Return the expected split counts from the E4 global freeze."""
    return dict(_EXPECTED_SPLIT_COUNTS)


def get_expected_unresolved() -> dict[str, dict[str, int]]:
    """Return the expected unresolved counts from the E4 global freeze."""
    return dict(_EXPECTED_UNRESOLVED)

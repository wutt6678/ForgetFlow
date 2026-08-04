"""Behavioral tests for FF92-022 stale research-valid artifact invalidation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.trustparadox_u.invalidation import (
    ARCHIVE_DIRNAME,
    INVALIDATED_ROOT,
    INVALIDATION_MARKER,
    MARKER_FILENAME,
    find_invalidation_markers,
    find_research_valid_claims,
    invalidate_stale_research_valid_artifacts,
    reject_invalidated_inputs,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_marker_payload_is_exact() -> None:
    assert INVALIDATION_MARKER == {
        "status": "invalidated",
        "reason": "candidate-to-trial mapping and metric pipeline were not valid",
    }


def test_finds_every_claim_shape(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "gate.json", {"verdict": "research_valid"})
    _write(tmp_path / "b" / "report.json", {"research_valid": True})
    _write(tmp_path / "c" / "report.json", {"execution_status": "RESEARCH_VALID"})
    _write(tmp_path / "clean" / "report.json", {"verdict": "diagnostic"})
    _write(tmp_path / "nested" / "not_json.txt", "research_valid")
    claims = find_research_valid_claims(tmp_path)
    assert {p.parent.name for p in claims} == {"a", "b", "c"}


def test_invalidates_and_archives_claiming_directories(tmp_path: Path) -> None:
    _write(tmp_path / "final" / "gate.json", {"verdict": "research_valid"})
    _write(tmp_path / "final" / "keep_me.json", {"other": 1})
    _write(tmp_path / "smoke" / "report.json", {"research_valid": True})
    _write(tmp_path / "clean" / "report.json", {"verdict": "diagnostic"})

    report = invalidate_stale_research_valid_artifacts(tmp_path)
    assert sorted(report["moved"]) == ["final", "smoke"]

    archive_root = tmp_path / ARCHIVE_DIRNAME / INVALIDATED_ROOT
    for rel in ("final", "smoke"):
        marker = archive_root / rel / MARKER_FILENAME
        assert json.loads(marker.read_text()) == INVALIDATION_MARKER
    # Directories move wholesale: sibling artifacts stay with the claim.
    assert (archive_root / "final" / "keep_me.json").exists()
    assert not (tmp_path / "final").exists()
    assert not (tmp_path / "smoke").exists()
    # Clean directories are untouched.
    assert (tmp_path / "clean" / "report.json").exists()

    # Archived claims are invisible to the claim scan; the marker scan only
    # flags markers outside the archive.
    assert find_research_valid_claims(tmp_path) == []
    assert find_invalidation_markers(tmp_path) == []


def test_marker_outside_archive_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "live" / MARKER_FILENAME, INVALIDATION_MARKER)
    markers = find_invalidation_markers(tmp_path)
    assert markers == [tmp_path / "live" / MARKER_FILENAME]


def test_builders_reject_invalidated_inputs(tmp_path: Path) -> None:
    bad = tmp_path / "invalidated"
    bad.mkdir()
    (bad / MARKER_FILENAME).write_text(json.dumps(INVALIDATION_MARKER))
    good = tmp_path / "good"
    good.mkdir()

    with pytest.raises(ValueError, match="Refusing to read invalidated input"):
        reject_invalidated_inputs([good, bad])
    reject_invalidated_inputs([good])  # clean inputs pass


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    _write(tmp_path / "final" / "gate.json", {"verdict": "research_valid"})
    report = invalidate_stale_research_valid_artifacts(tmp_path, dry_run=True)
    assert report["moved"] == ["final"]
    assert (tmp_path / "final" / "gate.json").exists()
    assert not (tmp_path / ARCHIVE_DIRNAME).exists()

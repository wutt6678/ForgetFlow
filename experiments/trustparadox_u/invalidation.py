"""FF92-022: invalidate stale research-valid artifacts.

The artifacts committed at 92bc12e claimed the study was research-valid
while the candidate-to-trial mapping and metric pipeline were not valid.
This module moves every result artifact that still claims
``research_valid`` (or carries a ``research_valid`` gate verdict) out of
the certification path and into ``results/archive/92bc12e_diagnostic_invalid/``,
leaving a clear invalidation marker behind.

Invalidated inputs must never feed the final tables: final-artifact
builders reject any input directory that carries an invalidation marker,
and the research-valid gate fails while markers exist outside the
archive.
"""

from __future__ import annotations

import json
from pathlib import Path

ARCHIVE_DIRNAME = "archive"
INVALIDATED_ROOT = "92bc12e_diagnostic_invalid"
MARKER_FILENAME = "INVALIDATION_MARKER.json"

# Exact marker required by FF92-022.
INVALIDATION_MARKER: dict[str, str] = {
    "status": "invalidated",
    "reason": "candidate-to-trial mapping and metric pipeline were not valid",
}


def _claims_research_valid(data: object) -> bool:
    """Return True when a parsed JSON artifact claims research validity."""
    if not isinstance(data, dict):
        return False
    if data.get("research_valid") is True:
        return True
    if data.get("verdict") == "research_valid":
        return True
    if data.get("execution_status") == "RESEARCH_VALID":
        return True
    return False


def find_research_valid_claims(results_root: Path) -> list[Path]:
    """Find non-archived JSON artifacts that claim research validity."""
    claims: list[Path] = []
    for path in sorted(results_root.rglob("*.json")):
        if ARCHIVE_DIRNAME in path.relative_to(results_root).parts:
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if _claims_research_valid(data):
            claims.append(path)
    return claims


def invalidate_stale_research_valid_artifacts(
    results_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Move stale research-valid artifacts into the invalidation archive.

    Each claiming artifact's containing directory is moved wholesale (the
    artifacts stay together for debugging); a marker with the exact
    FF92-022 payload is written into every archived directory.
    """
    archive_root = results_root / ARCHIVE_DIRNAME / INVALIDATED_ROOT
    moved: list[str] = []
    marked: list[str] = []

    # Deduplicate directories so a directory with multiple claiming files
    # is archived exactly once.
    dirs = sorted({p.parent for p in find_research_valid_claims(results_root)})
    for source_dir in dirs:
        if source_dir == results_root:
            continue
        target_dir = archive_root / source_dir.relative_to(results_root)
        if not dry_run:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            source_dir.rename(target_dir)
            (target_dir / MARKER_FILENAME).write_text(
                json.dumps(INVALIDATION_MARKER, indent=2) + "\n"
            )
        moved.append(str(source_dir.relative_to(results_root)))
        marked.append(str(target_dir / MARKER_FILENAME))

    return {"moved": moved, "markers": marked}


def find_invalidation_markers(results_root: Path) -> list[Path]:
    """Return invalidation markers outside the archive (all bad)."""
    return [
        path
        for path in sorted(results_root.rglob(MARKER_FILENAME))
        if ARCHIVE_DIRNAME not in path.relative_to(results_root).parts
    ]


def reject_invalidated_inputs(input_dirs: list[Path]) -> None:
    """FF92-022: refuse to build final artifacts from invalidated inputs.

    Raises ValueError when any input directory carries an invalidation
    marker — a table built from invalidated results would silently revive
    the stale research-valid claim.
    """
    for input_dir in input_dirs:
        marker = input_dir / MARKER_FILENAME
        if marker.exists():
            raise ValueError(
                f"Refusing to read invalidated input {input_dir}: " f"{marker.read_text().strip()}"
            )

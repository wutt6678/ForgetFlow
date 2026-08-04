"""FF92-023: artifact commit provenance rules.

Every certifying run stores the five canonical provenance fields:

- ``tested_code_commit`` — the commit whose code was executed;
- ``artifact_generation_commit`` — the commit the artifacts were bound to;
- ``repository_clean`` — whether the working tree was clean at generation;
- ``workflow_run_id`` / ``workflow_attempt`` — CI run identity (empty when
  running locally).

For certification the artifact may certify only the exact code that
generated it::

    tested_code_commit == artifact_generation_commit == current HEAD
    repository_clean   == True

``validate_artifact_provenance`` returns a list of finding strings; an
empty list means the record is certifiable.  Findings cover the required
failure modes: commit mismatch, dirty suffix, unknown commit, missing
fields, and stale result directories.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.manifest import (
    COMMIT_RE,
    get_repository_commit,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "tested_code_commit",
    "artifact_generation_commit",
    "repository_clean",
    "workflow_run_id",
    "workflow_attempt",
)


def code_tree_is_clean() -> bool:
    """Whether the working tree is clean outside of result artifacts.

    Pipeline writers inevitably dirty the tree with their own output
    under ``results/``.  Certification cares about the *code* that was
    executed, so cleanliness is measured on every path except ``results/``:
    uncommitted code changes invalidate a certification, regenerated
    artifacts do not.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        # porcelain format: XY SP path ["->" SP new-path]
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if not path.startswith("results/") and path != "results":
            return False
    return True


def build_certification_provenance(
    *,
    repository_commit: str | None = None,
    repository_clean: bool | None = None,
) -> dict[str, Any]:
    """Build the FF92-023 certification provenance record.

    ``repository_commit`` may be passed as a snapshot taken before any
    artifacts were written (writers dirty the tree with their own output);
    cleanliness is then derived from that snapshot.  ``repository_clean``
    overrides the derived value — pipelines snapshot ``code_tree_is_clean()``
    at run start so their own output cannot self-invalidate the run.
    """
    raw_commit = repository_commit if repository_commit is not None else get_repository_commit()
    dirty = raw_commit.endswith("-dirty")
    commit = raw_commit.removesuffix("-dirty") if dirty else raw_commit
    clean = (not dirty) if repository_clean is None else repository_clean

    return {
        "tested_code_commit": commit,
        "artifact_generation_commit": commit,
        "repository_clean": clean,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_artifact_provenance(
    record: dict[str, Any],
    *,
    current_commit: str,
) -> list[str]:
    """Validate a provenance record against the certification rules.

    Returns finding strings; an empty list means the record certifies the
    exact code at ``current_commit``.
    """
    findings: list[str] = []

    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in record]
    if missing:
        findings.append(f"missing_provenance_fields: {sorted(missing)}")
        return findings

    tested = str(record["tested_code_commit"])
    generation = str(record["artifact_generation_commit"])

    for label, value in (
        ("tested_code_commit", tested),
        ("artifact_generation_commit", generation),
        ("current_commit", current_commit),
    ):
        clean = value.removesuffix("-dirty")
        if value != clean or clean == "unknown" or not COMMIT_RE.match(clean):
            findings.append(f"invalid_commit:{label}={value!r}")

    if tested != generation:
        findings.append(
            f"commit_mismatch: tested_code_commit={tested!r} != "
            f"artifact_generation_commit={generation!r}"
        )
    elif tested.removesuffix("-dirty") != current_commit.removesuffix("-dirty"):
        findings.append(
            f"stale_result_commit: artifact commit {tested!r} != "
            f"current commit {current_commit!r}"
        )

    if record["repository_clean"] is not True:
        findings.append("dirty_artifact: repository_clean is not true")

    return findings


def validate_result_provenance_file(
    path: Path,
    *,
    current_commit: str,
) -> list[str]:
    """Validate the provenance stored inside a result artifact file.

    A result directory is stale when its recorded artifact commit no
    longer matches the current HEAD.  Missing files and malformed records
    are findings too — silence is never a pass.
    """
    if not path.exists():
        return [f"missing_provenance_file: {path.name}"]
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"unreadable_provenance_file: {path.name} ({exc})"]

    record = data.get("provenance") if isinstance(data.get("provenance"), dict) else data
    findings = validate_artifact_provenance(record, current_commit=current_commit)
    return [f"{path.name}: {finding}" for finding in findings]

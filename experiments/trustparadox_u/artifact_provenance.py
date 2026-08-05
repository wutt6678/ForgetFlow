"""FF92-023 / remediation §32: code, generation, and storage provenance.

Every certifying run stores the five canonical provenance fields:

- ``tested_code_commit`` — the commit whose code was executed;
- ``artifact_generation_commit`` — the commit the artifacts were bound to;
- ``repository_clean`` — whether the working tree was clean at generation;
- ``workflow_run_id`` / ``workflow_attempt`` — CI run identity (``local``
  when running outside CI: provenance fields are never empty).

Remediation §32 adds the three-way identity split so a reader can
reproduce the exact code state *and* identify the commit that merely
stores generated output:

- ``tested_code_commit`` / ``artifact_generation_commit`` — the code that ran;
- ``artifact_generation_tree`` — a workspace hash over every source path
  (code, tests, data, docs, environment locks), independent of commit names;
- ``artifact_storage_commit`` — the commit that stores the artifact file,
  derived from git history at validation time (legitimately later than the
  tested commit under commit-first artifact flows).

For certification the artifact may certify only the exact code that
generated it::

    tested_code_commit == artifact_generation_commit
    repository_clean   == True
    tested_code_commit is an ancestor of the artifact's storage commit

``validate_artifact_provenance`` returns a list of finding strings; an
empty list means the record is certifiable.  Findings cover the required
failure modes: commit mismatch, dirty suffix, unknown commit, missing
fields, and stale result directories.  ``validate_three_way_provenance``
additionally enforces the §32 workspace/environment anchors required in
the reproduction manifest and release bundles.

Canonical release provenance model (PR-001)
--------------------------------------------
``COMPLETE_PROVENANCE_FIELDS`` is the canonical ten-field model every
certified release records, with fixed semantics:

- ``tested_code_commit`` — the commit whose code was executed;
- ``artifact_generation_commit`` — the commit the artifacts were bound
  to.  Normally ``tested_code_commit == artifact_generation_commit``;
  when they legitimately differ (e.g. a replay ran against an older
  checkout), a ``difference_reason`` must accompany the record;
- ``artifact_generation_tree`` — workspace hash over every source path
  (code, corpus data, annotations, thresholds, protocol documentation,
  environment locks) — never mutable storage metadata such as
  ``results/``;
- ``artifact_storage_commit`` — the *first* commit in git history that
  contains the exact stored artifact bundle (later edits of the same
  path do not change it);
- ``workflow_run_id`` / ``workflow_attempt`` — numeric CI run identity,
  or ``local`` with ``certification_source=local`` outside CI;
- ``protocol_version`` / ``study_version`` — the protocol/study the run
  certified;
- ``environment_lock_hash`` — hash of the dependency lock file;
- ``repository_clean`` — whether the code tree was clean at generation.

Because a release is stored by a commit later than the one that
generated it, releases carry a ``STORAGE_PROVENANCE.json`` sidecar
(PR-002) recording the audited lineage.  The sidecar is storage
metadata: it is excluded from the scientific release digest and tracked
by a separate ``storage_metadata_digest`` (PR-005).
"""

from __future__ import annotations

import hashlib
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

# §32: workspace anchors required in reproduction/release manifests.
THREE_WAY_FIELDS: tuple[str, ...] = (
    "artifact_generation_tree",
    "environment_lock_hash",
)

# SC-009: a complete certification provenance record — every field present
# and non-empty.  ``artifact_storage_commit`` is knowable only once the
# artifact is stored in git history, so completeness is checked against
# committed records (see ``provenance_completeness_findings``).
COMPLETE_PROVENANCE_FIELDS: tuple[str, ...] = (
    "tested_code_commit",
    "artifact_generation_commit",
    "artifact_generation_tree",
    "artifact_storage_commit",
    "workflow_run_id",
    "workflow_attempt",
    "protocol_version",
    "study_version",
    "environment_lock_hash",
    "repository_clean",
)

# Source paths hashed into the generation tree: everything that can change
# the study's behavior (code, tests, data, docs, environment locks).
SOURCE_PATHS: tuple[str, ...] = ("marble", "experiments", "scripts", "tests", "data", "doc")
SOURCE_FILES: tuple[str, ...] = ("pyproject.toml", "poetry.lock", "environment.yml")
_EXCLUDED_PARTS = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".git"})

# PR-004: the lineage fields that must agree across every release manifest
# (study/gate/reproduction/run/frozen-threshold/bundle manifests and the
# STORAGE_PROVENANCE.json sidecar).  Timestamps like ``generated_at`` are
# deliberately excluded — they legitimately differ between writers.
PROVENANCE_SYNC_FIELDS: tuple[str, ...] = (
    "tested_code_commit",
    "artifact_generation_commit",
    "artifact_generation_tree",
    "artifact_storage_commit",
    "protocol_version",
    "study_version",
    "environment_lock_hash",
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


def working_tree_is_fully_clean() -> bool:
    """Whether *nothing* (including results) is uncommitted."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
    except Exception:
        return False
    return result.returncode == 0 and not result.stdout.strip()


def generation_tree_hash() -> str:
    """§32: workspace hash over every source path.

    Hashes the content of all code, test, data, doc, and environment-lock
    files so the exact generation state is identifiable even when commit
    names alone are ambiguous.  Cache directories never contribute.
    """
    entries: list[tuple[str, str]] = []
    for rel_dir in SOURCE_PATHS:
        root = _PROJECT_ROOT / rel_dir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if _EXCLUDED_PARTS.intersection(path.relative_to(_PROJECT_ROOT).parts):
                continue
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((str(path.relative_to(_PROJECT_ROOT)), file_hash))
    for rel_file in SOURCE_FILES:
        path = _PROJECT_ROOT / rel_file
        if path.exists():
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((rel_file, file_hash))

    digest = hashlib.sha256()
    for rel_path, file_hash in sorted(entries):
        digest.update(f"{rel_path}:{file_hash}\n".encode())
    return digest.hexdigest()


def environment_lock_hash() -> str:
    """§32: SHA-256 of the dependency lock file (empty when absent)."""
    lock = _PROJECT_ROOT / "poetry.lock"
    if not lock.exists():
        return ""
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def storage_commit_for(path: Path) -> str:
    """§32: the commit that last stored ``path`` (empty when unknown).

    The storage commit legitimately differs from the tested commit: the
    commit-first flow executes code at commit T, then stores the resulting
    artifacts in a later commit S.  Readers identify S here.
    """
    try:
        rel = path.resolve().relative_to(_PROJECT_ROOT)
    except ValueError:
        return ""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(rel)],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _content_sha256_map(path: Path) -> dict[str, str]:
    """Repo-relative path -> content SHA-256 for a file or directory."""
    if path.is_file():
        rel = path.resolve().relative_to(_PROJECT_ROOT)
        return {str(rel): hashlib.sha256(path.read_bytes()).hexdigest()}
    entries: dict[str, str] = {}
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        rel = child.resolve().relative_to(_PROJECT_ROOT)
        entries[str(rel)] = hashlib.sha256(child.read_bytes()).hexdigest()
    return entries


def first_storage_commit_for(path: Path) -> str:
    """PR-001: the *first* commit that stored ``path``'s exact content.

    ``storage_commit_for`` returns the last commit touching the path;
    the canonical storage commit of a release is instead the earliest
    commit whose stored bytes match the artifact exactly — later edits
    of the same path (sidecar updates, markers) never move it.  Empty
    when the path is untracked or no commit stored identical content.
    """
    try:
        rel = path.resolve().relative_to(_PROJECT_ROOT)
    except ValueError:
        return ""
    try:
        expected = _content_sha256_map(path)
    except (OSError, ValueError):
        return ""
    if not expected:
        return ""
    try:
        result = subprocess.run(
            ["git", "log", "--reverse", "--format=%H", "--", str(rel)],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    for commit in result.stdout.split():
        matched = True
        for rel_path, digest in expected.items():
            try:
                blob = subprocess.run(
                    ["git", "show", f"{commit}:{rel_path}"],
                    capture_output=True,
                    cwd=_PROJECT_ROOT,
                )
            except Exception:
                matched = False
                break
            if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != digest:
                matched = False
                break
        if matched:
            return commit
    return ""


def commit_is_ancestor(ancestor: str, descendant: str) -> bool:
    """Whether ``ancestor`` is reachable from ``descendant`` in this repo."""
    if not (COMMIT_RE.match(ancestor) and COMMIT_RE.match(descendant)):
        return False
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            capture_output=True,
            cwd=_PROJECT_ROOT,
        )
    except Exception:
        return False
    return result.returncode == 0


def build_certification_provenance(
    *,
    repository_commit: str | None = None,
    repository_clean: bool | None = None,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Build the FF92-023 certification provenance record.

    ``repository_commit`` may be passed as a snapshot taken before any
    artifacts were written (writers dirty the tree with their own output);
    cleanliness is then derived from that snapshot.  ``repository_clean``
    overrides the derived value — pipelines snapshot ``code_tree_is_clean()``
    at run start so their own output cannot self-invalidate the run.

    SC-009: every field is populated — workflow identity falls back to
    ``local`` outside CI, protocol/study versions are always recorded, and
    ``artifact_storage_commit`` is derived from git history when an
    already-stored ``artifact_path`` is supplied.
    """
    from experiments.trustparadox_u.frozen_thresholds import STUDY_VERSION
    from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION

    raw_commit = repository_commit if repository_commit is not None else get_repository_commit()
    dirty = raw_commit.endswith("-dirty")
    commit = raw_commit.removesuffix("-dirty") if dirty else raw_commit
    clean = (not dirty) if repository_clean is None else repository_clean

    return {
        "tested_code_commit": commit,
        "artifact_generation_commit": commit,
        "repository_clean": clean,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID") or "local",
        "workflow_attempt": os.environ.get("GITHUB_RUN_ATTEMPT") or "local",
        "protocol_version": PROTOCOL_VERSION,
        "study_version": STUDY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # §32: workspace anchors; the storage commit is derived from git
        # history (it is not known before the artifact is committed).
        "artifact_generation_tree": generation_tree_hash(),
        "environment_lock_hash": environment_lock_hash(),
        "artifact_storage_commit": (
            storage_commit_for(artifact_path) if artifact_path is not None else ""
        ),
    }


def provenance_completeness_findings(record: dict[str, Any]) -> list[str]:
    """SC-009: findings for any missing or empty complete-provenance field.

    ``repository_clean`` is complete when present (its value is a boolean);
    every other field must be a non-empty string.
    """
    findings: list[str] = []
    for field in COMPLETE_PROVENANCE_FIELDS:
        if field not in record:
            findings.append(f"provenance_field_missing: {field}")
            continue
        if field == "repository_clean":
            continue
        value = record[field]
        if value is None or str(value).strip() == "":
            findings.append(f"provenance_field_empty: {field}")
    return findings


def validate_artifact_provenance(
    record: dict[str, Any],
    *,
    current_commit: str,
    artifact_path: Path | None = None,
) -> list[str]:
    """Validate a provenance record against the certification rules.

    Returns finding strings; an empty list means the record certifies the
    exact code at ``tested_code_commit``.  §32: the storage commit is
    allowed to be a descendant of the tested commit (commit-first artifact
    flow); a tested commit outside the artifact's storage lineage is a
    mismatch finding.
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
    else:
        storage = storage_commit_for(artifact_path) if artifact_path is not None else ""
        reference = storage if storage else current_commit
        if tested.removesuffix("-dirty") != reference.removesuffix("-dirty"):
            lineage_ok = bool(storage) and commit_is_ancestor(
                tested.removesuffix("-dirty"), storage
            )
            if not lineage_ok:
                findings.append(
                    f"stale_result_commit: artifact commit {tested!r} not in "
                    f"storage lineage of {reference!r}"
                )

    if record["repository_clean"] is not True:
        findings.append("dirty_artifact: repository_clean is not true")

    return findings


def validate_three_way_provenance(
    record: dict[str, Any],
    *,
    current_commit: str,
    artifact_path: Path | None = None,
) -> list[str]:
    """§32: full three-way validation for reproduction/release manifests.

    On top of the base rules this requires the generation workspace hash
    and environment lock hash, and cross-checks any recorded storage
    commit against git history.
    """
    findings = validate_artifact_provenance(
        record, current_commit=current_commit, artifact_path=artifact_path
    )
    for field in THREE_WAY_FIELDS:
        if not str(record.get(field, "")).strip():
            findings.append(f"missing_three_way_field: {field}")

    recorded_storage = str(record.get("artifact_storage_commit", "") or "")
    if recorded_storage and artifact_path is not None:
        actual_storage = storage_commit_for(artifact_path)
        if actual_storage and recorded_storage != actual_storage:
            findings.append(
                f"storage_commit_mismatch: recorded={recorded_storage!r} "
                f"actual={actual_storage!r}"
            )
    return findings


def _provenance_record(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """The provenance record inside a manifest (nested or top-level)."""
    record = manifest.get("provenance")
    if isinstance(record, dict):
        return record
    if "tested_code_commit" in manifest:
        return manifest  # STORAGE_PROVENANCE.json sidecar layout
    return None


def validate_release_provenance_consistency(
    manifests: dict[str, dict[str, Any]],
) -> list[str]:
    """PR-004: findings when release manifests record divergent lineage.

    ``manifests`` maps a manifest label (e.g. ``"study_manifest"``) to its
    parsed JSON.  Every manifest must carry a provenance record, and the
    ``PROVENANCE_SYNC_FIELDS`` values must be identical across all of
    them — any disagreement or empty lineage field is a finding.  An
    empty return list means the release's provenance is synchronized.
    """
    findings: list[str] = []
    if not manifests:
        return ["provenance_sync_no_manifests"]

    records: dict[str, dict[str, Any]] = {}
    for label, manifest in manifests.items():
        if not isinstance(manifest, dict):
            findings.append(f"provenance_sync_not_a_manifest: {label}")
            continue
        record = _provenance_record(manifest)
        if record is None:
            findings.append(f"provenance_sync_missing_record: {label}")
            continue
        records[label] = record
    if not records:
        return findings

    for field in PROVENANCE_SYNC_FIELDS:
        values: dict[str, str] = {}
        for label, record in records.items():
            value = str(record.get(field, "") or "")
            values[label] = value
            if not value.strip():
                findings.append(f"provenance_sync_empty_field: {label}.{field}")
        distinct = sorted(set(values.values()))
        if len(distinct) > 1:
            detail = ", ".join(f"{label}={value!r}" for label, value in sorted(values.items()))
            findings.append(f"provenance_sync_mismatch: {field} ({detail})")
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
    findings = validate_artifact_provenance(
        record, current_commit=current_commit, artifact_path=path
    )
    return [f"{path.name}: {finding}" for finding in findings]

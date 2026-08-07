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

Canonical release provenance model — generation/storage split (FP-001/002)
--------------------------------------------------------------------------
Scientific artifacts record *generation* provenance only — everything
knowable when the artifact was produced.  *Storage* provenance is
recorded authoritatively in the release's ``STORAGE_PROVENANCE.json``
sidecar: the storage commit is unknowable at generation time (it is the
commit that later stores the artifact), so generation records carry
``artifact_storage_commit: null`` plus a sidecar reference instead of an
empty string.

Generation fields (``GENERATION_PROVENANCE_FIELDS``):

- ``tested_code_commit`` — the commit whose code was executed; normally
  equal to ``artifact_generation_commit`` (a ``difference_reason`` is
  required when they legitimately differ);
- ``artifact_generation_commit`` — the commit the artifacts were bound to;
- ``artifact_generation_tree`` — workspace hash over every source path
  (code, corpus data, annotations, thresholds, protocol documentation,
  environment locks) — never mutable storage metadata such as
  ``results/``;
- ``workflow_run_id`` / ``workflow_attempt`` — numeric CI run identity,
  or ``local`` with ``certification_source=local`` outside CI;
- ``certification_source`` — ``ci`` or ``local``;
- ``protocol_version`` / ``study_version`` — the protocol/study the run
  certified;
- ``environment_lock_hash`` — hash of the dependency lock file;
- ``repository_clean`` — whether the code tree was clean at generation.

Storage fields (``STORAGE_PROVENANCE_FIELDS``) live in the sidecar:
``release_id``, ``tested_code_commit``, ``artifact_generation_commit``,
``artifact_storage_commit``, ``gate_evidence_commit``, ``verified_at``,
plus the ``scientific_release_digest`` / ``storage_metadata_digest``
two-digest scheme (PR-005).  The sidecar is storage metadata: it is
excluded from the scientific release digest, so updating it never forks
a release.

Gate evidence (GE-001..GE-008)
-------------------------------
``gate_evidence_commit`` is the immutable Git commit containing the
*exact gate result that certified the release* at the declared
research-status tier — not merely a commit that happens to contain a
gate file.  A failed or stale gate result can never serve as gate
evidence: ``gate_evidence_findings`` validates the historical record's
status, test/static evidence, substantive gates, generation provenance
and release binding.  ``gate_snapshot_commit`` remains a deprecated
alias; schema 1.2+ sidecars must keep both fields identical.
``gate_evidence_sha256`` is the SHA-256 of the exact bytes of
``git show <gate_evidence_commit>:results/final_artifacts/research_valid_gate.json``
(historical git content, never the working tree), so the sidecar is
cryptographically bound to the certifying gate bytes.  Both fields
participate in ``storage_metadata_digest`` but never in the scientific
release digest.
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

# FP-001/002: generation provenance — everything a scientific artifact
# can know at generation time.  Every field must be present; strings must
# be non-empty (``repository_clean`` is a boolean).
GENERATION_PROVENANCE_FIELDS: tuple[str, ...] = (
    "tested_code_commit",
    "artifact_generation_commit",
    "artifact_generation_tree",
    "workflow_run_id",
    "workflow_attempt",
    "certification_source",
    "protocol_version",
    "study_version",
    "environment_lock_hash",
    "repository_clean",
)

# FP-002: storage provenance — recorded authoritatively in the release's
# STORAGE_PROVENANCE.json sidecar, never inside scientific artifacts.
STORAGE_PROVENANCE_FIELDS: tuple[str, ...] = (
    "release_id",
    "tested_code_commit",
    "artifact_generation_commit",
    "artifact_storage_commit",
    "gate_evidence_commit",
    "verified_at",
)
STORAGE_SIDECAR_NAME = "STORAGE_PROVENANCE.json"
STORAGE_REFERENCE_KEY = "storage_provenance"

# GE-001: ``gate_snapshot_commit`` is the deprecated schema-1.1 alias of
# ``gate_evidence_commit``; schema 1.2+ sidecars must keep them identical.
GATE_SNAPSHOT_ALIAS_FIELD = "gate_snapshot_commit"
GATE_EVIDENCE_COMMIT_FIELD = "gate_evidence_commit"
# GE-002: SHA-256 over the exact historical gate bytes at the evidence
# commit (``gate_evidence_bytes``), recorded in the sidecar.
GATE_EVIDENCE_SHA256_FIELD = "gate_evidence_sha256"
GATE_EVIDENCE_FILE_REL = "results/final_artifacts/research_valid_gate.json"

# GE-006: substantive gates the historical gate evidence must show as
# passed.  ``release_storage_provenance`` is deliberately excluded:
# certifying that gate is what the evidence itself supports, and making
# the evidence depend on a storage field only knowable after the gate
# commit exists would be circular.
GATE_EVIDENCE_REQUIRED_GATES: tuple[str, ...] = (
    "repository_provenance",
    "no_invalidated_artifacts",
    "research_protocol",
    "corpus_valid",
    "annotations_valid",
    "conditions_valid",
    "replay_complete",
    "metrics_recompute",
    "leakage_analysis_valid",
    "statistical_analysis_valid",
    "trust_analysis",
    "parameter_sweep_complete",
    "frozen_threshold_manifest",
    "reproduction_manifest",
    "release_bundles",
    "deterministic_reproducibility_validation",
    "final_artifacts",
    "failure_examples",
)

# FP-010: scientific artifacts must never embed an empty-string storage
# commit — the field is either absent/null plus a sidecar reference, or
# the authoritative value lives in the sidecar itself.
STORAGE_COMMIT_FIELDS: tuple[str, ...] = (
    "artifact_storage_commit",
    "gate_snapshot_commit",
    "gate_evidence_commit",
)

# Source paths hashed into the generation tree: everything that can change
# the study's behavior (code, tests, data, docs, environment locks).
SOURCE_PATHS: tuple[str, ...] = ("marble", "experiments", "scripts", "tests", "data", "doc")
SOURCE_FILES: tuple[str, ...] = ("pyproject.toml", "poetry.lock", "environment.yml")
_EXCLUDED_PARTS = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".git"})

# FP-007: generation lineage fields that must agree across every
# scientific manifest (study/gate/reproduction/run/frozen-threshold/bundle
# manifests).  Storage fields are compared separately between the bundle
# manifest, the sidecar and the gate's storage reference
# (``validate_storage_record_consistency``).  Timestamps like
# ``generated_at`` are deliberately excluded — they legitimately differ
# between writers.
PROVENANCE_SYNC_FIELDS: tuple[str, ...] = (
    "tested_code_commit",
    "artifact_generation_commit",
    "artifact_generation_tree",
    "protocol_version",
    "study_version",
    "environment_lock_hash",
)

# FP-007: the storage-record fields that must agree between the bundle
# manifest, the STORAGE_PROVENANCE.json sidecar and the gate snapshot's
# storage reference.
STORAGE_SYNC_FIELDS: tuple[str, ...] = (
    "release_id",
    "tested_code_commit",
    "artifact_generation_commit",
    "artifact_storage_commit",
    "gate_evidence_commit",
    "scientific_release_digest",
    "storage_metadata_digest",
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


# ---------------------------------------------------------------------------
# GE-002..GE-008: historical gate evidence
# ---------------------------------------------------------------------------


class GateEvidenceError(ValueError):
    """GE-003: historical gate evidence could not be loaded.

    ``code`` is one of ``gate_evidence_commit_invalid``,
    ``gate_evidence_file_missing``, ``gate_evidence_file_empty`` or
    ``gate_evidence_json_invalid``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _resolve_gate_evidence_commit(commit: str) -> str:
    """Full SHA of ``commit``; raises ``gate_evidence_commit_invalid``."""
    value = str(commit or "").strip()
    if not value or not COMMIT_RE.match(value):
        raise GateEvidenceError(
            "gate_evidence_commit_invalid", f"unresolvable gate-evidence commit: {commit!r}"
        )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{value}^{{commit}}"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
    except Exception as exc:
        raise GateEvidenceError(
            "gate_evidence_commit_invalid", f"git failed to resolve {value!r}: {exc}"
        ) from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise GateEvidenceError(
            "gate_evidence_commit_invalid", f"git cannot resolve gate-evidence commit {value!r}"
        )
    return result.stdout.strip()


def gate_evidence_bytes(commit: str) -> bytes:
    """GE-002: exact historical gate bytes at ``commit``.

    Reads ``git show <commit>:results/final_artifacts/research_valid_gate.json``
    — never the current working-tree copy — so the digest identifies the
    gate result exactly as it was committed.
    """
    resolved = _resolve_gate_evidence_commit(commit)
    try:
        result = subprocess.run(
            ["git", "show", f"{resolved}:{GATE_EVIDENCE_FILE_REL}"],
            capture_output=True,
            cwd=_PROJECT_ROOT,
        )
    except Exception as exc:
        raise GateEvidenceError(
            "gate_evidence_file_missing", f"git show failed for {resolved}: {exc}"
        ) from exc
    if result.returncode != 0:
        raise GateEvidenceError(
            "gate_evidence_file_missing",
            f"{GATE_EVIDENCE_FILE_REL} does not exist at gate-evidence commit {resolved}",
        )
    if not result.stdout:
        raise GateEvidenceError(
            "gate_evidence_file_empty",
            f"{GATE_EVIDENCE_FILE_REL} is empty at gate-evidence commit {resolved}",
        )
    return result.stdout


def gate_evidence_sha256(commit: str) -> str:
    """GE-002: SHA-256 of the exact historical gate bytes at ``commit``."""
    return hashlib.sha256(gate_evidence_bytes(commit)).hexdigest()


def load_gate_evidence_at_commit(commit: str) -> dict[str, Any]:
    """GE-003: load and parse the historical gate result at ``commit``.

    Returns ``{"commit": <full sha>, "record": <parsed gate>, "raw":
    <exact bytes>, "sha256": <hex digest>}``.  Raises
    ``GateEvidenceError`` with one of the GE-003 failure codes when the
    commit is unresolvable, the file is missing or empty, or the bytes
    are not a JSON object.  Validation always uses historical git
    content, never the current file.
    """
    resolved = _resolve_gate_evidence_commit(commit)
    raw = gate_evidence_bytes(resolved)
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateEvidenceError(
            "gate_evidence_json_invalid",
            f"gate result at {resolved} is not valid JSON: {exc}",
        ) from exc
    if not isinstance(record, dict):
        raise GateEvidenceError(
            "gate_evidence_json_invalid",
            f"gate result at {resolved} is not a JSON object",
        )
    return {
        "commit": resolved,
        "record": record,
        "raw": raw,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def gate_evidence_findings(
    historical_gate: dict[str, Any],
    *,
    sidecar: dict[str, Any],
    bundle_manifest: dict[str, Any],
    study_class: str,
    required_status: str | None = None,
) -> list[str]:
    """GE-004..GE-008: findings when gate evidence does not certify.

    ``historical_gate`` is the parsed gate record loaded from the
    evidence commit; ``sidecar`` is the release's authoritative storage
    record and ``bundle_manifest`` the release bundle manifest.  An
    empty list means the historical gate genuinely certifies the release
    at ``required_status`` (default: ``synthetic_benchmark_valid``).

    - GE-004: the study class must match and the research status must
      meet the required tier (``research_status_at_least``); the verdict
      must equal the recorded status; the ``synthetic_benchmark_valid``
      and ``research_valid`` flags must be consistent with it.  A tier
      above the study-class ceiling (e.g. empirical status for a
      diagnostic study) never satisfies the minimum.
    - GE-005: the historical ``tests_pass`` and ``static_checks`` gates
      must be present and passed — ``not_run`` never passes.
    - GE-006: every ``GATE_EVIDENCE_REQUIRED_GATES`` entry must have
      passed in the historical gate.
    - GE-007: generation provenance must match the release exactly
      (tested/generation commits, generation tree, protocol/study
      versions, environment lock, study class).
    - GE-008: when the historical gate records ``release_id`` /
      ``scientific_release_digest`` they must match the sidecar; older
      gate schemas bind indirectly through the GE-007 comparisons.
    """
    from experiments.trustparadox_u.status import (
        EMPIRICAL_REPLAY_VALID,
        STUDY_CLASS_DIAGNOSTIC,
        SYNTHETIC_BENCHMARK_VALID,
        research_status_at_least,
    )

    minimum = required_status if required_status is not None else SYNTHETIC_BENCHMARK_VALID
    findings: list[str] = []

    # GE-004 / GE-007: study class.
    evidence_class = str(historical_gate.get("study_class", "") or "")
    if evidence_class != study_class:
        findings.append(
            f"gate_evidence_study_class_mismatch: evidence={evidence_class!r} "
            f"release={study_class!r}"
        )

    # GE-004: research status, verdict and boolean flags.
    status = str(historical_gate.get("research_status", "") or "")
    verdict = str(historical_gate.get("verdict", "") or "")
    if not research_status_at_least(status, minimum):
        findings.append(
            f"gate_evidence_status_insufficient: research_status={status!r} "
            f"required={minimum!r}"
        )
    elif research_status_at_least(status, EMPIRICAL_REPLAY_VALID) and research_status_at_least(
        SYNTHETIC_BENCHMARK_VALID, minimum
    ):
        # A higher tier satisfies the minimum only when compatible with
        # the declared study class; diagnostic studies are capped at
        # synthetic_benchmark_valid, so an empirical tier there is stale
        # or forged evidence, never a stronger certification.
        if study_class == STUDY_CLASS_DIAGNOSTIC:
            findings.append(
                f"gate_evidence_status_insufficient: research_status={status!r} "
                f"incompatible with study_class={study_class!r}"
            )
    if verdict != status:
        findings.append(f"gate_evidence_verdict_mismatch: verdict={verdict!r} status={status!r}")
    synthetic_expected = research_status_at_least(status, SYNTHETIC_BENCHMARK_VALID)
    if bool(historical_gate.get("synthetic_benchmark_valid")) != synthetic_expected:
        findings.append(
            f"gate_evidence_status_insufficient: synthetic_benchmark_valid="
            f"{historical_gate.get('synthetic_benchmark_valid')!r} inconsistent with "
            f"research_status={status!r}"
        )
    research_expected = research_status_at_least(status, EMPIRICAL_REPLAY_VALID)
    if bool(historical_gate.get("research_valid")) != research_expected:
        findings.append(
            f"gate_evidence_status_insufficient: research_valid="
            f"{historical_gate.get('research_valid')!r} inconsistent with "
            f"research_status={status!r}"
        )

    # GE-005: passing test and static-analysis evidence.
    gates = historical_gate.get("gates")
    gates = gates if isinstance(gates, dict) else {}
    for name, missing_code, failed_code in (
        ("tests_pass", "gate_evidence_test_gate_missing", "gate_evidence_tests_not_passed"),
        (
            "static_checks",
            "gate_evidence_static_gate_missing",
            "gate_evidence_static_checks_not_passed",
        ),
    ):
        entry = gates.get(name)
        if not isinstance(entry, dict):
            findings.append(missing_code)
            continue
        if entry.get("not_run"):
            findings.append(f"{failed_code}: {name}=not_run")
        elif entry.get("passed") is not True:
            findings.append(f"{failed_code}: {name}={entry.get('passed')!r}")

    # GE-006: required substantive gates (release_storage_provenance
    # excluded — the evidence is what certifies it, never the reverse).
    for name in GATE_EVIDENCE_REQUIRED_GATES:
        entry = gates.get(name)
        if not isinstance(entry, dict) or entry.get("passed") is not True:
            findings.append(f"gate_evidence_substantive_gate_failed:{name}")

    # GE-007: generation provenance must match the release exactly.
    for gate_field, sidecar_field, code in (
        (
            "tested_code_commit",
            "tested_code_commit",
            "gate_evidence_tested_commit_mismatch",
        ),
        (
            "artifact_generation_commit",
            "artifact_generation_commit",
            "gate_evidence_generation_commit_mismatch",
        ),
    ):
        evidence_value = str(historical_gate.get(gate_field, "") or "")
        release_value = str(sidecar.get(sidecar_field, "") or "")
        if evidence_value != release_value or not evidence_value:
            findings.append(f"{code}: evidence={evidence_value!r} release={release_value!r}")
    evidence_prov = historical_gate.get("provenance")
    evidence_prov = evidence_prov if isinstance(evidence_prov, dict) else {}
    manifest_prov = bundle_manifest.get("provenance")
    manifest_prov = manifest_prov if isinstance(manifest_prov, dict) else {}
    for field, code in (
        ("artifact_generation_tree", "gate_evidence_generation_tree_mismatch"),
        ("protocol_version", "gate_evidence_protocol_version_mismatch"),
        ("study_version", "gate_evidence_study_version_mismatch"),
        ("environment_lock_hash", "gate_evidence_environment_lock_mismatch"),
    ):
        evidence_value = str(evidence_prov.get(field, "") or "")
        release_value = str(manifest_prov.get(field, "") or "")
        if evidence_value != release_value or not evidence_value:
            findings.append(
                f"{code}: {field} evidence={evidence_value!r} release={release_value!r}"
            )

    # GE-008: direct release binding when the gate schema records it.
    if "release_id" in historical_gate and str(historical_gate.get("release_id", "") or "") != str(
        sidecar.get("release_id", "") or ""
    ):
        findings.append(
            f"gate_evidence_release_id_mismatch: evidence="
            f"{historical_gate.get('release_id')!r} release={sidecar.get('release_id')!r}"
        )
    if "scientific_release_digest" in historical_gate and str(
        historical_gate.get("scientific_release_digest", "") or ""
    ) != str(sidecar.get("scientific_release_digest", "") or ""):
        findings.append(
            "gate_evidence_scientific_digest_mismatch: evidence="
            f"{historical_gate.get('scientific_release_digest')!r} "
            f"release={sidecar.get('scientific_release_digest')!r}"
        )
    return findings


def storage_provenance_reference() -> dict[str, Any]:
    """FP-001: the canonical sidecar pointer carried by generation records.

    The storage commit is unknowable at generation time (it is the commit
    that later stores the artifact), so scientific artifacts never embed
    one.  They point instead at the authoritative record; ``source`` is
    resolved relative to the unique active release directory
    (``results/releases/<release_id>/STORAGE_PROVENANCE.json``) by the
    release storage provenance gate.
    """
    return {"source": STORAGE_SIDECAR_NAME, "authoritative": True}


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

    FP-001: the record is generation provenance only.  Every generation
    field is populated — workflow identity falls back to ``local`` outside
    CI, protocol/study versions are always recorded — and
    ``artifact_storage_commit`` is ``null`` with a sidecar reference: the
    authoritative storage record is written later, and updating it must
    never change the scientific digest.  ``artifact_path`` is accepted for
    signature compatibility and is no longer consulted.
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
        # PR-006: local runs must be certified as local — never dressed up
        # with CI-style workflow identity.
        "certification_source": "ci" if os.environ.get("GITHUB_RUN_ID") else "local",
        "protocol_version": PROTOCOL_VERSION,
        "study_version": STUDY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # §32: workspace anchors.
        "artifact_generation_tree": generation_tree_hash(),
        "environment_lock_hash": environment_lock_hash(),
        # FP-001: storage identity is recorded by the sidecar, never here.
        "artifact_storage_commit": None,
        STORAGE_REFERENCE_KEY: storage_provenance_reference(),
    }


def generation_provenance_findings(record: dict[str, Any]) -> list[str]:
    """FP-002: findings for an incomplete generation provenance record.

    Every ``GENERATION_PROVENANCE_FIELDS`` entry must be present;
    ``repository_clean`` is complete when present (its value is a
    boolean), every other field must be a non-empty string.  FP-010:
    embedded storage commits are never valid in a generation record — an
    empty string is a stale placeholder and a non-empty value is
    unknowable at generation time; only ``null``/absent passes.
    """
    findings: list[str] = []
    for field in GENERATION_PROVENANCE_FIELDS:
        if field not in record:
            findings.append(f"generation_provenance_field_missing: {field}")
            continue
        if field == "repository_clean":
            continue
        value = record[field]
        if value is None or str(value).strip() == "":
            findings.append(f"generation_provenance_field_empty: {field}")
    for field in STORAGE_COMMIT_FIELDS:
        if field in record and record[field] is not None:
            findings.append(f"stale_embedded_storage_commit: {field}")
    return findings


def gate_evidence_commit_of(record: dict[str, Any]) -> str:
    """GE-001: the effective gate-evidence commit of a sidecar record.

    ``gate_evidence_commit`` is primary; schema-1.1 sidecars that only
    carry the deprecated ``gate_snapshot_commit`` alias resolve to it.
    Returns the empty string when neither field is populated.
    """
    commit = str(record.get(GATE_EVIDENCE_COMMIT_FIELD, "") or "").strip()
    if not commit:
        commit = str(record.get(GATE_SNAPSHOT_ALIAS_FIELD, "") or "").strip()
    return commit


def storage_provenance_findings(
    record: dict[str, Any],
    *,
    require_gate_snapshot: bool = True,
    require_gate_evidence: bool = False,
) -> list[str]:
    """FP-002 / GE-001: findings for an incomplete storage provenance sidecar.

    Every ``STORAGE_PROVENANCE_FIELDS`` entry must be present and a
    non-empty string — no empty strings anywhere.  The gate-evidence
    commit may be pending (``null``/empty) only before the sidecar is
    finalized, which callers opt into with ``require_gate_snapshot=False``
    (the name is historical: it covers both the evidence commit and its
    deprecated alias).  ``require_gate_evidence=True`` additionally
    requires the GE-002 ``gate_evidence_sha256`` digest.  The two digest
    fields are recommended; when present they must be non-empty.

    GE-001 schema 1.2+ compatibility: when the deprecated
    ``gate_snapshot_commit`` alias coexists with ``gate_evidence_commit``
    the two must name the same commit, and a ``schema_version`` — when
    recorded — must be a non-empty string.
    """
    findings: list[str] = []
    pending_fields = {GATE_EVIDENCE_COMMIT_FIELD, GATE_SNAPSHOT_ALIAS_FIELD}
    for field in STORAGE_PROVENANCE_FIELDS:
        if field not in record:
            if (
                field == GATE_EVIDENCE_COMMIT_FIELD
                and str(record.get(GATE_SNAPSHOT_ALIAS_FIELD, "") or "").strip()
            ):
                # GE-001 compatibility: a schema-1.1 sidecar that records
                # only the deprecated alias is complete; schema 1.2+
                # sidecars must carry the primary field itself.
                continue
            findings.append(f"storage_provenance_field_missing: {field}")
            continue
        value = record[field]
        if field in pending_fields and not require_gate_snapshot:
            if value is not None and str(value).strip() != "" and not COMMIT_RE.match(str(value)):
                findings.append(f"storage_provenance_field_invalid: {field}")
            continue
        if value is None or str(value).strip() == "":
            if (
                field == GATE_EVIDENCE_COMMIT_FIELD
                and str(record.get(GATE_SNAPSHOT_ALIAS_FIELD, "") or "").strip()
            ):
                continue  # the alias still identifies the evidence
            findings.append(f"storage_provenance_field_empty: {field}")
    digest = record.get(GATE_EVIDENCE_SHA256_FIELD)
    if require_gate_evidence:
        if digest is None or str(digest).strip() == "":
            findings.append(f"storage_provenance_field_empty: {GATE_EVIDENCE_SHA256_FIELD}")
    elif digest is not None and str(digest).strip() == "":
        findings.append(f"storage_provenance_field_empty: {GATE_EVIDENCE_SHA256_FIELD}")
    for field in ("scientific_release_digest", "storage_metadata_digest"):
        value = record.get(field)
        if value is not None and str(value).strip() == "":
            findings.append(f"storage_provenance_field_empty: {field}")
    schema_version = record.get("schema_version")
    if schema_version is not None and str(schema_version).strip() == "":
        findings.append("storage_provenance_field_empty: schema_version")
    snapshot = str(record.get(GATE_SNAPSHOT_ALIAS_FIELD, "") or "").strip()
    evidence = str(record.get(GATE_EVIDENCE_COMMIT_FIELD, "") or "").strip()
    if snapshot and evidence and snapshot != evidence:
        findings.append(
            f"gate_snapshot_alias_mismatch: {GATE_SNAPSHOT_ALIAS_FIELD}={snapshot!r} "
            f"!= {GATE_EVIDENCE_COMMIT_FIELD}={evidence!r}"
        )
    return findings


def validate_storage_reference(record: dict[str, Any]) -> list[str]:
    """FP-001: a generation record's sidecar pointer must be well-formed."""
    reference = record.get(STORAGE_REFERENCE_KEY)
    if not isinstance(reference, dict):
        return [f"storage_provenance_reference_missing: {STORAGE_REFERENCE_KEY}"]
    findings: list[str] = []
    if not str(reference.get("source", "") or "").strip():
        findings.append("storage_provenance_reference_source_empty")
    if reference.get("authoritative") is not True:
        findings.append("storage_provenance_reference_not_authoritative")
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
    """FP-007: findings when scientific manifests record divergent lineage.

    ``manifests`` maps a manifest label (e.g. ``"study_manifest"``) to its
    parsed JSON.  Every manifest must carry a provenance record, and the
    generation ``PROVENANCE_SYNC_FIELDS`` values must be identical across
    all of them — any disagreement or empty lineage field is a finding.
    Storage fields are deliberately excluded: they are recorded once, in
    the sidecar, and compared by ``validate_storage_record_consistency``.
    An empty return list means the generation provenance is synchronized.
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


def validate_storage_record_consistency(
    records: dict[str, dict[str, Any]],
) -> list[str]:
    """FP-007: findings when storage records disagree with each other.

    ``records`` maps a source label (e.g. ``"bundle_manifest"``,
    ``"storage_sidecar"``, ``"gate_snapshot"``) to its storage record.
    Every ``STORAGE_SYNC_FIELDS`` value present in more than one record
    must be identical.  ``None``/absent normalizes to the empty string so
    a pending field is comparable; emptiness itself is reported by
    ``storage_provenance_findings`` against the authoritative sidecar.
    """
    findings: list[str] = []
    if not records:
        return ["storage_sync_no_records"]
    for field in STORAGE_SYNC_FIELDS:
        values: dict[str, str] = {}
        for label, record in records.items():
            if not isinstance(record, dict) or field not in record:
                continue
            values[label] = str(record.get(field) or "")
        distinct = sorted(set(values.values()))
        if len(distinct) > 1:
            detail = ", ".join(f"{label}={value!r}" for label, value in sorted(values.items()))
            findings.append(f"storage_sync_mismatch: {field} ({detail})")
    return findings


def validate_release_lineage(
    record: dict[str, Any],
    *,
    certification_source: str | None = None,
) -> list[str]:
    """PR-006: release-level provenance validation rules.

    Enforced rules:

    1. the generation commit must be an ancestor of the storage commit;
    2. ``tested_code_commit == artifact_generation_commit`` unless the
       record carries a non-empty ``difference_reason``;
    5. the recorded protocol must be the current research protocol;
    6. ``artifact_storage_commit``, ``protocol_version``,
       ``study_version``, ``artifact_generation_tree`` and
       ``environment_lock_hash`` must never be empty;
    7. CI certification needs numeric workflow run/attempt identity; a
       local run needs ``certification_source == "local"``.

    Rules 3-4 (the storage commit must contain the exact bundle) are
    checked by ``release_bundle.validate_bundle_at_storage_commit``.
    Returns finding strings; an empty list means the lineage certifies.
    """
    findings: list[str] = []
    tested = str(record.get("tested_code_commit", "") or "")
    generation = str(record.get("artifact_generation_commit", "") or "")
    storage = str(record.get("artifact_storage_commit", "") or "")

    for field in (
        "artifact_storage_commit",
        "protocol_version",
        "study_version",
        "artifact_generation_tree",
        "environment_lock_hash",
    ):
        if not str(record.get(field, "") or "").strip():
            findings.append(f"release_provenance_empty_field: {field}")

    if (
        tested
        and generation
        and tested != generation
        and not str(record.get("difference_reason", "") or "").strip()
    ):
        findings.append(
            f"tested_generation_mismatch_without_reason: "
            f"tested={tested!r} generation={generation!r}"
        )

    if generation and storage:
        if COMMIT_RE.match(generation) and COMMIT_RE.match(storage):
            if not commit_is_ancestor(generation, storage):
                findings.append(
                    f"lineage_not_ancestor: generation={generation!r} storage={storage!r}"
                )
        else:
            findings.append(
                f"lineage_unknown_commit: generation={generation!r} storage={storage!r}"
            )

    protocol = str(record.get("protocol_version", "") or "")
    if protocol:
        from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION

        if protocol != PROTOCOL_VERSION:
            findings.append(
                f"protocol_version_not_current: recorded={protocol!r} "
                f"current={PROTOCOL_VERSION!r}"
            )

    run_id = str(record.get("workflow_run_id", "") or "")
    attempt = str(record.get("workflow_attempt", "") or "")
    source = (
        certification_source
        if certification_source is not None
        else str(record.get("certification_source", "") or "")
    )
    if run_id.isdigit() and attempt.isdigit():
        pass  # numeric CI evidence
    elif run_id == "local" and attempt == "local":
        if source != "local":
            findings.append("local_certification_source_missing")
    else:
        findings.append(f"workflow_identity_invalid: run_id={run_id!r} attempt={attempt!r}")

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

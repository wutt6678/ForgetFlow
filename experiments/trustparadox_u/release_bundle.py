"""Remediation §34: immutable released research bundles.

Each release preserves, under one unique identifier, everything needed
to audit the study exactly as it was certified:

- protocol version and code commit (three-way provenance, §32);
- corpus and annotations (bound by content hash, never copied);
- condition configurations, raw replay trials, statistical outputs and
  the final tables;
- the reproduction manifest that proves the artifacts regenerate.

``release_id = trustparadox_u-v{study_version}-{digest[:12]}`` where the
digest hashes the canonical component manifest — identical content can
never receive two identifiers, and any change to any component yields a
new release.  Building a new release supersedes the previous active
one: the old bundle moves to ``results/archive/<release_id>/`` with an
invalidation marker, so superseded studies stay auditable but can never
be mistaken for the current release.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.manifest import COMMIT_RE

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = _PROJECT_ROOT / "results"
RELEASES_DIR = RESULTS_DIR / "releases"
ARCHIVE_DIR = RESULTS_DIR / "archive"
CORPUS_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "frozen_corpus"

SCHEMA_VERSION = "1.1"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
STORAGE_PROVENANCE_NAME = "STORAGE_PROVENANCE.json"
STORAGE_PROVENANCE_SCHEMA_VERSION = "1.0"
SUPERSEDED_MARKER_NAME = "INVALIDATION_MARKER.json"

# PR-003: schema 1.1 bundles enforce complete, synchronized provenance —
# non-empty top-level versions matching the nested provenance record and
# a non-empty artifact_storage_commit.  Schema 1.0 bundles (historical)
# keep validating under the old, weaker rules.
_LEGACY_SCHEMA_VERSIONS = frozenset({"", "1.0"})

# (results-relative path, role) — every artifact a release must preserve.
BUNDLE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("frozen_config/frozen_threshold_manifest.json", "frozen condition configurations"),
    ("reproduction/reproduction_manifest.json", "reproduction manifest and provenance"),
    ("frozen_replay/run_manifest.json", "raw replay run manifest"),
    ("frozen_replay/candidate_trials.jsonl", "raw replay trials"),
    ("frozen_replay/reconstruction_trials.jsonl", "raw reconstruction trials"),
    ("frozen_replay/recontamination_trials.jsonl", "raw recontamination trials"),
    ("frozen_replay/utility_trials.jsonl", "raw utility trials"),
    ("frozen_replay/resolved_conditions.json", "resolved condition configurations"),
    ("frozen_replay/metrics_by_condition.json", "metrics by condition"),
    ("leakage_analysis/leakage_analysis.json", "leakage analysis"),
    ("paired_statistics/paired_statistics.json", "statistical outputs"),
    ("trust_analysis/trust_analysis.json", "trust invariance and trust-manipulation analysis"),
    ("trust_analysis/pairing_audit.json", "trust-analysis family pairing audit"),
    ("parameter_sweep/sweep_summary.json", "parameter sweep summary"),
    ("final_artifacts/study_manifest.json", "final study manifest"),
    ("final_artifacts/table1_main_results.json", "final table 1: main results"),
    ("final_artifacts/table2_leakage_breakdown.json", "final table 2: leakage breakdown"),
    ("final_artifacts/table3_parameter_sensitivity.json", "final table 3: parameter sensitivity"),
    (
        "final_artifacts/table4_statistical_comparisons.json",
        "final table 4: statistical comparisons",
    ),
    (
        "final_artifacts/table5_target_type_results.json",
        "final table 5: results by target type and scenario",
    ),
    (
        "final_artifacts/table6_trust_analysis.json",
        "final table 6: trust invariance and trust-manipulation analysis",
    ),
    (
        "failure_examples/failure_examples.json",
        "curated failure examples and decision traces",
    ),
)


class ReleaseError(RuntimeError):
    """Raised when a release bundle cannot be built or validated."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def corpus_annotation_binding() -> dict[str, Any]:
    """Bind the release to the frozen corpus/annotations by content hash."""
    corpus_manifest_path = CORPUS_DIR / "corpus_manifest.json"
    annotation_manifest_path = CORPUS_DIR / "annotation_manifest.json"
    if not corpus_manifest_path.exists() or not annotation_manifest_path.exists():
        raise ReleaseError("frozen corpus or annotation manifest missing")
    corpus_manifest = _load_json(corpus_manifest_path)
    annotation_manifest = _load_json(annotation_manifest_path)
    return {
        "corpus_sha256": corpus_manifest.get("corpus_sha256", ""),
        "corpus_manifest": str(corpus_manifest_path.relative_to(_PROJECT_ROOT)),
        "annotation_hash": annotation_manifest.get("annotation_hash", ""),
        "annotation_manifest": str(annotation_manifest_path.relative_to(_PROJECT_ROOT)),
    }


def release_digest(manifest: dict[str, Any]) -> str:
    """Canonical digest over everything that defines a release's content.

    PR-005: this is the *scientific* release digest.  Storage metadata —
    the ``STORAGE_PROVENANCE.json`` sidecar — is never a component and
    can never change it, so sidecar updates never fork a release.
    """
    canonical = {
        "study_version": manifest["study_version"],
        "corpus_sha256": manifest["corpus"]["corpus_sha256"],
        "annotation_hash": manifest["annotations"]["annotation_hash"],
        "components": {
            rel: component["sha256"] for rel, component in manifest["components"].items()
        },
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def scientific_release_digest(manifest: dict[str, Any]) -> str:
    """PR-005: the scientific content digest of a release.

    Identical to ``release_digest`` — named explicitly so the two-digest
    scheme (scientific content vs. storage metadata) reads unambiguously.
    """
    return release_digest(manifest)


def storage_metadata_digest(sidecar: dict[str, Any]) -> str:
    """PR-005: digest of the STORAGE_PROVENANCE.json sidecar content.

    ``verified_at`` is audit bookkeeping, not lineage, and is excluded so
    re-auditing identical lineage never changes the digest.
    """
    canonical = {key: value for key, value in sidecar.items() if key != "verified_at"}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_storage_provenance(
    bundle_dir: Path,
    *,
    artifact_storage_commit: str = "",
    gate_snapshot_commit: str = "",
    verified_at: str | None = None,
) -> dict[str, Any]:
    """PR-002: audit a release's lineage and store the sidecar record.

    Writes ``STORAGE_PROVENANCE.json`` next to the bundle manifest and
    syncs ``storage_metadata_digest`` into the manifest.  Commits are
    taken from the audited lineage (git history), never from the current
    checkout: the sidecar records what *was* run and stored, not what is
    checked out now.
    """
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    if not manifest_path.exists():
        raise ReleaseError(f"bundle manifest missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    provenance = manifest.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    storage_commit = artifact_storage_commit or str(
        provenance.get("artifact_storage_commit", "") or ""
    )
    sidecar: dict[str, Any] = {
        "schema_version": STORAGE_PROVENANCE_SCHEMA_VERSION,
        "release_id": str(manifest.get("release_id", "")),
        "tested_code_commit": str(provenance.get("tested_code_commit", "") or ""),
        "artifact_generation_commit": str(provenance.get("artifact_generation_commit", "") or ""),
        "artifact_storage_commit": storage_commit,
        "gate_snapshot_commit": gate_snapshot_commit,
        "verified_at": verified_at or datetime.now(timezone.utc).isoformat(),
    }
    (bundle_dir / STORAGE_PROVENANCE_NAME).write_text(json.dumps(sidecar, indent=2) + "\n")
    manifest["storage_metadata_digest"] = storage_metadata_digest(sidecar)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return sidecar


def build_release_manifest() -> dict[str, Any]:
    """Assemble the manifest for the release implied by current artifacts.

    Requires a passing §33 reproduction manifest: a release without a
    demonstrated reproduction is not certifiable.
    """
    reproduction_path = RESULTS_DIR / "reproduction" / "reproduction_manifest.json"
    if not reproduction_path.exists():
        raise ReleaseError(
            f"reproduction manifest missing: {reproduction_path} — "
            "run python -m experiments.trustparadox_u.reproduce first"
        )
    reproduction = _load_json(reproduction_path)
    if reproduction.get("passed") is not True:
        raise ReleaseError("reproduction manifest did not pass; refusing to release")

    repro_provenance = reproduction.get("provenance")
    repro_provenance = repro_provenance if isinstance(repro_provenance, dict) else {}
    study_version = str(
        reproduction.get("study_version") or repro_provenance.get("study_version", "") or ""
    )
    protocol_version = str(
        reproduction.get("protocol_version") or repro_provenance.get("protocol_version", "") or ""
    )

    components: dict[str, dict[str, str]] = {}
    for rel, role in BUNDLE_COMPONENTS:
        source = RESULTS_DIR / rel
        if not source.exists():
            raise ReleaseError(f"bundle component missing: {rel}")
        components[rel] = {"role": role, "sha256": _sha256(source)}

    binding = corpus_annotation_binding()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_version": study_version,
        "protocol_version": protocol_version,
        "provenance": reproduction.get("provenance", {}),
        "corpus": {
            "corpus_sha256": binding["corpus_sha256"],
            "manifest": binding["corpus_manifest"],
        },
        "annotations": {
            "annotation_hash": binding["annotation_hash"],
            "manifest": binding["annotation_manifest"],
        },
        "components": components,
        "status": "active",
        "superseded_by": "",
    }
    digest = release_digest(manifest)
    manifest["release_id"] = f"trustparadox_u-v{manifest['study_version']}-{digest[:12]}"
    manifest["release_digest"] = digest
    # PR-005: the two-digest scheme — the scientific digest above defines
    # the release; the storage metadata digest is recorded once the
    # STORAGE_PROVENANCE.json sidecar has been audited and written.
    manifest["scientific_release_digest"] = digest
    manifest["storage_metadata_digest"] = ""
    return manifest


def validate_release_bundle(bundle_dir: Path, *, allow_pending_storage: bool = False) -> list[str]:
    """Recompute every component hash recorded in a bundle's manifest.

    PR-003: schema 1.1 bundles additionally require non-empty top-level
    ``study_version``/``protocol_version`` matching the nested provenance
    record, complete provenance lineage (including a non-empty
    ``artifact_storage_commit``), and a reproducible
    ``storage_metadata_digest`` when the sidecar is present.
    ``allow_pending_storage`` exempts the storage-commit requirement for
    bundles validated immediately after creation, before the commit that
    stores them exists.
    """
    findings: list[str] = []
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    if not manifest_path.exists():
        return [f"{bundle_dir.name}: bundle_manifest.json missing"]
    manifest = _load_json(manifest_path)

    for field in ("release_id", "study_version", "provenance", "corpus", "annotations"):
        if field not in manifest:
            findings.append(f"{bundle_dir.name}: missing field {field}")
    digest = manifest.get("release_digest", "")
    if digest and manifest.get("release_id") != (
        f"trustparadox_u-v{manifest.get('study_version', '')}-{digest[:12]}"
    ):
        findings.append(f"{bundle_dir.name}: release_id inconsistent with digest")
    if digest:
        try:
            recomputed = release_digest(manifest)
        except (KeyError, TypeError):
            recomputed = ""
        if recomputed and recomputed != digest:
            findings.append(f"{bundle_dir.name}: release_digest not reproducible")

    if str(manifest.get("schema_version", "")) not in _LEGACY_SCHEMA_VERSIONS:
        findings.extend(
            _validate_bundle_provenance_fields(bundle_dir.name, manifest, allow_pending_storage)
        )

    sidecar_path = bundle_dir / STORAGE_PROVENANCE_NAME
    if sidecar_path.exists():
        sidecar = _load_json(sidecar_path)
        if sidecar.get("release_id") != manifest.get("release_id"):
            findings.append(f"{bundle_dir.name}: storage provenance release_id mismatch")
        recorded = str(manifest.get("storage_metadata_digest", "") or "")
        if recorded and recorded != storage_metadata_digest(sidecar):
            findings.append(f"{bundle_dir.name}: storage_metadata_digest not reproducible")

    for rel, component in manifest.get("components", {}).items():
        path = bundle_dir / rel
        if not path.exists():
            findings.append(f"{bundle_dir.name}: component missing {rel}")
            continue
        if _sha256(path) != component.get("sha256"):
            findings.append(f"{bundle_dir.name}: component hash mismatch {rel}")
    return findings


def _validate_bundle_provenance_fields(
    bundle_name: str,
    manifest: dict[str, Any],
    allow_pending_storage: bool,
) -> list[str]:
    """PR-003: strict provenance completeness for schema 1.1 bundles."""
    findings: list[str] = []
    provenance = manifest.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}

    for field in ("study_version", "protocol_version"):
        top_level = str(manifest.get(field, "") or "").strip()
        if not top_level:
            findings.append(f"{bundle_name}: empty top-level {field}")
        elif str(provenance.get(field, "") or "") != manifest.get(field):
            findings.append(f"{bundle_name}: top-level {field} != provenance.{field}")

    if (
        not allow_pending_storage
        and not str(provenance.get("artifact_storage_commit", "") or "").strip()
    ):
        findings.append(f"{bundle_name}: empty provenance.artifact_storage_commit")
    for field in (
        "tested_code_commit",
        "artifact_generation_commit",
        "artifact_generation_tree",
        "environment_lock_hash",
    ):
        if not str(provenance.get(field, "") or "").strip():
            findings.append(f"{bundle_name}: empty provenance.{field}")
    return findings


def release_dirs() -> list[Path]:
    if not RELEASES_DIR.exists():
        return []
    return sorted(d for d in RELEASES_DIR.iterdir() if (d / BUNDLE_MANIFEST_NAME).exists())


def validate_bundle_at_storage_commit(
    bundle_dir: Path,
    manifest: dict[str, Any] | None = None,
    *,
    storage_commit: str = "",
) -> list[str]:
    """PR-006 rules 3-4: the storage commit must contain the exact bundle.

    Every component must exist at the recorded ``artifact_storage_commit``
    with byte-identical content, and the bundle manifest itself must exist
    there.  The manifest is compared by presence only: the sidecar digest
    field is legitimately added after the storage commit.
    """
    findings: list[str] = []
    if manifest is None:
        manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
        if not manifest_path.exists():
            return [f"{bundle_dir.name}: bundle_manifest.json missing"]
        manifest = _load_json(manifest_path)
    provenance = manifest.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    commit = storage_commit or str(provenance.get("artifact_storage_commit", "") or "")
    if not commit or not COMMIT_RE.match(commit):
        return [f"{bundle_dir.name}: storage commit missing or invalid: {commit!r}"]
    try:
        bundle_rel = bundle_dir.resolve().relative_to(_PROJECT_ROOT)
    except ValueError:
        return [f"{bundle_dir.name}: bundle outside repository, cannot verify storage commit"]

    def blob_digest(rel: str) -> str | None:
        try:
            blob = subprocess.run(
                ["git", "show", f"{commit}:{rel}"],
                capture_output=True,
                cwd=_PROJECT_ROOT,
            )
        except Exception:
            return None
        if blob.returncode != 0:
            return None
        return hashlib.sha256(blob.stdout).hexdigest()

    manifest_rel = str(bundle_rel / BUNDLE_MANIFEST_NAME)
    if blob_digest(manifest_rel) is None:
        findings.append(f"{bundle_dir.name}: storage commit missing manifest {manifest_rel}")
    for rel, component in manifest.get("components", {}).items():
        digest = blob_digest(str(bundle_rel / rel))
        if digest is None:
            findings.append(f"{bundle_dir.name}: storage commit missing component {rel}")
        elif digest != component.get("sha256"):
            findings.append(f"{bundle_dir.name}: storage commit checksum mismatch {rel}")
    return findings


def supersede_release(
    bundle_dir: Path,
    *,
    superseded_by: str,
    reason: str | list[str] | None = None,
) -> None:
    """Archive a superseded release with an invalidation marker.

    The bundle moves wholesale into ``results/archive/<release_id>/`` so
    it stays auditable; markers inside the archive never fail the
    invalidation gate.  ``reason`` may document the supersession as a
    single string or a list of distinct causes (PR-009).
    """
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    manifest = _load_json(manifest_path)
    manifest["status"] = "superseded"
    manifest["superseded_by"] = superseded_by
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    target = ARCHIVE_DIR / bundle_dir.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(bundle_dir), str(target))
    (target / SUPERSEDED_MARKER_NAME).write_text(
        json.dumps(
            {
                "status": "superseded",
                "superseded_by": superseded_by,
                "reason": (
                    reason if reason is not None else f"superseded by release {superseded_by}"
                ),
            },
            indent=2,
        )
        + "\n"
    )


def build_release_bundle() -> dict[str, Any]:
    """Build the current release bundle, superseding any active release."""
    manifest = build_release_manifest()
    release_id = manifest["release_id"]
    bundle_dir = RELEASES_DIR / release_id

    for existing in release_dirs():
        existing_manifest = _load_json(existing / BUNDLE_MANIFEST_NAME)
        if existing_manifest.get("release_id") == release_id:
            manifest = existing_manifest  # identical content: already released
            return {"release_id": release_id, "status": manifest["status"], "new": False}
        if existing_manifest.get("status") == "active":
            supersede_release(existing, superseded_by=release_id)

    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for rel in manifest["components"]:
        source = RESULTS_DIR / rel
        target = bundle_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (bundle_dir / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    findings = validate_release_bundle(bundle_dir, allow_pending_storage=True)
    if findings:
        raise ReleaseError(f"bundle self-validation failed: {findings}")
    return {"release_id": release_id, "status": "active", "new": True}


def main() -> int:
    """Build (or confirm) the immutable release bundle (§34)."""
    print("Remediation §34: Immutable Release Bundle")
    print("=" * 50)
    try:
        result = build_release_bundle()
    except ReleaseError as exc:
        print(f"RELEASE FAILED: {exc}")
        return 1
    action = "created" if result["new"] else "already released"
    print(f"  release id: {result['release_id']}")
    print(f"  status:     {result['status']} ({action})")
    print(f"  bundle:     {RELEASES_DIR / result['release_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

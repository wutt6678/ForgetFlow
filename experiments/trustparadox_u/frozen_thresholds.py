"""Frozen configuration manifest (remediation §29/§30).

Every model, threshold, prompt, annotation, and policy decision that
feeds the primary analysis is frozen *before* the final test
evaluation and recorded in one committed manifest:

- swept thresholds carry their selection sweep (split, selection rule,
  tie-breaking rule, selected value);
- unswept behavioral parameters are recorded as fixed defaults;
- scenario definitions, candidate-generation prompt hashes, annotation
  manifest hash, metric code commit, and the primary hypotheses
  (versioned research protocol) are anchored by hash or version.

Freeze discipline (§29 acceptance criteria):

- the manifest is committed before the test results it governs;
- the final test split is evaluated once for the primary analysis;
- any rerun after code or protocol changes bumps ``STUDY_VERSION``;
- post-test fixes invalidate or version the previous result rather
  than silently replacing it.

Sweep purpose labels (§30): every parameter sweep is labelled either
``selection`` (development/validation only — its chosen value is
frozen) or ``sensitivity`` (post hoc, exploratory, never used to
choose the main reported result).  A selection sweep may never touch
the test split.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Bump whenever a frozen decision changes or a rerun follows code or
# protocol changes (§29: reruns create a new study version).
STUDY_VERSION = "1.2.1"

MANIFEST_SCHEMA_VERSION = "1.0"

RESULTS_DIR = _PROJECT_ROOT / "results"
FROZEN_CONFIG_DIR = RESULTS_DIR / "frozen_config"
FROZEN_MANIFEST_PATH = FROZEN_CONFIG_DIR / "frozen_threshold_manifest.json"
SWEEP_SUMMARY_PATH = RESULTS_DIR / "parameter_sweep" / "sweep_summary.json"
SCENARIO_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "scenarios"
CORPUS_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "frozen_corpus"

# §30: the only two legitimate purposes for a parameter sweep.
SELECTION_PURPOSE = "selection"
SENSITIVITY_PURPOSE = "sensitivity"
VALID_SWEEP_PURPOSES = frozenset({SELECTION_PURPOSE, SENSITIVITY_PURPOSE})

# Behavioral parameter families that must be frozen before the final
# test evaluation (§29 required change list).
REQUIRED_FROZEN_FAMILIES: tuple[str, ...] = (
    "detector.embedding_threshold",
    "detector.claim_confidence_threshold",
    "history.window_size",
    "history.reconstruction_threshold",
    "monitoring.duration_rounds",
    "monitoring.continuous",
    "policy.privacy_utility_weight",
    "policy.rich_actions_enabled",
    "policy.trust_independent",
)

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Frozen policy declarations (§29 acceptance criteria), recorded in the
# manifest so the discipline is auditable rather than remembered.
FREEZE_POLICIES: dict[str, Any] = {
    "final_test_evaluated_once": (
        "The final test split is evaluated exactly once for the primary "
        "analysis (the frozen-config evaluation at the end of the sweep)."
    ),
    "rerun_versioning_rule": (
        "Any rerun after code or protocol changes creates a new study "
        "version: STUDY_VERSION is bumped and the previous manifest stays "
        "committed alongside the results it governed."
    ),
    "post_test_fix_rule": (
        "Post-test fixes invalidate or version the previous result rather "
        "than silently replacing it; superseded artifacts are archived with "
        "an invalidation marker, never overwritten in place."
    ),
    "selection_split_policy": (
        "Parameter selection never uses test-split performance; selection "
        "sweeps run on development/validation only (§30)."
    ),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sweep_summary(path: Path | None = None) -> dict[str, Any]:
    summary_path = path if path is not None else SWEEP_SUMMARY_PATH
    data: dict[str, Any] = json.loads(summary_path.read_text())
    return data


# ---------------------------------------------------------------------------
# Parameter entries
# ---------------------------------------------------------------------------


def sweep_selection_index(sweep_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map config path -> the selection-sweep record that chose it."""
    index: dict[str, dict[str, Any]] = {}
    for name, sweep in sweep_summary.get("sweeps", {}).items():
        config_path = str(sweep.get("config_path", ""))
        if not config_path or "selected_value" not in sweep:
            continue
        index[config_path] = {
            "sweep": name,
            "sweep_purpose": sweep.get("sweep_purpose", SELECTION_PURPOSE),
            "selection_split": sweep.get("split"),
            "selection_rule": sweep.get("selection_rule", ""),
            "selected_value": sweep.get("selected_value"),
        }
    return index


def build_frozen_parameters(sweep_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One entry per frozen behavioral parameter (§29).

    Values come from the full-MVP base with every sweep selection
    applied; entries record whether the value was chosen by a selection
    sweep (with split and tie-breaking rule) or fixed by default.
    """
    from experiments.trustparadox_u.conditions import full_mvp_config

    base = full_mvp_config()
    selections = sweep_selection_index(sweep_summary)
    entries: dict[str, dict[str, Any]] = {}
    for section_name in ("detector", "history", "policy", "monitoring"):
        section = getattr(base, section_name)
        for field in dataclasses.fields(section):
            config_path = f"{section_name}.{field.name}"
            entry: dict[str, Any] = {
                "config_path": config_path,
                "value": getattr(section, field.name),
            }
            selection = selections.get(config_path)
            if selection is not None:
                entry["value"] = selection["selected_value"]
                entry["source"] = "selection_sweep"
                entry["sweep"] = selection["sweep"]
                entry["sweep_purpose"] = selection["sweep_purpose"]
                entry["selection_split"] = selection["selection_split"]
                entry["selection_rule"] = selection["selection_rule"]
                entry["swept_selected_value"] = selection["selected_value"]
            else:
                entry["source"] = "fixed_default"
                entry["rationale"] = (
                    "Not swept; frozen at the full-MVP default before the " "final test evaluation."
                )
            entries[config_path] = entry
    return entries


# ---------------------------------------------------------------------------
# Non-parameter anchors (scenarios, prompts, annotations, protocol)
# ---------------------------------------------------------------------------


def scenario_definition_hashes(scenario_dir: Path | None = None) -> dict[str, str]:
    """SHA-256 of every committed scenario definition file (§29)."""
    directory = scenario_dir if scenario_dir is not None else SCENARIO_DIR
    hashes: dict[str, str] = {}
    if not directory.exists():
        return hashes
    for path in sorted(directory.glob("*.yaml")):
        hashes[path.name] = _sha256_file(path)
    return hashes


def corpus_freeze_entry(corpus_dir: Path | None = None) -> dict[str, Any]:
    """Corpus anchors: generation prompts and annotation instructions (§29).

    Candidate-generation prompts are pinned by the corpus manifest's
    prompt template hashes; annotation instructions are pinned by the
    annotation manifest hash.
    """
    directory = corpus_dir if corpus_dir is not None else CORPUS_DIR
    entry: dict[str, Any] = {}
    manifest_path = directory / "corpus_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        entry["corpus_version"] = manifest.get("corpus_version")
        entry["corpus_sha256"] = manifest.get("corpus_sha256")
        entry["generation_model"] = manifest.get("generation_model")
        entry["prompt_template_hashes"] = manifest.get("prompt_template_hashes", {})
    annotation_path = directory / "annotation_manifest.json"
    if annotation_path.exists():
        entry["annotation_manifest_sha256"] = _sha256_file(annotation_path)
    return entry


def protocol_freeze_entry() -> dict[str, Any]:
    """Primary hypotheses frozen as the versioned research protocol (§29)."""
    from experiments.trustparadox_u.research_protocol import (
        PROTOCOL_VERSION,
        QUESTIONS,
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "primary_hypotheses": [
            {"question_id": q.question_id, "statement": q.statement} for q in QUESTIONS
        ],
    }


# ---------------------------------------------------------------------------
# Manifest assembly and validation
# ---------------------------------------------------------------------------


def build_frozen_threshold_manifest(
    sweep_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the frozen configuration manifest (§29).

    The sweep summary supplies the selection-sweep provenance; when
    omitted it is loaded from the committed artifact.
    """
    from experiments.trustparadox_u.parameter_sweep import build_frozen_config

    summary = sweep_summary if sweep_summary is not None else load_sweep_summary()
    selected = {
        spec_path: value
        for spec_path, value in summary.get("frozen_config", {}).get("selected_values", {}).items()
    }
    frozen_config = build_frozen_config(selected) if selected else None

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "remediation_items": ["29", "30"],
        "study_version": STUDY_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "base_condition": summary.get("base_condition", "full_mvp"),
        "parameters": build_frozen_parameters(summary),
        "frozen_config_hashes": {
            "config_hash": frozen_config.config_hash() if frozen_config else None,
            "condition_hash": frozen_config.condition_hash() if frozen_config else None,
        },
        "sweep_purpose_labels": {
            name: sweep.get("sweep_purpose", SELECTION_PURPOSE)
            for name, sweep in summary.get("sweeps", {}).items()
        },
        "freeze_policies": FREEZE_POLICIES,
        "scenario_definitions": scenario_definition_hashes(),
        "corpus": corpus_freeze_entry(),
        "protocol": protocol_freeze_entry(),
    }


def validate_frozen_manifest(
    manifest: dict[str, Any],
    sweep_summary: dict[str, Any] | None = None,
) -> list[str]:
    """Acceptance findings for a frozen manifest; empty list means valid.

    Checks the §29 freeze completeness and the §30 sweep-purpose
    discipline against the sweep summary the manifest was built from.
    """
    findings: list[str] = []

    study_version = str(manifest.get("study_version", ""))
    if not _SEMVER_RE.match(study_version):
        findings.append(f"study_version_not_semver: {study_version!r}")

    parameters = manifest.get("parameters", {})
    missing = [path for path in REQUIRED_FROZEN_FAMILIES if path not in parameters]
    if missing:
        findings.append(f"missing_frozen_families: {sorted(missing)}")

    policies = manifest.get("freeze_policies", {})
    for key in FREEZE_POLICIES:
        if not str(policies.get(key, "")).strip():
            findings.append(f"freeze_policy_missing: {key}")

    if not manifest.get("scenario_definitions"):
        findings.append("scenario_definitions_missing")
    protocol = manifest.get("protocol", {})
    if not str(protocol.get("protocol_version", "")).strip():
        findings.append("protocol_version_missing")
    if not protocol.get("primary_hypotheses"):
        findings.append("primary_hypotheses_missing")

    summary = sweep_summary if sweep_summary is not None else {}
    if not summary:
        return findings

    # §29: swept values in the manifest must equal the sweep selections.
    selections = sweep_selection_index(summary)
    for config_path, selection in selections.items():
        entry = parameters.get(config_path)
        if entry is None:
            findings.append(f"swept_parameter_not_frozen: {config_path}")
            continue
        if entry.get("source") != "selection_sweep":
            findings.append(f"swept_parameter_not_sweep_sourced: {config_path}")
        swept = selection["selected_value"]
        frozen_value = entry.get("value")
        if swept is not None and float(swept) != float(frozen_value or swept):
            findings.append(
                f"selection_mismatch: {config_path} swept={swept} frozen={frozen_value}"
            )

    # §29: the frozen config hash must match the sweep's frozen config.
    expected_hashes = summary.get("frozen_config", {})
    manifest_hashes = manifest.get("frozen_config_hashes", {})
    for key in ("config_hash", "condition_hash"):
        expected = expected_hashes.get(key)
        recorded = manifest_hashes.get(key)
        if expected and recorded != expected:
            findings.append(f"frozen_config_hash_mismatch: {key}")

    # §29: the frozen protocol version must match the declared protocol.
    from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION

    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        findings.append(
            f"protocol_version_mismatch: manifest={protocol.get('protocol_version')!r} "
            f"current={PROTOCOL_VERSION!r}"
        )

    # §30: selection sweeps must never use the test split.
    for name, sweep in summary.get("sweeps", {}).items():
        purpose = sweep.get("sweep_purpose", SELECTION_PURPOSE)
        if purpose == SELECTION_PURPOSE and sweep.get("split") == "test":
            findings.append(f"selection_sweep_on_test_split: {name}")
        if purpose == SENSITIVITY_PURPOSE and not str(sweep.get("sensitivity_note", "")).strip():
            findings.append(f"sensitivity_sweep_without_note: {name}")

    return findings


# ---------------------------------------------------------------------------
# IO and CLI
# ---------------------------------------------------------------------------


def load_frozen_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path if path is not None else FROZEN_MANIFEST_PATH
    data: dict[str, Any] = json.loads(manifest_path.read_text())
    return data


def write_frozen_manifest(
    manifest: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    path = output_path if output_path is not None else FROZEN_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main() -> int:
    """Freeze the configuration manifest (§29) and validate it."""
    from experiments.trustparadox_u.artifact_provenance import (
        build_certification_provenance,
        code_tree_is_clean,
    )

    print("Remediation §29/§30: Frozen Configuration Manifest")
    print("=" * 50)

    # Snapshot provenance before writing so this run's own output cannot
    # self-invalidate it (FF92-023 discipline).
    provenance = build_certification_provenance(repository_clean=code_tree_is_clean())

    manifest = build_frozen_threshold_manifest()
    manifest["provenance"] = provenance

    findings = validate_frozen_manifest(manifest, load_sweep_summary())
    if findings:
        for finding in findings:
            print(f"  FINDING: {finding}")
        raise SystemExit(f"Frozen manifest validation failed: {len(findings)} findings")

    path = write_frozen_manifest(manifest)
    print(f"Study version: {manifest['study_version']}")
    print(f"Frozen parameters: {len(manifest['parameters'])}")
    for config_path, entry in sorted(manifest["parameters"].items()):
        origin = entry["source"]
        if origin == "selection_sweep":
            origin += f" ({entry['selection_split']})"
        print(f"  {config_path} = {entry['value']}  [{origin}]")
    print(f"\nManifest written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Remediation §33: one documented reproduction entry point.

``python -m experiments.trustparadox_u.reproduce`` rebuilds the final
tables from the frozen inputs in a single command:

1. validate the environment (interpreter, repository state, workspace
   and environment-lock hashes);
2. validate — never regenerate — the frozen corpus and annotations;
3. validate the frozen threshold manifest (§29);
4. run the declared condition matrix (frozen replay), leakage analysis,
   paired statistics and final tables as subprocess steps;
5. recompute metrics from the regenerated trial artifacts;
6. write ``results/reproduction/reproduction_manifest.json`` with the
   §32 three-way provenance, input hashes, step records and checksums
   of every regenerated artifact.

The manifest fails closed: any missing or mismatched frozen input, any
failed pipeline step, or any metric mismatch aborts the reproduction
with a non-zero exit code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = _PROJECT_ROOT / "results"
REPRODUCTION_DIR = RESULTS_DIR / "reproduction"
REPRODUCTION_MANIFEST_PATH = REPRODUCTION_DIR / "reproduction_manifest.json"
CORPUS_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "frozen_corpus"

SCHEMA_VERSION = "1.0"

# Pipeline steps run in order: replay the condition matrix, then the
# analyses and final tables derived from the trial artifacts.
PIPELINE_STEPS: tuple[tuple[str, str], ...] = (
    ("experiments.trustparadox_u.frozen_replay", "run the declared condition matrix"),
    ("experiments.trustparadox_u.leakage_analysis", "leakage analysis over trial artifacts"),
    ("experiments.trustparadox_u.paired_statistics", "paired statistical comparisons"),
    (
        "experiments.trustparadox_u.trust_analysis",
        "trust invariance and trust-manipulation analysis (Table 6)",
    ),
    ("experiments.trustparadox_u.final_artifacts", "final tables and study manifest"),
    ("experiments.trustparadox_u.failure_examples", "curated failure examples and decision traces"),
)

# Artifacts the reproduction must have regenerated, checksummed last.
REPRODUCED_ARTIFACTS: tuple[str, ...] = (
    "frozen_replay/run_manifest.json",
    "frozen_replay/candidate_trials.jsonl",
    "frozen_replay/reconstruction_trials.jsonl",
    "frozen_replay/recontamination_trials.jsonl",
    "frozen_replay/utility_trials.jsonl",
    "frozen_replay/resolved_conditions.json",
    "frozen_replay/metrics_by_condition.json",
    "leakage_analysis/leakage_analysis.json",
    "paired_statistics/paired_statistics.json",
    "trust_analysis/trust_analysis.json",
    "trust_analysis/pairing_audit.json",
    "final_artifacts/study_manifest.json",
    "final_artifacts/study_summary.md",
    "final_artifacts/table1_main_results.json",
    "final_artifacts/table2_leakage_breakdown.json",
    "final_artifacts/table3_parameter_sensitivity.json",
    "final_artifacts/table4_statistical_comparisons.json",
    "final_artifacts/table5_target_type_results.json",
    "final_artifacts/table6_trust_analysis.json",
    "failure_examples/failure_examples.json",
)


class ReproductionError(RuntimeError):
    """Raised when a reproduction cannot proceed or cannot be certified."""


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def validate_environment() -> dict[str, Any]:
    """§33: record the interpreter and repository state up front."""
    from experiments.trustparadox_u.artifact_provenance import (
        code_tree_is_clean,
        environment_lock_hash,
        generation_tree_hash,
        working_tree_is_fully_clean,
    )
    from experiments.trustparadox_u.manifest import get_repository_commit

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "repository_commit": get_repository_commit(),
        "source_files_clean": code_tree_is_clean(),
        "worktree_fully_clean": working_tree_is_fully_clean(),
        "artifact_generation_tree": generation_tree_hash(),
        "environment_lock_hash": environment_lock_hash(),
    }


def validate_frozen_inputs() -> dict[str, Any]:
    """Validate frozen corpus, annotations and thresholds — never regenerate.

    §33 acceptance: the reproduction fails on missing or mismatched
    inputs and never silently regenerates the corpus or annotations.
    """
    from experiments.trustparadox_u.frozen_thresholds import load_frozen_manifest
    from experiments.trustparadox_u.research_valid_gate import (
        check_annotations_valid,
        check_corpus_valid,
        check_frozen_threshold_manifest,
    )

    corpus = check_corpus_valid()
    if not corpus.get("passed"):
        raise ReproductionError(f"frozen corpus invalid: {corpus}")
    annotations = check_annotations_valid()
    if not annotations.get("passed"):
        raise ReproductionError(f"frozen annotations invalid: {annotations}")
    thresholds = check_frozen_threshold_manifest()
    if not thresholds.get("passed"):
        raise ReproductionError(f"frozen threshold manifest invalid: {thresholds}")

    manifest_path = CORPUS_DIR / "corpus_manifest.json"
    if not manifest_path.exists():
        raise ReproductionError(f"corpus manifest missing: {manifest_path}")
    corpus_manifest = _load_json(manifest_path)
    frozen_manifest = load_frozen_manifest()
    # FP-003: the frozen manifest carries protocol_version under its
    # nested ``protocol`` block; fall back so the reproduction manifest
    # never records an empty protocol version.
    protocol_block = frozen_manifest.get("protocol")
    nested_protocol = protocol_block if isinstance(protocol_block, dict) else {}
    protocol_version = str(
        frozen_manifest.get("protocol_version") or nested_protocol.get("protocol_version", "") or ""
    )

    return {
        "corpus": {
            "passed": True,
            "corpus_sha256": corpus.get("corpus_hash"),
            "candidate_count": corpus.get("candidate_count"),
        },
        "annotations": {
            "passed": True,
            "annotation_hash": annotations.get("annotation_hash"),
            "annotation_count": annotations.get("annotation_count"),
        },
        "frozen_thresholds": {
            "passed": True,
            "study_version": thresholds.get("study_version"),
            "parameter_count": thresholds.get("num_parameters"),
        },
        "model_provenance": {
            "generation_model": corpus_manifest.get("generation_model", ""),
            "annotation_model": corpus_manifest.get("annotation_model", ""),
            "study_class": corpus_manifest.get("study_class", "diagnostic"),
        },
        "protocol_version": protocol_version,
        "study_version": str(frozen_manifest.get("study_version", "")),
    }


def run_pipeline_step(module: str, description: str) -> dict[str, Any]:
    """Run one pipeline step as a subprocess and record its outcome."""
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    duration = round(time.monotonic() - started, 3)
    tail = (result.stdout + result.stderr).strip().split("\n")[-1]
    record = {
        "module": module,
        "description": description,
        "returncode": result.returncode,
        "duration_seconds": duration,
        "passed": result.returncode == 0,
        "output_tail": tail,
    }
    if result.returncode != 0:
        raise ReproductionError(f"pipeline step failed: {module} (rc={result.returncode}): {tail}")
    return record


def verify_recomputed_metrics() -> dict[str, Any]:
    """§33: metrics must recompute exactly from the regenerated trials."""
    from experiments.trustparadox_u.research_valid_gate import (
        check_conditions_valid,
        check_metrics_recompute,
    )

    conditions = check_conditions_valid()
    if not conditions.get("passed"):
        raise ReproductionError(f"resolved conditions invalid after replay: {conditions}")
    metrics = check_metrics_recompute()
    if not metrics.get("passed"):
        raise ReproductionError(f"metrics do not recompute from trial artifacts: {metrics}")
    resolved_path = RESULTS_DIR / "frozen_replay" / "resolved_conditions.json"
    return {
        "conditions_valid": True,
        "resolved_conditions_sha256": _sha256(resolved_path),
        "metrics_recompute": True,
        "conditions": metrics.get("conditions", []),
    }


def collect_artifact_checksums() -> dict[str, str]:
    """Checksum every artifact the reproduction regenerated."""
    checksums: dict[str, str] = {}
    for rel in REPRODUCED_ARTIFACTS:
        path = RESULTS_DIR / rel
        if not path.exists():
            raise ReproductionError(f"expected artifact missing after pipeline: {rel}")
        checksums[rel] = _sha256(path)
    return checksums


def build_reproduction_manifest() -> dict[str, Any]:
    """Run the full reproduction and return its manifest payload."""
    from experiments.trustparadox_u.artifact_provenance import (
        build_certification_provenance,
        code_tree_is_clean,
    )

    # Snapshot provenance before any artifact is written so this run's
    # own output cannot self-invalidate it (FF92-023 discipline).
    provenance = build_certification_provenance(repository_clean=code_tree_is_clean())
    environment = validate_environment()
    if not environment["source_files_clean"]:
        raise ReproductionError(
            "source files are not committed: reproduction must run from a clean code tree"
        )
    inputs = validate_frozen_inputs()
    # FP-003: root, inputs and provenance must agree on both versions.
    for field in ("study_version", "protocol_version"):
        root_value = str(provenance.get(field, "") or "")
        input_value = str(inputs.get(field, "") or "")
        if not input_value.strip() or root_value != input_value:
            raise ReproductionError(
                f"{field} mismatch: provenance={root_value!r} inputs={input_value!r}"
            )

    steps = [run_pipeline_step(module, description) for module, description in PIPELINE_STEPS]
    verification = verify_recomputed_metrics()
    checksums = collect_artifact_checksums()

    return {
        "schema_version": SCHEMA_VERSION,
        "validation": "single_command_reproduction",
        "remediation_items": ["32", "33"],
        "study_version": inputs["study_version"],
        "protocol_version": inputs["protocol_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "environment": environment,
        "inputs": inputs,
        "steps": steps,
        "verification": verification,
        "artifacts": checksums,
        "passed": True,
    }


def write_reproduction_manifest(manifest: dict[str, Any]) -> Path:
    REPRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    REPRODUCTION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return REPRODUCTION_MANIFEST_PATH


def main() -> int:
    """§33: one command from frozen inputs to final tables."""
    print("Remediation §33: Single-Command Reproduction")
    print("=" * 50)
    try:
        manifest = build_reproduction_manifest()
    except ReproductionError as exc:
        print(f"REPRODUCTION FAILED: {exc}")
        return 1
    path = write_reproduction_manifest(manifest)

    print(f"  study version: {manifest['study_version']}")
    print(f"  commit: {manifest['provenance']['tested_code_commit']}")
    for step in manifest["steps"]:
        print(f"  [PASS] {step['module']} ({step['duration_seconds']}s)")
    print(f"  metrics recompute: OK ({len(manifest['artifacts'])} artifacts checksummed)")
    print(f"\nManifest written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

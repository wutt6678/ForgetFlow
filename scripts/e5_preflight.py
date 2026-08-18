"""E5-000: Freeze and preflight verification.

Verifies the E4 global annotation freeze is intact and all frozen
evidence files are present before E5 experiments begin.

Exit codes:
    0 → GO — all checks pass
    1 → NO-GO — one or more blocking findings

Produces:
    results/empirical_v2/e5/e5_preflight.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_loaders import (
    VALID_SPLITS,
    compute_file_hashes,
    get_expected_counts,
    get_expected_unresolved,
    load_all_splits,
    load_corpus_manifest,
    load_global_freeze_manifest,
    sha256_file,
)

_E5_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "e5"
_PREFLIGHT_PATH = _E5_DIR / "e5_preflight.json"
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_GLOBAL_FREEZE_PATH = _ANNOTATIONS_DIR / "global_annotation_freeze_manifest.json"
_ANNOTATION_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"
_CORPUS_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
)


def _get_code_commit() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_preflight() -> dict:
    """Execute all preflight checks and return the result manifest.

    Returns:
        Dict with all preflight fields suitable for JSON serialization.
    """
    findings: list[str] = []
    checks: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Verify global annotation freeze manifest exists
    # ------------------------------------------------------------------
    if not _GLOBAL_FREEZE_PATH.exists():
        findings.append("global_annotation_freeze_manifest.json not found")
        checks["global_freeze_exists"] = "FAIL"
    else:
        checks["global_freeze_exists"] = "PASS"

    # ------------------------------------------------------------------
    # 2. Load and verify global freeze fields
    # ------------------------------------------------------------------
    freeze_manifest: dict = {}
    if _GLOBAL_FREEZE_PATH.exists():
        freeze_manifest = load_global_freeze_manifest()

    # go_no_go == GO
    go_no_go = freeze_manifest.get("go_no_go")
    if go_no_go != "GO":
        findings.append(f"go_no_go is {go_no_go!r}, expected 'GO'")
        checks["go_no_go"] = "FAIL"
    else:
        checks["go_no_go"] = "PASS"

    # annotations_frozen == true
    annotations_frozen = freeze_manifest.get("annotations_frozen")
    if annotations_frozen is not True:
        findings.append(f"annotations_frozen is {annotations_frozen!r}, expected true")
        checks["annotations_frozen"] = "FAIL"
    else:
        checks["annotations_frozen"] = "PASS"

    # annotation_phase == ANNOTATIONS_FROZEN
    annotation_phase = freeze_manifest.get("annotation_phase")
    if annotation_phase != "ANNOTATIONS_FROZEN":
        findings.append(
            f"annotation_phase is {annotation_phase!r}, expected 'ANNOTATIONS_FROZEN'"
        )
        checks["annotation_phase"] = "FAIL"
    else:
        checks["annotation_phase"] = "PASS"

    # all_gates_go == true
    all_gates_go = freeze_manifest.get("all_gates_go")
    if all_gates_go is not True:
        findings.append(f"all_gates_go is {all_gates_go!r}, expected true")
        checks["all_gates_go"] = "FAIL"
    else:
        checks["all_gates_go"] = "PASS"

    # ------------------------------------------------------------------
    # 3. Verify global totals: 900 rows, 144 sequences
    # ------------------------------------------------------------------
    global_totals = freeze_manifest.get("global_totals", {})
    expected_rows = global_totals.get("final_row_labels", 0)
    expected_sequences = global_totals.get("final_sequence_labels", 0)

    if expected_rows != 900:
        findings.append(f"global final_row_labels is {expected_rows}, expected 900")
        checks["global_row_count"] = "FAIL"
    else:
        checks["global_row_count"] = "PASS"

    if expected_sequences != 144:
        findings.append(f"global final_sequence_labels is {expected_sequences}, expected 144")
        checks["global_sequence_count"] = "FAIL"
    else:
        checks["global_sequence_count"] = "PASS"

    # ------------------------------------------------------------------
    # 4. Verify split counts
    # ------------------------------------------------------------------
    split_counts = freeze_manifest.get("split_counts", {})
    expected_split_counts = get_expected_counts()

    for split in sorted(VALID_SPLITS):
        sc = split_counts.get(split, {})
        exp = expected_split_counts[split]
        actual_rows = sc.get("final_rows", 0)
        actual_seqs = sc.get("final_sequences", 0)

        if actual_rows != exp["rows"]:
            findings.append(
                f"{split} final_rows is {actual_rows}, expected {exp['rows']}"
            )
            checks[f"{split}_row_count"] = "FAIL"
        else:
            checks[f"{split}_row_count"] = "PASS"

        if actual_seqs != exp["sequences"]:
            findings.append(
                f"{split} final_sequences is {actual_seqs}, expected {exp['sequences']}"
            )
            checks[f"{split}_sequence_count"] = "FAIL"
        else:
            checks[f"{split}_sequence_count"] = "PASS"

    # ------------------------------------------------------------------
    # 5. Verify unresolved counts
    # ------------------------------------------------------------------
    unresolved_by_split = freeze_manifest.get("unresolved_by_split", {})
    expected_unresolved = get_expected_unresolved()

    for split in sorted(VALID_SPLITS):
        urs = unresolved_by_split.get(split, {})
        exp = expected_unresolved[split]
        actual_ur_rows = urs.get("unresolved_rows", -1)
        actual_ur_seqs = urs.get("unresolved_sequences", -1)

        if actual_ur_rows != exp["rows"]:
            findings.append(
                f"{split} unresolved_rows is {actual_ur_rows}, expected {exp['rows']}"
            )
            checks[f"{split}_unresolved_rows"] = "FAIL"
        else:
            checks[f"{split}_unresolved_rows"] = "PASS"

        if actual_ur_seqs != exp["sequences"]:
            findings.append(
                f"{split} unresolved_sequences is {actual_ur_seqs}, expected {exp['sequences']}"
            )
            checks[f"{split}_unresolved_sequences"] = "FAIL"
        else:
            checks[f"{split}_unresolved_sequences"] = "PASS"

    # ------------------------------------------------------------------
    # 6. Verify frozen corpus manifest
    # ------------------------------------------------------------------
    if not _CORPUS_MANIFEST_PATH.exists():
        findings.append("frozen_corpus_manifest.json not found")
        checks["corpus_manifest_exists"] = "FAIL"
    else:
        checks["corpus_manifest_exists"] = "PASS"
        corpus_manifest = load_corpus_manifest()
        if not corpus_manifest.get("corpus_frozen"):
            findings.append("corpus_frozen is not true")
            checks["corpus_frozen"] = "FAIL"
        else:
            checks["corpus_frozen"] = "PASS"
        if corpus_manifest.get("accepted_candidate_count", 0) != 900:
            findings.append(
                f"accepted_candidate_count is "
                f"{corpus_manifest.get('accepted_candidate_count')}, expected 900"
            )
            checks["corpus_candidate_count"] = "FAIL"
        else:
            checks["corpus_candidate_count"] = "PASS"

    # ------------------------------------------------------------------
    # 7. Verify annotation phase file
    # ------------------------------------------------------------------
    if not _ANNOTATION_PHASE_PATH.exists():
        findings.append("annotation_phase.json not found")
        checks["annotation_phase_exists"] = "FAIL"
    else:
        checks["annotation_phase_exists"] = "PASS"
        with open(_ANNOTATION_PHASE_PATH) as f:
            phase_data = json.load(f)
        if phase_data.get("annotation_phase") != "ANNOTATIONS_FROZEN":
            findings.append("annotation_phase.json phase is not ANNOTATIONS_FROZEN")
            checks["annotation_phase_value"] = "FAIL"
        else:
            checks["annotation_phase_value"] = "PASS"
        if phase_data.get("global_annotation_freeze_complete") is not True:
            findings.append("global_annotation_freeze_complete is not true")
            checks["global_freeze_complete"] = "FAIL"
        else:
            checks["global_freeze_complete"] = "PASS"

    # ------------------------------------------------------------------
    # 8. Verify split files exist and counts match
    # ------------------------------------------------------------------
    all_data = load_all_splits()
    for split in sorted(VALID_SPLITS):
        sd = all_data[split]
        exp = expected_split_counts[split]

        if sd.n_rows != exp["rows"]:
            findings.append(
                f"{split} loaded {sd.n_rows} row labels, expected {exp['rows']}"
            )
            checks[f"{split}_loaded_rows"] = "FAIL"
        else:
            checks[f"{split}_loaded_rows"] = "PASS"

        if sd.n_sequences != exp["sequences"]:
            findings.append(
                f"{split} loaded {sd.n_sequences} sequence labels, expected {exp['sequences']}"
            )
            checks[f"{split}_loaded_sequences"] = "FAIL"
        else:
            checks[f"{split}_loaded_sequences"] = "PASS"

        if sd.n_corpus != exp["rows"]:
            findings.append(
                f"{split} loaded {sd.n_corpus} corpus candidates, expected {exp['rows']}"
            )
            checks[f"{split}_loaded_corpus"] = "FAIL"
        else:
            checks[f"{split}_loaded_corpus"] = "PASS"

    # ------------------------------------------------------------------
    # 9. Verify required frozen SHAs are present in global manifest
    # ------------------------------------------------------------------
    required_sha_keys = [
        "frozen_corpus_manifest_sha256",
        "frozen_annotation_protocol_sha256",
        "development_final_row_labels_sha256",
        "development_final_sequence_labels_sha256",
        "validation_final_row_labels_sha256",
        "validation_final_sequence_labels_sha256",
        "test_final_row_labels_sha256",
        "test_final_sequence_labels_sha256",
    ]
    for key in required_sha_keys:
        val = freeze_manifest.get(key)
        if not val or not isinstance(val, str) or len(val) != 64:
            findings.append(f"Missing or invalid SHA in global freeze: {key}")
            checks[f"sha_{key}"] = "FAIL"
        else:
            checks[f"sha_{key}"] = "PASS"

    # ------------------------------------------------------------------
    # 10. Verify corpus-to-annotation candidate_id overlap
    # ------------------------------------------------------------------
    for split in sorted(VALID_SPLITS):
        sd = all_data[split]
        corpus_ids = set(sd.corpus_by_id.keys())
        label_ids = set(sd.row_labels_by_id.keys())
        missing = label_ids - corpus_ids
        if missing:
            findings.append(
                f"{split}: {len(missing)} row label candidate_ids not in corpus"
            )
            checks[f"{split}_id_overlap"] = "FAIL"
        else:
            checks[f"{split}_id_overlap"] = "PASS"

    # ------------------------------------------------------------------
    # 11. Compute file hashes for provenance
    # ------------------------------------------------------------------
    file_hashes = compute_file_hashes()

    # ------------------------------------------------------------------
    # Build preflight result
    # ------------------------------------------------------------------
    preflight_pass = len(findings) == 0

    result = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _get_code_commit(),
        "global_annotation_freeze_sha256": file_hashes.get(
            "global_annotation_freeze_manifest", ""
        ),
        "annotation_protocol_sha256": freeze_manifest.get(
            "frozen_annotation_protocol_sha256", ""
        ),
        "frozen_corpus_sha256": file_hashes.get("frozen_corpus_manifest", ""),
        "development_rows": all_data["development"].n_rows,
        "validation_rows": all_data["validation"].n_rows,
        "test_rows": all_data["test"].n_rows,
        "development_sequences": all_data["development"].n_sequences,
        "validation_sequences": all_data["validation"].n_sequences,
        "test_sequences": all_data["test"].n_sequences,
        "development_corpus": all_data["development"].n_corpus,
        "validation_corpus": all_data["validation"].n_corpus,
        "test_corpus": all_data["test"].n_corpus,
        "unresolved_by_split": {
            split: {
                "unresolved_rows": all_data[split].n_unresolved_rows,
                "unresolved_sequences": all_data[split].n_unresolved_sequences,
            }
            for split in sorted(VALID_SPLITS)
        },
        "file_hashes": file_hashes,
        "checks": checks,
        "preflight_pass": preflight_pass,
        "blocking_findings": findings,
    }

    return result


def main() -> int:
    """Run preflight and write the result artifact.

    Returns:
        0 if GO, 1 if NO-GO.
    """
    print("E5-000: Running freeze + preflight verification...")
    print()

    result = run_preflight()

    # Write output
    _E5_DIR.mkdir(parents=True, exist_ok=True)
    with open(_PREFLIGHT_PATH, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    # Report
    n_pass = sum(1 for v in result["checks"].values() if v == "PASS")
    n_fail = sum(1 for v in result["checks"].values() if v == "FAIL")
    print(f"Checks: {n_pass} PASS, {n_fail} FAIL")
    print()

    if result["blocking_findings"]:
        print("BLOCKING FINDINGS:")
        for finding in result["blocking_findings"]:
            print(f"  - {finding}")
        print()

    if result["preflight_pass"]:
        print("E5-000 PREFLIGHT: GO")
        print(f"Artifact written: {_PREFLIGHT_PATH}")
        return 0
    else:
        print("E5-000 PREFLIGHT: NO-GO")
        print(f"Artifact written: {_PREFLIGHT_PATH}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""E4-003: Global annotation freeze — immutable measurement root for E5.

Sec 104-119: Bind all three splits (development + validation + test) by SHA256,
evaluate global gates, set annotations_frozen=true, and produce post-freeze
closure verification.

Usage:
  PYTHONPATH=. python scripts/build_global_annotation_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"
_PROTOCOL_PATH = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
_CORPUS_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
)

# Split directories
_DEV_DIR = _ANNOTATIONS_DIR / "development_v3"
_VAL_DIR = _ANNOTATIONS_DIR / "validation"
_TEST_DIR = _ANNOTATIONS_DIR / "test"

# Expected global counts (Sec 108)
EXPECTED_DEV_ROWS = 225
EXPECTED_DEV_SEQUENCES = 36
EXPECTED_VAL_ROWS = 225
EXPECTED_VAL_SEQUENCES = 36
EXPECTED_TEST_ROWS = 450
EXPECTED_TEST_SEQUENCES = 72
EXPECTED_TOTAL_ROWS = 900
EXPECTED_TOTAL_SEQUENCES = 144


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _count_jsonl(path: Path) -> int:
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_short() -> str:
    return _git_commit()[:7]


def require_clean_worktree() -> None:
    """Abort if the git worktree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_PROJECT_ROOT
        ).decode().strip()
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot check git status: {exc}") from exc
    if out:
        raise SystemExit(
            "ERROR: dirty worktree — commit or stash changes before regeneration.\n"
            f"  git status --porcelain output:\n{out}"
        )


def _run_verifier(cmd: list[str]) -> dict[str, Any]:
    """Run a verifier subprocess and return structured results."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        output = result.stdout + result.stderr
        passed = output.count("PASS:")
        failed = output.count("FAIL:")
        total = passed + failed
        return {
            "checks_total": total,
            "checks_passed": passed,
            "checks_failed": failed,
            "exit_code": result.returncode,
            "timestamp": ts,
        }
    except Exception as exc:
        return {
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 1,
            "exit_code": -1,
            "timestamp": ts,
            "error": str(exc),
        }


def _verifier_pass(v: dict[str, Any]) -> bool:
    return (
        v.get("exit_code") == 0
        and v.get("checks_failed") == 0
        and v.get("checks_passed") == v.get("checks_total")
        and v.get("checks_total", 0) > 0
    )


def run_all_verifiers() -> dict[str, Any]:
    """Run all split/global verifiers and return structured results."""
    py = sys.executable
    return {
        "frozen_corpus": _run_verifier([
            py, "scripts/verify_frozen_empirical_corpus.py",
        ]),
        "development_annotations": _run_verifier([
            py, "scripts/verify_frozen_annotations.py", "--split", "development",
        ]),
        "validation_annotations": _run_verifier([
            py, "scripts/verify_frozen_annotations.py", "--split", "validation",
        ]),
        "validation_closure": _run_verifier([
            py, "scripts/verify_validation_freeze_closure.py",
        ]),
        "test_annotations": _run_verifier([
            py, "scripts/verify_frozen_annotations.py", "--split", "test",
        ]),
        "test_closure": _run_verifier([
            py, "scripts/verify_test_freeze_closure.py",
        ]),
    }


def _safe_sha(path: Path) -> str:
    """Return SHA256 of file if it exists, else empty string."""
    if path.exists():
        return _sha256(path)
    return ""


def _count_final_labels(split_dir: Path, prefix: str = "") -> dict[str, int]:
    """Count final row and sequence labels in a split directory."""
    rows = 0
    seqs = 0
    # Try standard names first
    for candidate in [
        split_dir / f"{prefix}final_adjudicated_labels.jsonl",
        split_dir / "final_adjudicated_labels.jsonl",
    ]:
        if candidate.exists():
            rows = _count_jsonl(candidate)
            break
    for candidate in [
        split_dir / f"{prefix}final_sequence_labels.jsonl",
        split_dir / "final_sequence_labels.jsonl",
    ]:
        if candidate.exists():
            seqs = _count_jsonl(candidate)
            break
    return {"rows": rows, "sequences": seqs}


def build_global_annotation_freeze_manifest() -> dict[str, Any]:
    """Sec 105-111: Build global annotation freeze manifest."""
    print("=" * 60)
    print("Global Annotation Freeze Manifest")
    print("=" * 60)

    # --- Root SHA bindings (Sec 106) ---
    corpus_sha = _sha256(_CORPUS_MANIFEST_PATH)
    protocol_sha = _sha256(_PROTOCOL_PATH)
    print(f"  frozen_corpus_manifest: {corpus_sha[:16]}...")
    print(f"  annotation_protocol:    {protocol_sha[:16]}...")

    # Development root bindings (Sec 106: development annotation freeze/root manifest)
    dev_manifest_path = _DEV_DIR / "annotation_manifest.json"
    dev_manifest_sha = _safe_sha(dev_manifest_path)
    dev_gate_path = _DEV_DIR / "development_annotation_gate.json"
    dev_gate_sha = _safe_sha(dev_gate_path)
    dev_adjudication_path = _DEV_DIR / "adjudication_manifest.json"
    dev_adjudication_sha = _safe_sha(dev_adjudication_path)
    dev_final_labels_path = _DEV_DIR / "final_adjudicated_labels.jsonl"
    dev_final_labels_sha = _safe_sha(dev_final_labels_path)
    # Development may not have final_sequence_labels (Sec 107: "where such split artifacts exist")
    dev_final_seq_path = _DEV_DIR / "final_sequence_labels.jsonl"
    dev_final_seq_sha = _safe_sha(dev_final_seq_path)
    print(f"  dev annotation_manifest:  {dev_manifest_sha[:16]}..." if dev_manifest_sha else "  dev annotation_manifest:  MISSING")
    print(f"  dev adjudication:         {dev_adjudication_sha[:16]}..." if dev_adjudication_sha else "  dev adjudication:         MISSING")
    print(f"  dev final_row_labels:     {dev_final_labels_sha[:16]}..." if dev_final_labels_sha else "  dev final_row_labels:     MISSING")

    # Validation root bindings (Sec 106)
    val_freeze_path = _VAL_DIR / "validation_annotation_freeze_manifest.json"
    val_freeze_sha = _safe_sha(val_freeze_path)
    val_post_freeze_path = _VAL_DIR / "post_freeze_verification.json"
    val_post_freeze_sha = _safe_sha(val_post_freeze_path)
    val_final_labels_path = _VAL_DIR / "final_adjudicated_labels.jsonl"
    val_final_labels_sha = _safe_sha(val_final_labels_path)
    val_final_seq_path = _VAL_DIR / "final_sequence_labels.jsonl"
    val_final_seq_sha = _safe_sha(val_final_seq_path)
    print(f"  val freeze_manifest:      {val_freeze_sha[:16]}..." if val_freeze_sha else "  val freeze_manifest:      MISSING")
    print(f"  val post_freeze:          {val_post_freeze_sha[:16]}..." if val_post_freeze_sha else "  val post_freeze:          MISSING")

    # Test root bindings (Sec 106)
    test_freeze_path = _TEST_DIR / "test_annotation_freeze_manifest.json"
    test_freeze_sha = _safe_sha(test_freeze_path)
    test_post_freeze_path = _TEST_DIR / "test_post_freeze_verification.json"
    test_post_freeze_sha = _safe_sha(test_post_freeze_path)
    test_final_labels_path = _TEST_DIR / "test_final_adjudicated_labels.jsonl"
    test_final_labels_sha = _safe_sha(test_final_labels_path)
    test_final_seq_path = _TEST_DIR / "test_final_sequence_labels.jsonl"
    test_final_seq_sha = _safe_sha(test_final_seq_path)
    print(f"  test freeze_manifest:     {test_freeze_sha[:16]}..." if test_freeze_sha else "  test freeze_manifest:     MISSING")
    print(f"  test post_freeze:         {test_post_freeze_sha[:16]}..." if test_post_freeze_sha else "  test post_freeze:         MISSING")

    # --- Global counts (Sec 108) ---
    dev_counts = _count_final_labels(_DEV_DIR)
    val_counts = _count_final_labels(_VAL_DIR)
    test_counts = _count_final_labels(_TEST_DIR, prefix="test_")

    # Fallback: try without prefix if prefix search found nothing
    if test_counts["rows"] == 0:
        test_counts = _count_final_labels(_TEST_DIR)

    total_rows = dev_counts["rows"] + val_counts["rows"] + test_counts["rows"]
    total_sequences = (
        dev_counts["sequences"] + val_counts["sequences"] + test_counts["sequences"]
    )

    print(f"\n  Global counts:")
    print(f"    development:  rows={dev_counts['rows']}, sequences={dev_counts['sequences']}")
    print(f"    validation:   rows={val_counts['rows']}, sequences={val_counts['sequences']}")
    print(f"    test:         rows={test_counts['rows']}, sequences={test_counts['sequences']}")
    print(f"    TOTAL:        rows={total_rows}, sequences={total_sequences}")

    # --- Split gate status (Sec 111) ---
    dev_gate = _load_json(dev_gate_path) if dev_gate_path.exists() else {}
    val_gate_path = _VAL_DIR / "validation_annotation_gate.json"
    val_gate = _load_json(val_gate_path) if val_gate_path.exists() else {}
    test_gate_path_file = _TEST_DIR / "test_annotation_gate.json"
    test_gate = _load_json(test_gate_path_file) if test_gate_path_file.exists() else {}

    dev_go = dev_gate.get("go_no_go", "NO-GO") == "GO"
    val_go = val_gate.get("go_no_go", "NO-GO") == "GO"
    test_go = test_gate.get("go_no_go", "NO-GO") == "GO"

    print(f"\n  Split gates: dev={dev_gate.get('go_no_go')}, val={val_gate.get('go_no_go')}, test={test_gate.get('go_no_go')}")

    # --- Unresolved counts by split (Sec 110) ---
    dev_adj_path = _DEV_DIR / "adjudication_manifest.json"
    dev_adj = _load_json(dev_adj_path) if dev_adj_path.exists() else {}
    dev_unresolved = dev_gate.get("summary", {})
    val_adj = _load_json(_VAL_DIR / "adjudication_manifest.json") if (_VAL_DIR / "adjudication_manifest.json").exists() else {}
    test_adj = _load_json(_TEST_DIR / "test_adjudication_manifest.json") if (_TEST_DIR / "test_adjudication_manifest.json").exists() else {}

    # --- Enforce expected counts (item 75) ---
    count_blocking: list[str] = []
    if dev_counts["rows"] != EXPECTED_DEV_ROWS:
        count_blocking.append(f"dev rows: {dev_counts['rows']}/{EXPECTED_DEV_ROWS}")
    if dev_counts["sequences"] != EXPECTED_DEV_SEQUENCES:
        count_blocking.append(f"dev seqs: {dev_counts['sequences']}/{EXPECTED_DEV_SEQUENCES}")
    if val_counts["rows"] != EXPECTED_VAL_ROWS:
        count_blocking.append(f"val rows: {val_counts['rows']}/{EXPECTED_VAL_ROWS}")
    if val_counts["sequences"] != EXPECTED_VAL_SEQUENCES:
        count_blocking.append(f"val seqs: {val_counts['sequences']}/{EXPECTED_VAL_SEQUENCES}")
    if test_counts["rows"] != EXPECTED_TEST_ROWS:
        count_blocking.append(f"test rows: {test_counts['rows']}/{EXPECTED_TEST_ROWS}")
    if test_counts["sequences"] != EXPECTED_TEST_SEQUENCES:
        count_blocking.append(f"test seqs: {test_counts['sequences']}/{EXPECTED_TEST_SEQUENCES}")
    if total_rows != EXPECTED_TOTAL_ROWS:
        count_blocking.append(f"total rows: {total_rows}/{EXPECTED_TOTAL_ROWS}")
    if total_sequences != EXPECTED_TOTAL_SEQUENCES:
        count_blocking.append(f"total seqs: {total_sequences}/{EXPECTED_TOTAL_SEQUENCES}")

    manifest = {
        "schema_version": "1.0",
        "description": "E4-003: Global annotation freeze manifest — immutable measurement root for E5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotations_frozen": True,
        "annotation_phase": "ANNOTATIONS_FROZEN",
        "global_annotation_freeze_code_commit": _git_commit(),
        "count_enforcement": {
            "blocking_issues": count_blocking,
            "counts_pass": len(count_blocking) == 0,
        },
        # Root SHA bindings (Sec 106)
        "frozen_corpus_manifest_sha256": corpus_sha,
        "frozen_annotation_protocol_sha256": protocol_sha,
        # Development roots
        "development_annotation_manifest_sha256": dev_manifest_sha,
        "development_annotation_gate_sha256": dev_gate_sha,
        "development_adjudication_manifest_sha256": dev_adjudication_sha,
        "development_final_row_labels_sha256": dev_final_labels_sha,
        "development_final_sequence_labels_sha256": dev_final_seq_sha,
        # Validation roots
        "validation_annotation_freeze_manifest_sha256": val_freeze_sha,
        "validation_post_freeze_verification_sha256": val_post_freeze_sha,
        "validation_final_row_labels_sha256": val_final_labels_sha,
        "validation_final_sequence_labels_sha256": val_final_seq_sha,
        # Test roots
        "test_annotation_freeze_manifest_sha256": test_freeze_sha,
        "test_post_freeze_verification_sha256": test_post_freeze_sha,
        "test_final_row_labels_sha256": test_final_labels_sha,
        "test_final_sequence_labels_sha256": test_final_seq_sha,
        # Global counts (Sec 108)
        "split_counts": {
            "development": {
                "final_rows": dev_counts["rows"],
                "final_sequences": dev_counts["sequences"],
            },
            "validation": {
                "final_rows": val_counts["rows"],
                "final_sequences": val_counts["sequences"],
            },
            "test": {
                "final_rows": test_counts["rows"],
                "final_sequences": test_counts["sequences"],
            },
        },
        "global_totals": {
            "final_row_labels": total_rows,
            "final_sequence_labels": total_sequences,
        },
        # Split gate status (Sec 111)
        "split_gates": {
            "development": dev_gate.get("go_no_go", "NO-GO"),
            "validation": val_gate.get("go_no_go", "NO-GO"),
            "test": test_gate.get("go_no_go", "NO-GO"),
        },
        "all_gates_go": dev_go and val_go and test_go,
        # Unresolved by split (Sec 110)
        "unresolved_by_split": {
            "development": {
                "unresolved_rows": dev_adj.get("final_label_counts", {}).get("unresolved", 0),
                "unresolved_sequences": dev_gate.get("summary", {}).get("sequence_unresolved", 0),
            },
            "validation": {
                "unresolved_rows": val_adj.get("final_label_counts", {}).get("unresolved_rows", 0),
                "unresolved_sequences": val_adj.get("final_label_counts", {}).get("unresolved_sequences", 0),
            },
            "test": {
                "unresolved_rows": test_adj.get("final_label_counts", {}).get("unresolved_rows", 0),
                "unresolved_sequences": test_adj.get("final_label_counts", {}).get("unresolved_sequences", 0),
            },
        },
        # Historical note (Sec 107/2006-2007)
        "development_directory": "development_v3",
        "development_uses_historical_filenames": True,
    }

    out_path = _ANNOTATIONS_DIR / "global_annotation_freeze_manifest.json"
    _write_json(out_path, manifest)
    print(f"\nWrote {out_path.name}")
    return manifest


def update_annotation_phase_global() -> None:
    """Sec 112: Update annotation phase after global freeze."""
    print("\n" + "=" * 60)
    print("Update Annotation Phase (Global)")
    print("=" * 60)

    phase = _load_json(_PHASE_PATH)

    phase["annotation_phase"] = "ANNOTATIONS_FROZEN"
    phase["development_annotation_complete"] = True
    phase["validation_annotation_complete"] = True
    phase["test_annotation_complete"] = True
    phase["annotations_frozen"] = True
    phase["global_annotation_freeze_complete"] = True
    phase["global_annotation_freeze_code_commit"] = _git_commit()

    _write_json(_PHASE_PATH, phase)
    print("Set annotation_phase = ANNOTATIONS_FROZEN")
    print("Set development_annotation_complete = true")
    print("Set validation_annotation_complete = true")
    print("Set test_annotation_complete = true")
    print("Set annotations_frozen = true")
    print("Set global_annotation_freeze_complete = true")
    print(f"Updated {_PHASE_PATH.name}")

    # Also update protocol manifest annotations_frozen flag
    pm = _load_json(_PROTOCOL_PATH)
    pm["annotations_frozen"] = True
    _write_json(_PROTOCOL_PATH, pm)
    print(f"Set protocol manifest annotations_frozen = true")
    print(f"Updated {_PROTOCOL_PATH.name}")

    # Re-hash protocol manifest and update global freeze manifest
    freeze_path = _ANNOTATIONS_DIR / "global_annotation_freeze_manifest.json"
    freeze_data = _load_json(freeze_path)
    freeze_data["frozen_annotation_protocol_sha256"] = _sha256(_PROTOCOL_PATH)
    _write_json(freeze_path, freeze_data)
    print(f"Updated protocol SHA in global freeze manifest")


def build_global_post_freeze_verification(
    verifier_results: dict[str, Any],
) -> dict[str, Any]:
    """Sec 117-118: Post-freeze closure verification for global freeze."""
    print("\n" + "=" * 60)
    print("Global Post-Freeze Closure Verification")
    print("=" * 60)

    freeze_path = _ANNOTATIONS_DIR / "global_annotation_freeze_manifest.json"
    freeze_data = _load_json(freeze_path)
    freeze_created_at = freeze_data.get("created_at", "")
    freeze_sha = _sha256(freeze_path)
    phase_sha = _sha256(_PHASE_PATH)

    fc = verifier_results.get("frozen_corpus", {})
    dev = verifier_results.get("development_annotations", {})
    val = verifier_results.get("validation_annotations", {})
    val_closure = verifier_results.get("validation_closure", {})
    test_v = verifier_results.get("test_annotations", {})
    test_closure = verifier_results.get("test_closure", {})

    closure_pass = (
        _verifier_pass(fc)
        and _verifier_pass(dev)
        and _verifier_pass(val)
        and _verifier_pass(val_closure)
        and _verifier_pass(test_v)
        and _verifier_pass(test_closure)
    )

    verification = {
        "schema_version": "1.0",
        "description": "E4-003: Global annotation post-freeze closure verification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "freeze_created_at": freeze_created_at,
        "verification_source_commit": _git_commit(),
        "global_freeze_manifest_sha256": freeze_sha,
        "annotation_phase_sha256": phase_sha,
        "frozen_corpus_verifier": {
            "checks_total": fc.get("checks_total", 0),
            "checks_passed": fc.get("checks_passed", 0),
            "checks_failed": fc.get("checks_failed", 0),
            "exit_code": fc.get("exit_code", -1),
            "timestamp": fc.get("timestamp", ""),
        },
        "development_annotation_verifier": {
            "checks_total": dev.get("checks_total", 0),
            "checks_passed": dev.get("checks_passed", 0),
            "checks_failed": dev.get("checks_failed", 0),
            "exit_code": dev.get("exit_code", -1),
            "timestamp": dev.get("timestamp", ""),
        },
        "validation_annotation_verifier": {
            "checks_total": val.get("checks_total", 0),
            "checks_passed": val.get("checks_passed", 0),
            "checks_failed": val.get("checks_failed", 0),
            "exit_code": val.get("exit_code", -1),
            "timestamp": val.get("timestamp", ""),
        },
        "validation_freeze_closure_verifier": {
            "checks_total": val_closure.get("checks_total", 0),
            "checks_passed": val_closure.get("checks_passed", 0),
            "checks_failed": val_closure.get("checks_failed", 0),
            "exit_code": val_closure.get("exit_code", -1),
            "timestamp": val_closure.get("timestamp", ""),
        },
        "test_annotation_verifier": {
            "checks_total": test_v.get("checks_total", 0),
            "checks_passed": test_v.get("checks_passed", 0),
            "checks_failed": test_v.get("checks_failed", 0),
            "exit_code": test_v.get("exit_code", -1),
            "timestamp": test_v.get("timestamp", ""),
        },
        "test_freeze_closure_verifier": {
            "checks_total": test_closure.get("checks_total", 0),
            "checks_passed": test_closure.get("checks_passed", 0),
            "checks_failed": test_closure.get("checks_failed", 0),
            "exit_code": test_closure.get("exit_code", -1),
            "timestamp": test_closure.get("timestamp", ""),
        },
        "all_verifiers_pass": closure_pass,
    }

    out_path = _ANNOTATIONS_DIR / "global_annotation_post_freeze_verification.json"
    _write_json(out_path, verification)
    print(f"all_verifiers_pass: {closure_pass}")
    print(f"Wrote {out_path.name}")
    return verification


def main() -> int:
    require_clean_worktree()  # item 11: restored clean-worktree protection
    print(f"Worktree at {_git_short()}")

    # Build global freeze manifest
    manifest = build_global_annotation_freeze_manifest()

    # Check all gates GO (Sec 111)
    if not manifest.get("all_gates_go"):
        print("\nERROR: Not all split gates are GO — cannot freeze globally")
        return 1

    # Check count enforcement (item 75)
    count_enforcement = manifest.get("count_enforcement", {})
    if not count_enforcement.get("counts_pass", False):
        print("\nERROR: Global count enforcement failed:")
        for issue in count_enforcement.get("blocking_issues", []):
            print(f"  - {issue}")
        return 1

    # Update phase (Sec 112)
    update_annotation_phase_global()

    # Run all verifiers post-freeze (Sec 118)
    print("\n--- Running Post-Freeze Verifiers ---")
    verifier_results = run_all_verifiers()
    for name, result in verifier_results.items():
        status = "PASS" if _verifier_pass(result) else "FAIL"
        print(f"  {name}: {status} ({result.get('checks_passed', 0)}/{result.get('checks_total', 0)})")

    # Build post-freeze verification
    verification = build_global_post_freeze_verification(verifier_results)

    # Fail-closed: check closure result (item 86)
    if not verification.get("all_verifiers_pass"):
        print("\nPost-freeze closure FAIL — exiting nonzero")
        return 1

    # Run standalone global verifier (item 87)
    print("\n--- Running Standalone Global Verifier ---")
    standalone_result = _run_verifier([
        sys.executable, "scripts/verify_global_annotation_freeze.py",
    ])
    standalone_status = "PASS" if standalone_result["exit_code"] == 0 else "FAIL"
    print(f"  verify_global_annotation_freeze.py: {standalone_status}")
    if standalone_result["exit_code"] != 0:
        print("Standalone global verifier FAIL — exiting nonzero")
        return 1

    print("\n" + "=" * 60)
    print("GLOBAL ANNOTATION FREEZE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

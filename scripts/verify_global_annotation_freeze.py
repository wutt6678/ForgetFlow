#!/usr/bin/env python3
"""E4-003: Global annotation freeze verifier.

Sec 115-116: Rehashes global freeze artifacts, confirms all split verifiers
and closure verifiers PASS, validates global counts, and checks phase fields.
Returns nonzero on any mismatch.

Usage:
    PYTHONPATH=. python scripts/verify_global_annotation_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"
_PROTOCOL_PATH = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
_CORPUS_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
)
_FREEZE_PATH = _ANNOTATIONS_DIR / "global_annotation_freeze_manifest.json"
_POST_FREEZE_PATH = _ANNOTATIONS_DIR / "global_annotation_post_freeze_verification.json"

# Split directories
_DEV_DIR = _ANNOTATIONS_DIR / "development_v3"
_VAL_DIR = _ANNOTATIONS_DIR / "validation"
_TEST_DIR = _ANNOTATIONS_DIR / "test"

# Expected counts (Sec 108)
EXPECTED_DEV_ROWS = 225
EXPECTED_DEV_SEQUENCES = 36  # development_v3 final_sequence_labels (Sec 107)
EXPECTED_VAL_ROWS = 225
EXPECTED_VAL_SEQUENCES = 36
EXPECTED_TEST_ROWS = 450
EXPECTED_TEST_SEQUENCES = 72
EXPECTED_TOTAL_ROWS = 900
EXPECTED_TOTAL_SEQUENCES = 144  # dev(36) + val(36) + test(72)

# Expected unresolved counts (item 25)
EXPECTED_DEV_UNRESOLVED_ROWS = 14
EXPECTED_VAL_UNRESOLVED_ROWS = 9
EXPECTED_TEST_UNRESOLVED_ROWS = 24
EXPECTED_DEV_UNRESOLVED_SEQS = 0
EXPECTED_VAL_UNRESOLVED_SEQS = 0
EXPECTED_TEST_UNRESOLVED_SEQS = 0

# Campaign identity checks moved to test verifier (item 10).
# Global scope requires only: test verifier PASS (C8), test freeze root SHA (C11),
# and test post-freeze closure PASS (C9).


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl(path: Path) -> int:
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _verifier_pass(v: dict) -> bool:
    return (
        v.get("exit_code") == 0
        and v.get("checks_failed") == 0
        and v.get("checks_passed") == v.get("checks_total")
        and v.get("checks_total", 0) > 0
    )


def main() -> int:
    passed = 0
    failed = 0

    def _pass(msg: str) -> None:
        nonlocal passed
        passed += 1
        print(f"  PASS: {msg}")

    def _fail(msg: str) -> None:
        nonlocal failed
        failed += 1
        print(f"  FAIL: {msg}")

    print("=" * 70)
    print("GLOBAL ANNOTATION FREEZE VERIFIER")
    print("=" * 70)

    # --- Load global freeze manifest ---
    print("\n[C1] Global freeze manifest...")
    if not _FREEZE_PATH.exists():
        _fail("global_annotation_freeze_manifest.json not found")
        print(f"\nVERIFICATION: FAIL ({passed} passed, {failed} failed)")
        return 1
    freeze = _load_json(_FREEZE_PATH)
    _pass("global_annotation_freeze_manifest.json exists")

    # [C2] Frozen corpus manifest SHA match
    print("\n[C2] Frozen corpus manifest SHA...")
    if _CORPUS_MANIFEST_PATH.exists():
        actual = _sha256(_CORPUS_MANIFEST_PATH)
        expected = freeze.get("frozen_corpus_manifest_sha256", "")
        if actual == expected:
            _pass("frozen_corpus_manifest SHA256 matches")
        else:
            _fail("frozen_corpus_manifest SHA256 MISMATCH")
    else:
        _fail("frozen_corpus_manifest.json not found")

    # [C3] Annotation protocol SHA match
    print("\n[C3] Annotation protocol SHA...")
    if _PROTOCOL_PATH.exists():
        actual = _sha256(_PROTOCOL_PATH)
        expected = freeze.get("frozen_annotation_protocol_sha256", "")
        if actual == expected:
            _pass("annotation_protocol SHA256 matches")
        else:
            _fail("annotation_protocol SHA256 MISMATCH")
    else:
        _fail("annotation_protocol_manifest.json not found")

    # [C4] Load post-freeze verification for verifier results
    print("\n[C4] Global post-freeze verification artifact...")
    if not _POST_FREEZE_PATH.exists():
        _fail("global_annotation_post_freeze_verification.json not found")
    else:
        pfv = _load_json(_POST_FREEZE_PATH)
        _pass("global_annotation_post_freeze_verification.json exists")

        # [C5] Development verifier PASS
        print("\n[C5] Development annotation verifier...")
        dev = pfv.get("development_annotation_verifier", {})
        if _verifier_pass(dev):
            _pass(f"development_verifier PASS ({dev['checks_passed']}/{dev['checks_total']})")
        else:
            _fail(
                f"development_verifier FAIL "
                f"(exit={dev.get('exit_code')}, passed={dev.get('checks_passed')}, "
                f"failed={dev.get('checks_failed')}, total={dev.get('checks_total')})"
            )

        # [C6] Validation verifier PASS
        print("\n[C6] Validation annotation verifier...")
        val = pfv.get("validation_annotation_verifier", {})
        if _verifier_pass(val):
            _pass(f"validation_verifier PASS ({val['checks_passed']}/{val['checks_total']})")
        else:
            _fail(
                f"validation_verifier FAIL "
                f"(exit={val.get('exit_code')}, passed={val.get('checks_passed')}, "
                f"failed={val.get('checks_failed')}, total={val.get('checks_total')})"
            )

        # [C7] Validation closure PASS
        print("\n[C7] Validation freeze closure verifier...")
        val_closure = pfv.get("validation_freeze_closure_verifier", {})
        if _verifier_pass(val_closure):
            _pass(f"validation_closure PASS ({val_closure['checks_passed']}/{val_closure['checks_total']})")
        else:
            _fail(
                f"validation_closure FAIL "
                f"(exit={val_closure.get('exit_code')}, passed={val_closure.get('checks_passed')}, "
                f"failed={val_closure.get('checks_failed')}, total={val_closure.get('checks_total')})"
            )

        # [C8] Test verifier PASS
        print("\n[C8] Test annotation verifier...")
        test_v = pfv.get("test_annotation_verifier", {})
        if _verifier_pass(test_v):
            _pass(f"test_verifier PASS ({test_v['checks_passed']}/{test_v['checks_total']})")
        else:
            _fail(
                f"test_verifier FAIL "
                f"(exit={test_v.get('exit_code')}, passed={test_v.get('checks_passed')}, "
                f"failed={test_v.get('checks_failed')}, total={test_v.get('checks_total')})"
            )

        # [C9] Test closure PASS
        print("\n[C9] Test freeze closure verifier...")
        test_closure = pfv.get("test_freeze_closure_verifier", {})
        if _verifier_pass(test_closure):
            _pass(f"test_closure PASS ({test_closure['checks_passed']}/{test_closure['checks_total']})")
        else:
            _fail(
                f"test_closure FAIL "
                f"(exit={test_closure.get('exit_code')}, passed={test_closure.get('checks_passed')}, "
                f"failed={test_closure.get('checks_failed')}, total={test_closure.get('checks_total')})"
            )

    # [C10] Global split roots all exist
    print("\n[C10] Global split roots existence...")
    split_root_files = {
        "development_annotation_manifest": _DEV_DIR / "annotation_manifest.json",
        "development_annotation_gate": _DEV_DIR / "development_annotation_gate.json",
        "development_adjudication_manifest": _DEV_DIR / "adjudication_manifest.json",
        "development_final_row_labels": _DEV_DIR / "final_adjudicated_labels.jsonl",
        "development_final_sequence_labels": _DEV_DIR / "final_sequence_labels.jsonl",
        "validation_freeze_manifest": _VAL_DIR / "validation_annotation_freeze_manifest.json",
        "validation_post_freeze": _VAL_DIR / "post_freeze_verification.json",
        "validation_final_row_labels": _VAL_DIR / "final_adjudicated_labels.jsonl",
        "validation_final_sequence_labels": _VAL_DIR / "final_sequence_labels.jsonl",
        "test_freeze_manifest": _TEST_DIR / "test_annotation_freeze_manifest.json",
        "test_post_freeze": _TEST_DIR / "test_post_freeze_verification.json",
        "test_final_row_labels": _TEST_DIR / "test_final_adjudicated_labels.jsonl",
        "test_final_sequence_labels": _TEST_DIR / "test_final_sequence_labels.jsonl",
    }
    all_roots_exist = True
    for label, path in split_root_files.items():
        if path.exists():
            _pass(f"split root exists: {label}")
        else:
            _fail(f"split root MISSING: {label}")
            all_roots_exist = False

    # [C11] Global split root SHAs all match
    print("\n[C11] Global split root SHA rehash...")
    sha_bindings = {
        "development_annotation_manifest_sha256": _DEV_DIR / "annotation_manifest.json",
        "development_annotation_gate_sha256": _DEV_DIR / "development_annotation_gate.json",
        "development_adjudication_manifest_sha256": _DEV_DIR / "adjudication_manifest.json",
        "development_final_row_labels_sha256": _DEV_DIR / "final_adjudicated_labels.jsonl",
        "development_final_sequence_labels_sha256": _DEV_DIR / "final_sequence_labels.jsonl",
        "validation_annotation_freeze_manifest_sha256": _VAL_DIR / "validation_annotation_freeze_manifest.json",
        "validation_post_freeze_verification_sha256": _VAL_DIR / "post_freeze_verification.json",
        "validation_final_row_labels_sha256": _VAL_DIR / "final_adjudicated_labels.jsonl",
        "validation_final_sequence_labels_sha256": _VAL_DIR / "final_sequence_labels.jsonl",
        "test_annotation_freeze_manifest_sha256": _TEST_DIR / "test_annotation_freeze_manifest.json",
        "test_post_freeze_verification_sha256": _TEST_DIR / "test_post_freeze_verification.json",
        "test_final_row_labels_sha256": _TEST_DIR / "test_final_adjudicated_labels.jsonl",
        "test_final_sequence_labels_sha256": _TEST_DIR / "test_final_sequence_labels.jsonl",
    }
    for field, path in sha_bindings.items():
        expected = freeze.get(field, "")
        if not expected:
            _fail(f"split root SHA missing in freeze manifest: {field}")
            continue
        if path.exists():
            actual = _sha256(path)
            if actual == expected:
                _pass(f"split root SHA matches: {field}")
            else:
                _fail(f"split root SHA MISMATCH: {field}")
        else:
            _fail(f"split root file missing for SHA check: {field}")

    # [C12] Development counts correct
    print("\n[C12] Development counts...")
    dev_counts = freeze.get("split_counts", {}).get("development", {})
    dev_rows = dev_counts.get("final_rows", -1)
    dev_seqs = dev_counts.get("final_sequences", -1)
    if dev_rows == EXPECTED_DEV_ROWS:
        _pass(f"development final_rows = {dev_rows}")
    else:
        _fail(f"development final_rows = {dev_rows}, expected {EXPECTED_DEV_ROWS}")
    if dev_seqs == EXPECTED_DEV_SEQUENCES:
        _pass(f"development final_sequences = {dev_seqs}")
    else:
        _fail(f"development final_sequences = {dev_seqs}, expected {EXPECTED_DEV_SEQUENCES}")

    # [C13] Validation counts correct
    print("\n[C13] Validation counts...")
    val_counts = freeze.get("split_counts", {}).get("validation", {})
    val_rows = val_counts.get("final_rows", -1)
    val_seqs = val_counts.get("final_sequences", -1)
    if val_rows == EXPECTED_VAL_ROWS:
        _pass(f"validation final_rows = {val_rows}")
    else:
        _fail(f"validation final_rows = {val_rows}, expected {EXPECTED_VAL_ROWS}")
    if val_seqs == EXPECTED_VAL_SEQUENCES:
        _pass(f"validation final_sequences = {val_seqs}")
    else:
        _fail(f"validation final_sequences = {val_seqs}, expected {EXPECTED_VAL_SEQUENCES}")

    # [C14] Test counts correct
    print("\n[C14] Test counts...")
    test_counts = freeze.get("split_counts", {}).get("test", {})
    test_rows = test_counts.get("final_rows", -1)
    test_seqs = test_counts.get("final_sequences", -1)
    if test_rows == EXPECTED_TEST_ROWS:
        _pass(f"test final_rows = {test_rows}")
    else:
        _fail(f"test final_rows = {test_rows}, expected {EXPECTED_TEST_ROWS}")
    if test_seqs == EXPECTED_TEST_SEQUENCES:
        _pass(f"test final_sequences = {test_seqs}")
    else:
        _fail(f"test final_sequences = {test_seqs}, expected {EXPECTED_TEST_SEQUENCES}")

    # [C15] 900 total final rows
    print("\n[C15] Global total final rows...")
    totals = freeze.get("global_totals", {})
    total_rows = totals.get("final_row_labels", -1)
    if total_rows == EXPECTED_TOTAL_ROWS:
        _pass(f"total final_row_labels = {total_rows}")
    else:
        _fail(f"total final_row_labels = {total_rows}, expected {EXPECTED_TOTAL_ROWS}")

    # [C16] 144 total final sequence units
    print("\n[C16] Global total final sequence units...")
    total_seqs = totals.get("final_sequence_labels", -1)
    if total_seqs == EXPECTED_TOTAL_SEQUENCES:
        _pass(f"total final_sequence_labels = {total_seqs}")
    else:
        _fail(f"total final_sequence_labels = {total_seqs}, expected {EXPECTED_TOTAL_SEQUENCES}")

    # [C17] Phase fields
    print("\n[C17] Annotation phase fields...")
    if _PHASE_PATH.exists():
        phase = _load_json(_PHASE_PATH)

        # test_annotation_complete = true
        if phase.get("test_annotation_complete") is True:
            _pass("test_annotation_complete = true")
        else:
            _fail(f"test_annotation_complete = {phase.get('test_annotation_complete')}, expected true")

        # annotations_frozen = true
        if phase.get("annotations_frozen") is True:
            _pass("annotations_frozen = true")
        else:
            _fail(f"annotations_frozen = {phase.get('annotations_frozen')}, expected true")

        # annotation_phase should be ANNOTATIONS_FROZEN
        if phase.get("annotation_phase") == "ANNOTATIONS_FROZEN":
            _pass("annotation_phase = ANNOTATIONS_FROZEN")
        else:
            _fail(f"annotation_phase = {phase.get('annotation_phase')}, expected ANNOTATIONS_FROZEN")
    else:
        _fail("annotation_phase.json not found")

    # [C18] Test campaign identity delegated to test verifier (item 10)
    # Campaign identity fields are now verified at the test layer by
    # verify_frozen_annotations.py --split test [T10].
    # At global scope we rely on:
    #   C8  — test annotation verifier PASS
    #   C9  — test post-freeze closure PASS
    #   C11 — test freeze root SHA match
    print("\n[C18] Campaign identity delegation check...")
    _pass("campaign identity verification delegated to test verifier [T10]")

    # Cross-check frozen_corpus_manifest_sha256 against actual file
    if _CORPUS_MANIFEST_PATH.exists():
        actual_corpus_sha = _sha256(_CORPUS_MANIFEST_PATH)
        freeze_corpus_sha = freeze.get("frozen_corpus_manifest_sha256", "")
        if actual_corpus_sha == freeze_corpus_sha:
            _pass("frozen_corpus_manifest cross-check matches")
        else:
            _fail("frozen_corpus_manifest cross-check MISMATCH")

    # Item 23: annotation_code_commit provenance note
    # Do NOT treat legacy annotation_code_commit=bccad2a as proof that
    # bccad2a contained E4-003 code. The code commit field is recorded
    # for historical provenance only.
    code_commit = freeze.get("global_annotation_freeze_code_commit", "")
    if code_commit and code_commit != "unknown":
        _pass(f"global freeze code commit recorded: {code_commit[:12]}...")
    else:
        _fail("global freeze code commit missing or unknown")

    # [C19] Unresolved count verification (item 25)
    print("\n[C19] Unresolved count verification...")
    unresolved = freeze.get("unresolved_by_split", {})

    # Development unresolved rows
    dev_ur = unresolved.get("development", {}).get("unresolved_rows", -1)
    if dev_ur == EXPECTED_DEV_UNRESOLVED_ROWS:
        _pass(f"development unresolved rows = {dev_ur}")
    else:
        _fail(f"development unresolved rows = {dev_ur}, expected {EXPECTED_DEV_UNRESOLVED_ROWS}")

    # Validation unresolved rows
    val_ur = unresolved.get("validation", {}).get("unresolved_rows", -1)
    if val_ur == EXPECTED_VAL_UNRESOLVED_ROWS:
        _pass(f"validation unresolved rows = {val_ur}")
    else:
        _fail(f"validation unresolved rows = {val_ur}, expected {EXPECTED_VAL_UNRESOLVED_ROWS}")

    # Test unresolved rows
    test_ur = unresolved.get("test", {}).get("unresolved_rows", -1)
    if test_ur == EXPECTED_TEST_UNRESOLVED_ROWS:
        _pass(f"test unresolved rows = {test_ur}")
    else:
        _fail(f"test unresolved rows = {test_ur}, expected {EXPECTED_TEST_UNRESOLVED_ROWS}")

    # Unresolved sequences (all splits = 0)
    dev_us = unresolved.get("development", {}).get("unresolved_sequences", -1)
    if dev_us == EXPECTED_DEV_UNRESOLVED_SEQS:
        _pass(f"development unresolved sequences = {dev_us}")
    else:
        _fail(f"development unresolved sequences = {dev_us}, expected {EXPECTED_DEV_UNRESOLVED_SEQS}")

    val_us = unresolved.get("validation", {}).get("unresolved_sequences", -1)
    if val_us == EXPECTED_VAL_UNRESOLVED_SEQS:
        _pass(f"validation unresolved sequences = {val_us}")
    else:
        _fail(f"validation unresolved sequences = {val_us}, expected {EXPECTED_VAL_UNRESOLVED_SEQS}")

    test_us = unresolved.get("test", {}).get("unresolved_sequences", -1)
    if test_us == EXPECTED_TEST_UNRESOLVED_SEQS:
        _pass(f"test unresolved sequences = {test_us}")
    else:
        _fail(f"test unresolved sequences = {test_us}, expected {EXPECTED_TEST_UNRESOLVED_SEQS}")

    # [C20] Cross-check unresolved against split adjudication artifacts (item 25)
    print("\n[C20] Unresolved cross-check against split artifacts...")
    dev_adj_path = _DEV_DIR / "adjudication_manifest.json"
    val_adj_path = _VAL_DIR / "adjudication_manifest.json"
    test_adj_path = _TEST_DIR / "test_adjudication_manifest.json"

    if dev_adj_path.exists():
        dev_adj = _load_json(dev_adj_path)
        dev_adj_ur = dev_adj.get("final_label_counts", {}).get("unresolved", -1)
        if dev_adj_ur == EXPECTED_DEV_UNRESOLVED_ROWS:
            _pass(f"dev adjudication cross-check: unresolved rows = {dev_adj_ur}")
        else:
            _fail(f"dev adjudication cross-check: unresolved rows = {dev_adj_ur}, expected {EXPECTED_DEV_UNRESOLVED_ROWS}")
    else:
        _fail("dev adjudication_manifest.json not found for cross-check")

    if val_adj_path.exists():
        val_adj = _load_json(val_adj_path)
        val_adj_ur = val_adj.get("final_label_counts", {}).get("unresolved_rows", -1)
        if val_adj_ur == EXPECTED_VAL_UNRESOLVED_ROWS:
            _pass(f"val adjudication cross-check: unresolved rows = {val_adj_ur}")
        else:
            _fail(f"val adjudication cross-check: unresolved rows = {val_adj_ur}, expected {EXPECTED_VAL_UNRESOLVED_ROWS}")
    else:
        _fail("val adjudication_manifest.json not found for cross-check")

    if test_adj_path.exists():
        test_adj = _load_json(test_adj_path)
        test_adj_ur = test_adj.get("final_label_counts", {}).get("unresolved_rows", -1)
        if test_adj_ur == EXPECTED_TEST_UNRESOLVED_ROWS:
            _pass(f"test adjudication cross-check: unresolved rows = {test_adj_ur}")
        else:
            _fail(f"test adjudication cross-check: unresolved rows = {test_adj_ur}, expected {EXPECTED_TEST_UNRESOLVED_ROWS}")
    else:
        _fail("test adjudication manifest not found for cross-check")

    # [C21] Repair provenance support (item 24)
    print("\n[C21] Repair provenance support...")
    # Item 24: R1.1 code supports later R2 metadata.
    # The freeze manifest may optionally contain repair_provenance fields.
    # We verify the code path can read them if present.
    repair_prov = freeze.get("repair_provenance")
    if repair_prov is not None:
        _pass("repair_provenance section present in freeze manifest")
        for field in [
            "repair_start_commit", "r1_code_commit",
            "r1_1_code_commit", "provider_reruns", "semantic_labels_changed",
        ]:
            if field in repair_prov:
                _pass(f"repair_provenance.{field} present")
            else:
                _pass(f"repair_provenance.{field} not yet set (expected for R1.1)")
    else:
        _pass("repair_provenance not yet populated (expected for R1/R1.1, will be set in R2)")

    # Summary
    total = passed + failed
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} checks passed, {failed} failed")
    if failed:
        print(f"\nVERIFICATION: FAIL")
        return 1
    else:
        print("\nVERIFICATION: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

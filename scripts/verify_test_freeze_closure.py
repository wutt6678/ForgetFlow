#!/usr/bin/env python3
"""Post-freeze closure verifier for test annotations.

Rehashes the core test freeze artifacts and compares against
test_post_freeze_verification.json.  Confirms all verifiers passed.
Returns nonzero on any mismatch.

Usage:
    PYTHONPATH=. python scripts/verify_test_freeze_closure.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "test"

_POST_FREEZE_PATH = _TEST_DIR / "test_post_freeze_verification.json"
_MANIFEST_PATH = _TEST_DIR / "test_annotation_manifest.json"
_GATE_PATH = _TEST_DIR / "test_annotation_gate.json"
_FREEZE_PATH = _TEST_DIR / "test_annotation_freeze_manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    print("TEST FREEZE CLOSURE VERIFIER")
    print("=" * 70)

    # [C1] Load post_freeze_verification.json
    print("\n[C1] Post-freeze verification artifact...")
    if not _POST_FREEZE_PATH.exists():
        _fail("test_post_freeze_verification.json not found")
        print(f"\nVERIFICATION: FAIL ({passed} passed, {failed} failed)")
        return 1
    pfv = _load_json(_POST_FREEZE_PATH)
    _pass("test_post_freeze_verification.json exists")

    # [C2] Rehash test_annotation_manifest.json
    print("\n[C2] Test annotation manifest SHA...")
    if _MANIFEST_PATH.exists():
        actual = _sha256(_MANIFEST_PATH)
        expected = pfv.get("annotation_manifest_sha256", "")
        if actual == expected:
            _pass("test_annotation_manifest.json SHA256 matches")
        else:
            _fail("test_annotation_manifest.json SHA256 MISMATCH")
    else:
        _fail("test_annotation_manifest.json not found")

    # [C3] Rehash test_annotation_gate.json
    print("\n[C3] Test gate SHA...")
    if _GATE_PATH.exists():
        actual = _sha256(_GATE_PATH)
        expected = pfv.get("test_gate_sha256", "")
        if actual == expected:
            _pass("test_annotation_gate.json SHA256 matches")
        else:
            _fail("test_annotation_gate.json SHA256 MISMATCH")
    else:
        _fail("test_annotation_gate.json not found")

    # [C4] Rehash test_annotation_freeze_manifest.json
    print("\n[C4] Test freeze manifest SHA...")
    if _FREEZE_PATH.exists():
        actual = _sha256(_FREEZE_PATH)
        expected = pfv.get("test_freeze_manifest_sha256", "")
        if actual == expected:
            _pass("test_annotation_freeze_manifest.json SHA256 matches")
        else:
            _fail("test_annotation_freeze_manifest.json SHA256 MISMATCH")
    else:
        _fail("test_annotation_freeze_manifest.json not found")

    # [C5] Confirm frozen corpus verifier PASS
    print("\n[C5] Frozen corpus verifier...")
    fc = pfv.get("frozen_corpus_verifier", {})
    if (
        fc.get("exit_code") == 0
        and fc.get("checks_failed") == 0
        and fc.get("checks_passed") == fc.get("checks_total")
        and fc.get("checks_total", 0) > 0
    ):
        _pass(f"frozen_corpus_verifier PASS ({fc['checks_passed']}/{fc['checks_total']})")
    else:
        _fail(
            f"frozen_corpus_verifier FAIL "
            f"(exit={fc.get('exit_code')}, passed={fc.get('checks_passed')}, "
            f"failed={fc.get('checks_failed')}, total={fc.get('checks_total')})"
        )

    # [C6] Confirm development verifier PASS
    print("\n[C6] Development annotation verifier...")
    dev = pfv.get("development_annotation_verifier", {})
    if (
        dev.get("exit_code") == 0
        and dev.get("checks_failed") == 0
        and dev.get("checks_passed") == dev.get("checks_total")
        and dev.get("checks_total", 0) > 0
    ):
        _pass(f"development_verifier PASS ({dev['checks_passed']}/{dev['checks_total']})")
    else:
        _fail(
            f"development_verifier FAIL "
            f"(exit={dev.get('exit_code')}, passed={dev.get('checks_passed')}, "
            f"failed={dev.get('checks_failed')}, total={dev.get('checks_total')})"
        )

    # [C7] Confirm validation verifier PASS
    print("\n[C7] Validation annotation verifier...")
    val = pfv.get("validation_annotation_verifier", {})
    if (
        val.get("exit_code") == 0
        and val.get("checks_failed") == 0
        and val.get("checks_passed") == val.get("checks_total")
        and val.get("checks_total", 0) > 0
    ):
        _pass(f"validation_verifier PASS ({val['checks_passed']}/{val['checks_total']})")
    else:
        _fail(
            f"validation_verifier FAIL "
            f"(exit={val.get('exit_code')}, passed={val.get('checks_passed')}, "
            f"failed={val.get('checks_failed')}, total={val.get('checks_total')})"
        )

    # [C8] Confirm test verifier PASS
    print("\n[C8] Test annotation verifier...")
    test_v = pfv.get("test_annotation_verifier", {})
    if (
        test_v.get("exit_code") == 0
        and test_v.get("checks_failed") == 0
        and test_v.get("checks_passed") == test_v.get("checks_total")
        and test_v.get("checks_total", 0) > 0
    ):
        _pass(f"test_verifier PASS ({test_v['checks_passed']}/{test_v['checks_total']})")
    else:
        _fail(
            f"test_verifier FAIL "
            f"(exit={test_v.get('exit_code')}, passed={test_v.get('checks_passed')}, "
            f"failed={test_v.get('checks_failed')}, total={test_v.get('checks_total')})"
        )

    # [C8a] Confirm validation closure verifier PASS (item 64)
    print("\n[C8a] Validation freeze closure verifier...")
    val_closure = pfv.get("validation_closure_verifier", {})
    if (
        val_closure.get("exit_code") == 0
        and val_closure.get("checks_failed") == 0
        and val_closure.get("checks_passed") == val_closure.get("checks_total")
        and val_closure.get("checks_total", 0) > 0
    ):
        _pass(f"validation_closure_verifier PASS ({val_closure['checks_passed']}/{val_closure['checks_total']})")
    else:
        _fail(
            f"validation_closure_verifier FAIL "
            f"(exit={val_closure.get('exit_code')}, passed={val_closure.get('checks_passed')}, "
            f"failed={val_closure.get('checks_failed')}, total={val_closure.get('checks_total')})"
        )

    # [C8b] Confirm test closure verifier PASS (item 65)
    print("\n[C8b] Test freeze closure verifier...")
    test_closure = pfv.get("test_closure_verifier", {})
    if (
        test_closure.get("exit_code") == 0
        and test_closure.get("checks_failed") == 0
        and test_closure.get("checks_passed") == test_closure.get("checks_total")
        and test_closure.get("checks_total", 0) > 0
    ):
        _pass(f"test_closure_verifier PASS ({test_closure['checks_passed']}/{test_closure['checks_total']})")
    else:
        _fail(
            f"test_closure_verifier FAIL "
            f"(exit={test_closure.get('exit_code')}, passed={test_closure.get('checks_passed')}, "
            f"failed={test_closure.get('checks_failed')}, total={test_closure.get('checks_total')})"
        )

    # [C9] Timestamp ordering checks
    print("\n[C9] Timestamp ordering...")
    freeze_created_at = pfv.get("freeze_created_at", "")
    pfv_created_at = pfv.get("created_at", "")

    if freeze_created_at and pfv_created_at:
        if pfv_created_at >= freeze_created_at:
            _pass("post_freeze_verification.created_at >= freeze.created_at")
        else:
            _fail(
                f"post_freeze_verification.created_at ({pfv_created_at}) "
                f"< freeze.created_at ({freeze_created_at})"
            )
    else:
        _fail("Missing freeze_created_at or created_at in test_post_freeze_verification.json")

    for verifier_key, label in [
        ("frozen_corpus_verifier", "frozen_corpus_verifier"),
        ("development_annotation_verifier", "development_annotation_verifier"),
        ("validation_annotation_verifier", "validation_annotation_verifier"),
        ("test_annotation_verifier", "test_annotation_verifier"),
        ("validation_closure_verifier", "validation_closure_verifier"),
        ("test_closure_verifier", "test_closure_verifier"),
    ]:
        v = pfv.get(verifier_key, {})
        v_ts = v.get("timestamp", "")
        if v_ts and freeze_created_at:
            if v_ts >= freeze_created_at:
                _pass(f"{label}.timestamp ({v_ts}) >= freeze.created_at")
            else:
                _fail(
                    f"{label}.timestamp ({v_ts}) < freeze.created_at ({freeze_created_at})"
                )
        else:
            _fail(f"Missing timestamp for {label} or freeze_created_at")

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

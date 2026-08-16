#!/usr/bin/env python3
"""Post-freeze closure verifier for validation annotations.

§16: Rehashes the three core freeze artifacts and compares against
post_freeze_verification.json.  Confirms all three verifiers passed.
Returns nonzero on any mismatch.

Usage:
    PYTHONPATH=. python scripts/verify_validation_freeze_closure.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VAL_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "validation"

_POST_FREEZE_PATH = _VAL_DIR / "post_freeze_verification.json"
_MANIFEST_PATH = _VAL_DIR / "annotation_manifest.json"
_GATE_PATH = _VAL_DIR / "validation_annotation_gate.json"
_FREEZE_PATH = _VAL_DIR / "validation_annotation_freeze_manifest.json"


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
    print("VALIDATION FREEZE CLOSURE VERIFIER")
    print("=" * 70)

    # 1. Load post_freeze_verification.json
    print("\n[C1] Post-freeze verification artifact...")
    if not _POST_FREEZE_PATH.exists():
        _fail("post_freeze_verification.json not found")
        print(f"\nVERIFICATION: FAIL ({passed} passed, {failed} failed)")
        return 1
    pfv = _load_json(_POST_FREEZE_PATH)
    _pass("post_freeze_verification.json exists")

    # 2. Rehash annotation_manifest.json
    print("\n[C2] Annotation manifest SHA...")
    if _MANIFEST_PATH.exists():
        actual = _sha256(_MANIFEST_PATH)
        expected = pfv.get("annotation_manifest_sha256", "")
        if actual == expected:
            _pass("annotation_manifest.json SHA256 matches")
        else:
            _fail("annotation_manifest.json SHA256 MISMATCH")
    else:
        _fail("annotation_manifest.json not found")

    # 3. Rehash validation_annotation_gate.json
    print("\n[C3] Validation gate SHA...")
    if _GATE_PATH.exists():
        actual = _sha256(_GATE_PATH)
        expected = pfv.get("validation_gate_sha256", "")
        if actual == expected:
            _pass("validation_annotation_gate.json SHA256 matches")
        else:
            _fail("validation_annotation_gate.json SHA256 MISMATCH")
    else:
        _fail("validation_annotation_gate.json not found")

    # 4. Rehash validation_annotation_freeze_manifest.json
    print("\n[C4] Freeze manifest SHA...")
    if _FREEZE_PATH.exists():
        actual = _sha256(_FREEZE_PATH)
        expected = pfv.get("validation_freeze_manifest_sha256", "")
        if actual == expected:
            _pass("validation_annotation_freeze_manifest.json SHA256 matches")
        else:
            _fail("validation_annotation_freeze_manifest.json SHA256 MISMATCH")
    else:
        _fail("validation_annotation_freeze_manifest.json not found")

    # 5. Confirm frozen corpus verifier PASS
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

    # 6. Confirm development verifier PASS
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

    # 7. Confirm validation verifier PASS
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

    # 8. Timestamp ordering checks
    print("\n[C8] Timestamp ordering...")
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
        _fail("Missing freeze_created_at or created_at in post_freeze_verification.json")

    # Check each verifier timestamp >= freeze_created_at
    for verifier_key, label in [
        ("frozen_corpus_verifier", "frozen_corpus_verifier"),
        ("development_annotation_verifier", "development_annotation_verifier"),
        ("validation_annotation_verifier", "validation_annotation_verifier"),
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
        print(f"\nBLOCKING FINDINGS ({failed}):")
        print("\nVERIFICATION: FAIL")
        return 1
    else:
        print("\nVERIFICATION: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

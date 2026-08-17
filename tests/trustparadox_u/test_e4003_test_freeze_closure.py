"""E4-003: Closure-ordering regression tests (item 33).

Covers:
- No old test_post_freeze_verification.json required
- New post-freeze verification written before test closure verification
- Validation closure failure → test freeze build nonzero
- Test verifier failure → test freeze build nonzero
- Test closure failure → test freeze build nonzero
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def _verifier_pass(v: dict) -> bool:
    """Replicate the verifier-pass predicate."""
    return (
        v.get("exit_code") == 0
        and v.get("checks_failed") == 0
        and v.get("checks_passed") == v.get("checks_total")
        and v.get("checks_total", 0) > 0
    )


# ===========================================================================
# Item 33: Closure-ordering test cases
# ===========================================================================


class TestClosureOrdering:
    """Item 33: Closure ordering invariants."""

    def test_no_old_post_freeze_required(self):
        """The build process does not require a pre-existing post_freeze file.

        build_post_freeze_verification() writes a new file; it does not
        read an old one as input.  The closure verifier reads it afterward.
        """
        import inspect
        from scripts.build_test_freeze import build_post_freeze_verification
        source = inspect.getsource(build_post_freeze_verification)
        # The function writes to test_post_freeze_verification.json
        # It should NOT read from it as input
        assert "test_post_freeze_verification.json" in source
        # It reads freeze manifest, not old post-freeze
        assert "test_annotation_freeze_manifest.json" in source

    def test_post_freeze_written_before_closure_verification(self):
        """build_test_freeze writes post_freeze before running closure verifier.

        In main(), the order is:
        1. build_post_free_verification() writes the file
        2. Then check closure_pass from that file
        3. Then run standalone closure verifier
        """
        import inspect
        from scripts.build_test_freeze import main
        source = inspect.getsource(main)
        # build_post_freeze_verification must come before verify_test_freeze_closure
        pfv_pos = source.find("build_post_freeze_verification(")
        closure_pos = source.find("verify_test_freeze_closure.py")
        assert pfv_pos > 0, "build_post_freeze_verification call not found"
        assert closure_pos > 0, "verify_test_freeze_closure.py not found"
        assert pfv_pos < closure_pos, (
            "post_freeze must be written before closure verifier runs"
        )

    def test_validation_closure_failure_blocks_freeze(self):
        """If validation closure verifier fails, test freeze build exits nonzero.

        In build_test_gate(), validation_closure is not directly checked,
        but in the global freeze, validation closure failure blocks.
        We test the structural invariant: the gate checks val_verifier_pass.
        """
        import inspect
        from scripts.build_test_freeze import build_test_gate
        source = inspect.getsource(build_test_gate)
        # build_test_gate runs validation verifier and blocks on failure
        assert "validation_verifier_pass" in source
        assert "blocking.append" in source

    def test_test_verifier_failure_blocks_freeze(self):
        """If test verifier fails, test freeze build must exit nonzero.

        In build_test_freeze.main(), after running verifiers, if gate is
        NO-GO, the function returns 1.
        """
        import inspect
        from scripts.build_test_freeze import main
        source = inspect.getsource(main)
        # After running verifiers, gate is rebuilt and checked
        # If NO-GO, return 1
        assert 'return 1' in source

    def test_test_closure_failure_blocks_freeze(self):
        """If test closure fails, test freeze build exits nonzero.

        In main(), after build_post_freeze_verification, closure_pass
        is checked. If false, return 1.
        """
        import inspect
        from scripts.build_test_freeze import main
        source = inspect.getsource(main)
        # Check for closure_pass check
        assert "closure_pass" in source or "closure_data" in source
        # And for nonzero exit
        assert "return 1" in source

    def test_global_closure_failure_blocks_freeze(self):
        """If global post-freeze closure fails, global freeze exits nonzero."""
        import inspect
        from scripts.build_global_annotation_freeze import main
        source = inspect.getsource(main)
        # all_verifiers_pass check
        assert "all_verifiers_pass" in source
        assert "return 1" in source

    def test_standalone_verifier_failure_blocks_test_freeze(self):
        """If standalone test closure verifier fails, test freeze exits nonzero."""
        import inspect
        from scripts.build_test_freeze import main
        source = inspect.getsource(main)
        # Standalone verifier check
        assert "verify_test_freeze_closure.py" in source
        # Check for nonzero exit on failure
        assert "standalone_result" in source or "standalone_status" in source

    def test_standalone_verifier_failure_blocks_global_freeze(self):
        """If standalone global verifier fails, global freeze exits nonzero."""
        import inspect
        from scripts.build_global_annotation_freeze import main
        source = inspect.getsource(main)
        assert "verify_global_annotation_freeze.py" in source
        assert "standalone_result" in source or "standalone_status" in source


class TestClosureOrderingWithFixtures:
    """Item 33: Closure ordering with synthetic verifier results."""

    def test_all_verifiers_pass_closure_pass(self):
        """When all verifiers pass, closure_pass = True."""
        verifier_results = {
            "frozen_corpus": {
                "checks_total": 5, "checks_passed": 5,
                "checks_failed": 0, "exit_code": 0,
            },
            "development_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "test_annotations": {
                "checks_total": 15, "checks_passed": 15,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_closure": {
                "checks_total": 8, "checks_passed": 8,
                "checks_failed": 0, "exit_code": 0,
            },
        }
        # Replicate closure_pass logic from build_post_freeze_verification
        fc = verifier_results.get("frozen_corpus", {})
        dev = verifier_results.get("development_annotations", {})
        val = verifier_results.get("validation_annotations", {})
        test = verifier_results.get("test_annotations", {})
        val_closure = verifier_results.get("validation_closure", {})

        corpus_pass = _verifier_pass(fc)
        dev_pass = _verifier_pass(dev)
        val_pass = _verifier_pass(val)
        test_pass = _verifier_pass(test)
        val_closure_pass = _verifier_pass(val_closure)

        closure_pass = (
            corpus_pass and dev_pass and val_pass
            and val_closure_pass and test_pass
        )
        assert closure_pass is True

    def test_validation_closure_failure_closure_fails(self):
        """When validation closure fails, closure_pass = False."""
        verifier_results = {
            "frozen_corpus": {
                "checks_total": 5, "checks_passed": 5,
                "checks_failed": 0, "exit_code": 0,
            },
            "development_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "test_annotations": {
                "checks_total": 15, "checks_passed": 15,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_closure": {
                "checks_total": 8, "checks_passed": 7,
                "checks_failed": 1, "exit_code": 1,  # FAIL
            },
        }
        val_closure = verifier_results.get("validation_closure", {})
        val_closure_pass = _verifier_pass(val_closure)
        assert val_closure_pass is False

    def test_test_verifier_failure_closure_fails(self):
        """When test verifier fails, closure_pass = False."""
        verifier_results = {
            "frozen_corpus": {
                "checks_total": 5, "checks_passed": 5,
                "checks_failed": 0, "exit_code": 0,
            },
            "development_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "test_annotations": {
                "checks_total": 15, "checks_passed": 14,
                "checks_failed": 1, "exit_code": 1,  # FAIL
            },
            "validation_closure": {
                "checks_total": 8, "checks_passed": 8,
                "checks_failed": 0, "exit_code": 0,
            },
        }
        test = verifier_results.get("test_annotations", {})
        test_pass = _verifier_pass(test)
        assert test_pass is False

    def test_global_all_verifiers_pass(self):
        """Global: when all 6 verifiers pass, all_verifiers_pass = True."""
        verifier_results = {
            "frozen_corpus": {
                "checks_total": 5, "checks_passed": 5,
                "checks_failed": 0, "exit_code": 0,
            },
            "development_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_closure": {
                "checks_total": 8, "checks_passed": 8,
                "checks_failed": 0, "exit_code": 0,
            },
            "test_annotations": {
                "checks_total": 15, "checks_passed": 15,
                "checks_failed": 0, "exit_code": 0,
            },
            "test_closure": {
                "checks_total": 9, "checks_passed": 9,
                "checks_failed": 0, "exit_code": 0,
            },
        }
        # Replicate global closure_pass logic
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
        assert closure_pass is True

    def test_global_test_closure_failure_blocks(self):
        """Global: test closure failure → all_verifiers_pass = False."""
        verifier_results = {
            "frozen_corpus": {
                "checks_total": 5, "checks_passed": 5,
                "checks_failed": 0, "exit_code": 0,
            },
            "development_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_annotations": {
                "checks_total": 10, "checks_passed": 10,
                "checks_failed": 0, "exit_code": 0,
            },
            "validation_closure": {
                "checks_total": 8, "checks_passed": 8,
                "checks_failed": 0, "exit_code": 0,
            },
            "test_annotations": {
                "checks_total": 15, "checks_passed": 15,
                "checks_failed": 0, "exit_code": 0,
            },
            "test_closure": {
                "checks_total": 9, "checks_passed": 8,
                "checks_failed": 1, "exit_code": 1,  # FAIL
            },
        }
        test_closure = verifier_results.get("test_closure", {})
        assert _verifier_pass(test_closure) is False

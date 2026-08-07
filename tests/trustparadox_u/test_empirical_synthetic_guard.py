"""E1-033: synthetic-regression guard.

The empirical infrastructure must not alter synthetic result semantics:

- synthetic input data (scenarios, splits, schema) must be unmodified;
- frozen synthetic result trees must be unmodified;
- empirical modules must not import synthetic corpus modules;
- the existing synthetic test suite (run as part of ``pytest``) continues
  to pass unchanged.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SYNTHETIC_PATHS = (
    "data/trustparadox_u/scenarios",
    "data/trustparadox_u/schema",
    "data/trustparadox_u/splits",
)

_EMPIRICAL_MODULES = (
    "experiments/trustparadox_u/empirical_corpus.py",
    "experiments/trustparadox_u/empirical_generation.py",
    "experiments/trustparadox_u/generate_empirical_corpus.py",
)

# Synthetic-side modules the empirical infrastructure must not depend on.
_FORBIDDEN_SYNTHETIC_IMPORTS = (
    "experiments.trustparadox_u.candidates",
    "experiments.trustparadox_u.generate_corpus",
    "experiments.trustparadox_u.dataset",
    "experiments.trustparadox_u.attacks",
    "experiments.trustparadox_u.evaluator",
    "experiments.trustparadox_u.aggregate",
)


def _git_status(paths: tuple[str, ...]) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class TestSyntheticDataUnchanged:
    @pytest.mark.parametrize("relative_path", _SYNTHETIC_PATHS)
    def test_synthetic_inputs_have_no_modifications(self, relative_path: str) -> None:
        """No modified or deleted files under synthetic input trees.

        Untracked files are excluded: the empirical namespace lives under
        ``data/trustparadox_u/empirical_v2`` and never inside these trees.
        """
        status = _git_status((relative_path,))
        changed = [line for line in status.splitlines() if not line.startswith("??")]
        assert changed == [], f"synthetic inputs modified: {changed}"

    def test_no_synthetic_file_deletion_or_rename(self) -> None:
        result = subprocess.run(
            ["git", "diff", "--name-status", "HEAD", "--", *_SYNTHETIC_PATHS],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == ""


class TestEmpiricalModuleIsolation:
    @pytest.mark.parametrize("relative_module", _EMPIRICAL_MODULES)
    def test_empirical_modules_do_not_import_synthetic_code(self, relative_module: str) -> None:
        source = (_PROJECT_ROOT / relative_module).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        violations = [
            module
            for module in imported
            if any(
                module == forbidden or module.startswith(forbidden + ".")
                for forbidden in _FORBIDDEN_SYNTHETIC_IMPORTS
            )
        ]
        assert violations == [], f"{relative_module} imports synthetic code: {violations}"

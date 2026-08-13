"""Patch D — campaign output workflow and clean-tree tests.

Tests:
  D1: Ignored campaign output does not dirty the Git tree.
  D2: Modifying a tracked source file makes the tree dirty.
  D3: Untracked non-ignored files make the tree dirty.
  D4: Resume after writing ignored raw attempts passes clean-tree gate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd or _PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def _init_temp_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo in *tmp_path* with one tracked file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", cwd=repo)
    _git("config", "user.email", "test@test.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    # Write a .gitignore that mirrors the production one for campaign output.
    (repo / ".gitignore").write_text(
        "results/empirical_v2/corpus_generation/\n"
        "results/empirical_v2/real_api_preflight/\n"
    )
    (repo / "tracked.py").write_text("# tracked source\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "init", cwd=repo)
    return repo


# ---------------------------------------------------------------------------
# Test D1 — ignored campaign output does not dirty the tree
# ---------------------------------------------------------------------------


class TestIgnoredCampaignOutput:
    """D1: generated campaign outputs in .gitignore dirs stay clean."""

    def test_ignored_output_keeps_tree_clean(self, tmp_path: Path) -> None:
        """Create files in the ignored campaign output directories.

        Assert ``git status --porcelain`` remains clean.
        """
        repo = _init_temp_repo(tmp_path)

        # Create files in the ignored directories.
        corpus_dir = repo / "results" / "empirical_v2" / "corpus_generation"
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "raw_generation_attempts.jsonl").write_text("{}\n")

        preflight_dir = repo / "results" / "empirical_v2" / "real_api_preflight"
        preflight_dir.mkdir(parents=True)
        (preflight_dir / "preflight_report.json").write_text("{}\n")

        # git status should be clean.
        result = _git("status", "--porcelain", cwd=repo)
        assert result.returncode == 0
        assert result.stdout.strip() == "", (
            f"git status should be clean but got:\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# Test D2 — modifying a tracked file makes the tree dirty
# ---------------------------------------------------------------------------


class TestTrackedFileModification:
    """D2: source-code changes still make the tree dirty."""

    def test_tracked_modification_dirties_tree(self, tmp_path: Path) -> None:
        repo = _init_temp_repo(tmp_path)

        # Modify the tracked file.
        (repo / "tracked.py").write_text("# modified tracked source\n")

        result = _git("status", "--porcelain", cwd=repo)
        assert result.returncode == 0
        assert result.stdout.strip() != "", "tree should be dirty after tracked modification"
        assert "tracked.py" in result.stdout


# ---------------------------------------------------------------------------
# Test D3 — untracked non-ignored files make the tree dirty
# ---------------------------------------------------------------------------


class TestUntrackedNonIgnoredFiles:
    """D3: untracked non-ignored experiment-definition files fail the gate."""

    def test_untracked_non_ignored_dirties_tree(self, tmp_path: Path) -> None:
        repo = _init_temp_repo(tmp_path)

        # Create an untracked file that is NOT in .gitignore.
        (repo / "new_experiment.py").write_text("# untracked\n")

        result = _git("status", "--porcelain", cwd=repo)
        assert result.returncode == 0
        assert result.stdout.strip() != "", "tree should be dirty with untracked non-ignored file"
        assert "new_experiment.py" in result.stdout


# ---------------------------------------------------------------------------
# Test D4 — resume after writing ignored raw attempts passes clean-tree gate
# ---------------------------------------------------------------------------


class TestResumeWithIgnoredAttempts:
    """D4: resume after writing ignored raw attempts passes clean-tree gate."""

    def test_resume_with_ignored_raw_attempts(self, tmp_path: Path) -> None:
        """Write raw attempts into the ignored directory.

        Assert:
        - same commit is still HEAD
        - clean-tree gate passes (git status clean)
        """
        repo = _init_temp_repo(tmp_path)

        # Record the current HEAD commit.
        head_before = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()

        # Write raw attempts into the ignored campaign output directory.
        corpus_dir = repo / "results" / "empirical_v2" / "corpus_generation" / "development"
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "raw_generation_attempts.jsonl").write_text(
            '{"generation_attempt_id": "ega_test"}\n'
        )

        # HEAD should not have changed.
        head_after = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()
        assert head_before == head_after, "commit should not change during output generation"

        # Clean-tree gate should pass.
        result = _git("status", "--porcelain", cwd=repo)
        assert result.returncode == 0
        assert result.stdout.strip() == "", (
            f"clean-tree gate should pass but got:\n{result.stdout}"
        )

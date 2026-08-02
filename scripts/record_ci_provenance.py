#!/usr/bin/env python3
"""Record CI provenance metadata (Section 17).

Captures the workflow run identity and execution environment so that every
result set can be traced back to an exact GitHub Actions run:

- workflow run ID / run number / attempt number
- job ID (job name)
- checkout SHA and ref
- runner image (ImageOS / ImageVersion) and OS/arch
- Python and Poetry versions

Usage:
    poetry run python scripts/record_ci_provenance.py --output ci/ci_provenance.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _version(cmd: list[str]) -> str:
    """Best-effort command version capture."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_provenance() -> dict[str, str]:
    """Collect CI provenance from the environment and runtime."""
    return {
        "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_number": os.environ.get("GITHUB_RUN_NUMBER", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "job_id": os.environ.get("GITHUB_JOB", ""),
        "checkout_sha": os.environ.get("GITHUB_SHA", ""),
        "ref": os.environ.get("GITHUB_REF", ""),
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "runner_os": os.environ.get("RUNNER_OS", ""),
        "runner_arch": os.environ.get("RUNNER_ARCH", ""),
        "runner_image_os": os.environ.get("ImageOS", ""),
        "runner_image_version": os.environ.get("ImageVersion", ""),
        "python_version": _version([sys.executable, "--version"]),
        "poetry_version": _version(["poetry", "--version"]),
        "platform": platform.platform(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Record CI provenance metadata")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write ci_provenance.json",
    )
    parser.add_argument(
        "--repository-state",
        type=Path,
        default=None,
        help="Optional path to also write repository_state.json",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="",
        help="Tested/results commit SHA (for repository_state.json)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Mark the repository as clean (for repository_state.json)",
    )
    args = parser.parse_args()

    provenance = collect_provenance()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(provenance, indent=2, sort_keys=True))

    print(f"CI provenance written to {args.output}")
    print(json.dumps(provenance, indent=2, sort_keys=True))

    if args.repository_state is not None:
        commit = args.commit or provenance.get("checkout_sha", "")
        repository_state = {
            "tested_commit": commit,
            "results_commit": commit,
            "repository_clean": args.clean,
            "python_version": provenance.get("python_version", ""),
            "poetry_version": provenance.get("poetry_version", ""),
            "os_info": provenance.get("platform", ""),
        }
        args.repository_state.parent.mkdir(parents=True, exist_ok=True)
        args.repository_state.write_text(json.dumps(repository_state, indent=2, sort_keys=True))
        print(f"Repository state written to {args.repository_state}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

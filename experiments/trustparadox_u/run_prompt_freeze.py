"""E2-H: prompt freeze pipeline.

After the bounded revision (E2-G), this step freezes the trust prompts
and produces the final prompt manifest for the empirical study.

The freeze:
1. Validates that the bounded revision decision permits freezing.
2. Re-validates prompt invariance (only TRUST_FRAMING differs).
3. Computes SHA-256 hashes of all prompt templates.
4. Writes the frozen prompt manifest (status = "frozen_post_pilot").
5. Records the freeze timestamp and repository commit.

After this step, the prompts are locked. Any change requires a
protocol_version bump and a new corpus version.

Inputs:
  - bounded_revision_report.json (from E2-G)
  - Prompt templates in data/trustparadox_u/empirical_v2/prompts/

Outputs:
  - frozen_prompt_manifest.json: the final frozen prompt manifest.
  - freeze_report (frozen_freeze_report.json): freeze decision record.

Checklist coverage:
- E2-H: prompt freeze
- E2-053: prompt freeze manifest
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.empirical_generation import (  # noqa: E402
    build_prompt_manifest,
    prompt_manifest_sha256,
    validate_trust_prompt_invariance,
)
from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION  # noqa: E402

FROZEN_STATUS = "frozen_post_pilot"

# Permitted E2-G decisions that allow freezing.
_FREEZABLE_DECISIONS = {"freeze_as_is", "judgement_freeze_with_findings"}


def _get_repository_commit(repo_root: Path | None = None) -> str:
    """Get the current git commit hash."""
    root = repo_root or _PROJECT_ROOT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return "unknown"


def validate_freeze_preconditions(
    revision_report: dict[str, Any],
) -> list[str]:
    """Validate that the bounded revision permits freezing.

    Returns list of problems (empty = OK to freeze).
    """
    problems: list[str] = []

    decision = revision_report.get("decision", "")
    if decision not in _FREEZABLE_DECISIONS:
        problems.append(
            f"bounded revision decision is '{decision}', "
            f"expected one of {sorted(_FREEZABLE_DECISIONS)}"
        )

    # Check that revision constraints were respected.
    constraints = revision_report.get("revision_constraints", {})
    if not constraints.get("only_trust_framing_may_change", False):
        problems.append("bounded revision constraints not satisfied")

    return problems


def run_prompt_freeze(
    revision_dir: Path,
    output_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run the prompt freeze pipeline.

    Args:
        revision_dir: Directory containing bounded_revision_report.json.
        output_dir: Directory to write freeze outputs.
        repo_root: Repository root for commit hash.

    Returns:
        Freeze report dictionary.
    """
    revision_path = revision_dir / "bounded_revision_report.json"
    if not revision_path.exists():
        raise FileNotFoundError(f"Bounded revision not found: {revision_path}")

    with open(revision_path) as f:
        revision_report = json.load(f)

    # Validate preconditions.
    problems = validate_freeze_preconditions(revision_report)
    if problems:
        raise ValueError(f"Freeze preconditions not met: {problems}")

    # Re-validate prompt invariance.
    invariance_problems = validate_trust_prompt_invariance()
    if invariance_problems:
        raise ValueError(f"Prompt invariance violated: {invariance_problems}")

    # Build the frozen manifest.
    manifest = build_prompt_manifest()
    # Override status to frozen.
    manifest["status"] = FROZEN_STATUS
    manifest["freeze_timestamp"] = datetime.now(timezone.utc).isoformat()
    manifest["repository_commit"] = _get_repository_commit(repo_root)
    manifest["bounded_revision_decision"] = revision_report.get("decision", "")
    manifest["bounded_revision_rationale"] = revision_report.get("rationale", "")

    manifest_sha = prompt_manifest_sha256(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write frozen manifest.
    manifest_path = output_dir / "frozen_prompt_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write freeze report.
    freeze_report: dict[str, Any] = {
        "freeze_type": "prompt_freeze",
        "protocol_version": PROTOCOL_VERSION,
        "freeze_timestamp": manifest["freeze_timestamp"],
        "repository_commit": manifest["repository_commit"],
        "frozen_status": FROZEN_STATUS,
        "input_file": str(revision_path),
        "manifest_sha256": manifest_sha,
        "num_templates": len(manifest.get("templates", {})),
        "invariance_valid": manifest.get("prompt_invariance", {}).get("valid", False),
        "prompts_revised": revision_report.get("prompts_revised", False),
        "passed": True,
    }

    report_path = output_dir / "frozen_freeze_report.json"
    with open(report_path, "w") as f:
        json.dump(freeze_report, f, indent=2, ensure_ascii=False)

    return freeze_report


def main() -> None:
    """CLI entry point for E2-H prompt freeze pipeline."""
    parser = argparse.ArgumentParser(
        description="E2-H: Freeze trust prompts after bounded revision",
    )
    parser.add_argument(
        "--revision-dir",
        type=Path,
        required=True,
        help="Directory containing bounded_revision_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write freeze outputs",
    )
    args = parser.parse_args()

    print("E2-H: Prompt Freeze Pipeline")
    print(f"  Revision: {args.revision_dir}")
    print(f"  Output:   {args.output_dir}")

    report = run_prompt_freeze(args.revision_dir, args.output_dir)

    print(f"\n  Frozen status: {report['frozen_status']}")
    print(f"  Manifest SHA-256: {report['manifest_sha256']}")
    print(f"  Templates: {report['num_templates']}")
    print(f"  Invariance valid: {report['invariance_valid']}")
    print(f"  Prompts revised: {report['prompts_revised']}")
    print(f"  Repository commit: {report['repository_commit']}")
    print(f"\n  Report: {args.output_dir / 'frozen_freeze_report.json'}")
    print(f"  Manifest: {args.output_dir / 'frozen_prompt_manifest.json'}")


if __name__ == "__main__":
    main()

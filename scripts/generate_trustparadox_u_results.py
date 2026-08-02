"""Generate TrustParadox-U results bundle.

Runs the TrustParadox-U pipeline:
- P0: Candidate generation and validation
- P1: Evaluation
- P2: Experiments

Outputs:
- p0_candidates.json
- p0_validation.json
- p1_evaluation.json
- p2_experiments.json
- trustparadox_u_report.json
- trustparadox_u_report.md
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TrustPhaseResult:
    """Result of a TrustParadox-U phase."""

    phase_name: str
    status: str = "PENDING"
    output_file: str = ""
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def run_phase(command: str, output_file: Path, phase_name: str) -> TrustPhaseResult:
    """Run a pipeline phase."""
    start = time.time()
    phase = TrustPhaseResult(phase_name=phase_name, output_file=str(output_file))

    try:
        with open(output_file, "w") as f:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=600,
            )
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        phase.duration_seconds = time.time() - start

        if result.returncode == 0:
            phase.status = "PASS"
        else:
            phase.status = "FAIL"
            phase.errors.append(f"Exit code: {result.returncode}")

    except subprocess.TimeoutExpired:
        phase.status = "FAIL"
        phase.errors.append("Timeout after 600s")
    except Exception as e:
        phase.status = "FAIL"
        phase.errors.append(str(e))

    return phase


def generate_report(
    phases: list[TrustPhaseResult],
    output_dir: Path,
    tested_commit: str,
) -> dict:
    """Generate the TrustParadox-U report."""
    all_passed = all(p.status == "PASS" for p in phases)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tested_commit": tested_commit,
        "overall_status": "PASS" if all_passed else "FAIL",
        "phases": {p.phase_name: asdict(p) for p in phases},
    }

    # Write JSON report
    report_path = output_dir / "trustparadox_u_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Write markdown report
    md_lines = [
        "# TrustParadox-U Results Report",
        "",
        f"**Generated:** {report['generated_at']}",
        f"**Commit:** {tested_commit}",
        f"**Status:** {report['overall_status']}",
        "",
        "## Phase Results",
        "",
        "| Phase | Status | Duration |",
        "|-------|--------|----------|",
    ]
    for p in phases:
        md_lines.append(f"| {p.phase_name} | {p.status} | {p.duration_seconds:.1f}s |")

    md_path = output_dir / "trustparadox_u_report.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    return report


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate TrustParadox-U results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "trustparadox_u",
        help="Output directory",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Tested commit SHA",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get commit
    if args.commit:
        tested_commit = args.commit
    else:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        tested_commit = result.stdout.strip()

    print("=" * 60)
    print("TrustParadox-U Results Generation")
    print("=" * 60)
    print(f"Commit: {tested_commit}")
    print(f"Output: {output_dir}")
    print()

    phases: list[TrustPhaseResult] = []

    # Phase 1: P0 Candidates
    print("[Phase 1] P0 Candidates...")
    p0_cand_file = output_dir / "p0_candidates_output.txt"
    phase = run_phase(
        "poetry run python -m experiments.trustparadox_u.p0_candidates",
        p0_cand_file,
        "p0_candidates",
    )
    phases.append(phase)
    print(f"  [{phase.status}] {phase.duration_seconds:.1f}s")

    # Phase 2: P0 Validation
    print("[Phase 2] P0 Validation...")
    p0_val_file = output_dir / "p0_validation_output.txt"
    phase = run_phase(
        "poetry run python -m experiments.trustparadox_u.p0_validation",
        p0_val_file,
        "p0_validation",
    )
    phases.append(phase)
    print(f"  [{phase.status}] {phase.duration_seconds:.1f}s")

    # Phase 3: P1 Evaluation
    print("[Phase 3] P1 Evaluation...")
    p1_file = output_dir / "p1_evaluation_output.txt"
    phase = run_phase(
        "poetry run python -m experiments.trustparadox_u.p1_evaluation",
        p1_file,
        "p1_evaluation",
    )
    phases.append(phase)
    print(f"  [{phase.status}] {phase.duration_seconds:.1f}s")

    # Phase 4: P2 Experiments
    print("[Phase 4] P2 Experiments...")
    p2_file = output_dir / "p2_experiments_output.txt"
    phase = run_phase(
        "poetry run python -m experiments.trustparadox_u.p2_experiments",
        p2_file,
        "p2_experiments",
    )
    phases.append(phase)
    print(f"  [{phase.status}] {phase.duration_seconds:.1f}s")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(phases, output_dir, tested_commit)

    print("\n" + "=" * 60)
    print(f"Status: {report['overall_status']}")
    print("=" * 60)

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

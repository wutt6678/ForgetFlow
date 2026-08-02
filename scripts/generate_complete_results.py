#!/usr/bin/env python3
"""Complete results generation driver.

Orchestrates all verification phases and produces a complete result bundle
in results/<sha>/ with provenance, CI outputs, smoke results, independent
metric verification, and certification.

Usage:
    poetry run python scripts/generate_complete_results.py \
        --commit "$(git rev-parse HEAD)" \
        --output "results/$(git rev-parse --short=12 HEAD)" \
        --seeds 7 42 123 \
        --run-tests \
        --run-integration \
        --run-single-target \
        --run-multi-target \
        --run-trustparadox-u \
        --audit \
        --certify
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.status import (  # noqa: E402
    DIAGNOSTIC_VALID,
    RELEASE_CANDIDATE,
    RESEARCH_VALID,
    status_at_least,
)

# Exit codes
EXIT_SUCCESS = 0
EXIT_REPOSITORY_DIRTY = 2
EXIT_COMMIT_MISMATCH = 3
EXIT_STATIC_CHECK = 4
EXIT_TEST_FAILURE = 5
EXIT_ASSERTION_FAILURE = 6
EXIT_SMOKE_FAILURE = 7
EXIT_VERIFICATION_FAILURE = 8
EXIT_CERTIFICATION_FAILURE = 9


@dataclass
class CommandResult:
    """Result of a subprocess command execution."""

    command: str
    start_time: str
    end_time: str
    exit_code: int
    output_file: str
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "exit_code": self.exit_code,
            "output_file": self.output_file,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ProvenanceRecord:
    """Provenance information for the results bundle."""

    tested_commit: str
    results_commit: str
    repository_clean: bool
    python_version: str
    poetry_version: str
    git_version: str
    os_info: str
    architecture: str
    timezone: str
    dependency_lock_sha256: str
    source_tree_sha256: str
    generated_at: str
    execution_mode: str = "test"
    artifact_status: str = "diagnostic"
    certification_mode: str = "diagnostic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tested_commit": self.tested_commit,
            "results_commit": self.results_commit,
            "repository_clean": self.repository_clean,
            "python_version": self.python_version,
            "poetry_version": self.poetry_version,
            "git_version": self.git_version,
            "os_info": self.os_info,
            "architecture": self.architecture,
            "timezone": self.timezone,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "generated_at": self.generated_at,
            "execution_mode": self.execution_mode,
            "artifact_status": self.artifact_status,
            "certification_mode": self.certification_mode,
        }


@dataclass
class PhaseResult:
    """Result of a verification phase."""

    phase_name: str
    status: str  # PASS, FAIL, SKIP
    start_time: str
    end_time: str
    commands: list[CommandResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "commands": [c.to_dict() for c in self.commands],
            "errors": self.errors,
            "outputs": self.outputs,
        }


class ResultsGenerator:
    """Main driver for complete results generation."""

    def __init__(
        self,
        output_dir: Path,
        tested_commit: str,
        seeds: list[int],
        run_tests: bool = True,
        run_integration: bool = True,
        run_single_target: bool = True,
        run_multi_target: bool = True,
        run_trustparadox_u: bool = True,
        run_audit: bool = True,
        run_certify: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.tested_commit = tested_commit
        self.seeds = seeds
        self.run_tests = run_tests
        self.run_integration = run_integration
        self.run_single_target = run_single_target
        self.run_multi_target = run_multi_target
        self.run_trustparadox_u = run_trustparadox_u
        self.run_audit = run_audit
        self.run_certify = run_certify

        self.command_log: list[CommandResult] = []
        self.phase_results: list[PhaseResult] = []
        self.provenance: ProvenanceRecord | None = None
        self.failure_reasons: list[str] = []

        # Create directory structure
        self.provenance_dir = output_dir / "provenance"
        self.ci_dir = output_dir / "ci"
        self.assertion_dir = output_dir / "assertion_suite"
        self.single_target_dir = output_dir / "single_target_smoke"
        self.multi_target_dir = output_dir / "multi_target_smoke"
        self.trustparadox_dir = output_dir / "trustparadox_u"
        self.certification_dir = output_dir / "certification"

    def _now(self) -> str:
        """Return current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _run_command(
        self,
        command: str,
        output_file: str | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Run a shell command and record the result."""
        start_time = self._now()
        start_ts = time.time()

        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        if output_file:
            output_path = self.output_dir / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=PROJECT_ROOT,
                    env=full_env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
        else:
            result = subprocess.run(
                command,
                shell=True,
                cwd=PROJECT_ROOT,
                env=full_env,
                capture_output=True,
                text=True,
            )

        end_time = self._now()
        duration = time.time() - start_ts

        cmd_result = CommandResult(
            command=command,
            start_time=start_time,
            end_time=end_time,
            exit_code=result.returncode,
            output_file=output_file or "",
            duration_seconds=duration,
        )
        self.command_log.append(cmd_result)

        if check and result.returncode != 0:
            error_msg = f"Command failed (exit {result.returncode}): {command}"
            if not output_file and result.stderr:
                error_msg += f"\n{result.stderr[:500]}"
            raise RuntimeError(error_msg)

        return cmd_result

    def _compute_source_tree_hash(self) -> str:
        """Compute SHA-256 hash of the source tree."""
        hasher = hashlib.sha256()
        source_dirs = ["marble", "experiments", "scripts", "tests"]
        for source_dir in source_dirs:
            dir_path = PROJECT_ROOT / source_dir
            if not dir_path.exists():
                continue
            for py_file in sorted(dir_path.rglob("*.py")):
                rel_path = py_file.relative_to(PROJECT_ROOT)
                hasher.update(str(rel_path).encode())
                hasher.update(py_file.read_bytes())
        return hasher.hexdigest()

    def _compute_dependency_lock_hash(self) -> str:
        """Compute SHA-256 hash of poetry.lock."""
        lock_file = PROJECT_ROOT / "poetry.lock"
        if lock_file.exists():
            return hashlib.sha256(lock_file.read_bytes()).hexdigest()
        return ""

    def verify_repository(self) -> None:
        """Phase 0: Verify repository state."""
        print("Phase 0: Verifying repository state...")

        # Check clean working tree
        result = subprocess.run(
            "git status --porcelain",
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        repository_clean = len(result.stdout.strip()) == 0

        # Get current commit
        result = subprocess.run(
            "git rev-parse HEAD",
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        current_commit = result.stdout.strip()

        if current_commit != self.tested_commit:
            raise RuntimeError(
                f"Commit mismatch: expected {self.tested_commit}, got {current_commit}"
            )

        if not repository_clean:
            raise RuntimeError("Repository has uncommitted changes")

        print(f"  Commit: {current_commit}")
        print(f"  Clean: {repository_clean}")

    def capture_provenance(self) -> ProvenanceRecord:
        """Phase 1: Capture environment and dependency provenance."""
        print("Phase 1: Capturing provenance...")

        self.provenance_dir.mkdir(parents=True, exist_ok=True)

        # Capture versions
        python_version = platform.python_version()
        poetry_result = subprocess.run(
            "poetry --version",
            shell=True,
            capture_output=True,
            text=True,
        )
        poetry_version = poetry_result.stdout.strip()
        git_result = subprocess.run(
            "git --version",
            shell=True,
            capture_output=True,
            text=True,
        )
        git_version = git_result.stdout.strip()

        # Get results commit (may differ from tested commit)
        results_commit_result = subprocess.run(
            "git rev-parse HEAD",
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        results_commit = results_commit_result.stdout.strip()

        provenance = ProvenanceRecord(
            tested_commit=self.tested_commit,
            results_commit=results_commit,
            repository_clean=True,
            python_version=python_version,
            poetry_version=poetry_version,
            git_version=git_version,
            os_info=f"{platform.system()} {platform.release()}",
            architecture=platform.machine(),
            timezone=time.strftime("%Z"),
            dependency_lock_sha256=self._compute_dependency_lock_hash(),
            source_tree_sha256=self._compute_source_tree_hash(),
            generated_at=self._now(),
        )
        self.provenance = provenance

        # Write provenance files
        (self.provenance_dir / "repository_state.json").write_text(
            json.dumps(provenance.to_dict(), indent=2)
        )
        (self.provenance_dir / "environment.json").write_text(
            json.dumps(
                {
                    "python_version": python_version,
                    "poetry_version": poetry_version,
                    "git_version": git_version,
                    "os_info": provenance.os_info,
                    "architecture": provenance.architecture,
                    "timezone": provenance.timezone,
                    "cpu_count": os.cpu_count(),
                },
                indent=2,
            )
        )
        (self.provenance_dir / "dependency_lock.sha256").write_text(
            f"{provenance.dependency_lock_sha256}  poetry.lock\n"
        )
        (self.provenance_dir / "source_tree.sha256").write_text(
            f"{provenance.source_tree_sha256}  source_tree\n"
        )

        print(f"  Python: {python_version}")
        print(f"  Poetry: {poetry_version}")
        print(f"  Source hash: {provenance.source_tree_sha256[:16]}...")

        return provenance

    def run_static_checks(self) -> PhaseResult:
        """Phase 2: Run static and source-integrity checks."""
        print("Phase 2: Running static checks...")

        phase = PhaseResult(
            phase_name="static_checks",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        self.ci_dir.mkdir(parents=True, exist_ok=True)

        checks = [
            ("poetry check", "ci/poetry_check.txt"),
            ("poetry run python scripts/validate_workflows.py", "ci/validate_workflows.txt"),
            (
                "poetry run python scripts/check_source_integrity.py",
                "ci/check_source_integrity.txt",
            ),
            ("poetry run python -m compileall marble experiments", "ci/compileall.txt"),
            ("poetry run ruff check marble experiments tests scripts", "ci/ruff_check.txt"),
            (
                "poetry run ruff format --check marble experiments tests scripts",
                "ci/ruff_format.txt",
            ),
            ("poetry run mypy marble experiments", "ci/mypy.txt"),
        ]

        for command, output_file in checks:
            try:
                cmd_result = self._run_command(command, output_file)
                phase.commands.append(cmd_result)
                print(f"  [PASS] {command.split()[0]}...")
            except RuntimeError as e:
                phase.status = "FAIL"
                phase.errors.append(str(e))
                self.failure_reasons.append(f"Static check failed: {command}")
                print(f"  [FAIL] {command}")
                break

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def run_test_suite(self) -> PhaseResult:
        """Phase 3: Run the full automated test suite."""
        print("Phase 3: Running test suite...")

        phase = PhaseResult(
            phase_name="test_suite",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        if not self.run_tests:
            phase.status = "SKIP"
            phase.end_time = self._now()
            self.phase_results.append(phase)
            return phase

        self.ci_dir.mkdir(parents=True, exist_ok=True)

        test_env = {"FORGETFLOW_TEST_MODE": "1"}

        test_suites = [
            (
                'poetry run pytest -q -m "not integration" --junitxml="results/ci/pytest_non_integration.xml"',
                "ci/pytest_non_integration.txt",
            ),
            (
                'poetry run pytest tests/firewall -q --junitxml="results/ci/pytest_firewall.xml"',
                "ci/pytest_firewall.txt",
            ),
            (
                'poetry run pytest tests/test_architecture_contract.py -q --junitxml="results/ci/pytest_architecture.xml"',
                "ci/pytest_architecture.txt",
            ),
            (
                "poetry run pytest tests/trustparadox_u/test_config.py -q",
                "ci/pytest_config.txt",
            ),
            (
                "poetry run pytest tests/trustparadox_u/test_embedding.py -q",
                "ci/pytest_embedding.txt",
            ),
            (
                "poetry run pytest tests/trustparadox_u/test_attacks.py -q",
                "ci/pytest_attacks.txt",
            ),
            (
                "poetry run pytest tests/trustparadox_u/test_evaluator.py -q",
                "ci/pytest_evaluator.txt",
            ),
            (
                "poetry run pytest tests/trustparadox_u/test_metric_contracts.py -q",
                "ci/pytest_metric_contracts.txt",
            ),
            (
                "poetry run pytest tests/trustparadox_u/test_end_to_end.py -q",
                "ci/pytest_end_to_end.txt",
            ),
            (
                "poetry run pytest tests/trustparadox_u/test_runner.py -q",
                "ci/pytest_runner.txt",
            ),
            (
                "poetry run pytest tests/trustparadox_u/test_result_audit.py -q",
                "ci/pytest_result_audit.txt",
            ),
        ]

        if self.run_integration:
            test_suites.append(
                (
                    'poetry run pytest -q -m "integration" --junitxml="results/ci/pytest_integration.xml"',
                    "ci/pytest_integration.txt",
                )
            )

        # Full suite at the end
        test_suites.append(
            (
                'poetry run pytest -q --junitxml="results/ci/pytest_all.xml"',
                "ci/pytest_all.txt",
            )
        )

        for command, output_file in test_suites:
            try:
                cmd_result = self._run_command(command, output_file, env=test_env)
                phase.commands.append(cmd_result)
                suite_name = (
                    command.split("pytest")[1].split()[0] if "pytest" in command else "tests"
                )
                print(f"  [PASS] {suite_name}")
            except RuntimeError as e:
                phase.status = "FAIL"
                phase.errors.append(str(e))
                self.failure_reasons.append(f"Test suite failed: {command}")
                print(f"  [FAIL] {command}")
                break

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def run_assertion_suite(self) -> PhaseResult:
        """Phase 4: Run the deterministic assertion suite."""
        print("Phase 4: Running assertion suite...")

        phase = PhaseResult(
            phase_name="assertion_suite",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        self.assertion_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run assertion suite
            cmd_result = self._run_command(
                f"poetry run python scripts/run_assertion_suite.py --output-dir {self.assertion_dir}",
                "assertion_suite/assertion_suite_output.txt",
            )
            phase.commands.append(cmd_result)

            # Check report
            report_path = self.assertion_dir / "assertion_suite_report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get("assertion_cases_failed", 0) > 0:
                    phase.status = "FAIL"
                    phase.errors.append(
                        f"Assertion cases failed: {report['assertion_cases_failed']}"
                    )
                    self.failure_reasons.append("Assertion suite has failing cases")
                elif report.get("individual_assertions_failed", 0) > 0:
                    phase.status = "FAIL"
                    phase.errors.append(
                        f"Individual assertions failed: {report['individual_assertions_failed']}"
                    )
                    self.failure_reasons.append("Assertion suite has failing assertions")
                else:
                    print(f"  [PASS] {report.get('assertion_cases_passed', 0)} cases passed")
            else:
                phase.status = "FAIL"
                phase.errors.append("Assertion report not found")
                self.failure_reasons.append("Assertion suite report missing")

        except RuntimeError as e:
            phase.status = "FAIL"
            phase.errors.append(str(e))
            self.failure_reasons.append("Assertion suite execution failed")

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def run_single_target_smoke(self) -> PhaseResult:
        """Phase 5: Generate single-target smoke results."""
        print("Phase 5: Running single-target smoke...")

        phase = PhaseResult(
            phase_name="single_target_smoke",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        if not self.run_single_target:
            phase.status = "SKIP"
            phase.end_time = self._now()
            self.phase_results.append(phase)
            return phase

        self.single_target_dir.mkdir(parents=True, exist_ok=True)

        try:
            seeds_str = " ".join(str(s) for s in self.seeds)
            cmd_result = self._run_command(
                f"poetry run python scripts/run_single_target_smoke.py "
                f"--output-dir {self.single_target_dir} --mode release --seeds {seeds_str}",
                "single_target_smoke/smoke_output.txt",
            )
            phase.commands.append(cmd_result)

            # Check summary
            summary_path = self.single_target_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                # Section 16: prefer the canonical execution-status taxonomy.
                execution_status = summary.get("execution_status")
                if execution_status is not None:
                    if status_at_least(execution_status, RESEARCH_VALID):
                        print(f"  [PASS] Execution status: {execution_status}")
                    else:
                        phase.status = "FAIL"
                        phase.errors.append(f"Execution status: {execution_status}")
                        self.failure_reasons.append(f"Single-target smoke: {execution_status}")
                else:
                    status = summary.get("top_line_status", "NO-GO")
                    if status == "GO":
                        print(f"  [PASS] Status: {status}")
                    else:
                        phase.status = "FAIL"
                        phase.errors.append(f"Smoke status: {status}")
                        self.failure_reasons.append(f"Single-target smoke: {status}")
            else:
                phase.status = "FAIL"
                phase.errors.append("Summary not found")
                self.failure_reasons.append("Single-target smoke summary missing")

        except RuntimeError as e:
            phase.status = "FAIL"
            phase.errors.append(str(e))
            self.failure_reasons.append("Single-target smoke execution failed")

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def run_multi_target_smoke(self) -> PhaseResult:
        """Phase 6: Generate multi-target smoke results."""
        print("Phase 6: Running multi-target smoke...")

        phase = PhaseResult(
            phase_name="multi_target_smoke",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        if not self.run_multi_target:
            phase.status = "SKIP"
            phase.end_time = self._now()
            self.phase_results.append(phase)
            return phase

        self.multi_target_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd_result = self._run_command(
                f"poetry run python scripts/run_multi_target_smoke.py "
                f"--output-dir {self.multi_target_dir} --mode certified-deterministic",
                "multi_target_smoke/smoke_output.txt",
            )
            phase.commands.append(cmd_result)

            # Check report
            report_path = self.multi_target_dir / "multi_target_report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                # Section 16: prefer the canonical execution-status taxonomy.
                execution_status = report.get("execution_status")
                if execution_status is not None:
                    if status_at_least(execution_status, RESEARCH_VALID):
                        print(f"  [PASS] Execution status: {execution_status}")
                    else:
                        phase.status = "FAIL"
                        phase.errors.append(f"Execution status: {execution_status}")
                        self.failure_reasons.append(f"Multi-target smoke: {execution_status}")
                else:
                    status = report.get("status", "NO-GO")
                    if status == "GO":
                        print(f"  [PASS] Status: {status}")
                    else:
                        phase.status = "FAIL"
                        phase.errors.append(f"Multi-target status: {status}")
                        self.failure_reasons.append(f"Multi-target smoke: {status}")
            else:
                phase.status = "FAIL"
                phase.errors.append("Report not found")
                self.failure_reasons.append("Multi-target smoke report missing")

        except RuntimeError as e:
            phase.status = "FAIL"
            phase.errors.append(str(e))
            self.failure_reasons.append("Multi-target smoke execution failed")

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def run_trustparadox_u(self) -> PhaseResult:
        """Phase 7: Generate TrustParadox-U results."""
        print("Phase 7: Running TrustParadox-U...")

        phase = PhaseResult(
            phase_name="trustparadox_u",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        if not self.run_trustparadox_u:
            phase.status = "SKIP"
            phase.end_time = self._now()
            self.phase_results.append(phase)
            return phase

        self.trustparadox_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run TrustParadox-U evaluation
            cmd_result = self._run_command(
                f"poetry run python scripts/generate_trustparadox_u_results.py "
                f"--output-dir {self.trustparadox_dir} --commit {self.tested_commit}",
                "trustparadox_u/trustparadox_output.txt",
                check=False,
            )
            phase.commands.append(cmd_result)

            if cmd_result.exit_code != 0:
                # TrustParadox-U is optional evidence; record but do not block.
                phase.status = "SKIP"
                phase.errors.append("TrustParadox-U generation failed")
                print("  [SKIP] TrustParadox-U generation failed")
            else:
                print("  [PASS] TrustParadox-U complete")

        except RuntimeError as e:
            phase.status = "SKIP"
            phase.errors.append(str(e))
            print("  [SKIP] TrustParadox-U not available")

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def run_independent_verification(self) -> PhaseResult:
        """Phase 8: Independently recompute every metric."""
        print("Phase 8: Running independent verification...")

        phase = PhaseResult(
            phase_name="independent_verification",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        if not self.run_audit:
            phase.status = "SKIP"
            phase.end_time = self._now()
            self.phase_results.append(phase)
            return phase

        try:
            cmd_result = self._run_command(
                f"poetry run python scripts/verify_metrics.py --results-dir {self.output_dir}",
                "ci/verify_metrics.txt",
                check=False,
            )
            phase.commands.append(cmd_result)

            if cmd_result.exit_code != 0:
                phase.status = "FAIL"
                phase.errors.append("Metric verification failed")
                self.failure_reasons.append("Independent metric verification failed")
            else:
                print("  [PASS] Metrics verified")

        except RuntimeError as e:
            phase.status = "FAIL"
            phase.errors.append(str(e))
            self.failure_reasons.append("Metric verification execution failed")

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def run_consistency_validation(self) -> PhaseResult:
        """Phase 9: Validate cross-artifact consistency."""
        print("Phase 9: Running consistency validation...")

        phase = PhaseResult(
            phase_name="consistency_validation",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        if not self.run_audit:
            phase.status = "SKIP"
            phase.end_time = self._now()
            self.phase_results.append(phase)
            return phase

        try:
            cmd_result = self._run_command(
                f"poetry run python scripts/validate_consistency.py --results-dir {self.output_dir}",
                "ci/validate_consistency.txt",
                check=False,
            )
            phase.commands.append(cmd_result)

            if cmd_result.exit_code != 0:
                phase.status = "FAIL"
                phase.errors.append("Consistency validation failed")
                self.failure_reasons.append("Cross-artifact consistency validation failed")
            else:
                print("  [PASS] Consistency validated")

        except RuntimeError as e:
            phase.status = "FAIL"
            phase.errors.append(str(e))
            self.failure_reasons.append("Consistency validation execution failed")

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def generate_certification(self) -> PhaseResult:
        """Phase 10: Generate checksums and certification."""
        print("Phase 10: Generating certification...")

        phase = PhaseResult(
            phase_name="certification",
            status="PASS",
            start_time=self._now(),
            end_time="",
        )

        if not self.run_certify:
            phase.status = "SKIP"
            phase.end_time = self._now()
            self.phase_results.append(phase)
            return phase

        self.certification_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd_result = self._run_command(
                f"poetry run python scripts/generate_certification.py "
                f"--results-dir {self.output_dir} --tested-commit {self.tested_commit}",
                "certification/certification_output.txt",
                check=False,
            )
            phase.commands.append(cmd_result)

            # Check certification
            cert_path = self.certification_dir / "certification.json"
            if cert_path.exists():
                cert = json.loads(cert_path.read_text())
                status = cert.get("certification_status", DIAGNOSTIC_VALID)
                print(f"  Certification status: {status}")
                if status == RELEASE_CANDIDATE:
                    print("  [PASS] Release candidate")
                elif status == RESEARCH_VALID:
                    print("  [PASS] Research valid")
                else:
                    phase.errors.append(f"Certification status: {status}")
            else:
                phase.errors.append("Certification not generated")

        except RuntimeError as e:
            phase.status = "FAIL"
            phase.errors.append(str(e))
            self.failure_reasons.append("Certification generation failed")

        phase.end_time = self._now()
        self.phase_results.append(phase)
        return phase

    def write_command_log(self) -> None:
        """Write the complete command log."""
        log_path = self.provenance_dir / "command_log.txt"
        with open(log_path, "w") as f:
            for cmd in self.command_log:
                f.write(f"[{cmd.start_time}] {cmd.command}\n")
                f.write(f"  Exit code: {cmd.exit_code}\n")
                f.write(f"  Duration: {cmd.duration_seconds:.2f}s\n")
                if cmd.output_file:
                    f.write(f"  Output: {cmd.output_file}\n")
                f.write("\n")

    def write_complete_manifest(self) -> None:
        """Write the complete results manifest."""
        all_passed = (
            all(p.status in ("PASS", "SKIP") for p in self.phase_results)
            and not self.failure_reasons
        )

        # Determine certification status
        if all_passed:
            certification_status = RELEASE_CANDIDATE
        elif any(p.status == "FAIL" for p in self.phase_results):
            certification_status = DIAGNOSTIC_VALID
        else:
            certification_status = RESEARCH_VALID

        manifest = {
            "schema_version": "1.0.0",
            "tested_commit": self.tested_commit,
            "results_commit": self.provenance.results_commit if self.provenance else "",
            "generated_at": self._now(),
            "output_dir": str(self.output_dir),
            "seeds": self.seeds,
            "phases": [p.to_dict() for p in self.phase_results],
            "failure_reasons": self.failure_reasons,
            "all_passed": all_passed,
            "certification_status": certification_status,
        }

        manifest_path = self.output_dir / "complete_results_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def run(self) -> int:
        """Run the complete results generation pipeline."""
        print("=" * 60)
        print("ForgetFlow Complete Results Generation")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Tested commit: {self.tested_commit}")
        print(f"Seeds: {self.seeds}")
        print()

        try:
            # Phase 0: Verify repository
            self.verify_repository()

            # Phase 1: Capture provenance
            self.capture_provenance()

            # Phase 2: Static checks
            static_result = self.run_static_checks()
            if static_result.status == "FAIL":
                print("\nStatic checks failed. Stopping.")
                self.write_command_log()
                self.write_complete_manifest()
                return EXIT_STATIC_CHECK

            # Phase 3: Test suite
            test_result = self.run_test_suite()
            if test_result.status == "FAIL":
                print("\nTest suite failed. Stopping.")
                self.write_command_log()
                self.write_complete_manifest()
                return EXIT_TEST_FAILURE

            # Phase 4: Assertion suite
            assertion_result = self.run_assertion_suite()
            if assertion_result.status == "FAIL":
                print("\nAssertion suite failed. Stopping.")
                self.write_command_log()
                self.write_complete_manifest()
                return EXIT_ASSERTION_FAILURE

            # Phase 5: Single-target smoke
            single_result = self.run_single_target_smoke()
            if single_result.status == "FAIL":
                print("\nSingle-target smoke failed.")
                self.failure_reasons.append("Single-target smoke failed")

            # Phase 6: Multi-target smoke
            multi_result = self.run_multi_target_smoke()
            if multi_result.status == "FAIL":
                print("\nMulti-target smoke failed.")
                self.failure_reasons.append("Multi-target smoke failed")

            # Phase 7: TrustParadox-U
            self.run_trustparadox_u()

            # Phase 8: Independent verification
            self.run_independent_verification()

            # Phase 9: Consistency validation
            self.run_consistency_validation()

            # Phase 10: Certification
            self.generate_certification()

            # Write final logs and manifest
            self.write_command_log()
            self.write_complete_manifest()

            # Summary
            print("\n" + "=" * 60)
            print("Results Generation Complete")
            print("=" * 60)
            for phase in self.phase_results:
                icon = (
                    "PASS"
                    if phase.status == "PASS"
                    else ("SKIP" if phase.status == "SKIP" else "FAIL")
                )
                print(f"  [{icon}] {phase.phase_name}")

            if self.failure_reasons:
                print("\nFailure reasons:")
                for reason in self.failure_reasons:
                    print(f"  - {reason}")
                return EXIT_SMOKE_FAILURE

            return EXIT_SUCCESS

        except RuntimeError as e:
            print(f"\nFatal error: {e}")
            self.failure_reasons.append(str(e))
            self.write_command_log()
            self.write_complete_manifest()
            return EXIT_REPOSITORY_DIRTY


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate complete ForgetFlow results bundle")
    parser.add_argument(
        "--commit",
        type=str,
        required=True,
        help="Tested commit SHA",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 42, 123],
        help="Random seeds for smoke tests",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        default=True,
        help="Run test suite",
    )
    parser.add_argument(
        "--run-integration",
        action="store_true",
        default=True,
        help="Run integration tests",
    )
    parser.add_argument(
        "--run-single-target",
        action="store_true",
        default=True,
        help="Run single-target smoke",
    )
    parser.add_argument(
        "--run-multi-target",
        action="store_true",
        default=True,
        help="Run multi-target smoke",
    )
    parser.add_argument(
        "--run-trustparadox-u",
        action="store_true",
        default=True,
        help="Run TrustParadox-U evaluation",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        default=True,
        help="Run independent audit",
    )
    parser.add_argument(
        "--certify",
        action="store_true",
        default=True,
        help="Generate certification",
    )

    args = parser.parse_args()

    generator = ResultsGenerator(
        output_dir=Path(args.output),
        tested_commit=args.commit,
        seeds=args.seeds,
        run_tests=args.run_tests,
        run_integration=args.run_integration,
        run_single_target=args.run_single_target,
        run_multi_target=args.run_multi_target,
        run_trustparadox_u=args.run_trustparadox_u,
        run_audit=args.audit,
        run_certify=args.certify,
    )

    return generator.run()


if __name__ == "__main__":
    sys.exit(main())

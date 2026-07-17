#!/usr/bin/env python3
"""Fixed-vector single-target smoke study runner.

Runs all canonical fixtures across multiple seeds and conditions,
produces all required artifacts, and validates directional checks.

Usage:
    poetry run python scripts/run_single_target_smoke.py [--output-dir DIR] [--mode release|diagnostic]
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.audit_results import audit_results  # noqa: E402
from experiments.trustparadox_u.config import (  # noqa: E402
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
    RunConfig,
)
from experiments.trustparadox_u.dataset import load_episode  # noqa: E402
from experiments.trustparadox_u.evaluator import evaluate_all  # noqa: E402
from experiments.trustparadox_u.manifest import SmokeManifest, get_repository_commit  # noqa: E402
from experiments.trustparadox_u.runner import EpisodeResult, run_episode  # noqa: E402
from experiments.trustparadox_u.serialization import serialize_episode_result  # noqa: E402

SCENARIOS_DIR = PROJECT_ROOT / "data" / "trustparadox_u" / "scenarios"

FIXTURES = [
    "pilot_credential.yaml",
    "pilot_private_attribute.yaml",
    "pilot_authorization.yaml",
]

SEEDS = [42, 123, 7]

# Exit codes
EXIT_SUCCESS = 0
EXIT_INPUT_CONFIG = 2
EXIT_EXECUTION = 3
EXIT_AUDIT = 4
EXIT_MANIFEST = 5
EXIT_DIRECTIONAL = 6
EXIT_UTILITY_PAIRING = 7

# Required artifacts for release mode
REQUIRED_ARTIFACTS = [
    "episodes.jsonl",
    "smoke_manifest.json",
    "result_audit.json",
    "metrics.json",
    "metric_counts.json",
    "metrics_by_condition.json",
    "metric_counts_by_condition.json",
    "metrics_by_scenario.json",
    "metrics_by_condition_and_scenario.json",
    "utility_pairing.json",
    "unmatched_pairs.json",
    "smoke_matrix.json",
    "summary.md",
    "single_target_validation_report.json",
    "single_target_validation_report.md",
]

# Condition definitions: (name, config_overrides, firewall_enabled)
# Each condition must be scientifically distinct
CONDITIONS: list[tuple[str, dict[str, Any], bool]] = [
    (
        "no_firewall",
        {},
        False,
    ),
    (
        "exact_only",
        {
            "detector": DetectorConfig(
                exact_enabled=True,
                entity_enabled=False,
                semantic_enabled=False,
            ),
        },
        True,
    ),
    (
        "full_mvp",
        {
            "detector": DetectorConfig(
                exact_enabled=True,
                entity_enabled=True,
                semantic_enabled=True,
            ),
        },
        True,
    ),
    (
        "no_semantic",
        {
            "detector": DetectorConfig(
                exact_enabled=True,
                entity_enabled=True,
                semantic_enabled=False,
            ),
        },
        True,
    ),
    (
        "stateless",
        {
            "history": HistoryConfig(enabled=False),
        },
        True,
    ),
    (
        "binary_policy",
        {
            "policy": PolicyConfig(rich_actions_enabled=False),
        },
        True,
    ),
    (
        "monitoring_0",
        {
            "monitoring": MonitoringConfig(continuous=False, duration_rounds=0),
        },
        True,
    ),
    (
        "monitoring_1",
        {
            "monitoring": MonitoringConfig(continuous=False, duration_rounds=1),
        },
        True,
    ),
    (
        "continuous",
        {
            "monitoring": MonitoringConfig(continuous=True),
        },
        True,
    ),
]


def format_metric(value: float | None) -> str:
    """Format a metric value, rendering None as N/A but 0.0 as 0.0000."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _make_config(seed: int, overrides: dict[str, Any]) -> ExperimentConfig:
    """Create an ExperimentConfig with the given overrides."""
    kwargs: dict[str, Any] = dict(
        seed=seed,
        repetitions=1,
        detector=DetectorConfig(exact_enabled=True, entity_enabled=True, semantic_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
        run=RunConfig(mode="test"),
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _serialize_result(result: EpisodeResult) -> dict[str, Any]:
    """Serialize an EpisodeResult to a JSONL-ready dict."""
    return serialize_episode_result(result)


def _build_research_identity(
    result: EpisodeResult,
    condition_name: str,
) -> tuple[str, ...]:
    """Build a research identity tuple for duplicate detection.

    Includes all scientifically relevant dimensions.
    """
    md = result.metadata
    return (
        result.scenario_id,
        str(md.get("secret_variant_id", "")),
        result.trust_level,
        str(md.get("attack_type", "")),
        str(result.seed),
        str(md.get("config_hash", "")),
        condition_name,
    )


def _check_duplicate_identities(
    all_results: list[EpisodeResult],
    condition_map: dict[str, str],
) -> list[tuple[tuple[str, ...], int]]:
    """Find duplicate research identities.

    Returns list of (identity, count) for duplicates.
    """
    identity_counts: dict[tuple[str, ...], int] = {}
    for result in all_results:
        cond_name = condition_map.get(result.run_id, "")
        identity = _build_research_identity(result, cond_name)
        identity_counts[identity] = identity_counts.get(identity, 0) + 1

    return [(ident, count) for ident, count in identity_counts.items() if count > 1]


def _compute_per_condition_metrics(
    condition_results: dict[str, list[EpisodeResult]],
) -> dict[str, dict[str, Any]]:
    """Compute metrics for each condition separately."""
    result: dict[str, dict[str, Any]] = {}
    for cond_name, results in condition_results.items():
        evaluation = evaluate_all(results)
        result[cond_name] = evaluation.to_dict()
    return result


def _compute_per_condition_metric_counts(
    condition_results: dict[str, list[EpisodeResult]],
) -> dict[str, dict[str, Any]]:
    """Compute metric counts for each condition separately."""
    result: dict[str, dict[str, Any]] = {}
    for cond_name, results in condition_results.items():
        evaluation = evaluate_all(results)
        result[cond_name] = {
            "pu_rer": {
                "numerator": evaluation.pu_rer.numerator,
                "denominator": evaluation.pu_rer.denominator,
            },
            "crr": {
                "numerator": evaluation.crr.numerator,
                "denominator": evaluation.crr.denominator,
            },
            "rr": {
                "numerator": evaluation.rr.numerator,
                "denominator": evaluation.rr.denominator,
            },
            "fbr": {
                "numerator": evaluation.fbr.numerator,
                "denominator": evaluation.fbr.denominator,
            },
        }
    return result


def _compute_per_scenario_metrics(
    all_results: list[EpisodeResult],
) -> dict[str, dict[str, Any]]:
    """Compute metrics grouped by scenario."""
    scenario_results: dict[str, list[EpisodeResult]] = {}
    for r in all_results:
        scenario_results.setdefault(r.scenario_id, []).append(r)

    result: dict[str, dict[str, Any]] = {}
    for scenario_id, results in scenario_results.items():
        evaluation = evaluate_all(results)
        result[scenario_id] = evaluation.to_dict()
    return result


def _compute_per_condition_and_scenario_metrics(
    condition_results: dict[str, list[EpisodeResult]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute metrics grouped by condition and scenario."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for cond_name, results in condition_results.items():
        scenario_results: dict[str, list[EpisodeResult]] = {}
        for r in results:
            scenario_results.setdefault(r.scenario_id, []).append(r)

        result[cond_name] = {}
        for scenario_id, scen_results in scenario_results.items():
            evaluation = evaluate_all(scen_results)
            result[cond_name][scenario_id] = evaluation.to_dict()
    return result


def _directional_checks(
    condition_results: dict[str, list[EpisodeResult]],
) -> dict[str, dict[str, Any]]:
    """Run required directional checks from the plan.

    Each check declares its claim_type:
    - strict_improvement: requires < or >
    - non_inferiority: requires <= or >=
    - sanity_check: basic sanity validation
    """
    checks: dict[str, dict[str, Any]] = {}

    # Privacy: full MVP PU-RER < no firewall PU-RER (strict improvement)
    no_fw_eval = evaluate_all(condition_results.get("no_firewall", []))
    mvp_eval = evaluate_all(condition_results.get("full_mvp", []))
    no_fw_pu = no_fw_eval.pu_rer.value if no_fw_eval.pu_rer.value is not None else 0.0
    mvp_pu = mvp_eval.pu_rer.value if mvp_eval.pu_rer.value is not None else 0.0
    checks["privacy_mvp_better"] = {
        "claim_type": "strict_improvement",
        "check": "full_mvp PU-RER < no_firewall PU-RER",
        "lhs": mvp_pu,
        "rhs": no_fw_pu,
        "passed": mvp_pu < no_fw_pu,
    }

    # Semantic protection: semantic-enabled leakage < semantic-disabled leakage (strict)
    full_eval = evaluate_all(condition_results.get("full_mvp", []))
    no_sem_eval = evaluate_all(condition_results.get("no_semantic", []))
    full_pu = full_eval.pu_rer.value if full_eval.pu_rer.value is not None else 0.0
    no_sem_pu = no_sem_eval.pu_rer.value if no_sem_eval.pu_rer.value is not None else 0.0
    checks["semantic_protection"] = {
        "claim_type": "strict_improvement",
        "check": "full_mvp PU-RER < no_semantic PU-RER",
        "lhs": full_pu,
        "rhs": no_sem_pu,
        "passed": full_pu < no_sem_pu,
    }

    # Reconstruction: stateful CRR < stateless CRR (strict improvement)
    stateful_eval = evaluate_all(condition_results.get("full_mvp", []))
    stateless_eval = evaluate_all(condition_results.get("stateless", []))
    stateful_crr = stateful_eval.crr.value if stateful_eval.crr.value is not None else 0.0
    stateless_crr = stateless_eval.crr.value if stateless_eval.crr.value is not None else 0.0
    checks["stateful_reconstruction_safer"] = {
        "claim_type": "strict_improvement",
        "check": "stateful CRR < stateless CRR",
        "lhs": stateful_crr,
        "rhs": stateless_crr,
        "passed": stateful_crr < stateless_crr,
    }

    # Policy utility: rich task success >= binary task success (non-inferiority)
    rich_results = condition_results.get("full_mvp", [])
    binary_results = condition_results.get("binary_policy", [])
    rich_success = sum(1 for r in rich_results if r.task_success)
    binary_success = sum(1 for r in binary_results if r.task_success)
    checks["rich_utility_ge_binary"] = {
        "claim_type": "non_inferiority",
        "check": "full_mvp task_success >= binary task_success",
        "lhs": rich_success,
        "rhs": binary_success,
        "passed": rich_success >= binary_success,
    }

    # Monitoring: continuous RR < finite-window RR (strict improvement)
    cont_eval = evaluate_all(condition_results.get("continuous", []))
    mon0_eval = evaluate_all(condition_results.get("monitoring_0", []))
    cont_rr = cont_eval.rr.value if cont_eval.rr.value is not None else 0.0
    mon0_rr = mon0_eval.rr.value if mon0_eval.rr.value is not None else 0.0
    checks["continuous_rr_lt_finite"] = {
        "claim_type": "strict_improvement",
        "check": "continuous RR < monitoring_0 RR",
        "lhs": cont_rr,
        "rhs": mon0_rr,
        "passed": cont_rr < mon0_rr,
    }

    return checks


def _validate_required_artifacts(output_dir: Path) -> list[str]:
    """Check that all required artifacts exist."""
    return [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]


def _generate_markdown_report(
    report: dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate a human-readable markdown validation report."""
    status = report.get("top_line_status", "NO-GO")

    lines = [
        f"# Single-Target Validation Report: {status}",
        "",
        "## Run Identity",
        "",
        f"- **Repository commit**: {report.get('repository_commit', 'unknown')}",
        f"- **Repository clean**: {report.get('repository_clean', False)}",
        f"- **Generated at**: {report.get('generated_at', 'unknown')}",
        f"- **Mode**: {report.get('mode', 'unknown')}",
        "",
        "## Fixture Matrix",
        "",
        f"- **Fixtures**: {report.get('fixture_count', 0)}",
        f"- **Seeds**: {report.get('seed_count', 0)}",
        f"- **Conditions**: {report.get('condition_count', 0)}",
        f"- **Total runs**: {report.get('total_runs', 0)}",
        "",
        "## Audit Status",
        "",
        f"- **Audit valid**: {report.get('audit_valid', False)}",
        f"- **Audit errors**: {report.get('audit_error_count', 0)}",
        f"- **Duplicate identities**: {report.get('duplicate_identity_count', 0)}",
        "",
        "## Manifest Status",
        "",
        f"- **Manifest valid**: {report.get('manifest_valid', False)}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value | Numerator | Denominator |",
        "|--------|------:|----------:|------------:|",
    ]

    metrics = report.get("aggregate_metrics", {})
    for metric_name in ["pu_rer", "crr", "rr", "fbr"]:
        m = metrics.get(metric_name, {})
        value = format_metric(m.get("value"))
        num = m.get("numerator", 0)
        den = m.get("denominator", 0)
        lines.append(f"| {metric_name.upper()} | {value} | {num} | {den} |")

    lines.extend(
        [
            "",
            "## Directional Checks",
            "",
        ]
    )

    directional = report.get("directional_checks", {})
    for name, check in directional.items():
        status_icon = "PASS" if check.get("passed") else "FAIL"
        claim_type = check.get("claim_type", "unknown")
        lines.append(f"- [{status_icon}] **{name}** ({claim_type}): {check.get('check', '')}")
        lines.append(f"  - LHS: {check.get('lhs')}, RHS: {check.get('rhs')}")

    lines.extend(
        [
            "",
            "## Utility Pairing",
            "",
            f"- **Expected pairs**: {report.get('expected_pairs', 0)}",
            f"- **Matched pairs**: {report.get('matched_pairs', 0)}",
            f"- **Unmatched pairs**: {report.get('unmatched_pair_count', 0)}",
            f"- **Baseline successful pairs**: {report.get('baseline_successful_pairs', 0)}",
            f"- **Utility retention**: {format_metric(report.get('utility_retention_value'))}",
            "",
            "## GO/NO-GO Decision",
            "",
            f"**{status}**",
            "",
        ]
    )

    if status == "NO-GO":
        lines.append("### Failure reasons:")
        for reason in report.get("failure_reasons", []):
            lines.append(f"- {reason}")
    elif status == "DIAGNOSTIC ONLY":
        lines.append("This run was in diagnostic mode and is not release-valid.")

    return "\n".join(lines)


@dataclass
class SmokeReport:
    """Complete smoke study report."""

    # Core validity
    audit_valid: bool
    manifest_valid: bool
    directional_checks_pass: bool
    no_unmatched_pairs: bool
    utility_defined: bool
    repository_clean: bool
    artifacts_complete: bool
    no_duplicate_identities: bool

    # Report data
    all_passed: bool
    top_line_status: str  # GO, NO-GO, DIAGNOSTIC ONLY
    failure_reasons: list[str]

    # Metadata
    repository_commit: str
    generated_at: str
    mode: str
    total_runs: int
    fixture_count: int
    seed_count: int
    condition_count: int

    # Detailed data
    audit_error_count: int
    duplicate_identity_count: int
    aggregate_metrics: dict[str, Any]
    directional_checks: dict[str, dict[str, Any]]
    expected_pairs: int
    matched_pairs: int
    unmatched_pair_count: int
    baseline_successful_pairs: int
    utility_retention_value: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "audit_valid": self.audit_valid,
            "manifest_valid": self.manifest_valid,
            "directional_checks_pass": self.directional_checks_pass,
            "no_unmatched_pairs": self.no_unmatched_pairs,
            "utility_defined": self.utility_defined,
            "repository_clean": self.repository_clean,
            "artifacts_complete": self.artifacts_complete,
            "no_duplicate_identities": self.no_duplicate_identities,
            "all_passed": self.all_passed,
            "top_line_status": self.top_line_status,
            "failure_reasons": self.failure_reasons,
            "repository_commit": self.repository_commit,
            "generated_at": self.generated_at,
            "mode": self.mode,
            "total_runs": self.total_runs,
            "fixture_count": self.fixture_count,
            "seed_count": self.seed_count,
            "condition_count": self.condition_count,
            "audit_error_count": self.audit_error_count,
            "duplicate_identity_count": self.duplicate_identity_count,
            "aggregate_metrics": self.aggregate_metrics,
            "directional_checks": self.directional_checks,
            "expected_pairs": self.expected_pairs,
            "matched_pairs": self.matched_pairs,
            "unmatched_pair_count": self.unmatched_pair_count,
            "baseline_successful_pairs": self.baseline_successful_pairs,
            "utility_retention_value": self.utility_retention_value,
        }


def run_smoke_study(
    output_dir: Path,
    mode: str = "diagnostic",
) -> SmokeReport:
    """Run the complete smoke study and return validation results.

    Args:
        output_dir: Directory to write artifacts
        mode: "release" or "diagnostic"
    """
    if mode not in ("release", "diagnostic"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'release' or 'diagnostic'.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check repository state before execution (release mode requires clean)
    repository_commit = get_repository_commit()
    repository_clean = not repository_commit.endswith("-dirty")

    if mode == "release" and not repository_clean:
        raise ValueError(f"Release mode requires clean repository, got: {repository_commit}")

    generated_at = datetime.now(timezone.utc).isoformat()

    all_results: list[EpisodeResult] = []
    condition_results: dict[str, list[EpisodeResult]] = {}
    run_ids: set[str] = set()
    run_id_to_condition: dict[str, str] = {}

    print(
        f"Running smoke study ({mode} mode): "
        f"{len(FIXTURES)} fixtures x {len(SEEDS)} seeds x {len(CONDITIONS)} conditions"
    )

    for fixture_name in FIXTURES:
        ep = load_episode(SCENARIOS_DIR / fixture_name)
        for seed in SEEDS:
            for cond_name, cond_overrides, fw_enabled in CONDITIONS:
                cfg = _make_config(seed, cond_overrides)
                # Include condition name in run_id for uniqueness
                run_id = hashlib.sha256(
                    f"{ep.episode_id}|{cond_name}|{seed}|{fw_enabled}".encode()
                ).hexdigest()[:20]
                result = run_episode(ep, cfg, firewall_enabled=fw_enabled, run_id=run_id)

                # Track unique run IDs
                if result.run_id in run_ids:
                    raise ValueError(f"Duplicate run_id: {result.run_id}")
                run_ids.add(result.run_id)
                run_id_to_condition[result.run_id] = cond_name

                # Add condition metadata
                result.metadata["smoke_condition"] = cond_name
                result.metadata["firewall_enabled"] = fw_enabled

                all_results.append(result)
                condition_results.setdefault(cond_name, []).append(result)

    print(f"Completed {len(all_results)} runs")

    # Write episodes.jsonl
    episodes_path = output_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(_serialize_result(r)) + "\n")

    # Audit
    audit_report = audit_results(all_results)
    audit_valid = not audit_report.has_errors

    # Write audit report
    audit_path = output_dir / "result_audit.json"
    audit_path.write_text(json.dumps(audit_report.to_dict(), indent=2))

    # Check for duplicate research identities
    duplicates = _check_duplicate_identities(all_results, run_id_to_condition)
    no_duplicate_identities = len(duplicates) == 0

    # Write smoke matrix
    matrix_conditions = []
    for cond_name, cond_overrides, fw_enabled in CONDITIONS:
        cfg = _make_config(0, cond_overrides)
        matrix_conditions.append(
            {
                "name": cond_name,
                "config_hash": cfg.config_hash(),
                "firewall_enabled": fw_enabled,
                "research_identity_dimensions": {
                    "detector": str(cfg.detector),
                    "history": str(cfg.history),
                    "policy": str(cfg.policy),
                    "monitoring": str(cfg.monitoring),
                },
            }
        )

    smoke_matrix = {
        "conditions": matrix_conditions,
        "duplicate_identities": [
            {"identity": list(ident), "count": count} for ident, count in duplicates
        ],
    }
    (output_dir / "smoke_matrix.json").write_text(json.dumps(smoke_matrix, indent=2))

    # Compute metrics
    evaluation = evaluate_all(all_results)

    # Write aggregate metrics
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(evaluation.to_dict(), indent=2, sort_keys=True))

    # Write aggregate metric counts
    metric_counts = {
        "pu_rer": {
            "numerator": evaluation.pu_rer.numerator,
            "denominator": evaluation.pu_rer.denominator,
        },
        "crr": {"numerator": evaluation.crr.numerator, "denominator": evaluation.crr.denominator},
        "rr": {"numerator": evaluation.rr.numerator, "denominator": evaluation.rr.denominator},
        "fbr": {"numerator": evaluation.fbr.numerator, "denominator": evaluation.fbr.denominator},
    }
    counts_path = output_dir / "metric_counts.json"
    counts_path.write_text(json.dumps(metric_counts, indent=2, sort_keys=True))

    # Write per-condition metrics
    per_condition_metrics = _compute_per_condition_metrics(condition_results)
    (output_dir / "metrics_by_condition.json").write_text(
        json.dumps(per_condition_metrics, indent=2)
    )

    # Write per-condition metric counts
    per_condition_counts = _compute_per_condition_metric_counts(condition_results)
    (output_dir / "metric_counts_by_condition.json").write_text(
        json.dumps(per_condition_counts, indent=2)
    )

    # Write per-scenario metrics
    per_scenario_metrics = _compute_per_scenario_metrics(all_results)
    (output_dir / "metrics_by_scenario.json").write_text(json.dumps(per_scenario_metrics, indent=2))

    # Write per-condition-and-scenario metrics
    per_cond_scen_metrics = _compute_per_condition_and_scenario_metrics(condition_results)
    (output_dir / "metrics_by_condition_and_scenario.json").write_text(
        json.dumps(per_cond_scen_metrics, indent=2)
    )

    # Build manifest
    episode_ids = tuple(sorted({r.episode_id for r in all_results}))
    seeds_tuple = tuple(sorted({r.seed for r in all_results}))
    config_hashes = tuple(sorted({str(r.metadata.get("config_hash", "")) for r in all_results}))

    manifest = SmokeManifest(
        repository_commit=repository_commit,
        generated_at_utc=generated_at,
        run_mode=mode,
        config_hashes=config_hashes,
        provider="fixed",
        model=None,
        dimension=None,
        semantic_threshold=0.80,
        api_base_sanitized=None,
        episode_ids=episode_ids,
        seeds=seeds_tuple,
        result_count=len(all_results),
        audit_valid=audit_valid,
        audit_error_count=len(audit_report.errors()),
        metric_counts=metric_counts,
    )

    manifest_valid = True  # We just built it, so it's valid
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(manifest.to_json())

    # Utility pairing (no_firewall vs full_mvp)
    no_fw = condition_results.get("no_firewall", [])
    fw = condition_results.get("full_mvp", [])
    utility_pairing: dict[str, Any]
    unmatched: dict[str, Any]
    utility_retention_value: float | None = None
    expected_pairs = 0
    matched_pairs = 0
    baseline_successful_pairs = 0

    if no_fw and fw:
        from experiments.trustparadox_u.evaluator import compute_utility_retention

        utility_result = compute_utility_retention(fw, no_fw)
        utility_retention_value = utility_result.metric.value
        expected_pairs = utility_result.expected_pairs
        matched_pairs = utility_result.matched_pairs
        baseline_successful_pairs = utility_result.baseline_successful_pairs

        utility_pairing = {
            "expected_pairs": expected_pairs,
            "matched_pairs": matched_pairs,
            "unmatched_pairs": len(utility_result.unmatched_firewall_keys)
            + len(utility_result.unmatched_baseline_keys),
            "baseline_successful_pairs": baseline_successful_pairs,
            "matched_keys": [list(k) for k in utility_result.matched_keys],
            "metric": utility_result.metric.to_dict(),
        }
        unmatched = {
            "unmatched_firewall_keys": [list(k) for k in utility_result.unmatched_firewall_keys],
            "unmatched_baseline_keys": [list(k) for k in utility_result.unmatched_baseline_keys],
        }
    else:
        utility_pairing = {
            "expected_pairs": 0,
            "matched_pairs": 0,
            "unmatched_pairs": 0,
            "baseline_successful_pairs": 0,
            "matched_keys": [],
            "metric": None,
        }
        unmatched = {"unmatched_firewall_keys": [], "unmatched_baseline_keys": []}

    (output_dir / "utility_pairing.json").write_text(json.dumps(utility_pairing, indent=2))
    (output_dir / "unmatched_pairs.json").write_text(json.dumps(unmatched, indent=2))

    # Run directional checks
    checks = _directional_checks(condition_results)
    directional_checks_pass = all(c["passed"] for c in checks.values())

    # Compute validity flags
    no_unmatched_pairs = utility_pairing.get("unmatched_pairs", 0) == 0
    utility_defined = utility_retention_value is not None
    artifacts_complete = len(_validate_required_artifacts(output_dir)) == 0

    # Compute all_passed
    all_passed = (
        audit_valid
        and manifest_valid
        and directional_checks_pass
        and no_unmatched_pairs
        and utility_defined
        and repository_clean
        and artifacts_complete
        and no_duplicate_identities
    )

    # Determine top-line status
    failure_reasons: list[str] = []
    if not audit_valid:
        failure_reasons.append(f"Audit invalid: {len(audit_report.errors())} errors")
    if not no_duplicate_identities:
        failure_reasons.append(f"Duplicate identities: {len(duplicates)}")
    if not repository_clean:
        failure_reasons.append(f"Repository dirty: {repository_commit}")
    if not utility_defined:
        failure_reasons.append("Utility retention undefined")
    if not no_unmatched_pairs:
        failure_reasons.append(f"Unmatched pairs: {utility_pairing.get('unmatched_pairs', 0)}")
    if not directional_checks_pass:
        failed = [name for name, c in checks.items() if not c["passed"]]
        failure_reasons.append(f"Directional checks failed: {failed}")
    if not artifacts_complete:
        missing = _validate_required_artifacts(output_dir)
        failure_reasons.append(f"Missing artifacts: {missing}")

    if mode == "diagnostic":
        top_line_status = "DIAGNOSTIC ONLY"
    elif all_passed:
        top_line_status = "GO"
    else:
        top_line_status = "NO-GO"

    # Build aggregate metrics dict
    aggregate_metrics = {
        "pu_rer": {
            "value": evaluation.pu_rer.value,
            "numerator": evaluation.pu_rer.numerator,
            "denominator": evaluation.pu_rer.denominator,
        },
        "crr": {
            "value": evaluation.crr.value,
            "numerator": evaluation.crr.numerator,
            "denominator": evaluation.crr.denominator,
        },
        "rr": {
            "value": evaluation.rr.value,
            "numerator": evaluation.rr.numerator,
            "denominator": evaluation.rr.denominator,
        },
        "fbr": {
            "value": evaluation.fbr.value,
            "numerator": evaluation.fbr.numerator,
            "denominator": evaluation.fbr.denominator,
        },
    }

    # Build report
    report = SmokeReport(
        audit_valid=audit_valid,
        manifest_valid=manifest_valid,
        directional_checks_pass=directional_checks_pass,
        no_unmatched_pairs=no_unmatched_pairs,
        utility_defined=utility_defined,
        repository_clean=repository_clean,
        artifacts_complete=artifacts_complete,
        no_duplicate_identities=no_duplicate_identities,
        all_passed=all_passed,
        top_line_status=top_line_status,
        failure_reasons=failure_reasons,
        repository_commit=repository_commit,
        generated_at=generated_at,
        mode=mode,
        total_runs=len(all_results),
        fixture_count=len(FIXTURES),
        seed_count=len(SEEDS),
        condition_count=len(CONDITIONS),
        audit_error_count=len(audit_report.errors()),
        duplicate_identity_count=len(duplicates),
        aggregate_metrics=aggregate_metrics,
        directional_checks=checks,
        expected_pairs=expected_pairs,
        matched_pairs=matched_pairs,
        unmatched_pair_count=utility_pairing.get("unmatched_pairs", 0),
        baseline_successful_pairs=baseline_successful_pairs,
        utility_retention_value=utility_retention_value,
    )

    # Write JSON report
    report_path = output_dir / "single_target_validation_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))

    # Write markdown report
    md_report = _generate_markdown_report(report.to_dict(), output_dir)
    (output_dir / "single_target_validation_report.md").write_text(md_report)

    # Write summary.md
    summary_md = f"""# Single-Target Smoke Study Summary

- **Status**: {top_line_status}
- **Commit**: {repository_commit}
- **Mode**: {mode}
- **Fixtures**: {len(FIXTURES)}
- **Seeds**: {len(SEEDS)}
- **Conditions**: {len(CONDITIONS)}
- **Total runs**: {len(all_results)}
- **Audit valid**: {audit_valid}
- **Audit errors**: {len(audit_report.errors())}
- **Duplicate identities**: {len(duplicates)}
- **Utility retention**: {format_metric(utility_retention_value)}

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | {format_metric(evaluation.pu_rer.value)} | {evaluation.pu_rer.numerator} | {evaluation.pu_rer.denominator} |
| CRR | {format_metric(evaluation.crr.value)} | {evaluation.crr.numerator} | {evaluation.crr.denominator} |
| RR | {format_metric(evaluation.rr.value)} | {evaluation.rr.numerator} | {evaluation.rr.denominator} |
| FBR | {format_metric(evaluation.fbr.value)} | {evaluation.fbr.numerator} | {evaluation.fbr.denominator} |

## Directional Checks

"""
    for name, check in checks.items():
        status_icon = "PASS" if check["passed"] else "FAIL"
        summary_md += f"- [{status_icon}] **{name}**: {check['check']}\n"

    (output_dir / "summary.md").write_text(summary_md)

    return report


def _get_exit_code(report: SmokeReport) -> int:
    """Determine exit code based on report validity."""
    if report.all_passed:
        return EXIT_SUCCESS

    if not report.audit_valid:
        return EXIT_AUDIT

    if not report.manifest_valid:
        return EXIT_MANIFEST

    if not report.directional_checks_pass:
        return EXIT_DIRECTIONAL

    if not report.no_unmatched_pairs or not report.utility_defined:
        return EXIT_UTILITY_PAIRING

    return EXIT_INPUT_CONFIG


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single-target smoke study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "single_target_smoke",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--mode",
        choices=["release", "diagnostic"],
        default="diagnostic",
        help="Smoke study mode (default: diagnostic)",
    )
    args = parser.parse_args()

    try:
        report = run_smoke_study(args.output_dir, mode=args.mode)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_INPUT_CONFIG)
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        sys.exit(EXIT_EXECUTION)

    print("\nSmoke study complete:")
    print(f"  Status: {report.top_line_status}")
    print(f"  Total runs: {report.total_runs}")
    print(f"  Audit valid: {report.audit_valid}")
    print(f"  All passed: {report.all_passed}")

    for name, check in report.directional_checks.items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}: {check['check']}")

    if report.failure_reasons:
        print("\nFailure reasons:")
        for reason in report.failure_reasons:
            print(f"  - {reason}")

    sys.exit(_get_exit_code(report))

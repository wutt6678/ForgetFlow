#!/usr/bin/env python3
"""Fixed-vector single-target smoke study runner.

Runs all canonical fixtures across multiple seeds and conditions,
produces all required artifacts, and validates directional checks.

Usage:
    poetry run python scripts/run_single_target_smoke.py [--output-dir DIR]
"""

from __future__ import annotations

import json
import sys
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
from experiments.trustparadox_u.manifest import get_repository_commit  # noqa: E402
from experiments.trustparadox_u.runner import EpisodeResult, run_episode  # noqa: E402
from experiments.trustparadox_u.serialization import serialize_episode_result  # noqa: E402

SCENARIOS_DIR = PROJECT_ROOT / "data" / "trustparadox_u" / "scenarios"

FIXTURES = [
    "pilot_credential.yaml",
    "pilot_private_attribute.yaml",
    "pilot_authorization.yaml",
]

SEEDS = [42, 123, 7]

# Condition definitions: (name, config_overrides, firewall_enabled)
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
        "rich_policy",
        {
            "policy": PolicyConfig(rich_actions_enabled=True),
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


def run_smoke_study(output_dir: Path) -> dict[str, Any]:
    """Run the complete smoke study and return validation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[EpisodeResult] = []
    condition_results: dict[str, list[EpisodeResult]] = {}
    run_ids: set[str] = set()

    print(
        f"Running smoke study: {len(FIXTURES)} fixtures x {len(SEEDS)} seeds x {len(CONDITIONS)} conditions"
    )

    for fixture_name in FIXTURES:
        ep = load_episode(SCENARIOS_DIR / fixture_name)
        for seed in SEEDS:
            for cond_name, cond_overrides, fw_enabled in CONDITIONS:
                cfg = _make_config(seed, cond_overrides)
                # Include condition name in run_id for uniqueness
                import hashlib

                run_id = hashlib.sha256(
                    f"{ep.episode_id}|{cond_name}|{seed}|{fw_enabled}".encode()
                ).hexdigest()[:20]
                result = run_episode(ep, cfg, firewall_enabled=fw_enabled, run_id=run_id)

                # Track unique run IDs
                if result.run_id in run_ids:
                    raise ValueError(f"Duplicate run_id: {result.run_id}")
                run_ids.add(result.run_id)

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

    # Write audit report
    audit_path = output_dir / "result_audit.json"
    audit_path.write_text(json.dumps(audit_report.to_dict(), indent=2))

    # Compute metrics
    evaluation = evaluate_all(all_results)

    # Write metrics
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(evaluation.to_dict(), indent=2, sort_keys=True))

    # Write metric counts
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

    # Build manifest (smoke study has heterogeneous conditions)
    from experiments.trustparadox_u.manifest import SmokeManifest

    commit = get_repository_commit()
    episode_ids = tuple(sorted({r.episode_id for r in all_results}))
    seeds_tuple = tuple(sorted({r.seed for r in all_results}))
    config_hashes = tuple(sorted({str(r.metadata.get("config_hash", "")) for r in all_results}))

    from datetime import datetime, timezone

    manifest = SmokeManifest(
        repository_commit=commit,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_mode="test",
        config_hashes=config_hashes,
        provider="fixed",
        model=None,
        dimension=None,
        semantic_threshold=0.80,
        api_base_sanitized=None,
        episode_ids=episode_ids,
        seeds=seeds_tuple,
        result_count=len(all_results),
        audit_valid=not audit_report.has_errors,
        audit_error_count=len(audit_report.errors()),
        metric_counts=metric_counts,
    )

    # Write manifest
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(manifest.to_json())

    # Utility pairing (no_firewall vs full_mvp)
    no_fw = condition_results.get("no_firewall", [])
    fw = condition_results.get("full_mvp", [])
    utility_pairing: dict[str, Any]
    unmatched: dict[str, Any]

    if no_fw and fw:
        from experiments.trustparadox_u.evaluator import compute_utility_retention

        utility_result = compute_utility_retention(fw, no_fw)
        utility_pairing = {
            "matched_keys": [list(k) for k in utility_result.matched_keys],
            "metric": utility_result.metric.to_dict(),
        }
        unmatched = {
            "unmatched_firewall_keys": [list(k) for k in utility_result.unmatched_firewall_keys],
            "unmatched_baseline_keys": [list(k) for k in utility_result.unmatched_baseline_keys],
        }
    else:
        utility_pairing = {"matched_keys": [], "metric": None}
        unmatched = {"unmatched_firewall_keys": [], "unmatched_baseline_keys": []}

    (output_dir / "utility_pairing.json").write_text(json.dumps(utility_pairing, indent=2))
    (output_dir / "unmatched_pairs.json").write_text(json.dumps(unmatched, indent=2))

    # Write summary.md
    summary_md = f"""# Single-Target Smoke Study Summary

- **Commit**: {commit}
- **Fixtures**: {len(FIXTURES)}
- **Seeds**: {len(SEEDS)}
- **Conditions**: {len(CONDITIONS)}
- **Total runs**: {len(all_results)}
- **Audit valid**: {not audit_report.has_errors}
- **Audit errors**: {len(audit_report.errors())}

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | {evaluation.pu_rer.value or 'N/A'} | {evaluation.pu_rer.numerator} | {evaluation.pu_rer.denominator} |
| CRR | {evaluation.crr.value or 'N/A'} | {evaluation.crr.numerator} | {evaluation.crr.denominator} |
| RR | {evaluation.rr.value or 'N/A'} | {evaluation.rr.numerator} | {evaluation.rr.denominator} |
| FBR | {evaluation.fbr.value or 'N/A'} | {evaluation.fbr.numerator} | {evaluation.fbr.denominator} |
"""
    (output_dir / "summary.md").write_text(summary_md)

    # Run directional checks
    checks = _directional_checks(condition_results)

    # Write validation report
    report = {
        "commit": commit,
        "total_runs": len(all_results),
        "unique_run_ids": len(run_ids),
        "audit_valid": not audit_report.has_errors,
        "audit_error_count": len(audit_report.errors()),
        "directional_checks": checks,
        "all_passed": all(c["passed"] for c in checks.values()),
    }
    report_path = output_dir / "single_target_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    return report


def _directional_checks(
    condition_results: dict[str, list[EpisodeResult]],
) -> dict[str, dict[str, Any]]:
    """Run required directional checks from the plan."""
    checks: dict[str, dict[str, Any]] = {}

    # Privacy: full MVP PU-RER < no firewall PU-RER
    no_fw_eval = evaluate_all(condition_results.get("no_firewall", []))
    mvp_eval = evaluate_all(condition_results.get("full_mvp", []))
    no_fw_pu = no_fw_eval.pu_rer.value if no_fw_eval.pu_rer.value is not None else 0.0
    mvp_pu = mvp_eval.pu_rer.value if mvp_eval.pu_rer.value is not None else 0.0
    checks["privacy_mvp_better"] = {
        "check": "full_mvp PU-RER <= no_firewall PU-RER",
        "mvp_pu_rer": mvp_pu,
        "no_fw_pu_rer": no_fw_pu,
        "passed": mvp_pu <= no_fw_pu,
    }

    # Semantic protection: semantic-enabled leakage < semantic-disabled leakage
    full_eval = evaluate_all(condition_results.get("full_mvp", []))
    no_sem_eval = evaluate_all(condition_results.get("no_semantic", []))
    full_pu = full_eval.pu_rer.value if full_eval.pu_rer.value is not None else 0.0
    no_sem_pu = no_sem_eval.pu_rer.value if no_sem_eval.pu_rer.value is not None else 0.0
    checks["semantic_protection"] = {
        "check": "full_mvp PU-RER <= no_semantic PU-RER",
        "full_pu_rer": full_pu,
        "no_sem_pu_rer": no_sem_pu,
        "passed": full_pu <= no_sem_pu,
    }

    # Reconstruction: stateful CRR <= stateless CRR (stateful is safer)
    stateful_eval = evaluate_all(condition_results.get("full_mvp", []))
    stateless_eval = evaluate_all(condition_results.get("stateless", []))
    stateful_crr = stateful_eval.crr.value if stateful_eval.crr.value is not None else 0.0
    stateless_crr = stateless_eval.crr.value if stateless_eval.crr.value is not None else 0.0
    checks["stateful_reconstruction_safer"] = {
        "check": "stateful CRR <= stateless CRR",
        "stateful_crr": stateful_crr,
        "stateless_crr": stateless_crr,
        "passed": stateful_crr <= stateless_crr,
    }

    # Policy utility: rich task success >= binary task success
    rich_results = condition_results.get("rich_policy", [])
    binary_results = condition_results.get("binary_policy", [])
    rich_success = sum(1 for r in rich_results if r.task_success)
    binary_success = sum(1 for r in binary_results if r.task_success)
    checks["rich_utility_ge_binary"] = {
        "check": "rich task_success >= binary task_success",
        "rich_success": rich_success,
        "binary_success": binary_success,
        "passed": rich_success >= binary_success,
    }

    # Monitoring: continuous RR <= finite-window RR
    cont_eval = evaluate_all(condition_results.get("continuous", []))
    mon0_eval = evaluate_all(condition_results.get("monitoring_0", []))
    cont_rr = cont_eval.rr.value if cont_eval.rr.value is not None else 0.0
    mon0_rr = mon0_eval.rr.value if mon0_eval.rr.value is not None else 0.0
    checks["continuous_rr_le_finite"] = {
        "check": "continuous RR <= monitoring_0 RR",
        "continuous_rr": cont_rr,
        "monitoring_0_rr": mon0_rr,
        "passed": cont_rr <= mon0_rr,
    }

    return checks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single-target smoke study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "single_target_smoke",
        help="Output directory for artifacts",
    )
    args = parser.parse_args()

    report = run_smoke_study(args.output_dir)

    print("\nSmoke study complete:")
    print(f"  Total runs: {report['total_runs']}")
    print(f"  Unique run IDs: {report['unique_run_ids']}")
    print(f"  Audit valid: {report['audit_valid']}")
    print(f"  All directional checks passed: {report['all_passed']}")

    for name, check in report["directional_checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}: {check['check']}")

    if not report["all_passed"]:
        sys.exit(1)

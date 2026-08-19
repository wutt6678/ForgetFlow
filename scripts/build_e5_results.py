"""E5-012: Build aggregated E5 results from raw outputs (Iteration 12).

Converts raw per-row/per-sequence outputs into:
- overall metrics per condition
- attack-type table
- trust-conditioned table
- utility table
- ablation table
- hyperparameter sensitivity table
- sequence table
- metric eligibility manifest

Plan references:
    §76  create build_e5_results.py
    §77  metric eligibility manifest
    §78  primary test table
    §79  attack robustness table
    §80  utility table
    §81  trust-conditioned table
    §82  ablation table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_ablation_study import (  # noqa: E402
    ablation_impacts_to_dict,
    ablation_metrics_to_dict,
    compute_ablation_impacts,
    run_ablation_study,
)
from experiments.trustparadox_u.e5_attack_analysis import (  # noqa: E402
    compute_attack_robustness,
    compute_hard_negative_analysis,
    compute_legitimate_task_analysis,
    compute_trust_conditioned,
    compute_trust_drift,
)
from experiments.trustparadox_u.e5_hyperparameter_study import (  # noqa: E402
    compute_threshold_sensitivity,
    compute_tradeoff_data,
    run_threshold_sweep,
    select_optimal_threshold,
    sensitivity_to_dict,
    tradeoff_to_dict,
)
from experiments.trustparadox_u.e5_statistics import (  # noqa: E402
    build_eligibility_manifest,
    compute_proportion_ci,
    eligibility_to_dict,
)

# ---------------------------------------------------------------------------
# Inline condition-metrics computation
# ---------------------------------------------------------------------------


def _compute_condition_counts(
    row_results: list[dict[str, Any]],
    row_labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute raw counts for one condition from row results and labels.

    Returns a dict with: n_eligible, n_excluded, n_leaking,
    n_leaking_blocked, n_non_leaking, n_fp, n_useful_eligible,
    n_useful_preserved, leakage_prevention, fbr, utility_retention.
    """
    results_by_id = {r["candidate_id"]: r for r in row_results}

    n_eligible = 0
    n_excluded = 0
    n_leaking = 0
    n_leaking_blocked = 0
    n_leaking_delivered = 0  # leaking AND allowed through
    n_non_leaking = 0
    n_fp = 0
    n_useful_eligible = 0
    n_useful_preserved = 0

    for cid, label in row_labels.items():
        if label.get("is_unresolved"):
            n_excluded += 1
            continue
        result = results_by_id.get(cid)
        if result is None:
            continue

        n_eligible += 1
        is_leaking = label.get("final_target_leakage") is True
        is_useful = label.get("final_task_useful") is True
        is_blocked = result.get("blocked", False)
        is_allowed = result.get("allowed", False)

        if is_leaking:
            n_leaking += 1
            if is_blocked:
                n_leaking_blocked += 1
            if is_allowed:
                n_leaking_delivered += 1
        elif label.get("final_target_leakage") is False:
            n_non_leaking += 1
            if is_blocked:
                n_fp += 1

        if is_useful:
            n_useful_eligible += 1
            if is_allowed:
                n_useful_preserved += 1

    # PU-RER = leakage-through rate = leaking delivered / leaking eligible (§23)
    purer = n_leaking_delivered / n_leaking if n_leaking > 0 else 0.0
    # Leakage Prevention Recall = leaking intercepted / leaking eligible
    leakage_prevention_recall = (
        n_leaking_blocked / n_leaking if n_leaking > 0 else 0.0
    )
    fbr = n_fp / n_non_leaking if n_non_leaking > 0 else 0.0
    utility_retention = (
        n_useful_preserved / n_useful_eligible
        if n_useful_eligible > 0
        else 0.0
    )

    return {
        "n_eligible": n_eligible,
        "n_excluded": n_excluded,
        "n_leaking": n_leaking,
        "n_leaking_blocked": n_leaking_blocked,
        "n_leaking_delivered": n_leaking_delivered,
        "n_non_leaking": n_non_leaking,
        "n_fp": n_fp,
        "n_useful_eligible": n_useful_eligible,
        "n_useful_preserved": n_useful_preserved,
        "PU-RER": purer,
        "leakage_prevention_recall": leakage_prevention_recall,
        "fbr": fbr,
        "utility_retention": utility_retention,
    }


def build_overall_metrics(
    row_results: list[dict[str, Any]],
    row_labels: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    condition: str,
    split: str = "test",
    tau_sem: float = 0.75,
) -> dict[str, Any]:
    """Build overall metrics for one condition.

    Args:
        row_results: Per-row evaluation results.
        row_labels: Per-row ground-truth labels.
        corpus: Per-row corpus metadata.
        condition: Condition ID (C0-C4).
        split: Data split name.
        tau_sem: Semantic threshold used.

    Returns:
        Dict with metrics and CI information.
    """
    metrics = _compute_condition_counts(row_results, row_labels)

    # Add confidence intervals for key proportions
    n_leaking = metrics["n_leaking"]
    n_leaking_delivered = metrics["n_leaking_delivered"]
    n_leaking_blocked = metrics["n_leaking_blocked"]
    n_non_leaking = metrics["n_non_leaking"]
    n_fp = metrics["n_fp"]
    n_useful = metrics["n_useful_eligible"]
    n_useful_preserved = metrics["n_useful_preserved"]

    # PU-RER CI uses leakage-through numerator (§25)
    ci_purer = compute_proportion_ci(
        "PU-RER", n_leaking_delivered, n_leaking,
        split=split, condition=condition,
    )
    ci_fbr = compute_proportion_ci(
        "FBR", n_fp, n_non_leaking,
        split=split, condition=condition,
    )
    ci_util = compute_proportion_ci(
        "utility_retention", n_useful_preserved, n_useful,
        split=split, condition=condition,
    )

    return {
        "condition": condition,
        "split": split,
        "tau_sem": tau_sem,
        "metrics": metrics,
        "confidence_intervals": {
            "PU-RER": {
                "p_hat": ci_purer.p_hat,
                "ci_lower": ci_purer.ci_lower,
                "ci_upper": ci_purer.ci_upper,
            },
            "FBR": {
                "p_hat": ci_fbr.p_hat,
                "ci_lower": ci_fbr.ci_lower,
                "ci_upper": ci_fbr.ci_upper,
            },
            "utility_retention": {
                "p_hat": ci_util.p_hat,
                "ci_lower": ci_util.ci_lower,
                "ci_upper": ci_util.ci_upper,
            },
        },
    }


def build_attack_table(
    row_results_by_condition: dict[str, list[dict[str, Any]]],
    row_labels: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build attack robustness table (plan §79).

    Args:
        row_results_by_condition: condition_id → list of row results.
        row_labels: Per-row labels.
        corpus: Per-row corpus.

    Returns:
        List of per-attack-type metric dicts.
    """
    robustness = compute_attack_robustness(
        row_results_by_condition, row_labels, corpus
    )
    return [
        {
            "attack_type": r.attack_type,
            "n": r.n,
            "n_leaking_eligible": r.n_leaking_eligible,
            "baseline_leakage_through": r.baseline_leakage_through,
            "forgetflow_leakage_through": r.forgetflow_leakage_through,
            "relative_leakage_reduction": r.relative_leakage_reduction,
            "utility_retention": r.utility_retention,
            "fbr": r.fbr,
        }
        for r in robustness
    ]


def build_trust_table(
    row_results: list[dict[str, Any]],
    row_labels: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build trust-conditioned table (plan §81).

    Args:
        row_results: Per-row results.
        row_labels: Per-row labels.
        corpus: Per-row corpus.

    Returns:
        Dict with per-trust metrics and drift.
    """
    trust_rows = compute_trust_conditioned(row_results, row_labels, corpus)
    drift = compute_trust_drift(trust_rows)

    return {
        "per_trust": [
            {
                "trust_level": r.trust_level,
                "n_eligible": r.n_eligible,
                "leakage_prevention": r.leakage_prevention,
                "utility_retention": r.utility_retention,
                "fbr": r.fbr,
            }
            for r in trust_rows
        ],
        "drift": {
            "leakage_rate_drift": drift.leakage_rate_drift,
            "fbr_drift": drift.fbr_drift,
            "utility_drift": drift.utility_drift,
        },
    }


def build_utility_table(
    row_results: list[dict[str, Any]],
    row_labels: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    """Build utility table (plan §80).

    Args:
        row_results: Per-row results.
        row_labels: Per-row labels.
        corpus: Per-row corpus.
        condition: Condition ID.

    Returns:
        Dict with utility metrics.
    """
    hard_neg = compute_hard_negative_analysis(row_results, row_labels, corpus)
    legit = compute_legitimate_task_analysis(row_results, row_labels, corpus)

    return {
        "condition": condition,
        "hard_negative": {
            "n_hard_negatives": hard_neg.n_hard_negatives,
            "n_blocked": hard_neg.n_blocked,
            "overblocking_rate": hard_neg.overblocking_rate,
        },
        "legitimate_task": {
            "n_legitimate": legit.n_legitimate,
            "n_preserved": legit.n_preserved,
            "utility_rate": legit.utility_rate,
        },
    }


def build_ablation_table(
    row_results: list[dict[str, Any]],
    row_labels: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    tau_sem: float = 0.75,
) -> dict[str, Any]:
    """Build ablation table (plan §82).

    Args:
        row_results: Per-row results.
        row_labels: Per-row labels.
        corpus: Per-row corpus.
        tau_sem: Semantic threshold.

    Returns:
        Dict with ablation metrics and impacts.
    """
    # Convert row_results list → features_by_id dict (§20-§26 new API)
    features_by_id: dict[str, dict[str, Any]] = {}
    for r in row_results:
        cid = r["candidate_id"]
        features_by_id[cid] = {
            "exact_match": r.get("exact_match", False),
            "alias_match": r.get("alias_match", False),
            "semantic_similarity": r.get("semantic_similarity", 0.0),
        }

    study = run_ablation_study(features_by_id, row_labels, tau_sem=tau_sem)
    impacts = compute_ablation_impacts(study)

    return {
        "ablations": ablation_metrics_to_dict(list(study.ablations)),
        "impacts": ablation_impacts_to_dict(impacts),
        "baseline_id": study.baseline_id,
    }


def build_hyperparameter_table(
    row_results: list[dict[str, Any]],
    row_labels: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build hyperparameter sensitivity table.

    Args:
        row_results: Per-row results.
        row_labels: Per-row labels.
        corpus: Per-row corpus.

    Returns:
        Dict with sensitivity, tradeoff, and recommendation.
    """
    swept = run_threshold_sweep(row_results)
    sens = compute_threshold_sensitivity(swept, row_labels, corpus)
    tradeoff = compute_tradeoff_data(sens)
    rec = select_optimal_threshold(sens)

    return {
        "sensitivity": sensitivity_to_dict(sens),
        "tradeoff": tradeoff_to_dict(tradeoff),
        "recommendation": {
            "tau_sem": rec.tau_sem,
            "leakage_recall": rec.leakage_recall,
            "fbr": rec.fbr,
            "utility_retention": rec.utility_retention,
            "reason": rec.reason,
        },
    }


def build_eligibility(
    overall_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build metric eligibility manifest (plan §77).

    Args:
        overall_results: List of overall metric dicts per condition.

    Returns:
        Eligibility manifest as list of dicts.
    """
    metrics_for_manifest: list[dict[str, Any]] = []

    for res in overall_results:
        cond = res["condition"]
        split = res["split"]
        m = res["metrics"]

        # PU-RER (leakage-through rate: delivered / eligible)
        metrics_for_manifest.append({
            "metric_name": "PU-RER",
            "split": split,
            "condition": cond,
            "n_total": m.get("n_eligible", 0),
            "n_eligible": m.get("n_leaking", 0),
            "n_excluded_unresolved": m.get("n_excluded", 0),
            "numerator": m.get("n_leaking_delivered", 0),
            "denominator": m.get("n_leaking", 0),
            "value": m.get("PU-RER", 0.0),
            "ci_lower": res["confidence_intervals"]["PU-RER"]["ci_lower"],
            "ci_upper": res["confidence_intervals"]["PU-RER"]["ci_upper"],
        })

        # FBR
        metrics_for_manifest.append({
            "metric_name": "FBR",
            "split": split,
            "condition": cond,
            "n_total": m.get("n_eligible", 0),
            "n_eligible": m.get("n_non_leaking", 0),
            "n_excluded_unresolved": m.get("n_excluded", 0),
            "numerator": m.get("n_fp", 0),
            "denominator": m.get("n_non_leaking", 0),
            "value": m.get("fbr", 0.0),
            "ci_lower": res["confidence_intervals"]["FBR"]["ci_lower"],
            "ci_upper": res["confidence_intervals"]["FBR"]["ci_upper"],
        })

        # Utility retention
        metrics_for_manifest.append({
            "metric_name": "utility_retention",
            "split": split,
            "condition": cond,
            "n_total": m.get("n_eligible", 0),
            "n_eligible": m.get("n_useful_eligible", 0),
            "n_excluded_unresolved": m.get("n_excluded", 0),
            "numerator": m.get("n_useful_preserved", 0),
            "denominator": m.get("n_useful_eligible", 0),
            "value": m.get("utility_retention", 0.0),
            "ci_lower": res["confidence_intervals"]["utility_retention"]["ci_lower"],
            "ci_upper": res["confidence_intervals"]["utility_retention"]["ci_upper"],
        })

    rows = build_eligibility_manifest(metrics_for_manifest)
    return eligibility_to_dict(rows)


def build_e5_results(
    row_results_by_condition: dict[str, list[dict[str, Any]]],
    row_labels: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    *,
    split: str = "test",
    tau_sem: float = 0.75,
) -> dict[str, Any]:
    """Build complete E5 aggregated results.

    Args:
        row_results_by_condition: condition_id → list of row results.
        row_labels: candidate_id → label dict.
        corpus: candidate_id → corpus dict.
        split: Data split name.
        tau_sem: Frozen semantic threshold.

    Returns:
        Complete E5 results dict with all tables.
    """
    # Overall metrics per condition
    overall: list[dict[str, Any]] = []
    for cond in sorted(row_results_by_condition.keys()):
        res = build_overall_metrics(
            row_results_by_condition[cond], row_labels, corpus,
            cond, split, tau_sem,
        )
        overall.append(res)

    # Attack table (uses all conditions for baseline comparison)
    attack_table = build_attack_table(
        row_results_by_condition, row_labels, corpus
    )

    # Trust table (full system C4)
    trust_table: dict[str, Any] = {}
    if "C4" in row_results_by_condition:
        trust_table = build_trust_table(
            row_results_by_condition["C4"], row_labels, corpus
        )

    # Utility table per condition
    utility_tables = []
    for cond in sorted(row_results_by_condition.keys()):
        ut = build_utility_table(
            row_results_by_condition[cond], row_labels, corpus, cond
        )
        utility_tables.append(ut)

    # Ablation table (uses C4 results)
    ablation_table: dict[str, Any] = {}
    if "C4" in row_results_by_condition:
        ablation_table = build_ablation_table(
            row_results_by_condition["C4"], row_labels, corpus, tau_sem
        )

    # Hyperparameter table (uses C4 results)
    hyperparameter_table: dict[str, Any] = {}
    if "C4" in row_results_by_condition:
        hyperparameter_table = build_hyperparameter_table(
            row_results_by_condition["C4"], row_labels, corpus
        )

    # Eligibility manifest
    eligibility = build_eligibility(overall)

    return {
        "split": split,
        "tau_sem": tau_sem,
        "overall": overall,
        "attack_table": attack_table,
        "trust_table": trust_table,
        "utility_tables": utility_tables,
        "ablation_table": ablation_table,
        "hyperparameter_table": hyperparameter_table,
        "eligibility_manifest": eligibility,
    }


def main() -> None:
    """CLI entry point for build_e5_results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build aggregated E5 results from raw outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw row results JSON files per condition.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to row labels JSON file.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for aggregated results JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split name (default: test).",
    )
    parser.add_argument(
        "--tau-sem",
        type=float,
        default=0.75,
        help="Frozen semantic threshold (default: 0.75).",
    )

    args = parser.parse_args()

    # Load inputs
    row_labels = json.loads(args.labels.read_text())
    corpus_data = json.loads(args.corpus.read_text())

    # Load per-condition results
    row_results_by_condition: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(args.input_dir.glob("*.json")):
        cond = f.stem
        row_results_by_condition[cond] = json.loads(f.read_text())

    # Build results
    results = build_e5_results(
        row_results_by_condition,
        row_labels,
        corpus_data,
        split=args.split,
        tau_sem=args.tau_sem,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"E5 results written to {args.output}")


if __name__ == "__main__":
    main()

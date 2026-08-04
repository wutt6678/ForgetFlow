"""FF92-018: Core hyperparameter sweep (one parameter at a time).

Sweeps real runtime hyperparameters on the full-MVP base configuration,
holding every other setting fixed:

- detector.embedding_threshold   (validation split; PU-RER vs FBR)
- detector.claim_confidence_threshold
                                   (validation split; claim PU-RER vs
                                    claim-control FBR)
- history.window_size            (development split; CRR vs memory/latency)
- monitoring.duration_rounds     (development split; RR vs monitoring cost;
                                  continuous=False so duration is active)

Protocol:
- one parameter at a time for interpretability;
- every other setting stays at the full MVP;
- sweeps never touch the test split; the frozen configuration is
  evaluated once on test at the end;
- failures raise (FF92-025 fail-fast); unevaluable metrics keep their
  None value and reason, never coerced to zero.

Split note: the FF92-010 template-family split places reconstruction
sequences and recontamination probes only in the development split, so
window_size and duration_rounds are swept there. Validation and test are
still strictly separated.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.candidates import (  # noqa: E402
    FrozenCandidate,
    load_frozen_corpus,
)
from experiments.trustparadox_u.conditions import (  # noqa: E402
    condition_diff_paths,
    full_mvp_config,
)
from experiments.trustparadox_u.config import ExperimentConfig  # noqa: E402
from experiments.trustparadox_u.dataset import TrustParadoxEpisode  # noqa: E402
from experiments.trustparadox_u.evaluator import EvalMetrics, evaluate_all  # noqa: E402
from experiments.trustparadox_u.frozen_replay import (  # noqa: E402
    _load_scenario_episode,
    build_trial_episode,
    build_trial_responder,
    partition_trial_units,
    target_spec_from_episode,
)
from experiments.trustparadox_u.runner import EpisodeResult, run_episode  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"
RESULTS_DIR = Path(__file__).parents[2] / "results" / "parameter_sweep"
SCENARIO_IDS = ("credential_001", "attribute_001", "auth_001")

# ---------------------------------------------------------------------------
# Sweep definitions (FF92-018 required core hyperparameters)
# ---------------------------------------------------------------------------

# FF92-010: reconstruction sequences and recontamination probes live only in
# the development split, so the parameters whose metric function needs them
# (CRR, RR) are swept there. Validation never mixes with test.
_NO_RECONSTRUCTION_IN_VALIDATION = (
    "The FF92-010 template-family split places no reconstruction/recontamination "
    "candidates in the validation split, so this parameter's metric is not "
    "evaluable there; swept on development instead."
)


@dataclass(frozen=True)
class SweepSpec:
    """One swept hyperparameter and its measurement protocol."""

    name: str
    config_path: str
    expected_diff_paths: frozenset[str]
    values: tuple[float, ...]
    split: str
    split_rationale: str
    metric_function: str
    primary_metric: str
    primary_population: str | None
    secondary_metric: str | None
    secondary_population: str | None
    selection_rule: str
    cost_proxy: str


SWEEP_SPECS: tuple[SweepSpec, ...] = (
    SweepSpec(
        name="embedding_threshold",
        config_path="detector.embedding_threshold",
        expected_diff_paths=frozenset({"detector.embedding_threshold"}),
        values=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
        split="validation",
        split_rationale="Exposure (paraphrase) and benign candidates are in validation.",
        metric_function="PU-RER vs FBR",
        primary_metric="pu_rer",
        primary_population=None,
        secondary_metric="fbr",
        secondary_population=None,
        selection_rule=(
            "Minimize PU-RER; ties broken by lower FBR, then closeness to the "
            "full-MVP default."
        ),
        cost_proxy="none",
    ),
    SweepSpec(
        name="claim_confidence_threshold",
        config_path="detector.claim_confidence_threshold",
        expected_diff_paths=frozenset({"detector.claim_confidence_threshold"}),
        values=(0.30, 0.50, 0.70, 0.85),
        split="validation",
        split_rationale="Claim (claim_past) and claim-control candidates are in validation.",
        metric_function="claim PU-RER vs claim-control FBR",
        primary_metric="pu_rer",
        primary_population="claim_past",
        secondary_metric="fbr",
        secondary_population="claim_question_control",
        selection_rule=(
            "Minimize claim PU-RER; ties broken by lower claim-control FBR, then "
            "closeness to the full-MVP default."
        ),
        cost_proxy="none",
    ),
    SweepSpec(
        name="history.window_size",
        config_path="history.window_size",
        expected_diff_paths=frozenset({"history.window_size"}),
        values=(1.0, 2.0, 3.0, 5.0, 8.0),
        split="development",
        split_rationale=_NO_RECONSTRUCTION_IN_VALIDATION,
        metric_function="CRR vs memory/latency",
        primary_metric="crr",
        primary_population=None,
        secondary_metric=None,
        secondary_population=None,
        selection_rule=(
            "Minimize CRR; ties broken by smaller window (memory cost), then "
            "closeness to the full-MVP default."
        ),
        cost_proxy="window_size (memory); elapsed_seconds (latency)",
    ),
    SweepSpec(
        name="monitoring.duration_rounds",
        config_path="monitoring.duration_rounds",
        # FF92-018: continuous=True makes duration_rounds a no-op, so the
        # sweep pins continuous=False and both paths differ from the base.
        expected_diff_paths=frozenset(
            {"monitoring.duration_rounds", "monitoring.continuous"}
        ),
        values=(0.0, 1.0, 3.0, 5.0),
        split="development",
        split_rationale=_NO_RECONSTRUCTION_IN_VALIDATION,
        metric_function="RR vs monitoring cost",
        primary_metric="rr",
        primary_population=None,
        secondary_metric=None,
        secondary_population=None,
        selection_rule=(
            "Minimize RR; ties broken by shorter duration (monitoring cost), "
            "then closeness to the full-MVP default."
        ),
        cost_proxy="duration_rounds; elapsed_seconds",
    ),
)


def split_corpus_path(split: str) -> Path:
    """Path of the frozen corpus file for one split."""
    path = CORPUS_DIR / f"frozen_corpus_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split corpus not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def build_sweep_config(spec: SweepSpec, value: float, seed: int = 42) -> ExperimentConfig:
    """Full-MVP base with exactly the swept parameter changed.

    Raises if the resulting config differs from the full MVP at any path
    outside the spec's documented paths — this is the runtime proof that
    one parameter at a time is varied and nothing inert is swept.
    """
    base = full_mvp_config(seed=seed)
    override = build_sweep_override(spec, value)
    config = dataclasses.replace(base, **override)
    diff = condition_diff_paths(base, config)
    # The diff must stay within the swept parameter's documented paths; a
    # swept value equal to the full-MVP default legitimately yields no diff.
    extra = diff - set(spec.expected_diff_paths)
    if extra:
        raise AssertionError(
            f"Sweep config for {spec.name}={value} differs from full MVP at "
            f"unexpected paths {sorted(extra)}; allowed: {sorted(spec.expected_diff_paths)}"
        )
    return config


def build_sweep_override(spec: SweepSpec, value: float) -> dict[str, Any]:
    """ExperimentConfig replace-overrides for one swept value."""
    if spec.config_path == "monitoring.duration_rounds":
        # continuous=False is what makes duration_rounds active at runtime.
        base = full_mvp_config()
        monitoring = dataclasses.replace(
            base.monitoring,
            continuous=False,
            duration_rounds=int(value),
        )
        return {"monitoring": monitoring}
    section, field_name = spec.config_path.split(".")
    base = full_mvp_config()
    section_obj = getattr(base, section)
    if isinstance(value, float) and field_name in ("window_size", "duration_rounds"):
        typed_value: Any = int(value)
    else:
        typed_value = value
    return {section: dataclasses.replace(section_obj, **{field_name: typed_value})}


def build_frozen_config(selected: dict[str, float], seed: int = 42) -> ExperimentConfig:
    """Apply every selected value on top of the full-MVP base."""
    config = full_mvp_config(seed=seed)
    for spec in SWEEP_SPECS:
        override = build_sweep_override(spec, selected[spec.config_path])
        config = dataclasses.replace(config, **override)
    return config


# ---------------------------------------------------------------------------
# Trial execution (frozen_replay infrastructure, fail-fast)
# ---------------------------------------------------------------------------


def run_trials(
    config: ExperimentConfig,
    candidates: list[FrozenCandidate],
    scenario_episodes: dict[str, TrustParadoxEpisode],
    run_id: str,
) -> tuple[list[EpisodeResult], float]:
    """Replay every trial unit under one config; any failure raises."""
    units = partition_trial_units(candidates)
    results: list[EpisodeResult] = []
    start = time.monotonic()
    for unit in units:
        candidate = unit.representative
        base_ep = scenario_episodes.get(candidate.scenario_id)
        if base_ep is None:
            raise ValueError(
                f"No base scenario episode loaded for {candidate.scenario_id!r} "
                f"(candidate {candidate.candidate_id!r})"
            )
        spec = target_spec_from_episode(base_ep, candidate)
        trial_ep = build_trial_episode(
            base_ep, candidate, spec, sequence_members=unit.members
        )
        responder = build_trial_responder(
            trial_ep, candidate, sequence_members=unit.members
        )
        result = run_episode(
            episode=trial_ep,
            config=config,
            responder=responder,
            run_id=run_id,
        )
        result.candidate_sample_id = candidate.candidate_id
        # Same lineage contract as frozen replay: construction defects raise.
        if result.trust_level != candidate.trust_level:
            raise ValueError(
                f"Trust level mismatch for {candidate.candidate_id!r}: "
                f"episode={result.trust_level!r} candidate={candidate.trust_level!r}"
            )
        results.append(result)
    return results, time.monotonic() - start


def filter_results_by_attack(
    results: list[EpisodeResult], attack_type: str
) -> list[EpisodeResult]:
    """Keep only trials whose episode attack type matches."""
    return [r for r in results if r.metadata.get("attack_type") == attack_type]


def extract_sweep_metrics(
    spec: SweepSpec, results: list[EpisodeResult]
) -> dict[str, dict[str, Any]]:
    """The metric pair belonging to the parameter's function.

    Values are kept exactly as produced — an unevaluable metric stays
    None with its reason; it is never coerced to zero.
    """
    if spec.name == "embedding_threshold":
        metrics = evaluate_all(results)
        return {"pu_rer": metrics.pu_rer.to_dict(), "fbr": metrics.fbr.to_dict()}
    if spec.name == "claim_confidence_threshold":
        claim = evaluate_all(filter_results_by_attack(results, "claim_past"))
        control = evaluate_all(
            filter_results_by_attack(results, "claim_question_control")
        )
        return {
            "claim_pu_rer": claim.pu_rer.to_dict(),
            "claim_control_fbr": control.fbr.to_dict(),
        }
    if spec.name == "history.window_size":
        metrics = evaluate_all(results)
        return {"crr": metrics.crr.to_dict()}
    if spec.name == "monitoring.duration_rounds":
        metrics = evaluate_all(results)
        return {"rr": metrics.rr.to_dict()}
    raise ValueError(f"Unknown sweep spec: {spec.name!r}")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_point(spec: SweepSpec, points: list[dict[str, Any]]) -> float:
    """Deterministic selection: minimize the primary leakage metric.

    Ties break on the secondary metric (when the spec defines one and it is
    evaluable), then closeness to the full-MVP default, then the smaller
    value. Raises when the primary metric is unevaluable at any point: the
    chosen split must support the parameter's metric function.
    """
    base = full_mvp_config()
    section, field_name = spec.config_path.split(".")
    default_value = float(getattr(getattr(base, section), field_name))

    primary_key = (
        "claim_pu_rer" if spec.name == "claim_confidence_threshold" else spec.primary_metric
    )
    secondary_key = (
        "claim_control_fbr"
        if spec.name == "claim_confidence_threshold"
        else spec.secondary_metric
    )

    def sort_key(point: dict[str, Any]) -> tuple[float, float, float, float]:
        primary = point["metrics"][primary_key]
        if not primary["evaluable"] or primary["value"] is None:
            raise ValueError(
                f"Primary metric {primary_key!r} is not evaluable for "
                f"{spec.name}={point['value']} on the {spec.split} split; "
                f"reason: {primary['reason']!r}"
            )
        secondary = 0.0
        if secondary_key is not None:
            sec = point["metrics"][secondary_key]
            if sec["evaluable"] and sec["value"] is not None:
                secondary = float(sec["value"])
        return (
            float(primary["value"]),
            secondary,
            abs(float(point["value"]) - default_value),
            float(point["value"]),
        )

    return float(min(points, key=sort_key)["value"])


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------


def load_scenario_episodes() -> dict[str, TrustParadoxEpisode]:
    episodes: dict[str, TrustParadoxEpisode] = {}
    for scenario_id in SCENARIO_IDS:
        episodes[scenario_id] = _load_scenario_episode(scenario_id)
    return episodes


def run_sweep(
    seed: int = 42,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """Run all one-at-a-time sweeps and the final frozen test evaluation."""
    scenario_episodes = load_scenario_episodes()
    split_candidates: dict[str, list[FrozenCandidate]] = {}
    sweeps: dict[str, Any] = {}
    selected: dict[str, float] = {}

    for spec in SWEEP_SPECS:
        if spec.split not in split_candidates:
            index = load_frozen_corpus(split_corpus_path(spec.split))
            candidates = list(index.candidates)
            if max_candidates is not None:
                candidates = candidates[:max_candidates]
            split_candidates[spec.split] = candidates
        candidates = split_candidates[spec.split]
        print(f"\nSweeping {spec.name} on {spec.split} ({len(candidates)} candidates)")

        points: list[dict[str, Any]] = []
        for value in spec.values:
            config = build_sweep_config(spec, value, seed=seed)
            run_id = f"sweep_{spec.name}_{value}"
            results, elapsed = run_trials(
                config, candidates, scenario_episodes, run_id
            )
            metrics = extract_sweep_metrics(spec, results)
            points.append(
                {
                    "value": value,
                    "config_hash": config.config_hash(),
                    "condition_hash": config.condition_hash(),
                    "metrics": metrics,
                    "num_trials": len(results),
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
            print(f"  {spec.name}={value}: {len(results)} trials, {elapsed:.1f}s")

        selected_value = select_point(spec, points)
        selected[spec.config_path] = selected_value
        sweeps[spec.name] = {
            "config_path": spec.config_path,
            "split": spec.split,
            "split_rationale": spec.split_rationale,
            "metric_function": spec.metric_function,
            "selection_rule": spec.selection_rule,
            "cost_proxy": spec.cost_proxy,
            "points": points,
            "selected_value": selected_value,
        }

    frozen_config = build_frozen_config(selected, seed=seed)
    final_evaluation = run_final_test_evaluation(
        frozen_config, scenario_episodes, seed=seed, max_candidates=max_candidates
    )

    return {
        "schema_version": "2.0",
        "repair_item": "FF92-018",
        "protocol": "one_parameter_at_a_time",
        "base_condition": "full_mvp",
        "seed": seed,
        "sweeps": sweeps,
        "frozen_config": {
            "selected_values": selected,
            "config_hash": frozen_config.config_hash(),
            "condition_hash": frozen_config.condition_hash(),
        },
        "final_test_evaluation": final_evaluation,
        "validation": build_sweep_validation(sweeps),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_final_test_evaluation(
    config: ExperimentConfig,
    scenario_episodes: dict[str, TrustParadoxEpisode],
    seed: int = 42,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen configuration exactly once on the test split."""
    index = load_frozen_corpus(split_corpus_path("test"))
    candidates = list(index.candidates)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    print(f"\nFinal evaluation of frozen config on test ({len(candidates)} candidates)")
    results, elapsed = run_trials(config, candidates, scenario_episodes, "sweep_final_test")
    metrics_eval: EvalMetrics = evaluate_all(results)
    metrics_dict = {
        "pu_rer": metrics_eval.pu_rer.to_dict(),
        "crr": metrics_eval.crr.to_dict(),
        "rr": metrics_eval.rr.to_dict(),
        "rr_clean": metrics_eval.rr_clean.to_dict(),
        "rr_at_risk": metrics_eval.rr_at_risk.to_dict(),
        "fbr": metrics_eval.fbr.to_dict(),
        "paired_policy_utility_retention": metrics_eval.paired_policy_utility_retention.to_dict(),
    }
    return {
        "split": "test",
        "config_hash": config.config_hash(),
        "condition_hash": config.condition_hash(),
        "metrics": metrics_dict,
        "num_trials": len(results),
        "elapsed_seconds": round(elapsed, 2),
    }


def build_sweep_validation(sweeps: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Artifact-level acceptance checks for FF92-018.

    distinct_condition_hashes: every swept value produces a distinct
    condition hash, i.e. the value reaches active runtime logic (no inert
    parameters). split_discipline: no sweep touches the test split and the
    final evaluation runs on test exactly once.
    """
    checks: dict[str, dict[str, Any]] = {}

    hash_problems: list[str] = []
    for name, sweep in sweeps.items():
        hashes = [p["condition_hash"] for p in sweep["points"]]
        if len(set(hashes)) != len(hashes):
            hash_problems.append(name)
    checks["distinct_condition_hashes"] = {
        "passed": not hash_problems,
        "failed_sweeps": hash_problems,
    }

    splits_used = sorted({sweep["split"] for sweep in sweeps.values()})
    checks["split_discipline"] = {
        "passed": "test" not in splits_used,
        "sweep_splits": splits_used,
        "final_evaluation_split": "test",
    }
    return checks


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------


def write_sweep_results(summary: dict[str, Any], output_dir: Path) -> None:
    """Write sweep_summary.json plus the full point detail."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    with (output_dir / "sweep_points.jsonl").open("w") as fh:
        for name, sweep in summary["sweeps"].items():
            for point in sweep["points"]:
                record = {"sweep": name, **point}
                fh.write(json.dumps(record) + "\n")

    print(f"\nSweep results written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the FF92-018 hyperparameter sweep."""
    import argparse

    parser = argparse.ArgumentParser(description="FF92-018 hyperparameter sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for sweep artifacts",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional per-split candidate limit (testing only)",
    )
    args = parser.parse_args()

    print("FF92-018: Core Hyperparameter Sweep (one parameter at a time)")
    print("=" * 60)

    summary = run_sweep(seed=args.seed, max_candidates=args.max_candidates)
    write_sweep_results(summary, args.output_dir)

    failed = [name for name, check in summary["validation"].items() if not check["passed"]]
    if failed:
        raise SystemExit(f"Sweep validation failed: {failed}")

    print("\nSelected (frozen) values:")
    for path, value in summary["frozen_config"]["selected_values"].items():
        print(f"  {path} = {value}")
    final = summary["final_test_evaluation"]["metrics"]
    print(
        "\nFinal test evaluation: "
        f"pu_rer={final['pu_rer']['value']} crr={final['crr']['value']} "
        f"rr={final['rr']['value']} fbr={final['fbr']['value']}"
    )
    print("\nExit criterion: PASSED (all sweeps completed; validation checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

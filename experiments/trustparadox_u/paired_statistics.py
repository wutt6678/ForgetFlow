"""FF92-017: Paired statistics computed from trial artifacts.

Paired unit: ``candidate_id`` (or ``sequence_id`` for reconstruction /
CRR comparisons).  Binary outcomes follow the scientific definitions:

    exposure        independent released exposure label is positive
    reconstruction  sequence trial recovered = true
    recontamination probe_recovered_target = true
    utility         task_success = true (paired utility trials)
    false_block     legitimate candidate blocked = true

Statistics (pure stdlib, exact implementations):

    exact McNemar test (exact binomial on discordant pairs)
    paired permutation test (sign-flip, seeded RNG)
    paired bootstrap 95% confidence interval (percentile method)

Every comparison reports n paired units, rate A, rate B, risk
difference, relative risk (where defined), 95% CI, p-values and an
effect size.  Duplicate pairing units fail loudly; unmatched units are
reported.  No shallow episode counter is used.
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.trial_artifacts import (  # noqa: E402
    STATUS_SUCCESS,
    CandidateTrial,
    UtilityTrial,
    load_trial_records,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parents[2] / "results" / "frozen_replay"
STATS_DIR = Path(__file__).parents[2] / "results" / "paired_statistics"

BASELINE_CONDITION = "no_firewall"

# Condition pairs to compare.  Utility and false-block comparisons only
# apply to pairs that include the baseline condition.
CONDITION_PAIRS: list[tuple[str, str]] = [
    ("no_firewall", "full_mvp"),
    ("no_firewall", "no_monitoring"),
    ("no_firewall", "no_claim_detection"),
    ("no_firewall", "binary_policy"),
    ("no_firewall", "one_time_monitoring"),
    ("full_mvp", "no_monitoring"),
    ("full_mvp", "no_claim_detection"),
    ("full_mvp", "binary_policy"),
    ("full_mvp", "one_time_monitoring"),
]

_LEGITIMATE_ATTACK_TYPES = frozenset({"legitimate_task", "benign_control"})

DEFAULT_SEED = 42
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000


# ---------------------------------------------------------------------------
# Statistical functions (exact, stdlib only)
# ---------------------------------------------------------------------------


def exact_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Exact McNemar test: two-sided exact binomial on discordant pairs.

    ``b`` = A positive / B negative, ``c`` = A negative / B positive.
    Under H0 the smaller discordant count follows Binom(b+c, 0.5); the
    two-sided p-value is twice the smaller tail, capped at 1.  This is an
    exact computation (math.comb), not an asymptotic approximation.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "discordant": 0, "p_value": 1.0, "test": "exact_mcnemar"}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    p_value = min(1.0, 2.0 * tail)
    return {
        "b": b,
        "c": c,
        "discordant": n,
        "p_value": p_value,
        "test": "exact_mcnemar",
    }


def paired_permmutation_test(
    differences: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    n_permutations: int = DEFAULT_PERMUTATIONS,
) -> dict[str, Any]:
    """Paired sign-flip permutation test on the mean paired difference.

    Deterministic given ``seed``.  Differences are in {-1, 0, 1}.
    """
    nonzero = [d for d in differences if d != 0]
    observed = sum(differences) / len(differences) if differences else 0.0
    if not nonzero:
        return {
            "statistic": observed,
            "p_value": 1.0,
            "n_permutations": 0,
            "seed": seed,
            "test": "paired_permmutation",
        }
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_permutations):
        flipped = sum(d if rng.random() >= 0.5 else -d for d in nonzero)
        permuted = flipped / len(differences)
        if abs(permuted) >= abs(observed) - 1e-12:
            extreme += 1
    return {
        "statistic": observed,
        "p_value": extreme / n_permutations,
        "n_permutations": n_permutations,
        "seed": seed,
        "test": "paired_permmutation",
    }


def paired_bootstrap_ci(
    differences: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    n_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean paired difference (risk diff)."""
    if not differences:
        return {
            "lower": None,
            "upper": None,
            "confidence_level": 1.0 - alpha,
            "n_iterations": 0,
            "seed": seed,
        }
    rng = random.Random(seed)
    n = len(differences)
    estimates: list[float] = []
    for _ in range(n_iterations):
        sample = sum(differences[rng.randrange(n)] for _ in range(n))
        estimates.append(sample / n)
    estimates.sort()
    lower = estimates[int((alpha / 2) * n_iterations)]
    upper = estimates[min(n_iterations - 1, int((1 - alpha / 2) * n_iterations))]
    return {
        "lower": lower,
        "upper": upper,
        "confidence_level": 1.0 - alpha,
        "n_iterations": n_iterations,
        "seed": seed,
    }


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


# ---------------------------------------------------------------------------
# Outcome indexing
# ---------------------------------------------------------------------------


def exposure_outcomes_by_condition(
    candidate_trials: Sequence[CandidateTrial],
) -> dict[str, dict[str, bool]]:
    """candidate_id → released-exposure-positive, per condition (attacks only)."""
    outcomes: dict[str, dict[str, bool]] = {}
    for trial in candidate_trials:
        if trial.attack_type in _LEGITIMATE_ATTACK_TYPES:
            continue
        if trial.result_status != STATUS_SUCCESS:
            continue
        condition = outcomes.setdefault(trial.condition_id, {})
        if trial.candidate_id in condition:
            raise ValueError(
                f"Duplicate candidate pairing unit {trial.candidate_id!r} "
                f"under condition {trial.condition_id!r}"
            )
        condition[trial.candidate_id] = trial.released_exposure_positive
    return outcomes


def reconstruction_outcomes_by_condition(
    reconstruction_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, bool]]:
    """(episode_id|sequence_id|forget_id) → recovered, per condition (eligible only).

    The same sequence runs once per trust level as a distinct trial
    episode, and the episode id is condition-independent, so it keeps
    pairing units unique while staying pairable across conditions.
    """
    outcomes: dict[str, dict[str, bool]] = {}
    for record in reconstruction_records:
        if not record.get("eligible"):
            continue
        unit_id = "|".join(
            str(record.get(key, "")) for key in ("episode_id", "sequence_id", "forget_id")
        )
        condition = outcomes.setdefault(record.get("condition", ""), {})
        if unit_id in condition:
            raise ValueError(f"Duplicate sequence pairing unit: {unit_id!r}")
        condition[unit_id] = bool(record.get("recovered"))
    return outcomes


def recontamination_outcomes_by_condition(
    recontamination_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, bool]]:
    """(episode_id|agent_id|forget_id) → probe_recovered_target, evaluable only.

    The probe label (candidate_id) is shared across every episode of a
    scenario, so the episode-scoped identity is the pairing unit.
    """
    outcomes: dict[str, dict[str, bool]] = {}
    for record in recontamination_records:
        if not record.get("probe_executed"):
            continue
        unit_id = "|".join(
            str(record.get(key, "")) for key in ("episode_id", "agent_id", "forget_id")
        )
        condition = outcomes.setdefault(record.get("condition", ""), {})
        if unit_id in condition:
            raise ValueError(f"Duplicate recontamination pairing unit: {unit_id!r}")
        condition[unit_id] = bool(record.get("probe_recovered_target"))
    return outcomes


def false_block_outcomes_by_condition(
    candidate_trials: Sequence[CandidateTrial],
) -> dict[str, dict[str, bool]]:
    """candidate_id → blocked_legitimate for legitimate candidates."""
    outcomes: dict[str, dict[str, bool]] = {}
    for trial in candidate_trials:
        if trial.attack_type not in _LEGITIMATE_ATTACK_TYPES:
            continue
        if trial.result_status != STATUS_SUCCESS:
            continue
        condition = outcomes.setdefault(trial.condition_id, {})
        if trial.candidate_id in condition:
            raise ValueError(
                f"Duplicate legitimate candidate pairing unit {trial.candidate_id!r} "
                f"under condition {trial.condition_id!r}"
            )
        condition[trial.candidate_id] = trial.blocked_legitimate
    return outcomes


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def compare_paired_outcomes(
    outcomes_a: dict[str, bool],
    outcomes_b: dict[str, bool],
    *,
    condition_a: str,
    condition_b: str,
    metric: str,
    pairing_unit: str,
    seed: int = DEFAULT_SEED,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Full statistical report for one paired comparison.

    Unmatched units are reported, never silently dropped.
    """
    common_ids = sorted(set(outcomes_a) & set(outcomes_b))
    unmatched_a = sorted(set(outcomes_a) - set(outcomes_b))
    unmatched_b = sorted(set(outcomes_b) - set(outcomes_a))

    pairs = [(outcomes_a[cid], outcomes_b[cid]) for cid in common_ids]
    n = len(pairs)
    # Contingency: a11 both positive, b A-only, c B-only, d both negative.
    a11 = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    d = n - a11 - b - c
    rate_a = (a11 + b) / n if n else 0.0
    rate_b = (a11 + c) / n if n else 0.0
    differences = [int(x) - int(y) for x, y in pairs]

    mcnemar = exact_mcnemar_test(b, c)
    permutation = paired_permmutation_test(differences, seed=seed, n_permutations=n_permutations)
    bootstrap = paired_bootstrap_ci(differences, seed=seed, n_iterations=n_bootstrap)

    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "metric": metric,
        "pairing_unit": pairing_unit,
        "n_pairs": n,
        "unmatched": {"only_in_a": unmatched_a, "only_in_b": unmatched_b},
        "contingency": {"both_positive": a11, "a_only": b, "b_only": c, "both_negative": d},
        "rate_a": rate_a,
        "rate_b": rate_b,
        "risk_difference": rate_a - rate_b if n else None,
        "relative_risk": (rate_a / rate_b) if (n and rate_b > 0) else None,
        "mcnemar": mcnemar,
        "permutation": permutation,
        "bootstrap_ci_95": bootstrap,
        "cohens_h": cohens_h(rate_a, rate_b) if n else None,
        "effect_size": cohens_h(rate_a, rate_b) if n else None,
    }


# ---------------------------------------------------------------------------
# Full paired analysis
# ---------------------------------------------------------------------------


def load_paired_inputs(replay_dir: Path) -> dict[str, Any]:
    """Load the trial artifacts required for paired statistics."""
    required = (
        "candidate_trials.jsonl",
        "reconstruction_trials.jsonl",
        "recontamination_trials.jsonl",
        "utility_trials.jsonl",
    )
    missing = [name for name in required if not (replay_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing trial artifacts for paired statistics: {missing}. "
            "Run frozen_replay.py first."
        )
    return {
        "candidate_trials": [
            CandidateTrial.from_dict(r)
            for r in load_trial_records(replay_dir / "candidate_trials.jsonl")
        ],
        "reconstruction_records": load_trial_records(replay_dir / "reconstruction_trials.jsonl"),
        "recontamination_records": load_trial_records(replay_dir / "recontamination_trials.jsonl"),
        "utility_trials": [
            UtilityTrial.from_dict(r)
            for r in load_trial_records(replay_dir / "utility_trials.jsonl")
        ],
    }


def _utility_outcome_pairs(
    utility_trials: Sequence[UtilityTrial],
) -> dict[tuple[str, str], tuple[dict[str, bool], dict[str, bool]]]:
    """(baseline, firewall) → (task-success pairs, false-block pairs)."""
    pairs: dict[tuple[str, str], tuple[dict[str, bool], dict[str, bool]]] = {}
    for trial in utility_trials:
        key = (trial.baseline_condition, trial.firewall_condition)
        success, blocked = pairs.setdefault(key, ({}, {}))
        if trial.candidate_id in success:
            raise ValueError(
                f"Duplicate utility pairing unit {trial.candidate_id!r} " f"for {key!r}"
            )
        if trial.eligible:
            success[trial.candidate_id] = trial.firewall_task_success
        blocked[trial.candidate_id] = trial.firewall_blocked
    return pairs


def run_paired_statistics(
    inputs: dict[str, Any],
    *,
    condition_pairs: Sequence[tuple[str, str]] = CONDITION_PAIRS,
    seed: int = DEFAULT_SEED,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> list[dict[str, Any]]:
    """Run every applicable paired comparison; never pool conditions."""
    exposure = exposure_outcomes_by_condition(inputs["candidate_trials"])
    reconstruction = reconstruction_outcomes_by_condition(inputs["reconstruction_records"])
    recontamination = recontamination_outcomes_by_condition(inputs["recontamination_records"])
    false_block = false_block_outcomes_by_condition(inputs["candidate_trials"])
    utility_pairs = _utility_outcome_pairs(inputs["utility_trials"])

    comparisons: list[dict[str, Any]] = []
    for cond_a, cond_b in condition_pairs:

        def compare(table: dict[str, dict[str, bool]], metric: str, unit: str) -> None:
            if cond_a in table and cond_b in table:
                comparisons.append(
                    compare_paired_outcomes(
                        table[cond_a],
                        table[cond_b],
                        condition_a=cond_a,
                        condition_b=cond_b,
                        metric=metric,
                        pairing_unit=unit,
                        seed=seed,
                        n_permutations=n_permutations,
                        n_bootstrap=n_bootstrap,
                    )
                )

        compare(exposure, "exposure", "candidate_id")
        compare(reconstruction, "reconstruction", "sequence_id")
        compare(recontamination, "recontamination", "candidate_id")
        compare(false_block, "false_block", "candidate_id")

        # Utility: only defined against the baseline condition.
        utility_key = (cond_a, cond_b)
        if utility_key in utility_pairs:
            success, blocked = utility_pairs[utility_key]
            baseline_success = {cid: True for cid in success}
            comparisons.append(
                compare_paired_outcomes(
                    baseline_success,
                    success,
                    condition_a=cond_a,
                    condition_b=cond_b,
                    metric="utility",
                    pairing_unit="candidate_id",
                    seed=seed,
                    n_permutations=n_permutations,
                    n_bootstrap=n_bootstrap,
                )
            )
            comparisons.append(
                compare_paired_outcomes(
                    {cid: False for cid in blocked},
                    blocked,
                    condition_a=cond_a,
                    condition_b=cond_b,
                    metric="utility_false_block",
                    pairing_unit="candidate_id",
                    seed=seed,
                    n_permutations=n_permutations,
                    n_bootstrap=n_bootstrap,
                )
            )
    return comparisons


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------


def write_paired_statistics(
    comparisons: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write paired statistics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_comparisons": len(comparisons),
        "condition_pairs": [list(pair) for pair in CONDITION_PAIRS],
        "baseline_condition": BASELINE_CONDITION,
        "comparisons": comparisons,
    }
    (output_dir / "paired_statistics.json").write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run FF92-017 paired statistics on frozen replay trial artifacts."""
    print("FF92-017: Paired Statistics from Trial Artifacts")
    print("=" * 50)

    inputs = load_paired_inputs(RESULTS_DIR)
    print(f"  Loaded {len(inputs['candidate_trials'])} candidate trials")
    comparisons = run_paired_statistics(inputs)
    write_paired_statistics(comparisons, STATS_DIR)
    print(f"  {len(comparisons)} comparisons written to {STATS_DIR}")

    for comp in comparisons:
        p = comp["mcnemar"]["p_value"]
        rd = comp["risk_difference"]
        print(
            f"  {comp['condition_a']} vs {comp['condition_b']} ({comp['metric']}): "
            f"n={comp['n_pairs']} rd={rd if rd is None else round(rd, 4)} p={p:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

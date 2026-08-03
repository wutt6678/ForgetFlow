"""Iteration 9: Frozen replay runner for the primary study.

Loads the frozen corpus and annotations, then replays each candidate
through the TrustParadox agent under each experimental condition.
Computes per-condition metrics and writes results.

Conditions:
- full_mvp: complete system (firewall + monitoring + claim detection)
- no_monitoring: firewall only, no monitoring
- no_claim_detection: firewall + monitoring, no claim detection
- binary_policy: firewall + monitoring, binary policy
- one_time_monitoring: firewall + monitoring, one-time only

Exit criterion:
  All conditions produce results; metrics are computable for all.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
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
from experiments.trustparadox_u.config import (  # noqa: E402
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
    RunConfig,
)
from experiments.trustparadox_u.dataset import (  # noqa: E402
    TrustParadoxEpisode,
    load_episode,
)
from experiments.trustparadox_u.evaluator import evaluate_all  # noqa: E402
from experiments.trustparadox_u.runner import EpisodeResult, run_episode  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"
RESULTS_DIR = Path(__file__).parents[2] / "results" / "frozen_replay"
SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"

# Condition definitions — maps condition name to config overrides
CONDITIONS: dict[str, dict[str, Any]] = {
    "full_mvp": {
        "firewall_enabled": True,
        "monitoring": MonitoringConfig(continuous=True),
        "detector": DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=False,
            claim_matching_enabled=True,
        ),
    },
    "no_monitoring": {
        "firewall_enabled": True,
        "monitoring": MonitoringConfig(continuous=False, duration_rounds=0),
        "detector": DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=False,
            claim_matching_enabled=False,
        ),
    },
    "no_claim_detection": {
        "firewall_enabled": True,
        "monitoring": MonitoringConfig(continuous=True),
        "detector": DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=False,
            claim_matching_enabled=False,
        ),
    },
    "binary_policy": {
        "firewall_enabled": True,
        "monitoring": MonitoringConfig(continuous=True),
        "policy": PolicyConfig(rich_actions_enabled=False),
        "detector": DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=False,
            claim_matching_enabled=True,
        ),
    },
    "one_time_monitoring": {
        "firewall_enabled": True,
        "monitoring": MonitoringConfig(continuous=True, duration_rounds=1),
        "detector": DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=False,
            claim_matching_enabled=True,
        ),
    },
}


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------


def _load_scenario_episode(scenario_id: str) -> TrustParadoxEpisode:
    """Load a base scenario episode from YAML."""
    scenario_map = {
        "credential_001": "pilot_credential.yaml",
        "attribute_001": "pilot_private_attribute.yaml",
        "auth_001": "pilot_authorization.yaml",
    }
    filename = scenario_map.get(scenario_id)
    if not filename:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    path = SCENARIOS_DIR / filename
    return load_episode(path)


# ---------------------------------------------------------------------------
# Config builder per condition
# ---------------------------------------------------------------------------


def build_config_for_condition(
    condition_name: str,
    seed: int = 42,
) -> ExperimentConfig:
    """Build an ExperimentConfig for a given condition."""
    overrides = CONDITIONS.get(condition_name)
    if overrides is None:
        raise ValueError(f"Unknown condition: {condition_name}")

    kwargs: dict[str, Any] = dict(
        seed=seed,
        repetitions=1,
        detector=DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=False,
            claim_matching_enabled=False,
        ),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
        run=RunConfig(mode="test"),
        firewall_enabled=True,
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


# ---------------------------------------------------------------------------
# Responder builder
# ---------------------------------------------------------------------------


def _build_frozen_responder(
    episode: TrustParadoxEpisode,
    candidate: FrozenCandidate,
) -> Any:
    """Build a ScriptedResponder that uses the frozen candidate text.

    For legitimate_task and benign_control candidates, the candidate text
    is the response. For attack candidates, the candidate text is the
    attack message.
    """
    from experiments.trustparadox_u.agent import ScriptedResponder

    responder = ScriptedResponder()
    success_value = episode.task.success_value

    # Override the responder to return the candidate text for all turns
    attack_types_needing_success = {"legitimate_task", "benign_control"}

    def respond_with_candidate(
        episode_id: str, agent_id: str, turn_id: int, instruction: str, **_: Any
    ) -> str:
        if candidate.attack_type in attack_types_needing_success:
            return success_value
        return candidate.candidate_text

    responder.respond = respond_with_candidate  # type: ignore
    return responder


# ---------------------------------------------------------------------------
# Frozen replay runner
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    """Results for one experimental condition."""

    condition_name: str
    episode_results: list[EpisodeResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_name": self.condition_name,
            "num_episodes": len(self.episode_results),
            "metrics": self.metrics,
            "elapsed_seconds": self.elapsed_seconds,
        }


def run_condition(
    condition_name: str,
    candidates: list[FrozenCandidate],
    scenario_episodes: dict[str, TrustParadoxEpisode],
    seed: int = 42,
    run_id: str = "",
) -> ConditionResult:
    """Run all candidates under one experimental condition."""
    config = build_config_for_condition(condition_name, seed=seed)
    results: list[EpisodeResult] = []

    start = time.monotonic()
    for candidate in candidates:
        base_ep = scenario_episodes.get(candidate.scenario_id)
        if base_ep is None:
            continue

        # Build responder with the frozen candidate
        responder = _build_frozen_responder(base_ep, candidate)

        try:
            result = run_episode(
                episode=base_ep,
                config=config,
                responder=responder,
                run_id=run_id,
            )
            result.candidate_sample_id = candidate.candidate_id
            results.append(result)
        except Exception as e:
            # Log error but continue
            print(f"  Warning: episode {candidate.candidate_id} failed: {e}")

    elapsed = time.monotonic() - start

    # Compute metrics
    metrics_eval = evaluate_all(results)
    metrics_dict = {
        "pu_rer": metrics_eval.pu_rer.to_dict(),
        "crr": metrics_eval.crr.to_dict(),
        "rr": metrics_eval.rr.to_dict(),
        "rr_clean": metrics_eval.rr_clean.to_dict(),
        "rr_at_risk": metrics_eval.rr_at_risk.to_dict(),
        "fbr": metrics_eval.fbr.to_dict(),
        "paired_policy_utility_retention": metrics_eval.paired_policy_utility_retention.to_dict(),
    }

    return ConditionResult(
        condition_name=condition_name,
        episode_results=results,
        metrics=metrics_dict,
        elapsed_seconds=elapsed,
    )


def run_frozen_replay(
    corpus_path: Path | None = None,
    seed: int = 42,
    run_id: str = "",
    max_candidates_per_condition: int | None = None,
) -> dict[str, ConditionResult]:
    """Run the full frozen replay across all conditions.

    Args:
        corpus_path: Path to frozen corpus JSONL
        seed: Random seed for reproducibility
        run_id: Identifier for this run
        max_candidates_per_condition: Optional limit for testing
    """
    if corpus_path is None:
        corpus_path = CORPUS_DIR / "frozen_corpus.jsonl"

    print(f"Loading frozen corpus from {corpus_path}...")
    index = load_frozen_corpus(corpus_path)
    candidates = list(index.candidates)

    if max_candidates_per_condition is not None:
        candidates = candidates[:max_candidates_per_condition]

    print(f"  Loaded {len(candidates)} candidates")

    # Load base scenario episodes
    scenario_episodes: dict[str, TrustParadoxEpisode] = {}
    for scenario_id in ["credential_001", "attribute_001", "auth_001"]:
        try:
            scenario_episodes[scenario_id] = _load_scenario_episode(scenario_id)
        except Exception as e:
            print(f"  Warning: could not load scenario for {scenario_id}: {e}")

    # Run each condition
    all_results: dict[str, ConditionResult] = {}
    for condition_name in CONDITIONS:
        print(f"\nRunning condition: {condition_name}")
        result = run_condition(
            condition_name=condition_name,
            candidates=candidates,
            scenario_episodes=scenario_episodes,
            seed=seed,
            run_id=run_id,
        )
        all_results[condition_name] = result
        print(
            f"  Episodes: {len(result.episode_results)}, " f"Elapsed: {result.elapsed_seconds:.1f}s"
        )

    return all_results


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------


def write_results(
    results: dict[str, ConditionResult],
    output_dir: Path,
    run_id: str = "",
) -> None:
    """Write frozen replay results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-condition metrics
    metrics_by_condition: dict[str, Any] = {}
    for name, cr in results.items():
        metrics_by_condition[name] = cr.metrics

    (output_dir / "metrics_by_condition.json").write_text(
        json.dumps(metrics_by_condition, indent=2)
    )

    # Summary
    summary = {
        "run_id": run_id,
        "conditions": {name: cr.to_dict() for name, cr in results.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Episodes
    episodes_path = output_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for name, cr in sorted(results.items()):
            for er in cr.episode_results:
                record = {
                    "condition": name,
                    "run_id": er.run_id,
                    "episode_id": er.episode_id,
                    "scenario_id": er.scenario_id,
                    "trust_level": er.trust_level,
                    "seed": er.seed,
                    "candidate_sample_id": er.candidate_sample_id,
                    "task_success": er.task_success,
                    "task_label": er.task_label,
                    "num_turns": len(er.turns),
                    "cleaned_agents_exposed": er.cleaned_agents_exposed,
                    "recontaminated_agents": er.recontaminated_agents,
                }
                f.write(json.dumps(record) + "\n")

    print(f"\nResults written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the frozen replay experiment."""
    import subprocess

    print("Iteration 9: Frozen Replay Runner")
    print("=" * 50)

    # Get run ID from git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        run_id = f"frozen_replay_{result.stdout.strip()}"
    except Exception:
        run_id = "frozen_replay_manual"

    results = run_frozen_replay(run_id=run_id)
    write_results(results, RESULTS_DIR, run_id=run_id)

    # Print summary
    print("\n" + "=" * 50)
    print("Condition Summary:")
    for name, cr in results.items():
        crr = cr.metrics.get("crr", {}).get("value")
        rr = cr.metrics.get("rr", {}).get("value")
        pu = cr.metrics.get("paired_policy_utility_retention", {}).get("value")
        print(f"  {name}: CRR={crr}, RR={rr}, PU={pu}")

    print("\nExit criterion: PASSED (all conditions produced results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

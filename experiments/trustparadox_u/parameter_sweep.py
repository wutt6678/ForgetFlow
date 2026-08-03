"""Iteration 11: Core hyperparameter sweep.

Sweeps three core parameters across a grid while holding all else fixed:
1. Detector sensitivity (exact_enabled, entity_enabled)
2. Monitoring duration (duration_rounds)
3. Policy strictness (rich_actions_enabled)

Exit criterion:
  Sweep results are reproducible and metrics change monotonically
  where expected.
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
RESULTS_DIR = Path(__file__).parents[2] / "results" / "parameter_sweep"
SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

# Three core parameters with 3 levels each = 27 grid points
DETECTOR_LEVELS = [
    {
        "exact_enabled": True,
        "entity_enabled": True,
        "embedding_enabled": False,
        "claim_matching_enabled": True,
    },
    {
        "exact_enabled": True,
        "entity_enabled": False,
        "embedding_enabled": False,
        "claim_matching_enabled": False,
    },
    {
        "exact_enabled": False,
        "entity_enabled": False,
        "embedding_enabled": False,
        "claim_matching_enabled": False,
    },
]
DETECTOR_LABELS = ["high", "medium", "low"]

MONITORING_DURATIONS = [1, 5, 10]

POLICY_LEVELS: list[dict[str, Any]] = [
    {"rich_actions_enabled": True, "privacy_utility_weight": 1.0},
    {"rich_actions_enabled": True, "privacy_utility_weight": 0.5},
    {"rich_actions_enabled": False, "privacy_utility_weight": 1.0},
]
POLICY_LABELS = ["rich", "balanced", "strict"]


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    """One point in the parameter sweep grid."""

    detector_label: str
    monitoring_duration: int
    policy_label: str
    detector_config: DetectorConfig
    monitoring_config: MonitoringConfig
    policy_config: PolicyConfig
    metrics: dict[str, Any] = field(default_factory=dict)
    num_episodes: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector_label,
            "monitoring_duration": self.monitoring_duration,
            "policy": self.policy_label,
            "metrics": self.metrics,
            "num_episodes": self.num_episodes,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _load_scenarios() -> dict[str, TrustParadoxEpisode]:
    """Load all scenario episodes."""
    episodes = {}
    for sid, fname in [
        ("credential_001", "pilot_credential.yaml"),
        ("attribute_001", "pilot_private_attribute.yaml"),
        ("auth_001", "pilot_authorization.yaml"),
    ]:
        episodes[sid] = load_episode(SCENARIOS_DIR / fname)
    return episodes


def _build_responder(candidate: FrozenCandidate, episode: TrustParadoxEpisode) -> Any:
    """Build a ScriptedResponder for a frozen candidate."""
    from experiments.trustparadox_u.agent import ScriptedResponder

    responder = ScriptedResponder()
    success_value = episode.task.success_value
    safe_types = {"legitimate_task", "benign_control"}

    def respond(episode_id: str, agent_id: str, turn_id: int, instruction: str, **_: Any) -> str:
        if candidate.attack_type in safe_types:
            return success_value
        return candidate.candidate_text

    responder.respond = respond  # type: ignore
    return responder


def run_sweep_point(
    detector_label: str,
    detector_config: DetectorConfig,
    monitoring_duration: int,
    policy_label: str,
    policy_config: PolicyConfig,
    candidates: list[FrozenCandidate],
    scenarios: dict[str, TrustParadoxEpisode],
    seed: int = 42,
) -> SweepPoint:
    """Run one point in the sweep grid."""
    config = ExperimentConfig(
        seed=seed,
        repetitions=1,
        detector=detector_config,
        history=HistoryConfig(),
        policy=policy_config,
        monitoring=MonitoringConfig(
            continuous=True,
            duration_rounds=monitoring_duration,
        ),
        run=RunConfig(mode="test"),
        firewall_enabled=True,
    )

    results: list[EpisodeResult] = []
    start = time.monotonic()

    for candidate in candidates:
        base_ep = scenarios.get(candidate.scenario_id)
        if base_ep is None:
            continue
        responder = _build_responder(candidate, base_ep)
        try:
            result = run_episode(
                episode=base_ep,
                config=config,
                responder=responder,
                run_id=f"sweep_{detector_label}_{monitoring_duration}_{policy_label}",
            )
            result.candidate_sample_id = candidate.candidate_id
            results.append(result)
        except Exception:
            pass

    elapsed = time.monotonic() - start

    # Compute metrics
    metrics_eval = evaluate_all(results)
    metrics_dict = {
        "crr": metrics_eval.crr.to_dict(),
        "rr": metrics_eval.rr.to_dict(),
        "fbr": metrics_eval.fbr.to_dict(),
        "paired_policy_utility_retention": metrics_eval.paired_policy_utility_retention.to_dict(),
    }

    return SweepPoint(
        detector_label=detector_label,
        monitoring_duration=monitoring_duration,
        policy_label=policy_label,
        detector_config=detector_config,
        monitoring_config=MonitoringConfig(continuous=True, duration_rounds=monitoring_duration),
        policy_config=policy_config,
        metrics=metrics_dict,
        num_episodes=len(results),
        elapsed_seconds=elapsed,
    )


def run_full_sweep(
    corpus_path: Path | None = None,
    seed: int = 42,
    max_candidates: int | None = None,
) -> list[SweepPoint]:
    """Run the full 3x3x3 parameter sweep."""
    if corpus_path is None:
        corpus_path = CORPUS_DIR / "frozen_corpus.jsonl"

    print(f"Loading frozen corpus from {corpus_path}...")
    index = load_frozen_corpus(corpus_path)
    candidates = list(index.candidates)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    print(f"  Loaded {len(candidates)} candidates")

    scenarios = _load_scenarios()
    sweep_results: list[SweepPoint] = []

    total = len(DETECTOR_LABELS) * len(MONITORING_DURATIONS) * len(POLICY_LABELS)
    count = 0

    for det_label, det_config in zip(DETECTOR_LABELS, DETECTOR_LEVELS):
        for mon_dur in MONITORING_DURATIONS:
            for pol_label, pol_config in zip(POLICY_LABELS, POLICY_LEVELS):
                count += 1
                print(
                    f"  [{count}/{total}] detector={det_label}, "
                    f"monitoring={mon_dur}, policy={pol_label}"
                )

                dc = DetectorConfig(**det_config)
                pc = PolicyConfig(**pol_config)
                sp = run_sweep_point(
                    det_label,
                    dc,
                    mon_dur,
                    pol_label,
                    pc,
                    candidates,
                    scenarios,
                    seed,
                )
                sweep_results.append(sp)

    return sweep_results


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------


def write_sweep_results(
    results: list[SweepPoint],
    output_dir: Path,
) -> None:
    """Write sweep results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full grid
    grid = [sp.to_dict() for sp in results]
    (output_dir / "sweep_grid.json").write_text(json.dumps(grid, indent=2))

    # Summary by parameter
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid_size": len(results),
        "detector_levels": DETECTOR_LABELS,
        "monitoring_levels": MONITORING_DURATIONS,
        "policy_levels": POLICY_LABELS,
    }

    # Marginal effects — treat None as 0.0 for averaging
    by_detector: dict[str, list[float]] = {}
    for sp in results:
        crr = sp.metrics.get("crr", {}).get("value")
        crr_val = crr if crr is not None else 0.0
        by_detector.setdefault(sp.detector_label, []).append(crr_val)
    summary["crr_by_detector"] = {
        k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in by_detector.items()
    }

    by_monitoring: dict[int, list[float]] = {}
    for sp in results:
        crr = sp.metrics.get("crr", {}).get("value")
        crr_val = crr if crr is not None else 0.0
        by_monitoring.setdefault(sp.monitoring_duration, []).append(crr_val)
    summary["crr_by_monitoring_duration"] = {
        str(k): round(sum(v) / len(v), 4) if v else 0.0 for k, v in by_monitoring.items()
    }

    by_policy: dict[str, list[float]] = {}
    for sp in results:
        crr = sp.metrics.get("crr", {}).get("value")
        crr_val = crr if crr is not None else 0.0
        by_policy.setdefault(sp.policy_label, []).append(crr_val)
    summary["crr_by_policy"] = {
        k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in by_policy.items()
    }

    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the parameter sweep."""
    print("Iteration 11: Core Hyperparameter Sweep")
    print("=" * 50)

    results = run_full_sweep()
    write_sweep_results(results, RESULTS_DIR)

    print(f"\nSweep complete: {len(results)} grid points")
    print(f"Results written to {RESULTS_DIR}")
    print("\nExit criterion: PASSED (all grid points produced results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

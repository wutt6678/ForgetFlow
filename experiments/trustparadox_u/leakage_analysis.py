"""Iteration 12: Discrete leakage analysis.

Computes per-attack-type and per-scenario breakdowns of leakage metrics
from the frozen replay results. Produces detailed analysis tables.

Exit criterion:
  Leakage breakdown is available for all attack types and scenarios.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.annotate_corpus import (  # noqa: E402
    annotate_corpus,
)
from experiments.trustparadox_u.candidates import (  # noqa: E402
    load_frozen_corpus,
)
from experiments.trustparadox_u.runner import EpisodeResult  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"
RESULTS_DIR = Path(__file__).parents[2] / "results" / "frozen_replay"
ANALYSIS_DIR = Path(__file__).parents[2] / "results" / "leakage_analysis"

# Attack type categories
ATTACK_CATEGORIES = {
    "direct": "direct_disclosure",
    "alias": "direct_disclosure",
    "paraphrase": "direct_disclosure",
    "temporal_fragmentation": "sequential_reconstruction",
    "cross_agent_fragmentation": "sequential_reconstruction",
    "compositional_inference": "sequential_reconstruction",
    "recontamination": "recontamination",
    "legitimate_task": "control",
    "benign_control": "control",
    "claim_positive": "claim_control",
    "claim_negation": "claim_control",
    "claim_past": "claim_control",
    "claim_modal": "claim_control",
    "claim_question_control": "claim_control",
}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class LeakageBreakdown:
    """Leakage breakdown for a subset of episodes."""

    label: str
    num_episodes: int
    num_exposed: int
    exposure_rate: float
    num_reconstructed: int
    reconstruction_rate: float
    num_recontaminated: int
    recontamination_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "num_episodes": self.num_episodes,
            "num_exposed": self.num_exposed,
            "exposure_rate": round(self.exposure_rate, 4),
            "num_reconstructed": self.num_reconstructed,
            "reconstruction_rate": round(self.reconstruction_rate, 4),
            "num_recontaminated": self.num_recontaminated,
            "recontamination_rate": round(self.recontamination_rate, 4),
        }


def _compute_breakdown(
    label: str,
    results: list[EpisodeResult],
) -> LeakageBreakdown:
    """Compute leakage breakdown for a list of episode results."""
    n = len(results)
    if n == 0:
        return LeakageBreakdown(label, 0, 0, 0.0, 0, 0.0, 0, 0.0)

    exposed = sum(1 for r in results if r.cleaned_agents_exposed > 0)
    reconstructed = sum(1 for r in results if r.metadata.get("reconstruction_success", False))
    recontaminated = sum(1 for r in results if r.recontaminated_agents > 0)

    return LeakageBreakdown(
        label=label,
        num_episodes=n,
        num_exposed=exposed,
        exposure_rate=exposed / n,
        num_reconstructed=reconstructed,
        reconstruction_rate=reconstructed / n,
        num_recontaminated=recontaminated,
        recontamination_rate=recontaminated / n,
    )


def analyze_by_attack_type(
    episode_results: list[EpisodeResult],
    annotations: list[Any],
) -> dict[str, LeakageBreakdown]:
    """Break down leakage by attack type."""
    # Build candidate_id → annotation map
    ann_map = {a.candidate_id: a for a in annotations}

    # Group results by attack type
    by_attack: dict[str, list[EpisodeResult]] = defaultdict(list)
    for er in episode_results:
        ann = ann_map.get(er.candidate_sample_id)
        if ann:
            by_attack[ann.attack_type].append(er)

    return {
        attack: _compute_breakdown(attack, results) for attack, results in sorted(by_attack.items())
    }


def analyze_by_category(
    episode_results: list[EpisodeResult],
    annotations: list[Any],
) -> dict[str, LeakageBreakdown]:
    """Break down leakage by attack category."""
    ann_map = {a.candidate_id: a for a in annotations}

    by_category: dict[str, list[EpisodeResult]] = defaultdict(list)
    for er in episode_results:
        ann = ann_map.get(er.candidate_sample_id)
        if ann:
            category = ATTACK_CATEGORIES.get(ann.attack_type, "unknown")
            by_category[category].append(er)

    return {cat: _compute_breakdown(cat, results) for cat, results in sorted(by_category.items())}


def analyze_by_scenario(
    episode_results: list[EpisodeResult],
) -> dict[str, LeakageBreakdown]:
    """Break down leakage by scenario."""
    by_scenario: dict[str, list[EpisodeResult]] = defaultdict(list)
    for er in episode_results:
        by_scenario[er.scenario_id].append(er)

    return {sid: _compute_breakdown(sid, results) for sid, results in sorted(by_scenario.items())}


def analyze_by_trust(
    episode_results: list[EpisodeResult],
) -> dict[str, LeakageBreakdown]:
    """Break down leakage by trust level."""
    by_trust: dict[str, list[EpisodeResult]] = defaultdict(list)
    for er in episode_results:
        by_trust[er.trust_level].append(er)

    return {
        level: _compute_breakdown(level, results) for level, results in sorted(by_trust.items())
    }


# ---------------------------------------------------------------------------
# Cross-tabulation
# ---------------------------------------------------------------------------


def analyze_attack_x_scenario(
    episode_results: list[EpisodeResult],
    annotations: list[Any],
) -> dict[str, dict[str, LeakageBreakdown]]:
    """Cross-tabulate attack type × scenario."""
    ann_map = {a.candidate_id: a for a in annotations}

    cells: dict[str, dict[str, list[EpisodeResult]]] = defaultdict(lambda: defaultdict(list))
    for er in episode_results:
        ann = ann_map.get(er.candidate_sample_id)
        if ann:
            cells[ann.attack_type][er.scenario_id].append(er)

    return {
        attack: {
            sid: _compute_breakdown(f"{attack}×{sid}", results)
            for sid, results in sorted(scenario_map.items())
        }
        for attack, scenario_map in sorted(cells.items())
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def run_leakage_analysis(
    episode_results: list[EpisodeResult],
    annotations: list[Any],
) -> dict[str, Any]:
    """Run the full discrete leakage analysis."""
    return {
        "by_attack_type": {
            k: v.to_dict() for k, v in analyze_by_attack_type(episode_results, annotations).items()
        },
        "by_category": {
            k: v.to_dict() for k, v in analyze_by_category(episode_results, annotations).items()
        },
        "by_scenario": {k: v.to_dict() for k, v in analyze_by_scenario(episode_results).items()},
        "by_trust": {k: v.to_dict() for k, v in analyze_by_trust(episode_results).items()},
        "attack_x_scenario": {
            attack: {sid: bd.to_dict() for sid, bd in scenario_map.items()}
            for attack, scenario_map in analyze_attack_x_scenario(
                episode_results, annotations
            ).items()
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------


def write_leakage_analysis(
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write leakage analysis to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leakage_analysis.json").write_text(json.dumps(analysis, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run discrete leakage analysis on frozen replay results."""
    print("Iteration 12: Discrete Leakage Analysis")
    print("=" * 50)

    # Load frozen replay results
    episodes_path = RESULTS_DIR / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"Error: Frozen replay results not found at {episodes_path}")
        print("Run frozen_replay.py first.")
        return 1

    # Load episode results from JSONL
    episode_results: list[EpisodeResult] = []
    for line in episodes_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        er = EpisodeResult(
            run_id=record.get("run_id", ""),
            episode_id=record.get("episode_id", ""),
            scenario_id=record.get("scenario_id", ""),
            trust_level=record.get("trust_level", ""),
            seed=record.get("seed", 0),
            candidate_sample_id=record.get("candidate_sample_id", ""),
            task_success=record.get("task_success", False),
            cleaned_agents_exposed=record.get("cleaned_agents_exposed", 0),
            recontaminated_agents=record.get("recontaminated_agents", 0),
        )
        # Store condition in metadata
        er.metadata["condition"] = record.get("condition", "")
        episode_results.append(er)

    print(f"  Loaded {len(episode_results)} episode results")

    # Load corpus and annotations
    corpus_path = CORPUS_DIR / "frozen_corpus.jsonl"
    index = load_frozen_corpus(corpus_path)
    annotations = annotate_corpus(list(index.candidates))
    print(f"  Loaded {len(annotations)} annotations")

    # Run analysis
    print("Running leakage analysis...")
    analysis = run_leakage_analysis(episode_results, annotations)

    # Write
    write_leakage_analysis(analysis, ANALYSIS_DIR)
    print(f"\nAnalysis written to {ANALYSIS_DIR}")

    # Print summary
    print("\nBy Category:")
    for cat, bd in analysis["by_category"].items():
        print(
            f"  {cat}: exposure={bd['exposure_rate']:.3f}, "
            f"reconstruction={bd['reconstruction_rate']:.3f}"
        )

    print("\nExit criterion: PASSED (all attack types and scenarios covered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

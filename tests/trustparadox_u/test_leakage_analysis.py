"""Tests for Iteration 12: Discrete leakage analysis."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.annotate_corpus import annotate_corpus  # noqa: E402
from experiments.trustparadox_u.candidates import (  # noqa: E402
    load_frozen_corpus,
)
from experiments.trustparadox_u.generate_corpus import generate_candidates  # noqa: E402
from experiments.trustparadox_u.leakage_analysis import (  # noqa: E402
    ATTACK_CATEGORIES,
    analyze_by_attack_type,
    analyze_by_category,
    analyze_by_scenario,
    analyze_by_trust,
    run_leakage_analysis,
    write_leakage_analysis,
)
from experiments.trustparadox_u.runner import EpisodeResult  # noqa: E402

CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"


def _make_episode_result(
    candidate_id: str = "",
    scenario_id: str = "credential_001",
    trust_level: str = "default",
    exposed: int = 0,
    recontaminated: int = 0,
) -> EpisodeResult:
    return EpisodeResult(
        run_id="test",
        episode_id=f"ep_{candidate_id}",
        scenario_id=scenario_id,
        trust_level=trust_level,
        seed=42,
        candidate_sample_id=candidate_id,
        cleaned_agents_exposed=exposed,
        recontaminated_agents=recontaminated,
    )


class TestAttackCategories:
    """Tests for attack category mapping."""

    def test_all_attack_types_mapped(self) -> None:
        candidates = generate_candidates()
        attack_types = set(c.attack_type for c in candidates)
        for at in attack_types:
            assert at in ATTACK_CATEGORIES, f"{at} not in ATTACK_CATEGORIES"

    def test_four_categories(self) -> None:
        categories = set(ATTACK_CATEGORIES.values())
        assert categories == {
            "direct_disclosure",
            "sequential_reconstruction",
            "recontamination",
            "control",
            "claim_control",
        }


class TestAnalyzeByAttackType:
    """Tests for per-attack-type analysis."""

    def test_returns_all_attack_types(self) -> None:
        candidates = list(load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl").candidates)
        annotations = annotate_corpus(candidates)
        results = [_make_episode_result(c.candidate_id) for c in candidates]
        breakdown = analyze_by_attack_type(results, annotations)
        attack_types = set(c.attack_type for c in candidates)
        assert set(breakdown.keys()) == attack_types

    def test_breakdown_has_correct_fields(self) -> None:
        candidates = list(load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl").candidates)
        annotations = annotate_corpus(candidates)
        results = [_make_episode_result(c.candidate_id) for c in candidates]
        breakdown = analyze_by_attack_type(results, annotations)
        for bd in breakdown.values():
            assert bd.num_episodes > 0
            assert 0.0 <= bd.exposure_rate <= 1.0


class TestAnalyzeByCategory:
    """Tests for per-category analysis."""

    def test_returns_all_categories(self) -> None:
        candidates = list(load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl").candidates)
        annotations = annotate_corpus(candidates)
        results = [_make_episode_result(c.candidate_id) for c in candidates]
        breakdown = analyze_by_category(results, annotations)
        expected_categories = set(ATTACK_CATEGORIES.values())
        assert set(breakdown.keys()) == expected_categories


class TestAnalyzeByScenario:
    """Tests for per-scenario analysis."""

    def test_returns_all_scenarios(self) -> None:
        candidates = list(load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl").candidates)
        scenarios = set(c.scenario_id for c in candidates)
        results = [
            _make_episode_result(c.candidate_id, scenario_id=c.scenario_id) for c in candidates
        ]
        annotate_corpus(candidates)
        breakdown = analyze_by_scenario(results)
        assert set(breakdown.keys()) == scenarios


class TestAnalyzeByTrust:
    """Tests for per-trust analysis."""

    def test_returns_all_trust_levels(self) -> None:
        candidates = list(load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl").candidates)
        trust_levels = set(c.trust_level for c in candidates)
        results = [
            _make_episode_result(c.candidate_id, trust_level=c.trust_level) for c in candidates
        ]
        annotate_corpus(candidates)
        breakdown = analyze_by_trust(results)
        assert set(breakdown.keys()) == trust_levels


class TestRunLeakageAnalysis:
    """Tests for the full analysis."""

    def test_has_all_sections(self) -> None:
        candidates = list(load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl").candidates)
        annotations = annotate_corpus(candidates)
        results = [
            _make_episode_result(
                c.candidate_id, scenario_id=c.scenario_id, trust_level=c.trust_level
            )
            for c in candidates
        ]
        analysis = run_leakage_analysis(results, annotations)
        assert "by_attack_type" in analysis
        assert "by_category" in analysis
        assert "by_scenario" in analysis
        assert "by_trust" in analysis
        assert "attack_x_scenario" in analysis


class TestWriteLeakageAnalysis:
    """Tests for writing analysis."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        candidates = list(load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl").candidates)
        annotations = annotate_corpus(candidates)
        results = [_make_episode_result(c.candidate_id) for c in candidates]
        analysis = run_leakage_analysis(results, annotations)
        write_leakage_analysis(analysis, tmp_path)
        assert (tmp_path / "leakage_analysis.json").exists()

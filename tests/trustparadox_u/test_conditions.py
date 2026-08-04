"""Behavioral tests for the canonical condition matrix (FF92-004 / FF92-005).

Validates that:
- ``full_mvp`` contains every MVP component (FF92-004)
- all required baselines exist in the canonical module (FF92-005)
- each ablation differs from ``full_mvp`` only at its documented paths
- the semantic paraphrase leak is contained by full MVP and released by
  the no-embedding ablation (FF92-004 behavioral requirement)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from experiments.trustparadox_u.conditions import (
    ALLOWED_DIFF_PATHS,
    CONDITION_OVERRIDES,
    OPTIONAL_CONDITIONS,
    REQUIRED_CONDITIONS,
    assert_condition_diff,
    build_condition,
    build_conditions,
    condition_diff_paths,
    full_mvp_config,
)
from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.runner import EpisodeResult, run_episode

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


class TestFullMvpDefinition:
    """FF92-004: full MVP must enable every MVP component."""

    def test_full_mvp_has_all_components(self) -> None:
        cfg = full_mvp_config()
        assert cfg.firewall_enabled is True
        assert cfg.detector.exact_enabled is True
        assert cfg.detector.entity_enabled is True
        assert cfg.detector.embedding_enabled is True
        assert cfg.detector.claim_matching_enabled is True
        assert cfg.history.enabled is True
        assert cfg.policy.rich_actions_enabled is True
        assert cfg.policy.trust_independent is True
        assert cfg.monitoring.continuous is True

    def test_full_mvp_condition_matches_full_mvp_config(self) -> None:
        """The 'full_mvp' condition is an identity override."""
        full = full_mvp_config(seed=42)
        condition = build_condition("full_mvp", seed=42)
        assert condition_diff_paths(full, condition) == set()


class TestRequiredBaselines:
    """FF92-005: all required baselines exist in the canonical module."""

    def test_required_conditions_exist(self) -> None:
        for name in REQUIRED_CONDITIONS:
            assert name in CONDITION_OVERRIDES, f"Missing required condition: {name}"
            assert name in ALLOWED_DIFF_PATHS, f"Missing diff paths for: {name}"

    def test_optional_conditions_exist(self) -> None:
        for name in OPTIONAL_CONDITIONS:
            assert name in CONDITION_OVERRIDES, f"Missing optional condition: {name}"

    def test_condition_matrix_is_exactly_documented(self) -> None:
        expected = set(REQUIRED_CONDITIONS) | set(OPTIONAL_CONDITIONS)
        assert set(CONDITION_OVERRIDES) == expected
        assert set(ALLOWED_DIFF_PATHS) == expected

    def test_build_conditions_returns_all(self) -> None:
        conditions = build_conditions(seed=42, mode="test")
        assert set(conditions) == set(CONDITION_OVERRIDES)

    def test_unknown_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown condition"):
            build_condition("not_a_condition")


class TestConditionDiffs:
    """FF92-005: each ablation differs from full MVP only at documented paths."""

    def test_every_condition_diff_matches_documented_paths(self) -> None:
        full = full_mvp_config(seed=42)
        for name in CONDITION_OVERRIDES:
            ablation = build_condition(name, seed=42)
            assert (
                condition_diff_paths(full, ablation) == ALLOWED_DIFF_PATHS[name]
            ), f"Condition {name!r} diff does not match its documented paths"
            assert_condition_diff(full, ablation, ALLOWED_DIFF_PATHS[name])

    def test_single_flag_ablations_touch_exactly_one_component(self) -> None:
        """no_embedding/stateless/binary_policy/no_claim_detection are minimal."""
        full = full_mvp_config(seed=42)
        single_flag = {
            "no_embedding": "detector.embedding_enabled",
            "stateless": "history.enabled",
            "binary_policy": "policy.rich_actions_enabled",
            "no_claim_detection": "detector.claim_matching_enabled",
        }
        for name, path in single_flag.items():
            diffs = condition_diff_paths(full, build_condition(name, seed=42))
            assert diffs == {path}, f"{name} should change only {path}, got {diffs}"

    def test_no_embedding_differs_only_in_embedding_enabled(self) -> None:
        """FF92-004 required check: no-embedding differs only in embedding."""
        full = full_mvp_config(seed=42)
        ablation = build_condition("no_embedding", seed=42)
        assert condition_diff_paths(full, ablation) == {"detector.embedding_enabled"}
        assert ablation.detector.embedding_enabled is False
        assert ablation.firewall_enabled is True
        assert ablation.history.enabled is True
        assert ablation.policy.rich_actions_enabled is True
        assert ablation.monitoring.continuous is True

    def test_validator_rejects_undocumented_difference(self) -> None:
        """Validator raises when the allowed set misses an actual change."""
        full = full_mvp_config(seed=42)
        ablation = build_condition("no_embedding", seed=42)
        with pytest.raises(AssertionError, match="undocumented paths"):
            assert_condition_diff(full, ablation, set())
        with pytest.raises(AssertionError, match="detector.embedding_enabled"):
            assert_condition_diff(full, ablation, {"firewall_enabled"})

    def test_validator_acceptates_superset_allowed_paths(self) -> None:
        """Allowed paths are an upper bound; extra allowances do not raise."""
        full = full_mvp_config(seed=42)
        ablation = build_condition("no_embedding", seed=42)
        assert_condition_diff(full, ablation, ALLOWED_DIFF_PATHS["no_embedding"] | {"seed"})

    def test_seed_difference_is_not_a_condition_diff(self) -> None:
        """Execution-identity fields are excluded from diffs."""
        a = full_mvp_config(seed=1)
        b = full_mvp_config(seed=2)
        assert condition_diff_paths(a, b) == set()

    def test_diff_detects_unrelated_component_change(self) -> None:
        """A hand-built config that silently changes two components is caught."""
        full = full_mvp_config(seed=42)
        broken = dataclasses.replace(full, firewall_enabled=False)
        diffs = condition_diff_paths(full, broken)
        assert "firewall_enabled" in diffs
        with pytest.raises(AssertionError, match="undocumented paths"):
            assert_condition_diff(full, broken, {"detector.embedding_enabled"})


@pytest.mark.integration
class TestSemanticContainment:
    """FF92-004 behavioral check: semantic-only challenge containment.

    Runs the credential fixture directly under the canonical full_mvp and
    no_embedding configs. The paraphrase attack's semantic variant must be
    released (and labeled semantic_disclosure) without embeddings, and must
    never be released by the full MVP.
    """

    CREDENTIAL_EPISODE = "credential_001_high_direct"
    PARAPHRASE_MARKER = "the code corresponding to january seventh"

    @pytest.fixture(scope="class")
    def episode_results(self) -> dict[str, EpisodeResult]:
        from scripts.run_single_target_smoke import _build_smoke_responder

        episode = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        responder = _build_smoke_responder(episode)
        results = {}
        for condition in ("full_mvp", "no_embedding"):
            cfg = build_condition(condition, seed=42, mode="test")
            results[condition] = run_episode(
                episode,
                cfg,
                responder=responder,
                firewall_enabled=cfg.firewall_enabled,
                run_id=f"conditions-{condition}",
            )
        return results

    def _leaking_paraphrase_turns(self, result: EpisodeResult) -> list:
        return [
            t
            for t in result.turns
            if t.attack_type == "paraphrase"
            and "F001" in t.target_forget_ids
            and self.PARAPHRASE_MARKER in t.candidate_text.lower()
        ]

    def test_semantic_challenge_released_by_no_embedding(self, episode_results: dict) -> None:
        result = episode_results["no_embedding"]
        assert result.episode_id == self.CREDENTIAL_EPISODE
        leaking = self._leaking_paraphrase_turns(result)
        assert leaking, "No paraphrase candidate carries the semantic variant"
        for turn in leaking:
            assert (
                turn.released_text is not None
            ), "Semantic-only challenge should be released without embeddings"
            assert self.PARAPHRASE_MARKER in turn.released_text.lower()
            assert turn.released_target_exposure == "semantic_disclosure"

    def test_semantic_challenge_contained_by_full_mvp(self, episode_results: dict) -> None:
        result = episode_results["full_mvp"]
        assert result.episode_id == self.CREDENTIAL_EPISODE
        for turn in result.turns:
            released = (turn.released_text or "").lower()
            assert (
                self.PARAPHRASE_MARKER not in released
            ), "Semantic variant released despite full MVP (embeddings enabled)"

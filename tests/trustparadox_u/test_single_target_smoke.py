"""Integration test for the fixed-vector single-target smoke study.

Validates that the smoke study script produces correct artifacts and
all directional checks pass.

This test runs the smoke study in-process and validates the results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.trustparadox_u.audit_results import audit_results
from experiments.trustparadox_u.config import (
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
    RunConfig,
)
from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.evaluator import evaluate_all
from experiments.trustparadox_u.runner import EpisodeResult, run_episode

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"

FIXTURES = [
    "pilot_credential.yaml",
    "pilot_private_attribute.yaml",
    "pilot_authorization.yaml",
]

SEEDS = [42, 123, 7]

CONDITIONS: list[tuple[str, dict, bool]] = [
    ("no_firewall", {}, False),
    (
        "exact_only",
        {
            "detector": DetectorConfig(
                exact_enabled=True, entity_enabled=False, semantic_enabled=False
            )
        },
        True,
    ),
    (
        "full_mvp",
        {
            "detector": DetectorConfig(
                exact_enabled=True, entity_enabled=True, semantic_enabled=True
            )
        },
        True,
    ),
    (
        "no_semantic",
        {
            "detector": DetectorConfig(
                exact_enabled=True, entity_enabled=True, semantic_enabled=False
            )
        },
        True,
    ),
    ("stateless", {"history": HistoryConfig(enabled=False)}, True),
    ("binary_policy", {"policy": PolicyConfig(rich_actions_enabled=False)}, True),
    ("rich_policy", {"policy": PolicyConfig(rich_actions_enabled=True)}, True),
    ("monitoring_0", {"monitoring": MonitoringConfig(continuous=False, duration_rounds=0)}, True),
    ("monitoring_1", {"monitoring": MonitoringConfig(continuous=False, duration_rounds=1)}, True),
    ("continuous", {"monitoring": MonitoringConfig(continuous=True)}, True),
]


def _make_config(seed: int, overrides: dict) -> ExperimentConfig:
    kwargs: dict = dict(
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


@pytest.fixture(scope="module")
def smoke_results() -> tuple[list[EpisodeResult], dict[str, list[EpisodeResult]]]:
    """Run the smoke study and return (all_results, condition_results)."""
    import hashlib

    all_results: list[EpisodeResult] = []
    condition_results: dict[str, list[EpisodeResult]] = {}

    for fixture_name in FIXTURES:
        ep = load_episode(SCENARIOS_DIR / fixture_name)
        for seed in SEEDS:
            for cond_name, cond_overrides, fw_enabled in CONDITIONS:
                cfg = _make_config(seed, cond_overrides)
                run_id = hashlib.sha256(
                    f"{ep.episode_id}|{cond_name}|{seed}|{fw_enabled}".encode()
                ).hexdigest()[:20]
                result = run_episode(ep, cfg, firewall_enabled=fw_enabled, run_id=run_id)
                result.metadata["smoke_condition"] = cond_name
                result.metadata["firewall_enabled"] = fw_enabled
                all_results.append(result)
                condition_results.setdefault(cond_name, []).append(result)

    return all_results, condition_results


@pytest.mark.integration
class TestSmokeStudyArtifacts:
    """Validate smoke study artifact production."""

    def test_total_run_count(self, smoke_results: tuple) -> None:
        """3 fixtures x 3 seeds x 10 conditions = 90 runs."""
        all_results, _ = smoke_results
        assert len(all_results) == 90

    def test_unique_run_ids(self, smoke_results: tuple) -> None:
        """All run IDs are unique."""
        all_results, _ = smoke_results
        run_ids = [r.run_id for r in all_results]
        assert len(run_ids) == len(set(run_ids))

    def test_all_fixtures_present(self, smoke_results: tuple) -> None:
        """All three canonical fixtures are represented."""
        all_results, _ = smoke_results
        episode_ids = {r.episode_id for r in all_results}
        assert len(episode_ids) == 3

    def test_all_conditions_present(self, smoke_results: tuple) -> None:
        """All 10 conditions are represented."""
        _, condition_results = smoke_results
        assert len(condition_results) == 10
        expected = {c[0] for c in CONDITIONS}
        assert set(condition_results.keys()) == expected

    def test_no_forbidden_string_leakage(self, smoke_results: tuple) -> None:
        """No forbidden strings appear in released text."""
        all_results, _ = smoke_results
        for result in all_results:
            forbidden = result.metadata.get("forbidden_strings", [])
            for turn in result.turns:
                if turn.released_text:
                    for f in forbidden:
                        assert f not in turn.released_text, (
                            f"Forbidden string {f!r} found in released text "
                            f"(condition={result.metadata.get('smoke_condition')}, "
                            f"episode={result.episode_id})"
                        )


@pytest.mark.integration
class TestSmokeStudyDirectionalChecks:
    """Validate required directional checks from the plan."""

    def test_privacy_mvp_better_than_no_firewall(self, smoke_results: tuple) -> None:
        """full MVP PU-RER <= no firewall PU-RER."""
        _, condition_results = smoke_results
        no_fw_eval = evaluate_all(condition_results["no_firewall"])
        mvp_eval = evaluate_all(condition_results["full_mvp"])
        no_fw_pu = no_fw_eval.pu_rer.value or 0.0
        mvp_pu = mvp_eval.pu_rer.value or 0.0
        assert mvp_pu <= no_fw_pu

    def test_semantic_protection(self, smoke_results: tuple) -> None:
        """semantic-enabled leakage <= semantic-disabled leakage."""
        _, condition_results = smoke_results
        full_eval = evaluate_all(condition_results["full_mvp"])
        no_sem_eval = evaluate_all(condition_results["no_semantic"])
        full_pu = full_eval.pu_rer.value or 0.0
        no_sem_pu = no_sem_eval.pu_rer.value or 0.0
        assert full_pu <= no_sem_pu

    def test_stateful_reconstruction_safer(self, smoke_results: tuple) -> None:
        """stateful CRR <= stateless CRR."""
        _, condition_results = smoke_results
        stateful_eval = evaluate_all(condition_results["full_mvp"])
        stateless_eval = evaluate_all(condition_results["stateless"])
        stateful_crr = stateful_eval.crr.value or 0.0
        stateless_crr = stateless_eval.crr.value or 0.0
        assert stateful_crr <= stateless_crr

    def test_rich_utility_ge_binary(self, smoke_results: tuple) -> None:
        """rich task_success >= binary task_success."""
        _, condition_results = smoke_results
        rich_success = sum(1 for r in condition_results["rich_policy"] if r.task_success)
        binary_success = sum(1 for r in condition_results["binary_policy"] if r.task_success)
        assert rich_success >= binary_success

    def test_continuous_rr_le_finite(self, smoke_results: tuple) -> None:
        """continuous RR <= finite-window RR."""
        _, condition_results = smoke_results
        cont_eval = evaluate_all(condition_results["continuous"])
        mon0_eval = evaluate_all(condition_results["monitoring_0"])
        cont_rr = cont_eval.rr.value or 0.0
        mon0_rr = mon0_eval.rr.value or 0.0
        assert cont_rr <= mon0_rr


@pytest.mark.integration
class TestSmokeStudyAudit:
    """Validate audit results for the smoke study."""

    def test_audit_no_unexpected_recontamination(self, smoke_results: tuple) -> None:
        """No unexpected recontamination pairs in smoke study."""
        all_results, _ = smoke_results
        report = audit_results(all_results)
        unexpected = [f for f in report.findings if f.code == "UNEXPECTED_RECONTAMINATION_PAIRS"]
        assert len(unexpected) == 0

    def test_audit_no_forbidden_string_violations(self, smoke_results: tuple) -> None:
        """No forbidden string violations in smoke study."""
        all_results, _ = smoke_results
        report = audit_results(all_results)
        violations = [f for f in report.findings if f.code == "FORBIDDEN_STRING_LEAK"]
        assert len(violations) == 0

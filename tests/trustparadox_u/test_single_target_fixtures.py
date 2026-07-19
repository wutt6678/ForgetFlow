"""Canonical single-target fixture verification tests.

These tests verify that the three canonical single-target fixtures
(credential, private_attribute, authorization) satisfy the single-target
validation suite requirements before they are used by other tests.
"""

from pathlib import Path

import pytest

from experiments.trustparadox_u.dataset import load_episode

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"

CANONICAL_FIXTURES = [
    pytest.param("pilot_credential.yaml", id="ST-FIXTURE-credential"),
    pytest.param("pilot_private_attribute.yaml", id="ST-FIXTURE-private_attribute"),
    pytest.param("pilot_authorization.yaml", id="ST-FIXTURE-authorization"),
]


class TestCanonicalFixtureLoading:
    """ST-FIXTURE-001: All canonical fixtures load successfully."""

    @pytest.mark.parametrize("filename", CANONICAL_FIXTURES)
    def test_fixture_loads(self, filename: str) -> None:
        ep = load_episode(SCENARIOS_DIR / filename)
        assert ep.episode_id
        assert ep.sensitive_items


class TestCanonicalFixtureSingleTarget:
    """ST-FIXTURE-002: Every canonical fixture contains exactly one sensitive item."""

    @pytest.mark.parametrize("filename", CANONICAL_FIXTURES)
    def test_exactly_one_sensitive_item(self, filename: str) -> None:
        ep = load_episode(SCENARIOS_DIR / filename)
        assert len(ep.sensitive_items) == 1


class TestCanonicalFixtureRecontaminationTargets:
    """ST-FIXTURE-003: Every recontamination attempt targets the episode's forget_id."""

    @pytest.mark.parametrize("filename", CANONICAL_FIXTURES)
    def test_recontamination_targets_match(self, filename: str) -> None:
        ep = load_episode(SCENARIOS_DIR / filename)
        valid_ids = {item.forget_id for item in ep.sensitive_items}

        for attack in ep.phases.post_forget:
            if attack.label.is_recontamination_attempt:
                for target_id in attack.label.target_forget_ids:
                    assert target_id in valid_ids, (
                        f"Recontamination target {target_id} not in "
                        f"episode forget_ids {valid_ids}"
                    )


class TestCanonicalFixturePermittedResiduals:
    """ST-FIXTURE-004: Every permitted residual is safe under leakage detectors."""

    @pytest.mark.parametrize("filename", CANONICAL_FIXTURES)
    def test_permitted_residuals_safe(self, filename: str) -> None:
        from marble.firewall.detectors import HybridDetector
        from marble.firewall.types import ForgetRecord

        ep = load_episode(SCENARIOS_DIR / filename)
        si = ep.sensitive_items[0]

        rec = ForgetRecord(
            forget_id=si.forget_id,
            canonical_target=si.canonical_target,
            target_type=si.target_type,
            aliases=si.aliases,
            semantic_variants=si.semantic_variants,
            permitted_residuals=si.permitted_residuals,
            active_from_turn=si.active_from_turn,
        )
        detector = HybridDetector(embedding_enabled=False)

        for residual in si.permitted_residuals:
            result = detector.detect(residual, [rec])
            assert (
                result.exact_score == 0.0
            ), f"Permitted residual '{residual}' triggers exact detection"
            assert (
                result.entity_score == 0.0
            ), f"Permitted residual '{residual}' triggers entity/alias detection"


class TestCanonicalFixtureDetectorPaths:
    """ST-FIXTURE-005: Every canonical target triggers the expected detector path."""

    @pytest.mark.parametrize("filename", CANONICAL_FIXTURES)
    def test_canonical_target_detected(self, filename: str) -> None:
        from marble.firewall.detectors import HybridDetector
        from marble.firewall.types import ForgetRecord

        ep = load_episode(SCENARIOS_DIR / filename)
        si = ep.sensitive_items[0]

        rec = ForgetRecord(
            forget_id=si.forget_id,
            canonical_target=si.canonical_target,
            target_type=si.target_type,
            aliases=si.aliases,
            semantic_variants=si.semantic_variants,
            permitted_residuals=si.permitted_residuals,
            active_from_turn=si.active_from_turn,
        )
        detector = HybridDetector(embedding_enabled=False)
        result = detector.detect(si.canonical_target, [rec])
        assert result.exact_score == 1.0
        assert si.forget_id in result.matched_forget_ids


class TestCanonicalFixtureTaskContracts:
    """ST-FIXTURE-006: Every task-success contract is explicit and valid."""

    @pytest.mark.parametrize("filename", CANONICAL_FIXTURES)
    def test_task_success_contract(self, filename: str) -> None:
        ep = load_episode(SCENARIOS_DIR / filename)
        assert ep.task.success_type in ("required_release", "exact_label")
        assert ep.task.success_value
        assert ep.task.success_label

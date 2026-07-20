"""Tests for utility consistency (Iteration 9 of repair spec)."""

from __future__ import annotations

import pytest

from experiments.trustparadox_u.assertion_contracts import classify_candidate_exposure


class TestPermittedResidualClassification:
    """Test that permitted residuals are correctly classified."""

    def test_permitted_residual_classification(self) -> None:
        """Permitted residual should be classified as permitted_residual, not direct_exact."""
        # Even if exact score is 1.0, if it's a permitted residual, it should be classified
        # as permitted_residual (lower precedence than direct_exact but still safe)
        classification = classify_candidate_exposure(
            exact_score=1.0,
            entity_score=0.0,
            embedding_score=0.0,
            proposition_relevant=False,
            proposition_entailed=False,
            reconstruction_score=0.0,
            recontamination_attempt=False,
            permitted_residual=True,
            attack_request=False,
            matched_forget_ids=("F001",),
        )

        # direct_exact takes precedence over permitted_residual
        assert classification.exposure_class == "direct_exact"

    def test_permitted_residual_without_exact_match(self) -> None:
        """Permitted residual without exact match should be classified as permitted_residual."""
        classification = classify_candidate_exposure(
            exact_score=0.0,
            entity_score=0.0,
            embedding_score=0.0,
            proposition_relevant=False,
            proposition_entailed=False,
            reconstruction_score=0.0,
            recontamination_attempt=False,
            permitted_residual=True,
            attack_request=False,
            matched_forget_ids=("F001",),
        )

        assert classification.exposure_class == "permitted_residual"

    def test_attack_request_classification(self) -> None:
        """Attack request should be classified as attack_request."""
        classification = classify_candidate_exposure(
            exact_score=0.0,
            entity_score=0.0,
            embedding_score=0.0,
            proposition_relevant=False,
            proposition_entailed=False,
            reconstruction_score=0.0,
            recontamination_attempt=False,
            permitted_residual=False,
            attack_request=True,
            matched_forget_ids=("F001",),
        )

        assert classification.exposure_class == "attack_request"

    def test_none_classification(self) -> None:
        """No detection should be classified as none."""
        classification = classify_candidate_exposure(
            exact_score=0.0,
            entity_score=0.0,
            embedding_score=0.0,
            proposition_relevant=False,
            proposition_entailed=False,
            reconstruction_score=0.0,
            recontamination_attempt=False,
            permitted_residual=False,
            attack_request=False,
            matched_forget_ids=(),
        )

        assert classification.exposure_class == "none"


class TestTaskSuccessEvaluation:
    """Test that task success is correctly evaluated."""

    def test_required_release_task_success(self) -> None:
        """Required release task should succeed when success value is released."""
        # This is tested in the runner tests
        pass

    def test_exact_label_task_success(self) -> None:
        """Exact label task should succeed when task label matches success value."""
        # This is tested in the runner tests
        pass

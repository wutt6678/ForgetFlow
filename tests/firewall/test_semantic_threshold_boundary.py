"""Semantic threshold boundary tests (ST-SEM-003).

This test suite verifies that the semantic threshold comparison is deterministic
and that an accidental comparator change cannot silently alter semantic enforcement.

Gap addressed: ST-SEM-003
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from marble.firewall.detectors import HybridDetector
from marble.firewall.types import ForgetRecord


class TestSemanticThresholdBoundary:
    """ST-SEM-003: Semantic threshold boundary tests.

    Contract: score >= threshold triggers detection.
    """

    @pytest.mark.parametrize(
        ("score", "expected_risky"),
        [
            pytest.param(0.79, False, id="ST-SEM-003-below"),
            pytest.param(0.80, True, id="ST-SEM-003-equal"),
            pytest.param(0.81, True, id="ST-SEM-003-above"),
        ],
    )
    def test_embedding_threshold_boundary(
        self, score: float, expected_risky: bool
    ) -> None:
        """Semantic score at threshold boundary triggers correct detection."""
        # Mock embedding provider to return controlled similarity
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [0.0] * 1024  # Dimension doesn't matter

        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=True,
            embedding_threshold=0.80,
            embedding_provider=mock_provider,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=(),
            semantic_variants=("Agent B holds emergency access",),
            permitted_residuals=(),
            active_from_turn=0,
        )

        # Mock _compute_semantic to return controlled score
        detector._compute_semantic = MagicMock(return_value=score)  # type: ignore

        result = detector.detect(
            text="Some text about Agent B",
            active_records=[record],
        )

        # Check if detection occurred
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]

        # Verify semantic score is recorded
        assert evidence.semantic_score == score

        # Verify detection decision matches expected
        assert evidence.matched == expected_risky

    @pytest.mark.parametrize(
        ("score", "expected_risky"),
        [
            pytest.param(0.80 - 1e-12, False, id="ST-SEM-003-precision-below"),
            pytest.param(0.80, True, id="ST-SEM-003-precision-equal"),
            pytest.param(0.80 + 1e-12, True, id="ST-SEM-003-precision-above"),
        ],
    )
    def test_embedding_threshold_precision(
        self, score: float, expected_risky: bool
    ) -> None:
        """Semantic threshold comparison is precise at floating-point boundary."""
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [0.0] * 1024

        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=True,
            embedding_threshold=0.80,
            embedding_provider=mock_provider,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=(),
            semantic_variants=("Agent B holds emergency access",),
            permitted_residuals=(),
            active_from_turn=0,
        )

        detector._compute_semantic = MagicMock(return_value=score)  # type: ignore

        result = detector.detect(
            text="Some text about Agent B",
            active_records=[record],
        )

        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        assert evidence.semantic_score == score
        assert evidence.matched == expected_risky

    def test_embedding_threshold_contract_documented(self) -> None:
        """Document that semantic threshold uses >= comparison."""
        # This test documents the contract: score >= threshold
        # The contract is implemented in detectors.py line 107:
        #   if sem_score >= self.embedding_threshold:
        #
        # This means:
        # - score = threshold - epsilon: NOT detected
        # - score = threshold: DETECTED
        # - score = threshold + epsilon: DETECTED
        #
        # This contract must not change without explicit review.
        assert True  # Documentation test

    @pytest.mark.parametrize(
        "threshold",
        [0.60, 0.70, 0.80, 0.90],
    )
    def test_embedding_threshold_various_values(self, threshold: float) -> None:
        """Semantic threshold works correctly at various values."""
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [0.0] * 1024

        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=True,
            embedding_threshold=threshold,
            embedding_provider=mock_provider,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=(),
            semantic_variants=("Agent B holds emergency access",),
            permitted_residuals=(),
            active_from_turn=0,
        )

        # Test at threshold
        detector._compute_semantic = MagicMock(return_value=threshold)  # type: ignore
        result = detector.detect(
            text="Some text about Agent B",
            active_records=[record],
        )
        assert len(result.record_evidence) > 0
        assert result.record_evidence[0].matched is True

        # Test below threshold
        detector._compute_semantic = MagicMock(return_value=threshold - 0.01)  # type: ignore
        result = detector.detect(
            text="Some text about Agent B",
            active_records=[record],
        )
        assert len(result.record_evidence) > 0
        assert result.record_evidence[0].matched is False

        # Test above threshold
        detector._compute_semantic = MagicMock(return_value=threshold + 0.01)  # type: ignore
        result = detector.detect(
            text="Some text about Agent B",
            active_records=[record],
        )
        assert len(result.record_evidence) > 0
        assert result.record_evidence[0].matched is True

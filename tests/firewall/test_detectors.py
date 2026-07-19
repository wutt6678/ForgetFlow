"""Tests for HybridDetector."""

from marble.firewall.detectors import HybridDetector, _normalize
from marble.firewall.types import ForgetRecord


def _rec(**kwargs: object) -> ForgetRecord:
    defaults = dict(
        forget_id="F001",
        canonical_target="0107",
        target_type="credential",
        aliases=("warehouse code",),
        semantic_variants=(),
        permitted_residuals=(),
        active_from_turn=0,
    )
    defaults.update(kwargs)
    return ForgetRecord(**defaults)  # type: ignore[arg-type]


class TestNormalize:
    def test_lowercase(self) -> None:
        assert _normalize("Hello World") == "hello world"

    def test_punctuation(self) -> None:
        assert "0107" in _normalize("The code is 0107!")

    def test_whitespace(self) -> None:
        assert _normalize("  too   many   spaces  ") == "too many spaces"


class TestHybridDetector:
    def test_exact_match(self) -> None:
        det = HybridDetector(embedding_enabled=False)
        result = det.detect("The code is 0107", [_rec()])
        assert result.exact_score == 1.0
        assert "F001" in result.matched_forget_ids

    def test_exact_case_insensitive(self) -> None:
        det = HybridDetector(embedding_enabled=False)
        result = det.detect("Code: 0107", [_rec()])
        assert result.exact_score == 1.0

    def test_alias_match(self) -> None:
        det = HybridDetector(embedding_enabled=False)
        result = det.detect("Tell me the warehouse code", [_rec()])
        assert result.entity_score == 1.0
        assert "F001" in result.matched_forget_ids

    def test_no_match(self) -> None:
        det = HybridDetector(embedding_enabled=False)
        result = det.detect("Hello world", [_rec()])
        assert result.exact_score == 0.0
        assert result.entity_score == 0.0

    def test_permitted_residual(self) -> None:
        det = HybridDetector(embedding_enabled=False)
        result = det.detect("Request a new temporary credential", [_rec()])
        assert result.exact_score == 0.0
        assert result.entity_score == 0.0

    def test_multiple_targets(self) -> None:
        r1 = _rec(forget_id="F001", canonical_target="0107")
        r2 = ForgetRecord(
            forget_id="F002",
            canonical_target="9999",
            target_type="pin",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )
        det = HybridDetector(embedding_enabled=False)
        result = det.detect("Codes are 0107 and 9999", [r1, r2])
        assert result.exact_score == 1.0
        assert "F001" in result.matched_forget_ids
        assert "F002" in result.matched_forget_ids

    def test_disabled_exact(self) -> None:
        det = HybridDetector(exact_enabled=False, embedding_enabled=False)
        result = det.detect("The code is 0107", [_rec()])
        assert result.exact_score == 0.0

    def test_disabled_entity(self) -> None:
        det = HybridDetector(entity_enabled=False, embedding_enabled=False)
        result = det.detect("Tell me the warehouse code", [_rec()])
        assert result.entity_score == 0.0

    def test_evidence_includes_match(self) -> None:
        det = HybridDetector(embedding_enabled=False)
        result = det.detect("The code is 0107", [_rec()])
        assert any("EXACT" in e for e in result.evidence)


class TestSemanticThresholdBoundary:
    """ST-SEM-003: Semantic threshold boundary behavior.

    The production comparison is score >= threshold.
    These tests freeze that contract.
    """

    def test_below_threshold_is_safe(self) -> None:
        """ST-SEM-003-below: Score below threshold is not risky."""
        from marble.firewall.policy import ForgetPolicy
        from marble.firewall.types import DetectorResult

        policy = ForgetPolicy(embedding_threshold=0.80)
        # Score 0.79 should be safe
        result = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.79,
            reconstruction_score=0.0,
            matched_forget_ids=(),
            evidence=(),
        )
        action, _, _ = policy.decide(result, [], "1.0")
        assert action == "allow"

    def test_at_threshold_is_risky(self) -> None:
        """ST-SEM-003-equal: Score at threshold is risky (>=)."""
        from marble.firewall.policy import ForgetPolicy
        from marble.firewall.types import DetectorResult

        policy = ForgetPolicy(embedding_threshold=0.80)
        result = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.80,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=("SEMANTIC",),
        )
        action, _, _ = policy.decide(result, [], "1.0")
        assert action in ("abstract", "block", "redact")

    def test_above_threshold_is_risky(self) -> None:
        """ST-SEM-003-above: Score above threshold is risky."""
        from marble.firewall.policy import ForgetPolicy
        from marble.firewall.types import DetectorResult

        policy = ForgetPolicy(embedding_threshold=0.80)
        result = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.81,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=("SEMANTIC",),
        )
        action, _, _ = policy.decide(result, [], "1.0")
        assert action in ("abstract", "block", "redact")

    def test_precision_boundary(self) -> None:
        """ST-SEM-003-precision: Boundary at machine precision."""
        from marble.firewall.policy import ForgetPolicy
        from marble.firewall.types import DetectorResult

        policy = ForgetPolicy(embedding_threshold=0.80)
        # Just below threshold
        below = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.80 - 1e-12,
            reconstruction_score=0.0,
            matched_forget_ids=(),
            evidence=(),
        )
        action_below, _, _ = policy.decide(below, [], "1.0")
        assert action_below == "allow"
        # At threshold
        at = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.80,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=("SEMANTIC",),
        )
        action_at, _, _ = policy.decide(at, [], "1.0")
        assert action_at in ("abstract", "block", "redact")

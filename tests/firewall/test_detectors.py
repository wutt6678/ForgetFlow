"""Tests for HybridDetector."""

import pytest
from marble.firewall.types import ForgetRecord
from marble.firewall.detectors import HybridDetector, _normalize


def _rec(**kwargs: object) -> ForgetRecord:
    defaults = dict(
        forget_id="F001", canonical_target="0107", target_type="credential",
        aliases=("warehouse code",), semantic_variants=(),
        permitted_residuals=(), active_from_turn=0,
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
        det = HybridDetector(semantic_enabled=False)
        result = det.detect("The code is 0107", [_rec()])
        assert result.exact_score == 1.0
        assert "F001" in result.matched_forget_ids

    def test_exact_case_insensitive(self) -> None:
        det = HybridDetector(semantic_enabled=False)
        result = det.detect("Code: 0107", [_rec()])
        assert result.exact_score == 1.0

    def test_alias_match(self) -> None:
        det = HybridDetector(semantic_enabled=False)
        result = det.detect("Tell me the warehouse code", [_rec()])
        assert result.entity_score == 1.0
        assert "F001" in result.matched_forget_ids

    def test_no_match(self) -> None:
        det = HybridDetector(semantic_enabled=False)
        result = det.detect("Hello world", [_rec()])
        assert result.exact_score == 0.0
        assert result.entity_score == 0.0

    def test_permitted_residual(self) -> None:
        det = HybridDetector(semantic_enabled=False)
        result = det.detect("Request a new temporary credential", [_rec()])
        assert result.exact_score == 0.0
        assert result.entity_score == 0.0

    def test_multiple_targets(self) -> None:
        r1 = _rec(forget_id="F001", canonical_target="0107")
        r2 = ForgetRecord(
            forget_id="F002", canonical_target="9999", target_type="pin",
            aliases=(), semantic_variants=(), permitted_residuals=(),
            active_from_turn=0,
        )
        det = HybridDetector(semantic_enabled=False)
        result = det.detect("Codes are 0107 and 9999", [r1, r2])
        assert result.exact_score == 1.0
        assert "F001" in result.matched_forget_ids
        assert "F002" in result.matched_forget_ids

    def test_disabled_exact(self) -> None:
        det = HybridDetector(exact_enabled=False, semantic_enabled=False)
        result = det.detect("The code is 0107", [_rec()])
        assert result.exact_score == 0.0

    def test_disabled_entity(self) -> None:
        det = HybridDetector(entity_enabled=False, semantic_enabled=False)
        result = det.detect("Tell me the warehouse code", [_rec()])
        assert result.entity_score == 0.0

    def test_evidence_includes_match(self) -> None:
        det = HybridDetector(semantic_enabled=False)
        result = det.detect("The code is 0107", [_rec()])
        assert any("EXACT" in e for e in result.evidence)

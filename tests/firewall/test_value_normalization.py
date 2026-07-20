"""Tests for canonical value normalization (Iteration 1 of repair spec)."""

from __future__ import annotations

import pytest

from marble.firewall.normalization import (
    CanonicalValue,
    canonicalize_sensitive_values,
    redact_canonical_value,
    text_contains_canonical_value,
)


class TestSpokenDigitNormalization:
    """Test spoken digit to numeric conversion."""

    def test_spoken_digits_match_numeric_target(self) -> None:
        """zero one zero seven should match 0107."""
        text = "The gold-store credential is zero one zero seven."
        assert text_contains_canonical_value(text, "0107")

    def test_hyphenated_spoken_digits_match_numeric_target(self) -> None:
        """zero-one-zero-seven should match 0107."""
        text = "The credential is zero-one-zero-seven."
        assert text_contains_canonical_value(text, "0107")

    def test_mixed_word_digit_alias_matches_target(self) -> None:
        """zero 1 zero 7 should match 0107."""
        text = "The code is zero 1 zero 7."
        assert text_contains_canonical_value(text, "0107")

    def test_separated_digits_match_target(self) -> None:
        """0 1 0 7 should match 0107."""
        text = "The code is 0 1 0 7."
        assert text_contains_canonical_value(text, "0107")

    def test_hyphenated_digits_match_target(self) -> None:
        """0-1-0-7 should match 0107."""
        text = "The code is 0-1-0-7."
        assert text_contains_canonical_value(text, "0107")

    def test_direct_match_still_works(self) -> None:
        """Direct substring match should still work."""
        text = "The credential is 0107."
        assert text_contains_canonical_value(text, "0107")

    def test_case_insensitive_match(self) -> None:
        """Matching should be case-insensitive."""
        text = "The code is ZERO ONE ZERO SEVEN."
        assert text_contains_canonical_value(text, "0107")

    def test_no_match_for_different_value(self) -> None:
        """Different values should not match."""
        text = "The code is zero one zero eight."
        assert not text_contains_canonical_value(text, "0107")

    def test_empty_text(self) -> None:
        """Empty text should not match."""
        assert not text_contains_canonical_value("", "0107")

    def test_empty_target(self) -> None:
        """Empty target should not match."""
        assert not text_contains_canonical_value("some text", "")


class TestCanonicalizeSensitiveValues:
    """Test the canonicalize_sensitive_values function."""

    def test_spoken_digit_sequence(self) -> None:
        """Spoken digit sequences should be normalized."""
        result = canonicalize_sensitive_values("zero one zero seven")
        assert "0107" in result.canonical_candidates

    def test_hyphenated_spoken_digits(self) -> None:
        """Hyphenated spoken digits should be normalized."""
        result = canonicalize_sensitive_values("zero-one-zero-seven")
        assert "0107" in result.canonical_candidates

    def test_separated_digits(self) -> None:
        """Separated digits should be normalized."""
        result = canonicalize_sensitive_values("0 1 0 7")
        assert "0107" in result.canonical_candidates

    def test_returns_canonical_value(self) -> None:
        """Should return a CanonicalValue object."""
        result = canonicalize_sensitive_values("test text")
        assert isinstance(result, CanonicalValue)
        assert result.raw_text == "test text"


class TestRedactCanonicalValue:
    """Test redaction of all canonical forms."""

    def test_redaction_removes_alias_value(self) -> None:
        """Redaction should remove spoken digit forms."""
        text = "The gold-store credential is zero one zero seven."
        result = redact_canonical_value(text, "0107")
        assert "zero one zero seven" not in result.lower()
        assert "0107" not in result
        assert "[REDACTED]" in result

    def test_redaction_removes_hyphenated_form(self) -> None:
        """Redaction should remove hyphenated spoken forms."""
        text = "The credential is zero-one-zero-seven."
        result = redact_canonical_value(text, "0107")
        assert "zero-one-zero-seven" not in result.lower()
        assert "[REDACTED]" in result

    def test_redaction_removes_separated_digits(self) -> None:
        """Redaction should remove separated digit forms."""
        text = "The code is 0 1 0 7."
        result = redact_canonical_value(text, "0107")
        assert "0 1 0 7" not in result
        assert "[REDACTED]" in result

    def test_redaction_removes_direct_form(self) -> None:
        """Redaction should remove direct numeric form."""
        text = "The credential is 0107."
        result = redact_canonical_value(text, "0107")
        assert "0107" not in result
        assert "[REDACTED]" in result

    def test_safe_residual_contains_no_canonical_value(self) -> None:
        """After redaction, no canonical form should remain."""
        text = "The gold-store credential is zero one zero seven. Please change it."
        result = redact_canonical_value(text, "0107")
        # Verify no form of the value remains
        assert not text_contains_canonical_value(result, "0107")

    def test_custom_replacement(self) -> None:
        """Should support custom replacement text."""
        text = "The code is 0107."
        result = redact_canonical_value(text, "0107", "[REMOVED]")
        assert "[REMOVED]" in result
        assert "0107" not in result

    def test_empty_text_returns_empty(self) -> None:
        """Empty text should return empty."""
        assert redact_canonical_value("", "0107") == ""

    def test_no_match_returns_original(self) -> None:
        """Text without target should be unchanged."""
        text = "The credential is 1234."
        result = redact_canonical_value(text, "0107")
        assert result == text


class TestPostTransformRecheck:
    """Test that post-transformation recheck catches spoken aliases."""

    def test_post_transform_recheck_catches_spoken_alias(self) -> None:
        """After partial redaction, spoken alias should still be detected."""
        # Simulate a partial redaction that missed the spoken form
        text = "The [REDACTED] is zero one zero seven."
        # The normalization should still detect the value
        assert text_contains_canonical_value(text, "0107")

    def test_full_redaction_passes_recheck(self) -> None:
        """After full redaction, no value should be detected."""
        text = "The gold-store credential is zero one zero seven."
        result = redact_canonical_value(text, "0107")
        assert not text_contains_canonical_value(result, "0107")

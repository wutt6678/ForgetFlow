"""Canonical value normalization for sensitive value detection.

This module provides normalization of sensitive values to their canonical forms,
enabling detection of spoken-digit aliases, hyphenated forms, and other variations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Mapping of spoken digit words to their numeric equivalents
_SPOKEN_DIGITS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "oh": "0",  # Common pronunciation for 0
}

# Build pattern for matching spoken digit sequences
_SPOKEN_DIGIT_PATTERN = re.compile(
    r"\b(" + "|".join(_SPOKEN_DIGITS.keys()) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CanonicalValue:
    """A normalized representation of a sensitive value.

    Attributes:
        raw_text: The original text containing the value
        normalized_text: The text after normalization
        canonical_candidates: Tuple of possible canonical forms
        spans: Tuple of (start, end) positions where values were found
    """

    raw_text: str
    normalized_text: str
    canonical_candidates: tuple[str, ...]
    spans: tuple[tuple[int, int], ...]


def _spoken_to_digits(text: str) -> str:
    """Convert spoken digit words to their numeric equivalents.

    Examples:
        "zero one zero seven" -> "0107"
        "zero-one-zero-seven" -> "0107"
        "zero 1 zero 7" -> "0101" (mixed)
    """
    result = text.lower()
    # Replace spoken digits with their numeric equivalents
    for word, digit in _SPOKEN_DIGITS.items():
        result = re.sub(rf"\b{word}\b", digit, result, flags=re.IGNORECASE)
    return result


def _remove_separators(text: str) -> str:
    """Remove separators between digits (spaces, hyphens, commas).

    Examples:
        "0 1 0 7" -> "0107"
        "01-07" -> "0107"
        "0,1,0,7" -> "0107"
    """
    # Remove hyphens, commas, and spaces between digits
    result = re.sub(r"[\s\-,]+", "", text)
    return result


def canonicalize_sensitive_values(
    text: str,
    *,
    value_type: str | None = None,
) -> CanonicalValue:
    """Normalize sensitive values in text to their canonical forms.

    This function handles:
    - Spoken digit words: "zero one zero seven" -> "0107"
    - Hyphenated forms: "zero-one-zero-seven" -> "0107"
    - Separated digits: "0 1 0 7" -> "0107"
    - Mixed word/digit: "zero 1 zero 7" -> "0107"
    - Punctuation and whitespace normalization

    Args:
        text: The text to normalize
        value_type: Optional hint for value type (e.g., "numeric_credential")

    Returns:
        CanonicalValue with normalized forms and span information
    """
    raw_text = text
    spans: list[tuple[int, int]] = []

    # Step 1: Check for spoken digit sequences
    spoken_matches = list(_SPOKEN_DIGIT_PATTERN.finditer(text))
    if spoken_matches:
        # Find contiguous sequences of spoken digits (with separators)
        normalized = _spoken_to_digits(text)
        # Remove separators between consecutive digits
        normalized = _remove_separators(normalized)
    else:
        normalized = text

    # Step 2: Also create a version with separators removed from original
    digits_only = _remove_separators(text)

    # Step 3: Build canonical candidates
    candidates: set[str] = set()
    candidates.add(normalized)
    candidates.add(digits_only)
    candidates.add(text.lower().strip())

    # Step 4: Find spans of potential sensitive values
    # Look for digit sequences (possibly with separators)
    digit_pattern = re.compile(r"\d[\d\s\-,]*\d|\d")
    for match in digit_pattern.finditer(text):
        spans.append((match.start(), match.end()))

    # Also look for spoken digit sequences
    if spoken_matches:
        # Find the full span of contiguous spoken digits
        start = spoken_matches[0].start()
        end = spoken_matches[-1].end()
        # Check if they're contiguous (separated by whitespace/hyphens only)
        between = text[start:end]
        if re.match(r"^[\w\s\-]+$", between):
            spans.append((start, end))

    return CanonicalValue(
        raw_text=raw_text,
        normalized_text=normalized,
        canonical_candidates=tuple(sorted(candidates)),
        spans=tuple(spans),
    )


def text_contains_canonical_value(
    text: str,
    target_value: str,
) -> bool:
    """Check if text contains a target value in any canonical form.

    This checks:
    1. Direct substring match (case-insensitive)
    2. Spoken digit form: "zero one zero seven" contains "0107"
    3. Separated digit form: "0 1 0 7" contains "0107"
    4. Hyphenated form: "zero-one-zero-seven" contains "0107"

    Args:
        text: The text to search
        target_value: The target value to find (e.g., "0107")

    Returns:
        True if the target value is found in any canonical form
    """
    if not text or not target_value:
        return False

    target_lower = target_value.lower().strip()
    text_lower = text.lower()

    # Check 1: Direct match
    if target_lower in text_lower:
        return True

    # Check 2: Normalize text and check again
    # Convert spoken digits to numbers
    normalized_text = _spoken_to_digits(text_lower)
    # Remove separators between digits
    normalized_compact = _remove_separators(normalized_text)

    if target_lower in normalized_compact:
        return True

    # Check 3: Also try removing separators from original text
    text_compact = _remove_separators(text_lower)
    if target_lower in text_compact:
        return True

    # Check 4: Convert target to spoken form and check
    target_spoken = _digits_to_spoken(target_lower)
    if target_spoken:
        # Check various spoken forms
        spoken_forms = [
            target_spoken,  # "zero one zero seven"
            target_spoken.replace(" ", "-"),  # "zero-one-zero-seven"
            " ".join(target_lower),  # "0 1 0 7"
            "-".join(target_lower),  # "0-1-0-7"
        ]
        for form in spoken_forms:
            if form in text_lower:
                return True

    return False


def _digits_to_spoken(digits: str) -> str:
    """Convert a digit string to spoken form.

    Example: "0107" -> "zero one zero seven"
    """
    reverse_map = {v: k for k, v in _SPOKEN_DIGITS.items() if k != "oh"}
    result = []
    for char in digits:
        if char in reverse_map:
            result.append(reverse_map[char])
        else:
            return ""  # Non-digit character, can't convert
    return " ".join(result)


def redact_canonical_value(
    text: str,
    target_value: str,
    replacement: str = "[REDACTED]",
) -> str:
    """Redact all forms of a target value from text.

    This removes:
    - Direct occurrences of the target
    - Spoken digit forms
    - Separated digit forms
    - Hyphenated forms

    Args:
        text: The text to redact
        target_value: The target value to redact
        replacement: The replacement text (default: "[REDACTED]")

    Returns:
        The redacted text with all forms of the target removed
    """
    if not text or not target_value:
        return text

    result = text
    target_lower = target_value.lower()

    # Build patterns to redact (order matters - longer patterns first)
    patterns: list[str] = []

    # Spoken form with spaces: "zero one zero seven"
    spoken_spaced = _digits_to_spoken(target_lower)
    if spoken_spaced:
        patterns.append(re.escape(spoken_spaced))
        # Spoken form with hyphens: "zero-one-zero-seven"
        patterns.append(re.escape(spoken_spaced.replace(" ", "-")))

    # Separated digit forms: "0 1 0 7", "0-1-0-7"
    separated_space = " ".join(target_lower)
    separated_hyphen = "-".join(target_lower)
    patterns.append(re.escape(separated_space))
    patterns.append(re.escape(separated_hyphen))

    # Direct form: "0107"
    patterns.append(re.escape(target_lower))

    # Apply patterns (case-insensitive)
    for pattern in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result

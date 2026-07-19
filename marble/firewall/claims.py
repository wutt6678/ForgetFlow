"""Structured claim representation for proposition-level detection.

This module provides:
- Claim: Normalized subject-predicate-object structure
- MessageContext: Sender/recipient identity for coreference resolution
- ClaimNormalizer: Extract claims from text with pronoun resolution
- PropositionMatcher: Compare claims for semantic equivalence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from marble.firewall.types import ForgetRecord


@dataclass(frozen=True)
class Claim:
    """Normalized proposition structure for privacy detection.

    A claim represents a normalized form of a statement that can be compared
    against forgotten targets. This enables detection of disclosures that
    change perspective, polarity, modality, or tense.

    Example:
        "You have emergency authority" →
        Claim(
            subject="AGENT_B",
            predicate="has",
            object="emergency authority",
            polarity="positive",
            modality="certain",
            temporal_status="current",
            source_text="You have emergency authority",
            confidence=0.95
        )
    """

    subject: str
    predicate: str
    object: str
    polarity: Literal["positive", "negative"]
    modality: Literal["certain", "possible", "requested", "conditional"]
    temporal_status: Literal["current", "past", "future", "revoked", "unknown"]
    source_text: str
    confidence: float


@dataclass(frozen=True)
class MessageContext:
    """Sender and recipient identity for pronoun resolution.

    This context enables resolution of:
    - First person (I, me, my) → sender
    - Second person (you, your, yourself) → recipient
    - Third person (he, she, they) → requires alias resolution

    Example:
        MessageContext(
            sender_id="COMMAND_AGENT",
            recipient_id="AGENT_B",
            sender_aliases=("Command Agent", "command"),
            recipient_aliases=("Agent B", "you", "your")
        )
    """

    sender_id: str
    recipient_id: str
    sender_aliases: tuple[str, ...] = ()
    recipient_aliases: tuple[str, ...] = ()


class CoreferenceResolver:
    """Resolve pronouns to named entities using message context."""

    # First-person pronouns → sender
    FIRST_PERSON = frozenset(
        {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
    )

    # Second-person pronouns → recipient
    SECOND_PERSON = frozenset({"you", "your", "yours", "yourself", "yourselves"})

    def resolve(
        self,
        text: str,
        context: MessageContext,
    ) -> str:
        """Resolve pronouns in text to named entities.

        Args:
            text: Input text with potential pronouns
            context: Sender/recipient identity context

        Returns:
            Text with pronouns replaced by named entities

        Example:
            >>> resolver = CoreferenceResolver()
            >>> ctx = MessageContext("COMMAND", "AGENT_B", (), ("Agent B",))
            >>> resolver.resolve("You have authority", ctx)
            "AGENT_B has authority"
        """
        words = text.split()
        resolved = []

        for word in words:
            lower = word.lower().strip(".,!?;:")

            if lower in self.FIRST_PERSON:
                resolved.append(context.sender_id)
            elif lower in self.SECOND_PERSON:
                resolved.append(context.recipient_id)
            else:
                resolved.append(word)

        return " ".join(resolved)


class ClaimNormalizer:
    """Extract normalized claims from text.

    This is a simplified implementation that handles basic SVO patterns.
    A production system would use NLP parsing or LLM extraction.
    """

    def __init__(self) -> None:
        self.coref_resolver = CoreferenceResolver()

    def normalize(
        self,
        text: str,
        context: MessageContext | None = None,
    ) -> list[Claim]:
        """Extract claims from text.

        Args:
            text: Input text to analyze
            context: Optional sender/recipient context for pronoun resolution

        Returns:
            List of extracted claims

        Example:
            >>> normalizer = ClaimNormalizer()
            >>> ctx = MessageContext("COMMAND", "AGENT_B", (), ("Agent B",))
            >>> claims = normalizer.normalize("You have authority", ctx)
            >>> len(claims)
            1
            >>> claims[0].subject
            'AGENT_B'
        """
        # Resolve pronouns if context provided
        if context:
            text = self.coref_resolver.resolve(text, context)

        # Simplified claim extraction
        # A production system would use dependency parsing or LLM
        claims = []

        # Detect polarity
        polarity = self._detect_polarity(text)

        # Detect modality
        modality = self._detect_modality(text)

        # Detect temporal status
        temporal = self._detect_temporal(text)

        # Extract basic SVO pattern (simplified)
        # This is a placeholder - real implementation needs NLP
        subject, predicate, obj = self._extract_svo(text)

        if subject and predicate:
            claims.append(
                Claim(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    polarity=polarity,
                    modality=modality,
                    temporal_status=temporal,
                    source_text=text,
                    confidence=0.7,  # Lower confidence for extracted claims
                )
            )

        return claims

    def _detect_polarity(self, text: str) -> Literal["positive", "negative"]:
        """Detect if claim is positive or negative."""
        lower = text.lower()
        negation_markers = [
            "not",
            "no",
            "never",
            "neither",
            "nobody",
            "nothing",
            "nowhere",
            "nor",
            "cannot",
            "can't",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "wouldn't",
            "shouldn't",
            "couldn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
        ]

        for marker in negation_markers:
            if marker in lower:
                return "negative"

        return "positive"

    def _detect_modality(
        self, text: str
    ) -> Literal["certain", "possible", "requested", "conditional"]:
        """Detect modality of claim."""
        lower = text.lower()

        # Questions/requests
        if "?" in text or any(w in lower for w in ["please", "could you", "would you"]):
            return "requested"

        # Possibility
        if any(w in lower for w in ["may", "might", "could", "possibly", "perhaps"]):
            return "possible"

        # Conditional
        if any(w in lower for w in ["if", "when", "unless", "provided"]):
            return "conditional"

        return "certain"

    def _detect_temporal(
        self, text: str
    ) -> Literal["current", "past", "future", "revoked", "unknown"]:
        """Detect temporal status of claim."""
        lower = text.lower()

        # Revoked
        if any(w in lower for w in ["revoked", "denied", "inactive", "removed"]):
            return "revoked"

        # Past
        if any(
            w in lower
            for w in ["previously", "formerly", "had", "was", "were", "used to"]
        ):
            return "past"

        # Future
        if any(w in lower for w in ["will", "shall", "going to", "will be"]):
            return "future"

        return "current"

    def _extract_svo(self, text: str) -> tuple[str, str, str]:
        """Extract subject-verb-object from text.

        This is a simplified placeholder. Real implementation needs NLP.
        """
        # Very basic pattern matching
        # Production system should use spaCy or similar
        words = text.split()

        if len(words) < 2:
            return "", "", ""

        # Assume first noun is subject, first verb is predicate, rest is object
        # This is extremely simplified
        subject = words[0] if words else ""
        predicate = words[1] if len(words) > 1 else ""
        obj = " ".join(words[2:]) if len(words) > 2 else ""

        return subject, predicate, obj


class PropositionMatcher:
    """Match claims against forgotten targets."""

    def match(
        self,
        claim: Claim,
        target_record: ForgetRecord,
        threshold: float = 0.8,
    ) -> tuple[bool, float]:
        """Check if claim matches forgotten target.

        Args:
            claim: Normalized claim to check
            target_record: Forgotten target record
            threshold: Similarity threshold for matching

        Returns:
            Tuple of (matches: bool, confidence: float)

        Example:
            >>> matcher = PropositionMatcher()
            >>> claim = Claim("AGENT_B", "has", "authority", "positive", ...)
            >>> record = ForgetRecord("F001", "Agent B has authority", ...)
            >>> matches, conf = matcher.match(claim, record)
        """
        # Simplified matching logic
        # Production system should use embedding similarity on normalized claims

        # Check if subjects match
        subject_match = self._subjects_match(claim.subject, target_record.canonical_target)

        # Check if predicates are compatible
        predicate_match = claim.predicate.lower() in ["has", "have", "is", "are", "holds"]

        # Check polarity
        polarity_match = claim.polarity == "positive"

        # Compute overall confidence
        if subject_match and predicate_match and polarity_match:
            return True, claim.confidence

        return False, 0.0

    def _subjects_match(self, claim_subject: str, target: str) -> bool:
        """Check if claim subject matches target."""
        # Simplified: check for substring or alias match
        lower_subject = claim_subject.lower()
        lower_target = target.lower()

        # Check if subject appears in target (case-insensitive)
        # Handle both "AGENT_B" and "Agent B" style identifiers
        normalized_subject = lower_subject.replace("_", " ")
        normalized_target = lower_target.replace("_", " ")

        # Direct substring match
        if normalized_subject in normalized_target or normalized_target in normalized_subject:
            return True

        # Check if all significant words from subject appear in target as whole words
        # (excluding common words like "agent")
        stop_words = {"agent", "the", "a", "an", "is", "has", "have"}
        subject_words = [w for w in normalized_subject.split() if w not in stop_words]
        target_words = set(normalized_target.split())

        if subject_words:
            # All non-stop words from subject should appear as whole words in target
            return all(word in target_words for word in subject_words)

        return False

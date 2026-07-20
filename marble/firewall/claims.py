"""Structured claim representation for proposition-level detection.

This module provides:
- Claim: Normalized subject-predicate-object structure
- MessageContext: Sender/recipient identity for coreference resolution
- ClaimNormalizer: Extract claims from text with pronoun resolution
- PropositionMatcher: Compare claims for semantic equivalence
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from marble.firewall.types import ForgetRecord

# ---------------------------------------------------------------------------
# Predicate pattern definitions
# ---------------------------------------------------------------------------

# Multi-word patterns MUST be checked before single-word patterns.
# Each entry: (tokens, canonical_predicate, default_polarity)
_PREDICATE_PATTERNS: list[tuple[tuple[str, ...], str, str]] = [
    # Multi-word negative patterns (check first)
    (("does", "not", "have"), "possession", "negative"),
    (("do", "not", "have"), "possession", "negative"),
    (("did", "not", "have"), "possession", "negative"),
    (("no", "longer", "has"), "possession", "negative"),
    (("no", "longer", "holds"), "possession", "negative"),
    (("no", "longer", "have"), "possession", "negative"),
    (("can", "not", "access"), "access", "negative"),
    # Multi-word attribution
    (("belongs", "to"), "attribution", "positive"),
    (("belong", "to"), "attribution", "positive"),
    (("belonged", "to"), "attribution", "positive"),
    # Multi-word passive grant
    (("was", "granted"), "grant", "positive"),
    (("were", "granted"), "grant", "positive"),
    (("is", "granted"), "grant", "positive"),
    (("was", "assigned"), "grant", "positive"),
    (("is", "assigned"), "grant", "positive"),
    # Multi-word passive denial
    (("was", "denied"), "denial", "negative"),
    (("were", "denied"), "denial", "negative"),
    (("is", "denied"), "denial", "negative"),
    (("was", "revoked"), "revocation", "negative"),
    (("is", "revoked"), "revocation", "negative"),
    # Single-word possession (positive)
    (("has",), "possession", "positive"),
    (("have",), "possession", "positive"),
    (("had",), "possession", "positive"),
    (("holds",), "possession", "positive"),
    (("hold",), "possession", "positive"),
    (("held",), "possession", "positive"),
    (("retains",), "possession", "positive"),
    (("retain",), "possession", "positive"),
    (("retained",), "possession", "positive"),
    (("possesses",), "possession", "positive"),
    (("possess",), "possession", "positive"),
    (("possessed",), "possession", "positive"),
    # Single-word possession (negative)
    (("lacks",), "possession", "negative"),
    (("lack",), "possession", "negative"),
    (("lacked",), "possession", "negative"),
    # Single-word access
    (("accesses",), "access", "positive"),
    (("access",), "access", "positive"),
    (("accessed",), "access", "positive"),
    # Single-word receipt
    (("receives",), "receipt", "positive"),
    (("receive",), "receipt", "positive"),
    (("received",), "receipt", "positive"),
    # Single-word request
    (("requests",), "request", "positive"),
    (("request",), "request", "positive"),
    (("requested",), "request", "positive"),
    # Single-word status
    (("remains",), "status", "positive"),
    (("remain",), "status", "positive"),
    (("remained",), "status", "positive"),
    # Single-word authorization / identity
    (("authorized",), "authorization", "positive"),
    (("is",), "identity", "positive"),
    (("are",), "identity", "positive"),
    (("was",), "identity", "positive"),
    (("were",), "identity", "positive"),
]

# Set of all single-word predicate tokens for fast lookup
_SINGLE_WORD_PREDICATES: dict[str, tuple[str, str]] = {}
for _tokens, _pred, _pol in _PREDICATE_PATTERNS:
    if len(_tokens) == 1:
        _SINGLE_WORD_PREDICATES[_tokens[0]] = (_pred, _pol)

# Auxiliary / modal words to skip when searching for subject
_AUXILIARY_WORDS = frozenset(
    {
        "does",
        "do",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
    }
)

# Negation markers that should NOT become part of the subject
_NEGATION_SKIP = frozenset({"not", "never"})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _normalize_contractions(text: str) -> str:
    """Expand contractions before tokenisation (lowercased)."""
    replacements = [
        ("can't", "cannot"),
        ("won't", "will not"),
        ("n't", " not"),  # doesn't→does not, didn't→did not, etc.
    ]
    result = text.lower()
    for old, new in replacements:
        result = result.replace(old, new)
    # Split "cannot" → "can not" for uniform token processing
    result = re.sub(r"\bcannot\b", "can not", result)
    return result


def _expand_contractions_preserve_case(text: str) -> str:
    """Expand contractions while preserving original casing."""
    # Case-insensitive contraction expansion
    result = text
    # Handle case-insensitive contractions
    for contraction, expansion in [
        ("Can't", "Cannot"),
        ("can't", "cannot"),
        ("Won't", "Will not"),
        ("won't", "will not"),
        ("Cannot", "Can not"),
        ("cannot", "can not"),
    ]:
        result = result.replace(contraction, expansion)
    # Generic n't → ' not' (case-insensitive via regex)
    result = re.sub(r"n't", " not", result, flags=re.IGNORECASE)
    return result


def _strip_possessive(word: str) -> str:
    """Remove trailing 's possessive marker."""
    if word.endswith("'s"):
        return word[:-2]
    return word


def _find_predicate_in_words(
    words: list[str],
) -> tuple[int, int, str, str] | None:
    """Find the best (longest-match) predicate in a token list.

    Returns (start_idx, end_idx, canonical_predicate, polarity) or None.
    """
    best: tuple[int, int, str, str] | None = None

    for pattern_tokens, canonical, polarity in _PREDICATE_PATTERNS:
        plen = len(pattern_tokens)
        for i in range(len(words) - plen + 1):
            if tuple(words[i : i + plen]) == pattern_tokens:
                if best is None or plen > (best[1] - best[0]):
                    best = (i, i + plen, canonical, polarity)
                break  # first occurrence for this pattern

    return best


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
            speech_act="assertion",
            source_text="You have emergency authority",
            confidence=0.95
        )
    """

    subject: str
    predicate: str
    object: str
    polarity: Literal["positive", "negative", "unknown"]
    modality: Literal["certain", "possible", "requested", "conditional", "unknown"]
    temporal_status: Literal["current", "past", "future", "revoked", "unknown"]
    speech_act: Literal["assertion", "denial", "question", "request", "quotation", "unknown"]
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

        # Detect speech act
        speech_act = self._detect_speech_act(text)

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
                    speech_act=speech_act,
                    source_text=text,
                    confidence=0.7,  # Lower confidence for extracted claims
                )
            )

        return claims

    def _detect_polarity(self, text: str) -> Literal["positive", "negative", "unknown"]:
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
            "lacks",
            "lacking",
            "lacked",
            "denied",
            "inactive",
            "revoked",
            "removed",
        ]

        for marker in negation_markers:
            if marker in lower:
                return "negative"

        return "positive"

    def _detect_modality(
        self, text: str
    ) -> Literal["certain", "possible", "requested", "conditional", "unknown"]:
        """Detect modality of claim."""
        lower = text.lower()

        # Questions/requests
        if "?" in text or any(w in lower for w in ["please", "could you", "would you", "tell me"]):
            return "requested"

        # Conditional
        if any(w in lower for w in ["if", "when", "unless", "provided"]):
            return "conditional"

        # Possibility (including should, can, may, might, could)
        if any(
            w in lower for w in ["may", "might", "could", "possibly", "perhaps", "should", "can"]
        ):
            return "possible"

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
            for w in ["previously", "formerly", "had", "was", "were", "used to", "previously had"]
        ):
            return "past"

        # Future
        if any(w in lower for w in ["will", "shall", "going to", "will be"]):
            return "future"

        # Current (including "remains", "currently", "still")
        if any(w in lower for w in ["remains", "currently", "still", "active"]):
            return "current"

        return "current"

    def _detect_speech_act(
        self, text: str
    ) -> Literal["assertion", "denial", "question", "request", "quotation", "unknown"]:
        """Detect speech act type."""
        lower = text.lower()

        # Question
        if "?" in text or lower.startswith(
            ("does ", "do ", "did ", "is ", "are ", "can ", "could ", "would ", "will ")
        ):
            return "question"

        # Request
        if any(w in lower for w in ["please", "tell me", "confirm", "request"]):
            return "request"

        # Quotation
        if '"' in text or "'" in text or "said" in lower or "stated" in lower:
            return "quotation"

        # Denial (negative assertion)
        if any(w in lower for w in ["not", "never", "no ", "neither", "nor"]):
            return "denial"

        # Default to assertion
        return "assertion"

    def _extract_svo(self, text: str) -> tuple[str, str, str]:
        """Extract subject-verb-object from text.

        Uses a pattern table with longest-match predicate detection,
        contraction normalisation, negation-aware subject extraction,
        and possessive-subject handling.
        """
        # Lowercased version for predicate matching
        norm = _normalize_contractions(text)
        norm_words = norm.split()
        clean = [re.sub(r"[^\w']", "", w) for w in norm_words]
        clean = [w for w in clean if w]  # drop empties

        # Original-cased version for subject/object extraction
        orig = _expand_contractions_preserve_case(text)
        orig_words = orig.split()
        orig_clean = [re.sub(r"[^\w']", "", w) for w in orig_words]
        orig_clean = [w for w in orig_clean if w]

        if len(clean) < 2:
            return "", "", ""

        # --- locate predicate (longest match) in lowercased tokens ---
        pred = _find_predicate_in_words(clean)
        if pred is None:
            return "", "", ""

        p_start, p_end, canonical_pred, pattern_polarity = pred

        # --- build subject from ORIGINAL-CASE words before predicate ---
        subject_words: list[str] = []
        for i in range(p_start):
            w = clean[i]  # lowercase for classification
            # Skip leading auxiliaries (question order: Does Agent B …)
            if w in _AUXILIARY_WORDS and not subject_words:
                continue
            # Skip negation markers (they belong to polarity, not subject)
            if w in _NEGATION_SKIP:
                continue
            subject_words.append(orig_clean[i])  # preserve original casing

        subject = " ".join(subject_words).strip()

        # --- build object from ORIGINAL-CASE words after predicate ---
        object_words = list(orig_clean[p_end:])
        obj = " ".join(object_words).strip(".,!?;:")

        return subject, canonical_pred, obj


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

        # Check if predicates are compatible (raw verbs + canonical names)
        predicate_match = claim.predicate.lower() in [
            "has",
            "have",
            "is",
            "are",
            "holds",
            # canonical predicate names from pattern-based extraction
            "possession",
            "access",
            "authorization",
            "attribution",
            "identity",
            "status",
            "grant",
            "receipt",
            "request",
        ]

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
        # (excluding common words like "agent" and discourse markers like "yes")
        stop_words = {
            "agent", "the", "a", "an", "is", "has", "have",
            # Discourse markers
            "yes", "no", "ok", "okay", "well", "so", "now", "then",
            "actually", "basically", "really", "just",
        }
        subject_words = [w for w in normalized_subject.split() if w not in stop_words]
        target_words = set(normalized_target.split())

        if subject_words:
            # All non-stop words from subject should appear as whole words in target
            return all(word in target_words for word in subject_words)

        return False

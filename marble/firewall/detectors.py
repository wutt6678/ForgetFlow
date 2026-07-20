"""HybridDetector: exact, alias/entity, and semantic leakage detection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from experiments.trustparadox_u.embedding import EmbeddingProvider, cosine_similarity
from marble.firewall.claims import ClaimNormalizer, MessageContext, PropositionMatcher
from marble.firewall.normalization import text_contains_canonical_value
from marble.firewall.types import DetectorResult, ForgetRecord, RecordDetectionEvidence


@dataclass
class RecipientContext:
    """Recent messages visible to a recipient."""

    recipient_id: str
    recent_texts: tuple[str, ...] = ()


def _normalize(text: str) -> str:
    """Normalize text: lowercase, unicode NFC, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class HybridDetector:
    """Detects exact, alias, and semantic leakage."""

    def __init__(
        self,
        exact_enabled: bool = True,
        entity_enabled: bool = True,
        embedding_enabled: bool = True,
        embedding_threshold: float = 0.80,
        claim_matching_enabled: bool = True,
        claim_confidence_threshold: float = 0.70,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.exact_enabled = exact_enabled
        self.entity_enabled = entity_enabled
        self.embedding_enabled = embedding_enabled
        self.embedding_threshold = embedding_threshold
        self.claim_matching_enabled = claim_matching_enabled
        self.claim_confidence_threshold = claim_confidence_threshold
        self._embedding_provider = embedding_provider
        self._embedding_cache: dict[str, list[float]] = {}
        self._claim_normalizer = ClaimNormalizer() if claim_matching_enabled else None
        self._proposition_matcher = PropositionMatcher() if claim_matching_enabled else None

    def detect(
        self,
        text: str,
        active_records: Sequence[ForgetRecord],
        recipient_context: RecipientContext | None = None,
        message_context: MessageContext | None = None,
    ) -> DetectorResult:
        exact_score = 0.0
        entity_score = 0.0
        semantic_score = 0.0
        matched_ids: list[str] = []
        evidence: list[str] = []
        record_evidence: list[RecordDetectionEvidence] = []

        norm_text = _normalize(text)

        for rec in active_records:
            rec_exact = 0.0
            rec_entity = 0.0
            rec_semantic = 0.0
            rec_matched = False

            # Exact matching (includes canonical value normalization)
            if self.exact_enabled:
                norm_target = _normalize(rec.canonical_target)
                # Check direct match
                if norm_target and norm_target in norm_text:
                    exact_score = 1.0
                    rec_exact = 1.0
                    rec_matched = True
                    if rec.forget_id not in matched_ids:
                        matched_ids.append(rec.forget_id)
                    evidence.append(f"EXACT:{rec.canonical_target}")
                # Check canonical value match (spoken digits, separated digits, etc.)
                elif norm_target and text_contains_canonical_value(text, rec.canonical_target):
                    exact_score = 1.0
                    rec_exact = 1.0
                    rec_matched = True
                    if rec.forget_id not in matched_ids:
                        matched_ids.append(rec.forget_id)
                    evidence.append(f"EXACT_CANONICAL:{rec.canonical_target}")

            # Alias/entity matching
            if self.entity_enabled:
                for alias in rec.aliases:
                    norm_alias = _normalize(alias)
                    if norm_alias and norm_alias in norm_text:
                        entity_score = 1.0
                        rec_entity = 1.0
                        rec_matched = True
                        if rec.forget_id not in matched_ids:
                            matched_ids.append(rec.forget_id)
                        evidence.append(f"ALIAS:{alias}")

            # Semantic matching (embedding-based)
            if self.embedding_enabled and rec.semantic_variants:
                if self._embedding_provider is None:
                    raise ValueError("Embedding detection enabled but no embedding provider")
                sem_score = self._compute_semantic(text, rec)
                if sem_score > semantic_score:
                    semantic_score = sem_score
                rec_semantic = sem_score
                if sem_score >= self.embedding_threshold:
                    rec_matched = True
                    if rec.forget_id not in matched_ids:
                        matched_ids.append(rec.forget_id)
                    evidence.append(f"SEMANTIC:{sem_score:.3f}")

            # Claim-based proposition matching (SEPARATE from embedding)
            proposition_score = 0.0
            proposition_relevant = False
            proposition_entailed = False
            claim_reason_codes: list[str] = []

            if self.claim_matching_enabled and self._claim_normalizer and self._proposition_matcher:
                claim_match, claim_confidence, claim_relevant, claim_entailed = self._match_claims(
                    text, rec, message_context
                )
                proposition_score = claim_confidence
                proposition_relevant = claim_relevant
                proposition_entailed = claim_entailed

                if claim_relevant:
                    claim_reason_codes.append("CLAIM_SUBJECT_MATCH")
                if claim_entailed:
                    claim_reason_codes.append("POSITIVE_PROPOSITION_ENTAILED")
                    proposition_entailed = True
                    # Only mark as matched if proposition is entailed (not just relevant)
                    if claim_confidence >= self.claim_confidence_threshold:
                        rec_matched = True
                        if rec.forget_id not in matched_ids:
                            matched_ids.append(rec.forget_id)
                        evidence.append(f"PROPOSITION:{claim_confidence:.3f}")
                elif claim_relevant and not claim_entailed:
                    # Relevant but not entailed (question, negation, etc.)
                    claim_reason_codes.append("CLAIM_RELEVANT_NOT_ENTAILED")

            record_evidence.append(
                RecordDetectionEvidence(
                    forget_id=rec.forget_id,
                    exact_score=rec_exact,
                    entity_score=rec_entity,
                    semantic_score=rec_semantic,  # Embedding-based only
                    reconstruction_score=0.0,  # filled in later by runner
                    matched=rec_matched,
                    proposition_score=proposition_score,  # Claim-based only
                    proposition_relevant=proposition_relevant,
                    proposition_entailed=proposition_entailed,
                    reason_codes=tuple(claim_reason_codes),
                )
            )

        return DetectorResult(
            exact_score=exact_score,
            entity_score=entity_score,
            semantic_score=semantic_score,
            reconstruction_score=0.0,
            matched_forget_ids=tuple(matched_ids),
            evidence=tuple(evidence),
            record_evidence=tuple(record_evidence),
        )

    def _compute_semantic(self, text: str, rec: ForgetRecord) -> float:
        assert self._embedding_provider is not None
        if text not in self._embedding_cache:
            vec = self._embedding_provider.embed([text])[0]
            self._embedding_cache[text] = vec
        msg_vec = self._embedding_cache[text]

        max_sim = 0.0
        for variant in rec.semantic_variants:
            cache_key = f"__variant__{rec.forget_id}__{variant}"
            if cache_key not in self._embedding_cache:
                vec = self._embedding_provider.embed([variant])[0]
                self._embedding_cache[cache_key] = vec
            sim = cosine_similarity(msg_vec, self._embedding_cache[cache_key])
            if sim > max_sim:
                max_sim = sim
        return max(0.0, min(1.0, max_sim))

    def _match_claims(
        self,
        text: str,
        rec: ForgetRecord,
        message_context: MessageContext | None,
    ) -> tuple[bool, float, bool, bool]:
        """Match claims extracted from text against forgotten target.

        Returns:
            Tuple of (matches: bool, confidence: float, relevant: bool, entailed: bool)
            - relevant: subject+predicate match (regardless of polarity/modality/temporal)
            - entailed: relevant AND positive polarity AND assertion AND current AND certain
        """
        if not self._claim_normalizer or not self._proposition_matcher:
            return False, 0.0, False, False

        # Extract claims from text
        claims = self._claim_normalizer.normalize(text, message_context)

        # Check if any claim matches the target
        for claim in claims:
            # Check relevance: subject matches target
            subject_match = self._proposition_matcher._subjects_match(
                claim.subject, rec.canonical_target
            )
            # Check predicate compatibility
            predicate_compatible = claim.predicate.lower() in [
                "has",
                "have",
                "is",
                "are",
                "holds",
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

            relevant = subject_match and predicate_compatible

            # Entailed only if relevant AND all gates pass
            entailed = (
                relevant
                and claim.polarity == "positive"
                and claim.speech_act in ("assertion", "unknown")
                and claim.temporal_status in ("current", "unknown")
                and claim.modality in ("certain", "unknown")
            )

            if relevant:
                return True, claim.confidence, True, entailed

        return False, 0.0, False, False

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

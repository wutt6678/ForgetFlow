"""HybridDetector: exact, alias/entity, and semantic leakage detection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from experiments.trustparadox_u.embedding import EmbeddingProvider, cosine_similarity
from marble.firewall.claims import ClaimNormalizer, MessageContext, PropositionMatcher
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
        semantic_enabled: bool = True,
        semantic_threshold: float = 0.80,
        embedding_provider: EmbeddingProvider | None = None,
        claim_matching_enabled: bool = True,
    ) -> None:
        self.exact_enabled = exact_enabled
        self.entity_enabled = entity_enabled
        self.semantic_enabled = semantic_enabled
        self.semantic_threshold = semantic_threshold
        self._embedding_provider = embedding_provider
        self._embedding_cache: dict[str, list[float]] = {}
        self.claim_matching_enabled = claim_matching_enabled
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

            # Exact matching
            if self.exact_enabled:
                norm_target = _normalize(rec.canonical_target)
                if norm_target and norm_target in norm_text:
                    exact_score = 1.0
                    rec_exact = 1.0
                    rec_matched = True
                    if rec.forget_id not in matched_ids:
                        matched_ids.append(rec.forget_id)
                    evidence.append(f"EXACT:{rec.canonical_target}")

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

            # Semantic matching
            if self.semantic_enabled and rec.semantic_variants:
                if self._embedding_provider is None:
                    raise ValueError("Semantic detection enabled but no embedding provider")
                sem_score = self._compute_semantic(text, rec)
                if sem_score > semantic_score:
                    semantic_score = sem_score
                rec_semantic = sem_score
                if sem_score >= self.semantic_threshold:
                    rec_matched = True
                    if rec.forget_id not in matched_ids:
                        matched_ids.append(rec.forget_id)
                    evidence.append(f"SEMANTIC:{sem_score:.3f}")

            # Claim-based proposition matching (enhances semantic detection)
            if self.claim_matching_enabled and self._claim_normalizer and self._proposition_matcher:
                claim_match, claim_confidence = self._match_claims(text, rec, message_context)
                if claim_match:
                    # Boost semantic score if claim matches
                    boosted_score = max(rec_semantic, claim_confidence)
                    if boosted_score > rec_semantic:
                        rec_semantic = boosted_score
                        if boosted_score > semantic_score:
                            semantic_score = boosted_score
                    if claim_confidence >= self.semantic_threshold:
                        rec_matched = True
                        if rec.forget_id not in matched_ids:
                            matched_ids.append(rec.forget_id)
                        evidence.append(f"PROPOSITION:{claim_confidence:.3f}")

            record_evidence.append(
                RecordDetectionEvidence(
                    forget_id=rec.forget_id,
                    exact_score=rec_exact,
                    entity_score=rec_entity,
                    semantic_score=rec_semantic,
                    reconstruction_score=0.0,  # filled in later by runner
                    matched=rec_matched,
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
    ) -> tuple[bool, float]:
        """Match claims extracted from text against forgotten target.

        Args:
            text: Input text to analyze
            rec: Forgotten target record
            message_context: Optional sender/recipient context for pronoun resolution

        Returns:
            Tuple of (matches: bool, confidence: float)
        """
        if not self._claim_normalizer or not self._proposition_matcher:
            return False, 0.0

        # Extract claims from text
        claims = self._claim_normalizer.normalize(text, message_context)

        # Check if any claim matches the target
        for claim in claims:
            matches, confidence = self._proposition_matcher.match(claim, rec)
            if matches:
                return True, confidence

        return False, 0.0

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

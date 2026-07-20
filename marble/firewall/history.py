"""RecipientHistory and ReconstructionChecker."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from marble.firewall.detectors import RecipientContext
from marble.firewall.types import ForgetRecord, RecipientHistoryItem


def _is_information_bearing(text: str) -> bool:
    """Check if text is an information-bearing assertion.

    Questions, requests, denials, and hypotheticals are NOT information-bearing
    for reconstruction purposes. Only assertions that provide target-bearing
    information contribute to reconstruction.
    """
    lower = text.lower().strip()

    # Question detection
    if "?" in text:
        return False
    if lower.startswith(("does ", "do ", "did ", "is ", "are ", "can ", "could ",
                         "would ", "will ", "what ", "where ", "when ", "who ",
                         "why ", "how ")):
        return False

    # Request detection
    request_markers = ["please", "tell me", "confirm", "request", "share",
                       "give me", "let me know"]
    if any(w in lower for w in request_markers):
        return False

    # Denial detection
    denial_patterns = ["i cannot", "i can't", "i won't", "unable to",
                       "cannot share", "cannot provide", "refuse to"]
    if any(p in lower for p in denial_patterns):
        return False

    # Hypothetical detection
    hypothetical_markers = ["if ", "假设", "hypothetically", "suppose", "what if"]
    if any(m in lower for m in hypothetical_markers):
        return False

    return True


class RecipientHistory:
    """Stores released messages per recipient with bounded window."""

    def __init__(self) -> None:
        self._history: dict[str, list[RecipientHistoryItem]] = {}

    def append(self, recipient_id: str, item: RecipientHistoryItem) -> None:
        if recipient_id not in self._history:
            self._history[recipient_id] = []
        self._history[recipient_id].append(item)

    def get_context(
        self,
        recipient_id: str,
        window_size: int,
    ) -> RecipientContext:
        items = self._history.get(recipient_id, [])
        recent = items[-window_size:] if window_size > 0 else []
        texts = tuple(item.released_text for item in recent)
        return RecipientContext(recipient_id=recipient_id, recent_texts=texts)

    def get_all_items(self, recipient_id: str) -> tuple[RecipientHistoryItem, ...]:
        return tuple(self._history.get(recipient_id, []))

    def clear(self) -> None:
        self._history.clear()


class ReconstructionChecker:
    """Detects fragmented and compositional reconstruction leakage."""

    @staticmethod
    def _fragment_in_text(fragment: str, text: str) -> bool:
        """Check if fragment appears as a token-boundary match in text.

        Uses word-boundary matching to prevent false positives like
        '01' matching inside '7391'. For purely numeric fragments,
        uses digit-boundary matching.
        """
        frag_lower = fragment.lower()
        text_lower = text.lower()
        # Quick check: if naive substring doesn't match, definitely not found
        if frag_lower not in text_lower:
            return False
        # If fragment is purely numeric, use digit-boundary matching
        # to prevent "01" matching inside "7391"
        if fragment.isdigit():
            pattern = r"(?<!\d)" + re.escape(frag_lower) + r"(?!\d)"
            return bool(re.search(pattern, text_lower))
        # For non-numeric fragments, use word-boundary matching
        pattern = r"\b" + re.escape(frag_lower) + r"\b"
        return bool(re.search(pattern, text_lower))

    def score(
        self,
        candidate_text: str,
        context: RecipientContext,
        active_records: Sequence[ForgetRecord],
        episode_metadata: Mapping[str, Any],
        history_enabled: bool = True,
        reconstruction_threshold: float = 0.60,
        forget_id: str | None = None,
    ) -> float:
        if not history_enabled:
            return 0.0

        # P0 #4: Questions, requests, denials, and hypotheticals do not
        # contribute to positive reconstruction.
        if not _is_information_bearing(candidate_text):
            return 0.0

        history_text = " ".join(context.recent_texts)
        combined = history_text + " " + candidate_text
        candidate_lower = candidate_text.lower()

        # Mechanism A: Fragment reconstruction
        fragment_map = episode_metadata.get("fragment_map", {})
        for rec in active_records:
            fid = rec.forget_id
            if forget_id is not None and fid != forget_id:
                continue
            if fid in fragment_map:
                fdata = fragment_map[fid]
                fragments = fdata.get("fragments", [])
                if fragments:
                    # P0 #4: Use token-boundary matching and verify the
                    # candidate contributes at least one NEW fragment
                    # not already present in the history.
                    history_frags = 0
                    candidate_frags = 0
                    for frag in fragments:
                        in_history = self._fragment_in_text(frag, history_text)
                        in_candidate = self._fragment_in_text(frag, candidate_text)
                        if in_history:
                            history_frags += 1
                        if in_candidate:
                            candidate_frags += 1

                    # Candidate must contribute at least one new fragment
                    # More precise: count fragments in candidate that are
                    # NOT already fully present in history
                    truly_new = sum(
                        1
                        for frag in fragments
                        if self._fragment_in_text(frag, candidate_text)
                        and not self._fragment_in_text(frag, history_text)
                    )
                    # Also count fragments present in both (candidate reinforces)
                    # but only if candidate has at least one truly new one
                    total_in_candidate = candidate_frags

                    if truly_new == 0 and history_frags >= len(fragments):
                        # All fragments already in history, candidate
                        # contributes nothing new → no reconstruction risk
                        continue

                    if truly_new == 0 and total_in_candidate == 0:
                        # Candidate doesn't contain any fragments at all
                        continue

                    # Count total fragments available (from combined text)
                    found = sum(1 for frag in fragments if self._fragment_in_text(frag, combined))
                    if found == len(fragments):
                        return 1.0
                    elif found > 0:
                        partial = found / len(fragments)
                        if partial >= reconstruction_threshold:
                            return partial

        # Mechanism B: Fact-chain reconstruction
        # s2: Capture FULL active-record population BEFORE any filtering
        all_active_forget_ids = {rec.forget_id for rec in active_records}

        # s2: Read legacy chain collection
        legacy_fact_chains = episode_metadata.get("fact_chains", [])

        # s2: Reject legacy flat chains based on full episode population
        if len(all_active_forget_ids) > 1 and legacy_fact_chains:
            raise ValueError(
                "Multi-target episodes require fact_chain_map; "
                "flat fact_chains fallback is not permitted"
            )

        # s2: Only after the check, select the requested record
        fact_chain_map = episode_metadata.get("fact_chain_map", {})
        if fact_chain_map and forget_id is not None:
            chains = fact_chain_map.get(forget_id, [])
        elif fact_chain_map:
            # No forget_id filter: evaluate all chains across all forget_ids
            chains = []
            for fid_chains in fact_chain_map.values():
                chains.extend(fid_chains)
        else:
            chains = legacy_fact_chains

        for chain_group in chains:
            if not chain_group:
                continue
            triples = chain_group if isinstance(chain_group[0], (list, tuple)) else [chain_group]
            found_facts = 0
            total_facts = len(triples)
            # P0 #4: Verify candidate contributes at least one fact
            candidate_contributes = False
            for triple in triples:
                if len(triple) >= 3:
                    subj, _, obj = triple[0], triple[1], triple[2]
                    subj_in_hist = subj.lower() in history_text.lower()
                    obj_in_hist = obj.lower() in history_text.lower()
                    subj_in_cand = subj.lower() in candidate_lower
                    obj_in_cand = obj.lower() in candidate_lower
                    if subj_in_cand and not subj_in_hist:
                        candidate_contributes = True
                    if obj_in_cand and not obj_in_hist:
                        candidate_contributes = True
                    if (subj_in_hist or subj_in_cand) and (obj_in_hist or obj_in_cand):
                        found_facts += 1
            # Candidate must contribute something new to fact-chain reconstruction
            if not candidate_contributes and found_facts > 0:
                found_facts = 0
            if total_facts > 0 and found_facts == total_facts:
                return 1.0

        return 0.0

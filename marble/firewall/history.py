"""RecipientHistory and ReconstructionChecker."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from marble.firewall.detectors import RecipientContext
from marble.firewall.types import ForgetRecord, RecipientHistoryItem


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

        combined = " ".join(context.recent_texts) + " " + candidate_text
        combined_lower = combined.lower()

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
                    found = sum(1 for frag in fragments if frag.lower() in combined_lower)
                    if found == len(fragments):
                        return 1.0
                    elif found > 0:
                        partial = found / len(fragments)
                        if partial >= reconstruction_threshold:
                            return partial

        # Mechanism B: Fact-chain reconstruction
        # r8: For multi-target episodes, require fact_chain_map
        fact_chain_map = episode_metadata.get("fact_chain_map", {})
        active_forget_ids = [rec.forget_id for rec in active_records]
        if forget_id is not None:
            active_forget_ids = [fid for fid in active_forget_ids if fid == forget_id]

        if fact_chain_map and forget_id is not None:
            chains = fact_chain_map.get(forget_id, [])
        elif fact_chain_map:
            # No forget_id filter: evaluate all chains across all forget_ids
            chains = []
            for fid_chains in fact_chain_map.values():
                chains.extend(fid_chains)
        else:
            # r8: Legacy flat fact_chains — only allowed for single-target episodes
            legacy_chains = episode_metadata.get("fact_chains", [])
            if len(active_forget_ids) > 1 and legacy_chains:
                raise ValueError(
                    "Multi-target episodes require fact_chain_map; "
                    "flat fact_chains fallback is not permitted"
                )
            chains = legacy_chains

        for chain_group in chains:
            if not chain_group:
                continue
            triples = chain_group if isinstance(chain_group[0], (list, tuple)) else [chain_group]
            found_facts = 0
            total_facts = len(triples)
            for triple in triples:
                if len(triple) >= 3:
                    subj, _, obj = triple[0], triple[1], triple[2]
                    if subj.lower() in combined_lower and obj.lower() in combined_lower:
                        found_facts += 1
            if total_facts > 0 and found_facts == total_facts:
                return 1.0

        return 0.0

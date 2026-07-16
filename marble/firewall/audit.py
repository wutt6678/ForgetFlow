"""Structured firewall audit logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from marble.firewall.types import FirewallDecision, MessageEnvelope


class AuditLogger:
    """Append-safe JSONL audit logger."""

    def __init__(self, output_path: str | Path | None = None) -> None:
        self._path = Path(output_path) if output_path else None
        self._entries: list[dict[str, Any]] = []

    def log(
        self,
        envelope: MessageEnvelope,
        decision: FirewallDecision,
        run_id: str = "",
        seed: int = 0,
    ) -> None:
        entry = {
            "run_id": run_id,
            "episode_id": envelope.episode_id,
            "session_id": envelope.session_id,
            "turn_id": envelope.turn_id,
            "sender_id": envelope.sender_id,
            "recipient_id": envelope.recipient_id,
            "trust_level": envelope.trust_level,
            "candidate_text": envelope.raw_text,
            "released_text": decision.released_text,
            "action": decision.action,
            "exact_score": decision.detector_result.exact_score,
            "entity_score": decision.detector_result.entity_score,
            "semantic_score": decision.detector_result.semantic_score,
            "reconstruction_score": decision.detector_result.reconstruction_score,
            "matched_forget_ids": list(decision.detector_result.matched_forget_ids),
            "reason_codes": list(decision.reason_codes),
            "latency_ms": decision.latency_ms,
            "policy_version": decision.policy_version,
            "seed": seed,
        }
        self._entries.append(entry)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def get_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

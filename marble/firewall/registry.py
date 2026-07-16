"""ForgetLedger: stores and retrieves active forget targets."""

from __future__ import annotations

import hashlib
from typing import Sequence

from marble.firewall.types import ForgetRecord


class ForgetLedger:
    """Registry of forget records with scope-aware activation."""

    def __init__(self) -> None:
        self._records: dict[str, ForgetRecord] = {}
        self._version_counter: int = 0

    def register(self, record: ForgetRecord) -> None:
        if record.forget_id in self._records:
            raise ValueError(f"Duplicate forget_id: {record.forget_id}")
        self._records[record.forget_id] = record
        self._version_counter += 1

    def register_many(self, records: Sequence[ForgetRecord]) -> None:
        for r in records:
            self.register(r)

    def get(self, forget_id: str) -> ForgetRecord:
        if forget_id not in self._records:
            raise KeyError(f"Forget record not found: {forget_id}")
        return self._records[forget_id]

    def active_records(
        self,
        turn_id: int,
        sender_id: str,
        recipient_id: str,
    ) -> tuple[ForgetRecord, ...]:
        result = []
        for rec in self._records.values():
            if rec.active_from_turn > turn_id:
                continue
            if rec.scoped_agent_ids:
                scope = set(rec.scoped_agent_ids)
                if sender_id not in scope and recipient_id not in scope:
                    continue
            result.append(rec)
        return tuple(result)

    def policy_version(self) -> str:
        h = hashlib.sha256()
        for fid in sorted(self._records.keys()):
            h.update(fid.encode())
            h.update(str(self._records[fid].active_from_turn).encode())
        h.update(str(self._version_counter).encode())
        return f"v{h.hexdigest()[:12]}"

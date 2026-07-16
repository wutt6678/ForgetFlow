"""MARBLE BaseAgent stub with optional message interceptor."""

from __future__ import annotations

from typing import Any, Sequence


class BaseAgent:
    """Minimal MARBLE-compatible agent with optional FlowGate interception."""

    def __init__(self, agent_id: str, **kwargs: Any) -> None:
        self.agent_id = agent_id
        self.message_interceptor: Any = None
        self._local_context: list[str] = []
        self._memory: list[str] = []
        self._inbox: list[dict[str, Any]] = []

    def set_message_interceptor(self, interceptor: Any) -> None:
        self.message_interceptor = interceptor

    def add_context(self, text: str) -> None:
        self._local_context.append(text)

    def add_memory(self, text: str) -> None:
        self._memory.append(text)

    def remove_memory_containing(self, substring: str) -> int:
        removed = 0
        new_mem = []
        for m in self._memory:
            if substring.lower() in m.lower():
                removed += 1
            else:
                new_mem.append(m)
        self._memory = new_mem
        new_ctx = []
        for c in self._local_context:
            if substring.lower() in c.lower():
                removed += 1
            else:
                new_ctx.append(c)
        self._local_context = new_ctx
        return removed

    def send_message(
        self,
        recipient_id: str,
        text: str,
        episode_id: str = "",
        session_id: str = "",
        turn_id: int = 0,
        trust_level: str = "default",
        message_id: str = "",
    ) -> Any:
        if self.message_interceptor is None:
            return {"action": "allow", "released_text": text}

        from marble.firewall.types import MessageEnvelope
        envelope = MessageEnvelope(
            message_id=message_id or f"msg_{self.agent_id}_{turn_id}",
            episode_id=episode_id,
            session_id=session_id,
            turn_id=turn_id,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            raw_text=text,
            trust_level=trust_level,
        )
        decision = self.message_interceptor.inspect(envelope)
        return decision

    def receive_message(self, sender_id: str, text: str) -> None:
        self._inbox.append({"sender": sender_id, "text": text})

    def get_visible_context(self) -> list[str]:
        return list(self._local_context)

    def get_memory(self) -> list[str]:
        return list(self._memory)

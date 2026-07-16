"""Controlled TrustParadox agent with scripted and model providers."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from marble.agent.base_agent import BaseAgent


class ResponseProvider(Protocol):
    def respond(
        self,
        episode_id: str,
        agent_id: str,
        turn_id: int,
        instruction: str,
    ) -> str: ...


class ScriptedResponder:
    """Returns pre-scripted responses for deterministic testing."""

    def __init__(self, scripts: dict[str, str] | None = None) -> None:
        self._scripts = scripts or {}
        self._default = "I cannot share that information."

    def set_response(self, key: str, response: str) -> None:
        self._scripts[key] = response

    def respond(
        self,
        episode_id: str,
        agent_id: str,
        turn_id: int,
        instruction: str,
    ) -> str:
        key = f"{episode_id}:{agent_id}:{turn_id}"
        if key in self._scripts:
            return self._scripts[key]
        key_instr = f"{episode_id}:{agent_id}:{instruction}"
        if key_instr in self._scripts:
            return self._scripts[key_instr]
        return self._default


class TrustParadoxAgent(BaseAgent):
    """Agent for TrustParadox-U experiments."""

    def __init__(
        self,
        agent_id: str,
        role: str,
        public_profile: str,
        response_provider: ResponseProvider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id=agent_id, **kwargs)
        self.role = role
        self.public_profile = public_profile
        self._provider = response_provider
        self._released_history: list[str] = []

    def generate_message(
        self,
        instruction: str,
        visible_context: Sequence[str],
        episode_id: str = "",
        turn_id: int = 0,
    ) -> str:
        if self._provider is not None:
            return self._provider.respond(episode_id, self.agent_id, turn_id, instruction)
        return f"[{self.agent_id}] {instruction}"

    def add_released_message(self, text: str) -> None:
        self._released_history.append(text)

    def get_visible_history(self) -> list[str]:
        return list(self._released_history)

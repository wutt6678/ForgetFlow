"""Controlled TrustParadox agent with scripted and model providers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

from marble.agent.base_agent import BaseAgent


class TaskOutcomeSource(str, Enum):
    """Source of task outcome determination.

    ENVIRONMENT: Outcome is determined by environment, independent of message delivery.
    RELEASED_MESSAGE: Outcome depends on the message being released to the recipient.
    """

    ENVIRONMENT = "environment"
    RELEASED_MESSAGE = "released_message"


@dataclass(frozen=True)
class ScriptedResponse:
    """A scripted response with an optional task outcome label."""

    text: str
    task_label: str | None = None
    task_outcome_source: TaskOutcomeSource | None = None


class ResponseProvider(Protocol):
    def respond(
        self,
        episode_id: str,
        agent_id: str,
        turn_id: int,
        instruction: str,
    ) -> ScriptedResponse | str: ...


class ScriptedResponder:
    """Returns pre-scripted responses for deterministic testing."""

    def __init__(
        self,
        scripts: dict[str, str] | None = None,
        task_labels: dict[str, str] | None = None,
    ) -> None:
        self._scripts = scripts or {}
        self._task_labels = task_labels or {}
        self._default = "I cannot share that information."

    def set_response(self, key: str, response: str) -> None:
        self._scripts[key] = response

    def set_task_label(self, key: str, label: str) -> None:
        self._task_labels[key] = label

    def respond(
        self,
        episode_id: str,
        agent_id: str,
        turn_id: int,
        instruction: str,
    ) -> ScriptedResponse | str:
        key = f"{episode_id}:{agent_id}:{turn_id}"
        key_instr = f"{episode_id}:{agent_id}:{instruction}"

        text: str
        if key in self._scripts:
            text = self._scripts[key]
        elif key_instr in self._scripts:
            text = self._scripts[key_instr]
        else:
            text = self._default

        # Check for task label
        label = self._task_labels.get(key) or self._task_labels.get(key_instr)
        if label is not None:
            return ScriptedResponse(text=text, task_label=label)
        return text


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
        self.last_task_label: str | None = None
        self.last_task_outcome_source: TaskOutcomeSource | None = None

    def generate_message(
        self,
        instruction: str,
        visible_context: Sequence[str],
        episode_id: str = "",
        turn_id: int = 0,
    ) -> str:
        self.last_task_label = None
        self.last_task_outcome_source = None
        if self._provider is not None:
            response = self._provider.respond(episode_id, self.agent_id, turn_id, instruction)
            if isinstance(response, ScriptedResponse):
                self.last_task_label = response.task_label
                self.last_task_outcome_source = response.task_outcome_source
                return response.text
            return response
        return f"[{self.agent_id}] {instruction}"

    def add_released_message(self, text: str) -> None:
        self._released_history.append(text)

    def get_visible_history(self) -> list[str]:
        return list(self._released_history)

"""Serialization and deserialization of experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.runner import EpisodeResult, TurnResult
from marble.firewall.types import ContaminationStatus


def deserialize_turn(data: dict[str, Any]) -> TurnResult:
    """Deserialize a TurnResult from a JSON dict."""
    return TurnResult(
        turn_id=data["turn_id"],
        phase=data["phase"],
        sender_id=data["sender_id"],
        recipient_id=data["recipient_id"],
        candidate_text=data["candidate_text"],
        released_text=data.get("released_text"),
        decision=data.get("decision"),
        attack_type=data.get("attack_type"),
        attack_step_index=data.get("attack_step_index"),
        is_attack_attempt=data.get("is_attack_attempt", False),
        is_legitimate_message=data.get("is_legitimate_message", False),
        is_reconstruction_attempt=data.get("is_reconstruction_attempt", False),
        is_recontamination_attempt=data.get("is_recontamination_attempt", False),
        target_exposed=data.get("target_exposed", False),
        target_reconstructed=data.get("target_reconstructed", False),
        target_reintroduced=data.get("target_reintroduced", False),
        task_relevant=data.get("task_relevant", False),
        task_contribution_successful=data.get("task_contribution_successful", False),
    )


def deserialize_contamination_status(data: dict[str, Any]) -> ContaminationStatus:
    """Deserialize a ContaminationStatus from a JSON dict."""
    # ContaminationStatus is an Enum, so we just need the value
    return ContaminationStatus(data["value"])


def deserialize_episode_result(data: dict[str, Any]) -> EpisodeResult:
    """Deserialize an EpisodeResult from a JSON dict."""
    turns = [deserialize_turn(t) for t in data.get("turns", [])]

    contamination_states = {}
    for agent_id, status_data in data.get("contamination_states", {}).items():
        contamination_states[agent_id] = deserialize_contamination_status(status_data)

    return EpisodeResult(
        run_id=data["run_id"],
        episode_id=data["episode_id"],
        scenario_id=data["scenario_id"],
        trust_level=data["trust_level"],
        seed=data["seed"],
        turns=turns,
        contamination_states=contamination_states,
        audit_entries=data.get("audit_entries", []),
        task_success=data.get("task_success", False),
        task_label=data.get("task_label"),
        cleaned_agents_exposed=data.get("cleaned_agents_exposed", 0),
        recontaminated_agents=data.get("recontaminated_agents", 0),
        metadata=data.get("metadata", {}),
    )


def load_episode_results(path: str | Path) -> list[EpisodeResult]:
    """Load episode results from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Episode file not found: {path}")

    results = []
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(deserialize_episode_result(data))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_num}: {exc}") from exc
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Malformed episode at line {line_num}: {exc}") from exc

    return results


def load_smoke_manifest(path: str | Path) -> dict[str, Any]:
    """Load a smoke manifest from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict in manifest, got {type(data).__name__}")
            return data
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed manifest JSON: {exc}") from exc

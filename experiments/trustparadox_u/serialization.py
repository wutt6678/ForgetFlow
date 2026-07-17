"""Serialization and deserialization of experiment results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from experiments.trustparadox_u.runner import EpisodeResult, TurnResult
from marble.firewall.types import (
    ContaminationStatus,
    DetectorResult,
    FirewallAction,
    FirewallDecision,
)

# Current schema version for episode results
RESULT_SCHEMA_VERSION = "1.0"


class UnsupportedSchemaVersionError(ValueError):
    """Raised when an unsupported schema version is encountered."""


def deserialize_detector_result(data: Mapping[str, Any] | Any) -> DetectorResult:
    """Deserialize a DetectorResult from a JSON dict."""
    if data is None:
        raise ValueError("DetectorResult payload is null")
    if not isinstance(data, Mapping):
        raise TypeError(f"DetectorResult payload must be a mapping, got {type(data).__name__}")

    return DetectorResult(
        exact_score=float(data.get("exact_score", 0.0)),
        entity_score=float(data.get("entity_score", 0.0)),
        semantic_score=float(data.get("semantic_score", 0.0)),
        reconstruction_score=float(data.get("reconstruction_score", 0.0)),
        matched_forget_ids=tuple(str(v) for v in data.get("matched_forget_ids", [])),
        evidence=tuple(str(v) for v in data.get("evidence", [])),
    )


def deserialize_firewall_decision(data: Mapping[str, Any] | None | Any) -> FirewallDecision | None:
    """Deserialize a FirewallDecision from a JSON dict."""
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise TypeError(
            f"FirewallDecision payload must be a mapping or null, got {type(data).__name__}"
        )

    detector_payload = data.get("detector_result")
    if detector_payload is None:
        raise ValueError("FirewallDecision is missing detector_result")

    return FirewallDecision(
        action=cast(FirewallAction, str(data["action"])),
        released_text=None if data.get("released_text") is None else str(data["released_text"]),
        detector_result=deserialize_detector_result(detector_payload),
        reason_codes=tuple(str(code) for code in data.get("reason_codes", [])),
        policy_version=str(data.get("policy_version", "")),
        latency_ms=float(data.get("latency_ms", 0.0)),
    )


def deserialize_contamination_status(data: Any) -> ContaminationStatus:
    """Deserialize a ContaminationStatus from various JSON forms.

    Supports:
    - Raw string value: "clean"
    - Mapping with value key: {"value": "clean"}
    - Enum-style string: "ContaminationStatus.clean"
    """
    if isinstance(data, str):
        value = data
    elif isinstance(data, Mapping):
        raw_value = data.get("value")
        if raw_value is None:
            raise ValueError("Contamination status mapping has no 'value'")
        value = str(raw_value)
    else:
        raise TypeError(f"Invalid contamination status payload: {type(data).__name__}")

    # Strip enum-style prefix
    if value.startswith("ContaminationStatus."):
        value = value.split(".", 1)[1]

    value = value.lower()

    try:
        return ContaminationStatus(value)
    except ValueError as exc:
        raise ValueError(f"Unknown contamination status: {value!r}") from exc


def deserialize_turn(data: dict[str, Any]) -> TurnResult:
    """Deserialize a TurnResult from a JSON dict."""
    return TurnResult(
        turn_id=data["turn_id"],
        phase=data["phase"],
        sender_id=data["sender_id"],
        recipient_id=data["recipient_id"],
        candidate_text=data["candidate_text"],
        released_text=data.get("released_text"),
        decision=deserialize_firewall_decision(data.get("decision")),
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
                # Handle schema versioning
                schema_version = str(data.get("schema_version", "0"))
                if schema_version == "1.0":
                    # Versioned envelope format
                    episode_data = data.get("episode", data)
                    results.append(deserialize_episode_result(episode_data))
                elif schema_version == "0":
                    # Legacy format (no version field)
                    results.append(deserialize_episode_result(data))
                else:
                    raise UnsupportedSchemaVersionError(
                        f"Unsupported schema version: {schema_version!r} at line {line_num}"
                    )
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_num}: {exc}") from exc
            except (KeyError, TypeError, ValueError) as exc:
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


def serialize_episode_result(result: EpisodeResult) -> dict[str, Any]:
    """Serialize an EpisodeResult to a schema-versioned dict.

    Returns a dict with schema_version and episode keys.
    """
    import dataclasses

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "episode": dataclasses.asdict(result),
    }

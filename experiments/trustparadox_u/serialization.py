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

# Schema version constants
UNVERSIONED_RESULT_SCHEMA = "0"
LEGACY_RESULT_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.1"


class UnsupportedSchemaVersionError(ValueError):
    """Raised when an unsupported schema version is encountered."""


def parse_schema_version(value: str) -> tuple[int, ...]:
    """Parse a schema version string into a numeric tuple for safe comparison.

    Raises ValueError for non-string or non-numeric inputs.
    """
    if not isinstance(value, str):
        raise ValueError("Schema version must be a string")

    parts = value.split(".")

    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid schema version: {value!r}")

    return tuple(int(part) for part in parts)


def deserialize_id_tuple(data: Mapping[str, object], field: str) -> tuple[str, ...]:
    """Deserialize and validate a tuple of forget ID strings.

    Rejects non-list values, non-string items, empty strings, and duplicates.
    """
    raw = data.get(field, [])

    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list, got {type(raw).__name__}")

    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"{field} must contain strings, got {type(item).__name__}")
        if not item:
            raise ValueError(f"{field} contains an empty ID")
        values.append(item)

    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicate IDs")

    return tuple(values)


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

    # Validate action is one of the valid FirewallAction values
    raw_action = str(data["action"])
    valid_actions = {"allow", "redact", "abstract", "block"}
    if raw_action not in valid_actions:
        raise ValueError(f"Invalid firewall action: {raw_action!r}")

    # Validate semantic constraints
    released_text = None if data.get("released_text") is None else str(data["released_text"])
    if raw_action == "block":
        if released_text is not None:
            raise ValueError("block action requires released_text to be null")
    else:
        if released_text is None:
            raise ValueError(f"{raw_action} action requires released_text to be non-null")

    return FirewallDecision(
        action=cast(FirewallAction, raw_action),
        released_text=released_text,
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
    # Parse and normalize ID fields using strict validation
    target_forget_ids = deserialize_id_tuple(data, "target_forget_ids")
    exposed_forget_ids = deserialize_id_tuple(data, "exposed_forget_ids")
    reconstructed_forget_ids = deserialize_id_tuple(data, "reconstructed_forget_ids")
    reintroduced_forget_ids = deserialize_id_tuple(data, "reintroduced_forget_ids")

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
        target_forget_ids=target_forget_ids,
        target_exposed=data.get("target_exposed", False),
        exposed_forget_ids=exposed_forget_ids,
        target_reconstructed=data.get("target_reconstructed", False),
        reconstructed_forget_ids=reconstructed_forget_ids,
        target_reintroduced=data.get("target_reintroduced", False),
        reintroduced_forget_ids=reintroduced_forget_ids,
        task_relevant=data.get("task_relevant", False),
        task_contribution_successful=data.get("task_contribution_successful", False),
    )


def deserialize_episode_result(
    data: dict[str, Any],
    schema_version: str = RESULT_SCHEMA_VERSION,
) -> EpisodeResult:
    """Deserialize an EpisodeResult from a JSON dict."""
    turns = [deserialize_turn(t) for t in data.get("turns", [])]

    contamination_states = {}
    for agent_id, status_data in data.get("contamination_states", {}).items():
        contamination_states[agent_id] = deserialize_contamination_status(status_data)

    # Validate and extract pair-based counters
    attempted_agent_record_pairs = int(data.get("attempted_agent_record_pairs", 0))
    recontaminated_agent_record_pairs = int(data.get("recontaminated_agent_record_pairs", 0))

    # Reject negative values
    if attempted_agent_record_pairs < 0:
        raise ValueError("attempted_agent_record_pairs must be non-negative")
    if recontaminated_agent_record_pairs < 0:
        raise ValueError("recontaminated_agent_record_pairs must be non-negative")

    # Extract final_contamination_states
    final_contamination_states: dict[tuple[str, str], str] = {}
    raw_fcs = data.get("final_contamination_states", [])
    if isinstance(raw_fcs, list):
        for entry in raw_fcs:
            if isinstance(entry, dict) and "agent_id" in entry and "forget_id" in entry:
                final_contamination_states[(str(entry["agent_id"]), str(entry["forget_id"]))] = str(
                    entry.get("status", "unknown")
                )
    elif isinstance(raw_fcs, dict):
        # Handle legacy dict form with tuple-like keys
        for k, v in raw_fcs.items():
            if isinstance(k, (list, tuple)) and len(k) == 2:
                final_contamination_states[(str(k[0]), str(k[1]))] = str(v)

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
        attempted_agent_record_pairs=attempted_agent_record_pairs,
        recontaminated_agent_record_pairs=recontaminated_agent_record_pairs,
        final_contamination_states=final_contamination_states,
        metadata=data.get("metadata", {}),
        schema_version=schema_version,
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
                schema_version = str(data.get("schema_version", UNVERSIONED_RESULT_SCHEMA))
                if schema_version == RESULT_SCHEMA_VERSION:
                    # Current versioned envelope format
                    episode_data = data.get("episode", data)
                    results.append(deserialize_episode_result(episode_data, schema_version))
                elif schema_version == LEGACY_RESULT_SCHEMA_VERSION:
                    # Legacy v1.0 format (may lack per-record ID fields)
                    episode_data = data.get("episode", data)
                    results.append(deserialize_episode_result(episode_data, schema_version))
                elif schema_version == UNVERSIONED_RESULT_SCHEMA:
                    # Unversioned legacy format (no version field)
                    results.append(deserialize_episode_result(data, schema_version))
                else:
                    raise UnsupportedSchemaVersionError(
                        f"Unsupported schema version: {schema_version!r} at line {line_num}"
                    )
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_num}: {exc}") from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed episode at line {line_num}: {exc}") from exc

    return results


def inspect_result_schema_versions(path: str | Path) -> set[str]:
    """Peek at schema versions in a JSONL file without full deserialization.

    Returns the set of distinct schema version strings found.
    Validates that each line is a JSON mapping with a parseable schema version.
    For schema >= 1.0, also validates the presence of an 'episode' mapping.
    """
    path = Path(path)
    versions: set[str] = set()
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path} at line {line_number}: {exc}") from exc

            if not isinstance(decoded, Mapping):
                raise ValueError(
                    f"Result envelope at line {line_number} in {path} "
                    f"must be a JSON object, got {type(decoded).__name__}"
                )

            raw_version = decoded.get("schema_version", UNVERSIONED_RESULT_SCHEMA)

            # Validate schema version is parseable
            try:
                sv_tuple = parse_schema_version(raw_version)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid schema version at line {line_number} in {path}: {exc}"
                ) from exc

            # For schema >= 1.0, require an 'episode' mapping
            if sv_tuple >= (1, 0):
                episode = decoded.get("episode")
                if episode is None:
                    raise ValueError(
                        f"Result envelope at line {line_number} in {path} "
                        f"with schema {raw_version!r} is missing 'episode' field"
                    )
                if not isinstance(episode, Mapping):
                    raise ValueError(
                        f"Result envelope at line {line_number} in {path} "
                        f"has 'episode' of type {type(episode).__name__}, "
                        f"expected a JSON object"
                    )

            versions.add(str(raw_version))
    return versions


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

    episode_data = dataclasses.asdict(result)
    # Convert final_contamination_states tuple keys to serializable form
    fcs = episode_data.get("final_contamination_states", {})
    serialized_fcs = [
        {"agent_id": k[0], "forget_id": k[1], "status": v} for k, v in sorted(fcs.items())
    ]
    episode_data["final_contamination_states"] = serialized_fcs

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "episode": episode_data,
    }

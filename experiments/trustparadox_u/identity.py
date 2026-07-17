"""Canonical experiment identity for pairing and deduplication.

Provides a single source of truth for pairing-key normalization,
shared by the evaluator, result auditor, and aggregation pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PairingKey = tuple[str, str, str, str, int]
RunIdentity = tuple[PairingKey, str]


@dataclass(frozen=True, order=True)
class ResearchRunIdentity:
    """Canonical research-run identity for duplicate detection.

    Includes all scientifically relevant dimensions that distinguish
    one experiment run from another.
    """

    scenario_id: str
    secret_variant_id: str
    trust_level: str
    attack_type: str
    seed: int
    condition_id: str


PAIRING_KEY_FIELDS = (
    "scenario_id",
    "secret_variant_id",
    "trust_level",
    "attack_type",
    "seed",
)


def normalize_identity_component(value: object) -> str:
    """Normalize a metadata component to a stable string.

    Lists, tuples, and sets are sorted and serialized as canonical JSON.
    Mappings are serialized with sorted keys.
    All other values are converted via ``str()``.
    """
    if isinstance(value, (list, tuple, set)):
        return json.dumps(
            sorted(normalize_identity_component(item) for item in value),
            separators=(",", ":"),
        )
    if isinstance(value, Mapping):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    return str(value)


def normalize_attack_type(value: object) -> str:
    """Normalize attack_type which may be a scalar or list."""
    return normalize_identity_component(value)


def normalize_pairing_key(value: object) -> PairingKey:
    """Normalize a pairing key to a canonical hashable tuple.

    Accepts:
    - A ``dict`` / ``Mapping`` with all ``PAIRING_KEY_FIELDS``.
    - A 5-element tuple already in canonical order.

    Raises ``TypeError`` for unsupported types.
    Raises ``ValueError`` when required fields are missing.
    """
    if isinstance(value, Mapping):
        missing = [f for f in PAIRING_KEY_FIELDS if f not in value]
        if missing:
            raise ValueError("Pairing key is missing required fields: " + ", ".join(missing))
        return _coerce_fields(value)

    if isinstance(value, tuple) and len(value) == 5:
        return (
            str(value[0]),
            normalize_identity_component(value[1]),
            str(value[2]),
            normalize_identity_component(value[3]),
            int(value[4]),
        )

    raise TypeError(f"Unsupported pairing key type: {type(value).__name__}")


def _coerce_fields(value: Mapping[str, Any]) -> PairingKey:
    """Extract and coerce the canonical fields from a mapping."""
    return (
        str(value["scenario_id"]),
        normalize_identity_component(value["secret_variant_id"]),
        str(value["trust_level"]),
        normalize_identity_component(value["attack_type"]),
        int(value["seed"]),
    )


def pairing_key_from_result(result: Any) -> PairingKey:
    """Build a canonical pairing key from an ``EpisodeResult``.

    The *result* must expose ``scenario_id``, ``trust_level``, ``seed``,
    and a ``metadata`` mapping containing ``secret_variant_id`` and
    ``attack_type``.
    """
    metadata = result.metadata
    return (
        str(result.scenario_id),
        normalize_identity_component(metadata["secret_variant_id"]),
        str(result.trust_level),
        normalize_attack_type(metadata["attack_type"]),
        int(result.seed),
    )


def run_identity_from_result(result: Any) -> RunIdentity:
    """Build a run identity for duplicate-result detection.

    Combines the pairing key with the config hash so that different
    experiment variants sharing the same pairing key are not flagged
    as duplicates.
    """
    config_hash = str(result.metadata.get("config_hash", ""))
    if not config_hash:
        raise ValueError("EpisodeResult metadata missing config_hash")
    return (pairing_key_from_result(result), config_hash)


def research_run_identity_from_result(result: Any) -> ResearchRunIdentity:
    """Build a ResearchRunIdentity from an EpisodeResult.

    Includes condition_id from smoke_condition metadata for proper
    distinction between experiment conditions.
    """
    metadata = result.metadata
    condition_id = str(metadata.get("smoke_condition", ""))
    if not condition_id:
        # Fall back to config_hash for non-smoke runs
        condition_id = str(metadata.get("config_hash", ""))
    if not condition_id:
        raise ValueError("EpisodeResult metadata missing both smoke_condition and config_hash")
    return ResearchRunIdentity(
        scenario_id=str(result.scenario_id),
        secret_variant_id=normalize_identity_component(metadata.get("secret_variant_id", "")),
        trust_level=str(result.trust_level),
        attack_type=normalize_attack_type(metadata.get("attack_type", "")),
        seed=int(result.seed),
        condition_id=condition_id,
    )

"""Canonical experiment identity for pairing and deduplication.

Provides a single source of truth for pairing-key normalization,
shared by the evaluator, result auditor, and aggregation pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PairingKey = tuple[str, str, str, str, int]

PAIRING_KEY_FIELDS = (
    "scenario_id",
    "secret_variant_id",
    "trust_level",
    "attack_type",
    "seed",
)


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
            str(value[1]),
            str(value[2]),
            str(value[3]),
            int(value[4]),
        )

    raise TypeError(f"Unsupported pairing key type: {type(value).__name__}")


def _coerce_fields(value: Mapping[str, Any]) -> PairingKey:
    """Extract and coerce the canonical fields from a mapping."""
    return (
        str(value["scenario_id"]),
        str(value["secret_variant_id"]),
        str(value["trust_level"]),
        str(value["attack_type"]),
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
        str(metadata["secret_variant_id"]),
        str(result.trust_level),
        str(metadata["attack_type"]),
        int(result.seed),
    )

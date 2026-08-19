"""E5 episode metadata construction (§10).

Builds the deterministic structural metadata (fragment_map, fact_chain_map)
that the ForgetFlow ReconstructionChecker and the post-firewall
reconstruction probe need.  The metadata is derived exclusively from
already-frozen corpus/sequence structures — never from the E4 outcome
labels or any condition-specific shortcut.

For the same sequence, C0-C4 and A0-A4 MUST use identical structural
metadata.  This is the single source of truth for what fragments and
fact chains participate in reconstruction.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    EmpiricalTargetSpec,
)


def build_e5_episode_metadata(
    sequence_label: Any,
    corpus_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic episode metadata for a single sequence.

    Uses the frozen ``EMPIRICAL_TARGET_REGISTRY`` plus the sequence's
    target identity to produce per-forget-id fragment_map and
    fact_chain_map entries. The ``sequence_label`` parameter is used
    only to determine the target family via the first candidate's
    ``secret_variant_id`` (and the target registry lookup).

    Args:
        sequence_label: Sequence annotation object (must expose
            ``ordered_candidate_ids`` and the corpus lookup is performed
            externally). Its first ordered candidate's corpus entry is
            used to look up the frozen target spec.
        corpus_by_id: Mapping from candidate_id → corpus entry with
            ``scenario_id`` and ``secret_variant_id``.

    Returns:
        Dict with ``fragment_map`` and ``fact_chain_map`` keyed by
        forget_id, ready to be passed to
        ``create_firewall_runner(..., episode_metadata=...)`` or
        ``run_reconstruction_probe(..., episode_metadata=...)``.

    Raises:
        KeyError: If no corpus entry is found for the first ordered
            candidate, or if the target spec cannot be located.
    """
    ordered_ids: Sequence[str] = getattr(sequence_label, "ordered_candidate_ids", ())
    if not ordered_ids:
        raise KeyError(
            "Sequence has no ordered_candidate_ids; cannot build episode metadata"
        )

    first_id = ordered_ids[0]
    first_corpus = corpus_by_id.get(first_id)
    if first_corpus is None:
        raise KeyError(
            f"Missing corpus entry for sequence first candidate {first_id!r}"
        )

    scenario_id = getattr(first_corpus, "scenario_id", "")
    secret_variant_id = getattr(first_corpus, "secret_variant_id", "")

    spec = _lookup_spec(scenario_id, secret_variant_id)

    fragment_map: dict[str, dict[str, Any]] = {
        spec.forget_id: {
            "fragments": list(spec.fragments),
            "canonical_target": spec.canonical_target,
        }
    }
    fact_chain_map: dict[str, list[list[str]]] = {
        spec.forget_id: [list(spec.fact_chain)]
    }

    return {
        "fragment_map": fragment_map,
        "fact_chain_map": fact_chain_map,
        "scenario_id": scenario_id,
        "secret_variant_id": secret_variant_id,
        "forget_id": spec.forget_id,
    }


def _lookup_spec(
    scenario_id: str,
    secret_variant_id: str,
) -> EmpiricalTargetSpec:
    """Look up a frozen empirical target spec.

    Tries the variant-id-keyed registry first, then falls back to a
    scenario+variant scan. Raises KeyError if no spec is found.
    """
    by_variant = {spec.secret_variant_id: spec for spec in EMPIRICAL_TARGET_REGISTRY}
    if secret_variant_id in by_variant:
        return by_variant[secret_variant_id]

    for spec in EMPIRICAL_TARGET_REGISTRY:
        if spec.scenario_id == scenario_id and spec.secret_variant_id == secret_variant_id:
            return spec

    raise KeyError(
        f"No empirical target spec for scenario_id={scenario_id!r}, "
        f"secret_variant_id={secret_variant_id!r}"
    )

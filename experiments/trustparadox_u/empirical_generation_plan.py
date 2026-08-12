"""E3-003/004: frozen generation configuration and deterministic generation plan.

The generation config captures every tunable parameter of the real-API
corpus-generation campaign.  The generation plan expands the config into
a flat list of ``GenerationPlanItem`` records — one per API attempt — so
the exact campaign size is known before any tokens are spent.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.trustparadox_u.artifact_provenance import (
    environment_lock_hash,
)
from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PHASE_FILE,
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    AttackType,
    EmpiricalTargetSpec,
    compute_target_registry_hash,
    empirical_candidate_family_id,
    empirical_sequence_family_id,
    generation_attempt_id,
)
from experiments.trustparadox_u.empirical_generation import (
    build_prompt_manifest,
    prompt_manifest_sha256,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "manifests"

# ---------------------------------------------------------------------------
# E3-003b: typed frozen generation configuration loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenGenerationConfig:
    """Typed view of ``full_generation_config.json``.

    This is the single source of truth for actual API request parameters
    during the full-corpus generation campaign.
    """

    generator_provider: str
    generator_model_requested: str
    generator_temperature: float
    generator_max_tokens: int
    request_timeout: float
    max_retries: int
    backoff_seconds: tuple[float, ...]
    retryable_statuses: tuple[str, ...]
    generation_replicates: int
    generator_seed_policy: str = "provider_default_unavailable"


def load_frozen_generation_config(
    path: Path = _MANIFESTS_DIR / "full_generation_config.json",
) -> FrozenGenerationConfig:
    """Load and parse the frozen generation config from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    retry = data.get("retry_policy", {})
    return FrozenGenerationConfig(
        generator_provider=data["generator_provider"],
        generator_model_requested=data["generator_model_requested"],
        generator_temperature=float(data["generator_temperature"]),
        generator_max_tokens=int(data["generator_max_tokens"]),
        request_timeout=float(data["request_timeout"]),
        max_retries=int(retry.get("max_retries", 2)),
        backoff_seconds=tuple(float(s) for s in retry.get("backoff_seconds", [2.0, 5.0])),
        retryable_statuses=tuple(retry.get("retryable_statuses", ["provider_error", "timeout"])),
        generation_replicates=int(data.get("generation_replicates", 1)),
        generator_seed_policy=data.get("generator_seed_policy", "provider_default_unavailable"),
    )


# ---------------------------------------------------------------------------
# E3-003: frozen generation configuration
# ---------------------------------------------------------------------------

#: Generator model frozen from E2.
GENERATOR_PROVIDER = "openai"
GENERATOR_MODEL_REQUESTED = "qwen3.7-plus"
GENERATOR_TEMPERATURE = 0.7
GENERATOR_MAX_TOKENS = 1024

#: Seed policy — the provider does not expose deterministic seeds.
GENERATOR_SEED_POLICY = "provider_default_unavailable"

#: Number of independent replicates per cell.
GENERATION_REPLICATES = 1

#: Retry policy for transient failures.
RETRY_POLICY = {
    "max_retries": 2,
    "backoff_seconds": [2.0, 5.0],
    "retryable_statuses": ["provider_error", "timeout"],
}

#: Request timeout in seconds.
REQUEST_TIMEOUT = 120

#: Frozen trust levels (protocol order).
FROZEN_TRUST_LEVELS: tuple[str, ...] = ("low", "default", "high")

#: Frozen primary attack types (no trust_discretion_task in Phase 3).
FROZEN_ATTACK_TYPES: tuple[str, ...] = (
    AttackType.DIRECT_DISCLOSURE.value,
    AttackType.SEMANTIC_PARAPHRASE.value,
    AttackType.ALIAS_OR_COREFERENCE.value,
    AttackType.RECONTAMINATION.value,
    AttackType.FRAGMENTATION_SEQUENCE.value,
    AttackType.COMPOSITIONAL_SEQUENCE.value,
    AttackType.HARD_NEGATIVE_CONTROL.value,
    AttackType.LEGITIMATE_TASK.value,
)

#: Protocol-defined scenario families.
FROZEN_SCENARIO_IDS: tuple[str, ...] = (
    "credential_001",
    "private_attribute_001",
    "authorization_001",
)

# ---------------------------------------------------------------------------
# E3-003: sample-count policy per scenario × variant × trust
# ---------------------------------------------------------------------------

#: Target sample counts per attack family (per scenario × variant × trust).
SAMPLE_COUNTS_BY_ATTACK: dict[str, int] = {
    "direct_disclosure": 2,
    "semantic_paraphrase": 4,
    "alias_or_coreference": 2,
    "hard_negative_control": 3,
    "recontamination": 2,
    "fragmentation_sequence": 2,  # 2 sequences
    "compositional_sequence": 2,  # 2 sequences
    "legitimate_task": 2,
}

# ---------------------------------------------------------------------------
# Attack applicability
# ---------------------------------------------------------------------------

#: alias_or_coreference is not applicable to credential scenarios.
_ALIAS_INAPPLICABLE_SCENARIOS: frozenset[str] = frozenset({"credential_001"})


def attack_is_applicable(attack_type: str, spec: EmpiricalTargetSpec) -> bool:
    """Whether *attack_type* is applicable for the given target spec.

    ``alias_or_coreference`` is excluded for credential scenarios because
    a four-digit code has no meaningful alias or coreference target.
    ``compositional_sequence`` requires a non-empty fact chain.
    """
    attack = AttackType(attack_type)
    if attack is AttackType.ALIAS_OR_COREFERENCE:
        return spec.scenario_id not in _ALIAS_INAPPLICABLE_SCENARIOS
    if attack is AttackType.COMPOSITIONAL_SEQUENCE:
        return len(spec.fact_chain) >= 2
    return True


# ---------------------------------------------------------------------------
# E3-004: deterministic generation plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationPlanItem:
    """One planned API attempt in the full-corpus generation campaign."""

    plan_item_id: str
    split: str
    scenario_id: str
    secret_variant_id: str
    trust_level: str
    attack_type: str
    sample_index: int
    generation_replicate: int
    sequence_id: str | None = None
    sequence_step_index: int | None = None
    sequence_step_count: int | None = None


def _sequence_step_count(attack_type: str, spec: EmpiricalTargetSpec) -> int | None:
    """Number of steps for a sequence attack, or None for non-sequence."""
    attack = AttackType(attack_type)
    if attack is AttackType.FRAGMENTATION_SEQUENCE:
        return len(spec.fragments)
    if attack is AttackType.COMPOSITIONAL_SEQUENCE:
        return len(spec.fact_chain)
    return None


def planned_units_for_spec(
    spec: EmpiricalTargetSpec,
    *,
    trust_levels: Sequence[str] = FROZEN_TRUST_LEVELS,
    attack_types: Sequence[str] = FROZEN_ATTACK_TYPES,
    sample_counts: dict[str, int] | None = None,
    replicates: int = GENERATION_REPLICATES,
) -> list[GenerationPlanItem]:
    """Expand one target spec into a flat list of planned API attempts.

    Each non-sequence item maps to exactly one API attempt.  Each sequence
    item maps to ``sequence_step_count`` attempts (one per step).
    """
    counts = sample_counts or SAMPLE_COUNTS_BY_ATTACK
    items: list[GenerationPlanItem] = []

    for attack_type in attack_types:
        if not attack_is_applicable(attack_type, spec):
            continue
        n_samples = counts.get(attack_type, 1)
        step_count = _sequence_step_count(attack_type, spec)
        is_sequence = step_count is not None

        for rep in range(replicates):
            for sample_idx in range(n_samples):
                if is_sequence:
                    for step_idx in range(step_count):  # type: ignore[arg-type]
                        for trust in trust_levels:
                            pid = generation_attempt_id(
                                scenario_id=spec.scenario_id,
                                secret_variant_id=spec.secret_variant_id,
                                trust_level=trust,
                                attack_type=attack_type,
                                sample_index=sample_idx,
                                generation_replicate=rep,
                                sequence_step_index=step_idx,
                            )
                            items.append(
                                GenerationPlanItem(
                                    plan_item_id=pid,
                                    split=spec.split,
                                    scenario_id=spec.scenario_id,
                                    secret_variant_id=spec.secret_variant_id,
                                    trust_level=trust,
                                    attack_type=attack_type,
                                    sample_index=sample_idx,
                                    generation_replicate=rep,
                                    sequence_id=empirical_sequence_family_id(
                                        scenario_id=spec.scenario_id,
                                        secret_variant_id=spec.secret_variant_id,
                                        attack_type=attack_type,
                                        sample_index=sample_idx,
                                        generation_replicate=rep,
                                    ),
                                    sequence_step_index=step_idx,
                                    sequence_step_count=step_count,
                                )
                            )
                else:
                    for trust in trust_levels:
                        pid = generation_attempt_id(
                            scenario_id=spec.scenario_id,
                            secret_variant_id=spec.secret_variant_id,
                            trust_level=trust,
                            attack_type=attack_type,
                            sample_index=sample_idx,
                            generation_replicate=rep,
                        )
                        items.append(
                            GenerationPlanItem(
                                plan_item_id=pid,
                                split=spec.split,
                                scenario_id=spec.scenario_id,
                                secret_variant_id=spec.secret_variant_id,
                                trust_level=trust,
                                attack_type=attack_type,
                                sample_index=sample_idx,
                                generation_replicate=rep,
                            )
                        )
    return items


def build_full_generation_plan(
    *,
    splits: Sequence[str] = ("development", "validation", "test"),
    scenario_ids: Sequence[str] | None = None,
) -> list[GenerationPlanItem]:
    """Expand the full corpus generation plan across all splits."""
    # Lazy import to break circular dependency with generate_empirical_corpus.
    from experiments.trustparadox_u.generate_empirical_corpus import specs_for_split

    items: list[GenerationPlanItem] = []
    for split in splits:
        specs = specs_for_split(split, scenario_ids=scenario_ids)
        for spec in specs:
            items.extend(planned_units_for_spec(spec))
    return items


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------


def validate_generation_plan(items: Sequence[GenerationPlanItem]) -> list[str]:
    """Check plan invariants; returns a list of findings (empty = valid)."""
    findings: list[str] = []

    # Uniqueness of plan_item_ids.
    ids = [it.plan_item_id for it in items]
    dup_ids = {pid for pid, cnt in Counter(ids).items() if cnt > 1}
    if dup_ids:
        findings.append(f"duplicate plan_item_ids: {len(dup_ids)}")

    # Uniqueness of candidate family IDs per trust level.
    # Candidate family IDs are trust-independent by design, so the same
    # family may appear once per trust level.  Duplicates within the
    # same trust level indicate a real collision.
    family_trust_pairs = [
        (
            empirical_candidate_family_id(
                scenario_id=it.scenario_id,
                secret_variant_id=it.secret_variant_id,
                attack_type=it.attack_type,
                sample_index=it.sample_index,
                generation_replicate=it.generation_replicate,
                sequence_step_index=it.sequence_step_index,
            ),
            it.trust_level,
        )
        for it in items
    ]
    dup_families = {pair for pair, cnt in Counter(family_trust_pairs).items() if cnt > 1}
    if dup_families:
        findings.append(f"duplicate candidate family IDs: {len(dup_families)}")

    # Each sequence item must have the correct step count.
    for it in items:
        if it.sequence_id is not None and it.sequence_step_count is None:
            findings.append(f"sequence item {it.plan_item_id} missing step count")

    # Plan must not contain firewall-specific fields.
    for it in items:
        if "firewall" in it.plan_item_id or "embedding" in it.plan_item_id:
            findings.append(f"plan item contains forbidden field: {it.plan_item_id}")

    return findings


def plan_summary(items: Sequence[GenerationPlanItem]) -> dict[str, object]:
    """Compute expected counts for the generation plan.

    Patch J: adds scientific unit counts that distinguish planned provider
    calls from independent scientific generation units.
    """
    by_split: dict[str, int] = dict(sorted(Counter(it.split for it in items).items()))
    by_scenario: dict[str, int] = dict(sorted(Counter(it.scenario_id for it in items).items()))
    by_variant: dict[str, int] = dict(sorted(Counter(it.secret_variant_id for it in items).items()))
    by_trust: dict[str, int] = dict(sorted(Counter(it.trust_level for it in items).items()))
    by_attack: dict[str, int] = dict(sorted(Counter(it.attack_type for it in items).items()))

    sequence_items = [it for it in items if it.sequence_id is not None]
    non_sequence_items = [it for it in items if it.sequence_id is None]

    # Count distinct sequences (unique (sequence_id, trust_level) pairs).
    distinct_sequences = len(
        {(it.sequence_id, it.trust_level) for it in sequence_items if it.sequence_id}
    )

    # Patch J: scientific unit counts.
    # Each non-sequence item is one scientific unit.
    # Each distinct (sequence_id, trust_level) group is one sequence scientific unit.
    non_sequence_unit_count = len(non_sequence_items)
    sequence_unit_count = distinct_sequences
    scientific_unit_count = non_sequence_unit_count + sequence_unit_count

    return {
        "total_planned_attempts": len(items),
        "non_sequence_attempts": len(non_sequence_items),
        "sequence_step_attempts": len(sequence_items),
        "distinct_sequences": distinct_sequences,
        "scientific_generation_unit_count": scientific_unit_count,
        "planned_provider_call_minimum": len(items),
        "sequence_scientific_unit_count": sequence_unit_count,
        "non_sequence_scientific_unit_count": non_sequence_unit_count,
        "by_split": by_split,
        "by_scenario": by_scenario,
        "by_variant": by_variant,
        "by_trust": by_trust,
        "by_attack": by_attack,
    }


def plan_sha256(items: Sequence[GenerationPlanItem]) -> str:
    """Deterministic SHA-256 over the sorted plan items."""
    records = sorted(
        (asdict(it) for it in items),
        key=lambda r: r["plan_item_id"],
    )
    lines = [
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for r in records
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Config + plan writers
# ---------------------------------------------------------------------------


def _repository_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _phase_manifest_sha256() -> str:
    if EMPIRICAL_PHASE_FILE.exists():
        return hashlib.sha256(EMPIRICAL_PHASE_FILE.read_bytes()).hexdigest()
    return ""


def build_generation_config() -> dict[str, object]:
    """Build the frozen generation configuration record."""
    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "generator_provider": GENERATOR_PROVIDER,
        "generator_model_requested": GENERATOR_MODEL_REQUESTED,
        "generator_temperature": GENERATOR_TEMPERATURE,
        "generator_max_tokens": GENERATOR_MAX_TOKENS,
        "generator_seed_policy": GENERATOR_SEED_POLICY,
        "generation_replicates": GENERATION_REPLICATES,
        "retry_policy": RETRY_POLICY,
        "request_timeout": REQUEST_TIMEOUT,
        "trust_levels": list(FROZEN_TRUST_LEVELS),
        "attack_types": list(FROZEN_ATTACK_TYPES),
        "sample_counts_by_attack": SAMPLE_COUNTS_BY_ATTACK,
        "scenario_ids": list(FROZEN_SCENARIO_IDS),
        "target_registry_sha256": compute_target_registry_hash(),
        "frozen_prompt_manifest_sha256": prompt_manifest_sha256(build_prompt_manifest()),
        "environment_lock_hash": environment_lock_hash(),
        "phase_manifest_sha256": _phase_manifest_sha256(),
        "created_from_commit": _repository_commit(),
    }


def write_generation_config(
    path: Path = _MANIFESTS_DIR / "full_generation_config.json",
) -> dict[str, object]:
    """Write the frozen config and return it."""
    config = build_generation_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config


def write_generation_plan(
    plan_path: Path = _MANIFESTS_DIR / "full_generation_plan.jsonl",
    summary_path: Path = _MANIFESTS_DIR / "full_generation_plan_summary.json",
) -> tuple[list[GenerationPlanItem], dict[str, object]]:
    """Build, validate, and write the full generation plan."""
    items = build_full_generation_plan()
    findings = validate_generation_plan(items)
    if findings:
        raise ValueError(f"generation plan validation failed: {findings}")

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), sort_keys=True, ensure_ascii=False) + "\n")

    summary = plan_summary(items)
    summary["plan_scientific_sha256"] = plan_sha256(items)
    summary["plan_file_sha256"] = hashlib.sha256(
        plan_path.read_bytes()
    ).hexdigest()
    # Backward compatibility: plan_sha256 remains the scientific hash.
    summary["plan_sha256"] = summary["plan_scientific_sha256"]
    summary["plan_item_count"] = len(items)
    summary["validation_findings"] = findings

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return items, summary

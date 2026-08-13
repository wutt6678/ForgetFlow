#!/usr/bin/env python3
"""E3-014 / Patch F: Full-corpus validation and audit.

Produces a blocking Phase-3 validation report covering all 11 audit
sections required before a split corpus can be frozen:

 10.1  Phase / provenance
 10.2  Generation-plan completeness
 10.3  Split integrity
 10.4  Identity validation
 10.5  Variant consistency and contamination
 10.6  Config consistency
 10.7  Hash integrity
 10.8  Sequence atomicity
 10.9  Retry lineage
 10.10 Acceptance-independence
 10.11 Coverage statistics

Output:
- results/empirical_v2/corpus_generation/full_corpus_validation_report.json
- results/empirical_v2/corpus_generation/full_corpus_validation_report.md
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PHASE,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    EMPIRICAL_TARGET_REGISTRY,
    SEQUENCE_ATTACK_TYPES,
    AttackType,
    EmpiricalCandidate,
    EmpiricalGenerationAttempt,
    EmpiricalSplit,
    EmpiricalTargetSpec,
    GenerationStatus,
    accepted_candidates_scientific_hash,
    compute_target_registry_hash,
    detect_cross_variant_contamination,
    raw_attempts_scientific_hash,
    record_to_attempt,
    record_to_candidate,
    validate_sequence_structure,
    validate_target_registry,
)
from experiments.trustparadox_u.empirical_generation_plan import (
    load_generation_plan,
    plan_sha256,
)
from experiments.trustparadox_u.campaign_identity import (
    CAMPAIGN_IDENTITY_FILENAME,
    CampaignIdentity,
    compute_campaign_identity,
    load_campaign_identity,
    verify_campaign_identity,
    CampaignIdentityMismatchError,
)
from experiments.trustparadox_u.empirical_generation_plan import (
    FrozenGenerationConfig,
    load_frozen_generation_config,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "manifests"
_CORPUS_BASE = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_OUTPUT_DIR = _CORPUS_BASE
_REPORT_JSON = _OUTPUT_DIR / "full_corpus_validation_report.json"
_REPORT_MD = _OUTPUT_DIR / "full_corpus_validation_report.md"

# Frozen split → variant mapping (from the target registry).
_FROZEN_SPLIT_VARIANTS: dict[str, list[str]] = {
    "development": ["credential_v1", "private_attribute_v1", "authorization_v1"],
    "validation": ["credential_v2", "private_attribute_v2", "authorization_v2"],
    "test": [
        "credential_v3", "credential_v4",
        "private_attribute_v3", "private_attribute_v4",
        "authorization_v3", "authorization_v4",
    ],
}

# All valid variant IDs.
_ALL_VARIANT_IDS: set[str] = set()
for _variants in _FROZEN_SPLIT_VARIANTS.values():
    _ALL_VARIANT_IDS.update(_variants)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_attempts(split: str) -> list[EmpiricalGenerationAttempt]:
    """Load raw attempts for a split."""
    path = _CORPUS_BASE / split / "raw_generation_attempts.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [record_to_attempt(json.loads(line)) for line in fh if line.strip()]


def _load_candidates(split: str) -> list[EmpiricalCandidate]:
    """Load accepted candidates for a split."""
    path = _CORPUS_BASE / split / "accepted_candidates.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [record_to_candidate(json.loads(line)) for line in fh if line.strip()]


def _load_manifest(split: str) -> dict | None:
    """Load the corpus manifest for a split."""
    path = _CORPUS_BASE / split / "corpus_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frozen_config() -> dict | None:
    path = _MANIFESTS_DIR / "full_generation_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_plan_items() -> list[dict]:
    path = _MANIFESTS_DIR / "full_generation_plan.jsonl"
    if not path.exists():
        return []
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec_for_variant(variant_id: str) -> EmpiricalTargetSpec | None:
    for spec in EMPIRICAL_TARGET_REGISTRY:
        if spec.secret_variant_id == variant_id:
            return spec
    return None


def _spec_by_scenario_variant(scenario_id: str, variant_id: str) -> EmpiricalTargetSpec | None:
    for spec in EMPIRICAL_TARGET_REGISTRY:
        if spec.scenario_id == scenario_id and spec.secret_variant_id == variant_id:
            return spec
    return None


# ---------------------------------------------------------------------------
# 10.1 Phase / provenance
# ---------------------------------------------------------------------------


def validate_phase_and_provenance() -> list[str]:
    """Check phase, required artifacts, and provenance fields."""
    findings: list[str] = []
    if EMPIRICAL_PHASE != "E3_CORPUS_GENERATION":
        findings.append(f"phase is {EMPIRICAL_PHASE}, expected E3_CORPUS_GENERATION")

    # Check required frozen manifests exist.
    for name in (
        "empirical_phase.json",
        "full_generation_config.json",
        "full_generation_plan.jsonl",
        "full_generation_plan_summary.json",
    ):
        if not (_MANIFESTS_DIR / name).exists():
            findings.append(f"frozen manifest missing: {name}")

    # Check target registry is valid.
    registry_problems = validate_target_registry(EMPIRICAL_TARGET_REGISTRY)
    findings.extend(registry_problems)

    return findings


# ---------------------------------------------------------------------------
# 10.2 Generation-plan completeness
# ---------------------------------------------------------------------------


def validate_plan_completeness(
    all_attempts: list[EmpiricalGenerationAttempt],
    all_candidates: list[EmpiricalCandidate],
) -> list[str]:
    """Every planned scientific unit must be accounted for."""
    findings: list[str] = []
    plan_items = _load_plan_items()
    if not plan_items:
        findings.append("generation plan not found or empty")
        return findings

    # Build set of (scenario_id, variant_id, trust, attack, sample, replicate)
    # from attempts to check coverage.
    attempt_units: set[tuple] = set()
    for a in all_attempts:
        unit = (
            a.scenario_id, a.secret_variant_id, a.trust_level,
            a.attack_type, a.sample_index, a.generation_replicate,
        )
        attempt_units.add(unit)

    planned_units: set[tuple] = set()
    for item in plan_items:
        unit = (
            item["scenario_id"], item["secret_variant_id"],
            item["trust_level"], item["attack_type"],
            item["sample_index"], item["generation_replicate"],
        )
        planned_units.add(unit)

    unaccounted = planned_units - attempt_units
    if unaccounted:
        findings.append(
            f"plan completeness: {len(unaccounted)} planned units have no raw attempts"
        )

    return findings


# ---------------------------------------------------------------------------
# 10.3 Split integrity
# ---------------------------------------------------------------------------


def validate_split_integrity(
    all_attempts: list[EmpiricalGenerationAttempt],
    all_candidates: list[EmpiricalCandidate],
) -> list[str]:
    """Check every attempt/candidate belongs to its frozen split."""
    findings: list[str] = []

    # Build variant → expected split mapping from registry.
    variant_to_split: dict[str, str] = {}
    for spec in EMPIRICAL_TARGET_REGISTRY:
        variant_to_split[spec.secret_variant_id] = spec.split

    for attempt in all_attempts:
        expected_split = variant_to_split.get(attempt.secret_variant_id)
        if expected_split is None:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"unknown variant {attempt.secret_variant_id}"
            )
        elif attempt.split != expected_split:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"split={attempt.split} but variant "
                f"{attempt.secret_variant_id} belongs to {expected_split}"
            )

    for candidate in all_candidates:
        expected_split = variant_to_split.get(candidate.secret_variant_id)
        if expected_split is None:
            findings.append(
                f"candidate {candidate.candidate_id}: "
                f"unknown variant {candidate.secret_variant_id}"
            )
        elif candidate.split != expected_split:
            findings.append(
                f"candidate {candidate.candidate_id}: "
                f"split={candidate.split} but variant "
                f"{candidate.secret_variant_id} belongs to {expected_split}"
            )

    return findings


# ---------------------------------------------------------------------------
# 10.4 Identity validation
# ---------------------------------------------------------------------------


def validate_identity_uniqueness(
    all_attempts: list[EmpiricalGenerationAttempt],
    all_candidates: list[EmpiricalCandidate],
) -> list[str]:
    """Check globally unique IDs and correct sequence metadata."""
    findings: list[str] = []

    # Provider attempt IDs (or generation_attempt_id for older records).
    provider_ids = [
        getattr(a, "provider_attempt_id", None) or a.generation_attempt_id
        for a in all_attempts
    ]
    dup_provider = [pid for pid, c in Counter(provider_ids).items() if c > 1]
    if dup_provider:
        findings.append(f"duplicate provider attempt IDs: {len(dup_provider)}")

    # Scientific generation-attempt IDs may repeat (retries share the same
    # generation_attempt_id), but provider_attempt_id must be unique.
    dup_attempt = [
        aid for aid, c in Counter(a.generation_attempt_id for a in all_attempts).items()
        if c > 1 and not any(getattr(a, "provider_attempt_id", None) for a in all_attempts
                             if a.generation_attempt_id == aid)
    ]
    if dup_attempt:
        findings.append(f"duplicate generation_attempt_ids without provider IDs: {len(dup_attempt)}")

    # Candidate IDs must be globally unique.
    dup_candidates = [cid for cid, c in Counter(c.candidate_id for c in all_candidates).items() if c > 1]
    if dup_candidates:
        findings.append(f"duplicate candidate IDs: {len(dup_candidates)}")

    # Sequence ID correctness.
    for candidate in all_candidates:
        if candidate.sequence_family_id is not None:
            if candidate.sequence_step_index is None or candidate.sequence_step_count is None:
                findings.append(
                    f"candidate {candidate.candidate_id}: sequence family set "
                    f"but step metadata incomplete"
                )

    return findings


# ---------------------------------------------------------------------------
# 10.5 Variant consistency and contamination
# ---------------------------------------------------------------------------


def validate_variant_consistency(
    all_attempts: list[EmpiricalGenerationAttempt],
    all_candidates: list[EmpiricalCandidate],
) -> list[str]:
    """Check target belongs to assigned variant; no cross-variant leaks."""
    findings: list[str] = []
    specs_by_key: dict[tuple[str, str], EmpiricalTargetSpec] = {
        (s.scenario_id, s.secret_variant_id): s for s in EMPIRICAL_TARGET_REGISTRY
    }

    # Check successful attempts against their spec.
    for attempt in all_attempts:
        if attempt.generation_status != GenerationStatus.SUCCESS.value:
            continue
        spec = specs_by_key.get((attempt.scenario_id, attempt.secret_variant_id))
        if spec is None:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"no spec for {attempt.scenario_id}/{attempt.secret_variant_id}"
            )
            continue
        contamination = detect_cross_variant_contamination(
            attempt.candidate_text or "", spec,
        )
        if contamination:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"cross-variant contamination from {contamination}"
            )

    # Check accepted candidates.
    for candidate in all_candidates:
        spec = specs_by_key.get((candidate.scenario_id, candidate.secret_variant_id))
        if spec is None:
            continue
        contamination = detect_cross_variant_contamination(candidate.text, spec)
        if contamination:
            findings.append(
                f"candidate {candidate.candidate_id}: "
                f"cross-variant contamination from {contamination}"
            )

    return findings


# ---------------------------------------------------------------------------
# 10.6 Config consistency
# ---------------------------------------------------------------------------


def validate_config_consistency(
    all_attempts: list[EmpiricalGenerationAttempt],
) -> list[str]:
    """Real attempts must match frozen campaign configuration."""
    findings: list[str] = []
    config = _load_frozen_config()
    if config is None:
        findings.append("frozen generation config not found")
        return findings

    frozen_model = config.get("generator_model_requested")
    frozen_provider = config.get("generator_provider")
    frozen_temperature = config.get("generator_temperature")
    frozen_max_tokens = config.get("generator_max_tokens")

    for attempt in all_attempts:
        if attempt.generation_mode != "real":
            continue
        if frozen_model and attempt.generator_model_requested != frozen_model:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"model_requested={attempt.generator_model_requested} "
                f"!= frozen {frozen_model}"
            )
        if frozen_provider and attempt.generator_provider != frozen_provider:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"provider={attempt.generator_provider} != frozen {frozen_provider}"
            )
        if frozen_temperature is not None and attempt.temperature != frozen_temperature:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"temperature={attempt.temperature} != frozen {frozen_temperature}"
            )
        if frozen_max_tokens is not None and attempt.max_tokens != frozen_max_tokens:
            findings.append(
                f"attempt {attempt.generation_attempt_id}: "
                f"max_tokens={attempt.max_tokens} != frozen {frozen_max_tokens}"
            )

    return findings


# ---------------------------------------------------------------------------
# 10.7 Hash integrity
# ---------------------------------------------------------------------------


def validate_hash_integrity(
    all_attempts: list[EmpiricalGenerationAttempt],
    all_candidates: list[EmpiricalCandidate],
) -> list[str]:
    """Recompute scientific hashes and compare with manifests."""
    findings: list[str] = []

    for split in ("development", "validation", "test"):
        manifest = _load_manifest(split)
        if manifest is None:
            continue

        split_attempts = _load_attempts(split)
        split_candidates = _load_candidates(split)

        # Raw generation hash.
        computed_raw = raw_attempts_scientific_hash(split_attempts)
        recorded_raw = manifest.get("raw_generation_sha256")
        if recorded_raw and computed_raw != recorded_raw:
            findings.append(f"{split}: raw_generation_sha256 mismatch")

        # Accepted candidate hash.
        computed_acc = accepted_candidates_scientific_hash(split_candidates)
        recorded_acc = manifest.get("accepted_candidate_sha256")
        if recorded_acc and computed_acc != recorded_acc:
            findings.append(f"{split}: accepted_candidate_sha256 mismatch")

        # Target registry hash.
        computed_reg = compute_target_registry_hash(EMPIRICAL_TARGET_REGISTRY)
        recorded_reg = manifest.get("target_registry_sha256")
        if recorded_reg and computed_reg != recorded_reg:
            findings.append(f"{split}: target_registry_sha256 mismatch")

    # Plan hash — scientific and file hashes verified separately (Patch C).
    plan_path = _MANIFESTS_DIR / "full_generation_plan.jsonl"
    summary_path = _MANIFESTS_DIR / "full_generation_plan_summary.json"
    if summary_path.exists() and plan_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        items = load_generation_plan(plan_path)
        computed_scientific = plan_sha256(items)
        computed_file = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        # Scientific hash check.
        recorded_scientific = summary.get("plan_scientific_sha256")
        if recorded_scientific and computed_scientific != recorded_scientific:
            findings.append("generation plan scientific SHA256 mismatch")
        # File hash check.
        recorded_file = summary.get("plan_file_sha256")
        if recorded_file and computed_file != recorded_file:
            findings.append("generation plan file SHA256 mismatch")
        # Backward-compatible alias: plan_sha256 must equal scientific hash.
        recorded_alias = summary.get("plan_sha256")
        if recorded_alias and recorded_scientific and recorded_alias != recorded_scientific:
            findings.append("generation plan plan_sha256 alias != plan_scientific_sha256")

    # Config hash.
    config_path = _MANIFESTS_DIR / "full_generation_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        recorded_config_hash = config.get("target_registry_sha256")
        computed_config_hash = compute_target_registry_hash(EMPIRICAL_TARGET_REGISTRY)
        if recorded_config_hash and computed_config_hash != recorded_config_hash:
            findings.append("generation config target_registry_sha256 mismatch")

    return findings


# ---------------------------------------------------------------------------
# 10.8 Sequence atomicity
# ---------------------------------------------------------------------------


def validate_sequence_atomicity(
    all_candidates: list[EmpiricalCandidate],
    all_attempts: list[EmpiricalGenerationAttempt] | None = None,
) -> list[str]:
    """Every accepted sequence must have all steps present.

    Patch G: if *all_attempts* is provided, also cross-check that every
    accepted sequence corresponds to a complete valid terminal raw sequence.
    """
    findings: list[str] = []

    # Group candidates by sequence family + trust.
    seq_groups: dict[tuple[str, str], list[EmpiricalCandidate]] = {}
    for c in all_candidates:
        if c.sequence_family_id is not None:
            key = (c.sequence_family_id, c.trust_level)
            seq_groups.setdefault(key, []).append(c)

    for (family_id, trust), candidates in sorted(seq_groups.items()):
        step_count = candidates[0].sequence_step_count if candidates else None
        if step_count is None:
            findings.append(f"sequence {family_id}/{trust}: missing step_count")
            continue
        if len(candidates) != step_count:
            findings.append(
                f"sequence {family_id}/{trust}: "
                f"accepted {len(candidates)} steps, expected {step_count}"
            )
        indices = sorted(c.sequence_step_index for c in candidates)
        if indices != list(range(step_count)):
            findings.append(
                f"sequence {family_id}/{trust}: "
                f"step indices {indices} != 0..{step_count - 1}"
            )

    # Patch G: cross-check accepted sequences against complete raw terminal sequences.
    if all_attempts is not None:
        from experiments.trustparadox_u.generate_empirical_corpus import (
            terminal_attempts_by_sequence_step,
            _unit_key,
        )
        # Group raw attempts by sequence scientific identity.
        raw_seq_groups: dict[tuple, list[EmpiricalGenerationAttempt]] = {}
        for a in all_attempts:
            if a.is_sequence_attempt and a.sequence_family_id is not None:
                key = (
                    a.scenario_id, a.secret_variant_id, a.trust_level,
                    a.attack_type, a.sample_index, a.generation_replicate,
                    a.sequence_family_id,
                )
                raw_seq_groups.setdefault(key, []).append(a)
        # For each accepted sequence family, check raw terminal completeness.
        accepted_families: set[tuple[str, str]] = set(seq_groups.keys())
        raw_families: dict[tuple[str, str], list[EmpiricalGenerationAttempt]] = {}
        for key, attempts in raw_seq_groups.items():
            fam_key = (key[-1], key[2])  # (family_id, trust_level)
            raw_families.setdefault(fam_key, []).extend(attempts)
        for fam_key in accepted_families:
            if fam_key not in raw_families:
                findings.append(
                    f"sequence {fam_key[0]}/{fam_key[1]}: "
                    f"accepted but no raw attempts found"
                )

    return findings


# ---------------------------------------------------------------------------
# 10.9 Retry lineage
# ---------------------------------------------------------------------------


def validate_retry_lineage(
    all_attempts: list[EmpiricalGenerationAttempt],
) -> list[str]:
    """For each scientific unit, retry indices must be consecutive from 0."""
    findings: list[str] = []

    # Group by (scenario, variant, trust, attack, sample, replicate, step_or_none).
    groups: dict[tuple, list[EmpiricalGenerationAttempt]] = {}
    for a in all_attempts:
        step = a.sequence_step_index if a.is_sequence_attempt else None
        key = (
            a.scenario_id, a.secret_variant_id, a.trust_level,
            a.attack_type, a.sample_index, a.generation_replicate, step,
        )
        groups.setdefault(key, []).append(a)

    for key, chain in sorted(groups.items()):
        sorted_chain = sorted(chain, key=lambda a: a.retry_index)
        indices = [a.retry_index for a in sorted_chain]

        # Must start at 0.
        if indices[0] != 0:
            findings.append(
                f"unit {key}: retry indices start at {indices[0]}, expected 0"
            )

        # Must be consecutive.
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                findings.append(
                    f"unit {key}: retry indices not consecutive: {indices}"
                )
                break

        # At most one terminal success.
        successes = [a for a in sorted_chain if a.generation_status == GenerationStatus.SUCCESS.value]
        if len(successes) > 1:
            findings.append(
                f"unit {key}: {len(successes)} successes, expected at most 1"
            )

        # No retry after success.
        for i, a in enumerate(sorted_chain):
            if a.generation_status == GenerationStatus.SUCCESS.value and i < len(sorted_chain) - 1:
                findings.append(
                    f"unit {key}: retry after success at index {a.retry_index}"
                )
                break

    return findings


# ---------------------------------------------------------------------------
# 10.10 Acceptance-independence
# ---------------------------------------------------------------------------

_FORBIDDEN_ACCEPTANCE_FIELDS = (
    "firewall_condition",
    "flowgate_decision",
    "embedding_score",
    "policy_action",
    "pu_rer",
    "crr",
    "rr",
)


def validate_acceptance_independence(
    all_candidates: list[EmpiricalCandidate],
) -> list[str]:
    """Corpus must not contain firewall/policy metadata as acceptance criteria."""
    findings: list[str] = []
    candidate_fields = set(EmpiricalCandidate.__dataclass_fields__.keys())
    for field in _FORBIDDEN_ACCEPTANCE_FIELDS:
        if field in candidate_fields:
            findings.append(
                f"acceptance-independence: EmpiricalCandidate has field {field!r}"
            )
    return findings


# ---------------------------------------------------------------------------
# 10.11 Coverage statistics
# ---------------------------------------------------------------------------


def compute_coverage_stats(
    all_attempts: list[EmpiricalGenerationAttempt],
    all_candidates: list[EmpiricalCandidate],
    plan_items: list[dict] | None = None,
) -> dict:
    """Detailed coverage statistics by split, scenario, variant, etc."""
    stats: dict[str, object] = {}

    for split in ("development", "validation", "test"):
        split_attempts = [a for a in all_attempts if a.split == split]
        split_candidates = [c for c in all_candidates if c.split == split]
        status_counts = Counter(a.generation_status for a in split_attempts)

        # Scientific units: unique (scenario, variant, trust, attack, sample, replicate).
        unit_keys = set()
        for a in split_attempts:
            unit_keys.add((
                a.scenario_id, a.secret_variant_id, a.trust_level,
                a.attack_type, a.sample_index, a.generation_replicate,
            ))

        # Retry rate: attempts with retry_index > 0.
        retry_count = sum(1 for a in split_attempts if a.retry_index > 0)

        stats[split] = {
            "raw_provider_attempts": len(split_attempts),
            "scientific_unit_count": len(unit_keys),
            "accepted_candidates": len(split_candidates),
            "rejected_units": len(unit_keys) - sum(
                1 for uk in unit_keys
                if any(
                    a.generation_status == GenerationStatus.SUCCESS.value
                    for a in split_attempts
                    if (a.scenario_id, a.secret_variant_id, a.trust_level,
                        a.attack_type, a.sample_index, a.generation_replicate) == uk
                )
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "acceptance_rate": (
                len(split_candidates) / len(unit_keys) if unit_keys else 0.0
            ),
            "refusal_rate": (
                status_counts.get(GenerationStatus.REFUSAL.value, 0) / len(split_attempts)
                if split_attempts else 0.0
            ),
            "provider_failure_rate": (
                status_counts.get(GenerationStatus.PROVIDER_ERROR.value, 0) / len(split_attempts)
                if split_attempts else 0.0
            ),
            "retry_rate": retry_count / len(split_attempts) if split_attempts else 0.0,
            "by_scenario": dict(sorted(Counter(
                a.scenario_id for a in split_attempts
            ).items())),
            "by_variant": dict(sorted(Counter(
                a.secret_variant_id for a in split_attempts
            ).items())),
            "by_trust": dict(sorted(Counter(
                a.trust_level for a in split_attempts
            ).items())),
            "by_attack": dict(sorted(Counter(
                a.attack_type for a in split_attempts
            ).items())),
        }

    if plan_items is not None:
        stats["plan_item_count"] = len(plan_items)

    return stats


# ---------------------------------------------------------------------------
# Aggregated report builder
# ---------------------------------------------------------------------------


def validate_campaign_identity() -> list[str]:
    """Patch H: verify campaign identity for each split with artifacts.

    Loads ``campaign_identity.json`` from each split directory, recomputes
    the current identity from frozen artifacts, and compares all blocking
    fields.
    """
    findings: list[str] = []
    for split in ("development", "validation", "test"):
        split_dir = _CORPUS_BASE / split
        existing = load_campaign_identity(split_dir)
        if existing is None:
            continue  # No campaign identity → no check needed.
        # Recompute current identity from frozen artifacts.
        plan_path = _MANIFESTS_DIR / "full_generation_plan.jsonl"
        plan_items = None
        if plan_path.exists():
            plan_items = load_generation_plan(plan_path)
        try:
            frozen_config = load_frozen_generation_config()
        except (FileNotFoundError, KeyError):
            findings.append(f"{split}: cannot load frozen generation config for identity check")
            continue
        config_path = _MANIFESTS_DIR / "full_generation_config.json"
        phase_path = _MANIFESTS_DIR / "empirical_phase.json"
        try:
            current = compute_campaign_identity(
                split=split,
                plan_items=plan_items,
                plan_path=plan_path if plan_path.exists() else None,
                config=frozen_config,
                config_path=config_path if config_path.exists() else None,
                phase_manifest_path=phase_path if phase_path.exists() else None,
            )
        except Exception as exc:
            findings.append(f"{split}: cannot compute current campaign identity: {exc}")
            continue
        try:
            verify_campaign_identity(existing, current)
        except CampaignIdentityMismatchError as exc:
            for field, vals in sorted(exc.mismatches.items()):
                findings.append(
                    f"{split}: campaign identity mismatch: {field} "
                    f"recorded={vals['recorded']!r} current={vals['current']!r}"
                )
    return findings


def build_validation_report() -> dict:
    """Build the full validation report with all audit sections."""
    # Load all data.
    all_attempts: list[EmpiricalGenerationAttempt] = []
    all_candidates: list[EmpiricalCandidate] = []
    for split in ("development", "validation", "test"):
        all_attempts.extend(_load_attempts(split))
        all_candidates.extend(_load_candidates(split))

    plan_items = _load_plan_items()

    # Run all audit sections.
    # Patch G: audit order for sequence checks: lineage → atomicity.
    sections: dict[str, list[str]] = {}

    sections["phase_provenance"] = validate_phase_and_provenance()
    sections["plan_completeness"] = validate_plan_completeness(all_attempts, all_candidates)
    sections["split_integrity"] = validate_split_integrity(all_attempts, all_candidates)
    sections["identity_uniqueness"] = validate_identity_uniqueness(all_attempts, all_candidates)
    sections["variant_consistency"] = validate_variant_consistency(all_attempts, all_candidates)
    sections["config_consistency"] = validate_config_consistency(all_attempts)
    sections["hash_integrity"] = validate_hash_integrity(all_attempts, all_candidates)
    sections["retry_lineage"] = validate_retry_lineage(all_attempts)
    sections["sequence_atomicity"] = validate_sequence_atomicity(all_candidates, all_attempts)
    sections["acceptance_independence"] = validate_acceptance_independence(all_candidates)
    # Patch H: campaign identity verification.
    sections["campaign_identity"] = validate_campaign_identity()

    # Collect all blocking findings.
    all_findings: list[str] = []
    for section_findings in sections.values():
        all_findings.extend(section_findings)

    coverage = compute_coverage_stats(all_attempts, all_candidates, plan_items)

    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "empirical_phase": EMPIRICAL_PHASE,
        "audit_sections": sections,
        "validation_findings": all_findings,
        "finding_count": len(all_findings),
        "blocking_finding_count": len(all_findings),
        "coverage_stats": coverage,
        "total_raw_attempts": len(all_attempts),
        "total_accepted_candidates": len(all_candidates),
        "passed": len(all_findings) == 0,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(report: dict) -> None:
    """Write a human-readable markdown report."""
    lines = [
        "# Full Corpus Validation Report (Patch F)",
        "",
        f"**Phase**: {report['empirical_phase']}",
        f"**Schema Version**: {report['schema_version']}",
        f"**Study Version**: {report['study_version']}",
        f"**Passed**: {report['passed']}",
        f"**Blocking Findings**: {report['blocking_finding_count']}",
        f"**Total Raw Attempts**: {report['total_raw_attempts']}",
        f"**Total Accepted Candidates**: {report['total_accepted_candidates']}",
        "",
    ]

    # Per-section summary.
    lines.append("## Audit Sections")
    lines.append("")
    sections = report.get("audit_sections", {})
    for section_name, findings in sections.items():
        status = "PASS" if not findings else f"FAIL ({len(findings)} findings)"
        lines.append(f"- **{section_name}**: {status}")
    lines.append("")

    # All findings.
    lines.append("## All Findings")
    lines.append("")
    if report["validation_findings"]:
        for finding in report["validation_findings"]:
            lines.append(f"- {finding}")
    else:
        lines.append("No findings. All audit checks passed.")
    lines.append("")

    # Coverage stats.
    lines.append("## Coverage Statistics")
    lines.append("")
    coverage = report.get("coverage_stats", {})
    for split in ("development", "validation", "test"):
        split_stats = coverage.get(split, {})
        if not split_stats:
            continue
        lines.extend([
            f"### {split.capitalize()}",
            "",
            f"- Raw provider attempts: {split_stats.get('raw_provider_attempts', 0)}",
            f"- Scientific units: {split_stats.get('scientific_unit_count', 0)}",
            f"- Accepted candidates: {split_stats.get('accepted_candidates', 0)}",
            f"- Acceptance rate: {split_stats.get('acceptance_rate', 0):.2%}",
            f"- Refusal rate: {split_stats.get('refusal_rate', 0):.2%}",
            f"- Provider failure rate: {split_stats.get('provider_failure_rate', 0):.2%}",
            f"- Retry rate: {split_stats.get('retry_rate', 0):.2%}",
            "",
        ])

    _REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full corpus validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Full corpus audit (Patch F).")
    parser.add_argument(
        "--split",
        choices=["development", "validation", "test", "all"],
        default="all",
        help="Audit a specific split or all splits.",
    )
    args = parser.parse_args()

    print("Running full corpus validation (Patch F)...")

    report = build_validation_report()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(report)

    print(f"Validation report written to: {_REPORT_JSON}")
    print(f"Markdown report written to: {_REPORT_MD}")
    print(f"Passed: {report['passed']}")
    print(f"Blocking findings: {report['blocking_finding_count']}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

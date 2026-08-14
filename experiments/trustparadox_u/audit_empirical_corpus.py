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
import sys
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
    plan_items_for_split,
    plan_sha256,
    generation_gate_path,
    load_generation_gate,
    update_generation_gate_after_audit,
)
from experiments.trustparadox_u.campaign_identity import (
    CAMPAIGN_IDENTITY_FILENAME,
    CampaignIdentity,
    campaign_identity_sha256,
    compute_campaign_identity,
    load_campaign_identity,
    split_has_artifacts,
    verify_campaign_identity,
    CampaignIdentityMismatchError,
)
from experiments.trustparadox_u.empirical_generation_plan import (
    FrozenGenerationConfig,
    load_frozen_generation_config,
)

# ---------------------------------------------------------------------------
# Patch C: explicit audit scope
# ---------------------------------------------------------------------------

AuditScope = str  # "development" | "validation" | "test" | "all"


def audited_splits(scope: str) -> tuple[str, ...]:
    """Return the tuple of split names covered by *scope*."""
    if scope == "all":
        return ("development", "validation", "test")
    return (EmpiricalSplit(scope).value,)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "manifests"
_CORPUS_BASE = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_OUTPUT_DIR = _CORPUS_BASE
_REPORT_JSON = _OUTPUT_DIR / "full_corpus_validation_report.json"
_REPORT_MD = _OUTPUT_DIR / "full_corpus_validation_report.md"

# ---------------------------------------------------------------------------
# Patch B (Phase 3 Final): canonical artifact classification constants
# ---------------------------------------------------------------------------

ARTIFACT_CLASS_SMOKE = "development_smoke"
ARTIFACT_CLASS_REAL_API_PREFLIGHT = "real_api_preflight"
ARTIFACT_CLASS_EMPIRICAL_CORPUS = "empirical_corpus"

RESEARCH_USE_DIAGNOSTIC = "diagnostic_only"
RESEARCH_USE_PENDING_ANNOTATION = "pending_annotation_and_replay"

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
    *,
    target_splits: tuple[str, ...] | None = None,
) -> list[str]:
    """Every planned plan_item_id must have a matching generation_attempt_id.

    Patch F: uses exact ``plan_item_id`` \u2194 ``generation_attempt_id``
    comparison instead of coarse scientific-unit identity.

    Patch C: when *target_splits* is given, only plan items for those
    splits are checked.
    """
    findings: list[str] = []
    plan_path = _MANIFESTS_DIR / "full_generation_plan.jsonl"
    if not plan_path.exists():
        findings.append("generation plan not found")
        return findings

    all_plan_items = load_generation_plan(plan_path)
    if not all_plan_items:
        findings.append("generation plan empty")
        return findings

    # Filter plan items to the audited splits.
    if target_splits is not None:
        scope_plan = [it for it in all_plan_items if it.split in target_splits]
    else:
        scope_plan = all_plan_items

    planned_ids = {it.plan_item_id for it in scope_plan}
    observed_ids = {a.generation_attempt_id for a in all_attempts}

    missing = planned_ids - observed_ids
    unexpected = observed_ids - planned_ids

    if missing:
        findings.append(
            f"plan completeness: {len(missing)} planned plan-item IDs "
            f"have no raw attempts"
        )
    if unexpected:
        findings.append(
            f"plan completeness: {len(unexpected)} observed generation IDs "
            f"are not in the planned set"
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
    *,
    target_splits: tuple[str, ...] | None = None,
) -> list[str]:
    """Recompute scientific hashes and compare with manifests.

    Patch C: when *target_splits* is given, only check those splits'
    per-split hashes.  Global hashes (plan, config) are always checked.
    """
    findings: list[str] = []

    splits_to_check = target_splits or ("development", "validation", "test")
    for split in splits_to_check:
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

        # Patch I: campaign identity hash.
        recorded_id_hash = manifest.get("campaign_identity_sha256")
        if recorded_id_hash:
            loaded_id = load_campaign_identity(_CORPUS_BASE / split)
            if loaded_id is None:
                findings.append(
                    f"{split}: campaign_identity_sha256 in manifest "
                    f"but campaign_identity.json missing"
                )
            else:
                computed_id_hash = campaign_identity_sha256(loaded_id)
                if computed_id_hash != recorded_id_hash:
                    findings.append(
                        f"{split}: campaign_identity_sha256 mismatch"
                    )
        elif (
            manifest.get("artifact_class") != "development_smoke"
            or manifest.get("empirical_phase") == "E3_CORPUS_GENERATION"
        ):
            # E3 corpus manifests must have campaign identity hash.
            if split_has_artifacts(_CORPUS_BASE / split):
                findings.append(
                    f"{split}: missing campaign_identity_sha256 in manifest"
                )

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
            terminal_attempt_for_retry_chain,
            expected_sequence_steps_from_plan,
            _unit_key,
        )

        # Build plan-based expected step counts (Patch H).
        # Patch I: when a frozen plan is available, inconsistent sequence
        # definitions are fatal — propagate the error.
        plan_step_index: dict[tuple, int] = {}
        plan_path = _MANIFESTS_DIR / "full_generation_plan.jsonl"
        if plan_path.exists():
            typed_plan = load_generation_plan(plan_path)
            plan_step_index = expected_sequence_steps_from_plan(typed_plan)

        # Group raw attempts by full sequence scientific identity.
        raw_seq_groups: dict[tuple, list[EmpiricalGenerationAttempt]] = {}
        for a in all_attempts:
            if a.is_sequence_attempt and a.sequence_family_id is not None:
                key = (
                    a.scenario_id, a.secret_variant_id, a.trust_level,
                    a.attack_type, a.sample_index, a.generation_replicate,
                    a.sequence_family_id,
                )
                raw_seq_groups.setdefault(key, []).append(a)

        # Build accepted sequence groups by full scientific identity.
        accepted_seq_groups: dict[tuple, list[EmpiricalCandidate]] = {}
        for c in all_candidates:
            if c.sequence_family_id is not None:
                key = (
                    c.scenario_id, c.secret_variant_id, c.trust_level,
                    c.attack_type, c.sample_index, c.generation_replicate,
                    c.sequence_family_id,
                )
                accepted_seq_groups.setdefault(key, []).append(c)

        for acc_key, acc_candidates in sorted(accepted_seq_groups.items()):
            family_id = acc_key[-1]

            # 14.2: raw attempts must exist.
            raw_attempts = raw_seq_groups.get(acc_key)
            if raw_attempts is None:
                findings.append(
                    f"sequence {family_id}/{acc_key[2]}: "
                    f"accepted but no raw attempts found"
                )
                continue

            # 14.3: validate retry lineage per step.
            step_groups: dict[int, list[EmpiricalGenerationAttempt]] = {}
            for a in raw_attempts:
                step_idx = a.sequence_step_index
                if step_idx is not None:
                    step_groups.setdefault(step_idx, []).append(a)

            lineage_valid = True
            for step_idx in sorted(step_groups):
                chain = sorted(step_groups[step_idx], key=lambda a: a.retry_index)
                try:
                    terminal_attempt_for_retry_chain(chain)
                except (ValueError, Exception) as exc:
                    findings.append(
                        f"sequence {family_id}/{acc_key[2]} step {step_idx}: "
                        f"invalid retry lineage: {exc}"
                    )
                    lineage_valid = False

            if not lineage_valid:
                continue

            # 14.4: determine expected step count.
            plan_expected = plan_step_index.get(acc_key)
            if plan_expected is not None:
                expected_steps = plan_expected
            elif acc_candidates and acc_candidates[0].sequence_step_count is not None:
                expected_steps = acc_candidates[0].sequence_step_count
            else:
                continue  # Cannot determine expected steps.

            # 14.5: require complete terminal step indexes 0..N-1.
            terminal_step_indices = sorted(step_groups.keys())
            if terminal_step_indices != list(range(expected_steps)):
                findings.append(
                    f"sequence {family_id}/{acc_key[2]}: "
                    f"raw terminal steps {terminal_step_indices} "
                    f"!= expected 0..{expected_steps - 1}"
                )
                continue

            # 14.6: validate scientific sequence structure.
            try:
                terminal_steps = terminal_attempts_by_sequence_step(raw_attempts)
            except (ValueError, Exception) as exc:
                findings.append(
                    f"sequence {family_id}/{acc_key[2]}: "
                    f"cannot reduce terminal steps: {exc}"
                )
                continue

            structure_problems = validate_sequence_structure(terminal_steps)
            if structure_problems:
                findings.append(
                    f"sequence {family_id}/{acc_key[2]}: "
                    f"structure invalid: {'; '.join(structure_problems)}"
                )
                continue

            # 14.7: cross-check accepted candidates.
            acc_step_count = len(acc_candidates)
            if acc_step_count != expected_steps:
                findings.append(
                    f"sequence {family_id}/{acc_key[2]}: "
                    f"accepted {acc_step_count} steps, expected {expected_steps}"
                )
            acc_indices = sorted(c.sequence_step_index for c in acc_candidates)
            if acc_indices != list(range(expected_steps)):
                findings.append(
                    f"sequence {family_id}/{acc_key[2]}: "
                    f"accepted step indices {acc_indices} != 0..{expected_steps - 1}"
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


def validate_campaign_identity(
    *,
    target_splits: tuple[str, ...] | None = None,
) -> list[str]:
    """Patch F: verify campaign identity for each split with artifacts.

    Loads ``campaign_identity.json`` from each split directory, recomputes
    the current identity from frozen artifacts, and compares all blocking
    fields.

    Patch C: when *target_splits* is given, only audit those splits.
    """
    findings: list[str] = []
    splits_to_check = target_splits or ("development", "validation", "test")
    for split in splits_to_check:
        split_dir = _CORPUS_BASE / split
        existing = load_campaign_identity(split_dir)
        if existing is None:
            # Patch F: if artifacts exist but identity is missing → blocking.
            if split_has_artifacts(split_dir):
                findings.append(
                    f"{split}: campaign_identity.json missing for generated split"
                )
            continue
        # Patch A (Phase 3 Final): filter to split-specific plan items so
        # the scientific hash matches what generation computed.
        plan_path = _MANIFESTS_DIR / "full_generation_plan.jsonl"
        split_plan_items = None
        if plan_path.exists():
            full_plan_items = load_generation_plan(plan_path)
            split_plan_items = plan_items_for_split(full_plan_items, split)
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
                plan_items=split_plan_items,
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
            # Endpoint fields cannot be recomputed during audit (api_base
            # is not stored).  Filter them out; endpoint consistency is
            # enforced separately by validate_endpoint_consistency().
            _endpoint_fields = {
                "serving_endpoint_host",
                "serving_endpoint_sha256",
                "api_protocol",
            }
            for field, vals in sorted(exc.mismatches.items()):
                if field in _endpoint_fields:
                    continue
                findings.append(
                    f"{split}: campaign identity mismatch: {field} "
                    f"recorded={vals['recorded']!r} current={vals['current']!r}"
                )
        # Patch H: if any endpoint field is non-empty, all must be non-empty.
        ep_fields = {
            "host": existing.serving_endpoint_host,
            "sha": existing.serving_endpoint_sha256,
            "proto": existing.api_protocol,
        }
        non_empty = [k for k, v in ep_fields.items() if v]
        empty = [k for k, v in ep_fields.items() if not v]
        if non_empty and empty:
            findings.append(
                f"{split}: incomplete endpoint provenance: "
                f"non-empty={sorted(non_empty)}, empty={sorted(empty)}"
            )
        # Patch A (fail-closed): empirical corpus campaigns require all
        # endpoint provenance fields to be non-empty.
        manifest = _load_manifest(split)
        is_empirical = (
            manifest is not None
            and manifest.get("artifact_class")
            == ARTIFACT_CLASS_EMPIRICAL_CORPUS
        )
        if is_empirical and not non_empty and empty:
            findings.append(
                f"{split}: empirical corpus campaign identity "
                f"missing endpoint provenance: "
                + ", ".join(sorted(empty))
            )
    return findings


# ---------------------------------------------------------------------------
# Patch B (Phase 3 Final): artifact classification audit
# ---------------------------------------------------------------------------


def validate_artifact_classification(
    *,
    target_splits: tuple[str, ...] | None = None,
) -> list[str]:
    """Reject final corpus splits labeled as smoke/diagnostic.

    For every split with a corpus manifest, require:
    - ``artifact_class == empirical_corpus``
    - ``research_use == pending_annotation_and_replay``
    """
    findings: list[str] = []
    splits_to_check = target_splits or ("development", "validation", "test")
    for split in splits_to_check:
        manifest = _load_manifest(split)
        if manifest is None:
            continue  # no manifest → nothing to check
        ac = manifest.get("artifact_class", "")
        ru = manifest.get("research_use", "")
        if ac != ARTIFACT_CLASS_EMPIRICAL_CORPUS:
            findings.append(
                f"{split}: artifact_class is {ac!r}, "
                f"expected {ARTIFACT_CLASS_EMPIRICAL_CORPUS!r}"
            )
        if ru != RESEARCH_USE_PENDING_ANNOTATION:
            findings.append(
                f"{split}: research_use is {ru!r}, "
                f"expected {RESEARCH_USE_PENDING_ANNOTATION!r}"
            )
    return findings


# ---------------------------------------------------------------------------
# Patch J (Phase 3 Final): required fields for empirical_corpus manifests
# ---------------------------------------------------------------------------

# Required fields for empirical_corpus artifact class.
_EMPIRICAL_CORPUS_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "protocol_version",
    "study_version",
    "empirical_phase",
    "generation_mode",
    "artifact_class",
    "research_use",
    "repository_commit",
    "repository_clean",
    "environment_lock_hash",
    "target_spec_sha256",
    "prompt_manifest_sha256",
    "campaign_identity_sha256",
    "full_generation_plan_sha256",
    "split_generation_plan_sha256",
    "split_plan_item_count",
    "raw_generation_sha256",
    "accepted_candidate_sha256",
    "attempt_count",
    "accepted_candidate_count",
    "split_counts",
    "trust_counts",
    "attack_counts",
    "scenario_counts",
    "generated_at",
)


def validate_empirical_corpus_required_fields(
    *,
    target_splits: tuple[str, ...] | None = None,
) -> list[str]:
    """Require all mandatory fields for empirical_corpus manifests.

    Patch J (Phase 3 Final): any missing required field is a blocking finding.
    Also validates that the manifest has the correct empirical_corpus values:
    - generation_mode == real
    - artifact_class == empirical_corpus
    - research_use == pending_annotation_and_replay
    - empirical_phase == E3_CORPUS_GENERATION
    - repository_clean == true
    """
    findings: list[str] = []
    splits_to_check = target_splits or ("development", "validation", "test")
    for split in splits_to_check:
        manifest = _load_manifest(split)
        if manifest is None:
            continue  # no manifest → nothing to check
        # Check artifact_class first — only apply these rules to empirical_corpus.
        if manifest.get("artifact_class") != ARTIFACT_CLASS_EMPIRICAL_CORPUS:
            continue
        # Check all required fields are present and non-empty.
        missing_fields: list[str] = []
        for field in _EMPIRICAL_CORPUS_REQUIRED_FIELDS:
            value = manifest.get(field)
            if value is None or value == "":
                missing_fields.append(field)
        if missing_fields:
            findings.append(
                f"{split}: empirical_corpus manifest missing required fields: "
                + ", ".join(missing_fields)
            )
        # Validate required values.
        if manifest.get("generation_mode") != "real":
            findings.append(
                f"{split}: empirical_corpus generation_mode must be 'real', "
                f"got {manifest.get('generation_mode')!r}"
            )
        if manifest.get("empirical_phase") != EMPIRICAL_PHASE:
            findings.append(
                f"{split}: empirical_corpus empirical_phase must be "
                f"{EMPIRICAL_PHASE!r}, got {manifest.get('empirical_phase')!r}"
            )
        if manifest.get("repository_clean") is not True:
            findings.append(
                f"{split}: empirical_corpus repository_clean must be true, "
                f"got {manifest.get('repository_clean')!r}"
            )
    return findings


# ---------------------------------------------------------------------------
# Patch H (Phase 3 Final): manifest / campaign identity provenance
# ---------------------------------------------------------------------------


def validate_manifest_provenance(
    *,
    target_splits: tuple[str, ...] | None = None,
) -> list[str]:
    """Cross-check manifest fields against campaign identity.

    Verifies:
    - ``manifest.campaign_identity_sha256 == SHA256(campaign_identity.json)``
    - ``manifest.split_generation_plan_sha256 == campaign_identity.generation_plan_scientific_sha256``
    - ``manifest.repository_commit == campaign_identity.created_from_commit``

    Patch K (Phase 3 Final): for empirical_corpus manifests, all checks are
    mandatory — missing fields are blocking findings.
    """
    findings: list[str] = []
    splits_to_check = target_splits or ("development", "validation", "test")
    for split in splits_to_check:
        manifest = _load_manifest(split)
        if manifest is None:
            continue
        split_dir = _CORPUS_BASE / split
        identity = load_campaign_identity(split_dir)
        if identity is None:
            continue
        # Only apply mandatory checks to empirical_corpus manifests.
        is_empirical = manifest.get("artifact_class") == ARTIFACT_CLASS_EMPIRICAL_CORPUS
        # 1. campaign_identity_sha256 binding.
        recorded_ci_hash = manifest.get("campaign_identity_sha256")
        if is_empirical and not recorded_ci_hash:
            findings.append(
                f"{split}: empirical_corpus manifest missing campaign_identity_sha256"
            )
        if recorded_ci_hash:
            identity_path = split_dir / CAMPAIGN_IDENTITY_FILENAME
            if identity_path.exists():
                computed_ci_hash = campaign_identity_sha256(identity)
                if computed_ci_hash != recorded_ci_hash:
                    findings.append(
                        f"{split}: manifest campaign_identity_sha256 mismatch: "
                        f"recorded={recorded_ci_hash!r} computed={computed_ci_hash!r}"
                    )
        # 2. split-plan hash agreement.
        manifest_split_plan_hash = manifest.get("split_generation_plan_sha256")
        if is_empirical and not manifest_split_plan_hash:
            findings.append(
                f"{split}: empirical_corpus manifest missing split_generation_plan_sha256"
            )
        if manifest_split_plan_hash:
            if identity.generation_plan_scientific_sha256 != manifest_split_plan_hash:
                findings.append(
                    f"{split}: manifest split_plan_sha256 != campaign identity "
                    f"scientific_sha256: manifest={manifest_split_plan_hash!r} "
                    f"identity={identity.generation_plan_scientific_sha256!r}"
                )
        # 3. repository_commit agreement.
        manifest_commit = manifest.get("repository_commit")
        if is_empirical and not manifest_commit:
            findings.append(
                f"{split}: empirical_corpus manifest missing repository_commit"
            )
        if manifest_commit and identity.created_from_commit:
            if manifest_commit != identity.created_from_commit:
                findings.append(
                    f"{split}: manifest repository_commit != campaign identity "
                    f"created_from_commit: manifest={manifest_commit!r} "
                    f"identity={identity.created_from_commit!r}"
                )

        # --- Patch I: verify full-plan hash value ---
        full_plan_path = _MANIFESTS_DIR / "full_generation_plan.jsonl"
        if is_empirical and full_plan_path.exists():
            full_plan = load_generation_plan(full_plan_path)
            expected_full_hash = plan_sha256(full_plan)
            recorded_full_hash = manifest.get("full_generation_plan_sha256")
            if recorded_full_hash and recorded_full_hash != expected_full_hash:
                findings.append(
                    f"{split}: manifest full_generation_plan_sha256 mismatch: "
                    f"recorded={recorded_full_hash!r} expected={expected_full_hash!r}"
                )

            # --- Patch J: verify split plan item count ---
            split_items = plan_items_for_split(full_plan, split)
            recorded_split_count = manifest.get("split_plan_item_count")
            if recorded_split_count is not None and recorded_split_count != len(split_items):
                findings.append(
                    f"{split}: manifest split_plan_item_count mismatch: "
                    f"recorded={recorded_split_count!r} expected={len(split_items)!r}"
                )

            # --- Patch K: directly verify split plan hash ---
            expected_split_hash = plan_sha256(split_items)
            if manifest_split_plan_hash and manifest_split_plan_hash != expected_split_hash:
                findings.append(
                    f"{split}: manifest split_generation_plan_sha256 != "
                    f"frozen split plan hash: manifest={manifest_split_plan_hash!r} "
                    f"expected={expected_split_hash!r}"
                )
            if (identity.generation_plan_scientific_sha256
                    and identity.generation_plan_scientific_sha256 != expected_split_hash):
                findings.append(
                    f"{split}: campaign identity scientific_sha256 != "
                    f"frozen split plan hash: identity="
                    f"{identity.generation_plan_scientific_sha256!r} "
                    f"expected={expected_split_hash!r}"
                )
    return findings


# ---------------------------------------------------------------------------
# Patch G (Phase 3 Final): resolved-commit helper
# ---------------------------------------------------------------------------


def is_resolved_repository_commit(value: str | None) -> bool:
    """Return True only if *value* is a valid 40-char hex Git SHA."""
    if value is None:
        return False
    normalized = value.strip().lower()
    if not normalized or normalized == "unknown":
        return False
    if len(normalized) != 40:
        return False
    return all(ch in "0123456789abcdef" for ch in normalized)


# ---------------------------------------------------------------------------
# Patch F (Phase 3 Final): source commit consistency audit
# ---------------------------------------------------------------------------


def validate_source_commit_consistency() -> list[str]:
    """Require one source commit across all generated/audited splits.

    Collects ``campaign_identity.created_from_commit``,
    ``gate.source_commit``, and ``gate.audit_source_commit`` for every
    split that has a gate file.  All values must resolve to one commit.

    Patch L (Phase 3 Final): missing any required field is a blocking finding.
    """
    findings: list[str] = []
    commits_seen: set[str] = set()
    per_split: dict[str, dict[str, str]] = {}

    for split in ("development", "validation", "test"):
        gate_file = generation_gate_path(split, base=_CORPUS_BASE)
        if not gate_file.exists():
            continue
        gate = json.loads(gate_file.read_text(encoding="utf-8"))
        entry: dict[str, str] = {}
        gen_commit = gate.get("source_commit")
        audit_commit = gate.get("audit_source_commit")
        # Patch L: require presence of source commit fields.
        if not gen_commit:
            findings.append(f"{split}: gate.source_commit missing")
        elif not is_resolved_repository_commit(gen_commit):
            findings.append(
                f"{split}: gate.source_commit is not a resolved commit "
                f"(value={gen_commit!r})"
            )
        else:
            entry["generation_commit"] = gen_commit
            commits_seen.add(gen_commit)
        if not audit_commit:
            findings.append(f"{split}: gate.audit_source_commit missing")
        elif not is_resolved_repository_commit(audit_commit):
            findings.append(
                f"{split}: gate.audit_source_commit is not a resolved commit "
                f"(value={audit_commit!r})"
            )
        else:
            entry["audit_commit"] = audit_commit
            commits_seen.add(audit_commit)
        # Campaign identity commit.
        identity = load_campaign_identity(_CORPUS_BASE / split)
        if identity is None:
            findings.append(f"{split}: campaign identity missing")
        elif not identity.created_from_commit:
            findings.append(f"{split}: campaign_identity.created_from_commit missing")
        elif not is_resolved_repository_commit(identity.created_from_commit):
            findings.append(
                f"{split}: campaign_identity.created_from_commit is not a "
                f"resolved commit "
                f"(value={identity.created_from_commit!r})"
            )
        else:
            entry["campaign_identity_commit"] = identity.created_from_commit
            commits_seen.add(identity.created_from_commit)
        if entry:
            per_split[split] = entry

    if len(commits_seen) > 1:
        findings.append(
            f"source commit inconsistency: found {len(commits_seen)} distinct "
            f"commits across splits: {sorted(commits_seen)}"
        )
    return findings


# ---------------------------------------------------------------------------
# Patch G/I: cross-split endpoint consistency audit
# ---------------------------------------------------------------------------


def validate_endpoint_consistency() -> list[str]:
    """Require one endpoint identity across all generated splits.

    Loads ``campaign_identity.json`` from each split and collects:
    - ``serving_endpoint_host``
    - ``serving_endpoint_sha256``
    - ``api_protocol``

    All splits must have identical values for these fields.
    """
    findings: list[str] = []
    hosts_seen: set[str] = set()
    sha_seen: set[str] = set()
    proto_seen: set[str] = set()
    per_split: dict[str, dict[str, str]] = {}

    for split in ("development", "validation", "test"):
        identity = load_campaign_identity(_CORPUS_BASE / split)
        if identity is None:
            # Split not generated yet; skip.
            continue
        # Patch C (fail-closed): for empirical corpus splits, require all
        # endpoint provenance fields to be non-empty.  Do not silently skip
        # a split with missing endpoint data.
        manifest = _load_manifest(split)
        is_empirical = (
            manifest is not None
            and manifest.get("artifact_class")
            == ARTIFACT_CLASS_EMPIRICAL_CORPUS
        )
        endpoint_values = {
            "serving_endpoint_host": identity.serving_endpoint_host,
            "serving_endpoint_sha256": identity.serving_endpoint_sha256,
            "api_protocol": identity.api_protocol,
        }
        missing = [k for k, v in endpoint_values.items() if not v]
        if is_empirical and missing:
            findings.append(
                f"{split}: missing endpoint provenance: "
                + ", ".join(missing)
            )
            continue
        entry: dict[str, str] = {}
        host = identity.serving_endpoint_host
        sha = identity.serving_endpoint_sha256
        proto = identity.api_protocol
        if host:
            entry["host"] = host
            hosts_seen.add(host)
        if sha:
            entry["sha"] = sha
            sha_seen.add(sha)
        if proto:
            entry["proto"] = proto
            proto_seen.add(proto)
        if entry:
            per_split[split] = entry

    # Require consistency across all splits that have endpoint data.
    if len(hosts_seen) > 1:
        findings.append(
            f"endpoint provenance inconsistency: "
            f"found {len(hosts_seen)} distinct serving_endpoint_host values "
            f"across splits: {sorted(hosts_seen)}"
        )
    if len(sha_seen) > 1:
        findings.append(
            "endpoint provenance inconsistency: "
            "found multiple serving_endpoint_sha256 values across splits"
        )
    if len(proto_seen) > 1:
        findings.append(
            "endpoint provenance inconsistency: "
            "found multiple api_protocol values across splits"
        )
    return findings


# ---------------------------------------------------------------------------
# Patch L: final combined audit — verify all split gates
# ---------------------------------------------------------------------------


def _validate_all_split_gates() -> list[str]:
    """Check that every split gate has generation_completed and audit_passed.

    Used only by the final combined audit (``--split all``).  Returns a
    list of blocking findings (empty when all gates are valid).
    """
    findings: list[str] = []
    for split in ("development", "validation", "test"):
        gate_file = generation_gate_path(split, base=_CORPUS_BASE)
        if not gate_file.exists():
            findings.append(f"{split}: generation gate file missing")
            continue
        gate = json.loads(gate_file.read_text(encoding="utf-8"))
        if gate.get("generation_completed") is not True:
            findings.append(
                f"{split}: gate generation_completed != true"
            )
        if gate.get("audit_passed") is not True:
            findings.append(
                f"{split}: gate audit_passed != true"
            )
        # Verify audit report hash is still valid.
        audit_path_str = gate.get("audit_report_path")
        recorded_hash = gate.get("audit_report_sha256")
        if not audit_path_str or not recorded_hash:
            findings.append(
                f"{split}: gate missing audit report hash"
            )
            continue
        audit_path = _CORPUS_BASE / audit_path_str
        if not audit_path.exists():
            findings.append(
                f"{split}: audit report file not found at {audit_path}"
            )
            continue
        computed = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        if computed != recorded_hash:
            findings.append(
                f"{split}: audit report SHA256 mismatch "
                f"(recorded={recorded_hash!r}, computed={computed!r})"
            )
        # Patch L (Phase 3 Final): verify corpus_manifest_sha256 binding.
        recorded_manifest_sha = gate.get("corpus_manifest_sha256")
        if not recorded_manifest_sha:
            findings.append(f"{split}: gate missing corpus_manifest_sha256")
        else:
            manifest_path = _CORPUS_BASE / split / "corpus_manifest.json"
            if not manifest_path.exists():
                findings.append(f"{split}: corpus_manifest.json not found")
            else:
                computed_manifest = hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
                if computed_manifest != recorded_manifest_sha:
                    findings.append(
                        f"{split}: corpus_manifest SHA256 mismatch "
                        f"(recorded={recorded_manifest_sha!r}, computed={computed_manifest!r})"
                    )
        # Patch L (Phase 3 Final): verify campaign_identity_sha256 binding.
        recorded_identity_sha = gate.get("campaign_identity_sha256")
        if not recorded_identity_sha:
            findings.append(f"{split}: gate missing campaign_identity_sha256")
        else:
            identity = load_campaign_identity(_CORPUS_BASE / split)
            if identity is None:
                findings.append(f"{split}: campaign_identity.json not found or malformed")
            else:
                computed_identity = campaign_identity_sha256(identity)
                if computed_identity != recorded_identity_sha:
                    findings.append(
                        f"{split}: campaign_identity SHA256 mismatch "
                        f"(recorded={recorded_identity_sha!r}, computed={computed_identity!r})"
                    )
    return findings


def build_validation_report(
    *,
    split_scope: str = "all",
) -> dict:
    """Build the full validation report with all audit sections.

    Patch C/D: when *split_scope* is a single split name, only that
    split's artifacts are loaded and audited.  ``"all"`` audits every
    split (the original behaviour).
    """
    target_splits = audited_splits(split_scope)

    # Load only the audited splits' data.
    all_attempts: list[EmpiricalGenerationAttempt] = []
    all_candidates: list[EmpiricalCandidate] = []
    for split in target_splits:
        all_attempts.extend(_load_attempts(split))
        all_candidates.extend(_load_candidates(split))

    # Filter frozen plan items to the audited splits.
    all_plan_items_raw = _load_plan_items()
    scope_plan_items = [
        it for it in all_plan_items_raw
        if it.get("split") in target_splits
    ] if split_scope != "all" else all_plan_items_raw

    # Run all audit sections.
    # Patch G: audit order for sequence checks: lineage → atomicity.
    sections: dict[str, list[str]] = {}

    sections["phase_provenance"] = validate_phase_and_provenance()
    sections["plan_completeness"] = validate_plan_completeness(
        all_attempts, all_candidates, target_splits=target_splits,
    )
    sections["split_integrity"] = validate_split_integrity(all_attempts, all_candidates)
    sections["identity_uniqueness"] = validate_identity_uniqueness(all_attempts, all_candidates)
    sections["variant_consistency"] = validate_variant_consistency(all_attempts, all_candidates)
    sections["config_consistency"] = validate_config_consistency(all_attempts)
    sections["hash_integrity"] = validate_hash_integrity(
        all_attempts, all_candidates, target_splits=target_splits,
    )
    sections["retry_lineage"] = validate_retry_lineage(all_attempts)
    sections["sequence_atomicity"] = validate_sequence_atomicity(all_candidates, all_attempts)
    sections["acceptance_independence"] = validate_acceptance_independence(all_candidates)
    # Patch H: campaign identity verification.
    sections["campaign_identity"] = validate_campaign_identity(
        target_splits=target_splits,
    )
    # Patch B (Phase 3 Final): artifact classification audit.
    sections["artifact_classification"] = validate_artifact_classification(
        target_splits=target_splits,
    )
    # Patch H (Phase 3 Final): manifest / campaign identity provenance.
    sections["manifest_provenance"] = validate_manifest_provenance(
        target_splits=target_splits,
    )
    # Patch J (Phase 3 Final): required fields for empirical_corpus manifests.
    sections["empirical_corpus_required_fields"] = validate_empirical_corpus_required_fields(
        target_splits=target_splits,
    )
    # Patch L: final combined audit must verify all split gates.
    if split_scope == "all":
        sections["split_gate_progression"] = _validate_all_split_gates()
        # Patch F (Phase 3 Final): source commit consistency.
        sections["source_commit_consistency"] = validate_source_commit_consistency()
        # Patch G/I: endpoint consistency across splits.
        sections["endpoint_consistency"] = validate_endpoint_consistency()

    # Collect all blocking findings.
    all_findings: list[str] = []
    for section_findings in sections.values():
        all_findings.extend(section_findings)

    coverage = compute_coverage_stats(
        all_attempts, all_candidates, scope_plan_items,
    )

    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "empirical_phase": EMPIRICAL_PHASE,
        "audit_scope": split_scope,
        "audited_splits": list(target_splits),
        "audit_sections": sections,
        "validation_findings": all_findings,
        "finding_count": len(all_findings),
        "blocking_finding_count": len(all_findings),
        "coverage_stats": coverage,
        "total_raw_attempts": len(all_attempts),
        "total_accepted_candidates": len(all_candidates),
        # Patch H: explicit evidence of which checks were run.
        "checks_run": [
            "phase_provenance",
            "plan_completeness",
            "split_integrity",
            "identity_uniqueness",
            "variant_consistency",
            "config_consistency",
            "hash_integrity",
            "retry_lineage",
            "sequence_atomicity",
            "acceptance_independence",
            "campaign_identity",
            "artifact_classification",
            "manifest_provenance",
            "empirical_corpus_required_fields",
            *([
                "split_gate_progression",
                "source_commit_consistency",
                "endpoint_consistency",
            ] if split_scope == "all" else []),
        ],
        "passed": len(all_findings) == 0,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(report: dict, *, md_path: Path | None = None) -> None:
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

    (md_path or _REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _source_commit() -> str:
    """Return the current repository HEAD commit hash."""
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
        cwd=_PROJECT_ROOT,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


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

    split_scope: str = args.split
    print(f"Running full corpus validation (Patch F) — scope={split_scope}...")

    report = build_validation_report(split_scope=split_scope)

    # Determine output paths: split-specific or global.
    if split_scope != "all":
        split_dir = _CORPUS_BASE / split_scope
        split_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = split_dir / "audit_report.json"
        report_md_path = split_dir / "audit_report.md"
    else:
        report_json_path = _REPORT_JSON
        report_md_path = _REPORT_MD

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(report, md_path=report_md_path)

    print(f"Validation report written to: {report_json_path}")
    print(f"Markdown report written to: {report_md_path}")
    print(f"Passed: {report['passed']}")
    print(f"Blocking findings: {report['blocking_finding_count']}")

    # Patch E: audit-to-gate promotion for split-specific audits.
    if split_scope != "all":
        audit_report_sha = hashlib.sha256(
            report_json_path.read_bytes(),
        ).hexdigest()
        source_commit = _source_commit()

        # Patch F+G (Phase 3 Final): compute exact artifact hashes for gate binding.
        split_dir = _CORPUS_BASE / split_scope
        corpus_manifest_path = split_dir / "corpus_manifest.json"
        if not corpus_manifest_path.exists():
            print(
                f"ERROR: {split_scope}: corpus_manifest.json not found — "
                f"cannot bind audit to exact artifacts",
                file=sys.stderr,
            )
            return 1
        corpus_manifest_sha = hashlib.sha256(
            corpus_manifest_path.read_bytes(),
        ).hexdigest()

        identity = load_campaign_identity(split_dir)
        if identity is None:
            print(
                f"ERROR: {split_scope}: campaign_identity.json not found or "
                f"malformed — cannot bind audit to exact artifacts",
                file=sys.stderr,
            )
            return 1
        campaign_identity_sha = campaign_identity_sha256(identity)

        # The generation gate must already exist with generation_completed=true.
        gate = load_generation_gate(split_scope)
        if gate is None or gate.get("generation_completed") is not True:
            print(
                f"ERROR: {split_scope}: generation gate missing or "
                f"generation_completed != true — cannot promote audit result",
                file=sys.stderr,
            )
            return 1

        # Patch M (Phase 3 Final): require source provenance fields to exist
        # and agree before promoting audit_passed=true.
        gate_source_commit = gate.get("source_commit", "")
        identity_commit = identity.created_from_commit or ""
        if report["passed"]:
            commit_problems: list[str] = []
            # Require presence of all source provenance fields.
            if not gate_source_commit:
                commit_problems.append(
                    "gate.source_commit is missing or empty"
                )
            elif not is_resolved_repository_commit(gate_source_commit):
                commit_problems.append(
                    f"gate.source_commit is not a resolved commit "
                    f"(value={gate_source_commit!r})"
                )
            if not identity_commit:
                commit_problems.append(
                    "campaign_identity.created_from_commit is missing or empty"
                )
            elif not is_resolved_repository_commit(identity_commit):
                commit_problems.append(
                    f"campaign_identity.created_from_commit is not a resolved "
                    f"commit (value={identity_commit!r})"
                )
            if not source_commit:
                commit_problems.append(
                    "current HEAD commit could not be resolved"
                )
            elif not is_resolved_repository_commit(source_commit):
                commit_problems.append(
                    f"current HEAD commit is not a resolved commit "
                    f"(value={source_commit!r})"
                )
            # Require equality when all present.
            if gate_source_commit and identity_commit and source_commit:
                if gate_source_commit != source_commit:
                    commit_problems.append(
                        f"gate.source_commit={gate_source_commit!r} != HEAD={source_commit!r}"
                    )
                if identity_commit != source_commit:
                    commit_problems.append(
                        f"campaign_identity.created_from_commit={identity_commit!r} != HEAD={source_commit!r}"
                    )
            if commit_problems:
                print(
                    f"ERROR: {split_scope}: audit promotion BLOCKED — "
                    f"source provenance issues:",
                    file=sys.stderr,
                )
                for m in commit_problems:
                    print(f"  {m}", file=sys.stderr)
                # Write the gate with audit_passed=false (diagnostic only).
                update_generation_gate_after_audit(
                    split=split_scope,
                    audit_passed=False,
                    audit_report_path=report_json_path.relative_to(_CORPUS_BASE),
                    audit_report_sha256=audit_report_sha,
                    source_commit=source_commit,
                    corpus_manifest_sha256=corpus_manifest_sha,
                    campaign_identity_sha256=campaign_identity_sha,
                )
                print(
                    f"Gate updated: {split_scope} audit FAILED (source provenance)",
                )
                return 1

        update_generation_gate_after_audit(
            split=split_scope,
            audit_passed=report["passed"],
            audit_report_path=report_json_path.relative_to(_CORPUS_BASE),
            audit_report_sha256=audit_report_sha,
            source_commit=source_commit,
            corpus_manifest_sha256=corpus_manifest_sha,
            campaign_identity_sha256=campaign_identity_sha,
        )
        status = "PASSED" if report["passed"] else "FAILED"
        print(f"Gate updated: {split_scope} audit {status}")
        print(f"  audit_report_sha256: {audit_report_sha}")
        print(f"  corpus_manifest_sha256: {corpus_manifest_sha}")
        print(f"  campaign_identity_sha256: {campaign_identity_sha}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

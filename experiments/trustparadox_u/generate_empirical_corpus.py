"""Development-only empirical corpus generation runner (E1-018/020/021).

Checklist coverage:

- E1-018: empirical corpus manifest (versions, provenance, scientific
  hashes, status/split/trust/attack/scenario counts).
- E1-019: canonical corpus hashing via the scientific hash helpers in
  ``empirical_corpus`` (records sorted deterministically before hashing).
- E1-020: CLI ``--split --mode --scenario --trust --attack --samples
  --output-dir`` with optional ``--generator-model``/``--temperature``.
  ``--split validation`` and ``--split test`` are hard-rejected.
- E1-021: phase lock — while ``EMPIRICAL_PHASE == "E1"`` only the
  development split may be generated (``EmpiricalPhaseLockedError``).
- E1-023: smoke artifacts (``artifact_class = development_smoke``,
  ``research_use = diagnostic_only``) are written to the requested
  output directory only; never into frozen replay/release paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import Enum
from pathlib import Path

from experiments.trustparadox_u.artifact_provenance import (
    environment_lock_hash,
    working_tree_is_fully_clean,
)
from experiments.trustparadox_u.empirical_corpus import (
    CROSS_VARIANT_CONTAMINATION,
    EMPIRICAL_PHASE,
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    EMPIRICAL_TARGET_REGISTRY,
    SEQUENCE_ATTACK_TYPES,
    AttackType,
    EmpiricalCandidate,
    EmpiricalGenerationAttempt,
    EmpiricalSplit,
    EmpiricalTargetSpec,
    GenerationMode,
    GenerationStatus,
    TrustLevel,
    accept_generation_attempt,
    accepted_candidates_scientific_hash,
    assert_generation_split_unlocked,
    candidate_to_record,
    compute_target_registry_hash,
    raw_attempts_scientific_hash,
    record_to_attempt,
    record_to_candidate,
    validate_sequence_structure,
    validate_target_registry,
)
from experiments.trustparadox_u.empirical_generation_plan import (
    FrozenGenerationConfig,
    load_frozen_generation_config,
)
from experiments.trustparadox_u.empirical_generation import (
    EmpiricalCandidateGenerator,
    MockEmpiricalGenerator,
    RawAttemptWriter,
    RealEmpiricalGenerator,
    attempt_from_response,
    build_generation_request,
    build_prompt_manifest,
    prompt_manifest_sha256,
    prompt_sha256,
    resolve_prompt_bundle,
    utc_now_iso,
    validate_trust_prompt_invariance,
)
from experiments.trustparadox_u.campaign_identity import (
    CAMPAIGN_IDENTITY_FILENAME,
    CampaignIdentityMismatchError,
    campaign_identity_sha256,
    compute_campaign_identity,
    load_campaign_identity,
    verify_campaign_identity,
    write_campaign_identity,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TARGET_SPECS_PATH = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "target_specs.jsonl"
)

# E1-022: the development smoke matrix uses exactly these attack families.
DEFAULT_SMOKE_ATTACKS: tuple[str, ...] = (
    AttackType.DIRECT_DISCLOSURE.value,
    AttackType.SEMANTIC_PARAPHRASE.value,
    AttackType.HARD_NEGATIVE_CONTROL.value,
)

RAW_ATTEMPTS_FILENAME = "raw_generation_attempts.jsonl"
ACCEPTED_CANDIDATES_FILENAME = "accepted_candidates.jsonl"
CORPUS_MANIFEST_FILENAME = "corpus_manifest.json"
PROMPT_MANIFEST_FILENAME = "prompt_manifest.json"
VALIDATION_REPORT_FILENAME = "validation_report.json"


# ---------------------------------------------------------------------------
# Patch D: atomic file writing helpers
# ---------------------------------------------------------------------------


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write JSONL atomically: write to .tmp then rename (Patch D)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    tmp_path.rename(path)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON atomically: write to .tmp then rename (Patch D)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    tmp_path.rename(path)


# ---------------------------------------------------------------------------
# Patch C: campaign-layer retry with frozen policy
# ---------------------------------------------------------------------------


def generate_with_retry(
    *,
    generator: EmpiricalCandidateGenerator,
    request: "EmpiricalGenerationRequest",
    retry_policy: Mapping[str, object] | None,
    raw_writer: RawAttemptWriter,
    spec: EmpiricalTargetSpec,
    trust_level: str,
    attack_type: str,
    sample_index: int,
    generation_mode: str,
    transport: str | None,
    generator_model_requested: str | None,
    max_tokens: int | None,
    trust_prompt_hash: str | None,
    attack_prompt_hash: str | None,
    temperature: float,
    start_retry_index: int = 0,
) -> list[EmpiricalGenerationAttempt]:
    """Call the provider with retry, writing every raw attempt (Patch C).

    If the generator exposes ``generate_once`` (single-call, never raises),
    each provider attempt is written to *raw_writer* immediately.  Otherwise
    falls back to the legacy ``generate()`` path (mock / single-shot).
    """
    import time as _time

    if not hasattr(generator, "generate_once"):
        response = generator.generate(request)
        attempt = attempt_from_response(
            request,
            response,
            generator_provider=getattr(generator, "provider", GenerationMode.MOCK.value),
            generation_mode=generation_mode,
            transport=transport,
            generator_model_requested=generator_model_requested,
            max_tokens=max_tokens,
            trust_prompt_hash=trust_prompt_hash,
            attack_prompt_hash=attack_prompt_hash,
        )
        raw_writer.write_attempt(attempt)
        return [attempt]

    if retry_policy is None:
        max_attempts = 1
        backoff: tuple[float, ...] = ()
        retryable: tuple[str, ...] = ()
    else:
        max_attempts = 1 + int(retry_policy.get("max_retries", 0))
        backoff = tuple(float(s) for s in retry_policy.get("backoff_seconds", []))
        retryable = tuple(retry_policy.get("retryable_statuses", ["provider_error"]))

    attempts: list[EmpiricalGenerationAttempt] = []
    for retry_index in range(start_retry_index, max_attempts):
        base_response = generator.generate_once(request)  # type: ignore[attr-defined]
        response = replace(base_response, retry_index=retry_index)

        attempt = attempt_from_response(
            request,
            response,
            generator_provider=getattr(generator, "provider", GenerationMode.REAL.value),
            generation_mode=generation_mode,
            transport=transport,
            generator_model_requested=generator_model_requested,
            max_tokens=max_tokens,
            trust_prompt_hash=trust_prompt_hash,
            attack_prompt_hash=attack_prompt_hash,
        )
        raw_writer.write_attempt(attempt)
        attempts.append(attempt)

        if response.status == GenerationStatus.SUCCESS.value:
            break
        if retry_index >= max_attempts - 1:
            break
        if response.status not in retryable:
            break
        if retry_index < len(backoff):
            _time.sleep(backoff[retry_index])

    return attempts


# ---------------------------------------------------------------------------
# Patch E: atomic sequence acceptance
# ---------------------------------------------------------------------------


def accept_sequence_attempts(
    attempts: Sequence[EmpiricalGenerationAttempt],
    spec: EmpiricalTargetSpec,
) -> tuple[bool, list[EmpiricalCandidate], list[str]]:
    """All-or-nothing sequence acceptance (Patch E).

    Returns ``(accepted, candidates, rejection_reasons)``.
    If every step passes individual acceptance, all candidates are returned.
    Otherwise NO candidates are returned.
    """
    expected = len(attempts)
    complete, problems = _is_sequence_complete(list(attempts), expected)
    if not complete:
        reason = f"sequence_incomplete: {'; '.join(problems)}"
        return False, [], [reason] * expected

    candidates: list[EmpiricalCandidate] = []
    step_reasons: list[str] = []
    for attempt in attempts:
        result = accept_generation_attempt(attempt, spec)
        if result.accepted and result.candidate is not None:
            candidates.append(result.candidate)
        else:
            step_reasons.append(result.reason)

    if len(candidates) == expected:
        return True, candidates, []
    return False, [], step_reasons


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def repository_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def target_spec_sha256() -> str:
    """SHA-256 of the frozen target-spec file (or the registry hash)."""
    if _TARGET_SPECS_PATH.exists():
        return hashlib.sha256(_TARGET_SPECS_PATH.read_bytes()).hexdigest()
    return compute_target_registry_hash(EMPIRICAL_TARGET_REGISTRY)


def development_spec_for_scenario(scenario_id: str) -> EmpiricalTargetSpec:
    matches = [
        spec
        for spec in EMPIRICAL_TARGET_REGISTRY
        if spec.scenario_id == scenario_id and spec.split == EmpiricalSplit.DEVELOPMENT.value
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one development spec for {scenario_id!r}")
    return matches[0]


def specs_for_split(
    split: str,
    scenario_ids: Sequence[str] | None = None,
) -> tuple[EmpiricalTargetSpec, ...]:
    """E3-002: return every target spec assigned to *split*.

    Uses the frozen ``split`` field in the target-spec registry — never
    infers the split from the variant suffix.  When *scenario_ids* is
    given the result is filtered to those scenarios only.

    Expected counts: development=3, validation=3, test=6.
    """
    split_value = EmpiricalSplit(split).value
    matches = tuple(
        spec
        for spec in EMPIRICAL_TARGET_REGISTRY
        if spec.split == split_value and (scenario_ids is None or spec.scenario_id in scenario_ids)
    )
    if not matches:
        raise ValueError(f"no target specs found for split {split!r}")
    return matches


def sequence_step_count_for(attack_type: str, spec: EmpiricalTargetSpec) -> int:
    attack = AttackType(attack_type)
    if attack is AttackType.FRAGMENTATION_SEQUENCE:
        return len(spec.fragments)
    if attack is AttackType.COMPOSITIONAL_SEQUENCE:
        return len(spec.fact_chain)
    raise ValueError(f"{attack_type!r} is not a sequence attack")


# ---------------------------------------------------------------------------
# E1-018: corpus manifest
# ---------------------------------------------------------------------------


def _counts_by(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_corpus_manifest(
    *,
    generation_mode: str,
    attempts: Sequence[EmpiricalGenerationAttempt],
    accepted: Sequence[EmpiricalCandidate],
    prompt_manifest: Mapping[str, object],
    artifact_class: str = "development_smoke",
    research_use: str = "diagnostic_only",
    campaign_identity_hash: str = "",
    full_generation_plan_sha256: str | None = None,
    split_generation_plan_sha256: str | None = None,
    split_plan_item_count: int | None = None,
) -> dict[str, object]:
    status_counts = Counter(attempt.generation_status for attempt in attempts)
    result: dict[str, object] = {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "empirical_phase": EMPIRICAL_PHASE,
        "generation_mode": generation_mode,
        "artifact_class": artifact_class,
        "research_use": research_use,
        "repository_commit": repository_commit(),
        "repository_clean": working_tree_is_fully_clean(),
        "environment_lock_hash": environment_lock_hash(),
        "target_spec_sha256": target_spec_sha256(),
        "prompt_manifest_sha256": prompt_manifest_sha256(prompt_manifest),
        "raw_generation_sha256": raw_attempts_scientific_hash(attempts),
        "accepted_candidate_sha256": accepted_candidates_scientific_hash(accepted),
        "attempt_count": len(attempts),
        "success_count": status_counts.get(GenerationStatus.SUCCESS.value, 0),
        "refusal_count": status_counts.get(GenerationStatus.REFUSAL.value, 0),
        "malformed_count": status_counts.get(GenerationStatus.MALFORMED.value, 0),
        "provider_error_count": status_counts.get(GenerationStatus.PROVIDER_ERROR.value, 0),
        "accepted_candidate_count": len(accepted),
        "split_counts": _counts_by([a.split for a in attempts]),
        "trust_counts": _counts_by([a.trust_level for a in attempts]),
        "attack_counts": _counts_by([a.attack_type for a in attempts]),
        "scenario_counts": _counts_by([a.scenario_id for a in attempts]),
        "generated_at": utc_now_iso(),
    }
    # Patch I: bind campaign identity hash when available.
    if campaign_identity_hash:
        result["campaign_identity_sha256"] = campaign_identity_hash
    # Patch K: bind split-plan identity fields when available.
    if full_generation_plan_sha256 is not None:
        result["full_generation_plan_sha256"] = full_generation_plan_sha256
    if split_generation_plan_sha256 is not None:
        result["split_generation_plan_sha256"] = split_generation_plan_sha256
    if split_plan_item_count is not None:
        result["split_plan_item_count"] = split_plan_item_count
    return result


# ---------------------------------------------------------------------------
# Validation report (smoke-level instance of the E1-035 structure)
# ---------------------------------------------------------------------------


def _load_attempts(path: Path) -> list[EmpiricalGenerationAttempt]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [record_to_attempt(json.loads(line)) for line in handle if line.strip()]


def _load_candidates(path: Path) -> list[EmpiricalCandidate]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [record_to_candidate(json.loads(line)) for line in handle if line.strip()]


def manifest_hashes_valid(output_dir: Path, manifest: Mapping[str, object]) -> bool:
    """Recompute scientific hashes from the written files and compare."""
    attempts = _load_attempts(output_dir / RAW_ATTEMPTS_FILENAME)
    candidates = _load_candidates(output_dir / ACCEPTED_CANDIDATES_FILENAME)
    return raw_attempts_scientific_hash(attempts) == manifest.get(
        "raw_generation_sha256"
    ) and accepted_candidates_scientific_hash(candidates) == manifest.get(
        "accepted_candidate_sha256"
    )


def sequence_validation_failures(attempts: Sequence[EmpiricalGenerationAttempt]) -> list[str]:
    """Group sequence attempts by stable scientific identity, reduce retries,
    then apply the E1-017 structural validator."""
    # Group by stable scientific identity (not just family_id).
    groups: dict[tuple, list[EmpiricalGenerationAttempt]] = {}
    for attempt in attempts:
        if attempt.is_sequence_attempt and attempt.sequence_family_id is not None:
            key = (
                attempt.scenario_id,
                attempt.secret_variant_id,
                attempt.trust_level,
                attempt.attack_type,
                attempt.sample_index,
                attempt.generation_replicate,
                attempt.sequence_family_id,
            )
            groups.setdefault(key, []).append(attempt)
    problems: list[str]
    problems = []
    for key, steps in sorted(groups.items()):
        family_id = key[-1]
        trust_level = key[2]
        # Collapse retries to terminal attempt per step.
        try:
            terminal_steps = terminal_attempts_by_sequence_step(steps)
        except ValueError as exc:
            problems.append(f"{family_id}/{trust_level}: retry-lineage error: {exc}")
            continue
        for problem in validate_sequence_structure(terminal_steps):
            problems.append(f"{family_id}/{trust_level}: {problem}")
    return problems


def build_validation_report(
    *,
    attempts: Sequence[EmpiricalGenerationAttempt],
    accepted: Sequence[EmpiricalCandidate],
    rejection_reasons: Sequence[str],
    duplicate_id_count: int,
    manifest: Mapping[str, object],
    output_dir: Path,
    phase_lock_valid: bool,
) -> dict[str, object]:
    rejection_counts = dict(sorted(Counter(rejection_reasons).items()))
    contamination_count = rejection_counts.get(CROSS_VARIANT_CONTAMINATION, 0)
    sequence_problems = sequence_validation_failures(attempts)
    target_registry_valid = not validate_target_registry(EMPIRICAL_TARGET_REGISTRY)
    prompt_invariance_valid = not validate_trust_prompt_invariance()
    manifest_hash_valid = manifest_hashes_valid(output_dir, manifest)
    e1_foundation_valid = all(
        (
            target_registry_valid,
            prompt_invariance_valid,
            manifest_hash_valid,
            phase_lock_valid,
            duplicate_id_count == 0,
            contamination_count == 0,
            not sequence_problems,
            len(attempts) > 0,
        )
    )
    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "empirical_phase": EMPIRICAL_PHASE,
        "artifact_class": manifest.get("artifact_class"),
        "research_use": manifest.get("research_use"),
        "attempt_count": len(attempts),
        "accepted_count": len(accepted),
        "rejected_count": len(rejection_reasons),
        "status_counts": dict(sorted(Counter(a.generation_status for a in attempts).items())),
        "rejection_counts": rejection_counts,
        "scenario_counts": _counts_by([a.scenario_id for a in attempts]),
        "trust_counts": _counts_by([a.trust_level for a in attempts]),
        "attack_counts": _counts_by([a.attack_type for a in attempts]),
        "duplicate_id_count": duplicate_id_count,
        "cross_variant_contamination_count": contamination_count,
        "sequence_validation_failures": sequence_problems,
        "target_registry_valid": target_registry_valid,
        "prompt_invariance_valid": prompt_invariance_valid,
        "manifest_hash_valid": manifest_hash_valid,
        "phase_lock_valid": phase_lock_valid,
        "e1_foundation_valid": e1_foundation_valid,
        "generated_at": utc_now_iso(),
    }


SEQUENCE_GENERATION_REPORT_FILENAME = "sequence_generation_report.json"


def _unit_key(attempt: EmpiricalGenerationAttempt) -> tuple:
    """Stable key identifying a generation unit (ignoring sequence steps)."""
    return (
        attempt.scenario_id,
        attempt.secret_variant_id,
        attempt.trust_level,
        attempt.attack_type,
        attempt.sample_index,
        attempt.generation_replicate,
    )


def _is_sequence_complete(
    attempts: list[EmpiricalGenerationAttempt],
    expected_step_count: int,
) -> tuple[bool, list[str]]:
    """E3-006: check whether all sequence steps are present and valid."""
    if len(attempts) != expected_step_count:
        return False, [f"expected {expected_step_count} steps, got {len(attempts)}"]
    indices = sorted(
        a.sequence_step_index if a.sequence_step_index is not None else 0 for a in attempts
    )
    if len(set(indices)) != len(indices):
        return False, ["duplicate step indices"]
    if indices != list(range(expected_step_count)):
        return False, [f"missing step indices: expected 0..{expected_step_count - 1}"]
    problems = validate_sequence_structure(attempts)
    if problems:
        return False, problems
    return True, []


# ---------------------------------------------------------------------------
# Patch D: rebuild accepted corpus from raw attempts
# ---------------------------------------------------------------------------


def terminal_attempt_for_retry_chain(
    attempts: Sequence[EmpiricalGenerationAttempt],
) -> EmpiricalGenerationAttempt:
    """Reduce a retry chain to its single terminal scientific attempt.

    Validates:
    - All attempts share the same generation_attempt_id (one scientific unit).
    - Retry indices are unique and consecutive from 0.
    - No retry occurs after a success.
    - Returns the attempt with the highest retry_index.
    """
    if not attempts:
        raise ValueError("terminal_attempt_for_retry_chain: empty attempt list")
    # All must share the same generation_attempt_id.
    ids = {a.generation_attempt_id for a in attempts}
    if len(ids) != 1:
        raise ValueError(
            f"terminal_attempt_for_retry_chain: mixed generation_attempt_ids: {ids}"
        )
    # Retry indices must be unique.
    retry_indices = sorted(a.retry_index for a in attempts)
    if len(set(retry_indices)) != len(retry_indices):
        raise ValueError(
            f"terminal_attempt_for_retry_chain: duplicate retry indices: {retry_indices}"
        )
    # Must start from 0 and be consecutive.
    if retry_indices[0] != 0:
        raise ValueError(
            f"terminal_attempt_for_retry_chain: retry indices do not start at 0: {retry_indices}"
        )
    if retry_indices != list(range(len(retry_indices))):
        raise ValueError(
            f"terminal_attempt_for_retry_chain: retry indices not consecutive: {retry_indices}"
        )
    # No retry after success.
    for a in sorted(attempts, key=lambda x: x.retry_index):
        if a.generation_status == GenerationStatus.SUCCESS.value:
            # This must be the last attempt.
            if a.retry_index != retry_indices[-1]:
                raise ValueError(
                    f"terminal_attempt_for_retry_chain: retry after success at "
                    f"retry_index={a.retry_index}"
                )
    return max(attempts, key=lambda a: a.retry_index)


def terminal_attempt_for_retry_segment(
    attempts: Sequence[EmpiricalGenerationAttempt],
    *,
    expected_start_retry_index: int,
) -> EmpiricalGenerationAttempt:
    """Reduce a *newly generated* retry segment to its terminal attempt.

    Unlike :func:`terminal_attempt_for_retry_chain`, this validator is used
    inside ``_generate_unit()`` for the segment produced during the *current*
    invocation.  A resumed segment may legitimately begin at ``retry1`` or
    ``retry2`` (whatever the resume state dictated), so the strict
    "must start at 0" rule from the complete-chain helper does not apply.

    Validates:
    - Segment is non-empty.
    - All attempts share the same ``generation_attempt_id``.
    - Retry indices are unique.
    - First retry index equals *expected_start_retry_index*.
    - Indices are consecutive within the segment.
    - No retry after success within the segment.
    - Returns the attempt with the highest retry_index.
    """
    if not attempts:
        raise ValueError("terminal_attempt_for_retry_segment: empty segment")
    # All must share the same generation_attempt_id.
    ids = {a.generation_attempt_id for a in attempts}
    if len(ids) != 1:
        raise ValueError(
            f"terminal_attempt_for_retry_segment: mixed generation_attempt_ids: {ids}"
        )
    # Retry indices must be unique.
    retry_indices = sorted(a.retry_index for a in attempts)
    if len(set(retry_indices)) != len(retry_indices):
        raise ValueError(
            f"terminal_attempt_for_retry_segment: duplicate retry indices: "
            f"{retry_indices}"
        )
    # Must start at the expected index.
    if retry_indices[0] != expected_start_retry_index:
        raise ValueError(
            f"terminal_attempt_for_retry_segment: expected start retry "
            f"{expected_start_retry_index}, got {retry_indices[0]}"
        )
    # Must be consecutive within the segment.
    expected = list(range(
        expected_start_retry_index,
        expected_start_retry_index + len(retry_indices),
    ))
    if retry_indices != expected:
        raise ValueError(
            f"terminal_attempt_for_retry_segment: retry indices not consecutive: "
            f"{retry_indices}"
        )
    # No retry after success within the segment.
    for a in sorted(attempts, key=lambda x: x.retry_index):
        if a.generation_status == GenerationStatus.SUCCESS.value:
            if a.retry_index != retry_indices[-1]:
                raise ValueError(
                    f"terminal_attempt_for_retry_segment: retry after success at "
                    f"retry_index={a.retry_index}"
                )
    return max(attempts, key=lambda a: a.retry_index)


def terminal_attempts_by_sequence_step(
    attempts: Sequence[EmpiricalGenerationAttempt],
    expected_step_count: int | None = None,
    *,
    require_complete: bool = False,
) -> list[EmpiricalGenerationAttempt]:
    """Group by sequence_step_index, reduce each chain to one terminal attempt.

    Returns terminal attempts sorted by sequence_step_index.

    If *expected_step_count* is given and *require_complete* is true,
    verifies that step indices == 0..expected_step_count-1.
    """
    step_groups: dict[int, list[EmpiricalGenerationAttempt]] = {}
    for attempt in attempts:
        step_idx = attempt.sequence_step_index
        if step_idx is None:
            raise ValueError(
                "terminal_attempts_by_sequence_step: non-sequence attempt"
            )
        step_groups.setdefault(step_idx, []).append(attempt)
    terminal_per_step: list[EmpiricalGenerationAttempt] = []
    for step_idx in sorted(step_groups):
        terminal_per_step.append(
            terminal_attempt_for_retry_chain(step_groups[step_idx])
        )
    if require_complete and expected_step_count is not None:
        actual_indices = sorted(step_groups.keys())
        expected_indices = list(range(expected_step_count))
        if actual_indices != expected_indices:
            raise ValueError(
                f"terminal_attempts_by_sequence_step: incomplete sequence: "
                f"expected steps {expected_indices}, got {actual_indices}"
            )
    return terminal_per_step


# ---------------------------------------------------------------------------
# Patch D: unit-completion state model for resume
# ---------------------------------------------------------------------------


class UnitResumeState(str, Enum):
    """Resume classification for a single scientific generation unit."""

    NOT_STARTED = "not_started"
    RETRY_PENDING = "retry_pending"
    COMPLETE_SUCCESS = "complete_success"
    COMPLETE_FAILURE = "complete_failure"
    SEQUENCE_PARTIAL = "sequence_partial"
    SEQUENCE_COMPLETE_SUCCESS = "sequence_complete_success"
    SEQUENCE_COMPLETE_FAILURE = "sequence_complete_failure"


def resolve_unit_resume_state(
    unit_key: tuple,
    existing_attempts: Sequence[EmpiricalGenerationAttempt],
    *,
    is_sequence: bool,
    expected_step_count: int,
    max_retries: int,
    retryable_statuses: Sequence[str],
) -> dict[str, object]:
    """Classify the resume state for one generation unit.

    Returns a dict with keys:
    - ``state``: a :class:`UnitResumeState` value
    - ``completed_step_indices``: set of step indices with terminal success
    - ``pending_step_indices``: set of step indices that need generation/retry
    - ``retry_index_to_continue_from``: int (for non-sequence retry continuation)
    - ``retry_budget_remaining``: int
    - ``terminal_status_by_step``: dict mapping step_index → status string
    """
    unit_attempts = [a for a in existing_attempts if _unit_key(a) == unit_key]

    if not is_sequence:
        return _resolve_non_sequence_state(
            unit_attempts,
            max_retries=max_retries,
            retryable_statuses=retryable_statuses,
        )
    return _resolve_sequence_state(
        unit_attempts,
        expected_step_count=expected_step_count,
        max_retries=max_retries,
        retryable_statuses=retryable_statuses,
    )


def _resolve_non_sequence_state(
    attempts: list[EmpiricalGenerationAttempt],
    *,
    max_retries: int,
    retryable_statuses: Sequence[str],
) -> dict[str, object]:
    """Resolve resume state for a non-sequence unit."""
    if not attempts:
        return {
            "state": UnitResumeState.NOT_STARTED,
            "completed_step_indices": set(),
            "pending_step_indices": set(),
            "retry_index_to_continue_from": 0,
            "retry_budget_remaining": max_retries + 1,
            "terminal_status_by_step": {},
        }

    terminal = terminal_attempt_for_retry_chain(attempts)
    max_retry = max(a.retry_index for a in attempts)
    budget_remaining = max(0, max_retries - max_retry)

    if terminal.generation_status == GenerationStatus.SUCCESS.value:
        return {
            "state": UnitResumeState.COMPLETE_SUCCESS,
            "completed_step_indices": set(),
            "pending_step_indices": set(),
            "retry_index_to_continue_from": max_retry + 1,
            "retry_budget_remaining": budget_remaining,
            "terminal_status_by_step": {None: terminal.generation_status},
        }

    if terminal.generation_status in retryable_statuses and budget_remaining > 0:
        return {
            "state": UnitResumeState.RETRY_PENDING,
            "completed_step_indices": set(),
            "pending_step_indices": set(),
            "retry_index_to_continue_from": max_retry + 1,
            "retry_budget_remaining": budget_remaining,
            "terminal_status_by_step": {None: terminal.generation_status},
        }

    return {
        "state": UnitResumeState.COMPLETE_FAILURE,
        "completed_step_indices": set(),
        "pending_step_indices": set(),
        "retry_index_to_continue_from": max_retry + 1,
        "retry_budget_remaining": budget_remaining,
        "terminal_status_by_step": {None: terminal.generation_status},
    }


def _resolve_sequence_state(
    attempts: list[EmpiricalGenerationAttempt],
    *,
    expected_step_count: int,
    max_retries: int,
    retryable_statuses: Sequence[str],
) -> dict[str, object]:
    """Resolve resume state for a sequence unit.

    Returns a dict with keys:
    - ``state``: a :class:`UnitResumeState` value
    - ``completed_step_indices``: set of step indices with terminal success
    - ``pending_step_indices``: set of step indices that need generation/retry
    - ``retry_start_by_step``: dict mapping each pending step to its retry start index
    - ``retry_index_to_continue_from``: int (kept for backward compat)
    - ``retry_budget_remaining``: int
    - ``terminal_status_by_step``: dict mapping step_index → status string
    """
    step_groups: dict[int, list[EmpiricalGenerationAttempt]] = {}
    for a in attempts:
        si = a.sequence_step_index
        if si is not None:
            step_groups.setdefault(si, []).append(a)

    terminal_by_step: dict[int, EmpiricalGenerationAttempt] = {}
    status_by_step: dict[int, str] = {}
    for si, group in step_groups.items():
        t = terminal_attempt_for_retry_chain(group)
        terminal_by_step[si] = t
        status_by_step[si] = t.generation_status

    completed = {
        si for si, t in terminal_by_step.items()
        if t.generation_status == GenerationStatus.SUCCESS.value
    }

    if len(completed) == expected_step_count:
        return {
            "state": UnitResumeState.SEQUENCE_COMPLETE_SUCCESS,
            "completed_step_indices": completed,
            "pending_step_indices": set(),
            "retry_start_by_step": {},
            "retry_index_to_continue_from": 0,
            "retry_budget_remaining": 0,
            "terminal_status_by_step": status_by_step,
        }

    pending: set[int] = set()
    retry_start_by_step: dict[int, int] = {}
    for si in range(expected_step_count):
        if si in completed:
            continue
        if si not in terminal_by_step:
            # Missing step: start at retry 0.
            pending.add(si)
            retry_start_by_step[si] = 0
            continue
        terminal = terminal_by_step[si]
        max_retry = max(a.retry_index for a in step_groups[si])
        budget = max(0, max_retries - max_retry)
        if terminal.generation_status in retryable_statuses and budget > 0:
            pending.add(si)
            retry_start_by_step[si] = max_retry + 1

    if pending:
        return {
            "state": UnitResumeState.SEQUENCE_PARTIAL,
            "completed_step_indices": completed,
            "pending_step_indices": pending,
            "retry_start_by_step": retry_start_by_step,
            "retry_index_to_continue_from": 0,
            "retry_budget_remaining": 0,
            "terminal_status_by_step": status_by_step,
        }

    return {
        "state": UnitResumeState.SEQUENCE_COMPLETE_FAILURE,
        "completed_step_indices": completed,
        "pending_step_indices": set(),
        "retry_start_by_step": {},
        "retry_index_to_continue_from": 0,
        "retry_budget_remaining": 0,
        "terminal_status_by_step": status_by_step,
    }


def _terminal_attempt(attempts: list[EmpiricalGenerationAttempt]) -> EmpiricalGenerationAttempt:
    """Return the attempt with the highest retry_index (last provider call).

    .. deprecated:: Use :func:`terminal_attempt_for_retry_chain` instead.
    """
    return max(attempts, key=lambda a: a.retry_index)


def rebuild_accepted_candidates(
    attempts: Sequence[EmpiricalGenerationAttempt],
    target_registry: Sequence[EmpiricalTargetSpec],
) -> tuple[list[EmpiricalCandidate], list[str]]:
    """Rebuild the accepted corpus from the full raw attempt log (Patch D).

    Treats ``raw_generation_attempts.jsonl`` as the authoritative event log
    and ``accepted_candidates.jsonl`` as a deterministic materialized view.

    Procedure:

    1. Group provider retries into scientific generation units.
    2. For non-sequence units, select the terminal successful attempt.
    3. For sequence units, group steps, select terminal per step, then
       apply sequence atomicity (all-or-nothing).
    4. Apply frozen acceptance rules.
    5. Return the full accepted corpus.
    """
    specs_by_key: dict[tuple[str, str], EmpiricalTargetSpec] = {
        (s.scenario_id, s.secret_variant_id): s for s in target_registry
    }

    # Group attempts by (unit_key, sequence_step_index | None).
    # For non-sequence: key = (unit_key, None)
    # For sequence: key = (unit_key, step_index)
    unit_step_groups: dict[tuple, list[EmpiricalGenerationAttempt]] = {}
    for attempt in attempts:
        uk = _unit_key(attempt)
        step = attempt.sequence_step_index if attempt.is_sequence_attempt else None
        group_key = (uk, step)
        unit_step_groups.setdefault(group_key, []).append(attempt)

    # Collect unique unit keys and classify as sequence or not.
    unit_keys_seen: dict[tuple, bool] = {}
    for (uk, step) in unit_step_groups:
        is_seq = step is not None
        if uk in unit_keys_seen:
            # Consistency: once marked sequence, always sequence.
            if is_seq:
                unit_keys_seen[uk] = True
        else:
            unit_keys_seen[uk] = is_seq

    accepted: list[EmpiricalCandidate] = []
    rejections: list[str] = []

    for uk, is_sequence in sorted(unit_keys_seen.items()):
        scenario_id, secret_variant_id, trust_level, attack_type, sample_index, gen_rep = uk
        spec = specs_by_key.get((scenario_id, secret_variant_id))
        if spec is None:
            rejections.append(f"unknown_target_spec:{scenario_id}/{secret_variant_id}")
            continue

        if not is_sequence:
            # Non-sequence: pick terminal attempt, run acceptance.
            group_key = (uk, None)
            chain = unit_step_groups.get(group_key, [])
            terminal = terminal_attempt_for_retry_chain(chain)
            result = accept_generation_attempt(terminal, spec)
            if result.accepted and result.candidate is not None:
                accepted.append(result.candidate)
            else:
                rejections.append(result.reason)
        else:
            # Sequence: collect terminal attempt per step, then atomicity.
            step_indices = sorted(
                step for (_, step) in unit_step_groups if _ == uk and step is not None
            )
            terminal_per_step: list[EmpiricalGenerationAttempt] = []
            for si in step_indices:
                chain = unit_step_groups[(uk, si)]
                terminal_per_step.append(terminal_attempt_for_retry_chain(chain))

            expected = len(step_indices)
            complete, problems = _is_sequence_complete(terminal_per_step, expected)
            if not complete:
                reason = f"sequence_incomplete: {'; '.join(problems)}"
                rejections.append(reason)
                continue

            # Apply acceptance to each terminal step.
            step_candidates: list[EmpiricalCandidate] = []
            step_failed = False
            for step_attempt in terminal_per_step:
                result = accept_generation_attempt(step_attempt, spec)
                if result.accepted and result.candidate is not None:
                    step_candidates.append(result.candidate)
                else:
                    rejections.append(result.reason)
                    step_failed = True

            if step_failed:
                # Atomic: discard all candidates for this sequence.
                continue
            accepted.extend(step_candidates)

    return accepted, rejections


def _build_sequence_report(
    *,
    planned: int,
    complete: int,
    accepted: int,
    rejected: int,
    rejection_reasons: list[str],
) -> dict[str, object]:
    """E3-006: build the sequence generation report.

    .. deprecated:: Use :func:`build_sequence_generation_report_from_attempts`
       for the authoritative report derived from complete raw history.
    """
    return {
        "planned_sequence_count": planned,
        "complete_sequence_count": complete,
        "incomplete_sequence_count": planned - complete,
        "accepted_sequence_count": accepted,
        "rejected_sequence_count": rejected,
        "rejection_reasons": dict(sorted(Counter(rejection_reasons).items())),
    }


def expected_sequence_steps_from_plan(
    plan_items: Sequence["GenerationPlanItem"],
) -> dict[tuple, int]:
    """Patch H: build a mapping from sequence scientific key to expected step count.

    For every planned sequence, all plan items must agree on
    ``sequence_step_count``.  If they disagree, raise a fatal error.

    Returns a dict keyed by
    ``(scenario_id, secret_variant_id, trust_level, attack_type,
      sample_index, generation_replicate, sequence_id)``
    mapping to the expected step count.
    """
    index: dict[tuple, int] = {}
    for item in plan_items:
        if item.sequence_id is None or item.sequence_step_count is None:
            continue
        seq_key = (
            item.scenario_id, item.secret_variant_id, item.trust_level,
            item.attack_type, item.sample_index, item.generation_replicate,
            item.sequence_id,
        )
        if seq_key in index:
            if index[seq_key] != item.sequence_step_count:
                raise ValueError(
                    f"inconsistent sequence_step_count for {seq_key}: "
                    f"{index[seq_key]} vs {item.sequence_step_count}"
                )
        else:
            index[seq_key] = item.sequence_step_count
    return index


def build_sequence_generation_report_from_attempts(
    *,
    attempts: Sequence[EmpiricalGenerationAttempt],
    plan_items: Sequence["GenerationPlanItem"] | None,
    target_registry: Sequence[EmpiricalTargetSpec],
) -> dict[str, object]:
    """Patch D: build the sequence report from complete raw history.

    Derives all sequence statistics from the full raw attempt log after
    retry-chain reduction, not from incremental runtime counters.
    """
    # Build plan-based expected step count index (Patch H).
    # Patch I: when a frozen plan is supplied, inconsistent sequence
    # definitions are fatal — no silent fallback.
    plan_step_index: dict[tuple, int] = {}
    if plan_items is not None:
        plan_step_index = expected_sequence_steps_from_plan(plan_items)

    # Determine planned sequence count from the plan if available.
    planned_sequence_count = 0
    sequence_plan_keys: set[tuple] = set()
    if plan_items is not None:
        for item in plan_items:
            if item.sequence_id is not None and item.sequence_step_index is not None:
                seq_key = (
                    item.scenario_id, item.secret_variant_id, item.trust_level,
                    item.attack_type, item.sample_index, item.generation_replicate,
                    item.sequence_id,
                )
                sequence_plan_keys.add(seq_key)
        planned_sequence_count = len(sequence_plan_keys)

    # Group raw attempts by sequence scientific identity.
    seq_groups: dict[tuple, list[EmpiricalGenerationAttempt]] = {}
    for a in attempts:
        if a.is_sequence_attempt and a.sequence_family_id is not None:
            key = (
                a.scenario_id, a.secret_variant_id, a.trust_level,
                a.attack_type, a.sample_index, a.generation_replicate,
                a.sequence_family_id,
            )
            seq_groups.setdefault(key, []).append(a)

    # If no plan, derive planned count from raw groups.
    if planned_sequence_count == 0:
        planned_sequence_count = len(seq_groups)

    # Classify each sequence.
    complete_count = 0
    incomplete_count = 0
    accepted_count = 0
    rejected_count = 0
    invalid_lineage_count = 0
    rejection_reasons: list[str] = []
    status_counts: dict[str, int] = {}

    specs_by_key: dict[tuple[str, str], EmpiricalTargetSpec] = {
        (s.scenario_id, s.secret_variant_id): s for s in target_registry
    }

    for key, group_attempts in sorted(seq_groups.items()):
        family_id = key[-1]
        # Patch H: expected step count from frozen plan, not observed max.
        plan_expected = plan_step_index.get(key)
        if plan_expected is not None:
            expected_steps = plan_expected
        else:
            # Fallback: use sequence_step_count recorded on attempts if all agree.
            attempt_step_counts = {
                a.sequence_step_count for a in group_attempts
                if a.sequence_step_count is not None
            }
            if len(attempt_step_counts) == 1:
                expected_steps = attempt_step_counts.pop()
            else:
                # Last-resort diagnostic fallback.
                step_indices_seen = {a.sequence_step_index for a in group_attempts}
                expected_steps = max(step_indices_seen) + 1 if step_indices_seen else 0
        if expected_steps == 0:
            incomplete_count += 1
            status_counts["incomplete"] = status_counts.get("incomplete", 0) + 1
            continue

        # Collapse retries per step.
        try:
            terminal_steps = terminal_attempts_by_sequence_step(group_attempts)
        except ValueError:
            invalid_lineage_count += 1
            status_counts["invalid_retry_lineage"] = status_counts.get("invalid_retry_lineage", 0) + 1
            rejection_reasons.append(f"sequence {family_id}: invalid retry lineage")
            continue

        # Check completeness.
        terminal_step_indices = sorted(a.sequence_step_index for a in terminal_steps)
        if terminal_step_indices != list(range(expected_steps)):
            incomplete_count += 1
            status_counts["incomplete"] = status_counts.get("incomplete", 0) + 1
            rejection_reasons.append(f"sequence {family_id}: incomplete steps")
            continue

        complete_count += 1

        # Check structural validity.
        problems = validate_sequence_structure(terminal_steps)
        if problems:
            rejected_count += 1
            status_counts["complete_rejected"] = status_counts.get("complete_rejected", 0) + 1
            rejection_reasons.append(f"sequence {family_id}: {'; '.join(problems)}")
            continue

        # Check acceptance for each step.
        scenario_id, secret_variant_id = key[0], key[1]
        spec = specs_by_key.get((scenario_id, secret_variant_id))
        if spec is None:
            rejected_count += 1
            status_counts["complete_rejected"] = status_counts.get("complete_rejected", 0) + 1
            rejection_reasons.append(f"sequence {family_id}: unknown target spec")
            continue

        all_accepted = True
        for step_attempt in terminal_steps:
            result = accept_generation_attempt(step_attempt, spec)
            if not result.accepted:
                all_accepted = False
                rejection_reasons.append(result.reason)
                break

        if all_accepted:
            accepted_count += 1
            status_counts["complete_accepted"] = status_counts.get("complete_accepted", 0) + 1
        else:
            rejected_count += 1
            status_counts["complete_rejected"] = status_counts.get("complete_rejected", 0) + 1

    # Account for planned sequences with no attempts yet.
    incomplete_count = planned_sequence_count - complete_count
    if incomplete_count < 0:
        incomplete_count = 0

    return {
        "planned_sequence_count": planned_sequence_count,
        "complete_sequence_count": complete_count,
        "incomplete_sequence_count": incomplete_count,
        "accepted_sequence_count": accepted_count,
        "rejected_sequence_count": rejected_count,
        "invalid_retry_lineage_count": invalid_lineage_count,
        "rejection_reasons": dict(sorted(Counter(rejection_reasons).items())),
        "sequence_status_counts": dict(sorted(status_counts.items())),
    }


def _load_generation_plan(path: Path, *, max_items: int | None = None) -> list["GenerationPlanItem"]:
    """Load a generation plan from a JSONL file.

    Delegates to the canonical :func:`load_generation_plan` in
    ``empirical_generation_plan``.
    """
    from experiments.trustparadox_u.empirical_generation_plan import load_generation_plan

    items = load_generation_plan(path)
    if max_items is not None:
        items = items[:max_items]
    return items


# ---------------------------------------------------------------------------
# E1-020: generation runner
# ---------------------------------------------------------------------------


def run_generation(
    *,
    split: str,
    mode: str,
    scenarios: Sequence[str],
    trust_levels: Sequence[str],
    attack_types: Sequence[str],
    samples: int,
    output_dir: Path,
    generator: EmpiricalCandidateGenerator,
    temperature: float = 0.7,
    artifact_class: str = "development_smoke",
    research_use: str = "diagnostic_only",
    resume: bool = False,
    plan_items: Sequence[GenerationPlanItem] | None = None,
    max_tokens: int | None = None,
    retry_policy: Mapping[str, object] | None = None,
    plan_path: Path | None = None,
    frozen_config: FrozenGenerationConfig | None = None,
    frozen_config_path: Path | None = None,
    phase_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Generate, retain, and accept empirical attempts for one split.

    E1-021: raises ``EmpiricalPhaseLockedError`` for validation/test while
    the empirical phase is E1.

    E3-005: requires a clean working tree for real-mode campaigns.

    E3-007: when *resume* is true, existing raw attempts are loaded and
    completed units are skipped.  Accepted candidates are rebuilt from
    all raw attempts using the current frozen acceptance code.
    """
    assert_generation_split_unlocked(split)
    EmpiricalSplit(split)  # validates the split value

    # Patch B.3: defensive mixed-plan rejection.
    if plan_items is not None:
        foreign = [
            item
            for item in plan_items
            if item.split != EmpiricalSplit(split).value
        ]
        if foreign:
            raise ValueError(
                f"run_generation received {len(foreign)} plan items "
                f"for splits other than {split!r}"
            )

    # Patch H: campaign guard — plan-driven real must use generate_once.
    if mode == "real" and plan_items is not None:
        if not hasattr(generator, "generate_once"):
            raise RuntimeError(
                "plan-driven real campaign requires generate_once(); "
                "legacy generate() with hidden retry is not permitted"
            )

    # E3-005: require clean committed tree for real generation campaigns.
    if mode == "real" and not working_tree_is_fully_clean():
        raise RuntimeError("refusing to start real generation with dirty working tree")

    # Patch E: campaign identity protection.
    if frozen_config is not None and mode == "real":
        current_identity = compute_campaign_identity(
            split=split,
            plan_items=list(plan_items) if plan_items else None,
            plan_path=plan_path,
            config=frozen_config,
            config_path=frozen_config_path,
            phase_manifest_path=phase_manifest_path,
        )
        existing_identity = load_campaign_identity(output_dir)
        if existing_identity is not None:
            # Resume: verify identity matches.
            verify_campaign_identity(existing_identity, current_identity)
        else:
            # First run: write identity before any provider request.
            write_campaign_identity(output_dir, current_identity)

    attempts: list[EmpiricalGenerationAttempt] = []
    accepted: list[EmpiricalCandidate] = []
    rejection_reasons: list[str] = []
    duplicate_id_count = 0

    # E3-007: load existing attempts for resume.
    existing_attempts: list[EmpiricalGenerationAttempt] = []
    if resume:
        existing_attempts = _load_attempts(output_dir / RAW_ATTEMPTS_FILENAME)
        # Patch D: rebuild accepted from complete raw history.
        rebuilt_accepted, rebuilt_rejections = rebuild_accepted_candidates(
            existing_attempts, EMPIRICAL_TARGET_REGISTRY,
        )
        accepted.extend(rebuilt_accepted)
        rejection_reasons.extend(rebuilt_rejections)

    # Patch D: derive retry policy parameters for resume-state classification.
    _rp_max_retries = int(retry_policy.get("max_retries", 0)) if retry_policy else 0
    _rp_retryable = tuple(
        retry_policy.get("retryable_statuses", ["provider_error"])
    ) if retry_policy else ("provider_error",)

    raw_writer = RawAttemptWriter(output_dir / RAW_ATTEMPTS_FILENAME)

    # Sequence tracking for E3-006 report.
    planned_sequences = 0
    complete_sequences = 0
    accepted_sequences = 0
    rejected_sequences = 0
    sequence_rejection_reasons: list[str] = []

    if plan_items is not None:
        # Plan-driven iteration (E3-007): group by unit and process.
        from experiments.trustparadox_u.empirical_generation_plan import (
            attack_is_applicable,
        )

        specs_by_key = {
            (s.scenario_id, s.secret_variant_id): s for s in specs_for_split(split)
        }
        groups: dict[tuple, list[GenerationPlanItem]] = {}
        for item in plan_items:
            key = (
                item.scenario_id,
                item.secret_variant_id,
                item.trust_level,
                item.attack_type,
                item.sample_index,
                item.generation_replicate,
            )
            groups.setdefault(key, []).append(item)

        for _key, group in sorted(groups.items()):
            first = group[0]
            spec_key = (first.scenario_id, first.secret_variant_id)
            spec = specs_by_key.get(spec_key)
            if spec is None:
                raise ValueError(
                    f"generation plan references unknown target spec: "
                    f"{first.scenario_id}/{first.secret_variant_id}"
                )
            # Hard invariants: plan item must match the resolved target spec.
            assert spec.scenario_id == first.scenario_id
            assert spec.secret_variant_id == first.secret_variant_id
            assert spec.split == first.split
            if not attack_is_applicable(first.attack_type, spec):
                continue
            is_seq = AttackType(first.attack_type) in SEQUENCE_ATTACK_TYPES
            if is_seq:
                planned_sequences += 1
            # Patch D: resume-state classification.
            skip_steps: frozenset[int] | None = None
            start_retry_index = 0
            retry_start_by_step: Mapping[int, int] | None = None
            if resume and existing_attempts:
                step_count = sequence_step_count_for(first.attack_type, spec) if is_seq else 0
                resume_info = resolve_unit_resume_state(
                    _key,
                    existing_attempts,
                    is_sequence=is_seq,
                    expected_step_count=step_count,
                    max_retries=_rp_max_retries,
                    retryable_statuses=_rp_retryable,
                )
                state = resume_info["state"]
                if state in (
                    UnitResumeState.COMPLETE_SUCCESS,
                    UnitResumeState.COMPLETE_FAILURE,
                    UnitResumeState.SEQUENCE_COMPLETE_SUCCESS,
                    UnitResumeState.SEQUENCE_COMPLETE_FAILURE,
                ):
                    continue
                if state is UnitResumeState.RETRY_PENDING:
                    start_retry_index = int(resume_info["retry_index_to_continue_from"])
                elif state is UnitResumeState.SEQUENCE_PARTIAL:
                    completed = resume_info["completed_step_indices"]
                    skip_steps = frozenset(completed)
                    retry_start_by_step = resume_info["retry_start_by_step"]
            new = _generate_unit(
                generator=generator,
                spec=spec,
                trust_level=first.trust_level,
                attack_type=first.attack_type,
                sample_index=first.sample_index,
                temperature=temperature,
                raw_writer=raw_writer,
                accepted=accepted,
                rejection_reasons=rejection_reasons,
                retry_policy=retry_policy,
                skip_steps=skip_steps,
                start_retry_index=start_retry_index,
                retry_start_by_step=retry_start_by_step,
                existing_attempts=existing_attempts,
            )
            attempts.extend(new)
            if is_seq:
                seq_ok, seq_cands, seq_reasons = accept_sequence_attempts(new, spec)
                if seq_ok:
                    complete_sequences += 1
                    accepted_sequences += 1
                else:
                    rejected_sequences += 1
                    sequence_rejection_reasons.extend(seq_reasons)
    else:
        # Standard nested-loop iteration.
        target_specs = specs_for_split(split, scenarios)
        for spec in target_specs:
            for trust_level in trust_levels:
                TrustLevel(trust_level)
                for attack_type in attack_types:
                    is_seq = AttackType(attack_type) in SEQUENCE_ATTACK_TYPES
                    for sample_index in range(samples):
                        unit_key = (
                            spec.scenario_id,
                            spec.secret_variant_id,
                            trust_level,
                            attack_type,
                            sample_index,
                            0,
                        )
                        # Patch D: resume-state classification.
                        if resume and existing_attempts:
                            step_count = sequence_step_count_for(attack_type, spec) if is_seq else 0
                            resume_info = resolve_unit_resume_state(
                                unit_key,
                                existing_attempts,
                                is_sequence=is_seq,
                                expected_step_count=step_count,
                                max_retries=_rp_max_retries,
                                retryable_statuses=_rp_retryable,
                            )
                            state = resume_info["state"]
                            if state in (
                                UnitResumeState.COMPLETE_SUCCESS,
                                UnitResumeState.COMPLETE_FAILURE,
                                UnitResumeState.SEQUENCE_COMPLETE_SUCCESS,
                                UnitResumeState.SEQUENCE_COMPLETE_FAILURE,
                            ):
                                continue
                            skip_steps = None
                            start_retry_index = 0
                            retry_start_by_step_loop: Mapping[int, int] | None = None
                            if state is UnitResumeState.RETRY_PENDING:
                                start_retry_index = int(resume_info["retry_index_to_continue_from"])
                            elif state is UnitResumeState.SEQUENCE_PARTIAL:
                                skip_steps = frozenset(resume_info["completed_step_indices"])
                                retry_start_by_step_loop = resume_info["retry_start_by_step"]
                        else:
                            skip_steps = None
                            start_retry_index = 0
                            retry_start_by_step_loop = None
                        if is_seq:
                            planned_sequences += 1
                        new = _generate_unit(
                            generator=generator,
                            spec=spec,
                            trust_level=trust_level,
                            attack_type=attack_type,
                            sample_index=sample_index,
                            temperature=temperature,
                            raw_writer=raw_writer,
                            accepted=accepted,
                            rejection_reasons=rejection_reasons,
                            retry_policy=retry_policy,
                            skip_steps=skip_steps,
                            start_retry_index=start_retry_index,
                            retry_start_by_step=retry_start_by_step_loop,
                            existing_attempts=existing_attempts,
                        )
                        attempts.extend(new)
                        if is_seq:
                            step_count = sequence_step_count_for(attack_type, spec)
                            seq_ok, seq_cands, seq_reasons = accept_sequence_attempts(
                                new, spec,
                            )
                            if seq_ok:
                                complete_sequences += 1
                                accepted_sequences += 1
                            else:
                                rejected_sequences += 1
                                sequence_rejection_reasons.extend(seq_reasons)

    # E3-007: all_attempts includes loaded + new for manifest/report.
    all_attempts = list(existing_attempts) + attempts

    # Patch D: rebuild accepted from ALL raw attempts (materialized view).
    if resume:
        final_accepted, final_rejections = rebuild_accepted_candidates(
            all_attempts, EMPIRICAL_TARGET_REGISTRY,
        )
        accepted.clear()
        accepted.extend(final_accepted)

    accepted_sorted = sorted(accepted, key=lambda c: c.candidate_id)
    _write_jsonl_atomic(
        output_dir / ACCEPTED_CANDIDATES_FILENAME,
        [candidate_to_record(candidate) for candidate in accepted_sorted],
    )

    prompt_manifest = build_prompt_manifest()
    _write_json_atomic(output_dir / PROMPT_MANIFEST_FILENAME, prompt_manifest)

    # Patch I: compute campaign identity hash for the manifest.
    _identity_hash = ""
    _loaded_id = load_campaign_identity(output_dir)
    if _loaded_id is not None:
        _identity_hash = campaign_identity_sha256(_loaded_id)

    # Patch K: compute split-plan identity hashes for the manifest.
    _full_plan_sha: str | None = None
    _split_plan_sha: str | None = None
    _split_plan_count: int | None = None
    if plan_items is not None:
        from experiments.trustparadox_u.empirical_generation_plan import (
            load_generation_plan as _load_full_plan,
            plan_sha256 as _plan_sha256,
        )
        _split_plan_sha = _plan_sha256(plan_items)
        _split_plan_count = len(plan_items)
        if plan_path is not None:
            _full_plan = _load_full_plan(plan_path)
            _full_plan_sha = _plan_sha256(_full_plan)

    manifest = build_corpus_manifest(
        generation_mode=generator.generation_mode,
        attempts=all_attempts,
        accepted=accepted_sorted,
        prompt_manifest=prompt_manifest,
        artifact_class=artifact_class,
        research_use=research_use,
        campaign_identity_hash=_identity_hash,
        full_generation_plan_sha256=_full_plan_sha,
        split_generation_plan_sha256=_split_plan_sha,
        split_plan_item_count=_split_plan_count,
    )
    _write_json_atomic(output_dir / CORPUS_MANIFEST_FILENAME, manifest)

    report = build_validation_report(
        attempts=all_attempts,
        accepted=accepted_sorted,
        rejection_reasons=rejection_reasons,
        duplicate_id_count=duplicate_id_count,
        manifest=manifest,
        output_dir=output_dir,
        phase_lock_valid=True,
    )
    _write_json_atomic(output_dir / VALIDATION_REPORT_FILENAME, report)

    # E3-006 / Patch D: sequence generation report from complete raw history.
    seq_report = build_sequence_generation_report_from_attempts(
        attempts=all_attempts,
        plan_items=plan_items,
        target_registry=EMPIRICAL_TARGET_REGISTRY,
    )
    _write_json_atomic(output_dir / SEQUENCE_GENERATION_REPORT_FILENAME, seq_report)

    return report


def _generate_unit(
    *,
    generator: EmpiricalCandidateGenerator,
    spec: EmpiricalTargetSpec,
    trust_level: str,
    attack_type: str,
    sample_index: int,
    temperature: float,
    raw_writer: RawAttemptWriter,
    accepted: list[EmpiricalCandidate],
    rejection_reasons: list[str],
    retry_policy: Mapping[str, object] | None = None,
    skip_steps: frozenset[int] | None = None,
    start_retry_index: int = 0,
    retry_start_by_step: Mapping[int, int] | None = None,
    existing_attempts: Sequence[EmpiricalGenerationAttempt] = (),
) -> list[EmpiricalGenerationAttempt]:
    """Generate one trial unit (one attempt, or one per sequence step).

    E3-005: records ``trust_prompt_hash``, ``attack_prompt_hash``, and
    ``max_tokens`` on every attempt.

    E3-006: sequence attempts are generated atomically — all steps are
    retained in the raw file, but none are accepted into the corpus
    unless the full sequence validates.

    Patch B: ``max_tokens`` comes from ``generator.max_tokens``.
    Patch C: retry is handled at the campaign layer via
    :func:`generate_with_retry`.
    Patch D: ``skip_steps`` allows partial-sequence resume (only generate
    steps not in the skip set); ``start_retry_index`` continues a partial
    retry chain.
    """
    is_sequence = AttackType(attack_type) in SEQUENCE_ATTACK_TYPES
    step_count = sequence_step_count_for(attack_type, spec) if is_sequence else None
    if step_count is not None:
        all_step_indices = list(range(step_count))
        active_steps = [s for s in all_step_indices if skip_steps is None or s not in skip_steps]
    else:
        all_step_indices = [None]
        active_steps = [None]

    gen_max_tokens = getattr(generator, "max_tokens", None)
    new_attempts: list[EmpiricalGenerationAttempt] = []
    terminal_by_step: dict[int | None, EmpiricalGenerationAttempt] = {}

    for step_index in all_step_indices:
        if step_index in active_steps:
            # Generate this step.
            bundle = resolve_prompt_bundle(
                trust_level,
                attack_type,
                spec,
                sequence_step_index=step_index,
                sequence_step_count=step_count,
            )
            request = build_generation_request(
                spec,
                trust_level,
                attack_type,
                sample_index,
                temperature=temperature,
                sequence_step_index=step_index,
                sequence_step_count=step_count,
            )
            if is_sequence:
                step_retry_index = (
                    retry_start_by_step.get(step_index, 0)
                    if retry_start_by_step is not None
                    else 0
                )
            else:
                step_retry_index = start_retry_index
            step_attempts = generate_with_retry(
                generator=generator,
                request=request,
                retry_policy=retry_policy,
                raw_writer=raw_writer,
                spec=spec,
                trust_level=trust_level,
                attack_type=attack_type,
                sample_index=sample_index,
                generation_mode=GenerationMode(generator.generation_mode).value,
                transport=getattr(generator, "transport", None),
                generator_model_requested=getattr(generator, "model_name", None),
                max_tokens=gen_max_tokens,
                trust_prompt_hash=prompt_sha256(bundle.trust_prompt),
                attack_prompt_hash=prompt_sha256(bundle.attack_prompt),
                temperature=temperature,
                start_retry_index=step_retry_index,
            )
            new_attempts.extend(step_attempts)
            if is_sequence:
                terminal_by_step[step_index] = terminal_attempt_for_retry_segment(
                    step_attempts,
                    expected_start_retry_index=step_retry_index,
                )
            else:
                terminal_by_step[None] = terminal_attempt_for_retry_segment(
                    step_attempts,
                    expected_start_retry_index=step_retry_index,
                )

    # Patch C: populate terminal_by_step for skipped steps from existing
    # attempts so that sequence completeness checks see all steps.
    if skip_steps and existing_attempts:
        for a in existing_attempts:
            si = a.sequence_step_index
            if si in skip_steps:
                # Group existing attempts by skipped step and pick terminal.
                step_chain = [
                    e for e in existing_attempts
                    if e.sequence_step_index == si
                    and e.scenario_id == spec.scenario_id
                    and e.secret_variant_id == spec.secret_variant_id
                    and e.trust_level == trust_level
                    and e.attack_type == attack_type
                    and e.sample_index == sample_index
                    and e.generation_replicate == (getattr(a, 'generation_replicate', 0))
                ]
                if step_chain and si not in terminal_by_step:
                    terminal_by_step[si] = terminal_attempt_for_retry_chain(step_chain)

    if is_sequence:
        # Collect terminal attempts for ALL steps (existing + new).
        all_terminal: list[EmpiricalGenerationAttempt] = []
        for si in sorted(terminal_by_step.keys(), key=lambda x: x if x is not None else -1):
            if si in terminal_by_step:
                all_terminal.append(terminal_by_step[si])
        # Patch E: truly atomic sequence acceptance (all-or-nothing).
        if len(all_terminal) == step_count:
            seq_ok, seq_candidates, seq_reasons = accept_sequence_attempts(
                all_terminal, spec,
            )
            if seq_ok:
                accepted.extend(seq_candidates)
            else:
                rejection_reasons.extend(seq_reasons)
    else:
        assert None in terminal_by_step
        terminal = terminal_by_step[None]
        result = accept_generation_attempt(terminal, spec)
        if result.accepted and result.candidate is not None:
            accepted.append(result.candidate)
        else:
            rejection_reasons.append(result.reason)

    return new_attempts


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write JSONL records (non-atomic; used for small diagnostic files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON payload (non-atomic; used for small diagnostic files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Empirical corpus generation (E1/E3).")
    parser.add_argument("--split", required=True, choices=[s.value for s in EmpiricalSplit])
    parser.add_argument("--mode", required=True, choices=["mock", "real"])
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario id (repeatable). Default: all empirical scenarios.",
    )
    parser.add_argument(
        "--trust",
        action="append",
        default=None,
        help="Trust level (repeatable). Default: low, default, high.",
    )
    parser.add_argument(
        "--attack",
        action="append",
        default=None,
        help="Attack type (repeatable). Default: the smoke attack families.",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generator-model", default=None)
    parser.add_argument(
        "--provider",
        default="openai",
        help="Serving provider recorded in attempt provenance (E2-002).",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key-env", default=None)
    # E3-007: resume and plan flags.
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume an interrupted campaign without duplicating attempts.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Path to a generation plan JSONL (E3-004).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print plan summary without generating.",
    )
    parser.add_argument(
        "--max-plan-items",
        type=int,
        default=None,
        help="Truncate the plan to N items (diagnostic/preflight only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        assert_generation_split_unlocked(args.split)
    except RuntimeError as exc:
        print(f"EmpiricalPhaseLockedError: {exc}", file=sys.stderr)
        return 2

    if args.samples < 1:
        print("--samples must be >= 1", file=sys.stderr)
        return 2

    scenarios = args.scenario or sorted({spec.scenario_id for spec in specs_for_split(args.split)})
    trust_levels = args.trust or [level.value for level in TrustLevel]
    attack_types = args.attack or list(DEFAULT_SMOKE_ATTACKS)

    # E3-007: load plan if provided.
    plan_items: list[GenerationPlanItem] | None = None
    full_plan_items: list[GenerationPlanItem] | None = None
    if args.plan is not None:
        full_plan_items = _load_generation_plan(args.plan, max_items=args.max_plan_items)
        # Patch B: filter to split-specific items before execution.
        from experiments.trustparadox_u.empirical_generation_plan import (
            plan_items_for_split,
        )
        plan_items = plan_items_for_split(full_plan_items, args.split)

    # E3-007: dry-run reports plan summary and exits.
    if args.dry_run:
        if plan_items is not None:
            from experiments.trustparadox_u.empirical_generation_plan import (
                plan_summary as _plan_summary,
            )
            _split_summary = _plan_summary(plan_items)
            print(json.dumps({
                "dry_run": True,
                "full_plan_item_count": len(full_plan_items) if full_plan_items else 0,
                "selected_split_plan_item_count": len(plan_items),
                "selected_split_scientific_unit_count": _split_summary.get(
                    "scientific_generation_unit_count", 0
                ),
                "selected_split_sequence_count": _split_summary.get(
                    "distinct_sequences", 0
                ),
            }))
        else:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "scenarios": len(scenarios),
                        "trust_levels": len(trust_levels),
                        "attack_types": len(attack_types),
                        "samples": args.samples,
                    }
                )
            )
        return 0

    # Patch J: write split execution-plan summary before any provider call.
    if plan_items is not None and not args.dry_run:
        from experiments.trustparadox_u.empirical_generation_plan import (
            plan_summary as _plan_summary,
            plan_sha256 as _plan_sha256,
        )
        _summary = _plan_summary(plan_items)
        _variants_in_split = len({it.secret_variant_id for it in plan_items})
        _foreign = [it for it in (full_plan_items or []) if it.split != args.split]
        exec_plan = {
            "split": args.split,
            "full_plan_item_count": len(full_plan_items) if full_plan_items else 0,
            "selected_plan_item_count": len(plan_items),
            "selected_sequence_step_item_count": _summary["sequence_step_attempts"],
            "selected_non_sequence_item_count": _summary["non_sequence_attempts"],
            "selected_scientific_sequence_count": _summary["distinct_sequences"],
            "selected_secret_variant_count": _variants_in_split,
            "foreign_plan_item_count": len(_foreign),
            "full_plan_scientific_sha256": (
                _plan_sha256(full_plan_items) if full_plan_items else None
            ),
            "split_plan_scientific_sha256": _plan_sha256(plan_items),
        }
        exec_plan_path = args.output_dir / "generation_execution_plan.json"
        exec_plan_path.parent.mkdir(parents=True, exist_ok=True)
        exec_plan_path.write_text(
            json.dumps(exec_plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Execution plan written to: {exec_plan_path}")

    generator: EmpiricalCandidateGenerator
    retry_policy: dict[str, object] | None = None
    frozen_config: FrozenGenerationConfig | None = None
    effective_temperature: float
    if args.mode == "mock":
        generator = MockEmpiricalGenerator()
        effective_temperature = args.temperature if args.temperature is not None else 0.7
    elif args.plan is not None:
        # Patch B: full plan-driven real campaign uses frozen config only.
        frozen_config = load_frozen_generation_config()
        # Patch G: reject incompatible temperature override.
        if (
            args.temperature is not None
            and args.temperature != frozen_config.generator_temperature
        ):
            print(
                f"ERROR: --temperature {args.temperature} conflicts with frozen "
                f"config temperature {frozen_config.generator_temperature}. "
                f"Plan-driven mode uses the frozen config only.",
                file=sys.stderr,
            )
            return 2
        effective_temperature = frozen_config.generator_temperature
        generator = RealEmpiricalGenerator.from_frozen_config(
            frozen_config,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
        )
        retry_policy = {
            "max_retries": frozen_config.max_retries,
            "backoff_seconds": list(frozen_config.backoff_seconds),
            "retryable_statuses": list(frozen_config.retryable_statuses),
        }
    else:
        if not args.generator_model:
            print("--generator-model is required for --mode real", file=sys.stderr)
            return 2
        effective_temperature = args.temperature if args.temperature is not None else 0.7
        generator = RealEmpiricalGenerator(
            provider=args.provider,
            model_name=args.generator_model,
            temperature=effective_temperature,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
        )

    # Patch E: paths for campaign identity computation.
    _manifests_dir = _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "manifests"
    _frozen_config_path = _manifests_dir / "full_generation_config.json"
    _phase_manifest_path = _manifests_dir / "empirical_phase.json"

    try:
        report = run_generation(
            split=args.split,
            mode=args.mode,
            scenarios=sorted(set(scenarios)),
            trust_levels=trust_levels,
            attack_types=attack_types,
            samples=args.samples,
            output_dir=args.output_dir,
            generator=generator,
            temperature=effective_temperature,
            resume=args.resume,
            plan_items=plan_items,
            max_tokens=getattr(generator, "max_tokens", None),
            retry_policy=retry_policy,
            plan_path=args.plan,
            frozen_config=frozen_config,
            frozen_config_path=_frozen_config_path,
            phase_manifest_path=_phase_manifest_path,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "attempt_count": report["attempt_count"],
                "accepted_count": report["accepted_count"],
                "rejected_count": report["rejected_count"],
                "e1_foundation_valid": report["e1_foundation_valid"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0 if report["e1_foundation_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

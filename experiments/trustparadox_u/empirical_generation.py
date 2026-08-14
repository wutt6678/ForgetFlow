"""Empirical candidate generation for the TrustParadox-U empirical replay study.

Checklist coverage (Iteration 1):

- E1-008: ``EmpiricalCandidateGenerator`` protocol plus provider-neutral
  ``EmpiricalGenerationRequest`` / ``EmpiricalGenerationResponse`` types.
  Nothing in the scientific schema is coupled to OpenAI response classes.
- E1-009: ``MockEmpiricalGenerator`` — deterministic, fixture-driven.
  ``generation_mode = mock``, ``study_class = diagnostic``. Never evidence.
- E1-010: ``RealEmpiricalGenerator`` — configuration-driven LiteLLM adapter.
  No hidden fallback to mock, no silent model substitution, provider/model
  mismatch fails, errors remain raw attempts, retries are traceable.
- E1-011/E1-012: prompt template loading, structured ``[SECTION]`` parsing,
  and the automated trust-prompt invariance check (only ``TRUST_FRAMING``
  may differ across low/default/high).
- E1-013: prompt hashing — sha256 over the exact resolved UTF-8 bytes for
  system/user/trust/attack prompts, plus a source-template prompt manifest
  labeled ``status = frozen_post_pilot``.
- E1-014: ``RawAttemptWriter`` — append-only JSONL writer that preserves
  failures, refusals, and malformed responses, flushes safely, and rejects
  duplicate generation-attempt IDs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    SEQUENCE_ATTACK_TYPES,
    AttackType,
    EmpiricalGenerationAttempt,
    EmpiricalTargetSpec,
    GenerationMode,
    GenerationStatus,
    TrustLevel,
    attempt_to_record,
    empirical_candidate_family_id,
    empirical_sequence_family_id,
    empirical_sequence_id,
    generation_attempt_id,
    provider_attempt_id as _compute_provider_attempt_id,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMPIRICAL_PROMPTS_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "prompts"

#: E2 repair §7: pilot-specific prompt namespace.
PILOT_PROMPTS_DIR = EMPIRICAL_PROMPTS_DIR / "pilot"

PROMPT_MANIFEST_STATUS = "frozen_post_pilot"

SYSTEM_PROMPT_FILE = "generator_system.txt"

#: E2 repair §7: primary pilot task template (discretion-based, no disclosure imperative).
PRIMARY_TASK_FILE = "primary_task.txt"

TRUST_PROMPT_FILES: Mapping[TrustLevel, str] = {
    TrustLevel.LOW: "trust_low.txt",
    TrustLevel.DEFAULT: "trust_default.txt",
    TrustLevel.HIGH: "trust_high.txt",
}

#: E2 repair §7: pilot trust prompt files (pilot/ subdirectory).
PILOT_TRUST_PROMPT_FILES: Mapping[TrustLevel, str] = {
    TrustLevel.LOW: "trust_low.txt",
    TrustLevel.DEFAULT: "trust_default.txt",
    TrustLevel.HIGH: "trust_high.txt",
}

ATTACK_TEMPLATE_FILES: Mapping[AttackType, str] = {
    AttackType.DIRECT_DISCLOSURE: "attacks/direct.txt",
    AttackType.SEMANTIC_PARAPHRASE: "attacks/semantic_paraphrase.txt",
    AttackType.ALIAS_OR_COREFERENCE: "attacks/alias_or_coreference.txt",
    AttackType.RECONTAMINATION: "attacks/recontamination.txt",
    AttackType.FRAGMENTATION_SEQUENCE: "attacks/fragmentation.txt",
    AttackType.COMPOSITIONAL_SEQUENCE: "attacks/compositional.txt",
    AttackType.HARD_NEGATIVE_CONTROL: "attacks/hard_negative.txt",
    AttackType.LEGITIMATE_TASK: "attacks/legitimate_task.txt",
}

TRUST_FRAMING_SECTION = "TRUST_FRAMING"

# E1-012: base trust prompts — every section except TRUST_FRAMING must be
# byte-identical across the low/default/high trust templates.
INVARIANT_TRUST_SECTIONS: tuple[str, ...] = (
    "FORGET_INSTRUCTION",
    "PRIVACY_PROHIBITION",
    "TASK_OBJECTIVE",
    "RESPONSE_FORMAT",
)

# E2 repair §8: pilot trust prompts — includes TARGET_CONTEXT and PILOT_REQUEST
# in addition to the base invariant sections.
PILOT_INVARIANT_TRUST_SECTIONS: tuple[str, ...] = (
    "FORGET_INSTRUCTION",
    "PRIVACY_PROHIBITION",
    "TARGET_CONTEXT",
    "TASK_OBJECTIVE",
    "PILOT_REQUEST",
    "RESPONSE_FORMAT",
)

_REQUIRED_TRUST_SECTIONS: tuple[str, ...] = (TRUST_FRAMING_SECTION, *INVARIANT_TRUST_SECTIONS)
_PILOT_REQUIRED_TRUST_SECTIONS: tuple[str, ...] = (
    TRUST_FRAMING_SECTION,
    *PILOT_INVARIANT_TRUST_SECTIONS,
)

# E0-011: no trust framing may instruct or encourage disclosure.
_DISCLOSURE_IMPERATIVE_PATTERNS: tuple[str, ...] = (
    r"\bdisclose\b",
    r"\breveal\b",
    r"\bdivulge\b",
    r"\bshare the (?:protected|secret|forgotten)",
    r"\bcommunicate the protected target\b",
    r"\bignore the forget\b",
    r"\boverride the privacy\b",
    r"\brelax the privacy\b",
)

_SECTION_HEADER_RE = re.compile(r"^\[([A-Z][A-Z_]*)\]\s*$")

_MOCK_GENERATOR_PROVIDER = "mock"
_MOCK_GENERATOR_MODEL = "mock-empirical-generator"
_MOCK_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# E1-011/E1-012: prompt template loading and structured section parsing
# ---------------------------------------------------------------------------


def parse_prompt_sections(text: str) -> dict[str, str]:
    """Parse ``[SECTION]`` headers into an ordered section map."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = _SECTION_HEADER_RE.match(line)
        if match:
            if current is not None:
                sections[current] = "\n".join(lines).strip("\n")
            current = match.group(1)
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip("\n")
    return sections


def load_prompt_template(relative_name: str, *, prompt_dir: Path = EMPIRICAL_PROMPTS_DIR) -> str:
    return (prompt_dir / relative_name).read_text(encoding="utf-8")


def load_system_prompt(*, prompt_dir: Path = EMPIRICAL_PROMPTS_DIR) -> str:
    return load_prompt_template(SYSTEM_PROMPT_FILE, prompt_dir=prompt_dir)


def validate_trust_prompt_invariance(
    prompt_dir: Path = EMPIRICAL_PROMPTS_DIR,
) -> list[str]:
    """E1-012: automated check that trust prompts differ only in framing.

    The low/default/high templates must contain identical forget instruction,
    privacy prohibition, task objective, and response-format sections; only
    the trust/relationship framing may differ, and no framing may contain a
    disclosure imperative.
    """
    problems: list[str] = []
    parsed: dict[TrustLevel, dict[str, str]] = {}
    for trust_level, filename in TRUST_PROMPT_FILES.items():
        try:
            text = load_prompt_template(filename, prompt_dir=prompt_dir)
        except OSError as exc:
            problems.append(f"cannot read {filename}: {exc}")
            continue
        sections = parse_prompt_sections(text)
        parsed[trust_level] = sections
        missing = [name for name in _REQUIRED_TRUST_SECTIONS if name not in sections]
        if missing:
            problems.append(f"{filename}: missing sections {sorted(missing)}")
        extra = [name for name in sections if name not in _REQUIRED_TRUST_SECTIONS]
        if extra:
            problems.append(f"{filename}: unexpected sections {sorted(extra)}")

    levels = sorted(level for level in parsed)
    for section in INVARIANT_TRUST_SECTIONS:
        reference: str | None = None
        for level in levels:
            value = parsed[level].get(section)
            if reference is None:
                reference = value
            elif value != reference:
                problems.append(
                    f"section {section} differs between trust prompts "
                    f"({levels[0].value} vs {level.value})"
                )

    framings = {level: parsed[level].get(TRUST_FRAMING_SECTION, "") for level in levels}
    for i, level_a in enumerate(levels):
        for level_b in levels[i + 1 :]:
            if framings[level_a] == framings[level_b]:
                problems.append(f"TRUST_FRAMING identical for {level_a.value} and {level_b.value}")
    for level, framing in framings.items():
        for pattern in _DISCLOSURE_IMPERATIVE_PATTERNS:
            if re.search(pattern, framing, flags=re.IGNORECASE):
                problems.append(
                    f"trust {level.value} framing contains disclosure imperative "
                    f"matching {pattern!r}"
                )
    return problems


def validate_pilot_prompt_invariance(
    prompt_dir: Path = PILOT_PROMPTS_DIR,
) -> list[str]:
    """E2 repair §8: pilot trust prompts differ only in TRUST_FRAMING.

    The pilot low/default/high templates must contain identical
    TARGET_CONTEXT, FORGET_INSTRUCTION, PRIVACY_PROHIBITION,
    TASK_OBJECTIVE, PILOT_REQUEST, and RESPONSE_FORMAT sections.
    """
    problems: list[str] = []
    parsed: dict[TrustLevel, dict[str, str]] = {}
    for trust_level, filename in PILOT_TRUST_PROMPT_FILES.items():
        try:
            text = load_prompt_template(filename, prompt_dir=prompt_dir)
        except OSError as exc:
            problems.append(f"cannot read {filename}: {exc}")
            continue
        sections = parse_prompt_sections(text)
        parsed[trust_level] = sections
        missing = [name for name in _PILOT_REQUIRED_TRUST_SECTIONS if name not in sections]
        if missing:
            problems.append(f"{filename}: missing sections {sorted(missing)}")
        extra = [name for name in sections if name not in _PILOT_REQUIRED_TRUST_SECTIONS]
        if extra:
            problems.append(f"{filename}: unexpected sections {sorted(extra)}")

    levels = sorted(level for level in parsed)
    for section in PILOT_INVARIANT_TRUST_SECTIONS:
        reference: str | None = None
        for level in levels:
            value = parsed[level].get(section)
            if reference is None:
                reference = value
            elif value != reference:
                problems.append(
                    f"pilot section {section} differs between trust prompts "
                    f"({levels[0].value} vs {level.value})"
                )

    framings = {level: parsed[level].get(TRUST_FRAMING_SECTION, "") for level in levels}
    for i, level_a in enumerate(levels):
        for level_b in levels[i + 1 :]:
            if framings[level_a] == framings[level_b]:
                problems.append(
                    f"pilot TRUST_FRAMING identical for {level_a.value} and {level_b.value}"
                )
    for level, framing in framings.items():
        for pattern in _DISCLOSURE_IMPERATIVE_PATTERNS:
            if re.search(pattern, framing, flags=re.IGNORECASE):
                problems.append(
                    f"pilot trust {level.value} framing contains disclosure imperative "
                    f"matching {pattern!r}"
                )
    return problems


# ---------------------------------------------------------------------------
# Prompt resolution (template + target spec -> exact prompt text)
# ---------------------------------------------------------------------------


def _join(values: Sequence[str]) -> str:
    return "; ".join(values)


def resolve_trust_prompt(
    trust_level: str,
    target_spec: EmpiricalTargetSpec,
    *,
    prompt_dir: Path = EMPIRICAL_PROMPTS_DIR,
) -> str:
    template = load_prompt_template(
        TRUST_PROMPT_FILES[TrustLevel(trust_level)], prompt_dir=prompt_dir
    )
    return template.format(forget_id=target_spec.forget_id)


def resolve_attack_prompt(
    attack_type: str,
    target_spec: EmpiricalTargetSpec,
    *,
    sequence_step_index: int | None = None,
    sequence_step_count: int | None = None,
    prompt_dir: Path = EMPIRICAL_PROMPTS_DIR,
) -> str:
    attack = AttackType(attack_type)
    is_sequence = attack in SEQUENCE_ATTACK_TYPES
    if is_sequence and (sequence_step_index is None or sequence_step_count is None):
        raise ValueError(f"sequence attack {attack.value} requires step index and count")
    if not is_sequence and sequence_step_index is not None:
        raise ValueError(f"non-sequence attack {attack.value} cannot carry step metadata")

    template = load_prompt_template(ATTACK_TEMPLATE_FILES[attack], prompt_dir=prompt_dir)
    format_kwargs: dict[str, object] = {
        "canonical_target": target_spec.canonical_target,
        "aliases": _join(target_spec.aliases),
        "permitted_residuals": _join(target_spec.permitted_residuals),
        "fact_chain": _join(target_spec.fact_chain),
        "fragments": _join(target_spec.fragments),
    }
    if sequence_step_index is not None and sequence_step_count is not None:
        format_kwargs["sequence_step_index"] = sequence_step_index + 1
        format_kwargs["sequence_step_count"] = sequence_step_count
    return template.format(**format_kwargs)


def build_user_prompt(
    trust_level: str,
    attack_type: str,
    target_spec: EmpiricalTargetSpec,
    *,
    sender_id: str | None = None,
    recipient_id: str | None = None,
    sequence_step_index: int | None = None,
    sequence_step_count: int | None = None,
    prompt_dir: Path = EMPIRICAL_PROMPTS_DIR,
) -> tuple[str, str, str]:
    """Resolve the user prompt. Returns ``(user_prompt, trust_prompt, attack_prompt)``."""
    trust_prompt = resolve_trust_prompt(trust_level, target_spec, prompt_dir=prompt_dir)
    attack_prompt = resolve_attack_prompt(
        attack_type,
        target_spec,
        sequence_step_index=sequence_step_index,
        sequence_step_count=sequence_step_count,
        prompt_dir=prompt_dir,
    )
    sender = sender_id or target_spec.custodian_agent_id
    recipient = recipient_id or target_spec.default_recipient_id
    context = (
        "[INTERACTION_CONTEXT]\n"
        f"Scenario: {target_spec.scenario_id}\n"
        f"Sender agent: {sender}\n"
        f"Recipient agent: {recipient}"
    )
    user_prompt = f"{trust_prompt}\n\n{attack_prompt}\n\n{context}"
    return user_prompt, trust_prompt, attack_prompt


# ---------------------------------------------------------------------------
# E1-013: prompt hashing and prompt manifest
# ---------------------------------------------------------------------------


def prompt_sha256(text: str) -> str:
    """SHA-256 over the exact UTF-8 bytes of a resolved prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedPromptBundle:
    """One fully resolved prompt instance with all four required hashes."""

    system_prompt: str
    user_prompt: str
    trust_prompt: str
    attack_prompt: str
    system_prompt_sha256: str
    user_prompt_sha256: str
    trust_prompt_sha256: str
    attack_prompt_sha256: str


def resolve_prompt_bundle(
    trust_level: str,
    attack_type: str,
    target_spec: EmpiricalTargetSpec,
    *,
    sender_id: str | None = None,
    recipient_id: str | None = None,
    sequence_step_index: int | None = None,
    sequence_step_count: int | None = None,
    prompt_dir: Path = EMPIRICAL_PROMPTS_DIR,
) -> ResolvedPromptBundle:
    system_prompt = load_system_prompt(prompt_dir=prompt_dir)
    user_prompt, trust_prompt, attack_prompt = build_user_prompt(
        trust_level,
        attack_type,
        target_spec,
        sender_id=sender_id,
        recipient_id=recipient_id,
        sequence_step_index=sequence_step_index,
        sequence_step_count=sequence_step_count,
        prompt_dir=prompt_dir,
    )
    return ResolvedPromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        trust_prompt=trust_prompt,
        attack_prompt=attack_prompt,
        system_prompt_sha256=prompt_sha256(system_prompt),
        user_prompt_sha256=prompt_sha256(user_prompt),
        trust_prompt_sha256=prompt_sha256(trust_prompt),
        attack_prompt_sha256=prompt_sha256(attack_prompt),
    )


def _all_template_files() -> tuple[str, ...]:
    return (
        SYSTEM_PROMPT_FILE,
        *TRUST_PROMPT_FILES.values(),
        *ATTACK_TEMPLATE_FILES.values(),
    )


def build_prompt_manifest(prompt_dir: Path = EMPIRICAL_PROMPTS_DIR) -> dict[str, object]:
    """E1-011/E1-013: source-template prompt manifest.

    Status is ``frozen_post_pilot`` after the E2 prompt freeze.
    """
    templates: dict[str, object] = {}
    for relative_name in _all_template_files():
        raw = (prompt_dir / relative_name).read_bytes()
        templates[relative_name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    invariance_problems = validate_trust_prompt_invariance(prompt_dir)
    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "status": PROMPT_MANIFEST_STATUS,
        "prompt_namespace": "data/trustparadox_u/empirical_v2/prompts",
        "templates": templates,
        "prompt_invariance": {
            "valid": not invariance_problems,
            "problems": invariance_problems,
        },
    }


def prompt_manifest_sha256(manifest: Mapping[str, object]) -> str:
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# E1-008: generator interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmpiricalGenerationRequest:
    """One generation request. Provider-neutral by construction."""

    target_spec: EmpiricalTargetSpec
    trust_level: str
    attack_type: str
    sample_index: int
    generation_replicate: int
    sender_id: str
    recipient_id: str
    system_prompt: str
    user_prompt: str
    temperature: float
    sequence_step_index: int | None = None
    sequence_step_count: int | None = None

    @property
    def scenario_id(self) -> str:
        return self.target_spec.scenario_id

    @property
    def secret_variant_id(self) -> str:
        return self.target_spec.secret_variant_id

    @property
    def split(self) -> str:
        return self.target_spec.split


@dataclass(frozen=True)
class EmpiricalGenerationResponse:
    """One generation outcome — errors are data, not exceptions."""

    raw_text: str | None
    request_id: str | None
    model_id: str
    model_revision: str | None
    status: str
    error_message: str | None = None
    retry_index: int = 0
    generated_at: str = ""
    latency_ms: float | None = None


class EmpiricalCandidateGenerator(Protocol):
    generation_mode: str

    def generate(
        self,
        request: EmpiricalGenerationRequest,
    ) -> EmpiricalGenerationResponse: ...


def build_generation_request(
    target_spec: EmpiricalTargetSpec,
    trust_level: str,
    attack_type: str,
    sample_index: int,
    *,
    generation_replicate: int = 0,
    temperature: float = 0.7,
    sequence_step_index: int | None = None,
    sequence_step_count: int | None = None,
    prompt_dir: Path = EMPIRICAL_PROMPTS_DIR,
) -> EmpiricalGenerationRequest:
    bundle = resolve_prompt_bundle(
        trust_level,
        attack_type,
        target_spec,
        sequence_step_index=sequence_step_index,
        sequence_step_count=sequence_step_count,
        prompt_dir=prompt_dir,
    )
    return EmpiricalGenerationRequest(
        target_spec=target_spec,
        trust_level=trust_level,
        attack_type=attack_type,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
        sender_id=target_spec.custodian_agent_id,
        recipient_id=target_spec.default_recipient_id,
        system_prompt=bundle.system_prompt,
        user_prompt=bundle.user_prompt,
        temperature=temperature,
        sequence_step_index=sequence_step_index,
        sequence_step_count=sequence_step_count,
    )


# ---------------------------------------------------------------------------
# E2 repair §5/§7: primary discretion-based trust pilot builder
# ---------------------------------------------------------------------------

#: E2 repair §5: primary pilot task identifier (discretion-based).
PRIMARY_PILOT_ATTACK_TYPE = "trust_discretion_task"


def build_pilot_user_prompt(
    trust_level: str,
    target_spec: EmpiricalTargetSpec,
    *,
    prompt_dir: Path = PILOT_PROMPTS_DIR,
) -> tuple[str, str]:
    """E2 repair §7: resolve the pilot user prompt from the pilot namespace.

    Returns ``(user_prompt, trust_prompt)``.  The pilot trust prompt
    already contains all required sections (TARGET_CONTEXT,
    FORGET_INSTRUCTION, PRIVACY_PROHIBITION, TASK_OBJECTIVE,
    PILOT_REQUEST, RESPONSE_FORMAT); no separate attack prompt is
    appended — the primary task is discretion-based.
    """
    trust_template = load_prompt_template(
        PILOT_TRUST_PROMPT_FILES[TrustLevel(trust_level)],
        prompt_dir=prompt_dir,
    )
    trust_prompt = trust_template.format(forget_id=target_spec.forget_id)
    user_prompt = trust_prompt
    return user_prompt, trust_prompt


def build_trust_pilot_request(
    target_spec: EmpiricalTargetSpec,
    trust_level: str,
    sample_index: int,
    *,
    generation_replicate: int = 0,
    temperature: float = 0.7,
    prompt_dir: Path = PILOT_PROMPTS_DIR,
) -> EmpiricalGenerationRequest:
    """E2 repair §5/§7: build a primary discretion-based pilot request.

    Uses the pilot/ prompt namespace and the trust_discretion_task
    identifier.  Does NOT route through the generic direct_disclosure
    attack builder.
    """
    system_prompt = load_system_prompt(prompt_dir=prompt_dir)
    user_prompt, trust_prompt = build_pilot_user_prompt(
        trust_level,
        target_spec,
        prompt_dir=prompt_dir,
    )
    return EmpiricalGenerationRequest(
        target_spec=target_spec,
        trust_level=trust_level,
        attack_type=PRIMARY_PILOT_ATTACK_TYPE,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
        sender_id=target_spec.custodian_agent_id,
        recipient_id=target_spec.default_recipient_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )


def build_pilot_prompt_manifest(
    prompt_dir: Path = PILOT_PROMPTS_DIR,
) -> dict[str, object]:
    """E2 repair §7/§13: prompt manifest for the pilot namespace.

    Covers the system prompt, the three pilot trust prompts, and the
    primary task template.
    """
    pilot_files = (
        SYSTEM_PROMPT_FILE,
        *PILOT_TRUST_PROMPT_FILES.values(),
        PRIMARY_TASK_FILE,
    )
    templates: dict[str, object] = {}
    for relative_name in pilot_files:
        raw = (prompt_dir / relative_name).read_bytes()
        templates[relative_name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    invariance_problems = validate_pilot_prompt_invariance(prompt_dir)
    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "status": PROMPT_MANIFEST_STATUS,
        "prompt_namespace": "data/trustparadox_u/empirical_v2/prompts/pilot",
        "templates": templates,
        "prompt_invariance": {
            "valid": not invariance_problems,
            "problems": invariance_problems,
        },
    }


def _status_flags(status: GenerationStatus) -> tuple[bool, bool, bool]:
    return (
        status is GenerationStatus.REFUSAL,
        status is GenerationStatus.MALFORMED,
        status is GenerationStatus.OFF_TOPIC,
    )


def attempt_from_response(
    request: EmpiricalGenerationRequest,
    response: EmpiricalGenerationResponse,
    *,
    generator_provider: str,
    seed: int | None = None,
    generation_mode: str = GenerationMode.MOCK.value,
    transport: str | None = None,
    generator_model_requested: str | None = None,
    max_tokens: int | None = None,
    trust_prompt_hash: str | None = None,
    attack_prompt_hash: str | None = None,
) -> EmpiricalGenerationAttempt:
    """Assemble the retained raw-attempt record from a request/response pair.

    E2-002: provider, transport, generation mode, and the requested vs
    returned model are recorded separately; ``generator_provider`` must
    never be "real".

    E3-005: ``latency_ms``, ``trust_prompt_hash``, ``attack_prompt_hash``,
    and ``max_tokens`` are populated when available.
    """
    spec = request.target_spec
    is_sequence = request.sequence_step_index is not None
    attempt_id = generation_attempt_id(
        scenario_id=spec.scenario_id,
        secret_variant_id=spec.secret_variant_id,
        trust_level=request.trust_level,
        attack_type=request.attack_type,
        sample_index=request.sample_index,
        generation_replicate=request.generation_replicate,
        sequence_step_index=request.sequence_step_index,
    )
    family_id = empirical_candidate_family_id(
        scenario_id=spec.scenario_id,
        secret_variant_id=spec.secret_variant_id,
        attack_type=request.attack_type,
        sample_index=request.sample_index,
        generation_replicate=request.generation_replicate,
        sequence_step_index=request.sequence_step_index,
    )
    sequence_family_id = (
        empirical_sequence_family_id(
            scenario_id=spec.scenario_id,
            secret_variant_id=spec.secret_variant_id,
            attack_type=request.attack_type,
            sample_index=request.sample_index,
            generation_replicate=request.generation_replicate,
        )
        if is_sequence
        else None
    )
    sequence_id = (
        empirical_sequence_id(sequence_family_id, request.trust_level)
        if sequence_family_id is not None
        else None
    )
    refusal, malformed, off_topic = _status_flags(GenerationStatus(response.status))
    prov_attempt_id = _compute_provider_attempt_id(
        scenario_id=spec.scenario_id,
        secret_variant_id=spec.secret_variant_id,
        trust_level=request.trust_level,
        attack_type=request.attack_type,
        sample_index=request.sample_index,
        generation_replicate=request.generation_replicate,
        retry_index=response.retry_index,
        sequence_step_index=request.sequence_step_index,
    )
    return EmpiricalGenerationAttempt(
        generation_attempt_id=attempt_id,
        scenario_id=spec.scenario_id,
        secret_variant_id=spec.secret_variant_id,
        split=spec.split,
        trust_level=request.trust_level,
        attack_type=request.attack_type,
        sample_index=request.sample_index,
        generation_replicate=request.generation_replicate,
        sender_id=request.sender_id,
        recipient_id=request.recipient_id,
        candidate_family_id=family_id,
        sequence_family_id=sequence_family_id,
        sequence_id=sequence_id,
        sequence_step_index=request.sequence_step_index,
        sequence_step_count=request.sequence_step_count,
        candidate_text=response.raw_text,
        generation_status=response.status,
        refusal=refusal,
        malformed=malformed,
        off_topic=off_topic,
        generator_provider=generator_provider,
        generator_model=response.model_id,
        generator_revision=response.model_revision,
        temperature=request.temperature,
        seed=seed,
        system_prompt_hash=prompt_sha256(request.system_prompt),
        user_prompt_hash=prompt_sha256(request.user_prompt),
        request_id=response.request_id,
        retry_index=response.retry_index,
        generated_at=response.generated_at or utc_now_iso(),
        generation_mode=GenerationMode(generation_mode).value,
        transport=transport,
        generator_model_requested=generator_model_requested,
        generator_model_returned=response.model_id,
        latency_ms=response.latency_ms,
        trust_prompt_hash=trust_prompt_hash,
        attack_prompt_hash=attack_prompt_hash,
        max_tokens=max_tokens,
        provider_attempt_id=prov_attempt_id,
    )


# ---------------------------------------------------------------------------
# E1-009: deterministic mock generator (diagnostic only)
# ---------------------------------------------------------------------------


def _mock_default_text(request: EmpiricalGenerationRequest) -> str:
    spec = request.target_spec
    attack = AttackType(request.attack_type)
    if attack is AttackType.DIRECT_DISCLOSURE:
        return f"As agreed between us: {spec.canonical_target}"
    if attack is AttackType.SEMANTIC_PARAPHRASE:
        return "To summarize what we discussed: " + "; ".join(spec.fact_chain) + "."
    if attack is AttackType.ALIAS_OR_COREFERENCE:
        return f"Regarding {spec.aliases[0]}: {spec.canonical_target}"
    if attack is AttackType.RECONTAMINATION:
        return f"Restoring the item that was forgotten: {spec.canonical_target}"
    if attack in SEQUENCE_ATTACK_TYPES:
        step = request.sequence_step_index or 0
        parts = spec.fragments if attack is AttackType.FRAGMENTATION_SEQUENCE else spec.fact_chain
        return parts[min(step, len(parts) - 1)]
    if attack is AttackType.HARD_NEGATIVE_CONTROL:
        return f"Note for the record: {spec.permitted_residuals[0]}"
    return f"Routine progress update on our shared work concerning {spec.aliases[0]}."


class MockEmpiricalGenerator:
    """Deterministic fixture-driven generator.

    Never empirical evidence: ``generation_mode = mock``,
    ``study_class = diagnostic``.
    """

    generation_mode = "mock"
    study_class = "diagnostic"

    def __init__(
        self,
        *,
        fixture_statuses: Mapping[str, str | GenerationStatus] | None = None,
        fixture_texts: Mapping[str, str] | None = None,
    ) -> None:
        self.fixture_statuses: dict[str, GenerationStatus] = {
            attempt_id: GenerationStatus(status)
            for attempt_id, status in (fixture_statuses or {}).items()
        }
        self.fixture_texts: dict[str, str] = dict(fixture_texts or {})

    def _attempt_id(self, request: EmpiricalGenerationRequest) -> str:
        return generation_attempt_id(
            scenario_id=request.target_spec.scenario_id,
            secret_variant_id=request.target_spec.secret_variant_id,
            trust_level=request.trust_level,
            attack_type=request.attack_type,
            sample_index=request.sample_index,
            generation_replicate=request.generation_replicate,
            sequence_step_index=request.sequence_step_index,
        )

    def generate(
        self,
        request: EmpiricalGenerationRequest,
    ) -> EmpiricalGenerationResponse:
        attempt_id = self._attempt_id(request)
        status = self.fixture_statuses.get(attempt_id, GenerationStatus.SUCCESS)
        error_message: str | None = None
        if status is GenerationStatus.SUCCESS:
            raw_text: str | None = self.fixture_texts.get(attempt_id) or _mock_default_text(request)
        elif status is GenerationStatus.MALFORMED:
            raw_text = self.fixture_texts.get(attempt_id, "%%MALFORMED {unterminated")
        elif status is GenerationStatus.PROVIDER_ERROR:
            raw_text = self.fixture_texts.get(attempt_id)
            error_message = "mock provider error"
        else:
            raw_text = self.fixture_texts.get(attempt_id)
        return EmpiricalGenerationResponse(
            raw_text=raw_text,
            request_id=f"mock-{attempt_id}",
            model_id=_MOCK_GENERATOR_MODEL,
            model_revision=None,
            status=status.value,
            error_message=error_message,
            retry_index=0,
            generated_at=_MOCK_TIMESTAMP,
            latency_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Patch I: timeout classification
# ---------------------------------------------------------------------------


def classify_generation_exception(exc: Exception) -> GenerationStatus:
    """Classify an exception into a :class:`GenerationStatus` (Patch I).

    At minimum:
    - ``TimeoutError`` (and LiteLLM timeout wrappers) → ``TIMEOUT``
    - Generic provider / HTTP failure → ``PROVIDER_ERROR``

    Both statuses are retained as raw attempts and are retryable according
    to the frozen retry policy.
    """
    # Python built-in timeout.
    if isinstance(exc, TimeoutError):
        return GenerationStatus.TIMEOUT

    # LiteLLM wraps some provider timeouts in its own exception types.
    # Check the class name hierarchy for timeout markers.
    exc_type = type(exc)
    for cls in exc_type.__mro__:
        name = cls.__name__.lower()
        if "timeout" in name:
            return GenerationStatus.TIMEOUT

    # Fall back to string heuristics for wrapped / opaque exceptions.
    text = str(exc).lower()
    if ("timeout" in text or "timed out" in text) and "rate" not in text:
        return GenerationStatus.TIMEOUT

    return GenerationStatus.PROVIDER_ERROR


# ---------------------------------------------------------------------------
# E1-010: real generator adapter
# ---------------------------------------------------------------------------


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("timeout", "rate limit", "rate_limit", "429", "500", "502", "503", "504")
    )


@dataclass
class RealEmpiricalGenerator:
    """Configuration-driven real-LLM adapter (LiteLLM transport).

    E1-010 guarantees: no hidden fallback to the mock generator, no silent
    model substitution (a returned model that does not match the configured
    model is recorded as a provider_error attempt), provider errors remain
    raw attempts, and each retry is traceable via ``retry_index``.
    """

    provider: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 512
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    transport: str = "litellm"

    generation_mode: str = field(default=GenerationMode.REAL.value, init=False)

    @classmethod
    def from_frozen_config(
        cls,
        config: "FrozenGenerationConfig",
        *,
        api_base: str | None = None,
        api_key_env: str | None = None,
    ) -> "RealEmpiricalGenerator":
        """Construct from a :class:`FrozenGenerationConfig` (Patch B).

        The frozen config is the single source of truth for actual API
        request parameters during the full-corpus generation campaign.
        """
        return cls(
            provider=config.generator_provider,
            model_name=config.generator_model_requested,
            temperature=config.generator_temperature,
            max_tokens=config.generator_max_tokens,
            timeout_seconds=config.request_timeout,
            max_retries=config.max_retries,
            api_base=api_base,
            api_key_env=api_key_env,
        )

    def generate_once(
        self,
        request: EmpiricalGenerationRequest,
    ) -> EmpiricalGenerationResponse:
        """Perform exactly one API call with no hidden retry loop (Patch C).

        Transient errors are caught and returned as ``PROVIDER_ERROR``
        responses so the campaign layer can retain every provider attempt
        in raw provenance.  Non-transient errors also produce a
        ``PROVIDER_ERROR`` response — this method never raises.
        """
        from litellm import completion

        api_key: str | None = None
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]

        try:
            # When using a custom OpenAI-compatible endpoint (api_base set),
            # litellm requires the model name to be prefixed with the provider
            # (e.g., "openai/qwen3.7-plus").
            litellm_model = self.model_name
            if self.api_base and "/" not in litellm_model:
                litellm_model = f"{self.provider}/{litellm_model}"

            kwargs: dict[str, object] = {
                "model": litellm_model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout_seconds,
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if api_key:
                kwargs["api_key"] = api_key

            start = time.monotonic()
            response = completion(**kwargs)  # type: ignore[arg-type]
            elapsed_ms = (time.monotonic() - start) * 1000.0

            returned_model = str(getattr(response, "model", "") or "")
            requested_model_name = self.model_name.split("/")[-1]
            returned_model_name = returned_model.split("/")[-1]
            if returned_model and requested_model_name not in returned_model_name:
                return EmpiricalGenerationResponse(
                    raw_text=None,
                    request_id=str(getattr(response, "id", "") or "") or None,
                    model_id=returned_model,
                    model_revision=None,
                    status=GenerationStatus.PROVIDER_ERROR.value,
                    error_message=(
                        "provider/model mismatch: requested "
                        f"{self.model_name!r}, received {returned_model!r}"
                    ),
                    retry_index=0,
                    generated_at=utc_now_iso(),
                    latency_ms=elapsed_ms,
                )

            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ValueError("provider returned empty response")

            revision = str(getattr(response, "system_fingerprint", "") or "") or None
            logger.debug("Empirical generation completed in %.1f ms", elapsed_ms)
            return EmpiricalGenerationResponse(
                raw_text=str(text),
                request_id=str(getattr(response, "id", "") or "") or None,
                model_id=returned_model or self.model_name,
                model_revision=revision,
                status=GenerationStatus.SUCCESS.value,
                retry_index=0,
                generated_at=utc_now_iso(),
                latency_ms=elapsed_ms,
            )
        except Exception as exc:
            status = classify_generation_exception(exc)
            return EmpiricalGenerationResponse(
                raw_text=None,
                request_id=None,
                model_id=self.model_name,
                model_revision=None,
                status=status.value,
                error_message=f"{type(exc).__name__}: {exc}",
                retry_index=0,
                generated_at=utc_now_iso(),
                latency_ms=None,
            )

    def generate(
        self,
        request: EmpiricalGenerationRequest,
    ) -> EmpiricalGenerationResponse:
        """Legacy hidden-retry generate (QUARANTINED — Patch H).

        .. deprecated::
            This method contains a hidden retry loop that is NOT traceable
            in raw provenance.  The full Phase-3 campaign MUST use
            :meth:`generate_once` exclusively.  This method is retained
            only for backward compatibility and diagnostic use.
        """
        from litellm import completion

        api_key: str | None = None
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]

        last_exc: Exception | None = None
        retry_index = 0
        for attempt in range(1 + self.max_retries):
            retry_index = attempt
            if attempt > 0:
                logger.info(
                    "Empirical generator retry %d/%d for model %s",
                    attempt,
                    self.max_retries,
                    self.model_name,
                )
            try:
                # When using a custom OpenAI-compatible endpoint (api_base set),
                # litellm requires the model name to be prefixed with the provider.
                litellm_model = self.model_name
                if self.api_base and "/" not in litellm_model:
                    litellm_model = f"{self.provider}/{litellm_model}"

                kwargs: dict[str, object] = {
                    "model": litellm_model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout_seconds,
                }
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                if api_key:
                    kwargs["api_key"] = api_key

                start = time.monotonic()
                response = completion(**kwargs)  # type: ignore[arg-type]
                elapsed_ms = (time.monotonic() - start) * 1000.0

                returned_model = str(getattr(response, "model", "") or "")
                # Normalize model names for comparison: LiteLLM uses "provider/model"
                # format but APIs may return just the model name.
                requested_model_name = self.model_name.split("/")[-1]
                returned_model_name = returned_model.split("/")[-1]
                if returned_model and requested_model_name not in returned_model_name:
                    # No silent model substitution: mismatch is a raw failure.
                    return EmpiricalGenerationResponse(
                        raw_text=None,
                        request_id=str(getattr(response, "id", "") or "") or None,
                        model_id=returned_model,
                        model_revision=None,
                        status=GenerationStatus.PROVIDER_ERROR.value,
                        error_message=(
                            "provider/model mismatch: requested "
                            f"{self.model_name!r}, received {returned_model!r}"
                        ),
                        retry_index=retry_index,
                        generated_at=utc_now_iso(),
                        latency_ms=elapsed_ms,
                    )

                text = response.choices[0].message.content
                if not text or not text.strip():
                    raise ValueError("provider returned empty response")

                revision = str(getattr(response, "system_fingerprint", "") or "") or None
                logger.debug("Empirical generation completed in %.1f ms", elapsed_ms)
                return EmpiricalGenerationResponse(
                    raw_text=str(text),
                    request_id=str(getattr(response, "id", "") or "") or None,
                    model_id=returned_model or self.model_name,
                    model_revision=revision,
                    status=GenerationStatus.SUCCESS.value,
                    retry_index=retry_index,
                    generated_at=utc_now_iso(),
                    latency_ms=elapsed_ms,
                )
            except Exception as exc:
                last_exc = exc
                if not _is_transient_error(exc) or attempt >= self.max_retries:
                    break

        return EmpiricalGenerationResponse(
            raw_text=None,
            request_id=None,
            model_id=self.model_name,
            model_revision=None,
            status=GenerationStatus.PROVIDER_ERROR.value,
            error_message=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown",
            retry_index=retry_index,
            generated_at=utc_now_iso(),
            latency_ms=None,
        )


# ---------------------------------------------------------------------------
# E1-014: raw generation-attempt writer
# ---------------------------------------------------------------------------


class RawAttemptWriter:
    """Append-only writer for ``raw_generation_attempts.jsonl``.

    Preserves every attempt (success, refusal, malformed, provider error),
    flushes after each record, and rejects duplicate provider-attempt IDs —
    including IDs already present in a pre-existing file.

    Patch C: uniqueness is enforced on ``provider_attempt_id`` (unique per
    provider call) rather than ``generation_attempt_id`` (shared across
    retries for the same scientific unit).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    # Patch C: prefer provider_attempt_id; fall back to
                    # generation_attempt_id for backward compatibility with
                    # pre-patch records.
                    attempt_id = record.get("provider_attempt_id") or record.get(
                        "generation_attempt_id"
                    )
                    if not isinstance(attempt_id, str) or not attempt_id:
                        raise ValueError(
                            f"{self.path} line {line_number}: missing provider_attempt_id"
                        )
                    if attempt_id in self._seen:
                        raise ValueError(
                            f"{self.path} line {line_number}: duplicate "
                            f"provider_attempt_id {attempt_id!r}"
                        )
                    self._seen.add(attempt_id)
        self._attempt_count = 0

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    def write_attempt(self, attempt: EmpiricalGenerationAttempt) -> None:
        # Patch C: use provider_attempt_id when available, fall back to
        # generation_attempt_id for backward compatibility.
        dedup_key = attempt.provider_attempt_id or attempt.generation_attempt_id
        if dedup_key in self._seen:
            raise ValueError(f"duplicate provider_attempt_id {dedup_key!r}")
        record = attempt_to_record(attempt)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._seen.add(dedup_key)
        self._attempt_count += 1

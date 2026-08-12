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
from experiments.trustparadox_u.empirical_generation import (
    EmpiricalCandidateGenerator,
    MockEmpiricalGenerator,
    RawAttemptWriter,
    RealEmpiricalGenerator,
    attempt_from_response,
    build_generation_request,
    build_prompt_manifest,
    prompt_manifest_sha256,
    utc_now_iso,
    validate_trust_prompt_invariance,
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
) -> dict[str, object]:
    status_counts = Counter(attempt.generation_status for attempt in attempts)
    return {
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
    """Group sequence attempts and apply the E1-017 structural validator."""
    groups: dict[tuple[str, str], list[EmpiricalGenerationAttempt]] = {}
    for attempt in attempts:
        if attempt.is_sequence_attempt and attempt.sequence_family_id is not None:
            key = (attempt.sequence_family_id, attempt.trust_level)
            groups.setdefault(key, []).append(attempt)
    problems: list[str] = []
    for (family_id, trust_level), steps in sorted(groups.items()):
        ordered = sorted(steps, key=lambda a: a.sequence_step_index or 0)
        for problem in validate_sequence_structure(ordered):
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
) -> dict[str, object]:
    """Generate, retain, and accept empirical attempts for one split.

    E1-021: raises ``EmpiricalPhaseLockedError`` for validation/test while
    the empirical phase is E1.
    """
    assert_generation_split_unlocked(split)
    EmpiricalSplit(split)  # validates the split value

    attempts: list[EmpiricalGenerationAttempt] = []
    accepted: list[EmpiricalCandidate] = []
    rejection_reasons: list[str] = []
    duplicate_id_count = 0

    raw_writer = RawAttemptWriter(output_dir / RAW_ATTEMPTS_FILENAME)
    target_specs = specs_for_split(split, scenarios)
    for spec in target_specs:
        for trust_level in trust_levels:
            TrustLevel(trust_level)
            for attack_type in attack_types:
                for sample_index in range(samples):
                    attempts.extend(
                        _generate_unit(
                            generator=generator,
                            spec=spec,
                            trust_level=trust_level,
                            attack_type=attack_type,
                            sample_index=sample_index,
                            temperature=temperature,
                            raw_writer=raw_writer,
                            accepted=accepted,
                            rejection_reasons=rejection_reasons,
                        )
                    )

    accepted_sorted = sorted(accepted, key=lambda c: c.candidate_id)
    _write_jsonl(
        output_dir / ACCEPTED_CANDIDATES_FILENAME,
        [candidate_to_record(candidate) for candidate in accepted_sorted],
    )

    prompt_manifest = build_prompt_manifest()
    _write_json(output_dir / PROMPT_MANIFEST_FILENAME, prompt_manifest)

    manifest = build_corpus_manifest(
        generation_mode=generator.generation_mode,
        attempts=attempts,
        accepted=accepted_sorted,
        prompt_manifest=prompt_manifest,
        artifact_class=artifact_class,
        research_use=research_use,
    )
    _write_json(output_dir / CORPUS_MANIFEST_FILENAME, manifest)

    report = build_validation_report(
        attempts=attempts,
        accepted=accepted_sorted,
        rejection_reasons=rejection_reasons,
        duplicate_id_count=duplicate_id_count,
        manifest=manifest,
        output_dir=output_dir,
        phase_lock_valid=True,
    )
    _write_json(output_dir / VALIDATION_REPORT_FILENAME, report)
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
) -> list[EmpiricalGenerationAttempt]:
    """Generate one trial unit (one attempt, or one per sequence step)."""
    is_sequence = AttackType(attack_type) in SEQUENCE_ATTACK_TYPES
    step_count = sequence_step_count_for(attack_type, spec) if is_sequence else None
    steps = range(step_count) if step_count is not None else (None,)

    unit_attempts: list[EmpiricalGenerationAttempt] = []
    for step_index in steps:
        request = build_generation_request(
            spec,
            trust_level,
            attack_type,
            sample_index,
            temperature=temperature,
            sequence_step_index=step_index,
            sequence_step_count=step_count,
        )
        response = generator.generate(request)
        # E2-002: provider/transport/mode are recorded separately; the
        # provider is never the generation mode ("mock"/"real").
        attempt = attempt_from_response(
            request,
            response,
            generator_provider=getattr(generator, "provider", GenerationMode.MOCK.value),
            generation_mode=GenerationMode(generator.generation_mode).value,
            transport=getattr(generator, "transport", None),
            generator_model_requested=getattr(generator, "model_name", response.model_id),
        )
        raw_writer.write_attempt(attempt)
        unit_attempts.append(attempt)

        result = accept_generation_attempt(attempt, spec)
        if result.accepted and result.candidate is not None:
            accepted.append(result.candidate)
        else:
            rejection_reasons.append(result.reason)
    return unit_attempts


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development-only empirical corpus generation (E1)."
    )
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key-env", default=None)
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

    generator: EmpiricalCandidateGenerator
    if args.mode == "mock":
        generator = MockEmpiricalGenerator()
    else:
        if not args.generator_model:
            print("--generator-model is required for --mode real", file=sys.stderr)
            return 2
        generator = RealEmpiricalGenerator(
            provider=args.provider,
            model_name=args.generator_model,
            temperature=args.temperature,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
        )

    report = run_generation(
        split=args.split,
        mode=args.mode,
        scenarios=sorted(set(scenarios)),
        trust_levels=trust_levels,
        attack_types=attack_types,
        samples=args.samples,
        output_dir=args.output_dir,
        generator=generator,
        temperature=args.temperature,
    )
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

"""E2-D: trust manipulation pilot runner.

This module executes the 90-attempt trust manipulation pilot:
- 10 families (sample indices 0-9)
- 3 trust levels (low, default, high)
- 3 scenarios (credential_001, private_attribute_001, authorization_001)
- 1 attack type (direct_disclosure)
- 1 secret variant per scenario (v1)

Total: 10 × 3 × 3 = 90 attempts.

The pilot uses the E2_PILOT_V1 manipulation prompts and records all
raw generation attempts with full provenance. The pilot dataset is
used for E2-E (labeling) and E2-F (analysis).

Checklist coverage:
- E2-007: 90-attempt pilot run
- E2-008: pilot prompt invariance (only TRUST_FRAMING differs)
- E2-009: raw retention for all attempts
- E2-010: provenance completeness
- artifact_class=pilot, research_use=pilot_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PHASE,
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    AttackType,
    EmpiricalCleanTreeRequiredError,
    EmpiricalGenerationAttempt,
    GenerationMode,
    TrustLevel,
    assert_clean_repository_tree,
    get_target_spec,
    raw_attempts_scientific_hash,
)
from experiments.trustparadox_u.empirical_generation import (
    RealEmpiricalGenerator,
    attempt_from_response,
    build_generation_request,
    build_prompt_manifest,
    prompt_manifest_sha256,
    utc_now_iso,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pilot design: 3 scenarios × 3 trust levels × 10 samples = 90 attempts.
PILOT_SCENARIOS: tuple[str, ...] = (
    "credential_v1",
    "private_attribute_v1",
    "authorization_v1",
)
PILOT_TRUST_LEVELS: tuple[str, ...] = (
    TrustLevel.LOW.value,
    TrustLevel.DEFAULT.value,
    TrustLevel.HIGH.value,
)
PILOT_NUM_SAMPLES: int = 10
PILOT_ATTACK_TYPE: str = AttackType.DIRECT_DISCLOSURE.value

RAW_ATTEMPTS_FILENAME = "raw_generation_attempts.jsonl"
VALIDATION_REPORT_FILENAME = "validation_report.json"
MANIFEST_FILENAME = "manifest.json"


def run_trust_pilot(
    *,
    generator_model: str,
    provider: str,
    output_dir: Path,
    transport: str = "litellm",
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: float = 60.0,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> dict:
    """Run 90-attempt trust pilot and return validation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # E2-004: clean tree gate before any real API call.
    try:
        repository_commit = assert_clean_repository_tree()
    except EmpiricalCleanTreeRequiredError as exc:
        return {
            "passed": False,
            "num_findings": 1,
            "error": f"dirty tree: {exc}",
            "findings": [f"dirty tree: {exc}"],
            "num_attempts": 0,
            "expected_attempts": 90,
            "raw_generation_sha256": None,
        }

    # Initialize generator.
    generator = RealEmpiricalGenerator(
        provider=provider,
        model_name=generator_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        transport=transport,
        api_base=api_base,
        api_key_env=api_key_env,
    )

    # Execute pilot: 3 scenarios × 3 trust × 10 samples = 90 attempts.
    attempts: list[EmpiricalGenerationAttempt] = []
    raw_path = output_dir / RAW_ATTEMPTS_FILENAME
    start_time = time.monotonic()

    for scenario_id in PILOT_SCENARIOS:
        spec = get_target_spec(scenario_id)
        for trust_level in PILOT_TRUST_LEVELS:
            for sample_index in range(PILOT_NUM_SAMPLES):
                request = build_generation_request(
                    spec,
                    trust_level,
                    PILOT_ATTACK_TYPE,
                    sample_index=sample_index,
                )
                response = generator.generate(request)
                attempt = attempt_from_response(
                    request,
                    response,
                    generator_provider=provider,
                    generation_mode=GenerationMode.REAL.value,
                    transport=transport,
                    generator_model_requested=generator_model,
                )
                attempts.append(attempt)

                # Write raw attempt immediately.
                record = {
                    field: getattr(attempt, field)
                    for field in EmpiricalGenerationAttempt.__dataclass_fields__
                }
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Progress indicator.
                elapsed = time.monotonic() - start_time
                print(
                    f"  [{len(attempts)}/90] {scenario_id} / {trust_level} / "
                    f"sample={sample_index} -> {response.status} ({elapsed:.1f}s)",
                    flush=True,
                )

    total_time = time.monotonic() - start_time

    # Validation.
    findings: list[str] = []
    for attempt in attempts:
        problems = attempt.validate()
        findings.extend(problems)

    # Check provenance completeness.
    for attempt in attempts:
        if attempt.generator_provider == "real":
            findings.append("provider must not be 'real'")
        if attempt.generation_mode != GenerationMode.REAL.value:
            findings.append(f"generation_mode mismatch: {attempt.generation_mode}")
        if attempt.transport is None:
            findings.append("transport is None")
        if attempt.generator_model_requested is None:
            findings.append("generator_model_requested is None")
        if attempt.generator_model_returned is None:
            findings.append("generator_model_returned is None")

    # Check attempt count.
    if len(attempts) != 90:
        findings.append(f"expected 90 attempts, got {len(attempts)}")

    # Compute scientific hash.
    raw_sha256 = raw_attempts_scientific_hash(attempts)

    # Build prompt manifest.
    prompt_manifest = build_prompt_manifest()
    prompt_sha256 = prompt_manifest_sha256(prompt_manifest)

    # Count successes and failures.
    num_success = sum(1 for a in attempts if a.generation_status == "success")
    num_provider_error = sum(1 for a in attempts if a.generation_status == "provider_error")

    # Write validation report.
    validation_report = {
        "passed": len(findings) == 0,
        "num_findings": len(findings),
        "findings": findings,
        "num_attempts": len(attempts),
        "expected_attempts": 90,
        "num_success": num_success,
        "num_provider_error": num_provider_error,
        "scenarios": list(PILOT_SCENARIOS),
        "trust_levels": list(PILOT_TRUST_LEVELS),
        "attack_type": PILOT_ATTACK_TYPE,
        "num_samples": PILOT_NUM_SAMPLES,
        "total_time_seconds": round(total_time, 2),
        "raw_generation_sha256": raw_sha256,
    }
    (output_dir / VALIDATION_REPORT_FILENAME).write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write manifest.
    manifest = {
        "artifact_class": "pilot",
        "research_use": "pilot_analysis",
        "study_version": EMPIRICAL_STUDY_VERSION,
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "empirical_phase": EMPIRICAL_PHASE,
        "repository_commit": repository_commit,
        "generator_provider": provider,
        "generator_model": generator_model,
        "generator_transport": transport,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_attempts": len(attempts),
        "expected_attempts": 90,
        "scenarios": list(PILOT_SCENARIOS),
        "trust_levels": list(PILOT_TRUST_LEVELS),
        "attack_type": PILOT_ATTACK_TYPE,
        "num_samples": PILOT_NUM_SAMPLES,
        "raw_generation_sha256": raw_sha256,
        "prompt_manifest_sha256": prompt_sha256,
        "generated_at": utc_now_iso(),
        "total_time_seconds": round(total_time, 2),
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-D: trust manipulation pilot")
    parser.add_argument(
        "--generator-model",
        required=True,
        help="Model name for real generation (e.g., openai/qwen-plus)",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="Generator provider name (default: openai)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/empirical_v2/e2_trust_pilot"),
        help="Output directory for pilot artifacts",
    )
    parser.add_argument(
        "--transport",
        default="litellm",
        help="Transport layer (default: litellm)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per generation (default: 512)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for custom endpoints",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable name containing the API key",
    )
    args = parser.parse_args()

    print("E2-D: Trust Manipulation Pilot")
    print(f"  Model: {args.generator_model}")
    print(f"  Provider: {args.provider}")
    print(f"  Output: {args.output_dir}")
    print("  Expected: 90 attempts (3 scenarios × 3 trust × 10 samples)")
    print()

    report = run_trust_pilot(
        generator_model=args.generator_model,
        provider=args.provider,
        output_dir=args.output_dir,
        transport=args.transport,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
    )

    print()
    if report["passed"]:
        print(f"PILOT COMPLETED: {report['num_attempts']}/{report['expected_attempts']} attempts")
        print(
            f"  Success: {report['num_success']}, Provider errors: {report['num_provider_error']}"
        )
        print(f"  Time: {report['total_time_seconds']:.1f}s")
        print(f"  raw_generation_sha256: {report['raw_generation_sha256']}")
        sys.exit(0)
    else:
        print(f"PILOT FAILED: {report['num_findings']} findings")
        for finding in report["findings"]:
            print(f"  - {finding}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""E2 repair §15-17: primary trust pilot runner (90 attempts, randomized).

This module executes the 90-attempt primary discretion-based trust pilot:
- 3 scenarios × 10 generation families × 3 trust levels = 90 requests
- Randomized execution order (deterministic seed)
- Preserves all outcomes (no selective regeneration)
- Uses the frozen pilot configuration

Checklist coverage:
- §15: 90-attempt primary pilot with required artifacts
- §16: preserve all outcomes (success/refusal/malformed/off_topic/error/timeout/retry)
- §17: no selective regeneration (only technical retry policy)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PHASE,
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    EmpiricalCleanTreeRequiredError,
    EmpiricalGenerationAttempt,
    GenerationMode,
    GenerationStatus,
    assert_clean_repository_tree,
    get_target_spec,
    raw_attempts_scientific_hash,
)
from experiments.trustparadox_u.empirical_generation import (
    RealEmpiricalGenerator,
    attempt_from_response,
    build_pilot_prompt_manifest,
    build_trust_pilot_request,
    prompt_manifest_sha256,
    utc_now_iso,
)
from experiments.trustparadox_u.empirical_pilot_config import (
    PrimaryPilotConfig,
    build_request_schedule,
    load_pilot_config,
    randomize_schedule,
    save_request_schedule,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: E2 repair §15: output directory for primary pilot artifacts.
PRIMARY_PILOT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_trust_pilot"

RAW_ATTEMPTS_FILENAME = "raw_generation_attempts.jsonl"
REQUEST_SCHEDULE_FILENAME = "request_schedule.json"
PILOT_MANIFEST_FILENAME = "pilot_manifest.json"
VALIDATION_REPORT_FILENAME = "validation_report.json"


def run_primary_pilot(
    *,
    config: PrimaryPilotConfig,
    output_dir: Path,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> dict:
    """E2 repair §15-17: execute the 90-attempt primary trust pilot.

    Returns a validation report dict.
    """
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
            "num_planned_requests": 0,
            "num_actual_attempts": 0,
            "num_successful_responses": 0,
            "num_eligible_responses": 0,
            "raw_generation_sha256": None,
        }

    # Build and randomize the request schedule.
    schedule = build_request_schedule(config)
    randomized_schedule = randomize_schedule(schedule, config.randomization_seed)

    # Save the request schedule.
    save_request_schedule(randomized_schedule, output_dir / REQUEST_SCHEDULE_FILENAME)

    # Initialize the real generator.
    generator = RealEmpiricalGenerator(
        provider=config.provider,
        model_name=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout,
        transport=config.transport,
        api_base=api_base,
        api_key_env=api_key_env,
    )

    # Execute each request in the randomized schedule.
    attempts: list[EmpiricalGenerationAttempt] = []
    raw_path = output_dir / RAW_ATTEMPTS_FILENAME

    for pilot_request in randomized_schedule:
        spec = get_target_spec(pilot_request.secret_variant_id)
        request = build_trust_pilot_request(
            spec,
            pilot_request.trust_level,
            sample_index=pilot_request.sample_index,
            generation_replicate=pilot_request.generation_replicate,
            temperature=config.temperature,
        )

        # Execute with retry policy.
        max_retries = config.retry_policy.get("max_retries", 3)
        retry_index = 0
        response = None

        while retry_index <= max_retries:
            try:
                response = generator.generate(request)
                break
            except Exception:
                retry_index += 1
                if retry_index > max_retries:
                    # Create a provider error response.
                    from experiments.trustparadox_u.empirical_generation import (
                        EmpiricalGenerationResponse,
                    )

                    response = EmpiricalGenerationResponse(
                        raw_text=None,
                        request_id=f"error-{retry_index}",
                        model_id=config.model,
                        model_revision=None,
                        status=GenerationStatus.PROVIDER_ERROR.value,
                        error_message="max retries exceeded",
                        retry_index=retry_index,
                        generated_at=utc_now_iso(),
                    )
                    break

        assert response is not None, "response must be set after retry loop"
        attempt = attempt_from_response(
            request,
            response,
            generator_provider=config.provider,
            generation_mode=GenerationMode.REAL.value,
            transport=config.transport,
            generator_model_requested=config.model,
        )
        attempts.append(attempt)

        # Write raw attempt immediately.
        record = {
            field: getattr(attempt, field)
            for field in EmpiricalGenerationAttempt.__dataclass_fields__
        }
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

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

    # Compute statistics.
    num_successful = sum(
        1 for a in attempts if a.generation_status == GenerationStatus.SUCCESS.value
    )
    num_eligible = sum(
        1
        for a in attempts
        if a.generation_status == GenerationStatus.SUCCESS.value
        and not a.refusal
        and not a.malformed
        and not a.off_topic
    )

    # Compute scientific hash.
    raw_sha256 = raw_attempts_scientific_hash(attempts)

    # Build pilot prompt manifest.
    pilot_prompt_manifest = build_pilot_prompt_manifest()
    prompt_sha256 = prompt_manifest_sha256(pilot_prompt_manifest)

    # Write pilot manifest.
    pilot_manifest = {
        "artifact_class": "primary_trust_pilot",
        "research_use": "primary_evidence",
        "study_version": EMPIRICAL_STUDY_VERSION,
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "empirical_phase": EMPIRICAL_PHASE,
        "repository_commit": repository_commit,
        "generator_provider": config.provider,
        "generator_model": config.model,
        "generator_transport": config.transport,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "retry_policy": config.retry_policy,
        "pilot_prompt_version": config.pilot_prompt_version,
        "randomization_seed": config.randomization_seed,
        "num_planned_requests": len(randomized_schedule),
        "num_actual_attempts": len(attempts),
        "num_successful_responses": num_successful,
        "num_eligible_responses": num_eligible,
        "raw_generation_sha256": raw_sha256,
        "pilot_prompt_manifest_sha256": prompt_sha256,
        "config_sha256": config.config_sha256(),
        "generated_at": utc_now_iso(),
    }
    (output_dir / PILOT_MANIFEST_FILENAME).write_text(
        json.dumps(pilot_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write validation report.
    validation_report = {
        "passed": len(findings) == 0,
        "num_findings": len(findings),
        "findings": findings,
        "num_planned_requests": len(randomized_schedule),
        "num_actual_attempts": len(attempts),
        "num_successful_responses": num_successful,
        "num_eligible_responses": num_eligible,
        "scenarios": list(config.development_target_variants),
        "trust_levels": list(config.trust_levels),
        "attack_type": "trust_discretion_task",
        "raw_generation_sha256": raw_sha256,
    }
    (output_dir / VALIDATION_REPORT_FILENAME).write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 primary trust pilot runner")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to pilot configuration file (default: auto-create)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PRIMARY_PILOT_OUTPUT_DIR,
        help="Output directory for pilot artifacts",
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

    # Load or create pilot config.
    if args.config_path:
        config = load_pilot_config(args.config_path)
    else:
        from experiments.trustparadox_u.empirical_pilot_config import (
            create_default_pilot_config,
            save_pilot_config,
        )

        config = create_default_pilot_config()
        # Save config into the output directory (gitignored) to avoid
        # dirtying the tree before the clean-tree gate.
        save_pilot_config(config, path=args.output_dir / "pilot_config.json")

    report = run_primary_pilot(
        config=config,
        output_dir=args.output_dir,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
    )

    if report["passed"]:
        print(f"PRIMARY PILOT PASSED: {report['num_actual_attempts']} attempts")
        print(f"  planned requests: {report['num_planned_requests']}")
        print(f"  successful responses: {report['num_successful_responses']}")
        print(f"  eligible responses: {report['num_eligible_responses']}")
        print(f"  raw_generation_sha256: {report['raw_generation_sha256']}")
        sys.exit(0)
    else:
        print(f"PRIMARY PILOT FAILED: {report['num_findings']} findings")
        for finding in report["findings"]:
            print(f"  - {finding}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""E2-B: connectivity smoke test for real empirical generation.

This module runs 3 real API calls (one per canonical scenario) to verify
that the provider is accessible, the model is available, and all provenance
fields can be recorded correctly. The smoke test is diagnostic-only and
does not contribute to the pilot dataset.

Checklist coverage:
- E2-006: 3-call connectivity smoke (3 scenarios × default trust × 1 neutral
  pilot prompt; credential_v1/private_attribute_v1/authorization_v1 only)
- artifact_class=connectivity_smoke, research_use=diagnostic_only
- Validates API access/provider/requested+returned model/request ID/timestamps/
  prompt hashes/raw logging/retry/failure retention
- Hard stop if provenance wrong
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PHASE,
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    AttackType,
    EmpiricalCleanTreeRequiredError,
    EmpiricalGenerationAttempt,
    EmpiricalSplit,
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

# E2-006: exactly these 3 scenarios for connectivity smoke.
CONNECTIVITY_SCENARIOS: tuple[str, ...] = (
    "credential_v1",
    "private_attribute_v1",
    "authorization_v1",
)

RAW_ATTEMPTS_FILENAME = "raw_generation_attempts.jsonl"
VALIDATION_REPORT_FILENAME = "validation_report.json"
MANIFEST_FILENAME = "manifest.json"


def run_connectivity_smoke(
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
    """Run 3-call connectivity smoke and return validation report."""
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

    # E2-006: 3 scenarios × default trust × 1 neutral attack.
    attempts: list[EmpiricalGenerationAttempt] = []
    raw_path = output_dir / RAW_ATTEMPTS_FILENAME

    for scenario_id in CONNECTIVITY_SCENARIOS:
        spec = get_target_spec(scenario_id)
        request = build_generation_request(
            spec,
            TrustLevel.DEFAULT.value,
            AttackType.DIRECT_DISCLOSURE.value,
            sample_index=0,
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

    # Compute scientific hash.
    raw_sha256 = raw_attempts_scientific_hash(attempts)

    # Build prompt manifest.
    prompt_manifest = build_prompt_manifest()
    prompt_sha256 = prompt_manifest_sha256(prompt_manifest)

    # Write validation report.
    validation_report = {
        "passed": len(findings) == 0,
        "num_findings": len(findings),
        "findings": findings,
        "num_attempts": len(attempts),
        "scenarios": list(CONNECTIVITY_SCENARIOS),
        "trust_level": TrustLevel.DEFAULT.value,
        "attack_type": AttackType.DIRECT_DISCLOSURE.value,
        "raw_generation_sha256": raw_sha256,
    }
    (output_dir / VALIDATION_REPORT_FILENAME).write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write manifest.
    manifest = {
        "artifact_class": "connectivity_smoke",
        "research_use": "diagnostic_only",
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
        "scenarios": list(CONNECTIVITY_SCENARIOS),
        "raw_generation_sha256": raw_sha256,
        "prompt_manifest_sha256": prompt_sha256,
        "generated_at": utc_now_iso(),
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-B: connectivity smoke test")
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
        default=Path("results/empirical_v2/e2_connectivity_smoke"),
        help="Output directory for smoke artifacts",
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

    report = run_connectivity_smoke(
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

    if report["passed"]:
        print(f"CONNECTIVITY SMOKE PASSED: {report['num_attempts']} attempts")
        print(f"  raw_generation_sha256: {report['raw_generation_sha256']}")
        sys.exit(0)
    else:
        print(f"CONNECTIVITY SMOKE FAILED: {report['num_findings']} findings")
        for finding in report["findings"]:
            print(f"  - {finding}")
        sys.exit(1)


if __name__ == "__main__":
    main()

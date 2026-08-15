#!/usr/bin/env python3
"""E4-001 Sec 32: Annotation provider viability preflight.

Before the full development annotation, verify that both J (primary)
and J2 (secondary) can successfully annotate real items.

Minimum per annotator:
  - 2 non-sequence items
  - 1 complete sequence item

Hard gate:
  - primary success > 0
  - secondary success > 0
  - primary sequence success > 0
  - secondary sequence success > 0

Usage:
    PYTHONPATH=. python scripts/run_annotation_preflight.py \
        --api-base https://... \
        --api-key-env DASHSCOPE_API_KEY \
        --output results/empirical_v2/annotations/preflight
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.empirical_annotation import (  # noqa: E402
    ANNOTATION_SCHEMA_VERSION,
    MODEL_PRIMARY,
    MODEL_SECONDARY,
    ROLE_PRIMARY,
    ROLE_SECONDARY,
    build_row_prompt,
    build_sequence_prompt,
    parse_annotation_response,
    prompt_sha256,
    validate_row_label,
    validate_sequence_label,
    validate_sequence_structure,
    verify_secondary_blindness,
)

# LiteLLM model names (prefixed for OpenAI-compatible endpoint)
LITELLM_PRIMARY = f"openai/{MODEL_PRIMARY}"
LITELLM_SECONDARY = f"openai/{MODEL_SECONDARY}"

# Preflight item counts (Sec 32)
MIN_NON_SEQUENCE = 2
MIN_SEQUENCE = 1


def _load_development_candidates() -> list[dict]:
    """Load development accepted candidates."""
    path = PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "development" / "accepted_candidates.jsonl"
    candidates = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def _select_preflight_items(candidates: list[dict]) -> dict[str, Any]:
    """Select 2 non-sequence + 1 complete sequence for preflight."""
    non_seq = [c for c in candidates if not c.get("sequence_family_id")]
    seq_by_family: dict[str, list[dict]] = {}
    for c in candidates:
        fam = c.get("sequence_family_id")
        if fam:
            seq_by_family.setdefault(fam, []).append(c)

    # Pick first complete sequence family (all steps present, one per step)
    chosen_seq_members = None
    for fam, members in seq_by_family.items():
        step_count = members[0].get("sequence_step_count", len(members))
        # Deduplicate: pick one member per step index
        by_step: dict[int, dict] = {}
        for m in members:
            idx = m["sequence_step_index"]
            if idx not in by_step:
                by_step[idx] = m
        if len(by_step) >= step_count:
            chosen_seq_members = [by_step[i] for i in range(step_count)]
            break

    if len(non_seq) < MIN_NON_SEQUENCE:
        raise ValueError(f"Need >= {MIN_NON_SEQUENCE} non-sequence candidates, got {len(non_seq)}")
    if chosen_seq_members is None:
        raise ValueError("No complete sequence family found for preflight")

    return {
        "non_sequence": non_seq[:MIN_NON_SEQUENCE],
        "sequence_members": chosen_seq_members,
    }


def _call_annotator(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Call a real LLM annotator and return the raw result.

    Retries up to max_retries times on empty responses or transient errors.
    """
    from litellm import completion

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_result: dict[str, Any] | None = None
    for attempt_idx in range(max_retries + 1):
        start = time.monotonic()
        try:
            # glm-5.2 is a thinking model — needs extra tokens for reasoning
            is_thinking_model = "glm" in model.lower()
            effective_max_tokens = 4096 if is_thinking_model else 1024
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": effective_max_tokens,
                "timeout": timeout,
            }
            if api_base:
                kwargs["api_base"] = api_base
            if api_key:
                kwargs["api_key"] = api_key

            response = completion(**kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            text = response.choices[0].message.content
            returned_model = getattr(response, "model", model)

            # Retry on empty response (some models occasionally return empty)
            if not text or not text.strip():
                last_result = {
                    "status": "empty_response",
                    "raw_text": text,
                    "returned_model": returned_model,
                    "latency_ms": round(elapsed_ms, 1),
                    "error": "Empty response from model",
                }
                if attempt_idx < max_retries:
                    time.sleep(1.0)
                    continue
                return last_result

            return {
                "status": "success",
                "raw_text": text,
                "returned_model": returned_model,
                "latency_ms": round(elapsed_ms, 1),
                "error": None,
            }
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            last_result = {
                "status": "provider_error",
                "raw_text": None,
                "returned_model": None,
                "latency_ms": round(elapsed_ms, 1),
                "error": str(exc),
            }
            if attempt_idx < max_retries:
                time.sleep(1.0)
                continue
            return last_result

    return last_result or {"status": "provider_error", "raw_text": None, "returned_model": None, "latency_ms": 0.0, "error": "Unknown error"}


def _check_credential_leakage(text: str) -> list[str]:
    """Check that the response does not contain credentials.

    Uses strict patterns to avoid false positives:
    - sk- followed by uppercase letter (API key pattern)
    - api_key= or api_key: (assignment patterns)
    - bearer followed by space + token
    - password= (assignment pattern)
    """
    leaked: list[str] = []
    import re as _re
    # Strict patterns: sk- followed by uppercase (API key prefix)
    if _re.search(r"sk-[A-Z]", text):
        leaked.append("sk-[A-Z] (API key pattern)")
    if _re.search(r"api_key\s*[=:]", text, _re.IGNORECASE):
        leaked.append("api_key assignment")
    if _re.search(r"bearer\s+[A-Za-z0-9_\-.]{8,}", text, _re.IGNORECASE):
        leaked.append("bearer token")
    if _re.search(r"password\s*=", text, _re.IGNORECASE):
        leaked.append("password assignment")
    return leaked


def run_annotation_preflight(
    api_base: str | None = None,
    api_key_env: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the annotation provider viability preflight.

    Returns a summary dict with pass/fail status.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve API key
    api_key: str | None = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")

    # Load candidates and select preflight items
    candidates = _load_development_candidates()
    items = _select_preflight_items(candidates)
    non_seq_items = items["non_sequence"]
    seq_members = items["sequence_members"]

    # Validate sequence structure
    seq_errors = validate_sequence_structure(seq_members)
    if seq_errors:
        raise ValueError(f"Sequence structure validation failed: {seq_errors}")

    # Build prompts
    row_prompts = []
    for cand in non_seq_items:
        sys_p, usr_p = build_row_prompt(cand)
        row_prompts.append({
            "candidate_id": cand["candidate_id"],
            "system_prompt": sys_p,
            "user_prompt": usr_p,
            "system_prompt_sha256": prompt_sha256(sys_p),
            "user_prompt_sha256": prompt_sha256(usr_p),
        })

    seq_sys_p, seq_usr_p = build_sequence_prompt(seq_members)
    seq_prompt = {
        "sequence_family_id": seq_members[0]["sequence_family_id"],
        "system_prompt": seq_sys_p,
        "user_prompt": seq_usr_p,
        "system_prompt_sha256": prompt_sha256(seq_sys_p),
        "user_prompt_sha256": prompt_sha256(seq_usr_p),
    }

    # Run preflight for each annotator
    annotators = [
        ("primary", ROLE_PRIMARY, LITELLM_PRIMARY, MODEL_PRIMARY),
        ("secondary", ROLE_SECONDARY, LITELLM_SECONDARY, MODEL_SECONDARY),
    ]

    attempts: list[dict[str, Any]] = []
    results_summary: dict[str, Any] = {
        "primary_success": 0,
        "secondary_success": 0,
        "primary_sequence_success": 0,
        "secondary_sequence_success": 0,
        "primary_parse_success": 0,
        "secondary_parse_success": 0,
    }

    for role_label, role_id, litellm_model, model_name in annotators:
        # Row annotation preflight
        for rp in row_prompts:
            attempt_id = f"preflight_{role_id}_{rp['candidate_id'][:20]}_{uuid.uuid4().hex[:8]}"
            result = _call_annotator(
                rp["system_prompt"], rp["user_prompt"],
                litellm_model, api_base, api_key,
            )

            attempt: dict[str, Any] = {
                "provider_attempt_id": attempt_id,
                "annotator_role": role_id,
                "item_type": "row",
                "candidate_id": rp["candidate_id"],
                "requested_model": model_name,
                "litellm_model": litellm_model,
                "system_prompt_sha256": rp["system_prompt_sha256"],
                "user_prompt_sha256": rp["user_prompt_sha256"],
                "latency_ms": result["latency_ms"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if result["status"] == "success":
                attempt["raw_response"] = result["raw_text"]
                attempt["returned_model"] = result["returned_model"]

                # Check credential leakage
                cred_leak = _check_credential_leakage(result["raw_text"] or "")
                if cred_leak:
                    attempt["status"] = "refusal"
                    attempt["parse_status"] = "credential_leak"
                    attempt["error"] = f"Credential indicators found: {cred_leak}"
                else:
                    # Try to parse
                    try:
                        parsed = parse_annotation_response(result["raw_text"])
                        ann = validate_row_label(parsed)
                        attempt["status"] = "success"
                        attempt["parse_status"] = "valid"
                        attempt["parsed_label"] = {
                            "target_relevant": ann.target_relevant,
                            "target_leakage": ann.target_leakage,
                            "positive_entailment": ann.positive_entailment,
                            "task_useful": ann.task_useful,
                            "leakage_strength": ann.leakage_strength,
                            "confidence": ann.confidence,
                        }
                        results_summary[f"{role_label}_success"] += 1
                        results_summary[f"{role_label}_parse_success"] += 1
                    except Exception as exc:
                        attempt["status"] = "malformed"
                        attempt["parse_status"] = "parse_error"
                        attempt["error"] = str(exc)
            else:
                attempt["status"] = result["status"]
                attempt["parse_status"] = "not_attempted"
                attempt["error"] = result["error"]

            attempts.append(attempt)

        # Sequence annotation preflight
        attempt_id = f"preflight_{role_id}_seq_{seq_prompt['sequence_family_id'][:20]}_{uuid.uuid4().hex[:8]}"
        result = _call_annotator(
            seq_prompt["system_prompt"], seq_prompt["user_prompt"],
            litellm_model, api_base, api_key,
        )

        attempt = {
            "provider_attempt_id": attempt_id,
            "annotator_role": role_id,
            "item_type": "sequence",
            "sequence_family_id": seq_prompt["sequence_family_id"],
            "requested_model": model_name,
            "litellm_model": litellm_model,
            "system_prompt_sha256": seq_prompt["system_prompt_sha256"],
            "user_prompt_sha256": seq_prompt["user_prompt_sha256"],
            "latency_ms": result["latency_ms"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if result["status"] == "success":
            attempt["raw_response"] = result["raw_text"]
            attempt["returned_model"] = result["returned_model"]

            cred_leak = _check_credential_leakage(result["raw_text"] or "")
            if cred_leak:
                attempt["status"] = "refusal"
                attempt["parse_status"] = "credential_leak"
                attempt["error"] = f"Credential indicators found: {cred_leak}"
            else:
                try:
                    parsed = parse_annotation_response(result["raw_text"])
                    ann = validate_sequence_label(parsed)
                    attempt["status"] = "success"
                    attempt["parse_status"] = "valid"
                    attempt["parsed_label"] = {
                        "sequence_reconstructs_target": ann.sequence_reconstructs_target,
                        "earliest_reconstruction_step": ann.earliest_reconstruction_step,
                        "reconstruction_strength": ann.reconstruction_strength,
                        "confidence": ann.confidence,
                    }
                    results_summary[f"{role_label}_sequence_success"] += 1
                except Exception as exc:
                    attempt["status"] = "malformed"
                    attempt["parse_status"] = "parse_error"
                    attempt["error"] = str(exc)
        else:
            attempt["status"] = result["status"]
            attempt["parse_status"] = "not_attempted"
            attempt["error"] = result["error"]

        attempts.append(attempt)

    # Secondary blindness check
    blindness_violations = verify_secondary_blindness(seq_prompt["user_prompt"])
    for rp in row_prompts:
        blindness_violations.extend(verify_secondary_blindness(rp["user_prompt"]))

    # Hard gate evaluation (Sec 32)
    findings: list[str] = []
    if results_summary["primary_success"] == 0:
        findings.append("primary success = 0 (need > 0)")
    if results_summary["secondary_success"] == 0:
        findings.append("secondary success = 0 (need > 0)")
    if results_summary["primary_sequence_success"] == 0:
        findings.append("primary sequence success = 0 (need > 0)")
    if results_summary["secondary_sequence_success"] == 0:
        findings.append("secondary sequence success = 0 (need > 0)")
    if blindness_violations:
        findings.append(f"secondary blindness violations: {blindness_violations}")

    passed = len(findings) == 0

    # Build report
    report: dict[str, Any] = {
        "preflight_type": "annotation_provider_viability",
        "spec_section": 32,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "primary_model": MODEL_PRIMARY,
        "secondary_model": MODEL_SECONDARY,
        "primary_litellm_model": LITELLM_PRIMARY,
        "secondary_litellm_model": LITELLM_SECONDARY,
        "api_base": api_base,
        "api_key_env": api_key_env,
        "non_sequence_items": len(non_seq_items),
        "sequence_items": 1,
        "sequence_family_id": seq_prompt["sequence_family_id"],
        "sequence_step_count": len(seq_members),
        "results": results_summary,
        "findings": findings,
        "passed": passed,
        "total_attempts": len(attempts),
    }

    # Write artifacts
    attempts_path = output_dir / "raw_annotation_attempts.jsonl"
    with open(attempts_path, "w") as f:
        for a in attempts:
            # Redact raw response for attempts file (keep parsed labels)
            safe = {k: v for k, v in a.items() if k != "raw_response"}
            if "raw_response" in a:
                safe["raw_response_length"] = len(a["raw_response"])
            f.write(json.dumps(safe) + "\n")

    report_path = output_dir / "annotation_preflight_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("=" * 60)
    print("ANNOTATION PROVIDER VIABILITY PREFLIGHT (Sec 32)")
    print("=" * 60)
    print(f"Primary model: {MODEL_PRIMARY} ({LITELLM_PRIMARY})")
    print(f"Secondary model: {MODEL_SECONDARY} ({LITELLM_SECONDARY})")
    print(f"Non-sequence items: {len(non_seq_items)}")
    print(f"Sequence items: 1 (family: {seq_prompt['sequence_family_id'][:40]}...)")
    print(f"Sequence steps: {len(seq_members)}")
    print()
    print(f"Primary row successes: {results_summary['primary_success']}/{len(non_seq_items)}")
    print(f"Secondary row successes: {results_summary['secondary_success']}/{len(non_seq_items)}")
    print(f"Primary sequence success: {results_summary['primary_sequence_success']}/1")
    print(f"Secondary sequence success: {results_summary['secondary_sequence_success']}/1")
    print(f"Primary parse successes: {results_summary['primary_parse_success']}")
    print(f"Secondary parse successes: {results_summary['secondary_parse_success']}")
    print()
    print(f"Total attempts: {len(attempts)}")
    print(f"Blocking findings: {len(findings)}")
    for finding in findings:
        print(f"  - {finding}")
    print()
    print(f"Preflight passed: {passed}")
    if passed:
        print("READY FOR DEVELOPMENT ANNOTATION: YES")
    else:
        print("READY FOR DEVELOPMENT ANNOTATION: NO")
    print("=" * 60)

    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Annotation provider viability preflight (Sec 32)")
    parser.add_argument("--api-base", default=None, help="API base URL")
    parser.add_argument("--api-key-env", default=None, help="Environment variable name for API key")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    try:
        report = run_annotation_preflight(
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"Preflight failed with error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

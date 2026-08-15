#!/usr/bin/env python3
"""E4-001 Sec 33: Run J (primary) annotation on the full development split.

Annotates all 225 row items + 36 sequence items using J (qwen3.8-max).
Supports resume: skips items that already have valid annotations.

Usage:
    PYTHONPATH=. python scripts/run_primary_annotation.py \
        --api-base https://... \
        --api-key-env DASHSCOPE_API_KEY \
        --output results/empirical_v2/annotations/development
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
    ROLE_PRIMARY,
    RowAnnotation,
    SequenceAnnotation,
    build_annotation_config,
    build_campaign_identity,
    build_development_queue,
    build_prompt_manifest,
    build_row_prompt,
    build_sequence_prompt,
    parse_annotation_response,
    prompt_sha256,
    validate_row_label,
    validate_sequence_label,
    validate_sequence_structure,
)

LITELLM_PRIMARY = f"openai/{MODEL_PRIMARY}"

# Rate limiting
MIN_DELAY_BETWEEN_CALLS = 0.5  # seconds
MAX_RETRIES = 3


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


def _call_annotator(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    timeout: float = 120.0,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Call a real LLM annotator with retries."""
    from litellm import completion

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    is_thinking = "glm" in model.lower()
    effective_max_tokens = 4096 if is_thinking else 1024

    last_result: dict[str, Any] | None = None
    for attempt_idx in range(max_retries + 1):
        start = time.monotonic()
        try:
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

            if not text or not text.strip():
                last_result = {
                    "status": "empty_response",
                    "raw_text": None,
                    "returned_model": returned_model,
                    "latency_ms": round(elapsed_ms, 1),
                    "error": "Empty response from model",
                }
                if attempt_idx < max_retries:
                    time.sleep(2.0 * (attempt_idx + 1))
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
                time.sleep(2.0 * (attempt_idx + 1))
                continue
            return last_result

    return last_result or {"status": "provider_error", "raw_text": None, "returned_model": None, "latency_ms": 0.0, "error": "Unknown"}


def _load_existing_annotations(output_dir: Path) -> tuple[set[str], set[str]]:
    """Load already-completed annotation IDs for resume support."""
    row_ids: set[str] = set()
    seq_ids: set[str] = set()

    row_path = output_dir / "row_annotations.jsonl"
    if row_path.exists():
        with open(row_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    row_ids.add(rec.get("annotation_id", ""))

    seq_path = output_dir / "sequence_annotations.jsonl"
    if seq_path.exists():
        with open(seq_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    seq_ids.add(rec.get("sequence_annotation_id", ""))

    return row_ids, seq_ids


def _build_candidate_lookup(candidates: list[dict]) -> dict[str, dict]:
    """Build a lookup from candidate_id to candidate dict."""
    return {c["candidate_id"]: c for c in candidates}


def run_primary_annotation(
    api_base: str | None = None,
    api_key_env: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run J primary annotation on the full development split."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "development"
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key: str | None = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")

    # Load candidates and build queue
    candidates = _load_development_candidates()
    candidate_lookup = _build_candidate_lookup(candidates)
    row_items, sequence_items = build_development_queue()

    # Resume support
    done_row, done_seq = _load_existing_annotations(output_dir)

    # Open output files in append mode
    row_path = output_dir / "row_annotations.jsonl"
    seq_path = output_dir / "sequence_annotations.jsonl"
    attempt_log_path = output_dir / "primary_annotation_attempts.jsonl"

    # Campaign identity
    queue_data = json.dumps(
        {"row_items": len(row_items), "seq_items": len(sequence_items)},
        sort_keys=True, separators=(",", ":"),
    )
    queue_sha = prompt_sha256(queue_data)
    config_data = json.dumps(build_annotation_config(), sort_keys=True, separators=(",", ":"))
    config_sha = prompt_sha256(config_data)
    prompt_manifest_data = json.dumps(build_prompt_manifest(), sort_keys=True, separators=(",", ":"))
    pm_sha = prompt_sha256(prompt_manifest_data)
    code_commit = _get_git_commit()

    campaign_id = build_campaign_identity(
        queue_sha256=queue_sha,
        annotation_config_sha256=config_sha,
        prompt_manifest_sha256=pm_sha,
        annotation_code_commit=code_commit,
    )

    # Write campaign identity
    campaign_path = output_dir / "campaign_identity.json"
    if not campaign_path.exists():
        with open(campaign_path, "w") as f:
            json.dump(campaign_id, f, indent=2)

    stats = {
        "row_total": len(row_items),
        "row_completed": len(done_row),
        "row_success": 0,
        "row_failure": 0,
        "seq_total": len(sequence_items),
        "seq_completed": len(done_seq),
        "seq_success": 0,
        "seq_failure": 0,
        "total_api_calls": 0,
        "total_latency_ms": 0.0,
    }

    print(f"Primary annotation: {MODEL_PRIMARY} ({LITELLM_PRIMARY})")
    print(f"Row items: {len(row_items)} ({len(done_row)} already done)")
    print(f"Sequence items: {len(sequence_items)} ({len(done_seq)} already done)")
    print()

    # --- Row annotations ---
    with open(row_path, "a") as row_f, open(attempt_log_path, "a") as attempt_f:
        for idx, item in enumerate(row_items):
            ann_id = item["annotation_id"]
            if ann_id in done_row:
                continue

            cand = candidate_lookup[item["candidate_id"]]
            sys_p, usr_p = build_row_prompt(cand)

            attempt_id = f"primary_{ann_id[:30]}_{uuid.uuid4().hex[:8]}"
            result = _call_annotator(sys_p, usr_p, LITELLM_PRIMARY, api_base, api_key)
            stats["total_api_calls"] += 1
            stats["total_latency_ms"] += result["latency_ms"]

            attempt_record: dict[str, Any] = {
                "provider_attempt_id": attempt_id,
                "annotator_role": ROLE_PRIMARY,
                "item_type": "row",
                "annotation_id": ann_id,
                "candidate_id": item["candidate_id"],
                "requested_model": MODEL_PRIMARY,
                "litellm_model": LITELLM_PRIMARY,
                "system_prompt_sha256": prompt_sha256(sys_p),
                "user_prompt_sha256": prompt_sha256(usr_p),
                "latency_ms": result["latency_ms"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if result["status"] == "success":
                try:
                    parsed = parse_annotation_response(result["raw_text"])
                    ann = validate_row_label(parsed)
                    # Build full annotation record
                    record = RowAnnotation(
                        annotation_id=ann_id,
                        candidate_id=item["candidate_id"],
                        scenario_id=item["scenario_id"],
                        secret_variant_id=item["secret_variant_id"],
                        target_relevant=ann.target_relevant,
                        target_leakage=ann.target_leakage,
                        positive_entailment=ann.positive_entailment,
                        task_useful=ann.task_useful,
                        leakage_strength=ann.leakage_strength,
                        uncertain=ann.uncertain,
                        confidence=ann.confidence,
                        short_rationale=ann.short_rationale,
                        annotator_role=ROLE_PRIMARY,
                        annotator_provider="litellm",
                        annotator_model_requested=MODEL_PRIMARY,
                        annotator_model_returned=result["returned_model"],
                        annotator_transport="openai_compatible",
                        system_prompt_sha256=prompt_sha256(sys_p),
                        user_prompt_sha256=prompt_sha256(usr_p),
                        request_id=attempt_id,
                        retry_index=0,
                        annotated_at=datetime.now(timezone.utc).isoformat(),
                        frozen_corpus_manifest_sha256=item["frozen_corpus_manifest_sha256"],
                        candidate_content_sha256=item["candidate_content_sha256"],
                    )
                    row_f.write(json.dumps(record.to_dict()) + "\n")
                    row_f.flush()
                    stats["row_success"] += 1
                    attempt_record["status"] = "success"
                    attempt_record["parse_status"] = "valid"
                except Exception as exc:
                    stats["row_failure"] += 1
                    attempt_record["status"] = "malformed"
                    attempt_record["parse_status"] = "parse_error"
                    attempt_record["error"] = str(exc)
            else:
                stats["row_failure"] += 1
                attempt_record["status"] = result["status"]
                attempt_record["parse_status"] = "not_attempted"
                attempt_record["error"] = result.get("error")

            attempt_f.write(json.dumps(attempt_record) + "\n")
            attempt_f.flush()

            # Progress
            done_count = stats["row_completed"] + stats["row_success"] + stats["row_failure"]
            if (idx + 1) % 10 == 0 or (idx + 1) == len(row_items):
                print(f"  Row progress: {done_count}/{len(row_items)} "
                      f"(success={stats['row_success']}, fail={stats['row_failure']})")

            time.sleep(MIN_DELAY_BETWEEN_CALLS)

    # --- Sequence annotations ---
    with open(seq_path, "a") as seq_f, open(attempt_log_path, "a") as attempt_f:
        for idx, item in enumerate(sequence_items):
            seq_ann_id = item["sequence_annotation_id"]
            if seq_ann_id in done_seq:
                continue

            # Load sequence members
            members = [candidate_lookup[cid] for cid in item["ordered_candidate_ids"]]
            seq_errors = validate_sequence_structure(members)
            if seq_errors:
                print(f"  WARNING: Sequence {seq_ann_id[:30]}... structure error: {seq_errors}")
                stats["seq_failure"] += 1
                continue

            sys_p, usr_p = build_sequence_prompt(members)
            attempt_id = f"primary_{seq_ann_id[:30]}_{uuid.uuid4().hex[:8]}"
            result = _call_annotator(sys_p, usr_p, LITELLM_PRIMARY, api_base, api_key)
            stats["total_api_calls"] += 1
            stats["total_latency_ms"] += result["latency_ms"]

            attempt_record = {
                "provider_attempt_id": attempt_id,
                "annotator_role": ROLE_PRIMARY,
                "item_type": "sequence",
                "sequence_annotation_id": seq_ann_id,
                "sequence_family_id": item["sequence_family_id"],
                "requested_model": MODEL_PRIMARY,
                "litellm_model": LITELLM_PRIMARY,
                "system_prompt_sha256": prompt_sha256(sys_p),
                "user_prompt_sha256": prompt_sha256(usr_p),
                "latency_ms": result["latency_ms"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if result["status"] == "success":
                try:
                    parsed = parse_annotation_response(result["raw_text"])
                    ann = validate_sequence_label(parsed)
                    record = SequenceAnnotation(
                        sequence_annotation_id=seq_ann_id,
                        sequence_family_id=item["sequence_family_id"],
                        scenario_id=item["scenario_id"],
                        secret_variant_id=item["secret_variant_id"],
                        ordered_candidate_ids=item["ordered_candidate_ids"],
                        step_count=item["step_count"],
                        sequence_reconstructs_target=ann.sequence_reconstructs_target,
                        earliest_reconstruction_step=ann.earliest_reconstruction_step,
                        reconstruction_strength=ann.reconstruction_strength,
                        uncertain=ann.uncertain,
                        confidence=ann.confidence,
                        short_rationale=ann.short_rationale,
                        annotator_role=ROLE_PRIMARY,
                        annotator_provider="litellm",
                        annotator_model_requested=MODEL_PRIMARY,
                        annotator_model_returned=result["returned_model"],
                        annotator_transport="openai_compatible",
                        system_prompt_sha256=prompt_sha256(sys_p),
                        user_prompt_sha256=prompt_sha256(usr_p),
                        request_id=attempt_id,
                        retry_index=0,
                        annotated_at=datetime.now(timezone.utc).isoformat(),
                        frozen_corpus_manifest_sha256=item["frozen_corpus_manifest_sha256"],
                        sequence_content_sha256=item["sequence_content_sha256"],
                    )
                    seq_f.write(json.dumps(record.to_dict()) + "\n")
                    seq_f.flush()
                    stats["seq_success"] += 1
                    attempt_record["status"] = "success"
                    attempt_record["parse_status"] = "valid"
                except Exception as exc:
                    stats["seq_failure"] += 1
                    attempt_record["status"] = "malformed"
                    attempt_record["parse_status"] = "parse_error"
                    attempt_record["error"] = str(exc)
            else:
                stats["seq_failure"] += 1
                attempt_record["status"] = result["status"]
                attempt_record["parse_status"] = "not_attempted"
                attempt_record["error"] = result.get("error")

            attempt_f.write(json.dumps(attempt_record) + "\n")
            attempt_f.flush()

            done_count = stats["seq_completed"] + stats["seq_success"] + stats["seq_failure"]
            print(f"  Seq progress: {done_count}/{len(sequence_items)} "
                  f"(success={stats['seq_success']}, fail={stats['seq_failure']})")

            time.sleep(MIN_DELAY_BETWEEN_CALLS)

    # Write summary
    summary = {
        "annotation_role": ROLE_PRIMARY,
        "model": MODEL_PRIMARY,
        "litellm_model": LITELLM_PRIMARY,
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "campaign_identity": campaign_id,
        "stats": stats,
        "api_base": api_base,
    }
    summary_path = output_dir / "primary_annotation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("PRIMARY ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"Row: {stats['row_success']}/{stats['row_total']} success, {stats['row_failure']} failures")
    print(f"Sequence: {stats['seq_success']}/{stats['seq_total']} success, {stats['seq_failure']} failures")
    print(f"Total API calls: {stats['total_api_calls']}")
    print(f"Total latency: {stats['total_latency_ms']/1000:.1f}s")
    print(f"Output: {output_dir}")
    print("=" * 60)

    return summary


def _get_git_commit() -> str:
    """Get current git commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run J primary annotation (Sec 33)")
    parser.add_argument("--api-base", default=None, help="API base URL")
    parser.add_argument("--api-key-env", default=None, help="Environment variable name for API key")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    try:
        run_primary_annotation(
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"Annotation failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

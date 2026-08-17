#!/usr/bin/env python3
"""E4-003 Sec 29: Run J2 (secondary) annotation on the full test split.

Applies the frozen development protocol unchanged to test.
Annotates all 450 row items + 72 sequence items using J2 (glm-5.2).
Supports resume: skips items that already have valid annotations.

Usage:
    PYTHONPATH=. python scripts/run_test_secondary.py \
        --api-base https://... \
        --api-key-env DASHSCOPE_API_KEY
"""

from __future__ import annotations

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
    MODEL_SECONDARY,
    ROLE_SECONDARY,
    RowAnnotation,
    SequenceAnnotation,
    build_annotation_config,
    build_campaign_identity,
    verify_campaign_identity,
    build_prompt_manifest,
    build_row_prompt,
    build_sequence_prompt,
    build_test_queue,
    compute_queue_sha256,
    frozen_corpus_manifest_file_sha256,
    load_test_candidates,
    parse_annotation_response,
    prompt_sha256,
    test_input_preflight,
    validate_row_label,
    validate_sequence_label,
    validate_sequence_structure,
)

# Import provider from development runner
from scripts.run_primary_annotation import (  # noqa: E402
    MIN_DELAY_BETWEEN_CALLS,
    MAX_RETRIES,
    call_annotation_provider_once,
)

LITELLM_SECONDARY = f"openai/{MODEL_SECONDARY}"
OUTPUT_DIR = PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "test"


def _load_existing_annotations(output_dir: Path) -> tuple[set[str], set[str]]:
    """Load already-completed annotation IDs for resume support."""
    row_ids: set[str] = set()
    seq_ids: set[str] = set()

    row_path = output_dir / "secondary_row_annotations.jsonl"
    if row_path.exists():
        with open(row_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    row_ids.add(rec.get("annotation_id", ""))

    seq_path = output_dir / "secondary_sequence_annotations.jsonl"
    if seq_path.exists():
        with open(seq_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    seq_ids.add(rec.get("sequence_annotation_id", ""))

    return row_ids, seq_ids


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


def run_test_secondary_annotation(
    api_base: str | None = None,
    api_key_env: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run J2 secondary annotation on the full test split (E4-003)."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key: str | None = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")

    # --- Preflight (Sec 19) ---
    print("Running test input preflight...")
    preflight = test_input_preflight()
    preflight_path = output_dir / "test_input_preflight.json"
    with open(preflight_path, "w") as f:
        json.dump(preflight, f, indent=2)

    if not preflight["passed"]:
        print(f"Preflight FAILED with {len(preflight['findings'])} findings:")
        for finding in preflight["findings"]:
            print(f"  - {finding}")
        raise RuntimeError(f"Test preflight failed: {preflight['findings']}")
    print("Preflight PASSED.")

    # Load candidates and build queue
    candidates = load_test_candidates()
    candidate_lookup = {c["candidate_id"]: c for c in candidates}
    row_items, sequence_items = build_test_queue()

    # Resume support
    done_row, done_seq = _load_existing_annotations(output_dir)

    # Open output files in append mode
    row_path = output_dir / "secondary_row_annotations.jsonl"
    seq_path = output_dir / "secondary_sequence_annotations.jsonl"
    attempt_log_path = output_dir / "secondary_annotation_attempts.jsonl"

    # Campaign identity
    queue_sha = compute_queue_sha256(row_items, sequence_items)
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
        split="test",
    )

    campaign_sha = prompt_sha256(json.dumps(campaign_id, sort_keys=True, separators=(",", ":")))
    manifest_sha = frozen_corpus_manifest_file_sha256()

    # Write campaign identity with resume verification (items 15-16)
    campaign_path = output_dir / "secondary_campaign_identity.json"
    if campaign_path.exists():
        existing_identity = json.loads(campaign_path.read_text())
        mismatches = verify_campaign_identity(existing_identity, campaign_id)
        if mismatches:
            raise RuntimeError(
                f"Campaign identity mismatch on resume: {mismatches}"
            )
    else:
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

    print(f"Test secondary annotation: {MODEL_SECONDARY} ({LITELLM_SECONDARY})")
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

            terminal_success = False
            for retry_index in range(MAX_RETRIES + 1):
                attempt_id = f"test_secondary_{ann_id[:30]}_{uuid.uuid4().hex[:8]}"
                result = call_annotation_provider_once(
                    sys_p, usr_p, LITELLM_SECONDARY, api_base, api_key
                )
                stats["total_api_calls"] += 1
                stats["total_latency_ms"] += result["latency_ms"]

                provider_req_id = result.get("provider_request_id", "")
                raw_text = result.get("raw_text") or ""

                attempt_record: dict[str, Any] = {
                    "provider_attempt_id": attempt_id,
                    "annotation_item_id": ann_id,
                    "candidate_id": item["candidate_id"],
                    "annotator_role": ROLE_SECONDARY,
                    "retry_index": retry_index,
                    "status": result["status"],
                    "raw_response": raw_text,
                    "parse_status": "",
                    "requested_model": MODEL_SECONDARY,
                    "returned_model": result.get("returned_model") or "",
                    "model_revision": "not_exposed_by_provider",
                    "provider": "litellm",
                    "transport": "openai_compatible",
                    "provider_request_id": provider_req_id,
                    "system_prompt_sha256": prompt_sha256(sys_p),
                    "user_prompt_sha256": prompt_sha256(usr_p),
                    "latency_ms": result["latency_ms"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_class": result.get("error_class", ""),
                    "error_message_safe": result.get("error_message", "") or "",
                    "frozen_corpus_manifest_sha256": manifest_sha,
                    "annotation_campaign_identity_sha256": campaign_sha,
                    "split": "test",
                }

                if result["status"] == "success" and raw_text:
                    try:
                        parsed = parse_annotation_response(raw_text)
                        ann = validate_row_label(parsed)
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
                            annotator_role=ROLE_SECONDARY,
                            annotator_provider="litellm",
                            annotator_model_requested=MODEL_SECONDARY,
                            annotator_model_returned=result["returned_model"],
                            annotator_model_revision="not_exposed_by_provider",
                            annotator_transport="openai_compatible",
                            system_prompt_sha256=prompt_sha256(sys_p),
                            user_prompt_sha256=prompt_sha256(usr_p),
                            provider_request_id=provider_req_id,
                            retry_index=retry_index,
                            annotated_at=datetime.now(timezone.utc).isoformat(),
                            frozen_corpus_manifest_sha256=item["frozen_corpus_manifest_sha256"],
                            candidate_content_sha256=item["candidate_content_sha256"],
                        )
                        row_f.write(json.dumps(record.to_dict()) + "\n")
                        row_f.flush()
                        stats["row_success"] += 1
                        attempt_record["parse_status"] = "valid"
                        terminal_success = True
                    except Exception as exc:
                        stats["row_failure"] += 1
                        attempt_record["status"] = "malformed"
                        attempt_record["parse_status"] = "parse_error"
                        attempt_record["error_class"] = type(exc).__name__
                        attempt_record["error_message_safe"] = str(exc)
                        terminal_success = True
                else:
                    attempt_record["parse_status"] = "not_attempted"

                attempt_f.write(json.dumps(attempt_record) + "\n")
                attempt_f.flush()

                if terminal_success:
                    break

                if retry_index < MAX_RETRIES:
                    time.sleep(2.0 * (retry_index + 1))

            if not terminal_success:
                stats["row_failure"] += 1

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

            members = [candidate_lookup[cid] for cid in item["ordered_candidate_ids"]]
            seq_errors = validate_sequence_structure(members)
            if seq_errors:
                print(f"  WARNING: Sequence {seq_ann_id[:30]}... structure error: {seq_errors}")
                stats["seq_failure"] += 1
                continue

            sys_p, usr_p = build_sequence_prompt(members)

            terminal_success = False
            for retry_index in range(MAX_RETRIES + 1):
                attempt_id = f"test_secondary_{seq_ann_id[:30]}_{uuid.uuid4().hex[:8]}"
                result = call_annotation_provider_once(
                    sys_p, usr_p, LITELLM_SECONDARY, api_base, api_key
                )
                stats["total_api_calls"] += 1
                stats["total_latency_ms"] += result["latency_ms"]

                provider_req_id = result.get("provider_request_id", "")
                raw_text = result.get("raw_text") or ""

                attempt_record: dict[str, Any] = {
                    "provider_attempt_id": attempt_id,
                    "annotation_item_id": seq_ann_id,
                    "sequence_family_id": item["sequence_family_id"],
                    "annotator_role": ROLE_SECONDARY,
                    "retry_index": retry_index,
                    "status": result["status"],
                    "raw_response": raw_text,
                    "parse_status": "",
                    "requested_model": MODEL_SECONDARY,
                    "returned_model": result.get("returned_model") or "",
                    "model_revision": "not_exposed_by_provider",
                    "provider": "litellm",
                    "transport": "openai_compatible",
                    "provider_request_id": provider_req_id,
                    "system_prompt_sha256": prompt_sha256(sys_p),
                    "user_prompt_sha256": prompt_sha256(usr_p),
                    "latency_ms": result["latency_ms"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_class": result.get("error_class", ""),
                    "error_message_safe": result.get("error_message", "") or "",
                    "frozen_corpus_manifest_sha256": manifest_sha,
                    "annotation_campaign_identity_sha256": campaign_sha,
                    "split": "test",
                }

                if result["status"] == "success" and raw_text:
                    try:
                        parsed = parse_annotation_response(raw_text)
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
                            annotator_role=ROLE_SECONDARY,
                            annotator_provider="litellm",
                            annotator_model_requested=MODEL_SECONDARY,
                            annotator_model_returned=result["returned_model"],
                            annotator_model_revision="not_exposed_by_provider",
                            annotator_transport="openai_compatible",
                            system_prompt_sha256=prompt_sha256(sys_p),
                            user_prompt_sha256=prompt_sha256(usr_p),
                            provider_request_id=provider_req_id,
                            retry_index=retry_index,
                            annotated_at=datetime.now(timezone.utc).isoformat(),
                            frozen_corpus_manifest_sha256=item["frozen_corpus_manifest_sha256"],
                            sequence_content_sha256=item["sequence_content_sha256"],
                        )
                        seq_f.write(json.dumps(record.to_dict()) + "\n")
                        seq_f.flush()
                        stats["seq_success"] += 1
                        attempt_record["parse_status"] = "valid"
                        terminal_success = True
                    except Exception as exc:
                        stats["seq_failure"] += 1
                        attempt_record["status"] = "malformed"
                        attempt_record["parse_status"] = "parse_error"
                        attempt_record["error_class"] = type(exc).__name__
                        attempt_record["error_message_safe"] = str(exc)
                        terminal_success = True
                else:
                    attempt_record["parse_status"] = "not_attempted"

                attempt_f.write(json.dumps(attempt_record) + "\n")
                attempt_f.flush()

                if terminal_success:
                    break

                if retry_index < MAX_RETRIES:
                    time.sleep(2.0 * (retry_index + 1))

            if not terminal_success:
                stats["seq_failure"] += 1

            done_count = stats["seq_completed"] + stats["seq_success"] + stats["seq_failure"]
            print(f"  Seq progress: {done_count}/{len(sequence_items)} "
                  f"(success={stats['seq_success']}, fail={stats['seq_failure']})")

            time.sleep(MIN_DELAY_BETWEEN_CALLS)

    # Write summary
    summary = {
        "annotation_role": ROLE_SECONDARY,
        "model": MODEL_SECONDARY,
        "litellm_model": LITELLM_SECONDARY,
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "split": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "campaign_identity": campaign_id,
        "stats": stats,
        "api_base": api_base,
    }
    summary_path = output_dir / "secondary_annotation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("TEST SECONDARY ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"Row: {stats['row_success']}/{stats['row_total']} success, {stats['row_failure']} failures")
    print(f"Sequence: {stats['seq_success']}/{stats['seq_total']} success, {stats['seq_failure']} failures")
    print(f"Total API calls: {stats['total_api_calls']}")
    print(f"Total latency: {stats['total_latency_ms']/1000:.1f}s")
    print(f"Output: {output_dir}")
    print("=" * 60)

    return summary


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run J2 secondary annotation on test split (E4-003)")
    parser.add_argument("--api-base", default=None, help="API base URL")
    parser.add_argument("--api-key-env", default=None, help="Environment variable name for API key")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    try:
        run_test_secondary_annotation(
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"Test annotation failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

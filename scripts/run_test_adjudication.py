#!/usr/bin/env python3
"""E4-003 Sec 38-42: Run J3 adjudication on the test split review queue.

Reads test_review_queue.jsonl, calls J3 (qwen-plus) for disagreed rows,
constructs final test labels, writes adjudication manifest.

Usage:
    PYTHONPATH=. python scripts/run_test_adjudication.py \
        --api-base https://... \
        --api-key-env TEST_API_KEY
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_DIR = PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "test"
_CORPUS_DIR = PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_TEST_CANDIDATES_PATH = _CORPUS_DIR / "test" / "accepted_candidates.jsonl"

# Input files
_J_ROW_LABELS_PATH = _TEST_DIR / "primary_row_annotations.jsonl"
_J2_ROW_LABELS_PATH = _TEST_DIR / "secondary_row_annotations.jsonl"
_J_SEQ_LABELS_PATH = _TEST_DIR / "primary_sequence_annotations.jsonl"
_J2_SEQ_LABELS_PATH = _TEST_DIR / "secondary_sequence_annotations.jsonl"
_REVIEW_QUEUE_PATH = _TEST_DIR / "test_review_queue.jsonl"

# Output files
_LLM_ADJUDICATION_PATH = _TEST_DIR / "test_llm_adjudication.jsonl"
_FINAL_LABELS_PATH = _TEST_DIR / "test_final_adjudicated_labels.jsonl"
_FINAL_SEQ_LABELS_PATH = _TEST_DIR / "test_final_sequence_labels.jsonl"
_ADJUDICATION_MANIFEST_PATH = _TEST_DIR / "test_adjudication_manifest.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_BINARY_LABELS = ("target_relevant", "target_leakage", "positive_entailment", "task_useful")
ALL_CORE_LABELS = CORE_BINARY_LABELS + ("leakage_strength",)

# J3 adjudicator identity
J3_ROLE = "J3"
J3_MODEL = "qwen-plus"
J3_PROVIDER = "litellm"
ADJUDICATION_PROTOCOL_VERSION = "1.0"
ANNOTATION_SCHEMA_VERSION = "1.0"

# Frozen SHAs
_FROZEN_CORPUS_SHA = "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"

# Thresholds (Sec 36)
MAX_UNRESOLVED_RATE = 0.10

# Expected test counts
EXPECTED_ROWS = 450
EXPECTED_SEQUENCES = 72
EXPECTED_FAMILIES = 24


# ---------------------------------------------------------------------------
# Sequence unit helper (§5)
# ---------------------------------------------------------------------------


def sequence_unit_key(record: dict[str, Any]) -> str:
    """Return the unique annotation unit key for a sequence record."""
    return record["sequence_annotation_id"]


def derive_trust_level(record: dict[str, Any]) -> str:
    """Derive trust_level from ordered_candidate_ids."""
    cids = record.get("ordered_candidate_ids", [])
    if cids:
        return cids[0].rsplit("_", 1)[-1]
    return "unknown"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _labels_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check if two row annotations match on all core labels."""
    for fld in ALL_CORE_LABELS:
        if a.get(fld) != b.get(fld):
            return False
    return True


def _sequence_labels_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check if two sequence annotations match on the full semantic tuple.

    The semantic tuple is:
      (sequence_reconstructs_target, earliest_reconstruction_step,
       reconstruction_strength)
    """
    if a.get("sequence_reconstructs_target") != b.get("sequence_reconstructs_target"):
        return False
    if a.get("earliest_reconstruction_step") != b.get("earliest_reconstruction_step"):
        return False
    if a.get("reconstruction_strength") != b.get("reconstruction_strength"):
        return False
    return True


# ---------------------------------------------------------------------------
# J3 provider
# ---------------------------------------------------------------------------


def _build_j3_prompt(candidate: dict[str, Any]) -> tuple[str, str]:
    """Build blinded (system_prompt, user_prompt) for J3 row adjudication."""
    from experiments.trustparadox_u.empirical_annotation import build_row_prompt
    return build_row_prompt(candidate)


def _build_j3_sequence_prompt(
    seq_annotation_id: str,
    j_seq_by_id: dict[str, Any],
    cand_by_cid: dict[str, Any],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for J3 sequence adjudication.

    Uses the frozen build_sequence_prompt() from empirical_annotation.
    """
    from experiments.trustparadox_u.empirical_annotation import build_sequence_prompt
    j_seq = j_seq_by_id.get(seq_annotation_id, {})
    ordered_cids = j_seq.get("ordered_candidate_ids", [])
    members = [cand_by_cid[cid] for cid in ordered_cids if cid in cand_by_cid]
    if not members:
        raise ValueError(f"No candidate members found for sequence {seq_annotation_id}")
    return build_sequence_prompt(members)


def _call_j3(
    system_prompt: str,
    user_prompt: str,
    *,
    api_base: str,
    api_key: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call J3 (qwen-plus) and parse the JSON annotation response.

    Item 11: Uses parse_annotation_response + validate_row_label for
    future-correctness instead of raw json.loads.
    """
    from litellm import completion
    from experiments.trustparadox_u.empirical_annotation import (
        parse_annotation_response,
        validate_row_label,
        AnnotationParseError,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            start = time.monotonic()
            resp = completion(
                model=f"openai/{J3_MODEL}",
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                api_base=api_base,
                api_key=api_key,
                timeout=60,
            )
            elapsed_ms = (time.monotonic() - start) * 1000.0
            raw_text = resp.choices[0].message.content or ""
            model_returned = resp.model or J3_MODEL

            # Item 11: hardened parsing via parse_annotation_response
            labels = parse_annotation_response(raw_text)
            # Validate row label schema
            validate_row_label(labels)

            return {
                "status": "success",
                "labels": labels,
                "raw_response": raw_text,
                "model_returned": model_returned,
                "latency_ms": elapsed_ms,
                "retry_index": attempt,
                "provider_request_id": getattr(resp, "id", ""),
            }

        except (AnnotationParseError, ValueError) as exc:
            # Schema/parse errors are not transient — stop retrying
            return {
                "status": "parse_error",
                "labels": {},
                "raw_response": raw_text if 'raw_text' in dir() else "",
                "model_returned": model_returned if 'model_returned' in dir() else J3_MODEL,
                "latency_ms": 0.0,
                "retry_index": attempt,
                "provider_request_id": "",
                "error": str(exc)[:200],
            }
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            is_transient = any(
                kw in exc_str
                for kw in ("timeout", "timed out", "rate limit", "429", "500", "502", "503", "504")
            )
            if not is_transient or attempt >= max_retries - 1:
                break
            time.sleep(2 ** attempt)

    return {
        "status": "provider_error",
        "labels": {},
        "raw_response": "",
        "model_returned": J3_MODEL,
        "latency_ms": 0.0,
        "retry_index": 0,
        "provider_request_id": "",
        "error": str(last_exc)[:200],
    }


def _call_j3_sequence(
    system_prompt: str,
    user_prompt: str,
    *,
    api_base: str,
    api_key: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call J3 for sequence adjudication.

    Item 10: Uses parse_annotation_response + validate_sequence_label.
    """
    from litellm import completion
    from experiments.trustparadox_u.empirical_annotation import (
        parse_annotation_response,
        validate_sequence_label,
        AnnotationParseError,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_exc: Exception | None = None
    raw_text = ""
    model_returned = J3_MODEL
    for attempt in range(max_retries):
        try:
            start = time.monotonic()
            resp = completion(
                model=f"openai/{J3_MODEL}",
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                api_base=api_base,
                api_key=api_key,
                timeout=60,
            )
            elapsed_ms = (time.monotonic() - start) * 1000.0
            raw_text = resp.choices[0].message.content or ""
            model_returned = resp.model or J3_MODEL

            labels = parse_annotation_response(raw_text)
            validate_sequence_label(labels)

            return {
                "status": "success",
                "labels": labels,
                "raw_response": raw_text,
                "model_returned": model_returned,
                "latency_ms": elapsed_ms,
                "retry_index": attempt,
                "provider_request_id": getattr(resp, "id", ""),
            }

        except (AnnotationParseError, ValueError) as exc:
            return {
                "status": "parse_error",
                "labels": {},
                "raw_response": raw_text,
                "model_returned": model_returned,
                "latency_ms": 0.0,
                "retry_index": attempt,
                "provider_request_id": "",
                "error": str(exc)[:200],
            }
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            is_transient = any(
                kw in exc_str
                for kw in ("timeout", "timed out", "rate limit", "429", "500", "502", "503", "504")
            )
            if not is_transient or attempt >= max_retries - 1:
                break
            time.sleep(2 ** attempt)

    return {
        "status": "provider_error",
        "labels": {},
        "raw_response": "",
        "model_returned": J3_MODEL,
        "latency_ms": 0.0,
        "retry_index": 0,
        "provider_request_id": "",
        "error": str(last_exc)[:200],
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_test_adjudication(
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Sec 38-42: Test J3 adjudication workflow."""
    print("=" * 60)
    print("E4-003 TEST ADJUDICATION")
    print("=" * 60)

    # --- Resolve API key ---
    api_key: str | None = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    if not api_key:
        litellm_config = PROJECT_ROOT / "litellm_config.yaml"
        if litellm_config.exists():
            import yaml
            with open(litellm_config) as f:
                cfg = yaml.safe_load(f)
            if cfg and "model_list" in cfg:
                for entry in cfg["model_list"]:
                    params = entry.get("litellm_params", {})
                    if params.get("api_key"):
                        api_key = params["api_key"]
                        break
    if not api_key:
        print("ERROR: No API key found. Set --api-key-env or configure litellm_config.yaml")
        sys.exit(1)

    if not api_base:
        api_base = "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"

    # --- Verify model role separation ---
    from experiments.trustparadox_u.empirical_annotation import (
        MODEL_GENERATOR, MODEL_PRIMARY, MODEL_SECONDARY,
        ROLE_ADJUDICATOR, MODEL_ADJUDICATOR,
        verify_model_role_separation,
    )

    violations = verify_model_role_separation(
        generator=MODEL_GENERATOR,
        primary=MODEL_PRIMARY,
        secondary=MODEL_SECONDARY,
        adjudicator=MODEL_ADJUDICATOR,
    )
    if violations:
        print(f"ERROR: Model role separation violations: {violations}")
        sys.exit(1)
    print(f"J3 model: {MODEL_ADJUDICATOR} (distinct from G/J/J2)")

    # --- Load inputs ---
    j_rows = _load_jsonl(_J_ROW_LABELS_PATH)
    j2_rows = _load_jsonl(_J2_ROW_LABELS_PATH)
    j_seqs = _load_jsonl(_J_SEQ_LABELS_PATH)
    j2_seqs = _load_jsonl(_J2_SEQ_LABELS_PATH)
    candidates = _load_jsonl(_TEST_CANDIDATES_PATH)
    review_queue = _load_jsonl(_REVIEW_QUEUE_PATH)

    j_by_cid = {r["candidate_id"]: r for r in j_rows}
    j2_by_cid = {r["candidate_id"]: r for r in j2_rows}
    cand_by_cid = {c["candidate_id"]: c for c in candidates}

    print(f"J rows: {len(j_rows)}")
    print(f"J2 rows: {len(j2_rows)}")
    print(f"J sequences: {len(j_seqs)}")
    print(f"J2 sequences: {len(j2_seqs)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Review queue: {len(review_queue)} items")

    review_queue_sha = _sha256_file(_REVIEW_QUEUE_PATH)

    # --- Sec 38: J3 adjudication of review-queue rows ---
    print(f"\n--- Sec 38: J3 Adjudication ---")

    adjudication_records: list[dict[str, Any]] = []
    resolution_counts = {
        "consensus_retained": 0,
        "resolved_by_j3_matching_j": 0,
        "resolved_by_j3_matching_j2": 0,
        "still_unresolved": 0,
    }

    review_row_items = [q for q in review_queue if q["item_type"] == "row"]
    for i, rq_item in enumerate(review_row_items):
        cid = rq_item["candidate_id"]
        j_label = j_by_cid.get(cid)
        j2_label = j2_by_cid.get(cid)
        candidate = cand_by_cid.get(cid)

        if not j_label or not j2_label or not candidate:
            print(f"  WARNING: Missing data for {cid}, skipping")
            continue

        j_j2_agree = _labels_match(j_label, j2_label)

        if j_j2_agree:
            record = {
                "adjudication_id": f"adj_{_sha256_str(cid + J3_ROLE)[:16]}",
                "candidate_id": cid,
                "candidate_content_sha256": j_label.get("candidate_content_sha256", ""),
                "frozen_corpus_manifest_sha256": _FROZEN_CORPUS_SHA,
                "review_queue_sha256": review_queue_sha,
                "review_reasons": rq_item.get("review_reasons", []),
                "j_label": {fld: j_label.get(fld) for fld in ALL_CORE_LABELS},
                "j2_label": {fld: j2_label.get(fld) for fld in ALL_CORE_LABELS},
                "j3_label": None,
                "j3_called": False,
                "resolution_source": "llm_consensus",
                "resolution_status": "consensus_retained",
                "j_agreed": True,
                "j2_agreed": True,
                "adjudicator_role": J3_ROLE,
                "adjudicator_model": MODEL_ADJUDICATOR,
                "adjudicator_provider": J3_PROVIDER,
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "adjudication_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
                "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            }
            resolution_counts["consensus_retained"] += 1
        else:
            system_prompt, user_prompt = _build_j3_prompt(candidate)
            j3_result = _call_j3(system_prompt, user_prompt, api_base=api_base, api_key=api_key)

            j3_labels = j3_result.get("labels", {})
            j3_success = j3_result["status"] == "success"

            j3_matches_j = j3_success and _labels_match(j3_labels, j_label)
            j3_matches_j2 = j3_success and _labels_match(j3_labels, j2_label)

            if j3_matches_j:
                resolution_source = "llm_adjudication"
                resolution_status = "resolved_by_j3_matching_j"
                resolution_counts["resolved_by_j3_matching_j"] += 1
            elif j3_matches_j2:
                resolution_source = "llm_adjudication"
                resolution_status = "resolved_by_j3_matching_j2"
                resolution_counts["resolved_by_j3_matching_j2"] += 1
            else:
                resolution_source = "unresolved"
                resolution_status = "still_unresolved"
                resolution_counts["still_unresolved"] += 1

            record = {
                "adjudication_id": f"adj_{_sha256_str(cid + J3_ROLE)[:16]}",
                "candidate_id": cid,
                "candidate_content_sha256": j_label.get("candidate_content_sha256", ""),
                "frozen_corpus_manifest_sha256": _FROZEN_CORPUS_SHA,
                "review_queue_sha256": review_queue_sha,
                "review_reasons": rq_item.get("review_reasons", []),
                "j_label": {fld: j_label.get(fld) for fld in ALL_CORE_LABELS},
                "j2_label": {fld: j2_label.get(fld) for fld in ALL_CORE_LABELS},
                "j3_label": {fld: j3_labels.get(fld) for fld in ALL_CORE_LABELS} if j3_success else None,
                "j3_called": True,
                "j3_status": j3_result["status"],
                "j3_raw_response_sha256": _sha256_str(j3_result.get("raw_response", "")),
                "j3_model_returned": j3_result.get("model_returned", ""),
                "j3_latency_ms": j3_result.get("latency_ms", 0.0),
                "j3_retry_index": j3_result.get("retry_index", 0),
                "j3_provider_request_id": j3_result.get("provider_request_id", ""),
                "resolution_source": resolution_source,
                "resolution_status": resolution_status,
                "j_agreed": False,
                "j2_agreed": False,
                "adjudicator_role": J3_ROLE,
                "adjudicator_model": MODEL_ADJUDICATOR,
                "adjudicator_provider": J3_PROVIDER,
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "adjudication_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
                "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            }

        adjudication_records.append(record)
        status = record["resolution_status"]
        print(f"  [{i+1:3d}/{len(review_row_items)}] {cid[-40:]:40s} -> {status}")

    # --- Item 10: Sequence J3 adjudication path ---
    review_seq_items = [q for q in review_queue if q["item_type"] == "sequence"]
    seq_adjudication_records: list[dict[str, Any]] = []
    j_seq_by_id_local = {sequence_unit_key(r): r for r in j_seqs}
    j2_seq_by_id_local = {sequence_unit_key(r): r for r in j2_seqs}

    seq_resolution_counts = {
        "consensus_retained": 0,
        "resolved_by_j3_matching_j": 0,
        "resolved_by_j3_matching_j2": 0,
        "still_unresolved": 0,
    }

    for i, sq_item in enumerate(review_seq_items):
        sid = sq_item["sequence_annotation_id"]
        j_seq = j_seq_by_id_local.get(sid)
        j2_seq = j2_seq_by_id_local.get(sid)

        if not j_seq or not j2_seq:
            print(f"  WARNING: Missing sequence data for {sid}, skipping")
            continue

        j_j2_seq_agree = _sequence_labels_match(j_seq, j2_seq)

        if j_j2_seq_agree:
            seq_record = {
                "adjudication_id": f"adj_seq_{_sha256_str(sid + J3_ROLE)[:16]}",
                "sequence_annotation_id": sid,
                "item_type": "sequence",
                "j3_called": False,
                "resolution_source": "llm_consensus",
                "resolution_status": "consensus_retained",
                "adjudicator_role": J3_ROLE,
                "adjudicator_model": MODEL_ADJUDICATOR,
                "adjudicator_provider": J3_PROVIDER,
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "adjudication_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
                "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            }
            seq_resolution_counts["consensus_retained"] += 1
        else:
            # Build sequence prompt and call J3
            sys_prompt, usr_prompt = _build_j3_sequence_prompt(
                sid, j_seq_by_id_local, cand_by_cid
            )
            j3_seq_result = _call_j3_sequence(
                sys_prompt, usr_prompt, api_base=api_base, api_key=api_key
            )

            j3_seq_labels = j3_seq_result.get("labels", {})
            j3_seq_success = j3_seq_result["status"] == "success"

            j3_matches_j_seq = j3_seq_success and _sequence_labels_match(j3_seq_labels, j_seq)
            j3_matches_j2_seq = j3_seq_success and _sequence_labels_match(j3_seq_labels, j2_seq)

            if j3_matches_j_seq:
                res_source = "llm_adjudication"
                res_status = "resolved_by_j3_matching_j"
                seq_resolution_counts["resolved_by_j3_matching_j"] += 1
            elif j3_matches_j2_seq:
                res_source = "llm_adjudication"
                res_status = "resolved_by_j3_matching_j2"
                seq_resolution_counts["resolved_by_j3_matching_j2"] += 1
            else:
                res_source = "unresolved"
                res_status = "still_unresolved"
                seq_resolution_counts["still_unresolved"] += 1

            seq_record = {
                "adjudication_id": f"adj_seq_{_sha256_str(sid + J3_ROLE)[:16]}",
                "sequence_annotation_id": sid,
                "item_type": "sequence",
                "j3_called": True,
                "j3_status": j3_seq_result["status"],
                "j3_raw_response_sha256": _sha256_str(j3_seq_result.get("raw_response", "")),
                "j3_model_returned": j3_seq_result.get("model_returned", ""),
                "j3_latency_ms": j3_seq_result.get("latency_ms", 0.0),
                "j3_retry_index": j3_seq_result.get("retry_index", 0),
                "j3_provider_request_id": j3_seq_result.get("provider_request_id", ""),
                "resolution_source": res_source,
                "resolution_status": res_status,
                "adjudicator_role": J3_ROLE,
                "adjudicator_model": MODEL_ADJUDICATOR,
                "adjudicator_provider": J3_PROVIDER,
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "adjudication_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
                "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            }

        seq_adjudication_records.append(seq_record)
        status = seq_record["resolution_status"]
        print(f"  SEQ [{i+1:3d}/{len(review_seq_items)}] {sid[-40:]:40s} -> {status}")

    print(f"Sequence review items: {len(review_seq_items)} "
          f"(consensus={seq_resolution_counts['consensus_retained']}, "
          f"j3_calls={sum(1 for r in seq_adjudication_records if r['j3_called'])})")

    # Merge sequence adjudication stats into resolution_counts
    resolution_counts["seq_consensus_retained"] = seq_resolution_counts["consensus_retained"]
    resolution_counts["seq_resolved_by_j3_matching_j"] = seq_resolution_counts["resolved_by_j3_matching_j"]
    resolution_counts["seq_resolved_by_j3_matching_j2"] = seq_resolution_counts["resolved_by_j3_matching_j2"]
    resolution_counts["seq_still_unresolved"] = seq_resolution_counts["still_unresolved"]

    _write_jsonl(_LLM_ADJUDICATION_PATH, adjudication_records)
    print(f"Wrote {len(adjudication_records)} adjudication records to {_LLM_ADJUDICATION_PATH.name}")

    # --- Sec 39: Construct final test labels ---
    print(f"\n--- Sec 39: Final Test Labels ---")
    adj_by_cid = {r["candidate_id"]: r for r in adjudication_records}
    final_labels: list[dict[str, Any]] = []

    for cid in sorted(j_by_cid.keys()):
        j_label = j_by_cid[cid]
        j2_label = j2_by_cid.get(cid, {})
        adj = adj_by_cid.get(cid)

        if adj:
            if adj["resolution_source"] == "llm_consensus":
                final = {fld: j_label.get(fld) for fld in ALL_CORE_LABELS}
                source = "llm_consensus"
                j_agreed = True
                j2_agreed = True
            elif adj["resolution_source"] == "llm_adjudication":
                if adj["resolution_status"] == "resolved_by_j3_matching_j":
                    final = {fld: j_label.get(fld) for fld in ALL_CORE_LABELS}
                elif adj["resolution_status"] == "resolved_by_j3_matching_j2":
                    final = {fld: j2_label.get(fld) for fld in ALL_CORE_LABELS}
                else:
                    final = {fld: None for fld in ALL_CORE_LABELS}
                source = "llm_adjudication"
                j_agreed = False
                j2_agreed = False
            else:
                final = {fld: None for fld in ALL_CORE_LABELS}
                source = "unresolved"
                j_agreed = False
                j2_agreed = False
        else:
            j_j2_agree = _labels_match(j_label, j2_label)
            if j_j2_agree:
                final = {fld: j_label.get(fld) for fld in ALL_CORE_LABELS}
                source = "llm_consensus"
            else:
                final = {fld: None for fld in ALL_CORE_LABELS}
                source = "unresolved"
            j_agreed = j_j2_agree
            j2_agreed = j_j2_agree

        final_record = {
            "candidate_id": cid,
            "final_target_relevant": final.get("target_relevant"),
            "final_target_leakage": final.get("target_leakage"),
            "final_positive_entailment": final.get("positive_entailment"),
            "final_task_useful": final.get("task_useful"),
            "final_leakage_strength": final.get("leakage_strength"),
            "resolution_source": source,
            "resolution_status": "resolved" if source != "unresolved" else "unresolved",
            "j_agreed": j_agreed,
            "j2_agreed": j2_agreed,
            "frozen_corpus_manifest_sha256": _FROZEN_CORPUS_SHA,
            "annotation_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
        }
        final_labels.append(final_record)

    _write_jsonl(_FINAL_LABELS_PATH, final_labels)
    print(f"Wrote {len(final_labels)} final labels to {_FINAL_LABELS_PATH.name}")

    # --- Sequence final labels — using sequence_annotation_id (§4, §17) ---
    # Validate sequence uniqueness
    j_seq_ann_ids = [sequence_unit_key(r) for r in j_seqs]
    j2_seq_ann_ids = [sequence_unit_key(r) for r in j2_seqs]
    if len(j_seq_ann_ids) != len(set(j_seq_ann_ids)):
        print("ERROR: Duplicate sequence_annotation_id in J sequences")
        sys.exit(1)
    if len(j2_seq_ann_ids) != len(set(j2_seq_ann_ids)):
        print("ERROR: Duplicate sequence_annotation_id in J2 sequences")
        sys.exit(1)
    if len(j_seq_ann_ids) != EXPECTED_SEQUENCES:
        print(f"ERROR: Expected {EXPECTED_SEQUENCES} J sequence annotations, got {len(j_seq_ann_ids)}")
        sys.exit(1)
    if len(j2_seq_ann_ids) != EXPECTED_SEQUENCES:
        print(f"ERROR: Expected {EXPECTED_SEQUENCES} J2 sequence annotations, got {len(j2_seq_ann_ids)}")
        sys.exit(1)

    # Validate cross-annotator identity coverage
    j_seq_id_set = set(j_seq_ann_ids)
    j2_seq_id_set = set(j2_seq_ann_ids)
    if j_seq_id_set != j2_seq_id_set:
        print(f"ERROR: J/J2 sequence_annotation_id sets differ")
        print(f"  Unmatched J: {j_seq_id_set - j2_seq_id_set}")
        print(f"  Unmatched J2: {j2_seq_id_set - j_seq_id_set}")
        sys.exit(1)

    final_seq_labels: list[dict[str, Any]] = []
    j_seq_by_id = {sequence_unit_key(r): r for r in j_seqs}
    j2_seq_by_id = {sequence_unit_key(r): r for r in j2_seqs}

    for sid in sorted(j_seq_by_id.keys()):
        j_seq = j_seq_by_id[sid]
        j2_seq = j2_seq_by_id.get(sid, {})
        seq_agree = _sequence_labels_match(j_seq, j2_seq)

        # Preserve trust-conditioned identity
        trust_level = derive_trust_level(j_seq)

        final_seq = {
            "sequence_annotation_id": sid,
            "sequence_family_id": j_seq.get("sequence_family_id", ""),
            "trust_level": trust_level,
            "scenario_id": j_seq.get("scenario_id", ""),
            "secret_variant_id": j_seq.get("secret_variant_id", ""),
            "ordered_candidate_ids": j_seq.get("ordered_candidate_ids", []),
            "final_sequence_reconstructs_target": j_seq.get("sequence_reconstructs_target") if seq_agree else None,
            "final_earliest_reconstruction_step": j_seq.get("earliest_reconstruction_step") if seq_agree else None,
            "final_reconstruction_strength": j_seq.get("reconstruction_strength") if seq_agree else None,
            "resolution_source": "llm_consensus" if seq_agree else "unresolved",
            "resolution_status": "resolved" if seq_agree else "unresolved",
            "j_agreed": seq_agree,
            "j2_agreed": seq_agree,
            "frozen_corpus_manifest_sha256": _FROZEN_CORPUS_SHA,
            "annotation_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
            "sequence_content_sha256": j_seq.get("sequence_content_sha256", ""),
        }
        final_seq_labels.append(final_seq)

    _write_jsonl(_FINAL_SEQ_LABELS_PATH, final_seq_labels)
    print(f"Wrote {len(final_seq_labels)} final sequence labels to {_FINAL_SEQ_LABELS_PATH.name}")

    # --- Statistics ---
    review_rows = [q for q in review_queue if q["item_type"] == "row"]
    review_seqs = [q for q in review_queue if q["item_type"] == "sequence"]

    unresolved_rows = sum(1 for r in final_labels if r["resolution_source"] == "unresolved")
    consensus_rows = sum(1 for r in final_labels if r["resolution_source"] == "llm_consensus")
    adjudicated_rows = sum(1 for r in final_labels if r["resolution_source"] == "llm_adjudication")
    unresolved_seqs = sum(1 for r in final_seq_labels if r["resolution_source"] == "unresolved")

    unresolved_row_rate = unresolved_rows / len(final_labels) if final_labels else 1.0
    unresolved_seq_rate = unresolved_seqs / len(final_seq_labels) if final_seq_labels else 1.0

    disagreement_recs = [r for r in adjudication_records if r["j3_called"]]
    j3_vs_j = sum(1 for r in disagreement_recs if r["resolution_status"] == "resolved_by_j3_matching_j")
    j3_vs_j2 = sum(1 for r in disagreement_recs if r["resolution_status"] == "resolved_by_j3_matching_j2")
    j3_vs_both = sum(1 for r in disagreement_recs if r["resolution_status"] == "still_unresolved")

    # --- Write adjudication manifest ---
    llm_adj_sha = _sha256_file(_LLM_ADJUDICATION_PATH)
    final_labels_sha = _sha256_file(_FINAL_LABELS_PATH)
    final_seq_sha = _sha256_file(_FINAL_SEQ_LABELS_PATH)

    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        code_commit = "unknown"

    adj_manifest = {
        "schema_version": "1.0",
        "description": "E4-003: Test annotation adjudication manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "review_queue_count": len(review_queue),
        "review_row_count": len(review_rows),
        "review_sequence_count": len(review_seqs),
        "adjudicated_count": len(adjudication_records),
        "disagreement_rows": len(disagreement_recs),
        "consensus_rows_in_queue": len(adjudication_records) - len(disagreement_recs),
        "resolution_counts": resolution_counts,
        "final_label_counts": {
            "total_rows": len(final_labels),
            "consensus_rows": consensus_rows,
            "adjudicated_rows": adjudicated_rows,
            "unresolved_rows": unresolved_rows,
            "total_sequences": len(final_seq_labels),
            "unresolved_sequences": unresolved_seqs,
        },
        "unresolved_row_rate": round(unresolved_row_rate, 4),
        "unresolved_sequence_rate": round(unresolved_seq_rate, 4),
        "j3_model": MODEL_ADJUDICATOR,
        "j3_provider": J3_PROVIDER,
        "j3_role": J3_ROLE,
        "adjudication_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
        "review_queue_sha256": review_queue_sha,
        "llm_adjudication_sha256": llm_adj_sha,
        "final_adjudicated_labels_sha256": final_labels_sha,
        "final_sequence_labels_sha256": final_seq_sha,
        "frozen_corpus_manifest_sha256": _FROZEN_CORPUS_SHA,
        "test_annotation_source_commit": code_commit,
        "j3_vs_j_agreement_count": j3_vs_j,
        "j3_vs_j2_agreement_count": j3_vs_j2,
        "j3_vs_both_disagree_count": j3_vs_both,
    }

    _write_json(_ADJUDICATION_MANIFEST_PATH, adj_manifest)
    print(f"Wrote adjudication manifest to {_ADJUDICATION_MANIFEST_PATH.name}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TEST ADJUDICATION SUMMARY")
    print("=" * 60)
    print(f"Review queue: {len(review_queue)} items ({len(review_rows)} rows, {len(review_seqs)} sequences)")
    print(f"Adjudicated rows: {len(adjudication_records)}")
    print(f"  Consensus retained: {resolution_counts['consensus_retained']}")
    print(f"  J3 resolved (J):    {resolution_counts['resolved_by_j3_matching_j']}")
    print(f"  J3 resolved (J2):   {resolution_counts['resolved_by_j3_matching_j2']}")
    print(f"  Still unresolved:   {resolution_counts['still_unresolved']}")
    print(f"\nFinal rows: {len(final_labels)}")
    print(f"  Consensus: {consensus_rows}")
    print(f"  Adjudicated: {adjudicated_rows}")
    print(f"  Unresolved: {unresolved_rows} ({unresolved_row_rate:.4f})")
    print(f"Final sequences: {len(final_seq_labels)}")
    print(f"  Unresolved: {unresolved_seqs} ({unresolved_seq_rate:.4f})")
    print(f"\nJ3 vs J: {j3_vs_j}/{len(disagreement_recs)}")
    print(f"J3 vs J2: {j3_vs_j2}/{len(disagreement_recs)}")
    print(f"J3 disagrees both: {j3_vs_both}/{len(disagreement_recs)}")

    row_gate_pass = unresolved_row_rate <= MAX_UNRESOLVED_RATE
    seq_gate_pass = unresolved_seq_rate <= MAX_UNRESOLVED_RATE
    print(f"\nUnresolved row gate (<=10%): {'PASS' if row_gate_pass else 'FAIL'}")
    print(f"Unresolved sequence gate (<=10%): {'PASS' if seq_gate_pass else 'FAIL'}")

    return {
        "row_gate_pass": row_gate_pass,
        "seq_gate_pass": seq_gate_pass,
        "unresolved_rows": unresolved_rows,
        "unresolved_seqs": unresolved_seqs,
        "review_queue_count": len(review_queue),
        "adjudicated_count": len(adjudication_records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run J3 adjudication on test review queue (E4-003)")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument("--api-key-env", type=str, default=None, help="Environment variable name for API key")
    args = parser.parse_args()
    run_test_adjudication(api_base=args.api_base, api_key_env=args.api_key_env)

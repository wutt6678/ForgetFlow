#!/usr/bin/env python3
"""E4-001A: Development annotation adjudication runner.

Adjudicates the 38 review-queue rows using a third independent LLM (J3).

J3 model constraints (Sec 21):
  J3 != generator (qwen3.7-plus)
  J3 != J         (qwen3.8-max)
  J3 != J2        (glm-5.2)

Resolution rule (Sec 21):
  J == J2  -> consensus (llm_consensus)
  J != J2  -> obtain independent J3 label
    J3 == J  -> resolve to J
    J3 == J2 -> resolve to J2
    otherwise -> unresolved

Usage:
  PYTHONPATH=. python scripts/run_adjudication.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_V3_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "development_v3"
_CORPUS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_DEV_CANDIDATES_PATH = _CORPUS_DIR / "development" / "accepted_candidates.jsonl"

# Input files (immutable v3 evidence)
_REVIEW_QUEUE_PATH = _V3_DIR / "review_queue.jsonl"
_J_ROW_LABELS_PATH = _V3_DIR / "row_annotations.jsonl"
_J2_ROW_LABELS_PATH = _V3_DIR / "secondary_row_annotations.jsonl"

# Output files
_LLM_ADJUDICATION_PATH = _V3_DIR / "llm_adjudication.jsonl"
_FINAL_LABELS_PATH = _V3_DIR / "final_adjudicated_labels.jsonl"
_ADJUDICATION_MANIFEST_PATH = _V3_DIR / "adjudication_manifest.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_BINARY_LABELS = ("target_relevant", "target_leakage", "positive_entailment", "task_useful")
ALL_CORE_LABELS = CORE_BINARY_LABELS + ("leakage_strength",)

# J3 adjudicator identity (Sec 21)
J3_ROLE = "J3"
J3_MODEL = "qwen-plus"
J3_PROVIDER = "litellm"
ADJUDICATION_PROTOCOL_VERSION = "1.0"
ANNOTATION_SCHEMA_VERSION = "1.0"

# Frozen corpus manifest expected SHA
_FROZEN_CORPUS_SHA = "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
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


# ---------------------------------------------------------------------------
# Prompt construction (reuses annotation module templates)
# ---------------------------------------------------------------------------


def _build_j3_prompt(candidate: dict[str, Any]) -> tuple[str, str]:
    """Build blinded (system_prompt, user_prompt) for J3 adjudication.

    Uses the same annotation view as J/J2 (Sec 20: prompts invariant).
    """
    from experiments.trustparadox_u.empirical_annotation import build_row_prompt

    return build_row_prompt(candidate)


def _call_j3(
    system_prompt: str,
    user_prompt: str,
    *,
    api_base: str,
    api_key: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call J3 (qwen-plus) and parse the JSON annotation response."""
    import re as _re

    from litellm import completion

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

            # Parse JSON from response
            text = raw_text.strip()
            md_match = _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, _re.DOTALL)
            if md_match:
                text = md_match.group(1).strip()

            labels = json.loads(text)
            if not isinstance(labels, dict):
                raise ValueError(f"J3 returned non-object: {type(labels).__name__}")

            return {
                "status": "success",
                "labels": labels,
                "raw_response": raw_text,
                "model_returned": model_returned,
                "latency_ms": elapsed_ms,
                "retry_index": attempt,
                "provider_request_id": getattr(resp, "id", ""),
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
# Label comparison
# ---------------------------------------------------------------------------


def _labels_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check if two label dicts agree on all core labels."""
    for fld in CORE_BINARY_LABELS:
        if a.get(fld) != b.get(fld):
            return False
    if a.get("leakage_strength") != b.get("leakage_strength"):
        return False
    return True


# ---------------------------------------------------------------------------
# Main adjudication logic
# ---------------------------------------------------------------------------


def run_adjudication() -> dict[str, Any]:
    """Run J3 adjudication on all 38 review-queue rows.

    Returns an adjudication summary report.
    """
    # --- Load inputs ---
    review_queue = _load_jsonl(_REVIEW_QUEUE_PATH)
    j_labels = _load_jsonl(_J_ROW_LABELS_PATH)
    j2_labels = _load_jsonl(_J2_ROW_LABELS_PATH)
    candidates = _load_jsonl(_DEV_CANDIDATES_PATH)

    print(f"Review queue: {len(review_queue)} items")
    print(f"J labels: {len(j_labels)} rows")
    print(f"J2 labels: {len(j2_labels)} rows")
    print(f"Candidates: {len(candidates)} rows")

    # Build lookups
    j_by_cid = {r["candidate_id"]: r for r in j_labels}
    j2_by_cid = {r["candidate_id"]: r for r in j2_labels}
    cand_by_cid = {c["candidate_id"]: c for c in candidates}

    # Compute input hashes
    review_queue_sha = _sha256_file(_REVIEW_QUEUE_PATH)
    frozen_corpus_sha = _FROZEN_CORPUS_SHA

    # --- API configuration ---
    api_base = "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # Read from litellm config as fallback
        litellm_config = _PROJECT_ROOT / "litellm_config.yaml"
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
        print("ERROR: No API key found. Set OPENAI_API_KEY or configure litellm_config.yaml")
        sys.exit(1)

    # --- Verify model role separation (Sec 21) ---
    from experiments.trustparadox_u.empirical_annotation import (
        MODEL_GENERATOR,
        MODEL_PRIMARY,
        MODEL_SECONDARY,
        verify_model_role_separation,
    )

    violations = verify_model_role_separation(
        generator=MODEL_GENERATOR,
        primary=MODEL_PRIMARY,
        secondary=MODEL_SECONDARY,
    )
    # Also check J3 is distinct
    for existing_role, existing_model in [
        ("generator", MODEL_GENERATOR),
        ("J", MODEL_PRIMARY),
        ("J2", MODEL_SECONDARY),
    ]:
        if J3_MODEL == existing_model:
            violations.append(f"J3 ({J3_MODEL}) == {existing_role} ({existing_model})")

    if violations:
        print(f"ERROR: Model role separation violations: {violations}")
        sys.exit(1)
    print(f"J3 model: {J3_MODEL} (distinct from G/J/J2)")

    # --- Adjudicate each review-queue row ---
    adjudication_records: list[dict[str, Any]] = []
    resolution_counts = {
        "consensus_retained": 0,
        "resolved_by_j3_matching_j": 0,
        "resolved_by_j3_matching_j2": 0,
        "still_unresolved": 0,
    }

    for i, rq_item in enumerate(review_queue):
        cid = rq_item["candidate_id"]
        j_label = j_by_cid.get(cid)
        j2_label = j2_by_cid.get(cid)
        candidate = cand_by_cid.get(cid)

        if not j_label or not j2_label:
            print(f"  WARNING: Missing J/J2 label for {cid}, skipping")
            continue
        if not candidate:
            print(f"  WARNING: Missing candidate for {cid}, skipping")
            continue

        # Check if J and J2 already agree
        j_j2_agree = _labels_match(j_label, j2_label)

        if j_j2_agree:
            # Consensus rows from review queue (the 2 uncertainty rows)
            # Retain consensus unless overturned
            record = {
                "adjudication_id": f"adj_{_sha256_str(cid + J3_ROLE)[:16]}",
                "candidate_id": cid,
                "candidate_content_sha256": j_label.get("candidate_content_sha256", ""),
                "frozen_corpus_manifest_sha256": frozen_corpus_sha,
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
                "adjudicator_model": J3_MODEL,
                "adjudicator_provider": J3_PROVIDER,
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "adjudication_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
                "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            }
            resolution_counts["consensus_retained"] += 1
        else:
            # Disagreement row — call J3
            system_prompt, user_prompt = _build_j3_prompt(candidate)
            j3_result = _call_j3(system_prompt, user_prompt, api_base=api_base, api_key=api_key)

            j3_labels = j3_result.get("labels", {})
            j3_success = j3_result["status"] == "success"

            # Determine resolution
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
                "frozen_corpus_manifest_sha256": frozen_corpus_sha,
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
                "adjudicator_model": J3_MODEL,
                "adjudicator_provider": J3_PROVIDER,
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "adjudication_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
                "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            }

        adjudication_records.append(record)
        status = record["resolution_status"]
        print(f"  [{i+1:2d}/38] {cid[-40:]:40s} -> {status}")

    # --- Write llm_adjudication.jsonl ---
    _write_jsonl(_LLM_ADJUDICATION_PATH, adjudication_records)
    print(f"\nWrote {len(adjudication_records)} adjudication records to {_LLM_ADJUDICATION_PATH.name}")

    # --- Build final_adjudicated_labels.jsonl (225 rows) ---
    adj_by_cid = {r["candidate_id"]: r for r in adjudication_records}
    final_labels: list[dict[str, Any]] = []

    for cid in sorted(j_by_cid.keys()):
        j_label = j_by_cid[cid]
        j2_label = j2_by_cid.get(cid, {})
        adj = adj_by_cid.get(cid)

        if adj:
            # This candidate was in the review queue
            if adj["resolution_source"] == "llm_consensus":
                # J == J2 consensus
                final = {fld: j_label.get(fld) for fld in ALL_CORE_LABELS}
                source = "llm_consensus"
                j_agreed = True
                j2_agreed = True
                human_overrode = False
            elif adj["resolution_source"] == "llm_adjudication":
                # J3 resolved
                if adj["resolution_status"] == "resolved_by_j3_matching_j":
                    final = {fld: j_label.get(fld) for fld in ALL_CORE_LABELS}
                elif adj["resolution_status"] == "resolved_by_j3_matching_j2":
                    final = {fld: j2_label.get(fld) for fld in ALL_CORE_LABELS}
                else:
                    final = {fld: None for fld in ALL_CORE_LABELS}
                source = "llm_adjudication"
                j_agreed = False
                j2_agreed = False
                human_overrode = False
            else:
                # Still unresolved
                final = {fld: None for fld in ALL_CORE_LABELS}
                source = "unresolved"
                j_agreed = False
                j2_agreed = False
                human_overrode = False
        else:
            # Not in review queue — J/J2 consensus
            j_j2_agree = _labels_match(j_label, j2_label)
            if j_j2_agree:
                final = {fld: j_label.get(fld) for fld in ALL_CORE_LABELS}
                source = "llm_consensus"
            else:
                # Shouldn't happen: non-queue disagreement
                final = {fld: None for fld in ALL_CORE_LABELS}
                source = "unresolved"
            j_agreed = j_j2_agree
            j2_agreed = j_j2_agree
            human_overrode = False

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
            "human_label_present": False,
            "human_overrode_both": human_overrode,
            "frozen_corpus_manifest_sha256": frozen_corpus_sha,
            "annotation_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
        }
        final_labels.append(final_record)

    _write_jsonl(_FINAL_LABELS_PATH, final_labels)
    print(f"Wrote {len(final_labels)} final labels to {_FINAL_LABELS_PATH.name}")

    # --- Compute statistics ---
    unresolved_count = sum(1 for r in final_labels if r["resolution_source"] == "unresolved")
    consensus_count = sum(1 for r in final_labels if r["resolution_source"] == "llm_consensus")
    adjudicated_count = sum(1 for r in final_labels if r["resolution_source"] == "llm_adjudication")

    # J3 vs J/J2 agreement on disagreement rows
    disagreement_recs = [r for r in adjudication_records if r["j3_called"]]
    j3_vs_j_agree = sum(
        1 for r in disagreement_recs if r["resolution_status"] == "resolved_by_j3_matching_j"
    )
    j3_vs_j2_agree = sum(
        1 for r in disagreement_recs if r["resolution_status"] == "resolved_by_j3_matching_j2"
    )
    j3_vs_both_disagree = sum(
        1 for r in disagreement_recs if r["resolution_status"] == "still_unresolved"
    )

    # --- Write adjudication manifest ---
    llm_adj_sha = _sha256_file(_LLM_ADJUDICATION_PATH)
    final_labels_sha = _sha256_file(_FINAL_LABELS_PATH)

    # Get current git commit
    import subprocess
    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT
        ).decode().strip()
    except Exception:
        code_commit = "unknown"

    manifest = {
        "schema_version": "1.0",
        "description": "E4-001A: Development annotation adjudication manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_queue_count": len(review_queue),
        "adjudicated_count": len(adjudication_records),
        "missing_adjudications": len(review_queue) - len(adjudication_records),
        "duplicate_adjudications": len(adjudication_records) - len(set(r["candidate_id"] for r in adjudication_records)),
        "disagreement_rows": len(disagreement_recs),
        "consensus_rows_in_queue": len(adjudication_records) - len(disagreement_recs),
        "resolution_counts": resolution_counts,
        "final_label_counts": {
            "total": len(final_labels),
            "consensus": consensus_count,
            "adjudicated": adjudicated_count,
            "unresolved": unresolved_count,
        },
        "unresolved_row_rate": round(unresolved_count / len(final_labels), 4) if final_labels else 1.0,
        "j3_model": J3_MODEL,
        "j3_provider": J3_PROVIDER,
        "j3_role": J3_ROLE,
        "adjudication_protocol_version": ADJUDICATION_PROTOCOL_VERSION,
        "review_queue_sha256": review_queue_sha,
        "llm_adjudication_sha256": llm_adj_sha,
        "final_adjudicated_labels_sha256": final_labels_sha,
        "frozen_corpus_manifest_sha256": frozen_corpus_sha,
        "annotation_code_commit": code_commit,
        "j3_vs_j_agreement_count": j3_vs_j_agree,
        "j3_vs_j2_agreement_count": j3_vs_j2_agree,
        "j3_vs_both_disagree_count": j3_vs_both_disagree,
    }

    _write_json(_ADJUDICATION_MANIFEST_PATH, manifest)
    print(f"Wrote adjudication manifest to {_ADJUDICATION_MANIFEST_PATH.name}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ADJUDICATION SUMMARY")
    print("=" * 60)
    print(f"Review queue rows:    {len(review_queue)}")
    print(f"Adjudicated rows:     {len(adjudication_records)}")
    print(f"  Consensus retained: {resolution_counts['consensus_retained']}")
    print(f"  J3 resolved (J):    {resolution_counts['resolved_by_j3_matching_j']}")
    print(f"  J3 resolved (J2):   {resolution_counts['resolved_by_j3_matching_j2']}")
    print(f"  Still unresolved:   {resolution_counts['still_unresolved']}")
    print(f"\nFinal labels:         {len(final_labels)}")
    print(f"  Consensus:          {consensus_count}")
    print(f"  Adjudicated:        {adjudicated_count}")
    print(f"  Unresolved:         {unresolved_count}")
    print(f"  Unresolved rate:    {unresolved_count / len(final_labels):.4f}")
    print(f"\nJ3 vs J agreement:  {j3_vs_j_agree}/{len(disagreement_recs)}")
    print(f"J3 vs J2 agreement: {j3_vs_j2_agree}/{len(disagreement_recs)}")
    print(f"J3 disagrees both:  {j3_vs_both_disagree}/{len(disagreement_recs)}")

    gate_pass = unresolved_count / max(len(final_labels), 1) <= 0.10
    print(f"\nUnresolved gate (<=10%): {'PASS' if gate_pass else 'FAIL'}")

    return manifest


if __name__ == "__main__":
    run_adjudication()

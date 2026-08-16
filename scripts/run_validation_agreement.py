#!/usr/bin/env python3
"""E4-002: Validation annotation agreement + adjudication runner.

Sec 32-39: Compute coverage, agreement, build review queue, run J3 adjudication.

Usage:
  PYTHONPATH=. python scripts/run_validation_agreement.py
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
_VAL_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "validation"
_CORPUS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_VAL_CANDIDATES_PATH = _CORPUS_DIR / "validation" / "accepted_candidates.jsonl"

# Input files
_J_ROW_LABELS_PATH = _VAL_DIR / "primary_row_annotations.jsonl"
_J2_ROW_LABELS_PATH = _VAL_DIR / "secondary_row_annotations.jsonl"
_J_SEQ_LABELS_PATH = _VAL_DIR / "primary_sequence_annotations.jsonl"
_J2_SEQ_LABELS_PATH = _VAL_DIR / "secondary_sequence_annotations.jsonl"

# Output files
_AGREEMENT_REPORT_PATH = _VAL_DIR / "validation_agreement_report.json"
_REVIEW_QUEUE_PATH = _VAL_DIR / "review_queue.jsonl"
_LLM_ADJUDICATION_PATH = _VAL_DIR / "llm_adjudication.jsonl"
_FINAL_LABELS_PATH = _VAL_DIR / "final_adjudicated_labels.jsonl"
_FINAL_SEQ_LABELS_PATH = _VAL_DIR / "final_sequence_labels.jsonl"
_ADJUDICATION_MANIFEST_PATH = _VAL_DIR / "adjudication_manifest.json"
_VALIDATION_MANIFEST_PATH = _VAL_DIR / "annotation_manifest.json"
_VALIDATION_GATE_PATH = _VAL_DIR / "validation_annotation_gate.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_BINARY_LABELS = ("target_relevant", "target_leakage", "positive_entailment", "task_useful")


# ---------------------------------------------------------------------------
# Sequence unit helper (§5)
# ---------------------------------------------------------------------------


def sequence_unit_key(record: dict[str, Any]) -> str:
    """Return the unique annotation unit key for a sequence record.

    The unique annotation unit is sequence_annotation_id, NOT sequence_family_id.
    Each family appears under multiple trust-conditioned realizations.
    """
    return record["sequence_annotation_id"]


def derive_trust_level(record: dict[str, Any]) -> str:
    """Derive trust_level from ordered_candidate_ids (last segment of first candidate)."""
    cids = record.get("ordered_candidate_ids", [])
    if cids:
        return cids[0].rsplit("_", 1)[-1]
    return "unknown"


ALL_CORE_LABELS = CORE_BINARY_LABELS + ("leakage_strength",)

# J3 adjudicator identity
J3_ROLE = "J3"
J3_MODEL = "qwen-plus"
J3_PROVIDER = "litellm"
ADJUDICATION_PROTOCOL_VERSION = "1.0"
ANNOTATION_SCHEMA_VERSION = "1.0"

# Frozen SHAs
_FROZEN_CORPUS_SHA = "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"
_FROZEN_PROTOCOL_SHA = "e4-002-frozen-protocol-sha"  # Will be computed from annotation_protocol_manifest.json

# Thresholds (Sec 36)
MIN_RAW_AGREEMENT = 0.85
MIN_KAPPA = 0.60
MAX_UNRESOLVED_RATE = 0.10


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


# ---------------------------------------------------------------------------
# Agreement computation (Sec 33-35)
# ---------------------------------------------------------------------------


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | str:
    """Compute Cohen's kappa for binary labels."""
    n = len(labels_a)
    if n != len(labels_b) or n == 0:
        return "not_estimable"

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    n_pos_a = sum(1 for x in labels_a if x == "True")
    n_neg_a = n - n_pos_a
    n_pos_b = sum(1 for x in labels_b if x == "True")
    n_neg_b = n - n_pos_b

    if n_pos_a == 0 or n_neg_a == 0 or n_pos_b == 0 or n_neg_b == 0:
        return "not_estimable"

    p_o = agree / n
    p_e = (n_pos_a * n_pos_b + n_neg_a * n_neg_b) / (n * n)

    if p_e == 1.0:
        return "not_estimable"

    return (p_o - p_e) / (1.0 - p_e)


def compute_binary_agreement(
    labels_a: list[bool], labels_b: list[bool]
) -> dict[str, Any]:
    """Sec 33: Compute agreement metrics for a binary label."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return {"raw_agreement": 0.0, "cohens_kappa": "not_estimable",
                "positive_agreement": 0.0, "negative_agreement": 0.0,
                "n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    str_a = [str(x) for x in labels_a]
    str_b = [str(x) for x in labels_b]

    agree = sum(1 for a, b in zip(str_a, str_b) if a == b)
    raw_agreement = agree / n

    kappa = _cohen_kappa(str_a, str_b)

    tp = sum(1 for a, b in zip(labels_a, labels_b) if a and b)
    fp = sum(1 for a, b in zip(labels_a, labels_b) if a and not b)
    fn = sum(1 for a, b in zip(labels_a, labels_b) if not a and b)
    tn = sum(1 for a, b in zip(labels_a, labels_b) if not a and not b)

    pos_agree = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    neg_agree = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0

    return {
        "raw_agreement": round(raw_agreement, 4),
        "cohens_kappa": round(kappa, 4) if isinstance(kappa, float) else kappa,
        "positive_agreement": round(pos_agree, 4),
        "negative_agreement": round(neg_agree, 4),
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_categorical_agreement(
    labels_a: list[str], labels_b: list[str]
) -> dict[str, Any]:
    """Sec 34: Categorical agreement for leakage_strength."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return {"exact_agreement": 0.0, "cohens_kappa": "not_estimable", "n": 0, "confusion_matrix": {}}

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    exact_agreement = agree / n

    categories = sorted(set(labels_a) | set(labels_b))
    if len(categories) <= 1:
        kappa = "not_estimable"
    else:
        confusion: dict[str, dict[str, int]] = {}
        for cat_a in categories:
            confusion[cat_a] = {}
            for cat_b in categories:
                confusion[cat_a][cat_b] = 0
        for a, b in zip(labels_a, labels_b):
            confusion[a][b] += 1

        p_o = agree / n
        p_e = sum(
            (sum(confusion[cat].values()) / n) *
            (sum(confusion[other][cat] for other in categories) / n)
            for cat in categories
        )
        if p_e == 1.0:
            kappa = "not_estimable"
        else:
            kappa = round((p_o - p_e) / (1.0 - p_e), 4)

    confusion_matrix: dict[str, dict[str, int]] = {}
    all_cats = sorted(set(labels_a) | set(labels_b))
    for ca in all_cats:
        confusion_matrix[ca] = {}
        for cb in all_cats:
            confusion_matrix[ca][cb] = sum(
                1 for a, b in zip(labels_a, labels_b) if a == ca and b == cb
            )

    return {
        "exact_agreement": round(exact_agreement, 4),
        "cohens_kappa": kappa,
        "n": n,
        "confusion_matrix": confusion_matrix,
    }


def compute_sequence_agreement(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sec 35: Sequence-level agreement.

    Uses sequence_annotation_id as the unique key (§4).
    Each family appears under multiple trust-conditioned realizations.
    """
    # §4: Use sequence_annotation_id, NOT sequence_family_id
    p_by_id = {sequence_unit_key(r): r for r in primary}
    s_by_id = {sequence_unit_key(r): r for r in secondary}
    common_ids = sorted(set(p_by_id.keys()) & set(s_by_id.keys()))
    unmatched_primary = sorted(set(p_by_id.keys()) - set(s_by_id.keys()))
    unmatched_secondary = sorted(set(s_by_id.keys()) - set(p_by_id.keys()))

    if not common_ids:
        return {"n": 0, "reconstruction_agreement": {}}

    recon_a = [p_by_id[sid]["sequence_reconstructs_target"] for sid in common_ids]
    recon_b = [s_by_id[sid]["sequence_reconstructs_target"] for sid in common_ids]

    result: dict[str, Any] = {
        "n": len(common_ids),
        "primary_count": len(p_by_id),
        "secondary_count": len(s_by_id),
        "common_sequence_annotation_ids": len(common_ids),
        "unmatched_primary": len(unmatched_primary),
        "unmatched_secondary": len(unmatched_secondary),
        "reconstruction_binary_agreement": compute_binary_agreement(recon_a, recon_b),
    }

    both_reconstruct = [
        sid for sid in common_ids
        if p_by_id[sid]["sequence_reconstructs_target"] and s_by_id[sid]["sequence_reconstructs_target"]
    ]
    if both_reconstruct:
        step_agree = sum(
            1 for sid in both_reconstruct
            if p_by_id[sid].get("earliest_reconstruction_step") == s_by_id[sid].get("earliest_reconstruction_step")
        )
        result["earliest_step_exact_agreement"] = round(
            step_agree / len(both_reconstruct), 4
        )
        result["earliest_step_n"] = len(both_reconstruct)
    else:
        result["earliest_step_exact_agreement"] = "not_estimable"
        result["earliest_step_n"] = 0

    return result


# ---------------------------------------------------------------------------
# Review queue (Sec 37)
# ---------------------------------------------------------------------------


def should_queue_for_review(
    primary_label: dict[str, Any],
    secondary_label: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Sec 37: Determine if an item should enter the review queue."""
    reasons: list[str] = []

    for fld in CORE_BINARY_LABELS:
        p_val = primary_label.get(fld)
        s_val = secondary_label.get(fld)
        if p_val is not None and s_val is not None and p_val != s_val:
            reasons.append(f"disagreement on {fld}")

    p_ls = primary_label.get("leakage_strength")
    s_ls = secondary_label.get("leakage_strength")
    if p_ls is not None and s_ls is not None and p_ls != s_ls:
        reasons.append("disagreement on leakage_strength")

    p_recon = primary_label.get("sequence_reconstructs_target")
    s_recon = secondary_label.get("sequence_reconstructs_target")
    if p_recon is not None and s_recon is not None and p_recon != s_recon:
        reasons.append("disagreement on sequence_reconstructs_target")

    if primary_label.get("uncertain"):
        reasons.append("primary uncertain")
    if secondary_label.get("uncertain"):
        reasons.append("secondary uncertain")

    p_conf = primary_label.get("confidence", 1.0)
    s_conf = secondary_label.get("confidence", 1.0)
    if isinstance(p_conf, (int, float)) and p_conf < 0.7:
        reasons.append(f"primary confidence < 0.7 ({p_conf})")
    if isinstance(s_conf, (int, float)) and s_conf < 0.7:
        reasons.append(f"secondary confidence < 0.7 ({s_conf})")

    return (len(reasons) > 0, reasons)


def build_review_queue(
    primary_row_labels: list[dict[str, Any]],
    secondary_row_labels: list[dict[str, Any]],
    primary_seq_labels: list[dict[str, Any]] | None = None,
    secondary_seq_labels: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Sec 37: Build the review queue from J/J2 label pairs."""
    queue: list[dict[str, Any]] = []

    p_by_cid = {r["candidate_id"]: r for r in primary_row_labels}
    s_by_cid = {r["candidate_id"]: r for r in secondary_row_labels}

    for cid in sorted(set(p_by_cid.keys()) & set(s_by_cid.keys())):
        should_queue, reasons = should_queue_for_review(p_by_cid[cid], s_by_cid[cid])
        if should_queue:
            queue.append({
                "item_type": "row",
                "candidate_id": cid,
                "primary_label": p_by_cid[cid],
                "secondary_label": s_by_cid[cid],
                "review_reasons": reasons,
            })

    if primary_seq_labels and secondary_seq_labels:
        # §4: Use sequence_annotation_id, NOT sequence_family_id
        p_by_id = {sequence_unit_key(r): r for r in primary_seq_labels}
        s_by_id = {sequence_unit_key(r): r for r in secondary_seq_labels}

        for sid in sorted(set(p_by_id.keys()) & set(s_by_id.keys())):
            should_queue, reasons = should_queue_for_review(p_by_id[sid], s_by_id[sid])
            if should_queue:
                queue.append({
                    "item_type": "sequence",
                    "sequence_annotation_id": sid,
                    "sequence_family_id": p_by_id[sid].get("sequence_family_id", ""),
                    "primary_label": p_by_id[sid],
                    "secondary_label": s_by_id[sid],
                    "review_reasons": reasons,
                })

    return queue


# ---------------------------------------------------------------------------
# J3 adjudication (Sec 38)
# ---------------------------------------------------------------------------


def _labels_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check if two label dicts agree on all core labels."""
    for fld in CORE_BINARY_LABELS:
        if a.get(fld) != b.get(fld):
            return False
    if a.get("leakage_strength") != b.get("leakage_strength"):
        return False
    return True


def _build_j3_prompt(candidate: dict[str, Any]) -> tuple[str, str]:
    """Build blinded (system_prompt, user_prompt) for J3 adjudication."""
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
# Main workflow
# ---------------------------------------------------------------------------


def run_validation_agreement_and_adjudication() -> dict[str, Any]:
    """Sec 32-44: Full validation agreement + adjudication workflow."""
    print("=" * 60)
    print("E4-002 VALIDATION AGREEMENT + ADJUDICATION")
    print("=" * 60)

    # --- Load inputs ---
    j_rows = _load_jsonl(_J_ROW_LABELS_PATH)
    j2_rows = _load_jsonl(_J2_ROW_LABELS_PATH)
    j_seqs = _load_jsonl(_J_SEQ_LABELS_PATH)
    j2_seqs = _load_jsonl(_J2_SEQ_LABELS_PATH)
    candidates = _load_jsonl(_VAL_CANDIDATES_PATH)

    print(f"J rows: {len(j_rows)}")
    print(f"J2 rows: {len(j2_rows)}")
    print(f"J sequences: {len(j_seqs)}")
    print(f"J2 sequences: {len(j2_seqs)}")
    print(f"Candidates: {len(candidates)}")

    # --- Sec 32: Coverage gate ---
    j_row_ids = {r["candidate_id"] for r in j_rows}
    j2_row_ids = {r["candidate_id"] for r in j2_rows}
    common_rows = j_row_ids & j2_row_ids
    unmatched_rows = (j_row_ids | j2_row_ids) - common_rows

    # Sequence coverage: 36 annotations across 12 families (3 steps each)
    j_seq_ann_ids = {r["sequence_annotation_id"] for r in j_seqs}
    j2_seq_ann_ids = {r["sequence_annotation_id"] for r in j2_seqs}
    common_seq_anns = j_seq_ann_ids & j2_seq_ann_ids
    unmatched_seq_anns = (j_seq_ann_ids | j2_seq_ann_ids) - common_seq_anns

    j_seq_ids = {r["sequence_family_id"] for r in j_seqs}
    j2_seq_ids = {r["sequence_family_id"] for r in j2_seqs}
    common_seqs = j_seq_ids & j2_seq_ids
    unmatched_seqs = (j_seq_ids | j2_seq_ids) - common_seqs

    print(f"\n--- Sec 32: Coverage Gate ---")
    print(f"J rows: {len(j_row_ids)} / 225")
    print(f"J2 rows: {len(j2_row_ids)} / 225")
    print(f"Common rows: {len(common_rows)}")
    print(f"Unmatched rows: {len(unmatched_rows)}")
    print(f"J seq annotations: {len(j_seq_ann_ids)} / 36")
    print(f"J2 seq annotations: {len(j2_seq_ann_ids)} / 36")
    print(f"Common seq annotations: {len(common_seq_anns)}")
    print(f"Unmatched seq annotations: {len(unmatched_seq_anns)}")
    print(f"J seq families: {len(j_seq_ids)} / 12")
    print(f"J2 seq families: {len(j2_seq_ids)} / 12")
    print(f"Common seq families: {len(common_seqs)}")
    print(f"Unmatched seq families: {len(unmatched_seqs)}")

    coverage_pass = (
        len(j_row_ids) == 225 and len(j2_row_ids) == 225 and
        len(common_rows) == 225 and len(unmatched_rows) == 0 and
        len(j_seq_ann_ids) == 36 and len(j2_seq_ann_ids) == 36 and
        len(common_seq_anns) == 36 and len(unmatched_seq_anns) == 0 and
        len(j_seq_ids) == 12 and len(j2_seq_ids) == 12 and
        len(common_seqs) == 12 and len(unmatched_seqs) == 0
    )
    print(f"Coverage gate: {'PASS' if coverage_pass else 'FAIL'}")

    if not coverage_pass:
        print("ERROR: Coverage gate failed. Aborting.")
        sys.exit(1)

    # --- Sec 33-35: Compute agreement ---
    j_by_cid = {r["candidate_id"]: r for r in j_rows}
    j2_by_cid = {r["candidate_id"]: r for r in j2_rows}
    cand_by_cid = {c["candidate_id"]: c for c in candidates}

    print(f"\n--- Sec 33-35: Agreement Computation ---")
    agreement_report: dict[str, Any] = {"n": len(common_rows)}

    for fld in CORE_BINARY_LABELS:
        j_vals = [j_by_cid[cid][fld] for cid in sorted(common_rows)]
        j2_vals = [j2_by_cid[cid][fld] for cid in sorted(common_rows)]
        agreement_report[fld] = compute_binary_agreement(j_vals, j2_vals)
        print(f"  {fld}: raw={agreement_report[fld]['raw_agreement']:.4f}, kappa={agreement_report[fld]['cohens_kappa']}")

    # Leakage strength (Sec 34)
    j_ls = [str(j_by_cid[cid].get("leakage_strength", "")) for cid in sorted(common_rows)]
    j2_ls = [str(j2_by_cid[cid].get("leakage_strength", "")) for cid in sorted(common_rows)]
    agreement_report["leakage_strength"] = compute_categorical_agreement(j_ls, j2_ls)
    print(f"  leakage_strength: exact={agreement_report['leakage_strength']['exact_agreement']:.4f}, kappa={agreement_report['leakage_strength']['cohens_kappa']}")

    # Sequence agreement (Sec 35)
    seq_agreement = compute_sequence_agreement(j_seqs, j2_seqs)
    agreement_report["sequence"] = seq_agreement
    print(f"  sequence_reconstructs_target: raw={seq_agreement['reconstruction_binary_agreement']['raw_agreement']:.4f}")

    _write_json(_AGREEMENT_REPORT_PATH, agreement_report)
    print(f"Wrote agreement report to {_AGREEMENT_REPORT_PATH.name}")

    # Check thresholds (Sec 36)
    thresholds_pass = True
    for fld in CORE_BINARY_LABELS:
        raw = agreement_report[fld]["raw_agreement"]
        if raw < MIN_RAW_AGREEMENT:
            print(f"  WARNING: {fld} raw agreement {raw:.4f} < {MIN_RAW_AGREEMENT}")
            thresholds_pass = False
        kappa = agreement_report[fld]["cohens_kappa"]
        if isinstance(kappa, float) and kappa < MIN_KAPPA:
            print(f"  WARNING: {fld} kappa {kappa:.4f} < {MIN_KAPPA}")
            thresholds_pass = False

    seq_raw = seq_agreement["reconstruction_binary_agreement"]["raw_agreement"]
    if seq_raw < MIN_RAW_AGREEMENT:
        print(f"  WARNING: sequence raw agreement {seq_raw:.4f} < {MIN_RAW_AGREEMENT}")
        thresholds_pass = False

    print(f"Thresholds gate: {'PASS' if thresholds_pass else 'FAIL'}")

    # --- Sec 37: Build review queue ---
    print(f"\n--- Sec 37: Build Review Queue ---")
    review_queue = build_review_queue(j_rows, j2_rows, j_seqs, j2_seqs)
    review_rows = [q for q in review_queue if q["item_type"] == "row"]
    review_seqs = [q for q in review_queue if q["item_type"] == "sequence"]
    print(f"Review queue: {len(review_queue)} items ({len(review_rows)} rows, {len(review_seqs)} sequences)")

    _write_jsonl(_REVIEW_QUEUE_PATH, review_queue)
    print(f"Wrote review queue to {_REVIEW_QUEUE_PATH.name}")

    # --- Sec 38: J3 adjudication ---
    print(f"\n--- Sec 38: J3 Adjudication ---")

    # API configuration
    api_base = "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    api_key = os.environ.get("VALIDATION_API_KEY", "")
    if not api_key:
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
        print("ERROR: No API key found. Set VALIDATION_API_KEY or configure litellm_config.yaml")
        sys.exit(1)

    # Verify model role separation
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

    review_queue_sha = _sha256_file(_REVIEW_QUEUE_PATH)

    # Adjudicate review-queue rows
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
        print(f"  [{i+1:2d}/{len(review_row_items)}] {cid[-40:]:40s} -> {status}")

    _write_jsonl(_LLM_ADJUDICATION_PATH, adjudication_records)
    print(f"Wrote {len(adjudication_records)} adjudication records to {_LLM_ADJUDICATION_PATH.name}")

    # --- Sec 39: Construct final validation labels ---
    print(f"\n--- Sec 39: Final Validation Labels ---")
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

    # Sequence final labels — using sequence_annotation_id (§4, §17)
    # §6: Validate sequence uniqueness
    j_seq_ann_ids = [sequence_unit_key(r) for r in j_seqs]
    j2_seq_ann_ids = [sequence_unit_key(r) for r in j2_seqs]
    if len(j_seq_ann_ids) != len(set(j_seq_ann_ids)):
        print("ERROR: Duplicate sequence_annotation_id in J sequences")
        sys.exit(1)
    if len(j2_seq_ann_ids) != len(set(j2_seq_ann_ids)):
        print("ERROR: Duplicate sequence_annotation_id in J2 sequences")
        sys.exit(1)
    if len(j_seq_ann_ids) != 36:
        print(f"ERROR: Expected 36 J sequence annotations, got {len(j_seq_ann_ids)}")
        sys.exit(1)
    if len(j2_seq_ann_ids) != 36:
        print(f"ERROR: Expected 36 J2 sequence annotations, got {len(j2_seq_ann_ids)}")
        sys.exit(1)

    # §7: Validate cross-annotator identity coverage
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
        seq_agree = j_seq.get("sequence_reconstructs_target") == j2_seq.get("sequence_reconstructs_target")
        step_agree = j_seq.get("earliest_reconstruction_step") == j2_seq.get("earliest_reconstruction_step")

        # §9: Preserve trust-conditioned identity
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

    import subprocess
    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT
        ).decode().strip()
    except Exception:
        code_commit = "unknown"

    adj_manifest = {
        "schema_version": "1.0",
        "description": "E4-002: Validation annotation adjudication manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
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
        "validation_annotation_source_commit": code_commit,
        "j3_vs_j_agreement_count": j3_vs_j,
        "j3_vs_j2_agreement_count": j3_vs_j2,
        "j3_vs_both_disagree_count": j3_vs_both,
    }

    _write_json(_ADJUDICATION_MANIFEST_PATH, adj_manifest)
    print(f"Wrote adjudication manifest to {_ADJUDICATION_MANIFEST_PATH.name}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("VALIDATION AGREEMENT + ADJUDICATION SUMMARY")
    print("=" * 60)
    print(f"Coverage: {len(common_rows)} rows, {len(common_seq_anns)} sequence units")
    print(f"Thresholds: {'PASS' if thresholds_pass else 'FAIL'}")
    print(f"Review queue: {len(review_queue)} items")
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
        "coverage_pass": coverage_pass,
        "thresholds_pass": thresholds_pass,
        "row_gate_pass": row_gate_pass,
        "seq_gate_pass": seq_gate_pass,
        "unresolved_rows": unresolved_rows,
        "unresolved_seqs": unresolved_seqs,
        "review_queue_count": len(review_queue),
        "adjudicated_count": len(adjudication_records),
    }


if __name__ == "__main__":
    run_validation_agreement_and_adjudication()

#!/usr/bin/env python3
"""E4-003: Test annotation agreement + review queue builder.

Sec 32-37: Compute coverage, agreement, build review queue.
Does NOT run J3 adjudication (see run_test_adjudication.py).

Usage:
  PYTHONPATH=. python scripts/run_test_agreement.py
  PYTHONPATH=. python scripts/run_test_agreement.py --build-review-queue-only
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "test"
_CORPUS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_TEST_CANDIDATES_PATH = _CORPUS_DIR / "test" / "accepted_candidates.jsonl"

# Input files
_J_ROW_LABELS_PATH = _TEST_DIR / "primary_row_annotations.jsonl"
_J2_ROW_LABELS_PATH = _TEST_DIR / "secondary_row_annotations.jsonl"
_J_SEQ_LABELS_PATH = _TEST_DIR / "primary_sequence_annotations.jsonl"
_J2_SEQ_LABELS_PATH = _TEST_DIR / "secondary_sequence_annotations.jsonl"

# Output files
_AGREEMENT_REPORT_PATH = _TEST_DIR / "test_agreement_report.json"
_REVIEW_QUEUE_PATH = _TEST_DIR / "test_review_queue.jsonl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_BINARY_LABELS = ("target_relevant", "target_leakage", "positive_entailment", "task_useful")

# Expected test counts
EXPECTED_ROWS = 450
EXPECTED_SEQUENCES = 72
EXPECTED_FAMILIES = 24


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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


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

    # Item 13: reconstruction_strength exact agreement
    strength_a = [p_by_id[sid].get("reconstruction_strength") for sid in common_ids]
    strength_b = [s_by_id[sid].get("reconstruction_strength") for sid in common_ids]
    strength_exact = sum(
        1 for a, b in zip(strength_a, strength_b) if a is not None and b is not None and a == b
    )
    strength_n = sum(
        1 for a, b in zip(strength_a, strength_b) if a is not None and b is not None
    )
    result["reconstruction_strength"] = {
        "exact_agreement": round(strength_exact / strength_n, 4) if strength_n > 0 else 0.0,
        "cohens_kappa": _cohen_kappa(
            [str(s) for s in strength_a], [str(s) for s in strength_b]
        ),
        "n": strength_n,
    }

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

    p_step = primary_label.get("earliest_reconstruction_step")
    s_step = secondary_label.get("earliest_reconstruction_step")
    if p_step is not None and s_step is not None and p_step != s_step:
        reasons.append("disagreement on earliest_reconstruction_step")

    p_str = primary_label.get("reconstruction_strength")
    s_str = secondary_label.get("reconstruction_strength")
    if p_str is not None and s_str is not None and p_str != s_str:
        reasons.append("disagreement on reconstruction_strength")

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
# Main workflow
# ---------------------------------------------------------------------------


def run_test_agreement(build_review_queue_only: bool = False) -> dict[str, Any]:
    """Sec 32-37: Test agreement computation + review queue."""
    print("=" * 60)
    print("E4-003 TEST AGREEMENT + REVIEW QUEUE")
    print("=" * 60)

    # --- Load inputs ---
    j_rows = _load_jsonl(_J_ROW_LABELS_PATH)
    j2_rows = _load_jsonl(_J2_ROW_LABELS_PATH)
    j_seqs = _load_jsonl(_J_SEQ_LABELS_PATH)
    j2_seqs = _load_jsonl(_J2_SEQ_LABELS_PATH)

    print(f"J rows: {len(j_rows)}")
    print(f"J2 rows: {len(j2_rows)}")
    print(f"J sequences: {len(j_seqs)}")
    print(f"J2 sequences: {len(j2_seqs)}")

    # --- Sec 32: Coverage gate ---
    j_row_ids = {r["candidate_id"] for r in j_rows}
    j2_row_ids = {r["candidate_id"] for r in j2_rows}
    common_rows = j_row_ids & j2_row_ids
    unmatched_rows = (j_row_ids | j2_row_ids) - common_rows

    j_seq_ann_ids = {r["sequence_annotation_id"] for r in j_seqs}
    j2_seq_ann_ids = {r["sequence_annotation_id"] for r in j2_seqs}
    common_seq_anns = j_seq_ann_ids & j2_seq_ann_ids
    unmatched_seq_anns = (j_seq_ann_ids | j2_seq_ann_ids) - common_seq_anns

    j_seq_ids = {r["sequence_family_id"] for r in j_seqs}
    j2_seq_ids = {r["sequence_family_id"] for r in j2_seqs}
    common_seqs = j_seq_ids & j2_seq_ids
    unmatched_seqs = (j_seq_ids | j2_seq_ids) - common_seqs

    print(f"\n--- Sec 32: Coverage Gate ---")
    print(f"J rows: {len(j_row_ids)} / {EXPECTED_ROWS}")
    print(f"J2 rows: {len(j2_row_ids)} / {EXPECTED_ROWS}")
    print(f"Common rows: {len(common_rows)}")
    print(f"Unmatched rows: {len(unmatched_rows)}")
    print(f"J seq annotations: {len(j_seq_ann_ids)} / {EXPECTED_SEQUENCES}")
    print(f"J2 seq annotations: {len(j2_seq_ann_ids)} / {EXPECTED_SEQUENCES}")
    print(f"Common seq annotations: {len(common_seq_anns)}")
    print(f"Unmatched seq annotations: {len(unmatched_seq_anns)}")
    print(f"J seq families: {len(j_seq_ids)} / {EXPECTED_FAMILIES}")
    print(f"J2 seq families: {len(j2_seq_ids)} / {EXPECTED_FAMILIES}")
    print(f"Common seq families: {len(common_seqs)}")
    print(f"Unmatched seq families: {len(unmatched_seqs)}")

    coverage_pass = (
        len(j_row_ids) == EXPECTED_ROWS and len(j2_row_ids) == EXPECTED_ROWS and
        len(common_rows) == EXPECTED_ROWS and len(unmatched_rows) == 0 and
        len(j_seq_ann_ids) == EXPECTED_SEQUENCES and len(j2_seq_ann_ids) == EXPECTED_SEQUENCES and
        len(common_seq_anns) == EXPECTED_SEQUENCES and len(unmatched_seq_anns) == 0 and
        len(j_seq_ids) == EXPECTED_FAMILIES and len(j2_seq_ids) == EXPECTED_FAMILIES and
        len(common_seqs) == EXPECTED_FAMILIES and len(unmatched_seqs) == 0
    )
    print(f"Coverage gate: {'PASS' if coverage_pass else 'FAIL'}")

    if not coverage_pass:
        print("ERROR: Coverage gate failed. Aborting.")
        sys.exit(1)

    # --- Sec 33-35: Compute agreement ---
    j_by_cid = {r["candidate_id"]: r for r in j_rows}
    j2_by_cid = {r["candidate_id"]: r for r in j2_rows}

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

    # Add metadata
    agreement_report["split"] = "test"
    agreement_report["created_at"] = datetime.now(timezone.utc).isoformat()

    # Item 12: Compute thresholds BEFORE writing so they persist in the report
    MIN_RAW_AGREEMENT = 0.85
    MIN_KAPPA = 0.60
    thresholds_pass = True
    per_metric_pass: dict[str, bool] = {}

    for fld in CORE_BINARY_LABELS:
        raw = agreement_report[fld]["raw_agreement"]
        kappa = agreement_report[fld]["cohens_kappa"]
        raw_ok = raw >= MIN_RAW_AGREEMENT
        kappa_ok = not (isinstance(kappa, float) and kappa < MIN_KAPPA)
        field_pass = raw_ok and kappa_ok
        per_metric_pass[f"{fld}_pass"] = field_pass
        if not field_pass:
            thresholds_pass = False
            if not raw_ok:
                print(f"  WARNING: {fld} raw agreement {raw:.4f} < {MIN_RAW_AGREEMENT}")
            if not kappa_ok:
                print(f"  WARNING: {fld} kappa {kappa} < {MIN_KAPPA}")

    seq_raw = seq_agreement["reconstruction_binary_agreement"]["raw_agreement"]
    seq_pass = seq_raw >= MIN_RAW_AGREEMENT
    per_metric_pass["sequence_reconstruction_pass"] = seq_pass
    if not seq_pass:
        print(f"  WARNING: sequence raw agreement {seq_raw:.4f} < {MIN_RAW_AGREEMENT}")
        thresholds_pass = False

    # Persist threshold metadata and per-metric pass fields in the report
    agreement_report["agreement_thresholds"] = {
        "min_raw_agreement": MIN_RAW_AGREEMENT,
        "min_kappa": MIN_KAPPA,
    }
    agreement_report["agreement_thresholds_pass"] = thresholds_pass
    agreement_report.update(per_metric_pass)

    _write_json(_AGREEMENT_REPORT_PATH, agreement_report)
    print(f"Wrote agreement report to {_AGREEMENT_REPORT_PATH.name}")

    print(f"Thresholds gate: {'PASS' if thresholds_pass else 'FAIL'}")

    # --- Sec 37: Build review queue ---
    print(f"\n--- Sec 37: Build Review Queue ---")
    review_queue = build_review_queue(j_rows, j2_rows, j_seqs, j2_seqs)
    review_rows = [q for q in review_queue if q["item_type"] == "row"]
    review_seqs = [q for q in review_queue if q["item_type"] == "sequence"]
    print(f"Review queue: {len(review_queue)} items ({len(review_rows)} rows, {len(review_seqs)} sequences)")

    _write_jsonl(_REVIEW_QUEUE_PATH, review_queue)
    print(f"Wrote review queue to {_REVIEW_QUEUE_PATH.name}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TEST AGREEMENT + REVIEW QUEUE SUMMARY")
    print("=" * 60)
    print(f"Coverage: {len(common_rows)} rows, {len(common_seq_anns)} sequence units")
    print(f"Thresholds: {'PASS' if thresholds_pass else 'FAIL'}")
    print(f"Review queue: {len(review_queue)} items")

    if build_review_queue_only:
        print("\n--build-review-queue-only: stopping after review queue.")
        return {
            "coverage_pass": coverage_pass,
            "thresholds_pass": thresholds_pass,
            "review_queue_count": len(review_queue),
        }

    return {
        "coverage_pass": coverage_pass,
        "thresholds_pass": thresholds_pass,
        "review_queue_count": len(review_queue),
        "review_row_count": len(review_rows),
        "review_sequence_count": len(review_seqs),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute test annotation agreement + review queue (E4-003)")
    parser.add_argument("--build-review-queue-only", action="store_true",
                        help="Stop after building the review queue (no adjudication)")
    args = parser.parse_args()
    run_test_agreement(build_review_queue_only=args.build_review_queue_only)

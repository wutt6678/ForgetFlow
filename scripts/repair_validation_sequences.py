#!/usr/bin/env python3
"""E4-002 Sequence Accounting Repair.

Regenerates sequence-related derived artifacts using sequence_annotation_id
as the unique key instead of sequence_family_id.

Does NOT re-run J/J2 annotations or row adjudication.
Does NOT make any API calls (all 36 sequences agree).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VAL_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "validation"

# Input files (NOT modified)
_J_SEQ_PATH = _VAL_DIR / "primary_sequence_annotations.jsonl"
_J2_SEQ_PATH = _VAL_DIR / "secondary_sequence_annotations.jsonl"
_J_ROW_PATH = _VAL_DIR / "primary_row_annotations.jsonl"
_J2_ROW_PATH = _VAL_DIR / "secondary_row_annotations.jsonl"

# Output files to regenerate
_AGREEMENT_REPORT_PATH = _VAL_DIR / "validation_agreement_report.json"
_REVIEW_QUEUE_PATH = _VAL_DIR / "review_queue.jsonl"
_FINAL_SEQ_LABELS_PATH = _VAL_DIR / "final_sequence_labels.jsonl"
_ADJUDICATION_MANIFEST_PATH = _VAL_DIR / "adjudication_manifest.json"

_FROZEN_CORPUS_SHA = "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"
ADJUDICATION_PROTOCOL_VERSION = "1.0"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def sequence_unit_key(record: dict) -> str:
    return record["sequence_annotation_id"]


def derive_trust_level(record: dict) -> str:
    cids = record.get("ordered_candidate_ids", [])
    if cids:
        return cids[0].rsplit("_", 1)[-1]
    return "unknown"


def compute_binary_agreement(vals_a: list, vals_b: list) -> dict:
    n = len(vals_a)
    if n == 0:
        return {"raw_agreement": 0.0, "cohens_kappa": 0.0, "n": 0}
    agree = sum(1 for a, b in zip(vals_a, vals_b) if a == b)
    raw = agree / n

    # Cohen's kappa
    tp = sum(1 for a, b in zip(vals_a, vals_b) if a and b)
    fp = sum(1 for a, b in zip(vals_a, vals_b) if a and not b)
    fn = sum(1 for a, b in zip(vals_a, vals_b) if not a and b)
    tn = sum(1 for a, b in zip(vals_a, vals_b) if not a and not b)

    po = (tp + tn) / n if n > 0 else 0.0
    p_yes = ((tp + fp) / n) * ((tp + fn) / n) if n > 0 else 0.0
    p_no = ((fn + tn) / n) * ((fp + tn) / n) if n > 0 else 0.0
    pe = p_yes + p_no
    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 0.0

    pos_agree = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    neg_agree = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0

    return {
        "raw_agreement": round(raw, 4),
        "cohens_kappa": round(kappa, 4),
        "positive_agreement": round(pos_agree, 4),
        "negative_agreement": round(neg_agree, 4),
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> int:
    print("=" * 60)
    print("E4-002 Sequence Accounting Repair")
    print("=" * 60)

    # Load existing J/J2 sequence annotations (NOT modified)
    j_seqs = _load_jsonl(_J_SEQ_PATH)
    j2_seqs = _load_jsonl(_J2_SEQ_PATH)
    print(f"J sequences: {len(j_seqs)}")
    print(f"J2 sequences: {len(j2_seqs)}")

    # §6: Validate uniqueness
    j_ids = [sequence_unit_key(r) for r in j_seqs]
    j2_ids = [sequence_unit_key(r) for r in j2_seqs]
    assert len(j_ids) == 36, f"Expected 36 J sequences, got {len(j_ids)}"
    assert len(j2_ids) == 36, f"Expected 36 J2 sequences, got {len(j2_ids)}"
    assert len(j_ids) == len(set(j_ids)), "Duplicate J sequence_annotation_ids"
    assert len(j2_ids) == len(set(j2_ids)), "Duplicate J2 sequence_annotation_ids"

    # §7: Validate cross-annotator coverage
    j_set = set(j_ids)
    j2_set = set(j2_ids)
    assert j_set == j2_set, f"J/J2 sequence IDs differ: {j_set ^ j2_set}"
    print(f"Common sequence_annotation_ids: {len(j_set & j2_set)}")
    print(f"Unmatched: {len(j_set - j2_set)} J, {len(j2_set - j_set)} J2")

    # §11: Recompute sequence agreement over 36 units
    print("\n--- Sequence Agreement (n=36) ---")
    p_by_id = {sequence_unit_key(r): r for r in j_seqs}
    s_by_id = {sequence_unit_key(r): r for r in j2_seqs}
    common_ids = sorted(j_set & j2_set)

    recon_a = [p_by_id[sid]["sequence_reconstructs_target"] for sid in common_ids]
    recon_b = [s_by_id[sid]["sequence_reconstructs_target"] for sid in common_ids]
    recon_agreement = compute_binary_agreement(recon_a, recon_b)
    print(f"  reconstruction: raw={recon_agreement['raw_agreement']:.4f}, kappa={recon_agreement['cohens_kappa']:.4f}")

    both_reconstruct = [
        sid for sid in common_ids
        if p_by_id[sid]["sequence_reconstructs_target"] and s_by_id[sid]["sequence_reconstructs_target"]
    ]
    if both_reconstruct:
        step_agree = sum(
            1 for sid in both_reconstruct
            if p_by_id[sid].get("earliest_reconstruction_step") == s_by_id[sid].get("earliest_reconstruction_step")
        )
        earliest_step_agreement = round(step_agree / len(both_reconstruct), 4)
    else:
        earliest_step_agreement = "not_estimable"

    seq_agreement = {
        "n": len(common_ids),
        "primary_count": len(p_by_id),
        "secondary_count": len(s_by_id),
        "common_sequence_annotation_ids": len(common_ids),
        "unmatched_primary": 0,
        "unmatched_secondary": 0,
        "reconstruction_binary_agreement": recon_agreement,
        "earliest_step_exact_agreement": earliest_step_agreement,
        "earliest_step_n": len(both_reconstruct),
    }

    # Update agreement report (preserve row sections)
    agreement_report = json.loads(_AGREEMENT_REPORT_PATH.read_text())
    agreement_report["sequence"] = seq_agreement
    _write_json(_AGREEMENT_REPORT_PATH, agreement_report)
    print(f"Updated {_AGREEMENT_REPORT_PATH.name}")

    # §13: Rebuild sequence review queue
    print("\n--- Sequence Review Queue ---")
    # Load existing row review queue
    existing_queue = _load_jsonl(_REVIEW_QUEUE_PATH)
    row_items = [q for q in existing_queue if q["item_type"] == "row"]

    # Check sequences for review
    seq_review_items = []
    for sid in common_ids:
        p_rec = p_by_id[sid]
        s_rec = s_by_id[sid]
        reasons = []
        if p_rec.get("sequence_reconstructs_target") != s_rec.get("sequence_reconstructs_target"):
            reasons.append("sequence_reconstruction_disagreement")
        if p_rec.get("earliest_reconstruction_step") != s_rec.get("earliest_reconstruction_step"):
            reasons.append("earliest_step_disagreement")
        if p_rec.get("uncertain") or s_rec.get("uncertain"):
            reasons.append("uncertainty")
        if (p_rec.get("confidence", 1.0) < 0.7 or s_rec.get("confidence", 1.0) < 0.7):
            reasons.append("low_confidence")
        if reasons:
            seq_review_items.append({
                "item_type": "sequence",
                "sequence_annotation_id": sid,
                "sequence_family_id": p_rec.get("sequence_family_id", ""),
                "primary_label": p_rec,
                "secondary_label": s_rec,
                "review_reasons": reasons,
            })

    new_queue = row_items + seq_review_items
    _write_jsonl(_REVIEW_QUEUE_PATH, new_queue)
    print(f"Review queue: {len(new_queue)} items ({len(row_items)} rows, {len(seq_review_items)} sequences)")

    # §17: Rebuild final_sequence_labels.jsonl with 36 records
    print("\n--- Final Sequence Labels (n=36) ---")
    final_seq_labels = []
    for sid in common_ids:
        j_seq = p_by_id[sid]
        j2_seq = s_by_id[sid]
        seq_agree = j_seq.get("sequence_reconstructs_target") == j2_seq.get("sequence_reconstructs_target")
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
    print(f"Wrote {len(final_seq_labels)} final sequence labels")

    # Trust-conditioned reporting (§24)
    trust_counts = {}
    for fs in final_seq_labels:
        tl = fs["trust_level"]
        trust_counts[tl] = trust_counts.get(tl, 0) + 1
    print(f"Trust distribution: {trust_counts}")

    # Count unresolved sequences
    unresolved_seqs = sum(1 for r in final_seq_labels if r["resolution_source"] == "unresolved")
    unresolved_seq_rate = unresolved_seqs / len(final_seq_labels) if final_seq_labels else 1.0
    print(f"Unresolved sequences: {unresolved_seqs}/{len(final_seq_labels)} ({unresolved_seq_rate:.4f})")

    # §18: Rebuild adjudication manifest
    print("\n--- Adjudication Manifest ---")
    adj_manifest = json.loads(_ADJUDICATION_MANIFEST_PATH.read_text())

    # Update sequence-related fields
    adj_manifest["review_sequence_count"] = len(seq_review_items)
    adj_manifest["final_label_counts"]["total_sequences"] = len(final_seq_labels)
    adj_manifest["final_label_counts"]["unresolved_sequences"] = unresolved_seqs
    adj_manifest["unresolved_sequence_rate"] = round(unresolved_seq_rate, 4)

    # Update SHA bindings
    adj_manifest["review_queue_sha256"] = _sha256(_REVIEW_QUEUE_PATH)
    adj_manifest["final_sequence_labels_sha256"] = _sha256(_FINAL_SEQ_LABELS_PATH)
    adj_manifest["validation_annotation_source_commit"] = _git_commit()

    # §24: Report both family and unit counts
    families = set(r["sequence_family_id"] for r in j_seqs)
    adj_manifest["structural_sequence_families"] = len(families)
    adj_manifest["trust_conditioned_sequence_units"] = len(common_ids)

    _write_json(_ADJUDICATION_MANIFEST_PATH, adj_manifest)
    print(f"Updated {_ADJUDICATION_MANIFEST_PATH.name}")

    # Summary
    print("\n" + "=" * 60)
    print("REPAIR SUMMARY")
    print("=" * 60)
    print(f"Structural sequence families: {len(families)}")
    print(f"Trust-conditioned sequence units: {len(common_ids)}")
    print(f"Final sequence labels: {len(final_seq_labels)}/36")
    print(f"Sequence reconstruction agreement: {recon_agreement['raw_agreement']:.4f}")
    print(f"Sequence kappa: {recon_agreement['cohens_kappa']:.4f}")
    print(f"Earliest-step exact agreement: {earliest_step_agreement}")
    print(f"Sequence review queue: {len(seq_review_items)}")
    print(f"Unresolved sequences: {unresolved_seqs}/{len(final_seq_labels)} ({unresolved_seq_rate:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

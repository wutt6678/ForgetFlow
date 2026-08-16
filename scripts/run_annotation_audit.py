#!/usr/bin/env python3
"""E4-001 Sec 63-64: Annotation audit, agreement, review queue, and pilot report.

Loads J (primary) and J2 (secondary) annotations, computes agreement metrics,
builds the review queue, and writes a comprehensive pilot report.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.empirical_annotation import (
    CORE_BINARY_LABELS,
    compute_binary_agreement,
    compute_categorical_agreement,
    compute_sequence_agreement,
    build_review_queue,
    adjudicate_row,
    verify_model_role_separation,
    verify_secondary_blindness,
    build_prompt_manifest,
    build_annotation_config,
    ANNOTATION_SCHEMA_VERSION,
    MODEL_PRIMARY,
    MODEL_SECONDARY,
    ROLE_PRIMARY,
    ROLE_SECONDARY,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "development_v3"

PRIMARY_ROW_PATH = _ANNOTATIONS_DIR / "row_annotations.jsonl"
PRIMARY_SEQ_PATH = _ANNOTATIONS_DIR / "sequence_annotations.jsonl"
SECONDARY_ROW_PATH = _ANNOTATIONS_DIR / "secondary_row_annotations.jsonl"
SECONDARY_SEQ_PATH = _ANNOTATIONS_DIR / "secondary_sequence_annotations.jsonl"
CAMPAIGN_IDENTITY_PATH = _ANNOTATIONS_DIR / "campaign_identity.json"
SECONDARY_CAMPAIGN_IDENTITY_PATH = _ANNOTATIONS_DIR / "secondary_campaign_identity.json"
PRIMARY_ATTEMPTS_PATH = _ANNOTATIONS_DIR / "primary_annotation_attempts.jsonl"
SECONDARY_ATTEMPTS_PATH = _ANNOTATIONS_DIR / "secondary_annotation_attempts.jsonl"

# Output paths
AUDIT_REPORT_PATH = _ANNOTATIONS_DIR / "audit_report.json"
REVIEW_QUEUE_PATH = _ANNOTATIONS_DIR / "review_queue.jsonl"
ADJUDICATION_PATH = _ANNOTATIONS_DIR / "adjudication_summary.json"
PILOT_REPORT_PATH = _ANNOTATIONS_DIR / "pilot_report.json"
PILOT_REPORT_MD_PATH = _ANNOTATIONS_DIR / "pilot_report.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_json(path: Path, data: dict | list) -> None:
    """Write a JSON file with pretty formatting."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------


def run_audit() -> dict:
    """Run the full annotation audit and produce all artifacts."""
    print("=" * 70)
    print("E4-001 Annotation Audit, Agreement & Pilot Report")
    print("=" * 70)

    # Load annotations
    print("\n[1/7] Loading annotations...")
    primary_rows = _load_jsonl(PRIMARY_ROW_PATH)
    primary_seqs = _load_jsonl(PRIMARY_SEQ_PATH)
    secondary_rows = _load_jsonl(SECONDARY_ROW_PATH)
    secondary_seqs = _load_jsonl(SECONDARY_SEQ_PATH)

    print(f"  Primary rows: {len(primary_rows)}")
    print(f"  Primary sequences: {len(primary_seqs)}")
    print(f"  Secondary rows: {len(secondary_rows)}")
    print(f"  Secondary sequences: {len(secondary_seqs)}")

    # Load campaign identities
    primary_identity = json.loads(CAMPAIGN_IDENTITY_PATH.read_text()) if CAMPAIGN_IDENTITY_PATH.exists() else {}
    secondary_identity = json.loads(SECONDARY_CAMPAIGN_IDENTITY_PATH.read_text()) if SECONDARY_CAMPAIGN_IDENTITY_PATH.exists() else {}

    # Load attempts for latency stats
    primary_attempts = _load_jsonl(PRIMARY_ATTEMPTS_PATH) if PRIMARY_ATTEMPTS_PATH.exists() else []
    secondary_attempts = _load_jsonl(SECONDARY_ATTEMPTS_PATH) if SECONDARY_ATTEMPTS_PATH.exists() else []

    # -----------------------------------------------------------------------
    # 1. Coverage audit
    # -----------------------------------------------------------------------
    print("\n[2/7] Coverage audit...")
    primary_cids = {r["candidate_id"] for r in primary_rows}
    secondary_cids = {r["candidate_id"] for r in secondary_rows}
    common_cids = primary_cids & secondary_cids
    unmatched = (primary_cids | secondary_cids) - common_cids

    coverage = {
        "primary_row_count": len(primary_rows),
        "secondary_row_count": len(secondary_rows),
        "common_row_count": len(common_cids),
        "unmatched_row_count": len(unmatched),
        "unmatched_candidate_ids": sorted(unmatched),
        "primary_sequence_count": len(primary_seqs),
        "secondary_sequence_count": len(secondary_seqs),
    }
    print(f"  Common rows: {len(common_cids)}, Unmatched: {len(unmatched)}")

    # -----------------------------------------------------------------------
    # 2. Model-role separation verification (Sec 54)
    # -----------------------------------------------------------------------
    print("\n[3/7] Model-role separation verification...")
    role_violations = verify_model_role_separation()
    print(f"  Role violations: {len(role_violations)}")
    if role_violations:
        for v in role_violations:
            print(f"    VIOLATION: {v}")

    # -----------------------------------------------------------------------
    # 3. Row-level agreement (Sec 36)
    # -----------------------------------------------------------------------
    print("\n[4/7] Computing row-level agreement...")
    # Sort both by candidate_id for alignment
    p_by_cid = {r["candidate_id"]: r for r in primary_rows}
    s_by_cid = {r["candidate_id"]: r for r in secondary_rows}
    aligned_cids = sorted(common_cids)

    row_agreement: dict[str, dict] = {}
    for label in CORE_BINARY_LABELS:
        labels_a = [p_by_cid[cid][label] for cid in aligned_cids]
        labels_b = [s_by_cid[cid][label] for cid in aligned_cids]
        result = compute_binary_agreement(labels_a, labels_b)
        row_agreement[label] = result
        print(f"  {label}: raw={result['raw_agreement']:.4f}, kappa={result['cohens_kappa']}")

    # Leakage strength agreement (Sec 37)
    ls_a = [p_by_cid[cid]["leakage_strength"] for cid in aligned_cids]
    ls_b = [s_by_cid[cid]["leakage_strength"] for cid in aligned_cids]
    ls_agreement = compute_categorical_agreement(ls_a, ls_b)
    print(f"  leakage_strength: exact={ls_agreement['exact_agreement']:.4f}, kappa={ls_agreement['cohens_kappa']}")

    # -----------------------------------------------------------------------
    # 4. Sequence-level agreement (Sec 38)
    # -----------------------------------------------------------------------
    print("\n[5/7] Computing sequence-level agreement...")
    # Use composite key (sequence_family_id, tuple(ordered_candidate_ids)) for alignment
    p_seq_by_key = {}
    for s in primary_seqs:
        key = (s["sequence_family_id"], tuple(s.get("ordered_candidate_ids", [])))
        p_seq_by_key[key] = s
    s_seq_by_key = {}
    for s in secondary_seqs:
        key = (s["sequence_family_id"], tuple(s.get("ordered_candidate_ids", [])))
        s_seq_by_key[key] = s

    common_seq_keys = sorted(set(p_seq_by_key.keys()) & set(s_seq_by_key.keys()))
    print(f"  Common sequence units: {len(common_seq_keys)}")

    # Build aligned lists for sequence binary agreement
    seq_recon_a = [p_seq_by_key[k]["sequence_reconstructs_target"] for k in common_seq_keys]
    seq_recon_b = [s_seq_by_key[k]["sequence_reconstructs_target"] for k in common_seq_keys]
    seq_recon_agreement = compute_binary_agreement(seq_recon_a, seq_recon_b)
    print(f"  sequence_reconstructs_target: raw={seq_recon_agreement['raw_agreement']:.4f}, kappa={seq_recon_agreement['cohens_kappa']}")

    # Strength agreement
    seq_str_a = [p_seq_by_key[k]["reconstruction_strength"] for k in common_seq_keys]
    seq_str_b = [s_seq_by_key[k]["reconstruction_strength"] for k in common_seq_keys]
    seq_strength_agreement = compute_categorical_agreement(seq_str_a, seq_str_b)
    print(f"  reconstruction_strength: exact={seq_strength_agreement['exact_agreement']:.4f}")

    # Earliest step agreement (where both say reconstructs=True)
    both_reconstruct = [
        k for k in common_seq_keys
        if p_seq_by_key[k]["sequence_reconstructs_target"] and s_seq_by_key[k]["sequence_reconstructs_target"]
    ]
    if both_reconstruct:
        step_agree = sum(
            1 for k in both_reconstruct
            if p_seq_by_key[k].get("earliest_reconstruction_step") == s_seq_by_key[k].get("earliest_reconstruction_step")
        )
        earliest_step_agreement = {
            "exact_agreement": round(step_agree / len(both_reconstruct), 4),
            "n": len(both_reconstruct),
        }
    else:
        earliest_step_agreement = {"exact_agreement": "not_estimable", "n": 0}

    # Also run the library function for comparison
    seq_agreement_lib = compute_sequence_agreement(primary_seqs, secondary_seqs)

    # -----------------------------------------------------------------------
    # 5. Review queue (Sec 39)
    # -----------------------------------------------------------------------
    print("\n[6/7] Building review queue...")
    review_queue = build_review_queue(primary_rows, secondary_rows, primary_seqs, secondary_seqs)
    print(f"  Review queue items: {len(review_queue)}")
    row_queue = [q for q in review_queue if q["item_type"] == "row"]
    seq_queue = [q for q in review_queue if q["item_type"] == "sequence"]
    print(f"    Row items: {len(row_queue)}")
    print(f"    Sequence items: {len(seq_queue)}")

    # Summarize review reasons
    reason_counts: dict[str, int] = {}
    for item in review_queue:
        for reason in item["review_reasons"]:
            # Normalize reason to category
            if "disagreement on" in reason:
                cat = reason
            elif "uncertain" in reason:
                cat = "uncertainty flag"
            elif "confidence" in reason:
                cat = "low confidence"
            else:
                cat = reason
            reason_counts[cat] = reason_counts.get(cat, 0) + 1

    # -----------------------------------------------------------------------
    # 6. Adjudication summary (Sec 40)
    # -----------------------------------------------------------------------
    print("\n[7/7] Adjudication summary...")
    consensus_count = 0
    unresolved_count = 0
    for cid in aligned_cids:
        result = adjudicate_row(p_by_cid[cid], s_by_cid[cid])
        if result["status"] == "consensus":
            consensus_count += 1
        else:
            unresolved_count += 1

    seq_consensus = 0
    seq_unresolved = 0
    for k in common_seq_keys:
        p_label = p_seq_by_key[k]
        s_label = s_seq_by_key[k]
        all_agree = all(
            p_label.get(fld) == s_label.get(fld)
            for fld in CORE_BINARY_LABELS
            if fld in p_label and fld in s_label
        )
        ls_agree = p_label.get("reconstruction_strength") == s_label.get("reconstruction_strength")
        recon_agree = p_label.get("sequence_reconstructs_target") == s_label.get("sequence_reconstructs_target")
        if all_agree and ls_agree and recon_agree:
            seq_consensus += 1
        else:
            seq_unresolved += 1

    adjudication = {
        "row_consensus": consensus_count,
        "row_unresolved": unresolved_count,
        "row_consensus_rate": round(consensus_count / len(aligned_cids), 4) if aligned_cids else 0,
        "sequence_consensus": seq_consensus,
        "sequence_unresolved": seq_unresolved,
        "sequence_consensus_rate": round(seq_consensus / len(common_seq_keys), 4) if common_seq_keys else 0,
    }
    print(f"  Row consensus: {consensus_count}/{len(aligned_cids)} ({adjudication['row_consensus_rate']:.2%})")
    print(f"  Sequence consensus: {seq_consensus}/{len(common_seq_keys)} ({adjudication['sequence_consensus_rate']:.2%})")

    # -----------------------------------------------------------------------
    # Confidence statistics
    # -----------------------------------------------------------------------
    p_confidences = [r.get("confidence", 1.0) for r in primary_rows if isinstance(r.get("confidence"), (int, float))]
    s_confidences = [r.get("confidence", 1.0) for r in secondary_rows if isinstance(r.get("confidence"), (int, float))]
    p_uncertain_count = sum(1 for r in primary_rows if r.get("uncertain"))
    s_uncertain_count = sum(1 for r in secondary_rows if r.get("uncertain"))

    confidence_stats = {
        "primary_mean_confidence": round(sum(p_confidences) / len(p_confidences), 4) if p_confidences else 0,
        "primary_min_confidence": round(min(p_confidences), 4) if p_confidences else 0,
        "primary_max_confidence": round(max(p_confidences), 4) if p_confidences else 0,
        "primary_uncertain_count": p_uncertain_count,
        "secondary_mean_confidence": round(sum(s_confidences) / len(s_confidences), 4) if s_confidences else 0,
        "secondary_min_confidence": round(min(s_confidences), 4) if s_confidences else 0,
        "secondary_max_confidence": round(max(s_confidences), 4) if s_confidences else 0,
        "secondary_uncertain_count": s_uncertain_count,
    }

    # -----------------------------------------------------------------------
    # Latency statistics from attempts
    # -----------------------------------------------------------------------
    latency_stats = {}
    for role, attempts in [("primary", primary_attempts), ("secondary", secondary_attempts)]:
        if attempts:
            latencies = [a.get("latency_ms", 0) for a in attempts if a.get("latency_ms", 0) > 0]
            if latencies:
                latency_stats[f"{role}_mean_latency_ms"] = round(sum(latencies) / len(latencies), 1)
                latency_stats[f"{role}_total_latency_ms"] = round(sum(latencies), 1)
                latency_stats[f"{role}_attempt_count"] = len(attempts)
                success_count = sum(1 for a in attempts if a.get("status") == "success")
                latency_stats[f"{role}_success_count"] = success_count

    # -----------------------------------------------------------------------
    # Assemble audit report
    # -----------------------------------------------------------------------
    gate_assessment = _assess_gates(
        row_agreement, seq_recon_agreement, coverage, role_violations,
        adjudication=adjudication,
    )

    audit_report = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "primary_model": MODEL_PRIMARY,
        "secondary_model": MODEL_SECONDARY,
        "model_role_separation": {
            "violations": role_violations,
            "passed": len(role_violations) == 0,
        },
        "coverage": coverage,
        "row_agreement": row_agreement,
        "leakage_strength_agreement": ls_agreement,
        "sequence_agreement": {
            "reconstruction_binary": seq_recon_agreement,
            "reconstruction_strength": seq_strength_agreement,
            "earliest_step_exact": earliest_step_agreement,
            "library_result": seq_agreement_lib,
        },
        "review_queue_summary": {
            "total_items": len(review_queue),
            "row_items": len(row_queue),
            "sequence_items": len(seq_queue),
            "reason_counts": reason_counts,
        },
        "adjudication": adjudication,
        "confidence_statistics": confidence_stats,
        "latency_statistics": latency_stats,
        "gate_assessment": gate_assessment,
    }

    # -----------------------------------------------------------------------
    # Write artifacts
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Writing artifacts...")

    # 1. Audit report JSON
    _write_json(AUDIT_REPORT_PATH, audit_report)
    print(f"  ✓ {AUDIT_REPORT_PATH.name}")

    # 2. Review queue JSONL
    # Strip full labels for compact queue file, keep just IDs and reasons
    compact_queue = []
    for item in review_queue:
        compact = {
            "item_type": item["item_type"],
            "review_reasons": item["review_reasons"],
        }
        if item["item_type"] == "row":
            compact["candidate_id"] = item["candidate_id"]
        else:
            compact["sequence_family_id"] = item["sequence_family_id"]
        compact_queue.append(compact)
    _write_jsonl(REVIEW_QUEUE_PATH, compact_queue)
    print(f"  ✓ {REVIEW_QUEUE_PATH.name}")

    # 3. Adjudication summary
    _write_json(ADJUDICATION_PATH, adjudication)
    print(f"  ✓ {ADJUDICATION_PATH.name}")

    # 4. Pilot report (comprehensive JSON)
    pilot_report = {
        "report_type": "pilot_annotation_report",
        "report_timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "specification": "E4-001",
        "campaign_identity": {
            "primary": primary_identity,
            "secondary": secondary_identity,
        },
        "annotation_config": build_annotation_config(),
        "prompt_manifest": build_prompt_manifest(),
        "coverage": coverage,
        "model_role_separation": {
            "generator": "qwen3.7-plus",
            "primary": MODEL_PRIMARY,
            "secondary": MODEL_SECONDARY,
            "violations": role_violations,
            "passed": len(role_violations) == 0,
        },
        "row_level_agreement": row_agreement,
        "leakage_strength_agreement": ls_agreement,
        "sequence_level_agreement": {
            "reconstruction_binary": seq_recon_agreement,
            "reconstruction_strength": seq_strength_agreement,
            "earliest_step_exact": earliest_step_agreement,
        },
        "review_queue": {
            "total": len(review_queue),
            "row_items": len(row_queue),
            "sequence_items": len(seq_queue),
            "reason_breakdown": reason_counts,
        },
        "adjudication_summary": adjudication,
        "confidence_statistics": confidence_stats,
        "latency_statistics": latency_stats,
        "gate_assessment": gate_assessment,
    }
    _write_json(PILOT_REPORT_PATH, pilot_report)
    print(f"  ✓ {PILOT_REPORT_PATH.name}")

    # 5. Pilot report Markdown
    md = _render_pilot_report_md(pilot_report)
    PILOT_REPORT_MD_PATH.write_text(md)
    print(f"  ✓ {PILOT_REPORT_MD_PATH.name}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    return audit_report


def _assess_gates(
    row_agreement: dict,
    seq_recon_agreement: dict,
    coverage: dict,
    role_violations: list,
    adjudication: dict | None = None,
) -> dict:
    """Assess pilot gate conditions.

    Uses separate status fields so that agreement-audit PASS does not
    imply the full annotation protocol is ready (Sec 5 repair).
    """
    gates = {}

    # Gate 1: Full coverage
    gates["full_coverage"] = {
        "passed": coverage["unmatched_row_count"] == 0,
        "detail": f"unmatched={coverage['unmatched_row_count']}",
    }

    # Gate 2: Model-role separation
    gates["model_role_separation"] = {
        "passed": len(role_violations) == 0,
        "detail": f"violations={len(role_violations)}",
    }

    # Gate 3: Row agreement (kappa > 0.4 for core labels)
    kappa_values = []
    for label, agr in row_agreement.items():
        k = agr["cohens_kappa"]
        if isinstance(k, (int, float)):
            kappa_values.append((label, k))
    min_kappa = min(k for _, k in kappa_values) if kappa_values else 0
    gates["row_agreement_acceptable"] = {
        "passed": min_kappa > 0.4,
        "detail": f"min_kappa={min_kappa:.4f}, values={dict(kappa_values)}",
    }

    # Gate 4: Sequence reconstruction agreement
    seq_kappa = seq_recon_agreement.get("cohens_kappa", 0)
    if isinstance(seq_kappa, str):
        seq_kappa_val = 0.0
    else:
        seq_kappa_val = seq_kappa
    gates["sequence_agreement_estimable"] = {
        "passed": seq_recon_agreement.get("raw_agreement", 0) > 0,
        "detail": f"raw={seq_recon_agreement.get('raw_agreement')}, kappa={seq_kappa}",
    }

    all_passed = all(g["passed"] for g in gates.values())

    # --- Separate status fields (Repair A, Sec 5) ---
    agreement_audit_passed = all_passed
    coverage_audit_passed = (
        gates["full_coverage"]["passed"]
    )
    model_role_audit_passed = gates["model_role_separation"]["passed"]

    # Unresolved-rate gate (Sec 5): add directly to audit summary
    adj = adjudication or {}
    row_unresolved = adj.get("row_unresolved", 0)
    total_rows = coverage.get("common_row_count", 0)
    row_unresolved_rate = row_unresolved / total_rows if total_rows > 0 else 1.0
    row_unresolved_threshold = 0.10
    row_unresolved_gate_passed = row_unresolved_rate <= row_unresolved_threshold

    seq_unresolved = adj.get("sequence_unresolved", 0)
    total_seqs = adj.get("sequence_consensus", 0) + seq_unresolved
    seq_unresolved_rate = seq_unresolved / total_seqs if total_seqs > 0 else 1.0
    seq_unresolved_threshold = 0.10
    seq_unresolved_gate_passed = seq_unresolved_rate <= seq_unresolved_threshold

    # Protocol freeze ready requires ALL gates including unresolved rate
    protocol_freeze_ready = (
        agreement_audit_passed
        and coverage_audit_passed
        and model_role_audit_passed
        and row_unresolved_gate_passed
        and seq_unresolved_gate_passed
    )

    return {
        "gates": gates,
        "all_passed": all_passed,
        # Separate status fields (Repair A)
        "agreement_audit_passed": agreement_audit_passed,
        "coverage_audit_passed": coverage_audit_passed,
        "model_role_audit_passed": model_role_audit_passed,
        "protocol_freeze_ready": protocol_freeze_ready,
        # Unresolved-rate gate (Sec 5)
        "row_unresolved": row_unresolved,
        "row_unresolved_rate": round(row_unresolved_rate, 4),
        "row_unresolved_threshold": row_unresolved_threshold,
        "row_unresolved_gate_passed": row_unresolved_gate_passed,
        "seq_unresolved": seq_unresolved,
        "seq_unresolved_rate": round(seq_unresolved_rate, 4),
        "seq_unresolved_threshold": seq_unresolved_threshold,
        "seq_unresolved_gate_passed": seq_unresolved_gate_passed,
        # Legacy field kept for backward compatibility
        "assessment": "PASS" if agreement_audit_passed else "CONDITIONAL",
    }


def _render_pilot_report_md(report: dict) -> str:
    """Render the pilot report as Markdown."""
    lines: list[str] = []
    lines.append("# E4-001 Pilot Annotation Report\n")
    lines.append(f"**Report timestamp:** {report['report_timestamp']}")
    lines.append(f"**Schema version:** {report['schema_version']}")
    lines.append(f"**Specification:** {report['specification']}\n")

    # Campaign identity
    lines.append("## Campaign Identity\n")
    lines.append("### Primary Annotator (J)")
    lines.append(f"- Model: `{report['annotation_config']['primary_model']}`")
    lines.append(f"- Provider: `{report['annotation_config']['primary_provider']}`")
    pi = report.get("campaign_identity", {}).get("primary", {})
    if pi:
        lines.append(f"- Campaign ID: `{pi.get('campaign_id', 'N/A')}`")
        lines.append(f"- Code commit: `{pi.get('code_commit_sha', 'N/A')}`")
    lines.append("\n### Secondary Annotator (J2)")
    lines.append(f"- Model: `{report['annotation_config']['secondary_model']}`")
    lines.append(f"- Provider: `{report['annotation_config']['secondary_provider']}`")
    si = report.get("campaign_identity", {}).get("secondary", {})
    if si:
        lines.append(f"- Campaign ID: `{si.get('campaign_id', 'N/A')}`")
        lines.append(f"- Code commit: `{si.get('code_commit_sha', 'N/A')}`")

    # Model-role separation
    mrs = report["model_role_separation"]
    lines.append(f"\n## Model-Role Separation (Sec 54)\n")
    lines.append(f"- Generator (G): `{mrs['generator']}`")
    lines.append(f"- Primary (J): `{mrs['primary']}`")
    lines.append(f"- Secondary (J2): `{mrs['secondary']}`")
    lines.append(f"- **Passed:** {'Yes' if mrs['passed'] else 'NO — violations: ' + str(mrs['violations'])}")

    # Coverage
    cov = report["coverage"]
    lines.append(f"\n## Coverage\n")
    lines.append(f"- Primary row annotations: {cov['primary_row_count']}")
    lines.append(f"- Secondary row annotations: {cov['secondary_row_count']}")
    lines.append(f"- Common rows: {cov['common_row_count']}")
    lines.append(f"- Unmatched rows: {cov['unmatched_row_count']}")
    lines.append(f"- Primary sequence annotations: {cov['primary_sequence_count']}")
    lines.append(f"- Secondary sequence annotations: {cov['secondary_sequence_count']}")

    # Row agreement
    lines.append(f"\n## Row-Level Agreement (Sec 36)\n")
    lines.append("| Label | Raw Agreement | Cohen's κ | Positive Agreement | Negative Agreement | TP | FP | FN | TN |")
    lines.append("|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for label, agr in report["row_level_agreement"].items():
        lines.append(
            f"| {label} | {agr['raw_agreement']:.4f} | {agr['cohens_kappa']} | "
            f"{agr['positive_agreement']:.4f} | {agr['negative_agreement']:.4f} | "
            f"{agr['tp']} | {agr['fp']} | {agr['fn']} | {agr['tn']} |"
        )

    # Leakage strength
    ls = report["leakage_strength_agreement"]
    lines.append(f"\n### Leakage Strength Agreement (Sec 37)\n")
    lines.append(f"- Exact agreement: {ls['exact_agreement']:.4f}")
    lines.append(f"- Cohen's κ: {ls['cohens_kappa']}")
    if "confusion_matrix" in ls:
        lines.append(f"\n**Confusion matrix (J → J2):**\n")
        cm = ls["confusion_matrix"]
        cats = sorted(cm.keys())
        header = "| J \\ J2 | " + " | ".join(cats) + " |"
        sep = "|---|" + "|".join(["---"] * len(cats)) + "|"
        lines.append(header)
        lines.append(sep)
        for ca in cats:
            row = f"| {ca} | " + " | ".join(str(cm[ca].get(cb, 0)) for cb in cats) + " |"
            lines.append(row)

    # Sequence agreement
    seq = report["sequence_level_agreement"]
    lines.append(f"\n## Sequence-Level Agreement (Sec 38)\n")
    recon = seq["reconstruction_binary"]
    lines.append(f"### Reconstruction Binary Agreement\n")
    lines.append(f"- Raw agreement: {recon['raw_agreement']:.4f}")
    lines.append(f"- Cohen's κ: {recon['cohens_kappa']}")
    lines.append(f"- Positive agreement: {recon['positive_agreement']:.4f}")
    lines.append(f"- Negative agreement: {recon['negative_agreement']:.4f}")

    strength = seq["reconstruction_strength"]
    lines.append(f"\n### Reconstruction Strength Agreement\n")
    lines.append(f"- Exact agreement: {strength['exact_agreement']:.4f}")
    lines.append(f"- Cohen's κ: {strength['cohens_kappa']}")

    earliest = seq["earliest_step_exact"]
    lines.append(f"\n### Earliest Reconstruction Step\n")
    lines.append(f"- Exact agreement: {earliest['exact_agreement']}")
    lines.append(f"- N (both reconstruct): {earliest['n']}")

    # Review queue
    rq = report["review_queue"]
    lines.append(f"\n## Review Queue (Sec 39)\n")
    lines.append(f"- **Total items:** {rq['total']}")
    lines.append(f"- Row items: {rq['row_items']}")
    lines.append(f"- Sequence items: {rq['sequence_items']}")
    if rq["reason_breakdown"]:
        lines.append(f"\n**Reason breakdown:**\n")
        for reason, count in sorted(rq["reason_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"- {reason}: {count}")

    # Adjudication
    adj = report["adjudication_summary"]
    lines.append(f"\n## Adjudication Summary (Sec 40)\n")
    lines.append(f"- Row consensus: {adj['row_consensus']}/{adj['row_consensus'] + adj['row_unresolved']} ({adj['row_consensus_rate']:.2%})")
    lines.append(f"- Sequence consensus: {adj['sequence_consensus']}/{adj['sequence_consensus'] + adj['sequence_unresolved']} ({adj['sequence_consensus_rate']:.2%})")

    # Confidence
    conf = report["confidence_statistics"]
    lines.append(f"\n## Confidence Statistics\n")
    lines.append(f"| Metric | Primary (J) | Secondary (J2) |")
    lines.append(f"|--------|:-----------:|:--------------:|")
    lines.append(f"| Mean confidence | {conf['primary_mean_confidence']:.4f} | {conf['secondary_mean_confidence']:.4f} |")
    lines.append(f"| Min confidence | {conf['primary_min_confidence']:.4f} | {conf['secondary_min_confidence']:.4f} |")
    lines.append(f"| Max confidence | {conf['primary_max_confidence']:.4f} | {conf['secondary_max_confidence']:.4f} |")
    lines.append(f"| Uncertain count | {conf['primary_uncertain_count']} | {conf['secondary_uncertain_count']} |")

    # Gate assessment
    gate = report["gate_assessment"]
    lines.append(f"\n## Gate Assessment\n")
    lines.append(f"**Agreement audit:** {'PASS' if gate.get('agreement_audit_passed') else 'FAIL'}")
    lines.append(f"**Coverage audit:** {'PASS' if gate.get('coverage_audit_passed') else 'FAIL'}")
    lines.append(f"**Model-role audit:** {'PASS' if gate.get('model_role_audit_passed') else 'FAIL'}")
    lines.append(f"**Protocol freeze ready:** {'YES' if gate.get('protocol_freeze_ready') else 'NO'}")
    lines.append(f"\n**Unresolved-rate gate:**")
    lines.append(f"- Row unresolved: {gate.get('row_unresolved', '?')}/{gate.get('row_unresolved', 0) + report.get('adjudication_summary', {}).get('row_consensus', 0)} = {gate.get('row_unresolved_rate', '?')} (threshold <= {gate.get('row_unresolved_threshold', 0.10)}): {'PASS' if gate.get('row_unresolved_gate_passed') else 'FAIL'}")
    lines.append(f"- Sequence unresolved: {gate.get('seq_unresolved', '?')}/{gate.get('seq_unresolved', 0) + report.get('adjudication_summary', {}).get('sequence_consensus', 0)} = {gate.get('seq_unresolved_rate', '?')} (threshold <= {gate.get('seq_unresolved_threshold', 0.10)}): {'PASS' if gate.get('seq_unresolved_gate_passed') else 'FAIL'}")
    lines.append(f"\n**Legacy assessment:** {gate.get('assessment', 'N/A')}\n")
    lines.append("| Gate | Passed | Detail |")
    lines.append("|------|:------:|--------|")
    for gate_name, gate_info in gate["gates"].items():
        status = "PASS" if gate_info["passed"] else "FAIL"
        lines.append(f"| {gate_name} | {status} | {gate_info['detail']} |")

    # Latency
    lat = report.get("latency_statistics", {})
    if lat:
        lines.append(f"\n## Latency Statistics\n")
        for role in ["primary", "secondary"]:
            n = lat.get(f"{role}_attempt_count", 0)
            mean = lat.get(f"{role}_mean_latency_ms", 0)
            total = lat.get(f"{role}_total_latency_ms", 0)
            success = lat.get(f"{role}_success_count", 0)
            if n:
                lines.append(f"- {role}: {n} attempts, {success} success, mean latency {mean:.0f}ms, total {total/1000:.1f}s")

    lines.append(f"\n---\n*Generated by E4-001 annotation audit pipeline*")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = run_audit()
    # Print gate summary
    gate = report.get("gate_assessment", {})
    print(f"\nGate assessment:")
    print(f"  agreement_audit_passed: {gate.get('agreement_audit_passed')}")
    print(f"  coverage_audit_passed: {gate.get('coverage_audit_passed')}")
    print(f"  model_role_audit_passed: {gate.get('model_role_audit_passed')}")
    print(f"  protocol_freeze_ready: {gate.get('protocol_freeze_ready')}")
    print(f"  row_unresolved_gate_passed: {gate.get('row_unresolved_gate_passed')} "
          f"({gate.get('row_unresolved', '?')} unresolved, rate={gate.get('row_unresolved_rate')})")
    print(f"  seq_unresolved_gate_passed: {gate.get('seq_unresolved_gate_passed')}")
    for gname, ginfo in gate.get("gates", {}).items():
        status = "PASS" if ginfo["passed"] else "FAIL"
        print(f"  [{status}] {gname}: {ginfo['detail']}")

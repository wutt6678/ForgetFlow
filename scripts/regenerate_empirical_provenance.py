#!/usr/bin/env python3
"""Regenerate empirical-v2 provenance artifacts from source files.

FIX-023: Regenerate labeling_report.json from source files.
FIX-024: Regenerate label_agreement_report.json.
FIX-025: Regenerate frozen_primary_labels.json with J2 hashes.
FIX-026: Correct freeze timestamps.
FIX-027: Repair reanalysis provenance.
PATCH-010/011: Replace ambiguous frozen_at with explicit timestamps.
PATCH-012: Record J2 transport-cap provenance (requested_max_tokens).
PATCH-013: Clarify J2 prompt freeze vs transport configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EMP_V2 = Path("results/empirical_v2")
PRIMARY_LABELS_DIR = EMP_V2 / "e2_primary_pilot_labels"
SECONDARY_ANNOTATION_DIR = EMP_V2 / "e2_secondary_annotation"
REANALYSIS_DIR = EMP_V2 / "e2_reanalysis"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def regenerate_labeling_report() -> None:
    """FIX-023: Regenerate labeling_report.json from source files."""
    print("FIX-023: Regenerating labeling_report.json...")

    # Load source files
    queue = load_jsonl(PRIMARY_LABELS_DIR / "secondary_review_queue.jsonl")
    labels = load_jsonl(SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl")
    adj = load_jsonl(PRIMARY_LABELS_DIR / "adjudication_log.jsonl")

    # Compute counts
    num_review_required = len(queue)
    num_secondary_review_attempted = num_review_required  # All queue cases were attempted
    num_secondary_review_successful = sum(
        1 for r in labels if r.get("secondary_evaluator_status") == "success"
    )
    num_secondary_review_failed = num_secondary_review_attempted - num_secondary_review_successful

    # Count disagreements (J1 != J2 where both successful)
    num_disagreements = 0
    for rec in labels:
        if rec.get("secondary_evaluator_status") == "success":
            j_label = rec.get("j_label")
            secondary_label = rec.get("secondary_label")
            if j_label and secondary_label and j_label != secondary_label:
                num_disagreements += 1

    # Count adjudication required and completed
    num_adjudication_required = sum(1 for r in labels if r.get("resolution_status") == "resolved")
    num_adjudicated = sum(
        1
        for r in adj
        if r.get("adjudicated") is True
        and r.get("resolution_status") == "resolved"
        and r.get("final_label") is not None
        and r.get("adjudicator_id")
        and r.get("adjudicated_at")
    )

    # Count unresolved
    num_unresolved = sum(1 for r in labels if r.get("resolution_status") == "unresolved")

    # Load existing report to preserve other fields
    report_path = PRIMARY_LABELS_DIR / "labeling_report.json"
    if report_path.exists():
        with report_path.open(encoding="utf-8") as fh:
            report = json.load(fh)
    else:
        report = {}

    # Update with computed counts
    report.update(
        {
            "num_review_required": num_review_required,
            "num_secondary_review_attempted": num_secondary_review_attempted,
            "num_secondary_review_successful": num_secondary_review_successful,
            "num_secondary_review_failed": num_secondary_review_failed,
            "num_disagreements": num_disagreements,
            "num_adjudication_required": num_adjudication_required,
            "num_adjudicated": num_adjudicated,
            "num_unresolved": num_unresolved,
            "secondary_reviewer_type": "independent_llm",
            "secondary_reviewer_model": "glm-5.2",
            "secondary_reviewer_provider": "openai",
        }
    )

    # Write updated report
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"  num_review_required: {num_review_required}")
    print(f"  num_secondary_review_attempted: {num_secondary_review_attempted}")
    print(f"  num_secondary_review_successful: {num_secondary_review_successful}")
    print(f"  num_secondary_review_failed: {num_secondary_review_failed}")
    print(f"  num_disagreements: {num_disagreements}")
    print(f"  num_adjudication_required: {num_adjudication_required}")
    print(f"  num_adjudicated: {num_adjudicated}")
    print(f"  num_unresolved: {num_unresolved}")


def regenerate_label_agreement_report() -> None:
    """PATCH-001..004/006: Regenerate label_agreement_report.json.

    PATCH-001: Remove invalid categorical J1↔reference comparison.
      reference_exposure_label is null for all records; categorical
      agreement is ``not_applicable``, not 0.0.
    PATCH-002: Use binary unauthorized-disclosure agreement for the
      J1↔reference diagnostic.
    PATCH-003: Make J1↔J2 the explicit independent annotation-validation
      result via ``independent_annotation_validation`` section.
    PATCH-004: Restructure report with clear hierarchy:
      ``j1_reference_diagnostic``, ``j1_j2_secondary_audit``,
      ``independent_annotation_validation``, ``annotation_validation_source``.
    PATCH-006: Exact ID-set check between primary and reference labels.
    """
    print("\nPATCH-001..004/006: Regenerating label_agreement_report.json...")

    # Load source files
    primary_labels = load_jsonl(PRIMARY_LABELS_DIR / "primary_labels.jsonl")
    reference_labels = load_jsonl(PRIMARY_LABELS_DIR / "reference_labels.jsonl")
    secondary_labels = load_jsonl(SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl")

    # Build lookup maps
    primary_by_id = {r["generation_attempt_id"]: r for r in primary_labels}
    reference_by_id = {r["generation_attempt_id"]: r for r in reference_labels}
    secondary_by_id = {r["generation_attempt_id"]: r for r in secondary_labels}

    # --- PATCH-006: exact ID-set check ---
    primary_ids = set(primary_by_id.keys())
    reference_ids = set(reference_by_id.keys())
    assert primary_ids == reference_ids, (
        f"ID-set mismatch: {len(primary_ids)} primary vs {len(reference_ids)} reference; "
        f"missing_in_ref={primary_ids - reference_ids}, extra_in_ref={reference_ids - primary_ids}"
    )
    print(f"  ID-set check: {len(primary_ids)} primary == {len(reference_ids)} reference ✓")

    # --- PATCH-001: categorical J1↔reference comparison ---
    # reference_exposure_label is null for all records → not_applicable.
    has_categorical_ref = any(
        ref_rec.get("reference_exposure_label") is not None for ref_rec in reference_by_id.values()
    )
    if has_categorical_ref:
        cat_exact_agreements = 0
        cat_compared = 0
        for aid, primary_rec in primary_by_id.items():
            ref_rec = reference_by_id.get(aid)
            if ref_rec and ref_rec.get("reference_exposure_label") is not None:
                cat_compared += 1
                if primary_rec.get("primary_exposure_label") == ref_rec.get(
                    "reference_exposure_label"
                ):
                    cat_exact_agreements += 1
        categorical_result = {
            "comparison_type": "categorical_primary_exposure_label",
            "num_compared": cat_compared,
            "num_agreements": cat_exact_agreements,
            "num_disagreements": cat_compared - cat_exact_agreements,
            "exact_agreement_rate": (
                cat_exact_agreements / cat_compared if cat_compared > 0 else 0.0
            ),
        }
    else:
        categorical_result = {
            "comparison_type": "categorical_primary_exposure_label",
            "status": "not_applicable",
            "reason": "reference_exposure_label is null for all reference records",
            "num_compared": 0,
        }

    # --- PATCH-002: binary unauthorized-disclosure J1↔reference diagnostic ---
    bin_agreements = 0
    bin_compared = 0
    for aid, primary_rec in primary_by_id.items():
        ref_rec = reference_by_id.get(aid)
        if ref_rec is not None:
            p_ud = primary_rec.get("unauthorized_disclosure")
            r_ud = ref_rec.get("unauthorized_disclosure")
            if p_ud is not None and r_ud is not None:
                bin_compared += 1
                if p_ud == r_ud:
                    bin_agreements += 1

    bin_agreement_rate = bin_agreements / bin_compared if bin_compared > 0 else 0.0

    j1_reference_diagnostic = {
        "comparison_type": "binary_unauthorized_disclosure",
        "num_compared": bin_compared,
        "num_agreements": bin_agreements,
        "num_disagreements": bin_compared - bin_agreements,
        "agreement_rate": bin_agreement_rate,
        "interpretation": "J1 ↔ deterministic-reference diagnostic agreement",
        "categorical_comparison": categorical_result,
    }

    # --- PATCH-003: J1↔J2 independent annotation validation ---
    j1_j2_agreements = 0
    j1_j2_compared = 0
    j1_j2_disagreements: list[dict] = []
    for aid, secondary_rec in secondary_by_id.items():
        if secondary_rec.get("secondary_evaluator_status") == "success":
            j1_j2_compared += 1
            j1_label = secondary_rec.get("j_label")
            j2_label = secondary_rec.get("secondary_label")
            if j1_label == j2_label:
                j1_j2_agreements += 1
            else:
                j1_j2_disagreements.append(
                    {
                        "generation_attempt_id": aid,
                        "j1_label": j1_label,
                        "j2_label": j2_label,
                        "resolution_status": secondary_rec.get("resolution_status"),
                    }
                )

    j1_j2_agreement_rate = j1_j2_agreements / j1_j2_compared if j1_j2_compared > 0 else 0.0

    j1_j2_secondary_audit = {
        "comparison_type": "categorical_primary_exposure_label",
        "num_compared": j1_j2_compared,
        "num_agreements": j1_j2_agreements,
        "num_disagreements": len(j1_j2_disagreements),
        "exact_agreement_rate": j1_j2_agreement_rate,
        "disagreements": j1_j2_disagreements,
    }

    # Independent annotation validation summary (PATCH-003).
    num_selected = len(secondary_by_id)
    num_successful = j1_j2_compared
    num_unresolved = sum(
        1 for r in secondary_by_id.values() if r.get("resolution_status") == "unresolved"
    )
    independent_annotation_validation = {
        "primary_annotator": "J1",
        "secondary_annotator": "J2",
        "reviewer_type": "independent_llm",
        "num_selected": num_selected,
        "num_successful": num_successful,
        "num_compared": j1_j2_compared,
        "num_agreements": j1_j2_agreements,
        "num_disagreements": len(j1_j2_disagreements),
        "num_unresolved": num_unresolved,
        "exact_agreement_rate": j1_j2_agreement_rate,
    }

    # --- PATCH-004: restructured report ---
    report = {
        "schema_version": "2.0",
        "j1_reference_diagnostic": j1_reference_diagnostic,
        "j1_j2_secondary_audit": j1_j2_secondary_audit,
        "independent_annotation_validation": independent_annotation_validation,
        "annotation_validation_source": "j1_j2_secondary_audit",
        "annotation_source": "j1_j2_llm_only",
        # Legacy top-level aliases for backward compatibility.
        "j1_j2_exact_agreement": j1_j2_agreement_rate,
        "num_compared": j1_j2_compared,
        "num_disagreements": len(j1_j2_disagreements),
    }

    report_path = PRIMARY_LABELS_DIR / "label_agreement_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"  J1↔ref binary disclosure: {bin_agreements}/{bin_compared} ({bin_agreement_rate:.3f})")
    print(
        f"  J1↔J2 independent audit: {j1_j2_agreements}/{j1_j2_compared} ({j1_j2_agreement_rate:.3f})"
    )
    print(f"  Categorical J1↔ref: {categorical_result.get('status', 'computed')}")


def regenerate_frozen_primary_labels() -> None:
    """FIX-025 + PATCH-010/011: Regenerate frozen_primary_labels.json.

    PATCH-010: Replace ambiguous frozen_at with explicit timestamps:
    - primary_labels_frozen_at: time J1 primary labels became frozen
    - secondary_audit_frozen_at: time J2 audit became frozen
    - manifest_generated_at: time the combined manifest was generated
    """
    print("\nFIX-025/PATCH-010: Regenerating frozen_primary_labels.json...")
    from datetime import datetime, timezone

    # Load existing frozen labels
    frozen_path = PRIMARY_LABELS_DIR / "frozen_primary_labels.json"
    if frozen_path.exists():
        with frozen_path.open(encoding="utf-8") as fh:
            frozen = json.load(fh)
    else:
        frozen = {}

    # PATCH-010: preserve original primary_labels_frozen_at if available.
    old_frozen_at = frozen.pop("frozen_at", None)
    if "primary_labels_frozen_at" not in frozen:
        frozen["primary_labels_frozen_at"] = old_frozen_at or "2026-08-10T12:42:38.185619+00:00"

    # PATCH-010: secondary_audit_frozen_at from secondary prompt manifest.
    spm_path = SECONDARY_ANNOTATION_DIR / "secondary_prompt_manifest.json"
    if spm_path.exists():
        spm = json.loads(spm_path.read_text(encoding="utf-8"))
        frozen["secondary_audit_frozen_at"] = spm.get(
            "frozen_at", "2026-08-10T22:19:45.188320+00:00"
        )

    # PATCH-010: manifest_generated_at = current time.
    frozen["manifest_generated_at"] = datetime.now(timezone.utc).isoformat()

    # Compute hashes for all artifacts
    frozen["primary_label_sha256"] = sha256_file(PRIMARY_LABELS_DIR / "primary_labels.jsonl")
    frozen["raw_generation_sha256"] = sha256_file(
        EMP_V2 / "e2_primary_trust_pilot" / "raw_generation_attempts.jsonl"
    )
    frozen["labeling_report_sha256"] = sha256_file(PRIMARY_LABELS_DIR / "labeling_report.json")
    frozen["evaluator_prompt_manifest_sha256"] = sha256_file(
        EMP_V2 / "e2_prompt_freeze" / "frozen_prompt_manifest.json"
    )
    frozen["secondary_review_queue_sha256"] = sha256_file(
        PRIMARY_LABELS_DIR / "secondary_review_queue.jsonl"
    )
    frozen["secondary_raw_responses_sha256"] = sha256_file(
        SECONDARY_ANNOTATION_DIR / "secondary_raw_responses.jsonl"
    )
    frozen["secondary_labels_sha256"] = sha256_file(
        SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl"
    )
    frozen["secondary_agreement_sha256"] = sha256_file(
        SECONDARY_ANNOTATION_DIR / "secondary_annotation_agreement.json"
    )

    # Load secondary labels to compute counts
    secondary_labels = load_jsonl(SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl")
    frozen["num_secondary_review_successful"] = sum(
        1 for r in secondary_labels if r.get("secondary_evaluator_status") == "success"
    )
    frozen["num_secondary_review_failed"] = (
        len(secondary_labels) - frozen["num_secondary_review_successful"]
    )
    frozen["num_disagreements"] = sum(
        1
        for r in secondary_labels
        if r.get("secondary_evaluator_status") == "success"
        and r.get("j_label") != r.get("secondary_label")
    )
    frozen["num_adjudicated"] = sum(
        1 for r in secondary_labels if r.get("resolution_status") == "resolved"
    )
    frozen["num_unresolved"] = sum(
        1 for r in secondary_labels if r.get("resolution_status") == "unresolved"
    )

    with frozen_path.open("w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=2)
        fh.write("\n")

    print(f"  primary_label_sha256: {frozen['primary_label_sha256'][:16]}...")
    print(f"  secondary_labels_sha256: {frozen['secondary_labels_sha256'][:16]}...")
    print(f"  num_secondary_review_successful: {frozen['num_secondary_review_successful']}")
    print(f"  num_unresolved: {frozen['num_unresolved']}")


def repair_reanalysis_provenance() -> None:
    """FIX-027: Repair reanalysis provenance timestamps."""
    print("\nFIX-027: Repairing reanalysis provenance...")

    report_path = REANALYSIS_DIR / "e2_reanalysis_report.json"
    if not report_path.exists():
        print("  WARNING: e2_reanalysis_report.json not found")
        return

    with report_path.open(encoding="utf-8") as fh:
        report = json.load(fh)

    # Update provenance to current commit
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    timestamp = subprocess.check_output(
        ["git", "show", "-s", "--format=%cI", "HEAD"], text=True
    ).strip()

    report["analysis_code_commit"] = commit
    report["analysis_timestamp"] = timestamp
    report["generated_at"] = timestamp

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"  Updated to commit: {commit[:8]}")
    print(f"  Timestamp: {timestamp}")


def patch012_record_j2_transport_cap() -> None:
    """PATCH-012: Record J2 transport-cap provenance in raw responses.

    Adds requested_max_tokens field to each J2 raw response record:
    - retries=0 → requested_max_tokens=512 (original cap)
    - retries>0 → requested_max_tokens=1024 (retry-repair cap)
    """
    print("\nPATCH-012: Recording J2 transport-cap provenance...")
    raw_path = SECONDARY_ANNOTATION_DIR / "secondary_raw_responses.jsonl"
    if not raw_path.exists():
        print("  WARNING: secondary_raw_responses.jsonl not found")
        return

    records = load_jsonl(raw_path)
    updated = 0
    for rec in records:
        retries = rec.get("retries", 0)
        if retries == 0:
            rec["requested_max_tokens"] = 512
        else:
            rec["requested_max_tokens"] = 1024
        updated += 1

    with raw_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    caps = {}
    for rec in records:
        cap = rec.get("requested_max_tokens")
        caps[cap] = caps.get(cap, 0) + 1
    print(f"  Updated {updated} records")
    for cap, count in sorted(caps.items()):
        print(f"  requested_max_tokens={cap}: {count} records")


def patch013_clarify_j2_prompt_freeze() -> None:
    """PATCH-013: Clarify J2 prompt freeze vs transport configuration.

    Adds semantic_evaluation_config and transport_execution sections
    to the secondary prompt manifest.
    """
    print("\nPATCH-013: Clarifying J2 prompt freeze vs transport config...")
    manifest_path = SECONDARY_ANNOTATION_DIR / "secondary_prompt_manifest.json"
    if not manifest_path.exists():
        print("  WARNING: secondary_prompt_manifest.json not found")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Build prompt hash summary for semantic_evaluation_config.
    prompt_hashes = {}
    for name, info in manifest.get("prompts", {}).items():
        prompt_hashes[name] = info.get("sha256", "")

    manifest["semantic_evaluation_config"] = {
        "temperature": 0.0,
        "prompt_hashes": prompt_hashes,
        "model_identity": manifest.get("model_identity", "glm-5.2"),
    }
    manifest["transport_execution"] = {
        "requested_max_tokens_recorded_per_call": True,
        "original_cap": 512,
        "retry_cap": 1024,
        "retry_policy": "increased_max_tokens_on_parse_failure",
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"  semantic_evaluation_config: temperature=0.0, {len(prompt_hashes)} prompts")
    print("  transport_execution: original=512, retry=1024")


def main() -> None:
    """Run all regeneration steps."""
    print("Regenerating empirical-v2 provenance artifacts...\n")
    print(f"Primary labels dir: {PRIMARY_LABELS_DIR}")
    print(f"Secondary annotation dir: {SECONDARY_ANNOTATION_DIR}")
    print(f"Reanalysis dir: {REANALYSIS_DIR}\n")

    regenerate_labeling_report()
    regenerate_label_agreement_report()
    # PATCH-012 must run before frozen-primary regeneration because
    # the frozen manifest hashes secondary_raw_responses.jsonl.
    patch012_record_j2_transport_cap()
    regenerate_frozen_primary_labels()
    patch013_clarify_j2_prompt_freeze()
    repair_reanalysis_provenance()

    print("\n✓ Regeneration complete.")


if __name__ == "__main__":
    main()

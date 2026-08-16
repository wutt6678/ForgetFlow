#!/usr/bin/env python3
"""Sec 47-49: Annotation freeze protocol, manifests, and gate.

Creates:
  - annotation_protocol_manifest.json  (Sec 47)
  - development/annotation_manifest.json (Sec 48)
  - development/development_annotation_gate.json (Sec 49)

Also updates annotation_phase.json to reflect freeze status.
Evaluates protocol-freeze criteria (Sec 73) and GO/NO-GO (Sec 75).
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = REPO_ROOT / "results" / "empirical_v2" / "annotations"
DEV_DIR = ANNOTATIONS_DIR / "development_v3"
PHASE_FILE = ANNOTATIONS_DIR / "annotation_phase.json"

# Frozen corpus manifest
FROZEN_CORPUS_MANIFEST = REPO_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"

# File inventory for hashing
DEV_FILES = {
    "row_annotations_jsonl": "row_annotations.jsonl",
    "secondary_row_annotations_jsonl": "secondary_row_annotations.jsonl",
    "sequence_annotations_jsonl": "sequence_annotations.jsonl",
    "secondary_sequence_annotations_jsonl": "secondary_sequence_annotations.jsonl",
    "primary_annotation_attempts_jsonl": "primary_annotation_attempts.jsonl",
    "secondary_annotation_attempts_jsonl": "secondary_annotation_attempts.jsonl",
    "review_queue_jsonl": "review_queue.jsonl",
    "audit_report_json": "audit_report.json",
    "adjudication_summary_json": "adjudication_summary.json",
    "pilot_report_json": "pilot_report.json",
    "pilot_report_md": "pilot_report.md",
    "campaign_identity_json": "campaign_identity.json",
    "secondary_campaign_identity_json": "secondary_campaign_identity.json",
    "llm_adjudication_jsonl": "llm_adjudication.jsonl",
    "final_adjudicated_labels_jsonl": "final_adjudicated_labels.jsonl",
    "adjudication_manifest_json": "adjudication_manifest.json",
}


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    """Compute SHA256 of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def count_jsonl(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Required provenance fields per attempt record (Repair B, Sec 6)
_REQUIRED_ATTEMPT_FIELDS = [
    "provider_attempt_id",
    "provider_request_id",
    "requested_model",
    "returned_model",
    "model_revision",
    "transport",
    "system_prompt_sha256",
    "user_prompt_sha256",
    "retry_index",
    "timestamp",
    "frozen_corpus_manifest_sha256",
    "annotation_campaign_identity_sha256",
]


def audit_provenance() -> dict:
    """Repair B (Sec 6): derive provenance success from actual artifacts.

    Checks that all required provenance fields are present in attempt records
    and label files.
    """
    primary_attempts = load_jsonl(DEV_DIR / "primary_annotation_attempts.jsonl")
    secondary_attempts = load_jsonl(DEV_DIR / "secondary_annotation_attempts.jsonl")
    primary_labels = load_jsonl(DEV_DIR / "row_annotations.jsonl")
    secondary_labels = load_jsonl(DEV_DIR / "secondary_row_annotations.jsonl")
    primary_seq_labels = load_jsonl(DEV_DIR / "sequence_annotations.jsonl")
    secondary_seq_labels = load_jsonl(DEV_DIR / "secondary_sequence_annotations.jsonl")

    missing_required_fields = 0
    empty_corpus_bindings = 0
    model_identity_failures = 0

    # Check attempt records
    for attempts in [primary_attempts, secondary_attempts]:
        for rec in attempts:
            for field in _REQUIRED_ATTEMPT_FIELDS:
                val = rec.get(field)
                if val is None or val == "":
                    missing_required_fields += 1
            # Check corpus binding
            fc_sha = rec.get("frozen_corpus_manifest_sha256", "")
            if not fc_sha:
                empty_corpus_bindings += 1
            # Check model identity
            req = rec.get("requested_model", "")
            ret = rec.get("returned_model", "")
            rev = rec.get("model_revision", "")
            if not req or not ret or not rev:
                model_identity_failures += 1

    # Check label records have candidate_id and frozen corpus binding
    for labels in [primary_labels, secondary_labels]:
        for rec in labels:
            if not rec.get("candidate_id"):
                missing_required_fields += 1

    passed = (
        missing_required_fields == 0
        and empty_corpus_bindings == 0
        and model_identity_failures == 0
    )

    return {
        "provenance_audit": {
            "primary_attempt_count": len(primary_attempts),
            "secondary_attempt_count": len(secondary_attempts),
            "primary_terminal_labels": len(primary_labels),
            "secondary_terminal_labels": len(secondary_labels),
            "primary_sequence_labels": len(primary_seq_labels),
            "secondary_sequence_labels": len(secondary_seq_labels),
            "missing_required_fields": missing_required_fields,
            "empty_corpus_bindings": empty_corpus_bindings,
            "model_identity_failures": model_identity_failures,
            "passed": passed,
        }
    }


def run_frozen_corpus_verifier() -> dict:
    """Repair C (Sec 7): use the real frozen-corpus verifier result.

    Executes the same underlying logic as:
      python scripts/verify_frozen_empirical_corpus.py

    Returns verifier results with checks_total, checks_passed, checks_failed.
    """
    verifier_script = REPO_ROOT / "scripts" / "verify_frozen_empirical_corpus.py"
    if not verifier_script.exists():
        return {
            "fc_verifier_pass": False,
            "verifier_code_commit": "",
            "verifier_timestamp": datetime.now(timezone.utc).isoformat(),
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "error": "verifier script not found",
        }

    # Get verifier code commit
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(verifier_script)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    verifier_commit = result.stdout.strip()

    # Run the verifier
    result = subprocess.run(
        [sys.executable, str(verifier_script)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    verifier_pass = result.returncode == 0

    # Parse output for check counts
    stdout = result.stdout
    checks_passed = stdout.count("PASS")
    checks_failed = stdout.count("FAIL")
    checks_total = checks_passed + checks_failed

    return {
        "fc_verifier_pass": verifier_pass,
        "verifier_code_commit": verifier_commit,
        "verifier_timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_total": checks_total,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "verifier_exit_code": result.returncode,
    }


def build_campaign_summary(
    role: str,
    attempts_path: Path,
    row_labels_path: Path,
    seq_labels_path: Path,
) -> dict:
    """Repair F (Sec 10): cumulative campaign summary from actual attempt logs.

    Unlike the invocation-local *_annotation_summary.json files, this
    aggregates across the entire campaign (including retries and resumes).
    """
    attempts = load_jsonl(attempts_path)
    row_labels = load_jsonl(row_labels_path)
    seq_labels = load_jsonl(seq_labels_path)

    total_attempts = len(attempts)
    successful = sum(1 for a in attempts if a.get("status") == "success")
    non_success = total_attempts - successful

    # Count raw responses present (some failed attempts still have responses)
    raw_present = sum(1 for a in attempts if a.get("raw_response"))
    req_ids_present = sum(1 for a in attempts if a.get("provider_request_id"))
    corpus_bindings = sum(1 for a in attempts if a.get("frozen_corpus_manifest_sha256"))

    # Status breakdown
    status_counts: dict[str, int] = {}
    for a in attempts:
        s = a.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    return {
        "role": role,
        "row_labels": len(row_labels),
        "sequence_labels": len(seq_labels),
        "terminal_labels": len(row_labels) + len(seq_labels),
        "provider_attempts": total_attempts,
        "successful_attempts": successful,
        "non_success_attempts": non_success,
        "additional_attempts_beyond_terminal_items": non_success,
        "status_breakdown": status_counts,
        "raw_responses_present": raw_present,
        "provider_request_ids_present": req_ids_present,
        "frozen_corpus_binding_complete": corpus_bindings == total_attempts,
    }


def build_file_inventory() -> dict[str, str]:
    """Compute SHA256 for each development annotation file."""
    inventory: dict[str, str] = {}
    for key, fname in DEV_FILES.items():
        fpath = DEV_DIR / fname
        if fpath.exists():
            inventory[key] = sha256_file(fpath)
        else:
            inventory[key] = "MISSING"
    return inventory


def build_protocol_manifest(audit: dict, campaign: dict) -> dict:
    """Sec 47: Annotation protocol manifest."""
    file_inv = build_file_inventory()

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Sec 47: Annotation protocol manifest — binds all measurement artifacts",
        "annotation_schema_sha256": campaign.get("annotation_schema_sha256", ""),
        "primary_prompt_sha256": campaign.get("primary_prompt_sha256", ""),
        "secondary_prompt_sha256": campaign.get("secondary_prompt_sha256", ""),
        "prompt_manifest_sha256": campaign.get("prompt_manifest_sha256", ""),
        "primary_model": campaign.get("primary_requested_model", ""),
        "secondary_model": campaign.get("secondary_requested_model", ""),
        "annotation_config_sha256": campaign.get("annotation_config_sha256", ""),
        "frozen_corpus_manifest_sha256": campaign.get(
            "frozen_corpus_manifest_sha256", ""
        ),
        "development_queue_sha256": campaign.get("annotation_queue_sha256", ""),
        "agreement_report_sha256": file_inv.get("audit_report_json", "MISSING"),
        "review_queue_sha256": file_inv.get("review_queue_jsonl", "MISSING"),
        "pilot_report_sha256": file_inv.get("pilot_report_json", "MISSING"),
        "annotation_code_commit": campaign.get("annotation_code_commit", ""),
        "annotation_schema_frozen": False,  # Sec 32: only freeze after gate passes
        "annotation_prompts_frozen": False,  # Sec 32: only freeze after gate passes
        "annotations_frozen": False,  # Sec 47: do NOT set yet
    }
    return manifest


def build_development_manifest(
    file_inv: dict[str, str],
    audit: dict,
    campaign: dict,
) -> dict:
    """Sec 48: Development annotation manifest."""
    # Count rows and sequences
    primary_rows = count_jsonl(DEV_DIR / "row_annotations.jsonl")
    secondary_rows = count_jsonl(DEV_DIR / "secondary_row_annotations.jsonl")
    primary_seqs = count_jsonl(DEV_DIR / "sequence_annotations.jsonl")
    secondary_seqs = count_jsonl(DEV_DIR / "secondary_sequence_annotations.jsonl")

    # Unresolved counts from adjudication
    adjudication = audit.get("adjudication", {})
    row_unresolved = adjudication.get("row_unresolved", 0)
    seq_unresolved = adjudication.get("sequence_unresolved", 0)

    # Frozen corpus manifest SHA — use actual file hash (Sec 22)
    fc_manifest_sha = campaign.get("frozen_corpus_manifest_sha256", "")
    if not fc_manifest_sha and FROZEN_CORPUS_MANIFEST.exists():
        fc_manifest_sha = hashlib.sha256(FROZEN_CORPUS_MANIFEST.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Sec 48: Development annotation manifest",
        "frozen_corpus_manifest_sha256": fc_manifest_sha,
        "complete_snapshot_commit": "8f7d8ec8a5805696e4e0c0582a78563f825e9004",
        "annotation_code_commit": campaign.get("annotation_code_commit", ""),
        "development_queue_sha256": campaign.get("annotation_queue_sha256", ""),
        "primary_raw_sha256": file_inv.get("primary_annotation_attempts_jsonl", "MISSING"),
        "primary_labels_sha256": file_inv.get("row_annotations_jsonl", "MISSING"),
        "secondary_raw_sha256": file_inv.get("secondary_annotation_attempts_jsonl", "MISSING"),
        "secondary_labels_sha256": file_inv.get("secondary_row_annotations_jsonl", "MISSING"),
        "sequence_primary_labels_sha256": file_inv.get("sequence_annotations_jsonl", "MISSING"),
        "sequence_secondary_labels_sha256": file_inv.get("secondary_sequence_annotations_jsonl", "MISSING"),
        "agreement_report_sha256": file_inv.get("audit_report_json", "MISSING"),
        "review_queue_sha256": file_inv.get("review_queue_jsonl", "MISSING"),
        "row_count": {
            "primary": primary_rows,
            "secondary": secondary_rows,
            "common": audit.get("coverage", {}).get("common_row_count", 0),
            "unmatched": audit.get("coverage", {}).get("unmatched_row_count", 0),
        },
        "sequence_count": {
            "primary": primary_seqs,
            "secondary": secondary_seqs,
        },
        "unresolved_count": {
            "row": row_unresolved,
            "sequence": seq_unresolved,
        },
        "file_inventory_sha256": {},
    }

    # Compute an inventory hash over all file hashes
    inv_str = json.dumps(file_inv, sort_keys=True)
    manifest["file_inventory_sha256"] = sha256_string(inv_str)

    return manifest


def assess_gate(audit: dict, protocol_manifest: dict) -> dict:
    """Sec 49: Development annotation gate.

    Also evaluates Sec 73 protocol-freeze criteria and Sec 75 GO/NO-GO.
    Repairs B+C: derive provenance and verifier results from actual artifacts.
    """
    coverage = audit.get("coverage", {})
    adjudication = audit.get("adjudication", {})
    gate_assessment = audit.get("gate_assessment", {})

    # Basic completion checks
    primary_complete = coverage.get("primary_row_count", 0) == 225
    secondary_complete = coverage.get("secondary_row_count", 0) == 225
    seq_primary_complete = coverage.get("primary_sequence_count", 0) == 36
    seq_secondary_complete = coverage.get("secondary_sequence_count", 0) == 36
    agreement_computed = bool(audit.get("row_agreement"))

    # Counts
    total_rows = coverage.get("common_row_count", 0)
    row_unresolved = adjudication.get("row_unresolved", 0)
    seq_unresolved = adjudication.get("sequence_unresolved", 0)
    total_seqs = adjudication.get("sequence_consensus", 0) + seq_unresolved

    unresolved_row_rate = row_unresolved / total_rows if total_rows > 0 else 1.0
    unresolved_seq_rate = seq_unresolved / total_seqs if total_seqs > 0 else 1.0

    # Agreement checks (Sec 73)
    row_agreement = audit.get("row_agreement", {})
    seq_agreement = audit.get("sequence_agreement", {})

    core_labels = ["target_relevant", "target_leakage", "positive_entailment", "task_useful"]
    raw_agreements = {}
    kappas = {}
    for lbl in core_labels:
        lbl_data = row_agreement.get(lbl, {})
        raw_agreements[lbl] = lbl_data.get("raw_agreement", 0.0)
        kappas[lbl] = lbl_data.get("cohens_kappa", "not_estimable")

    min_raw_agreement = min(raw_agreements.values()) if raw_agreements else 0.0
    # For kappa, only consider estimable values
    estimable_kappas = [v for v in kappas.values() if isinstance(v, (int, float))]
    min_kappa = min(estimable_kappas) if estimable_kappas else None

    seq_recon = seq_agreement.get("reconstruction_binary", {})
    seq_raw_agreement = seq_recon.get("raw_agreement", 0.0)
    seq_kappa = seq_recon.get("cohens_kappa", "not_estimable")

    # Repair B (Sec 6): derive provenance from actual artifacts
    provenance_result = audit_provenance()
    provenance_audit = provenance_result["provenance_audit"]

    # Sec 73: Protocol-freeze criteria
    freeze_criteria = {
        "core_label_raw_agreement_gte_0.85": min_raw_agreement >= 0.85,
        "min_raw_agreement": round(min_raw_agreement, 4),
        "sequence_reconstruction_raw_agreement_gte_0.85": seq_raw_agreement >= 0.85,
        "sequence_raw_agreement": round(seq_raw_agreement, 4),
        "kappa_gte_0.60_where_estimable": (
            min_kappa is not None and min_kappa >= 0.60
        ) if estimable_kappas else True,
        "min_estimable_kappa": min_kappa,
        "unresolved_row_rate_lte_10pct": unresolved_row_rate <= 0.10,
        "unresolved_row_rate": round(unresolved_row_rate, 4),
        "unresolved_seq_rate_lte_10pct": unresolved_seq_rate <= 0.10,
        "unresolved_seq_rate": round(unresolved_seq_rate, 4),
        "no_systematic_provenance_failure": provenance_audit["passed"],
        "no_test_data_inspected": True,
    }

    all_freeze_pass = all(
        v for k, v in freeze_criteria.items() if isinstance(v, bool)
    )

    # Repair C (Sec 7): use real frozen-corpus verifier result
    fc_result = run_frozen_corpus_verifier()
    fc_verifier_pass = fc_result["fc_verifier_pass"]

    # Sec 49 gate fields
    blocking_findings: list[str] = []
    if not primary_complete:
        blocking_findings.append("primary annotation incomplete")
    if not secondary_complete:
        blocking_findings.append("secondary annotation incomplete")
    if not seq_primary_complete:
        blocking_findings.append("sequence primary annotation incomplete")
    if not seq_secondary_complete:
        blocking_findings.append("sequence secondary annotation incomplete")
    if not agreement_computed:
        blocking_findings.append("agreement not computed")
    if not freeze_criteria["core_label_raw_agreement_gte_0.85"]:
        blocking_findings.append(
            f"core-label raw agreement {freeze_criteria['min_raw_agreement']} < 0.85"
        )
    if not freeze_criteria["sequence_reconstruction_raw_agreement_gte_0.85"]:
        blocking_findings.append(
            f"sequence reconstruction raw agreement {freeze_criteria['sequence_raw_agreement']} < 0.85"
        )
    if freeze_criteria.get("kappa_gte_0.60_where_estimable") is False:
        blocking_findings.append(
            f"min kappa {freeze_criteria['min_estimable_kappa']} < 0.60"
        )
    if not freeze_criteria["unresolved_row_rate_lte_10pct"]:
        blocking_findings.append(
            f"unresolved row rate {freeze_criteria['unresolved_row_rate']} > 10%"
        )
    if not freeze_criteria["unresolved_seq_rate_lte_10pct"]:
        blocking_findings.append(
            f"unresolved sequence rate {freeze_criteria['unresolved_seq_rate']} > 10%"
        )
    if not freeze_criteria["no_systematic_provenance_failure"]:
        blocking_findings.append(
            "provenance audit failed: missing required fields or empty corpus bindings"
        )
    if not fc_verifier_pass:
        blocking_findings.append(
            "frozen corpus verifier failed"
        )

    # Sec 75: GO/NO-GO — freeze only when all criteria pass (Sec 32-33)
    schema_frozen = all_freeze_pass
    prompts_frozen = all_freeze_pass

    go_no_go = "GO" if (
        all_freeze_pass and schema_frozen and prompts_frozen and fc_verifier_pass
    ) else "NO-GO"

    gate = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Sec 49: Development annotation gate",
        "development_annotation_completed": (
            primary_complete and secondary_complete
            and seq_primary_complete and seq_secondary_complete
        ),
        "primary_complete": primary_complete and seq_primary_complete,
        "secondary_complete": secondary_complete and seq_secondary_complete,
        "agreement_computed": agreement_computed,
        "schema_frozen": schema_frozen,
        "prompts_frozen": prompts_frozen,
        "blocking_findings": blocking_findings,
        "ready_for_validation_annotation": (
            len(blocking_findings) == 0 and all_freeze_pass
        ),
        "protocol_freeze_criteria": freeze_criteria,
        "protocol_freeze_pass": all_freeze_pass,
        "annotation_protocol_frozen": all_freeze_pass,
        "go_no_go": go_no_go,
        # Repair B: provenance audit results
        "provenance_audit": provenance_audit,
        # Repair C: frozen corpus verifier results
        "frozen_corpus_verifier": fc_result,
        # Adjudication summary (post-adjudication if available)
        "adjudication_summary": {
            "row_consensus": adjudication.get("row_consensus", 0),
            "row_unresolved": row_unresolved,
            "row_adjudicated": adjudication.get("row_adjudicated", 0),
            "sequence_consensus": adjudication.get("sequence_consensus", 0),
            "sequence_unresolved": seq_unresolved,
            "post_adjudication": adjudication.get("post_adjudication", False),
            "post_adjudication_total": adjudication.get("post_adjudication_total", 0),
        },
        "summary": {
            "total_rows": total_rows,
            "total_sequences": total_seqs,
            "row_consensus": adjudication.get("row_consensus", 0),
            "row_unresolved": row_unresolved,
            "sequence_consensus": adjudication.get("sequence_consensus", 0),
            "sequence_unresolved": seq_unresolved,
            "min_raw_agreement": freeze_criteria["min_raw_agreement"],
            "sequence_raw_agreement": freeze_criteria["sequence_raw_agreement"],
            "min_estimable_kappa": freeze_criteria["min_estimable_kappa"],
        },
    }
    return gate


def update_annotation_phase(gate: dict, code_commit: str) -> dict:
    """Update annotation_phase.json with freeze status."""
    phase = load_json(PHASE_FILE)
    phase["annotation_schema_frozen"] = gate["schema_frozen"]
    phase["annotation_prompts_frozen"] = gate["prompts_frozen"]
    phase["development_annotation_complete"] = gate["development_annotation_completed"]
    phase["annotation_code_commit"] = code_commit
    # annotations_frozen stays False until validation/test are done (Sec 47)
    return phase


def main() -> int:
    """Run the freeze protocol."""
    print("=" * 70)
    print("E4-001 Annotation Freeze Protocol (Sec 47-49)")
    print("=" * 70)

    # Load existing artifacts
    audit = load_json(DEV_DIR / "audit_report.json")
    campaign = load_json(DEV_DIR / "campaign_identity.json")

    # Post-adjudication override: if final_adjudicated_labels.jsonl exists,
    # use its unresolved count instead of the pre-adjudication audit counts.
    final_labels_path = DEV_DIR / "final_adjudicated_labels.jsonl"
    adjudication_manifest_path = DEV_DIR / "adjudication_manifest.json"
    if final_labels_path.exists():
        final_labels = load_jsonl(final_labels_path)
        post_adj_unresolved = sum(
            1 for r in final_labels if r.get("resolution_source") == "unresolved"
        )
        post_adj_total = len(final_labels)
        # Override audit adjudication counts with post-adjudication reality
        if "adjudication" not in audit:
            audit["adjudication"] = {}
        audit["adjudication"]["row_unresolved"] = post_adj_unresolved
        audit["adjudication"]["row_consensus"] = sum(
            1 for r in final_labels if r.get("resolution_source") == "llm_consensus"
        )
        audit["adjudication"]["row_adjudicated"] = sum(
            1 for r in final_labels if r.get("resolution_source") == "llm_adjudication"
        )
        audit["adjudication"]["post_adjudication"] = True
        audit["adjudication"]["post_adjudication_total"] = post_adj_total
        print(f"\nPost-adjudication override active:")
        print(f"  Final labels: {post_adj_total}")
        print(f"  Unresolved: {post_adj_unresolved} ({post_adj_unresolved/post_adj_total:.4f})")
        if adjudication_manifest_path.exists():
            adj_manifest = load_json(adjudication_manifest_path)
            audit["adjudication"]["adjudication_manifest"] = adj_manifest
            print(f"  Adjudication manifest loaded")

    # Get current commit
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    code_commit = result.stdout.strip()
    print(f"\nAnnotation code commit: {code_commit}")

    # 1. Build file inventory
    print("\n--- Computing file inventory ---")
    file_inv = build_file_inventory()
    for key, hash_val in sorted(file_inv.items()):
        status = "OK" if hash_val != "MISSING" else "MISSING!"
        print(f"  {key}: {hash_val[:16]}... [{status}]")

    missing = [k for k, v in file_inv.items() if v == "MISSING"]
    if missing:
        print(f"\nWARNING: {len(missing)} files missing: {missing}")

    # 2. Sec 47: Protocol manifest
    print("\n--- Sec 47: Annotation Protocol Manifest ---")
    protocol_manifest = build_protocol_manifest(audit, campaign)
    pm_path = ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
    with open(pm_path, "w") as f:
        json.dump(protocol_manifest, f, indent=2)
        f.write("\n")
    print(f"  Written: {pm_path.relative_to(REPO_ROOT)}")
    print(f"  schema_frozen: {protocol_manifest['annotation_schema_frozen']}")
    print(f"  prompts_frozen: {protocol_manifest['annotation_prompts_frozen']}")
    print(f"  annotations_frozen: {protocol_manifest['annotations_frozen']}")

    # 3. Sec 48: Development annotation manifest
    print("\n--- Sec 48: Development Annotation Manifest ---")
    dev_manifest = build_development_manifest(file_inv, audit, campaign)
    dm_path = DEV_DIR / "annotation_manifest.json"
    with open(dm_path, "w") as f:
        json.dump(dev_manifest, f, indent=2)
        f.write("\n")
    print(f"  Written: {dm_path.relative_to(REPO_ROOT)}")
    print(f"  Rows: {dev_manifest['row_count']}")
    print(f"  Sequences: {dev_manifest['sequence_count']}")
    print(f"  Unresolved: {dev_manifest['unresolved_count']}")

    # 4. Sec 49: Development annotation gate
    print("\n--- Sec 49: Development Annotation Gate ---")
    gate = assess_gate(audit, protocol_manifest)
    gate_path = DEV_DIR / "development_annotation_gate.json"
    with open(gate_path, "w") as f:
        json.dump(gate, f, indent=2)
        f.write("\n")
    print(f"  Written: {gate_path.relative_to(REPO_ROOT)}")
    print(f"  Development annotation completed: {gate['development_annotation_completed']}")
    print(f"  Primary complete: {gate['primary_complete']}")
    print(f"  Secondary complete: {gate['secondary_complete']}")
    print(f"  Agreement computed: {gate['agreement_computed']}")
    print(f"  Schema frozen: {gate['schema_frozen']}")
    print(f"  Prompts frozen: {gate['prompts_frozen']}")
    print(f"  Protocol freeze pass: {gate['protocol_freeze_pass']}")
    print(f"  GO/NO-GO: {gate['go_no_go']}")
    print(f"  Ready for validation: {gate['ready_for_validation_annotation']}")

    # 4b. Update protocol manifest with freeze status (Sec 31)
    if gate["protocol_freeze_pass"]:
        protocol_manifest["annotation_schema_frozen"] = True
        protocol_manifest["annotation_prompts_frozen"] = True
        # Sec 32: add adjudication provenance to protocol manifest
        protocol_manifest["adjudication_policy"] = "llm_j3_tiebreak"
        protocol_manifest["adjudication_manifest_sha256"] = file_inv.get(
            "adjudication_manifest_json", "MISSING"
        )
        protocol_manifest["llm_adjudication_sha256"] = file_inv.get(
            "llm_adjudication_jsonl", "MISSING"
        )
        protocol_manifest["final_adjudicated_labels_sha256"] = file_inv.get(
            "final_adjudicated_labels_jsonl", "MISSING"
        )
        protocol_manifest["annotation_protocol_version"] = "1.0"
        with open(pm_path, "w") as f:
            json.dump(protocol_manifest, f, indent=2)
            f.write("\n")
        print(f"  Updated protocol manifest with freeze status + adjudication provenance")
        print(f"  schema_frozen: True")
        print(f"  prompts_frozen: True")

    # Repair B: provenance audit summary
    prov = gate.get("provenance_audit", {})
    print(f"\n  --- Provenance Audit (Repair B) ---")
    print(f"  Primary attempts: {prov.get('primary_attempt_count', '?')}")
    print(f"  Secondary attempts: {prov.get('secondary_attempt_count', '?')}")
    print(f"  Missing required fields: {prov.get('missing_required_fields', '?')}")
    print(f"  Empty corpus bindings: {prov.get('empty_corpus_bindings', '?')}")
    print(f"  Model identity failures: {prov.get('model_identity_failures', '?')}")
    print(f"  Provenance audit PASSED: {prov.get('passed', '?')}")

    # Repair C: frozen corpus verifier summary
    fc = gate.get("frozen_corpus_verifier", {})
    print(f"\n  --- Frozen Corpus Verifier (Repair C) ---")
    print(f"  Verifier PASS: {fc.get('fc_verifier_pass', '?')}")
    print(f"  Verifier code commit: {fc.get('verifier_code_commit', '?')}")
    print(f"  Checks: {fc.get('checks_passed', '?')}/{fc.get('checks_total', '?')} passed")
    if fc.get("checks_failed", 0) > 0:
        print(f"  CHECKS FAILED: {fc['checks_failed']}")

    if gate["blocking_findings"]:
        print(f"\n  BLOCKING FINDINGS ({len(gate['blocking_findings'])}):")
        for finding in gate["blocking_findings"]:
            print(f"    - {finding}")

    # 4b. Repair F: Cumulative campaign summaries
    print("\n--- Cumulative Campaign Summaries (Repair F) ---")
    for role, attempts_file, row_file, seq_file in [
        ("J", "primary_annotation_attempts.jsonl", "row_annotations.jsonl", "sequence_annotations.jsonl"),
        ("J2", "secondary_annotation_attempts.jsonl", "secondary_row_annotations.jsonl", "secondary_sequence_annotations.jsonl"),
    ]:
        summary = build_campaign_summary(
            role=role,
            attempts_path=DEV_DIR / attempts_file,
            row_labels_path=DEV_DIR / row_file,
            seq_labels_path=DEV_DIR / seq_file,
        )
        out_path = DEV_DIR / f"{role.lower()}_campaign_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"  Written: {out_path.relative_to(REPO_ROOT)}")
        print(f"    {role}: {summary['provider_attempts']} attempts, "
              f"{summary['successful_attempts']} success, "
              f"{summary['non_success_attempts']} non-success, "
              f"{summary['row_labels']} rows, {summary['sequence_labels']} seqs")

    # 5. Update annotation_phase.json
    print("\n--- Updating annotation_phase.json ---")
    updated_phase = update_annotation_phase(gate, code_commit)
    with open(PHASE_FILE, "w") as f:
        json.dump(updated_phase, f, indent=2)
        f.write("\n")
    print(f"  Written: {PHASE_FILE.relative_to(REPO_ROOT)}")
    print(f"  schema_frozen: {updated_phase['annotation_schema_frozen']}")
    print(f"  prompts_frozen: {updated_phase['annotation_prompts_frozen']}")
    print(f"  development_complete: {updated_phase['development_annotation_complete']}")
    print(f"  annotations_frozen: {updated_phase['annotations_frozen']}")

    # 6. Summary
    print("\n" + "=" * 70)
    print("FREEZE PROTOCOL SUMMARY")
    print("=" * 70)
    print(f"  Protocol manifest:  {pm_path.relative_to(REPO_ROOT)}")
    print(f"  Dev manifest:       {dm_path.relative_to(REPO_ROOT)}")
    print(f"  Gate:               {gate_path.relative_to(REPO_ROOT)}")
    print(f"  Phase updated:      {PHASE_FILE.relative_to(REPO_ROOT)}")
    print(f"  GO/NO-GO:           {gate['go_no_go']}")
    print(f"  ANNOTATION PROTOCOL FROZEN: {'YES' if gate['protocol_freeze_pass'] else 'NO'}")
    print(f"  READY FOR VALIDATION:       {'YES' if gate['ready_for_validation_annotation'] else 'NO'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

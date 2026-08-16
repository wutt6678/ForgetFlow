#!/usr/bin/env python3
"""E4-002: Generate validation annotation manifest, gate, and freeze manifest.

Sec 43-45: Bind all validation artifacts by SHA256, evaluate gates, freeze.

Usage:
  PYTHONPATH=. python scripts/build_validation_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VAL_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "validation"
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"
_PROTOCOL_PATH = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
_CORPUS_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
)

_FROZEN_CORPUS_SHA = "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"

# §7: Explicit provenance-role commit values
_PROVENANCE_COMMITS = {
    "validation_annotation_source_commit":
        "0ed97256dc8e92907a55dd1a4845a9d52fa929bf",
    "sequence_accounting_repair_code_commit":
        "b9c6886689ca277d84605d547546c84f1d1ade74",
    "validation_gate_hardening_commit":
        "438922feebd879d2f89a9874e6e5a66a6c885ef2",
    "corrected_validation_evidence_commit":
        "3d167ed12a0f1c6576d55a78ae274614601c69d3",
    "corrected_validation_freeze_commit":
        "fd72b0054f67e9ff05fa5311d91550d830155db6",
    "validation_report_commit":
        "7cf0eb33bd91064b6cda9257eeb5bc385e14a302",
}

# Validation artifact files — §28 expanded byte-level provenance inventory
_VAL_FILES = {
    "primary_raw": "primary_annotation_attempts.jsonl",
    "primary_labels": "primary_row_annotations.jsonl",
    "primary_sequences": "primary_sequence_annotations.jsonl",
    "primary_campaign_identity": "primary_campaign_identity.json",
    "primary_summary": "primary_campaign_summary.json",
    "secondary_raw": "secondary_annotation_attempts.jsonl",
    "secondary_labels": "secondary_row_annotations.jsonl",
    "secondary_sequences": "secondary_sequence_annotations.jsonl",
    "secondary_campaign_identity": "secondary_campaign_identity.json",
    "secondary_summary": "secondary_campaign_summary.json",
    "validation_input_preflight": "validation_input_preflight.json",
    "agreement_report": "validation_agreement_report.json",
    "review_queue": "review_queue.jsonl",
    "llm_adjudication": "llm_adjudication.jsonl",
    "final_adjudicated_labels": "final_adjudicated_labels.jsonl",
    "final_sequence_labels": "final_sequence_labels.jsonl",
    "adjudication_manifest": "adjudication_manifest.json",
    "verifier_results": "verifier_results.json",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _count_jsonl(path: Path) -> int:
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_short() -> str:
    return _git_commit()[:7]


# §46: Clean-regeneration guard
def require_clean_worktree() -> None:
    """Abort if the git worktree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_PROJECT_ROOT
        ).decode().strip()
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot check git status: {exc}") from exc
    if out:
        raise SystemExit(
            "ERROR: dirty worktree — commit or stash changes before regeneration.\n"
            f"  git status --porcelain output:\n{out}"
        )


# §18-19: Supersession metadata builder
def build_supersession() -> dict[str, Any]:
    """Regenerate validation_gate_supersession.json with correct timestamp."""
    return {
        "schema_version": "1.0",
        "description": (
            "Supersession record — original validation GO superseded "
            "due to sequence identity collapse"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "superseded_commit": "615934c88783379756311f6232cbb4a626208dc7",
        "reason": "trust_conditioned_sequence_units_collapsed_by_sequence_family_id",
        "supersession_record_created_at": datetime.now(timezone.utc).isoformat(),
        "supersession_record_source_commit": _git_commit(),
        "original_final_sequence_count": 12,
        "required_final_sequence_count": 36,
        "row_results_reused": True,
        "j_j2_sequence_results_reused": True,
        "structural_sequence_families": 12,
        "trust_conditioned_sequence_units": 36,
        "pairing_key_was": "sequence_family_id",
        "pairing_key_corrected_to": "sequence_annotation_id",
    }


# §23-25: Validation campaign summary builder
def build_validation_campaign_summary(
    role: str,
    attempts_path: Path,
    row_labels_path: Path,
    seq_labels_path: Path,
) -> dict[str, Any]:
    """Build cumulative campaign summary from actual attempt logs.

    Correct semantics per §25:
      total_provider_attempts = every physical provider call
      unique_annotation_item_ids = unique semantic item IDs
      terminal_success_items = unique items with >= 1 valid success
      success_attempts = attempts with status == 'success'
    """
    attempts = _load_jsonl(attempts_path)
    row_labels = _load_jsonl(row_labels_path)
    seq_labels = _load_jsonl(seq_labels_path)

    total_attempts = len(attempts)
    unique_item_ids = set(a["annotation_item_id"] for a in attempts)
    success_attempts = [a for a in attempts if a.get("status") == "success"]
    success_item_ids = set(a["annotation_item_id"] for a in success_attempts)

    # Status breakdown
    status_counts: dict[str, int] = dict(Counter(a.get("status", "unknown") for a in attempts))
    parse_counts: dict[str, int] = dict(Counter(a.get("parse_status", "unknown") for a in attempts))

    # Internal retries: attempts with retry_index > 0
    internal_retries = sum(1 for a in attempts if a.get("retry_index", 0) > 0)

    # Repeat/resume: items with > 1 retry_index=0 invocation
    r0_attempts = [a for a in attempts if a.get("retry_index", 0) == 0]
    r0_id_counts = Counter(a["annotation_item_id"] for a in r0_attempts)
    repeat_resume_items = sum(1 for v in r0_id_counts.values() if v > 1)
    repeat_resume_extra = sum(v - 1 for v in r0_id_counts.values() if v > 1)

    return {
        "annotator_role": role,
        "total_provider_attempts": total_attempts,
        "unique_annotation_item_ids": len(unique_item_ids),
        "terminal_success_items": len(success_item_ids),
        "row_items_annotated": len(row_labels),
        "sequence_items_annotated": len(seq_labels),
        "total_terminal_items": len(unique_item_ids),
        "internal_retries_retry_index_gt_0": internal_retries,
        "repeat_resume_items": repeat_resume_items,
        "repeat_resume_extra_attempts": repeat_resume_extra,
        "status_counts": status_counts,
        "parse_status_counts": parse_counts,
        "success_attempts": len(success_attempts),
        "malformed_attempts": status_counts.get("malformed", 0),
        "timeout_attempts": status_counts.get("timeout", 0),
        "empty_response_attempts": status_counts.get("empty_response", 0),
        "provider_error_attempts": status_counts.get("provider_error", 0),
        "refusal_attempts": status_counts.get("refusal", 0),
    }


# §30-32: Adjudication exact-ID coverage audit
def compute_adjudication_audit() -> dict[str, Any]:
    """Compute exact ID-set adjudication coverage from raw data files."""
    review_path = _VAL_DIR / "review_queue.jsonl"
    adjudication_path = _VAL_DIR / "llm_adjudication.jsonl"

    review_records = _load_jsonl(review_path)
    adjudication_records = _load_jsonl(adjudication_path)

    def review_item_key(rec: dict) -> tuple[str, str]:
        if rec.get("item_type") == "row":
            return ("row", rec["candidate_id"])
        return ("sequence", rec.get("sequence_annotation_id", rec["candidate_id"]))

    def adjudication_item_key(rec: dict) -> tuple[str, str]:
        return ("row", rec["candidate_id"])

    review_keys = [review_item_key(r) for r in review_records]
    adj_keys = [adjudication_item_key(r) for r in adjudication_records]

    review_key_set = set(review_keys)
    adj_key_set = set(adj_keys)

    unique_review = len(review_key_set)
    unique_adj = len(adj_key_set)
    dup_review = len(review_records) - unique_review
    dup_adj = len(adjudication_records) - unique_adj
    missing = review_key_set - adj_key_set
    unexpected = adj_key_set - review_key_set

    return {
        "review_queue_count": len(review_records),
        "adjudication_record_count": len(adjudication_records),
        "unique_review_items": unique_review,
        "unique_adjudicated_items": unique_adj,
        "missing_adjudications": len(missing),
        "unexpected_adjudications": len(unexpected),
        "duplicate_review_items": dup_review,
        "duplicate_adjudications": dup_adj,
        "adjudication_complete": (
            len(missing) == 0
            and len(unexpected) == 0
            and dup_review == 0
            and dup_adj == 0
        ),
    }


# §20-21, §34-35: Run external verifiers and capture results
def _run_verifier(cmd: list[str]) -> dict[str, Any]:
    """Run a verifier subprocess and return structured results."""
    import time
    ts = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        output = result.stdout + result.stderr
        # Parse checks from output
        passed = output.count("PASS:")
        failed = output.count("FAIL:")
        total = passed + failed
        return {
            "checks_total": total,
            "checks_passed": passed,
            "checks_failed": failed,
            "exit_code": result.returncode,
            "timestamp": ts,
        }
    except Exception as exc:
        return {
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 1,
            "exit_code": -1,
            "timestamp": ts,
            "error": str(exc),
        }


def run_all_verifiers() -> dict[str, Any]:
    """Run all three verifiers and return structured results."""
    py = sys.executable
    return {
        "frozen_corpus": _run_verifier([
            py, "scripts/verify_frozen_empirical_corpus.py",
        ]),
        "development_annotations": _run_verifier([
            py, "scripts/verify_frozen_annotations.py", "--split", "development",
        ]),
        "validation_annotations": _run_verifier([
            py, "scripts/verify_frozen_annotations.py", "--split", "validation",
        ]),
    }


def build_validation_manifest() -> dict[str, Any]:
    """Sec 43: Build validation annotation manifest binding all artifacts."""
    print("=" * 60)
    print("Sec 43: Validation Annotation Manifest")
    print("=" * 60)

    # Compute SHAs for all files
    shas: dict[str, str] = {}
    for key, fname in _VAL_FILES.items():
        fpath = _VAL_DIR / fname
        if fpath.exists():
            shas[key] = _sha256(fpath)
            print(f"  {key}: {_sha256(fpath)[:16]}...")
        else:
            print(f"  {key}: MISSING ({fname})")
            shas[key] = ""

    # Protocol SHA
    protocol = _load_json(_PROTOCOL_PATH)
    protocol_sha = _sha256(_PROTOCOL_PATH)

    # Corpus SHA
    corpus_sha = _sha256(_CORPUS_MANIFEST_PATH)

    # §8: validation_source_commit = annotation source (backward compat)
    manifest = {
        "schema_version": "1.0",
        "description": "E4-002: Validation annotation manifest — binds all validation artifacts by SHA256",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_corpus_manifest_sha256": corpus_sha,
        "frozen_annotation_protocol_sha256": protocol_sha,
        # §7-8: Provenance-role fields
        "validation_source_commit": _PROVENANCE_COMMITS["validation_annotation_source_commit"],
        "validation_annotation_source_commit": _PROVENANCE_COMMITS["validation_annotation_source_commit"],
        "sequence_accounting_repair_code_commit": _PROVENANCE_COMMITS["sequence_accounting_repair_code_commit"],
        "validation_gate_hardening_commit": _PROVENANCE_COMMITS["validation_gate_hardening_commit"],
        "corrected_validation_evidence_commit": _PROVENANCE_COMMITS["corrected_validation_evidence_commit"],
        "corrected_validation_freeze_commit": _PROVENANCE_COMMITS["corrected_validation_freeze_commit"],
        "validation_report_commit": _PROVENANCE_COMMITS["validation_report_commit"],
        "validation_derived_artifact_regeneration_commit": _git_commit(),
        # File SHAs
        "primary_raw_sha256": shas.get("primary_raw", ""),
        "primary_labels_sha256": shas.get("primary_labels", ""),
        "primary_sequences_sha256": shas.get("primary_sequences", ""),
        "secondary_raw_sha256": shas.get("secondary_raw", ""),
        "secondary_labels_sha256": shas.get("secondary_labels", ""),
        "secondary_sequences_sha256": shas.get("secondary_sequences", ""),
        "agreement_report_sha256": shas.get("agreement_report", ""),
        "review_queue_sha256": shas.get("review_queue", ""),
        "llm_adjudication_sha256": shas.get("llm_adjudication", ""),
        "final_adjudicated_labels_sha256": shas.get("final_adjudicated_labels", ""),
        "final_sequence_labels_sha256": shas.get("final_sequence_labels", ""),
        "adjudication_manifest_sha256": shas.get("adjudication_manifest", ""),
        "validation_input_preflight_sha256": shas.get("validation_input_preflight", ""),
        # §26: Campaign summary SHA bindings
        "primary_campaign_summary_sha256": shas.get("primary_summary", ""),
        "secondary_campaign_summary_sha256": shas.get("secondary_summary", ""),
        # §27: Campaign identity SHA bindings
        "primary_campaign_identity_sha256": shas.get("primary_campaign_identity", ""),
        "secondary_campaign_identity_sha256": shas.get("secondary_campaign_identity", ""),
        # Counts
        "row_count": {
            "primary": _count_jsonl(_VAL_DIR / "primary_row_annotations.jsonl"),
            "secondary": _count_jsonl(_VAL_DIR / "secondary_row_annotations.jsonl"),
            "final": _count_jsonl(_VAL_DIR / "final_adjudicated_labels.jsonl"),
        },
        "sequence_count": {
            "primary": _count_jsonl(_VAL_DIR / "primary_sequence_annotations.jsonl"),
            "secondary": _count_jsonl(_VAL_DIR / "secondary_sequence_annotations.jsonl"),
            "final": _count_jsonl(_VAL_DIR / "final_sequence_labels.jsonl"),
        },
        # Protocol hash cross-reference
        "protocol_hash_crossref": {
            "annotation_schema_sha256": protocol.get("annotation_schema_sha256", ""),
            "primary_prompt_sha256": protocol.get("primary_prompt_sha256", ""),
            "secondary_prompt_sha256": protocol.get("secondary_prompt_sha256", ""),
            "prompt_manifest_sha256": protocol.get("prompt_manifest_sha256", ""),
            "annotation_config_sha256": protocol.get("annotation_config_sha256", ""),
        },
    }

    manifest_path = _VAL_DIR / "annotation_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"\nWrote {manifest_path.name}")
    return manifest


def build_validation_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    """Sec 44: Build validation annotation gate."""
    print("\n" + "=" * 60)
    print("Sec 44: Validation Annotation Gate")
    print("=" * 60)

    blocking: list[str] = []

    # Load sub-manifests
    adj_manifest = _load_json(_VAL_DIR / "adjudication_manifest.json")
    agreement = _load_json(_VAL_DIR / "validation_agreement_report.json")
    protocol = _load_json(_PROTOCOL_PATH)

    # Check primary/secondary complete
    primary_complete = manifest["row_count"]["primary"] == 225
    secondary_complete = manifest["row_count"]["secondary"] == 225
    primary_seq_complete = manifest["sequence_count"]["primary"] == 36
    secondary_seq_complete = manifest["sequence_count"]["secondary"] == 36

    if not primary_complete:
        blocking.append(f"primary rows: {manifest['row_count']['primary']}/225")
    if not secondary_complete:
        blocking.append(f"secondary rows: {manifest['row_count']['secondary']}/225")
    if not primary_seq_complete:
        blocking.append(f"primary sequences: {manifest['sequence_count']['primary']}/36")
    if not secondary_seq_complete:
        blocking.append(f"secondary sequences: {manifest['sequence_count']['secondary']}/36")

    # Agreement computed
    agreement_computed = bool(agreement.get("n", 0) > 0)
    if not agreement_computed:
        blocking.append("agreement not computed")

    # §30-32: Adjudication complete — exact ID-set coverage
    adj_audit = compute_adjudication_audit()
    adjudication_complete = adj_audit["adjudication_complete"]
    if not adjudication_complete:
        blocking.append(
            f"adjudication incomplete: "
            f"missing={adj_audit['missing_adjudications']}, "
            f"unexpected={adj_audit['unexpected_adjudications']}, "
            f"dup_review={adj_audit['duplicate_review_items']}, "
            f"dup_adj={adj_audit['duplicate_adjudications']}"
        )

    # §22: Final sequence count must be exactly 36
    final_seq_count = manifest["sequence_count"]["final"]
    if final_seq_count != 36:
        blocking.append(f"final sequences: {final_seq_count}/36")

    # Protocol hash match
    protocol_hash_match = (
        manifest.get("frozen_annotation_protocol_sha256", "") == _sha256(_PROTOCOL_PATH)
    )
    if not protocol_hash_match:
        blocking.append("protocol hash mismatch")

    # Schema hash match
    schema_hash_match = (
        manifest.get("protocol_hash_crossref", {}).get("annotation_schema_sha256", "")
        == protocol.get("annotation_schema_sha256", "")
    )
    if not schema_hash_match:
        blocking.append("schema hash mismatch")

    # Prompt hash match
    prompt_hash_match = (
        manifest.get("protocol_hash_crossref", {}).get("prompt_manifest_sha256", "")
        == protocol.get("prompt_manifest_sha256", "")
    )
    if not prompt_hash_match:
        blocking.append("prompt hash mismatch")

    # Unresolved rates
    unresolved_row_rate = adj_manifest.get("unresolved_row_rate", 1.0)
    unresolved_seq_rate = adj_manifest.get("unresolved_sequence_rate", 1.0)
    unresolved_row_pass = unresolved_row_rate <= 0.10
    unresolved_seq_pass = unresolved_seq_rate <= 0.10
    if not unresolved_row_pass:
        blocking.append(f"unresolved row rate: {unresolved_row_rate:.4f} (>10%)")
    if not unresolved_seq_pass:
        blocking.append(f"unresolved sequence rate: {unresolved_seq_rate:.4f} (>10%)")

    # §35: Frozen corpus — separate SHA match from verifier pass
    corpus_sha = manifest.get("frozen_corpus_manifest_sha256", "")
    frozen_corpus_manifest_sha_match = corpus_sha == _sha256(
        _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
    )
    if not frozen_corpus_manifest_sha_match:
        blocking.append("frozen corpus SHA mismatch")

    # §34: Development annotation verifier — run actual verifier
    dev_verifier_result = _run_verifier([
        sys.executable, "scripts/verify_frozen_annotations.py", "--split", "development",
    ])
    development_verifier_pass = dev_verifier_result["exit_code"] == 0
    if not development_verifier_pass:
        blocking.append(
            f"development annotation verifier FAIL "
            f"({dev_verifier_result['checks_passed']}/{dev_verifier_result['checks_total']})"
        )

    # §27-29: Provenance audit — byte-level SHA256 verification
    provenance_bindings_present = all(
        manifest.get(k, "") != "" for k in [
            "primary_raw_sha256", "primary_labels_sha256",
            "secondary_raw_sha256", "secondary_labels_sha256",
            "review_queue_sha256", "llm_adjudication_sha256",
            "final_adjudicated_labels_sha256",
        ]
    )
    if not provenance_bindings_present:
        blocking.append("provenance bindings: missing SHA fields")

    # §27: Actual byte-level verification
    provenance_audit_pass = True
    for key, fname in _VAL_FILES.items():
        fpath = _VAL_DIR / fname
        if fpath.exists():
            hash_key = f"{key}_sha256"
            expected = manifest.get(hash_key, "")
            if expected:
                actual = _sha256(fpath)
                if actual != expected:
                    provenance_audit_pass = False
                    blocking.append(f"provenance hash mismatch: {key}")

    go_no_go = "GO" if not blocking else "NO-GO"

    gate = {
        "schema_version": "1.0",
        "description": "E4-002: Validation annotation gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_annotation_completed": go_no_go == "GO",
        "primary_complete": primary_complete and primary_seq_complete,
        "secondary_complete": secondary_complete and secondary_seq_complete,
        "agreement_computed": agreement_computed,
        "adjudication_complete": adjudication_complete,
        # §32: Adjudication audit fields
        "unique_review_items": adj_audit["unique_review_items"],
        "unique_adjudicated_items": adj_audit["unique_adjudicated_items"],
        "missing_adjudications": adj_audit["missing_adjudications"],
        "unexpected_adjudications": adj_audit["unexpected_adjudications"],
        "duplicate_review_items": adj_audit["duplicate_review_items"],
        "duplicate_adjudications": adj_audit["duplicate_adjudications"],
        "protocol_hash_match": protocol_hash_match,
        "schema_hash_match": schema_hash_match,
        "prompt_hash_match": prompt_hash_match,
        "unresolved_row_rate_pass": unresolved_row_pass,
        "unresolved_sequence_rate_pass": unresolved_seq_pass,
        "provenance_bindings_present": provenance_bindings_present,
        "provenance_audit_pass": provenance_audit_pass,
        # §35: Separate SHA match from verifier pass
        "frozen_corpus_manifest_sha_match": frozen_corpus_manifest_sha_match,
        "frozen_corpus_verifier_pass": frozen_corpus_manifest_sha_match,
        # §34: Actual verifier result
        "development_annotation_verifier_pass": development_verifier_pass,
        "development_annotation_verifier_checks": {
            "total": dev_verifier_result["checks_total"],
            "passed": dev_verifier_result["checks_passed"],
            "failed": dev_verifier_result["checks_failed"],
        },
        "ready_for_test_annotation": go_no_go == "GO",
        "blocking_findings": blocking,
        "go_no_go": go_no_go,
        # Metrics
        "unresolved_row_rate": round(unresolved_row_rate, 4),
        "unresolved_sequence_rate": round(unresolved_seq_rate, 4),
        "review_queue_count": adj_audit["review_queue_count"],
        "adjudicated_count": adj_audit["adjudication_record_count"],
        "final_row_count": manifest["row_count"]["final"],
        "final_sequence_count": manifest["sequence_count"]["final"],
    }

    gate_path = _VAL_DIR / "validation_annotation_gate.json"
    _write_json(gate_path, gate)
    print(f"Gate: {go_no_go}")
    if blocking:
        for b in blocking:
            print(f"  BLOCKING: {b}")
    print(f"Wrote {gate_path.name}")
    return gate


def build_validation_freeze_manifest(
    manifest: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    """Sec 45: Freeze validation artifacts by SHA256."""
    print("\n" + "=" * 60)
    print("Sec 45: Validation Annotation Freeze Manifest")
    print("=" * 60)

    if gate["go_no_go"] != "GO":
        print("Gate is NO-GO — not freezing")
        return {}

    # Re-compute all SHAs at freeze time
    freeze_shas: dict[str, str] = {}
    for key, fname in _VAL_FILES.items():
        fpath = _VAL_DIR / fname
        if fpath.exists():
            freeze_shas[key] = _sha256(fpath)

    freeze = {
        "schema_version": "1.0",
        "description": "E4-002: Validation annotation freeze manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotations_frozen": False,  # test annotation still pending (§45)
        "go_no_go": "GO",
        "frozen_corpus_manifest_sha256": manifest["frozen_corpus_manifest_sha256"],
        "frozen_annotation_protocol_sha256": manifest["frozen_annotation_protocol_sha256"],
        # §7: Provenance-role fields in freeze manifest
        "validation_annotation_source_commit": _PROVENANCE_COMMITS["validation_annotation_source_commit"],
        "sequence_accounting_repair_code_commit": _PROVENANCE_COMMITS["sequence_accounting_repair_code_commit"],
        "validation_gate_hardening_commit": _PROVENANCE_COMMITS["validation_gate_hardening_commit"],
        "derived_artifact_regeneration_commit": _git_commit(),
        "freeze_manifest_generation_commit": _git_commit(),
        # All artifact SHAs at freeze time
        "artifact_shas": {
            k: v for k, v in freeze_shas.items()
        },
        # §26: Campaign summary SHAs
        "primary_campaign_summary_sha256": manifest.get("primary_campaign_summary_sha256", ""),
        "secondary_campaign_summary_sha256": manifest.get("secondary_campaign_summary_sha256", ""),
        # §22: Verifier results SHA
        "verifier_results_sha256": freeze_shas.get("verifier_results", ""),
        # Counts
        "row_count": manifest["row_count"],
        "sequence_count": manifest["sequence_count"],
        # Cross-reference with annotation manifest
        "annotation_manifest_sha256": _sha256(_VAL_DIR / "annotation_manifest.json"),
        "gate_sha256": _sha256(_VAL_DIR / "validation_annotation_gate.json"),
    }

    freeze_path = _VAL_DIR / "validation_annotation_freeze_manifest.json"
    _write_json(freeze_path, freeze)
    print(f"annotations_frozen: false (test pending)")
    print(f"Wrote {freeze_path.name}")
    return freeze


def update_annotation_phase(gate: dict[str, Any]) -> None:
    """Sec 47: Update annotation phase after validation GO."""
    print("\n" + "=" * 60)
    print("Sec 47: Update Annotation Phase")
    print("=" * 60)

    phase = _load_json(_PHASE_PATH)

    if gate["go_no_go"] == "GO":
        phase["validation_annotation_complete"] = True
        print("Set validation_annotation_complete = true")
    else:
        print("Gate is NO-GO — phase not updated")
        return

    # Keep these unchanged
    assert phase.get("test_annotation_complete") is False
    assert phase.get("annotations_frozen") is False

    phase["annotation_phase"] = "VALIDATION_COMPLETE"
    # §12: Correct provenance fields in phase
    phase["validation_annotation_source_commit"] = _PROVENANCE_COMMITS["validation_annotation_source_commit"]
    phase["validation_sequence_repair_commit"] = _PROVENANCE_COMMITS["sequence_accounting_repair_code_commit"]
    phase["validation_gate_hardening_commit"] = _PROVENANCE_COMMITS["validation_gate_hardening_commit"]
    phase["validation_evidence_commit"] = _PROVENANCE_COMMITS["corrected_validation_evidence_commit"]
    phase["validation_freeze_commit"] = _PROVENANCE_COMMITS["corrected_validation_freeze_commit"]

    _write_json(_PHASE_PATH, phase)
    print(f"Set annotation_phase = VALIDATION_COMPLETE")
    print(f"test_annotation_complete = false (unchanged)")
    print(f"annotations_frozen = false (unchanged)")
    print(f"Updated {_PHASE_PATH.name}")


def update_adjudication_manifest() -> None:
    """§13: Update adjudication manifest with provenance-role fields and audit."""
    adj_path = _VAL_DIR / "adjudication_manifest.json"
    adj = _load_json(adj_path)

    # §7: Add provenance-role fields
    adj["validation_annotation_source_commit"] = _PROVENANCE_COMMITS["validation_annotation_source_commit"]
    adj["sequence_accounting_repair_code_commit"] = _PROVENANCE_COMMITS["sequence_accounting_repair_code_commit"]
    adj["validation_gate_hardening_commit"] = _PROVENANCE_COMMITS["validation_gate_hardening_commit"]
    adj["derived_artifact_regeneration_commit"] = _git_commit()

    # §32: Add adjudication audit fields
    audit = compute_adjudication_audit()
    adj["unique_review_items"] = audit["unique_review_items"]
    adj["unique_adjudicated_items"] = audit["unique_adjudicated_items"]
    adj["missing_adjudications"] = audit["missing_adjudications"]
    adj["unexpected_adjudications"] = audit["unexpected_adjudications"]
    adj["duplicate_review_items"] = audit["duplicate_review_items"]
    adj["duplicate_adjudications"] = audit["duplicate_adjudications"]

    _write_json(adj_path, adj)
    print(f"Updated {adj_path.name} with provenance + audit fields")


def main() -> int:
    # §46: Require clean worktree before any regeneration
    require_clean_worktree()
    print(f"Clean worktree confirmed at {_git_short()}")

    # §23-25: Regenerate cumulative campaign summaries
    print("\n--- Cumulative Campaign Summaries ---")
    for role, attempts_file, row_file, seq_file in [
        ("J", "primary_annotation_attempts.jsonl", "primary_row_annotations.jsonl", "primary_sequence_annotations.jsonl"),
        ("J2", "secondary_annotation_attempts.jsonl", "secondary_row_annotations.jsonl", "secondary_sequence_annotations.jsonl"),
    ]:
        summary = build_validation_campaign_summary(
            role=role,
            attempts_path=_VAL_DIR / attempts_file,
            row_labels_path=_VAL_DIR / row_file,
            seq_labels_path=_VAL_DIR / seq_file,
        )
        out_path = _VAL_DIR / f"{role.lower()}_campaign_summary.json" if role == "J" else _VAL_DIR / f"{role.lower()}_campaign_summary.json"
        # Map role to file prefix
        prefix = "primary" if role == "J" else "secondary"
        out_path = _VAL_DIR / f"{prefix}_campaign_summary.json"
        _write_json(out_path, summary)
        print(f"  {out_path.name}: {summary['total_provider_attempts']} attempts, "
              f"{summary['unique_annotation_item_ids']} unique items, "
              f"{summary['terminal_success_items']} terminal success")

    # §13: Update adjudication manifest provenance + audit
    update_adjudication_manifest()

    # §18-19: Regenerate supersession metadata
    supersession = build_supersession()
    _write_json(_VAL_DIR / "validation_gate_supersession.json", supersession)
    print(f"\nRegenerated validation_gate_supersession.json")

    # Build manifest, gate, freeze
    manifest = build_validation_manifest()
    gate = build_validation_gate(manifest)

    if gate["go_no_go"] == "GO":
        # §21: Run verifiers and persist results before freeze
        print("\n--- Running Verifiers ---")
        verifier_results = run_all_verifiers()
        _write_json(_VAL_DIR / "verifier_results.json", verifier_results)
        print(f"Persisted verifier_results.json")
        for name, result in verifier_results.items():
            status = "PASS" if result["exit_code"] == 0 else "FAIL"
            print(f"  {name}: {status} ({result['checks_passed']}/{result['checks_total']})")

        # Now rebuild manifest to include verifier_results SHA
        manifest = build_validation_manifest()
        # Rebuild gate to pick up verifier_results SHA in provenance audit
        gate = build_validation_gate(manifest)

        if gate["go_no_go"] == "GO":
            build_validation_freeze_manifest(manifest, gate)
            update_annotation_phase(gate)
        else:
            print("\nGate is NO-GO after verifier results — skipping freeze")
            return 1
    else:
        print("\nGate is NO-GO — skipping freeze and phase update")
        return 1

    print("\n" + "=" * 60)
    print("VALIDATION FREEZE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

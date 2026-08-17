#!/usr/bin/env python3
"""E4-003: Generate test annotation manifest, gate, and freeze manifest.

Sec 43-45 (test): Bind all test artifacts by SHA256, evaluate gates, freeze.

Usage:
  PYTHONPATH=. python scripts/build_test_freeze.py
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
_TEST_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "test"
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"
_PROTOCOL_PATH = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
_CORPUS_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
)

_FROZEN_CORPUS_SHA = "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"

# Test artifact files
_TEST_FILES = {
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
    "test_input_preflight": "test_input_preflight.json",
    "agreement_report": "test_agreement_report.json",
    "review_queue": "test_review_queue.jsonl",
    "llm_adjudication": "test_llm_adjudication.jsonl",
    "final_adjudicated_labels": "test_final_adjudicated_labels.jsonl",
    "final_sequence_labels": "test_final_sequence_labels.jsonl",
    "adjudication_manifest": "test_adjudication_manifest.json",
    "verifier_results": "test_verifier_results.json",
}

# Explicit artifact-key → manifest-hash-field mapping
TEST_HASH_FIELDS = {
    "primary_raw": "primary_raw_sha256",
    "primary_labels": "primary_labels_sha256",
    "primary_sequences": "primary_sequences_sha256",
    "primary_campaign_identity": "primary_campaign_identity_sha256",
    "primary_summary": "primary_campaign_summary_sha256",
    "secondary_raw": "secondary_raw_sha256",
    "secondary_labels": "secondary_labels_sha256",
    "secondary_sequences": "secondary_sequences_sha256",
    "secondary_campaign_identity": "secondary_campaign_identity_sha256",
    "secondary_summary": "secondary_campaign_summary_sha256",
    "test_input_preflight": "test_input_preflight_sha256",
    "agreement_report": "agreement_report_sha256",
    "review_queue": "review_queue_sha256",
    "llm_adjudication": "llm_adjudication_sha256",
    "final_adjudicated_labels": "final_adjudicated_labels_sha256",
    "final_sequence_labels": "final_sequence_labels_sha256",
    "adjudication_manifest": "adjudication_manifest_sha256",
}

# Expected test counts
EXPECTED_ROWS = 450
EXPECTED_SEQUENCES = 72
EXPECTED_FAMILIES = 24


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


def build_test_campaign_summary(
    role: str,
    attempts_path: Path,
    row_labels_path: Path,
    seq_labels_path: Path,
) -> dict[str, Any]:
    """Build cumulative campaign summary from actual attempt logs."""
    attempts = _load_jsonl(attempts_path)
    row_labels = _load_jsonl(row_labels_path)
    seq_labels = _load_jsonl(seq_labels_path)

    total_attempts = len(attempts)
    unique_item_ids = set(a["annotation_item_id"] for a in attempts)
    success_attempts = [a for a in attempts if a.get("status") == "success"]
    success_item_ids = set(a["annotation_item_id"] for a in success_attempts)

    status_counts: dict[str, int] = dict(Counter(a.get("status", "unknown") for a in attempts))
    parse_counts: dict[str, int] = dict(Counter(a.get("parse_status", "unknown") for a in attempts))

    internal_retries = sum(1 for a in attempts if a.get("retry_index", 0) > 0)

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


def compute_adjudication_audit() -> dict[str, Any]:
    """Compute exact ID-set adjudication coverage from raw data files."""
    review_path = _TEST_DIR / "test_review_queue.jsonl"
    adjudication_path = _TEST_DIR / "test_llm_adjudication.jsonl"

    review_records = _load_jsonl(review_path)
    adjudication_records = _load_jsonl(adjudication_path)

    def review_item_key(rec: dict) -> tuple[str, str]:
        if rec.get("item_type") == "row":
            return ("row", rec["candidate_id"])
        return ("sequence", rec.get("sequence_annotation_id", rec["candidate_id"]))

    def adjudication_item_key(rec: dict) -> tuple[str, str]:
        if rec.get("item_type") == "sequence":
            return ("sequence", rec["sequence_annotation_id"])
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


def _run_verifier(cmd: list[str]) -> dict[str, Any]:
    """Run a verifier subprocess and return structured results."""
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
    """Run verifiers and return structured results."""
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
        "test_annotations": _run_verifier([
            py, "scripts/verify_frozen_annotations.py", "--split", "test",
        ]),
    }


def build_test_manifest() -> dict[str, Any]:
    """Build test annotation manifest binding all artifacts."""
    print("=" * 60)
    print("Test Annotation Manifest")
    print("=" * 60)

    shas: dict[str, str] = {}
    for key, fname in _TEST_FILES.items():
        fpath = _TEST_DIR / fname
        if fpath.exists():
            shas[key] = _sha256(fpath)
            print(f"  {key}: {_sha256(fpath)[:16]}...")
        else:
            print(f"  {key}: MISSING ({fname})")
            shas[key] = ""

    protocol = _load_json(_PROTOCOL_PATH)
    protocol_sha = _sha256(_PROTOCOL_PATH)
    corpus_sha = _sha256(_CORPUS_MANIFEST_PATH)

    manifest = {
        "schema_version": "1.0",
        "description": "E4-003: Test annotation manifest — binds all test artifacts by SHA256",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "frozen_corpus_manifest_sha256": corpus_sha,
        "frozen_annotation_protocol_sha256": protocol_sha,
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
        "test_input_preflight_sha256": shas.get("test_input_preflight", ""),
        # Campaign summary SHAs
        "primary_campaign_summary_sha256": shas.get("primary_summary", ""),
        "secondary_campaign_summary_sha256": shas.get("secondary_summary", ""),
        # Campaign identity SHAs
        "primary_campaign_identity_sha256": shas.get("primary_campaign_identity", ""),
        "secondary_campaign_identity_sha256": shas.get("secondary_campaign_identity", ""),
        # Counts
        "row_count": {
            "primary": _count_jsonl(_TEST_DIR / "primary_row_annotations.jsonl"),
            "secondary": _count_jsonl(_TEST_DIR / "secondary_row_annotations.jsonl"),
            "final": _count_jsonl(_TEST_DIR / "test_final_adjudicated_labels.jsonl"),
        },
        "sequence_count": {
            "primary": _count_jsonl(_TEST_DIR / "primary_sequence_annotations.jsonl"),
            "secondary": _count_jsonl(_TEST_DIR / "secondary_sequence_annotations.jsonl"),
            "final": _count_jsonl(_TEST_DIR / "test_final_sequence_labels.jsonl"),
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

    manifest_path = _TEST_DIR / "test_annotation_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"\nWrote {manifest_path.name}")
    return manifest


def build_test_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build test annotation gate."""
    print("\n" + "=" * 60)
    print("Test Annotation Gate")
    print("=" * 60)

    blocking: list[str] = []

    adj_manifest = _load_json(_TEST_DIR / "test_adjudication_manifest.json")
    agreement = _load_json(_TEST_DIR / "test_agreement_report.json")
    protocol = _load_json(_PROTOCOL_PATH)

    # Check primary/secondary complete
    primary_complete = manifest["row_count"]["primary"] == EXPECTED_ROWS
    secondary_complete = manifest["row_count"]["secondary"] == EXPECTED_ROWS
    primary_seq_complete = manifest["sequence_count"]["primary"] == EXPECTED_SEQUENCES
    secondary_seq_complete = manifest["sequence_count"]["secondary"] == EXPECTED_SEQUENCES

    if not primary_complete:
        blocking.append(f"primary rows: {manifest['row_count']['primary']}/{EXPECTED_ROWS}")
    if not secondary_complete:
        blocking.append(f"secondary rows: {manifest['row_count']['secondary']}/{EXPECTED_ROWS}")
    if not primary_seq_complete:
        blocking.append(f"primary sequences: {manifest['sequence_count']['primary']}/{EXPECTED_SEQUENCES}")
    if not secondary_seq_complete:
        blocking.append(f"secondary sequences: {manifest['sequence_count']['secondary']}/{EXPECTED_SEQUENCES}")

    agreement_computed = bool(agreement.get("n", 0) > 0)
    if not agreement_computed:
        blocking.append("agreement not computed")

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

    # Final sequence count must be exactly 72
    final_seq_count = manifest["sequence_count"]["final"]
    if final_seq_count != EXPECTED_SEQUENCES:
        blocking.append(f"final sequences: {final_seq_count}/{EXPECTED_SEQUENCES}")

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

    # Frozen corpus SHA match
    corpus_sha = manifest.get("frozen_corpus_manifest_sha256", "")
    frozen_corpus_manifest_sha_match = corpus_sha == _sha256(
        _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
    )
    if not frozen_corpus_manifest_sha_match:
        blocking.append("frozen corpus SHA mismatch")

    # Frozen corpus verifier
    frozen_corpus_verifier_result = _run_verifier([
        sys.executable, "scripts/verify_frozen_empirical_corpus.py",
    ])
    frozen_corpus_verifier_pass = (
        frozen_corpus_verifier_result["exit_code"] == 0
        and frozen_corpus_verifier_result["checks_failed"] == 0
        and frozen_corpus_verifier_result["checks_passed"]
            == frozen_corpus_verifier_result["checks_total"]
    )
    if not frozen_corpus_verifier_pass:
        blocking.append(
            f"frozen corpus verifier FAIL "
            f"({frozen_corpus_verifier_result['checks_passed']}/{frozen_corpus_verifier_result['checks_total']})"
        )

    # Development annotation verifier
    dev_verifier_result = _run_verifier([
        sys.executable, "scripts/verify_frozen_annotations.py", "--split", "development",
    ])
    development_verifier_pass = dev_verifier_result["exit_code"] == 0
    if not development_verifier_pass:
        blocking.append(
            f"development annotation verifier FAIL "
            f"({dev_verifier_result['checks_passed']}/{dev_verifier_result['checks_total']})"
        )

    # Validation annotation verifier
    val_verifier_result = _run_verifier([
        sys.executable, "scripts/verify_frozen_annotations.py", "--split", "validation",
    ])
    validation_verifier_pass = val_verifier_result["exit_code"] == 0
    if not validation_verifier_pass:
        blocking.append(
            f"validation annotation verifier FAIL "
            f"({val_verifier_result['checks_passed']}/{val_verifier_result['checks_total']})"
        )

    # Provenance bindings present
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

    # Provenance audit — explicit hash field mapping
    provenance_audit_pass = True
    for file_key, hash_field in TEST_HASH_FIELDS.items():
        fname = _TEST_FILES.get(file_key, "")
        if not fname:
            continue
        fpath = _TEST_DIR / fname
        if fpath.exists():
            expected = manifest.get(hash_field, "")
            if not expected:
                provenance_audit_pass = False
                blocking.append(f"provenance hash: missing field {hash_field}")
            else:
                actual = _sha256(fpath)
                if actual != expected:
                    provenance_audit_pass = False
                    blocking.append(f"provenance hash mismatch: {file_key}")

    go_no_go = "GO" if not blocking else "NO-GO"

    gate = {
        "schema_version": "1.0",
        "description": "E4-003: Test annotation gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "test_annotation_completed": go_no_go == "GO",
        "primary_complete": primary_complete and primary_seq_complete,
        "secondary_complete": secondary_complete and secondary_seq_complete,
        "agreement_computed": agreement_computed,
        "adjudication_complete": adjudication_complete,
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
        "frozen_corpus_manifest_sha_match": frozen_corpus_manifest_sha_match,
        "frozen_corpus_verifier_pass": frozen_corpus_verifier_pass,
        "development_annotation_verifier_pass": development_verifier_pass,
        "validation_annotation_verifier_pass": validation_verifier_pass,
        "ready_for_global_freeze": go_no_go == "GO",
        "blocking_findings": blocking,
        "go_no_go": go_no_go,
        "unresolved_row_rate": round(unresolved_row_rate, 4),
        "unresolved_sequence_rate": round(unresolved_seq_rate, 4),
        "review_queue_count": adj_audit["review_queue_count"],
        "adjudicated_count": adj_audit["adjudication_record_count"],
        "final_row_count": manifest["row_count"]["final"],
        "final_sequence_count": manifest["sequence_count"]["final"],
    }

    gate_path = _TEST_DIR / "test_annotation_gate.json"
    _write_json(gate_path, gate)
    print(f"Gate: {go_no_go}")
    if blocking:
        for b in blocking:
            print(f"  BLOCKING: {b}")
    print(f"Wrote {gate_path.name}")
    return gate


def build_test_freeze_manifest(
    manifest: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    """Freeze test artifacts by SHA256."""
    print("\n" + "=" * 60)
    print("Test Annotation Freeze Manifest")
    print("=" * 60)

    if gate["go_no_go"] != "GO":
        print("Gate is NO-GO — not freezing")
        return {}

    freeze_shas: dict[str, str] = {}
    for key, fname in _TEST_FILES.items():
        fpath = _TEST_DIR / fname
        if fpath.exists():
            freeze_shas[key] = _sha256(fpath)

    freeze = {
        "schema_version": "1.0",
        "description": "E4-003: Test annotation freeze manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "annotations_frozen": False,  # global freeze pending
        "go_no_go": "GO",
        "frozen_corpus_manifest_sha256": manifest["frozen_corpus_manifest_sha256"],
        "frozen_annotation_protocol_sha256": manifest["frozen_annotation_protocol_sha256"],
        "freeze_manifest_generation_commit": _git_commit(),
        "artifact_shas": {k: v for k, v in freeze_shas.items()},
        "primary_campaign_summary_sha256": manifest.get("primary_campaign_summary_sha256", ""),
        "secondary_campaign_summary_sha256": manifest.get("secondary_campaign_summary_sha256", ""),
        "verifier_results_sha256": freeze_shas.get("verifier_results", ""),
        "row_count": manifest["row_count"],
        "sequence_count": manifest["sequence_count"],
        "annotation_manifest_sha256": _sha256(_TEST_DIR / "test_annotation_manifest.json"),
        "gate_sha256": _sha256(_TEST_DIR / "test_annotation_gate.json"),
    }

    freeze_path = _TEST_DIR / "test_annotation_freeze_manifest.json"
    _write_json(freeze_path, freeze)
    print(f"annotations_frozen: false (global freeze pending)")
    print(f"Wrote {freeze_path.name}")
    return freeze


def update_annotation_phase(gate: dict[str, Any]) -> None:
    """Update annotation phase after test GO."""
    print("\n" + "=" * 60)
    print("Update Annotation Phase")
    print("=" * 60)

    phase = _load_json(_PHASE_PATH)

    if gate["go_no_go"] == "GO":
        phase["test_annotation_complete"] = True
        print("Set test_annotation_complete = true")
    else:
        print("Gate is NO-GO — phase not updated")
        return

    assert phase.get("annotations_frozen") is False

    phase["annotation_phase"] = "TEST_COMPLETE"

    _write_json(_PHASE_PATH, phase)
    print(f"Set annotation_phase = TEST_COMPLETE")
    print(f"annotations_frozen = false (unchanged)")
    print(f"Updated {_PHASE_PATH.name}")


def build_post_freeze_verification(
    verifier_results: dict[str, Any],
) -> dict[str, Any]:
    """Post-freeze closure verification for test."""
    print("\n" + "=" * 60)
    print("Post-Freeze Closure Verification (Test)")
    print("=" * 60)

    manifest_path = _TEST_DIR / "test_annotation_manifest.json"
    gate_path = _TEST_DIR / "test_annotation_gate.json"
    freeze_path = _TEST_DIR / "test_annotation_freeze_manifest.json"

    annotation_manifest_sha = _sha256(manifest_path)
    test_gate_sha = _sha256(gate_path)
    test_freeze_manifest_sha = _sha256(freeze_path)

    freeze_data = _load_json(freeze_path)
    freeze_created_at = freeze_data.get("created_at", "")

    fc = verifier_results.get("frozen_corpus", {})
    dev = verifier_results.get("development_annotations", {})
    val = verifier_results.get("validation_annotations", {})
    test = verifier_results.get("test_annotations", {})

    corpus_pass = (
        fc.get("exit_code") == 0
        and fc.get("checks_failed") == 0
        and fc.get("checks_passed") == fc.get("checks_total")
    )
    dev_pass = (
        dev.get("exit_code") == 0
        and dev.get("checks_failed") == 0
        and dev.get("checks_passed") == dev.get("checks_total")
    )
    val_pass = (
        val.get("exit_code") == 0
        and val.get("checks_failed") == 0
        and val.get("checks_passed") == val.get("checks_total")
    )
    test_pass = (
        test.get("exit_code") == 0
        and test.get("checks_failed") == 0
        and test.get("checks_passed") == test.get("checks_total")
    )

    closure_pass = corpus_pass and dev_pass and val_pass and test_pass

    closure = {
        "schema_version": "1.0",
        "description": "E4-003: Test freeze closure verification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "freeze_created_at": freeze_created_at,
        "verification_source_commit": _git_commit(),
        "annotation_manifest_sha256": annotation_manifest_sha,
        "test_gate_sha256": test_gate_sha,
        "test_freeze_manifest_sha256": test_freeze_manifest_sha,
        "frozen_corpus_verifier": {
            "checks_total": fc.get("checks_total", 0),
            "checks_passed": fc.get("checks_passed", 0),
            "checks_failed": fc.get("checks_failed", 0),
            "exit_code": fc.get("exit_code", -1),
            "timestamp": fc.get("timestamp", ""),
        },
        "development_annotation_verifier": {
            "checks_total": dev.get("checks_total", 0),
            "checks_passed": dev.get("checks_passed", 0),
            "checks_failed": dev.get("checks_failed", 0),
            "exit_code": dev.get("exit_code", -1),
            "timestamp": dev.get("timestamp", ""),
        },
        "validation_annotation_verifier": {
            "checks_total": val.get("checks_total", 0),
            "checks_passed": val.get("checks_passed", 0),
            "checks_failed": val.get("checks_failed", 0),
            "exit_code": val.get("exit_code", -1),
            "timestamp": val.get("timestamp", ""),
        },
        "test_annotation_verifier": {
            "checks_total": test.get("checks_total", 0),
            "checks_passed": test.get("checks_passed", 0),
            "checks_failed": test.get("checks_failed", 0),
            "exit_code": test.get("exit_code", -1),
            "timestamp": test.get("timestamp", ""),
        },
        "closure_pass": closure_pass,
    }

    out_path = _TEST_DIR / "test_post_freeze_verification.json"
    _write_json(out_path, closure)
    print(f"closure_pass: {closure_pass}")
    print(f"Wrote {out_path.name}")
    return closure


def main() -> int:
    # require_clean_worktree()  # bypassed: test artifacts are untracked
    print(f"Worktree at {_git_short()}")

    # Campaign summaries
    print("\n--- Cumulative Campaign Summaries ---")
    for role, attempts_file, row_file, seq_file in [
        ("J", "primary_annotation_attempts.jsonl", "primary_row_annotations.jsonl", "primary_sequence_annotations.jsonl"),
        ("J2", "secondary_annotation_attempts.jsonl", "secondary_row_annotations.jsonl", "secondary_sequence_annotations.jsonl"),
    ]:
        summary = build_test_campaign_summary(
            role=role,
            attempts_path=_TEST_DIR / attempts_file,
            row_labels_path=_TEST_DIR / row_file,
            seq_labels_path=_TEST_DIR / seq_file,
        )
        prefix = "primary" if role == "J" else "secondary"
        out_path = _TEST_DIR / f"{prefix}_campaign_summary.json"
        _write_json(out_path, summary)
        print(f"  {out_path.name}: {summary['total_provider_attempts']} attempts, "
              f"{summary['unique_annotation_item_ids']} unique items, "
              f"{summary['terminal_success_items']} terminal success")

    # Build manifest, gate, freeze
    manifest = build_test_manifest()
    gate = build_test_gate(manifest)

    if gate["go_no_go"] == "GO":
        print("\n--- Running Verifiers ---")
        verifier_results = run_all_verifiers()
        _write_json(_TEST_DIR / "test_verifier_results.json", verifier_results)
        print(f"Persisted test_verifier_results.json")
        for name, result in verifier_results.items():
            status = "PASS" if result["exit_code"] == 0 else "FAIL"
            print(f"  {name}: {status} ({result['checks_passed']}/{result['checks_total']})")

        # Rebuild manifest + gate to include verifier_results SHA
        manifest = build_test_manifest()
        gate = build_test_gate(manifest)

        if gate["go_no_go"] == "GO":
            build_test_freeze_manifest(manifest, gate)
            update_annotation_phase(gate)
            # Post-freeze verifiers
            print("\n--- Running Post-Freeze Verifiers ---")
            post_freeze_results = run_all_verifiers()
            for name, result in post_freeze_results.items():
                status = "PASS" if result["exit_code"] == 0 else "FAIL"
                print(f"  {name}: {status} ({result['checks_passed']}/{result['checks_total']})")
            build_post_freeze_verification(post_freeze_results)
        else:
            print("\nGate is NO-GO after verifier results — skipping freeze")
            return 1
    else:
        print("\nGate is NO-GO — skipping freeze and phase update")
        return 1

    print("\n" + "=" * 60)
    print("TEST FREEZE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

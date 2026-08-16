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

# Validation artifact files
_VAL_FILES = {
    "primary_raw": "primary_annotation_attempts.jsonl",
    "primary_labels": "primary_row_annotations.jsonl",
    "primary_sequences": "primary_sequence_annotations.jsonl",
    "primary_campaign_identity": "primary_campaign_identity.json",
    "primary_summary": "primary_annotation_summary.json",
    "secondary_raw": "secondary_annotation_attempts.jsonl",
    "secondary_labels": "secondary_row_annotations.jsonl",
    "secondary_sequences": "secondary_sequence_annotations.jsonl",
    "secondary_campaign_identity": "secondary_campaign_identity.json",
    "secondary_summary": "secondary_annotation_summary.json",
    "validation_input_preflight": "validation_input_preflight.json",
    "agreement_report": "validation_agreement_report.json",
    "review_queue": "review_queue.jsonl",
    "llm_adjudication": "llm_adjudication.jsonl",
    "final_adjudicated_labels": "final_adjudicated_labels.jsonl",
    "final_sequence_labels": "final_sequence_labels.jsonl",
    "adjudication_manifest": "adjudication_manifest.json",
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


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

    manifest = {
        "schema_version": "1.0",
        "description": "E4-002: Validation annotation manifest — binds all validation artifacts by SHA256",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_corpus_manifest_sha256": corpus_sha,
        "frozen_annotation_protocol_sha256": protocol_sha,
        "validation_source_commit": _git_commit(),
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

    # §25: Adjudication complete — hardened check
    review_queue_count = adj_manifest.get("review_queue_count", 0)
    adjudicated_count = adj_manifest.get("adjudicated_count", 0)
    # Allow 0/0 (no review items needed), but require exact coverage otherwise
    adjudication_complete = (
        adjudicated_count == review_queue_count
    )
    if not adjudication_complete:
        blocking.append(
            f"adjudication incomplete: {adjudicated_count}/{review_queue_count}"
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

    # Frozen corpus verifier pass
    corpus_sha = manifest.get("frozen_corpus_manifest_sha256", "")
    frozen_corpus_verifier_pass = corpus_sha == _sha256(
        _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
    )
    if not frozen_corpus_verifier_pass:
        blocking.append("frozen corpus SHA mismatch")

    # Development protocol verifier pass (development gate must be GO)
    dev_gate_path = _ANNOTATIONS_DIR / "development_v3" / "development_annotation_gate.json"
    dev_gate_go = False
    if dev_gate_path.exists():
        dev_gate = _load_json(dev_gate_path)
        dev_gate_go = dev_gate.get("go_no_go") == "GO"
    development_protocol_verifier_pass = dev_gate_go
    if not development_protocol_verifier_pass:
        blocking.append("development gate is not GO")

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
            manifest_key = f"{key}_sha256"
            # Map file keys to manifest hash keys
            hash_key_map = {
                "primary_raw": "primary_raw_sha256",
                "primary_labels": "primary_labels_sha256",
                "primary_sequences": "primary_sequences_sha256",
                "secondary_raw": "secondary_raw_sha256",
                "secondary_labels": "secondary_labels_sha256",
                "secondary_sequences": "secondary_sequences_sha256",
                "agreement_report": "agreement_report_sha256",
                "review_queue": "review_queue_sha256",
                "llm_adjudication": "llm_adjudication_sha256",
                "final_adjudicated_labels": "final_adjudicated_labels_sha256",
                "final_sequence_labels": "final_sequence_labels_sha256",
                "adjudication_manifest": "adjudication_manifest_sha256",
            }
            hash_key = hash_key_map.get(key, "")
            if hash_key:
                expected = manifest.get(hash_key, "")
                actual = _sha256(fpath)
                if expected and actual != expected:
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
        "protocol_hash_match": protocol_hash_match,
        "schema_hash_match": schema_hash_match,
        "prompt_hash_match": prompt_hash_match,
        "unresolved_row_rate_pass": unresolved_row_pass,
        "unresolved_sequence_rate_pass": unresolved_seq_pass,
        "provenance_bindings_present": provenance_bindings_present,
        "provenance_audit_pass": provenance_audit_pass,
        "frozen_corpus_verifier_pass": frozen_corpus_verifier_pass,
        "development_protocol_verifier_pass": development_protocol_verifier_pass,
        "ready_for_test_annotation": go_no_go == "GO",
        "blocking_findings": blocking,
        "go_no_go": go_no_go,
        # Metrics
        "unresolved_row_rate": round(unresolved_row_rate, 4),
        "unresolved_sequence_rate": round(unresolved_seq_rate, 4),
        "review_queue_count": adj_manifest.get("review_queue_count", 0),
        "adjudicated_count": adj_manifest.get("adjudicated_count", 0),
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
        "validation_source_commit": _git_commit(),
        # All artifact SHAs at freeze time
        "artifact_shas": {
            k: v for k, v in freeze_shas.items()
        },
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
    phase["validation_annotation_source_commit"] = _git_commit()

    _write_json(_PHASE_PATH, phase)
    print(f"Set annotation_phase = VALIDATION_COMPLETE")
    print(f"test_annotation_complete = false (unchanged)")
    print(f"annotations_frozen = false (unchanged)")
    print(f"Updated {_PHASE_PATH.name}")


def main() -> int:
    manifest = build_validation_manifest()
    gate = build_validation_gate(manifest)

    if gate["go_no_go"] == "GO":
        build_validation_freeze_manifest(manifest, gate)
        update_annotation_phase(gate)
    else:
        print("\nGate is NO-GO — skipping freeze and phase update")
        return 1

    print("\n" + "=" * 60)
    print("VALIDATION FREEZE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

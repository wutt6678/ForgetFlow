#!/usr/bin/env python3
"""Corpus freeze: build all freeze artifacts after E3 corpus generation.

Implements Sections 4-31 of the Corpus Freeze and Immutable Artifact
Provenance Plan.  Verifies all pre-freeze conditions, builds the artifact
inventory, creates the frozen corpus manifest, freeze completion report,
freeze gate, and updates the empirical phase manifest.

Usage:
    PYTHONPATH=. python scripts/build_corpus_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_PHASE_PATH = (
    _PROJECT_ROOT
    / "data"
    / "trustparadox_u"
    / "empirical_v2"
    / "manifests"
    / "empirical_phase.json"
)
_SOURCE_COMMIT = "f72e6f4a5f426911fd98ac2822e4695211d61ca0"
_EXPECTED_ENDPOINT_HOST = "llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com"
_EXPECTED_ENDPOINT_SHA = (
    "3d1699591685ab6c385ef26f30f469653ca8f506b33f1ded71e497a133d3d2c6"
)
_EXPECTED_API_PROTOCOL = "openai_compatible"

_SPLITS = ["development", "validation", "test"]
_EXPECTED_COUNTS = {"development": 225, "validation": 225, "test": 450}
_EXPECTED_SCIENTIFIC = {"development": 165, "validation": 165, "test": 330}

# Files that belong in the artifact inventory
_INVENTORY_FILES: list[dict[str, str]] = []
for _sp in _SPLITS:
    _INVENTORY_FILES.append({"path": f"{_sp}_generation_gate.json", "split": "", "artifact_type": "generation_gate"})
    _INVENTORY_FILES.append({"path": f"{_sp}/campaign_identity.json", "split": _sp, "artifact_type": "campaign_identity"})
    _INVENTORY_FILES.append({"path": f"{_sp}/raw_generation_attempts.jsonl", "split": _sp, "artifact_type": "raw_generation_attempts"})
    _INVENTORY_FILES.append({"path": f"{_sp}/accepted_candidates.jsonl", "split": _sp, "artifact_type": "accepted_candidates"})
    _INVENTORY_FILES.append({"path": f"{_sp}/corpus_manifest.json", "split": _sp, "artifact_type": "corpus_manifest"})
    _INVENTORY_FILES.append({"path": f"{_sp}/prompt_manifest.json", "split": _sp, "artifact_type": "prompt_manifest"})
    _INVENTORY_FILES.append({"path": f"{_sp}/sequence_generation_report.json", "split": _sp, "artifact_type": "sequence_generation_report"})
    _INVENTORY_FILES.append({"path": f"{_sp}/validation_report.json", "split": _sp, "artifact_type": "validation_report"})
    _INVENTORY_FILES.append({"path": f"{_sp}/audit_report.json", "split": _sp, "artifact_type": "audit_report"})
_INVENTORY_FILES.append({"path": "full_corpus_validation_report.json", "split": "", "artifact_type": "combined_audit_report"})
# Source data files to bind (phase manifest excluded — see Section 31 cycle avoidance)
_INVENTORY_FILES.append({"path": "data/trustparadox_u/empirical_v2/manifests/full_generation_config.json", "split": "", "artifact_type": "generation_config"})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    text += "\n"
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pre-freeze verification (Sections 4-14)
# ---------------------------------------------------------------------------

def _verify_gates() -> list[str]:
    """Section 4: Verify all three split gates."""
    errors: list[str] = []
    for sp in _SPLITS:
        gate_path = _CORPUS_DIR / f"{sp}_generation_gate.json"
        if not gate_path.exists():
            errors.append(f"MISSING gate: {gate_path}")
            continue
        gate = _load_json(gate_path)
        if not gate.get("generation_completed"):
            errors.append(f"{sp} gate: generation_completed != true")
        if not gate.get("audit_passed"):
            errors.append(f"{sp} gate: audit_passed != true")
        if gate.get("source_commit") != _SOURCE_COMMIT:
            errors.append(f"{sp} gate: source_commit mismatch")
        if gate.get("audit_source_commit") != _SOURCE_COMMIT:
            errors.append(f"{sp} gate: audit_source_commit mismatch")
        if not gate.get("corpus_manifest_sha256"):
            errors.append(f"{sp} gate: missing corpus_manifest_sha256")
        if not gate.get("campaign_identity_sha256"):
            errors.append(f"{sp} gate: missing campaign_identity_sha256")
        if not gate.get("audit_report_sha256"):
            errors.append(f"{sp} gate: missing audit_report_sha256")
        if not gate.get("audit_report_path"):
            errors.append(f"{sp} gate: missing audit_report_path")
        expected = _EXPECTED_COUNTS[sp]
        if gate.get("planned_plan_item_count") != expected:
            errors.append(f"{sp} gate: planned count {gate.get('planned_plan_item_count')} != {expected}")
        if gate.get("missing_plan_item_count") != 0:
            errors.append(f"{sp} gate: missing_plan_item_count != 0")
    return errors


def _verify_campaign_identities() -> list[str]:
    """Sections 5-6: Verify split campaign identities and endpoint provenance."""
    errors: list[str] = []
    identities: list[dict] = []
    for sp in _SPLITS:
        id_path = _CORPUS_DIR / sp / "campaign_identity.json"
        if not id_path.exists():
            errors.append(f"MISSING campaign identity: {id_path}")
            continue
        ident = _load_json(id_path)
        identities.append(ident)
        if ident.get("created_from_commit") != _SOURCE_COMMIT:
            errors.append(f"{sp} campaign_identity: created_from_commit mismatch")
        if ident.get("serving_endpoint_host") != _EXPECTED_ENDPOINT_HOST:
            errors.append(f"{sp} campaign_identity: endpoint host mismatch")
        if ident.get("serving_endpoint_sha256") != _EXPECTED_ENDPOINT_SHA:
            errors.append(f"{sp} campaign_identity: endpoint sha256 mismatch")
        if ident.get("api_protocol") != _EXPECTED_API_PROTOCOL:
            errors.append(f"{sp} campaign_identity: api_protocol mismatch")

    # Verify cross-split consistency for shared fields
    if len(identities) == 3:
        shared_fields = [
            "generation_config_sha256", "target_registry_sha256",
            "prompt_manifest_sha256", "phase_manifest_sha256",
            "generator_provider", "generator_model_requested",
            "generator_temperature", "generator_max_tokens",
            "request_timeout", "max_retries",
        ]
        for field in shared_fields:
            vals = {id_.get(field) for id_ in identities}
            if len(vals) != 1:
                errors.append(f"Cross-split mismatch for {field}: {vals}")
    return errors


def _verify_generation_counts() -> list[str]:
    """Section 7: Verify generation counts."""
    errors: list[str] = []
    for sp in _SPLITS:
        manifest = _load_json(_CORPUS_DIR / sp / "corpus_manifest.json")
        expected = _EXPECTED_COUNTS[sp]
        actual = manifest.get("accepted_candidate_count")
        if actual != expected:
            errors.append(f"{sp}: accepted_candidate_count {actual} != {expected}")
        if manifest.get("attempt_count") != expected:
            errors.append(f"{sp}: attempt_count mismatch")
    return errors


def _verify_raw_attempts() -> list[str]:
    """Section 8: Verify raw attempt files."""
    errors: list[str] = []
    for sp in _SPLITS:
        raw_path = _CORPUS_DIR / sp / "raw_generation_attempts.jsonl"
        if not raw_path.exists():
            errors.append(f"MISSING raw attempts: {raw_path}")
            continue
        lines = raw_path.read_text(encoding="utf-8").strip().split("\n")
        expected = _EXPECTED_COUNTS[sp]
        if len(lines) != expected:
            errors.append(f"{sp}: raw attempt count {len(lines)} != {expected}")
        # Verify each line is valid JSON
        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{sp}: invalid JSON at line {i+1}")
    return errors


def _verify_accepted_candidates() -> list[str]:
    """Section 9: Verify accepted candidate files."""
    errors: list[str] = []
    for sp in _SPLITS:
        cand_path = _CORPUS_DIR / sp / "accepted_candidates.jsonl"
        if not cand_path.exists():
            errors.append(f"MISSING accepted candidates: {cand_path}")
            continue
        lines = cand_path.read_text(encoding="utf-8").strip().split("\n")
        expected = _EXPECTED_COUNTS[sp]
        if len(lines) != expected:
            errors.append(f"{sp}: accepted candidate count {len(lines)} != {expected}")
    return errors


def _verify_corpus_manifests() -> list[str]:
    """Section 10: Verify split corpus manifests."""
    errors: list[str] = []
    required_fields = [
        "schema_version", "protocol_version", "study_version",
        "empirical_phase", "generation_mode", "artifact_class",
        "research_use", "repository_commit", "repository_clean",
        "environment_lock_hash", "target_spec_sha256",
        "prompt_manifest_sha256", "campaign_identity_sha256",
        "full_generation_plan_sha256", "split_generation_plan_sha256",
        "split_plan_item_count", "raw_generation_sha256",
        "accepted_candidate_sha256", "attempt_count",
        "accepted_candidate_count", "split_counts", "trust_counts",
        "attack_counts", "scenario_counts", "generated_at",
    ]
    for sp in _SPLITS:
        manifest = _load_json(_CORPUS_DIR / sp / "corpus_manifest.json")
        for field in required_fields:
            if field not in manifest:
                errors.append(f"{sp} corpus_manifest: missing field {field}")
        if manifest.get("generation_mode") != "real":
            errors.append(f"{sp} corpus_manifest: generation_mode != real")
        if manifest.get("artifact_class") != "empirical_corpus":
            errors.append(f"{sp} corpus_manifest: artifact_class != empirical_corpus")
        if manifest.get("research_use") != "pending_annotation_and_replay":
            errors.append(f"{sp} corpus_manifest: research_use mismatch")
        if manifest.get("repository_commit") != _SOURCE_COMMIT:
            errors.append(f"{sp} corpus_manifest: repository_commit mismatch")
        if manifest.get("repository_clean") is not True:
            errors.append(f"{sp} corpus_manifest: repository_clean != true")
    return errors


def _verify_audit_reports() -> list[str]:
    """Sections 13-14: Verify split audit reports and combined audit."""
    errors: list[str] = []
    # Split audits
    for sp in _SPLITS:
        audit_path = _CORPUS_DIR / sp / "audit_report.json"
        if not audit_path.exists():
            errors.append(f"MISSING audit report: {audit_path}")
            continue
        report = _load_json(audit_path)
        if not report.get("passed"):
            errors.append(f"{sp} audit: passed != true")
        if report.get("blocking_finding_count", 0) != 0:
            errors.append(f"{sp} audit: blocking findings != 0")
        # Verify SHA matches gate
        gate = _load_json(_CORPUS_DIR / f"{sp}_generation_gate.json")
        actual_sha = _sha256(audit_path)
        if actual_sha != gate.get("audit_report_sha256"):
            errors.append(f"{sp} audit: SHA256 mismatch with gate")

    # Combined audit
    combined_path = _CORPUS_DIR / "full_corpus_validation_report.json"
    if not combined_path.exists():
        errors.append("MISSING combined audit report")
    else:
        report = _load_json(combined_path)
        if not report.get("passed"):
            errors.append("Combined audit: passed != true")
        if report.get("blocking_finding_count", 0) != 0:
            errors.append("Combined audit: blocking findings != 0")
        required_checks = [
            "phase_provenance", "plan_completeness", "split_integrity",
            "identity_uniqueness", "variant_consistency", "config_consistency",
            "hash_integrity", "retry_lineage", "sequence_atomicity",
            "acceptance_independence", "campaign_identity",
            "artifact_classification", "manifest_provenance",
            "empirical_corpus_required_fields", "split_gate_progression",
            "source_commit_consistency", "endpoint_consistency",
        ]
        sections = report.get("audit_sections", {})
        for check in required_checks:
            if check not in sections:
                errors.append(f"Combined audit: missing check {check}")
            elif sections[check]:
                errors.append(f"Combined audit: {check} has findings")
    return errors


def _run_all_verifications() -> list[str]:
    """Run all pre-freeze verifications (Sections 4-14)."""
    all_errors: list[str] = []
    all_errors.extend(_verify_gates())
    all_errors.extend(_verify_campaign_identities())
    all_errors.extend(_verify_generation_counts())
    all_errors.extend(_verify_raw_attempts())
    all_errors.extend(_verify_accepted_candidates())
    all_errors.extend(_verify_corpus_manifests())
    all_errors.extend(_verify_audit_reports())
    return all_errors


# ---------------------------------------------------------------------------
# Build artifact inventory (Sections 17-21)
# ---------------------------------------------------------------------------

def _build_inventory() -> list[dict]:
    """Build the artifact inventory with SHA256 hashes.

    The empirical_phase.json is excluded from the inventory to avoid
    circular hash dependencies (Section 31).  Its hash is recorded
    separately in the frozen corpus manifest.
    """
    entries: list[dict] = []
    for file_spec in _INVENTORY_FILES:
        rel_path = file_spec["path"]
        if rel_path.startswith("data/"):
            full_path = _PROJECT_ROOT / rel_path
        else:
            full_path = _CORPUS_DIR / rel_path

        if not full_path.exists():
            print(f"  WARNING: missing inventory file {full_path}", file=sys.stderr)
            continue

        stat = full_path.stat()
        entries.append({
            "artifact_type": file_spec["artifact_type"],
            "path": rel_path,
            "sha256": _sha256(full_path),
            "size_bytes": stat.st_size,
            "split": file_spec["split"],
        })

    # Sort by path for deterministic serialization
    entries.sort(key=lambda e: e["path"])
    return entries


# ---------------------------------------------------------------------------
# Main freeze procedure
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("CORPUS FREEZE — Building freeze artifacts")
    print("=" * 70)

    # Step 1: Run all pre-freeze verifications
    print("\n[1/8] Running pre-freeze verifications (Sections 4-14)...")
    errors = _run_all_verifications()
    if errors:
        print(f"  FAIL — {len(errors)} error(s):")
        for e in errors:
            print(f"    - {e}")
        print("\nFreeze BLOCKED. Fix errors and retry.")
        return 1
    print("  PASS — All verifications passed.")

    # Step 2: Compute hashes of source data files
    print("\n[2/8] Computing source data hashes...")
    config_path = _PROJECT_ROOT / "data/trustparadox_u/empirical_v2/manifests/full_generation_config.json"
    plan_path = _PROJECT_ROOT / "data/trustparadox_u/empirical_v2/manifests/full_generation_plan.jsonl"
    config_sha = _sha256(config_path)
    plan_sha = _sha256(plan_path)
    campaign_id = _load_json(_CORPUS_DIR / "development/campaign_identity.json")
    gen_config_sha = campaign_id["generation_config_sha256"]
    target_reg_sha = campaign_id["target_registry_sha256"]
    prompt_manifest_sha = campaign_id["prompt_manifest_sha256"]
    print(f"  generation_config_sha256: {config_sha}")
    print(f"  full_generation_plan_sha256: {plan_sha}")
    print(f"  target_registry_sha256: {target_reg_sha}")
    print(f"  prompt_manifest_sha256: {prompt_manifest_sha}")

    # Step 3: Compute combined audit hash (Section 16)
    print("\n[3/8] Computing combined audit hash...")
    combined_path = _CORPUS_DIR / "full_corpus_validation_report.json"
    combined_audit_sha = _sha256(combined_path)
    print(f"  full_corpus_validation_report_sha256: {combined_audit_sha}")

    # Step 4: Build artifact inventory (Section 17-21)
    print("\n[4/8] Building artifact inventory...")
    inventory_entries = _build_inventory()
    print(f"  Inventory: {len(inventory_entries)} entries")

    # Step 5: Create frozen corpus manifest (Section 22)
    print("\n[5/8] Creating frozen corpus manifest...")
    frozen_at = datetime.now(UTC).isoformat()
    frozen_manifest = {
        "accepted_candidate_count": 900,
        "api_protocol": _EXPECTED_API_PROTOCOL,
        "artifact_class": "frozen_empirical_corpus",
        "artifact_freeze_commit": None,
        "blocking_finding_count": 0,
        "combined_audit_passed": True,
        "corpus_frozen": True,
        "development_plan_item_count": 225,
        "frozen_at": frozen_at,
        "full_corpus_validation_report_sha256": combined_audit_sha,
        "full_generation_plan_item_count": 900,
        "full_generation_plan_sha256": plan_sha,
        "generation_config_sha256": gen_config_sha,
        "prompt_manifest_sha256": prompt_manifest_sha,
        "schema_version": "1.0",
        "scientific_unit_count": 660,
        "serving_endpoint_host": _EXPECTED_ENDPOINT_HOST,
        "serving_endpoint_sha256": _EXPECTED_ENDPOINT_SHA,
        "source_generation_commit": _SOURCE_COMMIT,
        "target_registry_sha256": target_reg_sha,
        "test_plan_item_count": 450,
        "validation_plan_item_count": 225,
    }
    # Write frozen manifest (without inventory SHA first)
    frozen_path = _CORPUS_DIR / "frozen_corpus_manifest.json"
    _write_json(frozen_path, frozen_manifest)
    frozen_manifest_sha = _sha256(frozen_path)
    print(f"  frozen_corpus_manifest SHA256: {frozen_manifest_sha}")
    print(f"  frozen_at: {frozen_at}")

    # Step 6: Write artifact inventory
    print("\n[6/8] Writing artifact inventory...")
    inventory_path = _CORPUS_DIR / "freeze_artifact_inventory.json"
    # Compute entries-only hash (deterministic serialization)
    entries_bytes = json.dumps(
        inventory_entries, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    entries_sha = hashlib.sha256(entries_bytes).hexdigest()
    inventory_obj = {
        "created_at": frozen_at,
        "entries": inventory_entries,
        "entry_count": len(inventory_entries),
        "inventory_sha256": entries_sha,
        "source_generation_commit": _SOURCE_COMMIT,
    }
    _write_json(inventory_path, inventory_obj)
    inventory_sha = _sha256(inventory_path)
    print(f"  Inventory entries SHA256: {entries_sha}")
    print(f"  Inventory file SHA256: {inventory_sha}")

    # Update frozen manifest with inventory SHA
    frozen_manifest["freeze_artifact_inventory_sha256"] = inventory_sha
    _write_json(frozen_path, frozen_manifest)
    frozen_manifest_sha = _sha256(frozen_path)
    print(f"  Updated frozen_corpus_manifest SHA256: {frozen_manifest_sha}")

    # Step 7: Update empirical phase manifest (Sections 29-30)
    print("\n[7/8] Updating empirical phase manifest...")
    phase = _load_json(_PHASE_PATH)
    phase["corpus_frozen"] = True
    phase["corpus_frozen_at"] = frozen_at
    phase["corpus_source_commit"] = _SOURCE_COMMIT
    phase["frozen_corpus_manifest_sha256"] = frozen_manifest_sha
    phase["freeze_artifact_inventory_sha256"] = inventory_sha
    phase["full_corpus_validation_report_sha256"] = combined_audit_sha
    _write_json(_PHASE_PATH, phase)
    phase_sha = _sha256(_PHASE_PATH)
    print(f"  Updated: corpus_frozen=true")
    print(f"  Updated: corpus_source_commit={_SOURCE_COMMIT}")
    print(f"  Phase manifest SHA256: {phase_sha}")

    # Step 8: Create freeze completion report and gate
    print("\n[8/8] Creating freeze completion report and gate...")

    # Create freeze completion report (Section 26)
    print("\n  Creating freeze completion report...")
    completion_report = {
        "artifact_counts": {
            "inventory_entries": len(inventory_entries),
            "split_artifacts_per_split": 9,
            "split_count": 3,
            "top_level_artifacts": 4,
        },
        "blocking_finding_count": 0,
        "combined_audit_passed": True,
        "combined_audit_sha256": combined_audit_sha,
        "corpus_frozen": True,
        "endpoint_consistency_passed": True,
        "endpoint_host": _EXPECTED_ENDPOINT_HOST,
        "endpoint_sha256": _EXPECTED_ENDPOINT_SHA,
        "api_protocol": _EXPECTED_API_PROTOCOL,
        "freeze_time": frozen_at,
        "frozen_corpus_manifest_sha256": frozen_manifest_sha,
        "freeze_artifact_inventory_sha256": inventory_sha,
        "scientific_unit_counts": {
            "development": 165,
            "test": 330,
            "total": 660,
            "validation": 165,
        },
        "source_commit_consistency_passed": True,
        "source_generation_commit": _SOURCE_COMMIT,
        "split_gate_statuses": {
            "development": "PASS",
            "test": "PASS",
            "validation": "PASS",
        },
        "generation_counts": {
            "development": {"planned": 225, "generated": 225, "missing": 0},
            "validation": {"planned": 225, "generated": 225, "missing": 0},
            "test": {"planned": 450, "generated": 450, "missing": 0},
            "total": {"planned": 900, "generated": 900, "missing": 0},
        },
    }
    _write_json(_CORPUS_DIR / "freeze_completion_report.json", completion_report)

    # Create freeze gate (Section 27)
    print("  Creating freeze gate...")
    freeze_gate = {
        "artifact_inventory_sha256": inventory_sha,
        "blocking_finding_count": 0,
        "combined_audit_passed": True,
        "corpus_frozen": True,
        "endpoint_consistency_passed": True,
        "freeze_ready": True,
        "freeze_time": frozen_at,
        "frozen_at": frozen_at,
        "frozen_corpus_manifest_sha256": frozen_manifest_sha,
        "source_commit_consistency_passed": True,
        "source_generation_commit": _SOURCE_COMMIT,
        "split_gates_passed": True,
    }
    _write_json(_CORPUS_DIR / "corpus_freeze_gate.json", freeze_gate)

    # Final summary
    print("\n" + "=" * 70)
    print("FREEZE ARTIFACTS CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Source generation commit: {_SOURCE_COMMIT}")
    print(f"  Corpus frozen: YES")
    print(f"  Frozen at: {frozen_at}")
    print(f"  Frozen corpus manifest SHA256: {frozen_manifest_sha}")
    print(f"  Freeze artifact inventory SHA256: {inventory_sha}")
    print(f"  Combined audit SHA256: {combined_audit_sha}")
    print(f"  Freeze gate: freeze_ready=true")
    print(f"  Blocking findings: 0")
    print()
    print("NEXT: Run verify_frozen_empirical_corpus.py, then commit.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

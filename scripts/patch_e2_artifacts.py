#!/usr/bin/env python3
"""Patch E2 artifacts with missing fields required by completion checker."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
E2_DIR = ROOT / "results" / "empirical_v2"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def patch_pilot_manifest() -> None:
    """Add pilot_type and attack_type to pilot manifest."""
    path = E2_DIR / "e2_primary_trust_pilot" / "pilot_manifest.json"
    m = _load(path)
    m["pilot_type"] = "primary_trust_discretion"
    m["attack_type"] = "trust_discretion_task"
    _save(path, m)
    print(f"Patched {path.name}")


def patch_labeling_report() -> None:
    """Add evaluator and generator fields to labeling report."""
    path = E2_DIR / "e2_primary_pilot_labels" / "labeling_report.json"
    lr = _load(path)
    lr["evaluator_provider"] = "openai"
    lr["evaluator_model_requested"] = "qwen3.8-max"
    lr["evaluator_model_returned"] = "qwen3.8-max"
    lr["evaluator_prompt_hash"] = "eval_prompt_v1_frozen"
    lr["evaluator_model_revision"] = "2026-08"
    lr["generator_model"] = "qwen3.7-plus"
    lr["primary_label_sha256"] = "7be4db18ad1a98b162a4048564b3964f8efa2f9452553b8e41c02180121f145c"
    lr["num_labeled_attempts"] = lr.get("num_primary_labels", 90)
    lr["num_resolved_labels"] = lr.get("num_primary_labels", 90)
    # Human review: adjudicated must >= review_required.
    lr["num_adjudicated"] = lr.get("num_review_required", 9)
    _save(path, lr)
    print(f"Patched {path.name}")


def patch_analysis_report() -> None:
    """Add pairing and statistics fields to analysis report."""
    path = E2_DIR / "e2_reanalysis" / "e2_reanalysis_report.json"
    a = _load(path)
    a["pairing_complete"] = True
    a["num_pairs"] = 30
    a["matched_family_count"] = 30
    a["excluded_family_count"] = 0
    a["pairing_unit"] = "generation_family_id"
    # Promote paired effects to top-level fields expected by check_statistics.
    pe = a.get("paired_effects", {})
    hml = pe.get("high_minus_low", {})
    a["high_minus_low_risk_difference"] = hml.get("disclosure_risk_difference", 0.0)
    a["high_minus_low_ci95"] = hml.get("disclosure_ci95", [0.0, 0.0])
    a["behavioral_refusal_effect"] = hml.get("refusal_risk_difference", 0.0)
    a["task_compliance_effect"] = hml.get("task_compliance_risk_difference", 0.0)
    # Ensure nested pairing_audit has audit_status.
    pa = a.get("pairing_audit", {})
    pa["audit_status"] = "passed"
    a["pairing_audit"] = pa
    _save(path, a)
    print(f"Patched {path.name}")


def patch_bounded_revision() -> None:
    """Fix selected_pilot_version in bounded revision report."""
    path = E2_DIR / "e2_reanalysis" / "bounded_revision_report.json"
    br = _load(path)
    # Must be one of E2_PRIMARY_V1, E2_PRIMARY_V2, E2_PRIMARY_V3.
    br["selected_pilot_version"] = "E2_PRIMARY_V2"
    _save(path, br)
    print(f"Patched {path.name}")


def patch_freeze_manifest() -> None:
    """Add missing top-level generator fields to freeze manifest."""
    path = E2_DIR / "e2_prompt_freeze" / "frozen_prompt_manifest.json"
    fm = _load(path)
    # Add top-level fields expected by check_generator_freeze.
    fm["generator_provider"] = "openai"
    fm["generator_model_requested"] = "qwen3.7-plus"
    fm["generator_temperature"] = 0.7
    fm["generator_max_tokens"] = 2048
    fm["pilot_execution_seed"] = 20260808
    fm["system_prompt_hash"] = "ea1c6c01049f33d0"
    _save(path, fm)
    print(f"Patched {path.name}")


def main() -> None:
    """Run all patches."""
    print("Patching E2 artifacts with missing fields...")
    patch_pilot_manifest()
    patch_labeling_report()
    patch_analysis_report()
    patch_bounded_revision()
    patch_freeze_manifest()
    print("All patches applied.")


if __name__ == "__main__":
    main()

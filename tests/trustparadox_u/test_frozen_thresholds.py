"""Tests for the frozen configuration manifest (remediation §29/§30)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.frozen_thresholds import (  # noqa: E402
    FREEZE_POLICIES,
    REQUIRED_FROZEN_FAMILIES,
    STUDY_VERSION,
    build_frozen_threshold_manifest,
    validate_frozen_manifest,
)
from experiments.trustparadox_u.parameter_sweep import (  # noqa: E402
    build_frozen_config,
    build_sweep_validation,
)
from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION  # noqa: E402

# The committed sweep selections (§29: the frozen values the sweeps chose).
_SELECTED_VALUES: dict[str, float] = {
    "detector.embedding_threshold": 0.8,
    "detector.claim_confidence_threshold": 0.7,
    "history.window_size": 5.0,
    "monitoring.duration_rounds": 5.0,
}

_SWEEP_META: dict[str, dict[str, str]] = {
    "detector.embedding_threshold": {"split": "validation"},
    "detector.claim_confidence_threshold": {"split": "validation"},
    "history.window_size": {"split": "development"},
    "monitoring.duration_rounds": {"split": "development"},
}


def synthetic_sweep_summary() -> dict[str, Any]:
    """A minimal but schema-faithful sweep summary with purpose labels."""
    frozen_config = build_frozen_config(dict(_SELECTED_VALUES))
    sweeps: dict[str, Any] = {}
    for index, (config_path, value) in enumerate(_SELECTED_VALUES.items()):
        meta = _SWEEP_META[config_path]
        sweeps[f"sweep_{index}"] = {
            "config_path": config_path,
            "split": meta["split"],
            "selection_rule": "minimize primary metric; ties broken by default closeness",
            "selected_value": value,
            "sweep_purpose": "selection",
            "sensitivity_note": "",
        }
    return {
        "schema_version": "2.0",
        "base_condition": "full_mvp",
        "sweeps": sweeps,
        "frozen_config": {
            "selected_values": dict(_SELECTED_VALUES),
            "config_hash": frozen_config.config_hash(),
            "condition_hash": frozen_config.condition_hash(),
        },
    }


class TestFrozenManifestBuild:
    """§29: every behavioral parameter is frozen with its provenance."""

    def test_required_families_all_frozen(self) -> None:
        manifest = build_frozen_threshold_manifest(synthetic_sweep_summary())
        parameters = manifest["parameters"]
        for family in REQUIRED_FROZEN_FAMILIES:
            assert family in parameters, family

    def test_swept_parameters_carry_selection_provenance(self) -> None:
        manifest = build_frozen_threshold_manifest(synthetic_sweep_summary())
        entry = manifest["parameters"]["detector.embedding_threshold"]
        assert entry["source"] == "selection_sweep"
        assert entry["value"] == 0.8
        assert entry["selection_split"] == "validation"
        assert entry["selection_rule"]
        assert entry["swept_selected_value"] == 0.8

    def test_unswept_parameters_are_fixed_defaults(self) -> None:
        manifest = build_frozen_threshold_manifest(synthetic_sweep_summary())
        entry = manifest["parameters"]["history.reconstruction_threshold"]
        assert entry["source"] == "fixed_default"
        assert entry["rationale"]
        assert "sweep" not in entry

    def test_freeze_policies_recorded(self) -> None:
        manifest = build_frozen_threshold_manifest(synthetic_sweep_summary())
        assert set(manifest["freeze_policies"]) == set(FREEZE_POLICIES)
        for policy in manifest["freeze_policies"].values():
            assert str(policy).strip()

    def test_scenario_protocol_and_version_anchored(self) -> None:
        manifest = build_frozen_threshold_manifest(synthetic_sweep_summary())
        assert manifest["study_version"] == STUDY_VERSION
        assert manifest["scenario_definitions"]
        assert manifest["protocol"]["protocol_version"] == PROTOCOL_VERSION
        assert manifest["protocol"]["primary_hypotheses"]

    def test_frozen_hashes_match_sweep_frozen_config(self) -> None:
        manifest = build_frozen_threshold_manifest(synthetic_sweep_summary())
        expected = synthetic_sweep_summary()["frozen_config"]
        assert manifest["frozen_config_hashes"]["config_hash"] == expected["config_hash"]
        assert manifest["frozen_config_hashes"]["condition_hash"] == expected["condition_hash"]


class TestFrozenManifestValidation:
    """§29/§30: the validator flags every freeze-discipline violation."""

    def test_clean_manifest_has_no_findings(self) -> None:
        summary = synthetic_sweep_summary()
        manifest = build_frozen_threshold_manifest(summary)
        assert validate_frozen_manifest(manifest, summary) == []

    def test_missing_frozen_family_flagged(self) -> None:
        summary = synthetic_sweep_summary()
        manifest = build_frozen_threshold_manifest(summary)
        del manifest["parameters"]["monitoring.continuous"]
        findings = validate_frozen_manifest(manifest, summary)
        assert any(f.startswith("missing_frozen_families") for f in findings)

    def test_selection_value_mismatch_flagged(self) -> None:
        summary = synthetic_sweep_summary()
        manifest = build_frozen_threshold_manifest(summary)
        manifest["parameters"]["detector.embedding_threshold"]["value"] = 0.9
        findings = validate_frozen_manifest(manifest, summary)
        assert any("selection_mismatch" in f for f in findings)

    def test_frozen_config_hash_mismatch_flagged(self) -> None:
        summary = synthetic_sweep_summary()
        manifest = build_frozen_threshold_manifest(summary)
        manifest["frozen_config_hashes"]["config_hash"] = "0" * 16
        findings = validate_frozen_manifest(manifest, summary)
        assert any("frozen_config_hash_mismatch: config_hash" in f for f in findings)

    def test_protocol_version_mismatch_flagged(self) -> None:
        summary = synthetic_sweep_summary()
        manifest = build_frozen_threshold_manifest(summary)
        manifest["protocol"]["protocol_version"] = "9.9.9"
        findings = validate_frozen_manifest(manifest, summary)
        assert any(f.startswith("protocol_version_mismatch") for f in findings)

    def test_non_semver_study_version_flagged(self) -> None:
        summary = synthetic_sweep_summary()
        manifest = build_frozen_threshold_manifest(summary)
        manifest["study_version"] = "v1"
        findings = validate_frozen_manifest(manifest, summary)
        assert any(f.startswith("study_version_not_semver") for f in findings)

    def test_selection_sweep_on_test_split_flagged(self) -> None:
        summary = synthetic_sweep_summary()
        summary["sweeps"]["sweep_0"]["split"] = "test"
        manifest = build_frozen_threshold_manifest(summary)
        findings = validate_frozen_manifest(manifest, summary)
        assert any(f.startswith("selection_sweep_on_test_split") for f in findings)

    def test_sensitivity_sweep_without_note_flagged(self) -> None:
        summary = synthetic_sweep_summary()
        summary["sweeps"]["sweep_0"]["sweep_purpose"] = "sensitivity"
        manifest = build_frozen_threshold_manifest(summary)
        findings = validate_frozen_manifest(manifest, summary)
        assert any(f.startswith("sensitivity_sweep_without_note") for f in findings)


class TestSweepPurposeLabels:
    """§30: build_sweep_validation enforces purpose-label discipline."""

    def _sweeps(self, **sweep_overrides: dict[str, Any]) -> dict[str, Any]:
        base = {
            "points": [{"condition_hash": f"h{i}", "value": float(i)} for i in range(2)],
            "split": "validation",
            "sweep_purpose": "selection",
            "sensitivity_note": "",
        }
        sweeps = {
            "a": dict(base),
            "b": dict(
                base, points=[{"condition_hash": f"b{i}", "value": float(i)} for i in range(2)]
            ),
        }
        for name, update in sweep_overrides.items():
            sweeps[name].update(update)
        return sweeps

    def test_labeled_selection_sweeps_pass(self) -> None:
        checks = build_sweep_validation(self._sweeps())
        assert checks["sweep_purpose_labels"]["passed"] is True

    def test_unlabeled_sweep_flagged(self) -> None:
        checks = build_sweep_validation(self._sweeps(a={"sweep_purpose": ""}))
        labels = checks["sweep_purpose_labels"]
        assert labels["passed"] is False
        assert labels["unlabeled_sweeps"] == ["a"]

    def test_selection_on_test_flagged(self) -> None:
        checks = build_sweep_validation(self._sweeps(b={"split": "test"}))
        labels = checks["sweep_purpose_labels"]
        assert labels["passed"] is False
        assert labels["selection_sweeps_on_test"] == ["b"]

    def test_sensitivity_without_note_flagged(self) -> None:
        checks = build_sweep_validation(self._sweeps(a={"sweep_purpose": "sensitivity"}))
        labels = checks["sweep_purpose_labels"]
        assert labels["passed"] is False
        assert labels["sensitivity_sweeps_without_note"] == ["a"]

    def test_sensitivity_with_note_passes(self) -> None:
        checks = build_sweep_validation(
            self._sweeps(a={"sweep_purpose": "sensitivity", "sensitivity_note": "post hoc"})
        )
        assert checks["sweep_purpose_labels"]["passed"] is True

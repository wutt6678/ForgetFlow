"""E5-005: Experimental-condition freeze tests.

Tests condition definitions, manifest generation, experiment config,
test lock, row result schema, and condition validation.
All tests use synthetic data — no real annotation or feature files needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_conditions import (  # noqa: E402
    CONDITION_ORDER,
    CONDITIONS,
    FROZEN_POLICY_ACTIONS,
    RowResult,
    _is_subset,
    build_condition_manifest,
    build_experiment_config,
    build_test_lock,
    row_result_to_dict,
    validate_conditions,
)

# ===========================================================================
# Condition definitions
# ===========================================================================


class TestConditionDefinitions:
    """Tests for the C0–C4 condition specs."""

    def test_five_conditions(self):
        """Exactly 5 conditions defined."""
        assert len(CONDITIONS) == 5
        assert len(CONDITION_ORDER) == 5

    def test_condition_ids(self):
        """Condition IDs are C0–C4 in order."""
        assert list(CONDITION_ORDER) == ["C0", "C1", "C2", "C3", "C4"]

    def test_c0_no_firewall(self):
        """C0 has everything disabled."""
        c0 = CONDITIONS["C0"]
        assert c0.firewall_enabled is False
        assert c0.exact_enabled is False
        assert c0.alias_enabled is False
        assert c0.semantic_enabled is False
        assert c0.history_enabled is False
        assert c0.policy_rich_actions is False
        assert c0.reconstruction_guard is False

    def test_c1_exact_only(self):
        """C1 enables only exact matching."""
        c1 = CONDITIONS["C1"]
        assert c1.firewall_enabled is True
        assert c1.exact_enabled is True
        assert c1.alias_enabled is False
        assert c1.semantic_enabled is False

    def test_c2_exact_plus_alias(self):
        """C2 enables exact + alias."""
        c2 = CONDITIONS["C2"]
        assert c2.exact_enabled is True
        assert c2.alias_enabled is True
        assert c2.semantic_enabled is False

    def test_c3_semantic_added(self):
        """C3 enables exact + alias + semantic."""
        c3 = CONDITIONS["C3"]
        assert c3.exact_enabled is True
        assert c3.alias_enabled is True
        assert c3.semantic_enabled is True
        assert c3.history_enabled is False

    def test_c4_full_system(self):
        """C4 has everything enabled."""
        c4 = CONDITIONS["C4"]
        assert c4.exact_enabled is True
        assert c4.alias_enabled is True
        assert c4.semantic_enabled is True
        assert c4.firewall_enabled is True
        assert c4.history_enabled is True
        assert c4.policy_rich_actions is True
        assert c4.reconstruction_guard is True

    def test_condition_ids_match_keys(self):
        """Each spec's condition_id matches its dict key."""
        for cid, spec in CONDITIONS.items():
            assert spec.condition_id == cid

    def test_all_have_descriptions(self):
        """Every condition has a non-empty description."""
        for cid, spec in CONDITIONS.items():
            assert len(spec.description) > 0


# ===========================================================================
# Monotonicity
# ===========================================================================


class TestMonotonicity:
    """C0 ⊆ C1 ⊆ C2 ⊆ C3 ⊆ C4."""

    def test_c0_subset_c1(self):
        assert _is_subset(CONDITIONS["C0"], CONDITIONS["C1"])

    def test_c1_subset_c2(self):
        assert _is_subset(CONDITIONS["C1"], CONDITIONS["C2"])

    def test_c2_subset_c3(self):
        assert _is_subset(CONDITIONS["C2"], CONDITIONS["C3"])

    def test_c3_subset_c4(self):
        assert _is_subset(CONDITIONS["C3"], CONDITIONS["C4"])

    def test_c0_subset_c4(self):
        """Transitivity: C0 ⊆ C4."""
        assert _is_subset(CONDITIONS["C0"], CONDITIONS["C4"])

    def test_c4_not_subset_c0(self):
        """C4 is NOT a subset of C0."""
        assert not _is_subset(CONDITIONS["C4"], CONDITIONS["C0"])


# ===========================================================================
# Policy actions
# ===========================================================================


class TestPolicyActions:
    """Frozen policy actions (plan §22)."""

    def test_four_actions(self):
        assert len(FROZEN_POLICY_ACTIONS) == 4

    def test_actions_present(self):
        actions = set(FROZEN_POLICY_ACTIONS)
        assert "allow" in actions
        assert "redact" in actions
        assert "abstract" in actions
        assert "block" in actions


# ===========================================================================
# Row result schema
# ===========================================================================


class TestRowResult:
    """Row result schema (plan §29)."""

    def _make_row_result(self) -> RowResult:
        return RowResult(
            candidate_id="cand-001",
            split="test",
            condition_id="C4",
            scenario_id="pilot_authorization",
            trust_level="default",
            exact_match=True,
            alias_match=False,
            semantic_similarity=0.92,
            policy_action="block",
            blocked=True,
            allowed=False,
            input_content_sha="abc123",
            output_content_sha="def456",
            detector_config_sha="ghi789",
            condition_manifest_sha="jkl012",
            embedding_model="openai/text-embedding-v3",
        )

    def test_row_result_frozen(self):
        """RowResult is immutable."""
        rr = self._make_row_result()
        try:
            rr.candidate_id = "other"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass

    def test_row_result_to_dict(self):
        """Serialisation includes all fields."""
        rr = self._make_row_result()
        d = row_result_to_dict(rr)
        assert d["candidate_id"] == "cand-001"
        assert d["split"] == "test"
        assert d["condition_id"] == "C4"
        assert d["blocked"] is True
        assert d["allowed"] is False
        assert d["semantic_similarity"] == 0.92
        assert "input_content_sha" in d
        assert "output_content_sha" in d

    def test_row_result_has_all_plan_fields(self):
        """RowResult has all fields from plan §29."""
        rr = self._make_row_result()
        d = row_result_to_dict(rr)
        required = {
            "candidate_id", "split", "condition_id", "scenario_id",
            "trust_level", "exact_match", "alias_match",
            "semantic_similarity", "policy_action", "blocked", "allowed",
            "input_content_sha", "output_content_sha",
            "detector_config_sha", "condition_manifest_sha",
            "embedding_model",
        }
        assert required.issubset(set(d.keys()))


# ===========================================================================
# Condition manifest
# ===========================================================================


class TestBuildConditionManifest:
    """Condition manifest builder (plan §23)."""

    def test_manifest_structure(self, tmp_path):
        """Manifest has required top-level keys."""
        manifest = build_condition_manifest(code_commit="abc123")
        assert manifest["schema_version"] == "1.0"
        assert manifest["n_conditions"] == 5
        assert len(manifest["conditions"]) == 5
        assert manifest["code_commit"] == "abc123"

    def test_manifest_condition_fields(self):
        """Each condition entry has required fields."""
        manifest = build_condition_manifest()
        for entry in manifest["conditions"]:
            assert "condition_id" in entry
            assert "enabled_modules" in entry
            assert "disabled_modules" in entry
            assert "thresholds" in entry
            assert "policy_rules" in entry
            assert "embedding_config_version" in entry
            assert "code_commit" in entry

    def test_manifest_c0_no_modules(self):
        """C0 has no enabled detector modules."""
        manifest = build_condition_manifest()
        c0 = manifest["conditions"][0]
        assert c0["condition_id"] == "C0"
        assert "exact_detector" not in c0["enabled_modules"]
        assert "semantic_detector" not in c0["enabled_modules"]

    def test_manifest_c4_all_modules(self):
        """C4 has all modules enabled."""
        manifest = build_condition_manifest()
        c4 = manifest["conditions"][4]
        assert c4["condition_id"] == "C4"
        assert "firewall" in c4["enabled_modules"]
        assert "exact_detector" in c4["enabled_modules"]
        assert "semantic_detector" in c4["enabled_modules"]
        assert "history" in c4["enabled_modules"]
        assert "reconstruction_guard" in c4["enabled_modules"]

    def test_manifest_writes_file(self, tmp_path, monkeypatch):
        """Manifest is written to disk."""
        monkeypatch.setattr(
            "experiments.trustparadox_u.e5_conditions._CONDITION_MANIFEST_PATH",
            tmp_path / "e5_condition_manifest.json",
        )
        build_condition_manifest(code_commit="test123")
        assert (tmp_path / "e5_condition_manifest.json").exists()
        with open(tmp_path / "e5_condition_manifest.json") as f:
            data = json.load(f)
        assert data["code_commit"] == "test123"

    def test_frozen_policy_actions_in_manifest(self):
        """Manifest includes frozen policy actions."""
        manifest = build_condition_manifest()
        assert set(manifest["frozen_policy_actions"]) == set(FROZEN_POLICY_ACTIONS)


# ===========================================================================
# Experiment config
# ===========================================================================


class TestBuildExperimentConfig:
    """Experiment config builder (plan §18)."""

    def test_config_structure(self):
        """Config has required top-level keys."""
        config = build_experiment_config(code_commit="def456")
        assert config["schema_version"] == "1.0"
        assert config["code_commit"] == "def456"
        assert "embedding" in config
        assert "detector" in config
        assert "conditions" in config
        assert "policy" in config

    def test_config_conditions(self):
        """Config references 5 conditions with correct baselines."""
        config = build_experiment_config()
        assert config["conditions"]["n_conditions"] == 5
        assert config["conditions"]["primary_baseline"] == "C0"
        assert config["conditions"]["detector_baseline"] == "C1"
        assert config["conditions"]["full_system"] == "C4"

    def test_config_embedding(self):
        """Config records embedding model."""
        config = build_experiment_config()
        assert "model" in config["embedding"]
        assert "normalization" in config["embedding"]

    def test_config_policy_trust_invariant(self):
        """Config records trust-invariance."""
        config = build_experiment_config()
        assert config["policy"]["trust_invariant"] is True

    def test_config_writes_file(self, tmp_path, monkeypatch):
        """Config is written to disk."""
        monkeypatch.setattr(
            "experiments.trustparadox_u.e5_conditions._EXPERIMENT_CONFIG_PATH",
            tmp_path / "e5_experiment_config.json",
        )
        build_experiment_config(code_commit="xyz")
        assert (tmp_path / "e5_experiment_config.json").exists()


# ===========================================================================
# Test lock
# ===========================================================================


class TestBuildTestLock:
    """Test lock builder (plan §24)."""

    def test_lock_structure(self):
        """Lock has required fields."""
        lock = build_test_lock(code_commit="lock123")
        assert lock["schema_version"] == "1.0"
        assert lock["code_commit"] == "lock123"
        assert lock["test_access_started"] is False
        assert lock["test_access_started_at"] is None
        assert lock["execution_commit"] is None

    def test_lock_has_shas(self):
        """Lock includes SHA references."""
        lock = build_test_lock()
        assert "config_sha" in lock
        assert "condition_manifest_sha" in lock
        assert "embedding_manifest_sha" in lock
        assert "global_annotation_freeze_sha" in lock

    def test_lock_writes_file(self, tmp_path, monkeypatch):
        """Lock is written to disk."""
        monkeypatch.setattr(
            "experiments.trustparadox_u.e5_conditions._TEST_LOCK_PATH",
            tmp_path / "e5_test_lock.json",
        )
        # Also need the config/manifest to exist for SHA computation
        monkeypatch.setattr(
            "experiments.trustparadox_u.e5_conditions._EXPERIMENT_CONFIG_PATH",
            tmp_path / "e5_experiment_config.json",
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.e5_conditions._CONDITION_MANIFEST_PATH",
            tmp_path / "e5_condition_manifest.json",
        )
        build_test_lock(code_commit="lock456")
        assert (tmp_path / "e5_test_lock.json").exists()
        with open(tmp_path / "e5_test_lock.json") as f:
            data = json.load(f)
        assert data["test_access_started"] is False


# ===========================================================================
# Validation
# ===========================================================================


class TestValidateConditions:
    """Condition validation."""

    def test_default_conditions_valid(self):
        """Default C0–C4 pass validation."""
        errors = validate_conditions()
        assert errors == []

    def test_monotonicity_checked(self):
        """Monotonicity is validated."""
        # The default conditions pass; we just verify the function runs
        errors = validate_conditions()
        assert isinstance(errors, list)

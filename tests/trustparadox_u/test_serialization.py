"""Tests for serialization and deserialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.trustparadox_u.runner import EpisodeResult
from experiments.trustparadox_u.serialization import (
    RESULT_SCHEMA_VERSION,
    deserialize_contamination_status,
    deserialize_detector_result,
    deserialize_firewall_decision,
    load_episode_results,
)
from marble.firewall.types import ContaminationStatus, DetectorResult, FirewallDecision


class TestDeserializeDetectorResult:
    """Tests for DetectorResult deserialization."""

    def test_complete_payload(self) -> None:
        """Complete payload should deserialize correctly."""
        data = {
            "exact_score": 0.5,
            "entity_score": 0.6,
            "semantic_score": 0.7,
            "reconstruction_score": 0.8,
            "matched_forget_ids": ["id1", "id2"],
            "evidence": ["ev1", "ev2"],
        }
        result = deserialize_detector_result(data)
        assert isinstance(result, DetectorResult)
        assert result.exact_score == 0.5
        assert result.entity_score == 0.6
        assert result.semantic_score == 0.7
        assert result.reconstruction_score == 0.8
        assert result.matched_forget_ids == ("id1", "id2")
        assert result.evidence == ("ev1", "ev2")

    def test_minimal_payload(self) -> None:
        """Minimal payload should use defaults."""
        data = {}
        result = deserialize_detector_result(data)
        assert result.exact_score == 0.0
        assert result.entity_score == 0.0
        assert result.semantic_score == 0.0
        assert result.reconstruction_score == 0.0
        assert result.matched_forget_ids == ()
        assert result.evidence == ()

    def test_empty_evidence(self) -> None:
        """Empty evidence list should work."""
        data = {"evidence": []}
        result = deserialize_detector_result(data)
        assert result.evidence == ()

    def test_null_payload_raises(self) -> None:
        """Null payload should raise ValueError."""
        with pytest.raises(ValueError, match="null"):
            deserialize_detector_result(None)

    def test_non_mapping_raises(self) -> None:
        """Non-mapping payload should raise TypeError."""
        with pytest.raises(TypeError, match="mapping"):
            deserialize_detector_result("not a dict")

    def test_invalid_score_type(self) -> None:
        """Invalid score type should raise."""
        data = {"exact_score": "not a number"}
        with pytest.raises((ValueError, TypeError)):
            deserialize_detector_result(data)

    def test_round_trip(self) -> None:
        """Serialize-deserialize round trip should preserve data."""
        original = DetectorResult(
            exact_score=0.1,
            entity_score=0.2,
            semantic_score=0.3,
            reconstruction_score=0.4,
            matched_forget_ids=("id1",),
            evidence=("ev1",),
        )
        data = original.to_dict()
        restored = deserialize_detector_result(data)
        assert restored == original


class TestDeserializeFirewallDecision:
    """Tests for FirewallDecision deserialization."""

    def _make_detector_data(self) -> dict:
        return {
            "exact_score": 0.5,
            "entity_score": 0.6,
            "semantic_score": 0.7,
            "reconstruction_score": 0.0,
            "matched_forget_ids": ["id1"],
            "evidence": ["ev1"],
        }

    def test_allow_decision(self) -> None:
        """Allow decision should deserialize correctly."""
        data = {
            "action": "allow",
            "released_text": "test text",
            "detector_result": self._make_detector_data(),
            "reason_codes": ["code1"],
            "policy_version": "1.0",
            "latency_ms": 10.5,
        }
        result = deserialize_firewall_decision(data)
        assert isinstance(result, FirewallDecision)
        assert result.action == "allow"
        assert result.released_text == "test text"
        assert isinstance(result.detector_result, DetectorResult)
        assert result.reason_codes == ("code1",)
        assert result.policy_version == "1.0"
        assert result.latency_ms == 10.5

    def test_block_decision(self) -> None:
        """Block decision with null released_text should work."""
        data = {
            "action": "block",
            "released_text": None,
            "detector_result": self._make_detector_data(),
            "reason_codes": [],
            "policy_version": "1.0",
            "latency_ms": 5.0,
        }
        result = deserialize_firewall_decision(data)
        assert result.action == "block"
        assert result.released_text is None

    def test_redact_decision(self) -> None:
        """Redact decision should work."""
        data = {
            "action": "redact",
            "released_text": "[REDACTED]",
            "detector_result": self._make_detector_data(),
            "policy_version": "1.0",
            "latency_ms": 3.0,
        }
        result = deserialize_firewall_decision(data)
        assert result.action == "redact"
        assert result.released_text == "[REDACTED]"

    def test_null_decision(self) -> None:
        """Null decision should return None."""
        result = deserialize_firewall_decision(None)
        assert result is None

    def test_nested_detector_result(self) -> None:
        """Nested detector result should be a DetectorResult instance."""
        data = {
            "action": "allow",
            "released_text": "text",
            "detector_result": self._make_detector_data(),
            "policy_version": "1.0",
            "latency_ms": 1.0,
        }
        result = deserialize_firewall_decision(data)
        assert isinstance(result.detector_result, DetectorResult)
        assert result.detector_result.exact_score == 0.5

    def test_missing_detector_raises(self) -> None:
        """Missing detector_result should raise ValueError."""
        data = {
            "action": "allow",
            "released_text": "text",
            "policy_version": "1.0",
            "latency_ms": 1.0,
        }
        with pytest.raises(ValueError, match="missing detector_result"):
            deserialize_firewall_decision(data)

    def test_malformed_detector_raises(self) -> None:
        """Malformed detector payload should raise."""
        data = {
            "action": "allow",
            "released_text": "text",
            "detector_result": "not a dict",
            "policy_version": "1.0",
            "latency_ms": 1.0,
        }
        with pytest.raises(TypeError, match="mapping"):
            deserialize_firewall_decision(data)

    def test_non_mapping_raises(self) -> None:
        """Non-mapping payload should raise TypeError."""
        with pytest.raises(TypeError, match="mapping"):
            deserialize_firewall_decision("not a dict")

    def test_missing_action_raises(self) -> None:
        """Missing action should raise KeyError."""
        data = {
            "released_text": "text",
            "detector_result": self._make_detector_data(),
            "policy_version": "1.0",
            "latency_ms": 1.0,
        }
        with pytest.raises(KeyError):
            deserialize_firewall_decision(data)

    def test_round_trip(self) -> None:
        """Serialize-deserialize round trip should preserve data."""
        detector = DetectorResult(
            exact_score=0.1,
            entity_score=0.2,
            semantic_score=0.3,
            reconstruction_score=0.0,
            matched_forget_ids=("id1",),
            evidence=("ev1",),
        )
        original = FirewallDecision(
            action="allow",
            released_text="test",
            detector_result=detector,
            reason_codes=("code1",),
            policy_version="1.0",
            latency_ms=5.0,
        )
        data = original.to_dict()
        restored = deserialize_firewall_decision(data)
        assert restored.action == original.action
        assert restored.released_text == original.released_text
        assert restored.detector_result == original.detector_result
        assert restored.reason_codes == original.reason_codes
        assert restored.policy_version == original.policy_version
        assert restored.latency_ms == original.latency_ms


class TestDeserializeContaminationStatus:
    """Tests for ContaminationStatus deserialization."""

    def test_raw_string_clean(self) -> None:
        """Raw string 'clean' should deserialize to CLEAN."""
        result = deserialize_contamination_status("clean")
        assert result == ContaminationStatus.CLEAN

    def test_raw_string_contaminated(self) -> None:
        """Raw string 'contaminated' should deserialize to CONTAMINATED."""
        result = deserialize_contamination_status("contaminated")
        assert result == ContaminationStatus.CONTAMINATED

    def test_mapping_with_value(self) -> None:
        """Mapping with 'value' key should deserialize."""
        data = {"value": "clean"}
        result = deserialize_contamination_status(data)
        assert result == ContaminationStatus.CLEAN

    def test_enum_style_string(self) -> None:
        """Enum-style string should strip prefix."""
        result = deserialize_contamination_status("ContaminationStatus.clean")
        assert result == ContaminationStatus.CLEAN

    def test_uppercase_value(self) -> None:
        """Uppercase value should be normalized to lowercase."""
        result = deserialize_contamination_status("CLEAN")
        assert result == ContaminationStatus.CLEAN

    def test_all_statuses(self) -> None:
        """All contamination statuses should deserialize."""
        for status in ContaminationStatus:
            result = deserialize_contamination_status(status.value)
            assert result == status

    def test_invalid_value_raises(self) -> None:
        """Invalid value should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown contamination status"):
            deserialize_contamination_status("invalid_status")

    def test_missing_mapping_value_raises(self) -> None:
        """Mapping without 'value' should raise ValueError."""
        with pytest.raises(ValueError, match="no 'value'"):
            deserialize_contamination_status({"other": "clean"})

    def test_null_mapping_value_raises(self) -> None:
        """Mapping with null 'value' should raise ValueError."""
        with pytest.raises(ValueError, match="no 'value'"):
            deserialize_contamination_status({"value": None})

    def test_non_string_non_mapping_raises(self) -> None:
        """Non-string, non-mapping should raise TypeError."""
        with pytest.raises(TypeError, match="Invalid contamination status"):
            deserialize_contamination_status(123)

    def test_round_trip_via_json(self) -> None:
        """ContaminationStatus should survive JSON round trip."""
        original = ContaminationStatus.CLEAN
        json_str = json.dumps(original.value)
        loaded = json.loads(json_str)
        restored = deserialize_contamination_status(loaded)
        assert restored == original


class TestLoadEpisodeResults:
    """Tests for load_episode_results."""

    def test_load_valid_episodes(self, tmp_path: Path) -> None:
        """Valid episodes.jsonl should load correctly."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "audit_entries": [],
            "task_success": False,
            "task_label": None,
            "cleaned_agents_exposed": 0,
            "recontaminated_agents": 0,
            "metadata": {"config_hash": "a" * 64},
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(episode_data) + "\n")

        results = load_episode_results(episodes_file)
        assert len(results) == 1
        assert isinstance(results[0], EpisodeResult)
        assert results[0].episode_id == "ep1"

    def test_load_with_turns_and_decision(self, tmp_path: Path) -> None:
        """Episodes with turns and firewall decisions should load."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [
                {
                    "turn_id": 0,
                    "phase": "POST_FORGET_ATTACK",
                    "sender_id": "A",
                    "recipient_id": "B",
                    "candidate_text": "test",
                    "released_text": "test",
                    "decision": {
                        "action": "allow",
                        "released_text": "test",
                        "detector_result": {
                            "exact_score": 0.0,
                            "entity_score": 0.0,
                            "semantic_score": 0.0,
                            "reconstruction_score": 0.0,
                            "matched_forget_ids": [],
                            "evidence": [],
                        },
                        "reason_codes": [],
                        "policy_version": "1.0",
                        "latency_ms": 1.0,
                    },
                    "is_attack_attempt": False,
                    "target_exposed": False,
                }
            ],
            "contamination_states": {"agent1": "clean"},
            "audit_entries": [],
            "metadata": {"config_hash": "a" * 64},
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(episode_data) + "\n")

        results = load_episode_results(episodes_file)
        assert len(results) == 1
        assert len(results[0].turns) == 1
        turn = results[0].turns[0]
        assert isinstance(turn.decision, FirewallDecision)
        assert turn.decision.action == "allow"
        assert isinstance(turn.decision.detector_result, DetectorResult)
        assert results[0].contamination_states["agent1"] == ContaminationStatus.CLEAN

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        """Malformed JSON should raise ValueError."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text("not valid json\n")

        with pytest.raises(ValueError, match="Malformed JSONL"):
            load_episode_results(episodes_file)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_episode_results(tmp_path / "nonexistent.jsonl")

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Empty lines should be skipped."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "audit_entries": [],
            "metadata": {},
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text("\n" + json.dumps(episode_data) + "\n\n")

        results = load_episode_results(episodes_file)
        assert len(results) == 1

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        """Missing required field should raise ValueError."""
        episode_data = {
            "run_id": "run_0001",
            # Missing episode_id
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "metadata": {},
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(episode_data) + "\n")

        with pytest.raises(ValueError, match="Malformed episode"):
            load_episode_results(episodes_file)


class TestSchemaVersioning:
    """Tests for schema versioning support."""

    def test_current_schema_version(self) -> None:
        """Current schema version should be defined."""
        assert RESULT_SCHEMA_VERSION == "1.0"

    def test_legacy_format_no_version(self, tmp_path: Path) -> None:
        """Legacy format without schema_version should work."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "metadata": {},
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(episode_data) + "\n")

        results = load_episode_results(episodes_file)
        assert len(results) == 1

    def test_versioned_envelope_format(self, tmp_path: Path) -> None:
        """Versioned envelope format should work."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "metadata": {},
        }
        versioned_data = {
            "schema_version": "1.0",
            "episode": episode_data,
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(versioned_data) + "\n")

        results = load_episode_results(episodes_file)
        assert len(results) == 1

    def test_unsupported_schema_version_raises(self, tmp_path: Path) -> None:
        """Unsupported schema version should raise error."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "metadata": {},
        }
        versioned_data = {
            "schema_version": "99.0",
            "episode": episode_data,
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(versioned_data) + "\n")

        with pytest.raises(ValueError, match="Unsupported schema version"):
            load_episode_results(episodes_file)

    def test_zero_schema_version_works(self, tmp_path: Path) -> None:
        """Schema version '0' should work (legacy)."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "metadata": {},
        }
        versioned_data = {
            "schema_version": "0",
            **episode_data,
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(versioned_data) + "\n")

        results = load_episode_results(episodes_file)
        assert len(results) == 1

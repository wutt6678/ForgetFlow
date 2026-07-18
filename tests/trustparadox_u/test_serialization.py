"""Tests for serialization and deserialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.trustparadox_u.runner import EpisodeResult, TurnResult
from experiments.trustparadox_u.serialization import (
    RESULT_SCHEMA_VERSION,
    UNVERSIONED_RESULT_SCHEMA,
    deserialize_contamination_status,
    deserialize_detector_result,
    deserialize_firewall_decision,
    deserialize_record_evidence,
    load_episode_results,
    serialize_episode_result,
)
from marble.firewall.types import (
    ContaminationStatus,
    DetectorResult,
    FirewallDecision,
)


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
        assert RESULT_SCHEMA_VERSION == "1.1"

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

    def test_pair_counters_round_trip_nonzero(self, tmp_path: Path) -> None:
        """Nonzero pair counters should survive serialization."""
        result = EpisodeResult(
            run_id="run_0001",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
            attempted_agent_record_pairs=5,
            recontaminated_agent_record_pairs=2,
        )
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        assert len(loaded) == 1
        assert loaded[0].attempted_agent_record_pairs == 5
        assert loaded[0].recontaminated_agent_record_pairs == 2

    def test_pair_counters_round_trip_zero(self, tmp_path: Path) -> None:
        """Zero pair counters should survive serialization."""
        result = EpisodeResult(
            run_id="run_0001",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
            attempted_agent_record_pairs=0,
            recontaminated_agent_record_pairs=0,
        )
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        assert len(loaded) == 1
        assert loaded[0].attempted_agent_record_pairs == 0
        assert loaded[0].recontaminated_agent_record_pairs == 0

    def test_negative_attempted_pairs_rejected(self, tmp_path: Path) -> None:
        """Negative attempted_agent_record_pairs should be rejected."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "attempted_agent_record_pairs": -1,
            "recontaminated_agent_record_pairs": 0,
            "metadata": {},
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(episode_data) + "\n")

        with pytest.raises(ValueError, match="attempted_agent_record_pairs must be non-negative"):
            load_episode_results(episodes_file)

    def test_negative_recontaminated_pairs_rejected(self, tmp_path: Path) -> None:
        """Negative recontaminated_agent_record_pairs should be rejected."""
        episode_data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "attempted_agent_record_pairs": 0,
            "recontaminated_agent_record_pairs": -1,
            "metadata": {},
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(episode_data) + "\n")

        with pytest.raises(
            ValueError, match="recontaminated_agent_record_pairs must be non-negative"
        ):
            load_episode_results(episodes_file)

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


class TestMultiTargetDiskPipeline:
    """Section 10: Full multi-target disk-pipeline coverage."""

    def test_exposed_forget_ids_round_trip(self, tmp_path: Path) -> None:
        """exposed_forget_ids survives disk round trip."""
        result = EpisodeResult(
            run_id="run_0001",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="secret",
                released_text="secret",
                target_exposed=True,
                exposed_forget_ids=("F001", "F002"),
                target_forget_ids=("F001",),
                is_attack_attempt=True,
                is_recontamination_attempt=True,
            )
        )
        result.attempted_agent_record_pairs = 2
        result.recontaminated_agent_record_pairs = 1

        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        assert len(loaded) == 1
        turn = loaded[0].turns[0]
        assert turn.exposed_forget_ids == ("F001", "F002")
        assert turn.target_forget_ids == ("F001",)
        assert turn.target_exposed is True
        assert turn.is_recontamination_attempt is True

    def test_pair_counts_round_trip(self, tmp_path: Path) -> None:
        """Pair counts survive disk round trip."""
        result = EpisodeResult(
            run_id="run_0001",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
            attempted_agent_record_pairs=3,
            recontaminated_agent_record_pairs=1,
        )
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        assert loaded[0].attempted_agent_record_pairs == 3
        assert loaded[0].recontaminated_agent_record_pairs == 1

    def test_rr_unchanged_after_loading(self, tmp_path: Path) -> None:
        """RR metric is unchanged after disk round trip."""
        result = EpisodeResult(
            run_id="run_0001",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
            attempted_agent_record_pairs=4,
            recontaminated_agent_record_pairs=2,
        )
        from experiments.trustparadox_u.evaluator import compute_rr

        rr_before = compute_rr([result])

        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")
        loaded = load_episode_results(episodes_file)
        rr_after = compute_rr(loaded)

        assert rr_before.value == rr_after.value
        assert rr_before.numerator == rr_after.numerator
        assert rr_before.denominator == rr_after.denominator

    def test_unrelated_records_unchanged(self, tmp_path: Path) -> None:
        """Unrelated records remain unchanged after round trip."""
        result = EpisodeResult(
            run_id="run_0001",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="msg",
                released_text="msg",
                target_exposed=False,
                exposed_forget_ids=(),
                target_forget_ids=("F001",),
                is_attack_attempt=True,
            )
        )
        result.contamination_states = {
            "B:F002": ContaminationStatus.CLEAN,
        }

        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")
        loaded = load_episode_results(episodes_file)

        assert loaded[0].turns[0].exposed_forget_ids == ()
        assert loaded[0].contamination_states["B:F002"] == ContaminationStatus.CLEAN

    def test_unexpected_pairs_cause_audit_failure(self, tmp_path: Path) -> None:
        """Unexpected pairs cause audit failure after loading."""
        from experiments.trustparadox_u.audit_results import audit_episode_result

        result = EpisodeResult(
            run_id="run_0001",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
        )
        result.metadata = {
            "config_hash": "a" * 64,
            "forbidden_strings": [],
            "unexpected_recontaminated_pair_count": 2,
        }

        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")
        loaded = load_episode_results(episodes_file)

        findings = audit_episode_result(loaded[0])
        assert any(f.code == "UNEXPECTED_RECONTAMINATION_PAIRS" for f in findings)

    def test_aggregation_produces_bounded_metrics(self, tmp_path: Path) -> None:
        """Aggregation produces bounded metrics after disk round trip."""
        from experiments.trustparadox_u.evaluator import compute_rr

        results = []
        for i in range(3):
            r = EpisodeResult(
                run_id=f"run_{i:04d}",
                episode_id=f"ep{i}",
                scenario_id="s1",
                trust_level="default",
                seed=42 + i,
                attempted_agent_record_pairs=2,
                recontaminated_agent_record_pairs=1,
            )
            results.append(r)

        episodes_file = tmp_path / "episodes.jsonl"
        with open(episodes_file, "w") as f:
            for r in results:
                f.write(json.dumps(serialize_episode_result(r)) + "\n")

        loaded = load_episode_results(episodes_file)
        metric = compute_rr(loaded)
        assert metric.value is not None
        assert 0.0 <= metric.value <= 1.0
        assert metric.numerator <= metric.denominator


class TestNewFieldDiskRoundTrip:
    """Disk round trip preserves reintroduced_forget_ids and reconstructed_forget_ids."""

    def test_reintroduced_forget_ids_round_trip(self, tmp_path: Path) -> None:
        """reintroduced_forget_ids survives serialization round trip."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="A",
            recipient_id="B",
            candidate_text="test",
            released_text="test",
            is_recontamination_attempt=True,
            target_forget_ids=("F001",),
            exposed_forget_ids=("F001",),
            reintroduced_forget_ids=("F001",),
            target_reintroduced=True,
        )
        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
            turns=[turn],
        )
        result.metadata = {"config_hash": "a" * 64}

        episodes_file = tmp_path / "episodes.jsonl"
        with open(episodes_file, "w") as f:
            f.write(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        assert len(loaded) == 1
        loaded_turn = loaded[0].turns[0]
        assert loaded_turn.reintroduced_forget_ids == ("F001",)
        assert loaded_turn.target_reintroduced is True

    def test_reconstructed_forget_ids_round_trip(self, tmp_path: Path) -> None:
        """reconstructed_forget_ids survives serialization round trip."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="A",
            recipient_id="B",
            candidate_text="test",
            released_text="test",
            is_reconstruction_attempt=True,
            reconstructed_forget_ids=("F001", "F002"),
            target_reconstructed=True,
        )
        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
            turns=[turn],
        )
        result.metadata = {"config_hash": "a" * 64}

        episodes_file = tmp_path / "episodes.jsonl"
        with open(episodes_file, "w") as f:
            f.write(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        loaded_turn = loaded[0].turns[0]
        assert loaded_turn.reconstructed_forget_ids == ("F001", "F002")
        assert loaded_turn.target_reconstructed is True

    def test_missing_new_fields_default_to_empty(self, tmp_path: Path) -> None:
        """Old format without new fields deserializes with empty tuples."""
        old_turn = {
            "turn_id": 0,
            "phase": "POST_FORGET_ATTACK",
            "sender_id": "A",
            "recipient_id": "B",
            "candidate_text": "test",
            "released_text": "test",
        }
        old_result = {
            "run_id": "r1",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [old_turn],
            "contamination_states": {},
            "audit_entries": [],
            "metadata": {"config_hash": "a" * 64},
        }

        episodes_file = tmp_path / "episodes.jsonl"
        with open(episodes_file, "w") as f:
            f.write(json.dumps(old_result) + "\n")

        loaded = load_episode_results(episodes_file)
        loaded_turn = loaded[0].turns[0]
        assert loaded_turn.reintroduced_forget_ids == ()
        assert loaded_turn.reconstructed_forget_ids == ()
        assert loaded_turn.target_reintroduced is False
        assert loaded_turn.target_reconstructed is False


class TestDeserializeIdTuple:
    """Section 8: Strict deserialization of per-record ID fields."""

    def test_valid_list(self) -> None:
        """Valid list of strings should deserialize."""
        from experiments.trustparadox_u.serialization import deserialize_id_tuple

        data = {"field": ["a", "b", "c"]}
        result = deserialize_id_tuple(data, "field")
        assert result == ("a", "b", "c")

    def test_empty_list(self) -> None:
        """Empty list should return empty tuple."""
        from experiments.trustparadox_u.serialization import deserialize_id_tuple

        data = {"field": []}
        result = deserialize_id_tuple(data, "field")
        assert result == ()

    def test_missing_field(self) -> None:
        """Missing field should return empty tuple."""
        from experiments.trustparadox_u.serialization import deserialize_id_tuple

        data = {"other": ["a"]}
        result = deserialize_id_tuple(data, "field")
        assert result == ()

    def test_non_list_raises(self) -> None:
        """Non-list value should raise ValueError."""
        from experiments.trustparadox_u.serialization import deserialize_id_tuple

        data = {"field": "not-a-list"}
        with pytest.raises(ValueError, match="must be a list"):
            deserialize_id_tuple(data, "field")


class TestParseSchemaVersion:
    """Section 3: Numeric schema version parsing."""

    def test_simple_version(self) -> None:
        """Simple version string parses correctly."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        assert parse_schema_version("1.1") == (1, 1)

    def test_single_digit(self) -> None:
        """Single digit version parses correctly."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        assert parse_schema_version("0") == (0,)

    def test_multi_part_version(self) -> None:
        """Multi-part version parses correctly."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        assert parse_schema_version("1.2.3") == (1, 2, 3)

    def test_numeric_comparison_safe(self) -> None:
        """Numeric comparison handles 1.10 > 1.2 correctly."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        assert parse_schema_version("1.10") > parse_schema_version("1.2")

    def test_1_0_less_than_1_1(self) -> None:
        """1.0 < 1.1 numerically."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        assert parse_schema_version("1.0") < parse_schema_version("1.1")

    def test_equal_versions(self) -> None:
        """Equal versions compare equal."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        assert parse_schema_version("1.1") == parse_schema_version("1.1")

    def test_malformed_raises(self) -> None:
        """Malformed version string raises ValueError."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        with pytest.raises(ValueError, match="Invalid schema version"):
            parse_schema_version("abc")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        with pytest.raises(ValueError, match="Invalid schema version"):
            parse_schema_version("")

    def test_non_string_raises(self) -> None:
        """Non-string input raises ValueError."""
        from experiments.trustparadox_u.serialization import parse_schema_version

        with pytest.raises(ValueError, match="must be a string"):
            parse_schema_version(123)  # type: ignore[arg-type]


class TestSchemaVersionConstants:
    """Schema version constants are defined correctly."""

    def test_constants_defined(self) -> None:
        """All schema version constants are defined."""
        from experiments.trustparadox_u.serialization import (
            LEGACY_RESULT_SCHEMA_VERSION,
            RESULT_SCHEMA_VERSION,
        )

        assert UNVERSIONED_RESULT_SCHEMA == "0"
        assert LEGACY_RESULT_SCHEMA_VERSION == "1.0"
        assert RESULT_SCHEMA_VERSION == "1.1"

    def test_new_records_use_current_schema(self, tmp_path: Path) -> None:
        """Newly serialized records use schema 1.1."""
        from experiments.trustparadox_u.serialization import (
            RESULT_SCHEMA_VERSION,
            serialize_episode_result,
        )

        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
        )
        data = serialize_episode_result(result)
        assert data["schema_version"] == RESULT_SCHEMA_VERSION
        assert data["schema_version"] == "1.1"

    def test_non_string_item_raises(self) -> None:
        """Non-string items should raise ValueError."""
        from experiments.trustparadox_u.serialization import deserialize_id_tuple

        data = {"field": ["a", 123]}
        with pytest.raises(ValueError, match="must contain strings"):
            deserialize_id_tuple(data, "field")

    def test_empty_string_raises(self) -> None:
        """Empty string IDs should raise ValueError."""
        from experiments.trustparadox_u.serialization import deserialize_id_tuple

        data = {"field": ["a", ""]}
        with pytest.raises(ValueError, match="empty ID"):
            deserialize_id_tuple(data, "field")

    def test_duplicates_raise(self) -> None:
        """Duplicate IDs should raise ValueError."""
        from experiments.trustparadox_u.serialization import deserialize_id_tuple

        data = {"field": ["a", "b", "a"]}
        with pytest.raises(ValueError, match="duplicate"):
            deserialize_id_tuple(data, "field")

    def test_dict_value_raises(self) -> None:
        """Dict value should raise ValueError."""
        from experiments.trustparadox_u.serialization import deserialize_id_tuple

        data = {"field": {"nested": True}}
        with pytest.raises(ValueError, match="must be a list"):
            deserialize_id_tuple(data, "field")


class TestInspectResultSchemaVersionsHardened:
    """Section 5: Hardened schema envelope inspection."""

    def _inspect(self, path: Path) -> set[str]:
        from experiments.trustparadox_u.serialization import inspect_result_schema_versions

        return inspect_result_schema_versions(path)

    def test_list_envelope_rejected(self, tmp_path: Path) -> None:
        """List envelope is rejected."""
        f = tmp_path / "episodes.jsonl"
        f.write_text("[]\n")
        with pytest.raises(ValueError, match="must be a JSON object"):
            self._inspect(f)

    def test_string_envelope_rejected(self, tmp_path: Path) -> None:
        """String envelope is rejected."""
        f = tmp_path / "episodes.jsonl"
        f.write_text('"episode"\n')
        with pytest.raises(ValueError, match="must be a JSON object"):
            self._inspect(f)

    def test_null_envelope_rejected(self, tmp_path: Path) -> None:
        """Null envelope is rejected."""
        f = tmp_path / "episodes.jsonl"
        f.write_text("null\n")
        with pytest.raises(ValueError, match="must be a JSON object"):
            self._inspect(f)

    def test_integer_envelope_rejected(self, tmp_path: Path) -> None:
        """Integer envelope is rejected."""
        f = tmp_path / "episodes.jsonl"
        f.write_text("42\n")
        with pytest.raises(ValueError, match="must be a JSON object"):
            self._inspect(f)

    def test_numeric_schema_version_rejected(self, tmp_path: Path) -> None:
        """Numeric (non-string) schema version is rejected."""
        f = tmp_path / "episodes.jsonl"
        f.write_text(json.dumps({"schema_version": 1.1, "episode": {}}) + "\n")
        with pytest.raises(ValueError, match="Invalid schema version"):
            self._inspect(f)

    def test_malformed_schema_string_rejected(self, tmp_path: Path) -> None:
        """Malformed schema version string is rejected."""
        f = tmp_path / "episodes.jsonl"
        f.write_text(json.dumps({"schema_version": "abc", "episode": {}}) + "\n")
        with pytest.raises(ValueError, match="Invalid schema version"):
            self._inspect(f)

    def test_missing_episode_object_rejected(self, tmp_path: Path) -> None:
        """Schema 1.1 without episode field is rejected."""
        f = tmp_path / "episodes.jsonl"
        f.write_text(json.dumps({"schema_version": "1.1", "run_id": "r1"}) + "\n")
        with pytest.raises(ValueError, match="missing 'episode' field"):
            self._inspect(f)

    def test_episode_wrong_type_rejected(self, tmp_path: Path) -> None:
        """Schema 1.1 with non-mapping episode is rejected."""
        f = tmp_path / "episodes.jsonl"
        f.write_text(json.dumps({"schema_version": "1.1", "episode": "not-a-dict"}) + "\n")
        with pytest.raises(ValueError, match="expected a JSON object"):
            self._inspect(f)

    def test_line_number_in_error(self, tmp_path: Path) -> None:
        """Error message includes line number."""
        f = tmp_path / "episodes.jsonl"
        valid = json.dumps(
            {
                "schema_version": "1.1",
                "episode": {
                    "run_id": "r1",
                    "episode_id": "ep1",
                    "scenario_id": "s1",
                    "trust_level": "default",
                    "seed": 42,
                    "turns": [],
                    "contamination_states": {},
                    "metadata": {},
                },
            }
        )
        f.write_text(valid + "\n" + "[]\n")
        with pytest.raises(ValueError, match="line 2"):
            self._inspect(f)

    def test_valid_envelope_passes(self, tmp_path: Path) -> None:
        """Valid envelope passes inspection."""
        f = tmp_path / "episodes.jsonl"
        f.write_text(
            json.dumps(
                {
                    "schema_version": "1.1",
                    "episode": {"run_id": "r1", "episode_id": "ep1"},
                }
            )
            + "\n"
        )
        versions = self._inspect(f)
        assert versions == {"1.1"}

    def test_unversioned_no_episode_required(self, tmp_path: Path) -> None:
        """Unversioned schema 0 does not require episode wrapper."""
        f = tmp_path / "episodes.jsonl"
        f.write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "episode_id": "ep1",
                    "scenario_id": "s1",
                    "trust_level": "default",
                    "seed": 42,
                    "turns": [],
                    "contamination_states": {},
                    "metadata": {},
                }
            )
            + "\n"
        )
        versions = self._inspect(f)
        assert versions == {"0"}


class TestFinalContaminationStates:
    """final_contamination_states serialization round trip."""

    def test_final_states_round_trip(self, tmp_path: Path) -> None:
        """final_contamination_states survives disk round trip."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
        )
        result.final_contamination_states = {
            ("CK", "F001"): "at_risk",
            ("CK", "F002"): "clean",
            ("SK", "F001"): "unknown",
        }

        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        assert len(loaded) == 1
        fcs = loaded[0].final_contamination_states
        assert fcs[("CK", "F001")] == "at_risk"
        assert fcs[("CK", "F002")] == "clean"
        assert fcs[("SK", "F001")] == "unknown"

    def test_empty_final_states_round_trip(self, tmp_path: Path) -> None:
        """Empty final_contamination_states survives disk round trip."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
        )

        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        assert loaded[0].final_contamination_states == {}

    def test_duplicate_final_states_rejected(self, tmp_path: Path) -> None:
        """Duplicate final_contamination_states are rejected."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "episode_id": "ep1",
                    "scenario_id": "s1",
                    "trust_level": "default",
                    "schema_version": "1.1",
                    "seed": 42,
                    "turns": [],
                    "contamination_states": {},
                    "metadata": {},
                    "final_contamination_states": [
                        {"agent_id": "CK", "forget_id": "F001", "status": "clean"},
                        {"agent_id": "CK", "forget_id": "F001", "status": "at_risk"},
                    ],
                }
            )
            + "\n"
        )
        with pytest.raises(ValueError, match="Duplicate final state"):
            load_episode_results(episodes_file)

    def test_invalid_status_rejected(self, tmp_path: Path) -> None:
        """Invalid contamination status is rejected."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "episode_id": "ep1",
                    "scenario_id": "s1",
                    "trust_level": "default",
                    "schema_version": "1.1",
                    "seed": 42,
                    "turns": [],
                    "contamination_states": {},
                    "metadata": {},
                    "final_contamination_states": [
                        {"agent_id": "CK", "forget_id": "F001", "status": "invalid_status"},
                    ],
                }
            )
            + "\n"
        )
        with pytest.raises(ValueError, match="Invalid status"):
            load_episode_results(episodes_file)

    def test_empty_agent_id_rejected(self, tmp_path: Path) -> None:
        """Empty agent_id in final_contamination_states is rejected."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "episode_id": "ep1",
                    "scenario_id": "s1",
                    "trust_level": "default",
                    "schema_version": "1.1",
                    "seed": 42,
                    "turns": [],
                    "contamination_states": {},
                    "metadata": {},
                    "final_contamination_states": [
                        {"agent_id": "", "forget_id": "F001", "status": "clean"},
                    ],
                }
            )
            + "\n"
        )
        with pytest.raises(ValueError, match="Invalid agent_id"):
            load_episode_results(episodes_file)

    def test_empty_forget_id_rejected(self, tmp_path: Path) -> None:
        """Empty forget_id in final_contamination_states is rejected."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "episode_id": "ep1",
                    "scenario_id": "s1",
                    "trust_level": "default",
                    "schema_version": "1.1",
                    "seed": 42,
                    "turns": [],
                    "contamination_states": {},
                    "metadata": {},
                    "final_contamination_states": [
                        {"agent_id": "CK", "forget_id": "", "status": "clean"},
                    ],
                }
            )
            + "\n"
        )
        with pytest.raises(ValueError, match="Invalid forget_id"):
            load_episode_results(episodes_file)

    def test_split_rr_fields_round_trip(self, tmp_path: Path) -> None:
        """Split RR fields survive disk round trip."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
            attempted_clean_pairs=10,
            recontaminated_clean_pairs=3,
            attempted_at_risk_pairs=5,
            escalated_at_risk_pairs=2,
        )

        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")

        loaded = load_episode_results(episodes_file)
        assert len(loaded) == 1
        assert loaded[0].attempted_clean_pairs == 10
        assert loaded[0].recontaminated_clean_pairs == 3
        assert loaded[0].attempted_at_risk_pairs == 5
        assert loaded[0].escalated_at_risk_pairs == 2


class TestDeserializeRecordEvidence:
    """Tests for strict per-record detector evidence deserialization (s13)."""

    def test_valid_evidence(self) -> None:
        """Valid evidence should deserialize correctly."""
        raw = [
            {
                "forget_id": "F001",
                "exact_score": 1.0,
                "entity_score": 0.5,
                "semantic_score": 0.3,
                "reconstruction_score": 0.0,
                "matched": True,
            },
            {
                "forget_id": "F002",
                "exact_score": 0.0,
                "entity_score": 0.0,
                "semantic_score": 0.0,
                "reconstruction_score": 0.0,
                "matched": False,
            },
        ]
        result = deserialize_record_evidence(raw, ("F001",))
        assert len(result) == 2
        assert result[0].forget_id == "F001"
        assert result[0].matched is True
        assert result[1].forget_id == "F002"
        assert result[1].matched is False

    def test_empty_evidence(self) -> None:
        """Empty list should deserialize to empty tuple."""
        result = deserialize_record_evidence([], ())
        assert result == ()

    def test_duplicate_forget_id(self) -> None:
        """Duplicate forget IDs should raise ValueError."""
        raw = [
            {"forget_id": "F001", "exact_score": 0.0, "matched": False},
            {"forget_id": "F001", "exact_score": 0.5, "matched": True},
        ]
        with pytest.raises(ValueError, match="duplicate forget_id"):
            deserialize_record_evidence(raw, ())

    def test_missing_forget_id(self) -> None:
        """Missing forget_id should raise ValueError."""
        raw = [{"exact_score": 0.5, "matched": False}]
        with pytest.raises(ValueError, match="nonempty string"):
            deserialize_record_evidence(raw, ())

    def test_empty_forget_id(self) -> None:
        """Empty forget_id should raise ValueError."""
        raw = [{"forget_id": "", "exact_score": 0.5, "matched": False}]
        with pytest.raises(ValueError, match="nonempty string"):
            deserialize_record_evidence(raw, ())

    def test_invalid_score_type(self) -> None:
        """Non-numeric score should raise ValueError."""
        raw = [{"forget_id": "F001", "exact_score": "high", "matched": False}]
        with pytest.raises(ValueError, match="must be numeric"):
            deserialize_record_evidence(raw, ())

    def test_bool_score_rejected(self) -> None:
        """Boolean score should raise ValueError."""
        raw = [{"forget_id": "F001", "exact_score": True, "matched": False}]
        with pytest.raises(ValueError, match="must be numeric, not bool"):
            deserialize_record_evidence(raw, ())

    def test_out_of_range_score(self) -> None:
        """Score outside [0, 1] should raise ValueError."""
        raw = [{"forget_id": "F001", "exact_score": 1.5, "matched": False}]
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            deserialize_record_evidence(raw, ())

    def test_negative_score(self) -> None:
        """Negative score should raise ValueError."""
        raw = [{"forget_id": "F001", "exact_score": -0.1, "matched": False}]
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            deserialize_record_evidence(raw, ())

    def test_non_boolean_matched(self) -> None:
        """Non-boolean matched should raise ValueError."""
        raw = [{"forget_id": "F001", "exact_score": 0.5, "matched": 1}]
        with pytest.raises(ValueError, match="must be boolean"):
            deserialize_record_evidence(raw, ())

    def test_mismatch_with_matched_ids(self) -> None:
        """matched_forget_ids must match evidence with matched=True."""
        raw = [
            {"forget_id": "F001", "exact_score": 1.0, "matched": True},
        ]
        # F001 is matched in evidence but matched_forget_ids says F002
        with pytest.raises(ValueError, match="does not match"):
            deserialize_record_evidence(raw, ("F002",))

    def test_non_object_entry(self) -> None:
        """Non-object entry should raise ValueError."""
        raw = ["not_an_object"]
        with pytest.raises(ValueError, match="must be an object"):
            deserialize_record_evidence(raw, ())

    def test_non_list_input(self) -> None:
        """Non-list input should raise ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            deserialize_record_evidence("not_a_list", ())

    def test_integer_score_accepted(self) -> None:
        """Integer scores (0 and 1) should be accepted."""
        raw = [
            {"forget_id": "F001", "exact_score": 1, "entity_score": 0, "matched": True},
        ]
        result = deserialize_record_evidence(raw, ("F001",))
        assert result[0].exact_score == 1.0
        assert result[0].entity_score == 0.0


class TestLegacyFinalStateRestriction:
    """Tests for s14: legacy dict form restricted to old schemas."""

    def test_schema_1_1_rejects_dict_form(self, tmp_path: Path) -> None:
        """Schema 1.1 must reject legacy dict form."""
        episode_data = {
            "run_id": "r1",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "final_contamination_states": {
                '["CK", "F001"]': "clean",
            },
        }
        record = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "episode": episode_data,
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(record) + "\n")
        with pytest.raises(ValueError, match="must be a list for schema"):
            load_episode_results(episodes_file)

    def test_legacy_schema_accepts_dict_form(self, tmp_path: Path) -> None:
        """Schema 1.0 should still accept legacy dict form."""
        episode_data = {
            "run_id": "r1",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "final_contamination_states": {},
        }
        record = {
            "schema_version": "1.0",
            "episode": episode_data,
        }
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(record) + "\n")
        # Should not raise
        results = load_episode_results(episodes_file)
        assert len(results) == 1

    def test_final_probe_turn_round_trip(self, tmp_path: Path) -> None:
        """FINAL_PROBE turn with exposed_forget_ids and state changes survives round trip."""
        from experiments.trustparadox_u.runner import (
            ContaminationStateChange,
            EpisodeResult,
            TurnResult,
        )
        from experiments.trustparadox_u.serialization import (
            load_episode_results,
            serialize_episode_result,
        )

        result = EpisodeResult(
            run_id="run_0001",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="default",
            seed=42,
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="FINAL_PROBE",
                sender_id="CK",
                recipient_id="CK",
                candidate_text="Nothing.",
                released_text="Nothing.",
                target_exposed=False,
                exposed_forget_ids=(),
                contamination_state_changes=(
                    ContaminationStateChange(
                        agent_id="CK",
                        forget_id="F001",
                        before="contaminated",
                        after="clean",
                        reason="verified_cleanup",
                    ),
                ),
            )
        )
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(json.dumps(serialize_episode_result(result)) + "\n")
        loaded = load_episode_results(episodes_file)
        assert len(loaded) == 1
        turn = loaded[0].turns[0]
        assert turn.phase == "FINAL_PROBE"
        assert turn.target_exposed is False
        assert turn.exposed_forget_ids == ()
        assert len(turn.contamination_state_changes) == 1
        change = turn.contamination_state_changes[0]
        assert change.agent_id == "CK"
        assert change.forget_id == "F001"
        assert change.before == "contaminated"
        assert change.after == "clean"

"""Integration tests for runner-to-aggregation pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from experiments.trustparadox_u.aggregate import main as aggregate_main
from experiments.trustparadox_u.runner import EpisodeResult, TurnResult
from experiments.trustparadox_u.serialization import load_episode_results
from marble.firewall.types import ContaminationStatus, DetectorResult, FirewallDecision


def _create_realistic_episode() -> EpisodeResult:
    """Create a realistic EpisodeResult with all nested structures."""
    detector = DetectorResult(
        exact_score=0.0,
        entity_score=0.1,
        semantic_score=0.2,
        reconstruction_score=0.0,
        matched_forget_ids=("forget_001",),
        evidence=("entity match",),
    )
    decision = FirewallDecision(
        action="allow",
        released_text="This is a test message",
        detector_result=detector,
        reason_codes=("low_score",),
        policy_version="1.0",
        latency_ms=5.5,
    )
    turn = TurnResult(
        turn_id=0,
        phase="POST_FORGET_ATTACK",
        sender_id="agent_A",
        recipient_id="agent_B",
        candidate_text="Original message",
        released_text="This is a test message",
        decision=decision,
        attack_type="direct",
        attack_step_index=0,
        is_attack_attempt=True,
        is_legitimate_message=False,
        is_reconstruction_attempt=False,
        is_recontamination_attempt=False,
        target_exposed=False,
        target_reconstructed=False,
        target_reintroduced=False,
        task_relevant=True,
        task_contribution_successful=True,
    )
    result = EpisodeResult(
        run_id="run_0001",
        episode_id="ep_001",
        scenario_id="scenario_1",
        trust_level="default",
        seed=42,
        turns=[turn],
        contamination_states={
            "agent_A": ContaminationStatus.CLEAN,
            "agent_B": ContaminationStatus.CONTAMINATED,
        },
        audit_entries=[],
        task_success=True,
        task_label="test_task",
        cleaned_agents_exposed=0,
        recontaminated_agents=0,
        metadata={
            "config_hash": "a" * 64,
            "forbidden_strings": ["secret"],
            "secret_variant_id": "sv1",
            "attack_type": "direct",
            "seed": 42,
            "run_mode": "test",
            "semantic_threshold": 0.8,
            "firewall_enabled": True,
        },
    )
    return result


def _write_episode_and_manifest(
    tmp_path: Path,
    episode: EpisodeResult,
    commit: str = "a" * 40,
) -> tuple[Path, Path]:
    """Write episode results and manifest to tmp_path."""
    from experiments.trustparadox_u.paths import EPISODE_RESULTS_FILENAME

    # Write episodes.jsonl
    episodes_file = tmp_path / EPISODE_RESULTS_FILENAME
    episode_dict = {
        "run_id": episode.run_id,
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "trust_level": episode.trust_level,
        "seed": episode.seed,
        "turns": [
            {
                "turn_id": t.turn_id,
                "phase": t.phase,
                "sender_id": t.sender_id,
                "recipient_id": t.recipient_id,
                "candidate_text": t.candidate_text,
                "released_text": t.released_text,
                "decision": t.decision.to_dict() if t.decision else None,
                "attack_type": t.attack_type,
                "attack_step_index": t.attack_step_index,
                "is_attack_attempt": t.is_attack_attempt,
                "is_legitimate_message": t.is_legitimate_message,
                "is_reconstruction_attempt": t.is_reconstruction_attempt,
                "is_recontamination_attempt": t.is_recontamination_attempt,
                "target_exposed": t.target_exposed,
                "target_reconstructed": t.target_reconstructed,
                "target_reintroduced": t.target_reintroduced,
                "task_relevant": t.task_relevant,
                "task_contribution_successful": t.task_contribution_successful,
            }
            for t in episode.turns
        ],
        "contamination_states": {
            agent_id: status.value for agent_id, status in episode.contamination_states.items()
        },
        "audit_entries": episode.audit_entries,
        "task_success": episode.task_success,
        "task_label": episode.task_label,
        "cleaned_agents_exposed": episode.cleaned_agents_exposed,
        "recontaminated_agents": episode.recontaminated_agents,
        "metadata": episode.metadata,
    }
    episodes_file.write_text(json.dumps(episode_dict) + "\n")

    # Write smoke_manifest.json
    manifest_file = tmp_path / "smoke_manifest.json"
    manifest = {
        "repository_commit": commit,
        "generated_at_utc": "2024-01-01T00:00:00+00:00",
        "run_mode": "test",
        "config_hashes": [episode.metadata["config_hash"]],
        "provider": None,
        "model": None,
        "dimension": None,
        "semantic_threshold": episode.metadata["semantic_threshold"],
        "api_base_sanitized": None,
        "episode_ids": [episode.episode_id],
        "seeds": [episode.seed],
        "result_count": 1,
        "audit_valid": True,
        "audit_error_count": 0,
        "metric_counts": {},
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))

    return episodes_file, manifest_file


class TestRunnerToAggregationIntegration:
    """Integration tests for the full runner-to-aggregation pipeline."""

    def test_realistic_episode_aggregates_successfully(self, tmp_path: Path) -> None:
        """A realistic episode should aggregate successfully."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode = _create_realistic_episode()
        _write_episode_and_manifest(input_dir, episode)

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            exit_code = aggregate_main()

        assert exit_code == 0
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "audit_report.json").exists()

        # Verify metrics structure
        metrics = json.loads((output_dir / "metrics.json").read_text())
        assert "pu_rer" in metrics
        assert "crr" in metrics
        assert "rr" in metrics

    def test_deserialized_decision_has_correct_attributes(self, tmp_path: Path) -> None:
        """Deserialized firewall decisions should have correct attributes."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        episode = _create_realistic_episode()
        episodes_file, _ = _write_episode_and_manifest(input_dir, episode)

        results = load_episode_results(episodes_file)
        assert len(results) == 1
        assert len(results[0].turns) == 1

        turn = results[0].turns[0]
        assert turn.decision is not None
        assert isinstance(turn.decision, FirewallDecision)
        assert turn.decision.action == "allow"
        assert turn.decision.released_text == "This is a test message"
        assert isinstance(turn.decision.detector_result, DetectorResult)
        assert turn.decision.detector_result.exact_score == 0.0

    def test_deserialized_contamination_states_correct(self, tmp_path: Path) -> None:
        """Deserialized contamination states should be correct."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        episode = _create_realistic_episode()
        episodes_file, _ = _write_episode_and_manifest(input_dir, episode)

        results = load_episode_results(episodes_file)
        assert len(results) == 1

        assert "agent_A" in results[0].contamination_states
        assert results[0].contamination_states["agent_A"] == ContaminationStatus.CLEAN
        assert results[0].contamination_states["agent_B"] == ContaminationStatus.CONTAMINATED

    def test_malformed_decision_fails_gracefully(self, tmp_path: Path) -> None:
        """Malformed decision should fail with clear error."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode = _create_realistic_episode()
        episodes_file, manifest_file = _write_episode_and_manifest(input_dir, episode)

        # Corrupt the decision
        episode_dict = json.loads(episodes_file.read_text())
        episode_dict["turns"][0]["decision"] = {"invalid": "structure"}
        episodes_file.write_text(json.dumps(episode_dict) + "\n")

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            exit_code = aggregate_main()

        assert exit_code == 3  # RESULT_LOAD

    def test_malformed_contamination_status_fails(self, tmp_path: Path) -> None:
        """Malformed contamination status should fail with clear error."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode = _create_realistic_episode()
        episodes_file, manifest_file = _write_episode_and_manifest(input_dir, episode)

        # Corrupt the contamination status
        episode_dict = json.loads(episodes_file.read_text())
        episode_dict["contamination_states"]["agent_A"] = "invalid_status"
        episodes_file.write_text(json.dumps(episode_dict) + "\n")

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            exit_code = aggregate_main()

        assert exit_code == 3  # RESULT_LOAD

    def test_duplicate_results_blocked(self, tmp_path: Path) -> None:
        """Duplicate results should be blocked."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode = _create_realistic_episode()
        episodes_file, manifest_file = _write_episode_and_manifest(input_dir, episode)

        # Add duplicate episode
        episode_dict = json.loads(episodes_file.read_text())
        episodes_file.write_text(json.dumps(episode_dict) + "\n" + json.dumps(episode_dict) + "\n")

        # Update manifest to expect 2 results
        manifest = json.loads(manifest_file.read_text())
        manifest["result_count"] = 2
        manifest_file.write_text(json.dumps(manifest))

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            exit_code = aggregate_main()

        # Should fail due to duplicate run identity (audit error)
        assert exit_code == 4  # AUDIT

    def test_wrong_manifest_commit_fails(self, tmp_path: Path) -> None:
        """Wrong manifest commit should fail validation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode = _create_realistic_episode()
        _write_episode_and_manifest(input_dir, episode, commit="b" * 40)

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            # Mock get_repository_commit to return different commit
            with patch(
                "experiments.trustparadox_u.aggregate.validate_manifest_or_raise"
            ) as mock_validate:
                mock_validate.side_effect = ValueError("Commit mismatch")
                exit_code = aggregate_main()

        assert exit_code == 1  # General ValueError

    def test_wrong_config_hash_fails(self, tmp_path: Path) -> None:
        """Wrong config hash in manifest should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode = _create_realistic_episode()
        episodes_file, manifest_file = _write_episode_and_manifest(input_dir, episode)

        # Change config hash in manifest
        manifest = json.loads(manifest_file.read_text())
        manifest["config_hashes"] = ["b" * 64]
        manifest_file.write_text(json.dumps(manifest))

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            exit_code = aggregate_main()

        # Should fail due to config hash mismatch (manifest validation)
        assert exit_code == 5  # MANIFEST_VALIDATION

    def test_missing_results_file_fails(self, tmp_path: Path) -> None:
        """Missing results file should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Only write manifest, no episodes
        manifest_file = input_dir / "smoke_manifest.json"
        manifest = {
            "repository_commit": "a" * 40,
            "generated_at_utc": "2024-01-01T00:00:00+00:00",
            "run_mode": "test",
            "config_hashes": ["a" * 64],
            "provider": None,
            "model": None,
            "dimension": None,
            "semantic_threshold": 0.8,
            "api_base_sanitized": None,
            "episode_ids": ["ep_001"],
            "seeds": [42],
            "result_count": 1,
            "audit_valid": True,
            "audit_error_count": 0,
            "metric_counts": {},
        }
        manifest_file.write_text(json.dumps(manifest))

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            exit_code = aggregate_main()

        assert exit_code == 2  # INPUT_MISSING

    def test_missing_manifest_fails(self, tmp_path: Path) -> None:
        """Missing manifest should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode = _create_realistic_episode()
        episodes_file, _ = _write_episode_and_manifest(input_dir, episode)

        # Remove manifest
        manifest_file = input_dir / "smoke_manifest.json"
        manifest_file.unlink()

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            exit_code = aggregate_main()

        assert exit_code == 5  # MANIFEST_VALIDATION

    def test_invalid_observed_dimension_fails(self, tmp_path: Path) -> None:
        """Invalid observed dimension should fail validation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode = _create_realistic_episode()
        episodes_file, manifest_file = _write_episode_and_manifest(input_dir, episode)

        # Set invalid dimension in manifest
        manifest = json.loads(manifest_file.read_text())
        manifest["dimension"] = -1
        manifest_file.write_text(json.dumps(manifest))

        with patch.object(
            sys,
            "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir)],
        ):
            exit_code = aggregate_main()

        # Should fail due to invalid dimension (manifest validation)
        assert exit_code == 5  # MANIFEST_VALIDATION

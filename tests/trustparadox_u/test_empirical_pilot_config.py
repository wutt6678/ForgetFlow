"""Tests for E2 primary pilot configuration and request schedule."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.trustparadox_u.empirical_pilot_config import (
    PILOT_EXECUTION_SEED,
    PilotRequest,
    build_request_schedule,
    create_default_pilot_config,
    load_request_schedule,
    randomize_schedule,
    save_pilot_config,
    save_request_schedule,
)


class TestPrimaryPilotConfig:
    """Tests for primary pilot configuration."""

    def test_create_default_config(self) -> None:
        """Test creating the default pilot configuration."""
        config = create_default_pilot_config()
        assert config.schema_version == "1.0.0"
        assert config.protocol_version == "2.0.0"
        assert config.study_version == "2.0.0"
        assert config.provider == "openai"
        assert config.model == "openai/qwen3.7-plus"
        assert config.transport == "litellm"
        assert config.temperature == 0.7
        assert config.max_tokens == 512
        assert config.timeout == 60.0
        assert config.retry_policy == {"max_retries": 3, "backoff_factor": 2.0}
        assert len(config.development_target_variants) == 3
        assert len(config.trust_levels) == 3
        assert config.samples_per_scenario == 10
        assert config.randomization_seed == PILOT_EXECUTION_SEED

    def test_config_to_dict(self) -> None:
        """Test converting config to dict."""
        config = create_default_pilot_config()
        data = config.to_dict()
        assert isinstance(data, dict)
        assert data["schema_version"] == "1.0.0"
        assert data["provider"] == "openai"
        assert isinstance(data["development_target_variants"], list)
        assert isinstance(data["trust_levels"], list)

    def test_config_sha256(self) -> None:
        """Test config SHA-256 computation."""
        config = create_default_pilot_config()
        hash1 = config.config_sha256()
        hash2 = config.config_sha256()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_config_sha256_changes_with_content(self) -> None:
        """Test that config SHA-256 changes when content changes."""
        config1 = create_default_pilot_config(temperature=0.7)
        config2 = create_default_pilot_config(temperature=0.8)
        assert config1.config_sha256() != config2.config_sha256()

    def test_save_and_load_config(self, tmp_path: Path) -> None:
        """Test saving and loading pilot configuration."""
        config = create_default_pilot_config()
        config_path = tmp_path / "pilot_config.json"
        save_pilot_config(config, config_path)
        assert config_path.exists()

        # Verify the file is valid JSON
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["schema_version"] == "1.0.0"


class TestRequestSchedule:
    """Tests for request schedule building and randomization."""

    def test_build_request_schedule(self) -> None:
        """Test building the 90-request schedule."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)
        assert len(schedule) == 90  # 6 variants × 10 samples × 3 trust levels

    def test_schedule_structure(self) -> None:
        """Test schedule structure and indexing."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)

        # Check first request
        first = schedule[0]
        assert first.request_order_index == 0
        assert first.generation_family_id.startswith("credential_001_f")
        assert first.trust_level in ["low", "default", "high"]
        assert first.sample_index == 0
        assert first.generation_replicate == 0

        # Check last request
        last = schedule[-1]
        assert last.request_order_index == 89

    def test_schedule_has_all_trust_levels(self) -> None:
        """Test that schedule includes all trust levels for each family."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)

        # Group by generation family
        families: dict[str, set[str]] = {}
        for req in schedule:
            families.setdefault(req.generation_family_id, set()).add(req.trust_level)

        # Each family should have all 3 trust levels
        for family_id, trust_levels in families.items():
            assert trust_levels == {
                "low",
                "default",
                "high",
            }, f"Family {family_id} missing trust levels"

    def test_schedule_has_30_families(self) -> None:
        """Test that schedule has exactly 30 generation families."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)

        family_ids = {req.generation_family_id for req in schedule}
        assert len(family_ids) == 30

    def test_randomize_schedule_deterministic(self) -> None:
        """Test that randomization is deterministic with same seed."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)

        randomized1 = randomize_schedule(schedule, config.randomization_seed)
        randomized2 = randomize_schedule(schedule, config.randomization_seed)

        # Same seed should produce identical order
        for req1, req2 in zip(randomized1, randomized2):
            assert req1.generation_family_id == req2.generation_family_id
            assert req1.trust_level == req2.trust_level

    def test_randomize_schedule_different_seed(self) -> None:
        """Test that different seeds produce different orders."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)

        randomized1 = randomize_schedule(schedule, 12345)
        randomized2 = randomize_schedule(schedule, 67890)

        # Different seeds should (almost certainly) produce different orders
        order1 = [(r.generation_family_id, r.trust_level) for r in randomized1]
        order2 = [(r.generation_family_id, r.trust_level) for r in randomized2]
        assert order1 != order2

    def test_randomize_schedule_reindexes(self) -> None:
        """Test that randomization re-indexes requests."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)
        randomized = randomize_schedule(schedule, config.randomization_seed)

        # Check that request_order_index is 0..89 in order
        indices = [req.request_order_index for req in randomized]
        assert indices == list(range(90))

    def test_randomize_schedule_not_trivially_grouped(self) -> None:
        """Test that randomized schedule is not trivially grouped by trust."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)
        randomized = randomize_schedule(schedule, config.randomization_seed)

        # Extract trust levels in order
        trust_sequence = [req.trust_level for req in randomized[:30]]

        # Should not be all the same trust level in first 30
        assert len(set(trust_sequence)) > 1

    def test_save_and_load_schedule(self, tmp_path: Path) -> None:
        """Test saving and loading request schedule."""
        config = create_default_pilot_config()
        schedule = build_request_schedule(config)
        randomized = randomize_schedule(schedule, config.randomization_seed)

        schedule_path = tmp_path / "request_schedule.json"
        save_request_schedule(randomized, schedule_path)
        assert schedule_path.exists()

        loaded = load_request_schedule(schedule_path)
        assert len(loaded) == 90

        # Verify content matches
        for orig, load in zip(randomized, loaded):
            assert orig.generation_family_id == load.generation_family_id
            assert orig.trust_level == load.trust_level
            assert orig.request_order_index == load.request_order_index


class TestPilotRequest:
    """Tests for PilotRequest dataclass."""

    def test_pilot_request_to_dict(self) -> None:
        """Test converting PilotRequest to dict."""
        req = PilotRequest(
            request_order_index=0,
            generation_family_id="credential_001_f000",
            scenario_id="credential_001",
            secret_variant_id="credential_v1",
            trust_level="low",
            sample_index=0,
            generation_replicate=0,
            pilot_prompt_version="abc123",
        )
        data = req.to_dict()
        assert isinstance(data, dict)
        assert data["request_order_index"] == 0
        assert data["generation_family_id"] == "credential_001_f000"
        assert data["trust_level"] == "low"

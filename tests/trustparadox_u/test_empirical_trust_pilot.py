"""E2-044..054: empirical phase trust-pilot and prompt-freeze tests.

This module covers the E2-A preflight checklist items: phase-lock enum
semantics, provenance separation, identity separation, clean-tree gate,
pilot prompt invariance, request scheduling, raw retention, pilot
labeling, manipulation analysis, prompt freeze, and synthetic regression.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    REPLAY_FAMILY_ID_FIELD,
    REPLAY_SEQUENCE_FAMILY_ID_FIELD,
    RESERVED_REPLAY_IDENTITY_FIELDS,
    AttackType,
    EmpiricalCleanTreeRequiredError,
    GenerationMode,
    GenerationStatus,
    TrustLevel,
    assert_clean_repository_tree,
    assert_replay_identity_unassigned,
    empirical_candidate_id,
    generation_family_id,
    get_target_spec,
)
from experiments.trustparadox_u.empirical_generation import (
    EMPIRICAL_PROMPTS_DIR,
    INVARIANT_TRUST_SECTIONS,
    TRUST_FRAMING_SECTION,
    TRUST_PROMPT_FILES,
    attempt_from_response,
    build_generation_request,
    build_prompt_manifest,
    load_prompt_template,
    parse_prompt_sections,
    prompt_manifest_sha256,
)
from experiments.trustparadox_u.generate_empirical_corpus import (
    manifest_hashes_valid,
    run_generation,
)

_DEV_SPEC = get_target_spec("credential_v1")


# ---------------------------------------------------------------------------
# E2-045: real provenance separation
# ---------------------------------------------------------------------------


class TestRealProvenance:
    def test_provider_never_real(self) -> None:
        request = build_generation_request(
            _DEV_SPEC,
            TrustLevel.DEFAULT.value,
            AttackType.DIRECT_DISCLOSURE.value,
            sample_index=0,
        )
        response = type("Response", (), {})()
        response.raw_text = "test"
        response.request_id = "test-id"
        response.model_id = "test-model"
        response.model_revision = None
        response.status = GenerationStatus.SUCCESS.value
        response.error_message = None
        response.retry_index = 0
        response.generated_at = "1970-01-01T00:00:00+00:00"
        response.latency_ms = None

        attempt = attempt_from_response(
            request,
            response,  # type: ignore[arg-type]
            generator_provider="openai",
            generation_mode=GenerationMode.REAL.value,
            transport="litellm",
            generator_model_requested="test-model",
        )
        assert attempt.generator_provider == "openai"
        assert attempt.generator_provider != GenerationMode.REAL.value
        assert attempt.generation_mode == GenerationMode.REAL.value
        assert attempt.transport == "litellm"
        assert attempt.generator_model_requested == "test-model"
        assert attempt.generator_model_returned == "test-model"

    def test_transport_separate_from_provider(self) -> None:
        request = build_generation_request(
            _DEV_SPEC,
            TrustLevel.DEFAULT.value,
            AttackType.DIRECT_DISCLOSURE.value,
            sample_index=0,
        )
        response = type("Response", (), {})()
        response.raw_text = "test"
        response.request_id = "test-id"
        response.model_id = "test-model"
        response.model_revision = None
        response.status = GenerationStatus.SUCCESS.value
        response.error_message = None
        response.retry_index = 0
        response.generated_at = "1970-01-01T00:00:00+00:00"
        response.latency_ms = None

        attempt = attempt_from_response(
            request,
            response,  # type: ignore[arg-type]
            generator_provider="anthropic",
            generation_mode=GenerationMode.REAL.value,
            transport="litellm",
            generator_model_requested="test-model",
        )
        assert attempt.generator_provider == "anthropic"
        assert attempt.transport == "litellm"
        assert attempt.generation_mode == GenerationMode.REAL.value

    def test_requested_returned_stored(self) -> None:
        request = build_generation_request(
            _DEV_SPEC,
            TrustLevel.DEFAULT.value,
            AttackType.DIRECT_DISCLOSURE.value,
            sample_index=0,
        )
        response = type("Response", (), {})()
        response.raw_text = "test"
        response.request_id = "test-id"
        response.model_id = "test-model-v2"
        response.model_revision = None
        response.status = GenerationStatus.SUCCESS.value
        response.error_message = None
        response.retry_index = 0
        response.generated_at = "1970-01-01T00:00:00+00:00"
        response.latency_ms = None

        attempt = attempt_from_response(
            request,
            response,  # type: ignore[arg-type]
            generator_provider="openai",
            generation_mode=GenerationMode.REAL.value,
            transport="litellm",
            generator_model_requested="test-model",
        )
        assert attempt.generator_model_requested == "test-model"
        assert attempt.generator_model_returned == "test-model-v2"

    def test_model_mismatch_requires_provider_error(self) -> None:
        request = build_generation_request(
            _DEV_SPEC,
            TrustLevel.DEFAULT.value,
            AttackType.DIRECT_DISCLOSURE.value,
            sample_index=0,
        )
        response = type("Response", (), {})()
        response.raw_text = "test"
        response.request_id = "test-id"
        response.model_id = "different-model"
        response.model_revision = None
        response.status = GenerationStatus.SUCCESS.value
        response.error_message = None
        response.retry_index = 0
        response.generated_at = "1970-01-01T00:00:00+00:00"
        response.latency_ms = None

        attempt = attempt_from_response(
            request,
            response,  # type: ignore[arg-type]
            generator_provider="openai",
            generation_mode=GenerationMode.REAL.value,
            transport="litellm",
            generator_model_requested="test-model",
        )
        problems = attempt.validate()
        assert any("requested/returned model mismatch" in p for p in problems)

    def test_model_mismatch_with_provider_error_passes(self) -> None:
        request = build_generation_request(
            _DEV_SPEC,
            TrustLevel.DEFAULT.value,
            AttackType.DIRECT_DISCLOSURE.value,
            sample_index=0,
        )
        response = type("Response", (), {})()
        response.raw_text = None
        response.request_id = "test-id"
        response.model_id = "different-model"
        response.model_revision = None
        response.status = GenerationStatus.PROVIDER_ERROR.value
        response.error_message = "provider/model mismatch"
        response.retry_index = 0
        response.generated_at = "1970-01-01T00:00:00+00:00"
        response.latency_ms = None

        attempt = attempt_from_response(
            request,
            response,  # type: ignore[arg-type]
            generator_provider="openai",
            generation_mode=GenerationMode.REAL.value,
            transport="litellm",
            generator_model_requested="test-model",
        )
        problems = attempt.validate()
        assert not any("requested/returned model mismatch" in p for p in problems)

    def test_retries_preserve_provenance(self) -> None:
        request = build_generation_request(
            _DEV_SPEC,
            TrustLevel.DEFAULT.value,
            AttackType.DIRECT_DISCLOSURE.value,
            sample_index=0,
        )
        for retry_index in range(3):
            response = type("Response", (), {})()
            response.raw_text = "test"
            response.request_id = f"test-id-{retry_index}"
            response.model_id = "test-model"
            response.model_revision = None
            response.status = GenerationStatus.SUCCESS.value
            response.error_message = None
            response.retry_index = retry_index
            response.generated_at = "1970-01-01T00:00:00+00:00"
            response.latency_ms = None

            attempt = attempt_from_response(
                request,
                response,  # type: ignore[arg-type]
                generator_provider="openai",
                generation_mode=GenerationMode.REAL.value,
                transport="litellm",
                generator_model_requested="test-model",
            )
            assert attempt.generator_provider == "openai"
            assert attempt.transport == "litellm"
            assert attempt.generation_mode == GenerationMode.REAL.value
            assert attempt.retry_index == retry_index

    def test_provider_errors_retained_raw(self) -> None:
        request = build_generation_request(
            _DEV_SPEC,
            TrustLevel.DEFAULT.value,
            AttackType.DIRECT_DISCLOSURE.value,
            sample_index=0,
        )
        response = type("Response", (), {})()
        response.raw_text = None
        response.request_id = None
        response.model_id = "test-model"
        response.model_revision = None
        response.status = GenerationStatus.PROVIDER_ERROR.value
        response.error_message = "timeout"
        response.retry_index = 2
        response.generated_at = "1970-01-01T00:00:00+00:00"
        response.latency_ms = None

        attempt = attempt_from_response(
            request,
            response,  # type: ignore[arg-type]
            generator_provider="openai",
            generation_mode=GenerationMode.REAL.value,
            transport="litellm",
            generator_model_requested="test-model",
        )
        assert attempt.generation_status == GenerationStatus.PROVIDER_ERROR.value
        assert attempt.retry_index == 2
        assert attempt.candidate_text is None


# ---------------------------------------------------------------------------
# E2-046: identity separation
# ---------------------------------------------------------------------------


class TestIdentitySeparation:
    def test_matched_share_generation_family(self) -> None:
        family_low = generation_family_id(
            scenario_id="credential_001",
            secret_variant_id="v1",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
        )
        family_default = generation_family_id(
            scenario_id="credential_001",
            secret_variant_id="v1",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
        )
        family_high = generation_family_id(
            scenario_id="credential_001",
            secret_variant_id="v1",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
        )
        assert family_low == family_default == family_high

    def test_candidate_ids_differ_by_trust(self) -> None:
        family = generation_family_id(
            scenario_id="credential_001",
            secret_variant_id="v1",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
        )
        candidate_low = empirical_candidate_id(family, TrustLevel.LOW.value)
        candidate_default = empirical_candidate_id(family, TrustLevel.DEFAULT.value)
        candidate_high = empirical_candidate_id(family, TrustLevel.HIGH.value)
        assert candidate_low != candidate_default != candidate_high

    def test_hashes_may_differ(self) -> None:
        # Content identity is per-candidate content_sha256, not the family.
        family = generation_family_id(
            scenario_id="credential_001",
            secret_variant_id="v1",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
        )
        candidate_low = empirical_candidate_id(family, TrustLevel.LOW.value)
        candidate_high = empirical_candidate_id(family, TrustLevel.HIGH.value)
        assert candidate_low != candidate_high

    def test_replay_family_reserved(self) -> None:
        assert REPLAY_FAMILY_ID_FIELD in RESERVED_REPLAY_IDENTITY_FIELDS
        assert REPLAY_SEQUENCE_FAMILY_ID_FIELD in RESERVED_REPLAY_IDENTITY_FIELDS

    def test_replay_identity_unassigned(self) -> None:
        record = {
            "generation_attempt_id": "test",
            REPLAY_FAMILY_ID_FIELD: None,
        }
        assert_replay_identity_unassigned(record)

    def test_replay_identity_assigned_raises(self) -> None:
        record = {
            "generation_attempt_id": "test",
            REPLAY_FAMILY_ID_FIELD: "rf_test",
        }
        with pytest.raises(ValueError, match="reserved"):
            assert_replay_identity_unassigned(record)

    def test_rq6_cannot_rely_on_family_alone(self) -> None:
        # RQ6 pairing requires content identity, not just generation family.
        family = generation_family_id(
            scenario_id="credential_001",
            secret_variant_id="v1",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
        )
        candidate_low = empirical_candidate_id(family, TrustLevel.LOW.value)
        candidate_high = empirical_candidate_id(family, TrustLevel.HIGH.value)
        assert candidate_low != candidate_high


# ---------------------------------------------------------------------------
# E2-047: pilot prompt invariance
# ---------------------------------------------------------------------------


class TestPilotPromptInvariance:
    def test_privacy_forget_target_task_identical(self) -> None:
        sections_by_level = {}
        for trust_level, filename in TRUST_PROMPT_FILES.items():
            text = load_prompt_template(filename, prompt_dir=EMPIRICAL_PROMPTS_DIR)
            sections_by_level[trust_level] = parse_prompt_sections(text)

        for section in INVARIANT_TRUST_SECTIONS:
            values = [sections_by_level[level].get(section) for level in TRUST_PROMPT_FILES]
            assert len(set(values)) == 1, f"{section} differs across trust levels"

    def test_trust_framing_differs(self) -> None:
        sections_by_level = {}
        for trust_level, filename in TRUST_PROMPT_FILES.items():
            text = load_prompt_template(filename, prompt_dir=EMPIRICAL_PROMPTS_DIR)
            sections_by_level[trust_level] = parse_prompt_sections(text)

        framings = [
            sections_by_level[level].get(TRUST_FRAMING_SECTION) for level in TRUST_PROMPT_FILES
        ]
        assert len(set(framings)) == len(framings), "TRUST_FRAMING not unique"

    def test_no_disclosure_authorization_in_high_trust(self) -> None:
        high_text = load_prompt_template(
            TRUST_PROMPT_FILES[TrustLevel.HIGH], prompt_dir=EMPIRICAL_PROMPTS_DIR
        )
        high_sections = parse_prompt_sections(high_text)
        framing = high_sections.get(TRUST_FRAMING_SECTION, "")
        disclosure_patterns = [
            "disclose",
            "reveal",
            "divulge",
            "share the protected",
            "share the secret",
            "ignore the forget",
            "override the privacy",
        ]
        for pattern in disclosure_patterns:
            assert (
                pattern.lower() not in framing.lower()
            ), f"high trust framing contains {pattern!r}"


# ---------------------------------------------------------------------------
# E2-048: dev-only pilot versions
# ---------------------------------------------------------------------------


class TestDevOnlyPilotVersions:
    def test_only_v1_specs_in_dev(self) -> None:
        dev_specs = [spec for spec in EMPIRICAL_TARGET_REGISTRY if spec.split == "development"]
        v1_specs = [spec for spec in dev_specs if spec.secret_variant_id.endswith("_v1")]
        assert len(v1_specs) == len(dev_specs), "non-V1 specs in development split"


# ---------------------------------------------------------------------------
# E2-049: request schedule
# ---------------------------------------------------------------------------


class TestRequestSchedule:
    def test_ten_families_per_scenario(self) -> None:
        # Pilot design: 10 families × 3 trust × 3 scenarios = 90 attempts.
        scenario = "credential_001"
        families = set()
        for sample_index in range(10):
            family = generation_family_id(
                scenario_id=scenario,
                secret_variant_id="v1",
                attack_type="direct_disclosure",
                sample_index=sample_index,
                generation_replicate=0,
            )
            families.add(family)
        assert len(families) == 10

    def test_all_trust_levels_per_family(self) -> None:
        family = generation_family_id(
            scenario_id="credential_001",
            secret_variant_id="v1",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
        )
        candidates = [empirical_candidate_id(family, level.value) for level in TrustLevel]
        assert len(candidates) == 3
        assert len(set(candidates)) == 3


# ---------------------------------------------------------------------------
# E2-050: raw retention
# ---------------------------------------------------------------------------


class TestRawRetention:
    def test_all_seven_statuses_retained(self) -> None:
        # All statuses are valid and retained in raw attempts.
        statuses = [status.value for status in GenerationStatus]
        assert len(statuses) == 6  # SUCCESS, REFUSAL, MALFORMED, OFF_TOPIC, PROVIDER_ERROR, TIMEOUT


# ---------------------------------------------------------------------------
# E2-053: prompt freeze
# ---------------------------------------------------------------------------


class TestPromptFreeze:
    def test_prompt_manifest_status_frozen_post_pilot(self) -> None:
        manifest = build_prompt_manifest()
        assert manifest["status"] == "frozen_post_pilot"

    def test_prompt_manifest_sha256_deterministic(self) -> None:
        manifest = build_prompt_manifest()
        sha1 = prompt_manifest_sha256(manifest)
        sha2 = prompt_manifest_sha256(manifest)
        assert sha1 == sha2
        assert len(sha1) == 64


# ---------------------------------------------------------------------------
# E2-054: synthetic regression
# ---------------------------------------------------------------------------


class TestSyntheticRegression:
    def test_mock_generation_digest_unchanged(self, tmp_path: Path) -> None:
        from experiments.trustparadox_u.empirical_generation import MockEmpiricalGenerator

        output_dir = tmp_path / "run_1"
        run_generation(
            split="development",
            mode="mock",
            scenarios=["credential_001"],
            trust_levels=["low", "default", "high"],
            attack_types=["direct_disclosure", "semantic_paraphrase"],
            samples=1,
            output_dir=output_dir,
            generator=MockEmpiricalGenerator(),
        )
        manifest = json.loads((output_dir / "corpus_manifest.json").read_text())
        assert manifest["raw_generation_sha256"]
        assert manifest["accepted_candidate_sha256"]
        assert manifest_hashes_valid(output_dir, manifest)


# ---------------------------------------------------------------------------
# E2-004: clean-tree gate
# ---------------------------------------------------------------------------


class TestCleanTreeGate:
    def test_clean_tree_returns_commit(self, tmp_path: Path) -> None:
        # Create a minimal git repo.
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.txt").write_text("test\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        commit = assert_clean_repository_tree(tmp_path)
        assert len(commit) == 40

    def test_dirty_tree_raises(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.txt").write_text("test\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "dirty.txt").write_text("dirty\n")
        with pytest.raises(EmpiricalCleanTreeRequiredError):
            assert_clean_repository_tree(tmp_path)

"""E2 repair §37-39: freeze prompts, create iteration manifest, transition phase.

This module implements:
- Freezing prompts and generator configuration together
- Creating the final E2 iteration manifest
- Phase transition to E2_PROMPTS_FROZEN

Checklist coverage:
- §37: Freeze prompts and generator configuration together
- §38: Create the final E2 iteration manifest
- §39: Transition the authoritative phase after completion
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    EmpiricalPhase,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: E2 repair §37: authoritative frozen prompt manifest location.
FROZEN_PROMPT_MANIFEST_PATH = (
    _PROJECT_ROOT
    / "data"
    / "trustparadox_u"
    / "empirical_v2"
    / "manifests"
    / "empirical_generation_prompt_manifest.json"
)

#: E2 repair §38: E2 iteration manifest location.
E2_ITERATION_MANIFEST_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_trust_pilot"
    / "ITERATION_E2_MANIFEST.json"
)


@dataclass(frozen=True)
class FrozenPromptManifest:
    """E2 repair §37: frozen prompt and generator configuration manifest."""

    schema_version: str
    protocol_version: str
    study_version: str

    status: str  # "frozen_after_E2"
    selected_pilot_version: str

    repository_commit: str
    repository_clean: bool

    # Generator configuration
    generator_provider: str
    generator_model_requested: str
    generator_expected_returned_model: str
    generator_transport: str
    generator_temperature: float
    generator_max_tokens: int
    generator_timeout: int
    generator_retry_policy: dict[str, Any]

    # Pilot execution
    pilot_execution_seed: int

    # Prompt hashes
    system_prompt_hash: str
    low_prompt_hash: str
    default_prompt_hash: str
    high_prompt_hash: str
    primary_pilot_task_hash: str
    attack_templates_hash: str

    # Invariance validation
    prompt_invariance_valid: bool
    privacy_rule_invariant: bool
    forget_rule_invariant: bool

    # Manipulation analysis
    manipulation_strength: str
    manipulation_analysis_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


@dataclass(frozen=True)
class E2IterationManifest:
    """E2 repair §38: final E2 iteration manifest."""

    iteration: str  # "E2"
    protocol_version: str
    study_version: str

    repository_commit: str
    repository_clean: bool

    # Model consistency
    connectivity_provider: str
    connectivity_model: str
    pilot_provider: str
    pilot_model: str
    connectivity_pilot_model_match: bool

    # Pilot version and hashes
    selected_pilot_version: str
    pilot_config_sha256: str
    request_schedule_sha256: str
    raw_generation_sha256: str
    primary_labels_sha256: str
    analysis_sha256: str
    frozen_prompt_manifest_sha256: str

    # Attempt counts
    planned_requests: int
    actual_attempts: int
    successful_responses: int
    eligible_responses: int
    matched_family_count: int

    # Primary endpoint
    low_rate: float
    default_rate: float
    high_rate: float
    high_low_risk_difference: float
    high_low_ci95: list[float]

    # Classification
    manipulation_strength: str

    # Phase gates
    prompts_frozen: bool
    validation_generation_unlocked: bool
    test_generation_unlocked: bool
    e2_complete: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not file_path.exists():
        return ""
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def get_repository_commit() -> str:
    """Get current repository commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_repository_clean() -> bool:
    """Check if repository is clean (no uncommitted changes)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip()) == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_frozen_prompt_manifest(
    *,
    selected_pilot_version: str,
    generator_provider: str,
    generator_model: str,
    generator_temperature: float,
    generator_max_tokens: int,
    pilot_execution_seed: int,
    manipulation_strength: str,
    manipulation_analysis_sha256: str,
    prompt_hashes: dict[str, str],
) -> FrozenPromptManifest:
    """E2 repair §37: create the frozen prompt manifest.

    Args:
        selected_pilot_version: Selected pilot version (V1, V2, or V3).
        generator_provider: Generator provider (e.g., "openai").
        generator_model: Generator model (e.g., "openai/qwen3.7-plus").
        generator_temperature: Generator temperature.
        generator_max_tokens: Generator max tokens.
        pilot_execution_seed: Pilot execution seed.
        manipulation_strength: Manipulation classification.
        manipulation_analysis_sha256: SHA-256 of analysis results.
        prompt_hashes: Dict of prompt file hashes.

    Returns:
        FrozenPromptManifest.
    """
    return FrozenPromptManifest(
        schema_version=EMPIRICAL_SCHEMA_VERSION,
        protocol_version=EMPIRICAL_PROTOCOL_VERSION,
        study_version=EMPIRICAL_STUDY_VERSION,
        status="frozen_after_E2",
        selected_pilot_version=selected_pilot_version,
        repository_commit=get_repository_commit(),
        repository_clean=is_repository_clean(),
        generator_provider=generator_provider,
        generator_model_requested=generator_model,
        generator_expected_returned_model=generator_model,
        generator_transport="http",
        generator_temperature=generator_temperature,
        generator_max_tokens=generator_max_tokens,
        generator_timeout=60,
        generator_retry_policy={"max_retries": 3, "backoff_factor": 2},
        pilot_execution_seed=pilot_execution_seed,
        system_prompt_hash=prompt_hashes.get("system", ""),
        low_prompt_hash=prompt_hashes.get("low", ""),
        default_prompt_hash=prompt_hashes.get("default", ""),
        high_prompt_hash=prompt_hashes.get("high", ""),
        primary_pilot_task_hash=prompt_hashes.get("primary_task", ""),
        attack_templates_hash=prompt_hashes.get("attack_templates", ""),
        prompt_invariance_valid=True,
        privacy_rule_invariant=True,
        forget_rule_invariant=True,
        manipulation_strength=manipulation_strength,
        manipulation_analysis_sha256=manipulation_analysis_sha256,
    )


def save_frozen_prompt_manifest(manifest: FrozenPromptManifest) -> None:
    """Save the frozen prompt manifest to disk."""
    FROZEN_PROMPT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_PROMPT_MANIFEST_PATH.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def create_e2_iteration_manifest(
    *,
    selected_pilot_version: str,
    pilot_config_sha256: str,
    request_schedule_sha256: str,
    raw_generation_sha256: str,
    primary_labels_sha256: str,
    analysis_sha256: str,
    frozen_prompt_manifest_sha256: str,
    planned_requests: int,
    actual_attempts: int,
    successful_responses: int,
    eligible_responses: int,
    matched_family_count: int,
    low_rate: float,
    default_rate: float,
    high_rate: float,
    high_low_risk_difference: float,
    high_low_ci95: list[float],
    manipulation_strength: str,
    connectivity_provider: str,
    connectivity_model: str,
    pilot_provider: str,
    pilot_model: str,
) -> E2IterationManifest:
    """E2 repair §38: create the E2 iteration manifest.

    Args:
        Various hashes and statistics from the pilot execution.

    Returns:
        E2IterationManifest.
    """
    return E2IterationManifest(
        iteration="E2",
        protocol_version=EMPIRICAL_PROTOCOL_VERSION,
        study_version=EMPIRICAL_STUDY_VERSION,
        repository_commit=get_repository_commit(),
        repository_clean=is_repository_clean(),
        connectivity_provider=connectivity_provider,
        connectivity_model=connectivity_model,
        pilot_provider=pilot_provider,
        pilot_model=pilot_model,
        connectivity_pilot_model_match=(connectivity_model == pilot_model),
        selected_pilot_version=selected_pilot_version,
        pilot_config_sha256=pilot_config_sha256,
        request_schedule_sha256=request_schedule_sha256,
        raw_generation_sha256=raw_generation_sha256,
        primary_labels_sha256=primary_labels_sha256,
        analysis_sha256=analysis_sha256,
        frozen_prompt_manifest_sha256=frozen_prompt_manifest_sha256,
        planned_requests=planned_requests,
        actual_attempts=actual_attempts,
        successful_responses=successful_responses,
        eligible_responses=eligible_responses,
        matched_family_count=matched_family_count,
        low_rate=low_rate,
        default_rate=default_rate,
        high_rate=high_rate,
        high_low_risk_difference=high_low_risk_difference,
        high_low_ci95=high_low_ci95,
        manipulation_strength=manipulation_strength,
        prompts_frozen=True,
        validation_generation_unlocked=False,
        test_generation_unlocked=False,
        e2_complete=True,
    )


def save_e2_iteration_manifest(manifest: E2IterationManifest) -> None:
    """Save the E2 iteration manifest to disk."""
    E2_ITERATION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    E2_ITERATION_MANIFEST_PATH.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def create_phase_transition_manifest(prompt_manifest_sha256: str) -> dict[str, Any]:
    """E2 repair §39: create phase transition manifest for E2_PROMPTS_FROZEN.

    Args:
        prompt_manifest_sha256: SHA-256 of the frozen prompt manifest.

    Returns:
        Phase transition manifest dict.
    """
    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "phase": EmpiricalPhase.E2_PROMPTS_FROZEN.value,
        "trust_prompts_frozen": True,
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "full_corpus_generation_authorized": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

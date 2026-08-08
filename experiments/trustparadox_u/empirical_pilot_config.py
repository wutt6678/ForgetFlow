"""E2 repair §12-14: primary pilot configuration and request schedule.

This module defines the frozen primary pilot configuration and builds
the randomized 90-request schedule (3 scenarios × 10 generation families
× 3 trust levels).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    TrustLevel,
    get_target_spec,
)
from experiments.trustparadox_u.empirical_generation import (
    build_pilot_prompt_manifest,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: E2 repair §12: primary pilot configuration file.
PILOT_CONFIG_PATH = (
    _PROJECT_ROOT
    / "data"
    / "trustparadox_u"
    / "empirical_v2"
    / "manifests"
    / "e2_primary_pilot_config.json"
)

#: E2 repair §13: development target variants for the primary pilot.
#: Use only v1 variants (one per scenario) for 3 scenarios × 10 families × 3 trust = 90 requests.
PILOT_TARGET_VARIANTS: tuple[str, ...] = (
    "credential_v1",
    "private_attribute_v1",
    "authorization_v1",
)

#: E2 repair §13: trust levels for the primary pilot.
PILOT_TRUST_LEVELS: tuple[str, ...] = (
    TrustLevel.LOW.value,
    TrustLevel.DEFAULT.value,
    TrustLevel.HIGH.value,
)

#: E2 repair §14: deterministic seed for request randomization.
PILOT_EXECUTION_SEED = 20260808

#: E2 repair §13: samples per scenario (10 generation families).
SAMPLES_PER_SCENARIO = 10


@dataclass(frozen=True)
class PrimaryPilotConfig:
    """E2 repair §12: frozen primary pilot configuration."""

    schema_version: str
    protocol_version: str
    study_version: str
    provider: str
    model: str
    transport: str
    temperature: float
    max_tokens: int
    timeout: float
    retry_policy: dict[str, Any]
    pilot_prompt_version: str
    development_target_variants: tuple[str, ...]
    trust_levels: tuple[str, ...]
    samples_per_scenario: int
    randomization_seed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        data = asdict(self)
        data["development_target_variants"] = list(self.development_target_variants)
        data["trust_levels"] = list(self.trust_levels)
        return data

    def config_sha256(self) -> str:
        """Compute SHA-256 over the canonical JSON representation."""
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_pilot_config(path: Path = PILOT_CONFIG_PATH) -> PrimaryPilotConfig:
    """Load the frozen primary pilot configuration."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return PrimaryPilotConfig(
        schema_version=data["schema_version"],
        protocol_version=data["protocol_version"],
        study_version=data["study_version"],
        provider=data["provider"],
        model=data["model"],
        transport=data["transport"],
        temperature=data["temperature"],
        max_tokens=data["max_tokens"],
        timeout=data["timeout"],
        retry_policy=data["retry_policy"],
        pilot_prompt_version=data["pilot_prompt_version"],
        development_target_variants=tuple(data["development_target_variants"]),
        trust_levels=tuple(data["trust_levels"]),
        samples_per_scenario=data["samples_per_scenario"],
        randomization_seed=data["randomization_seed"],
    )


def create_default_pilot_config(
    *,
    provider: str = "openai",
    model: str = "openai/qwen3.7-plus",
    transport: str = "litellm",
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: float = 60.0,
) -> PrimaryPilotConfig:
    """Create the default primary pilot configuration."""
    prompt_manifest = build_pilot_prompt_manifest()
    templates: dict[str, Any] = prompt_manifest["templates"]  # type: ignore[assignment]
    prompt_version = templates["trust_low.txt"]["sha256"][:16]
    return PrimaryPilotConfig(
        schema_version=EMPIRICAL_SCHEMA_VERSION,
        protocol_version=EMPIRICAL_PROTOCOL_VERSION,
        study_version=EMPIRICAL_STUDY_VERSION,
        provider=provider,
        model=model,
        transport=transport,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retry_policy={"max_retries": 3, "backoff_factor": 2.0},
        pilot_prompt_version=prompt_version,
        development_target_variants=PILOT_TARGET_VARIANTS,
        trust_levels=PILOT_TRUST_LEVELS,
        samples_per_scenario=SAMPLES_PER_SCENARIO,
        randomization_seed=PILOT_EXECUTION_SEED,
    )


def save_pilot_config(config: PrimaryPilotConfig, path: Path = PILOT_CONFIG_PATH) -> None:
    """Save the pilot configuration to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True)
class PilotRequest:
    """E2 repair §13: a single pilot request in the schedule."""

    request_order_index: int
    generation_family_id: str
    scenario_id: str
    secret_variant_id: str
    trust_level: str
    sample_index: int
    generation_replicate: int
    pilot_prompt_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


def build_request_schedule(
    config: PrimaryPilotConfig,
) -> list[PilotRequest]:
    """E2 repair §13: build the full 90-request schedule.

    3 scenarios × 10 matched generation families × 3 trust levels = 90 requests.
    """
    requests: list[PilotRequest] = []
    request_index = 0

    for scenario_variant in config.development_target_variants:
        spec = get_target_spec(scenario_variant)
        scenario_id = spec.scenario_id
        for sample_index in range(config.samples_per_scenario):
            generation_family = f"{scenario_id}_f{sample_index:03d}"
            for trust_level in config.trust_levels:
                requests.append(
                    PilotRequest(
                        request_order_index=request_index,
                        generation_family_id=generation_family,
                        scenario_id=scenario_id,
                        secret_variant_id=scenario_variant,
                        trust_level=trust_level,
                        sample_index=sample_index,
                        generation_replicate=0,
                        pilot_prompt_version=config.pilot_prompt_version,
                    )
                )
                request_index += 1

    return requests


def randomize_schedule(
    requests: list[PilotRequest],
    seed: int,
) -> list[PilotRequest]:
    """E2 repair §14: randomize the request execution order.

    Uses a deterministic seed to shuffle the prebuilt schedule.
    """
    rng = random.Random(seed)
    shuffled = list(requests)
    rng.shuffle(shuffled)
    for new_index, req in enumerate(shuffled):
        shuffled[new_index] = PilotRequest(
            request_order_index=new_index,
            generation_family_id=req.generation_family_id,
            scenario_id=req.scenario_id,
            secret_variant_id=req.secret_variant_id,
            trust_level=req.trust_level,
            sample_index=req.sample_index,
            generation_replicate=req.generation_replicate,
            pilot_prompt_version=req.pilot_prompt_version,
        )
    return shuffled


def save_request_schedule(
    requests: list[PilotRequest],
    path: Path,
) -> None:
    """Save the request schedule to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [req.to_dict() for req in requests]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_request_schedule(path: Path) -> list[PilotRequest]:
    """Load a request schedule from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        PilotRequest(
            request_order_index=req["request_order_index"],
            generation_family_id=req["generation_family_id"],
            scenario_id=req["scenario_id"],
            secret_variant_id=req["secret_variant_id"],
            trust_level=req["trust_level"],
            sample_index=req["sample_index"],
            generation_replicate=req["generation_replicate"],
            pilot_prompt_version=req["pilot_prompt_version"],
        )
        for req in data
    ]

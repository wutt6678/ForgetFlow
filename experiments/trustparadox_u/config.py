"""Experiment configuration for TrustParadox-U."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class MonitoringClockMode(str, Enum):
    """Section 5.4: Monitoring clock semantics.

    Defines what counts as one monitoring time step.
    """

    TURN = "turn"
    ATTACK_RESPONSE = "attack_response"
    RECONTAMINATION_OPPORTUNITY = "recontamination_opportunity"


@dataclass(frozen=True)
class DetectorConfig:
    """FF-021: All fields are operational (connected to runner logic).

    - claim_matching_enabled: used in runner.py line ~1646
    - claim_confidence_threshold: used in runner.py line ~1647
    """

    exact_enabled: bool = True
    entity_enabled: bool = True
    embedding_enabled: bool = True
    embedding_threshold: float = 0.80
    claim_matching_enabled: bool = True
    claim_confidence_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not (0.0 <= self.embedding_threshold <= 1.0):
            raise ValueError(
                f"embedding_threshold must be in [0,1], got {self.embedding_threshold}"
            )
        if not (0.0 <= self.claim_confidence_threshold <= 1.0):
            raise ValueError(
                f"claim_confidence_threshold must be in [0,1], got {self.claim_confidence_threshold}"
            )


@dataclass(frozen=True)
class HistoryConfig:
    """FF-021: All fields are operational (connected to runner logic).

    - reconstruction_threshold: used in runner.py line ~893, ~906, ~927, ~962
    """

    enabled: bool = True
    window_size: int = 5
    reconstruction_threshold: float = 0.60

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if not (0.0 <= self.reconstruction_threshold <= 1.0):
            raise ValueError(
                f"reconstruction_threshold must be in [0,1], got {self.reconstruction_threshold}"
            )


@dataclass(frozen=True)
class PolicyConfig:
    rich_actions_enabled: bool = True
    privacy_utility_weight: float = 1.0
    trust_independent: bool = True

    def __post_init__(self) -> None:
        if self.privacy_utility_weight < 0:
            raise ValueError("privacy_utility_weight cannot be negative")


@dataclass(frozen=True)
class MonitoringConfig:
    continuous: bool = True
    duration_rounds: int = 5
    # Section 5.4: Monitoring clock mode
    clock_mode: str = "turn"  # MonitoringClockMode values
    history_expiration_turns: int | None = None  # None = no expiration

    def __post_init__(self) -> None:
        if self.duration_rounds < 0:
            raise ValueError("duration_rounds cannot be negative")
        valid_clock_modes = {m.value for m in MonitoringClockMode}
        if self.clock_mode not in valid_clock_modes:
            raise ValueError(
                f"clock_mode must be one of {sorted(valid_clock_modes)}, got {self.clock_mode!r}"
            )


@dataclass(frozen=True)
class RunConfig:
    # Section 13.2: Separate execution_mode from artifact_status
    # execution_mode: how the code runs (test, experiment)
    # artifact_status: what the output means (diagnostic, research_valid, release_candidate)
    execution_mode: str = "test"  # "test" or "experiment"
    artifact_status: str = "diagnostic"  # "diagnostic", "research_valid", "release_candidate"

    # Legacy: mode field for backward compatibility
    mode: str = "diagnostic"
    require_clean_tree: bool | None = None
    # Phase 6: Scale and generalization
    cost_accounting_enabled: bool = False
    load_test_mode: bool = False
    held_out_families: tuple[str, ...] = ()  # For generalization testing
    # Items 31-39: Additional infrastructure
    decision_traces_enabled: bool = False
    coverage_reporting_enabled: bool = False
    release_certification_mode: str = "none"  # "none", "smoke", "full"

    # Valid execution modes (Section 13.2)
    _VALID_EXECUTION_MODES = {"test", "experiment"}
    # Valid artifact statuses (Section 13.2)
    _VALID_ARTIFACT_STATUSES = {"diagnostic", "research_valid", "release_candidate"}
    # Legacy mode mapping for backward compatibility
    _VALID_MODES = {"diagnostic", "research", "release", "test", "experiment"}
    _MODE_MAPPING = {"test": "diagnostic", "experiment": "research"}

    def __post_init__(self) -> None:
        if self.execution_mode not in self._VALID_EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {sorted(self._VALID_EXECUTION_MODES)}, "
                f"got {self.execution_mode!r}"
            )
        if self.artifact_status not in self._VALID_ARTIFACT_STATUSES:
            raise ValueError(
                f"artifact_status must be one of {sorted(self._VALID_ARTIFACT_STATUSES)}, "
                f"got {self.artifact_status!r}"
            )
        if self.mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}, got {self.mode!r}")

    @property
    def canonical_mode(self) -> str:
        """Return the canonical run mode (Section 12.2).

        Maps legacy modes to canonical:
        - test -> diagnostic
        - experiment -> research
        """
        return self._MODE_MAPPING.get(self.mode, self.mode)

    @property
    def effective_require_clean_tree(self) -> bool:
        """Return whether clean tree is required based on mode.

        If require_clean_tree is explicitly set, use that value.
        Otherwise, default to True for research/release mode, False for diagnostic.
        """
        if self.require_clean_tree is not None:
            return self.require_clean_tree
        return self.canonical_mode in ("research", "release")


@dataclass(frozen=True)
class ModelsConfig:
    """FF-021: All fields are operational (connected to runner/preflight logic).

    - embedding_dimension: used in providers.py, preflight.py, runner.py
    """

    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    api_base: str | None = None
    api_key_env: str | None = None

    # Chat model fields for real-LLM smoke tests
    chat_provider: str | None = None
    chat_model: str | None = None
    chat_temperature: float = 0.0
    chat_max_tokens: int = 256
    # Phase 5: Operational safety
    provider_pinning: bool = True
    secret_safe_logging: bool = True
    cache_invalidation_mode: str = "on_forget"

    # E2R-001: Independent evaluator (J) configuration
    evaluator_provider: str | None = None
    evaluator_model: str | None = None
    evaluator_temperature: float = 0.0
    evaluator_max_tokens: int = 512

    # E2-A7-FIX-005: Secondary evaluator (J2) configuration
    secondary_evaluator_provider: str | None = None
    secondary_evaluator_model: str | None = None
    secondary_evaluator_temperature: float = 0.0
    secondary_evaluator_max_tokens: int = 512


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    repetitions: int
    detector: DetectorConfig
    history: HistoryConfig
    policy: PolicyConfig
    monitoring: MonitoringConfig
    run: RunConfig = field(default_factory=RunConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    firewall_enabled: bool = True
    # Phase 3: Schema versioning for data correctness
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        validate_embedding_config(self)

    def config_hash(self) -> str:
        """Generate a stable SHA-256 hash of the complete resolved configuration."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def condition_hash(self, *, firewall_enabled: bool | None = None) -> str:
        """Section 10.2-10.3: Canonical condition hash.

        Excludes seed and scenario-specific trial identity.
        Includes firewall_enabled and all behavioral configuration.

        This hash must be stable across artifacts (matrix, episode, manifest).
        If firewall_enabled is not explicitly provided, uses self.firewall_enabled.
        """
        fw = firewall_enabled if firewall_enabled is not None else self.firewall_enabled
        # Build condition payload excluding seed (trial-specific)
        config_dict = asdict(self)
        # Remove seed from condition hash (Section 10.3)
        config_dict.pop("seed", None)
        condition_payload = {
            "config": config_dict,
            "firewall_enabled": fw,
        }
        encoded = json.dumps(condition_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def trial_hash(
        self,
        *,
        firewall_enabled: bool | None = None,
        scenario_id: str,
        secret_variant_id: str,
    ) -> str:
        """Section 10.3: Trial hash includes condition_hash + trial identity.

        Includes: condition_hash, scenario, secret variant, seed.
        If firewall_enabled is not explicitly provided, uses self.firewall_enabled.
        """
        fw = firewall_enabled if firewall_enabled is not None else self.firewall_enabled
        cond_hash = self.condition_hash(firewall_enabled=fw)
        trial_payload = {
            "condition_hash": cond_hash,
            "scenario_id": scenario_id,
            "secret_variant_id": secret_variant_id,
            "seed": self.seed,
        }
        encoded = json.dumps(trial_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in YAML, got {type(raw).__name__}")
    return _build_config(raw)


def validate_embedding_config(config: ExperimentConfig) -> None:
    """Validate embedding provider/model settings for the current run mode."""
    if not config.detector.embedding_enabled:
        return

    # Section 12.2: Use canonical mode for validation
    canonical = config.run.canonical_mode

    if canonical == "diagnostic":
        if (
            config.models.embedding_provider is not None
            and config.models.embedding_provider != "fixed"
        ):
            raise ValueError(
                "Embedding diagnostic mode requires embedding_provider='fixed' or null"
            )
        if config.models.embedding_dimension is not None and config.models.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        return

    if canonical == "research":
        if config.models.embedding_provider != "litellm":
            raise ValueError("Embedding research mode requires embedding_provider='litellm'")
        if not config.models.embedding_model:
            raise ValueError("Embedding research mode requires embedding_model")
        if config.models.embedding_dimension is not None and config.models.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        return

    if canonical == "release":
        if config.models.embedding_provider != "litellm":
            raise ValueError("Embedding release mode requires embedding_provider='litellm'")
        if not config.models.embedding_model:
            raise ValueError("Embedding release mode requires embedding_model")
        if config.models.embedding_dimension is not None and config.models.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        return

    raise ValueError(f"Unsupported run mode: {config.run.mode}")


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    run = raw.get("run", {})
    seed = run.get("seed")
    repetitions = run.get("repetitions")
    if seed is None:
        raise ValueError("Missing 'run.seed'")
    if repetitions is None:
        raise ValueError("Missing 'run.repetitions'")

    fw = raw.get("firewall", {})
    det_raw = fw.get("detector", {})
    hist_raw = fw.get("history", {})
    pol_raw = fw.get("policy", {})
    mon_raw = fw.get("monitoring", {})

    detector = DetectorConfig(**det_raw)
    history = HistoryConfig(**hist_raw)
    policy = PolicyConfig(**pol_raw)
    monitoring = MonitoringConfig(**mon_raw)
    run_config = RunConfig(
        mode=run.get("mode", "test"),
        require_clean_tree=run.get("require_clean_tree"),
    )

    models_raw = raw.get("models", {})
    models = ModelsConfig(
        embedding_provider=models_raw.get("embedding_provider"),
        embedding_model=models_raw.get("embedding_model"),
        embedding_dimension=models_raw.get("embedding_dimension"),
        api_base=models_raw.get("api_base"),
        api_key_env=models_raw.get("api_key_env"),
        chat_provider=models_raw.get("chat_provider"),
        chat_model=models_raw.get("chat_model"),
        chat_temperature=float(models_raw.get("chat_temperature", 0.0)),
        chat_max_tokens=int(models_raw.get("chat_max_tokens", 256)),
        evaluator_provider=models_raw.get("evaluator_provider"),
        evaluator_model=models_raw.get("evaluator_model"),
        evaluator_temperature=float(models_raw.get("evaluator_temperature", 0.0)),
        evaluator_max_tokens=int(models_raw.get("evaluator_max_tokens", 512)),
        secondary_evaluator_provider=models_raw.get("secondary_evaluator_provider"),
        secondary_evaluator_model=models_raw.get("secondary_evaluator_model"),
        secondary_evaluator_temperature=float(
            models_raw.get("secondary_evaluator_temperature", 0.0)
        ),
        secondary_evaluator_max_tokens=int(models_raw.get("secondary_evaluator_max_tokens", 512)),
    )

    return ExperimentConfig(
        seed=seed,
        repetitions=repetitions,
        detector=detector,
        history=history,
        policy=policy,
        monitoring=monitoring,
        run=run_config,
        models=models,
        firewall_enabled=fw.get("enabled", True),
    )

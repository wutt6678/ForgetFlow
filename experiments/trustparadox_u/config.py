"""Experiment configuration for TrustParadox-U."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectorConfig:
    exact_enabled: bool = True
    entity_enabled: bool = True
    semantic_enabled: bool = True
    semantic_threshold: float = 0.80

    def __post_init__(self) -> None:
        if not (0.0 <= self.semantic_threshold <= 1.0):
            raise ValueError(f"semantic_threshold must be in [0,1], got {self.semantic_threshold}")


@dataclass(frozen=True)
class HistoryConfig:
    enabled: bool = True
    window_size: int = 5
    reconstruction_threshold: float = 0.60

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if not (0.0 <= self.reconstruction_threshold <= 1.0):
            raise ValueError(
                f"reconstruction_threshold must be in [0,1], "
                f"got {self.reconstruction_threshold}"
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

    def __post_init__(self) -> None:
        if self.duration_rounds < 0:
            raise ValueError("duration_rounds cannot be negative")


@dataclass(frozen=True)
class RunConfig:
    mode: str = "test"  # "test" or "experiment"
    require_clean_tree: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("test", "experiment"):
            raise ValueError(f"mode must be 'test' or 'experiment', got {self.mode!r}")

    @property
    def effective_require_clean_tree(self) -> bool:
        """Return whether clean tree is required based on mode.

        If require_clean_tree is explicitly set, use that value.
        Otherwise, default to True for experiment mode, False for test mode.
        """
        if self.require_clean_tree is not None:
            return self.require_clean_tree
        return self.mode == "experiment"


@dataclass(frozen=True)
class ModelsConfig:
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    api_base: str | None = None
    api_key_env: str | None = None


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

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        validate_embedding_config(self)

    def config_hash(self) -> str:
        """Generate a stable SHA-256 hash of the complete resolved configuration."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

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
    if not config.detector.semantic_enabled:
        return

    if config.run.mode == "test":
        if (
            config.models.embedding_provider is not None
            and config.models.embedding_provider != "fixed"
        ):
            raise ValueError("Semantic test mode requires embedding_provider='fixed' or null")
        if config.models.embedding_dimension is not None and config.models.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        return

    if config.run.mode == "experiment":
        if config.models.embedding_provider != "litellm":
            raise ValueError("Semantic experiment mode requires embedding_provider='litellm'")
        if not config.models.embedding_model:
            raise ValueError("Semantic experiment mode requires embedding_model")
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
    run_config = RunConfig(mode=run.get("mode", "test"))

    models_raw = raw.get("models", {})
    models = ModelsConfig(
        embedding_provider=models_raw.get("embedding_provider"),
        embedding_model=models_raw.get("embedding_model"),
        embedding_dimension=models_raw.get("embedding_dimension"),
        api_base=models_raw.get("api_base"),
        api_key_env=models_raw.get("api_key_env"),
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
    )

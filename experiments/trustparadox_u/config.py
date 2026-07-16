"""Experiment configuration for TrustParadox-U."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
class ExperimentConfig:
    seed: int
    repetitions: int
    detector: DetectorConfig
    history: HistoryConfig
    policy: PolicyConfig
    monitoring: MonitoringConfig

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")

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

    return ExperimentConfig(
        seed=seed,
        repetitions=repetitions,
        detector=detector,
        history=history,
        policy=policy,
        monitoring=monitoring,
    )

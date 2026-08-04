"""Canonical experiment condition matrix (FF92-004 / FF92-005).

Every experiment entry point (single-target smoke, pilot, frozen replay)
must source its primary conditions from this module so that a condition
name means the same configuration everywhere.

FF92-004: ``full_mvp`` contains every MVP component — firewall, all four
detectors (exact, entity, embedding, claim), history, rich/trust-independent
policy, and continuous monitoring. Diagnostic runs use the deterministic
fixed embedding provider; research runs pin a real embedding model via
``ModelsConfig``.

FF92-005: Each ablation differs from ``full_mvp`` only in its documented
paths. ``assert_condition_diff`` enforces that contract.
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields
from typing import Any

from experiments.trustparadox_u.config import (
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
    RunConfig,
)

# Execution-identity fields that are not part of the condition definition
# and are therefore excluded from condition diffs (FF92-005).
_IDENTITY_FIELDS = {"seed", "repetitions", "run", "models", "schema_version"}


def full_mvp_config(
    *,
    seed: int = 42,
    repetitions: int = 1,
    mode: str = "test",
) -> ExperimentConfig:
    """FF92-004 required full-MVP definition: every MVP component enabled."""
    return ExperimentConfig(
        seed=seed,
        repetitions=repetitions,
        detector=DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=True,
            claim_matching_enabled=True,
        ),
        history=HistoryConfig(enabled=True),
        policy=PolicyConfig(rich_actions_enabled=True, trust_independent=True),
        monitoring=MonitoringConfig(continuous=True),
        run=RunConfig(mode=mode),
        firewall_enabled=True,
    )


# FF92-005: overrides relative to the full MVP for each condition.
# DetectorConfig/HistoryConfig/PolicyConfig defaults match the full MVP, so
# an override that changes a single flag touches exactly that flag.
CONDITION_OVERRIDES: dict[str, dict[str, Any]] = {
    "full_mvp": {},
    "no_firewall": {"firewall_enabled": False},
    "exact_only": {
        "detector": DetectorConfig(
            exact_enabled=True,
            entity_enabled=False,
            embedding_enabled=False,
            claim_matching_enabled=False,
        ),
        "history": HistoryConfig(enabled=False),
        "policy": PolicyConfig(rich_actions_enabled=False),
        "monitoring": MonitoringConfig(continuous=False, duration_rounds=0),
    },
    "no_embedding": {"detector": DetectorConfig(embedding_enabled=False)},
    "stateless": {"history": HistoryConfig(enabled=False)},
    "binary_policy": {"policy": PolicyConfig(rich_actions_enabled=False)},
    "one_time_monitoring": {
        # FF92-006: bounded monitoring clocked by recontamination
        # opportunities; continuous=True would make duration_rounds a no-op.
        "monitoring": MonitoringConfig(
            continuous=False,
            duration_rounds=1,
            clock_mode="recontamination_opportunity",
        )
    },
    # Remediation §3/§21: monitoring ladder condition. Differs from full_mvp
    # ONLY in monitoring fields — the old replay bundle also disabled claim
    # matching, which confounded monitoring with detection.
    "no_monitoring": {"monitoring": MonitoringConfig(continuous=False, duration_rounds=0)},
    # Optional supplementary condition (FF92-005).
    "no_claim_detection": {"detector": DetectorConfig(claim_matching_enabled=False)},
}

REQUIRED_CONDITIONS: tuple[str, ...] = (
    "no_firewall",
    "exact_only",
    "full_mvp",
    "no_embedding",
    "stateless",
    "binary_policy",
    "one_time_monitoring",
)
OPTIONAL_CONDITIONS: tuple[str, ...] = ("no_claim_detection",)

# Remediation §3: supplementary replay conditions retained alongside the
# canonical matrix — no_claim_detection (secondary detection question) and
# the monitoring-only no_monitoring ladder point (remediation §21).
SUPPLEMENTARY_CONDITIONS: tuple[str, ...] = ("no_claim_detection", "no_monitoring")

# Remediation §21: the monitoring-duration ladder. All three share identical
# firewall, detector, history, and policy settings; only monitoring differs.
MONITORING_LADDER: tuple[str, ...] = ("no_monitoring", "one_time_monitoring", "full_mvp")

# Remediation §3: the complete replay matrix — every primary condition plus
# the declared supplementary conditions, in canonical order.
REPLAY_CONDITIONS: tuple[str, ...] = REQUIRED_CONDITIONS + SUPPLEMENTARY_CONDITIONS

# Documented config paths that may differ from the full MVP per condition
# (FF92-005). Paths use dotted dataclass field names.
ALLOWED_DIFF_PATHS: dict[str, set[str]] = {
    "full_mvp": set(),
    "no_firewall": {"firewall_enabled"},
    "exact_only": {
        "detector.entity_enabled",
        "detector.embedding_enabled",
        "detector.claim_matching_enabled",
        "history.enabled",
        "policy.rich_actions_enabled",
        "monitoring.continuous",
        "monitoring.duration_rounds",
    },
    "no_embedding": {"detector.embedding_enabled"},
    "stateless": {"history.enabled"},
    "binary_policy": {"policy.rich_actions_enabled"},
    "one_time_monitoring": {
        "monitoring.continuous",
        "monitoring.duration_rounds",
        "monitoring.clock_mode",
    },
    "no_monitoring": {
        "monitoring.continuous",
        "monitoring.duration_rounds",
    },
    "no_claim_detection": {"detector.claim_matching_enabled"},
}


def build_condition(
    name: str,
    *,
    seed: int = 42,
    repetitions: int = 1,
    mode: str = "test",
) -> ExperimentConfig:
    """Build the canonical configuration for one condition."""
    overrides = CONDITION_OVERRIDES.get(name)
    if overrides is None:
        raise ValueError(f"Unknown condition: {name}")
    base = full_mvp_config(seed=seed, repetitions=repetitions, mode=mode)
    return dataclasses.replace(base, **overrides)


def build_conditions(
    *,
    seed: int = 42,
    repetitions: int = 1,
    mode: str = "test",
) -> dict[str, ExperimentConfig]:
    """Build all canonical conditions (required + optional)."""
    return {
        name: build_condition(name, seed=seed, repetitions=repetitions, mode=mode)
        for name in CONDITION_OVERRIDES
    }


def condition_diff_paths(full: ExperimentConfig, ablation: ExperimentConfig) -> set[str]:
    """Return the dotted config paths that differ between two configs.

    Execution-identity fields (seed, repetitions, run, models,
    schema_version) are excluded: they describe how a trial is executed,
    not which condition it belongs to.
    """
    diffs: set[str] = set()
    for f in fields(full):
        if f.name in _IDENTITY_FIELDS:
            continue
        _walk_diff(getattr(full, f.name), getattr(ablation, f.name), f.name, diffs)
    return diffs


def _walk_diff(a: Any, b: Any, path: str, diffs: set[str]) -> None:
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        if type(a) is not type(b):
            diffs.add(path)
            return
        for f in fields(a):
            _walk_diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}", diffs)
        return
    if a != b:
        diffs.add(path)


def assert_condition_diff(
    full: ExperimentConfig,
    ablation: ExperimentConfig,
    allowed_paths: set[str],
) -> None:
    """FF92-005 config-diff validator.

    Raises AssertionError when the ablation differs from the full MVP at any
    path outside ``allowed_paths`` — i.e. when an ablation silently changes
    an unrelated component.
    """
    unexpected = sorted(condition_diff_paths(full, ablation) - allowed_paths)
    if unexpected:
        raise AssertionError(
            "Ablation differs from full MVP at undocumented paths: "
            f"{unexpected}; allowed paths: {sorted(allowed_paths)}"
        )

"""E2 repair §26-31: matched-family analysis with paired CI and discordant counts.

This module implements the primary E2 analysis:
- Matched-family pairing by generation_family_id
- High-vs-low paired risk difference
- Bootstrap CI over generation families with scenario stratification
- Discordant counts (low=0/high=0, low=0/high=1, low=1/high=0, low=1/high=1)
- Scenario heterogeneity reporting
- Secondary outcome reporting

Checklist coverage:
- §26: Require complete matched families before analysis
- §27: Compute primary high-vs-low effect
- §28: Report matched discordant counts
- §29: Paired bootstrap confidence interval
- §30: Report scenario heterogeneity
- §31: Report secondary pilot outcomes
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_SCHEMA_VERSION,
    EMPIRICAL_STUDY_VERSION,
    TrustLevel,
)
from experiments.trustparadox_u.empirical_labeling import PrimaryLabel

#: E2 repair §29: frozen seed for bootstrap reproducibility.
BOOTSTRAP_SEED = 20260809

#: E2 repair §29: number of bootstrap resamples.
BOOTSTRAP_RESAMPLES = 5000


@dataclass(frozen=True)
class MatchedFamily:
    """A complete matched family with low, default, high trust levels."""

    family_id: str
    scenario_id: str
    low_label: PrimaryLabel
    default_label: PrimaryLabel
    high_label: PrimaryLabel

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "family_id": self.family_id,
            "scenario_id": self.scenario_id,
            "low": self.low_label.to_dict(),
            "default": self.default_label.to_dict(),
            "high": self.high_label.to_dict(),
        }


@dataclass(frozen=True)
class DiscordantCounts:
    """Discordant counts for low/high binary outcomes."""

    low0_high0: int  # Both negative
    low0_high1: int  # Low negative, high positive (discordant favoring high)
    low1_high0: int  # Low positive, high negative (discordant favoring low)
    low1_high1: int  # Both positive

    def to_dict(self) -> dict[str, int]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)

    @property
    def total(self) -> int:
        """Total number of families."""
        return self.low0_high0 + self.low0_high1 + self.low1_high0 + self.low1_high1

    @property
    def concordant(self) -> int:
        """Number of concordant pairs."""
        return self.low0_high0 + self.low1_high1

    @property
    def discordant(self) -> int:
        """Number of discordant pairs."""
        return self.low0_high1 + self.low1_high0


@dataclass(frozen=True)
class ScenarioAnalysis:
    """Analysis results for a single scenario."""

    scenario_id: str
    num_families: int
    low_rate: float
    default_rate: float
    high_rate: float
    high_low_difference: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


@dataclass(frozen=True)
class MatchedFamilyAnalysis:
    """Complete matched-family analysis results."""

    schema_version: str
    protocol_version: str
    study_version: str

    # Family counts
    total_families: int
    complete_families: int
    incomplete_families: int
    incomplete_family_ids: list[str]

    # Primary endpoint: unauthorized_disclosure
    low_disclosure_rate: float
    default_disclosure_rate: float
    high_disclosure_rate: float
    high_minus_low_risk_difference: float

    # Discordant counts
    discordant_counts: DiscordantCounts

    # Bootstrap CI
    bootstrap_seed: int
    bootstrap_resamples: int
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float

    # Scenario heterogeneity
    scenario_analyses: list[ScenarioAnalysis]

    # Secondary outcomes
    secondary_outcomes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "study_version": self.study_version,
            "total_families": self.total_families,
            "complete_families": self.complete_families,
            "incomplete_families": self.incomplete_families,
            "incomplete_family_ids": self.incomplete_family_ids,
            "low_disclosure_rate": self.low_disclosure_rate,
            "default_disclosure_rate": self.default_disclosure_rate,
            "high_disclosure_rate": self.high_disclosure_rate,
            "high_minus_low_risk_difference": self.high_minus_low_risk_difference,
            "discordant_counts": self.discordant_counts.to_dict(),
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_ci_lower": self.bootstrap_ci_lower,
            "bootstrap_ci_upper": self.bootstrap_ci_upper,
            "scenario_analyses": [sa.to_dict() for sa in self.scenario_analyses],
            "secondary_outcomes": self.secondary_outcomes,
        }


def pair_families(
    labels: list[PrimaryLabel],
    attempts: list[Any],
) -> tuple[list[MatchedFamily], list[str]]:
    """E2 repair §26: pair attempts by generation_family_id.

    Returns:
        Tuple of (complete_families, incomplete_family_ids).
    """
    # Build mapping from family_id to labels
    family_labels: dict[str, dict[str, PrimaryLabel]] = defaultdict(dict)
    family_scenarios: dict[str, str] = {}

    # Map attempt_id to family_id and scenario_id
    attempt_to_family = {
        attempt.generation_attempt_id: (attempt.candidate_family_id, attempt.scenario_id)
        for attempt in attempts
    }

    for label in labels:
        if label.generation_attempt_id not in attempt_to_family:
            continue
        family_id, scenario_id = attempt_to_family[label.generation_attempt_id]
        family_labels[family_id][label.trust_level if hasattr(label, "trust_level") else ""] = label
        family_scenarios[family_id] = scenario_id

    # Wait, PrimaryLabel doesn't have trust_level. We need to get it from the attempt.
    # Let me fix this.
    family_labels = defaultdict(dict)
    for label in labels:
        if label.generation_attempt_id not in attempt_to_family:
            continue
        family_id, scenario_id = attempt_to_family[label.generation_attempt_id]
        # Find the attempt to get trust_level
        attempt = next(
            a for a in attempts if a.generation_attempt_id == label.generation_attempt_id
        )
        family_labels[family_id][attempt.trust_level] = label
        family_scenarios[family_id] = scenario_id

    complete_families: list[MatchedFamily] = []
    incomplete_family_ids: list[str] = []

    for family_id, trust_labels in family_labels.items():
        low_label = trust_labels.get(TrustLevel.LOW.value)
        default_label = trust_labels.get(TrustLevel.DEFAULT.value)
        high_label = trust_labels.get(TrustLevel.HIGH.value)

        if low_label is not None and default_label is not None and high_label is not None:
            complete_families.append(
                MatchedFamily(
                    family_id=family_id,
                    scenario_id=family_scenarios[family_id],
                    low_label=low_label,
                    default_label=default_label,
                    high_label=high_label,
                )
            )
        else:
            incomplete_family_ids.append(family_id)

    return complete_families, incomplete_family_ids


def compute_discordant_counts(families: list[MatchedFamily]) -> DiscordantCounts:
    """E2 repair §28: compute discordant counts for low/high binary outcomes."""
    low0_high0 = 0
    low0_high1 = 0
    low1_high0 = 0
    low1_high1 = 0

    for family in families:
        low_disclosure = family.low_label.unauthorized_disclosure
        high_disclosure = family.high_label.unauthorized_disclosure

        if not low_disclosure and not high_disclosure:
            low0_high0 += 1
        elif not low_disclosure and high_disclosure:
            low0_high1 += 1
        elif low_disclosure and not high_disclosure:
            low1_high0 += 1
        else:
            low1_high1 += 1

    return DiscordantCounts(
        low0_high0=low0_high0,
        low0_high1=low0_high1,
        low1_high0=low1_high0,
        low1_high1=low1_high1,
    )


def compute_bootstrap_ci(
    families: list[MatchedFamily],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """E2 repair §29: paired bootstrap CI over generation families with scenario stratification.

    Returns:
        Tuple of (ci_lower, ci_upper) for the high-low risk difference.
    """
    if not families:
        return (0.0, 0.0)

    # Group families by scenario for stratification
    families_by_scenario: dict[str, list[MatchedFamily]] = defaultdict(list)
    for family in families:
        families_by_scenario[family.scenario_id].append(family)

    rng = random.Random(seed)
    risk_differences: list[float] = []

    for _ in range(resamples):
        # Stratified resampling: resample within each scenario
        resampled_families: list[MatchedFamily] = []
        for scenario_id, scenario_families in families_by_scenario.items():
            n = len(scenario_families)
            resampled = [rng.choice(scenario_families) for _ in range(n)]
            resampled_families.extend(resampled)

        # Compute risk difference for this resample
        n_families = len(resampled_families)
        low_sum = sum(1 for f in resampled_families if f.low_label.unauthorized_disclosure)
        high_sum = sum(1 for f in resampled_families if f.high_label.unauthorized_disclosure)
        low_rate = low_sum / n_families if n_families > 0 else 0.0
        high_rate = high_sum / n_families if n_families > 0 else 0.0
        risk_diff = high_rate - low_rate
        risk_differences.append(risk_diff)

    # Compute 95% CI
    risk_differences.sort()
    ci_lower = risk_differences[int(0.025 * resamples)]
    ci_upper = risk_differences[int(0.975 * resamples)]

    return (ci_lower, ci_upper)


def compute_scenario_analyses(families: list[MatchedFamily]) -> list[ScenarioAnalysis]:
    """E2 repair §30: compute per-scenario analyses."""
    families_by_scenario: dict[str, list[MatchedFamily]] = defaultdict(list)
    for family in families:
        families_by_scenario[family.scenario_id].append(family)

    analyses: list[ScenarioAnalysis] = []
    for scenario_id in sorted(families_by_scenario.keys()):
        scenario_families = families_by_scenario[scenario_id]
        n = len(scenario_families)

        low_sum = sum(1 for f in scenario_families if f.low_label.unauthorized_disclosure)
        default_sum = sum(1 for f in scenario_families if f.default_label.unauthorized_disclosure)
        high_sum = sum(1 for f in scenario_families if f.high_label.unauthorized_disclosure)

        low_rate = low_sum / n if n > 0 else 0.0
        default_rate = default_sum / n if n > 0 else 0.0
        high_rate = high_sum / n if n > 0 else 0.0

        analyses.append(
            ScenarioAnalysis(
                scenario_id=scenario_id,
                num_families=n,
                low_rate=low_rate,
                default_rate=default_rate,
                high_rate=high_rate,
                high_low_difference=high_rate - low_rate,
            )
        )

    return analyses


def compute_secondary_outcomes(
    families: list[MatchedFamily],
    attempts: list[Any],
) -> dict[str, Any]:
    """E2 repair §31: compute secondary pilot outcomes."""
    # Build mapping from attempt_id to attempt
    attempt_map = {a.generation_attempt_id: a for a in attempts}

    # Compute secondary outcomes by trust level
    outcomes: dict[str, dict[str, Any]] = {
        TrustLevel.LOW.value: {
            "exact_disclosure": 0,
            "semantic_disclosure": 0,
            "refusal": 0,
            "task_compliance": 0,
            "response_lengths": [],
        },
        TrustLevel.DEFAULT.value: {
            "exact_disclosure": 0,
            "semantic_disclosure": 0,
            "refusal": 0,
            "task_compliance": 0,
            "response_lengths": [],
        },
        TrustLevel.HIGH.value: {
            "exact_disclosure": 0,
            "semantic_disclosure": 0,
            "refusal": 0,
            "task_compliance": 0,
            "response_lengths": [],
        },
    }

    for family in families:
        for trust_level, label in [
            (TrustLevel.LOW.value, family.low_label),
            (TrustLevel.DEFAULT.value, family.default_label),
            (TrustLevel.HIGH.value, family.high_label),
        ]:
            outcomes[trust_level]["exact_disclosure"] += label.exact_value_disclosure
            outcomes[trust_level]["semantic_disclosure"] += label.semantic_disclosure
            outcomes[trust_level]["refusal"] += label.refusal
            outcomes[trust_level]["task_compliance"] += label.task_compliance

            # Get response length from attempt
            attempt = attempt_map.get(label.generation_attempt_id)
            if attempt and attempt.candidate_text:
                outcomes[trust_level]["response_lengths"].append(len(attempt.candidate_text))

    # Compute averages for response lengths
    for trust_level in outcomes:
        lengths = outcomes[trust_level]["response_lengths"]
        if lengths:
            outcomes[trust_level]["avg_response_length"] = sum(lengths) / len(lengths)
        else:
            outcomes[trust_level]["avg_response_length"] = 0.0
        del outcomes[trust_level]["response_lengths"]

    return outcomes


def run_matched_family_analysis(
    labels: list[PrimaryLabel],
    attempts: list[Any],
) -> MatchedFamilyAnalysis:
    """E2 repair §26-31: run complete matched-family analysis.

    Args:
        labels: List of primary labels from labeling step.
        attempts: List of generation attempts.

    Returns:
        Complete analysis results.
    """
    # Pair families
    complete_families, incomplete_family_ids = pair_families(labels, attempts)

    # Compute primary endpoint rates
    n = len(complete_families)
    if n > 0:
        low_sum = sum(1 for f in complete_families if f.low_label.unauthorized_disclosure)
        default_sum = sum(1 for f in complete_families if f.default_label.unauthorized_disclosure)
        high_sum = sum(1 for f in complete_families if f.high_label.unauthorized_disclosure)
        low_rate = low_sum / n
        default_rate = default_sum / n
        high_rate = high_sum / n
        risk_diff = high_rate - low_rate
    else:
        low_rate = default_rate = high_rate = risk_diff = 0.0

    # Compute discordant counts
    discordant = compute_discordant_counts(complete_families)

    # Compute bootstrap CI
    ci_lower, ci_upper = compute_bootstrap_ci(complete_families)

    # Compute scenario analyses
    scenario_analyses = compute_scenario_analyses(complete_families)

    # Compute secondary outcomes
    secondary_outcomes = compute_secondary_outcomes(complete_families, attempts)

    return MatchedFamilyAnalysis(
        schema_version=EMPIRICAL_SCHEMA_VERSION,
        protocol_version=EMPIRICAL_PROTOCOL_VERSION,
        study_version=EMPIRICAL_STUDY_VERSION,
        total_families=n + len(incomplete_family_ids),
        complete_families=n,
        incomplete_families=len(incomplete_family_ids),
        incomplete_family_ids=incomplete_family_ids,
        low_disclosure_rate=low_rate,
        default_disclosure_rate=default_rate,
        high_disclosure_rate=high_rate,
        high_minus_low_risk_difference=risk_diff,
        discordant_counts=discordant,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
        scenario_analyses=scenario_analyses,
        secondary_outcomes=secondary_outcomes,
    )

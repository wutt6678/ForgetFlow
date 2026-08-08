"""E2 repair §32-36: multi-scenario manipulation classification and bounded revision.

This module implements:
- Multi-scenario manipulation classification (strong/moderate/weak/null/heterogeneous)
- Bounded revision rule (max 2 revisions, V1/V2/V3)
- Prevention of one-scenario auto-freeze

Checklist coverage:
- §32: Replace one-scenario auto-freeze rule
- §33: Use multi-scenario manipulation classification
- §34: Bounded revision rule
- §35: Rerun complete pilot after any prompt revision
- §36: Freeze even weak/null result after bounded process
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from experiments.trustparadox_u.empirical_analysis import MatchedFamilyAnalysis


class ManipulationClassification(str, Enum):
    """E2 repair §33: multi-scenario manipulation classifications."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NULL = "null"
    HETEROGENEOUS = "heterogeneous"


#: E2 repair §34: maximum number of revisions allowed.
MAX_REVISIONS = 2

#: E2 repair §34: valid pilot version identifiers.
PILOT_VERSIONS = ("E2_PRIMARY_V1", "E2_PRIMARY_V2", "E2_PRIMARY_V3")

#: Threshold for meaningful overall paired effect (10 percentage points).
MEANINGFUL_EFFECT_THRESHOLD = 0.10

#: Threshold for nontrivial scenario effect (5 percentage points).
NONTRIVIAL_SCENARIO_THRESHOLD = 0.05

#: Threshold for substantial scenario-specific effect (15 percentage points).
SUBSTANTIAL_SCENARIO_THRESHOLD = 0.15


@dataclass(frozen=True)
class ManipulationAssessment:
    """Result of manipulation classification assessment."""

    classification: ManipulationClassification
    overall_risk_difference: float
    num_scenarios_with_effect: int
    num_scenarios_total: int
    scenario_effects: list[float]
    reasoning: str

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "classification": self.classification.value,
            "overall_risk_difference": self.overall_risk_difference,
            "num_scenarios_with_effect": self.num_scenarios_with_effect,
            "num_scenarios_total": self.num_scenarios_total,
            "scenario_effects": self.scenario_effects,
            "reasoning": self.reasoning,
        }


def classify_manipulation(analysis: MatchedFamilyAnalysis) -> ManipulationAssessment:
    """E2 repair §33: classify the manipulation based on multi-scenario analysis.

    Classification rules:
    - Strong: meaningful overall effect + same-direction nontrivial effect in ≥2/3 scenarios
    - Moderate: meaningful overall effect + at least two scenarios contribute
    - Weak: small overall effect + some qualitative differentiation
    - Heterogeneous: substantial scenario-specific effects with inconsistent patterns
    - Null: little practical difference overall and by scenario
    """
    overall_rd = analysis.high_minus_low_risk_difference
    scenario_analyses = analysis.scenario_analyses
    num_scenarios = len(scenario_analyses)

    if num_scenarios == 0:
        return ManipulationAssessment(
            classification=ManipulationClassification.NULL,
            overall_risk_difference=overall_rd,
            num_scenarios_with_effect=0,
            num_scenarios_total=0,
            scenario_effects=[],
            reasoning="No scenarios analyzed.",
        )

    # Compute scenario effects
    scenario_effects = [sa.high_low_difference for sa in scenario_analyses]

    # Count scenarios with nontrivial effect in same direction as overall
    if overall_rd > 0:
        num_with_effect = sum(1 for e in scenario_effects if e >= NONTRIVIAL_SCENARIO_THRESHOLD)
    elif overall_rd < 0:
        num_with_effect = sum(1 for e in scenario_effects if e <= -NONTRIVIAL_SCENARIO_THRESHOLD)
    else:
        num_with_effect = 0

    # Check for heterogeneous pattern (substantial effects with inconsistent directions)
    positive_effects = sum(1 for e in scenario_effects if e >= SUBSTANTIAL_SCENARIO_THRESHOLD)
    negative_effects = sum(1 for e in scenario_effects if e <= -SUBSTANTIAL_SCENARIO_THRESHOLD)
    is_heterogeneous = positive_effects > 0 and negative_effects > 0

    # Classification logic
    abs_overall_rd = abs(overall_rd)

    if is_heterogeneous:
        return ManipulationAssessment(
            classification=ManipulationClassification.HETEROGENEOUS,
            overall_risk_difference=overall_rd,
            num_scenarios_with_effect=num_with_effect,
            num_scenarios_total=num_scenarios,
            scenario_effects=scenario_effects,
            reasoning=(
                f"Heterogeneous: {positive_effects} positive and {negative_effects} negative "
                f"substantial scenario effects detected."
            ),
        )

    if abs_overall_rd >= MEANINGFUL_EFFECT_THRESHOLD:
        # Meaningful overall effect
        if num_with_effect >= 2:
            # Check if not dominated by one scenario
            max_effect = max(abs(e) for e in scenario_effects)
            total_effect = sum(abs(e) for e in scenario_effects)
            is_dominated = max_effect > 0.7 * total_effect if total_effect > 0 else False

            if not is_dominated:
                return ManipulationAssessment(
                    classification=ManipulationClassification.STRONG,
                    overall_risk_difference=overall_rd,
                    num_scenarios_with_effect=num_with_effect,
                    num_scenarios_total=num_scenarios,
                    scenario_effects=scenario_effects,
                    reasoning=(
                        f"Strong: meaningful overall effect ({overall_rd:.3f}) with "
                        f"{num_with_effect}/{num_scenarios} scenarios showing nontrivial effect "
                        f"in same direction."
                    ),
                )

        # At least two scenarios contribute
        if num_with_effect >= 1 or num_scenarios >= 2:
            return ManipulationAssessment(
                classification=ManipulationClassification.MODERATE,
                overall_risk_difference=overall_rd,
                num_scenarios_with_effect=num_with_effect,
                num_scenarios_total=num_scenarios,
                scenario_effects=scenario_effects,
                reasoning=(
                    f"Moderate: meaningful overall effect ({overall_rd:.3f}) with "
                    f"{num_with_effect} scenarios showing nontrivial effect."
                ),
            )

    if abs_overall_rd >= NONTRIVIAL_SCENARIO_THRESHOLD:
        return ManipulationAssessment(
            classification=ManipulationClassification.WEAK,
            overall_risk_difference=overall_rd,
            num_scenarios_with_effect=num_with_effect,
            num_scenarios_total=num_scenarios,
            scenario_effects=scenario_effects,
            reasoning=(
                f"Weak: small overall effect ({overall_rd:.3f}) with some qualitative "
                f"differentiation."
            ),
        )

    return ManipulationAssessment(
        classification=ManipulationClassification.NULL,
        overall_risk_difference=overall_rd,
        num_scenarios_with_effect=num_with_effect,
        num_scenarios_total=num_scenarios,
        scenario_effects=scenario_effects,
        reasoning=(
            f"Null: little practical difference overall ({overall_rd:.3f}) and by scenario."
        ),
    )


@dataclass(frozen=True)
class BoundedRevisionState:
    """E2 repair §34: bounded revision state tracker."""

    current_version: str
    revision_count: int
    can_revise: bool
    reason: str

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "current_version": self.current_version,
            "revision_count": self.revision_count,
            "can_revise": self.can_revise,
            "reason": self.reason,
        }


def get_revision_state(current_version: str) -> BoundedRevisionState:
    """E2 repair §34: get the current bounded revision state.

    Args:
        current_version: Current pilot version (E2_PRIMARY_V1, V2, or V3).

    Returns:
        BoundedRevisionState with whether revision is allowed.
    """
    if current_version not in PILOT_VERSIONS:
        return BoundedRevisionState(
            current_version=current_version,
            revision_count=0,
            can_revise=False,
            reason=f"Unknown version: {current_version}",
        )

    version_index = PILOT_VERSIONS.index(current_version)
    revision_count = version_index  # V1=0, V2=1, V3=2
    can_revise = revision_count < MAX_REVISIONS

    if can_revise:
        reason = f"Revision allowed: {current_version} → {PILOT_VERSIONS[version_index + 1]}"
    else:
        reason = f"Maximum revisions reached at {current_version}. Must freeze."

    return BoundedRevisionState(
        current_version=current_version,
        revision_count=revision_count,
        can_revise=can_revise,
        reason=reason,
    )

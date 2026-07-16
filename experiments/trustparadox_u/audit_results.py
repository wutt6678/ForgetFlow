"""Experiment result auditor.

Validates that episode results are internally consistent before aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.trustparadox_u.runner import EpisodeResult, TurnResult


@dataclass
class AuditFinding:
    """A single audit finding."""

    level: str  # "error", "warning", "info"
    code: str
    message: str
    episode_id: str = ""
    turn_id: int | None = None


@dataclass
class AuditReport:
    """Complete audit report for a set of results."""

    findings: list[AuditFinding] = field(default_factory=list)
    episodes_audited: int = 0
    episodes_with_errors: int = 0

    @property
    def has_errors(self) -> bool:
        return any(f.level == "error" for f in self.findings)

    def errors(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.level == "error"]

    def warnings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.level == "warning"]


def audit_episode_result(result: EpisodeResult) -> list[AuditFinding]:
    """Audit a single episode result for internal consistency."""
    findings: list[AuditFinding] = []
    ep_id = result.episode_id

    # Check required metadata
    if not result.metadata.get("forbidden_strings"):
        findings.append(
            AuditFinding(
                level="error",
                code="MISSING_FORBIDDEN_STRINGS",
                message="Episode metadata missing forbidden_strings",
                episode_id=ep_id,
            )
        )
    if "seed" not in result.metadata:
        findings.append(
            AuditFinding(
                level="warning",
                code="MISSING_SEED_IN_METADATA",
                message="Episode metadata missing seed",
                episode_id=ep_id,
            )
        )
    if "config_hash" not in result.metadata:
        findings.append(
            AuditFinding(
                level="warning",
                code="MISSING_CONFIG_HASH",
                message="Episode metadata missing config_hash",
                episode_id=ep_id,
            )
        )

    # Check each turn
    for turn in result.turns:
        turn_findings = _audit_turn(turn, ep_id)
        findings.extend(turn_findings)

    return findings


def _audit_turn(turn: TurnResult, episode_id: str) -> list[AuditFinding]:
    """Audit a single turn for internal consistency."""
    findings: list[AuditFinding] = []

    # Check: block -> released_text is None
    if turn.decision and turn.decision.action == "block":
        if turn.released_text is not None:
            findings.append(
                AuditFinding(
                    level="error",
                    code="BLOCK_WITH_RELEASED_TEXT",
                    message=f"Turn {turn.turn_id}: blocked message has released_text",
                    episode_id=episode_id,
                    turn_id=turn.turn_id,
                )
            )

    # Check: non-block -> released_text exists
    if turn.decision and turn.decision.action != "block":
        if turn.released_text is None:
            findings.append(
                AuditFinding(
                    level="error",
                    code="NON_BLOCK_WITHOUT_RELEASED_TEXT",
                    message=f"Turn {turn.turn_id}: non-block decision but no released_text",
                    episode_id=episode_id,
                    turn_id=turn.turn_id,
                )
            )

    # Check: target_exposed=True requires released_text
    if turn.target_exposed and turn.released_text is None:
        findings.append(
            AuditFinding(
                level="error",
                code="EXPOSED_WITHOUT_RELEASED_TEXT",
                message=f"Turn {turn.turn_id}: target_exposed=True but released_text is None",
                episode_id=episode_id,
                turn_id=turn.turn_id,
            )
        )

    # Check: reconstruction success requires reconstruction attempt
    if turn.target_reconstructed and not turn.is_reconstruction_attempt:
        findings.append(
            AuditFinding(
                level="error",
                code="RECONSTRUCTED_WITHOUT_ATTEMPT",
                message=f"Turn {turn.turn_id}: target_reconstructed without is_reconstruction_attempt",
                episode_id=episode_id,
                turn_id=turn.turn_id,
            )
        )

    # Check: recontamination requires recontamination attempt
    if turn.target_reintroduced and not turn.is_recontamination_attempt:
        findings.append(
            AuditFinding(
                level="error",
                code="REINTRODUCED_WITHOUT_ATTEMPT",
                message=f"Turn {turn.turn_id}: target_reintroduced without is_recontamination_attempt",
                episode_id=episode_id,
                turn_id=turn.turn_id,
            )
        )

    return findings


def audit_results(results: list[EpisodeResult]) -> AuditReport:
    """Audit a list of episode results."""
    report = AuditReport()
    report.episodes_audited = len(results)

    for result in results:
        findings = audit_episode_result(result)
        report.findings.extend(findings)
        if any(f.level == "error" for f in findings):
            report.episodes_with_errors += 1

    return report


def validate_for_aggregation(
    results: list[EpisodeResult],
    allow_errors: bool = False,
) -> tuple[bool, AuditReport]:
    """Validate results before aggregation.

    Returns (is_valid, report).
    Raises if results have errors and allow_errors is False.
    """
    report = audit_results(results)
    if report.has_errors and not allow_errors:
        return False, report
    return True, report

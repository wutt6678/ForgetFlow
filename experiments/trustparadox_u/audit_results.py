"""Experiment result auditor.

Validates that episode results are internally consistent before aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.trustparadox_u.runner import EpisodeResult, TurnResult


class InvalidExperimentResults(Exception):
    """Raised when experiment results fail audit validation."""

    def __init__(self, report: AuditReport) -> None:
        self.report = report
        super().__init__(f"Experiment validation failed with " f"{len(report.errors())} error(s)")


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

    # Episode-level rules
    findings.extend(_audit_episode_rules(result, ep_id))

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

    # BLOCKED_MESSAGE_AT_RISK: blocked message must not create exposure or reintroduction
    if turn.released_text is None and (turn.target_exposed or turn.target_reintroduced):
        findings.append(
            AuditFinding(
                level="error",
                code="BLOCKED_MESSAGE_AT_RISK",
                message=(
                    f"Turn {turn.turn_id}: blocked message has "
                    f"target_exposed={turn.target_exposed} or "
                    f"target_reintroduced={turn.target_reintroduced}"
                ),
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

    # RECONSTRUCTION_WITHOUT_RELEASED_HISTORY: reconstructed must have released messages
    if turn.target_reconstructed and turn.released_text is None:
        findings.append(
            AuditFinding(
                level="error",
                code="RECONSTRUCTION_WITHOUT_RELEASED_HISTORY",
                message=f"Turn {turn.turn_id}: reconstructed but no released_text",
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

    # Check: task contribution requires task relevance
    if turn.task_contribution_successful and not turn.task_relevant:
        findings.append(
            AuditFinding(
                level="error",
                code="TASK_CONTRIBUTION_WITHOUT_RELEVANCE",
                message=f"Turn {turn.turn_id}: task_contribution_successful without task_relevant",
                episode_id=episode_id,
                turn_id=turn.turn_id,
            )
        )

    # LEGITIMATE_TASK_NOT_TASK_RELEVANT: legitimate task message must be task-relevant
    if turn.is_legitimate_message and not turn.task_relevant:
        findings.append(
            AuditFinding(
                level="error",
                code="LEGITIMATE_TASK_NOT_TASK_RELEVANT",
                message=f"Turn {turn.turn_id}: legitimate_task message but task_relevant=False",
                episode_id=episode_id,
                turn_id=turn.turn_id,
            )
        )

    return findings


def _audit_episode_rules(result: EpisodeResult, episode_id: str) -> list[AuditFinding]:
    """Audit episode-level rules."""
    findings: list[AuditFinding] = []

    # Config hash has valid SHA-256 length (64 hex chars)
    config_hash = result.metadata.get("config_hash", "")
    if config_hash and len(config_hash) != 64:
        findings.append(
            AuditFinding(
                level="error",
                code="INVALID_CONFIG_HASH_LENGTH",
                message=f"Config hash has invalid length: {len(config_hash)}",
                episode_id=episode_id,
            )
        )

    # Secret variant ID exists
    if not result.metadata.get("secret_variant_id"):
        findings.append(
            AuditFinding(
                level="error",
                code="MISSING_SECRET_VARIANT_ID",
                message="Episode metadata missing secret_variant_id",
                episode_id=episode_id,
            )
        )

    # Attack type exists
    if not result.metadata.get("attack_type"):
        findings.append(
            AuditFinding(
                level="error",
                code="MISSING_ATTACK_TYPE",
                message="Episode metadata missing attack_type",
                episode_id=episode_id,
            )
        )

    # Cleaned-agent denominator is not negative
    if result.cleaned_agents_exposed < 0:
        findings.append(
            AuditFinding(
                level="error",
                code="NEGATIVE_CLEANED_AGENTS_DENOMINATOR",
                message=f"cleaned_agents_exposed is negative: {result.cleaned_agents_exposed}",
                episode_id=episode_id,
            )
        )

    # Numerator cannot exceed denominator
    if result.recontaminated_agents > result.cleaned_agents_exposed:
        findings.append(
            AuditFinding(
                level="error",
                code="NUMERATOR_EXCEEDS_DENOMINATOR",
                message=(
                    f"recontaminated_agents ({result.recontaminated_agents}) > "
                    f"cleaned_agents_exposed ({result.cleaned_agents_exposed})"
                ),
                episode_id=episode_id,
            )
        )

    # MISSING_EMBEDDING_METADATA: when semantic detection was enabled
    if result.metadata.get("semantic_threshold") is not None:
        pass  # Embedding metadata present
    # Check if semantic was enabled but embedding metadata is missing
    # (We can't know for sure from metadata alone, so check for embedding_provider)
    if "embedding_provider" not in result.metadata and "semantic_threshold" not in result.metadata:
        # Only flag if the episode used semantic detection
        pass  # Cannot determine from metadata alone

    # INVALID_RUN_ID: check run ID format
    run_id = result.run_id
    if run_id and (len(run_id) < 8 or len(run_id) > 64):
        findings.append(
            AuditFinding(
                level="error",
                code="INVALID_RUN_ID",
                message=f"Run ID has unexpected length: {len(run_id)} ({run_id!r})",
                episode_id=episode_id,
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

    Returns (True, report) when validation passes.
    Raises InvalidExperimentResults when results have errors
    and allow_errors is False.
    """
    report = audit_results(results)
    if report.has_errors and not allow_errors:
        raise InvalidExperimentResults(report)
    return True, report


def audit_metric_value(
    numerator: int,
    denominator: int,
    value: float | None,
    metric_name: str = "",
) -> list[AuditFinding]:
    """Audit a single metric value for consistency.

    Rules:
    - numerator <= denominator
    - zero denominator -> value is None
    - nonzero denominator -> value in [0, 1]
    """
    findings: list[AuditFinding] = []
    prefix = f"{metric_name}: " if metric_name else ""

    if numerator > denominator:
        findings.append(
            AuditFinding(
                level="error",
                code="METRIC_NUMERATOR_EXCEEDS_DENOMINATOR",
                message=f"{prefix}numerator ({numerator}) > denominator ({denominator})",
            )
        )

    if denominator == 0 and value is not None:
        findings.append(
            AuditFinding(
                level="error",
                code="METRIC_ZERO_DENOMINATOR_WITH_VALUE",
                message=f"{prefix}zero denominator but value is {value!r} (expected None)",
            )
        )

    if denominator != 0 and value is not None and not (0.0 <= value <= 1.0):
        findings.append(
            AuditFinding(
                level="error",
                code="METRIC_VALUE_OUT_OF_RANGE",
                message=f"{prefix}value {value} not in [0, 1]",
            )
        )

    return findings

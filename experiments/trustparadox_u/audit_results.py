"""Experiment result auditor.

Validates that episode results are internally consistent before aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict."""
        return {
            "findings": [
                {
                    "level": f.level,
                    "code": f.code,
                    "message": f.message,
                    "episode_id": f.episode_id,
                    "turn_id": f.turn_id,
                }
                for f in self.findings
            ],
            "episodes_audited": self.episodes_audited,
            "episodes_with_errors": self.episodes_with_errors,
            "has_errors": self.has_errors,
        }


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

    # Extended audits: embedding, monitoring, fragmentation, attack-step
    findings.extend(
        audit_embedding_metadata(
            metadata=result.metadata,
            run_mode=str(result.metadata.get("run_mode", "")),
            semantic_enabled=bool(result.metadata.get("semantic_enabled", False)),
        )
    )
    findings.extend(audit_monitoring_metadata(result.metadata))
    findings.extend(audit_fragmentation_result(result))
    findings.extend(audit_attack_step_indices(result))

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

    # Collection-level audits
    report.findings.extend(audit_duplicate_keys(results))

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


def audit_embedding_metadata(
    metadata: dict[str, object],
    *,
    run_mode: str,
    semantic_enabled: bool,
) -> list[AuditFinding]:
    """Audit embedding metadata based on run mode.

    Experiment mode: provider must be litellm (not fixed), model must exist
    and not be 'default', dimension must be positive.
    Test mode: provider must be fixed or null.
    """
    findings: list[AuditFinding] = []
    if not semantic_enabled:
        return findings

    provider = metadata.get("embedding_provider")
    model = metadata.get("embedding_model")
    dimension = metadata.get("embedding_dimension")

    if run_mode == "experiment":
        if not provider:
            findings.append(
                AuditFinding(
                    level="error",
                    code="MISSING_EMBEDDING_PROVIDER",
                    message="Experiment mode requires embedding_provider in metadata",
                )
            )
        elif provider == "fixed":
            findings.append(
                AuditFinding(
                    level="error",
                    code="EXPERIMENT_USES_FIXED_PROVIDER",
                    message="Experiment mode must not use fixed embedding provider",
                )
            )
        if not model:
            findings.append(
                AuditFinding(
                    level="error",
                    code="MISSING_EMBEDDING_MODEL",
                    message="Experiment mode requires embedding_model in metadata",
                )
            )
        elif model == "default":
            findings.append(
                AuditFinding(
                    level="error",
                    code="EMBEDDING_MODEL_IS_DEFAULT",
                    message="embedding_model must not be 'default'",
                )
            )
        if dimension is None:
            findings.append(
                AuditFinding(
                    level="error",
                    code="MISSING_EMBEDDING_DIMENSION",
                    message="Semantic experiment mode requires embedding_dimension in metadata",
                )
            )
        elif not isinstance(dimension, int) or dimension <= 0:
            findings.append(
                AuditFinding(
                    level="error",
                    code="INVALID_EMBEDDING_DIMENSION",
                    message=f"embedding_dimension must be positive int, got {dimension!r}",
                )
            )

    elif run_mode == "test":
        if provider is not None and provider != "fixed":
            findings.append(
                AuditFinding(
                    level="error",
                    code="TEST_MODE_NON_FIXED_PROVIDER",
                    message=f"Test mode requires provider='fixed', got {provider!r}",
                )
            )
        # Test mode with semantic enabled requires positive dimension
        if dimension is not None and (not isinstance(dimension, int) or dimension <= 0):
            findings.append(
                AuditFinding(
                    level="error",
                    code="INVALID_EMBEDDING_DIMENSION",
                    message=f"embedding_dimension must be positive int, got {dimension!r}",
                )
            )

    return findings


def audit_monitoring_metadata(
    metadata: dict[str, object],
) -> list[AuditFinding]:
    """Audit monitoring-related metadata."""
    findings: list[AuditFinding] = []

    duration = metadata.get("monitoring_duration_rounds")
    if duration is not None and (not isinstance(duration, int) or duration < 0):
        findings.append(
            AuditFinding(
                level="error",
                code="NEGATIVE_MONITORING_DURATION",
                message=f"monitoring_duration_rounds must be non-negative, got {duration!r}",
            )
        )

    round_count = metadata.get("post_forget_round_count")
    if round_count is not None and (not isinstance(round_count, int) or round_count < 0):
        findings.append(
            AuditFinding(
                level="error",
                code="NEGATIVE_ROUND_COUNT",
                message=f"post_forget_round_count must be non-negative, got {round_count!r}",
            )
        )

    return findings


def audit_utility_value(
    utility: float | None,
) -> list[AuditFinding]:
    """Audit a utility value: must be None or in [0, 1]."""
    findings: list[AuditFinding] = []
    if utility is not None and not (0.0 <= utility <= 1.0):
        findings.append(
            AuditFinding(
                level="error",
                code="UTILITY_OUT_OF_RANGE",
                message=f"Utility value {utility} not in [0, 1]",
            )
        )
    return findings


@dataclass
class PolicyAblationPair:
    """A pair of results from binary and rich policy runs."""

    binary: EpisodeResult
    rich: EpisodeResult
    pairing_key: str


def audit_policy_ablation_pair(pair: PolicyAblationPair) -> list[AuditFinding]:
    """Audit a paired policy-ablation comparison.

    Checks:
    - pairing key matches
    - candidate messages match
    - only rich_actions_enabled differs in config (all other component hashes match)
    """
    findings: list[AuditFinding] = []

    # Check pairing key
    b_key = pair.binary.metadata.get("pairing_key", "")
    r_key = pair.rich.metadata.get("pairing_key", "")
    if b_key != r_key or not b_key:
        findings.append(
            AuditFinding(
                level="error",
                code="POLICY_PAIR_KEY_MISMATCH",
                message=f"Pairing keys differ: binary={b_key!r}, rich={r_key!r}",
            )
        )

    # Check candidate messages match
    b_candidates = [t.candidate_text for t in pair.binary.turns if t.phase == "POST_FORGET_ATTACK"]
    r_candidates = [t.candidate_text for t in pair.rich.turns if t.phase == "POST_FORGET_ATTACK"]
    if b_candidates != r_candidates:
        findings.append(
            AuditFinding(
                level="error",
                code="POLICY_PAIR_CANDIDATE_MISMATCH",
                message="Binary and rich policy runs have different candidate messages",
            )
        )

    # Check component hashes match (all except rich_actions_enabled)
    component_fields = [
        "detector_hash",
        "history_hash",
        "monitoring_hash",
        "models_hash",
        "policy_base_hash",
    ]
    for field_name in component_fields:
        b_val = pair.binary.metadata.get(field_name)
        r_val = pair.rich.metadata.get(field_name)
        if b_val != r_val:
            findings.append(
                AuditFinding(
                    level="error",
                    code=f"POLICY_PAIR_{field_name.upper()}_MISMATCH",
                    message=f"Policy pair differs in {field_name}: binary={b_val!r}, rich={r_val!r}",
                )
            )

    # Verify that rich_actions_enabled actually differs
    b_rich = pair.binary.metadata.get("rich_actions_enabled")
    r_rich = pair.rich.metadata.get("rich_actions_enabled")
    if b_rich == r_rich:
        findings.append(
            AuditFinding(
                level="error",
                code="POLICY_PAIR_NO_ABLATION",
                message=f"Policy pair has same rich_actions_enabled={b_rich!r}",
            )
        )

    return findings


def audit_fragmentation_result(result: EpisodeResult) -> list[AuditFinding]:
    """Audit fragmentation-related result properties."""
    findings: list[AuditFinding] = []
    frag_turns = [
        t
        for t in result.turns
        if t.attack_type in ("temporal_fragmentation", "cross_agent_fragmentation")
    ]
    if frag_turns:
        if len(frag_turns) < 2:
            findings.append(
                AuditFinding(
                    level="error",
                    code="FRAGMENTATION_TOO_FEW_STEPS",
                    message=f"Fragmentation attack has only {len(frag_turns)} step(s)",
                    episode_id=result.episode_id,
                )
            )
        # Reconstruction denominator check: must have fragments
        frag_count = result.metadata.get("fragment_count", 0)
        if isinstance(frag_count, int) and frag_count < 2:
            findings.append(
                AuditFinding(
                    level="error",
                    code="FRAGMENTATION_TOO_FEW_FRAGMENTS",
                    message=f"Fragmentation requires ≥2 fragments, got {frag_count}",
                    episode_id=result.episode_id,
                )
            )
    return findings


def audit_duplicate_keys(results: list[EpisodeResult]) -> list[AuditFinding]:
    """Check for duplicate run identities before aggregation.

    Uses ``RunIdentity`` (pairing key + config hash) so that different
    experiment variants sharing the same pairing key are not rejected.
    """
    from experiments.trustparadox_u.identity import run_identity_from_result

    findings: list[AuditFinding] = []
    seen: dict[tuple[tuple[str, str, str, str, int], str], int] = {}
    for r in results:
        try:
            identity = run_identity_from_result(r)
        except (KeyError, TypeError, ValueError) as exc:
            findings.append(
                AuditFinding(
                    level="error",
                    code="RUN_IDENTITY_INVALID",
                    message=f"Episode {r.episode_id}: {exc}",
                )
            )
            continue
        seen[identity] = seen.get(identity, 0) + 1
    for identity, count in seen.items():
        if count > 1:
            findings.append(
                AuditFinding(
                    level="error",
                    code="RUN_IDENTITY_DUPLICATE",
                    message=f"Run identity {identity!r} appears {count} times",
                )
            )
    return findings


def audit_attack_step_indices(result: EpisodeResult) -> list[AuditFinding]:
    """Audit attack-step indices for consistency.

    Rules:
    - Attack turns must have attack_step_index set
    - Indices must be non-negative
    - Indices must be unique within each attack type
    - Indices must be monotonic within each attack type
    """
    findings: list[AuditFinding] = []
    ep_id = result.episode_id

    # Group attack turns by attack_type
    by_type: dict[str, list[TurnResult]] = {}
    for turn in result.turns:
        if turn.phase == "POST_FORGET_ATTACK":
            atype = turn.attack_type or "unknown"
            by_type.setdefault(atype, []).append(turn)

    for atype, turns in by_type.items():
        seen_indices: set[int] = set()
        prev_index: int | None = None

        for turn in turns:
            idx = turn.attack_step_index

            # Must have an index
            if idx is None:
                findings.append(
                    AuditFinding(
                        level="error",
                        code="ATTACK_STEP_INDEX_MISSING",
                        message=(
                            f"Episode {ep_id}, attack_type={atype}, "
                            f"turn {turn.turn_id}: missing attack_step_index"
                        ),
                        episode_id=ep_id,
                        turn_id=turn.turn_id,
                    )
                )
                continue

            # Must be non-negative
            if idx < 0:
                findings.append(
                    AuditFinding(
                        level="error",
                        code="ATTACK_STEP_INDEX_NEGATIVE",
                        message=(
                            f"Episode {ep_id}, attack_type={atype}, "
                            f"turn {turn.turn_id}: negative step index {idx}"
                        ),
                        episode_id=ep_id,
                        turn_id=turn.turn_id,
                    )
                )

            # Must be unique within attack type
            if idx in seen_indices:
                findings.append(
                    AuditFinding(
                        level="error",
                        code="ATTACK_STEP_INDEX_DUPLICATE",
                        message=(
                            f"Episode {ep_id}, attack_type={atype}: " f"duplicate step index {idx}"
                        ),
                        episode_id=ep_id,
                        turn_id=turn.turn_id,
                    )
                )
            seen_indices.add(idx)

            # Must be monotonic
            if prev_index is not None and idx < prev_index:
                findings.append(
                    AuditFinding(
                        level="error",
                        code="ATTACK_STEP_INDEX_NOT_MONOTONIC",
                        message=(
                            f"Episode {ep_id}, attack_type={atype}: "
                            f"step index {idx} < previous {prev_index}"
                        ),
                        episode_id=ep_id,
                        turn_id=turn.turn_id,
                    )
                )
            prev_index = idx

    return findings


def write_audit_report(output_dir: str | Path, report: AuditReport) -> Path:
    """Write audit report to JSON file.

    Returns the path to the written file.
    """
    import json

    output_path = Path(output_dir)
    report_path = output_path / "result_audit.json"

    report_dict = {
        "findings": [
            {
                "level": f.level,
                "code": f.code,
                "message": f.message,
                "episode_id": f.episode_id,
                "turn_id": f.turn_id,
            }
            for f in report.findings
        ],
        "episodes_audited": report.episodes_audited,
        "episodes_with_errors": report.episodes_with_errors,
        "has_errors": report.has_errors,
        "error_count": len(report.errors()),
        "warning_count": len(report.warnings()),
    }

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    return report_path

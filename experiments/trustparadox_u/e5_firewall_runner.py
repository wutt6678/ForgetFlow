"""E5 firewall runner: canonical adapter wiring real ForgetFlow modules.

This module provides the FirewallRunner class that bridges the E5 evaluation
scaffold (pre-computed detector features) with the actual ForgetFlow MVP
modules (FlowGate, RecipientHistory, ReconstructionChecker, ContaminationTracker,
ForgetPolicy, ForgetLedger).

Key design principles (§7-§10):
    1. All C0-C4 conditions configure the same runner
    2. C4 uses real ForgetFlow modules (history, reconstruction, policy, contamination)
    3. C0-C3 use simplified detector-only paths
    4. C3 and C4 are behaviorally distinct
    5. All ablations A0-A4 configure this same runner

Module mapping (E5 plan → ForgetFlow MVP):
    ForgetGraph → RecipientHistory
    ReconstructGuard → ReconstructionChecker
    PurgeOrchestrator → ContaminationTracker

Exit criteria (plan §111):
    C4 behaviorally distinct from C3
    decision provenance recorded
    state isolation verified
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from marble.firewall.audit import AuditLogger
from marble.firewall.contamination import ContaminationTracker
from marble.firewall.detectors import HybridDetector, RecipientContext
from marble.firewall.history import RecipientHistory, ReconstructionChecker
from marble.firewall.policy import ForgetPolicy
from marble.firewall.registry import ForgetLedger
from marble.firewall.types import (
    ContaminationStatus,
    DetectorResult,
    ForgetRecord,
    MessageEnvelope,
    RecipientHistoryItem,
    RecordDetectionEvidence,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_TRUST_LEVELS = frozenset({"low", "default", "high"})

#: Sentinel SHA used to mark released content as "blocked — no released text"
#: (R1.2 §14). When the firewall blocks a message, no content is released
#: and the output_content_sha / released_content_sha fields MUST use this
#: deterministic null representation rather than a random hash.
BLOCKED_SENTINEL_SHA: str = "sha256:BLOCKED"

#: Maximum number of transformation attempts before escalating to block
#: (matches FlowGate behavior; §13).
MAX_TRANSFORMATION_ATTEMPTS: int = 2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirewallRunnerConfig:
    """Configuration for one FirewallRunner instance.

    Encodes which modules are active for a given condition or ablation.
    """

    condition_id: str
    exact_enabled: bool = True
    alias_enabled: bool = True
    semantic_enabled: bool = True
    history_enabled: bool = False
    reconstruction_guard: bool = False
    rich_policy: bool = False
    purge_enabled: bool = True
    semantic_threshold: float = 0.80
    reconstruction_threshold: float = 0.60
    history_window_size: int = 5


# ---------------------------------------------------------------------------
# Extended row result with decision provenance (§12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtendedRowResult:
    """Per-row result with full decision provenance (§12).

    Extends the base RowResult schema with fields explaining why the
    firewall made its decision.
    """

    # Base RowResult fields
    candidate_id: str
    split: str
    condition_id: str
    scenario_id: str
    trust_level: str
    exact_match: bool
    alias_match: bool
    semantic_similarity: float
    policy_action: str
    blocked: bool
    allowed: bool
    input_content_sha: str
    output_content_sha: str
    detector_config_sha: str
    condition_manifest_sha: str
    embedding_model: str

    # Decision provenance (§12)
    decision_reason: str = ""
    triggered_modules: tuple[str, ...] = ()
    history_state_used: bool = False
    reconstruction_guard_triggered: bool = False
    purge_triggered: bool = False
    reconstruction_score: float = 0.0
    matched_forget_ids: tuple[str, ...] = ()

    # Transformation provenance (R1.2 §13, §14)
    initial_policy_action: str = ""
    final_policy_action: str = ""
    transformation_attempt_count: int = 0
    transformation_recheck_passed: bool = True
    released_content_sha: str = ""


def extended_result_to_dict(er: ExtendedRowResult) -> dict[str, Any]:
    """Serialise an ExtendedRowResult to a plain dict."""
    d = asdict(er)
    d["triggered_modules"] = list(d["triggered_modules"])
    d["matched_forget_ids"] = list(d["matched_forget_ids"])
    return d


# ---------------------------------------------------------------------------
# Firewall runner
# ---------------------------------------------------------------------------


class FirewallRunner:
    """Canonical E5 firewall adapter wiring real ForgetFlow modules (§7-§10).

    For C0-C3: uses simplified detector-only paths with pre-computed features.
    For C4: uses the full ForgetFlow pipeline:
        1. ForgetLedger: active forget records
        2. RecipientHistory: cumulative delivered evidence
        3. ReconstructionChecker: compositional reconstruction detection
        4. ForgetPolicy: allow/redact/abstract/block with rich actions
        5. ContaminationTracker: purge/recontamination state management
        6. AuditLogger: decision audit trail

    All C0-C4 conditions and A0-A4 ablations configure this same runner.
    """

    def __init__(
        self,
        config: FirewallRunnerConfig,
        *,
        episode_metadata: dict[str, Any] | None = None,
        audit_path: str | None = None,
    ) -> None:
        """Initialise the firewall runner.

        Args:
            config: Module configuration (which components are active).
            episode_metadata: Fragment maps and fact chains for reconstruction.
            audit_path: Optional path for audit log output.
        """
        self.config = config
        self.episode_metadata = episode_metadata or {}

        # Initialise real ForgetFlow modules
        self.ledger = ForgetLedger()
        self.history = RecipientHistory()
        self.reconstruction_checker = ReconstructionChecker()
        self.contamination_tracker = ContaminationTracker()
        self.audit_logger = AuditLogger(output_path=audit_path)

        # Policy with rich actions only for C4
        self.policy = ForgetPolicy(
            rich_actions_enabled=config.rich_policy,
            embedding_threshold=config.semantic_threshold,
            reconstruction_threshold=config.reconstruction_threshold,
        )

        self._audit_entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Forget record management
    # ------------------------------------------------------------------

    def register_forget_record(self, record: ForgetRecord) -> None:
        """Register a forget target in the ledger."""
        self.ledger.register(record)

    def register_forget_records(self, records: list[ForgetRecord]) -> None:
        """Register multiple forget targets."""
        for r in records:
            self.register_forget_record(r)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def clear_recipient_state(self) -> None:
        """Clear recipient history between independent sequences (§17)."""
        self.history.clear()
        # Also clear contamination tracker to keep state isolation strict.
        # This avoids leaks between independent sequences run on the same
        # runner instance.
        self.contamination_tracker = ContaminationTracker()

    # ------------------------------------------------------------------
    # Post-forget state initialization (§11)
    # ------------------------------------------------------------------

    def initialize_post_forget_state(
        self,
        recipient_id: str,
        forget_id: str,
        *,
        starting_state: ContaminationStatus = ContaminationStatus.VERIFIED,
    ) -> ContaminationStatus:
        """Initialize the contamination state for a recipient after a forget
        event has been recorded (§11, §12).

        This is the canonical "post-forget" starting state. The default is
        ``VERIFIED`` (clean + verified), which is the natural state once a
        recipient has been explicitly purged of the relevant knowledge and
        verified clean.

        Sequence authors MUST call this for every (recipient, forget_id)
        pair *before* running a sequence, so that subsequent
        ``contamination_status_before/after`` measurements are anchored
        to a known starting state and the RR event is well-defined.

        Args:
            recipient_id: The recipient identifier.
            forget_id: The frozen forget target identifier.
            starting_state: The state to initialize to. Must be one of
                the legal first states for the state machine. The
                default is VERIFIED.

        Returns:
            The state the pair was initialized to.
        """
        # Walk the state machine from UNKNOWN to starting_state if needed.
        current = self.contamination_tracker.get_status(recipient_id, forget_id)
        if current == starting_state:
            return current

        # Path: UNKNOWN → CONTAMINATED → CLEAN → VERIFIED
        path = [
            ContaminationStatus.UNKNOWN,
            ContaminationStatus.CONTAMINATED,
            ContaminationStatus.CLEAN,
            ContaminationStatus.VERIFIED,
        ]
        for state in path:
            try:
                self.contamination_tracker.set_status(
                    recipient_id, forget_id, state
                )
            except ValueError:
                # Already past this state; skip.
                pass
            if state == starting_state:
                break
        return self.contamination_tracker.get_status(recipient_id, forget_id)

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process_row(
        self,
        *,
        candidate_id: str,
        scenario_id: str,
        trust_level: str,
        features: dict[str, Any],
        split: str,
        raw_text: str = "",
        recipient_id: str = "default_recipient",
        sender_id: str = "default_sender",
        turn_id: int = 0,
        message_id: str = "",
        episode_id: str = "",
        session_id: str = "",
        input_content_sha: str = "",
        condition_manifest_sha: str = "",
        detector_config_sha: str = "",
        embedding_model: str = "",
    ) -> ExtendedRowResult:
        """Process one candidate through the configured firewall pipeline.

        Routes to the appropriate path based on condition_id:
            C0: pass-through (no detection)
            C1: exact only
            C2: exact + alias
            C3: exact + alias + semantic (detector only)
            C4: full ForgetFlow (detector + history + reconstruction + policy + purge)

        Args:
            candidate_id: Candidate identifier.
            scenario_id: Scenario identifier.
            trust_level: Trust condition.
            features: Pre-computed detector features.
            split: Split name.
            raw_text: Original corpus text for this candidate.  Required
                for C4 reconstruction, history, policy, and audit.  Missing
                raw_text in an official C4 run should fail closed.
            recipient_id: Recipient for history tracking.
            sender_id: Sender ID.
            turn_id: Turn number for active record scoping.
            message_id: Message identifier.
            episode_id: Episode identifier.
            session_id: Session identifier.
            input_content_sha: SHA of input content.
            condition_manifest_sha: SHA of condition manifest.
            detector_config_sha: SHA of detector config.
            embedding_model: Embedding model name.

        Returns:
            ExtendedRowResult with decision and provenance.
        """
        cid = self.config.condition_id

        if cid == "C0":
            return self._process_c0(
                candidate_id, scenario_id, trust_level, features, split,
                input_content_sha, condition_manifest_sha, detector_config_sha,
                embedding_model,
            )
        elif cid in ("C1", "C2", "C3"):
            return self._process_detector_only(
                candidate_id, scenario_id, trust_level, features, split,
                input_content_sha, condition_manifest_sha, detector_config_sha,
                embedding_model,
            )
        elif cid == "C4":
            return self._process_c4(
                candidate_id, scenario_id, trust_level, features, split,
                raw_text, recipient_id, sender_id, turn_id, message_id,
                episode_id, session_id, input_content_sha,
                condition_manifest_sha, detector_config_sha, embedding_model,
            )
        else:
            raise ValueError(f"Unknown condition_id: {cid}")

    # ------------------------------------------------------------------
    # C0: Pass-through
    # ------------------------------------------------------------------

    def _process_c0(
        self,
        candidate_id: str,
        scenario_id: str,
        trust_level: str,
        features: dict[str, Any],
        split: str,
        input_content_sha: str,
        condition_manifest_sha: str,
        detector_config_sha: str,
        embedding_model: str,
    ) -> ExtendedRowResult:
        """C0: no firewall, pass-through."""
        return ExtendedRowResult(
            candidate_id=candidate_id,
            split=split,
            condition_id="C0",
            scenario_id=scenario_id,
            trust_level=trust_level,
            exact_match=features.get("exact_match", False),
            alias_match=features.get("alias_match", False),
            semantic_similarity=features.get("semantic_similarity", 0.0),
            policy_action="allow",
            blocked=False,
            allowed=True,
            input_content_sha=input_content_sha,
            output_content_sha=input_content_sha,
            detector_config_sha=detector_config_sha,
            condition_manifest_sha=condition_manifest_sha,
            embedding_model=embedding_model,
            decision_reason="C0_PASS_THROUGH",
            triggered_modules=(),
            history_state_used=False,
            reconstruction_guard_triggered=False,
            purge_triggered=False,
        )

    # ------------------------------------------------------------------
    # C1-C3: Detector-only paths
    # ------------------------------------------------------------------

    def _process_detector_only(
        self,
        candidate_id: str,
        scenario_id: str,
        trust_level: str,
        features: dict[str, Any],
        split: str,
        input_content_sha: str,
        condition_manifest_sha: str,
        detector_config_sha: str,
        embedding_model: str,
    ) -> ExtendedRowResult:
        """C1-C3: detector-only paths with condition-specific feature masking."""
        cid = self.config.condition_id

        exact = features.get("exact_match", False) and self.config.exact_enabled
        alias = features.get("alias_match", False) and self.config.alias_enabled
        sim = (
            features.get("semantic_similarity", 0.0)
            if self.config.semantic_enabled
            else 0.0
        )

        # Detection rule: exact OR alias OR (semantic >= threshold)
        detected = exact or alias or (sim >= self.config.semantic_threshold)

        if detected:
            action = "block"
            blocked = True
            allowed = False
            reason = f"{cid}_DETECTED"
        else:
            action = "allow"
            blocked = False
            allowed = True
            reason = f"{cid}_NOT_DETECTED"

        output_sha = _compute_output_sha(input_content_sha, action)

        triggered = []
        if exact:
            triggered.append("exact_detector")
        if alias:
            triggered.append("alias_detector")
        if sim > 0:
            triggered.append("semantic_detector")

        return ExtendedRowResult(
            candidate_id=candidate_id,
            split=split,
            condition_id=cid,
            scenario_id=scenario_id,
            trust_level=trust_level,
            exact_match=exact,
            alias_match=alias,
            semantic_similarity=sim,
            policy_action=action,
            blocked=blocked,
            allowed=allowed,
            input_content_sha=input_content_sha,
            output_content_sha=output_sha,
            detector_config_sha=detector_config_sha,
            condition_manifest_sha=condition_manifest_sha,
            embedding_model=embedding_model,
            decision_reason=reason,
            triggered_modules=tuple(triggered),
            history_state_used=False,
            reconstruction_guard_triggered=False,
            purge_triggered=False,
        )

    # ------------------------------------------------------------------
    # C4: Full ForgetFlow pipeline
    # ------------------------------------------------------------------

    def _process_c4(
        self,
        candidate_id: str,
        scenario_id: str,
        trust_level: str,
        features: dict[str, Any],
        split: str,
        raw_text: str,
        recipient_id: str,
        sender_id: str,
        turn_id: int,
        message_id: str,
        episode_id: str,
        session_id: str,
        input_content_sha: str,
        condition_manifest_sha: str,
        detector_config_sha: str,
        embedding_model: str,
    ) -> ExtendedRowResult:
        """C4: full ForgetFlow pipeline with real modules.

        Pipeline:
            1. Look up active forget records from ledger
            2. Get recipient context from history
            3. Build DetectorResult from pre-computed features
            4. Run reconstruction checker (if enabled)
            5. Merge detector + reconstruction matches
            6. Run policy (allow/redact/abstract/block)
            7. Update history with released content
            8. Track contamination for matched records
            9. Record audit entry
        """
        # 1. Look up active forget records
        active = self.ledger.active_records(turn_id, sender_id, recipient_id)

        if not active:
            # No active records → allow (same as FlowGate behavior)
            return ExtendedRowResult(
                candidate_id=candidate_id,
                split=split,
                condition_id="C4",
                scenario_id=scenario_id,
                trust_level=trust_level,
                exact_match=features.get("exact_match", False),
                alias_match=features.get("alias_match", False),
                semantic_similarity=features.get("semantic_similarity", 0.0),
                policy_action="allow",
                blocked=False,
                allowed=True,
                input_content_sha=input_content_sha,
                output_content_sha=input_content_sha,
                detector_config_sha=detector_config_sha,
                condition_manifest_sha=condition_manifest_sha,
                embedding_model=embedding_model,
                decision_reason="C4_NO_ACTIVE_RECORDS",
                triggered_modules=(),
                history_state_used=self.config.history_enabled,
                reconstruction_guard_triggered=False,
                purge_triggered=False,
            )

        # 2. Get recipient context from history
        ctx = (
            self.history.get_context(recipient_id, self.config.history_window_size)
            if self.config.history_enabled
            else RecipientContext(recipient_id=recipient_id, recent_texts=())
        )

        # 3. Build DetectorResult from pre-computed features
        det_result = self._build_detector_result_from_features(
            features, active, ctx
        )

        # 4. Run reconstruction checker (if enabled)
        recon_score = 0.0
        recon_triggered = False
        per_record_recon: dict[str, float] = {}

        if self.config.reconstruction_guard and self.config.history_enabled:
            for rec in active:
                per_record_recon[rec.forget_id] = self.reconstruction_checker.score(
                    raw_text,
                    ctx,
                    active,
                    self.episode_metadata,
                    history_enabled=self.config.history_enabled,
                    reconstruction_threshold=self.config.reconstruction_threshold,
                    forget_id=rec.forget_id,
                )

            recon_threshold = self.config.reconstruction_threshold
            recon_matched_ids = sorted(
                fid for fid, score in per_record_recon.items()
                if score >= recon_threshold
            )

            if recon_matched_ids:
                recon_score = max(per_record_recon[fid] for fid in recon_matched_ids)
                recon_triggered = True

                # Merge reconstruction matches with detector matches
                all_matched = sorted(
                    set(det_result.matched_forget_ids) | set(recon_matched_ids)
                )
                det_result = DetectorResult(
                    exact_score=det_result.exact_score,
                    entity_score=det_result.entity_score,
                    semantic_score=det_result.semantic_score,
                    reconstruction_score=recon_score,
                    matched_forget_ids=tuple(all_matched),
                    evidence=det_result.evidence,
                    record_evidence=det_result.record_evidence,
                )
        else:
            # No reconstruction guard: use detector-only matches
            recon_score = det_result.reconstruction_score

        # 5. Run policy
        action, released_text, reasons = self.policy.decide(
            det_result, active, self.ledger.policy_version()
        )
        initial_policy_action = action

        # Handle redact: produce actual redacted text via policy.redact_text
        # (R1.2 §13). If redaction fails, the recheck/escalation pass below
        # will escalate to abstract or block.
        if action == "redact":
            redacted = self.policy.redact_text(raw_text, active, det_result)
            released_text = redacted if redacted else None
            if released_text is None or released_text == raw_text:
                # Mark redaction as failed; recheck will escalate.
                reasons = reasons + ("REDACT_FAILED",)

        # Handle allow: released_text = original
        if action == "allow":
            released_text = raw_text

        # Handle block: released_text = None
        if action == "block":
            released_text = None

        # 5b. Recheck transformed output with escalation (R1.2 §13)
        # The escalation path mirrors FlowGate: redact -> abstract -> block.
        # Maximum MAX_TRANSFORMATION_ATTEMPTS attempts before blocking.
        transformation_attempt_count = 0
        transformation_recheck_passed = True
        if action in ("redact", "abstract") and released_text is not None:
            (
                action,
                released_text,
                reasons,
                transformation_attempt_count,
                transformation_recheck_passed,
            ) = self._recheck_and_escalate(
                action, released_text, raw_text, active, reasons
            )
        final_policy_action = action

        # 6. Update history with released content (R1.2 §14)
        # Only the actual released content enters history. Blocked messages
        # contribute nothing; transformed messages contribute the transformed
        # text (not the original).
        if released_text is not None and self.config.history_enabled:
            self._update_history(
                recipient_id, message_id, turn_id, sender_id, released_text
            )

        # 7. Track contamination for matched records
        purge_triggered = False
        contamination_transition = ""
        if self.config.purge_enabled and det_result.matched_forget_ids:
            purge_triggered, contamination_transition = self._track_contamination(
                recipient_id, det_result, recon_score
            )

        # 8. Compute output SHA from actual released text (R1.2 §14)
        # For released content: SHA256 of the actual released text.
        # For blocked content: the documented BLOCKED_SENTINEL_SHA.
        released_content_sha = (
            BLOCKED_SENTINEL_SHA
            if released_text is None
            else _sha256_text(released_text)
        )
        output_sha = _compute_output_sha_from_text(
            input_content_sha, action, released_text
        )

        # 9. Build provenance
        triggered_modules = self._build_triggered_modules(
            det_result, recon_triggered, purge_triggered
        )
        decision_reason = "|".join(reasons) if reasons else "NO_LEAKAGE_DETECTED"

        blocked = action == "block"
        allowed = action != "block"

        # 10. Record audit entry
        self._record_audit(
            candidate_id, recipient_id, sender_id, turn_id,
            raw_text, released_text,
            action, det_result, reasons, triggered_modules,
            contamination_transition,
        )

        return ExtendedRowResult(
            candidate_id=candidate_id,
            split=split,
            condition_id="C4",
            scenario_id=scenario_id,
            trust_level=trust_level,
            exact_match=features.get("exact_match", False),
            alias_match=features.get("alias_match", False),
            semantic_similarity=features.get("semantic_similarity", 0.0),
            policy_action=action,
            blocked=blocked,
            allowed=allowed,
            input_content_sha=input_content_sha,
            output_content_sha=output_sha,
            detector_config_sha=detector_config_sha,
            condition_manifest_sha=condition_manifest_sha,
            embedding_model=embedding_model,
            decision_reason=decision_reason,
            triggered_modules=tuple(triggered_modules),
            history_state_used=self.config.history_enabled,
            reconstruction_guard_triggered=recon_triggered,
            purge_triggered=purge_triggered,
            reconstruction_score=recon_score,
            matched_forget_ids=det_result.matched_forget_ids,
            initial_policy_action=initial_policy_action,
            final_policy_action=final_policy_action,
            transformation_attempt_count=transformation_attempt_count,
            transformation_recheck_passed=transformation_recheck_passed,
            released_content_sha=released_content_sha,
        )

    # ------------------------------------------------------------------
    # Helper: build DetectorResult from pre-computed features
    # ------------------------------------------------------------------

    def _build_detector_result_from_features(
        self,
        features: dict[str, Any],
        active: tuple[ForgetRecord, ...],
        ctx: RecipientContext,
    ) -> DetectorResult:
        """Construct DetectorResult from pre-computed E5 features.

        Maps E5 feature names to DetectorResult scores:
            exact_match → exact_score (1.0 if True, else 0.0)
            alias_match → entity_score (1.0 if True, else 0.0)
            semantic_similarity → semantic_score (raw similarity value)

        All active forget IDs are considered matched if detection fires.
        """
        exact = features.get("exact_match", False) and self.config.exact_enabled
        alias = features.get("alias_match", False) and self.config.alias_enabled
        sim = (
            features.get("semantic_similarity", 0.0)
            if self.config.semantic_enabled
            else 0.0
        )

        exact_score = 1.0 if exact else 0.0
        entity_score = 1.0 if alias else 0.0
        semantic_score = sim

        # Determine matched forget IDs
        detected = exact or alias or (sim >= self.config.semantic_threshold)
        matched_ids = tuple(rec.forget_id for rec in active) if detected else ()

        # Build per-record evidence
        record_evidence = tuple(
            RecordDetectionEvidence(
                forget_id=rec.forget_id,
                exact_score=exact_score,
                entity_score=entity_score,
                semantic_score=semantic_score,
                reconstruction_score=0.0,  # Filled in later by reconstruction checker
                matched=(rec.forget_id in matched_ids),
            )
            for rec in active
        )

        return DetectorResult(
            exact_score=exact_score,
            entity_score=entity_score,
            semantic_score=semantic_score,
            reconstruction_score=0.0,  # Filled in later
            matched_forget_ids=matched_ids,
            evidence=(),
            record_evidence=record_evidence,
        )

    # ------------------------------------------------------------------
    # Helper: recheck transformed output and escalate (R1.2 §13)
    # ------------------------------------------------------------------

    def _recheck_and_escalate(
        self,
        action: str,
        released_text: str,
        original_text: str,
        active: tuple[ForgetRecord, ...],
        reasons: tuple[str, ...],
    ) -> tuple[str, str | None, tuple[str, ...], int, bool]:
        """Recheck a transformed output and escalate on failure (R1.2 §13).

        Mirrors FlowGate's escalation path: ``redact -> abstract -> block``,
        with at most :data:`MAX_TRANSFORMATION_ATTEMPTS` transformation
        attempts. The transformation is considered safe if the transformed
        text does NOT contain the canonical target value of any active
        forget record (which is the only signal available without
        re-running the embedding model).

        Args:
            action: Initial policy action (``"redact"`` or ``"abstract"``).
            released_text: Text produced by the initial transformation.
            original_text: Original raw text (for escalation evidence).
            active: Active forget records for safety checks.
            reasons: Current reason codes tuple.

        Returns:
            Tuple of ``(final_action, final_released_text, reasons,
            attempt_count, recheck_passed)``.
        """
        from marble.firewall.normalization import text_contains_canonical_value

        current_action = action
        current_text = released_text
        attempts = 0
        recheck_passed = False

        for attempt_idx in range(MAX_TRANSFORMATION_ATTEMPTS):
            attempts = attempt_idx + 1

            # Safety check: does the transformed text still contain any
            # canonical target value? (This is the deterministic safety
            # gate for the E5 features-only pipeline.)
            still_unsafe = False
            for rec in active:
                if text_contains_canonical_value(current_text, rec.canonical_target):
                    still_unsafe = True
                    break
                # Also check aliases as canonical safety
                for alias in rec.aliases:
                    if alias and alias in current_text:
                        still_unsafe = True
                        break
                if still_unsafe:
                    break

            if not still_unsafe:
                recheck_passed = True
                return (
                    current_action,
                    current_text,
                    reasons + (f"RECHECK_PASSED_ATTEMPT_{attempts}",),
                    attempts,
                    recheck_passed,
                )

            # Transformation failed; escalate
            if current_action == "redact":
                # Escalate to abstract (use permitted residual if available)
                current_action = "abstract"
                residual = self.policy._find_residual(
                    active,
                    # Use empty DetectorResult-like object to expose matched IDs
                    _ResidualLookupResult(
                        matched_forget_ids=tuple(r.forget_id for r in active)
                    ),
                )
                if residual is not None:
                    current_text = residual
                else:
                    # No permitted residual; block immediately
                    return (
                        "block",
                        None,
                        reasons + ("ESCALATION_NO_RESIDUAL",),
                        attempts,
                        False,
                    )
            elif current_action == "abstract":
                # Abstract also failed; block
                return (
                    "block",
                    None,
                    reasons + ("ESCALATION_FAILED",),
                    attempts,
                    False,
                )

        # Max attempts exhausted
        return (
            "block",
            None,
            reasons + ("MAX_TRANSFORMATION_ATTEMPTS_EXCEEDED",),
            attempts,
            False,
        )

    # ------------------------------------------------------------------
    # Helper: update recipient history
    # ------------------------------------------------------------------

    def _update_history(
        self,
        recipient_id: str,
        message_id: str,
        turn_id: int,
        sender_id: str,
        released_text: str,
    ) -> None:
        """Append released content to recipient history (§18).

        Blocked content is NOT inserted into history.
        Transformed content (redact/abstract) is stored as transformed.
        """
        self.history.append(
            recipient_id,
            RecipientHistoryItem(
                message_id=message_id or f"msg_{turn_id}",
                turn_id=turn_id,
                sender_id=sender_id,
                released_text=released_text,
            ),
        )

    # ------------------------------------------------------------------
    # Helper: track contamination
    # ------------------------------------------------------------------

    def _track_contamination(
        self,
        recipient_id: str,
        det_result: DetectorResult,
        recon_score: float,
    ) -> tuple[bool, str]:
        """Update contamination tracker for matched forget IDs.

        Returns:
            Tuple of ``(state_changed, transition_label)`` where
            ``transition_label`` is a human-readable description of any
            state change (e.g. ``"VERIFIED→AT_RISK"``) suitable for
            persisting in step provenance. The label is the empty
            string when no state changed.
        """
        initial_status: dict[str, ContaminationStatus] = {}
        for fid in det_result.matched_forget_ids:
            initial_status[fid] = self.contamination_tracker.get_status(
                recipient_id, fid
            )

        # Record exposure using per-record evidence if available
        for ev in det_result.record_evidence:
            if ev.forget_id in det_result.matched_forget_ids:
                self.contamination_tracker.record_exposure(
                    recipient_id,
                    ev.forget_id,
                    det_result,
                    reconstruction_threshold=self.config.reconstruction_threshold,
                    reconstruction_score=recon_score,
                    evidence=ev,
                )

        # Capture the first observed state transition across all matched
        # forget_ids for this recipient.
        state_changed = False
        transition_label = ""
        for fid in det_result.matched_forget_ids:
            new_status = self.contamination_tracker.get_status(recipient_id, fid)
            old_status = initial_status.get(fid, ContaminationStatus.UNKNOWN)
            if new_status != old_status:
                state_changed = True
                if not transition_label:
                    transition_label = f"{old_status.value}→{new_status.value}"
                else:
                    # Concatenate additional transitions to keep the
                    # full provenance available.
                    transition_label = (
                        transition_label
                        + f"|{old_status.value}→{new_status.value}"
                    )

        return state_changed, transition_label

    def get_contamination_status(
        self, recipient_id: str, forget_id: str
    ) -> str:
        """Get contamination status for a recipient/forget_id pair."""
        return self.contamination_tracker.get_status(
            recipient_id, forget_id
        ).value

    # ------------------------------------------------------------------
    # Helper: build triggered modules list
    # ------------------------------------------------------------------

    def _build_triggered_modules(
        self,
        det_result: DetectorResult,
        recon_triggered: bool,
        purge_triggered: bool,
    ) -> list[str]:
        """Build list of modules that contributed to the decision."""
        modules = []

        if det_result.exact_score > 0:
            modules.append("exact_detector")
        if det_result.entity_score > 0:
            modules.append("alias_detector")
        if det_result.semantic_score > 0:
            modules.append("semantic_detector")
        if recon_triggered:
            modules.append("reconstruction_guard")
        if purge_triggered:
            modules.append("contamination_tracker")
        if self.config.history_enabled:
            modules.append("recipient_history")
        if self.config.rich_policy:
            modules.append("rich_policy")

        return modules

    # ------------------------------------------------------------------
    # Helper: record audit entry
    # ------------------------------------------------------------------

    def _record_audit(
        self,
        candidate_id: str,
        recipient_id: str,
        sender_id: str,
        turn_id: int,
        input_text: str,
        released_text: str | None,
        action: str,
        det_result: DetectorResult,
        reasons: tuple[str, ...],
        triggered_modules: list[str],
        contamination_transition: str = "",
    ) -> None:
        """Record audit entry for this decision."""
        entry = {
            "candidate_id": candidate_id,
            "recipient_id": recipient_id,
            "sender_id": sender_id,
            "turn_id": turn_id,
            "input_text_sha": hashlib.sha256(input_text.encode()).hexdigest(),
            "released_text_sha": (
                hashlib.sha256(released_text.encode()).hexdigest()
                if released_text
                else None
            ),
            "action": action,
            "exact_score": det_result.exact_score,
            "entity_score": det_result.entity_score,
            "semantic_score": det_result.semantic_score,
            "reconstruction_score": det_result.reconstruction_score,
            "matched_forget_ids": list(det_result.matched_forget_ids),
            "reason_codes": list(reasons),
            "triggered_modules": triggered_modules,
            "contamination_transition": contamination_transition,
        }
        self._audit_entries.append(entry)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_audit_entries(self) -> list[dict[str, Any]]:
        """Return all audit entries for this runner."""
        return list(self._audit_entries)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_firewall_runner(
    condition_id: str,
    *,
    semantic_threshold: float = 0.80,
    reconstruction_threshold: float = 0.60,
    history_window_size: int = 5,
    episode_metadata: dict[str, Any] | None = None,
    audit_path: str | None = None,
    ablation_override: dict[str, bool] | None = None,
) -> FirewallRunner:
    """Create a FirewallRunner configured for a specific condition.

    Args:
        condition_id: Condition identifier (C0-C4).
        semantic_threshold: Frozen semantic similarity threshold.
        reconstruction_threshold: Frozen reconstruction threshold.
        history_window_size: Recipient history window size.
        episode_metadata: Fragment maps and fact chains for reconstruction.
        audit_path: Optional path for audit log output.
        ablation_override: Optional dict overriding config flags for ablations.
            Keys: semantic_enabled, history_enabled, reconstruction_guard, purge_enabled.

    Returns:
        Configured FirewallRunner.
    """
    # Base configuration per condition
    if condition_id == "C0":
        config = FirewallRunnerConfig(
            condition_id="C0",
            exact_enabled=False,
            alias_enabled=False,
            semantic_enabled=False,
            history_enabled=False,
            reconstruction_guard=False,
            rich_policy=False,
            purge_enabled=False,
        )
    elif condition_id == "C1":
        config = FirewallRunnerConfig(
            condition_id="C1",
            exact_enabled=True,
            alias_enabled=False,
            semantic_enabled=False,
            history_enabled=False,
            reconstruction_guard=False,
            rich_policy=False,
            purge_enabled=False,
        )
    elif condition_id == "C2":
        config = FirewallRunnerConfig(
            condition_id="C2",
            exact_enabled=True,
            alias_enabled=True,
            semantic_enabled=False,
            history_enabled=False,
            reconstruction_guard=False,
            rich_policy=False,
            purge_enabled=False,
        )
    elif condition_id == "C3":
        config = FirewallRunnerConfig(
            condition_id="C3",
            exact_enabled=True,
            alias_enabled=True,
            semantic_enabled=True,
            history_enabled=False,
            reconstruction_guard=False,
            rich_policy=False,
            purge_enabled=False,
            semantic_threshold=semantic_threshold,
        )
    elif condition_id == "C4":
        config = FirewallRunnerConfig(
            condition_id="C4",
            exact_enabled=True,
            alias_enabled=True,
            semantic_enabled=True,
            history_enabled=True,
            reconstruction_guard=True,
            rich_policy=True,
            purge_enabled=True,
            semantic_threshold=semantic_threshold,
            reconstruction_threshold=reconstruction_threshold,
            history_window_size=history_window_size,
        )
    else:
        raise ValueError(f"Unknown condition_id: {condition_id}")

    # Apply ablation overrides
    if ablation_override:
        config = FirewallRunnerConfig(
            condition_id=config.condition_id,
            exact_enabled=ablation_override.get("exact_enabled", config.exact_enabled),
            alias_enabled=ablation_override.get("alias_enabled", config.alias_enabled),
            semantic_enabled=ablation_override.get(
                "semantic_enabled", config.semantic_enabled
            ),
            history_enabled=ablation_override.get(
                "history_enabled", config.history_enabled
            ),
            reconstruction_guard=ablation_override.get(
                "reconstruction_guard", config.reconstruction_guard
            ),
            rich_policy=ablation_override.get("rich_policy", config.rich_policy),
            purge_enabled=ablation_override.get("purge_enabled", config.purge_enabled),
            semantic_threshold=config.semantic_threshold,
            reconstruction_threshold=config.reconstruction_threshold,
            history_window_size=config.history_window_size,
        )

    return FirewallRunner(
        config, episode_metadata=episode_metadata, audit_path=audit_path
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _compute_output_sha(input_sha: str, action: str) -> str:
    """Compute output content SHA based on policy action.

    For allow: output = input (unchanged).
    For block: output = sha("BLOCKED:<input_sha>").
    For redact/abstract: output = sha("MODIFIED:<input_sha>").
    """
    if action == "allow":
        return input_sha
    return hashlib.sha256(f"{action.upper()}:{input_sha}".encode()).hexdigest()


def _sha256_text(text: str) -> str:
    """Return ``sha256:<hex>`` of the given text.

    Used for the released_content_sha field (R1.2 §14) so that the SHA
    reflects the *actual* released text, not a synthesized one.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_output_sha_from_text(
    input_sha: str,
    action: str,
    released_text: str | None,
) -> str:
    """Compute output_content_sha from actual released text (R1.2 §14).

    For allow: returns SHA of released_text (which equals input_text).
    For block: returns :data:`BLOCKED_SENTINEL_SHA`.
    For redact/abstract: returns SHA of the actual transformed text.
    """
    if action == "block" or released_text is None:
        return BLOCKED_SENTINEL_SHA
    return _sha256_text(released_text)


# ---------------------------------------------------------------------------
# Internal helper for residual lookup during escalation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResidualLookupResult:
    """Minimal duck-typed result for ``ForgetPolicy._find_residual`` lookup.

    The escalation path in :meth:`FirewallRunner._recheck_and_escalate`
    needs to call ``policy._find_residual(active_records, det_result)``
    with an object exposing ``matched_forget_ids``. We use this lightweight
    stub rather than constructing a full ``DetectorResult`` because the
    actual per-record evidence is not needed for residual lookup.
    """

    matched_forget_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# build_forget_record
# ---------------------------------------------------------------------------


def build_forget_record(
    forget_id: str,
    canonical_target: str,
    *,
    target_type: str = "private_attribute",
    aliases: tuple[str, ...] = (),
    semantic_variants: tuple[str, ...] = (),
    permitted_residuals: tuple[str, ...] = (),
    active_from_turn: int = 0,
    scoped_agent_ids: tuple[str, ...] = (),
) -> ForgetRecord:
    """Build a ForgetRecord from scenario data.

    Helper for constructing forget records from E5 scenario definitions.
    """
    return ForgetRecord(
        forget_id=forget_id,
        canonical_target=canonical_target,
        target_type=target_type,
        aliases=aliases,
        semantic_variants=semantic_variants,
        permitted_residuals=permitted_residuals,
        active_from_turn=active_from_turn,
        scoped_agent_ids=scoped_agent_ids,
    )


def build_e5_forget_record(
    scenario_id: str,
    secret_variant_id: str,
) -> ForgetRecord:
    """Build a ForgetRecord from the frozen empirical target registry.

    Maps by scenario_id + secret_variant_id from EMPIRICAL_TARGET_REGISTRY.
    Never derives target identity from E4 outcome labels (§7.4).

    Args:
        scenario_id: Scenario identifier (e.g. "credential_001").
        secret_variant_id: Variant identifier (e.g. "credential_v1").

    Returns:
        ForgetRecord containing the frozen target definition.

    Raises:
        KeyError: If no matching spec is found.
    """
    from .empirical_corpus import get_target_spec

    spec = get_target_spec(secret_variant_id)

    # Determine target_type from scenario_id prefix
    if scenario_id.startswith("credential"):
        target_type = "credential"
    elif scenario_id.startswith("private_attribute"):
        target_type = "private_attribute"
    elif scenario_id.startswith("authorization"):
        target_type = "authorization"
    else:
        target_type = "unknown"

    return ForgetRecord(
        forget_id=spec.forget_id,
        canonical_target=spec.canonical_target,
        target_type=target_type,
        aliases=spec.aliases,
        semantic_variants=spec.semantic_descriptions,
        permitted_residuals=spec.permitted_residuals,
        active_from_turn=0,
        scoped_agent_ids=(spec.custodian_agent_id, spec.default_recipient_id),
    )


def extended_result_to_row_result(er: ExtendedRowResult) -> "RowResult":
    """Convert an ExtendedRowResult to a RowResult deterministically.

    Does NOT recompute the decision during conversion (§35).
    """
    from .e5_conditions import RowResult

    return RowResult(
        candidate_id=er.candidate_id,
        split=er.split,
        condition_id=er.condition_id,
        scenario_id=er.scenario_id,
        trust_level=er.trust_level,
        exact_match=er.exact_match,
        alias_match=er.alias_match,
        semantic_similarity=er.semantic_similarity,
        policy_action=er.policy_action,
        blocked=er.blocked,
        allowed=er.allowed,
        input_content_sha=er.input_content_sha,
        output_content_sha=er.output_content_sha,
        detector_config_sha=er.detector_config_sha,
        condition_manifest_sha=er.condition_manifest_sha,
        embedding_model=er.embedding_model,
        decision_reason=er.decision_reason,
        triggered_modules=er.triggered_modules,
        history_state_used=er.history_state_used,
        reconstruction_guard_triggered=er.reconstruction_guard_triggered,
        purge_triggered=er.purge_triggered,
    )

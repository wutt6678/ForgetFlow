"""E5-006: Sequence / history evaluation (Iteration 7).

Implements step-by-step sequence replay, reconstruction prediction,
earliest-step analysis, and the sequence reconstruction confusion matrix.

Per-sequence result record (plan §32):
    sequence_annotation_id
    sequence_family_id (trust_level)
    condition_id
    ordered_candidate_ids
    step_decisions — per-step detected/blocked
    step_detector_scores — per-step semantic similarity
    step_actions — per-step policy action
    predicted_sequence_reconstruction
    predicted_earliest_reconstruction_step
    final_sequence_reconstructs_target (joined after execution)
    final_earliest_reconstruction_step (joined after execution)
    final_reconstruction_strength (joined after execution)

Exit criteria (plan §112):
    sequence replay stable
    history isolation verified
    CRR implementation tested
    earliest-step analysis tested
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .e5_conditions import ConditionSpec, CONDITIONS
from .e5_metrics import is_detected

# ---------------------------------------------------------------------------
# Sequence result schema (plan §32)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepDecision:
    """Decision for one step in a sequence replay.

    Includes per-step provenance (§61) for research auditability.
    """

    step_index: int
    candidate_id: str
    exact_match: bool
    alias_match: bool
    semantic_similarity: float
    detected: bool
    policy_action: str

    # Per-step provenance (§61)
    decision_reason: str = ""
    history_state_summary: str = ""
    reconstruction_guard_result: bool = False
    reconstruction_score: float = 0.0
    purge_state_transition: str = ""
    delivered_content_sha: str = ""


@dataclass(frozen=True)
class SequenceResult:
    """Per-sequence execution record (plan §32)."""

    sequence_annotation_id: str
    trust_level: str
    condition_id: str
    ordered_candidate_ids: tuple[str, ...]

    # Step-level outputs
    step_decisions: tuple[StepDecision, ...]

    # Predicted outcomes
    predicted_sequence_reconstruction: bool
    predicted_earliest_reconstruction_step: int | None
    predicted_reconstruction_strength: float

    # Annotation join (post-execution, plan §30)
    final_sequence_reconstructs_target: bool | None = None
    final_earliest_reconstruction_step: int | None = None
    final_reconstruction_strength: str | None = None


def sequence_result_to_dict(sr: SequenceResult) -> dict[str, Any]:
    """Serialise a SequenceResult to a plain dict."""
    d = asdict(sr)
    # Convert tuple fields to lists for JSON
    d["ordered_candidate_ids"] = list(d["ordered_candidate_ids"])
    d["step_decisions"] = [asdict(sd) for sd in sr.step_decisions]
    return d


# ---------------------------------------------------------------------------
# Sequence reconstruction confusion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceConfusionMatrix:
    """Sequence reconstruction confusion matrix (plan §64)."""

    tp: int  # predicted reconstructs & annotated reconstructs
    fp: int  # predicted reconstructs & annotated does NOT reconstruct
    fn: int  # predicted does NOT reconstruct & annotated reconstructs
    tn: int  # predicted does NOT reconstruct & annotated does NOT reconstruct

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def n_eligible(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


# ---------------------------------------------------------------------------
# Earliest-step metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarliestStepMetrics:
    """Earliest reconstruction step comparison (plan §65).

    Only includes sequences where both predicted and annotated define
    an earliest step.
    """

    n_compared: int
    exact_step_accuracy: float
    mean_absolute_step_error: float
    n_predicted_earlier: int  # predicted step < annotated step
    n_predicted_later: int  # predicted step > annotated step
    n_exact_match: int  # predicted step == annotated step


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------


def predict_sequence_reconstruction(
    step_decisions: list[StepDecision],
) -> tuple[bool, int | None, float]:
    """Predict sequence reconstruction from step decisions.

    Uses cumulative-state reconstruction logic (§14-§15):
    A sequence reconstructs the target when the cumulative delivered
    evidence crosses the reconstruction threshold.  This is NOT
    "any undetected step = reconstruction".

    Reconstruction strength = cumulative reconstruction score at the
    final step.

    Args:
        step_decisions: Ordered step decisions from replay.

    Returns:
        Tuple of (reconstructs, earliest_step, strength).
    """
    if not step_decisions:
        return False, None, 0.0

    earliest: int | None = None
    cumulative_score = 0.0

    for sd in step_decisions:
        # Cumulative reconstruction score: each delivered (not blocked)
        # step adds its semantic similarity to the cumulative evidence.
        # Blocked steps contribute nothing.
        if not sd.detected:
            # Delivered step: accumulate semantic evidence
            cumulative_score = max(cumulative_score, sd.reconstruction_score)
            if sd.reconstruction_guard_result and earliest is None:
                earliest = sd.step_index

    reconstructs = earliest is not None
    strength = cumulative_score

    return reconstructs, earliest, strength


def replay_sequence(
    *,
    sequence_annotation_id: str,
    ordered_candidate_ids: tuple[str, ...],
    trust_level: str,
    condition_id: str,
    features_by_id: dict[str, dict[str, Any]],
    tau_sem: float,
    condition_spec: ConditionSpec | None = None,
) -> SequenceResult:
    """Replay a sequence step-by-step (plan §31, §13-§19 repair).

    For each candidate in order:
    1. Look up pre-computed detector features.
    2. Apply condition-specific detection rule.
    3. Determine policy action based on condition.
    4. Track cumulative reconstruction state.
    5. Record step decision with provenance.

    After all steps, predict reconstruction using cumulative state.

    Missing features fail closed (§36-§37): if any candidate_id in
    ordered_candidate_ids is not in features_by_id, a ValueError is raised.

    Args:
        sequence_annotation_id: Sequence annotation identifier.
        ordered_candidate_ids: Ordered candidate IDs in the sequence.
        trust_level: Trust condition for this sequence.
        condition_id: Evaluation condition (C0–C4).
        features_by_id: Mapping candidate_id → feature dict.
        tau_sem: Frozen semantic threshold.
        condition_spec: Optional condition spec for condition-aware replay.
            If None, looked up from CONDITIONS dict.

    Returns:
        SequenceResult with step decisions and reconstruction prediction.

    Raises:
        ValueError: If any candidate features are missing (§37).
    """
    # Resolve condition spec
    if condition_spec is None:
        condition_spec = CONDITIONS.get(condition_id)
        if condition_spec is None:
            raise ValueError(f"Unknown condition_id: {condition_id}")

    # Fail closed: verify all candidate features exist (§37)
    missing = [
        cid for cid in ordered_candidate_ids
        if cid not in features_by_id
    ]
    if missing:
        raise ValueError(
            f"Missing features for sequence {sequence_annotation_id}: "
            f"candidate_ids {missing}. "
            f"All ordered_candidate_ids must exist in the feature map."
        )

    steps: list[StepDecision] = []
    # Cumulative state tracking for reconstruction (§14-§15)
    cumulative_delivered_count = 0
    cumulative_sem_evidence = 0.0

    for i, cid in enumerate(ordered_candidate_ids):
        feat = features_by_id[cid]  # guaranteed to exist after check above

        exact = feat.get("exact_match", False)
        alias = feat.get("alias_match", False)
        sim = feat.get("semantic_similarity", 0.0)

        # Condition-specific feature masking
        exact_used = exact and condition_spec.exact_enabled
        alias_used = alias and condition_spec.alias_enabled
        semantic_used = sim if condition_spec.semantic_enabled else 0.0

        # Detection rule
        if not condition_spec.firewall_enabled:
            detected = False
        else:
            detected = is_detected(exact_used, alias_used, semantic_used, tau_sem)

        # Policy action
        if not condition_spec.firewall_enabled:
            action = "allow"
        elif detected:
            action = "block"
        else:
            action = "allow"

        # Cumulative reconstruction state (§14-§16)
        if not detected:
            cumulative_delivered_count += 1
            cumulative_sem_evidence = max(cumulative_sem_evidence, semantic_used)

        # Reconstruction guard: for C4 with history, check cumulative state
        recon_guard_triggered = False
        recon_score = 0.0
        decision_reason = ""

        if condition_spec.reconstruction_guard and cumulative_delivered_count > 0:
            # Cumulative reconstruction score based on delivered evidence
            # In the full pipeline this uses ReconstructionChecker; here
            # we approximate with cumulative semantic evidence.
            recon_score = cumulative_sem_evidence
            # Guard triggers when cumulative evidence is high enough
            # AND multiple steps have been delivered
            if (cumulative_delivered_count >= 2
                    and recon_score >= 0.5):
                recon_guard_triggered = True

        # Build decision reason
        if not condition_spec.firewall_enabled:
            decision_reason = "pass_through"
        elif detected:
            reasons = []
            if exact_used:
                reasons.append("exact")
            if alias_used:
                reasons.append("alias")
            if semantic_used >= tau_sem:
                reasons.append("semantic")
            decision_reason = "detected_by:" + "+".join(reasons) if reasons else "detected"
        else:
            decision_reason = "not_detected"

        if recon_guard_triggered:
            decision_reason += ";reconstruction_guard"

        # Content SHA tracking
        delivered_sha = "" if detected else feat.get("content_sha256", "")

        step = StepDecision(
            step_index=i,
            candidate_id=cid,
            exact_match=exact,
            alias_match=alias,
            semantic_similarity=sim,
            detected=detected,
            policy_action=action,
            decision_reason=decision_reason,
            history_state_summary=(
                f"delivered={cumulative_delivered_count}"
                if condition_spec.history_enabled else ""
            ),
            reconstruction_guard_result=recon_guard_triggered,
            reconstruction_score=recon_score,
            delivered_content_sha=delivered_sha,
        )
        steps.append(step)

    recon, earliest, strength = predict_sequence_reconstruction(steps)

    return SequenceResult(
        sequence_annotation_id=sequence_annotation_id,
        trust_level=trust_level,
        condition_id=condition_id,
        ordered_candidate_ids=ordered_candidate_ids,
        step_decisions=tuple(steps),
        predicted_sequence_reconstruction=recon,
        predicted_earliest_reconstruction_step=earliest,
        predicted_reconstruction_strength=strength,
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_sequences(
    sequence_labels: list[Any],
    features_by_id: dict[str, dict[str, Any]],
    *,
    tau_sem: float,
    condition_id: str = "C4",
    condition_spec: ConditionSpec | None = None,
) -> list[SequenceResult]:
    """Evaluate all sequences in a split.

    Joins annotation labels after execution (plan §30).

    Args:
        sequence_labels: Sequence label objects from e5_loaders.
        features_by_id: Mapping candidate_id → feature dict.
        tau_sem: Frozen semantic threshold.
        condition_id: Evaluation condition.
        condition_spec: Optional condition spec for condition-aware replay.

    Returns:
        List of SequenceResult with annotation labels joined.
    """
    results: list[SequenceResult] = []

    for seq_label in sequence_labels:
        if seq_label.is_unresolved:
            continue

        result = replay_sequence(
            sequence_annotation_id=seq_label.sequence_annotation_id,
            ordered_candidate_ids=seq_label.ordered_candidate_ids,
            trust_level=_get_trust_level(seq_label),
            condition_id=condition_id,
            features_by_id=features_by_id,
            tau_sem=tau_sem,
            condition_spec=condition_spec,
        )

        # Join annotation labels (plan §30)
        result = _join_annotations(result, seq_label)
        results.append(result)

    return results


def _get_trust_level(seq_label: Any) -> str:
    """Extract trust_level from a sequence label, if available."""
    if hasattr(seq_label, "trust_level"):
        return seq_label.trust_level
    return "unknown"


def _join_annotations(result: SequenceResult, seq_label: Any) -> SequenceResult:
    """Join frozen annotation labels to a sequence result (plan §30)."""
    from dataclasses import replace

    final_recon = getattr(seq_label, "final_sequence_reconstructs_target", None)
    final_earliest = getattr(seq_label, "final_earliest_reconstruction_step", None)
    final_strength = getattr(seq_label, "final_reconstruction_strength", None)

    return replace(
        result,
        final_sequence_reconstructs_target=final_recon,
        final_earliest_reconstruction_step=final_earliest,
        final_reconstruction_strength=final_strength,
    )


# ---------------------------------------------------------------------------
# Confusion matrix and metrics
# ---------------------------------------------------------------------------


def compute_sequence_confusion_matrix(
    results: list[SequenceResult],
) -> SequenceConfusionMatrix:
    """Compute the sequence reconstruction confusion matrix (plan §64).

    Only includes results where both predicted and annotated values
    are available.
    """
    tp = fp = fn = tn = 0

    for r in results:
        if r.final_sequence_reconstructs_target is None:
            continue

        pred = r.predicted_sequence_reconstruction
        actual = r.final_sequence_reconstructs_target

        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1

    return SequenceConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def compute_earliest_step_metrics(
    results: list[SequenceResult],
) -> EarliestStepMetrics:
    """Compare predicted vs annotated earliest reconstruction step (plan §65).

    Only includes sequences where:
    - Both predicted and annotated define an earliest step.
    - The annotated sequence reconstructs the target.
    """
    errors: list[int] = []
    n_earlier = 0
    n_later = 0
    n_exact = 0

    for r in results:
        if r.final_sequence_reconstructs_target is None:
            continue
        if not r.final_sequence_reconstructs_target:
            continue
        if r.predicted_earliest_reconstruction_step is None:
            continue
        if r.final_earliest_reconstruction_step is None:
            continue

        pred_step = r.predicted_earliest_reconstruction_step
        actual_step = r.final_earliest_reconstruction_step
        error = pred_step - actual_step

        errors.append(abs(error))

        if error < 0:
            n_earlier += 1
        elif error > 0:
            n_later += 1
        else:
            n_exact += 1

    n = len(errors)
    exact_acc = n_exact / n if n > 0 else 0.0
    mae = sum(errors) / n if n > 0 else 0.0

    return EarliestStepMetrics(
        n_compared=n,
        exact_step_accuracy=exact_acc,
        mean_absolute_step_error=mae,
        n_predicted_earlier=n_earlier,
        n_predicted_later=n_later,
        n_exact_match=n_exact,
    )

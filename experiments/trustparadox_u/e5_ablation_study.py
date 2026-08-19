"""E5-009: Component ablation study (Iteration 10 + repair).

Defines four component ablations plus the full-system baseline.
Each ablation reruns the canonical firewall with exactly one component
disabled (§20-§26 repair).

Plan references:
    §55  component ablations
    §56  ablation rule (remove one component, hold others constant)
    §57  ablation metrics (PU-RER, RR, CRR, utility, FBR, attack breakdown)
    §58  ablation interpretation

Ablation set:
    A0  Full ForgetFlow (baseline)
    A1  − Semantic detector
    A2  − Recipient/history-aware state (ForgetGraph → RecipientHistory)
    A3  − ReconstructGuard (→ ReconstructionChecker)
    A4  − Purge/recontamination handling (→ ContaminationTracker)

Repair constraints (§20-§26):
    All ablations rerun the canonical firewall, not post-hoc modification.
    Attack labels are NOT used as runtime inputs (§23).
    A2 actually disables history state (§24).
    A3 actually disables reconstruction guard (§25).
    A4 actually disables purge state management (§26).

Exit criteria (plan §115):
    four core ablations complete
    same frozen test/evaluation config
    no ablation-specific test tuning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Ablation definitions (plan §55)
# ---------------------------------------------------------------------------

ABLATION_IDS: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4")

ABLATION_DESCRIPTIONS: dict[str, str] = {
    "A0": "Full ForgetFlow (all components)",
    "A1": "− Semantic detector",
    "A2": "− Recipient/history-aware state (ForgetGraph)",
    "A3": "− ReconstructGuard",
    "A4": "− Purge/recontamination handling",
}

# Which component each ablation disables (None = full system)
ABLATION_DISABLED_COMPONENT: dict[str, str | None] = {
    "A0": None,
    "A1": "semantic",
    "A2": "history",
    "A3": "reconstruction_guard",
    "A4": "purge",
}


# ---------------------------------------------------------------------------
# Ablation specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationSpec:
    """One ablation configuration."""

    ablation_id: str
    description: str
    disabled_component: str | None
    semantic_enabled: bool
    history_enabled: bool
    reconstruction_guard: bool
    purge_enabled: bool


def get_ablation_specs() -> list[AblationSpec]:
    """Return the full list of ablation specs."""
    return [
        AblationSpec(
            ablation_id="A0",
            description=ABLATION_DESCRIPTIONS["A0"],
            disabled_component=None,
            semantic_enabled=True,
            history_enabled=True,
            reconstruction_guard=True,
            purge_enabled=True,
        ),
        AblationSpec(
            ablation_id="A1",
            description=ABLATION_DESCRIPTIONS["A1"],
            disabled_component="semantic",
            semantic_enabled=False,
            history_enabled=True,
            reconstruction_guard=True,
            purge_enabled=True,
        ),
        AblationSpec(
            ablation_id="A2",
            description=ABLATION_DESCRIPTIONS["A2"],
            disabled_component="history",
            semantic_enabled=True,
            history_enabled=False,
            reconstruction_guard=True,
            purge_enabled=True,
        ),
        AblationSpec(
            ablation_id="A3",
            description=ABLATION_DESCRIPTIONS["A3"],
            disabled_component="reconstruction_guard",
            semantic_enabled=True,
            history_enabled=True,
            reconstruction_guard=False,
            purge_enabled=True,
        ),
        AblationSpec(
            ablation_id="A4",
            description=ABLATION_DESCRIPTIONS["A4"],
            disabled_component="purge",
            semantic_enabled=True,
            history_enabled=True,
            reconstruction_guard=True,
            purge_enabled=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Real ablation execution via FirewallRunner (§20-§26)
# ---------------------------------------------------------------------------


def get_ablation_override(spec: AblationSpec) -> dict[str, bool]:
    """Convert an AblationSpec to a FirewallRunner ablation_override dict.

    This maps the ablation's disabled component to the corresponding
    FirewallRunnerConfig flag override.
    """
    override: dict[str, bool] = {}
    if not spec.semantic_enabled:
        override["semantic_enabled"] = False
    if not spec.history_enabled:
        override["history_enabled"] = False
    if not spec.reconstruction_guard:
        override["reconstruction_guard"] = False
    if not spec.purge_enabled:
        override["purge_enabled"] = False
    return override


def run_ablation_via_firewall(
    spec: AblationSpec,
    features_by_id: dict[str, dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    *,
    tau_sem: float,
    reconstruction_threshold: float = 0.60,
    episode_metadata: dict[str, Any] | None = None,
    raw_text_by_id: dict[str, str] | None = None,
    corpus_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run one ablation by re-executing the canonical firewall (§20).

    Creates a FirewallRunner with the ablation's component disabled
    and processes every candidate through it.  This is a REAL re-execution,
    not a post-hoc modification of row outputs.

    Args:
        spec: Ablation specification.
        features_by_id: Pre-computed detector features.
        row_labels_by_id: candidate_id -> label dict.
        tau_sem: Frozen semantic threshold.
        reconstruction_threshold: Frozen reconstruction threshold.
        episode_metadata: Fragment maps and fact chains.
        raw_text_by_id: candidate_id -> raw corpus text (§6).
        corpus_by_id: candidate_id -> corpus dict for forget record
            registration (§7).  Keys: scenario_id, secret_variant_id.

    Returns:
        List of row result dicts from the ablated firewall run.
    """
    from .e5_firewall_runner import (
        build_e5_forget_record,
        create_firewall_runner,
        extended_result_to_dict,
    )

    override = get_ablation_override(spec)
    runner = create_firewall_runner(
        condition_id="C4",  # All ablations start from C4 config
        semantic_threshold=tau_sem,
        reconstruction_threshold=reconstruction_threshold,
        episode_metadata=episode_metadata,
        ablation_override=override if override else None,
    )

    results: list[dict[str, Any]] = []
    for cid, features in sorted(features_by_id.items()):
        label = row_labels_by_id.get(cid, {})

        # Register forget target from corpus metadata (§7)
        if corpus_by_id is not None:
            corpus = corpus_by_id.get(cid, {})
            scenario_id = corpus.get("scenario_id", label.get("scenario_id", ""))
            secret_variant_id = corpus.get(
                "secret_variant_id", label.get("secret_variant_id", "")
            )
            if scenario_id and secret_variant_id:
                try:
                    forget_record = build_e5_forget_record(
                        scenario_id=scenario_id,
                        secret_variant_id=secret_variant_id,
                    )
                    runner.register_forget_record(forget_record)
                except KeyError:
                    pass

        # Get raw text (§6)
        raw_text = ""
        if raw_text_by_id is not None:
            raw_text = raw_text_by_id.get(cid, "")

        er = runner.process_row(
            candidate_id=cid,
            scenario_id=label.get("scenario_id", ""),
            trust_level=label.get("trust_level", "default"),
            features=features,
            split=label.get("split", "development"),
            raw_text=raw_text,
            input_content_sha=features.get("content_sha256", ""),
            recipient_id=label.get("recipient_id", "default"),
        )
        results.append(extended_result_to_dict(er))

        # Reset state between independent rows (§8)
        runner.clear_recipient_state()

    return results


# ---------------------------------------------------------------------------
# Ablation metrics (plan §57)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationMetrics:
    """Metrics for one ablation (plan §57)."""

    ablation_id: str
    description: str
    n_eligible: int
    n_leaking: int
    n_leaking_blocked: int
    n_non_leaking: int
    n_fp: int
    n_useful_eligible: int
    n_useful_preserved: int
    leakage_prevention: float  # PU-RER proxy: recall among leaking rows
    fbr: float  # false blocking rate
    utility_retention: float
    attack_type_breakdown: dict[str, dict[str, float]]


def compute_ablation_metrics(
    ablated_results: list[dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
    ablation_id: str,
    description: str,
) -> AblationMetrics:
    """Compute metrics for one ablation (plan §57).

    Args:
        ablated_results: Row result dicts after ablation applied.
        row_labels_by_id: candidate_id → label dict.
        corpus_by_id: candidate_id → corpus dict.
        ablation_id: Ablation identifier.
        description: Human-readable description.

    Returns:
        AblationMetrics with all plan §57 metrics.
    """
    results_by_id = {r["candidate_id"]: r for r in ablated_results}

    n_eligible = 0
    n_leaking = 0
    n_leaking_blocked = 0
    n_non_leaking = 0
    n_fp = 0
    n_useful_eligible = 0
    n_useful_preserved = 0

    # Per-attack-type tracking
    attack_counts: dict[str, dict[str, int]] = {}

    for cid, label in row_labels_by_id.items():
        if label.get("is_unresolved"):
            continue
        result = results_by_id.get(cid)
        if result is None:
            continue

        corpus = corpus_by_id.get(cid, {})
        attack_type = corpus.get("attack_type", "unknown")

        # Initialise attack-type bucket
        if attack_type not in attack_counts:
            attack_counts[attack_type] = {
                "n": 0, "n_leaking": 0, "n_blocked": 0,
                "n_non_leaking": 0, "n_fp": 0,
                "n_useful": 0, "n_useful_preserved": 0,
            }
        ac = attack_counts[attack_type]

        n_eligible += 1
        ac["n"] += 1

        is_leaking = label.get("final_target_leakage") is True
        is_useful = label.get("final_task_useful") is True
        is_blocked = result.get("blocked", False)
        is_allowed = result.get("allowed", False)

        if is_leaking:
            n_leaking += 1
            ac["n_leaking"] += 1
            if is_blocked:
                n_leaking_blocked += 1
                ac["n_blocked"] += 1
        elif label.get("final_target_leakage") is False:
            n_non_leaking += 1
            ac["n_non_leaking"] += 1
            if is_blocked:
                n_fp += 1
                ac["n_fp"] += 1

        if is_useful:
            n_useful_eligible += 1
            ac["n_useful"] += 1
            if is_allowed:
                n_useful_preserved += 1
                ac["n_useful_preserved"] += 1

    leakage_prevention = (
        n_leaking_blocked / n_leaking if n_leaking > 0 else 0.0
    )
    fbr = n_fp / n_non_leaking if n_non_leaking > 0 else 0.0
    utility_retention = (
        n_useful_preserved / n_useful_eligible
        if n_useful_eligible > 0
        else 0.0
    )

    # Compute per-attack-type metrics
    attack_breakdown: dict[str, dict[str, float]] = {}
    for at, counts in attack_counts.items():
        at_leak_prev = (
            counts["n_blocked"] / counts["n_leaking"]
            if counts["n_leaking"] > 0
            else 0.0
        )
        at_fbr = (
            counts["n_fp"] / counts["n_non_leaking"]
            if counts["n_non_leaking"] > 0
            else 0.0
        )
        at_util = (
            counts["n_useful_preserved"] / counts["n_useful"]
            if counts["n_useful"] > 0
            else 0.0
        )
        attack_breakdown[at] = {
            "n": counts["n"],
            "leakage_prevention": at_leak_prev,
            "fbr": at_fbr,
            "utility_retention": at_util,
        }

    return AblationMetrics(
        ablation_id=ablation_id,
        description=description,
        n_eligible=n_eligible,
        n_leaking=n_leaking,
        n_leaking_blocked=n_leaking_blocked,
        n_non_leaking=n_non_leaking,
        n_fp=n_fp,
        n_useful_eligible=n_useful_eligible,
        n_useful_preserved=n_useful_preserved,
        leakage_prevention=leakage_prevention,
        fbr=fbr,
        utility_retention=utility_retention,
        attack_type_breakdown=attack_breakdown,
    )


# ---------------------------------------------------------------------------
# Full ablation study runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationStudyResult:
    """Complete ablation study result."""

    ablations: tuple[AblationMetrics, ...]
    baseline_id: str  # "A0"

    @property
    def baseline(self) -> AblationMetrics:
        """Return the baseline (A0) metrics."""
        for a in self.ablations:
            if a.ablation_id == self.baseline_id:
                return a
        raise ValueError(f"Baseline {self.baseline_id} not found")


def run_ablation_study(
    features_by_id: dict[str, dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    *,
    tau_sem: float,
    reconstruction_threshold: float = 0.60,
    episode_metadata: dict[str, Any] | None = None,
    raw_text_by_id: dict[str, str] | None = None,
    corpus_by_id: dict[str, dict[str, Any]] | None = None,
) -> AblationStudyResult:
    """Run the full ablation study via real re-execution (§20-§26).

    Each ablation reruns the canonical firewall with one component
    disabled.  Attack labels are NOT used as runtime inputs (§23).

    Args:
        features_by_id: Pre-computed detector features.
        row_labels_by_id: candidate_id -> label dict.
        tau_sem: Frozen semantic threshold.
        reconstruction_threshold: Frozen reconstruction threshold.
        episode_metadata: Fragment maps and fact chains.
        raw_text_by_id: candidate_id -> raw corpus text (§6).
        corpus_by_id: candidate_id -> corpus dict for target registration (§7).

    Returns:
        AblationStudyResult with metrics for all ablations.
    """
    specs = get_ablation_specs()
    ablations: list[AblationMetrics] = []

    for spec in specs:
        # Real re-execution via FirewallRunner (§20)
        ablated_results = run_ablation_via_firewall(
            spec=spec,
            features_by_id=features_by_id,
            row_labels_by_id=row_labels_by_id,
            tau_sem=tau_sem,
            reconstruction_threshold=reconstruction_threshold,
            episode_metadata=episode_metadata,
            raw_text_by_id=raw_text_by_id,
            corpus_by_id=corpus_by_id,
        )

        # Build corpus_by_id from labels (no attack_type injection)
        corpus_by_id = {
            cid: {"attack_type": label.get("attack_type", "unknown")}
            for cid, label in row_labels_by_id.items()
        }

        metrics = compute_ablation_metrics(
            ablated_results=ablated_results,
            row_labels_by_id=row_labels_by_id,
            corpus_by_id=corpus_by_id,
            ablation_id=spec.ablation_id,
            description=spec.description,
        )
        ablations.append(metrics)

    return AblationStudyResult(
        ablations=tuple(ablations),
        baseline_id="A0",
    )


# ---------------------------------------------------------------------------
# Relative impact computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationImpact:
    """Relative impact of removing one component."""

    ablation_id: str
    disabled_component: str | None
    leakage_prevention_delta: float  # baseline - ablation
    fbr_delta: float  # ablation - baseline (positive = worse)
    utility_delta: float  # baseline - ablation (positive = better in baseline)


def compute_ablation_impacts(
    study: AblationStudyResult,
) -> list[AblationImpact]:
    """Compute relative impact of each ablation vs baseline.

    Args:
        study: Complete ablation study result.

    Returns:
        List of AblationImpact, one per non-baseline ablation.
    """
    baseline = study.baseline
    impacts: list[AblationImpact] = []

    for ablation in study.ablations:
        if ablation.ablation_id == baseline.ablation_id:
            continue

        disabled = ABLATION_DISABLED_COMPONENT.get(ablation.ablation_id)
        impacts.append(AblationImpact(
            ablation_id=ablation.ablation_id,
            disabled_component=disabled,
            leakage_prevention_delta=(
                baseline.leakage_prevention - ablation.leakage_prevention
            ),
            fbr_delta=ablation.fbr - baseline.fbr,
            utility_delta=baseline.utility_retention - ablation.utility_retention,
        ))

    return impacts


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def ablation_metrics_to_dict(
    metrics: list[AblationMetrics],
) -> list[dict[str, Any]]:
    """Serialise ablation metrics to list of dicts."""
    return [
        {
            "ablation_id": m.ablation_id,
            "description": m.description,
            "n_eligible": m.n_eligible,
            "n_leaking": m.n_leaking,
            "n_leaking_blocked": m.n_leaking_blocked,
            "n_non_leaking": m.n_non_leaking,
            "n_fp": m.n_fp,
            "n_useful_eligible": m.n_useful_eligible,
            "n_useful_preserved": m.n_useful_preserved,
            "leakage_prevention": m.leakage_prevention,
            "fbr": m.fbr,
            "utility_retention": m.utility_retention,
            "attack_type_breakdown": m.attack_type_breakdown,
        }
        for m in metrics
    ]


def ablation_impacts_to_dict(
    impacts: list[AblationImpact],
) -> list[dict[str, Any]]:
    """Serialise ablation impacts to list of dicts."""
    return [
        {
            "ablation_id": i.ablation_id,
            "disabled_component": i.disabled_component,
            "leakage_prevention_delta": i.leakage_prevention_delta,
            "fbr_delta": i.fbr_delta,
            "utility_delta": i.utility_delta,
        }
        for i in impacts
    ]

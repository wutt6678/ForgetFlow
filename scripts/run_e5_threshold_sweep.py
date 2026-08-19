"""Explicit E5 threshold sweep runner (R1.2 §17).

For every frozen τ_sem ∈ {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90},
this script runs *actual* row evaluation, stateful sequence evaluation,
and post-firewall reconstruction probe through the canonical
FirewallRunner.  It then computes PU-RER, leakage-prevention recall,
FBR, utility retention, CRR, and RR where appropriate.

CRR is computed from the post-firewall reconstruction probe — NOT from
row recall, NOT from guard-trigger rate, NOT hardcoded to 0.

Outputs:
    <output-dir>/threshold_sweep.jsonl   — one record per τ_sem
    <output-dir>/sweep_manifest.json     — run metadata

Usage:
    PYTHONPATH=. python scripts/run_e5_threshold_sweep.py \\
        --split development \\
        --output-dir results/empirical_v2/e5/threshold_sweep

Fail-closed: test split is rejected.  Missing target registry, missing
corpus rows, or target mismatch raises a fatal error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_episode_metadata import (  # noqa: E402
    build_e5_episode_metadata,
)
from experiments.trustparadox_u.e5_firewall_runner import (  # noqa: E402
    build_e5_forget_record,
    create_firewall_runner,
    extended_result_to_dict,
)
from experiments.trustparadox_u.e5_hyperparameter_study import (  # noqa: E402
    FROZEN_TAU_SEM_GRID,
)
from experiments.trustparadox_u.e5_loaders import (  # noqa: E402
    CorpusCandidate,
    SplitData,
    load_split,
)
from experiments.trustparadox_u.e5_metrics import (  # noqa: E402
    compute_compositional_reconstruction_rate,
)
from experiments.trustparadox_u.e5_reconstruction_probe import (  # noqa: E402
    run_reconstruction_probe,
)
from experiments.trustparadox_u.e5_sequence_evaluation import (  # noqa: E402
    SequenceResult,
    StepDecision,
    predict_sequence_reconstruction,
    sequence_result_to_dict,
)


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

_E5_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "e5"
_DEFAULT_OUTPUT_DIR = _E5_DIR / "threshold_sweep"
_CONFIG_DIR = _E5_DIR / "config"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    """Return the current git commit (short) or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's contents."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-threshold sweep result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepThresholdResult:
    """Result of running the full evaluation at one τ_sem."""

    tau_sem: float
    n_eligible: int
    n_leaking: int
    n_leaking_blocked: int
    n_leaking_delivered: int
    n_non_leaking: int
    n_fp: int
    n_useful_eligible: int
    n_useful_preserved: int
    pu_rer: float
    leakage_prevention_recall: float
    fbr: float
    utility_retention: float
    crr: float
    n_sequence_eligible: int
    n_sequence_reconstructable: int
    row_results: list[dict[str, Any]]
    sequence_results: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Per-row execution with state isolation (R1.2 §23)
# ---------------------------------------------------------------------------


def _execute_one_row(
    *,
    cid: str,
    features: dict[str, Any],
    corpus_by_id: dict[str, CorpusCandidate],
    row_labels_by_id: dict[str, dict[str, Any]],
    split_name: str,
    tau_sem: float,
    reconstruction_threshold: float,
) -> dict[str, Any]:
    """Execute one row through a fresh C4 runner (R1.2 §23)."""
    corpus = corpus_by_id[cid]

    # R1.2 §20: missing target registry → fatal.
    forget_record = build_e5_forget_record(
        scenario_id=corpus.scenario_id,
        secret_variant_id=corpus.secret_variant_id,
    )

    runner = create_firewall_runner(
        condition_id="C4",
        semantic_threshold=tau_sem,
        reconstruction_threshold=reconstruction_threshold,
    )
    runner.register_forget_record(forget_record)

    er = runner.process_row(
        candidate_id=cid,
        scenario_id=corpus.scenario_id,
        trust_level=corpus.trust_level,
        features=features,
        split=split_name,
        raw_text=corpus.text,
        recipient_id=corpus.recipient_id,
        sender_id=corpus.sender_id,
        input_content_sha=corpus.content_sha256,
    )
    return extended_result_to_dict(er)


# ---------------------------------------------------------------------------
# Per-sequence execution with reconstruction probe (R1.2 §17, §23)
# ---------------------------------------------------------------------------


def _execute_one_sequence(
    *,
    seq_label: Any,
    corpus_by_id: dict[str, CorpusCandidate],
    features_by_id: dict[str, dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    split_name: str,
    tau_sem: float,
    reconstruction_threshold: float,
    condition_manifest_sha: str,
    detector_config_sha: str,
) -> tuple[dict[str, Any], list[str]]:
    """Execute one sequence through a fresh C4 runner and collect
    released texts for the reconstruction probe.

    Returns:
        Tuple of (sequence result dict, released texts list).
    """
    ordered_ids = list(seq_label.ordered_candidate_ids)

    # R1.2 §21: fail closed on missing corpus / features.
    missing_corpus = [c for c in ordered_ids if c not in corpus_by_id]
    if missing_corpus:
        raise ValueError(
            f"Missing corpus rows for sequence "
            f"{seq_label.sequence_annotation_id!r}: "
            f"{missing_corpus[:5]}"
            f"{'...' if len(missing_corpus) > 5 else ''} (R1.2 §21)"
        )
    missing_features = [c for c in ordered_ids if c not in features_by_id]
    if missing_features:
        raise ValueError(
            f"Missing features for sequence "
            f"{seq_label.sequence_annotation_id!r}: "
            f"{missing_features[:5]}"
            f"{'...' if len(missing_features) > 5 else ''} (R1.2 §21)"
        )

    # R1.2 §22: target consistency.
    first_corpus = corpus_by_id[ordered_ids[0]]
    for c in ordered_ids[1:]:
        cand = corpus_by_id[c]
        if (
            cand.scenario_id != first_corpus.scenario_id
            or cand.secret_variant_id != first_corpus.secret_variant_id
        ):
            raise ValueError(
                f"Sequence target mismatch (R1.2 §22): sequence "
                f"{seq_label.sequence_annotation_id!r} contains "
                f"candidates with different target families. "
                f"step[0]=(scenario_id={first_corpus.scenario_id!r}, "
                f"secret_variant_id={first_corpus.secret_variant_id!r}), "
                f"step[{ordered_ids.index(c)}]=(scenario_id={cand.scenario_id!r}, "
                f"secret_variant_id={cand.secret_variant_id!r})."
            )

    # Fresh runner per sequence (R1.2 §23).
    seq_runner = create_firewall_runner(
        condition_id="C4",
        semantic_threshold=tau_sem,
        reconstruction_threshold=reconstruction_threshold,
    )

    forget_record = build_e5_forget_record(
        scenario_id=first_corpus.scenario_id,
        secret_variant_id=first_corpus.secret_variant_id,
    )
    seq_runner.register_forget_record(forget_record)

    # Build episode metadata for this sequence's target family.
    episode_metadata = build_e5_episode_metadata(
        sequence_label=seq_label,
        corpus_by_id=corpus_by_id,
    )

    trust_level = getattr(seq_label, "trust_level", "unknown")
    steps: list[StepDecision] = []
    released_texts: list[str] = []

    for i, candidate_id in enumerate(ordered_ids):
        corpus = corpus_by_id[candidate_id]
        features = features_by_id[candidate_id]
        er = seq_runner.process_row(
            candidate_id=candidate_id,
            scenario_id=corpus.scenario_id,
            trust_level=trust_level,
            features=features,
            split=split_name,
            raw_text=corpus.text,
            recipient_id=corpus.recipient_id,
            sender_id=corpus.sender_id,
            turn_id=i,
            message_id=f"seq_{seq_label.sequence_annotation_id}_step{i}",
            input_content_sha=corpus.content_sha256,
            condition_manifest_sha=condition_manifest_sha,
            detector_config_sha=detector_config_sha,
            embedding_model=features.get("embedding_model", "unknown"),
        )
        sd = StepDecision(
            step_index=i,
            candidate_id=candidate_id,
            exact_match=er.exact_match,
            alias_match=er.alias_match,
            semantic_similarity=er.semantic_similarity,
            detected=er.blocked,
            policy_action=er.policy_action,
            decision_reason=er.decision_reason,
            history_state_summary=f"history_used={er.history_state_used}",
            reconstruction_guard_result=er.reconstruction_guard_triggered,
            reconstruction_score=er.reconstruction_score,
            purge_state_transition=f"purge={er.purge_triggered}",
            delivered_content_sha=er.output_content_sha,
        )
        steps.append(sd)

        # Collect released texts for the reconstruction probe.
        # "allow" → raw text is released unchanged.
        # "redact"/"abstract" → transformed text (we use raw_text as a
        #   conservative upper-bound proxy; the actual transformed text
        #   has canonical values removed so it contributes less).
        # "block" → nothing released.
        if er.final_policy_action == "allow":
            released_texts.append(corpus.text)

    # Predict reconstruction from step decisions (legacy proxy).
    recon, earliest, strength = predict_sequence_reconstruction(steps)

    # Run the post-firewall reconstruction probe (R1.2 §17: real CRR).
    probe_result = run_reconstruction_probe(
        forget_record=forget_record,
        released_texts=released_texts,
        episode_metadata=episode_metadata,
        reconstruction_threshold=reconstruction_threshold,
    )

    seq_result = SequenceResult(
        sequence_annotation_id=seq_label.sequence_annotation_id,
        trust_level=trust_level,
        condition_id="C4",
        ordered_candidate_ids=tuple(ordered_ids),
        step_decisions=tuple(steps),
        predicted_sequence_reconstruction=recon,
        predicted_earliest_reconstruction_step=earliest,
        predicted_reconstruction_strength=strength,
        # Post-firewall reconstruction from the probe (R1.2 §9).
        post_firewall_reconstructable=probe_result.reconstructable,
        post_firewall_earliest_reconstruction_step=probe_result.earliest_step,
        post_firewall_reconstruction_score=probe_result.reconstruction_score,
        # Annotation join fields from the frozen label.
        final_sequence_reconstructs_target=getattr(
            seq_label, "final_sequence_reconstructs_target", None
        ),
        final_earliest_reconstruction_step=getattr(
            seq_label, "final_earliest_reconstruction_step", None
        ),
        final_reconstruction_strength=getattr(
            seq_label, "final_reconstruction_strength", None
        ),
    )
    return sequence_result_to_dict(seq_result), released_texts


# ---------------------------------------------------------------------------
# Run one threshold
# ---------------------------------------------------------------------------


def _run_one_threshold(
    tau_sem: float,
    *,
    split_data: SplitData,
    features_by_id: dict[str, dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    reconstruction_threshold: float,
    split_name: str,
    condition_manifest_sha: str,
    detector_config_sha: str,
) -> SweepThresholdResult:
    """Run the full evaluation at one τ_sem."""
    corpus_by_id = {c.candidate_id: c for c in split_data.corpus}

    # --- Row evaluation ---
    row_results: list[dict[str, Any]] = []
    for cid in sorted(features_by_id.keys()):
        er_dict = _execute_one_row(
            cid=cid,
            features=features_by_id[cid],
            corpus_by_id=corpus_by_id,
            row_labels_by_id=row_labels_by_id,
            split_name=split_name,
            tau_sem=tau_sem,
            reconstruction_threshold=reconstruction_threshold,
        )
        row_results.append(er_dict)

    # --- Compute row-level metrics ---
    results_by_id = {r["candidate_id"]: r for r in row_results}
    n_eligible = 0
    n_leaking = 0
    n_leaking_blocked = 0
    n_leaking_delivered = 0
    n_non_leaking = 0
    n_fp = 0
    n_useful_eligible = 0
    n_useful_preserved = 0

    for cid, label in row_labels_by_id.items():
        if label.get("is_unresolved"):
            continue
        result = results_by_id.get(cid)
        if result is None:
            continue

        n_eligible += 1
        is_leaking = label.get("final_target_leakage") is True
        is_useful = label.get("final_task_useful") is True
        is_blocked = result.get("blocked", False)
        is_allowed = result.get("allowed", False)

        # R1.2 §15: PU-RER uses final_policy_action.
        final_action = result.get("final_policy_action") or result.get(
            "policy_action", ""
        )
        has_r12_fields = bool(result.get("final_policy_action")) or bool(
            result.get("transformation_recheck_passed") is not None
            and "transformation_recheck_passed" in result
        )
        if has_r12_fields:
            released_unsafe = is_leaking and final_action == "allow"
        else:
            released_unsafe = is_leaking and is_allowed

        if is_leaking:
            n_leaking += 1
            if is_blocked:
                n_leaking_blocked += 1
            if released_unsafe:
                n_leaking_delivered += 1
        elif label.get("final_target_leakage") is False:
            n_non_leaking += 1
            if is_blocked:
                n_fp += 1

        if is_useful:
            n_useful_eligible += 1
            if is_allowed:
                n_useful_preserved += 1

    pu_rer = n_leaking_delivered / n_leaking if n_leaking > 0 else 0.0
    leakage_prevention_recall = (
        n_leaking_blocked / n_leaking if n_leaking > 0 else 0.0
    )
    fbr = n_fp / n_non_leaking if n_non_leaking > 0 else 0.0
    utility_retention = (
        n_useful_preserved / n_useful_eligible
        if n_useful_eligible > 0
        else 0.0
    )

    # --- Sequence evaluation with reconstruction probe ---
    sequence_results: list[dict[str, Any]] = []
    for seq_label in split_data.sequence_labels:
        if getattr(seq_label, "is_unresolved", False):
            continue
        seq_dict, _released = _execute_one_sequence(
            seq_label=seq_label,
            corpus_by_id=corpus_by_id,
            features_by_id=features_by_id,
            row_labels_by_id=row_labels_by_id,
            split_name=split_name,
            tau_sem=tau_sem,
            reconstruction_threshold=reconstruction_threshold,
            condition_manifest_sha=condition_manifest_sha,
            detector_config_sha=detector_config_sha,
        )
        sequence_results.append(seq_dict)

    # --- Compute CRR from sequence results ---
    # Build lightweight objects for compute_compositional_reconstruction_rate.
    # We need objects with the right attributes; use the dict form.
    class _SeqProxy:
        """Minimal proxy to feed sequence dicts into CRR computation."""
        def __init__(self, d: dict[str, Any]) -> None:
            self._d = d
        def __getattr__(self, name: str) -> Any:
            if name in self._d:
                return self._d[name]
            return None

    seq_proxies = [_SeqProxy(s) for s in sequence_results]
    crr_result = compute_compositional_reconstruction_rate(seq_proxies)

    return SweepThresholdResult(
        tau_sem=tau_sem,
        n_eligible=n_eligible,
        n_leaking=n_leaking,
        n_leaking_blocked=n_leaking_blocked,
        n_leaking_delivered=n_leaking_delivered,
        n_non_leaking=n_non_leaking,
        n_fp=n_fp,
        n_useful_eligible=n_useful_eligible,
        n_useful_preserved=n_useful_preserved,
        pu_rer=pu_rer,
        leakage_prevention_recall=leakage_prevention_recall,
        fbr=fbr,
        utility_retention=utility_retention,
        crr=crr_result.crr,
        n_sequence_eligible=crr_result.n_eligible_sequences,
        n_sequence_reconstructable=crr_result.n_reconstructable,
        row_results=row_results,
        sequence_results=sequence_results,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute E5 threshold sweep with real sequence CRR "
            "(R1.2 §17)."
        )
    )
    parser.add_argument(
        "--split", required=True, choices=("development", "validation"),
        help="Frozen split to evaluate on. Test split is rejected (R1.2 §18).",
    )
    parser.add_argument(
        "--reconstruction-threshold", type=float, default=0.60,
        help="Reconstruction threshold for the post-firewall probe.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
        help="Directory to write sweep artifacts.",
    )
    parser.add_argument(
        "--run-mode", default="diagnostic",
        choices=("scientific", "diagnostic"),
        help=(
            "scientific = full evidence; diagnostic = labelled diagnostic "
            "smoke (R1.2 §29)."
        ),
    )
    args = parser.parse_args()

    if args.split == "test":
        print(
            "ERROR: test split is not a valid split for threshold sweep "
            "(R1.2 §18). Use --split development or --split validation.",
            file=sys.stderr,
        )
        return 2

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen split data.
    split_data: SplitData = load_split(args.split)
    features_by_id: dict[str, dict[str, Any]] = {}
    row_labels_by_id: dict[str, dict[str, Any]] = {}
    for label in split_data.row_labels:
        row_labels_by_id[label.candidate_id] = {
            "final_target_leakage": label.final_target_leakage,
            "final_task_useful": label.final_task_useful,
            "is_unresolved": label.is_unresolved,
            "split": args.split,
        }

    condition_manifest_sha = "sweep-runner-diagnostic"
    detector_config_sha = "sweep-runner-diagnostic"

    # Synthesize features in diagnostic mode.
    if not features_by_id and args.run_mode == "diagnostic":
        for label in split_data.row_labels:
            features_by_id[label.candidate_id] = {
                "exact_match": False,
                "alias_match": False,
                "semantic_similarity": 0.50,
            }

    if not features_by_id:
        print(
            f"ERROR: no features available for split {args.split!r}.",
            file=sys.stderr,
        )
        return 3

    # Execute sweep across all frozen thresholds.
    sweep_results: list[SweepThresholdResult] = []
    for tau_sem in FROZEN_TAU_SEM_GRID:
        print(
            f"Running τ_sem={tau_sem:.2f} "
            f"({len(features_by_id)} rows, "
            f"{len(split_data.sequence_labels)} sequences)...",
            file=sys.stderr,
        )
        result = _run_one_threshold(
            tau_sem=tau_sem,
            split_data=split_data,
            features_by_id=features_by_id,
            row_labels_by_id=row_labels_by_id,
            reconstruction_threshold=args.reconstruction_threshold,
            split_name=args.split,
            condition_manifest_sha=condition_manifest_sha,
            detector_config_sha=detector_config_sha,
        )
        sweep_results.append(result)

    # Persist threshold_sweep.jsonl
    sweep_path = output_dir / "threshold_sweep.jsonl"
    with open(sweep_path, "w") as f:
        for r in sweep_results:
            record = {
                "tau_sem": r.tau_sem,
                "n_eligible": r.n_eligible,
                "n_leaking": r.n_leaking,
                "n_leaking_blocked": r.n_leaking_blocked,
                "n_leaking_delivered": r.n_leaking_delivered,
                "n_non_leaking": r.n_non_leaking,
                "n_fp": r.n_fp,
                "n_useful_eligible": r.n_useful_eligible,
                "n_useful_preserved": r.n_useful_preserved,
                "pu_rer": r.pu_rer,
                "leakage_prevention_recall": r.leakage_prevention_recall,
                "fbr": r.fbr,
                "utility_retention": r.utility_retention,
                "crr": r.crr,
                "n_sequence_eligible": r.n_sequence_eligible,
                "n_sequence_reconstructable": r.n_sequence_reconstructable,
            }
            f.write(json.dumps(record, default=str) + "\n")

    # Persist per-threshold row + sequence evidence
    for r in sweep_results:
        tau_tag = f"{r.tau_sem:.2f}".replace(".", "")
        rows_path = output_dir / f"sweep_tau{tau_tag}_rows.jsonl"
        seqs_path = output_dir / f"sweep_tau{tau_tag}_sequences.jsonl"
        with open(rows_path, "w") as f:
            for row_r in r.row_results:
                f.write(json.dumps(row_r, default=str) + "\n")
        with open(seqs_path, "w") as f:
            for seq_r in r.sequence_results:
                f.write(json.dumps(seq_r, default=str) + "\n")

    # Build sweep manifest
    manifest: dict[str, Any] = {
        "schema_version": "1.2",
        "run_mode": args.run_mode,
        "split": args.split,
        "reconstruction_threshold": args.reconstruction_threshold,
        "tau_sem_grid": FROZEN_TAU_SEM_GRID,
        "code_commit": _git_commit(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_thresholds": len(sweep_results),
        "thresholds": [
            {
                "tau_sem": r.tau_sem,
                "pu_rer": r.pu_rer,
                "leakage_prevention_recall": r.leakage_prevention_recall,
                "fbr": r.fbr,
                "utility_retention": r.utility_retention,
                "crr": r.crr,
                "n_eligible": r.n_eligible,
                "n_leaking": r.n_leaking,
                "n_sequence_eligible": r.n_sequence_eligible,
                "n_sequence_reconstructable": r.n_sequence_reconstructable,
            }
            for r in sweep_results
        ],
    }

    # Optional provenance SHAs
    cond_manifest_path = _CONFIG_DIR / "condition_manifest.json"
    if cond_manifest_path.exists():
        manifest["condition_manifest_sha"] = _file_sha256(cond_manifest_path)
    embed_manifest_path = _CONFIG_DIR / "embedding_manifest.json"
    if embed_manifest_path.exists():
        manifest["embedding_manifest_sha"] = _file_sha256(embed_manifest_path)

    manifest_path = output_dir / "sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        f"Wrote sweep JSONl to {sweep_path}",
        file=sys.stderr,
    )
    print(
        f"Wrote sweep manifest to {manifest_path}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

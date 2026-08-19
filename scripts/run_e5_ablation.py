"""Explicit E5 ablation runner (R1.2 §16, §23).

This script executes the four named component ablations (A0-A4) and the
full-system baseline A0 through the *canonical* :class:`FirewallRunner`.
It does NOT re-implement ablation logic, and it does NOT reconstruct
pseudo-features from primary row results.

Per R1.2 §16: each ablation receives the same corpus, features, target
records, episode metadata, tau_sem, reconstruction threshold, and
history window — only the named module is disabled.

Per R1.2 §23: each independent row gets a fresh runner; the
state-isolation rule is that no contamination, history, or audit state
may leak across independent rows or sequences.

Usage:
    PYTHONPATH=. python scripts/run_e5_ablation.py \\
        --split development \\
        --output-dir results/empirical_v2/e5/ablation

Outputs:
    <output-dir>/ablation_manifest.json
    <output-dir>/ablation_<AID>_rows.jsonl
    <output-dir>/ablation_<AID>_sequences.jsonl
    <output-dir>/ablation_summary.json

Fail-closed: missing target registry, missing corpus row, or target
mismatch raises a fatal error and aborts the entire ablation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_ablation_study import (  # noqa: E402
    ABLATION_DESCRIPTIONS,
    ABLATION_DISABLED_COMPONENT,
    ABLATION_IDS,
    AblationSpec,
    compute_ablation_metrics,
    get_ablation_override,
    get_ablation_specs,
)
from experiments.trustparadox_u.e5_episode_metadata import (  # noqa: E402
    build_e5_episode_metadata,
)
from experiments.trustparadox_u.e5_firewall_runner import (  # noqa: E402
    build_e5_forget_record,
    create_firewall_runner,
    extended_result_to_dict,
)
from experiments.trustparadox_u.e5_loaders import (  # noqa: E402
    CorpusCandidate,
    SplitData,
    load_split,
)
from experiments.trustparadox_u.e5_metrics import (  # noqa: E402
    compute_row_metrics,
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
_DEFAULT_OUTPUT_DIR = _E5_DIR / "ablation"
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


@dataclass(frozen=True)
class AblationRunResult:
    """Result of executing one ablation through the canonical runner."""

    ablation_id: str
    spec: AblationSpec
    row_results: list[dict[str, Any]]
    sequence_results: list[dict[str, Any]]
    row_count: int
    sequence_count: int
    runner_config_sha: str
    metrics: dict[str, Any]


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
    spec: AblationSpec,
    tau_sem: float,
    reconstruction_threshold: float,
    episode_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Execute one row through an ablation's runner with full state
    isolation (R1.2 §23).

    A fresh runner is created per row, so contamination state, history,
    and audit state cannot leak across rows. The forget target is
    registered from the frozen registry (R1.2 §20: missing target
    raises a fatal error).
    """
    label = row_labels_by_id[cid]
    corpus = corpus_by_id[cid]

    # R1.2 §20: missing target registry mapping fails closed.
    forget_record = build_e5_forget_record(
        scenario_id=corpus.scenario_id,
        secret_variant_id=corpus.secret_variant_id,
    )

    # Fresh runner per row (R1.2 §23: state isolation).
    override = get_ablation_override(spec)
    runner = create_firewall_runner(
        condition_id="C4",  # all ablations start from C4 config
        semantic_threshold=tau_sem,
        reconstruction_threshold=reconstruction_threshold,
        episode_metadata=episode_metadata,
        ablation_override=override if override else None,
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
# Per-sequence execution with state isolation (R1.2 §23)
# ---------------------------------------------------------------------------


def _execute_one_sequence(
    *,
    seq_label: Any,
    corpus_by_id: dict[str, CorpusCandidate],
    features_by_id: dict[str, dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    split_name: str,
    spec: AblationSpec,
    tau_sem: float,
    reconstruction_threshold: float,
    episode_metadata: dict[str, Any],
    condition_manifest_sha: str,
    detector_config_sha: str,
) -> dict[str, Any]:
    """Execute one sequence through an ablation's runner with full state
    isolation (R1.2 §23).

    R1.2a: Uses the shared execute_e5_sequence() helper which builds
    per-sequence metadata, creates runner WITH metadata, and runs the
    post-firewall reconstruction probe for CRR.
    """
    from experiments.trustparadox_u.e5_sequence_evaluation import (
        execute_e5_sequence,
        sequence_result_to_dict,
    )

    # R1.2a §12-§14: use the shared executor with ablation overrides.
    # The shared executor builds per-sequence episode metadata internally
    # (R1.2a §13: no split-global metadata reuse).
    override = get_ablation_override(spec)
    seq_result, _released = execute_e5_sequence(
        condition_id="C4",
        seq_label=seq_label,
        corpus_by_id=corpus_by_id,
        features_by_id=features_by_id,
        tau_sem=tau_sem,
        reconstruction_threshold=reconstruction_threshold,
        split_name=split_name,
        ablation_override=override if override else None,
        condition_manifest_sha=condition_manifest_sha,
        detector_config_sha=detector_config_sha,
    )

    return sequence_result_to_dict(seq_result)


# ---------------------------------------------------------------------------
# Run one ablation
# ---------------------------------------------------------------------------


def run_one_ablation(
    spec: AblationSpec,
    split_data: SplitData,
    features_by_id: dict[str, dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    *,
    tau_sem: float,
    reconstruction_threshold: float,
    split_name: str,
    condition_manifest_sha: str,
    detector_config_sha: str,
) -> AblationRunResult:
    """Run a single ablation through the canonical runner.

    Each row gets a fresh runner (R1.2 §23). Each sequence gets a
    fresh runner shared across its steps. No state is reused across
    independent rows or sequences.
    """
    corpus_by_id = {c.candidate_id: c for c in split_data.corpus}

    # Build episode metadata once for the whole ablation (R1.2 §10:
    # identical structural metadata across C0-C4 and A0-A4 for the
    # same target family). Use the first sequence's first ordered
    # candidate as the target family anchor.
    if split_data.sequence_labels:
        # Pick the first non-unresolved sequence.
        anchor_seq = None
        for sl in split_data.sequence_labels:
            if not getattr(sl, "is_unresolved", False):
                anchor_seq = sl
                break
        if anchor_seq is not None and getattr(
            anchor_seq, "ordered_candidate_ids", ()
        ):
            episode_metadata = build_e5_episode_metadata(
                sequence_label=anchor_seq,
                corpus_by_id=corpus_by_id,
            )
        else:
            episode_metadata = {}
    else:
        # No sequences; build empty episode metadata.
        episode_metadata = {}

    # Row execution
    row_results: list[dict[str, Any]] = []
    for cid in sorted(features_by_id.keys()):
        er_dict = _execute_one_row(
            cid=cid,
            features=features_by_id[cid],
            corpus_by_id=corpus_by_id,
            row_labels_by_id=row_labels_by_id,
            split_name=split_name,
            spec=spec,
            tau_sem=tau_sem,
            reconstruction_threshold=reconstruction_threshold,
            episode_metadata=episode_metadata,
        )
        row_results.append(er_dict)

    # Sequence execution (independent sequences get fresh runners)
    sequence_results: list[dict[str, Any]] = []
    for seq_label in split_data.sequence_labels:
        if getattr(seq_label, "is_unresolved", False):
            continue
        seq_dict = _execute_one_sequence(
            seq_label=seq_label,
            corpus_by_id=corpus_by_id,
            features_by_id=features_by_id,
            row_labels_by_id=row_labels_by_id,
            split_name=split_name,
            spec=spec,
            tau_sem=tau_sem,
            reconstruction_threshold=reconstruction_threshold,
            episode_metadata=episode_metadata,
            condition_manifest_sha=condition_manifest_sha,
            detector_config_sha=detector_config_sha,
        )
        sequence_results.append(seq_dict)

    # Compute metrics from the actual execution results.
    metrics = compute_ablation_metrics(
        ablated_results=row_results,
        row_labels_by_id=row_labels_by_id,
        corpus_by_id={c.candidate_id: vars(c) for c in split_data.corpus},
        ablation_id=spec.ablation_id,
        description=spec.description,
    )

    # Compute a config SHA for the ablation (the spec fields are the
    # canonical ablation configuration; this lets downstream consumers
    # bind a result back to its exact configuration).
    config_payload = json.dumps(
        {
            "ablation_id": spec.ablation_id,
            "disabled_component": spec.disabled_component,
            "semantic_enabled": spec.semantic_enabled,
            "history_enabled": spec.history_enabled,
            "reconstruction_guard": spec.reconstruction_guard,
            "purge_enabled": spec.purge_enabled,
            "tau_sem": tau_sem,
            "reconstruction_threshold": reconstruction_threshold,
        },
        sort_keys=True,
    )
    config_sha = hashlib.sha256(config_payload.encode()).hexdigest()

    return AblationRunResult(
        ablation_id=spec.ablation_id,
        spec=spec,
        row_results=row_results,
        sequence_results=sequence_results,
        row_count=len(row_results),
        sequence_count=len(sequence_results),
        runner_config_sha=config_sha,
        metrics={
            "n_eligible": metrics.n_eligible,
            "n_leaking": metrics.n_leaking,
            "n_leaking_blocked": metrics.n_leaking_blocked,
            "n_non_leaking": metrics.n_non_leaking,
            "n_fp": metrics.n_fp,
            "n_useful_eligible": metrics.n_useful_eligible,
            "n_useful_preserved": metrics.n_useful_preserved,
            "fbr": metrics.fbr,
            "utility_retention": metrics.utility_retention,
        },
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute E5 component ablations A0-A4 through the canonical "
            "FirewallRunner (R1.2 §16, §23)."
        )
    )
    parser.add_argument(
        "--split", required=True, choices=("development", "validation"),
        help="Frozen split to evaluate on. Test split is rejected (R1.2 §18).",
    )
    parser.add_argument(
        "--tau-sem", type=float, default=0.80,
        help="Frozen semantic threshold.",
    )
    parser.add_argument(
        "--reconstruction-threshold", type=float, default=0.60,
        help="Frozen reconstruction threshold.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
        help="Directory to write ablation artifacts.",
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
            "ERROR: test split is not a valid split for ablation execution "
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

    # R1.2 §16: same corpus, features, target records, episode metadata,
    # tau_sem, reconstruction threshold, history window for every
    # ablation. We pass identical inputs to each.
    condition_manifest_sha = "ablation-runner-diagnostic"
    detector_config_sha = "ablation-runner-diagnostic"

    # We need features; use a synthetic placeholder that the runner
    # can interpret deterministically without re-running the embedding
    # backend. The diagnostic mode is meant to validate the runner
    # wiring (R1.2 §29).
    if not features_by_id and args.run_mode == "diagnostic":
        # Synthesize a single fake candidate per row label to exercise
        # the runner without touching the embedding backend.
        for label in split_data.row_labels:
            features_by_id[label.candidate_id] = {
                "exact_match": False,
                "alias_match": False,
                "semantic_similarity": 0.50,
            }

    if not features_by_id:
        print(
            f"ERROR: no features available for split {args.split!r}. "
            f"In diagnostic mode this runner can synthesise minimal "
            f"features, but no features were found in the loaded split.",
            file=sys.stderr,
        )
        return 3

    # Execute all five ablations.
    results: list[AblationRunResult] = []
    for spec in get_ablation_specs():
        print(
            f"Running {spec.ablation_id} ({spec.description})...",
            file=sys.stderr,
        )
        run_result = run_one_ablation(
            spec=spec,
            split_data=split_data,
            features_by_id=features_by_id,
            row_labels_by_id=row_labels_by_id,
            tau_sem=args.tau_sem,
            reconstruction_threshold=args.reconstruction_threshold,
            split_name=args.split,
            condition_manifest_sha=condition_manifest_sha,
            detector_config_sha=detector_config_sha,
        )
        results.append(run_result)

        # Persist per-ablation row + sequence evidence
        rows_path = output_dir / f"ablation_{spec.ablation_id}_rows.jsonl"
        seqs_path = output_dir / f"ablation_{spec.ablation_id}_sequences.jsonl"
        with open(rows_path, "w") as f:
            for r in run_result.row_results:
                f.write(json.dumps(r, default=str) + "\n")
        with open(seqs_path, "w") as f:
            for s in run_result.sequence_results:
                f.write(json.dumps(s, default=str) + "\n")

    # Build the ablation manifest
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_mode": args.run_mode,
        "split": args.split,
        "tau_sem": args.tau_sem,
        "reconstruction_threshold": args.reconstruction_threshold,
        "code_commit": _git_commit(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ablations": [
            {
                "ablation_id": r.ablation_id,
                "disabled_component": ABLATION_DISABLED_COMPONENT.get(
                    r.ablation_id
                ),
                "description": r.spec.description,
                "runner_config_sha": r.runner_config_sha,
                "row_count": r.row_count,
                "sequence_count": r.sequence_count,
                "rows_path": f"ablation_{r.ablation_id}_rows.jsonl",
                "sequences_path": (
                    f"ablation_{r.ablation_id}_sequences.jsonl"
                ),
            }
            for r in results
        ],
        "summary": {
            r.ablation_id: r.metrics for r in results
        },
    }

    # Condition manifest SHA and embedding manifest SHA, when available.
    cond_manifest_path = _CONFIG_DIR / "condition_manifest.json"
    if cond_manifest_path.exists():
        manifest["condition_manifest_sha"] = _file_sha256(cond_manifest_path)
    embed_manifest_path = _CONFIG_DIR / "embedding_manifest.json"
    if embed_manifest_path.exists():
        manifest["embedding_manifest_sha"] = _file_sha256(embed_manifest_path)

    # Persist manifest + summary
    manifest_path = output_dir / "ablation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "schema_version": "1.0",
                "run_mode": args.run_mode,
                "split": args.split,
                "summary": manifest["summary"],
            },
            f,
            indent=2,
            default=str,
        )

    print(
        f"Wrote ablation manifest to {manifest_path}",
        file=sys.stderr,
    )
    print(
        f"Wrote ablation summary to {summary_path}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

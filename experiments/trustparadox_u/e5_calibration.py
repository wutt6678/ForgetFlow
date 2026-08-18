"""E5-003: Development calibration — threshold sweep and selection.

This module implements Iteration 4 of the E5 plan.  It:

1. Loads development-split detector features (from E5-002).
2. Loads development-split frozen annotation labels (from ``e5_loaders``).
3. Sweeps τ_sem over a small candidate grid.
4. At each threshold, joins features to labels and computes row-level
   and sequence-level metrics via ``e5_metrics``.
5. Selects the best threshold using the selection rule (plan §11.2).
6. Writes three output artifacts under ``results/empirical_v2/e5/calibration/``:
   - ``threshold_sweep.jsonl`` — one JSON record per candidate threshold.
   - ``selected_config.json`` — the chosen configuration.
   - ``calibration_report.json`` — full report with all sweep results.

Exit criteria (plan §109):
  - development threshold sweep complete
  - selected configuration deterministic
  - calibration report generated
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .e5_loaders import SplitData, load_split
from .e5_metrics import (
    TAU_SEM_GRID,
    RowMetrics,
    SequenceMetrics,
    ThresholdSelection,
    compute_row_metrics,
    compute_sequence_metrics,
    select_threshold,
)
from .embedding_backend import E5_EMBEDDING_MODEL, E5_EMBEDDING_NORMALIZATION
from .semantic_detector import _DETECTOR_VERSION, load_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_CALIBRATION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "empirical_v2"
    / "e5"
    / "calibration"
)

_THRESHOLD_SWEEP_PATH = _CALIBRATION_DIR / "threshold_sweep.jsonl"
_SELECTED_CONFIG_PATH = _CALIBRATION_DIR / "selected_config.json"
_CALIBRATION_REPORT_PATH = _CALIBRATION_DIR / "calibration_report.json"


# ---------------------------------------------------------------------------
# Sweep result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepResult:
    """Combined row + sequence metrics for one threshold candidate."""

    tau_sem: float
    row_metrics: RowMetrics
    sequence_metrics: SequenceMetrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "tau_sem": self.tau_sem,
            "leakage_precision": self.row_metrics.leakage_precision,
            "leakage_recall": self.row_metrics.leakage_recall,
            "leakage_f1": self.row_metrics.leakage_f1,
            "false_blocking_rate": self.row_metrics.false_blocking_rate,
            "utility_retention": self.row_metrics.utility_retention,
            "sequence_reconstruction_recall": (
                self.sequence_metrics.sequence_reconstruction_recall
            ),
            "sequence_leakage_rate": (
                self.sequence_metrics.sequence_leakage_rate
            ),
            "n_eligible_rows": self.row_metrics.n_eligible,
            "n_unresolved_rows_excluded": (
                self.row_metrics.n_unresolved_excluded
            ),
            "tp": self.row_metrics.counts.tp,
            "fp": self.row_metrics.counts.fp,
            "fn": self.row_metrics.counts.fn,
            "tn": self.row_metrics.counts.tn,
            "n_reconstructing_sequences": (
                self.sequence_metrics.n_reconstructing_sequences
            ),
            "n_reconstructing_caught": (
                self.sequence_metrics.n_reconstructing_caught
            ),
            "n_eligible_sequences": (
                self.sequence_metrics.n_eligible_sequences
            ),
        }


# ---------------------------------------------------------------------------
# Core calibration logic
# ---------------------------------------------------------------------------


def _build_features_by_id(
    features: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index feature records by candidate_id."""
    return {f["candidate_id"]: f for f in features}


def run_threshold_sweep(
    split_data: SplitData,
    features: list[dict[str, Any]],
    *,
    tau_grid: list[float] | None = None,
) -> list[SweepResult]:
    """Run the threshold sweep over the development split.

    Args:
        split_data: Frozen SplitData for the development split.
        features: Detector feature dicts for the development split.
        tau_grid: Candidate τ_sem values.  Defaults to ``TAU_SEM_GRID``.

    Returns:
        List of SweepResult, one per threshold candidate.
    """
    if tau_grid is None:
        tau_grid = list(TAU_SEM_GRID)

    labels_by_id = split_data.row_labels_by_id
    features_by_id = _build_features_by_id(features)
    sequence_labels = list(split_data.sequence_labels)

    results: list[SweepResult] = []

    for tau in tau_grid:
        row_m = compute_row_metrics(features, labels_by_id, tau)
        seq_m = compute_sequence_metrics(sequence_labels, features_by_id, tau)
        results.append(SweepResult(tau_sem=tau, row_metrics=row_m, sequence_metrics=seq_m))
        logger.info(
            "τ=%.2f  P=%.3f  R=%.3f  F1=%.3f  FBR=%.3f  UR=%.3f  SR=%.3f",
            tau,
            row_m.leakage_precision,
            row_m.leakage_recall,
            row_m.leakage_f1,
            row_m.false_blocking_rate,
            row_m.utility_retention,
            seq_m.sequence_reconstruction_recall,
        )

    return results


def run_calibration(
    *,
    split: str = "development",
    tau_grid: list[float] | None = None,
    code_commit: str = "unknown",
    output_dir: Path = _CALIBRATION_DIR,
) -> dict[str, Any]:
    """Run the full development calibration pipeline.

    This is the main entry point for E5-003.

    Args:
        split: Must be ``"development"``.
        tau_grid: Optional custom grid (defaults to ``TAU_SEM_GRID``).
        code_commit: Git commit hash for provenance.
        output_dir: Where to write output artifacts.

    Returns:
        The calibration report dict.

    Raises:
        ValueError: If *split* is not ``"development"``.
    """
    if split != "development":
        raise ValueError(
            f"Calibration must use the development split, got {split!r}"
        )

    # Load frozen data
    split_data = load_split(split)
    features = load_features(split)

    logger.info(
        "Loaded %d features, %d row labels (%d unresolved), %d sequence labels",
        len(features),
        split_data.n_rows,
        split_data.n_unresolved_rows,
        split_data.n_sequences,
    )

    # Sweep
    sweep_results = run_threshold_sweep(split_data, features, tau_grid=tau_grid)

    # Select best threshold
    row_metrics_list = [sr.row_metrics for sr in sweep_results]
    selection = select_threshold(row_metrics_list)

    logger.info(
        "Selected τ=%.2f  (recall=%.3f, FBR=%.3f, UR=%.3f)",
        selection.selected_tau,
        selection.leakage_recall,
        selection.false_blocking_rate,
        selection.utility_retention,
    )

    # Build the best sweep result for the selected threshold
    best_sweep = next(
        sr for sr in sweep_results if sr.tau_sem == selection.selected_tau
    )

    # Write artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_threshold_sweep(sweep_results, output_dir / "threshold_sweep.jsonl")
    _write_selected_config(
        selection=selection,
        best_sweep=best_sweep,
        code_commit=code_commit,
        output_path=output_dir / "selected_config.json",
    )
    report = _write_calibration_report(
        sweep_results=sweep_results,
        selection=selection,
        split_data=split_data,
        n_features=len(features),
        code_commit=code_commit,
        output_path=output_dir / "calibration_report.json",
    )

    return report


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_threshold_sweep(
    results: list[SweepResult],
    path: Path,
) -> Path:
    """Write threshold_sweep.jsonl — one record per candidate threshold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sr in results:
            f.write(json.dumps(sr.to_dict()) + "\n")
    logger.info("Wrote threshold sweep: %s (%d records)", path, len(results))
    return path


def _write_selected_config(
    *,
    selection: ThresholdSelection,
    best_sweep: SweepResult,
    code_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    """Write selected_config.json (plan §13.1)."""
    config: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "semantic_threshold": selection.selected_tau,
        "embedding_model": E5_EMBEDDING_MODEL,
        "normalization": E5_EMBEDDING_NORMALIZATION,
        "detector_version": _DETECTOR_VERSION,
        "policy_configuration": {
            "detection_rule": (
                "exact_match OR alias_match "
                "OR (semantic_similarity >= semantic_threshold)"
            ),
            "trust_invariant": True,
        },
        "selection_rule": selection.selection_rule,
        "development_metric_summary": {
            "leakage_precision": best_sweep.row_metrics.leakage_precision,
            "leakage_recall": best_sweep.row_metrics.leakage_recall,
            "leakage_f1": best_sweep.row_metrics.leakage_f1,
            "false_blocking_rate": best_sweep.row_metrics.false_blocking_rate,
            "utility_retention": best_sweep.row_metrics.utility_retention,
            "sequence_reconstruction_recall": (
                best_sweep.sequence_metrics.sequence_reconstruction_recall
            ),
            "sequence_leakage_rate": (
                best_sweep.sequence_metrics.sequence_leakage_rate
            ),
        },
        "code_commit": code_commit,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    logger.info("Wrote selected config: %s", output_path)
    return config


def _write_calibration_report(
    *,
    sweep_results: list[SweepResult],
    selection: ThresholdSelection,
    split_data: SplitData,
    n_features: int,
    code_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    """Write calibration_report.json — full report."""
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split_data.split,
        "detector_version": _DETECTOR_VERSION,
        "embedding_model": E5_EMBEDDING_MODEL,
        "code_commit": code_commit,
        "data_summary": {
            "n_features": n_features,
            "n_row_labels": split_data.n_rows,
            "n_unresolved_rows": split_data.n_unresolved_rows,
            "n_sequence_labels": split_data.n_sequences,
            "n_unresolved_sequences": split_data.n_unresolved_sequences,
        },
        "tau_grid": [sr.tau_sem for sr in sweep_results],
        "sweep_results": [sr.to_dict() for sr in sweep_results],
        "selection": {
            "selected_tau": selection.selected_tau,
            "leakage_recall": selection.leakage_recall,
            "false_blocking_rate": selection.false_blocking_rate,
            "utility_retention": selection.utility_retention,
            "selection_rule": selection.selection_rule,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info("Wrote calibration report: %s", output_path)
    return report

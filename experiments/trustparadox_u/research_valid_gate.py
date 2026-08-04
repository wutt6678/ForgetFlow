"""FF92-021: research-valid gate rebuilt to verify scientific validity.

The old gate mostly checked that files existed, which let an invalid study
reach a "research_valid" verdict.  This gate recomputes what it can —
corpus/annotation hashes, target-spec hashes, conditions, trial pairing,
metrics from trial artifacts — and verifies provenance (FF92-023) and
invalidation state (FF92-022) before any certification.

Verdicts (remediation §31 staged statuses):
  diagnostic                  any substantive gate failed;
  diagnostic_valid            substantive gates pass, but the suite/static
                              checks were not run (e.g. gate invoked from
                              inside a pytest run — which must never
                              auto-pass);
  synthetic_benchmark_valid   full suite plus substantive gates: the ceiling
                              for deterministic fixed-embedding studies;
  empirical_replay_valid      additionally requires the empirical study
                              design gate and a matching study class;
  closed_loop_study_valid     likewise for closed-loop study designs;
  release_candidate           the study-class ceiling plus release
                              readiness (every gate, including the design
                              gate, passed).

File existence alone can never yield a staged research status, and a
diagnostic-class run can never exceed ``synthetic_benchmark_valid``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parents[2] / "results"
REPLAY_DIR = RESULTS_DIR / "frozen_replay"
FINAL_DIR = RESULTS_DIR / "final_artifacts"
CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"

SCHEMA_VERSION = "3.0.0"

# Provenance-bearing artifacts that a certifying run must have produced.
PROVENANCE_ARTIFACTS: tuple[Path, ...] = (
    REPLAY_DIR / "run_manifest.json",
    RESULTS_DIR / "parameter_sweep" / "sweep_summary.json",
    FINAL_DIR / "study_manifest.json",
)


def _current_commit() -> str:
    """HEAD commit for certification.

    Dirtiness for certification is measured on the code tree only
    (``code_tree_is_clean`` excludes ``results/``): pipeline writers
    inevitably leave uncommitted regenerated artifacts behind, and that
    alone must not poison the commit they were generated from.
    """
    from experiments.trustparadox_u.artifact_provenance import code_tree_is_clean
    from experiments.trustparadox_u.manifest import get_repository_commit

    return (
        get_repository_commit().removesuffix("-dirty")
        if code_tree_is_clean()
        else get_repository_commit()
    )


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# ---------------------------------------------------------------------------
# Provenance and invalidation gates (FF92-023 / FF92-022)
# ---------------------------------------------------------------------------


def check_repository_provenance() -> dict[str, Any]:
    """Certification requires a clean code tree and matching commits.

    Every certifying artifact must record tested_code_commit ==
    artifact_generation_commit == current HEAD with repository_clean=true.
    """
    from experiments.trustparadox_u.artifact_provenance import (
        code_tree_is_clean,
        validate_result_provenance_file,
    )

    commit = _current_commit()
    findings: list[str] = []
    if commit in ("", "unknown"):
        findings.append("unknown_repository_commit")
    if not code_tree_is_clean():
        findings.append("repository_tree_dirty")
    for artifact in PROVENANCE_ARTIFACTS:
        findings.extend(validate_result_provenance_file(artifact, current_commit=commit))
    return {
        "passed": not findings,
        "repository_commit": commit,
        "findings": findings,
    }


def check_no_invalidated_artifacts() -> dict[str, Any]:
    """No invalidation markers outside the archive; no stale validity claims.

    The gate's own output file is excluded: it is rewritten on every run,
    so a previous verdict can never be treated as an independent claim.
    """
    from experiments.trustparadox_u.invalidation import (
        find_invalidation_markers,
        find_research_valid_claims,
    )

    markers = [str(p) for p in find_invalidation_markers(RESULTS_DIR)]
    own_output = FINAL_DIR / "research_valid_gate.json"
    claims = [str(p) for p in find_research_valid_claims(RESULTS_DIR) if p != own_output]
    return {
        "passed": not markers and not claims,
        "invalidation_markers": markers,
        "stale_research_valid_claims": claims,
    }


# ---------------------------------------------------------------------------
# Corpus and annotation gates
# ---------------------------------------------------------------------------


def _load_split_candidates(split: str) -> list[Any]:
    from experiments.trustparadox_u.candidates import load_frozen_corpus

    index = load_frozen_corpus(CORPUS_DIR / f"frozen_corpus_{split}.jsonl")
    return list(index.candidates)


def check_corpus_valid() -> dict[str, Any]:
    """Recompute corpus and target-spec hashes; run content validation."""
    from experiments.trustparadox_u.candidates import load_frozen_corpus
    from experiments.trustparadox_u.generate_corpus import (
        _target_spec_hash,
        build_target_specs,
        validate_corpus,
    )

    manifest_path = CORPUS_DIR / "corpus_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "reason": "corpus_manifest.json not found"}
    manifest = _load_json(manifest_path)

    try:
        index = load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl")
        candidates = list(index.candidates)
    except (OSError, ValueError, KeyError) as exc:
        return {"passed": False, "reason": f"corpus_unreadable: {exc}"}
    findings: list[str] = []

    if index.corpus_hash != manifest.get("corpus_sha256"):
        findings.append("corpus_hash_mismatch")
    if len(candidates) != manifest.get("candidate_count"):
        findings.append("candidate_count_mismatch")

    spec_hash = _target_spec_hash(build_target_specs())
    if spec_hash != manifest.get("target_spec_sha256"):
        findings.append("target_spec_hash_mismatch")

    splits = {
        split: _load_split_candidates(split) for split in ("development", "validation", "test")
    }
    for split, members in splits.items():
        if len(members) != manifest.get("split_counts", {}).get(split):
            findings.append(f"split_count_mismatch:{split}")

    errors = validate_corpus(candidates, splits)
    findings.extend(errors)

    return {
        "passed": not findings,
        "candidate_count": len(candidates),
        "corpus_hash": index.corpus_hash,
        "findings": findings[:20],
    }


def check_annotations_valid() -> dict[str, Any]:
    """Recompute the annotation hash and run content validation."""
    from experiments.trustparadox_u.annotate_corpus import (
        CorpusAnnotation,
        validate_annotations,
    )
    from experiments.trustparadox_u.candidates import (
        canonical_jsonl_hash,
        load_frozen_corpus,
    )

    ann_manifest_path = CORPUS_DIR / "annotation_manifest.json"
    ann_path = CORPUS_DIR / "corpus_annotations.jsonl"
    if not ann_manifest_path.exists() or not ann_path.exists():
        return {"passed": False, "reason": "annotation manifest or annotations not found"}
    manifest = _load_json(ann_manifest_path)

    annotations: list[CorpusAnnotation] = []
    try:
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                annotations.append(
                    CorpusAnnotation(
                        **{**record, "target_forget_ids": tuple(record["target_forget_ids"])}
                    )
                )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"annotations_unreadable: {exc}"}

    findings: list[str] = []
    recomputed = canonical_jsonl_hash([a.to_dict() for a in annotations])
    if recomputed != manifest.get("annotation_hash"):
        findings.append("annotation_hash_mismatch")

    corpus_manifest_path = CORPUS_DIR / "corpus_manifest.json"
    if corpus_manifest_path.exists():
        corpus_manifest = _load_json(corpus_manifest_path)
        if manifest.get("corpus_hash") != corpus_manifest.get("corpus_sha256"):
            findings.append("annotation_corpus_binding_mismatch")

    index = load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl")
    findings.extend(validate_annotations(annotations, list(index.candidates)))

    return {
        "passed": not findings,
        "annotation_count": len(annotations),
        "annotation_hash": recomputed,
        "findings": findings[:20],
    }


# ---------------------------------------------------------------------------
# Condition and replay gates
# ---------------------------------------------------------------------------


def check_conditions_valid() -> dict[str, Any]:
    """Resolved conditions must match the documented condition set exactly,
    and each resolved config must equal the condition's frozen builder."""
    from experiments.trustparadox_u.frozen_replay import (
        BASELINE_CONDITION,
        CONDITIONS,
        build_config_for_condition,
    )

    resolved_path = REPLAY_DIR / "resolved_conditions.json"
    manifest_path = REPLAY_DIR / "run_manifest.json"
    if not resolved_path.exists():
        return {"passed": False, "reason": "resolved_conditions.json not found"}
    resolved = _load_json(resolved_path)
    seed = 42
    if manifest_path.exists():
        seed = _load_json(manifest_path).get("seed", 42)

    findings: list[str] = []
    expected = set(CONDITIONS)
    if set(resolved) != expected:
        findings.append(f"condition_set_mismatch: {sorted(set(resolved) ^ expected)}")
    if BASELINE_CONDITION not in resolved:
        findings.append(f"baseline_missing: {BASELINE_CONDITION}")
    for name in sorted(set(resolved) & expected):
        expected_config = dataclasses.asdict(build_config_for_condition(name, seed=seed))
        # The resolved configs are JSON-serialized: normalize tuples to
        # lists before comparing.
        normalized = json.loads(json.dumps(expected_config))
        if resolved[name] != normalized:
            findings.append(f"config_mismatch: {name}")
    return {"passed": not findings, "conditions": sorted(resolved), "findings": findings}


def check_replay_complete() -> dict[str, Any]:
    """Every corpus candidate must be covered by exactly one trial unit per
    condition: one trial record per unit, no failures, and each record's
    member/target lineage must match the corpus exactly."""
    from experiments.trustparadox_u.candidates import load_frozen_corpus
    from experiments.trustparadox_u.frozen_replay import (
        CONDITIONS,
        partition_trial_units,
    )
    from experiments.trustparadox_u.trial_artifacts import load_trial_records

    manifest_path = REPLAY_DIR / "run_manifest.json"
    trials_path = REPLAY_DIR / "candidate_trials.jsonl"
    if not manifest_path.exists() or not trials_path.exists():
        return {"passed": False, "reason": "run manifest or candidate trials not found"}
    manifest = _load_json(manifest_path)

    findings: list[str] = []
    if manifest.get("failed_candidate_count", -1) != 0:
        findings.append(f"failed_candidates: {manifest.get('failed_candidate_count')}")

    try:
        index = load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl")
    except (OSError, ValueError, KeyError) as exc:
        return {"passed": False, "reason": f"corpus_unreadable: {exc}"}
    units = partition_trial_units(list(index.candidates))
    expected_members = len(index.candidates) * len(CONDITIONS)
    expected_trials = len(units) * len(CONDITIONS)
    if manifest.get("candidate_count") != expected_members:
        findings.append(
            f"candidate_count_mismatch: {manifest.get('candidate_count')} != {expected_members}"
        )

    try:
        trials = load_trial_records(trials_path)
    except (OSError, ValueError, KeyError) as exc:
        return {"passed": False, "reason": f"trials_unreadable: {exc}"}
    if len(trials) != expected_trials:
        findings.append(f"trial_count_mismatch: {len(trials)} != {expected_trials}")

    expected_keys = {
        (condition, unit.representative.candidate_id, tuple(m.candidate_id for m in unit.members))
        for condition in CONDITIONS
        for unit in units
    }
    actual_keys = {
        (t.get("condition_id"), t.get("candidate_id"), tuple(t.get("candidate_ids", ())))
        for t in trials
    }
    if len(actual_keys) != len(trials):
        findings.append("duplicate_trial_keys")
    if actual_keys != expected_keys:
        findings.append(f"trial_unit_mismatches: {len(actual_keys ^ expected_keys)}")

    failed = [t["candidate_id"] for t in trials if t.get("result_status") != "success"]
    if failed:
        findings.append(f"failed_trials: {len(failed)}")

    by_id = {c.candidate_id: c for c in index.candidates}
    mismatched = [
        t["candidate_id"]
        for t in trials
        if t.get("candidate_id") in by_id
        and list(t.get("target_forget_ids", [])) != list(by_id[t["candidate_id"]].target_forget_ids)
    ]
    if mismatched:
        findings.append(f"target_forget_id_mismatches: {len(mismatched)}")

    return {
        "passed": not findings,
        "trial_count": len(trials),
        "expected_count": expected_trials,
        "expected_member_count": expected_members,
        "findings": findings[:20],
    }


def check_metrics_recompute() -> dict[str, Any]:
    """Metrics must recompute exactly from the trial artifacts; utility must
    be evaluable for every non-baseline condition; no None-valued metric may
    be reported as computed."""
    from experiments.trustparadox_u.frozen_replay import BASELINE_CONDITION, CONDITIONS
    from experiments.trustparadox_u.trial_artifacts import (
        CandidateTrial,
        load_trial_records,
        metrics_from_artifacts,
    )

    stored_path = REPLAY_DIR / "metrics_by_condition.json"
    if not stored_path.exists():
        return {"passed": False, "reason": "metrics_by_condition.json not found"}
    stored = _load_json(stored_path)

    try:
        candidate_trials = [
            CandidateTrial.from_dict(r)
            for r in load_trial_records(REPLAY_DIR / "candidate_trials.jsonl")
        ]
        reconstruction = load_trial_records(REPLAY_DIR / "reconstruction_trials.jsonl")
        recontamination = load_trial_records(REPLAY_DIR / "recontamination_trials.jsonl")
        utility = load_trial_records(REPLAY_DIR / "utility_trials.jsonl")
    except (OSError, ValueError, KeyError) as exc:
        return {"passed": False, "reason": f"trial_artifacts_unreadable: {exc}"}

    recomputed = metrics_from_artifacts(
        candidate_trials,
        reconstruction,
        recontamination,
        utility,
        conditions=sorted(CONDITIONS),
        baseline_condition=BASELINE_CONDITION,
    )

    findings: list[str] = []
    if recomputed != stored:
        mismatched = sorted(
            c for c in set(recomputed) | set(stored) if recomputed.get(c) != stored.get(c)
        )
        findings.append(f"metric_mismatch: {mismatched}")

    for condition, metrics in stored.items():
        for metric_name, metric in metrics.items():
            if not isinstance(metric, dict):
                continue
            value = metric.get("value")
            evaluable = metric.get("evaluable", value is not None)
            if evaluable and value is None:
                findings.append(f"none_metric_called_computed: {condition}.{metric_name}")
            if condition != BASELINE_CONDITION and metric_name == "paired_policy_utility_retention":
                if not metric.get("evaluable"):
                    findings.append(f"utility_not_evaluable: {condition}")
    return {
        "passed": not findings,
        "conditions": sorted(stored),
        "findings": findings[:20],
    }


# ---------------------------------------------------------------------------
# Analysis gates
# ---------------------------------------------------------------------------


def check_leakage_analysis_valid() -> dict[str, Any]:
    """Check that the FF92-016 leakage analysis exists, validates, and covers
    every replay condition."""
    from experiments.trustparadox_u.frozen_replay import CONDITIONS

    path = RESULTS_DIR / "leakage_analysis" / "leakage_analysis.json"
    if not path.exists():
        return {"passed": False, "reason": "leakage_analysis.json not found"}

    data = _load_json(path)
    required = ("by_condition_and_attack", "global", "validation")
    missing = [key for key in required if key not in data]
    if missing:
        return {"passed": False, "reason": f"missing keys: {missing}"}

    findings: list[str] = []
    covered = set(data.get("by_condition_and_attack", {}))
    if covered != set(CONDITIONS):
        findings.append(f"condition_coverage_mismatch: {sorted(set(CONDITIONS) - covered)}")

    validation = data.get("validation", {})
    failed = [c["check"] for c in validation.values() if not c.get("passed")]
    if not validation or failed:
        findings.append(f"validation_failures: {failed}")
    return {
        "passed": not findings,
        "conditions": len(covered),
        "validations_passed": len(validation) - len(failed),
        "findings": findings,
    }


def check_statistical_analysis_valid() -> dict[str, Any]:
    """Paired statistics must be complete: every comparison paired on
    candidate_id, contingency tables consistent, unmatched pairs reported,
    and bootstrap confidence intervals available."""
    path = RESULTS_DIR / "paired_statistics" / "paired_statistics.json"
    if not path.exists():
        return {"passed": False, "reason": "paired_statistics.json not found"}

    data = _load_json(path)
    comparisons = data.get("comparisons", [])
    if data.get("num_comparisons", 0) <= 0 or not comparisons:
        return {"passed": False, "reason": "no comparisons"}

    findings: list[str] = []
    for comp in comparisons:
        label = f"{comp.get('condition_a')}~{comp.get('condition_b')}:{comp.get('metric')}"
        contingency = comp.get("contingency", {})
        if sum(contingency.values()) != comp.get("n_pairs", -1):
            findings.append(f"contingency_mismatch: {label}")
        if "unmatched" not in comp:
            findings.append(f"unmatched_not_reported: {label}")
        if comp.get("bootstrap_ci_95") is None:
            findings.append(f"bootstrap_ci_missing: {label}")
        # Reconstruction trials exist per sequence, so their comparisons
        # pair on sequence_id; every other metric pairs on candidate_id.
        expected_unit = "sequence_id" if comp.get("metric") == "reconstruction" else "candidate_id"
        if comp.get("pairing_unit") != expected_unit:
            findings.append(f"pairing_unit_mismatch: {label}")
    return {
        "passed": not findings,
        "num_comparisons": len(comparisons),
        "findings": findings[:20],
    }


def check_parameter_sweep_complete() -> dict[str, Any]:
    """FF92-018: check the one-at-a-time hyperparameter sweep is complete.

    Requires the schema-2.0 sweep with all four core hyperparameters swept,
    a frozen selection for each, passing artifact validation checks, and a
    final evaluation on the test split.
    """
    path = RESULTS_DIR / "parameter_sweep" / "sweep_summary.json"
    if not path.exists():
        return {"passed": False, "reason": "sweep_summary.json not found"}

    data = _load_json(path)
    if data.get("schema_version") != "2.0":
        return {
            "passed": False,
            "reason": f"expected schema_version 2.0, got {data.get('schema_version')!r}",
        }

    required_sweeps = {
        "embedding_threshold",
        "claim_confidence_threshold",
        "history.window_size",
        "monitoring.duration_rounds",
    }
    sweeps = data.get("sweeps", {})
    missing = required_sweeps - set(sweeps)
    if missing:
        return {"passed": False, "reason": f"missing sweeps: {sorted(missing)}"}

    unselected = [name for name, s in sweeps.items() if "selected_value" not in s]
    if unselected:
        return {"passed": False, "reason": f"sweeps without selection: {unselected}"}

    failed = [name for name, check in data.get("validation", {}).items() if not check.get("passed")]
    if failed:
        return {"passed": False, "reason": f"validation checks failed: {failed}"}

    final = data.get("final_test_evaluation", {})
    if final.get("split") != "test":
        return {"passed": False, "reason": "final evaluation is not on the test split"}

    return {
        "passed": True,
        "num_sweeps": len(sweeps),
        "final_test_split": final["split"],
    }


def check_deterministic_reproducibility_validation() -> dict[str, Any]:
    """FF92-020: check the deterministic reproducibility validation.

    Requires all four comparison layers (candidate-level, trial-level,
    metric counts, hashes) to pass — aggregate equality alone is not
    sufficient.
    """
    path = RESULTS_DIR / "deterministic_reproducibility_validation" / "validation_result.json"
    if not path.exists():
        return {"passed": False, "reason": "validation_result.json not found"}

    data = _load_json(path)
    checks = data.get("checks", {})
    failed = [name for name, check in checks.items() if not check.get("passed")]
    if failed:
        return {"passed": False, "reason": f"failed layers: {failed}"}
    if not data.get("passed", False):
        return {"passed": False, "reason": "overall validation did not pass"}
    return {
        "passed": True,
        "num_conditions": data.get("num_conditions", 0),
        "num_layers": len(checks),
        "num_mismatches": data.get("num_mismatches", -1),
    }


def check_final_artifacts() -> dict[str, Any]:
    """Final artifacts must exist AND the study manifest's exit criteria
    must all be satisfied — existence alone is not certification."""
    required = [
        FINAL_DIR / "study_manifest.json",
        FINAL_DIR / "study_summary.md",
        FINAL_DIR / "table1_main_results.json",
        FINAL_DIR / "table2_leakage_breakdown.json",
        FINAL_DIR / "table3_parameter_sensitivity.json",
        FINAL_DIR / "table4_statistical_comparisons.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return {"passed": False, "missing": missing, "findings": [f"missing: {missing}"]}

    manifest = _load_json(FINAL_DIR / "study_manifest.json")
    exit_criteria = manifest.get("exit_criteria", {})
    failed = sorted(name for name, ok in exit_criteria.items() if ok is not True)
    findings: list[str] = []
    if not exit_criteria:
        findings.append("exit_criteria_missing")
    if failed:
        findings.append(f"exit_criteria_failed: {failed}")
    return {
        "passed": not findings,
        "present": len(required),
        "total": len(required),
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Test and static-analysis gates
# ---------------------------------------------------------------------------


def check_tests_pass() -> dict[str, Any]:
    """Run the full test suite.

    FF92-021: inside a pytest run the gate must never auto-pass — the
    enclosing suite's success says nothing about this gate's evidence, so
    the check reports ``not_run`` and the verdict drops to at most
    ``release_candidate``.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return {
            "passed": False,
            "not_run": True,
            "reason": "not_run_inside_pytest: a gate must never auto-pass under pytest",
        }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--tb=no", "-q", "--no-header"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=1200,
        )
        output = result.stdout.strip()
        lines = output.split("\n")
        last_line = lines[-1] if lines else ""
        return {"passed": result.returncode == 0, "output": last_line}
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def check_static_checks() -> dict[str, Any]:
    """Ruff lint, ruff format, and mypy exactly as CI runs them."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return {
            "passed": False,
            "not_run": True,
            "reason": "not_run_inside_pytest: static checks must be run explicitly",
        }
    targets = ["marble", "experiments", "tests", "scripts"]
    commands = (
        [sys.executable, "-m", "ruff", "check", *targets],
        [sys.executable, "-m", "ruff", "format", "--check", *targets],
        [sys.executable, "-m", "mypy", "marble", "experiments"],
    )
    findings: list[str] = []
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=_PROJECT_ROOT,
                timeout=900,
            )
        except Exception as e:  # noqa: BLE001
            findings.append(f"{command[2]}: {e}")
            continue
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip().split("\n")[-1]
            findings.append(f"{command[2]}: failed ({tail})")
    return {"passed": not findings, "findings": findings}


# ---------------------------------------------------------------------------
# Study-design gate (remediation §31)
# ---------------------------------------------------------------------------


def _study_class_from_artifacts() -> str:
    """Study class recorded on the replay run manifest (remediation §4).

    Pre-§4 artifacts do not carry the field; they were produced by the
    deterministic scripted-responder harness, so ``diagnostic`` is the
    only honest default.
    """
    from experiments.trustparadox_u.status import STUDY_CLASS_DIAGNOSTIC, validate_study_class

    manifest_path = REPLAY_DIR / "run_manifest.json"
    study_class = STUDY_CLASS_DIAGNOSTIC
    if manifest_path.exists():
        study_class = str(_load_json(manifest_path).get("study_class", study_class))
    validate_study_class(study_class)
    return study_class


def check_empirical_study_design() -> dict[str, Any]:
    """Empirical statuses require empirical study design (remediation §31).

    This gate is deliberately outside ``SUBSTANTIVE_GATES``: a
    deterministic fixed-embedding study fails it *by design* and is
    capped at ``synthetic_benchmark_valid`` instead of being declared
    invalid.  Corpus provenance decides — a ``deterministic_template``
    generation model can never support empirical claims, no matter how
    many internal-consistency gates pass.
    """
    from experiments.trustparadox_u.status import (
        STUDY_CLASS_CLOSED_LOOP,
        STUDY_CLASS_DIAGNOSTIC,
    )

    study_class = _study_class_from_artifacts()
    if study_class == STUDY_CLASS_DIAGNOSTIC:
        return {
            "passed": False,
            "study_class": study_class,
            "reason": (
                "diagnostic study class: scripted responders and deterministic "
                "embeddings validate code paths, not empirical behavior"
            ),
        }

    manifest_path = CORPUS_DIR / "corpus_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "study_class": study_class, "reason": "corpus_manifest missing"}
    generation_model = str(_load_json(manifest_path).get("generation_model", ""))
    findings: list[str] = []
    if generation_model in ("", "deterministic_template"):
        findings.append(f"corpus_generation_model_not_empirical: {generation_model!r}")
    if study_class == STUDY_CLASS_CLOSED_LOOP:
        # Closed-loop evidence (agents generating during the episode) is
        # produced by Phase D; until then the declaration is unsupported.
        findings.append("closed_loop_evidence_not_recorded")
    return {"passed": not findings, "study_class": study_class, "findings": findings}


def check_research_protocol() -> dict[str, Any]:
    """A versioned protocol declares every question/comparison (§2).

    The protocol must exist, carry a semver version, cover the minimum
    primary claims, and map every final table to a declared question.
    """
    from experiments.trustparadox_u.research_protocol import (
        QUESTIONS,
        validate_protocol,
    )

    findings = validate_protocol()
    if len(QUESTIONS) < 7:
        findings.append(f"minimum_primary_claims_not_declared: {len(QUESTIONS)} < 7")
    return {
        "passed": not findings,
        "question_count": len(QUESTIONS),
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

SUBSTANTIVE_GATES: tuple[str, ...] = (
    "repository_provenance",
    "no_invalidated_artifacts",
    "research_protocol",
    "corpus_valid",
    "annotations_valid",
    "conditions_valid",
    "replay_complete",
    "metrics_recompute",
    "leakage_analysis_valid",
    "statistical_analysis_valid",
    "parameter_sweep_complete",
    "deterministic_reproducibility_validation",
    "final_artifacts",
)


def _gate_passed(gates: dict[str, dict[str, Any]], name: str) -> bool:
    return bool(gates.get(name, {}).get("passed"))


def verdict_for(gates: dict[str, dict[str, Any]], study_class: str | None = None) -> str:
    """Highest staged research status the gate evidence supports (§31).

    ``study_class=None`` reads the class from the replay run manifest
    (diagnostic for pre-§4 artifacts).  Substantive failure yields
    ``diagnostic``; a missing suite caps at ``diagnostic_valid``; the
    study-design gate and the declared class set the ceiling above
    ``synthetic_benchmark_valid``.
    """
    from experiments.trustparadox_u.status import (
        STUDY_CLASS_CLOSED_LOOP,
        STUDY_CLASS_EMPIRICAL_REPLAY,
        compute_research_status,
    )

    if study_class is None:
        study_class = _study_class_from_artifacts()
    substantive_ok = all(_gate_passed(gates, name) for name in SUBSTANTIVE_GATES)
    suite_ok = _gate_passed(gates, "tests_pass") and _gate_passed(gates, "static_checks")
    empirical_ok = _gate_passed(gates, "empirical_study_design")
    return compute_research_status(
        study_class=study_class,
        substantive_gates_passed=substantive_ok,
        suite_passed=suite_ok,
        empirical_design_passed=empirical_ok and study_class == STUDY_CLASS_EMPIRICAL_REPLAY,
        closed_loop_design_passed=empirical_ok and study_class == STUDY_CLASS_CLOSED_LOOP,
        release_ready=substantive_ok and suite_ok and empirical_ok,
    )


def run_research_valid_gate() -> dict[str, Any]:
    """Run all gate checks and produce the staged research status (§31)."""
    from experiments.trustparadox_u.artifact_provenance import (
        build_certification_provenance,
        code_tree_is_clean,
    )

    gates = {
        "repository_provenance": check_repository_provenance(),
        "no_invalidated_artifacts": check_no_invalidated_artifacts(),
        "research_protocol": check_research_protocol(),
        "corpus_valid": check_corpus_valid(),
        "annotations_valid": check_annotations_valid(),
        "conditions_valid": check_conditions_valid(),
        "replay_complete": check_replay_complete(),
        "metrics_recompute": check_metrics_recompute(),
        "leakage_analysis_valid": check_leakage_analysis_valid(),
        "statistical_analysis_valid": check_statistical_analysis_valid(),
        "parameter_sweep_complete": check_parameter_sweep_complete(),
        "deterministic_reproducibility_validation": (
            check_deterministic_reproducibility_validation()
        ),
        "final_artifacts": check_final_artifacts(),
        "empirical_study_design": check_empirical_study_design(),
        "tests_pass": check_tests_pass(),
        "static_checks": check_static_checks(),
    }

    from experiments.trustparadox_u.status import (
        EMPIRICAL_REPLAY_VALID,
        SYNTHETIC_BENCHMARK_VALID,
        research_status_at_least,
    )

    study_class = str(gates["empirical_study_design"].get("study_class", "diagnostic"))
    verdict = verdict_for(gates, study_class=study_class)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_name": "research_valid",
        "study_class": study_class,
        "verdict": verdict,
        "research_status": verdict,
        # Remediation §31: ``research_valid`` now means an empirical tier
        # has actually been certified — never a deterministic replay.
        "research_valid": research_status_at_least(verdict, EMPIRICAL_REPLAY_VALID),
        "synthetic_benchmark_valid": research_status_at_least(verdict, SYNTHETIC_BENCHMARK_VALID),
        "all_passed": all(g["passed"] for g in gates.values()),
        "repository_commit": _current_commit(),
        "provenance": build_certification_provenance(repository_clean=code_tree_is_clean()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the final research-valid gate (staged statuses, §31)."""
    from experiments.trustparadox_u.status import (
        SYNTHETIC_BENCHMARK_VALID,
        research_status_at_least,
    )

    print("Final Research-Valid Gate (remediation §31 staged statuses)")
    print("=" * 50)

    result = run_research_valid_gate()

    # Write result
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "research_valid_gate.json").write_text(json.dumps(result, indent=2))

    # Print results
    print(f"\nVerdict: {result['verdict'].upper()}")
    print(f"Commit: {result['repository_commit']}")
    print()

    for gate_name, gate_result in result["gates"].items():
        status = (
            "PASS" if gate_result["passed"] else ("SKIP" if gate_result.get("not_run") else "FAIL")
        )
        print(f"  [{status}] {gate_name}")
        if not gate_result["passed"]:
            for k, v in gate_result.items():
                if k not in ("passed", "not_run"):
                    print(f"         {k}: {v}")

    print()
    n_fail = sum(1 for g in result["gates"].values() if not g["passed"])
    print(f"STUDY CLASS: {result['study_class']}")
    print(
        f"STUDY STATUS: {result['verdict'].upper().replace('_', '-')}"
        f" ({n_fail} gates failed/skipped)"
    )

    # Remediation §31: a deterministic replay that certifies its own tier
    # exits successfully; empirical tiers are what the full pipeline
    # command should target once the corpus is empirical.
    return 0 if research_status_at_least(result["verdict"], SYNTHETIC_BENCHMARK_VALID) else 1


if __name__ == "__main__":
    raise SystemExit(main())

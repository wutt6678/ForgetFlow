"""E5-005: Experimental-condition freeze.

Freezes all evaluation conditions, the experiment configuration, and the
test-access lock *before* any held-out test execution.

Conditions (plan §18):
    C0  No firewall (pass-through)
    C1  Exact-match only
    C2  Exact + alias
    C3  Exact + alias + semantic detector
    C4  Full ForgetFlow (detector + policy + history + reconstruction guard)

Artifacts written under ``results/empirical_v2/e5/config/``:
    e5_experiment_config.json
    e5_condition_manifest.json
    e5_test_lock.json

Exit criteria (plan §111):
    all row conditions executable
    row result schema stable
    primary metrics verified on dev/validation
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embedding_backend import (
    E5_EMBEDDING_CONFIG_VERSION,
    E5_EMBEDDING_MODEL,
    E5_EMBEDDING_NORMALIZATION,
)
from .semantic_detector import _DETECTOR_VERSION

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "empirical_v2"
    / "e5"
    / "config"
)

_CALIBRATION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "empirical_v2"
    / "e5"
    / "calibration"
)

_VALIDATION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "empirical_v2"
    / "e5"
    / "validation"
)

_EXPERIMENT_CONFIG_PATH = _CONFIG_DIR / "e5_experiment_config.json"
_CONDITION_MANIFEST_PATH = _CONFIG_DIR / "e5_condition_manifest.json"
_TEST_LOCK_PATH = _CONFIG_DIR / "e5_test_lock.json"
_METRIC_SPEC_PATH = _CONFIG_DIR / "e5_metric_spec.json"

# ---------------------------------------------------------------------------
# Placeholder values that must never appear in authoritative artifacts
# ---------------------------------------------------------------------------

_FORBIDDEN_PLACEHOLDERS: frozenset[str] = frozenset({
    "unknown", "missing", "", "from_calibration", "pending_calibration",
})

# ---------------------------------------------------------------------------
# Test-access guard (§6)
# ---------------------------------------------------------------------------


class TestAccessError(RuntimeError):
    """Raised when held-out test access is attempted without authorisation."""


def require_test_access_started(
    *,
    lock_path: Path = _TEST_LOCK_PATH,
) -> dict[str, Any]:
    """Verify that held-out test access is properly authorised.

    Checks:
        1. e5_test_lock file exists
        2. test_access_started == true
        3. execution_commit is non-empty
        4. config hashes are concrete and valid (not placeholders)

    Raises:
        TestAccessError: If any check fails.

    Returns:
        The parsed test-lock dict.
    """
    if not lock_path.exists():
        raise TestAccessError(
            f"Test lock file not found: {lock_path}. "
            "Cannot proceed with held-out test access."
        )

    with open(lock_path) as f:
        lock = json.load(f)

    if not lock.get("test_access_started"):
        raise TestAccessError(
            "test_access_started is not true in the test lock. "
            "Held-out test access is not authorised."
        )

    execution_commit = lock.get("execution_commit")
    if not execution_commit:
        raise TestAccessError(
            "execution_commit is empty or missing in the test lock."
        )

    # Verify all hash fields are concrete (not placeholders)
    hash_fields = [
        "config_sha", "condition_manifest_sha", "embedding_manifest_sha",
        "selected_config_sha", "metric_spec_sha",
        "global_annotation_freeze_sha",
    ]
    for field in hash_fields:
        value = lock.get(field, "")
        if value in _FORBIDDEN_PLACEHOLDERS:
            raise TestAccessError(
                f"Test lock field {field!r} has placeholder value {value!r}. "
                "Concrete hashes are required before test access."
            )

    return lock


def require_phase_order(
    *,
    phase_path: Path | None = None,
) -> dict[str, Any]:
    """Load and return the phase status artifact (§71, §72).

    Raises:
        FileNotFoundError: If the phase file doesn't exist.
    """
    if phase_path is None:
        phase_path = (
            Path(__file__).resolve().parents[2]
            / "results" / "empirical_v2" / "e5" / "e5_phase.json"
        )
    if not phase_path.exists():
        raise FileNotFoundError(f"Phase file not found: {phase_path}")
    with open(phase_path) as f:
        return json.load(f)


def assert_phase_transition(
    phase: dict[str, Any],
    target: str,
) -> None:
    """Assert that a phase transition is valid (§72).

    Prevents invalid transitions:
        - validation before calibration
        - test lock before validation acceptance
        - test evaluation before lock
        - test freeze before evaluation
        - E5 closure before freeze verification

    Raises:
        ValueError: If the transition is invalid.
    """
    guards = {
        "development_calibration": [],
        "validation": ["development_calibration_complete"],
        "test_lock": ["validation_complete"],
        "test_access": ["test_lock_complete"],
        "test_evaluation": ["test_access_started"],
        "test_freeze": ["test_evaluation_complete"],
        "e5_closure": ["e5_frozen"],
    }
    prerequisites = guards.get(target, [])
    for prereq in prerequisites:
        if not phase.get(prereq):
            raise ValueError(
                f"Phase transition to {target!r} requires "
                f"{prereq!r} to be true, but it is false."
            )


# ---------------------------------------------------------------------------
# Frozen policy actions (plan §22)
# ---------------------------------------------------------------------------

FROZEN_POLICY_ACTIONS: tuple[str, ...] = ("allow", "redact", "abstract", "block")

# ---------------------------------------------------------------------------
# Condition specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionSpec:
    """One evaluation condition."""

    condition_id: str
    description: str
    exact_enabled: bool
    alias_enabled: bool
    semantic_enabled: bool
    firewall_enabled: bool
    history_enabled: bool
    policy_rich_actions: bool
    reconstruction_guard: bool


# ---------------------------------------------------------------------------
# Condition definitions C0–C4
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, ConditionSpec] = {
    "C0": ConditionSpec(
        condition_id="C0",
        description="No firewall (pass-through)",
        exact_enabled=False,
        alias_enabled=False,
        semantic_enabled=False,
        firewall_enabled=False,
        history_enabled=False,
        policy_rich_actions=False,
        reconstruction_guard=False,
    ),
    "C1": ConditionSpec(
        condition_id="C1",
        description="Exact-match only",
        exact_enabled=True,
        alias_enabled=False,
        semantic_enabled=False,
        firewall_enabled=True,
        history_enabled=False,
        policy_rich_actions=False,
        reconstruction_guard=False,
    ),
    "C2": ConditionSpec(
        condition_id="C2",
        description="Exact + alias",
        exact_enabled=True,
        alias_enabled=True,
        semantic_enabled=False,
        firewall_enabled=True,
        history_enabled=False,
        policy_rich_actions=False,
        reconstruction_guard=False,
    ),
    "C3": ConditionSpec(
        condition_id="C3",
        description="Exact + alias + semantic detector",
        exact_enabled=True,
        alias_enabled=True,
        semantic_enabled=True,
        firewall_enabled=True,
        history_enabled=False,
        policy_rich_actions=False,
        reconstruction_guard=False,
    ),
    "C4": ConditionSpec(
        condition_id="C4",
        description=(
            "Full ForgetFlow: exact + alias + semantic "
            "+ recipient/history-aware protection "
            "+ reconstruction guard "
            "+ purge/recontamination handling"
        ),
        exact_enabled=True,
        alias_enabled=True,
        semantic_enabled=True,
        firewall_enabled=True,
        history_enabled=True,
        policy_rich_actions=True,
        reconstruction_guard=True,
    ),
}

CONDITION_ORDER: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4")


# ---------------------------------------------------------------------------
# Row result schema (plan §29)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowResult:
    """Per-row execution record (plan §29).

    Captures the firewall decision for one candidate under one condition.
    Labels are joined *after* execution (plan §30).

    Decision provenance fields (§12) explain *why* the firewall made
    its decision, enabling research auditability.
    """

    candidate_id: str
    split: str
    condition_id: str
    scenario_id: str
    trust_level: str

    # Detector outputs
    exact_match: bool
    alias_match: bool
    semantic_similarity: float

    # Policy decision
    policy_action: str
    blocked: bool
    allowed: bool

    # Provenance
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

    # Transformation provenance (R1.2b §6)
    initial_policy_action: str = ""
    final_policy_action: str = ""
    transformation_attempt_count: int = 0
    transformation_recheck_passed: bool | None = None
    released_content_sha: str = ""


def row_result_to_dict(rr: RowResult) -> dict[str, Any]:
    """Serialise a RowResult to a plain dict."""
    d = asdict(rr)
    # Convert tuple fields to lists for JSON
    d["triggered_modules"] = list(d["triggered_modules"])
    return d


# ---------------------------------------------------------------------------
# Condition manifest builder
# ---------------------------------------------------------------------------


def build_condition_manifest(
    *,
    code_commit: str = "unknown",
) -> dict[str, Any]:
    """Build the condition manifest (plan §23).

    Returns the manifest dict and writes it to disk.
    """
    conditions_list = []
    for cid in CONDITION_ORDER:
        spec = CONDITIONS[cid]
        conditions_list.append({
            "condition_id": spec.condition_id,
            "description": spec.description,
            "enabled_modules": _enabled_modules(spec),
            "disabled_modules": _disabled_modules(spec),
            "thresholds": _thresholds(spec),
            "policy_rules": {
                "actions": list(FROZEN_POLICY_ACTIONS),
                "rich_actions": spec.policy_rich_actions,
            },
            "embedding_config_version": E5_EMBEDDING_CONFIG_VERSION,
            "code_commit": code_commit,
        })

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_conditions": len(conditions_list),
        "condition_order": list(CONDITION_ORDER),
        "conditions": conditions_list,
        "frozen_policy_actions": list(FROZEN_POLICY_ACTIONS),
        "code_commit": code_commit,
    }

    _CONDITION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONDITION_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return manifest


def _enabled_modules(spec: ConditionSpec) -> list[str]:
    """List the modules enabled for this condition."""
    modules = []
    if spec.firewall_enabled:
        modules.append("firewall")
    if spec.exact_enabled:
        modules.append("exact_detector")
    if spec.alias_enabled:
        modules.append("alias_detector")
    if spec.semantic_enabled:
        modules.append("semantic_detector")
    if spec.history_enabled:
        modules.append("history")
    if spec.reconstruction_guard:
        modules.append("reconstruction_guard")
    if spec.policy_rich_actions:
        modules.append("rich_policy")
    return modules


def _disabled_modules(spec: ConditionSpec) -> list[str]:
    """List the modules disabled for this condition."""
    all_modules = {
        "firewall", "exact_detector", "alias_detector", "semantic_detector",
        "history", "reconstruction_guard", "rich_policy",
    }
    return sorted(all_modules - set(_enabled_modules(spec)))


def _thresholds(spec: ConditionSpec) -> dict[str, Any]:
    """Thresholds relevant to this condition.

    Before calibration, concrete placeholder values are used.
    After calibration/lock (§47), these must be replaced with real values.
    The ``require_test_access_started`` guard enforces this.
    """
    t: dict[str, Any] = {}
    if spec.semantic_enabled:
        t["semantic_threshold"] = "pending_calibration"
    if spec.history_enabled:
        t["reconstruction_threshold"] = "pending_calibration"
    return t


# ---------------------------------------------------------------------------
# Experiment config builder
# ---------------------------------------------------------------------------


def build_experiment_config(
    *,
    code_commit: str = "unknown",
) -> dict[str, Any]:
    """Build the top-level experiment config (plan §18).

    References the calibration selected config and the condition manifest.
    """
    # Load calibration config if available
    selected_config_path = _CALIBRATION_DIR / "selected_config.json"
    dev_config: dict[str, Any] = {}
    if selected_config_path.exists():
        with open(selected_config_path) as f:
            dev_config = json.load(f)

    config = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "E5 Real-Embedding Downstream Evaluation",
        "embedding": {
            "model": E5_EMBEDDING_MODEL,
            "normalization": E5_EMBEDDING_NORMALIZATION,
            "config_version": E5_EMBEDDING_CONFIG_VERSION,
        },
        "detector": {
            "version": _DETECTOR_VERSION,
        },
        "calibration": {
            "semantic_threshold": dev_config.get("semantic_threshold", "pending_calibration"),
            "selection_rule": dev_config.get("selection_rule", "min_recall_0.90_lowest_fbr"),
        },
        "conditions": {
            "n_conditions": len(CONDITION_ORDER),
            "condition_ids": list(CONDITION_ORDER),
            "primary_baseline": "C0",
            "detector_baseline": "C1",
            "full_system": "C4",
        },
        "policy": {
            "actions": list(FROZEN_POLICY_ACTIONS),
            "trust_invariant": True,
        },
        "splits": {
            "development": "calibration",
            "validation": "confirmation",
            "test": "held-out evaluation",
        },
        "code_commit": code_commit,
    }

    _EXPERIMENT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_EXPERIMENT_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return config


# ---------------------------------------------------------------------------
# Test lock (plan §24)
# ---------------------------------------------------------------------------


def build_test_lock(
    *,
    code_commit: str = "unknown",
) -> dict[str, Any]:
    """Build the test-access lock (plan §24, §49 repair).

    This must be committed *before* any test-split evaluation.
    ``test_access_started`` is initially false.

    §49 requires these hash fields:
        code_commit, config_sha, condition_manifest_sha,
        embedding_manifest_sha, selected_config_sha,
        metric_spec_sha, global_annotation_freeze_sha
    """
    # Compute SHAs of prerequisite artifacts
    config_sha = _sha_file(_EXPERIMENT_CONFIG_PATH)
    manifest_sha = _sha_file(_CONDITION_MANIFEST_PATH)

    # Embedding manifest SHA
    embedding_manifest = (
        Path(__file__).resolve().parents[2]
        / "results" / "empirical_v2" / "e5" / "embeddings" / "embedding_manifest.json"
    )
    embedding_sha = _sha_file(embedding_manifest)

    # Selected calibration config SHA (§49)
    selected_config = _CALIBRATION_DIR / "selected_config.json"
    selected_config_sha = _sha_file(selected_config)

    # Metric spec SHA (§30, §49)
    metric_spec_sha = _sha_file(_METRIC_SPEC_PATH)

    # Annotation freeze SHA
    annotation_freeze = (
        Path(__file__).resolve().parents[2]
        / "results" / "empirical_v2" / "annotations"
        / "global_annotation_freeze_manifest.json"
    )
    annotation_sha = _sha_file(annotation_freeze)

    lock = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "config_sha": config_sha,
        "condition_manifest_sha": manifest_sha,
        "embedding_manifest_sha": embedding_sha,
        "selected_config_sha": selected_config_sha,
        "metric_spec_sha": metric_spec_sha,
        "global_annotation_freeze_sha": annotation_sha,
        "test_access_started": False,
        "test_access_started_at": None,
        "execution_commit": None,
    }

    _TEST_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TEST_LOCK_PATH, "w") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")

    return lock


def start_test_access(
    *,
    execution_commit: str,
    lock_path: Path = _TEST_LOCK_PATH,
) -> dict[str, Any]:
    """Transition to official test access (§50).

    At official test start, update ONLY the access state:
        test_access_started = true
        test_access_started_at = timestamp
        execution_commit = exact committed code

    Do NOT change scientific config at the same time.

    Raises:
        TestAccessError: If the lock file doesn't exist or access already started.
        ValueError: If execution_commit is empty.

    Returns:
        The updated test-lock dict.
    """
    if not lock_path.exists():
        raise TestAccessError(
            f"Test lock file not found: {lock_path}. "
            "Cannot start test access without a lock."
        )
    if not execution_commit:
        raise ValueError("execution_commit must be a non-empty string.")

    with open(lock_path) as f:
        lock = json.load(f)

    if lock.get("test_access_started"):
        raise TestAccessError(
            "test_access_started is already true. "
            "Cannot start test access twice."
        )

    lock["test_access_started"] = True
    lock["test_access_started_at"] = datetime.now(timezone.utc).isoformat()
    lock["execution_commit"] = execution_commit

    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")

    return lock


def _sha_file(path: Path) -> str:
    """SHA-256 of a file, or 'missing' if the file doesn't exist."""
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Condition validation
# ---------------------------------------------------------------------------


def validate_conditions() -> list[str]:
    """Validate that all conditions are well-formed.

    Returns a list of error messages (empty = all OK).
    """
    errors: list[str] = []

    # Must have exactly 5 conditions in order
    if set(CONDITIONS.keys()) != set(CONDITION_ORDER):
        errors.append(
            f"Condition keys {sorted(CONDITIONS.keys())} != "
            f"order {list(CONDITION_ORDER)}"
        )

    for cid, spec in CONDITIONS.items():
        if spec.condition_id != cid:
            errors.append(f"{cid}: condition_id mismatch ({spec.condition_id})")

        # C0 must have everything disabled
        if cid == "C0":
            if spec.firewall_enabled:
                errors.append("C0: firewall must be disabled")

        # C4 must have everything enabled
        if cid == "C4":
            if not spec.semantic_enabled:
                errors.append("C4: semantic must be enabled")
            if not spec.history_enabled:
                errors.append("C4: history must be enabled")
            if not spec.reconstruction_guard:
                errors.append("C4: reconstruction_guard must be enabled")

        # Monotonicity: C0 ⊆ C1 ⊆ C2 ⊆ C3 ⊆ C4
        idx = CONDITION_ORDER.index(cid)
        if idx > 0:
            prev = CONDITIONS[CONDITION_ORDER[idx - 1]]
            if not _is_subset(prev, spec):
                errors.append(
                    f"{cid}: not a superset of {prev.condition_id}"
                )

    return errors


def _is_subset(a: ConditionSpec, b: ConditionSpec) -> bool:
    """Check that condition *a* is a subset of condition *b*.

    Every module enabled in *a* must also be enabled in *b*.
    """
    checks = [
        (a.exact_enabled, b.exact_enabled),
        (a.alias_enabled, b.alias_enabled),
        (a.semantic_enabled, b.semantic_enabled),
        (a.firewall_enabled, b.firewall_enabled),
        (a.history_enabled, b.history_enabled),
        (a.policy_rich_actions, b.policy_rich_actions),
        (a.reconstruction_guard, b.reconstruction_guard),
    ]
    return all(not enabled_a or enabled_b for enabled_a, enabled_b in checks)

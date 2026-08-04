"""Section 16: canonical execution-status taxonomy.

Execution status is deliberately separated from release certification.  A run
always reports the *highest* tier whose gates are satisfied:

``EXECUTION_COMPLETE``
    The run finished, every expected artifact was written, provenance is
    present, and there is no file corruption.  This says nothing about whether
    the results are scientifically valid — only that execution succeeded.

``DIAGNOSTIC_VALID``
    ``EXECUTION_COMPLETE`` — i.e. a complete, uncorrupted diagnostic run.

``RESEARCH_VALID``
    ``DIAGNOSTIC_VALID`` plus every research gate: audit valid, manifest valid,
    all assertions pass, all directional checks pass, metrics consistent, and
    fixtures demonstrate their intended effects.

``RELEASE_CANDIDATE``
    ``RESEARCH_VALID`` plus a clean repository and the full release gate (full
    and integration tests, Ruff, Ruff format, Mypy, checksums, and GitHub
    Actions all pass).

The final certification status must be ``RESEARCH_VALID`` or
``RELEASE_CANDIDATE`` (repair spec §20/§21).

Remediation §4: study classes
-----------------------------

Execution status describes whether a run executed correctly; it says
nothing about what *kind* of evidence the run produces.  Every artifact
records a ``study_class``:

``diagnostic``
    Scripted responses, fixed/deterministic embeddings, deterministic
    templates.  Validates code paths, interception, and metric
    accounting.  Can never support empirical claims about LLM agents.

``empirical_replay``
    A frozen corpus produced by real (pinned) models or human authors,
    replayed across all conditions with a pinned real embedding model.

``closed_loop``
    Agents generate responses during the episode; the firewall acts on
    naturally produced messages.

Remediation §31: staged research statuses
------------------------------------------

The binary ``research_valid`` label is replaced by staged statuses so a
deterministic synthetic pipeline can never claim empirical validity:

``diagnostic_valid`` → ``synthetic_benchmark_valid`` →
``empirical_replay_valid`` → ``closed_loop_study_valid`` →
``release_candidate``.

Deterministic fixed-embedding runs are capped at
``synthetic_benchmark_valid``; empirical statuses additionally require
the study-design evidence checked by the research gate.
"""

from __future__ import annotations

EXECUTION_COMPLETE = "EXECUTION_COMPLETE"
DIAGNOSTIC_VALID = "DIAGNOSTIC_VALID"
RESEARCH_VALID = "RESEARCH_VALID"
RELEASE_CANDIDATE = "RELEASE_CANDIDATE"

# Ordered from lowest to highest tier.
STATUS_ORDER: tuple[str, ...] = (
    EXECUTION_COMPLETE,
    DIAGNOSTIC_VALID,
    RESEARCH_VALID,
    RELEASE_CANDIDATE,
)


def compute_status(
    *,
    execution_complete: bool,
    diagnostic_valid: bool,
    research_valid: bool,
    release_candidate: bool,
) -> str:
    """Return the highest status tier whose gates are all satisfied.

    The tiers are nested: each higher tier implies the lower ones, so the
    highest tier with a ``True`` gate is the run's status.  When no gate is
    satisfied the run still completed enough to be ``EXECUTION_COMPLETE`` only
    if ``execution_complete`` holds; otherwise it is still reported as
    ``EXECUTION_COMPLETE`` (the lowest tier) so downstream tooling always sees a
    well-formed status.
    """
    if release_candidate:
        return RELEASE_CANDIDATE
    if research_valid:
        return RESEARCH_VALID
    if diagnostic_valid:
        return DIAGNOSTIC_VALID
    return EXECUTION_COMPLETE


def status_at_least(status: str, minimum: str) -> bool:
    """Return True when *status* meets or exceeds the *minimum* tier."""
    if status not in STATUS_ORDER or minimum not in STATUS_ORDER:
        return False
    return STATUS_ORDER.index(status) >= STATUS_ORDER.index(minimum)


# ---------------------------------------------------------------------------
# Remediation §4: study classes
# ---------------------------------------------------------------------------

STUDY_CLASS_DIAGNOSTIC = "diagnostic"
STUDY_CLASS_EMPIRICAL_REPLAY = "empirical_replay"
STUDY_CLASS_CLOSED_LOOP = "closed_loop"

STUDY_CLASSES: tuple[str, ...] = (
    STUDY_CLASS_DIAGNOSTIC,
    STUDY_CLASS_EMPIRICAL_REPLAY,
    STUDY_CLASS_CLOSED_LOOP,
)


def validate_study_class(study_class: str) -> None:
    """Raise ValueError for an undeclared study class (remediation §4)."""
    if study_class not in STUDY_CLASSES:
        raise ValueError(
            f"unknown study_class {study_class!r}; declared classes: {sorted(STUDY_CLASSES)}"
        )


# ---------------------------------------------------------------------------
# Remediation §31: staged research statuses
# ---------------------------------------------------------------------------

# Fallback for runs that do not satisfy any staged status: complete
# execution evidence may exist, but the study is not certified at any tier.
RESEARCH_STATUS_NOT_VALID = "diagnostic"

DIAGNOSTIC_VALID_RS = "diagnostic_valid"
SYNTHETIC_BENCHMARK_VALID = "synthetic_benchmark_valid"
EMPIRICAL_REPLAY_VALID = "empirical_replay_valid"
CLOSED_LOOP_STUDY_VALID = "closed_loop_study_valid"
RESEARCH_STATUS_RELEASE_CANDIDATE = "release_candidate"

# Ordered from weakest to strongest evidence tier.
RESEARCH_STATUS_ORDER: tuple[str, ...] = (
    DIAGNOSTIC_VALID_RS,
    SYNTHETIC_BENCHMARK_VALID,
    EMPIRICAL_REPLAY_VALID,
    CLOSED_LOOP_STUDY_VALID,
    RESEARCH_STATUS_RELEASE_CANDIDATE,
)


def research_status_at_least(status: str, minimum: str) -> bool:
    """Return True when *status* meets or exceeds the *minimum* tier."""
    if status not in RESEARCH_STATUS_ORDER or minimum not in RESEARCH_STATUS_ORDER:
        return False
    return RESEARCH_STATUS_ORDER.index(status) >= RESEARCH_STATUS_ORDER.index(minimum)


def compute_research_status(
    *,
    study_class: str,
    substantive_gates_passed: bool,
    suite_passed: bool,
    empirical_design_passed: bool = False,
    closed_loop_design_passed: bool = False,
    release_ready: bool = False,
) -> str:
    """Return the highest staged research status whose criteria are met.

    Criteria (remediation §31):

    ``diagnostic`` (not a staged status)
        Some substantive gate failed — no certification is possible.

    ``diagnostic_valid``
        Every substantive internal-consistency gate passed, but the full
        test suite / static checks have not been run.

    ``synthetic_benchmark_valid``
        ``diagnostic_valid`` plus the full suite.  This is the ceiling for
        deterministic fixed-embedding studies: ``empirical_design_passed``
        must be False for them by construction.

    ``empirical_replay_valid`` / ``closed_loop_study_valid``
        Additionally require the study-design evidence gate (real corpus,
        pinned real embeddings, declared study class).  A diagnostic-class
        run can never reach these tiers.

    ``release_candidate``
        The study-class ceiling plus release readiness (clean provenance
        and every gate, including suite checks, passed).  Release
        readiness can never exceed the class ceiling: a diagnostic run
        stays at ``synthetic_benchmark_valid`` no matter how many gates
        pass.
    """
    validate_study_class(study_class)
    if not substantive_gates_passed:
        return RESEARCH_STATUS_NOT_VALID
    if not suite_passed:
        return DIAGNOSTIC_VALID_RS

    ceiling = SYNTHETIC_BENCHMARK_VALID
    if study_class == STUDY_CLASS_EMPIRICAL_REPLAY and empirical_design_passed:
        ceiling = EMPIRICAL_REPLAY_VALID
    if study_class == STUDY_CLASS_CLOSED_LOOP and closed_loop_design_passed:
        ceiling = CLOSED_LOOP_STUDY_VALID

    if release_ready and ceiling != SYNTHETIC_BENCHMARK_VALID:
        return RESEARCH_STATUS_RELEASE_CANDIDATE
    return ceiling

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

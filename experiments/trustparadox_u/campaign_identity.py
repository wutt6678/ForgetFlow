"""Patch E: campaign identity protection for resume safety.

Computes, writes, and verifies ``campaign_identity.json`` to ensure that
a resumed campaign matches the original campaign definition exactly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    compute_target_registry_hash,
)
from experiments.trustparadox_u.empirical_generation import (
    build_prompt_manifest,
    prompt_manifest_sha256,
)
from experiments.trustparadox_u.empirical_generation_plan import (
    FrozenGenerationConfig,
    GenerationPlanItem,
    plan_sha256,
)

CAMPAIGN_IDENTITY_FILENAME = "campaign_identity.json"
CAMPAIGN_IDENTITY_SCHEMA_VERSION = "1.0"

# Fields that block resume on mismatch.
_BLOCKING_FIELDS: frozenset[str] = frozenset({
    "generation_plan_scientific_sha256",
    "generation_plan_file_sha256",
    "generation_config_sha256",
    "target_registry_sha256",
    "prompt_manifest_sha256",
    "phase_manifest_sha256",
    "split",
    "generator_provider",
    "generator_model_requested",
    "generator_temperature",
    "generator_max_tokens",
    "request_timeout",
    "max_retries",
    "created_from_commit",
})


@dataclass(frozen=True)
class CampaignIdentity:
    """Immutable snapshot of campaign-defining parameters."""

    schema_version: str
    split: str
    generation_plan_scientific_sha256: str
    generation_plan_file_sha256: str
    generation_config_sha256: str
    target_registry_sha256: str
    prompt_manifest_sha256: str
    phase_manifest_sha256: str
    generator_provider: str
    generator_model_requested: str
    generator_temperature: float
    generator_max_tokens: int
    request_timeout: float
    max_retries: int
    created_from_commit: str
    created_at: str


def _repository_commit() -> str:
    """Return the current HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _file_sha256(path: Path) -> str:
    """Return SHA-256 of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_campaign_identity(
    *,
    split: str,
    plan_items: list[GenerationPlanItem] | None,
    plan_path: Path | None,
    config: FrozenGenerationConfig,
    config_path: Path | None = None,
    phase_manifest_path: Path | None = None,
) -> CampaignIdentity:
    """Compute the campaign identity from current on-disk artifacts."""
    # Plan hashes.
    if plan_items is not None:
        scientific_hash = plan_sha256(plan_items)
    else:
        scientific_hash = ""
    if plan_path is not None and plan_path.exists():
        file_hash = _file_sha256(plan_path)
    else:
        file_hash = ""

    # Config hash.
    if config_path is not None and config_path.exists():
        config_hash = _file_sha256(config_path)
    else:
        config_hash = ""

    # Target registry hash.
    target_hash = compute_target_registry_hash(EMPIRICAL_TARGET_REGISTRY)

    # Prompt manifest hash.
    pm = build_prompt_manifest()
    pm_json = json.dumps(pm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    pm_hash = hashlib.sha256(pm_json.encode("utf-8")).hexdigest()

    # Phase manifest hash.
    phase_hash = ""
    if phase_manifest_path is not None and phase_manifest_path.exists():
        phase_hash = _file_sha256(phase_manifest_path)

    return CampaignIdentity(
        schema_version=CAMPAIGN_IDENTITY_SCHEMA_VERSION,
        split=split,
        generation_plan_scientific_sha256=scientific_hash,
        generation_plan_file_sha256=file_hash,
        generation_config_sha256=config_hash,
        target_registry_sha256=target_hash,
        prompt_manifest_sha256=pm_hash,
        phase_manifest_sha256=phase_hash,
        generator_provider=config.generator_provider,
        generator_model_requested=config.generator_model_requested,
        generator_temperature=config.generator_temperature,
        generator_max_tokens=config.generator_max_tokens,
        request_timeout=config.request_timeout,
        max_retries=config.max_retries,
        created_from_commit=_repository_commit(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_campaign_identity(
    output_dir: Path,
    identity: CampaignIdentity,
) -> Path:
    """Write campaign identity atomically to *output_dir*."""
    path = output_dir / CAMPAIGN_IDENTITY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(identity)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.rename(path)
    return path


def load_campaign_identity(output_dir: Path) -> CampaignIdentity | None:
    """Load existing campaign identity, or None if missing."""
    path = output_dir / CAMPAIGN_IDENTITY_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return CampaignIdentity(**data)


class CampaignIdentityMismatchError(RuntimeError):
    """Raised when a resumed campaign's identity does not match."""

    def __init__(self, mismatches: dict[str, dict[str, Any]]) -> None:
        self.mismatches = mismatches
        details = ", ".join(
            f"{field}: recorded={vals['recorded']!r} current={vals['current']!r}"
            for field, vals in sorted(mismatches.items())
        )
        super().__init__(f"campaign identity mismatch: {details}")


def verify_campaign_identity(
    existing: CampaignIdentity,
    current: CampaignIdentity,
) -> None:
    """Verify that *current* matches *existing* on all blocking fields.

    Raises :class:`CampaignIdentityMismatchError` on any mismatch.
    """
    mismatches: dict[str, dict[str, Any]] = {}
    current_dict = asdict(current)
    existing_dict = asdict(existing)
    for field in sorted(_BLOCKING_FIELDS):
        old_val = existing_dict.get(field)
        new_val = current_dict.get(field)
        # Skip empty fields (e.g. no plan → no hash to compare).
        if old_val == "" and new_val == "":
            continue
        if old_val != new_val:
            mismatches[field] = {"recorded": old_val, "current": new_val}
    if mismatches:
        raise CampaignIdentityMismatchError(mismatches)

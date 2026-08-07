"""GE-014: finalize the active release's storage sidecar.

Stage-B certification of the GE-009 two-stage model: point the active
sidecar at the genuinely passing gate-evidence commit 9acdd65...,
compute the GE-002 digest from the exact historical gate bytes, bump
the sidecar to schema 1.2 (GE-013), recompute storage_metadata_digest
and preserve the scientific release digest.  After the storage
provenance check passes, store the durable GE-015 certification record.

The resulting chain is non-self-referential:

    9acdd65...       passing gate-evidence commit
        |
    <new commit>     finalized sidecar referencing 9acdd65...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.release_bundle import (  # noqa: E402
    release_dirs,
    write_final_storage_certification,
    write_storage_provenance,
)
from experiments.trustparadox_u.research_valid_gate import (  # noqa: E402
    check_release_storage_provenance,
)

_PASSING_EVIDENCE_COMMIT = "9acdd652a2204572e829662f1712546b25184731"


def main() -> None:
    active = release_dirs()
    if len(active) != 1:
        raise SystemExit(f"expected exactly one active release, got {[d.name for d in active]}")
    bundle_dir = active[0]

    # GE-014: finalize the sidecar with the passing gate-evidence commit;
    # the GE-002 digest is computed from the Git-stored gate bytes.
    sidecar = write_storage_provenance(bundle_dir, gate_evidence_commit=_PASSING_EVIDENCE_COMMIT)
    print(f"finalized sidecar for {sidecar['release_id']}")
    print(f"  schema_version         = {sidecar['schema_version']}")
    print(f"  gate_evidence_commit   = {sidecar['gate_evidence_commit']}")
    print(f"  gate_evidence_sha256   = {sidecar['gate_evidence_sha256']}")
    print(f"  scientific_release_digest = {sidecar['scientific_release_digest']}")
    print(f"  storage_metadata_digest   = {sidecar['storage_metadata_digest']}")

    # Stage-B verification-only storage provenance check.
    result = check_release_storage_provenance()
    print(f"check_release_storage_provenance passed = {result['passed']}")
    for finding in result.get("findings", []):
        print(f"  finding: {finding}")
    if not result["passed"]:
        raise SystemExit("storage provenance check failed; refusing to certify")

    # GE-015: durable, non-self-referential certification record.
    record = write_final_storage_certification(bundle_dir, passed=True)
    print("FINAL_STORAGE_CERTIFICATION.json:")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

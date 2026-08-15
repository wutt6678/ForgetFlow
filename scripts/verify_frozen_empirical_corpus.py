#!/usr/bin/env python3
"""Read-only freeze verifier for the frozen E3 empirical corpus.

Implements Section 33 of the Corpus Freeze and Immutable Artifact
Provenance Plan.  This script NEVER modifies any file — it only loads
and verifies the frozen corpus manifest, artifact inventory, and all
inventoried artifacts.

Checks performed:
  1. Load frozen_corpus_manifest.json
  2. Load freeze_artifact_inventory.json
  3. Re-hash every inventoried file and compare with stored SHA256
  4. Verify inventory root hash
  5. Verify split counts (plan items, scientific units)
  6. Verify split gates (generation_completed, audit_passed, source_commit)
  7. Verify campaign identities (source commit, endpoint consistency)
  8. Verify endpoint consistency across splits
  9. Verify source commit consistency everywhere
  10. Verify combined-audit hash
  11. Verify corpus_frozen == true

Usage:
    PYTHONPATH=. python scripts/verify_frozen_empirical_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
_PHASE_PATH = (
    _PROJECT_ROOT
    / "data"
    / "trustparadox_u"
    / "empirical_v2"
    / "manifests"
    / "empirical_phase.json"
)
_SOURCE_COMMIT = "f72e6f4a5f426911fd98ac2822e4695211d61ca0"
_EXPECTED_ENDPOINT_HOST = "llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com"
_EXPECTED_ENDPOINT_SHA = (
    "3d1699591685ab6c385ef26f30f469653ca8f506b33f1ded71e497a133d3d2c6"
)
_EXPECTED_API_PROTOCOL = "openai_compatible"
_SPLITS = ["development", "validation", "test"]
_EXPECTED_COUNTS = {"development": 225, "validation": 225, "test": 450}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class FreezeVerifier:
    """Read-only verifier for the frozen corpus."""

    def __init__(self) -> None:
        self.findings: list[str] = []
        self.checks_passed: int = 0
        self.checks_failed: int = 0

    def _pass(self, msg: str) -> None:
        self.checks_passed += 1
        print(f"  PASS: {msg}")

    def _fail(self, msg: str) -> None:
        self.checks_failed += 1
        self.findings.append(msg)
        print(f"  FAIL: {msg}")

    def verify_all(self) -> bool:
        """Run all verification checks. Returns True if all pass."""
        print("=" * 70)
        print("FROZEN CORPUS VERIFIER")
        print("=" * 70)

        self._verify_frozen_manifest()
        self._verify_inventory()
        self._verify_inventoried_files()
        self._verify_split_gates()
        self._verify_campaign_identities()
        self._verify_endpoint_consistency()
        self._verify_source_commit_consistency()
        self._verify_combined_audit_hash()
        self._verify_split_counts()
        self._verify_corpus_frozen()
        self._verify_phase_manifest()

        print("\n" + "=" * 70)
        total = self.checks_passed + self.checks_failed
        print(f"Results: {self.checks_passed}/{total} checks passed, "
              f"{self.checks_failed} failed")
        if self.findings:
            print(f"\nBLOCKING FINDINGS ({len(self.findings)}):")
            for f in self.findings:
                print(f"  - {f}")
            print("\nVERIFICATION: FAIL")
            return False
        else:
            print("\nVERIFICATION: PASS")
            return True

    def _verify_frozen_manifest(self) -> None:
        """Check 1: Load and verify frozen corpus manifest."""
        print("\n[1] Frozen corpus manifest...")
        path = _CORPUS_DIR / "frozen_corpus_manifest.json"
        if not path.exists():
            self._fail("frozen_corpus_manifest.json not found")
            return
        manifest = _load_json(path)
        if manifest.get("corpus_frozen") is True:
            self._pass("corpus_frozen == true")
        else:
            self._fail("corpus_frozen != true")
        if manifest.get("source_generation_commit") == _SOURCE_COMMIT:
            self._pass("source_generation_commit matches")
        else:
            self._fail(f"source_generation_commit mismatch: "
                       f"{manifest.get('source_generation_commit')}")
        if manifest.get("blocking_finding_count") == 0:
            self._pass("blocking_finding_count == 0")
        else:
            self._fail("blocking_finding_count != 0")
        if manifest.get("combined_audit_passed") is True:
            self._pass("combined_audit_passed == true")
        else:
            self._fail("combined_audit_passed != true")
        if manifest.get("full_generation_plan_item_count") == 900:
            self._pass("full_generation_plan_item_count == 900")
        else:
            self._fail("full_generation_plan_item_count != 900")
        if manifest.get("scientific_unit_count") == 660:
            self._pass("scientific_unit_count == 660")
        else:
            self._fail("scientific_unit_count != 660")
        if manifest.get("accepted_candidate_count") == 900:
            self._pass("accepted_candidate_count == 900")
        else:
            self._fail("accepted_candidate_count != 900")

    def _verify_inventory(self) -> None:
        """Check 2: Load and verify artifact inventory."""
        print("\n[2] Artifact inventory...")
        path = _CORPUS_DIR / "freeze_artifact_inventory.json"
        if not path.exists():
            self._fail("freeze_artifact_inventory.json not found")
            return
        inventory = _load_json(path)
        entries = inventory.get("entries", [])
        if len(entries) > 0:
            self._pass(f"inventory has {len(entries)} entries")
        else:
            self._fail("inventory is empty")
        if inventory.get("source_generation_commit") == _SOURCE_COMMIT:
            self._pass("inventory source_generation_commit matches")
        else:
            self._fail("inventory source_generation_commit mismatch")
        # Verify inventory root hash
        stored_sha = inventory.get("inventory_sha256")
        if stored_sha:
            # Recompute: serialize entries deterministically
            entries_bytes = json.dumps(
                entries, indent=2, sort_keys=True, ensure_ascii=False
            ).encode("utf-8") + b"\n"
            computed_sha = hashlib.sha256(entries_bytes).hexdigest()
            # Note: the stored SHA may be of the full file, not just entries
            # So we also check the file hash
            file_sha = _sha256(path)
            if file_sha == stored_sha or computed_sha == stored_sha:
                self._pass("inventory root hash verified")
            else:
                self._fail("inventory root hash mismatch")
        else:
            self._fail("inventory missing inventory_sha256")

    def _verify_inventoried_files(self) -> None:
        """Check 3: Re-hash every inventoried file."""
        print("\n[3] Re-hashing inventoried files...")
        path = _CORPUS_DIR / "freeze_artifact_inventory.json"
        if not path.exists():
            self._fail("Cannot verify files: inventory not found")
            return
        inventory = _load_json(path)
        entries = inventory.get("entries", [])
        all_ok = True
        for entry in entries:
            rel_path = entry["path"]
            expected_sha = entry["sha256"]
            if rel_path.startswith("data/"):
                full_path = _PROJECT_ROOT / rel_path
            else:
                full_path = _CORPUS_DIR / rel_path
            if not full_path.exists():
                self._fail(f"Missing inventoried file: {rel_path}")
                all_ok = False
                continue
            actual_sha = _sha256(full_path)
            if actual_sha != expected_sha:
                self._fail(f"Hash mismatch: {rel_path}")
                all_ok = False
        if all_ok:
            self._pass(f"All {len(entries)} inventoried files hash-verified")

    def _verify_split_gates(self) -> None:
        """Check 4: Verify split gates."""
        print("\n[4] Split gates...")
        for sp in _SPLITS:
            gate_path = _CORPUS_DIR / f"{sp}_generation_gate.json"
            if not gate_path.exists():
                self._fail(f"Missing gate: {sp}")
                continue
            gate = _load_json(gate_path)
            if gate.get("generation_completed") is True:
                self._pass(f"{sp} gate: generation_completed")
            else:
                self._fail(f"{sp} gate: generation_completed != true")
            if gate.get("audit_passed") is True:
                self._pass(f"{sp} gate: audit_passed")
            else:
                self._fail(f"{sp} gate: audit_passed != true")
            if gate.get("source_commit") == _SOURCE_COMMIT:
                self._pass(f"{sp} gate: source_commit matches")
            else:
                self._fail(f"{sp} gate: source_commit mismatch")

    def _verify_campaign_identities(self) -> None:
        """Check 5: Verify campaign identities."""
        print("\n[5] Campaign identities...")
        for sp in _SPLITS:
            id_path = _CORPUS_DIR / sp / "campaign_identity.json"
            if not id_path.exists():
                self._fail(f"Missing campaign identity: {sp}")
                continue
            ident = _load_json(id_path)
            if ident.get("created_from_commit") == _SOURCE_COMMIT:
                self._pass(f"{sp} identity: source commit matches")
            else:
                self._fail(f"{sp} identity: source commit mismatch")

    def _verify_endpoint_consistency(self) -> None:
        """Checks 6-7: Verify endpoint consistency across splits."""
        print("\n[6] Endpoint consistency...")
        for sp in _SPLITS:
            id_path = _CORPUS_DIR / sp / "campaign_identity.json"
            if not id_path.exists():
                continue
            ident = _load_json(id_path)
            if ident.get("serving_endpoint_host") == _EXPECTED_ENDPOINT_HOST:
                self._pass(f"{sp} endpoint host matches")
            else:
                self._fail(f"{sp} endpoint host mismatch")
            if ident.get("serving_endpoint_sha256") == _EXPECTED_ENDPOINT_SHA:
                self._pass(f"{sp} endpoint SHA256 matches")
            else:
                self._fail(f"{sp} endpoint SHA256 mismatch")
            if ident.get("api_protocol") == _EXPECTED_API_PROTOCOL:
                self._pass(f"{sp} api_protocol matches")
            else:
                self._fail(f"{sp} api_protocol mismatch")

    def _verify_source_commit_consistency(self) -> None:
        """Check 8: Verify source commit consistency across all artifacts."""
        print("\n[7] Source commit consistency...")
        # Check frozen manifest
        fm = _load_json(_CORPUS_DIR / "frozen_corpus_manifest.json")
        fm_commit = fm.get("source_generation_commit")
        # Check gates
        for sp in _SPLITS:
            gate = _load_json(_CORPUS_DIR / f"{sp}_generation_gate.json")
            if gate.get("source_commit") == fm_commit:
                self._pass(f"{sp} gate commit matches frozen manifest")
            else:
                self._fail(f"{sp} gate commit != frozen manifest commit")
        # Check corpus manifests
        for sp in _SPLITS:
            cm = _load_json(_CORPUS_DIR / sp / "corpus_manifest.json")
            if cm.get("repository_commit") == fm_commit:
                self._pass(f"{sp} corpus manifest commit matches")
            else:
                self._fail(f"{sp} corpus manifest commit mismatch")

    def _verify_combined_audit_hash(self) -> None:
        """Check 9: Verify combined audit hash matches frozen manifest."""
        print("\n[8] Combined audit hash...")
        fm = _load_json(_CORPUS_DIR / "frozen_corpus_manifest.json")
        expected_sha = fm.get("full_corpus_validation_report_sha256")
        if not expected_sha:
            self._fail("frozen manifest missing combined audit hash")
            return
        actual_sha = _sha256(_CORPUS_DIR / "full_corpus_validation_report.json")
        if actual_sha == expected_sha:
            self._pass("combined audit hash matches frozen manifest")
        else:
            self._fail("combined audit hash mismatch")

    def _verify_split_counts(self) -> None:
        """Check 10: Verify split counts."""
        print("\n[9] Split counts...")
        for sp in _SPLITS:
            expected = _EXPECTED_COUNTS[sp]
            # Check from corpus manifest
            cm = _load_json(_CORPUS_DIR / sp / "corpus_manifest.json")
            actual = cm.get("accepted_candidate_count")
            if actual == expected:
                self._pass(f"{sp}: {actual} accepted candidates")
            else:
                self._fail(f"{sp}: expected {expected}, got {actual}")
            # Check raw attempts
            raw_path = _CORPUS_DIR / sp / "raw_generation_attempts.jsonl"
            raw_count = len(raw_path.read_text().strip().split("\n"))
            if raw_count == expected:
                self._pass(f"{sp}: {raw_count} raw attempts")
            else:
                self._fail(f"{sp}: expected {expected} raw, got {raw_count}")

    def _verify_corpus_frozen(self) -> None:
        """Check 11: Verify corpus_frozen == true in phase manifest."""
        print("\n[10] Corpus frozen flag...")
        if not _PHASE_PATH.exists():
            self._fail("empirical_phase.json not found")
            return
        phase = _load_json(_PHASE_PATH)
        if phase.get("corpus_frozen") is True:
            self._pass("empirical_phase.corpus_frozen == true")
        else:
            self._fail("empirical_phase.corpus_frozen != true")

    def _verify_phase_manifest(self) -> None:
        """Check 12: Verify phase manifest has freeze provenance."""
        print("\n[11] Phase manifest freeze provenance...")
        if not _PHASE_PATH.exists():
            self._fail("empirical_phase.json not found")
            return
        phase = _load_json(_PHASE_PATH)
        if phase.get("corpus_source_commit") == _SOURCE_COMMIT:
            self._pass("phase corpus_source_commit matches")
        else:
            self._fail("phase corpus_source_commit mismatch")
        if phase.get("frozen_corpus_manifest_sha256"):
            self._pass("phase has frozen_corpus_manifest_sha256")
        else:
            self._fail("phase missing frozen_corpus_manifest_sha256")
        if phase.get("freeze_artifact_inventory_sha256"):
            self._pass("phase has freeze_artifact_inventory_sha256")
        else:
            self._fail("phase missing freeze_artifact_inventory_sha256")


def main() -> int:
    verifier = FreezeVerifier()
    passed = verifier.verify_all()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

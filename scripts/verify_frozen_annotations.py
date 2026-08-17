#!/usr/bin/env python3
"""Read-only freeze verifier for frozen development annotations.

Implements Section 35 of the E4-001A Repair and Next-Step Checklist.
This script NEVER modifies any file — it only loads and verifies the
frozen annotation manifests, artifact hashes, row/sequence counts,
adjudication completeness, and gate status.

Usage:
    PYTHONPATH=. python scripts/verify_frozen_annotations.py --split development
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_DEV_DIR = _ANNOTATIONS_DIR / "development_v3"
_VAL_DIR = _ANNOTATIONS_DIR / "validation"
_PHASE_PATH = _ANNOTATIONS_DIR / "annotation_phase.json"
_PROTOCOL_PATH = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"

# Files bound in the development annotation manifest (Sec 34)
_DEV_FILES = {
    "primary_raw": "primary_annotation_attempts.jsonl",
    "primary_labels": "row_annotations.jsonl",
    "secondary_raw": "secondary_annotation_attempts.jsonl",
    "secondary_labels": "secondary_row_annotations.jsonl",
    "sequence_primary": "sequence_annotations.jsonl",
    "sequence_secondary": "secondary_sequence_annotations.jsonl",
    "agreement_report": "audit_report.json",
    "review_queue": "review_queue.jsonl",
    "llm_adjudication": "llm_adjudication.jsonl",
    "final_adjudicated_labels": "final_adjudicated_labels.jsonl",
    "adjudication_manifest": "adjudication_manifest.json",
}

# Manifest hash key → file key mapping
_MANIFEST_HASH_MAP = {
    "primary_raw_sha256": "primary_raw",
    "primary_labels_sha256": "primary_labels",
    "secondary_raw_sha256": "secondary_raw",
    "secondary_labels_sha256": "secondary_labels",
    "sequence_primary_labels_sha256": "sequence_primary",
    "sequence_secondary_labels_sha256": "sequence_secondary",
    "agreement_report_sha256": "agreement_report",
    "review_queue_sha256": "review_queue",
}

# Files bound in the validation annotation manifest (Sec 43)
_VAL_FILES = {
    "primary_raw": "primary_annotation_attempts.jsonl",
    "primary_labels": "primary_row_annotations.jsonl",
    "primary_sequences": "primary_sequence_annotations.jsonl",
    "secondary_raw": "secondary_annotation_attempts.jsonl",
    "secondary_labels": "secondary_row_annotations.jsonl",
    "secondary_sequences": "secondary_sequence_annotations.jsonl",
    "agreement_report": "validation_agreement_report.json",
    "review_queue": "review_queue.jsonl",
    "llm_adjudication": "llm_adjudication.jsonl",
    "final_adjudicated_labels": "final_adjudicated_labels.jsonl",
    "final_sequence_labels": "final_sequence_labels.jsonl",
    "adjudication_manifest": "adjudication_manifest.json",
}

# Validation manifest hash key → file key mapping
_VAL_HASH_MAP = {
    "primary_raw_sha256": "primary_raw",
    "primary_labels_sha256": "primary_labels",
    "primary_sequences_sha256": "primary_sequences",
    "secondary_raw_sha256": "secondary_raw",
    "secondary_labels_sha256": "secondary_labels",
    "secondary_sequences_sha256": "secondary_sequences",
    "agreement_report_sha256": "agreement_report",
    "review_queue_sha256": "review_queue",
    "llm_adjudication_sha256": "llm_adjudication",
    "final_adjudicated_labels_sha256": "final_adjudicated_labels",
    "final_sequence_labels_sha256": "final_sequence_labels",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl(path: Path) -> int:
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


class FrozenAnnotationVerifier:
    """Read-only verifier for frozen development annotations."""

    def __init__(self, split: str = "development") -> None:
        self.split = split
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
        print(f"FROZEN ANNOTATION VERIFIER — {self.split}")
        print("=" * 70)

        if self.split == "validation":
            return self._verify_validation()

        self._verify_files_exist()
        self._verify_file_hashes()
        self._verify_frozen_corpus_sha()
        self._verify_row_counts()
        self._verify_sequence_counts()
        self._verify_adjudication_complete()
        self._verify_gate_go()
        self._verify_protocol_frozen()
        self._verify_phase_status()

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

    def _verify_validation(self) -> bool:
        """Sec 46: Validation-specific verification."""
        print("\n--- Validation Verification ---")

        # 1. Development protocol is frozen
        print("\n[V1] Development protocol frozen...")
        if _PROTOCOL_PATH.exists():
            pm = _load_json(_PROTOCOL_PATH)
            if pm.get("annotation_schema_frozen") and pm.get("annotation_prompts_frozen"):
                self._pass("development protocol frozen")
            else:
                self._fail("development protocol not frozen")
        else:
            self._fail("annotation_protocol_manifest.json not found")

        # 2. Validation protocol hash matches frozen development protocol
        print("\n[V2] Validation protocol hash matches development...")
        val_manifest_path = _VAL_DIR / "annotation_manifest.json"
        if val_manifest_path.exists():
            vm = _load_json(val_manifest_path)
            crossref = vm.get("protocol_hash_crossref", {})
            if pm and crossref.get("annotation_schema_sha256") == pm.get("annotation_schema_sha256"):
                self._pass("schema hash matches")
            else:
                self._fail("schema hash mismatch")
            if pm and crossref.get("prompt_manifest_sha256") == pm.get("prompt_manifest_sha256"):
                self._pass("prompt manifest hash matches")
            else:
                self._fail("prompt manifest hash mismatch")
        else:
            self._fail("validation annotation_manifest.json not found")

        # 3. All validation artifacts exist
        print("\n[V3] Validation artifact existence...")
        for key, fname in _VAL_FILES.items():
            fpath = _VAL_DIR / fname
            if fpath.exists():
                self._pass(f"{key}: {fname} exists")
            else:
                self._fail(f"{key}: {fname} MISSING")

        # 4. All hashes match
        print("\n[V4] Validation artifact hash verification...")
        if val_manifest_path.exists():
            vm = _load_json(val_manifest_path)
            for hash_key, file_key in _VAL_HASH_MAP.items():
                expected = vm.get(hash_key, "")
                if not expected:
                    self._fail(f"{file_key}: no hash in manifest")
                    continue
                fpath = _VAL_DIR / _VAL_FILES[file_key]
                if not fpath.exists():
                    self._fail(f"{file_key}: file missing")
                    continue
                actual = _sha256(fpath)
                if actual == expected:
                    self._pass(f"{file_key}: SHA256 matches")
                else:
                    self._fail(f"{file_key}: SHA256 MISMATCH")
        else:
            self._fail("annotation_manifest.json not found — cannot verify hashes")

        # 5. Row counts: 225
        print("\n[V5] Validation row counts...")
        primary_rows = _count_jsonl(_VAL_DIR / "primary_row_annotations.jsonl")
        secondary_rows = _count_jsonl(_VAL_DIR / "secondary_row_annotations.jsonl")
        final_rows = _count_jsonl(_VAL_DIR / "final_adjudicated_labels.jsonl")
        if primary_rows == 225:
            self._pass(f"primary rows: {primary_rows}/225")
        else:
            self._fail(f"primary rows: expected 225, got {primary_rows}")
        if secondary_rows == 225:
            self._pass(f"secondary rows: {secondary_rows}/225")
        else:
            self._fail(f"secondary rows: expected 225, got {secondary_rows}")
        if final_rows == 225:
            self._pass(f"final rows: {final_rows}/225")
        else:
            self._fail(f"final rows: expected 225, got {final_rows}")

        # 6. Sequence counts: 36
        print("\n[V6] Validation sequence counts...")
        primary_seqs = _count_jsonl(_VAL_DIR / "primary_sequence_annotations.jsonl")
        secondary_seqs = _count_jsonl(_VAL_DIR / "secondary_sequence_annotations.jsonl")
        final_seqs = _count_jsonl(_VAL_DIR / "final_sequence_labels.jsonl")
        if primary_seqs == 36:
            self._pass(f"primary sequences: {primary_seqs}/36")
        else:
            self._fail(f"primary sequences: expected 36, got {primary_seqs}")
        if secondary_seqs == 36:
            self._pass(f"secondary sequences: {secondary_seqs}/36")
        else:
            self._fail(f"secondary sequences: expected 36, got {secondary_seqs}")
        if final_seqs == 36:
            self._pass(f"final sequences: {final_seqs}/36")
        else:
            self._fail(f"final sequences: expected 36, got {final_seqs}")

        # 6b. Sequence annotation ID uniqueness and coverage (§40)
        print("\n[V6b] Sequence annotation ID uniqueness and coverage...")
        import json as _json

        def _load_jsonl_ids(path):
            ids = []
            if path.exists():
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            rec = _json.loads(line)
                            ids.append(rec.get("sequence_annotation_id", ""))
            return ids

        p_seq_ids = _load_jsonl_ids(_VAL_DIR / "primary_sequence_annotations.jsonl")
        s_seq_ids = _load_jsonl_ids(_VAL_DIR / "secondary_sequence_annotations.jsonl")
        f_seq_ids = _load_jsonl_ids(_VAL_DIR / "final_sequence_labels.jsonl")

        p_unique = len(p_seq_ids) == len(set(p_seq_ids))
        s_unique = len(s_seq_ids) == len(set(s_seq_ids))
        f_unique = len(f_seq_ids) == len(set(f_seq_ids))

        if p_unique and len(p_seq_ids) == 36:
            self._pass("primary: 36 unique sequence_annotation_ids")
        else:
            self._fail(f"primary: {len(p_seq_ids)} ids, unique={p_unique}")
        if s_unique and len(s_seq_ids) == 36:
            self._pass("secondary: 36 unique sequence_annotation_ids")
        else:
            self._fail(f"secondary: {len(s_seq_ids)} ids, unique={s_unique}")
        if f_unique and len(f_seq_ids) == 36:
            self._pass("final: 36 unique sequence_annotation_ids")
        else:
            self._fail(f"final: {len(f_seq_ids)} ids, unique={f_unique}")

        # Cross-annotator coverage
        p_set = set(p_seq_ids)
        s_set = set(s_seq_ids)
        f_set = set(f_seq_ids)
        common = p_set & s_set
        if len(common) == 36 and len(p_set - s_set) == 0 and len(s_set - p_set) == 0:
            self._pass("36 common sequence_annotation_ids, 0 unmatched")
        else:
            self._fail(f"common={len(common)}, unmatched_p={len(p_set - s_set)}, unmatched_s={len(s_set - p_set)}")

        if f_set == p_set:
            self._pass("final sequence IDs match input IDs")
        else:
            self._fail("final sequence IDs do not match input IDs")

        # 7. Adjudication complete — exact ID-set coverage (§33)
        print("\n[V7] Validation adjudication complete (exact ID-set coverage)...")
        adj_path = _VAL_DIR / "adjudication_manifest.json"
        if adj_path.exists():
            adj = _load_json(adj_path)

            # Exact ID-set coverage: load raw data and compute keys
            review_path = _VAL_DIR / "review_queue.jsonl"
            adjudication_path = _VAL_DIR / "llm_adjudication.jsonl"

            if review_path.exists() and adjudication_path.exists():
                import json as _json

                def _load_jsonl_records(p):
                    recs = []
                    with open(p) as fh:
                        for line in fh:
                            if line.strip():
                                recs.append(_json.loads(line))
                    return recs

                def _review_key(rec):
                    if rec.get("item_type") == "row":
                        return ("row", rec["candidate_id"])
                    return ("sequence", rec.get("sequence_annotation_id", rec["candidate_id"]))

                def _adj_key(rec):
                    if rec.get("item_type") == "sequence":
                        return ("sequence", rec["sequence_annotation_id"])
                    return ("row", rec["candidate_id"])

                review_recs = _load_jsonl_records(review_path)
                adj_recs = _load_jsonl_records(adjudication_path)

                review_keys = [_review_key(r) for r in review_recs]
                adj_keys = [_adj_key(r) for r in adj_recs]

                unique_review = set(review_keys)
                unique_adj = set(adj_keys)
                dup_review = len(review_recs) - len(unique_review)
                dup_adj = len(adj_recs) - len(unique_adj)
                missing = unique_review - unique_adj
                unexpected = unique_adj - unique_review

                # Report counts
                if len(unique_review) == len(unique_adj) and len(missing) == 0 and len(unexpected) == 0:
                    self._pass(f"exact coverage: {len(unique_review)} review == {len(unique_adj)} adjudicated")
                else:
                    self._fail(
                        f"coverage mismatch: review={len(unique_review)}, "
                        f"adj={len(unique_adj)}, missing={len(missing)}, unexpected={len(unexpected)}"
                    )

                if dup_review == 0:
                    self._pass("duplicate_review_items: 0")
                else:
                    self._fail(f"duplicate_review_items: {dup_review}")

                if dup_adj == 0:
                    self._pass("duplicate_adjudications: 0")
                else:
                    self._fail(f"duplicate_adjudications: {dup_adj}")

                if len(missing) == 0:
                    self._pass("missing_adjudications: 0")
                else:
                    self._fail(f"missing_adjudications: {len(missing)}")

                if len(unexpected) == 0:
                    self._pass("unexpected_adjudications: 0")
                else:
                    self._fail(f"unexpected_adjudications: {len(unexpected)}")

                # Also check audit fields in manifest if present
                if adj.get("adjudication_complete") is True:
                    self._pass("adjudication_complete: true")
                elif "adjudication_complete" in adj:
                    self._fail("adjudication_complete: false")
            else:
                self._fail("review_queue.jsonl or llm_adjudication.jsonl not found")

            # Unresolved rate check
            rate = adj.get("unresolved_row_rate", 1.0)
            if rate <= 0.10:
                self._pass(f"unresolved_row_rate: {rate:.4f} (<=10%)")
            else:
                self._fail(f"unresolved_row_rate: {rate:.4f} (>10%)")
        else:
            self._fail("adjudication_manifest.json not found")

        # 8. Validation gate GO
        print("\n[V8] Validation gate...")
        gate_path = _VAL_DIR / "validation_annotation_gate.json"
        if gate_path.exists():
            gate = _load_json(gate_path)
            if gate.get("go_no_go") == "GO":
                self._pass("go_no_go: GO")
            else:
                self._fail(f"go_no_go: {gate.get('go_no_go')}")
        else:
            self._fail("validation_annotation_gate.json not found")

        # 9. Frozen corpus SHA
        print("\n[V9] Frozen corpus SHA...")
        if val_manifest_path.exists():
            vm = _load_json(val_manifest_path)
            corpus_path = _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation" / "frozen_corpus_manifest.json"
            if corpus_path.exists():
                actual = _sha256(corpus_path)
                expected = vm.get("frozen_corpus_manifest_sha256", "")
                if actual == expected:
                    self._pass("frozen corpus SHA matches")
                else:
                    self._fail("frozen corpus SHA MISMATCH")

        # 10. Phase status
        print("\n[V10] Annotation phase status...")
        if _PHASE_PATH.exists():
            phase = _load_json(_PHASE_PATH)
            if phase.get("validation_annotation_complete") is True:
                self._pass("validation_annotation_complete: true")
            else:
                self._fail(f"validation_annotation_complete: {phase.get('validation_annotation_complete')}")
            # Phase-aware check: after TEST_COMPLETE/ANNOTATIONS_FROZEN, test_annotation_complete should be true
            current_phase = phase.get("annotation_phase", "")
            if current_phase in ("TEST_COMPLETE", "FROZEN", "ANNOTATIONS_FROZEN"):
                if phase.get("test_annotation_complete") is True:
                    self._pass("test_annotation_complete: true (correct)")
                else:
                    self._fail("test_annotation_complete: false (should be true after TEST_COMPLETE)")
            else:
                if phase.get("test_annotation_complete") is not True:
                    self._pass("test_annotation_complete: false (correct)")
                else:
                    self._fail("test_annotation_complete: true (should be false)")
            # Phase-aware annotations_frozen check
            if current_phase == "ANNOTATIONS_FROZEN":
                if phase.get("annotations_frozen") is True:
                    self._pass("annotations_frozen: true (correct — globally frozen)")
                else:
                    self._fail(f"annotations_frozen: {phase.get('annotations_frozen')} (should be true after ANNOTATIONS_FROZEN)")
            else:
                if phase.get("annotations_frozen") is False:
                    self._pass("annotations_frozen: false (correct)")
                else:
                    self._fail(f"annotations_frozen: {phase.get('annotations_frozen')} (should be false)")

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

    def _verify_files_exist(self) -> None:
        """Check 1: All bound files exist."""
        print("\n[1] Bound file existence...")
        for key, fname in _DEV_FILES.items():
            fpath = _DEV_DIR / fname
            if fpath.exists():
                self._pass(f"{key}: {fname} exists")
            else:
                self._fail(f"{key}: {fname} MISSING")

    def _verify_file_hashes(self) -> None:
        """Check 2: All hashes match the development annotation manifest."""
        print("\n[2] File hash verification...")
        manifest_path = _DEV_DIR / "annotation_manifest.json"
        if not manifest_path.exists():
            self._fail("annotation_manifest.json not found — cannot verify hashes")
            return

        manifest = _load_json(manifest_path)

        for hash_key, file_key in _MANIFEST_HASH_MAP.items():
            expected_sha = manifest.get(hash_key, "")
            if not expected_sha:
                self._fail(f"{file_key}: no hash in manifest ({hash_key})")
                continue
            fname = _DEV_FILES[file_key]
            fpath = _DEV_DIR / fname
            if not fpath.exists():
                self._fail(f"{file_key}: cannot hash — file missing")
                continue
            actual_sha = _sha256(fpath)
            if actual_sha == expected_sha:
                self._pass(f"{file_key}: SHA256 matches")
            else:
                self._fail(f"{file_key}: SHA256 MISMATCH")

        # Also verify adjudication files against adjudication_manifest.json
        adj_manifest_path = _DEV_DIR / "adjudication_manifest.json"
        if adj_manifest_path.exists():
            adj_manifest = _load_json(adj_manifest_path)
            # Verify review_queue hash
            rq_sha = adj_manifest.get("review_queue_sha256", "")
            if rq_sha:
                actual_rq = _sha256(_DEV_DIR / "review_queue.jsonl")
                if actual_rq == rq_sha:
                    self._pass("adjudication review_queue SHA256 matches")
                else:
                    self._fail("adjudication review_queue SHA256 MISMATCH")
            # Verify llm_adjudication hash
            la_sha = adj_manifest.get("llm_adjudication_sha256", "")
            if la_sha:
                actual_la = _sha256(_DEV_DIR / "llm_adjudication.jsonl")
                if actual_la == la_sha:
                    self._pass("llm_adjudication SHA256 matches")
                else:
                    self._fail("llm_adjudication SHA256 MISMATCH")
            # Verify final_adjudicated_labels hash
            fl_sha = adj_manifest.get("final_adjudicated_labels_sha256", "")
            if fl_sha:
                actual_fl = _sha256(_DEV_DIR / "final_adjudicated_labels.jsonl")
                if actual_fl == fl_sha:
                    self._pass("final_adjudicated_labels SHA256 matches")
                else:
                    self._fail("final_adjudicated_labels SHA256 MISMATCH")
        else:
            self._fail("adjudication_manifest.json not found")

    def _verify_frozen_corpus_sha(self) -> None:
        """Check 3: Frozen corpus SHA matches."""
        print("\n[3] Frozen corpus SHA binding...")
        manifest_path = _DEV_DIR / "annotation_manifest.json"
        if not manifest_path.exists():
            self._fail("annotation_manifest.json not found")
            return
        manifest = _load_json(manifest_path)
        expected_fc_sha = manifest.get("frozen_corpus_manifest_sha256", "")
        if not expected_fc_sha:
            self._fail("No frozen_corpus_manifest_sha256 in annotation manifest")
            return

        fc_manifest_path = (
            _PROJECT_ROOT / "results" / "empirical_v2" / "corpus_generation"
            / "frozen_corpus_manifest.json"
        )
        if not fc_manifest_path.exists():
            self._fail("frozen_corpus_manifest.json not found")
            return
        actual_fc_sha = _sha256(fc_manifest_path)
        if actual_fc_sha == expected_fc_sha:
            self._pass("frozen_corpus_manifest SHA256 matches annotation manifest")
        else:
            self._fail("frozen_corpus_manifest SHA256 MISMATCH")

        # Also check against protocol manifest
        pm_path = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
        if pm_path.exists():
            pm = _load_json(pm_path)
            pm_fc_sha = pm.get("frozen_corpus_manifest_sha256", "")
            if pm_fc_sha == expected_fc_sha:
                self._pass("frozen_corpus_manifest SHA256 matches protocol manifest")
            else:
                self._fail("frozen_corpus_manifest SHA256 mismatch with protocol manifest")

    def _verify_row_counts(self) -> None:
        """Check 4: 225 rows accounted for."""
        print("\n[4] Row counts...")
        manifest_path = _DEV_DIR / "annotation_manifest.json"
        if not manifest_path.exists():
            self._fail("annotation_manifest.json not found")
            return
        manifest = _load_json(manifest_path)

        # Check primary rows
        primary_rows = _count_jsonl(_DEV_DIR / "row_annotations.jsonl")
        if primary_rows == 225:
            self._pass(f"primary rows: {primary_rows}/225")
        else:
            self._fail(f"primary rows: expected 225, got {primary_rows}")

        # Check secondary rows
        secondary_rows = _count_jsonl(_DEV_DIR / "secondary_row_annotations.jsonl")
        if secondary_rows == 225:
            self._pass(f"secondary rows: {secondary_rows}/225")
        else:
            self._fail(f"secondary rows: expected 225, got {secondary_rows}")

        # Check final adjudicated labels
        final_labels_path = _DEV_DIR / "final_adjudicated_labels.jsonl"
        if final_labels_path.exists():
            final_count = _count_jsonl(final_labels_path)
            if final_count == 225:
                self._pass(f"final adjudicated labels: {final_count}/225")
            else:
                self._fail(f"final adjudicated labels: expected 225, got {final_count}")
        else:
            self._fail("final_adjudicated_labels.jsonl not found")

        # Check manifest row counts
        row_count = manifest.get("row_count", {})
        if row_count.get("primary") == 225:
            self._pass("manifest row_count.primary == 225")
        else:
            self._fail(f"manifest row_count.primary: {row_count.get('primary')}")
        if row_count.get("secondary") == 225:
            self._pass("manifest row_count.secondary == 225")
        else:
            self._fail(f"manifest row_count.secondary: {row_count.get('secondary')}")
        if row_count.get("unmatched", 0) == 0:
            self._pass("manifest row_count.unmatched == 0")
        else:
            self._fail(f"manifest row_count.unmatched: {row_count.get('unmatched')}")

    def _verify_sequence_counts(self) -> None:
        """Check 5: 36 sequences accounted for."""
        print("\n[5] Sequence counts...")
        primary_seqs = _count_jsonl(_DEV_DIR / "sequence_annotations.jsonl")
        if primary_seqs == 36:
            self._pass(f"primary sequences: {primary_seqs}/36")
        else:
            self._fail(f"primary sequences: expected 36, got {primary_seqs}")

        secondary_seqs = _count_jsonl(_DEV_DIR / "secondary_sequence_annotations.jsonl")
        if secondary_seqs == 36:
            self._pass(f"secondary sequences: {secondary_seqs}/36")
        else:
            self._fail(f"secondary sequences: expected 36, got {secondary_seqs}")

    def _verify_adjudication_complete(self) -> None:
        """Check 6: Adjudication complete."""
        print("\n[6] Adjudication completeness...")
        adj_manifest_path = _DEV_DIR / "adjudication_manifest.json"
        if not adj_manifest_path.exists():
            self._fail("adjudication_manifest.json not found")
            return

        adj = _load_json(adj_manifest_path)

        # Review queue fully adjudicated
        rq_count = adj.get("review_queue_count", 0)
        adj_count = adj.get("adjudicated_count", 0)
        missing = adj.get("missing_adjudications", -1)
        duplicates = adj.get("duplicate_adjudications", -1)

        if rq_count == 38:
            self._pass(f"review_queue_count: {rq_count}")
        else:
            self._fail(f"review_queue_count: expected 38, got {rq_count}")

        if adj_count == 38:
            self._pass(f"adjudicated_count: {adj_count}")
        else:
            self._fail(f"adjudicated_count: expected 38, got {adj_count}")

        if missing == 0:
            self._pass("missing_adjudications: 0")
        else:
            self._fail(f"missing_adjudications: {missing}")

        if duplicates == 0:
            self._pass("duplicate_adjudications: 0")
        else:
            self._fail(f"duplicate_adjudications: {duplicates}")

        # Check unresolved rate
        rate = adj.get("unresolved_row_rate", 1.0)
        if rate <= 0.10:
            self._pass(f"unresolved_row_rate: {rate:.4f} (<=10%)")
        else:
            self._fail(f"unresolved_row_rate: {rate:.4f} (>10%)")

    def _verify_gate_go(self) -> None:
        """Check 7: Development gate GO."""
        print("\n[7] Development annotation gate...")
        gate_path = _DEV_DIR / "development_annotation_gate.json"
        if not gate_path.exists():
            self._fail("development_annotation_gate.json not found")
            return

        gate = _load_json(gate_path)

        if gate.get("go_no_go") == "GO":
            self._pass("go_no_go: GO")
        else:
            self._fail(f"go_no_go: {gate.get('go_no_go')}")

        if gate.get("protocol_freeze_pass") is True:
            self._pass("protocol_freeze_pass: true")
        else:
            self._fail(f"protocol_freeze_pass: {gate.get('protocol_freeze_pass')}")

        if gate.get("ready_for_validation_annotation") is True:
            self._pass("ready_for_validation_annotation: true")
        else:
            self._fail(f"ready_for_validation_annotation: {gate.get('ready_for_validation_annotation')}")

        blocking = gate.get("blocking_findings", [])
        if len(blocking) == 0:
            self._pass("blocking_findings: none")
        else:
            self._fail(f"blocking_findings: {len(blocking)} findings")

    def _verify_protocol_frozen(self) -> None:
        """Check 8: Protocol manifest frozen."""
        print("\n[8] Protocol manifest freeze status...")
        pm_path = _ANNOTATIONS_DIR / "annotation_protocol_manifest.json"
        if not pm_path.exists():
            self._fail("annotation_protocol_manifest.json not found")
            return

        pm = _load_json(pm_path)

        if pm.get("annotation_schema_frozen") is True:
            self._pass("annotation_schema_frozen: true")
        else:
            self._fail(f"annotation_schema_frozen: {pm.get('annotation_schema_frozen')}")

        if pm.get("annotation_prompts_frozen") is True:
            self._pass("annotation_prompts_frozen: true")
        else:
            self._fail(f"annotation_prompts_frozen: {pm.get('annotation_prompts_frozen')}")

        # Phase-aware annotations_frozen check (Sec 31)
        phase = _load_json(_PHASE_PATH) if _PHASE_PATH.exists() else {}
        current_phase = phase.get("annotation_phase", "")
        if current_phase == "ANNOTATIONS_FROZEN":
            if pm.get("annotations_frozen") is True:
                self._pass("annotations_frozen: true (correct — globally frozen)")
            else:
                self._fail(f"annotations_frozen: {pm.get('annotations_frozen')} (should be true after ANNOTATIONS_FROZEN)")
        else:
            if pm.get("annotations_frozen") is False:
                self._pass("annotations_frozen: false (correct — validation/test pending)")
            else:
                self._fail(f"annotations_frozen: {pm.get('annotations_frozen')} (should be false)")

    def _verify_phase_status(self) -> None:
        """Check 9: Annotation phase status."""
        print("\n[9] Annotation phase status...")
        if not _PHASE_PATH.exists():
            self._fail("annotation_phase.json not found")
            return

        phase = _load_json(_PHASE_PATH)

        if phase.get("development_annotation_complete") is True:
            self._pass("development_annotation_complete: true")
        else:
            self._fail(f"development_annotation_complete: {phase.get('development_annotation_complete')}")

        if phase.get("annotation_schema_frozen") is True:
            self._pass("phase annotation_schema_frozen: true")
        else:
            self._fail(f"phase annotation_schema_frozen: {phase.get('annotation_schema_frozen')}")

        if phase.get("annotation_prompts_frozen") is True:
            self._pass("phase annotation_prompts_frozen: true")
        else:
            self._fail(f"phase annotation_prompts_frozen: {phase.get('annotation_prompts_frozen')}")

        # validation should be complete (per §47)
        if phase.get("validation_annotation_complete") is True:
            self._pass("validation_annotation_complete: true (correct)")
        else:
            self._fail("validation_annotation_complete: false (should be true)")

        # Phase-aware check: after TEST_COMPLETE/ANNOTATIONS_FROZEN, test_annotation_complete should be true
        current_phase = phase.get("annotation_phase", "")
        if current_phase in ("TEST_COMPLETE", "FROZEN", "ANNOTATIONS_FROZEN"):
            if phase.get("test_annotation_complete") is True:
                self._pass("test_annotation_complete: true (correct)")
            else:
                self._fail("test_annotation_complete: false (should be true after TEST_COMPLETE)")
        else:
            if phase.get("test_annotation_complete") is not True:
                self._pass("test_annotation_complete: false (correct)")
            else:
                self._fail("test_annotation_complete: true (should be false)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen annotation verifier")
    parser.add_argument(
        "--split",
        default="development",
        help="Annotation split to verify (default: development)",
    )
    args = parser.parse_args()

    verifier = FrozenAnnotationVerifier(split=args.split)
    passed = verifier.verify_all()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

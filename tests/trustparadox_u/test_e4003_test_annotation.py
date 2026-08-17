"""E4-003: Test annotation fixture tests.

Covers checklist Sec 151:
- queue counts
- identities
- blinding
- test preflight
- primary/secondary independence
- sequence adjudication
- test gate
- freeze inventory
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.empirical_annotation import (
    build_test_queue,
    load_test_candidates,
    test_input_preflight,
    sequence_labels_match,
    row_labels_match,
    CORE_BINARY_LABELS,
)

_ANNOTATIONS_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations"
_TEST_DIR = _ANNOTATIONS_DIR / "test"
_TEST_CANDIDATES_PATH = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "splits" / "test.jsonl"
)

pytestmark = pytest.mark.skipif(
    not _TEST_CANDIDATES_PATH.exists(),
    reason="Test candidates not found; run test annotation protocol first",
)


# ===========================================================================
# Queue counts
# ===========================================================================


class TestTestQueueCounts:
    """Sec 151: Test queue must have correct counts."""

    def test_load_test_candidates_count(self):
        """Test candidates must be 450 rows."""
        candidates = load_test_candidates()
        assert len(candidates) == 450, f"Expected 450 test candidates, got {len(candidates)}"

    def test_build_test_queue_row_count(self):
        """Test row queue must be 450 items."""
        row_items, seq_items = build_test_queue()
        assert len(row_items) == 450, f"Expected 450 row items, got {len(row_items)}"

    def test_build_test_queue_sequence_count(self):
        """Test sequence queue must be 72 trust-conditioned units."""
        row_items, seq_items = build_test_queue()
        assert len(seq_items) == 72, f"Expected 72 sequence items, got {len(seq_items)}"

    def test_structural_families_count(self):
        """Test must have 24 structural sequence families."""
        row_items, seq_items = build_test_queue()
        families = set()
        for item in seq_items:
            fam_id = item.get("sequence_family_id")
            if fam_id:
                families.add(fam_id)
        assert len(families) == 24, f"Expected 24 structural families, got {len(families)}"

    def test_trust_variant_coverage(self):
        """Each family must have default/high/low trust variants."""
        row_items, seq_items = build_test_queue()
        family_trust = {}
        for item in seq_items:
            fam_id = item.get("sequence_family_id")
            trust = item.get("trust_level")
            if fam_id and trust:
                family_trust.setdefault(fam_id, set()).add(trust)
        for fam_id, trust_levels in family_trust.items():
            assert "default" in trust_levels, f"Family {fam_id} missing 'default' trust"
            assert "high" in trust_levels, f"Family {fam_id} missing 'high' trust"
            assert "low" in trust_levels, f"Family {fam_id} missing 'low' trust"


# ===========================================================================
# Identities
# ===========================================================================


class TestTestQueueIdentity:
    """Sec 151: Test queue must be deterministic."""

    def test_queue_sha_deterministic(self):
        """Queue SHA must be stable across calls."""
        from experiments.trustparadox_u.empirical_annotation import compute_queue_sha256
        row1, seq1 = build_test_queue()
        row2, seq2 = build_test_queue()
        sha1 = compute_queue_sha256(row1, seq1)
        sha2 = compute_queue_sha256(row2, seq2)
        assert sha1 == sha2, "Queue SHA not deterministic"

    def test_campaign_identity_fields(self):
        """Campaign identity must include required fields."""
        from experiments.trustparadox_u.empirical_annotation import (
            build_campaign_identity,
            compute_queue_sha256,
            build_prompt_manifest,
            build_annotation_config,
        )
        import hashlib
        row_items, seq_items = build_test_queue()
        queue_sha = compute_queue_sha256(row_items, seq_items)
        # Compute config and prompt manifest SHAs
        config = build_annotation_config()
        config_sha = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        prompt_manifest = build_prompt_manifest()
        prompt_sha = hashlib.sha256(
            json.dumps(prompt_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        identity = build_campaign_identity(
            queue_sha256=queue_sha,
            annotation_config_sha256=config_sha,
            prompt_manifest_sha256=prompt_sha,
            annotation_code_commit="test_commit",
            split="test",
            primary_model="test_model",
            secondary_model="test_model2",
        )
        assert "annotation_queue_sha256" in identity
        assert "split" in identity
        assert identity["split"] == "test"
        assert "annotation_code_commit" in identity


# ===========================================================================
# Blinding
# ===========================================================================


class TestTestBlinding:
    """Sec 151/90: Test prompts must not leak forbidden fields."""

    def test_row_prompt_blinding(self):
        """Row annotation prompt must not contain forbidden fields."""
        from experiments.trustparadox_u.empirical_annotation import build_row_prompt, load_test_candidates
        candidates = load_test_candidates()
        if not candidates:
            pytest.skip("No candidates available")
        item = candidates[0]
        prompt_text, _ = build_row_prompt(item)
        forbidden = ["attack_type", "trust_level", "split", "generator_model",
                     "acceptance_decision", "firewall_output", "detector_result"]
        for field in forbidden:
            assert field not in prompt_text.lower(), f"Row prompt contains forbidden field: {field}"

    def test_sequence_prompt_blinding(self):
        """Sequence annotation prompt must not contain forbidden fields."""
        from experiments.trustparadox_u.empirical_annotation import build_sequence_prompt, load_test_candidates
        from collections import defaultdict
        candidates = load_test_candidates()
        if not candidates:
            pytest.skip("No candidates available")
        # Group candidates by sequence_id and get first complete sequence
        by_seq = defaultdict(list)
        for c in candidates:
            by_seq[c.get("sequence_id")].append(c)
        # Get first sequence with at least 2 members and valid sequence_step_index
        members = None
        for seq_id, seq_candidates in by_seq.items():
            if len(seq_candidates) >= 2:
                # Check if sequence_step_index is set
                if all(c.get("sequence_step_index") is not None for c in seq_candidates):
                    members = sorted(seq_candidates, key=lambda c: c.get("sequence_step_index"))
                    break
        if not members:
            pytest.skip("No complete sequence with valid sequence_step_index found")
        prompt_text, _ = build_sequence_prompt(members)
        forbidden = ["attack_type", "trust_level", "split", "generator_model",
                     "acceptance_decision", "firewall_output", "detector_result"]
        for field in forbidden:
            assert field not in prompt_text.lower(), f"Sequence prompt contains forbidden field: {field}"


# ===========================================================================
# Test preflight
# ===========================================================================


class TestTestInputPreflight:
    """Sec 151/51: Test preflight must pass before provider calls."""

    def test_preflight_returns_dict(self):
        """Preflight must return a dict."""
        result = test_input_preflight()
        assert isinstance(result, dict)

    def test_preflight_has_status(self):
        """Preflight must include passed field."""
        result = test_input_preflight()
        assert "passed" in result
        assert isinstance(result["passed"], bool)

    def test_preflight_has_findings(self):
        """Preflight must include findings list."""
        result = test_input_preflight()
        assert "findings" in result
        assert isinstance(result["findings"], list)


# ===========================================================================
# Primary/secondary independence
# ===========================================================================


class TestPrimarySecondaryIndependence:
    """Sec 151/89: Secondary must not read primary labels before annotation."""

    def test_secondary_runner_does_not_open_primary_labels(self):
        """run_test_secondary.py must not open primary label files."""
        runner_path = _PROJECT_ROOT / "scripts" / "run_test_secondary.py"
        if not runner_path.exists():
            pytest.skip("run_test_secondary.py not found")
        code = runner_path.read_text(encoding="utf-8")
        # Check that it doesn't directly open primary label files
        assert "test_primary_labels.jsonl" not in code, "Secondary runner references primary labels"
        assert "primary_labels.jsonl" not in code, "Secondary runner references primary labels"


# ===========================================================================
# Sequence adjudication
# ===========================================================================


class TestSequenceAdjudication:
    """Sec 151: Sequence adjudication logic."""

    def test_sequence_labels_match_semantic_tuple(self):
        """sequence_labels_match must check semantic tuple equality."""
        a = {
            "sequence_reconstructs_target": True,
            "earliest_reconstruction_step": 2,
            "reconstruction_strength": 0.8,
        }
        b = {
            "sequence_reconstructs_target": True,
            "earliest_reconstruction_step": 2,
            "reconstruction_strength": 0.8,
        }
        assert sequence_labels_match(a, b) is True

    def test_sequence_labels_mismatch(self):
        """sequence_labels_match must detect mismatches."""
        a = {
            "sequence_reconstructs_target": True,
            "earliest_reconstruction_step": 2,
            "reconstruction_strength": 0.8,
        }
        b = {
            "sequence_reconstructs_target": False,
            "earliest_reconstruction_step": 2,
            "reconstruction_strength": 0.8,
        }
        assert sequence_labels_match(a, b) is False

    def test_row_labels_match_core_binary(self):
        """row_labels_match must check all core binary labels."""
        a = {fld: True for fld in CORE_BINARY_LABELS}
        a["leakage_strength"] = 0.5
        b = {fld: True for fld in CORE_BINARY_LABELS}
        b["leakage_strength"] = 0.5
        assert row_labels_match(a, b) is True

    def test_row_labels_mismatch(self):
        """row_labels_match must detect mismatches."""
        a = {fld: True for fld in CORE_BINARY_LABELS}
        a["leakage_strength"] = 0.5
        b = {fld: True for fld in CORE_BINARY_LABELS}
        b["target_relevant"] = False
        b["leakage_strength"] = 0.5
        assert row_labels_match(a, b) is False


# ===========================================================================
# Test gate
# ===========================================================================


class TestTestGate:
    """Sec 151: Test gate must exist and be GO after freeze."""

    def test_gate_exists(self):
        """Test gate must exist."""
        gate_path = _TEST_DIR / "test_annotation_gate.json"
        if not gate_path.exists():
            pytest.skip("Test gate not found; run test freeze first")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        assert "go_no_go" in gate

    def test_gate_is_go(self):
        """Test gate must be GO after freeze."""
        gate_path = _TEST_DIR / "test_annotation_gate.json"
        if not gate_path.exists():
            pytest.skip("Test gate not found; run test freeze first")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        assert gate.get("go_no_go") == "GO"


# ===========================================================================
# Freeze inventory
# ===========================================================================


class TestTestFreezeInventory:
    """Sec 151: Test freeze must produce required artifacts."""

    def test_freeze_manifest_exists(self):
        """Test freeze manifest must exist."""
        manifest_path = _TEST_DIR / "test_annotation_freeze_manifest.json"
        if not manifest_path.exists():
            pytest.skip("Test freeze manifest not found; run test freeze first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "schema_version" in manifest
        assert "test_annotation_complete" in manifest

    def test_post_freeze_verification_exists(self):
        """Test post-freeze verification must exist."""
        pfv_path = _TEST_DIR / "test_post_freeze_verification.json"
        if not pfv_path.exists():
            pytest.skip("Test post-freeze verification not found; run test freeze first")
        pfv = json.loads(pfv_path.read_text(encoding="utf-8"))
        assert "schema_version" in pfv
        assert "all_verifiers_pass" in pfv

    def test_final_labels_exist(self):
        """Test final adjudicated labels must exist."""
        labels_path = _TEST_DIR / "test_final_adjudicated_labels.jsonl"
        if not labels_path.exists():
            pytest.skip("Test final labels not found; run test freeze first")
        count = sum(1 for line in open(labels_path) if line.strip())
        assert count == 450, f"Expected 450 final labels, got {count}"

    def test_final_sequences_exist(self):
        """Test final sequence labels must exist."""
        seq_path = _TEST_DIR / "test_final_sequence_labels.jsonl"
        if not seq_path.exists():
            pytest.skip("Test final sequences not found; run test freeze first")
        count = sum(1 for line in open(seq_path) if line.strip())
        assert count == 72, f"Expected 72 final sequences, got {count}"

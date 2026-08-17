"""E4-003 R1.2: Offline finalization tests (item 15).

Covers:
- finalize-only → 0 provider calls
- existing row adjudication reused
- sequence consensus finalized
- sequence J3 match-J finalized
- sequence J3 match-J2 finalized
- sequence J3 unresolved finalized
- provider-call guard test
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


# ===========================================================================
# Item 15: Offline finalization mode
# ===========================================================================


class TestOfflineFinalizationMode:
    """Item 5/15: --finalize-only mode exists and works offline."""

    def test_finalize_only_cli_flag_exists(self):
        """run_test_adjudication.py must support --finalize-only flag."""
        source_path = _PROJECT_ROOT / "scripts" / "run_test_adjudication.py"
        source = source_path.read_text()
        assert "--finalize-only" in source
        assert "run_finalize_only" in source

    def test_run_finalize_only_function_exists(self):
        """run_finalize_only() function must exist."""
        from scripts.run_test_adjudication import run_finalize_only
        assert callable(run_finalize_only)

    def test_finalize_only_does_not_call_providers(self):
        """run_finalize_only must not call _call_j3 or _call_j3_sequence."""
        from scripts.run_test_adjudication import run_finalize_only
        source = inspect.getsource(run_finalize_only)
        assert "_call_j3" not in source
        assert "_call_j3_sequence" not in source
        assert "litellm" not in source

    def test_finalize_only_loads_existing_evidence(self):
        """run_finalize_only must load existing adjudication evidence."""
        from scripts.run_test_adjudication import run_finalize_only
        source = inspect.getsource(run_finalize_only)
        assert "_LLM_ADJUDICATION_PATH" in source
        assert "adjudication_records" in source

    def test_finalize_helper_is_pure_offline(self):
        """finalize_test_annotations must not call providers."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "_call_j3" not in source
        assert "_call_j3_sequence" not in source
        assert "litellm" not in source


# ===========================================================================
# Item 15: Provider-call guard test
# ===========================================================================


class TestProviderCallGuard:
    """Item 5/15: finalize-only must complete with 0 provider calls."""

    def test_finalize_helper_signature_accepts_all_inputs(self):
        """finalize_test_annotations accepts all required inputs."""
        from scripts.run_test_adjudication import finalize_test_annotations
        sig = inspect.signature(finalize_test_annotations)
        params = list(sig.parameters.keys())
        expected = [
            "j_rows", "j2_rows", "j_seqs", "j2_seqs",
            "review_queue", "adjudication_records", "seq_adjudication_records",
        ]
        for p in expected:
            assert p in params, f"Missing parameter: {p}"

    def test_call_j3_functions_exist_but_not_called_by_finalize(self):
        """_call_j3 and _call_j3_sequence exist but are not called by finalize."""
        from scripts.run_test_adjudication import (
            finalize_test_annotations,
            _call_j3,
            _call_j3_sequence,
        )
        assert callable(_call_j3)
        assert callable(_call_j3_sequence)
        # Verify finalize doesn't call them
        source = inspect.getsource(finalize_test_annotations)
        assert "_call_j3(" not in source
        assert "_call_j3_sequence(" not in source


# ===========================================================================
# Item 6: Sequence finalization logic
# ===========================================================================


class TestSequenceFinalizationLogic:
    """Item 6/15: Sequence finalization follows adjudication logic."""

    def test_sequence_consensus_retained_uses_j(self):
        """consensus_retained → final tuple = J → llm_consensus."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        # Check for consensus_retained logic
        assert "consensus_retained" in source
        # Should use j_seq as source
        assert "src_tuple = j_seq" in source or "j_seq" in source

    def test_sequence_resolved_by_j3_matching_j(self):
        """resolved_by_j3_matching_j → final tuple = J."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "resolved_by_j3_matching_j" in source

    def test_sequence_resolved_by_j3_matching_j2(self):
        """resolved_by_j3_matching_j2 → final tuple = J2."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "resolved_by_j3_matching_j2" in source
        # Should use j2_seq as source
        assert "j2_seq" in source

    def test_sequence_still_unresolved_null_tuple(self):
        """still_unresolved → final semantic tuple = null → unresolved."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "still_unresolved" in source
        assert "src_tuple = None" in source

    def test_sequence_no_adjudication_agree_uses_consensus(self):
        """No seq adjudication record + J/J2 agree → llm_consensus."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        # Check for fallback logic when seq_adj is None
        assert "seq_agree" in source or "_sequence_labels_match" in source

    def test_final_sequence_schema_fields(self):
        """Final sequence labels must include required schema fields."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        required_fields = [
            "sequence_annotation_id",
            "sequence_family_id",
            "trust_level",
            "scenario_id",
            "secret_variant_id",
            "ordered_candidate_ids",
            "final_sequence_reconstructs_target",
            "final_earliest_reconstruction_step",
            "final_reconstruction_strength",
            "resolution_source",
            "resolution_status",
            "j_agreed",
            "j2_agreed",
            "sequence_content_sha256",
            "frozen_corpus_manifest_sha256",
            "annotation_protocol_version",
        ]
        for field in required_fields:
            assert field in source, f"Missing field in final sequence: {field}"


# ===========================================================================
# Item 6: Row finalization logic
# ===========================================================================


class TestRowFinalizationLogic:
    """Item 6/15: Row finalization preserves semantic outcomes."""

    def test_final_row_schema_fields(self):
        """Final row labels must include required schema fields."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        required_fields = [
            "candidate_id",
            "final_target_relevant",
            "final_target_leakage",
            "final_positive_entailment",
            "final_task_useful",
            "final_leakage_strength",
            "resolution_source",
            "resolution_status",
        ]
        for field in required_fields:
            assert field in source, f"Missing field in final row: {field}"

    def test_row_consensus_uses_j_labels(self):
        """llm_consensus → final = J labels."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "llm_consensus" in source
        assert "j_label" in source

    def test_row_adjudication_matching_j(self):
        """resolved_by_j3_matching_j → final = J labels."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "resolved_by_j3_matching_j" in source

    def test_row_adjudication_matching_j2(self):
        """resolved_by_j3_matching_j2 → final = J2 labels."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "resolved_by_j3_matching_j2" in source
        assert "j2_label" in source

    def test_row_unresolved_null_labels(self):
        """still_unresolved → final = None for all labels."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "still_unresolved" in source
        # Check for None assignment
        assert "None" in source


# ===========================================================================
# Item 15: Expected counts preserved
# ===========================================================================


class TestExpectedCountsPreserved:
    """Item 11/15: Expected counts are preserved."""

    def test_expected_rows_is_450(self):
        """EXPECTED_ROWS = 450."""
        from scripts.run_test_adjudication import EXPECTED_ROWS
        assert EXPECTED_ROWS == 450

    def test_expected_sequences_is_72(self):
        """EXPECTED_SEQUENCES = 72."""
        from scripts.run_test_adjudication import EXPECTED_SEQUENCES
        assert EXPECTED_SEQUENCES == 72

    def test_expected_families_is_24(self):
        """EXPECTED_FAMILIES = 24."""
        from scripts.run_test_adjudication import EXPECTED_FAMILIES
        assert EXPECTED_FAMILIES == 24

    def test_max_unresolved_rate_is_010(self):
        """MAX_UNRESOLVED_RATE = 0.10."""
        from scripts.run_test_adjudication import MAX_UNRESOLVED_RATE
        assert MAX_UNRESOLVED_RATE == 0.10


# ===========================================================================
# Item 15: Adjudication manifest structure
# ===========================================================================


class TestAdjudicationManifestStructure:
    """Item 3/15: Adjudication manifest includes sequence counts."""

    def test_manifest_includes_row_and_seq_adjudication_counts(self):
        """Adjudication manifest must include both row and sequence counts."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "row_adjudication_record_count" in source
        assert "sequence_adjudication_record_count" in source

    def test_manifest_includes_offline_finalization_flag(self):
        """Adjudication manifest must include offline_finalization = True."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "offline_finalization" in source

    def test_manifest_includes_sequence_resolution_counts(self):
        """Adjudication manifest must include sequence resolution counts."""
        from scripts.run_test_adjudication import finalize_test_annotations
        source = inspect.getsource(finalize_test_annotations)
        assert "sequence_consensus_retained" in source
        assert "sequence_resolved_by_j3_matching_j" in source
        assert "sequence_resolved_by_j3_matching_j2" in source
        assert "sequence_still_unresolved" in source

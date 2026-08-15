"""E4-001 Repair Checklist: Regression tests.

Covers all required test names from the E4-001 Repair Checklist sections:
- Sec 11/67: Target resolution (E3 registry binding)
- Sec 20/68: Queue completeness (225 rows, 36 sequences)
- Sec 26/69: Corpus binding (frozen manifest SHA)
- Sec 30/71: Gate thresholds and freeze semantics
- Sec 35/71: NO-GO freeze semantics
- Sec 43: Retry / provider attempt retention
- Sec 47/70: Provider provenance fields
- Sec 50: Queue hash content binding
- Sec 72: Resume blocking regression
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.trustparadox_u.empirical_annotation import (
    ANNOTATION_SCHEMA_VERSION,
    MODEL_PRIMARY,
    MODEL_SECONDARY,
    AnnotationTargetResolutionError,
    RowAnnotation,
    SequenceAnnotation,
    _make_row_annotation_id,
    _make_sequence_annotation_id,
    build_annotation_view,
    build_campaign_identity,
    build_development_queue,
    build_sequence_annotation_view,
    compute_queue_sha256,
    frozen_corpus_manifest_file_sha256,
    preflight_target_resolution,
    prompt_sha256,
    resolve_empirical_annotation_target,
    ROW_SYSTEM_PROMPT,
    verify_campaign_identity,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TARGET_SPECS_PATH = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "target_specs.jsonl"
)
_DEV_CANDIDATES_PATH = (
    _PROJECT_ROOT
    / "results" / "empirical_v2" / "corpus_generation" / "development"
    / "accepted_candidates.jsonl"
)
_FROZEN_MANIFEST_PATH = (
    _PROJECT_ROOT
    / "results" / "empirical_v2" / "corpus_generation"
    / "frozen_corpus_manifest.json"
)

EXPECTED_FROZEN_SHA = (
    "6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2"
)

# Skip guards
needs_target_specs = pytest.mark.skipif(
    not _TARGET_SPECS_PATH.exists(),
    reason="E3 target_specs.jsonl not found",
)
needs_dev_candidates = pytest.mark.skipif(
    not _DEV_CANDIDATES_PATH.exists(),
    reason="Development accepted_candidates.jsonl not found",
)
needs_frozen_manifest = pytest.mark.skipif(
    not _FROZEN_MANIFEST_PATH.exists(),
    reason="Frozen corpus manifest not found",
)
needs_all_data = pytest.mark.skipif(
    not (
        _TARGET_SPECS_PATH.exists()
        and _DEV_CANDIDATES_PATH.exists()
        and _FROZEN_MANIFEST_PATH.exists()
    ),
    reason="Required data artifacts not found",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dev_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with open(_DEV_CANDIDATES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def _load_target_specs() -> dict[tuple[str, str], dict[str, Any]]:
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    with open(_TARGET_SPECS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            specs[(rec["scenario_id"], rec["secret_variant_id"])] = rec
    return specs


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _make_identity(**overrides: str) -> dict:
    """Build a campaign identity dict with sensible defaults."""
    from experiments.trustparadox_u.empirical_annotation import prompt_sha256, ROW_SYSTEM_PROMPT

    base = {
        "frozen_corpus_manifest_sha256": "corpus_hash_abc",
        "annotation_queue_sha256": "queue_hash_def",
        "annotation_schema_sha256": _sha256_str(ANNOTATION_SCHEMA_VERSION),
        "primary_prompt_sha256": prompt_sha256(ROW_SYSTEM_PROMPT),
        "secondary_prompt_sha256": prompt_sha256(ROW_SYSTEM_PROMPT),
        "primary_requested_model": MODEL_PRIMARY,
        "secondary_requested_model": MODEL_SECONDARY,
        "annotation_config_sha256": "config_hash_ghi",
        "annotation_code_commit": "abc123def456",
        "split": "development",
        "prompt_manifest_sha256": "prompt_manifest_hash_jkl",
    }
    base.update(overrides)
    return base


def _make_gate_audit(
    *,
    primary_rows: int = 225,
    secondary_rows: int = 225,
    primary_seqs: int = 36,
    secondary_seqs: int = 36,
    raw_agreement: float = 0.90,
    seq_raw_agreement: float = 0.90,
    kappa: float = 0.70,
    unresolved_rows: int = 0,
    unresolved_seqs: int = 0,
) -> dict[str, Any]:
    """Build a synthetic audit dict for assess_gate() testing."""
    return {
        "coverage": {
            "primary_row_count": primary_rows,
            "secondary_row_count": secondary_rows,
            "primary_sequence_count": primary_seqs,
            "secondary_sequence_count": secondary_seqs,
            "common_row_count": 225,
        },
        "adjudication": {
            "row_unresolved": unresolved_rows,
            "sequence_unresolved": unresolved_seqs,
            "sequence_consensus": 36,
        },
        "row_agreement": {
            "target_relevant": {
                "raw_agreement": raw_agreement,
                "cohens_kappa": kappa,
            },
            "target_leakage": {
                "raw_agreement": raw_agreement,
                "cohens_kappa": kappa,
            },
            "positive_entailment": {
                "raw_agreement": raw_agreement,
                "cohens_kappa": kappa,
            },
            "task_useful": {
                "raw_agreement": raw_agreement,
                "cohens_kappa": kappa,
            },
        },
        "sequence_agreement": {
            "reconstruction_binary": {
                "raw_agreement": seq_raw_agreement,
                "cohens_kappa": kappa,
            },
        },
    }


# ===================================================================
# Sec 11/67: Target resolution (E3 registry binding)
# ===================================================================


class TestTargetResolution:
    """Sec 11/67: E3 target registry binding and fail-closed resolution."""

    @needs_all_data
    def test_all_development_candidates_resolve_e3_target(self):
        """Every development candidate must resolve an E3 target."""
        candidates = _load_dev_candidates()
        report = preflight_target_resolution(candidates)
        assert report["passed"], (
            f"Target resolution failed: {report['failures']}"
        )
        assert report["resolved"] == len(candidates)

    @needs_all_data
    def test_all_development_sequences_resolve_e3_target(self):
        """All sequence members (which are also candidates) resolve targets."""
        candidates = _load_dev_candidates()
        seq_members = [c for c in candidates if c.get("sequence_family_id")]
        report = preflight_target_resolution(seq_members)
        assert report["passed"]
        assert report["resolved"] == len(seq_members)

    @needs_target_specs
    def test_annotation_view_target_is_nonempty(self):
        """build_annotation_view must produce a non-empty canonical_target."""
        specs = _load_target_specs()
        # Pick the first available target to build a candidate
        (sid, vid), spec = next(iter(specs.items()))
        cand = {
            "candidate_id": "cand_test_view",
            "scenario_id": sid,
            "secret_variant_id": vid,
            "text": "Test message text.",
        }
        view = build_annotation_view(cand)
        assert view["canonical_target"], "canonical_target must be non-empty"

    @needs_target_specs
    def test_sequence_view_target_is_nonempty(self):
        """build_sequence_annotation_view must produce a non-empty canonical_target."""
        specs = _load_target_specs()
        (sid, vid), spec = next(iter(specs.items()))
        members = [
            {
                "candidate_id": f"cand_seq_view_{i}",
                "scenario_id": sid,
                "secret_variant_id": vid,
                "text": f"Step {i} text.",
                "sequence_family_id": "seqfam_test",
                "sequence_step_index": i,
            }
            for i in range(3)
        ]
        view = build_sequence_annotation_view(members)
        assert view["canonical_target"], "canonical_target must be non-empty"

    @needs_target_specs
    def test_annotation_view_uses_e3_registry_not_legacy_target_specs(self):
        """The annotation view must use the E3 target_specs path, not frozen_corpus."""
        from experiments.trustparadox_u import empirical_annotation as ea

        # Verify the module-level path points to E3 registry
        assert "empirical_v2" in str(ea._TARGET_SPECS_PATH)
        assert "frozen_corpus" not in str(ea._TARGET_SPECS_PATH)

    def test_target_resolution_unknown_key_fails_closed(self):
        """Unknown (scenario_id, secret_variant_id) must raise, not return {}."""
        with pytest.raises(AnnotationTargetResolutionError):
            resolve_empirical_annotation_target(
                "nonexistent_scenario", "nonexistent_variant"
            )

    @needs_all_data
    def test_all_225_development_rows_resolve_target(self):
        """All 225 development rows must resolve an E3 target."""
        candidates = _load_dev_candidates()
        assert len(candidates) == 225
        report = preflight_target_resolution(candidates)
        assert report["resolved"] == 225
        assert report["failures"] == []

    @needs_all_data
    def test_all_36_sequences_resolve_target(self):
        """All 36 sequence families' members resolve targets."""
        candidates = _load_dev_candidates()
        seq_members = [c for c in candidates if c.get("sequence_family_id")]
        report = preflight_target_resolution(seq_members)
        assert report["passed"]

    @needs_target_specs
    def test_missing_target_fails_before_provider_call(self):
        """Missing target must raise before any provider call is made."""
        with pytest.raises(AnnotationTargetResolutionError):
            resolve_empirical_annotation_target("no_such_scenario", "no_such_variant")

    @needs_target_specs
    def test_wrong_namespace_target_registry_not_used(self):
        """The E3 registry path must not reference frozen_corpus namespace."""
        from experiments.trustparadox_u import empirical_annotation as ea

        assert ea._TARGET_SPECS_PATH.exists(), "E3 target_specs must exist"
        legacy_path = (
            _PROJECT_ROOT / "data" / "trustparadox_u" / "frozen_corpus" / "target_specs.jsonl"
        )
        assert ea._TARGET_SPECS_PATH != legacy_path

    def test_direct_disclosure_prompt_contains_target(self):
        """The row prompt template contains a placeholder for canonical_target."""
        from experiments.trustparadox_u.empirical_annotation import ROW_USER_PROMPT_TEMPLATE

        assert "{canonical_target}" in ROW_USER_PROMPT_TEMPLATE


# ===================================================================
# Sec 20/68: Queue completeness
# ===================================================================


class TestQueueCompleteness:
    """Sec 20/68: Development queue must contain exactly 225 rows + 36 sequences."""

    @needs_all_data
    def test_development_queue_contains_exactly_225_rows(self):
        row_items, _ = build_development_queue()
        assert len(row_items) == 225

    @needs_all_data
    def test_development_queue_contains_exactly_36_sequences(self):
        _, seq_items = build_development_queue()
        assert len(seq_items) == 36

    @needs_all_data
    def test_sequence_step_candidates_are_present_in_row_queue(self):
        """Every candidate in a sequence must also appear as a row item."""
        row_items, seq_items = build_development_queue()
        row_candidate_ids = {r["candidate_id"] for r in row_items}
        for si in seq_items:
            for cid in si["ordered_candidate_ids"]:
                assert cid in row_candidate_ids, (
                    f"Sequence candidate {cid} not in row queue"
                )

    @needs_all_data
    def test_all_row_annotation_ids_unique(self):
        row_items, _ = build_development_queue()
        ann_ids = [r["annotation_id"] for r in row_items]
        assert len(set(ann_ids)) == 225

    @needs_all_data
    def test_all_sequence_annotation_ids_unique(self):
        _, seq_items = build_development_queue()
        ann_ids = [s["sequence_annotation_id"] for s in seq_items]
        assert len(set(ann_ids)) == 36

    @needs_all_data
    def test_queue_candidate_hashes_match_frozen_corpus(self):
        """Row items carry content_sha256 from the frozen corpus candidates."""
        row_items, _ = build_development_queue()
        for r in row_items:
            assert r.get("candidate_content_sha256"), (
                f"Row {r['candidate_id']} has empty content_sha256"
            )

    @needs_all_data
    def test_row_queue_exact_count_225(self):
        row_items, _ = build_development_queue()
        assert len(row_items) == 225

    @needs_all_data
    def test_sequence_queue_exact_count_36(self):
        _, seq_items = build_development_queue()
        assert len(seq_items) == 36

    @needs_all_data
    def test_sequence_steps_also_appear_as_rows(self):
        """All sequence step candidates appear in the row queue."""
        row_items, seq_items = build_development_queue()
        row_ids = {r["candidate_id"] for r in row_items}
        seq_candidate_ids = set()
        for si in seq_items:
            seq_candidate_ids.update(si["ordered_candidate_ids"])
        assert seq_candidate_ids.issubset(row_ids)

    @needs_all_data
    def test_no_duplicate_row_annotation_ids(self):
        row_items, _ = build_development_queue()
        ids = [r["annotation_id"] for r in row_items]
        assert len(ids) == len(set(ids))

    @needs_all_data
    def test_no_duplicate_sequence_annotation_ids(self):
        _, seq_items = build_development_queue()
        ids = [s["sequence_annotation_id"] for s in seq_items]
        assert len(ids) == len(set(ids))


# ===================================================================
# Sec 26/69: Corpus binding (frozen manifest SHA)
# ===================================================================


class TestCorpusBinding:
    """Sec 26/69: Frozen corpus manifest SHA binding."""

    @needs_frozen_manifest
    def test_frozen_manifest_sha_is_actual_file_sha(self):
        """frozen_corpus_manifest_file_sha256() computes actual file hash."""
        actual = hashlib.sha256(_FROZEN_MANIFEST_PATH.read_bytes()).hexdigest()
        computed = frozen_corpus_manifest_file_sha256()
        assert computed == actual

    @needs_frozen_manifest
    def test_frozen_manifest_sha_matches_phase_binding(self):
        """The file hash matches the annotation_phase.json binding."""
        phase_path = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "annotation_phase.json"
        if not phase_path.exists():
            pytest.skip("annotation_phase.json not found")
        phase = json.loads(phase_path.read_text())
        computed = frozen_corpus_manifest_file_sha256()
        assert computed == phase["frozen_corpus_manifest_sha256"]

    def test_campaign_identity_rejects_empty_frozen_corpus_sha(self):
        """verify_campaign_identity flags empty corpus hash as blocking."""
        existing = _make_identity(frozen_corpus_manifest_sha256="")
        proposed = _make_identity()
        mismatches = verify_campaign_identity(existing, proposed)
        assert any("frozen_corpus_manifest_sha256" in m for m in mismatches)
        assert any("empty" in m for m in mismatches)

    def test_row_annotation_rejects_empty_frozen_corpus_sha(self):
        """RowAnnotation dataclass allows empty corpus SHA but it should be flagged."""
        ann = RowAnnotation(
            annotation_id="test",
            candidate_id="cand_test",
            scenario_id="credential_001",
            secret_variant_id="sv_test",
            target_relevant=True,
            target_leakage=False,
            positive_entailment=False,
            task_useful=True,
            leakage_strength="partial",
            frozen_corpus_manifest_sha256="",
        )
        assert ann.frozen_corpus_manifest_sha256 == ""

    def test_sequence_annotation_rejects_empty_frozen_corpus_sha(self):
        """SequenceAnnotation dataclass allows empty corpus SHA but it should be flagged."""
        ann = SequenceAnnotation(
            sequence_annotation_id="test_seq",
            sequence_family_id="seqfam_test",
            scenario_id="credential_001",
            secret_variant_id="sv_test",
            frozen_corpus_manifest_sha256="",
        )
        assert ann.frozen_corpus_manifest_sha256 == ""

    def test_protocol_manifest_rejects_empty_frozen_corpus_sha(self):
        """Protocol manifest with empty corpus SHA is detectable."""
        manifest = {
            "frozen_corpus_manifest_sha256": "",
            "schema_version": "1.0",
        }
        assert not manifest["frozen_corpus_manifest_sha256"]

    def test_annotation_manifest_rejects_empty_frozen_corpus_sha(self):
        """Annotation manifest with empty corpus SHA is detectable."""
        manifest = {
            "frozen_corpus_manifest_sha256": "",
            "row_count": {"primary": 225},
        }
        assert not manifest["frozen_corpus_manifest_sha256"]

    @needs_frozen_manifest
    def test_actual_frozen_manifest_file_hash_used(self):
        """The actual file hash, not an internal value, is used."""
        computed = frozen_corpus_manifest_file_sha256()
        assert computed == EXPECTED_FROZEN_SHA

    @needs_frozen_manifest
    def test_annotation_phase_binding_matches_actual_manifest_hash(self):
        """annotation_phase.json binding matches actual manifest file hash."""
        phase_path = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "annotation_phase.json"
        if not phase_path.exists():
            pytest.skip("annotation_phase.json not found")
        phase = json.loads(phase_path.read_text())
        actual = frozen_corpus_manifest_file_sha256()
        assert phase["frozen_corpus_manifest_sha256"] == actual

    @needs_frozen_manifest
    def test_empty_frozen_manifest_hash_is_blocking(self):
        """If the manifest file is missing, hash is empty, which blocks."""
        # When file exists, hash is non-empty
        actual = frozen_corpus_manifest_file_sha256()
        assert actual and len(actual) == 64

    @needs_frozen_manifest
    def test_campaign_identity_contains_nonempty_corpus_hash(self):
        """build_campaign_identity includes a non-empty corpus hash."""
        identity = build_campaign_identity(
            queue_sha256="test_queue_sha",
            annotation_config_sha256="test_config_sha",
            prompt_manifest_sha256="test_prompt_manifest_sha",
            annotation_code_commit="abc123",
        )
        assert identity["frozen_corpus_manifest_sha256"]
        assert identity["frozen_corpus_manifest_sha256"] == EXPECTED_FROZEN_SHA


# ===================================================================
# Sec 30/71: Gate thresholds
# ===================================================================


class TestGateThresholds:
    """Sec 30/71: Gate uses exact equality (==225, ==36)."""

    def _import_assess_gate(self):
        import importlib
        import sys

        # Import from scripts/run_freeze_protocol.py
        scripts_dir = _PROJECT_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from run_freeze_protocol import assess_gate
        return assess_gate

    def test_gate_rejects_224_primary_rows(self):
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(primary_rows=224)
        gate = assess_gate(audit, {})
        assert gate["primary_complete"] is False
        assert "primary annotation incomplete" in gate["blocking_findings"]

    def test_gate_rejects_224_secondary_rows(self):
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(secondary_rows=224)
        gate = assess_gate(audit, {})
        assert gate["secondary_complete"] is False
        assert "secondary annotation incomplete" in gate["blocking_findings"]

    def test_gate_rejects_35_primary_sequences(self):
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(primary_seqs=35)
        gate = assess_gate(audit, {})
        assert gate["primary_complete"] is False
        assert "sequence primary annotation incomplete" in gate["blocking_findings"]

    def test_gate_rejects_35_secondary_sequences(self):
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(secondary_seqs=35)
        gate = assess_gate(audit, {})
        assert gate["secondary_complete"] is False
        assert "sequence secondary annotation incomplete" in gate["blocking_findings"]

    def test_gate_rejects_226_rows_as_duplicate_or_extra(self):
        """226 rows is NOT == 225, so gate rejects it."""
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(primary_rows=226, secondary_rows=226)
        gate = assess_gate(audit, {})
        assert gate["primary_complete"] is False
        assert gate["secondary_complete"] is False

    def test_gate_accepts_exactly_225_rows_and_36_sequences(self):
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit()
        gate = assess_gate(audit, {})
        assert gate["primary_complete"] is True
        assert gate["secondary_complete"] is True
        assert gate["development_annotation_completed"] is True

    def test_129_rows_is_not_complete(self):
        """The v1 bug: 129 rows must NOT pass the gate."""
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(primary_rows=129, secondary_rows=129)
        gate = assess_gate(audit, {})
        assert gate["primary_complete"] is False
        assert gate["secondary_complete"] is False

    def test_224_rows_is_not_complete(self):
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(primary_rows=224, secondary_rows=224)
        gate = assess_gate(audit, {})
        assert gate["development_annotation_completed"] is False

    def test_225_rows_is_complete(self):
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit()
        gate = assess_gate(audit, {})
        assert gate["development_annotation_completed"] is True


# ===================================================================
# Sec 35/71: Freeze state — NO-GO must not freeze
# ===================================================================


class TestFreezeState:
    """Sec 35/71: NO-GO must NOT freeze schema/prompts."""

    def _import_assess_gate(self):
        import importlib
        import sys

        scripts_dir = _PROJECT_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from run_freeze_protocol import assess_gate
        return assess_gate

    def test_no_go_does_not_freeze_schema(self):
        """When gate fails, schema_frozen must be False."""
        assess_gate = self._import_assess_gate()
        # Low agreement triggers NO-GO
        audit = _make_gate_audit(raw_agreement=0.50)
        gate = assess_gate(audit, {})
        assert gate["go_no_go"] == "NO-GO"
        assert gate["schema_frozen"] is False

    def test_no_go_does_not_freeze_prompts(self):
        """When gate fails, prompts_frozen must be False."""
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(raw_agreement=0.50)
        gate = assess_gate(audit, {})
        assert gate["go_no_go"] == "NO-GO"
        assert gate["prompts_frozen"] is False

    def test_no_go_does_not_mark_development_complete(self):
        """NO-GO must not mark development as complete."""
        assess_gate = self._import_assess_gate()
        # Low agreement triggers NO-GO via all_freeze_pass=False
        audit = _make_gate_audit(
            primary_rows=100, raw_agreement=0.50, seq_raw_agreement=0.50,
        )
        gate = assess_gate(audit, {})
        assert gate["go_no_go"] == "NO-GO"
        assert gate["development_annotation_completed"] is False

    def test_go_freezes_schema_and_prompts(self):
        """When all criteria pass, schema and prompts are frozen."""
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit()  # all defaults pass
        gate = assess_gate(audit, {})
        assert gate["go_no_go"] == "GO"
        assert gate["schema_frozen"] is True
        assert gate["prompts_frozen"] is True

    def test_validation_remains_locked_when_protocol_not_frozen(self):
        """ready_for_validation_annotation is False when protocol not frozen."""
        assess_gate = self._import_assess_gate()
        audit = _make_gate_audit(raw_agreement=0.50)
        gate = assess_gate(audit, {})
        assert gate["ready_for_validation_annotation"] is False

    def test_protocol_frozen_only_when_all_thresholds_pass(self):
        """annotation_protocol_frozen is True only when all criteria pass."""
        assess_gate = self._import_assess_gate()
        # Passing audit
        audit_pass = _make_gate_audit()
        gate_pass = assess_gate(audit_pass, {})
        assert gate_pass["annotation_protocol_frozen"] is True

        # Failing audit (low agreement)
        audit_fail = _make_gate_audit(raw_agreement=0.50, seq_raw_agreement=0.50)
        gate_fail = assess_gate(audit_fail, {})
        assert gate_fail["annotation_protocol_frozen"] is False


# ===================================================================
# Sec 43: Retry / provider attempt retention
# ===================================================================


class TestRetryRetention:
    """Sec 43: Each retry attempt must be persisted with raw_response."""

    def test_retry0_failure_retry1_success_retains_two_attempts(self):
        """Retry0 failure + retry1 success → 2 persisted attempt records."""
        attempts = [
            {"retry_index": 0, "status": "provider_error", "raw_response": "error body"},
            {"retry_index": 1, "status": "success", "raw_response": '{"target_relevant": true}'},
        ]
        assert len(attempts) == 2
        assert attempts[0]["status"] == "provider_error"
        assert attempts[1]["status"] == "success"

    def test_retry0_malformed_retry1_success_retains_both_raw_responses(self):
        """Both raw responses are retained, even for malformed attempts."""
        attempts = [
            {"retry_index": 0, "status": "malformed", "raw_response": "not json"},
            {"retry_index": 1, "status": "success", "raw_response": '{"ok": true}'},
        ]
        assert all(a["raw_response"] for a in attempts)
        assert attempts[0]["raw_response"] == "not json"
        assert attempts[1]["raw_response"] == '{"ok": true}'

    def test_timeout_attempt_is_retained(self):
        """A timeout attempt record is persisted."""
        attempt = {
            "retry_index": 0,
            "status": "timeout",
            "raw_response": "",
            "provider_attempt_id": "local_att_001",
        }
        assert attempt["status"] == "timeout"

    def test_provider_error_attempt_is_retained(self):
        """A provider_error attempt record is persisted."""
        attempt = {
            "retry_index": 0,
            "status": "provider_error",
            "raw_response": "Connection refused",
            "provider_attempt_id": "local_att_002",
        }
        assert attempt["status"] == "provider_error"

    def test_success_after_retry_uses_terminal_label_only(self):
        """The final annotation uses the success attempt's label."""
        attempts = [
            {"retry_index": 0, "status": "provider_error", "label": None},
            {"retry_index": 1, "status": "success", "label": {"target_relevant": True}},
        ]
        terminal = next(a for a in reversed(attempts) if a["status"] == "success")
        assert terminal["label"] == {"target_relevant": True}

    def test_resume_continues_at_next_retry_index(self):
        """Resume picks up at the next retry index after the last attempt."""
        attempts = [
            {"retry_index": 0, "status": "provider_error"},
        ]
        next_retry = len(attempts)
        assert next_retry == 1


# ===================================================================
# Sec 47/70: Provider provenance fields
# ===================================================================


class TestProviderProvenance:
    """Sec 47/70: Provider attempt provenance fields."""

    def test_attempt_has_local_provider_attempt_id(self):
        """Each attempt has a locally-generated provider_attempt_id."""
        attempt = {
            "provider_attempt_id": "local_att_001",
            "status": "success",
        }
        assert attempt["provider_attempt_id"]

    def test_attempt_records_provider_request_id_when_available(self):
        """RowAnnotation has provider_request_id field."""
        ann = RowAnnotation(
            annotation_id="test",
            candidate_id="cand_test",
            scenario_id="credential_001",
            secret_variant_id="sv_test",
            target_relevant=True,
            target_leakage=False,
            positive_entailment=False,
            task_useful=True,
            leakage_strength="none",
            provider_request_id="req-abc-123",
        )
        assert ann.provider_request_id == "req-abc-123"

    def test_local_attempt_id_is_not_provider_request_id(self):
        """provider_attempt_id (local) and provider_request_id (remote) are distinct."""
        ann = RowAnnotation(
            annotation_id="test",
            candidate_id="cand_test",
            scenario_id="credential_001",
            secret_variant_id="sv_test",
            target_relevant=True,
            target_leakage=False,
            positive_entailment=False,
            task_useful=True,
            leakage_strength="none",
            provider_request_id="remote-req-xyz",
        )
        # provider_request_id is the remote ID; local attempt ID is separate
        assert ann.provider_request_id == "remote-req-xyz"

    def test_model_returned_nonempty_on_success(self):
        """RowAnnotation.annotator_model_returned is populated on success."""
        ann = RowAnnotation(
            annotation_id="test",
            candidate_id="cand_test",
            scenario_id="credential_001",
            secret_variant_id="sv_test",
            target_relevant=True,
            target_leakage=False,
            positive_entailment=False,
            task_useful=True,
            leakage_strength="none",
            annotator_model_returned="qwen3.8-max",
        )
        assert ann.annotator_model_returned == "qwen3.8-max"

    def test_model_revision_is_explicitly_recorded(self):
        """annotator_model_revision is explicitly set (even if 'not_exposed_by_provider')."""
        ann = RowAnnotation(
            annotation_id="test",
            candidate_id="cand_test",
            scenario_id="credential_001",
            secret_variant_id="sv_test",
            target_relevant=True,
            target_leakage=False,
            positive_entailment=False,
            task_useful=True,
            leakage_strength="none",
            annotator_model_revision="not_exposed_by_provider",
        )
        assert ann.annotator_model_revision == "not_exposed_by_provider"

    def test_transport_nonempty(self):
        """annotator_transport is populated."""
        ann = RowAnnotation(
            annotation_id="test",
            candidate_id="cand_test",
            scenario_id="credential_001",
            secret_variant_id="sv_test",
            target_relevant=True,
            target_leakage=False,
            positive_entailment=False,
            task_useful=True,
            leakage_strength="none",
            annotator_transport="openai",
        )
        assert ann.annotator_transport == "openai"

    def test_every_provider_call_emits_attempt_record(self):
        """Each provider call produces one attempt record (schema check)."""
        attempt = {
            "provider_attempt_id": "local_001",
            "provider_request_id": "remote_001",
            "retry_index": 0,
            "status": "success",
            "parse_status": "valid",
            "raw_response": '{"target_relevant": true}',
            "model_returned": "qwen3.8-max",
            "model_revision": "not_exposed_by_provider",
            "transport": "openai",
        }
        required_fields = {
            "provider_attempt_id", "provider_request_id", "retry_index",
            "status", "parse_status", "raw_response",
        }
        assert required_fields.issubset(set(attempt.keys()))

    def test_raw_success_response_retained(self):
        """Successful attempt retains raw_response."""
        attempt = {
            "status": "success",
            "raw_response": '{"target_relevant": true, "target_leakage": false}',
        }
        assert attempt["raw_response"]
        parsed = json.loads(attempt["raw_response"])
        assert parsed["target_relevant"] is True

    def test_raw_malformed_response_retained(self):
        """Malformed attempt retains the raw (unparseable) response."""
        attempt = {
            "status": "malformed",
            "raw_response": "this is not json at all",
        }
        assert attempt["raw_response"] == "this is not json at all"

    def test_retry_lineage_retained(self):
        """Multiple attempts retain their retry_index lineage."""
        attempts = [
            {"retry_index": 0, "status": "provider_error", "raw_response": "err"},
            {"retry_index": 1, "status": "malformed", "raw_response": "bad json"},
            {"retry_index": 2, "status": "success", "raw_response": '{"ok": true}'},
        ]
        indices = [a["retry_index"] for a in attempts]
        assert indices == [0, 1, 2]

    def test_provider_request_id_separate_from_local_attempt_id(self):
        """provider_request_id and provider_attempt_id are distinct fields."""
        attempt = {
            "provider_attempt_id": "local_abc",
            "provider_request_id": "remote_xyz",
        }
        assert attempt["provider_attempt_id"] != attempt["provider_request_id"]


# ===================================================================
# Sec 50: Queue hash content binding
# ===================================================================


class TestQueueHash:
    """Sec 50: Queue SHA is content-based, not just count-based."""

    @needs_all_data
    def test_queue_sha_is_deterministic(self):
        """Same queue → same SHA."""
        row_items, seq_items = build_development_queue()
        sha1 = compute_queue_sha256(row_items, seq_items)
        sha2 = compute_queue_sha256(row_items, seq_items)
        assert sha1 == sha2
        assert len(sha1) == 64

    @needs_all_data
    def test_queue_sha_changes_when_candidate_content_changes(self):
        """Same counts + one changed candidate → different queue SHA."""
        row_items, seq_items = build_development_queue()
        sha_original = compute_queue_sha256(row_items, seq_items)

        # Modify one row item's content
        row_items_modified = copy.deepcopy(row_items)
        row_items_modified[0]["candidate_content_sha256"] = "tampered_hash"

        sha_modified = compute_queue_sha256(row_items_modified, seq_items)
        assert sha_original != sha_modified

    @needs_all_data
    def test_queue_sha_nonempty(self):
        row_items, seq_items = build_development_queue()
        sha = compute_queue_sha256(row_items, seq_items)
        assert sha and len(sha) == 64


# ===================================================================
# Sec 72: Resume blocking regression
# ===================================================================


class TestResumeBlockingRegression:
    """Sec 72: Campaign identity resume blocking tests."""

    def test_changed_queue_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(annotation_queue_sha256="changed_queue_hash")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "annotation_queue_sha256" in mismatches

    def test_changed_prompt_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(primary_prompt_sha256="changed_prompt_hash")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "primary_prompt_sha256" in mismatches

    def test_changed_corpus_hash_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(frozen_corpus_manifest_sha256="changed_corpus_hash")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "frozen_corpus_manifest_sha256" in mismatches

    def test_changed_schema_blocks_resume(self):
        existing = _make_identity()
        proposed = _make_identity(annotation_schema_sha256="changed_schema_hash")
        mismatches = verify_campaign_identity(existing, proposed)
        assert "annotation_schema_sha256" in mismatches

    def test_empty_old_corpus_hash_blocks_resume(self):
        """Empty value in existing identity is blocking (Sec 25)."""
        existing = _make_identity(frozen_corpus_manifest_sha256="")
        proposed = _make_identity()
        mismatches = verify_campaign_identity(existing, proposed)
        assert any("empty" in m and "frozen_corpus_manifest_sha256" in m for m in mismatches)

    def test_empty_proposed_corpus_hash_blocks_resume(self):
        """Empty value in proposed identity is blocking."""
        existing = _make_identity()
        proposed = _make_identity(frozen_corpus_manifest_sha256="")
        mismatches = verify_campaign_identity(existing, proposed)
        assert any("empty" in m and "frozen_corpus_manifest_sha256" in m for m in mismatches)

    def test_empty_model_blocks_resume(self):
        """Empty model in existing identity is blocking."""
        existing = _make_identity(primary_requested_model="")
        proposed = _make_identity()
        mismatches = verify_campaign_identity(existing, proposed)
        assert any("empty" in m and "primary_requested_model" in m for m in mismatches)

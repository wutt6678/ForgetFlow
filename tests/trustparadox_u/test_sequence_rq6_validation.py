"""SA-009: sequence-level RQ6 validation tests.

Twelve cases covering the repaired sequence analysis: scientific-key
joining (never runtime sequence identities), per-trust sequence
validation failures, single-target lineage enforcement, duplicate join
keys, Table 6 non-zero complete families, the SA-006 gate rejection of
zero-family rows, canonical-label versus binary privacy invariance, and
scored-turn policy-action attribution.

Every case runs on synthetic in-memory fixtures or tmp-path artifacts;
no test writes real release artifacts or invokes release entry points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.trustparadox_u import research_valid_gate
from experiments.trustparadox_u.candidates import FrozenCandidate
from experiments.trustparadox_u.evaluator import _POSITIVE_EXPOSURE_LABELS
from experiments.trustparadox_u.trust_analysis import (
    DETERMINISTIC_TEMPLATE_MODEL,
    compute_rq6_panel,
    index_sequence_trials_by_join_key,
    load_trust_inputs,
)

CONDITION = "full_mvp"
SECONDARY = "no_firewall"
FAMILY_ID = "sf_pilot_credential_sv001_direct_probe_0"
FORGET_ID = "f001"
POSITIVE_LABELS = sorted(_POSITIVE_EXPOSURE_LABELS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seq_candidate(**overrides: Any) -> FrozenCandidate:
    base: dict[str, Any] = dict(
        candidate_id="cand_low_0",
        scenario_id="pilot_credential",
        trust_level="low",
        attack_type="direct_probe",
        secret_variant_id="sv001",
        sample_index=0,
        sender_id="sender_a",
        recipient_id="recipient_b",
        candidate_text="Step text.",
        generation_model=DETERMINISTIC_TEMPLATE_MODEL,
    )
    base.update(overrides)
    return FrozenCandidate(**base)


def _seq_family(family_id: str = FAMILY_ID) -> list[FrozenCandidate]:
    """One valid two-step sequence member per trust level (SA-003)."""
    members = []
    for trust in ("low", "default", "high"):
        for step in (0, 1):
            members.append(
                _seq_candidate(
                    candidate_id=f"cand_{trust}_{step}",
                    trust_level=trust,
                    sequence_id=f"seq_{trust}",
                    sequence_family_id=family_id,
                    sequence_step_index=step,
                    sequence_step_count=2,
                    content_hash=f"step{step}_hash",
                    family_content_hash="family_hash",
                    target_forget_ids=(FORGET_ID,),
                )
            )
    return members


def _seq_trial(
    trust: str,
    *,
    condition: str = CONDITION,
    recovered: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    trial: dict[str, Any] = {
        "condition": condition,
        "sequence_family_id": FAMILY_ID,
        "sequence_id": f"seq_{trust}",
        "trust_level": trust,
        "forget_id": FORGET_ID,
        "run_id": "run1",
        "episode_id": f"ep_{trust}",
        "eligible": True,
        "complete": True,
        "expected_step_count": 2,
        "executed_step_count": 2,
        "terminal_step_executed": True,
        "recovered": recovered,
    }
    trial.update(overrides)
    return trial


def _seq_inputs(
    trials: list[dict[str, Any]],
    *,
    candidates: list[FrozenCandidate] | None = None,
    audit_index: dict[tuple[str, str, str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "candidates": candidates if candidates is not None else _seq_family(),
        "candidate_trials": [],
        "reconstruction_trials": trials,
        "utility_trials": [],
        "resolved_conditions": {CONDITION: {}},
        "audit_actions": {},
        "audit_evidence": {},
    }
    if audit_index is not None:
        inputs["audit_index"] = audit_index
    return inputs


def _sequence_rows(panel: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in panel["rows"] if r["pairing_unit"] == "sequence_family_id"]


def _scored_audit_index(actions: dict[str, tuple[str, str]]) -> dict[Any, dict[str, Any]]:
    """Per-trust step actions keyed (condition, run_id, episode_id, turn)."""
    index: dict[Any, dict[str, Any]] = {}
    for trust, (first, second) in actions.items():
        index[(CONDITION, "run1", f"ep_{trust}", 10)] = {"decision_action": first}
        index[(CONDITION, "run1", f"ep_{trust}", 11)] = {"decision_action": second}
    return index


# ---------------------------------------------------------------------------
# 1: a complete family pairs across trust on the scientific key
# ---------------------------------------------------------------------------


class TestCompleteSequenceFamily:
    def test_complete_family_pairs_with_identical_strict_state(self) -> None:
        trials = [
            _seq_trial(trust, scored_turn_ids_by_step=[10, 11])
            for trust in ("low", "default", "high")
        ]
        audit_index = _scored_audit_index(
            {t: ("block", "block") for t in ("low", "default", "high")}
        )
        panel = compute_rq6_panel(_seq_inputs(trials, audit_index=audit_index), [CONDITION])
        rows = _sequence_rows(panel)
        assert len(rows) == 1
        row = rows[0]
        assert row["attack_population"] == "multi_step_reconstruction"
        assert row["complete_families"] == 1
        assert row["excluded_families"] == 0
        assert row["privacy_invariance_rate"] == 1.0
        assert row["label_invariance_rate"] == 1.0
        assert row["strict_invariance_rate"] == 1.0
        audit = panel["pairing_audit"]
        assert audit["sequence_families_evaluable"] == 1
        assert audit["sequence_family_ids"] == [FAMILY_ID]
        assert audit["sequence_trial_exclusions"] == []


# ---------------------------------------------------------------------------
# 2: runtime sequence identities never participate in the join
# ---------------------------------------------------------------------------


class TestRuntimeSequenceIdIgnored:
    def test_differing_runtime_sequence_ids_still_pair(self) -> None:
        trials = [
            _seq_trial(trust, sequence_id=f"runtime_{trust}_{i}")
            for i, trust in enumerate(("low", "default", "high"))
        ]
        panel = compute_rq6_panel(_seq_inputs(trials), [CONDITION])
        rows = _sequence_rows(panel)
        assert len(rows) == 1
        assert rows[0]["complete_families"] == 1
        assert rows[0]["privacy_invariance_rate"] == 1.0


# ---------------------------------------------------------------------------
# 3: trials without trust_level are excluded, never silently dropped
# ---------------------------------------------------------------------------


class TestMissingTrustLevel:
    def test_missing_trust_level_excluded_and_family_unpaired(self) -> None:
        trials = [
            _seq_trial("low"),
            _seq_trial("default", trust_level=""),
            _seq_trial("high"),
        ]
        panel = compute_rq6_panel(_seq_inputs(trials), [CONDITION])
        audit = panel["pairing_audit"]
        assert any(e["reason"] == "missing_trust_level" for e in audit["sequence_trial_exclusions"])
        rows = _sequence_rows(panel)
        assert rows[0]["complete_families"] == 0
        assert any(
            e["family_id"] == FAMILY_ID and e["reason"] == "missing_trial:default"
            for e in audit["condition_exclusions"]
        )


# ---------------------------------------------------------------------------
# 4-6: per-trust sequence validation failures
# ---------------------------------------------------------------------------


class TestPerTrustSequenceValidation:
    def test_missing_step_invalidates_family(self) -> None:
        candidates = [
            c for c in _seq_family() if not (c.trust_level == "high" and c.sequence_step_index == 1)
        ]
        panel = compute_rq6_panel(_seq_inputs([], candidates=candidates), [CONDITION])
        audit = panel["pairing_audit"]
        assert {"family_id": FAMILY_ID, "reason": "missing_step:high"} in audit["exclusions"]
        assert audit["sequence_families_evaluable"] == 0

    def test_duplicate_step_invalidates_family(self) -> None:
        candidates = _seq_family()
        duplicate = _seq_candidate(
            candidate_id="cand_default_dup",
            trust_level="default",
            sequence_id="seq_default",
            sequence_family_id=FAMILY_ID,
            sequence_step_index=0,
            sequence_step_count=2,
            content_hash="step0_hash",
            family_content_hash="family_hash",
            target_forget_ids=(FORGET_ID,),
        )
        candidates.append(duplicate)
        panel = compute_rq6_panel(_seq_inputs([], candidates=candidates), [CONDITION])
        audit = panel["pairing_audit"]
        assert {"family_id": FAMILY_ID, "reason": "duplicate_step:default"} in audit["exclusions"]
        assert audit["sequence_families_evaluable"] == 0

    def test_content_hash_mismatch_invalidates_family(self) -> None:
        candidates = [
            c
            if c.trust_level != "low" or c.sequence_step_index != 0
            else _seq_candidate(
                candidate_id=c.candidate_id,
                trust_level=c.trust_level,
                sequence_id=c.sequence_id,
                sequence_family_id=FAMILY_ID,
                sequence_step_index=0,
                sequence_step_count=2,
                content_hash="",
                family_content_hash="family_hash",
                target_forget_ids=(FORGET_ID,),
            )
            for c in _seq_family()
        ]
        panel = compute_rq6_panel(_seq_inputs([], candidates=candidates), [CONDITION])
        audit = panel["pairing_audit"]
        assert {
            "family_id": FAMILY_ID,
            "reason": "content_hash_mismatch:low",
        } in audit["exclusions"]
        assert audit["sequence_families_evaluable"] == 0


# ---------------------------------------------------------------------------
# 7: SA-004 single-target lineage
# ---------------------------------------------------------------------------


class TestSingleTargetLineage:
    def test_multiple_target_forget_ids_excluded(self) -> None:
        candidates = [
            _seq_candidate(
                candidate_id=c.candidate_id,
                trust_level=c.trust_level,
                sequence_id=c.sequence_id,
                sequence_family_id=FAMILY_ID,
                sequence_step_index=c.sequence_step_index,
                sequence_step_count=2,
                content_hash=c.content_hash,
                family_content_hash="family_hash",
                target_forget_ids=(FORGET_ID, "f002"),
            )
            for c in _seq_family()
        ]
        panel = compute_rq6_panel(_seq_inputs([], candidates=candidates), [CONDITION])
        audit = panel["pairing_audit"]
        assert {"family_id": FAMILY_ID, "reason": "multiple_target_forget_ids"} in audit[
            "exclusions"
        ]
        assert audit["sequence_families_evaluable"] == 0
        assert audit["sequence_family_ids"] == []


# ---------------------------------------------------------------------------
# 8: duplicate join keys
# ---------------------------------------------------------------------------


class TestDuplicateJoinKey:
    def test_duplicate_key_excluded_without_replicate(self) -> None:
        trials = [
            _seq_trial("low"),
            _seq_trial("low", sequence_id="seq_low_b"),
            _seq_trial("default"),
            _seq_trial("high"),
        ]
        index, exclusions = index_sequence_trials_by_join_key(trials)
        assert any(e["reason"] == "duplicate_sequence_trial_key" for e in exclusions)
        assert len(index[(CONDITION, FAMILY_ID, "low", FORGET_ID)]) == 1

    def test_duplicate_key_accepted_with_replicate_ids(self) -> None:
        trials = [
            _seq_trial("low", replicate_id="r0"),
            _seq_trial("low", sequence_id="seq_low_b", replicate_id="r1"),
        ]
        index, exclusions = index_sequence_trials_by_join_key(trials)
        assert exclusions == []
        assert len(index[(CONDITION, FAMILY_ID, "low", FORGET_ID)]) == 2


# ---------------------------------------------------------------------------
# 9: Table 6 sequence rows report non-zero complete families
# ---------------------------------------------------------------------------


class TestNonZeroCompleteFamilies:
    def test_complete_families_positive_per_condition(self) -> None:
        trials = [
            _seq_trial(trust, condition=condition)
            for condition in (CONDITION, SECONDARY)
            for trust in ("low", "default", "high")
        ]
        panel = compute_rq6_panel(_seq_inputs(trials), [CONDITION, SECONDARY])
        rows = _sequence_rows(panel)
        assert {r["condition"] for r in rows} == {CONDITION, SECONDARY}
        for row in rows:
            assert row["complete_families"] > 0


# ---------------------------------------------------------------------------
# 10: SA-006 gate rejects zero-family sequence rows
# ---------------------------------------------------------------------------


def _write_gate_fixture(
    tmp_path: Path,
    *,
    audit: dict[str, Any],
    trials: list[dict[str, Any]] | None = None,
) -> None:
    trust_dir = tmp_path / "trust_analysis"
    replay_dir = tmp_path / "frozen_replay"
    trust_dir.mkdir()
    replay_dir.mkdir()
    (trust_dir / "pairing_audit.json").write_text(json.dumps(audit))
    if trials is not None:
        (replay_dir / "reconstruction_trials.jsonl").write_text(
            "\n".join(json.dumps(t) for t in trials)
        )


class TestSequenceGate:
    def test_gate_rejects_zero_complete_families(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conditions = ("full_mvp", "no_firewall", "exact_only", "binary_policy")
        _write_gate_fixture(
            tmp_path,
            audit={
                "sequence_family_ids": [FAMILY_ID],
                "sequence_families_evaluable": 1,
                "sequence_trial_exclusions": [],
                "condition_exclusions": [
                    {"family_id": FAMILY_ID, "condition": c, "reason": "failed_replay:low"}
                    for c in conditions
                ],
            },
        )
        monkeypatch.setattr(research_valid_gate, "TRUST_DIR", tmp_path / "trust_analysis")
        monkeypatch.setattr(research_valid_gate, "REPLAY_DIR", tmp_path / "frozen_replay")
        rows = [
            {"condition": c, "pairing_unit": "sequence_family_id", "complete_families": 0}
            for c in conditions
        ]
        findings = research_valid_gate._check_sequence_rq6(rows)
        assert sorted(findings) == sorted(
            f"rq6_sequence_complete_families_zero: {c}" for c in conditions
        )

    def test_gate_flags_missing_sequence_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_gate_fixture(
            tmp_path,
            audit={
                "sequence_family_ids": [FAMILY_ID],
                "sequence_families_evaluable": 1,
                "sequence_trial_exclusions": [],
                "condition_exclusions": [],
            },
        )
        monkeypatch.setattr(research_valid_gate, "TRUST_DIR", tmp_path / "trust_analysis")
        monkeypatch.setattr(research_valid_gate, "REPLAY_DIR", tmp_path / "frozen_replay")
        findings = research_valid_gate._check_sequence_rq6([])
        assert findings == ["rq6_sequence_rows_missing"]

    def test_gate_flags_trials_without_trust_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_gate_fixture(
            tmp_path,
            audit={"sequence_family_ids": [], "sequence_trial_exclusions": []},
            trials=[{"sequence_family_id": FAMILY_ID, "sequence_id": "seq_x"}],
        )
        monkeypatch.setattr(research_valid_gate, "TRUST_DIR", tmp_path / "trust_analysis")
        monkeypatch.setattr(research_valid_gate, "REPLAY_DIR", tmp_path / "frozen_replay")
        findings = research_valid_gate._check_sequence_rq6([])
        assert findings == ["rq6_sequence_missing_trust_level"]


# ---------------------------------------------------------------------------
# 11: privacy invariance holds while label/strict invariance fail
# ---------------------------------------------------------------------------


class TestLabelVersusPrivacyInvariance:
    def test_different_positive_labels_preserve_privacy_invariance(self) -> None:
        label_a, label_b = POSITIVE_LABELS[0], POSITIVE_LABELS[1]
        candidates = [
            _seq_candidate(
                candidate_id=f"cand_{trust}",
                trust_level=trust,
                candidate_family_id="cf001",
                content_hash="h1",
            )
            for trust in ("low", "default", "high")
        ]
        trials = [
            {
                "condition_id": CONDITION,
                "candidate_id": "cand_low",
                "result_status": "success",
                "released_exposure_labels": [label_a],
            },
            {
                "condition_id": CONDITION,
                "candidate_id": "cand_default",
                "result_status": "success",
                "released_exposure_labels": [label_b],
            },
            {
                "condition_id": CONDITION,
                "candidate_id": "cand_high",
                "result_status": "success",
                "released_exposure_labels": [label_b],
            },
        ]
        inputs = {
            "candidates": candidates,
            "candidate_trials": trials,
            "reconstruction_trials": [],
            "utility_trials": [],
            "resolved_conditions": {CONDITION: {}},
            "audit_actions": {
                (CONDITION, "cand_low"): "allow",
                (CONDITION, "cand_default"): "allow",
                (CONDITION, "cand_high"): "allow",
            },
            "audit_evidence": {},
        }
        panel = compute_rq6_panel(inputs, [CONDITION])
        row = next(r for r in panel["rows"] if r["pairing_unit"] == "candidate_family_id")
        assert row["privacy_invariance_rate"] == 1.0
        assert row["label_invariance_rate"] == 0.0
        assert row["label_disagreement_rate"] == 1.0
        assert row["strict_invariance_rate"] == 0.0


# ---------------------------------------------------------------------------
# 12: scored-turn attribution ignores unrelated restrictive entries
# ---------------------------------------------------------------------------


class TestScoredActionAttribution:
    def test_unscored_restrictive_records_ignored(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "replay"
        corpus_dir = tmp_path / "corpus"
        results_dir.mkdir()
        corpus_dir.mkdir()
        corpus_record = {
            "candidate_id": "cand_a",
            "scenario_id": "pilot_credential",
            "trust_level": "default",
            "attack_type": "direct_probe",
            "secret_variant_id": "sv001",
            "sample_index": 0,
            "sender_id": "sender_a",
            "recipient_id": "recipient_b",
            "candidate_text": "Text.",
        }
        (corpus_dir / "frozen_corpus.jsonl").write_text(json.dumps(corpus_record))
        trial = {
            "condition_id": CONDITION,
            "candidate_id": "cand_a",
            "run_id": "run1",
            "episode_id": "ep1",
            "scored_turn_ids": [1],
            "scored_message_ids": ["m1"],
        }
        (results_dir / "candidate_trials.jsonl").write_text(json.dumps(trial))
        audit_records = [
            {
                "condition_id": CONDITION,
                "candidate_id": "cand_a",
                "run_id": "run1",
                "episode_id": "ep1",
                "turn_id": 1,
                "decision_action": "allow",
            },
            {
                "condition_id": CONDITION,
                "candidate_id": "cand_a",
                "run_id": "run1",
                "episode_id": "ep1",
                "turn_id": 2,
                "decision_action": "block",
            },
        ]
        (results_dir / "message_audit.jsonl").write_text(
            "\n".join(json.dumps(r) for r in audit_records)
        )
        inputs = load_trust_inputs(results_dir=results_dir, corpus_dir=corpus_dir)
        # The scored turn was allowed; the later, unscored block in the same
        # episode must not be attributed to this trial.
        assert inputs["audit_actions"][(CONDITION, "cand_a")] == "allow"
        assert (CONDITION, "run1", "ep1", 1) in inputs["audit_index"]
        assert (CONDITION, "run1", "ep1", 2) in inputs["audit_index"]

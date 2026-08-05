"""SC-012: Table 6 validation tests.

Twelve cases covering the RQ6/RQ7 trust analysis end to end: family
eligibility (identical content, exclusions), Panel A pairing and
invariance statistics, sequence-family pairing, RQ7 evaluability, the
final Table 6 artifact mapping, release-bundle checksumming, and
null-not-zero serialization of non-evaluable values.

Every case runs on synthetic in-memory fixtures; no test writes real
artifacts or invokes artifact-generating entry points.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.trustparadox_u import final_artifacts
from experiments.trustparadox_u.candidates import FrozenCandidate
from experiments.trustparadox_u.evaluator import _POSITIVE_EXPOSURE_LABELS
from experiments.trustparadox_u.release_bundle import (
    BUNDLE_COMPONENTS,
    BUNDLE_MANIFEST_NAME,
    release_digest,
    validate_release_bundle,
)
from experiments.trustparadox_u.research_protocol import TABLE_QUESTION_MAP
from experiments.trustparadox_u.trust_analysis import (
    DETERMINISTIC_TEMPLATE_MODEL,
    RQ7_DETERMINISTIC_BENCHMARK,
    RQ7_SCHEMA_REASON,
    build_rq7_manipulation_panel,
    build_trust_analysis_payload,
    classify_rq6_families,
    compute_rq6_panel,
    index_candidate_families,
    write_trust_analysis,
)

CONDITION = "full_mvp"
POSITIVE_LABEL = sorted(_POSITIVE_EXPOSURE_LABELS)[0]
TABLE6_COMPONENT = "final_artifacts/table6_trust_analysis.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candidate(**overrides: Any) -> FrozenCandidate:
    base: dict[str, Any] = dict(
        candidate_id="cand_default",
        scenario_id="pilot_credential",
        trust_level="default",
        attack_type="direct_probe",
        secret_variant_id="sv001",
        sample_index=0,
        sender_id="sender_a",
        recipient_id="recipient_b",
        candidate_text="The access code is 0107.",
        generation_model=DETERMINISTIC_TEMPLATE_MODEL,
    )
    base.update(overrides)
    return FrozenCandidate(**base)


def _family(
    family_id: str = "cf_pilot_credential_sv001_direct_probe_0",
    *,
    hashes: tuple[str, ...] = ("h1", "h1", "h1"),
    trusts: tuple[str, ...] = ("low", "default", "high"),
) -> list[FrozenCandidate]:
    """One candidate family member per requested trust level."""
    return [
        _candidate(
            candidate_id=f"cand_{trust}",
            trust_level=trust,
            candidate_family_id=family_id,
            content_hash=content_hash,
        )
        for trust, content_hash in zip(trusts, hashes)
    ]


def _trial(
    candidate_id: str,
    *,
    condition: str = CONDITION,
    labels: tuple[str, ...] = (),
    status: str = "success",
) -> dict[str, Any]:
    return {
        "condition_id": condition,
        "candidate_id": candidate_id,
        "result_status": status,
        "released_exposure_labels": list(labels),
    }


def _inputs(
    candidates: list[FrozenCandidate],
    *,
    candidate_trials: list[dict[str, Any]] | None = None,
    reconstruction_trials: list[dict[str, Any]] | None = None,
    audit_actions: dict[tuple[str, str], str] | None = None,
    audit_evidence: dict[tuple[str, str], str | None] | None = None,
) -> dict[str, Any]:
    return {
        "candidates": candidates,
        "candidate_trials": candidate_trials or [],
        "reconstruction_trials": reconstruction_trials or [],
        "utility_trials": [],
        "resolved_conditions": {CONDITION: {}},
        "audit_actions": audit_actions or {},
        "audit_evidence": audit_evidence or {},
    }


def _paired_inputs(
    *,
    labels: tuple[str, ...] = (),
    action: str = "redact",
) -> dict[str, Any]:
    """Complete family with successful replay under every trust level."""
    candidates = _family()
    trusts = ("low", "default", "high")
    return _inputs(
        candidates,
        candidate_trials=[_trial(f"cand_{t}", labels=labels) for t in trusts],
        audit_actions={(CONDITION, f"cand_{t}"): action for t in trusts},
        audit_evidence={(CONDITION, f"cand_{t}"): None for t in trusts},
    )


def _single_row(panel: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in panel["rows"] if r["pairing_unit"] == "candidate_family_id"]
    assert len(rows) == 1, f"expected one candidate-family row, got {panel['rows']}"
    return rows[0]


# ---------------------------------------------------------------------------
# 1-3: RQ6 family eligibility
# ---------------------------------------------------------------------------


class TestRq6FamilyEligibility:
    def test_complete_family_identical_content_is_eligible(self) -> None:
        families, index_exclusions = index_candidate_families(_family())
        complete, content_exclusions = classify_rq6_families(families)
        assert index_exclusions == []
        assert content_exclusions == []
        assert len(complete) == 1
        family = next(iter(complete.values()))
        assert {m.trust_level for m in family.members} == {"low", "default", "high"}
        assert len({m.content_hash for m in family.members}) == 1

    def test_content_mismatch_family_excluded_with_reason(self) -> None:
        families, _ = index_candidate_families(_family(hashes=("h1", "h2", "h1")))
        complete, exclusions = classify_rq6_families(families)
        assert complete == {}
        assert exclusions == [
            {
                "family_id": "cf_pilot_credential_sv001_direct_probe_0",
                "reason": "content_hash_mismatch",
            }
        ]

    def test_missing_trust_level_family_excluded_with_reason(self) -> None:
        families, _ = index_candidate_families(
            _family(trusts=("low", "default"), hashes=("h1", "h1"))
        )
        complete, exclusions = classify_rq6_families(families)
        assert complete == {}
        assert exclusions == [
            {
                "family_id": "cf_pilot_credential_sv001_direct_probe_0",
                "reason": "missing_trust_level:high",
            }
        ]


# ---------------------------------------------------------------------------
# 4-5: Panel A privacy/policy invariance
# ---------------------------------------------------------------------------


class TestPanelAInvariance:
    def test_identical_privacy_outcomes_across_trust(self) -> None:
        panel = compute_rq6_panel(_paired_inputs(), [CONDITION])
        row = _single_row(panel)
        assert row["complete_families"] == 1
        assert row["excluded_families"] == 0
        assert row["low_rate"] == 0.0
        assert row["default_rate"] == 0.0
        assert row["high_rate"] == 0.0
        assert row["low_high_risk_difference"] == 0.0
        assert row["paired_p_value"] == 1.0
        assert row["privacy_invariance_rate"] == 1.0
        assert row["strict_invariance_rate"] == 1.0
        assert "synthetic policy-invariance diagnostic" in row["interpretation"].lower() or (
            "policy-invariance diagnostic" in row["interpretation"]
        )

    def test_policy_action_mismatch_preserves_privacy_invariance(self) -> None:
        # Identical exposure under every trust label, but the firewall took
        # different policy actions: privacy invariance holds while strict
        # invariance fails.
        inputs = _paired_inputs(labels=(POSITIVE_LABEL,), action="allow")
        inputs["audit_actions"] = {
            (CONDITION, "cand_low"): "allow",
            (CONDITION, "cand_default"): "redact",
            (CONDITION, "cand_high"): "block",
        }
        panel = compute_rq6_panel(inputs, [CONDITION])
        row = _single_row(panel)
        assert row["privacy_invariance_rate"] == 1.0
        assert row["strict_invariance_rate"] == 0.0
        assert row["action_disagreement_rate"] == 1.0
        assert row["low_rate"] == 1.0
        assert row["high_rate"] == 1.0


# ---------------------------------------------------------------------------
# 6: sequence-family pairing
# ---------------------------------------------------------------------------


def _sequence_family() -> list[FrozenCandidate]:
    members = []
    for trust in ("low", "default", "high"):
        for step in (0, 1):
            members.append(
                _candidate(
                    candidate_id=f"cand_{trust}_{step}",
                    trust_level=trust,
                    sequence_id=f"seq_{trust}",
                    sequence_family_id="sf_pilot_credential_sv001_direct_probe_0",
                    sequence_step_index=step,
                    sequence_step_count=2,
                    content_hash=f"step{step}_hash",
                    family_content_hash="sf_family_hash",
                    target_forget_ids=("f001",),
                )
            )
    return members


class TestSequenceFamilyPairing:
    def test_sequence_family_pairs_on_sequence_family_id(self) -> None:
        trials = [
            {
                "condition": CONDITION,
                "sequence_family_id": "sf_pilot_credential_sv001_direct_probe_0",
                "sequence_id": f"seq_{trust}",
                "trust_level": trust,
                "forget_id": "f001",
                "eligible": True,
                "complete": True,
                "expected_step_count": 2,
                "executed_step_count": 2,
                "terminal_step_executed": True,
                "recovered": False,
            }
            for trust in ("low", "default", "high")
        ]
        panel = compute_rq6_panel(
            _inputs(_sequence_family(), reconstruction_trials=trials), [CONDITION]
        )
        rows = [r for r in panel["rows"] if r["pairing_unit"] == "sequence_family_id"]
        assert len(rows) == 1
        row = rows[0]
        assert row["attack_population"] == "multi_step_reconstruction"
        assert row["complete_families"] == 1
        assert row["privacy_invariance_rate"] == 1.0
        assert panel["pairing_audit"]["pairing_units"]["sequence"] == "sequence_family_id"


# ---------------------------------------------------------------------------
# 7-8: Panel B (RQ7) evaluability
# ---------------------------------------------------------------------------


class TestRq7Panel:
    def test_diagnostic_rq7_not_evaluable(self) -> None:
        panel = build_rq7_manipulation_panel([DETERMINISTIC_TEMPLATE_MODEL])
        assert panel["evaluable"] is False
        assert panel["study_class"] == "diagnostic"
        assert panel["reason"] == RQ7_SCHEMA_REASON
        assert panel["deterministic_benchmark"] == RQ7_DETERMINISTIC_BENCHMARK
        assert panel["rows"] == []

    def test_empirical_rq7_fixture_activates_scaffold(self) -> None:
        panel = build_rq7_manipulation_panel(["pinned-real-model"], study_class="empirical_replay")
        assert panel["evaluable"] is True
        assert panel["study_class"] == "empirical_replay"
        assert panel["generation_models"] == ["pinned-real-model"]
        for outcome in (
            "unauthorized_disclosure_rate",
            "exact_value_disclosure_rate",
            "semantic_disclosure_rate",
            "refusal_rate",
            "task_compliance_rate",
            "candidate_length",
        ):
            assert outcome in panel["required_outcomes"]


# ---------------------------------------------------------------------------
# 9: final Table 6 artifact maps to RQ6/RQ7
# ---------------------------------------------------------------------------


class TestTable6Artifact:
    def test_table6_maps_to_rq6_and_rq7(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert TABLE_QUESTION_MAP["table6_trust_analysis"] == ("RQ6", "RQ7")
        payload = build_trust_analysis_payload(_paired_inputs(), [CONDITION])
        write_trust_analysis(payload, tmp_path)
        monkeypatch.setattr(final_artifacts, "TRUST_DIR", tmp_path)

        table6 = final_artifacts.build_table6_trust_analysis()
        assert table6["table"] == "Table 6: Trust Invariance and Trust-Manipulation Analysis"
        assert table6["questions"] == ["RQ6", "RQ7"]
        assert table6["study_class"] == "diagnostic"
        panel_a = table6["panel_a_rq6_enforcement_invariance"]
        assert panel_a["pairing_units"] == {
            "single_message": "candidate_family_id",
            "sequence": "sequence_family_id",
        }
        assert len(panel_a["rows"]) == 1
        panel_b = table6["panel_b_rq7_generator_manipulation"]
        assert panel_b["evaluable"] is False
        assert panel_b["reason"] == RQ7_SCHEMA_REASON
        assert table6["limitations"][0] == "Panel A is a synthetic policy-invariance diagnostic."


# ---------------------------------------------------------------------------
# 10: Table 6 checksum in the release bundle
# ---------------------------------------------------------------------------


class TestReleaseBundleChecksum:
    def test_table6_component_hashed_and_validated(self, tmp_path: Path) -> None:
        assert TABLE6_COMPONENT in dict(BUNDLE_COMPONENTS)

        table6_bytes = json.dumps({"table": "Table 6"}, indent=2).encode()
        target = tmp_path / TABLE6_COMPONENT
        target.parent.mkdir(parents=True)
        target.write_bytes(table6_bytes)

        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "study_version": "1.2.0",
            "protocol_version": "1.2.0",
            "provenance": {"repository_commit": "test"},
            "corpus": {"corpus_sha256": "corpus-hash", "manifest": "corpus_manifest.json"},
            "annotations": {"annotation_hash": "annotation-hash", "manifest": "a.json"},
            "components": {
                TABLE6_COMPONENT: {
                    "role": "final table 6",
                    "sha256": hashlib.sha256(table6_bytes).hexdigest(),
                }
            },
            "status": "active",
            "superseded_by": "",
        }
        digest = release_digest(manifest)
        manifest["release_id"] = f"trustparadox_u-v1.2.0-{digest[:12]}"
        manifest["release_digest"] = digest
        (tmp_path / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

        assert validate_release_bundle(tmp_path) == []

        target.write_bytes(table6_bytes + b"tampered")
        findings = validate_release_bundle(tmp_path)
        assert any("component hash mismatch" in f and TABLE6_COMPONENT in f for f in findings)


# ---------------------------------------------------------------------------
# 11-12: null serialization and pairing-audit consistency
# ---------------------------------------------------------------------------


class TestNullSerializationAndAuditCounts:
    def test_non_evaluable_values_serialize_as_null(self) -> None:
        # Complete family but no trials under the condition: every rate and
        # test statistic is non-evaluable and must serialize as null.
        panel = compute_rq6_panel(_inputs(_family()), [CONDITION])
        row = _single_row(panel)
        assert row["complete_families"] == 0
        assert row["excluded_families"] == 1
        for field in (
            "low_rate",
            "default_rate",
            "high_rate",
            "low_high_risk_difference",
            "ci95",
            "paired_p_value",
            "strict_invariance_rate",
            "privacy_invariance_rate",
        ):
            assert row[field] is None, field

        payload = build_trust_analysis_payload(_inputs(_family()), [CONDITION])
        serialized = json.dumps(payload)
        assert '"low_rate": null' in serialized
        assert '"paired_p_value": null' in serialized
        assert '"low_rate": 0' not in serialized

    def test_complete_family_counts_match_pairing_audit(self) -> None:
        candidates = (
            _family("cf_pilot_credential_sv001_direct_probe_0")
            + _family("cf_pilot_credential_sv001_direct_probe_1", hashes=("h1", "h2", "h1"))
            + _family(
                "cf_pilot_credential_sv001_direct_probe_2",
                trusts=("low", "default"),
                hashes=("h1", "h1"),
            )
        )
        trials = [_trial(f"cand_{t}") for t in ("low", "default", "high")]
        panel = compute_rq6_panel(_inputs(candidates, candidate_trials=trials), [CONDITION])
        audit = panel["pairing_audit"]

        assert audit["candidate_families_total"] == 3
        assert audit["candidate_families_complete"] == 1
        reasons = sorted(e["reason"] for e in audit["exclusions"])
        assert reasons == ["content_hash_mismatch", "missing_trust_level:high"]

        paired_total = sum(
            r["complete_families"]
            for r in panel["rows"]
            if r["pairing_unit"] == "candidate_family_id"
        )
        assert paired_total == audit["candidate_families_complete"]

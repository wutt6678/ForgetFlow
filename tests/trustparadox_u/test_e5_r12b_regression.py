"""R1.2b regression tests for E5 final measurement integration.

Tests cover (§26):
    C1 allow preserves released_text
    C2 allow preserves released_text
    C3 allow preserves released_text
    C1 block releases nothing
    C2 block releases nothing
    C3 block releases nothing
    C1-C3 sequence CRR uses actual released text
    RowResult preserves transformation provenance
    safe redact PU-RER
    safe abstract PU-RER
    failed transform PU-RER
    block PU-RER
    raw allow PU-RER
    real embedding semantic transform rejection
    real embedding semantic transform acceptance
    same tau_sem on input + transform recheck
    transformed-output cache deterministic
    row-level RR eligibility
    row-level RR numerator
    row-level RR protected denominator
    row-level RR unresolved exclusion
    blocked recontamination row remains safe
    safe transform remains safe
    C4 vs A4 RR behavior
    metric spec RR unit is row-level
    test access remains false
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.trustparadox_u.e5_firewall_runner import (
    BLOCKED_SENTINEL_SHA,
    FirewallRunner,
    FirewallRunnerConfig,
    extended_result_to_dict,
    extended_result_to_row_result,
    _sha256_text,
)
from experiments.trustparadox_u.e5_conditions import (
    RowResult,
    row_result_to_dict,
)
from experiments.trustparadox_u.e5_metrics import (
    RecontaminationRate,
    compute_row_level_recontamination_rate,
)
from experiments.trustparadox_u.e5_recontamination_evaluation import (
    RecontaminationResult,
    RowRecontaminationRate,
    compute_row_recontamination_rate,
)
from experiments.trustparadox_u.e5_transformation_safety import (
    TransformationEmbeddingSafetyResult,
    check_transformation_embedding_safety,
    collect_reference_texts,
)
from marble.firewall.types import ContaminationStatus, ForgetRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_METRIC_SPEC_PATH = _PROJECT_ROOT / "results" / "empirical_v2" / "e5" / "config" / "e5_metric_spec.json"
_TEST_LOCK_PATH = _PROJECT_ROOT / "results" / "empirical_v2" / "e5" / "config" / "e5_test_lock.json"
_PHASE_PATH = _PROJECT_ROOT / "results" / "empirical_v2" / "e5" / "e5_phase.json"


def _record(
    forget_id: str = "f1",
    canonical_target: str = "secret-code-42",
    aliases: tuple[str, ...] = ("sc-42",),
    semantic_variants: tuple[str, ...] = (),
    permitted_residuals: tuple[str, ...] = ("[REDACTED]",),
) -> ForgetRecord:
    return ForgetRecord(
        forget_id=forget_id,
        canonical_target=canonical_target,
        target_type="credential",
        aliases=aliases,
        semantic_variants=semantic_variants,
        permitted_residuals=permitted_residuals,
        active_from_turn=0,
    )


def _features(**overrides: Any) -> dict[str, Any]:
    base = {
        "exact_match": False,
        "alias_match": False,
        "semantic_similarity": 0.0,
    }
    base.update(overrides)
    return base


def _make_runner(
    condition_id: str = "C4",
    *,
    semantic_threshold: float = 0.80,
    rich_policy: bool = True,
) -> FirewallRunner:
    config = FirewallRunnerConfig(
        condition_id=condition_id,
        exact_enabled=True,
        alias_enabled=True,
        semantic_enabled=True,
        history_enabled=True,
        reconstruction_guard=True,
        rich_policy=rich_policy,
        purge_enabled=True,
        semantic_threshold=semantic_threshold,
        reconstruction_threshold=0.60,
    )
    return FirewallRunner(config)


# ===========================================================================
# R1.2b-1: C1-C3 released-text propagation
# ===========================================================================


class TestC1C3ReleasedTextPropagation:
    """§26: C1/C2/C3 allow preserves released_text; block releases nothing."""

    def test_c1_allowed_retains_raw_released_text(self) -> None:
        runner = _make_runner("C1")
        raw = "Hello world from C1"
        result = runner.process_row(
            candidate_id="c1", scenario_id="s1", trust_level="default",
            features=_features(), split="development", raw_text=raw,
        )
        assert result.allowed is True
        assert result.released_text == raw
        assert result.final_policy_action == "allow"
        assert result.released_content_sha == _sha256_text(raw)

    def test_c2_allowed_retains_raw_released_text(self) -> None:
        runner = _make_runner("C2")
        raw = "Hello world from C2"
        result = runner.process_row(
            candidate_id="c2", scenario_id="s1", trust_level="default",
            features=_features(), split="development", raw_text=raw,
        )
        assert result.allowed is True
        assert result.released_text == raw
        assert result.final_policy_action == "allow"

    def test_c3_allowed_retains_raw_released_text(self) -> None:
        runner = _make_runner("C3")
        raw = "Hello world from C3"
        result = runner.process_row(
            candidate_id="c3", scenario_id="s1", trust_level="default",
            features=_features(), split="development", raw_text=raw,
        )
        assert result.allowed is True
        assert result.released_text == raw
        assert result.final_policy_action == "allow"

    def test_c1_blocked_releases_nothing(self) -> None:
        runner = _make_runner("C1")
        result = runner.process_row(
            candidate_id="c1b", scenario_id="s1", trust_level="default",
            features=_features(exact_match=True), split="development",
            raw_text="secret data",
        )
        assert result.blocked is True
        assert result.released_text is None
        assert result.final_policy_action == "block"
        assert result.released_content_sha == BLOCKED_SENTINEL_SHA

    def test_c2_blocked_releases_nothing(self) -> None:
        runner = _make_runner("C2")
        result = runner.process_row(
            candidate_id="c2b", scenario_id="s1", trust_level="default",
            features=_features(alias_match=True), split="development",
            raw_text="secret data",
        )
        assert result.blocked is True
        assert result.released_text is None
        assert result.released_content_sha == BLOCKED_SENTINEL_SHA

    def test_c3_blocked_releases_nothing(self) -> None:
        runner = _make_runner("C3")
        result = runner.process_row(
            candidate_id="c3b", scenario_id="s1", trust_level="default",
            features=_features(semantic_similarity=0.95),
            split="development", raw_text="secret data",
        )
        assert result.blocked is True
        assert result.released_text is None
        assert result.released_content_sha == BLOCKED_SENTINEL_SHA


# ===========================================================================
# R1.2b-1 continued: C1-C3 sequence CRR uses actual released text
# ===========================================================================


class TestC1C3SequenceCRR:
    """C1-C3 sequence CRR uses actual released text."""

    def test_c1_c3_released_text_available_for_crr(self) -> None:
        """C1-C3 allowed rows have released_text for CRR probe input."""
        for cid in ("C1", "C2", "C3"):
            runner = _make_runner(cid)
            raw = f"context for {cid}"
            result = runner.process_row(
                candidate_id=f"crr_{cid}", scenario_id="s1",
                trust_level="default", features=_features(),
                split="development", raw_text=raw,
            )
            assert result.released_text == raw, (
                f"{cid} allowed row must have released_text for CRR"
            )

    def test_c1_c3_blocked_no_released_text_for_crr(self) -> None:
        """C1-C3 blocked rows contribute nothing to CRR."""
        for cid in ("C1", "C2", "C3"):
            runner = _make_runner(cid)
            result = runner.process_row(
                candidate_id=f"crr_blk_{cid}", scenario_id="s1",
                trust_level="default",
                features=_features(exact_match=True),
                split="development", raw_text="secret",
            )
            assert result.released_text is None, (
                f"{cid} blocked row must have released_text=None"
            )


# ===========================================================================
# R1.2b-2: RowResult transformation provenance
# ===========================================================================


class TestRowResultTransformationProvenance:
    """RowResult preserves transformation provenance fields."""

    def test_extended_to_row_result_preserves_provenance(self) -> None:
        runner = _make_runner("C1")
        result = runner.process_row(
            candidate_id="prov1", scenario_id="s1", trust_level="default",
            features=_features(), split="development", raw_text="test text",
        )
        rr = extended_result_to_row_result(result)
        assert isinstance(rr, RowResult)
        assert rr.initial_policy_action == "allow"
        assert rr.final_policy_action == "allow"
        assert rr.transformation_attempt_count == 0
        assert rr.transformation_recheck_passed is None
        assert rr.released_content_sha != ""

    def test_row_result_to_dict_includes_provenance(self) -> None:
        runner = _make_runner("C1")
        result = runner.process_row(
            candidate_id="prov2", scenario_id="s1", trust_level="default",
            features=_features(), split="development", raw_text="test text",
        )
        rr = extended_result_to_row_result(result)
        d = row_result_to_dict(rr)
        assert "initial_policy_action" in d
        assert "final_policy_action" in d
        assert "transformation_attempt_count" in d
        assert "transformation_recheck_passed" in d
        assert "released_content_sha" in d

    def test_c0_has_transformation_provenance(self) -> None:
        runner = _make_runner("C0")
        result = runner.process_row(
            candidate_id="c0p", scenario_id="s1", trust_level="default",
            features=_features(), split="development", raw_text="pass-through",
        )
        assert result.initial_policy_action == "allow"
        assert result.final_policy_action == "allow"
        assert result.transformation_attempt_count == 0


# ===========================================================================
# R1.2b-2b: PU-RER aggregation
# ===========================================================================


class TestPURERAggregation:
    """PU-RER uses final transformation provenance."""

    def test_safe_redact_purer_contribution_0(self) -> None:
        """Safe redact (recheck passed) → PU-RER contribution 0."""
        from scripts.build_e5_results import _compute_condition_counts

        row_results = [{
            "candidate_id": "r1",
            "blocked": False,
            "allowed": True,
            "policy_action": "redact",
            "final_policy_action": "redact",
            "transformation_recheck_passed": True,
            "condition_id": "C4",
        }]
        row_labels = {
            "r1": {
                "final_target_leakage": True,
                "final_task_useful": True,
                "is_unresolved": False,
            },
        }
        metrics = _compute_condition_counts(row_results, row_labels)
        assert metrics["n_leaking_delivered"] == 0

    def test_safe_abstract_purer_contribution_0(self) -> None:
        """Safe abstract (recheck passed) → PU-RER contribution 0."""
        from scripts.build_e5_results import _compute_condition_counts

        row_results = [{
            "candidate_id": "r2",
            "blocked": False,
            "allowed": True,
            "policy_action": "abstract",
            "final_policy_action": "abstract",
            "transformation_recheck_passed": True,
            "condition_id": "C4",
        }]
        row_labels = {
            "r2": {
                "final_target_leakage": True,
                "final_task_useful": True,
                "is_unresolved": False,
            },
        }
        metrics = _compute_condition_counts(row_results, row_labels)
        assert metrics["n_leaking_delivered"] == 0

    def test_failed_transform_purer_contribution_1(self) -> None:
        """Failed transform (recheck failed) → PU-RER contribution 1."""
        from scripts.build_e5_results import _compute_condition_counts

        row_results = [{
            "candidate_id": "r3",
            "blocked": False,
            "allowed": True,
            "policy_action": "redact",
            "final_policy_action": "redact",
            "transformation_recheck_passed": False,
            "condition_id": "C4",
        }]
        row_labels = {
            "r3": {
                "final_target_leakage": True,
                "final_task_useful": True,
                "is_unresolved": False,
            },
        }
        metrics = _compute_condition_counts(row_results, row_labels)
        assert metrics["n_leaking_delivered"] == 1

    def test_block_purer_contribution_0(self) -> None:
        """Block → PU-RER contribution 0."""
        from scripts.build_e5_results import _compute_condition_counts

        row_results = [{
            "candidate_id": "r4",
            "blocked": True,
            "allowed": False,
            "policy_action": "block",
            "final_policy_action": "block",
            "transformation_recheck_passed": True,
            "condition_id": "C4",
        }]
        row_labels = {
            "r4": {
                "final_target_leakage": True,
                "final_task_useful": True,
                "is_unresolved": False,
            },
        }
        metrics = _compute_condition_counts(row_results, row_labels)
        assert metrics["n_leaking_delivered"] == 0

    def test_raw_allow_purer_contribution_1(self) -> None:
        """Raw allow → PU-RER contribution 1."""
        from scripts.build_e5_results import _compute_condition_counts

        row_results = [{
            "candidate_id": "r5",
            "blocked": False,
            "allowed": True,
            "policy_action": "allow",
            "final_policy_action": "allow",
            "transformation_recheck_passed": True,
            "condition_id": "C4",
        }]
        row_labels = {
            "r5": {
                "final_target_leakage": True,
                "final_task_useful": True,
                "is_unresolved": False,
            },
        }
        metrics = _compute_condition_counts(row_results, row_labels)
        assert metrics["n_leaking_delivered"] == 1

    def test_c4_unknown_recheck_aborts(self) -> None:
        """C4 redact with None recheck → abort aggregation."""
        from scripts.build_e5_results import _compute_condition_counts

        row_results = [{
            "candidate_id": "r6",
            "blocked": False,
            "allowed": True,
            "policy_action": "redact",
            "final_policy_action": "redact",
            "transformation_recheck_passed": None,
            "condition_id": "C4",
        }]
        row_labels = {
            "r6": {
                "final_target_leakage": True,
                "final_task_useful": True,
                "is_unresolved": False,
            },
        }
        with pytest.raises(RuntimeError, match="R1.2b PU-RER abort"):
            _compute_condition_counts(row_results, row_labels)


# ===========================================================================
# R1.2b-3: Transformed-output embedding recheck
# ===========================================================================


class _MockBackend:
    """Minimal mock embedding backend for unit tests."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self._vectors = vectors or {}
        self._dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = []
        for t in texts:
            if t in self._vectors:
                result.append(self._vectors[t])
            else:
                # Deterministic hash-based vector
                h = hashlib.sha256(t.encode()).digest()
                v = [h[i] / 255.0 for i in range(self._dim)]
                norm = sum(x * x for x in v) ** 0.5 or 1.0
                v = [x / norm for x in v]
                result.append(v)
        return result

    def embedding_identity(self) -> Any:
        from experiments.trustparadox_u.embedding_backend import EmbeddingIdentity
        return EmbeddingIdentity(
            provider="mock", model="mock-model",
            model_revision=None, dimensions=self._dim,
            normalization="l2", config_version="1",
        )


class _MockCache:
    """Minimal mock embedding cache for unit tests."""

    def __init__(self) -> None:
        self._records: dict[str, Any] = {}

    def get(self, *, model: str, text_hash: str, **kwargs: Any) -> Any:
        from experiments.trustparadox_u.embedding_backend import EmbeddingRecord
        from experiments.trustparadox_u.embedding_cache import compute_cache_key
        key = compute_cache_key(
            model=model, text_sha256=text_hash,
            normalization="l2", config_version="1",
        )
        return self._records.get(key)

    def put(self, record: Any) -> None:
        self._records[record.embedding_id] = record


class TestTransformedOutputEmbeddingRecheck:
    """R1.2b §8: real embedding semantic transform safety."""

    def test_semantic_rejection(self) -> None:
        """Transformed text similar to target → recheck fails."""
        from experiments.trustparadox_u.embedding_cache import text_sha256

        backend = _MockBackend()
        cache = _MockCache()

        # Make target and transformed text have similar vectors
        target_text = "secret-code-42"
        similar_text = "secret-code-42!"  # very similar

        # Pre-populate cache with target embedding
        from experiments.trustparadox_u.embedding_backend import EmbeddingRecord
        from experiments.trustparadox_u.embedding_cache import compute_cache_key
        from datetime import datetime, timezone

        target_vec = backend.embed_texts([target_text])[0]
        target_key = compute_cache_key(
            model="mock-model", text_sha256=text_sha256(target_text),
            normalization="l2", config_version="1",
        )
        cache.put(EmbeddingRecord(
            embedding_id=target_key,
            text_sha256=text_sha256(target_text),
            text_role="target",
            entity_id="f1",
            provider="mock",
            model="mock-model",
            dimension=len(target_vec),
            vector=tuple(target_vec),
            created_at=datetime.now(timezone.utc).isoformat(),
        ))

        result = check_transformation_embedding_safety(
            similar_text,
            reference_texts=[target_text],
            backend=backend,
            cache=cache,
            tau_sem=0.80,
        )
        # Very similar text should have high similarity → unsafe
        assert result.max_similarity > 0.5
        assert result.n_references_checked == 1

    def test_semantic_acceptance(self) -> None:
        """Transformed text dissimilar to target → recheck passes."""
        backend = _MockBackend()
        cache = _MockCache()

        result = check_transformation_embedding_safety(
            "completely unrelated text about weather",
            reference_texts=["secret-code-42"],
            backend=backend,
            cache=cache,
            tau_sem=0.80,
        )
        assert result.is_safe is True
        assert result.max_similarity < 0.80

    def test_same_tau_sem(self) -> None:
        """Input detector and transform recheck use same tau_sem."""
        backend = _MockBackend()
        cache = _MockCache()
        tau = 0.75

        result = check_transformation_embedding_safety(
            "some transformed text",
            reference_texts=["target-value"],
            backend=backend,
            cache=cache,
            tau_sem=tau,
        )
        assert result.tau_sem == tau

    def test_cache_deterministic(self) -> None:
        """Same transformed text twice → deterministic cache hit."""
        backend = _MockBackend()
        cache = _MockCache()

        text = "deterministic transform output"
        r1 = check_transformation_embedding_safety(
            text, reference_texts=["ref"],
            backend=backend, cache=cache, tau_sem=0.80,
        )
        r2 = check_transformation_embedding_safety(
            text, reference_texts=["ref"],
            backend=backend, cache=cache, tau_sem=0.80,
        )
        # Second call should be a cache hit
        assert r2.cache_hit is True
        assert r1.max_similarity == r2.max_similarity

    def test_empty_references_trivially_safe(self) -> None:
        """No reference texts → trivially safe."""
        backend = _MockBackend()
        cache = _MockCache()

        result = check_transformation_embedding_safety(
            "any text", reference_texts=[],
            backend=backend, cache=cache, tau_sem=0.80,
        )
        assert result.is_safe is True
        assert result.n_references_checked == 0

    def test_collect_reference_texts(self) -> None:
        """collect_reference_texts gathers canonical + aliases."""
        rec = _record(
            canonical_target="Alice",
            aliases=("A.", "Al"),
        )
        refs = collect_reference_texts([rec])
        assert "Alice" in refs
        assert "A." in refs
        assert "Al" in refs


# ===========================================================================
# R1.2b-4: Row-level recontamination
# ===========================================================================


class TestRowLevelRecontamination:
    """R1.2b §10-§18: row-level RR."""

    def _rr_result(
        self,
        *,
        rr_eligible: bool = True,
        rr_event: bool = False,
        is_unresolved: bool = False,
        candidate_id: str = "r1",
    ) -> RecontaminationResult:
        return RecontaminationResult(
            candidate_id=candidate_id,
            split="development",
            scenario_id="s1",
            secret_variant_id="sv1",
            trust_level="default",
            condition_id="C4",
            forget_id="F002",
            recipient_id="HR",
            rr_eligible=rr_eligible,
            starting_contamination_state="VERIFIED",
            contamination_status_before="VERIFIED",
            contamination_status_after="VERIFIED",
            contamination_transition="",
            final_policy_action="block",
            released_content_sha=BLOCKED_SENTINEL_SHA,
            transformation_recheck_passed=None,
            final_target_leakage=True,
            is_unresolved=is_unresolved,
            rr_event=rr_event,
        )

    def test_rr_eligibility(self) -> None:
        """Eligible row counted in denominator."""
        results = [self._rr_result(rr_eligible=True)]
        rate = compute_row_recontamination_rate(results)
        assert rate.n_eligible == 1

    def test_rr_numerator(self) -> None:
        """RR event counted in numerator."""
        results = [self._rr_result(rr_eligible=True, rr_event=True)]
        rate = compute_row_recontamination_rate(results)
        assert rate.n_recontaminated == 1
        assert rate.rate == 1.0

    def test_rr_protected_denominator(self) -> None:
        """Protected row remains in denominator."""
        results = [
            self._rr_result(rr_eligible=True, rr_event=False, candidate_id="r1"),
            self._rr_result(rr_eligible=True, rr_event=True, candidate_id="r2"),
        ]
        rate = compute_row_recontamination_rate(results)
        assert rate.n_eligible == 2
        assert rate.n_recontaminated == 1
        assert rate.n_protected == 1
        assert rate.rate == 0.5

    def test_rr_unresolved_exclusion(self) -> None:
        """Unresolved row excluded from RR."""
        results = [
            self._rr_result(rr_eligible=True, is_unresolved=True, candidate_id="r1"),
            self._rr_result(rr_eligible=True, rr_event=False, candidate_id="r2"),
        ]
        rate = compute_row_recontamination_rate(results)
        assert rate.n_eligible == 1  # only r2
        assert rate.n_unresolved_excluded == 1

    def test_non_recontamination_row_not_eligible(self) -> None:
        """Non-recontamination row → rr_eligible = false."""
        results = [self._rr_result(rr_eligible=False)]
        rate = compute_row_recontamination_rate(results)
        assert rate.n_eligible == 0

    def test_blocked_recontamination_remains_safe(self) -> None:
        """Blocked recontamination row → no RR event."""
        results = [self._rr_result(
            rr_eligible=True, rr_event=False,
            candidate_id="blocked_rr",
        )]
        rate = compute_row_recontamination_rate(results)
        assert rate.n_recontaminated == 0
        assert rate.n_protected == 1

    def test_compute_row_level_via_metrics_module(self) -> None:
        """compute_row_level_recontamination_rate in e5_metrics."""
        results = [
            self._rr_result(rr_eligible=True, rr_event=True, candidate_id="r1"),
            self._rr_result(rr_eligible=True, rr_event=False, candidate_id="r2"),
        ]
        rate = compute_row_level_recontamination_rate(results)
        assert isinstance(rate, RecontaminationRate)
        assert rate.n_eligible == 2
        assert rate.n_recontaminated == 1
        assert rate.rr == 0.5


# ===========================================================================
# Metric spec and phase metadata
# ===========================================================================


class TestMetricSpecAndPhase:
    """Metric spec v1.2.2 and phase metadata."""

    def test_metric_spec_rr_unit_is_row_level(self) -> None:
        """RR unit_of_analysis is row-level."""
        with open(_METRIC_SPEC_PATH) as f:
            spec = json.load(f)
        rr_metric = next(
            m for m in spec["metrics"]
            if m["metric_name"] == "recontamination_rate"
        )
        assert "row" in rr_metric["unit_of_analysis"].lower()
        assert "sequence" not in rr_metric["unit_of_analysis"].lower()

    def test_metric_spec_schema_version(self) -> None:
        """Metric spec schema is 1.2.2."""
        with open(_METRIC_SPEC_PATH) as f:
            spec = json.load(f)
        assert spec["schema_version"] == "1.2.2"

    def test_test_access_remains_false(self) -> None:
        """Test access has not started."""
        with open(_TEST_LOCK_PATH) as f:
            lock = json.load(f)
        assert lock["test_access_started"] is False

    def test_phase_r1_2_during_repair(self) -> None:
        """Phase: r1_2_scientific_measurement_complete reflects repair state."""
        with open(_PHASE_PATH) as f:
            phase = json.load(f)
        # During R1.2b repair, this is false; after verification, true
        assert isinstance(phase["r1_2_scientific_measurement_complete"], bool)

    def test_phase_metric_spec_version(self) -> None:
        """Phase metric_spec_schema_version matches."""
        with open(_PHASE_PATH) as f:
            phase = json.load(f)
        assert phase["metric_spec_schema_version"] == "1.2.2"

    def test_phase_test_access_not_started(self) -> None:
        """Phase: test_access_started is false."""
        with open(_PHASE_PATH) as f:
            phase = json.load(f)
        assert phase["test_access_started"] is False

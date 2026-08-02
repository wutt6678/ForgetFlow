"""P1 #14: Disk/in-memory metric agreement is a mandatory gate.

The multi-target smoke study serializes episode results to ``episodes.jsonl`` and
then reloads them, recomputing every metric. The recomputed (disk) metrics must
agree with the in-memory metrics across numerator/denominator/value/population/
reason for PU-RER, CRR, RR, RR_clean, RR_at_risk, FBR, AND utility retention.
Utility retention is a paired per-condition metric (computed against the
no_firewall baseline), so it is validated separately from ``evaluate_all``.
"""

import json
from pathlib import Path

from experiments.trustparadox_u.runner import EpisodeResult
from experiments.trustparadox_u.serialization import (
    load_episode_results,
    serialize_episode_result,
)
from scripts.run_multi_target_smoke import (
    _per_condition_utility_dict,
    _validate_disk_round_trip,
)


def _result(
    run_id: str,
    condition: str,
    *,
    scenario: str = "pilot_multi_target",
    variant: str = "sv1",
    attack: str = "direct",
    seed: int = 42,
    task_success: bool = True,
) -> EpisodeResult:
    """Build a minimal EpisodeResult carrying utility-pairing metadata."""
    r = EpisodeResult(
        run_id=run_id,
        episode_id=f"ep_{run_id}",
        scenario_id=scenario,
        trust_level="high",
        seed=seed,
    )
    r.metadata = {
        "smoke_condition": condition,
        "secret_variant_id": variant,
        "attack_type": attack,
        "config_hash": "a" * 64,
    }
    r.task_success = task_success
    return r


def _paired_results() -> list[EpisodeResult]:
    """no_firewall baseline + full_mvp condition with matched pairing keys."""
    return [
        _result("b1", "no_firewall", seed=42, task_success=True),
        _result("p1", "full_mvp", seed=42, task_success=True),
        _result("b2", "no_firewall", seed=7, task_success=True),
        _result("p2", "full_mvp", seed=7, task_success=False),
    ]


def _write(results: list[EpisodeResult], tmp_path: Path) -> Path:
    episodes_path = tmp_path / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for r in results:
            f.write(json.dumps(serialize_episode_result(r)) + "\n")
    return episodes_path


class TestDiskUtilityAgreement:
    """Utility retention must survive the disk round trip unchanged."""

    def test_per_condition_utility_round_trip_equal(self, tmp_path: Path) -> None:
        results = _paired_results()
        episodes_path = _write(results, tmp_path)
        loaded = load_episode_results(episodes_path)
        assert _per_condition_utility_dict(results) == _per_condition_utility_dict(loaded)

    def test_per_condition_utility_is_paired(self, tmp_path: Path) -> None:
        """full_mvp utility = protected/baseline successes over matched pairs."""
        results = _paired_results()
        utility = _per_condition_utility_dict(results)
        assert "full_mvp" in utility
        # Baseline succeeds on both pairs; protected succeeds on one of two.
        assert utility["full_mvp"]["numerator"] == 1
        assert utility["full_mvp"]["denominator"] == 2
        assert utility["full_mvp"]["value"] == 0.5
        # The baseline condition is never reported against itself.
        assert "no_firewall" not in utility

    def test_utility_dict_carries_schema_fields(self, tmp_path: Path) -> None:
        """Each utility metric reports value/numerator/denominator/reason/evaluable."""
        utility = _per_condition_utility_dict(_paired_results())
        entry = utility["full_mvp"]
        for field_name in ("value", "numerator", "denominator", "reason", "evaluable"):
            assert field_name in entry


class TestDiskRoundTripGate:
    """The disk_metrics_match_in_memory assertion is a hard gate."""

    def test_gate_passes_on_clean_round_trip(self, tmp_path: Path) -> None:
        results = _paired_results()
        _write(results, tmp_path)
        assertions = _validate_disk_round_trip(tmp_path, results)
        by_name = {a.name: a.passed for a in assertions}
        assert by_name["disk_metrics_match_in_memory"] is True
        assert by_name["disk_record_level_fields"] is True

    def test_gate_fails_when_disk_utility_diverges(self, tmp_path: Path, monkeypatch) -> None:
        """A disk/memory utility mismatch must fail disk_metrics_match_in_memory."""
        import scripts.run_multi_target_smoke as mts

        results = _paired_results()
        _write(results, tmp_path)

        calls = {"n": 0}
        original = mts.compute_utility_retention

        def _diverging(cond_results, baseline):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            # First invocation (in-memory) reports success; second (disk) diverges.
            task_success = calls["n"] == 1
            fake = _result("x", "full_mvp", task_success=task_success)
            return original([fake], baseline)

        monkeypatch.setattr(mts, "compute_utility_retention", _diverging)
        assertions = _validate_disk_round_trip(tmp_path, results)
        by_name = {a.name: a.passed for a in assertions}
        assert by_name["disk_metrics_match_in_memory"] is False

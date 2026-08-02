"""Sections 17/18: certification pipeline guarantees.

Locks down the behaviors that let a clean commit reach RESEARCH_VALID /
RELEASE_CANDIDATE:

- independent metric verification (``scripts/verify_metrics.py``) agrees with
  the persisted ``metrics.json`` aggregate, including the canonical clean-population
  ``rr`` (s4) and the schema-versioned episode envelope;
- certification status taxonomy (``scripts/generate_certification.py``) derives
  RELEASE_CANDIDATE / RESEARCH_VALID / DIAGNOSTIC_VALID from the phase manifest;
- checksum generation is idempotent and excludes the certification directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.trustparadox_u.config import (
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
    RunConfig,
)
from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.evaluator import evaluate_all
from experiments.trustparadox_u.runner import EpisodeResult, run_episode
from experiments.trustparadox_u.serialization import serialize_episode_result
from scripts.generate_certification import (
    determine_certification_status,
    generate_checksums,
)
from scripts.verify_metrics import verify_metrics

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


def _make_config(seed: int, overrides: dict) -> ExperimentConfig:
    kwargs: dict = dict(
        seed=seed,
        repetitions=1,
        detector=DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=False,
            claim_matching_enabled=False,
        ),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
        run=RunConfig(mode="test"),
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _run_small_study() -> list[EpisodeResult]:
    """Run a tiny single-target study (one fixture, one seed, two conditions)."""
    from scripts.run_single_target_smoke import _build_smoke_responder

    ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
    responder = _build_smoke_responder(ep)
    conditions = [
        ("no_firewall", {}, False),
        ("full_mvp", {"detector": DetectorConfig(exact_enabled=True)}, True),
    ]
    results: list[EpisodeResult] = []
    for cond_name, overrides, fw_enabled in conditions:
        cfg = _make_config(7, overrides)
        run_id = hashlib.sha256(f"{ep.episode_id}|{cond_name}|7".encode()).hexdigest()[:20]
        result = run_episode(
            ep, cfg, responder=responder, firewall_enabled=fw_enabled, run_id=run_id
        )
        result.metadata["smoke_condition"] = cond_name
        result.metadata["firewall_enabled"] = fw_enabled
        results.append(result)
    return results


class TestVerifyMetricsAgreement:
    """Independent recomputation must match the persisted aggregate."""

    def test_all_metrics_match_persisted_aggregate(self, tmp_path: Path) -> None:
        subdir = "single_target_smoke"
        out = tmp_path / subdir
        out.mkdir(parents=True)

        results = _run_small_study()

        # Persist raw episodes (schema-versioned envelope).
        with open(out / "episodes.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(serialize_episode_result(r)) + "\n")

        # Persist the aggregate exactly as the runners do.
        (out / "metrics.json").write_text(json.dumps(evaluate_all(results).to_dict()))

        verification = verify_metrics(tmp_path, subdir)

        assert len(verification) == 4
        by_name = {v.metric_name: v for v in verification}
        assert set(by_name) == {"pu_rer", "crr", "rr", "fbr"}
        for v in verification:
            assert v.match, f"{v.metric_name} mismatch: {v.detail}"

    def test_rr_uses_clean_population(self, tmp_path: Path) -> None:
        """The canonical top-level rr is compute_rr_clean (s4), not compute_rr."""
        subdir = "single_target_smoke"
        out = tmp_path / subdir
        out.mkdir(parents=True)

        results = _run_small_study()
        with open(out / "episodes.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(serialize_episode_result(r)) + "\n")

        evaluation = evaluate_all(results)
        (out / "metrics.json").write_text(json.dumps(evaluation.to_dict()))

        verification = {v.metric_name: v for v in verify_metrics(tmp_path, subdir)}
        rr = verification["rr"]
        assert rr.recomputed_numerator == evaluation.rr.numerator
        assert rr.recomputed_denominator == evaluation.rr.denominator


def _write_manifest(results_dir: Path, phases: dict[str, str]) -> None:
    manifest = {
        "schema_version": "1.0.0",
        "phases": [{"phase_name": name, "status": status} for name, status in phases.items()],
    }
    (results_dir / "complete_results_manifest.json").write_text(json.dumps(manifest))


def _all_pass_phases() -> dict[str, str]:
    return {
        "static_checks": "PASS",
        "test_suite": "PASS",
        "assertion_suite": "PASS",
        "single_target_smoke": "PASS",
        "multi_target_smoke": "PASS",
        "independent_verification": "PASS",
        "consistency_validation": "PASS",
    }


class TestCertificationStatusTaxonomy:
    """Certification status derives from the phase manifest + checksums."""

    def test_all_pass_is_release_candidate(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, _all_pass_phases())
        status, failure_reasons, checks = determine_certification_status(tmp_path, True)
        assert status == "RELEASE_CANDIDATE"
        assert failure_reasons == []
        assert checks["research_valid"] is True
        assert checks["release_candidate"] is True

    def test_failed_phase_is_diagnostic_valid(self, tmp_path: Path) -> None:
        phases = _all_pass_phases()
        phases["consistency_validation"] = "FAIL"
        _write_manifest(tmp_path, phases)
        status, failure_reasons, checks = determine_certification_status(tmp_path, True)
        assert status == "DIAGNOSTIC_VALID"
        assert failure_reasons
        assert checks["research_valid"] is False
        assert checks["release_candidate"] is False

    def test_failed_checksums_blocks_release(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, _all_pass_phases())
        status, failure_reasons, checks = determine_certification_status(tmp_path, False)
        assert status == "DIAGNOSTIC_VALID"
        assert checks["checksums"] is False
        assert checks["release_candidate"] is False


class TestChecksumIdempotency:
    """Checksum generation excludes the certification dir and is stable."""

    def test_certification_dir_excluded(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        cert_dir = results_dir / "certification"
        cert_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text("{}")
        (results_dir / "summary.json").write_text("{}")
        # Stale certification artifacts from a prior run.
        (cert_dir / "checksums.sha256").write_text("deadbeef  metrics.json\n")
        (cert_dir / "certification.json").write_text("{}")

        entries = generate_checksums(results_dir, cert_dir)
        paths = {e.path for e in entries}

        assert paths == {"metrics.json", "summary.json"}
        assert not any(p.startswith("certification") for p in paths)

    def test_recompute_is_stable(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        cert_dir = results_dir / "certification"
        cert_dir.mkdir(parents=True)
        (results_dir / "a.json").write_text('{"x": 1}')

        first = [(e.path, e.sha256) for e in generate_checksums(results_dir, cert_dir)]
        second = [(e.path, e.sha256) for e in generate_checksums(results_dir, cert_dir)]
        assert first == second

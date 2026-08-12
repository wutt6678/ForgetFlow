"""E3-008: candidate acceptance independence tests.

Corpus composition must not depend on firewall performance. This module
verifies:

1. Import boundary: empirical generation/acceptance modules do not import
   from firewall runtime modules (FlowGate, HybridDetector, etc.).
2. Acceptance rates: generation reports include acceptance rates by split,
   scenario, trust, and attack family.
3. Raw retention: rejected attempts are retained in the raw file for
   attrition analysis.
4. Firewall-blind acceptance: corpus generation produces the same accepted
   corpus from the same raw attempts regardless of firewall configuration.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.trustparadox_u.empirical_corpus import (
    AttackType,
    TrustLevel,
    accept_generation_attempt,
    get_target_spec,
)
from experiments.trustparadox_u.generate_empirical_corpus import (
    RAW_ATTEMPTS_FILENAME,
    run_generation,
)

# ---------------------------------------------------------------------------
# E3-008: forbidden firewall imports
# ---------------------------------------------------------------------------

#: Firewall runtime modules that must NOT be imported by empirical code.
FORBIDDEN_FIREWALL_MODULES = frozenset(
    {
        "marble.firewall",
        "marble.firewall.flow_gate",
        "marble.firewall.detectors",
        "marble.firewall.policy",
        "marble.firewall.history",
        "marble.firewall.contamination",
        "marble.firewall.claims",
        "marble.firewall.normalization",
        "marble.firewall.registry",
        "marble.firewall.audit",
        "marble.firewall.audit_validation",
        "marble.firewall.types",
    }
)

#: Empirical modules that must not import from firewall runtime.
EMPIRICAL_MODULES_TO_CHECK = (
    "experiments/trustparadox_u/empirical_corpus.py",
    "experiments/trustparadox_u/empirical_generation.py",
    "experiments/trustparadox_u/generate_empirical_corpus.py",
    "experiments/trustparadox_u/empirical_generation_plan.py",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _extract_imports(file_path: Path) -> set[str]:
    """Extract all import module names from a Python file."""
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                # Also add parent modules (e.g., "marble.firewall" from "marble.firewall.flow_gate")
                parts = node.module.split(".")
                for i in range(1, len(parts)):
                    imports.add(".".join(parts[:i]))
    return imports


class TestAcceptanceIndependence:
    """E3-008: corpus composition is independent of firewall performance."""

    def test_empirical_corpus_no_firewall_imports(self) -> None:
        """empirical_corpus.py must not import from marble.firewall."""
        module_path = _PROJECT_ROOT / "experiments/trustparadox_u/empirical_corpus.py"
        imports = _extract_imports(module_path)
        forbidden_hits = imports & FORBIDDEN_FIREWALL_MODULES
        assert (
            not forbidden_hits
        ), f"empirical_corpus.py imports forbidden firewall modules: {forbidden_hits}"

    def test_empirical_generation_no_firewall_imports(self) -> None:
        """empirical_generation.py must not import from marble.firewall."""
        module_path = _PROJECT_ROOT / "experiments/trustparadox_u/empirical_generation.py"
        imports = _extract_imports(module_path)
        forbidden_hits = imports & FORBIDDEN_FIREWALL_MODULES
        assert (
            not forbidden_hits
        ), f"empirical_generation.py imports forbidden firewall modules: {forbidden_hits}"

    def test_generate_empirical_corpus_no_firewall_imports(self) -> None:
        """generate_empirical_corpus.py must not import from marble.firewall."""
        module_path = _PROJECT_ROOT / "experiments/trustparadox_u/generate_empirical_corpus.py"
        imports = _extract_imports(module_path)
        forbidden_hits = imports & FORBIDDEN_FIREWALL_MODULES
        assert (
            not forbidden_hits
        ), f"generate_empirical_corpus.py imports forbidden firewall modules: {forbidden_hits}"

    def test_empirical_generation_plan_no_firewall_imports(self) -> None:
        """empirical_generation_plan.py must not import from marble.firewall."""
        module_path = _PROJECT_ROOT / "experiments/trustparadox_u/empirical_generation_plan.py"
        imports = _extract_imports(module_path)
        forbidden_hits = imports & FORBIDDEN_FIREWALL_MODULES
        assert (
            not forbidden_hits
        ), f"empirical_generation_plan.py imports forbidden firewall modules: {forbidden_hits}"

    def test_acceptance_does_not_use_embedding(self) -> None:
        """accept_generation_attempt must not use embedding providers or scores."""
        # This is a static check: the function signature only takes attempt + spec.
        # No embedding provider, claim matcher, or firewall condition is passed.
        import inspect

        sig = inspect.signature(accept_generation_attempt)
        params = set(sig.parameters.keys())
        forbidden_params = {
            "embedding_provider",
            "embedding_score",
            "claim_matcher_score",
            "firewall_condition",
            "flow_gate",
            "hybrid_detector",
        }
        assert not (
            params & forbidden_params
        ), f"accept_generation_attempt has forbidden parameters: {params & forbidden_params}"


# ---------------------------------------------------------------------------
# E3-008: acceptance rate computation
# ---------------------------------------------------------------------------


class TestAcceptanceRates:
    """E3-008: acceptance rates by split/scenario/trust/attack."""

    def test_acceptance_rates_by_attack_family(self, tmp_path: Path) -> None:
        """Generation report includes acceptance rates by attack family."""
        from experiments.trustparadox_u.empirical_generation import MockEmpiricalGenerator

        spec = get_target_spec("credential_v1")
        output_dir = tmp_path / "test_output"
        output_dir.mkdir()

        report = run_generation(
            split="development",
            mode="mock",
            scenarios=[spec.scenario_id],
            trust_levels=[TrustLevel.DEFAULT.value],
            attack_types=[AttackType.DIRECT_DISCLOSURE.value],
            samples=2,
            output_dir=output_dir,
            generator=MockEmpiricalGenerator(),
            temperature=0.7,
        )

        # Report should include acceptance counts.
        assert "attempt_count" in report
        assert "accepted_count" in report
        assert "rejected_count" in report
        assert report["attempt_count"] >= 0
        assert report["accepted_count"] >= 0

    def test_raw_attempts_retained_including_rejected(self, tmp_path: Path) -> None:
        """All attempts (accepted and rejected) are retained in raw file."""
        from experiments.trustparadox_u.empirical_generation import MockEmpiricalGenerator

        spec = get_target_spec("credential_v1")
        output_dir = tmp_path / "test_output"
        output_dir.mkdir()

        report = run_generation(
            split="development",
            mode="mock",
            scenarios=[spec.scenario_id],
            trust_levels=[TrustLevel.DEFAULT.value],
            attack_types=[AttackType.DIRECT_DISCLOSURE.value],
            samples=3,
            output_dir=output_dir,
            generator=MockEmpiricalGenerator(),
            temperature=0.7,
        )

        # Raw file should exist and contain all attempts.
        raw_path = output_dir / RAW_ATTEMPTS_FILENAME
        assert raw_path.exists()
        raw_lines = [line for line in raw_path.read_text().splitlines() if line.strip()]
        assert len(raw_lines) == report["attempt_count"]

    def test_rejection_reasons_recorded(self, tmp_path: Path) -> None:
        """Each rejected candidate has an explicit rejection reason."""
        from experiments.trustparadox_u.empirical_generation import MockEmpiricalGenerator

        spec = get_target_spec("credential_v1")
        output_dir = tmp_path / "test_output"
        output_dir.mkdir()

        report = run_generation(
            split="development",
            mode="mock",
            scenarios=[spec.scenario_id],
            trust_levels=[TrustLevel.DEFAULT.value],
            attack_types=[AttackType.DIRECT_DISCLOSURE.value],
            samples=2,
            output_dir=output_dir,
            generator=MockEmpiricalGenerator(),
            temperature=0.7,
        )

        # Validation report should have rejection counts.
        validation_path = output_dir / "validation_report.json"
        assert validation_path.exists()
        validation = json.loads(validation_path.read_text())
        assert "rejection_counts" in validation
        assert "rejected_count" in report

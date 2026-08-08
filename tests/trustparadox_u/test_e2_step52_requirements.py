"""E2 repair §52: required unit/integration tests before rerun.

This test file fills remaining coverage gaps for the Step 52 requirements:

Protocol tests
- no E2 module imports synthetic PROTOCOL_VERSION
- all E2 writers use empirical 2.0.0 constants

Phase tests
- E2 pilot locks validation/test (covered in test_empirical_phase_lock.py)
- E2 frozen still locks (covered in test_empirical_phase_lock.py)
- completion checker reads the actual phase file (covered in test_e2_completion_check.py)

Pilot-prompt tests
- primary E2 builder does not use direct disclosure (covered in test_e2_completion_check.py)
- pilot namespace actually changes resolved requests
- no primary-pilot disclosure imperative
- only trust framing differs

Schedule tests (covered in test_empirical_pilot_config.py)
- 90 requests; 30 families; 3 trust levels per family; deterministic randomization

Connectivity tests (covered in test_e2_completion_check.py)
- connectivity and pilot model/config must match
- mismatch blocks pilot certification

Annotation tests
- evaluator does not receive trust
- J != G (covered in test_empirical_labeling.py)
- credential/proposition cases (covered in test_empirical_labeling.py)

Analysis tests (covered in test_empirical_analysis.py)
- paired risk difference; discordant counts; bootstrap CI; scenario stratification

Revision tests
- one-scenario-only effect does not auto-freeze (covered in test_empirical_classification.py)
- two-scenario consistent effect may freeze
- all-flat triggers revision
- final V3 null freezes transparently
"""

from __future__ import annotations

import ast
from pathlib import Path

from experiments.trustparadox_u.empirical_analysis import (
    DiscordantCounts,
    MatchedFamilyAnalysis,
    ScenarioAnalysis,
)
from experiments.trustparadox_u.empirical_classification import (
    ManipulationClassification,
    classify_manipulation,
    get_revision_state,
)
from experiments.trustparadox_u.empirical_corpus import get_target_spec
from experiments.trustparadox_u.empirical_generation import (
    PILOT_PROMPTS_DIR,
    PRIMARY_PILOT_ATTACK_TYPE,
    build_trust_pilot_request,
)

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "trustparadox_u"


# ---------------------------------------------------------------------------
# Protocol tests (§52: Protocol tests)
# ---------------------------------------------------------------------------


class TestProtocolImports:
    """Verify no E2 module imports synthetic PROTOCOL_VERSION."""

    def _get_e2_module_files(self) -> list[Path]:
        """Get all E2-specific module files."""
        prefixes = ("empirical_", "run_e2_")
        return [
            f for f in _EXPERIMENTS_DIR.glob("*.py") if any(f.name.startswith(p) for p in prefixes)
        ]

    def test_no_e2_module_imports_synthetic_protocol_version(self) -> None:
        """No E2 module imports PROTOCOL_VERSION from research_protocol."""
        violations: list[str] = []
        for filepath in self._get_e2_module_files():
            source = filepath.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "research_protocol" in node.module:
                        for alias in node.names:
                            if alias.name == "PROTOCOL_VERSION":
                                violations.append(
                                    f"{filepath.name}: imports PROTOCOL_VERSION "
                                    f"from {node.module}"
                                )
        assert (
            not violations
        ), f"E2 modules must not import synthetic PROTOCOL_VERSION: {violations}"

    def test_e2_modules_use_empirical_constants(self) -> None:
        """All E2 writers use empirical 2.0.0 constants."""
        from experiments.trustparadox_u.empirical_corpus import (
            EMPIRICAL_PROTOCOL_VERSION,
            EMPIRICAL_STUDY_VERSION,
        )

        assert EMPIRICAL_PROTOCOL_VERSION == "2.0.0"
        assert EMPIRICAL_STUDY_VERSION == "2.0.0"

    def test_completion_checker_uses_empirical_versions(self) -> None:
        """Completion checker uses empirical version constants."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            CompletionReport,
        )

        report = CompletionReport()
        assert report.protocol_version == "2.0.0"
        assert report.study_version == "2.0.0"


# ---------------------------------------------------------------------------
# Pilot-prompt tests (§52: Pilot-prompt tests)
# ---------------------------------------------------------------------------


class TestPilotPrompts:
    """Verify pilot prompt namespace and content requirements."""

    def test_pilot_builder_uses_pilot_namespace(self) -> None:
        """build_trust_pilot_request uses PILOT_PROMPTS_DIR, not base prompts."""
        spec = get_target_spec("credential_v1")
        request = build_trust_pilot_request(spec, "default", sample_index=0)
        # The request should use the pilot attack type, not direct_disclosure
        assert request.attack_type == PRIMARY_PILOT_ATTACK_TYPE
        assert request.attack_type == "trust_discretion_task"
        assert request.attack_type != "direct_disclosure"

    def test_pilot_prompts_directory_exists(self) -> None:
        """Pilot prompts directory exists and contains required files."""
        assert PILOT_PROMPTS_DIR.exists()
        assert (PILOT_PROMPTS_DIR / "generator_system.txt").exists()
        assert (PILOT_PROMPTS_DIR / "primary_task.txt").exists()
        assert (PILOT_PROMPTS_DIR / "trust_low.txt").exists()
        assert (PILOT_PROMPTS_DIR / "trust_default.txt").exists()
        assert (PILOT_PROMPTS_DIR / "trust_high.txt").exists()

    def test_no_disclosure_imperative_in_pilot_prompts(self) -> None:
        """Primary pilot prompts do not contain disclosure imperatives."""
        primary_task = (PILOT_PROMPTS_DIR / "primary_task.txt").read_text(encoding="utf-8")
        # Must not contain explicit disclosure instructions
        disclosure_phrases = [
            "reveal the",
            "disclose the",
            "tell the recipient",
            "share the secret",
            "provide the credential",
            "state the password",
        ]
        for phrase in disclosure_phrases:
            assert (
                phrase not in primary_task.lower()
            ), f"Pilot primary_task.txt contains disclosure imperative: '{phrase}'"

    def test_pilot_prompt_uses_discretion_language(self) -> None:
        """Pilot prompts use discretion-based language."""
        primary_task = (PILOT_PROMPTS_DIR / "primary_task.txt").read_text(encoding="utf-8")
        # Should contain discretion-based language
        assert "personal judgment" in primary_task or "exercise" in primary_task
        assert "without being explicitly told" in primary_task

    def test_only_trust_framing_differs_across_levels(self) -> None:
        """Trust-level prompts differ only in [TRUST_FRAMING] section."""
        trust_low = (PILOT_PROMPTS_DIR / "trust_low.txt").read_text(encoding="utf-8")
        trust_default = (PILOT_PROMPTS_DIR / "trust_default.txt").read_text(encoding="utf-8")
        trust_high = (PILOT_PROMPTS_DIR / "trust_high.txt").read_text(encoding="utf-8")

        def _non_trust_sections(text: str) -> str:
            """Extract everything except the [TRUST_FRAMING] section."""
            lines = text.split("\n")
            result: list[str] = []
            skip = False
            for line in lines:
                if line.startswith("[TRUST_FRAMING]"):
                    skip = True
                    continue
                if skip and line.startswith("["):
                    skip = False
                if not skip:
                    result.append(line)
            return "\n".join(result)

        # Non-trust sections should be identical across all three
        low_rest = _non_trust_sections(trust_low)
        default_rest = _non_trust_sections(trust_default)
        high_rest = _non_trust_sections(trust_high)

        assert (
            low_rest == default_rest
        ), "trust_low and trust_default differ outside [TRUST_FRAMING]"
        assert (
            default_rest == high_rest
        ), "trust_default and trust_high differ outside [TRUST_FRAMING]"

    def test_trust_framing_actually_differs(self) -> None:
        """The [TRUST_FRAMING] section is different for each trust level."""
        trust_low = (PILOT_PROMPTS_DIR / "trust_low.txt").read_text(encoding="utf-8")
        trust_high = (PILOT_PROMPTS_DIR / "trust_high.txt").read_text(encoding="utf-8")

        def _trust_section(text: str) -> str:
            lines = text.split("\n")
            result: list[str] = []
            capture = False
            for line in lines:
                if line.startswith("[TRUST_FRAMING]"):
                    capture = True
                    continue
                if capture and line.startswith("["):
                    break
                if capture:
                    result.append(line)
            return "\n".join(result)

        assert _trust_section(trust_low) != _trust_section(trust_high)


# ---------------------------------------------------------------------------
# Annotation independence tests (§52: Annotation tests)
# ---------------------------------------------------------------------------


class TestEvaluatorIndependence:
    """Verify evaluator independence requirements."""

    def test_labeling_module_does_not_receive_trust(self) -> None:
        """The labeling module functions do not accept trust_level as parameter."""
        import inspect

        from experiments.trustparadox_u.empirical_labeling import (
            label_credential_attempt,
            label_proposition_attempt,
        )

        cred_sig = inspect.signature(label_credential_attempt)
        prop_sig = inspect.signature(label_proposition_attempt)

        assert (
            "trust_level" not in cred_sig.parameters
        ), "label_credential_attempt must not receive trust_level"
        assert (
            "trust_level" not in prop_sig.parameters
        ), "label_proposition_attempt must not receive trust_level"

    def test_labeling_module_does_not_receive_trust_prompt(self) -> None:
        """The labeling module functions do not accept trust_prompt as parameter."""
        import inspect

        from experiments.trustparadox_u.empirical_labeling import (
            label_credential_attempt,
            label_proposition_attempt,
        )

        cred_sig = inspect.signature(label_credential_attempt)
        prop_sig = inspect.signature(label_proposition_attempt)

        assert "trust_prompt" not in cred_sig.parameters
        assert "trust_prompt" not in prop_sig.parameters


# ---------------------------------------------------------------------------
# Revision tests (§52: Revision tests)
# ---------------------------------------------------------------------------


def _make_analysis(
    *,
    overall_rd: float = 0.0,
    scenario_effects: list[float] | None = None,
) -> MatchedFamilyAnalysis:
    """Create a test analysis for classification tests."""
    if scenario_effects is None:
        scenario_effects = [0.0, 0.0, 0.0]

    scenario_analyses = [
        ScenarioAnalysis(
            scenario_id=f"scenario_{i}",
            num_families=10,
            low_rate=0.0,
            default_rate=0.0,
            high_rate=effect,
            high_low_difference=effect,
        )
        for i, effect in enumerate(scenario_effects)
    ]

    return MatchedFamilyAnalysis(
        schema_version="1.0.0",
        protocol_version="2.0.0",
        study_version="2.0.0",
        total_families=30,
        complete_families=30,
        incomplete_families=0,
        incomplete_family_ids=[],
        low_disclosure_rate=0.0,
        default_disclosure_rate=0.0,
        high_disclosure_rate=overall_rd,
        high_minus_low_risk_difference=overall_rd,
        discordant_counts=DiscordantCounts(0, 0, 0, 30),
        bootstrap_seed=20260809,
        bootstrap_resamples=5000,
        bootstrap_ci_lower=0.0,
        bootstrap_ci_upper=0.0,
        scenario_analyses=scenario_analyses,
        secondary_outcomes={},
    )


class TestRevisionRequirements:
    """Verify revision requirements from §52."""

    def test_two_scenario_consistent_effect_may_freeze(self) -> None:
        """Two-scenario consistent effect classifies as STRONG (may freeze)."""
        analysis = _make_analysis(
            overall_rd=0.20,
            scenario_effects=[0.15, 0.20, 0.18],
        )
        assessment = classify_manipulation(analysis)
        # Two or more scenarios with consistent nontrivial effect → STRONG
        assert assessment.classification == ManipulationClassification.STRONG

    def test_all_flat_triggers_revision(self) -> None:
        """All-flat (null) result should trigger revision if not at V3."""
        analysis = _make_analysis(
            overall_rd=0.01,
            scenario_effects=[0.01, 0.00, 0.02],
        )
        assessment = classify_manipulation(analysis)
        assert assessment.classification == ManipulationClassification.NULL

        # At V1, revision is allowed
        state_v1 = get_revision_state("E2_PRIMARY_V1")
        assert state_v1.can_revise is True

        # At V2, revision is allowed
        state_v2 = get_revision_state("E2_PRIMARY_V2")
        assert state_v2.can_revise is True

    def test_v3_null_freezes_transparently(self) -> None:
        """V3 null freezes transparently without pipeline failure."""
        analysis = _make_analysis(
            overall_rd=0.01,
            scenario_effects=[0.01, 0.00, 0.02],
        )
        assessment = classify_manipulation(analysis)
        assert assessment.classification == ManipulationClassification.NULL

        # At V3, cannot revise - must freeze
        state_v3 = get_revision_state("E2_PRIMARY_V3")
        assert state_v3.can_revise is False
        assert state_v3.revision_count == 2
        assert "Maximum revisions reached" in state_v3.reason

        # The classification is still valid (not an error)
        assert assessment.classification.value == "null"

    def test_one_scenario_only_not_auto_strong(self) -> None:
        """One-scenario-only effect does not auto-classify as STRONG."""
        analysis = _make_analysis(
            overall_rd=0.20,
            scenario_effects=[0.55, 0.02, 0.01],
        )
        assessment = classify_manipulation(analysis)
        # One scenario dominates → not STRONG
        assert assessment.classification != ManipulationClassification.STRONG


# ---------------------------------------------------------------------------
# Pilot execution follows schedule (§52: Schedule tests)
# ---------------------------------------------------------------------------


class TestPilotExecutionSchedule:
    """Verify pilot runner follows the schedule."""

    def test_pilot_runner_imports_schedule(self) -> None:
        """Primary pilot runner module imports schedule functions."""
        from experiments.trustparadox_u import run_e2_primary_pilot

        # Verify the module has the expected functions
        assert hasattr(run_e2_primary_pilot, "run_primary_pilot")

    def test_pilot_runner_uses_pilot_config(self) -> None:
        """Primary pilot runner uses empirical_pilot_config."""
        from experiments.trustparadox_u import run_e2_primary_pilot

        # Verify the module references pilot config
        source_file = Path(run_e2_primary_pilot.__file__)
        source = source_file.read_text(encoding="utf-8")
        assert "build_request_schedule" in source or "randomize_schedule" in source
        assert "PILOT_EXECUTION_SEED" in source or "randomization_seed" in source

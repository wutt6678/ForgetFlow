"""Architecture contract tests to prevent regression to mixed APIs."""

import ast
from pathlib import Path

import pytest


class TestArchitectureContract:
    """Tests to enforce canonical MVP architecture."""

    def test_canonical_imports_succeed(self) -> None:
        """All canonical imports should work."""
        from experiments.trustparadox_u.runner import run_episode
        from marble.firewall.contamination import ContaminationTracker
        from marble.firewall.detectors import HybridDetector
        from marble.firewall.flow_gate import FlowGate
        from marble.firewall.history import RecipientHistory, ReconstructionChecker
        from marble.firewall.policy import ForgetPolicy
        from marble.firewall.registry import ForgetLedger
        from marble.firewall.types import (
            DetectorResult,
            FirewallDecision,
            ForgetRecord,
            MessageEnvelope,
        )

        assert ForgetRecord is not None
        assert MessageEnvelope is not None
        assert DetectorResult is not None
        assert FirewallDecision is not None
        assert ForgetLedger is not None
        assert HybridDetector is not None
        assert RecipientHistory is not None
        assert ReconstructionChecker is not None
        assert ForgetPolicy is not None
        assert FlowGate is not None
        assert ContaminationTracker is not None
        assert run_episode is not None

    def test_legacy_modules_do_not_exist(self) -> None:
        """Legacy advanced modules should not exist in active package."""
        legacy_modules = [
            "marble/firewall/flowgate.py",
            "marble/firewall/ledger.py",
            "marble/firewall/recipient_state.py",
            "marble/firewall/reconstruction.py",
            "marble/firewall/transformer.py",
            "marble/engine/trustparadox_u_engine.py",
            "marble/engine/trustparadox_u_scheduler.py",
        ]
        for module_path in legacy_modules:
            assert not Path(module_path).exists(), f"Legacy module should not exist: {module_path}"

    def test_runner_imports_canonical_flow_gate(self) -> None:
        """Runner should import canonical flow_gate, not legacy flowgate."""
        runner_path = Path("experiments/trustparadox_u/runner.py")
        source = runner_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "flowgate" in node.module.lower():
                    pytest.fail(f"Runner imports from legacy module: {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "flowgate" in alias.name.lower():
                        pytest.fail(f"Runner imports legacy module: {alias.name}")

    def test_no_source_imports_decision_action(self) -> None:
        """No source should import DecisionAction (advanced API)."""
        forbidden = "DecisionAction"
        for py_file in Path("marble").rglob("*.py"):
            source = py_file.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        assert alias.name != forbidden, f"{py_file} imports forbidden {forbidden}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name != forbidden, f"{py_file} imports forbidden {forbidden}"

    def test_no_source_imports_forget_request(self) -> None:
        """No source should import ForgetRequest (advanced API)."""
        forbidden = "ForgetRequest"
        for py_file in Path("marble").rglob("*.py"):
            source = py_file.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        assert alias.name != forbidden, f"{py_file} imports forbidden {forbidden}"

    def test_no_source_imports_advanced_engine(self) -> None:
        """No source should import advanced engine modules."""
        forbidden_modules = {
            "marble.engine.trustparadox_u_engine",
            "marble.engine.trustparadox_u_scheduler",
            "marble.transactions",
        }
        for py_file in list(Path("marble").rglob("*.py")) + list(Path("experiments").rglob("*.py")):
            source = py_file.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module in forbidden_modules:
                        pytest.fail(f"{py_file} imports forbidden module: {node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in forbidden_modules:
                            pytest.fail(f"{py_file} imports forbidden module: {alias.name}")

    def test_one_flowgate_class_exists(self) -> None:
        """Only one FlowGate class should exist in the codebase."""
        flowgate_files = []
        for py_file in Path("marble").rglob("*.py"):
            if "flowgate" in py_file.name.lower() or "flow_gate" in py_file.name.lower():
                flowgate_files.append(py_file)

        # Should only have flow_gate.py, not flowgate.py
        assert len(flowgate_files) == 1, f"Expected 1 FlowGate file, found: {flowgate_files}"
        assert flowgate_files[0].name == "flow_gate.py"

    def test_one_forgetledger_class_exists(self) -> None:
        """Only one ForgetLedger class should exist in the codebase."""
        ledger_files = []
        for py_file in Path("marble").rglob("*.py"):
            if "ledger" in py_file.name.lower() or "registry" in py_file.name.lower():
                source = py_file.read_text()
                if "class ForgetLedger" in source:
                    ledger_files.append(py_file)

        assert len(ledger_files) == 1, f"Expected 1 ForgetLedger file, found: {ledger_files}"
        assert ledger_files[0].name == "registry.py"

    def test_one_contaminationtracker_class_exists(self) -> None:
        """Only one ContaminationTracker class should exist in the codebase."""
        tracker_files = []
        for py_file in Path("marble").rglob("*.py"):
            source = py_file.read_text()
            if "class ContaminationTracker" in source:
                tracker_files.append(py_file)

        assert (
            len(tracker_files) == 1
        ), f"Expected 1 ContaminationTracker file, found: {tracker_files}"
        assert tracker_files[0].name == "contamination.py"

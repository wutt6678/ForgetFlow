#!/usr/bin/env python3
"""Check source code integrity and prevent duplicate definitions."""

import ast
import sys
from pathlib import Path


class IntegrityError(Exception):
    """Raised when source integrity check fails."""


FORBIDDEN_IMPORTS = {
    "marble.firewall.flowgate",
    "marble.firewall.ledger",
    "marble.firewall.recipient_state",
    "marble.firewall.reconstruction",
    "marble.firewall.transformer",
    "marble.engine.trustparadox_u_engine",
    "marble.engine.trustparadox_u_scheduler",
    "marble.transactions",
}

FORBIDDEN_NAMES = {
    "ActionCandidate",
    "AuditEntry",
    "ChannelType",
    "DecisionAction",
    "EncryptedDescriptor",
    "ForgetRequest",
    "StoredForgetRecord",
    "TargetDetection",
    "TargetType",
    "ERROR_BLOCK",
    "released_envelope",
    "released_payload",
}


def check_file(path: Path) -> list[str]:
    """Check a single Python file for integrity issues.

    Returns a list of error messages.
    """
    errors = []

    try:
        source = path.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{path}: syntax error: {e}"]
    except Exception as e:
        return [f"{path}: read error: {e}"]

    # Check for duplicate class definitions
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    seen_classes = set()
    for cls in classes:
        if cls in seen_classes:
            errors.append(f"{path}: duplicate class definition: {cls}")
        seen_classes.add(cls)

    # Check for duplicate function definitions at top level
    functions = [
        node.name for node in ast.iter_child_nodes(tree) if isinstance(node, ast.FunctionDef)
    ]
    seen_functions = set()
    for func in functions:
        if func in seen_functions:
            errors.append(f"{path}: duplicate function definition: {func}")
        seen_functions.add(func)

    # Check for forbidden imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    errors.append(f"{path}: forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module in FORBIDDEN_IMPORTS:
                errors.append(f"{path}: forbidden import from: {node.module}")

    # Check for forbidden names in assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in FORBIDDEN_NAMES:
                    errors.append(f"{path}: forbidden name assignment: {target.id}")
        elif isinstance(node, ast.ClassDef):
            if node.name in FORBIDDEN_NAMES:
                errors.append(f"{path}: forbidden class name: {node.name}")
        elif isinstance(node, ast.FunctionDef):
            if node.name in FORBIDDEN_NAMES:
                errors.append(f"{path}: forbidden function name: {node.name}")

    return errors


def main() -> int:
    """Check all Python files in marble/ and experiments/."""
    directories = [Path("marble"), Path("experiments")]
    all_errors = []

    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            errors = check_file(path)
            all_errors.extend(errors)

    if all_errors:
        print("Source integrity check FAILED:")
        for error in all_errors:
            print(f"  {error}")
        return 1

    print("Source integrity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

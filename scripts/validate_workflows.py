#!/usr/bin/env python3
"""Validate GitHub Actions workflow files for syntax and duplicate keys."""

import sys
from pathlib import Path

import yaml


class DuplicateKeyError(Exception):
    """Raised when a YAML file contains duplicate keys."""


class DuplicateKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys."""


def _check_duplicate_key(loader: DuplicateKeyLoader, node: yaml.MappingNode) -> dict:
    """Check for duplicate keys in a YAML mapping node."""
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in mapping:
            raise DuplicateKeyError(
                f"Duplicate key '{key}' found at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node)
    return mapping


DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _check_duplicate_key
)


def validate_workflow(path: Path) -> bool:
    """Validate a single workflow file.

    Returns True if valid, False otherwise.
    """
    try:
        with path.open() as f:
            data = yaml.load(f, Loader=DuplicateKeyLoader)

        if not isinstance(data, dict):
            print(f"ERROR: {path}: root is not a mapping")
            return False

        # 'on' is a reserved YAML keyword that gets parsed as True
        required_keys = {"name", "jobs"}
        missing = required_keys - set(data.keys())
        # Check for 'on' key (may be parsed as True in YAML)
        has_on = "on" in data or True in data
        if missing or not has_on:
            print(f"ERROR: {path}: missing required keys: name, on, or jobs")
            return False

        if not isinstance(data["jobs"], dict):
            print(f"ERROR: {path}: 'jobs' is not a mapping")
            return False

        print(f"valid: {path}")
        return True

    except DuplicateKeyError as e:
        print(f"ERROR: {path}: {e}")
        return False
    except yaml.YAMLError as e:
        print(f"ERROR: {path}: YAML parse error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {path}: {e}")
        return False


def main() -> int:
    """Validate all workflow files."""
    workflows_dir = Path(".github/workflows")

    if not workflows_dir.exists():
        print(f"ERROR: {workflows_dir} does not exist")
        return 1

    workflow_files = list(workflows_dir.glob("*.y*ml"))

    if not workflow_files:
        print(f"ERROR: No workflow files found in {workflows_dir}")
        return 1

    all_valid = True
    for path in workflow_files:
        if not validate_workflow(path):
            all_valid = False

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())

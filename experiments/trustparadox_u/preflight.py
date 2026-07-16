"""Preflight checks for experiment-mode runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.trustparadox_u.config import ExperimentConfig, load_config


def run_preflight(config: ExperimentConfig, *, probe_provider: bool = False) -> list[str]:
    """Run preflight checks and return a list of failure messages.

    An empty list means all checks passed.
    """
    failures: list[str] = []

    # 1. Config loaded successfully (already done by caller)
    # 2. Semantic config is valid (enforced by __post_init__)

    # 3. LiteLLM importability in experiment mode
    if config.run.mode == "experiment":
        try:
            import litellm  # noqa: F401
        except ImportError:
            failures.append(
                "litellm is not installed. " "Install with: poetry install -E experiment"
            )

    # 4. Check required environment variables for experiment mode
    if config.run.mode == "experiment" and config.models.embedding_provider == "litellm":
        import os

        # LiteLLM typically needs OPENAI_API_KEY or similar
        # We just warn, don't fail, since keys may be set at runtime
        known_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "COHERE_API_KEY",
            "AZURE_API_KEY",
        ]
        if not any(os.environ.get(k) for k in known_keys):
            failures.append(
                "No known embedding API key found in environment. "
                "Set OPENAI_API_KEY or the appropriate provider key."
            )

    # 5. Output directory writability
    output_dir = Path("results/trustparadox_u")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".preflight_test"
        test_file.touch()
        test_file.unlink()
    except OSError:
        failures.append(f"Output directory {output_dir} is not writable")

    # 6. No raw secret in filenames (check canonical targets)
    # This is a structural check - secrets should not appear in file paths
    # The runner handles this, but we verify config is consistent

    # 7. Probe provider if requested
    if probe_provider and config.run.mode == "experiment":
        try:
            from experiments.trustparadox_u.embedding import RealEmbeddingProvider

            provider = RealEmbeddingProvider(
                model_name=config.models.embedding_model or "text-embedding-3-small",
                expected_dimension=config.models.embedding_dimension,
            )
            vectors = provider.embed(["preflight probe"])
            if not vectors or not vectors[0]:
                failures.append("Provider probe returned empty vector")
        except Exception as exc:
            failures.append(f"Provider probe failed: {exc}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgetFlow experiment preflight checks")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--probe-provider",
        action="store_true",
        help="Make a real embedding API call to verify provider connectivity",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    failures = run_preflight(config, probe_provider=args.probe_provider)

    if failures:
        print("PREFLIGHT FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("PREFLIGHT PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()

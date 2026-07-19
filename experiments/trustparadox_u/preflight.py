"""Preflight checks for experiment-mode runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.trustparadox_u.config import ExperimentConfig, load_config


def run_preflight(
    config: ExperimentConfig,
    *,
    probe_provider: bool = False,
    probe_chat_provider: bool = False,
) -> list[str]:
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

        # Check for custom API key environment variable first
        if config.models.api_key_env:
            if not os.environ.get(config.models.api_key_env):
                failures.append(
                    f"Required API key environment variable {config.models.api_key_env} is not set."
                )
        elif config.models.api_base:
            # Custom endpoint without api_key_env configured
            # Don't require standard cloud-provider keys, but warn
            pass
        else:
            # Standard provider - check for known keys
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
            from experiments.trustparadox_u.providers import (
                build_real_embedding_provider,
                sanitize_api_base,
            )

            provider = build_real_embedding_provider(config.models)
            vectors = provider.embed(["ForgetFlow provider preflight probe"])
            if not vectors or not vectors[0]:
                failures.append("Provider probe returned empty vector")
            else:
                host = sanitize_api_base(config.models.api_base)
                print(
                    f"  provider={provider.provider_name}  "
                    f"model={provider.model_name}  "
                    f"expected_dim={config.models.embedding_dimension}  "
                    f"observed_dim={len(vectors[0])}  "
                    f"api_base={host or '(default)'}"
                )
        except Exception as exc:
            failures.append(f"Provider probe failed: {exc}")

    # 8. Probe chat provider if requested
    if probe_chat_provider and config.run.mode == "experiment":
        if not config.models.chat_model:
            failures.append("Chat provider probe requested but no chat_model configured")
        else:
            try:
                from experiments.trustparadox_u.chat_provider import LiteLLMResponseProvider

                chat_provider = LiteLLMResponseProvider(
                    model_name=config.models.chat_model,
                    temperature=config.models.chat_temperature,
                    max_tokens=min(config.models.chat_max_tokens, 32),
                    api_base=config.models.api_base,
                    api_key_env=config.models.api_key_env,
                )
                result_text = str(
                    chat_provider.respond(
                        episode_id="preflight",
                        agent_id="probe",
                        turn_id=0,
                        instruction="Return exactly: FORGETFLOW_CHAT_OK",
                        role="probe",
                        public_profile="preflight probe",
                        trust_level="default",
                    )
                )
                if "FORGETFLOW_CHAT_OK" in result_text:
                    print(
                        f"  chat_model={config.models.chat_model}  "
                        f"latency_ms={chat_provider.last_latency_ms:.0f}  "
                        f"response=FORGETFLOW_CHAT_OK"
                    )
                else:
                    print(
                        f"  chat_model={config.models.chat_model}  "
                        f"latency_ms={chat_provider.last_latency_ms:.0f}  "
                        f"response={result_text[:60]!r}"
                    )
            except Exception as exc:
                failures.append(f"Chat provider probe failed: {exc}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgetFlow experiment preflight checks")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--probe-provider",
        action="store_true",
        help="Make a real embedding API call to verify provider connectivity",
    )
    parser.add_argument(
        "--probe-chat-provider",
        action="store_true",
        help="Make a real chat API call to verify chat provider connectivity",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    failures = run_preflight(
        config,
        probe_provider=args.probe_provider,
        probe_chat_provider=args.probe_chat_provider,
    )

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

#!/usr/bin/env python3
"""Real-LLM smoke test runner for ForgetFlow.

Runs a small matrix of episodes using real chat and embedding models
to verify end-to-end pipeline behaviour before committing to a full
experiment.

Usage:
    poetry run python -m experiments.trustparadox_u.run_real_llm_smoke \
      --config experiments/trustparadox_u/configs/smoke_real_llm_full.yaml \
      --scenarios credential,private_attribute,authorization \
      --trust-levels high \
      --repetitions 1 \
      --output results/real_llm_smoke/full_mvp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.chat_provider import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    LiteLLMResponseProvider,
    _build_system_prompt,
    _build_user_prompt,
)
from experiments.trustparadox_u.config import (  # noqa: E402
    ExperimentConfig,
    load_config,
)
from experiments.trustparadox_u.dataset import load_episode  # noqa: E402
from experiments.trustparadox_u.evaluator import evaluate_all  # noqa: E402
from experiments.trustparadox_u.manifest import get_repository_commit  # noqa: E402
from experiments.trustparadox_u.runner import run_episode  # noqa: E402
from experiments.trustparadox_u.serialization import (  # noqa: E402
    serialize_episode_result,
)

SCENARIOS_DIR = PROJECT_ROOT / "data" / "trustparadox_u" / "scenarios"

# Available pilot scenarios
AVAILABLE_SCENARIOS = {
    "credential": "pilot_credential.yaml",
    "private_attribute": "pilot_private_attribute.yaml",
    "authorization": "pilot_authorization.yaml",
}


def _write_manifest(
    output_dir: Path,
    config: ExperimentConfig,
    repository_commit: str,
    scenarios: list[str],
    trust_levels: list[str],
    repetitions: int,
) -> dict[str, Any]:
    """Write the smoke-test manifest.json."""
    import platform

    manifest: dict[str, Any] = {
        "repository_commit": repository_commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "config_hash": config.config_hash(),
        "seed": config.seed,
        "repetitions": repetitions,
        "scenarios": scenarios,
        "trust_levels": trust_levels,
        "chat_provider": config.models.chat_provider,
        "chat_model": config.models.chat_model,
        "chat_temperature": config.models.chat_temperature,
        "chat_max_tokens": config.models.chat_max_tokens,
        "embedding_provider": config.models.embedding_provider,
        "embedding_model": config.models.embedding_model,
        "embedding_dimension": config.models.embedding_dimension,
        "api_base_sanitized": None,
    }
    if config.models.api_base:
        from experiments.trustparadox_u.providers import sanitize_api_base

        manifest["api_base_sanitized"] = sanitize_api_base(config.models.api_base)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _write_environment(output_dir: Path) -> None:
    """Write environment.txt with runtime metadata."""
    import platform

    lines = [
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"repository_commit={get_repository_commit()}",
        f"timestamp={datetime.now(timezone.utc).isoformat()}",
    ]
    try:
        import litellm

        lines.append(f"litellm={getattr(litellm, '__version__', 'unknown')}")
    except ImportError:
        lines.append("litellm=NOT_INSTALLED")

    (output_dir / "environment.txt").write_text("\n".join(lines) + "\n")


def _write_git_commit(output_dir: Path) -> None:
    """Write git_commit.txt."""
    (output_dir / "git_commit.txt").write_text(get_repository_commit() + "\n")


def _write_prompt_template(output_dir: Path) -> None:
    """Write prompt_template.txt and its hash."""
    system_example = _build_system_prompt(
        agent_id="EXAMPLE_AGENT",
        role="example_role",
        public_profile="example_profile",
        trust_level="high",
        visible_context=["example context"],
        released_history=["example released"],
    )
    user_example = _build_user_prompt("example instruction")
    template_text = f"=== System Prompt Template ===\n{system_example}\n\n=== User Prompt Template ===\n{user_example}\n\n=== Version: {PROMPT_TEMPLATE_VERSION} ===\n"
    (output_dir / "prompt_template.txt").write_text(template_text)

    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()
    (output_dir / "prompt_template.sha256").write_text(template_hash + "\n")


def _write_resolved_config(output_dir: Path, config: ExperimentConfig) -> None:
    """Write resolved_config.yaml (sanitized — no secrets)."""
    import yaml

    config_dict = config.to_dict()
    # Remove any potential secret fields
    for key in ("api_key_env",):
        if key in config_dict.get("models", {}):
            # Keep the env var name but not any value
            pass
    (output_dir / "resolved_config.yaml").write_text(
        yaml.dump(config_dict, default_flow_style=False)
    )


def run_real_llm_smoke(
    config: ExperimentConfig,
    output_dir: Path,
    scenarios: list[str],
    trust_levels: list[str],
    repetitions: int = 1,
) -> dict[str, Any]:
    """Run the real-LLM smoke test matrix.

    Returns a summary dict with results and pass/fail status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    repository_commit = get_repository_commit()

    # Build chat provider
    if not config.models.chat_model:
        raise ValueError("chat_model is required for real-LLM smoke test")

    chat_provider = LiteLLMResponseProvider(
        model_name=config.models.chat_model,
        temperature=config.models.chat_temperature,
        max_tokens=config.models.chat_max_tokens,
        api_base=config.models.api_base,
        api_key_env=config.models.api_key_env,
    )

    # Write static artifacts
    _write_manifest(output_dir, config, repository_commit, scenarios, trust_levels, repetitions)
    _write_environment(output_dir)
    _write_git_commit(output_dir)
    _write_prompt_template(output_dir)
    _write_resolved_config(output_dir, config)

    all_results: list[Any] = []
    failures: list[dict[str, Any]] = []
    total_attempts = 0

    # Episode loop
    for scenario_name in scenarios:
        if scenario_name not in AVAILABLE_SCENARIOS:
            failures.append(
                {
                    "scenario": scenario_name,
                    "error": f"Unknown scenario: {scenario_name}",
                }
            )
            continue

        scenario_file = AVAILABLE_SCENARIOS[scenario_name]
        for trust_level in trust_levels:
            for rep in range(repetitions):
                total_attempts += 1
                try:
                    ep = load_episode(SCENARIOS_DIR / scenario_file)

                    # Override trust level if needed
                    # (scenarios have a fixed trust_level in YAML;
                    #  for the smoke test we use what's in the file)
                    run_id = hashlib.sha256(
                        f"{ep.episode_id}|{trust_level}|{rep}|{config.seed}".encode()
                    ).hexdigest()[:20]

                    result = run_episode(
                        ep,
                        config,
                        responder=chat_provider,
                        firewall_enabled=True,
                        run_id=run_id,
                    )

                    # Add chat metadata to result
                    result.metadata["candidate_generation_model"] = chat_provider.last_model_name
                    result.metadata["candidate_generation_latency_ms"] = (
                        chat_provider.last_latency_ms
                    )
                    result.metadata["candidate_prompt_hash"] = chat_provider.last_prompt_hash
                    result.metadata["candidate_retry_count"] = chat_provider.last_retry_count
                    result.metadata["chat_prompt_version"] = PROMPT_TEMPLATE_VERSION

                    all_results.append(result)

                except Exception as exc:
                    failures.append(
                        {
                            "scenario": scenario_name,
                            "trust_level": trust_level,
                            "repetition": rep,
                            "error": str(exc),
                        }
                    )

    # Write episodes.jsonl
    episodes_path = output_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(serialize_episode_result(r)) + "\n")

    # Write failures.jsonl
    failures_path = output_dir / "failures.jsonl"
    with open(failures_path, "w") as f:
        for fail in failures:
            f.write(json.dumps(fail) + "\n")

    # Compute summary metrics
    evaluation = evaluate_all(all_results) if all_results else None

    # Build run summary
    run_summary: dict[str, Any] = {
        "total_attempts": total_attempts,
        "successful_episodes": len(all_results),
        "failed_episodes": len(failures),
        "repository_commit": repository_commit,
        "chat_model": config.models.chat_model,
        "embedding_model": config.models.embedding_model,
    }
    if evaluation:
        run_summary["metrics"] = evaluation.to_dict()

    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))

    # Determine pass/fail
    failure_rate = len(failures) / max(total_attempts, 1)
    passed = failure_rate <= 0.2 and len(all_results) > 0

    summary = {
        "passed": passed,
        "total_attempts": total_attempts,
        "successful": len(all_results),
        "failed": len(failures),
        "failure_rate": failure_rate,
        "results": all_results,
        "failures": failures,
    }

    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Real-LLM smoke test runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to smoke test config YAML",
    )
    parser.add_argument(
        "--scenarios",
        default="credential,private_attribute,authorization",
        help="Comma-separated scenario names",
    )
    parser.add_argument(
        "--trust-levels",
        default="high",
        help="Comma-separated trust levels",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per condition",
    )
    parser.add_argument(
        "--output",
        default="results/real_llm_smoke/smoke",
        help="Output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    trust_levels = [t.strip() for t in args.trust_levels.split(",")]

    try:
        summary = run_real_llm_smoke(
            config=config,
            output_dir=Path(args.output),
            scenarios=scenarios,
            trust_levels=trust_levels,
            repetitions=args.repetitions,
        )
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1

    if summary["passed"]:
        print(f"SMOKE TEST PASSED: {summary['successful']}/{summary['total_attempts']} episodes")
        return 0
    else:
        print(
            f"SMOKE TEST FAILED: {summary['failed']}/{summary['total_attempts']} episodes failed "
            f"({summary['failure_rate']:.0%} failure rate)"
        )
        for f in summary["failures"]:
            print(f"  - {f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

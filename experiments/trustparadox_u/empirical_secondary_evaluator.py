"""E2-A7-FIX-005: secondary independent evaluator J2.

J2 is a scientifically independent reviewer distinct from both G (generator)
and J1 (primary evaluator).  J2 uses a different model family (GLM-5.2)
from J1 (qwen3.8-max) and G (qwen3.7-plus).

J2 receives ONLY:
- candidate text
- target semantics (canonical target, aliases, permitted residuals)
- annotation schema

J2 must NOT receive:
- firewall condition, expected label, J1 confidence, J1 rationale,
  reference-oracle output, embedding score, detector evidence.

This module reuses the blinded request construction and output parsing
from the primary evaluator module but loads J2-specific prompts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_STUDY_VERSION,
    GENERATOR_MODEL_IDENTITY,
    EVALUATOR_MODEL_IDENTITY,
    SECONDARY_EVALUATOR_MODEL_IDENTITY,
    SECONDARY_EVALUATOR_ROLE,
)
from experiments.trustparadox_u.empirical_evaluator import (
    EvaluatorRequest,
    EVALUATOR_STATUSES,
    PROHIBITED_EVALUATOR_FIELDS,
    PRIMARY_EXPOSURE_LABELS,
    REQUIRED_JUDGMENT_FIELDS,
    BOOLEAN_JUDGMENT_FIELDS,
    EvaluatorParseError,
    _extract_json_from_text,
    parse_evaluator_output,
    build_evaluator_request,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: E2-A7-FIX-005: J2 prompt directory (independent from J1 prompts).
SECONDARY_PROMPT_DIR = (
    _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "prompts" / "secondary_evaluator"
)

#: E2-A7-FIX-005: output directory for secondary annotation artifacts.
SECONDARY_ANNOTATION_DIR = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_secondary_annotation"
)

#: E2-A7-FIX-005: J2 must be distinct from G and J1.
assert SECONDARY_EVALUATOR_MODEL_IDENTITY != EVALUATOR_MODEL_IDENTITY, (
    f"J2 model {SECONDARY_EVALUATOR_MODEL_IDENTITY} must differ from "
    f"J1 model {EVALUATOR_MODEL_IDENTITY}"
)
assert SECONDARY_EVALUATOR_MODEL_IDENTITY != GENERATOR_MODEL_IDENTITY, (
    f"J2 model {SECONDARY_EVALUATOR_MODEL_IDENTITY} must differ from "
    f"generator G {GENERATOR_MODEL_IDENTITY}"
)


# ---------------------------------------------------------------------------
# J2 prompt loading and hashing
# ---------------------------------------------------------------------------


def load_secondary_system_prompt() -> str:
    """Load the frozen J2 system prompt."""
    path = SECONDARY_PROMPT_DIR / "secondary_system.txt"
    return path.read_text(encoding="utf-8")


def load_secondary_task_prompt(target_type: str) -> str:
    """Load the J2 task prompt for a target type."""
    if target_type == "credential":
        path = SECONDARY_PROMPT_DIR / "credential_task.txt"
    else:
        path = SECONDARY_PROMPT_DIR / "proposition_task.txt"
    return path.read_text(encoding="utf-8")


def secondary_prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """SHA-256 of the combined J2 prompts."""
    combined = f"{system_prompt}\n---\n{user_prompt}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def secondary_prompt_manifest() -> dict[str, Any]:
    """E2-A7-FIX-017: compute hashes for all J2 prompt files."""
    manifest: dict[str, Any] = {}
    for prompt_file in sorted(SECONDARY_PROMPT_DIR.rglob("*.txt")):
        content = prompt_file.read_text(encoding="utf-8")
        rel_path = str(prompt_file.relative_to(SECONDARY_PROMPT_DIR))
        manifest[rel_path] = {
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "size_bytes": len(content.encode("utf-8")),
        }
    schema_file = SECONDARY_PROMPT_DIR / "reviewer_schema.json"
    if schema_file.exists():
        content = schema_file.read_text(encoding="utf-8")
        manifest["reviewer_schema.json"] = {
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "size_bytes": len(content.encode("utf-8")),
        }
    return manifest


# ---------------------------------------------------------------------------
# Secondary evaluator provider (E2-A7-FIX-005)
# ---------------------------------------------------------------------------


@dataclass
class SecondaryEvaluatorProvider:
    """E2-A7-FIX-005: chat-model provider for secondary evaluator J2.

    Calls the J2 model via LiteLLM with retry logic.  J2 uses a different
    model family from J1 and G to ensure scientific independence.
    """

    model_name: str
    provider: str = "openai"
    temperature: float = 0.0
    max_tokens: int = 512
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 2

    # Recorded metadata from the most recent call
    last_latency_ms: float = 0.0
    last_model_returned: str = ""
    last_request_id: str = ""

    def evaluate(
        self,
        request: EvaluatorRequest,
    ) -> dict[str, Any]:
        """Run a single J2 evaluation with retry logic.

        Returns a dict with:
        - raw_output: the raw model response text
        - status: one of EVALUATOR_STATUSES
        - model_returned: the actual model identity returned
        - request_id: the API request ID
        - latency_ms: wall-clock latency
        - retries: number of retries used
        - parsed: parsed judgment dict (if status=success)
        - parse_error: parse error message (if status=malformed)
        - system_prompt_hash: SHA-256 of the J2 system prompt
        - user_prompt_hash: SHA-256 of the J2 user prompt
        """
        from litellm import completion

        system_prompt = load_secondary_system_prompt()
        task_prompt = load_secondary_task_prompt(request.target_type)
        user_prompt = request.to_user_prompt(task_prompt)

        # Blinding check (E2R-027 — same rules apply to J2)
        violations = request.verify_blinding()
        if violations:
            return {
                "raw_output": "",
                "status": "provider_error",
                "model_returned": "",
                "request_id": "",
                "latency_ms": 0.0,
                "retries": 0,
                "parsed": None,
                "parse_error": f"blinding violation: {violations}",
                "system_prompt_hash": "",
                "user_prompt_hash": "",
            }

        sys_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        usr_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Resolve API key
        api_key: str | None = None
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)

        last_exc: Exception | None = None
        retries_used = 0

        for attempt in range(1 + self.max_retries):
            if attempt > 0:
                retries_used = attempt
                logger.info("J2 evaluator retry %d/%d", attempt, self.max_retries)
            try:
                start = time.monotonic()
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout_seconds,
                }
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                if api_key:
                    kwargs["api_key"] = api_key

                response = completion(**kwargs)
                elapsed_ms = (time.monotonic() - start) * 1000.0
                self.last_latency_ms = elapsed_ms

                raw_text: str = response.choices[0].message.content or ""
                model_returned = getattr(response, "model", self.model_name)
                request_id = getattr(response, "id", "")
                self.last_model_returned = model_returned
                self.last_request_id = request_id

                if not raw_text.strip():
                    return {
                        "raw_output": "",
                        "status": "empty",
                        "model_returned": model_returned,
                        "request_id": request_id,
                        "latency_ms": elapsed_ms,
                        "retries": retries_used,
                        "parsed": None,
                        "parse_error": "J2 evaluator returned empty response",
                        "system_prompt_hash": sys_hash,
                        "user_prompt_hash": usr_hash,
                    }

                try:
                    parsed = parse_evaluator_output(raw_text, request.generation_attempt_id)
                    return {
                        "raw_output": raw_text,
                        "status": "success",
                        "model_returned": model_returned,
                        "request_id": request_id,
                        "latency_ms": elapsed_ms,
                        "retries": retries_used,
                        "parsed": parsed,
                        "parse_error": None,
                        "system_prompt_hash": sys_hash,
                        "user_prompt_hash": usr_hash,
                    }
                except EvaluatorParseError as parse_exc:
                    if attempt >= self.max_retries:
                        return {
                            "raw_output": raw_text,
                            "status": "malformed",
                            "model_returned": model_returned,
                            "request_id": request_id,
                            "latency_ms": elapsed_ms,
                            "retries": retries_used,
                            "parsed": None,
                            "parse_error": str(parse_exc),
                            "system_prompt_hash": sys_hash,
                            "user_prompt_hash": usr_hash,
                        }
                    last_exc = parse_exc
                    continue

            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                is_transient = any(
                    kw in exc_str
                    for kw in (
                        "timeout",
                        "timed out",
                        "rate limit",
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                    )
                )
                if not is_transient or attempt >= self.max_retries:
                    status = "timeout" if "timeout" in exc_str else "provider_error"
                    return {
                        "raw_output": "",
                        "status": status,
                        "model_returned": "",
                        "request_id": "",
                        "latency_ms": 0.0,
                        "retries": retries_used,
                        "parsed": None,
                        "parse_error": str(exc),
                        "system_prompt_hash": sys_hash,
                        "user_prompt_hash": usr_hash,
                    }

        return {
            "raw_output": "",
            "status": "provider_error",
            "model_returned": "",
            "request_id": "",
            "latency_ms": 0.0,
            "retries": retries_used,
            "parsed": None,
            "parse_error": str(last_exc) if last_exc else "unknown error",
            "system_prompt_hash": sys_hash,
            "user_prompt_hash": usr_hash,
        }

"""Perform secondary review of the E2 pilot label audit queue.

E2-A7-FIX-005: replaces the placeholder secondary review with real J2
independent evaluator calls via LiteLLM.

E2-A7-FIX-007: persists raw J2 responses to secondary_raw_responses.jsonl.
E2-A7-FIX-008: generates secondary labels separately with full provenance.
E2-A7-FIX-009: explicit disagreement resolution (Option C: mark unresolved).
E2-A7-FIX-010: generates J1-J2 agreement report (no human agreement metrics).
E2-A7-FIX-017: freezes J2 prompt manifest.
E2-A7-FIX-035: computes secondary annotation agreement statistics.

This script:
1. Loads the secondary_review_queue.jsonl (cases requiring review).
2. For each case, loads the candidate text and target spec.
3. Builds a blinded EvaluatorRequest (no trust level, no J1 info).
4. Calls J2 (glm-5.2) via SecondaryEvaluatorProvider for independent review.
5. Writes all required artifacts:
   - secondary_raw_responses.jsonl (FIX-007)
   - secondary_labels.jsonl (FIX-008)
   - secondary_prompt_manifest.json (FIX-017)
   - secondary_annotation_agreement.json (FIX-035)
   - Updates label_agreement_report.json (FIX-010)
   - Updates adjudication_log.jsonl (FIX-009)
   - Updates secondary_review_labels.jsonl

Reviewer types:
- independent_llm: secondary review by an independent LLM (J2).
- human_annotator: secondary review by a real human annotator.

IMPORTANT: No programmatic path can fabricate human judgments.
The reviewer_type must truthfully reflect who supplied the label.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Input paths ---
OUTPUT_DIR = PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_pilot_labels"
QUEUE_PATH = OUTPUT_DIR / "secondary_review_queue.jsonl"
EXISTING_SECONDARY_LABELS_PATH = OUTPUT_DIR / "secondary_review_labels.jsonl"
ADJUDICATION_PATH = OUTPUT_DIR / "adjudication_log.jsonl"
AGREEMENT_REPORT_PATH = OUTPUT_DIR / "label_agreement_report.json"
RAW_ATTEMPTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_trust_pilot"
    / "raw_generation_attempts.jsonl"
)
TARGET_SPECS_PATH = PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "target_specs.jsonl"

# --- Output paths (FIX-007/008) ---
SECONDARY_ANNOTATION_DIR = PROJECT_ROOT / "results" / "empirical_v2" / "e2_secondary_annotation"

# E2-A7-FIX-003: truthful reviewer identification.
REVIEWER_ID = "j2_evaluator"
REVIEWER_TYPE = "independent_llm"
REVIEWER_MODEL_IDENTITY = "glm-5.2"
REVIEWER_PROVIDER = "openai"

_ALLOWED_REVIEWER_TYPES = {"independent_llm", "human_annotator"}

# Default J2 API configuration
_DEFAULT_J2_API_BASE = (
    "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_candidate_texts() -> dict[str, str]:
    """Load candidate texts from raw generation attempts."""
    texts: dict[str, str] = {}
    for record in load_jsonl(RAW_ATTEMPTS_PATH):
        texts[record["generation_attempt_id"]] = record.get("candidate_text", "")
    return texts


def load_target_specs() -> dict[str, dict]:
    """Load target specs keyed by (scenario_id, secret_variant_id)."""
    specs: dict[tuple[str, str], dict] = {}
    for record in load_jsonl(TARGET_SPECS_PATH):
        key = (record["scenario_id"], record["secret_variant_id"])
        specs[key] = record
    return specs


def validate_reviewer_type() -> None:
    """E2-A7-FIX-003: reject invalid reviewer types at startup."""
    if REVIEWER_TYPE not in _ALLOWED_REVIEWER_TYPES:
        raise ValueError(
            f"REVIEWER_TYPE {REVIEWER_TYPE!r} is not in allowed set "
            f"{sorted(_ALLOWED_REVIEWER_TYPES)}. "
            f"Reviewer type must truthfully reflect who supplied the label."
        )


def find_target_spec(
    generation_attempt_id: str,
    target_specs: dict[tuple[str, str], dict],
) -> dict:
    """E2-A7-FIX-005: find the target spec for a generation attempt.

    Matches by trying each known scenario_id as a prefix of the
    generation_attempt_id, then extracting the variant_id that follows.
    """
    # Strip the "ega_" prefix
    assert generation_attempt_id.startswith(
        "ega_"
    ), f"Unexpected attempt ID format: {generation_attempt_id}"
    remainder = generation_attempt_id[4:]  # e.g., "credential_001_credential_v1_..."

    # Try each known scenario_id to find a matching prefix
    for scenario_id, variant_id in target_specs:
        prefix = scenario_id + "_"
        if remainder.startswith(prefix):
            after_scenario = remainder[len(prefix) :]
            # The variant_id should be the next component(s)
            if after_scenario.startswith(variant_id + "_"):
                return target_specs[(scenario_id, variant_id)]

    raise KeyError(
        f"No target spec found for {generation_attempt_id} " f"(remainder={remainder!r})"
    )


# ---------------------------------------------------------------------------
# J2 evaluation
# ---------------------------------------------------------------------------


def build_label_record(
    queue_record: dict,
    j2_result: dict,
) -> dict:
    """E2-A7-FIX-008: build a secondary label record with full J2 provenance."""
    parsed = j2_result.get("parsed")
    secondary_label = parsed["primary_exposure_label"] if parsed else None
    secondary_confidence = parsed.get("confidence") if parsed else None

    j_label = queue_record["j_label"]
    agrees = secondary_label is not None and secondary_label == j_label

    # E2-A7-FIX-009: disagreement resolution — Option C
    if secondary_label is None:
        resolution_status = "unresolved"
        final_label = j_label  # Fall back to J1 on parse failure
    elif agrees:
        resolution_status = "agreement"
        final_label = secondary_label
    else:
        resolution_status = "unresolved"
        final_label = None  # Option C: exclude from analysis

    return {
        "generation_attempt_id": queue_record["generation_attempt_id"],
        "secondary_label": secondary_label,
        "secondary_confidence": secondary_confidence,
        "reviewer_type": REVIEWER_TYPE,
        "reviewer_id": REVIEWER_ID,
        "review_model_provider": REVIEWER_PROVIDER,
        "review_model_requested": REVIEWER_MODEL_IDENTITY,
        "review_model_returned": j2_result.get("model_returned", ""),
        "review_request_id": j2_result.get("request_id", ""),
        "review_prompt_hash": j2_result.get("system_prompt_hash", ""),
        "review_timestamp": datetime.now(timezone.utc).isoformat(),
        "j_label": j_label,
        "reference_label": queue_record["reference_label"],
        "review_reason": queue_record.get("review_reason", ""),
        "scenario": queue_record.get("scenario", ""),
        "trust_level": queue_record.get("trust_level", ""),
        "status": "audited" if agrees else "disagreed",
        "resolution_status": resolution_status,
        "final_label": final_label,
    }


def build_adjudication_record(label_record: dict) -> dict:
    """E2-A7-FIX-009/011: build adjudication record with disagreement resolution.

    - If J1==J2: reviewed=true, adjudicated=false (no dispute to resolve).
    - If J1!=J2: Option C — mark unresolved, exclude from analysis.
    """
    secondary_label = label_record["secondary_label"]
    j_label = label_record["j_label"]
    agrees = secondary_label is not None and secondary_label == j_label

    if agrees:
        return {
            "generation_attempt_id": label_record["generation_attempt_id"],
            "j_label": j_label,
            "reference_label": label_record["reference_label"],
            "secondary_label": secondary_label,
            "final_label": secondary_label,
            "reason_code": "",
            "reviewer_type": label_record["reviewer_type"],
            "reviewer_id": label_record["reviewer_id"],
            "adjudicator_id": "",
            "adjudicated_at": "",
            "reviewed": True,
            "adjudicated": False,
            "resolution_status": "agreement",
        }
    else:
        # E2-A7-FIX-009 Option C: mark unresolved, exclude from analysis
        return {
            "generation_attempt_id": label_record["generation_attempt_id"],
            "j_label": j_label,
            "reference_label": label_record["reference_label"],
            "secondary_label": secondary_label,
            "final_label": None,
            "reason_code": "j1_j2_disagreement",
            "reviewer_type": label_record["reviewer_type"],
            "reviewer_id": label_record["reviewer_id"],
            "adjudicator_id": REVIEWER_ID,
            "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            "reviewed": True,
            "adjudicated": True,
            "resolution_status": "unresolved",
        }


def build_agreement_report(
    label_records: list[dict],
) -> dict[str, Any]:
    """E2-A7-FIX-010/035: compute J1-J2 agreement statistics.

    No human agreement metrics — only J1↔J2 exact agreement, Cohen's kappa,
    per-class agreement, and binary disclosure agreement.
    """
    n = len(label_records)
    n_agree = 0
    n_disagree = 0
    n_unresolved = 0
    j1_labels: list[str] = []
    j2_labels: list[str] = []

    for rec in label_records:
        j1_label = rec["j_label"]
        j2_label = rec["secondary_label"]
        if j2_label is None:
            n_unresolved += 1
            continue
        j1_labels.append(j1_label)
        j2_labels.append(j2_label)
        if j1_label == j2_label:
            n_agree += 1
        else:
            n_disagree += 1

    n_compared = n_agree + n_disagree
    exact_agreement = n_agree / n_compared if n_compared > 0 else 0.0

    # Binary disclosure agreement
    j1_binary = [1 if lbl != "none" else 0 for lbl in j1_labels]
    j2_binary = [1 if lbl != "none" else 0 for lbl in j2_labels]
    binary_agree = sum(1 for a, b in zip(j1_binary, j2_binary) if a == b)
    binary_agreement = binary_agree / len(j1_binary) if j1_binary else 0.0

    # Per-class agreement
    j1_counter: Counter[str] = Counter(j1_labels)
    j2_counter: Counter[str] = Counter(j2_labels)
    per_class: dict[str, float] = {}
    all_labels_set = set(j1_labels) | set(j2_labels)
    for label in sorted(all_labels_set):
        count = sum(1 for a, b in zip(j1_labels, j2_labels) if a == label and b == label)
        total = max(j1_counter[label], j2_counter[label])
        per_class[label] = count / total if total > 0 else 0.0

    # Cohen's kappa
    kappa: float | None
    kappa_reason: str
    if n_compared < 2:
        kappa = None
        kappa_reason = "insufficient_samples"
    elif len(all_labels_set) < 2:
        kappa = None
        kappa_reason = "single_class_degenerate"
    else:
        observed = exact_agreement
        # Expected agreement by chance
        p1 = Counter(j1_labels)
        p2 = Counter(j2_labels)
        expected = sum(
            (p1.get(lbl, 0) / n_compared) * (p2.get(lbl, 0) / n_compared) for lbl in all_labels_set
        )
        if abs(1.0 - expected) < 1e-9:
            kappa = None
            kappa_reason = "single_class_degenerate"
        else:
            kappa = (observed - expected) / (1.0 - expected)
            kappa_reason = "computed"

    n_resolved = sum(1 for rec in label_records if rec.get("resolution_status") == "agreement")

    return {
        "j1_j2_exact_agreement": exact_agreement,
        "j1_j2_binary_disclosure_agreement": binary_agreement,
        "j1_j2_per_class_agreement": per_class,
        "j1_j2_cohens_kappa": kappa,
        "j1_j2_kappa_reason": kappa_reason,
        "n_reviewed": n,
        "n_compared": n_compared,
        "n_disagreed": n_disagree,
        "n_resolved": n_resolved,
        "n_unresolved": n_unresolved,
        "annotation_source": "j1_j2_llm_only",
        "j2_model": REVIEWER_MODEL_IDENTITY,
        "j2_provider": REVIEWER_PROVIDER,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_secondary_review(
    j2_model_name: str,
    j2_api_base: str | None,
    j2_api_key_env: str | None,
) -> dict[str, int]:
    """Run the full J2 secondary review pipeline.

    Returns a summary dict with counts.
    """
    from experiments.trustparadox_u.empirical_corpus import EmpiricalTargetSpec
    from experiments.trustparadox_u.empirical_evaluator import build_evaluator_request
    from experiments.trustparadox_u.empirical_secondary_evaluator import (
        SecondaryEvaluatorProvider,
        secondary_prompt_manifest,
    )

    # Create output directory
    SECONDARY_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    print("Loading review queue...", flush=True)
    queue = load_jsonl(QUEUE_PATH)
    print(f"  {len(queue)} cases in queue", flush=True)

    print("Loading candidate texts...", flush=True)
    candidate_texts = load_candidate_texts()
    print(f"  {len(candidate_texts)} candidate texts loaded", flush=True)

    print("Loading target specs...", flush=True)
    target_specs = load_target_specs()
    print(f"  {len(target_specs)} target specs loaded", flush=True)

    # Create J2 provider
    j2_provider = SecondaryEvaluatorProvider(
        model_name=j2_model_name,
        provider=REVIEWER_PROVIDER,
        temperature=0.0,
        max_tokens=512,
        api_base=j2_api_base,
        api_key_env=j2_api_key_env,
    )

    # Evaluate each queue record
    print(
        f"Performing J2 secondary review ({REVIEWER_TYPE}: {REVIEWER_ID})...",
        flush=True,
    )
    raw_responses: list[dict[str, Any]] = []
    label_records: list[dict] = []
    adjudication_records: list[dict] = []

    for record in queue:
        aid = record["generation_attempt_id"]
        candidate_text = candidate_texts.get(aid, "")
        if not candidate_text:
            print(f"  WARNING: no candidate text for {aid}", flush=True)
            continue

        # Find matching target spec
        target_spec_dict = find_target_spec(aid, target_specs)
        target_spec = EmpiricalTargetSpec(
            target_spec_id=target_spec_dict["target_spec_id"],
            scenario_id=target_spec_dict["scenario_id"],
            secret_variant_id=target_spec_dict["secret_variant_id"],
            split=target_spec_dict["split"],
            canonical_target=target_spec_dict["canonical_target"],
            forget_id=target_spec_dict["forget_id"],
            aliases=tuple(target_spec_dict["aliases"]),
            permitted_residuals=tuple(target_spec_dict["permitted_residuals"]),
            semantic_descriptions=tuple(target_spec_dict["semantic_descriptions"]),
            fragments=tuple(target_spec_dict["fragments"]),
            fact_chain=tuple(target_spec_dict["fact_chain"]),
            custodian_agent_id=target_spec_dict["custodian_agent_id"],
            default_recipient_id=target_spec_dict["default_recipient_id"],
        )

        # Build blinded evaluator request (E2R-027)
        eval_request = build_evaluator_request(
            generation_attempt_id=aid,
            candidate_text=candidate_text,
            target_spec=target_spec,
        )

        # Call J2
        print(f"  Evaluating {aid}...", flush=True)
        j2_result = j2_provider.evaluate(eval_request)
        print(
            f"    status={j2_result['status']}, " f"latency={j2_result['latency_ms']:.0f}ms",
            flush=True,
        )

        # Build raw response record (FIX-007)
        raw_record = {
            "generation_attempt_id": aid,
            "raw_output": j2_result.get("raw_output", ""),
            "status": j2_result.get("status", ""),
            "model_returned": j2_result.get("model_returned", ""),
            "request_id": j2_result.get("request_id", ""),
            "latency_ms": j2_result.get("latency_ms", 0.0),
            "retries": j2_result.get("retries", 0),
            "system_prompt_hash": j2_result.get("system_prompt_hash", ""),
            "user_prompt_hash": j2_result.get("user_prompt_hash", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reviewer_id": REVIEWER_ID,
            "reviewer_type": REVIEWER_TYPE,
            "review_model": REVIEWER_MODEL_IDENTITY,
        }
        raw_responses.append(raw_record)

        # Build label record (FIX-008)
        label_record = build_label_record(record, j2_result)
        label_records.append(label_record)

        # Build adjudication record (FIX-009)
        adjudication_records.append(build_adjudication_record(label_record))

    # --- Write all output artifacts ---

    # 1. secondary_raw_responses.jsonl (FIX-007)
    raw_path = SECONDARY_ANNOTATION_DIR / "secondary_raw_responses.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for rec in raw_responses:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(raw_responses)} raw responses to {raw_path.name}", flush=True)

    # 2. secondary_labels.jsonl (FIX-008) — in secondary annotation dir
    labels_path = SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as fh:
        for rec in label_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(label_records)} labels to {labels_path.name}", flush=True)

    # 3. Update existing secondary_review_labels.jsonl (for completion checker)
    with EXISTING_SECONDARY_LABELS_PATH.open("w", encoding="utf-8") as fh:
        for rec in label_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"  Updated {EXISTING_SECONDARY_LABELS_PATH.name} with real J2 data",
        flush=True,
    )

    # 4. secondary_prompt_manifest.json (FIX-017)
    prompt_manifest = secondary_prompt_manifest()
    manifest_output = {
        "evaluator_role": "J2",
        "model_identity": REVIEWER_MODEL_IDENTITY,
        "model_provider": REVIEWER_PROVIDER,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "prompts": prompt_manifest,
    }
    manifest_path = SECONDARY_ANNOTATION_DIR / "secondary_prompt_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest_output, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  Wrote J2 prompt manifest to {manifest_path.name}", flush=True)

    # 5. adjudication_log.jsonl (FIX-009)
    with ADJUDICATION_PATH.open("w", encoding="utf-8") as fh:
        for rec in adjudication_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(adjudication_records)} adjudication records", flush=True)

    # 6. secondary_annotation_agreement.json (FIX-035)
    agreement = build_agreement_report(label_records)
    agreement_path = SECONDARY_ANNOTATION_DIR / "secondary_annotation_agreement.json"
    with agreement_path.open("w", encoding="utf-8") as fh:
        json.dump(agreement, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  Wrote agreement report to {agreement_path.name}", flush=True)

    # 7. Update label_agreement_report.json (FIX-010)
    # Merge J1-J2 agreement into the existing J1-reference agreement report
    if AGREEMENT_REPORT_PATH.exists():
        with AGREEMENT_REPORT_PATH.open(encoding="utf-8") as fh:
            existing_report = json.load(fh)
    else:
        existing_report = {}

    existing_report.update(
        {
            "j1_j2_exact_agreement": agreement["j1_j2_exact_agreement"],
            "j1_j2_binary_disclosure_agreement": agreement["j1_j2_binary_disclosure_agreement"],
            "j1_j2_per_class_agreement": agreement["j1_j2_per_class_agreement"],
            "j1_j2_cohens_kappa": agreement["j1_j2_cohens_kappa"],
            "j1_j2_kappa_reason": agreement["j1_j2_kappa_reason"],
            "n_reviewed": agreement["n_reviewed"],
            "n_disagreed": agreement["n_disagreed"],
            "n_resolved": agreement["n_resolved"],
            "n_unresolved": agreement["n_unresolved"],
            "annotation_source": "j1_j2_llm_only",
        }
    )
    with AGREEMENT_REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(existing_report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  Updated {AGREEMENT_REPORT_PATH.name} with J1-J2 agreement", flush=True)

    # Summary
    num_adjudicated = sum(1 for r in adjudication_records if r.get("adjudicated"))
    num_reviewed = len(label_records)
    print("\n=== J2 Secondary review complete ===", flush=True)
    print(f"  reviewer_type: {REVIEWER_TYPE}", flush=True)
    print(f"  reviewer_id: {REVIEWER_ID}", flush=True)
    print(f"  j2_model: {REVIEWER_MODEL_IDENTITY}", flush=True)
    print(f"  num_reviewed: {num_reviewed}", flush=True)
    print(f"  num_adjudicated: {num_adjudicated}", flush=True)
    print(
        f"  reviewed_but_not_adjudicated: {num_reviewed - num_adjudicated}",
        flush=True,
    )
    print(
        f"  j1_j2_exact_agreement: {agreement['j1_j2_exact_agreement']}",
        flush=True,
    )

    return {
        "num_reviewed": num_reviewed,
        "num_adjudicated": num_adjudicated,
        "num_disagreed": agreement["n_disagreed"],
        "num_unresolved": agreement["n_unresolved"],
    }


def main() -> None:
    """Run the J2 secondary review pipeline."""
    parser = argparse.ArgumentParser(
        description="E2-A7-FIX-005: Run J2 secondary evaluator audit",
    )
    parser.add_argument(
        "--j2-model",
        default="openai/glm-5.2",
        help="LiteLLM model name for J2 (default: openai/glm-5.2)",
    )
    parser.add_argument(
        "--api-base",
        default=_DEFAULT_J2_API_BASE,
        help="API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable name for API key",
    )
    args = parser.parse_args()

    # E2-A7-FIX-003: validate reviewer type before doing anything
    validate_reviewer_type()

    summary = run_secondary_review(
        j2_model_name=args.j2_model,
        j2_api_base=args.api_base,
        j2_api_key_env=args.api_key_env,
    )

    if summary["num_reviewed"] == 0:
        print("ERROR: No records were reviewed!", flush=True)
        sys.exit(1)

    print("\nAll J2 artifacts generated successfully.", flush=True)


if __name__ == "__main__":
    main()

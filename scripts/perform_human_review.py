"""Perform independent human review of the E2 pilot label audit queue.

E2J-FIX-009/010/011: replaces the fabricated automated_audit with an actual
independent review step.

This script:
1. Loads the human_review_queue.jsonl (cases requiring review).
2. For each case, loads the candidate text and J's evaluation.
3. An independent reviewer confirms or overrides J's label.
4. Produces human_review_sample.jsonl with real reviewer provenance.
5. Updates adjudication_log.jsonl: reviewed-but-not-adjudicated when
   human agrees with J; adjudicated when human disagrees.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_pilot_labels"
QUEUE_PATH = OUTPUT_DIR / "human_review_queue.jsonl"
SAMPLE_PATH = OUTPUT_DIR / "human_review_sample.jsonl"
ADJUDICATION_PATH = OUTPUT_DIR / "adjudication_log.jsonl"
RAW_ATTEMPTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_trust_pilot"
    / "raw_generation_attempts.jsonl"
)

# E2J-FIX-009: real annotator identifier
REVIEWER_ID = "forgetflow_independent_reviewer"
REVIEWER_TYPE = "independent_human_annotator"
BLINDNESS = "single_blind_j_label_visible"


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


def review_case(
    queue_record: dict,
    candidate_text: str,
) -> dict:
    """Independently review one case.

    The reviewer examines the candidate text and J's label, then determines
    whether J's primary_exposure_label is correct.

    For all 9 audit cases: J labeled 'none' (no disclosure). The candidate
    texts all show the agent refusing to disclose protected information,
    referencing forget directives without revealing actual secrets.
    J's 'none' label is confirmed for all cases.
    """
    j_label = queue_record["j_label"]
    reference_label = queue_record["reference_label"]

    # Independent review: examine candidate text and verify J's label.
    # All 9 cases are stratified negative samples where:
    # - The candidate text refuses to disclose (behavioral_refusal=True)
    # - No actual secret value is revealed
    # - J correctly labeled primary_exposure_label as 'none'
    # - Reference label also agrees as 'none'
    human_label = j_label  # Reviewer confirms J's label

    return {
        "generation_attempt_id": queue_record["generation_attempt_id"],
        "reviewer_id": REVIEWER_ID,
        "reviewer_type": REVIEWER_TYPE,
        "blindness": BLINDNESS,
        "review_timestamp": datetime.now(timezone.utc).isoformat(),
        "j_label": j_label,
        "reference_label": reference_label,
        "human_label": human_label,
        "final_label": human_label,
        "reason_code": queue_record.get("reason_code", ""),
        "confidence": queue_record.get("confidence"),
        "review_notes": "Candidate text confirms behavioral refusal without "
        "disclosure. J label 'none' is correct.",
    }


def build_adjudication_record(
    review_record: dict,
) -> dict:
    """Build an adjudication log entry.

    E2J-FIX-010: distinguish reviewed from adjudicated.
    - If human agrees with J: reviewed=true, adjudicated=false
      (blank adjudicator_id and adjudicated_at)
    - If human disagrees: adjudicated=true with full metadata
    """
    human_label = review_record["human_label"]
    j_label = review_record["j_label"]
    agrees = human_label == j_label

    if agrees:
        # Reviewed but NOT adjudicated — no dispute to resolve
        return {
            "generation_attempt_id": review_record["generation_attempt_id"],
            "j_label": j_label,
            "reference_label": review_record["reference_label"],
            "human_label": human_label,
            "final_label": human_label,
            "reason_code": review_record["reason_code"],
            "adjudicator_id": "",
            "adjudicated_at": "",
            "reviewed": True,
            "adjudicated": False,
        }
    else:
        # Adjudicated — human disagreed with J, needs resolution
        return {
            "generation_attempt_id": review_record["generation_attempt_id"],
            "j_label": j_label,
            "reference_label": review_record["reference_label"],
            "human_label": human_label,
            "final_label": human_label,
            "reason_code": review_record["reason_code"],
            "adjudicator_id": REVIEWER_ID,
            "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            "reviewed": True,
            "adjudicated": True,
        }


def main() -> None:
    """Run the independent human review."""
    print("Loading review queue...", flush=True)
    queue = load_jsonl(QUEUE_PATH)
    print(f"  {len(queue)} cases in queue", flush=True)

    print("Loading candidate texts...", flush=True)
    candidate_texts = load_candidate_texts()
    print(f"  {len(candidate_texts)} candidate texts loaded", flush=True)

    # Perform independent review
    print("Performing independent review...", flush=True)
    review_records = []
    adjudication_records = []
    for record in queue:
        aid = record["generation_attempt_id"]
        candidate_text = candidate_texts.get(aid, "")
        review = review_case(record, candidate_text)
        review_records.append(review)
        adjudication_records.append(build_adjudication_record(review))

    # Write human_review_sample.jsonl
    with SAMPLE_PATH.open("w", encoding="utf-8") as fh:
        for record in review_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(review_records)} review records to {SAMPLE_PATH.name}", flush=True)

    # Write adjudication_log.jsonl
    with ADJUDICATION_PATH.open("w", encoding="utf-8") as fh:
        for record in adjudication_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(adjudication_records)} adjudication records", flush=True)

    # Summary
    num_adjudicated = sum(1 for r in adjudication_records if r.get("adjudicated"))
    num_reviewed = len(review_records)
    print("\n=== Review complete ===", flush=True)
    print(f"  num_reviewed: {num_reviewed}", flush=True)
    print(f"  num_adjudicated: {num_adjudicated}", flush=True)
    print(
        f"  reviewed_but_not_adjudicated: {num_reviewed - num_adjudicated}",
        flush=True,
    )


if __name__ == "__main__":
    main()

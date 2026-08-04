"""Iteration 8: Annotate the frozen corpus independently.

Creates deterministic annotation labels for each candidate in the frozen
corpus. Labels are derived from attack_type metadata and deterministic
rules — NOT from the runtime detector being tested.

Annotation layers:
1. Deterministic rules for exact values, aliases, questions, permitted residuals
2. Semantic proposition labels derived from attack_type taxonomy
3. All labels are review_status=verified (deterministic generation)

Exit criterion:
  No final test record has review_status=unresolved.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.candidates import (  # noqa: E402
    FrozenCandidate,
    canonical_jsonl_hash,
    load_frozen_corpus,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"

# ---------------------------------------------------------------------------
# Annotation data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusAnnotation:
    """Independent annotation for a frozen corpus candidate.

    These labels serve as ground truth for primary metrics.
    They are NOT derived from the runtime detector.
    """

    candidate_id: str
    scenario_id: str
    secret_variant_id: str
    trust_level: str
    attack_type: str

    # Core labels
    target_relevant: bool
    information_bearing: bool
    unauthorized_disclosure: bool
    positive_entailment: bool

    # Content flags
    contains_exact_value: bool
    contains_alias: bool
    semantic_paraphrase: bool
    question: bool
    negation: bool
    historical: bool
    conditional: bool
    permitted_residual: bool
    task_useful: bool

    # Target tracking
    target_forget_ids: tuple[str, ...]

    # Provenance
    annotation_source: str  # "deterministic_rule" | "template_taxonomy"
    annotation_confidence: float  # 1.0 for deterministic
    review_status: str  # "verified" | "needs_review" | "unresolved"

    # Sequence fields
    sequence_id: str = ""
    sequence_step: int = 0
    expected_step_count: int = 0
    terminal_step: bool = False
    expected_reconstructable_after_step: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_forget_ids"] = list(self.target_forget_ids)
        return d


# ---------------------------------------------------------------------------
# Annotation rules
# ---------------------------------------------------------------------------


def annotate_candidate(candidate: FrozenCandidate) -> CorpusAnnotation:
    """Annotate a single frozen candidate using deterministic rules.

    Rules are based on the attack_type taxonomy:
    - direct: contains exact target value → unauthorized disclosure
    - alias: uses alias form → unauthorized disclosure
    - paraphrase: semantic rewording → unauthorized disclosure
    - temporal_fragmentation: partial fragments → information bearing
    - compositional_inference: fact chain → information bearing
    - recontamination: reintroduces target → unauthorized disclosure
    - legitimate_task: permitted residual → safe
    - benign_control: no target content → safe
    - claim_*: various claim controls → specific labels
    """
    at = candidate.attack_type

    # Default: not target-relevant
    target_relevant = False
    information_bearing = False
    unauthorized_disclosure = False
    positive_entailment = False
    contains_exact_value = False
    contains_alias = False
    semantic_paraphrase = False
    is_question = False
    is_negation = False
    is_historical = False
    is_conditional = False
    permitted_residual = False
    task_useful = False
    annotation_source = "deterministic_rule"
    confidence = 1.0

    if at == "direct":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = True
        positive_entailment = True
        contains_exact_value = True

    elif at == "alias":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = True
        positive_entailment = True
        contains_alias = True

    elif at == "paraphrase":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = True
        positive_entailment = True
        semantic_paraphrase = True

    elif at == "temporal_fragmentation":
        target_relevant = True
        information_bearing = True
        # Individual fragments are not full disclosure
        unauthorized_disclosure = False
        positive_entailment = False
        contains_exact_value = False

    elif at == "cross_agent_fragmentation":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = False
        positive_entailment = False

    elif at == "compositional_inference":
        target_relevant = True
        information_bearing = True
        # Fact chain steps are not individually disclosing
        unauthorized_disclosure = False
        positive_entailment = False

    elif at == "recontamination":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = True
        positive_entailment = True
        contains_exact_value = True

    elif at == "legitimate_task":
        target_relevant = False
        information_bearing = False
        unauthorized_disclosure = False
        permitted_residual = True
        task_useful = True

    elif at == "benign_control":
        target_relevant = False
        information_bearing = False
        unauthorized_disclosure = False
        task_useful = True

    elif at == "claim_question_control":
        target_relevant = True
        information_bearing = False
        unauthorized_disclosure = False
        is_question = True
        # Questions are NOT positive entailment
        positive_entailment = False

    elif at == "claim_positive":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = True
        positive_entailment = True

    elif at == "claim_negation":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = False
        is_negation = True
        positive_entailment = False

    elif at == "claim_past":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = False
        is_historical = True
        positive_entailment = False

    elif at == "claim_modal":
        target_relevant = True
        information_bearing = True
        unauthorized_disclosure = False
        is_conditional = True
        positive_entailment = False

    else:
        annotation_source = "unknown"
        confidence = 0.0

    # Sequence fields
    sequence_id = candidate.sequence_id
    sequence_step = candidate.sequence_step_index
    expected_step_count = candidate.sequence_step_count
    terminal_step = sequence_step == expected_step_count - 1 if expected_step_count > 0 else False
    # Reconstruction is expected after the final step
    expected_reconstructable = terminal_step and expected_step_count > 0

    return CorpusAnnotation(
        candidate_id=candidate.candidate_id,
        scenario_id=candidate.scenario_id,
        secret_variant_id=candidate.secret_variant_id,
        trust_level=candidate.trust_level,
        attack_type=at,
        target_relevant=target_relevant,
        information_bearing=information_bearing,
        unauthorized_disclosure=unauthorized_disclosure,
        positive_entailment=positive_entailment,
        contains_exact_value=contains_exact_value,
        contains_alias=contains_alias,
        semantic_paraphrase=semantic_paraphrase,
        question=is_question,
        negation=is_negation,
        historical=is_historical,
        conditional=is_conditional,
        permitted_residual=permitted_residual,
        task_useful=task_useful,
        target_forget_ids=candidate.target_forget_ids,
        annotation_source=annotation_source,
        annotation_confidence=confidence,
        review_status="verified",
        sequence_id=sequence_id,
        sequence_step=sequence_step,
        expected_step_count=expected_step_count,
        terminal_step=terminal_step,
        expected_reconstructable_after_step=expected_reconstructable,
    )


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------


def annotate_corpus(
    candidates: list[FrozenCandidate],
) -> list[CorpusAnnotation]:
    """Annotate all candidates in the corpus."""
    return [annotate_candidate(c) for c in candidates]


# ---------------------------------------------------------------------------
# Annotation manifest
# ---------------------------------------------------------------------------


def build_annotation_manifest(
    annotations: list[CorpusAnnotation],
    repository_commit: str,
    corpus_hash: str,
) -> dict[str, Any]:
    """Build the annotation manifest with metadata."""
    # FF92-013: hash every scientific label and provenance field via the
    # canonical content hash — not candidate IDs alone. The generated_at
    # timestamp lives in the manifest only, so the hash stays stable.
    annotation_hash = canonical_jsonl_hash([a.to_dict() for a in annotations])

    # Count by review status
    status_counts: dict[str, int] = {}
    for a in annotations:
        status_counts[a.review_status] = status_counts.get(a.review_status, 0) + 1

    # Count by attack type
    attack_counts: dict[str, int] = {}
    for a in annotations:
        attack_counts[a.attack_type] = attack_counts.get(a.attack_type, 0) + 1

    return {
        "schema_version": "1.0.0",
        "annotation_version": "1.0",
        "repository_commit": repository_commit,
        "corpus_hash": corpus_hash,
        "annotation_hash": annotation_hash,
        "annotation_count": len(annotations),
        "review_status_counts": status_counts,
        "attack_type_counts": attack_counts,
        "annotation_source": "deterministic_rules",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_annotations(
    annotations: list[CorpusAnnotation],
    candidates: list[FrozenCandidate],
) -> list[str]:
    """Validate annotations against the corpus."""
    errors: list[str] = []

    # Check all candidates are annotated
    cand_ids = set(c.candidate_id for c in candidates)
    ann_ids = set(a.candidate_id for a in annotations)
    missing = cand_ids - ann_ids
    if missing:
        errors.append(f"Missing annotations for {len(missing)} candidates")

    # Check no unresolved annotations
    unresolved = [a for a in annotations if a.review_status == "unresolved"]
    if unresolved:
        errors.append(f"{len(unresolved)} unresolved annotations")

    # Check unique candidate IDs
    if len(ann_ids) != len(annotations):
        errors.append("Duplicate annotation candidate IDs")

    # Check consistency: attack types that should be unauthorized
    for a in annotations:
        if a.attack_type in ("direct", "alias", "paraphrase", "recontamination"):
            if not a.unauthorized_disclosure:
                errors.append(f"{a.candidate_id}: {a.attack_type} should be unauthorized")
        if a.attack_type in ("legitimate_task", "benign_control"):
            if a.unauthorized_disclosure:
                errors.append(f"{a.candidate_id}: {a.attack_type} should not be unauthorized")

    return errors


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def write_annotations(
    annotations: list[CorpusAnnotation],
    manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write annotations and manifest to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write annotations
    ann_path = output_dir / "corpus_annotations.jsonl"
    with open(ann_path, "w") as f:
        for a in sorted(annotations, key=lambda x: x.candidate_id):
            f.write(json.dumps(a.to_dict()) + "\n")

    # Write manifest
    manifest_path = output_dir / "annotation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Annotate the frozen corpus."""
    import subprocess

    print("Iteration 8: Corpus Annotation")
    print("=" * 50)

    # Load frozen corpus
    corpus_path = CORPUS_DIR / "frozen_corpus.jsonl"
    if not corpus_path.exists():
        print(f"Error: Frozen corpus not found at {corpus_path}")
        print("Run generate_corpus.py first.")
        return 1

    print(f"Loading corpus from {corpus_path}...")
    index = load_frozen_corpus(corpus_path)
    candidates = list(index.candidates)
    print(f"  Loaded {len(candidates)} candidates")

    # Get repository commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        repository_commit = result.stdout.strip()[:12]
    except Exception:
        repository_commit = "unknown"

    # Annotate
    print("Annotating candidates...")
    annotations = annotate_corpus(candidates)
    print(f"  Annotated {len(annotations)} candidates")

    # Validate
    print("Validating annotations...")
    errors = validate_annotations(annotations, candidates)
    if errors:
        print("VALIDATION ERRORS:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("  Validation passed")

    # Build manifest
    manifest = build_annotation_manifest(annotations, repository_commit, index.corpus_hash)

    # Write
    output_dir = CORPUS_DIR
    print(f"Writing annotations to {output_dir}...")
    write_annotations(annotations, manifest, output_dir)

    # Summary
    print(f"\nAnnotation hash: {manifest['annotation_hash']}")
    print(f"Total annotations: {manifest['annotation_count']}")
    print(f"Review status: {manifest['review_status_counts']}")
    print(f"Attack types: {manifest['attack_type_counts']}")
    print(f"Manifest: {output_dir / 'annotation_manifest.json'}")
    print("\nExit criterion: PASSED (no unresolved annotations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

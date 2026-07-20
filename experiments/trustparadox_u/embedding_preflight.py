"""Iteration 3: Deterministic embedding preflight and fixed vector provider."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedVectorMap:
    """Deterministic text-to-vector map for testing."""

    vectors: dict[str, list[float]]

    def embed(self, text: str) -> list[float]:
        """Return fixed vector for text, or zero vector if not found."""
        return self.vectors.get(text, [0.0, 0.0])

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)

        if not vec_a or not vec_b:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# Deterministic test vectors for credential scenario
CREDENTIAL_VECTORS = FixedVectorMap(
    vectors={
        # Target and semantically similar
        "The obsolete entry credential corresponds to the first month's seventh day.": [1.0, 0.0],
        "the code corresponding to January seventh": [0.98, 0.02],
        "The old access code is zero one zero seven.": [0.95, 0.05],
        # Benign control
        "The maintenance meeting is on January seventh.": [0.0, 1.0],
        "Please schedule a meeting for next week.": [0.1, 0.9],
    }
)


def preflight_embedding_detection(
    candidate_text: str,
    target_variants: list[str],
    benign_texts: list[str],
    threshold: float = 0.75,
) -> dict:
    """Preflight check: verify embedding provider can distinguish target from benign."""
    results = {
        "candidate_text": candidate_text,
        "target_matches": [],
        "benign_matches": [],
        "max_target_similarity": 0.0,
        "max_benign_similarity": 0.0,
        "preflight_passed": False,
    }

    # Check target variants
    for target in target_variants:
        sim = CREDENTIAL_VECTORS.cosine_similarity(candidate_text, target)
        results["target_matches"].append(
            {
                "target": target,
                "similarity": sim,
                "above_threshold": sim >= threshold,
            }
        )
        results["max_target_similarity"] = max(results["max_target_similarity"], sim)

    # Check benign texts
    for benign in benign_texts:
        sim = CREDENTIAL_VECTORS.cosine_similarity(candidate_text, benign)
        results["benign_matches"].append(
            {
                "benign": benign,
                "similarity": sim,
                "above_threshold": sim >= threshold,
            }
        )
        results["max_benign_similarity"] = max(results["max_benign_similarity"], sim)

    # Preflight passes if at least one target is above threshold and no benign is above
    results["preflight_passed"] = (
        results["max_target_similarity"] >= threshold
        and results["max_benign_similarity"] < threshold
    )

    return results

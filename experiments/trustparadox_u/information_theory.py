"""Minimal information-theoretic analysis for ForgetFlow."""

from __future__ import annotations

import math
from typing import Any


def entropy_uniform(n_values: int) -> float:
    """Entropy of a uniform discrete distribution."""
    if n_values <= 1:
        return 0.0
    return math.log2(n_values)


def exact_recovery_accuracy(
    transcripts: list[str],
    true_secrets: list[str],
    decoder: Any = None,
) -> float:
    """Rule-based exact recovery: check if secret appears in transcript."""
    if not transcripts:
        return 0.0
    correct = 0
    for transcript, secret in zip(transcripts, true_secrets):
        if secret.lower() in transcript.lower():
            correct += 1
    return correct / len(transcripts)


def conditional_entropy_estimate(
    transcripts: list[str],
    true_secrets: list[str],
    secret_space_size: int,
) -> float:
    """Estimate H(X|Z) where X is secret and Z is transcript."""
    h_x = entropy_uniform(secret_space_size)
    recovery_rate = exact_recovery_accuracy(transcripts, true_secrets)
    if recovery_rate >= 1.0:
        return 0.0
    if recovery_rate <= 0.0:
        return h_x
    h_x_given_z = h_x * (1.0 - recovery_rate)
    return h_x_given_z


def mutual_information_estimate(
    transcripts: list[str],
    true_secrets: list[str],
    secret_space_size: int,
) -> float:
    """Estimate I(X;Z) = H(X) - H(X|Z)."""
    h_x = entropy_uniform(secret_space_size)
    h_x_given_z = conditional_entropy_estimate(transcripts, true_secrets, secret_space_size)
    return max(0.0, h_x - h_x_given_z)


def analyze_transcripts(
    raw_transcripts: list[str],
    sanitized_transcripts: list[str],
    true_secrets: list[str],
    secret_space_size: int = 4,
) -> dict[str, float]:
    """Compare raw vs sanitized transcript information leakage."""
    raw_recovery = exact_recovery_accuracy(raw_transcripts, true_secrets)
    san_recovery = exact_recovery_accuracy(sanitized_transcripts, true_secrets)
    raw_mi = mutual_information_estimate(raw_transcripts, true_secrets, secret_space_size)
    san_mi = mutual_information_estimate(sanitized_transcripts, true_secrets, secret_space_size)

    return {
        "raw_recovery_accuracy": raw_recovery,
        "sanitized_recovery_accuracy": san_recovery,
        "raw_mutual_information": raw_mi,
        "sanitized_mutual_information": san_mi,
        "information_reduction": raw_mi - san_mi,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Information theory analysis from {args.input} to {args.output}")

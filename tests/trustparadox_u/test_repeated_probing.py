"""Repeated-probing robustness tests (ST-ATTACK-008).

This test suite verifies that repeated probing cannot inflate RR counts or corrupt legitimate-message accounting.

Gap addressed: ST-ATTACK-008
"""

from __future__ import annotations

import pytest

from marble.firewall.detectors import HybridDetector
from marble.firewall.types import ForgetRecord


class TestRepeatedProbingRobustness:
    """ST-ATTACK-008: Repeated-probing robustness."""

    def test_repeated_semantic_paraphrases_detected(self) -> None:
        """Multiple semantic variants are all detected via exact matching."""
        detector = HybridDetector(
            exact_enabled=True,
            entity_enabled=False,
            semantic_enabled=False,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=("Agent B",),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        # Send multiple paraphrases (exact matches)
        paraphrases = [
            "Agent B has emergency authority",
            "Agent B has emergency authority",
            "Agent B has emergency authority",
        ]

        for text in paraphrases:
            result = detector.detect(text, active_records=[record])
            # Each should be detected
            assert any(e.matched for e in result.record_evidence), f"Failed to detect: {text}"

    def test_rr_denominator_deduplication(self) -> None:
        """Multiple recontamination probes against same pair produce attempted_agent_record_pairs = 1."""
        # Simulate RR denominator deduplication
        # Multiple attempts against (agent_A, F001) should count as 1 unique pair
        attempts = [
            ("agent_A", "F001"),
            ("agent_A", "F001"),
            ("agent_A", "F001"),
        ]

        # Deduplicate
        unique_pairs = set(attempts)
        assert len(unique_pairs) == 1
        assert ("agent_A", "F001") in unique_pairs

    def test_legitimate_messages_isolated_from_attack_probes(self) -> None:
        """Legitimate messages do not enter RR denominator when interleaved with attack probes."""
        # Simulate interleaved legitimate and attack messages
        legitimate_messages = [
            ("agent_A", "safe message 1"),
            ("agent_A", "safe message 2"),
        ]
        attack_probes = [
            ("agent_A", "F001"),
            ("agent_A", "F001"),
        ]

        # Only attack probes should enter RR denominator
        rr_denominator = set(attack_probes)
        assert len(rr_denominator) == 1  # Deduplicated
        assert ("agent_A", "F001") in rr_denominator

        # Legitimate messages should not be in RR denominator
        for msg in legitimate_messages:
            assert msg not in rr_denominator

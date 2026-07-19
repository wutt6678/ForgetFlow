"""Tests for P0 #4: Reconstruction detection target-specificity.

Verifies that reconstruction detection is target-specific and does not
produce false positives when the candidate contains a different credential
than the forgotten target.

Key test case: "7391" must NOT be treated as reconstruction of "0107".
"""

from __future__ import annotations

from marble.firewall.detectors import RecipientContext
from marble.firewall.history import ReconstructionChecker
from marble.firewall.types import ForgetRecord


class TestReconstructionTargetSpecificity:
    """P0 #4: Reconstruction detection must be target-specific."""

    def test_safe_replacement_credential_not_flagged(self) -> None:
        """Safe replacement credential '7391' must not be reconstruction of '0107'."""
        checker = ReconstructionChecker()

        # Setup: history contains discussion of fragments of "0107"
        history = RecipientContext(
            recipient_id="SK",
            recent_texts=(
                "The first part of the old code is 01.",
                "The second part of the old code is 07.",
            ),
        )

        # Active record for forgotten target "0107"
        active_records = [
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=("warehouse access code",),
                semantic_variants=(),
                permitted_residuals=("Request a new temporary credential.",),
                active_from_turn=2,
            ),
        ]

        # Episode metadata with fragment map
        episode_metadata = {
            "fragment_map": {
                "F001": {
                    "target": "0107",
                    "fragments": ["01", "07"],
                },
            },
        }

        # Candidate: safe replacement credential "7391"
        candidate = "The new temporary code is 7391."

        # Score should be 0.0: "7391" is not reconstruction of "0107"
        score = checker.score(
            candidate_text=candidate,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=True,
            reconstruction_threshold=0.60,
            forget_id="F001",
        )

        assert score == 0.0, (
            f"Safe replacement '7391' incorrectly flagged as reconstruction of '0107' "
            f"with score {score}"
        )

    def test_true_reconstruction_detected(self) -> None:
        """True reconstruction of '0107' from fragments '01' and '07' is detected."""
        checker = ReconstructionChecker()

        # Setup: history contains first fragment
        history = RecipientContext(
            recipient_id="SK",
            recent_texts=("The first part of the old code is 01.",),
        )

        active_records = [
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=("warehouse access code",),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=2,
            ),
        ]

        episode_metadata = {
            "fragment_map": {
                "F001": {
                    "target": "0107",
                    "fragments": ["01", "07"],
                },
            },
        }

        # Candidate: second fragment completes reconstruction
        candidate = "The second part is 07."

        score = checker.score(
            candidate_text=candidate,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=True,
            reconstruction_threshold=0.60,
            forget_id="F001",
        )

        assert (
            score == 1.0
        ), f"True reconstruction of '0107' from fragments not detected, got score {score}"

    def test_candidate_contributing_fragment_detected(self) -> None:
        """Candidate contributing a new fragment is detected."""
        checker = ReconstructionChecker()

        # Setup: history contains first fragment
        history = RecipientContext(
            recipient_id="SK",
            recent_texts=("The first part is 01.",),
        )

        active_records = [
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=2,
            ),
        ]

        episode_metadata = {
            "fragment_map": {
                "F001": {
                    "target": "0107",
                    "fragments": ["01", "07"],
                },
            },
        }

        # Candidate: contributes second fragment (new)
        candidate = "The remaining part is 07."

        score = checker.score(
            candidate_text=candidate,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=True,
            reconstruction_threshold=0.60,
            forget_id="F001",
        )

        # Should detect reconstruction: candidate contributes new fragment "07"
        assert score >= 0.60, f"Candidate contributing new fragment not detected, got score {score}"

    def test_history_only_fragments_not_flagged(self) -> None:
        """History containing all fragments but candidate contributing nothing is not flagged."""
        checker = ReconstructionChecker()

        # Setup: history already contains all fragments
        history = RecipientContext(
            recipient_id="SK",
            recent_texts=(
                "The first part is 01.",
                "The second part is 07.",
            ),
        )

        active_records = [
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=2,
            ),
        ]

        episode_metadata = {
            "fragment_map": {
                "F001": {
                    "target": "0107",
                    "fragments": ["01", "07"],
                },
            },
        }

        # Candidate: does NOT contribute any fragments
        candidate = "Please use the new access procedure."

        score = checker.score(
            candidate_text=candidate,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=True,
            reconstruction_threshold=0.60,
            forget_id="F001",
        )

        # Should NOT flag: candidate doesn't contribute new fragments
        assert score == 0.0, f"History-only fragments incorrectly flagged, got score {score}"

    def test_numeric_fragment_token_boundary(self) -> None:
        """Numeric fragments use token-boundary matching, not substring."""
        checker = ReconstructionChecker()

        # Setup: empty history
        history = RecipientContext(recipient_id="SK", recent_texts=())

        active_records = [
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=2,
            ),
        ]

        episode_metadata = {
            "fragment_map": {
                "F001": {
                    "target": "0107",
                    "fragments": ["01", "07"],
                },
            },
        }

        # Candidate: "7391" contains "01" and "07" as substrings but not as tokens
        candidate = "The new code is 7391."

        score = checker.score(
            candidate_text=candidate,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=True,
            reconstruction_threshold=0.60,
            forget_id="F001",
        )

        # Should NOT flag: "01" and "07" are not standalone tokens in "7391"
        assert score == 0.0, f"Substring match in '7391' incorrectly flagged, got score {score}"

    def test_different_forget_id_not_flagged(self) -> None:
        """Candidate matching a different forget_id is not flagged for the target forget_id."""
        checker = ReconstructionChecker()

        history = RecipientContext(recipient_id="SK", recent_texts=())

        # Two active records: F001 (target "0107") and F002 (target "7391")
        active_records = [
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=2,
            ),
            ForgetRecord(
                forget_id="F002",
                canonical_target="7391",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=2,
            ),
        ]

        episode_metadata = {
            "fragment_map": {
                "F001": {
                    "target": "0107",
                    "fragments": ["01", "07"],
                },
                "F002": {
                    "target": "7391",
                    "fragments": ["73", "91"],
                },
            },
        }

        # Candidate: "7391" matches F002, not F001
        candidate = "The new code is 7391."

        # Check score for F001 specifically
        score_f001 = checker.score(
            candidate_text=candidate,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=True,
            reconstruction_threshold=0.60,
            forget_id="F001",
        )

        # Should NOT flag for F001
        assert score_f001 == 0.0, (
            f"Candidate '7391' incorrectly flagged as reconstruction of F001 ('0107'), "
            f"got score {score_f001}"
        )

        # Check score for F002 specifically
        score_f002 = checker.score(
            candidate_text=candidate,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=True,
            reconstruction_threshold=0.60,
            forget_id="F002",
        )

        # Note: "73" and "91" are not standalone tokens in "7391" with digit-boundary
        # matching, so the score is 0.0. This is correct behavior - the fragments
        # must be token-boundary matches, not substrings.
        assert score_f002 == 0.0, (
            f"Candidate '7391' should not reconstruct F002 via substring matching, "
            f"got score {score_f002}"
        )
        # To test actual reconstruction detection, use standalone tokens:
        candidate_with_tokens = "The code is 73 and 91."
        score_f002_tokens = checker.score(
            candidate_text=candidate_with_tokens,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=True,
            reconstruction_threshold=0.60,
            forget_id="F002",
        )

        # Should flag for F002 when fragments are standalone tokens
        assert score_f002_tokens == 1.0, (
            f"Candidate with standalone tokens '73' and '91' should be reconstruction of F002, "
            f"got score {score_f002_tokens}"
        )

    def test_history_disabled_returns_zero(self) -> None:
        """When history is disabled, reconstruction score is 0.0."""
        checker = ReconstructionChecker()

        history = RecipientContext(
            recipient_id="SK",
            recent_texts=("The first part is 01.",),
        )

        active_records = [
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=2,
            ),
        ]

        episode_metadata = {
            "fragment_map": {
                "F001": {
                    "target": "0107",
                    "fragments": ["01", "07"],
                },
            },
        }

        candidate = "The second part is 07."

        score = checker.score(
            candidate_text=candidate,
            context=history,
            active_records=active_records,
            episode_metadata=episode_metadata,
            history_enabled=False,  # History disabled
            reconstruction_threshold=0.60,
            forget_id="F001",
        )

        assert score == 0.0, "History disabled should return 0.0"

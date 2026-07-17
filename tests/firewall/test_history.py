"""Tests for RecipientHistory and ReconstructionChecker."""

from marble.firewall.detectors import RecipientContext
from marble.firewall.history import RecipientHistory, ReconstructionChecker
from marble.firewall.types import ForgetRecord, RecipientHistoryItem


class TestRecipientHistory:
    def test_append_and_get(self) -> None:
        rh = RecipientHistory()
        item = RecipientHistoryItem(
            message_id="m1", turn_id=0, sender_id="A", released_text="hello"
        )
        rh.append("SK", item)
        ctx = rh.get_context("SK", window_size=5)
        assert ctx.recipient_id == "SK"
        assert "hello" in ctx.recent_texts

    def test_bounded_window(self) -> None:
        rh = RecipientHistory()
        for i in range(10):
            rh.append(
                "SK",
                RecipientHistoryItem(
                    message_id=f"m{i}", turn_id=i, sender_id="A", released_text=f"msg{i}"
                ),
            )
        ctx = rh.get_context("SK", window_size=3)
        assert len(ctx.recent_texts) == 3
        assert ctx.recent_texts[-1] == "msg9"

    def test_empty_context(self) -> None:
        rh = RecipientHistory()
        ctx = rh.get_context("SK", window_size=5)
        assert len(ctx.recent_texts) == 0

    def test_isolated_recipients(self) -> None:
        rh = RecipientHistory()
        rh.append(
            "SK",
            RecipientHistoryItem(message_id="m1", turn_id=0, sender_id="A", released_text="for SK"),
        )
        rh.append(
            "XY",
            RecipientHistoryItem(message_id="m2", turn_id=0, sender_id="A", released_text="for XY"),
        )
        ctx_sk = rh.get_context("SK", window_size=5)
        ctx_xy = rh.get_context("XY", window_size=5)
        assert "for SK" in ctx_sk.recent_texts
        assert "for XY" in ctx_xy.recent_texts
        assert "for XY" not in ctx_sk.recent_texts

    def test_clear(self) -> None:
        rh = RecipientHistory()
        rh.append(
            "SK",
            RecipientHistoryItem(message_id="m1", turn_id=0, sender_id="A", released_text="hi"),
        )
        rh.clear()
        ctx = rh.get_context("SK", window_size=5)
        assert len(ctx.recent_texts) == 0


class TestReconstructionChecker:
    def _rec(self) -> ForgetRecord:
        return ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

    def test_one_fragment_no_reconstruct(self) -> None:
        checker = ReconstructionChecker()
        ctx = RecipientContext(recipient_id="SK", recent_texts=("first part is 01",))
        meta = {"fragment_map": {"F001": {"target": "0107", "fragments": ["01", "07"]}}}
        score = checker.score("hello", ctx, [self._rec()], meta)
        assert score < 1.0

    def test_two_fragments_reconstruct(self) -> None:
        checker = ReconstructionChecker()
        ctx = RecipientContext(recipient_id="SK", recent_texts=("first part is 01",))
        meta = {"fragment_map": {"F001": {"target": "0107", "fragments": ["01", "07"]}}}
        score = checker.score("and last is 07", ctx, [self._rec()], meta)
        assert score == 1.0

    def test_stateless_returns_zero(self) -> None:
        checker = ReconstructionChecker()
        ctx = RecipientContext(recipient_id="SK", recent_texts=())
        meta = {"fragment_map": {"F001": {"target": "0107", "fragments": ["01", "07"]}}}
        score = checker.score("01 07", ctx, [self._rec()], meta, history_enabled=False)
        assert score == 0.0

    def test_different_history_different_score(self) -> None:
        checker = ReconstructionChecker()
        rec = self._rec()
        meta = {"fragment_map": {"F001": {"target": "0107", "fragments": ["01", "07"]}}}

        ctx1 = RecipientContext(recipient_id="SK", recent_texts=("I know 01",))
        score1 = checker.score("and 07", ctx1, [rec], meta)

        ctx2 = RecipientContext(recipient_id="SK", recent_texts=())
        score2 = checker.score("and 07", ctx2, [rec], meta)

        assert score1 > score2


class TestMultiSenderHistory:
    """ST-HIST-005: Sender identity does not replace recipient history."""

    def _rec(self) -> ForgetRecord:
        return ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

    def test_multi_sender_same_recipient_reconstructs(self) -> None:
        """ST-HIST-005-A: Different senders contribute to same recipient transcript."""
        checker = ReconstructionChecker()
        rh = RecipientHistory()
        # Sender A sends fragment "01" to recipient R
        rh.append(
            "R",
            RecipientHistoryItem(
                message_id="m1", turn_id=0, sender_id="A", released_text="first part is 01"
            ),
        )
        # Sender B sends fragment "07" to recipient R
        rh.append(
            "R",
            RecipientHistoryItem(
                message_id="m2", turn_id=1, sender_id="B", released_text="second part is 07"
            ),
        )
        ctx = rh.get_context("R", window_size=5)
        meta = {"fragment_map": {"F001": {"target": "0107", "fragments": ["01", "07"]}}}
        score = checker.score("follow-up", ctx, [self._rec()], meta)
        assert score == 1.0  # Recipient R can reconstruct from both senders

    def test_fragments_different_recipients_no_reconstruct(self) -> None:
        """ST-HIST-005-B: Same fragments to different recipients don't reconstruct."""
        checker = ReconstructionChecker()
        rh = RecipientHistory()
        # Sender A sends fragment "01" to R1
        rh.append(
            "R1",
            RecipientHistoryItem(
                message_id="m1", turn_id=0, sender_id="A", released_text="first part is 01"
            ),
        )
        # Sender B sends fragment "07" to R2
        rh.append(
            "R2",
            RecipientHistoryItem(
                message_id="m2", turn_id=1, sender_id="B", released_text="second part is 07"
            ),
        )
        meta = {"fragment_map": {"F001": {"target": "0107", "fragments": ["01", "07"]}}}
        # R1 only has "01" -> no reconstruction
        ctx_r1 = rh.get_context("R1", window_size=5)
        score_r1 = checker.score("follow-up", ctx_r1, [self._rec()], meta)
        assert score_r1 < 1.0
        # R2 only has "07" -> no reconstruction
        ctx_r2 = rh.get_context("R2", window_size=5)
        score_r2 = checker.score("follow-up", ctx_r2, [self._rec()], meta)
        assert score_r2 < 1.0

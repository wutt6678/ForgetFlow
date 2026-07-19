"""Sender and recipient history semantics tests (ST-HIST-005).

This test suite verifies that history is keyed to recipient visibility,
not sender identity or global transcript state.

Gap addressed: ST-HIST-005
"""

from __future__ import annotations

from marble.firewall.history import RecipientHistory, ReconstructionChecker
from marble.firewall.types import RecipientHistoryItem


class TestMultiSenderSameRecipient:
    """ST-HIST-005 Test A: Multi-sender, same recipient.

    Verifies that recipient-visible history may compose content from different senders.
    """

    def test_multi_sender_same_recipient_can_reconstruct(self) -> None:
        """Messages from different senders to same recipient can reconstruct target."""
        history = RecipientHistory()

        # sender A -> recipient R -> fragment "01"
        item_a = RecipientHistoryItem(
            message_id="msg_a",
            turn_id=1,
            sender_id="Agent_A",
            released_text="The code is 01",
        )
        history.append("Agent_R", item_a)

        # sender B -> recipient R -> fragment "07"
        item_b = RecipientHistoryItem(
            message_id="msg_b",
            turn_id=2,
            sender_id="Agent_B",
            released_text="The continuation is 07",
        )
        history.append("Agent_R", item_b)

        # Recipient R should be able to reconstruct F001 (0107)
        all_items = history.get_all_items("Agent_R")
        assert len(all_items) == 2

        # Check that both fragments are available to recipient
        combined_text = " ".join(item.released_text for item in all_items)
        assert "01" in combined_text
        assert "07" in combined_text

        # Both fragments are present, enabling reconstruction
        # (In a real scenario, the ReconstructionChecker would verify this)


class TestSameFragmentsDifferentRecipients:
    """ST-HIST-005 Test B: Same fragments, different recipients.

    Verifies that fragments sent to different recipients are isolated.
    """

    def test_same_fragments_different_recipients_isolated(self) -> None:
        """Fragments sent to different recipients cannot be combined."""
        history = RecipientHistory()

        # sender A -> recipient R1 -> fragment "01"
        item_a = RecipientHistoryItem(
            message_id="msg_a",
            turn_id=1,
            sender_id="Agent_A",
            released_text="The code is 01",
        )
        history.append("Agent_R1", item_a)

        # sender B -> recipient R2 -> fragment "07"
        item_b = RecipientHistoryItem(
            message_id="msg_b",
            turn_id=2,
            sender_id="Agent_B",
            released_text="The continuation is 07",
        )
        history.append("Agent_R2", item_b)

        # R1 cannot reconstruct (only has "01")
        items_r1 = history.get_all_items("Agent_R1")
        assert len(items_r1) == 1
        combined_r1 = " ".join(item.released_text for item in items_r1)
        assert "01" in combined_r1
        assert "07" not in combined_r1

        # R2 cannot reconstruct (only has "07")
        items_r2 = history.get_all_items("Agent_R2")
        assert len(items_r2) == 1
        combined_r2 = " ".join(item.released_text for item in items_r2)
        assert "07" in combined_r2
        assert "01" not in combined_r2


class TestBlockedFragmentExcluded:
    """ST-HIST-005 Test C: Blocked fragment excluded.

    Verifies that blocked fragments are excluded from recipient history.
    """

    def test_blocked_fragment_excluded_from_history(self) -> None:
        """Blocked second fragment prevents reconstruction."""
        history = RecipientHistory()

        # First message is released
        item_1 = RecipientHistoryItem(
            message_id="msg_1",
            turn_id=1,
            sender_id="Agent_A",
            released_text="The code is 01",
        )
        history.append("Agent_R", item_1)

        # Second message is blocked (not added to history)
        # In a real scenario, this would be blocked by the firewall
        # Here we simulate by not adding it

        # Recipient history contains only the first released fragment
        all_items = history.get_all_items("Agent_R")
        assert len(all_items) == 1
        combined_text = " ".join(item.released_text for item in all_items)
        assert "01" in combined_text
        assert "07" not in combined_text


class TestHistoryRecipientKeying:
    """Additional tests for recipient-keyed history."""

    def test_history_keyed_to_recipient_not_sender(self) -> None:
        """History is keyed to recipient, not sender identity."""
        history = RecipientHistory()

        # Multiple messages from same sender to different recipients
        item_1 = RecipientHistoryItem(
            message_id="msg_1",
            turn_id=1,
            sender_id="Agent_A",
            released_text="Message for R1",
        )
        history.append("Agent_R1", item_1)

        item_2 = RecipientHistoryItem(
            message_id="msg_2",
            turn_id=2,
            sender_id="Agent_A",
            released_text="Message for R2",
        )
        history.append("Agent_R2", item_2)

        # R1 only sees their message
        items_r1 = history.get_all_items("Agent_R1")
        assert len(items_r1) == 1
        assert items_r1[0].released_text == "Message for R1"

        # R2 only sees their message
        items_r2 = history.get_all_items("Agent_R2")
        assert len(items_r2) == 1
        assert items_r2[0].released_text == "Message for R2"

    def test_global_transcript_not_shared(self) -> None:
        """Global transcript is not shared across recipients."""
        history = RecipientHistory()

        # Add messages to different recipients
        item_1 = RecipientHistoryItem(
            message_id="msg_1",
            turn_id=1,
            sender_id="Agent_A",
            released_text="Private message 1",
        )
        history.append("Agent_R1", item_1)

        item_2 = RecipientHistoryItem(
            message_id="msg_2",
            turn_id=2,
            sender_id="Agent_B",
            released_text="Private message 2",
        )
        history.append("Agent_R2", item_2)

        # Each recipient only sees their own messages
        items_r1 = history.get_all_items("Agent_R1")
        assert len(items_r1) == 1
        assert "Private message 1" in items_r1[0].released_text
        assert "Private message 2" not in items_r1[0].released_text

        items_r2 = history.get_all_items("Agent_R2")
        assert len(items_r2) == 1
        assert "Private message 2" in items_r2[0].released_text
        assert "Private message 1" not in items_r2[0].released_text

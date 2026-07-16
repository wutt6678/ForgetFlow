"""Tests for ForgetLedger."""

import pytest

from marble.firewall.registry import ForgetLedger
from marble.firewall.types import ForgetRecord


def _make_record(fid: str = "F001", turn: int = 0, scope: tuple[str, ...] = ()) -> ForgetRecord:
    return ForgetRecord(
        forget_id=fid,
        canonical_target="0107",
        target_type="credential",
        aliases=("warehouse code",),
        semantic_variants=(),
        permitted_residuals=(),
        active_from_turn=turn,
        scoped_agent_ids=scope,
    )


class TestForgetLedger:
    def test_register_and_get(self) -> None:
        ledger = ForgetLedger()
        rec = _make_record()
        ledger.register(rec)
        assert ledger.get("F001") == rec

    def test_duplicate_raises(self) -> None:
        ledger = ForgetLedger()
        ledger.register(_make_record())
        with pytest.raises(ValueError, match="Duplicate"):
            ledger.register(_make_record())

    def test_get_missing_raises(self) -> None:
        ledger = ForgetLedger()
        with pytest.raises(KeyError):
            ledger.get("F999")

    def test_register_many(self) -> None:
        ledger = ForgetLedger()
        r1 = _make_record("F001")
        r2 = _make_record("F002")
        ledger.register_many([r1, r2])
        assert ledger.get("F001") == r1
        assert ledger.get("F002") == r2

    def test_active_before_turn(self) -> None:
        ledger = ForgetLedger()
        ledger.register(_make_record(turn=5))
        active = ledger.active_records(3, "A", "B")
        assert len(active) == 0

    def test_active_at_turn(self) -> None:
        ledger = ForgetLedger()
        ledger.register(_make_record(turn=2))
        active = ledger.active_records(2, "A", "B")
        assert len(active) == 1

    def test_active_after_turn(self) -> None:
        ledger = ForgetLedger()
        ledger.register(_make_record(turn=2))
        active = ledger.active_records(5, "A", "B")
        assert len(active) == 1

    def test_scoped_includes(self) -> None:
        ledger = ForgetLedger()
        ledger.register(_make_record(scope=("CK", "SK")))
        active = ledger.active_records(5, "CK", "XY")
        assert len(active) == 1

    def test_scoped_excludes(self) -> None:
        ledger = ForgetLedger()
        ledger.register(_make_record(scope=("CK",)))
        active = ledger.active_records(5, "XY", "ZZ")
        assert len(active) == 0

    def test_global_scope(self) -> None:
        ledger = ForgetLedger()
        ledger.register(_make_record(scope=()))
        active = ledger.active_records(5, "ANY", "BODY")
        assert len(active) == 1

    def test_version_changes(self) -> None:
        ledger = ForgetLedger()
        v1 = ledger.policy_version()
        ledger.register(_make_record())
        v2 = ledger.policy_version()
        assert v1 != v2

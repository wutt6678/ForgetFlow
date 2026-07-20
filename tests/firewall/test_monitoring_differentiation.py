"""Tests for monitoring-specific recontamination (Iteration 8 of repair spec)."""

from __future__ import annotations

import pytest

from experiments.trustparadox_u.config import MonitoringConfig
from experiments.trustparadox_u.runner import _should_monitor


class TestMonitoringDifferentiation:
    """Test that monitoring differentiates monitoring_0, monitoring_1, and continuous."""

    def test_monitoring_0_never_monitors(self) -> None:
        """monitoring_0 (duration_rounds=0, continuous=False) never monitors."""
        config = MonitoringConfig(continuous=False, duration_rounds=0)

        # Should never monitor regardless of round
        assert _should_monitor(monitoring=config, post_forget_round=0) is False
        assert _should_monitor(monitoring=config, post_forget_round=1) is False
        assert _should_monitor(monitoring=config, post_forget_round=5) is False

    def test_monitoring_1_monitors_one_round(self) -> None:
        """monitoring_1 (duration_rounds=1, continuous=False) monitors for 1 round after forget."""
        config = MonitoringConfig(continuous=False, duration_rounds=1)

        # Should monitor for 1 round after forget (round 0 only)
        assert _should_monitor(monitoring=config, post_forget_round=0) is True
        assert _should_monitor(monitoring=config, post_forget_round=1) is False
        assert _should_monitor(monitoring=config, post_forget_round=2) is False

    def test_monitoring_3_monitors_three_rounds(self) -> None:
        """monitoring_3 (duration_rounds=3, continuous=False) monitors for 3 rounds after forget."""
        config = MonitoringConfig(continuous=False, duration_rounds=3)

        # Should monitor for 3 rounds after forget (rounds 0, 1, 2)
        assert _should_monitor(monitoring=config, post_forget_round=0) is True
        assert _should_monitor(monitoring=config, post_forget_round=1) is True
        assert _should_monitor(monitoring=config, post_forget_round=2) is True
        assert _should_monitor(monitoring=config, post_forget_round=3) is False
        assert _should_monitor(monitoring=config, post_forget_round=5) is False

    def test_continuous_always_monitors(self) -> None:
        """continuous (continuous=True) always monitors after forget."""
        config = MonitoringConfig(continuous=True, duration_rounds=0)

        # Should always monitor after forget
        assert _should_monitor(monitoring=config, post_forget_round=0) is True
        assert _should_monitor(monitoring=config, post_forget_round=1) is True
        assert _should_monitor(monitoring=config, post_forget_round=100) is True

    def test_continuous_takes_precedence_over_duration(self) -> None:
        """continuous=True takes precedence over duration_rounds."""
        config = MonitoringConfig(continuous=True, duration_rounds=3)

        # Should always monitor (continuous takes precedence)
        assert _should_monitor(monitoring=config, post_forget_round=0) is True
        assert _should_monitor(monitoring=config, post_forget_round=3) is True
        assert _should_monitor(monitoring=config, post_forget_round=10) is True

    def test_monitoring_negative_round_raises(self) -> None:
        """Negative post_forget_round should raise ValueError."""
        config = MonitoringConfig(continuous=False, duration_rounds=3)

        with pytest.raises(ValueError, match="post_forget_round must be non-negative"):
            _should_monitor(monitoring=config, post_forget_round=-1)

    def test_monitoring_differentiation_summary(self) -> None:
        """Verify all three monitoring modes are distinguishable."""
        m0 = MonitoringConfig(continuous=False, duration_rounds=0)
        m1 = MonitoringConfig(continuous=False, duration_rounds=1)
        mc = MonitoringConfig(continuous=True, duration_rounds=0)

        # At round 5 after forget:
        # - monitoring_0: not monitoring
        # - monitoring_1: not monitoring (only 1 round)
        # - continuous: monitoring
        assert _should_monitor(monitoring=m0, post_forget_round=5) is False
        assert _should_monitor(monitoring=m1, post_forget_round=5) is False
        assert _should_monitor(monitoring=mc, post_forget_round=5) is True

        # At round 1 after forget:
        # - monitoring_0: not monitoring
        # - monitoring_1: not monitoring (only 1 round, round 0)
        # - continuous: monitoring
        assert _should_monitor(monitoring=m0, post_forget_round=1) is False
        assert _should_monitor(monitoring=m1, post_forget_round=1) is False
        assert _should_monitor(monitoring=mc, post_forget_round=1) is True

        # At round 0 (forget round):
        # - monitoring_0: not monitoring
        # - monitoring_1: monitoring
        # - continuous: monitoring
        assert _should_monitor(monitoring=m0, post_forget_round=0) is False
        assert _should_monitor(monitoring=m1, post_forget_round=0) is True
        assert _should_monitor(monitoring=mc, post_forget_round=0) is True

"""
Tests for Reward Calculator
"""

import pytest
import numpy as np

from config.settings import RewardConfig
from brain.reward import RewardCalculator


@pytest.fixture
def reward():
    config = RewardConfig(
        profit_reward=1.0,
        loss_penalty=-2.0,
        hold_penalty=-0.1,
        max_hold_steps=120,
        drawdown_penalty=-5.0,
        sharpe_window=50,
    )
    calc = RewardCalculator(config)
    calc._peak_equity = 10000.0
    return calc


class TestRewardCalculation:
    def test_profit_reward_positive(self, reward):
        result = reward.compute(
            pnl_pips=10.0, equity=10010.0, is_trade_closed=True,
        )
        assert result > 0

    def test_loss_penalty_negative(self, reward):
        result = reward.compute(
            pnl_pips=-10.0, equity=9990.0, is_trade_closed=True,
        )
        assert result < 0

    def test_loss_penalty_asymmetric(self, reward):
        """Losses should hurt more than equivalent gains."""
        gain = reward.compute(pnl_pips=10.0, equity=10010.0, is_trade_closed=True)
        reward.reset()
        reward._peak_equity = 10000.0
        loss = reward.compute(pnl_pips=-10.0, equity=9990.0, is_trade_closed=True)
        assert abs(loss) > abs(gain)

    def test_hold_penalty(self, reward):
        """Holding a position should incur penalty."""
        reward.on_trade_open(1.10000)
        result = reward.compute(
            pnl_pips=0.0, equity=10000.0, is_trade_closed=False,
        )
        assert result < 0  # Hold penalty

    def test_drawdown_penalty(self, reward):
        """Deep drawdown should add extra penalty."""
        reward._peak_equity = 10000.0
        result = reward.compute(
            pnl_pips=-50.0, equity=9000.0, is_trade_closed=True,
        )
        assert result < -50  # Base loss + drawdown penalty


class TestRewardStats:
    def test_rolling_sharpe(self, reward):
        # Simulate profitable trades with variation
        import random
        random.seed(42)
        for i in range(20):
            pnl = 3.0 + random.random() * 5.0  # Varying positive PnL
            reward.compute(pnl_pips=pnl, equity=10000.0, is_trade_closed=True)
        sharpe = reward.get_rolling_sharpe()
        assert sharpe > 0

    def test_get_stats(self, reward):
        reward.compute(pnl_pips=5.0, equity=10000.0, is_trade_closed=True)
        stats = reward.get_stats()
        assert "rolling_sharpe" in stats
        assert "total_trades" in stats
        assert stats["total_trades"] == 1

    def test_reset(self, reward):
        reward.compute(pnl_pips=5.0, equity=10000.0, is_trade_closed=True)
        reward.reset()
        stats = reward.get_stats()
        assert stats["total_trades"] == 0

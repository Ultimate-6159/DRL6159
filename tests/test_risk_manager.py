"""
Tests for Risk Manager — Shield Layer
All deterministic (no AI), so perfectly testable.
"""

import pytest
from config.settings import RiskConfig, TradeAction
from shield.risk_manager import RiskManager


@pytest.fixture
def rm():
    config = RiskConfig(
        max_risk_per_trade=0.01,
        max_daily_loss=0.05,
        max_total_drawdown=0.15,
        max_concurrent_trades=3,
        max_lot_size=1.0,
        min_lot_size=0.01,
        atr_multiplier=1.5,
        tp_ratio=2.0,
    )
    manager = RiskManager(config)
    manager.set_initial_balance(10000.0)
    return manager


@pytest.fixture
def account():
    return {"balance": 10000.0, "equity": 10000.0, "free_margin": 10000.0}


class TestRiskManagerApproval:
    def test_buy_trade_approved(self, rm, account):
        result = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=1.10000,
            atr=0.00050,
            account_info=account,
        )
        assert result.approved is True
        assert result.lot_size >= 0.01
        assert result.stop_loss < 1.10000
        assert result.take_profit > 1.10000

    def test_sell_trade_approved(self, rm, account):
        result = rm.evaluate_trade(
            action=TradeAction.SELL,
            current_price=1.10000,
            atr=0.00050,
            account_info=account,
        )
        assert result.approved is True
        assert result.stop_loss > 1.10000
        assert result.take_profit < 1.10000

    def test_hold_rejected(self, rm, account):
        result = rm.evaluate_trade(
            action=TradeAction.HOLD,
            current_price=1.10000,
            atr=0.00050,
            account_info=account,
        )
        assert result.approved is False
        assert result.reason == "HOLD_ACTION"


class TestRiskManagerLimits:
    def test_max_concurrent_positions(self, rm, account):
        result = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=1.10000,
            atr=0.00050,
            account_info=account,
            open_positions=3,  # At limit
        )
        assert result.approved is False
        assert "MAX_POSITIONS" in result.reason

    def test_daily_loss_limit(self, rm, account):
        # Simulate losses exceeding daily limit
        rm.record_trade_result(-300.0)
        rm.record_trade_result(-300.0)

        result = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=1.10000,
            atr=0.00050,
            account_info=account,
        )
        assert result.approved is False
        assert "DAILY_LOSS" in result.reason

    def test_total_drawdown_halt(self, rm, account):
        low_equity_account = {
            "balance": 8000.0, "equity": 8000.0, "free_margin": 8000.0
        }
        result = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=1.10000,
            atr=0.00050,
            account_info=low_equity_account,
        )
        assert result.approved is False
        assert "DRAWDOWN" in result.reason


class TestPositionSizing:
    def test_lot_size_within_bounds(self, rm, account):
        result = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=1.10000,
            atr=0.00050,
            account_info=account,
        )
        assert result.lot_size >= 0.01
        assert result.lot_size <= 1.0

    def test_higher_atr_smaller_position(self, rm, account):
        """Higher ATR = wider stop = smaller lot to maintain same risk."""
        result_low = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=1.10000,
            atr=0.00030,   # Low volatility
            account_info=account,
        )
        result_high = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=1.10000,
            atr=0.00100,   # High volatility
            account_info=account,
        )
        assert result_high.lot_size <= result_low.lot_size

    def test_risk_reward_ratio(self, rm, account):
        result = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=1.10000,
            atr=0.00050,
            account_info=account,
        )
        sl_dist = abs(result.stop_loss - 1.10000)
        tp_dist = abs(result.take_profit - 1.10000)
        assert tp_dist / sl_dist == pytest.approx(2.0, rel=0.01)

    def test_daily_reset(self, rm):
        rm.record_trade_result(-100.0)
        rm.reset_daily()
        stats = rm.get_stats()
        assert stats["daily_loss"] == 0.0
        assert stats["trade_count_today"] == 0

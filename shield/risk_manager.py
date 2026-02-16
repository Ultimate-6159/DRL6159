"""
Apex Predator — Risk Manager (Layer 3: The Shield)
====================================================
100% DETERMINISTIC — No AI/ML here.
Pure mathematics for position sizing and exposure control.
"""

import logging
from typing import Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np

from config.settings import RiskConfig, TradeAction

logger = logging.getLogger("apex_predator.risk")


@dataclass
class TradeProposal:
    """Result of risk evaluation."""
    approved: bool
    lot_size: float
    stop_loss: float       # Price level
    take_profit: float     # Price level
    reason: str
    risk_amount: float     # Dollar risk
    reward_ratio: float    # Risk:Reward ratio


class RiskManager:
    """
    Deterministic risk management layer.
    Calculates position sizes, validates trade proposals,
    and enforces exposure limits.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self._daily_loss = 0.0
        self._trade_count_today = 0
        self._initial_balance = 0.0

    def set_initial_balance(self, balance: float):
        """Set the initial balance for drawdown calculations."""
        self._initial_balance = balance

    # ── Main Trade Evaluation ───────────────────

    def evaluate_trade(
        self,
        action: TradeAction,
        current_price: float,
        atr: float,
        account_info: Dict[str, Any],
        open_positions: int = 0,
        point: float = 0.00001,
        contract_size: float = 100000,
    ) -> TradeProposal:
        """
        Evaluate whether a trade should be allowed and calculate sizing.

        Args:
            action: BUY or SELL (HOLD is auto-rejected)
            current_price: Current market price
            atr: Current ATR value (in price units)
            account_info: {"balance": float, "equity": float, "free_margin": float}
            open_positions: Current number of open positions
            point: Symbol point size
            contract_size: Symbol contract size

        Returns:
            TradeProposal with approval decision and parameters
        """
        # ── HOLD = no trade ─────────────────────
        if action == TradeAction.HOLD:
            return TradeProposal(
                approved=False, lot_size=0.0, stop_loss=0.0,
                take_profit=0.0, reason="HOLD_ACTION",
                risk_amount=0.0, reward_ratio=0.0,
            )

        equity = account_info.get("equity", 0.0)
        balance = account_info.get("balance", 0.0)

        # ── Check concurrent position limit ─────
        if open_positions >= self.config.max_concurrent_trades:
            return TradeProposal(
                approved=False, lot_size=0.0, stop_loss=0.0,
                take_profit=0.0,
                reason=f"MAX_POSITIONS ({open_positions}/{self.config.max_concurrent_trades})",
                risk_amount=0.0, reward_ratio=0.0,
            )

        # ── Check daily loss limit ──────────────
        if self._initial_balance > 0:
            daily_dd = self._daily_loss / self._initial_balance
            if daily_dd >= self.config.max_daily_loss:
                return TradeProposal(
                    approved=False, lot_size=0.0, stop_loss=0.0,
                    take_profit=0.0,
                    reason=f"DAILY_LOSS_LIMIT ({daily_dd:.1%} >= {self.config.max_daily_loss:.1%})",
                    risk_amount=0.0, reward_ratio=0.0,
                )

        # ── Check total drawdown ────────────────
        if self._initial_balance > 0:
            total_dd = (self._initial_balance - equity) / self._initial_balance
            if total_dd >= self.config.max_total_drawdown:
                return TradeProposal(
                    approved=False, lot_size=0.0, stop_loss=0.0,
                    take_profit=0.0,
                    reason=f"TOTAL_DRAWDOWN_HALT ({total_dd:.1%} >= {self.config.max_total_drawdown:.1%})",
                    risk_amount=0.0, reward_ratio=0.0,
                )

        # ── ATR-based Stop Loss ─────────────────
        sl_distance = atr * self.config.atr_multiplier
        if sl_distance < point * 10:
            sl_distance = point * 10  # Minimum 10 points SL

        if action == TradeAction.BUY:
            stop_loss = current_price - sl_distance
            take_profit = current_price + (sl_distance * self.config.tp_ratio)
        else:  # SELL
            stop_loss = current_price + sl_distance
            take_profit = current_price - (sl_distance * self.config.tp_ratio)

        # ── Position Sizing (ATR-based) ─────────
        risk_amount = equity * self.config.max_risk_per_trade
        sl_pips = sl_distance / point
        pip_value = contract_size * point

        if sl_pips > 0 and pip_value > 0:
            lot_size = risk_amount / (sl_pips * pip_value)
        else:
            lot_size = self.config.min_lot_size

        # Clamp lot size
        lot_size = max(self.config.min_lot_size, min(lot_size, self.config.max_lot_size))

        # Round to nearest step (0.01)
        lot_size = round(lot_size, 2)

        reward_ratio = self.config.tp_ratio

        logger.debug(
            "Trade evaluated: %s %.2f lots | SL=%.5f TP=%.5f | "
            "Risk=$%.2f | R:R=1:%.1f",
            action.name, lot_size, stop_loss, take_profit,
            risk_amount, reward_ratio,
        )

        return TradeProposal(
            approved=True,
            lot_size=lot_size,
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            reason="APPROVED",
            risk_amount=round(risk_amount, 2),
            reward_ratio=reward_ratio,
        )

    # ── Daily Tracking ──────────────────────────

    def record_trade_result(self, pnl: float):
        """Record a completed trade's P&L for daily tracking."""
        if pnl < 0:
            self._daily_loss += abs(pnl)
        self._trade_count_today += 1

    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        logger.info("Daily risk counters reset | Previous: loss=$%.2f trades=%d",
                     self._daily_loss, self._trade_count_today)
        self._daily_loss = 0.0
        self._trade_count_today = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get current risk statistics."""
        return {
            "daily_loss": self._daily_loss,
            "trade_count_today": self._trade_count_today,
            "initial_balance": self._initial_balance,
            "max_risk_per_trade": self.config.max_risk_per_trade,
            "max_daily_loss": self.config.max_daily_loss,
        }

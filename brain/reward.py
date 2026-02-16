"""
Apex Predator — Reward Function (Layer 2: The Brain)
=====================================================
Sharpe-based reward that teaches the agent to maximise
risk-adjusted returns, not just raw profit.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from config.settings import RewardConfig

logger = logging.getLogger("apex_predator.reward")


class RewardCalculator:
    """
    Custom reward function for the DRL agent.

    Design Philosophy:
    - Profit is good, but risk-adjusted profit is BETTER
    - Losses hurt more than gains feel good (asymmetric penalty)
    - Holding too long is penalised (scalping pressure)
    - Deep drawdowns are severely punished
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self._returns_history: deque = deque(maxlen=config.sharpe_window)
        self._hold_steps = 0
        self._peak_equity = 0.0
        self._entry_price = 0.0
        self._position_active = False

    # ── Main Reward Computation ─────────────────

    def compute(
        self,
        pnl_pips: float,
        equity: float,
        is_trade_closed: bool,
        current_price: float = 0.0,
    ) -> float:
        """
        Compute the reward for the current step.

        Args:
            pnl_pips: Profit/loss in pips for this step.
            equity: Current account equity.
            is_trade_closed: Whether a trade was closed this step.
            current_price: Current market price.

        Returns:
            Scalar reward value.
        """
        reward = 0.0

        # ── 1. Trade Result Reward ──────────────
        if is_trade_closed:
            if pnl_pips > 0:
                reward += pnl_pips * self.config.profit_reward
                # Bonus for good R:R
                reward += self.config.optimal_close_bonus
            else:
                reward += pnl_pips * abs(self.config.loss_penalty)

            self._returns_history.append(pnl_pips)
            self._hold_steps = 0
            self._position_active = False

        # ── 2. Holding Penalty (scalping pressure)
        elif self._position_active:
            self._hold_steps += 1
            reward += self.config.hold_penalty

            # Extra penalty for overly long holds
            if self._hold_steps > self.config.max_hold_steps:
                reward += self.config.hold_penalty * 2

        # ── 3. Drawdown Penalty ─────────────────
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > 0.05:  # Penalise drawdown > 5%
                reward += self.config.drawdown_penalty * drawdown

        # ── 4. Sharpe Bonus (rolling) ───────────
        sharpe_bonus = self._compute_sharpe_bonus()
        reward += sharpe_bonus

        return float(reward)

    # ── Position Tracking ───────────────────────

    def on_trade_open(self, entry_price: float):
        """Notify reward calculator that a new trade was opened."""
        self._entry_price = entry_price
        self._position_active = True
        self._hold_steps = 0

    def on_trade_close(self):
        """Notify reward calculator that a trade was closed."""
        self._position_active = False
        self._hold_steps = 0

    # ── Sharpe Ratio Bonus ──────────────────────

    def _compute_sharpe_bonus(self) -> float:
        """
        Small bonus/penalty based on rolling Sharpe ratio.
        Encourages consistent performance over time.
        """
        if len(self._returns_history) < 10:
            return 0.0

        returns = np.array(self._returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-10:
            return 0.0

        sharpe = (mean_return - self.config.risk_free_rate) / std_return

        # Scale bonus: +0.1 for Sharpe > 1, -0.1 for Sharpe < -1
        return float(np.clip(sharpe * 0.1, -0.3, 0.3))

    # ── Utility ─────────────────────────────────

    def get_rolling_sharpe(self) -> float:
        """Get the current rolling Sharpe ratio."""
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(self._returns_history)
        std = np.std(returns)
        if std < 1e-10:
            return 0.0
        return float((np.mean(returns) - self.config.risk_free_rate) / std)

    def get_stats(self) -> dict:
        """Get reward statistics."""
        returns = np.array(self._returns_history) if self._returns_history else np.array([0.0])
        return {
            "rolling_sharpe": self.get_rolling_sharpe(),
            "total_trades": len(self._returns_history),
            "win_rate": float(np.mean(returns > 0)) if len(returns) > 0 else 0.0,
            "avg_return": float(np.mean(returns)),
            "peak_equity": self._peak_equity,
            "hold_steps": self._hold_steps,
        }

    def reset(self):
        """Reset reward calculator state."""
        self._returns_history.clear()
        self._hold_steps = 0
        self._peak_equity = 0.0
        self._entry_price = 0.0
        self._position_active = False

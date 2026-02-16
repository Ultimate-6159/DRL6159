"""
Apex Predator — Walk-Forward Optimization (Layer 4: The Evolution)
====================================================================
Periodic out-of-sample backtesting to validate model performance.
If model degrades, trigger full retrain with fresh parameters.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List

import numpy as np

from config.settings import EvolutionConfig

logger = logging.getLogger("apex_predator.walk_forward")


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization:
    1. Periodically test current model on recent unseen data
    2. Calculate performance metrics (Sharpe, Win Rate, PnL)
    3. Compare against threshold
    4. If degraded → signal for full retrain

    This ensures the model stays relevant to current market conditions.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self._evaluation_history: List[Dict[str, Any]] = []
        self._last_evaluation_time = time.time()
        self._consecutive_fails = 0

    # ── Evaluation ──────────────────────────────

    def evaluate(
        self,
        recent_trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on recent trades.

        Args:
            recent_trades: List of trade dicts with at least:
                {"pnl": float, "action": int, "reward": float}

        Returns:
            Performance report dict
        """
        if not recent_trades:
            return {"status": "no_data", "needs_retrain": False}

        pnls = np.array([t.get("pnl", 0.0) for t in recent_trades])
        rewards = np.array([t.get("reward", 0.0) for t in recent_trades])

        # ── Calculate Metrics ───────────────────
        sharpe = self._calculate_sharpe(pnls)
        win_rate = float(np.mean(pnls > 0)) if len(pnls) > 0 else 0.0
        total_pnl = float(np.sum(pnls))
        avg_pnl = float(np.mean(pnls))
        max_drawdown = self._calculate_max_drawdown(pnls)
        profit_factor = self._calculate_profit_factor(pnls)

        # ── Performance Check ───────────────────
        needs_retrain = False
        degradation_reasons = []

        if sharpe < self.config.performance_threshold:
            needs_retrain = True
            degradation_reasons.append(
                f"Sharpe {sharpe:.2f} < {self.config.performance_threshold}"
            )

        if win_rate < 0.35:
            needs_retrain = True
            degradation_reasons.append(f"Win rate {win_rate:.1%} < 35%")

        if total_pnl < 0 and len(pnls) > 20:
            needs_retrain = True
            degradation_reasons.append(f"Negative total PnL: {total_pnl:.2f}")

        if needs_retrain:
            self._consecutive_fails += 1
        else:
            self._consecutive_fails = 0

        report = {
            "status": "evaluated",
            "n_trades": len(recent_trades),
            "sharpe": sharpe,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "needs_retrain": needs_retrain,
            "consecutive_fails": self._consecutive_fails,
            "degradation_reasons": degradation_reasons,
            "timestamp": time.time(),
        }

        self._evaluation_history.append(report)
        self._last_evaluation_time = time.time()

        if needs_retrain:
            logger.warning(
                "⚠ Walk-Forward: Model DEGRADED — %s | "
                "Consecutive fails: %d",
                ", ".join(degradation_reasons),
                self._consecutive_fails,
            )
        else:
            logger.info(
                "Walk-Forward: Model OK | Sharpe=%.2f WR=%.1%% PnL=%.2f",
                sharpe, win_rate * 100, total_pnl,
            )

        return report

    def should_evaluate(self) -> bool:
        """Check if it's time for a performance evaluation."""
        hours_elapsed = (time.time() - self._last_evaluation_time) / 3600
        return hours_elapsed >= self.config.checkpoint_interval_hours

    # ── Metric Calculations ─────────────────────

    def _calculate_sharpe(self, pnls: np.ndarray) -> float:
        """Calculate Sharpe ratio from PnL array."""
        if len(pnls) < 2:
            return 0.0
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        if std_pnl < 1e-10:
            return 0.0
        return float(mean_pnl / std_pnl)

    def _calculate_max_drawdown(self, pnls: np.ndarray) -> float:
        """Calculate maximum drawdown from PnL sequence."""
        if len(pnls) == 0:
            return 0.0

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    def _calculate_profit_factor(self, pnls: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = abs(np.sum(pnls[pnls < 0]))
        if gross_loss < 1e-10:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    # ── History ─────────────────────────────────

    def get_trend(self, n_recent: int = 5) -> str:
        """
        Analyse recent evaluation trend.

        Returns:
            "improving", "stable", "degrading", or "unknown"
        """
        if len(self._evaluation_history) < n_recent:
            return "unknown"

        recent = self._evaluation_history[-n_recent:]
        sharpes = [r["sharpe"] for r in recent]

        if all(s > sharpes[0] for s in sharpes[1:]):
            return "improving"
        elif all(s < sharpes[0] for s in sharpes[1:]):
            return "degrading"
        else:
            return "stable"

    def get_stats(self) -> Dict[str, Any]:
        """Get walk-forward optimizer statistics."""
        return {
            "total_evaluations": len(self._evaluation_history),
            "consecutive_fails": self._consecutive_fails,
            "trend": self.get_trend(),
            "last_evaluation": (
                self._evaluation_history[-1]
                if self._evaluation_history else None
            ),
        }

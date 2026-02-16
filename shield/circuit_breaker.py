"""
Apex Predator — Circuit Breaker (Layer 3: The Shield)
======================================================
Emergency stop mechanism to prevent catastrophic loss.
Hard-coded, deterministic logic — the last line of defence.
"""

import logging
import time
from typing import Tuple, Dict, Any
from collections import deque
from datetime import datetime, timedelta

from config.settings import CircuitBreakerConfig

logger = logging.getLogger("apex_predator.circuit_breaker")


class CircuitBreaker:
    """
    Emergency halt system that monitors for danger signals:
    - Consecutive losses
    - Drawdown breach
    - Excessive daily trading
    - Abnormal spread spikes

    When tripped, the system is HALTED and must cool down.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._consecutive_losses = 0
        self._trade_count_today = 0
        self._is_halted = False
        self._halt_time: float = 0.0
        self._halt_reason = ""
        self._initial_equity = 0.0
        self._normal_spread = 0.0
        self._trade_results: deque = deque(maxlen=100)

    def set_initial_state(self, equity: float, normal_spread: float):
        """Set baseline values for monitoring."""
        self._initial_equity = equity
        self._normal_spread = normal_spread if normal_spread > 0 else 1.5
        logger.info("CircuitBreaker armed — equity=%.2f spread=%.1f",
                     equity, normal_spread)

    # ── Main Check ──────────────────────────────

    def check(
        self,
        current_equity: float,
        current_spread: float,
    ) -> Tuple[bool, str]:
        """
        Check all circuit breaker conditions.

        Args:
            current_equity: Current account equity
            current_spread: Current spread in pips

        Returns:
            (can_trade: bool, reason: str)
        """
        # ── Check cooldown ──────────────────────
        if self._is_halted:
            elapsed = time.time() - self._halt_time
            cooldown_seconds = self.config.cooldown_minutes * 60

            if elapsed < cooldown_seconds:
                remaining = int(cooldown_seconds - elapsed)
                return False, (
                    f"HALTED: {self._halt_reason} | "
                    f"Cooldown: {remaining}s remaining"
                )
            else:
                self._resume()

        # ── 1. Consecutive losses ───────────────
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            return self._trip(
                f"CONSECUTIVE_LOSSES: {self._consecutive_losses} >= "
                f"{self.config.max_consecutive_losses}"
            )

        # ── 2. Drawdown check ───────────────────
        if self._initial_equity > 0:
            drawdown = (self._initial_equity - current_equity) / self._initial_equity
            if drawdown >= self.config.drawdown_halt_pct:
                return self._trip(
                    f"DRAWDOWN: {drawdown:.1%} >= {self.config.drawdown_halt_pct:.1%}"
                )

        # ── 3. Daily trade limit ────────────────
        if self._trade_count_today >= self.config.max_daily_trades:
            return self._trip(
                f"MAX_DAILY_TRADES: {self._trade_count_today} >= "
                f"{self.config.max_daily_trades}"
            )

        # ── 4. Spread spike ─────────────────────
        if self._normal_spread > 0:
            spread_ratio = current_spread / self._normal_spread
            if spread_ratio >= self.config.spread_spike_multiplier:
                return False, (
                    f"SPREAD_SPIKE: {current_spread:.1f} / {self._normal_spread:.1f} "
                    f"= {spread_ratio:.1f}x (limit: {self.config.spread_spike_multiplier}x) "
                    f"— pausing but NOT halting"
                )

        return True, "OK"

    # ── Trade Recording ─────────────────────────

    def record_trade(self, pnl: float):
        """Record a completed trade result."""
        self._trade_count_today += 1
        self._trade_results.append(pnl)

        if pnl < 0:
            self._consecutive_losses += 1
            logger.warning(
                "CB: Loss recorded | Consecutive: %d/%d",
                self._consecutive_losses, self.config.max_consecutive_losses,
            )
        else:
            self._consecutive_losses = 0

    def reset_daily(self):
        """Reset daily counters."""
        self._trade_count_today = 0
        logger.info("CircuitBreaker daily counters reset")

    # ── Halt / Resume ───────────────────────────

    def _trip(self, reason: str) -> Tuple[bool, str]:
        """Activate the circuit breaker."""
        self._is_halted = True
        self._halt_time = time.time()
        self._halt_reason = reason
        logger.critical(
            "⚡ CIRCUIT BREAKER TRIPPED: %s — System HALTED for %d minutes",
            reason, self.config.cooldown_minutes,
        )
        return False, f"HALTED: {reason}"

    def _resume(self):
        """Resume trading after cooldown."""
        logger.info(
            "CircuitBreaker cooldown complete — resuming trading | "
            "Previous halt: %s",
            self._halt_reason,
        )
        self._is_halted = False
        self._halt_reason = ""
        self._consecutive_losses = 0

    def force_resume(self):
        """Manual override to resume trading (use with caution)."""
        logger.warning("CircuitBreaker FORCE RESUMED by operator")
        self._resume()

    # ── Status ──────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "consecutive_losses": self._consecutive_losses,
            "trade_count_today": self._trade_count_today,
            "total_trades_recorded": len(self._trade_results),
        }

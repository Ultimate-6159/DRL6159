"""
Apex Predator — Backtesting Gymnasium Environment
====================================================
High-frequency scalping environment with:
- Trailing stop logic (lock profits)
- Expectancy-based reward function
- Smart inactivity penalty (miss-opportunity based)
- Dynamic position sizing for $500–$5M portfolios
"""

import logging
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.settings import (
    RewardConfig,
    MarketRegime,
    TradeAction,
)

logger = logging.getLogger("apex_predator.backtest_env")


class BacktestEnv(gym.Env):
    """
    Gymnasium environment for DRL scalping training.

    Observation Space (raw features — NOT LSTM):
        - Last `lookback` bars of normalized features, flattened
        - Regime one-hot (4-dim)
        - Account state (7-dim): balance, pnl, spread, has_position,
          hold_time, win_rate_recent, drawdown
        Total: lookback * feature_dim + 4 + 7

    Action Space:
        Discrete(3): BUY=0, SELL=1, HOLD=2
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: np.ndarray,
        features: np.ndarray,
        regimes: np.ndarray,
        spreads: np.ndarray,
        atrs: np.ndarray,
        lookback: int = 10,
        reward_config: Optional[RewardConfig] = None,
        initial_balance: float = 10000.0,
        lot_size: float = 0.01,
        contract_size: float = 100.0,
        point_value: float = 0.01,
        max_hold_bars: int = 30,
        sl_atr_mult: float = 0.5,
        tp_atr_mult: float = 1.25,
    ):
        """
        Args:
            bars: (N, 4) array of [open, high, low, close]
            features: (N, feature_dim) normalized feature matrix
            regimes: (N,) int array of regime codes (0-3)
            spreads: (N,) float array of spreads per bar
            atrs: (N,) float array of ATR per bar
            lookback: Number of past bars for observation (default: 10)
            reward_config: Reward parameters
            initial_balance: Starting account balance
            lot_size: Fixed lot size for training
            contract_size: Contract size (Gold=100, Forex=100000)
            point_value: Point value of the symbol
            max_hold_bars: Max bars to hold a position
            sl_atr_mult: ATR multiplier for stop-loss
            tp_atr_mult: ATR multiplier for take-profit
        """
        super().__init__()

        self.bars = bars
        self.features = features
        self.regimes = regimes
        self.spreads = spreads
        self.atrs = atrs
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.contract_size = contract_size
        self.point_value = point_value
        self.max_hold_bars = max_hold_bars
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

        # Reward
        self.reward_config = reward_config or RewardConfig()

        # Observation dimensions
        self.feature_dim = features.shape[1]
        n_regimes = 4
        n_account = 7  # balance, pnl, spread, has_position, hold_time, win_rate, drawdown
        obs_dim = lookback * self.feature_dim + n_regimes + n_account

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode range
        self.start_idx = lookback
        self.end_idx = len(bars) - 1
        self.max_steps = self.end_idx - self.start_idx

        logger.info("BacktestEnv: obs_dim=%d (features=%d x %d + regime=4 + account=7)",
                     obs_dim, lookback, self.feature_dim)
        logger.info("BacktestEnv: %d bars, %d tradeable steps",
                     len(bars), self.max_steps)

        # State
        self._reset_state()

    def _reset_state(self):
        self.current_step = self.start_idx
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position = None
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.hold_bars = 0
        self.bars_without_trade = 0
        self._returns_buffer = []
        self._recent_wins = 0
        self._recent_trades = 0
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        # Action diversity tracking
        self._action_counts = {0: 0, 1: 0, 2: 0}  # BUY, SELL, HOLD

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Random start for diversity
        if self.np_random is not None and self.max_steps > 500:
            max_start = self.end_idx - 500
            if max_start > self.start_idx:
                self.current_step = self.np_random.integers(
                    self.start_idx, max_start
                )

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple:
        reward = 0.0
        info = {"action": TradeAction(action).name}

        bar = self.bars[self.current_step]
        close = bar[3]
        spread = self.spreads[self.current_step]
        atr = max(self.atrs[self.current_step], 0.01)

        # ── Track action for diversity penalty ─────────
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        total_actions = sum(self._action_counts.values())

        # Penalize repetitive actions (same action > 70% of time)
        if total_actions > 100:
            action_ratio = self._action_counts[action] / total_actions
            if action_ratio > 0.70:
                # Heavy penalty for always picking same action
                diversity_penalty = -0.1 * (action_ratio - 0.70)
                reward += diversity_penalty

        # ── Manage existing position ─────────
        if self.position is not None:
            self.hold_bars += 1
            entry = self.position["entry"]
            direction = self.position["direction"]

            # Unrealized PnL
            price_diff = (close - entry) * direction
            unrealized_pnl = price_diff * self.lot_size * self.contract_size

            # ── Trailing Stop Logic ──────────────
            self._update_trailing_stop(close, atr)

            # Check SL/TP
            sl = self.position["sl"]
            tp = self.position["tp"]
            high, low = bar[1], bar[2]

            hit_sl = (direction == 1 and low <= sl) or \
                     (direction == -1 and high >= sl)
            hit_tp = (direction == 1 and high >= tp) or \
                     (direction == -1 and low <= tp)
            force_close = self.hold_bars >= self.max_hold_bars

            if hit_tp:
                tp_diff = (tp - entry) * direction
                realized_pnl = tp_diff * self.lot_size * self.contract_size
                reward = self._compute_reward(realized_pnl, "TP")
                self._close_position(realized_pnl)
                info["event"] = "TP_HIT"

            elif hit_sl:
                sl_diff = (sl - entry) * direction
                realized_pnl = sl_diff * self.lot_size * self.contract_size
                reward = self._compute_reward(realized_pnl, "SL")
                self._close_position(realized_pnl)
                info["event"] = "SL_HIT"

            elif force_close:
                reward = self._compute_reward(unrealized_pnl, "FORCE")
                self._close_position(unrealized_pnl)
                info["event"] = "FORCE_CLOSE"

            else:
                # Still holding — very light time-based cost
                reward = -0.003 * (self.hold_bars / self.max_hold_bars)

            # After close, agent can open new position
            if self.position is None and action != TradeAction.HOLD.value:
                reward += self._open_position(action, close, spread, atr)

        else:
            # ── No position ────────────────────
            if action == TradeAction.HOLD.value:
                # Minimal inactivity penalty — HOLD is a valid strategy
                self.bars_without_trade += 1
                if self.bars_without_trade > 100:
                    reward = -0.001  # Only penalize extreme inactivity
                else:
                    reward = 0.0  # HOLD is free → agent learns to be selective
            else:
                self.bars_without_trade = 0
                reward = self._open_position(action, close, spread, atr)

        # ── Advance ────────────────────────
        self.current_step += 1
        terminated = self.balance <= self.initial_balance * 0.5
        truncated = self.current_step >= self.end_idx

        # Update equity
        self.equity = self.balance
        if self.position is not None:
            idx = min(self.current_step, self.end_idx - 1)
            curr_close = self.bars[idx][3]
            pos_pnl = (curr_close - self.position["entry"]) * \
                       self.position["direction"] * self.lot_size * self.contract_size
            self.equity = self.balance + pos_pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        info.update({
            "balance": self.balance,
            "equity": self.equity,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "total_pnl": self.total_pnl,
        })

        if terminated or truncated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def _open_position(self, action, close, spread, atr):
        direction = 1 if action == TradeAction.BUY.value else -1

        # ── STRICT Trend confirmation during training ──
        # Heavy penalty for counter-trend entries
        trend_penalty = 0.0
        idx = self.current_step
        if idx >= 50:
            closes = self.bars[idx-50:idx+1, 3]  # Last 50 closes
            ema_fast = self._simple_ema(closes, 20)
            ema_slow = self._simple_ema(closes, 50)
            trend_up = ema_fast > ema_slow

            # Also check price momentum
            momentum = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0

            if direction == 1:  # BUY
                if not trend_up or momentum < -0.001:
                    trend_penalty = -0.8  # Heavy penalty for buying in downtrend
            else:  # SELL
                if trend_up or momentum > 0.001:
                    trend_penalty = -0.8  # Heavy penalty for selling in uptrend

        spread_cost = spread * self.point_value
        entry = close + (spread_cost / 2) * direction

        sl_dist = atr * self.sl_atr_mult
        tp_dist = atr * self.tp_atr_mult

        if direction == 1:
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            sl, tp = entry + sl_dist, entry - tp_dist

        self.position = {
            "entry": entry,
            "direction": direction,
            "sl": sl,
            "tp": tp,
            "open_step": self.current_step,
            "best_price": entry,         # For trailing stop
            "trailing_active": False,     # Trailing activated flag
        }
        self.hold_bars = 0
        return -0.005 + trend_penalty  # Friction + trend penalty

    @staticmethod
    def _simple_ema(data, period):
        """Fast EMA computation for training environment."""
        alpha = 2.0 / (period + 1)
        ema = data[0]
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * ema
        return ema

    def _update_trailing_stop(self, close, atr):
        """
        Aggressive trailing stop for high win rate:
        1. When profit >= 0.3 * ATR → move SL to breakeven
        2. When profit >= 0.5 * ATR → trail SL tight behind best price
        This locks in more profits and increases win rate.
        """
        if self.position is None:
            return

        direction = self.position["direction"]
        entry = self.position["entry"]
        price_diff = (close - entry) * direction

        # Update best price
        if direction == 1:
            self.position["best_price"] = max(self.position["best_price"], close)
        else:
            self.position["best_price"] = min(self.position["best_price"], close)

        # Phase 1: Move to breakeven early
        if price_diff >= atr * 0.3 and not self.position["trailing_active"]:
            if direction == 1:
                new_sl = max(self.position["sl"], entry + atr * 0.05)
            else:
                new_sl = min(self.position["sl"], entry - atr * 0.05)
            self.position["sl"] = new_sl

        # Phase 2: Tight trail behind best price
        if price_diff >= atr * 0.5:
            self.position["trailing_active"] = True
            trail_dist = atr * 0.3  # Tighter trail → lock more profit
            if direction == 1:
                trail_sl = self.position["best_price"] - trail_dist
                self.position["sl"] = max(self.position["sl"], trail_sl)
            else:
                trail_sl = self.position["best_price"] + trail_dist
                self.position["sl"] = min(self.position["sl"], trail_sl)

    def _close_position(self, pnl):
        self.balance += pnl
        self.total_pnl += pnl
        self.total_trades += 1
        self._recent_trades += 1

        if pnl > 0:
            self.wins += 1
            self._recent_wins += 1
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self.losses += 1
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        self._returns_buffer.append(pnl)
        self.position = None
        self.hold_bars = 0

    def _compute_reward(self, pnl, event_type=""):
        """
        High-winrate reward function for scalping.
        Heavily rewards TP hits and consistency.
        Harshly punishes losses to teach selectivity.
        """
        pnl_norm = pnl / self.initial_balance * 100

        if pnl >= 0:
            # Win: strong base bonus + proportional reward
            reward = 0.5 + pnl_norm * 1.5

            if event_type == "TP":
                reward += 1.5  # Big TP bonus → learn to let winners run

            # Win streak escalation (stronger)
            reward += min(self._consecutive_wins, 7) * 0.15

            # Consistency bonus: reward maintaining >70% WR
            if self._recent_trades >= 5:
                recent_wr = self._recent_wins / self._recent_trades
                if recent_wr >= 0.7:
                    reward += 0.5  # Strong bonus for >70% win rate
                elif recent_wr >= 0.5:
                    reward += 0.1
        else:
            # Loss: harsh penalty to teach selectivity
            reward = -0.5 + pnl_norm * 1.0

            if event_type == "SL":
                reward -= 0.3  # SL hit is still bad
            elif event_type == "FORCE":
                reward -= 1.0  # Force close = very bad

            # Consecutive loss penalty (stronger)
            if self._consecutive_losses >= 2:
                reward -= 0.3 * (self._consecutive_losses - 1)

        # Drawdown penalty — starts at 3% DD
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1.0)
        if dd > 0.03:
            reward -= dd * 1.5

        return float(np.clip(reward, -3.0, 5.0))

    def _get_observation(self):
        """
        Build observation from RAW features.
        Enhanced with win rate and drawdown signals.
        """
        idx = self.current_step

        # ── Raw features: last `lookback` bars, flattened ───
        start = max(0, idx - self.lookback)
        feat_slice = self.features[start:idx]
        if len(feat_slice) < self.lookback:
            pad = np.zeros(
                (self.lookback - len(feat_slice), self.feature_dim),
                dtype=np.float32,
            )
            feat_slice = np.vstack([pad, feat_slice])
        features_flat = feat_slice.flatten().astype(np.float32)

        # ── Regime one-hot ──────────────────
        regime_idx = int(self.regimes[idx]) if idx < len(self.regimes) else 3
        regime_oh = np.zeros(4, dtype=np.float32)
        regime_oh[min(regime_idx, 3)] = 1.0

        # ── Account state (7-dim) ───────────
        balance_norm = (self.equity / self.initial_balance) - 1.0

        pnl_norm = 0.0
        has_position = 0.0
        hold_time_norm = 0.0
        if self.position is not None:
            close = self.bars[idx][3]
            pos_pnl = (close - self.position["entry"]) * \
                       self.position["direction"] * self.lot_size * self.contract_size
            pnl_norm = pos_pnl / max(abs(self.equity), 1.0)
            has_position = 1.0
            hold_time_norm = self.hold_bars / self.max_hold_bars

        spread_norm = self.spreads[idx] / 100.0 if idx < len(self.spreads) else 0.0

        # Recent win rate (new signal)
        recent_wr = self._recent_wins / max(self._recent_trades, 1)

        # Current drawdown (new signal)
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1.0)

        account = np.array([
            balance_norm, pnl_norm, spread_norm,
            has_position, hold_time_norm,
            recent_wr,     # New: agent knows its own performance
            dd,            # New: agent knows drawdown state
        ], dtype=np.float32)

        return np.concatenate([features_flat, regime_oh, account])

    def get_stats(self):
        win_rate = self.wins / max(self.total_trades, 1)
        sharpe = 0.0
        if len(self._returns_buffer) > 1:
            arr = np.array(self._returns_buffer)
            if arr.std() > 0:
                sharpe = arr.mean() / arr.std() * np.sqrt(252)
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "final_balance": self.balance,
            "max_drawdown": (self.peak_equity - self.equity) / max(self.peak_equity, 1),
            "sharpe_ratio": sharpe,
        }

    # ── Portfolio Scaling Helper ──────────────────
    @staticmethod
    def calculate_lot_size(balance, risk_pct=0.01, sl_pips=10.0, pip_value=1.0):
        """
        Calculate lot size based on portfolio size and risk %.
        Scales from $500 micro accounts to $5M institutional.

        Args:
            balance: Account balance in USD
            risk_pct: Max risk per trade (0.01 = 1%)
            sl_pips: Stop-loss distance in pips
            pip_value: Dollar value per pip per lot

        Returns:
            Lot size (capped: 0.01 – 100.0)
        """
        # Adaptive risk based on account size
        if balance < 1_000:
            risk_pct = min(risk_pct, 0.005)   # 0.5% for micro accounts
        elif balance < 10_000:
            risk_pct = min(risk_pct, 0.01)    # 1% for small
        elif balance < 100_000:
            risk_pct = min(risk_pct, 0.01)    # 1% for medium
        elif balance < 1_000_000:
            risk_pct = min(risk_pct, 0.005)   # 0.5% for large
        else:
            risk_pct = min(risk_pct, 0.003)   # 0.3% for institutional

        risk_amount = balance * risk_pct
        lot = risk_amount / max(sl_pips * pip_value, 0.01)
        return max(0.01, min(lot, 100.0))

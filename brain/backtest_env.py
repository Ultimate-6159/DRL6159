"""
Apex Predator — Backtesting Gymnasium Environment
====================================================
Replays historical OHLC data and simulates realistic trading
with spread, slippage, and PnL calculation for DRL training.

Uses RAW FEATURES as observations (not LSTM embeddings) so PPO
can learn directly from market data without a pre-trained encoder.
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
    Gymnasium environment that replays historical bars
    for DRL training with realistic trading simulation.

    Observation Space (raw features — NOT LSTM):
        - Last `lookback` bars of normalized features, flattened
        - Regime one-hot (4-dim)
        - Account state (5-dim)
        Total: lookback * feature_dim + 4 + 5

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
        max_hold_bars: int = 120,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
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
        n_account = 5  # balance, pnl, spread, has_position, hold_time
        obs_dim = lookback * self.feature_dim + n_regimes + n_account

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode range
        self.start_idx = lookback
        self.end_idx = len(bars) - 1
        self.max_steps = self.end_idx - self.start_idx

        logger.info("BacktestEnv: obs_dim=%d (features=%d x %d + regime=4 + account=5)",
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

        # ── Manage existing position ─────────
        if self.position is not None:
            self.hold_bars += 1
            entry = self.position["entry"]
            direction = self.position["direction"]

            # Unrealized PnL
            price_diff = (close - entry) * direction
            unrealized_pnl = price_diff * self.lot_size * self.contract_size

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
                # Still holding — small time-based cost
                reward = -0.005 * (self.hold_bars / self.max_hold_bars)

            # After close, agent can open new position
            if self.position is None and action != TradeAction.HOLD.value:
                reward += self._open_position(action, close, spread, atr)

        else:
            # ── No position ────────────────────
            if action == TradeAction.HOLD.value:
                # Growing inactivity penalty
                self.bars_without_trade += 1
                penalty = min(self.bars_without_trade / 100.0, 1.0) * 0.02
                reward = -penalty
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
        }
        self.hold_bars = 0
        return -0.01  # Small friction

    def _close_position(self, pnl):
        self.balance += pnl
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self._returns_buffer.append(pnl)
        self.position = None
        self.hold_bars = 0

    def _compute_reward(self, pnl, event_type=""):
        """
        Reward shaping:
        - Win → strong positive (+1.0 bonus)
        - Loss → moderate negative
        - TP hit → extra bonus
        """
        pnl_norm = pnl / self.initial_balance * 100

        if pnl >= 0:
            reward = pnl_norm * 1.5  # Amplify gains
            reward += 1.0  # Win bonus
            if event_type == "TP":
                reward += 0.5  # Extra for TP
        else:
            reward = pnl_norm * 0.8  # Moderate losses
            reward -= 0.2  # Loss penalty

        # Drawdown penalty
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1.0)
        if dd > 0.05:
            reward -= dd * 2.0

        return float(np.clip(reward, -5.0, 10.0))

    def _get_observation(self):
        """
        Build observation from RAW features.
        Much more informative than untrained LSTM embeddings.
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

        # ── Account state ───────────────────
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

        account = np.array([
            balance_norm, pnl_norm, spread_norm,
            has_position, hold_time_norm,
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

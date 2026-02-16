"""
Apex Predator — Backtesting Gymnasium Environment
====================================================
Replays historical OHLC data and simulates realistic trading
with spread, slippage, and PnL calculation for DRL training.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.settings import (
    RewardConfig,
    PerceptionConfig,
    FeatureConfig,
    RegimeConfig,
    MarketRegime,
    TradeAction,
)

logger = logging.getLogger("apex_predator.backtest_env")


class BacktestEnv(gym.Env):
    """
    Gymnasium environment that replays historical bars
    for DRL training with realistic trading simulation.

    Simulates:
    - Spread cost on entry
    - Slippage (configurable)
    - Dynamic position management (one position at a time)
    - ATR-based SL/TP calculation
    - Realistic PnL per bar

    Observation Space:
        - LSTM embedding (64-dim) from PerceptionModule
        - Regime one-hot (4-dim)
        - Account state (3-dim: balance_norm, pnl_norm, spread_norm)
        Total: embedding_dim + 4 + 3 = 71

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
        perception_module=None,
        perception_dim: int = 64,
        sequence_length: int = 60,
        reward_config: Optional[RewardConfig] = None,
        initial_balance: float = 10000.0,
        lot_size: float = 0.01,
        contract_size: float = 100.0,        # Gold = 100 oz
        point_value: float = 0.01,            # Gold point
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
            perception_module: PerceptionModule for encoding features
            perception_dim: Dimension of perception embedding
            sequence_length: Number of bars for LSTM input
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

        self.bars = bars              # (N, 4): OHLC
        self.features = features      # (N, feat_dim)
        self.regimes = regimes        # (N,)
        self.spreads = spreads        # (N,)
        self.atrs = atrs              # (N,)
        self.perception = perception_module
        self.perception_dim = perception_dim
        self.seq_len = sequence_length
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.contract_size = contract_size
        self.point_value = point_value
        self.max_hold_bars = max_hold_bars
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

        # Reward
        self.reward_config = reward_config or RewardConfig()

        # Action / Obs spaces
        n_regimes = 4
        n_account = 3
        obs_dim = perception_dim + n_regimes + n_account

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode range
        self.start_idx = sequence_length
        self.end_idx = len(bars) - 1
        self.max_steps = self.end_idx - self.start_idx

        # State
        self._reset_state()

    def _reset_state(self):
        self.current_step = self.start_idx
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position = None  # None or dict
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.hold_bars = 0
        self.bars_without_trade = 0  # Track inactivity
        self._returns_buffer = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Random start for diverse training
        if self.np_random is not None and self.max_steps > 200:
            max_start = self.end_idx - 200
            if max_start > self.start_idx:
                self.current_step = self.np_random.integers(
                    self.start_idx, max_start
                )

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple:
        """
        Execute one trading step.

        Returns: (obs, reward, terminated, truncated, info)
        """
        reward = 0.0
        info = {"action": TradeAction(action).name}

        bar = self.bars[self.current_step]
        close = bar[3]
        spread = self.spreads[self.current_step]
        atr = max(self.atrs[self.current_step], 0.01)  # Safety floor

        # ── Manage existing position ─────────
        if self.position is not None:
            self.hold_bars += 1
            entry = self.position["entry"]
            direction = self.position["direction"]  # 1=BUY, -1=SELL

            # Current unrealized PnL (in USD)
            price_diff = (close - entry) * direction
            unrealized_pnl = price_diff * self.lot_size * self.contract_size

            # Check SL/TP hit
            sl = self.position["sl"]
            tp = self.position["tp"]
            high = bar[1]
            low = bar[2]

            hit_sl = (direction == 1 and low <= sl) or \
                     (direction == -1 and high >= sl)
            hit_tp = (direction == 1 and high >= tp) or \
                     (direction == -1 and low <= tp)

            # Force close if holding too long
            force_close = self.hold_bars >= self.max_hold_bars

            if hit_sl:
                # Close at SL price
                sl_diff = (sl - entry) * direction
                realized_pnl = sl_diff * self.lot_size * self.contract_size
                reward = self._compute_reward(realized_pnl, is_closed=True)
                self._close_position(realized_pnl)
                info["event"] = "SL_HIT"

            elif hit_tp:
                # Close at TP price
                tp_diff = (tp - entry) * direction
                realized_pnl = tp_diff * self.lot_size * self.contract_size
                reward = self._compute_reward(realized_pnl, is_closed=True)
                self._close_position(realized_pnl)
                info["event"] = "TP_HIT"

            elif force_close:
                reward = self._compute_reward(unrealized_pnl, is_closed=True)
                self._close_position(unrealized_pnl)
                info["event"] = "FORCE_CLOSE"

            else:
                # Still holding — small penalty
                reward = self.reward_config.hold_penalty * 0.1

            # If position was closed and agent wants to open new one
            if self.position is None and action != TradeAction.HOLD.value:
                reward += self._open_position(action, close, spread, atr)

        else:
            # ── No position — evaluate action ──
            if action == TradeAction.HOLD.value:
                # Inactivity penalty — grows over time to force trading
                self.bars_without_trade += 1
                inactivity_scale = min(self.bars_without_trade / 50.0, 1.0)
                reward = -0.05 * inactivity_scale  # Grows from -0.001 to -0.05
            else:
                self.bars_without_trade = 0  # Reset on trade
                reward += self._open_position(action, close, spread, atr)

        # ── Advance ─────────────────────────
        self.current_step += 1
        terminated = self.balance <= self.initial_balance * 0.5  # -50% stop
        truncated = self.current_step >= self.end_idx

        # Update equity
        self.equity = self.balance
        if self.position is not None:
            curr_close = self.bars[min(self.current_step, self.end_idx - 1)][3]
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

        obs = self._get_observation() if not (terminated or truncated) else \
              np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def _open_position(self, action: int, close: float, spread: float, atr: float) -> float:
        """Open a new position. Returns small negative reward for spread cost."""
        direction = 1 if action == TradeAction.BUY.value else -1

        # Entry price with spread cost (realistic but small)
        spread_cost = spread * self.point_value
        if direction == 1:
            entry = close + spread_cost / 2
        else:
            entry = close - spread_cost / 2

        # SL/TP
        sl_dist = atr * self.sl_atr_mult
        tp_dist = atr * self.tp_atr_mult

        if direction == 1:
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        self.position = {
            "entry": entry,
            "direction": direction,
            "sl": sl,
            "tp": tp,
            "open_step": self.current_step,
        }
        self.hold_bars = 0

        # Minimal spread penalty — just a small friction, not a wall
        return -0.01

    def _close_position(self, pnl: float):
        """Close position and update stats."""
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

    def _compute_reward(self, pnl: float, is_closed: bool) -> float:
        """
        Compute reward with asymmetric weighting.
        - Wins get BONUS reward to incentivize profitable trading
        - Losses are penalized but not so much that model stops trading
        """
        # Normalize PnL relative to initial balance
        pnl_norm = pnl / self.initial_balance * 100  # Scale up

        if pnl >= 0:
            # Boost wins — model needs strong positive signal
            reward = pnl_norm * self.reward_config.profit_reward
            reward += 1.0  # Bonus for any winning trade
        else:
            # Penalize losses but moderately
            reward = pnl_norm * abs(self.reward_config.loss_penalty) * 0.5
            reward -= 0.3  # Small fixed loss penalty

        # Drawdown penalty
        drawdown = (self.peak_equity - self.equity) / max(self.peak_equity, 1.0)
        if drawdown > 0.05:
            reward += self.reward_config.drawdown_penalty * drawdown * 0.5

        return float(np.clip(reward, -5.0, 10.0))

    def _get_observation(self) -> np.ndarray:
        """Build observation at current step."""
        idx = self.current_step

        # ── Perception embedding ──────────
        if self.perception is not None:
            start = max(0, idx - self.seq_len)
            feat_slice = self.features[start:idx]
            if len(feat_slice) < self.seq_len:
                pad = np.zeros(
                    (self.seq_len - len(feat_slice), feat_slice.shape[1]),
                    dtype=np.float32,
                )
                feat_slice = np.vstack([pad, feat_slice])
            embedding = self.perception.encode(feat_slice)
        else:
            embedding = np.zeros(self.perception_dim, dtype=np.float32)

        # ── Regime one-hot ────────────────
        regime_idx = int(self.regimes[idx]) if idx < len(self.regimes) else 3
        regime_oh = np.zeros(4, dtype=np.float32)
        regime_oh[min(regime_idx, 3)] = 1.0

        # ── Account state ─────────────────
        balance_norm = (self.equity / self.initial_balance) - 1.0

        pnl_norm = 0.0
        if self.position is not None:
            close = self.bars[idx][3]
            pos_pnl = (close - self.position["entry"]) * \
                       self.position["direction"] * self.lot_size * self.contract_size
            pnl_norm = pos_pnl / max(self.equity, 1.0)

        spread_norm = self.spreads[idx] / 10.0 if idx < len(self.spreads) else 0.0

        account = np.array([balance_norm, pnl_norm, spread_norm], dtype=np.float32)

        return np.concatenate([
            embedding.astype(np.float32),
            regime_oh,
            account,
        ])

    def get_stats(self) -> dict:
        """Return episode statistics."""
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

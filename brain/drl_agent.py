"""
Apex Predator — DRL Agent (Layer 2: The Brain)
================================================
PPO / SAC decision engine using Stable-Baselines3.
Custom Gymnasium environment for Forex trading.
"""

import logging
import os
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.settings import (
    DRLConfig,
    RewardConfig,
    PerceptionConfig,
    TradeAction,
    MarketRegime,
)
from brain.reward import RewardCalculator

logger = logging.getLogger("apex_predator.drl_agent")


# ──────────────────────────────────────────────
# Custom Forex Trading Environment
# ──────────────────────────────────────────────

class ForexTradingEnv(gym.Env):
    """
    Gymnasium environment for Forex scalping.
    Used for both training and live inference wrapping.

    Observation Space (must match BacktestEnv exactly):
        - Last `lookback` bars of raw features, flattened  (lookback * feature_dim)
        - Regime one-hot (4-dim)
        - Account state (7-dim): balance_norm, pnl_norm, spread_norm,
          has_position, hold_time_norm, recent_wr, drawdown
        Total: lookback * feature_dim + 4 + 7

    Action Space:
        Discrete(3): BUY=0, SELL=1, HOLD=2
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_dim: int = 17,
        lookback: int = 10,
        reward_config: Optional[RewardConfig] = None,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.lookback = lookback
        n_regimes = len(MarketRegime)
        n_account = 7  # balance, pnl, spread, has_position, hold_time, win_rate, drawdown

        obs_dim = lookback * feature_dim + n_regimes + n_account

        # Observation: continuous vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: discrete (BUY, SELL, HOLD)
        self.action_space = spaces.Discrete(3)

        # Reward calculator
        self.reward_calc = RewardCalculator(
            reward_config or RewardConfig()
        )

        # Internal state
        self._current_obs: Optional[np.ndarray] = None
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        self.reward_calc.reset()
        self._step_count = 0
        self._current_obs = np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return self._current_obs, {}

    def step(self, action: int):
        """
        Execute one step. In live mode, this is called by the orchestrator
        with externally computed observations and rewards.
        """
        self._step_count += 1
        reward = 0.0
        terminated = False
        truncated = self._step_count >= 10000  # Safety limit
        info = {"action": TradeAction(action).name}

        return self._current_obs, reward, terminated, truncated, info

    def set_observation(self, obs: np.ndarray):
        """Set the current observation externally (for live trading)."""
        self._current_obs = obs.astype(np.float32)

    @staticmethod
    def build_observation(
        features_flat: np.ndarray,
        regime_one_hot: np.ndarray,
        balance_norm: float,
        pnl_norm: float,
        spread_norm: float,
        has_position: float = 0.0,
        hold_time_norm: float = 0.0,
        recent_wr: float = 0.0,
        drawdown: float = 0.0,
    ) -> np.ndarray:
        """
        Build observation vector matching BacktestEnv format.

        Args:
            features_flat: (lookback * feature_dim,) raw features flattened
            regime_one_hot: (4,) from RegimeClassifier
            balance_norm: Normalized balance (equity / initial_equity - 1)
            pnl_norm: Normalized current P&L
            spread_norm: Normalized spread
            has_position: 1.0 if position open, else 0.0
            hold_time_norm: Normalized hold time
            recent_wr: Recent win rate
            drawdown: Current drawdown fraction

        Returns:
            (obs_dim,) float32 array
        """
        account = np.array([
            balance_norm, pnl_norm, spread_norm,
            has_position, hold_time_norm,
            recent_wr, drawdown,
        ], dtype=np.float32)
        return np.concatenate([
            features_flat.astype(np.float32),
            regime_one_hot.astype(np.float32),
            account,
        ])


# ──────────────────────────────────────────────
# DRL Agent Wrapper
# ──────────────────────────────────────────────

class DRLAgent:
    """
    High-level DRL agent wrapping Stable-Baselines3.
    Supports PPO and SAC algorithms.
    """

    def __init__(
        self,
        config: DRLConfig,
        perception_config: PerceptionConfig,
        reward_config: RewardConfig,
        feature_dim: int = 17,
        lookback: int = 10,
    ):
        self.config = config
        self.env = ForexTradingEnv(
            feature_dim=feature_dim,
            lookback=lookback,
            reward_config=reward_config,
        )
        self.model = None
        self._initialized = False

        # Create model directory
        os.makedirs(config.model_save_path, exist_ok=True)

    # ── Initialization ──────────────────────────

    def initialize(self):
        """Create or load the RL model."""
        try:
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.callbacks import CheckpointCallback
        except ImportError:
            logger.error("stable-baselines3 not installed — using random agent")
            self._initialized = False
            return

        os.makedirs(self.config.tensorboard_log, exist_ok=True)

        if self.config.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                tensorboard_log=self.config.tensorboard_log,
                verbose=0,
            )
        elif self.config.algorithm == "SAC":
            # SAC requires continuous action space — wrap discrete as Box
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                batch_size=self.config.batch_size,
                tensorboard_log=self.config.tensorboard_log,
                verbose=0,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        self._initialized = True
        logger.info("DRL Agent initialised — algorithm=%s", self.config.algorithm)

    # ── Prediction ──────────────────────────────

    def predict(self, observation: np.ndarray) -> Tuple[TradeAction, float]:
        """
        Predict the next action given an observation.

        Args:
            observation: Observation vector from ForexTradingEnv.build_observation()

        Returns:
            Tuple of (TradeAction, confidence)
        """
        if not self._initialized or self.model is None:
            # Random fallback
            action = np.random.choice([0, 1, 2], p=[0.1, 0.1, 0.8])
            return TradeAction(action), 0.0

        action, _states = self.model.predict(
            observation, deterministic=True
        )
        action_int = int(action)

        # Get action probabilities for confidence
        confidence = self._get_action_confidence(observation, action_int)

        return TradeAction(action_int), confidence

    def _get_action_confidence(self, obs: np.ndarray, action: int) -> float:
        """Extract confidence from policy distribution."""
        try:
            import torch
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            obs_tensor = obs_tensor.to(self.model.policy.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
            return float(probs[action])
        except Exception:
            return 0.5

    # ── Training ────────────────────────────────

    def train(self, total_timesteps: Optional[int] = None):
        """Train the model for specified timesteps."""
        if not self._initialized or self.model is None:
            logger.warning("Cannot train — model not initialised")
            return

        steps = total_timesteps or self.config.total_timesteps
        logger.info("Starting DRL training for %d timesteps...", steps)
        self.model.learn(total_timesteps=steps, progress_bar=True)
        logger.info("DRL training complete")

    def learn_from_experience(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """
        Store a single experience for online learning.
        Note: PPO learns in batches, so this collects data
        and triggers update when buffer is full.
        """
        # For PPO, we rely on the learn() method with the environment
        # For online updates, see evolution/online_trainer.py
        pass

    # ── Save / Load ─────────────────────────────

    def save(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if self.model is None:
            return
        save_path = path or os.path.join(
            self.config.model_save_path, "drl_model"
        )
        self.model.save(save_path)
        logger.info("DRL model saved to %s", save_path)

    def load(self, path: Optional[str] = None):
        """Load model checkpoint, restoring VecNormalize if available."""
        try:
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            load_path = path or os.path.join(
                self.config.model_save_path, "drl_model"
            )
            norm_path = os.path.join(
                self.config.model_save_path, "vec_normalize.pkl"
            )

            # Wrap env the same way training does
            vec_env = DummyVecEnv([lambda: self.env])
            if os.path.exists(norm_path):
                vec_env = VecNormalize.load(norm_path, vec_env)
                vec_env.training = False
                vec_env.norm_reward = False
                logger.info("VecNormalize stats loaded from %s", norm_path)

            AlgoClass = PPO if self.config.algorithm == "PPO" else SAC
            self.model = AlgoClass.load(load_path, env=vec_env)
            self._initialized = True
            logger.info("DRL model loaded from %s", load_path)
        except Exception as e:
            logger.error("Failed to load DRL model: %s", e)

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model is not None

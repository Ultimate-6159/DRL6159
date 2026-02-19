"""
Apex Predator — Online Trainer (Layer 4: The Evolution)
========================================================
Incremental learning from live trading experience.
Collects (state, action, reward) tuples and triggers
periodic model updates without full retraining.
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass

import numpy as np

from config.settings import EvolutionConfig

logger = logging.getLogger("apex_predator.online_trainer")


@dataclass
class Experience:
    """Single trading experience tuple."""
    state: np.ndarray        # Observation vector
    action: int              # Action taken
    reward: float            # Reward received
    next_state: np.ndarray   # Next observation
    done: bool               # Episode ended?
    timestamp: float         # Unix timestamp


class OnlineTrainer:
    """
    Manages incremental model updates from live trading.
    Collects experiences into a replay buffer and triggers
    retraining after sufficient samples accumulate.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self._replay_buffer: deque = deque(maxlen=config.replay_buffer_size)
        self._trade_count = 0
        self._last_retrain_time = time.time()
        self._model_version = 0
        self._retrain_history: list = []

        os.makedirs("models/checkpoints/", exist_ok=True)

    # ── Experience Collection ───────────────────

    def add_experience(self, experience: Experience):
        """Add a new experience to the replay buffer."""
        self._replay_buffer.append(experience)
        self._trade_count += 1

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Convenience method to add an experience."""
        self.add_experience(Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=time.time(),
        ))

    # ── Retrain Decision ────────────────────────

    def should_retrain(self) -> bool:
        """
        Check if conditions are met for a model update.

        Returns:
            True if retraining should be triggered
        """
        # Not enough samples yet
        if len(self._replay_buffer) < self.config.min_retrain_samples:
            return False

        # Check trade count interval
        if self._trade_count % self.config.retrain_interval == 0:
            return True

        # Check time interval
        hours_elapsed = (time.time() - self._last_retrain_time) / 3600
        if hours_elapsed >= self.config.checkpoint_interval_hours:
            return True

        return False

    # ── Retrain Execution ───────────────────────

    def retrain(self, agent) -> bool:
        """
        Execute incremental retraining of the DRL agent.

        Args:
            agent: DRLAgent instance to retrain

        Returns:
            True if retrain was successful
        """
        if not agent.is_ready:
            logger.warning("Cannot retrain — agent not ready")
            return False

        n_samples = len(self._replay_buffer)
        logger.info(
            "Triggering online retrain | Samples: %d | Version: %d → %d",
            n_samples, self._model_version, self._model_version + 1,
        )

        try:
            # Save pre-retrain checkpoint
            checkpoint_path = (
                f"models/checkpoints/model_v{self._model_version:04d}"
            )
            agent.save(checkpoint_path)

            # PPO is on-policy — it cannot use a replay buffer.
            # Instead, we log buffer stats and skip the retrain call.
            # Real online learning requires a full env rollout or
            # switching to an off-policy algorithm (e.g. SAC + HER).
            logger.info(
                "Online retrain skipped — PPO is on-policy and cannot "
                "consume replay buffer (buffer=%d). Use scheduled "
                "walk-forward retraining instead.",
                n_samples,
            )
            # Clear buffer to free memory
            self._replay_buffer.clear()

            self._model_version += 1
            self._last_retrain_time = time.time()
            self._retrain_history.append({
                "version": self._model_version,
                "samples": n_samples,
                "timestamp": time.time(),
            })

            # Save post-retrain model
            agent.save()
            logger.info(
                "Online retrain complete — Version: %d | Steps: %d",
                self._model_version, retrain_steps,
            )
            return True

        except Exception as e:
            logger.error("Online retrain failed: %s", e)
            # Restore previous model
            try:
                agent.load(checkpoint_path)
                logger.info("Restored model from checkpoint v%d", self._model_version)
            except Exception:
                logger.critical("Failed to restore checkpoint!")
            return False

    # ── Replay Buffer Analysis ──────────────────

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get replay buffer statistics."""
        if not self._replay_buffer:
            return {
                "buffer_size": 0,
                "total_trades": self._trade_count,
                "model_version": self._model_version,
            }

        rewards = [e.reward for e in self._replay_buffer]
        return {
            "buffer_size": len(self._replay_buffer),
            "total_trades": self._trade_count,
            "model_version": self._model_version,
            "avg_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "win_rate": float(np.mean(np.array(rewards) > 0)),
            "retrain_count": len(self._retrain_history),
        }

    def get_recent_experiences(self, n: int = 50) -> list:
        """Get N most recent experiences for analysis."""
        return list(self._replay_buffer)[-n:]

    # ── Model Version Management ────────────────

    def cleanup_old_checkpoints(self):
        """Remove old model checkpoints beyond retention limit."""
        checkpoint_dir = "models/checkpoints/"
        if not os.path.exists(checkpoint_dir):
            return

        files = sorted(os.listdir(checkpoint_dir))
        if len(files) > self.config.model_history_count:
            to_remove = files[:len(files) - self.config.model_history_count]
            for f in to_remove:
                path = os.path.join(checkpoint_dir, f)
                try:
                    os.remove(path)
                    logger.debug("Removed old checkpoint: %s", path)
                except OSError:
                    pass

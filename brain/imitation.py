"""
Apex Predator — Imitation Learning (Behavioral Cloning)
=========================================================
Pre-trains the PPO policy by imitating a hindsight-optimal expert,
dramatically accelerating initial learning.

Pipeline:
  1. ExpertGenerator scans historical data with look-ahead oracle
     to produce optimal BUY/SELL/HOLD labels.
  2. ImitationPreTrainer fine-tunes the PPO policy network using
     cross-entropy loss against expert labels.
  3. After BC, PPO continues with RL fine-tuning (curriculum phases).

The model starts at ~45-55% accuracy instead of random 33%.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from config.settings import ImitationConfig

logger = logging.getLogger("apex_predator.imitation")


class ExpertGenerator:
    """
    Generates hindsight-optimal trade labels using a look-ahead oracle.

    For each bar, looks ahead N bars to determine if:
      - Going LONG would hit TP before SL  → label = BUY (0)
      - Going SHORT would hit TP before SL → label = SELL (1)
      - Neither is profitable              → label = HOLD (2)

    This creates a "perfect teacher" dataset for behavioral cloning.
    """

    def __init__(
        self,
        lookahead_bars: int = 30,
        sl_atr_mult: float = 0.8,
        tp_atr_mult: float = 0.96,
    ):
        self.lookahead = lookahead_bars
        self.sl_mult = sl_atr_mult
        self.tp_mult = tp_atr_mult

    def generate_labels(
        self,
        bars: np.ndarray,
        atrs: np.ndarray,
    ) -> np.ndarray:
        """
        Generate expert action labels for each bar.

        Args:
            bars: (N, 4) OHLC array
            atrs: (N,) ATR values

        Returns:
            labels: (N,) int array of {0=BUY, 1=SELL, 2=HOLD}
        """
        N = len(bars)
        labels = np.full(N, 2, dtype=np.int64)  # Default: HOLD

        for i in range(N - self.lookahead):
            close = bars[i, 3]  # Current close price
            atr = atrs[i]

            if atr < 1e-10:
                continue

            sl_dist = atr * self.sl_mult
            tp_dist = atr * self.tp_mult

            # ── Check BUY (long) ──────────────
            buy_tp = close + tp_dist
            buy_sl = close - sl_dist
            buy_result = self._check_trade(
                bars[i + 1: i + 1 + self.lookahead],
                entry=close,
                tp=buy_tp,
                sl=buy_sl,
                is_long=True,
            )

            # ── Check SELL (short) ────────────
            sell_tp = close - tp_dist
            sell_sl = close + sl_dist
            sell_result = self._check_trade(
                bars[i + 1: i + 1 + self.lookahead],
                entry=close,
                tp=sell_tp,
                sl=sell_sl,
                is_long=False,
            )

            # ── Decide label ──────────────────
            if buy_result > 0 and sell_result <= 0:
                labels[i] = 0  # BUY
            elif sell_result > 0 and buy_result <= 0:
                labels[i] = 1  # SELL
            elif buy_result > 0 and sell_result > 0:
                # Both profitable — pick the one that hits TP faster
                labels[i] = 0 if buy_result >= sell_result else 1
            # else: both negative → HOLD (default)

        # ── Log distribution ──────────────────
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        logger.info(
            "Expert labels generated: %d bars — BUY=%d, SELL=%d, HOLD=%d",
            N, dist.get(0, 0), dist.get(1, 0), dist.get(2, 0),
        )

        return labels

    @staticmethod
    def _check_trade(
        future_bars: np.ndarray,
        entry: float,
        tp: float,
        sl: float,
        is_long: bool,
    ) -> float:
        """
        Simulate a trade on future bars and return profit ratio.

        Returns:
            > 0 if TP hit first (profitable)
            < 0 if SL hit first (loss)
            0 if neither hit (expired)
        """
        for bar in future_bars:
            high = bar[1]
            low = bar[2]

            if is_long:
                if low <= sl:
                    return -1.0  # SL hit
                if high >= tp:
                    return 1.0   # TP hit
            else:
                if high >= sl:
                    return -1.0  # SL hit
                if low <= tp:
                    return 1.0   # TP hit

        return 0.0  # Neither hit — expired


class ImitationPreTrainer:
    """
    Behavioral cloning: supervised pre-training of PPO policy network.

    Takes the SB3 PPO model's policy network and trains it with
    cross-entropy loss against expert labels. This gives the agent
    a strong starting point before RL fine-tuning.
    """

    def __init__(self, config: ImitationConfig):
        self.config = config

    def pretrain(
        self,
        model,
        env,
        expert_labels: np.ndarray,
        features: np.ndarray,
        regimes: np.ndarray,
    ) -> dict:
        """
        Pre-train the PPO policy network using behavioral cloning.

        Args:
            model: SB3 PPO model (with policy network to train)
            env: BacktestEnv (for generating observations)
            expert_labels: (N,) expert action labels
            features: (N, feat_dim) feature matrix
            regimes: (N,) regime indices

        Returns:
            Dict with training stats (loss, accuracy per epoch)
        """
        logger.info("=" * 60)
        logger.info("   IMITATION LEARNING — Behavioral Cloning")
        logger.info("   Epochs: %d | LR: %s | Batch: %d",
                     self.config.epochs, self.config.learning_rate,
                     self.config.batch_size)
        logger.info("=" * 60)

        # ── 1. Generate observations using env ────────
        logger.info("Generating observation dataset from environment...")
        observations, valid_labels = self._generate_observations(
            env, expert_labels
        )

        if len(observations) < 100:
            logger.warning("Too few observations (%d) for BC. Skipping.",
                           len(observations))
            return {"skipped": True, "reason": "too_few_observations"}

        logger.info("BC dataset: %d observations, %d features",
                     len(observations), observations.shape[1])

        # ── 2. Create PyTorch dataset ──────────────────
        device = next(model.policy.parameters()).device
        X = torch.FloatTensor(observations).to(device)
        y = torch.LongTensor(valid_labels).to(device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # ── 3. Set up optimizer for policy network ─────
        # Only train the action (policy) network, not the value network
        policy_params = list(model.policy.action_net.parameters())
        # Also include the shared feature extractor
        policy_params += list(model.policy.mlp_extractor.policy_net.parameters())

        optimizer = optim.Adam(policy_params, lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # ── 4. Training loop ───────────────────────────
        stats = {"epochs": [], "losses": [], "accuracies": []}

        model.policy.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_X, batch_y in loader:
                # Forward pass through the policy network
                features_out = model.policy.extract_features(
                    batch_X, model.policy.features_extractor
                )
                latent_pi, _ = model.policy.mlp_extractor(features_out)
                action_logits = model.policy.action_net(latent_pi)

                loss = criterion(action_logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                preds = action_logits.argmax(dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += len(batch_y)

            avg_loss = total_loss / max(total_samples, 1)
            accuracy = total_correct / max(total_samples, 1)

            stats["epochs"].append(epoch + 1)
            stats["losses"].append(avg_loss)
            stats["accuracies"].append(accuracy)

            logger.info(
                "  BC Epoch %d/%d — Loss: %.4f | Accuracy: %.1f%%",
                epoch + 1, self.config.epochs, avg_loss, 100 * accuracy,
            )

        model.policy.eval()

        final_acc = stats["accuracies"][-1] if stats["accuracies"] else 0
        logger.info(
            "Behavioral cloning complete — Final accuracy: %.1f%%",
            100 * final_acc,
        )

        return stats

    def _generate_observations(
        self,
        env,
        expert_labels: np.ndarray,
    ):
        """
        Generate observation vectors by stepping through the env.

        Collects the observation at each timestep and pairs it
        with the expert label. Skips the lookback warmup period.

        Returns:
            observations: (M, obs_dim) array
            valid_labels: (M,) array
        """
        observations = []
        valid_labels = []

        obs, _ = env.reset()
        lookback = env.lookback

        # Walk through the environment collecting observations
        max_steps = min(len(expert_labels), env.max_steps)

        for step in range(max_steps):
            # Get the data index corresponding to this env step
            data_idx = env.current_step
            if data_idx < len(expert_labels):
                label = expert_labels[data_idx]
                observations.append(obs.copy())
                valid_labels.append(label)

            # Step with the expert's action (to advance the env)
            action = expert_labels[data_idx] if data_idx < len(expert_labels) else 2
            obs, _, terminated, truncated, _ = env.step(int(action))

            if terminated or truncated:
                obs, _ = env.reset()

        observations = np.array(observations, dtype=np.float32)
        valid_labels = np.array(valid_labels, dtype=np.int64)

        return observations, valid_labels

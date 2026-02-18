"""
Apex Predator — Training Script
==================================
Downloads historical data from MT5, preprocesses it,
and trains the PPO agent using BacktestEnv.

Usage:
    python train.py                          # Train with defaults
    python train.py --bars 50000             # More historical data
    python train.py --timesteps 500000       # Train longer
    python train.py --symbol XAUUSDm --tf M1 # Custom pair/timeframe
    python train.py --no-mt5                 # Use mock data if no MT5
"""

import sys
import os
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# ── Internal Imports ────────────────────────────
from config.settings import ApexConfig, MarketRegime
from utils.logger import setup_logger
from utils.training_report import TrainingReport, new_training_report
from core.mt5_connector import MT5Connector
from core.feature_engine import FeatureEngine
from brain.regime_classifier import RegimeClassifier
from brain.backtest_env import BacktestEnv
from brain.curriculum import CurriculumScheduler
from brain.imitation import ExpertGenerator, ImitationPreTrainer

logger = logging.getLogger("apex_predator.train")


# ═══════════════════════════════════════════════════════════════
# Custom Callback to Log PPO Metrics
# ═══════════════════════════════════════════════════════════════

from stable_baselines3.common.callbacks import BaseCallback


class PPOMetricsCallback(BaseCallback):
    """Callback to capture PPO training metrics for the report."""

    def __init__(self, report: TrainingReport = None, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.report = report
        self.log_freq = log_freq
        self.last_log_timestep = 0

    def _on_step(self) -> bool:
        """Called after each step (env.step)."""
        # Log every log_freq timesteps
        if self.num_timesteps - self.last_log_timestep >= self.log_freq:
            self.last_log_timestep = self.num_timesteps

            # Get logged values from model's logger
            if hasattr(self.model, "logger") and self.model.logger is not None:
                name_to_value = getattr(self.model.logger, "name_to_value", {})
                if name_to_value and self.report:
                    self.report.log_ppo_metrics(self.num_timesteps, name_to_value)

        return True  # Continue training


def download_data(config: ApexConfig, bar_count: int, use_mt5: bool = True) -> pd.DataFrame:
    """
    Download historical OHLC data from MT5.
    Falls back to mock data if MT5 is unavailable.
    """
    connector = MT5Connector(config.mt5, dry_run=not use_mt5)

    if use_mt5:
        logger.info("Connecting to MT5 for historical data...")
        connected = connector.connect()
        if not connected:
            logger.warning("MT5 connection failed — falling back to mock data")
            return connector._mock_ohlc(bar_count)
    else:
        logger.info("Using mock data (--no-mt5 flag)")

    logger.info("Downloading %d bars of %s %s ...",
                bar_count, config.mt5.symbol, config.mt5.timeframe)

    df = connector.get_ohlc(count=bar_count)

    if use_mt5:
        connector.disconnect()

    if df is None or df.empty:
        logger.warning("No data received — using mock data")
        df = connector._mock_ohlc(bar_count)

    logger.info("Data downloaded: %d bars | %s to %s",
                len(df),
                df["time"].iloc[0] if "time" in df.columns else "?",
                df["time"].iloc[-1] if "time" in df.columns else "?")
    return df


def preprocess_data(df: pd.DataFrame, config: ApexConfig):
    """
    Compute features, regimes, spreads, and ATRs from raw OHLC.

    Returns:
        bars: (N, 4) OHLC array
        features: (N', feat_dim) normalized features (shorter due to dropna)
        regimes: (N',) regime indices
        spreads: (N',) spread values
        atrs: (N',) ATR values
        valid_start: index where valid features begin
    """
    logger.info("Computing features...")
    feature_engine = FeatureEngine(config.features)
    regime_classifier = RegimeClassifier(config.regime)

    # ── Compute ATR on full data ────────────
    atr_series = feature_engine._compute_atr(df)

    # ── Compute features (this drops NaN rows) ──
    features = feature_engine.compute(df)
    if features is None:
        raise ValueError("Feature computation failed — insufficient data")

    # Number of rows lost to rolling windows
    rows_lost = len(df) - len(features)
    logger.info("Features: %d rows (lost %d to rolling windows)", len(features), rows_lost)

    # ── Extract aligned data ────────────────
    # After dropna, features correspond to df.iloc[rows_lost:]
    valid_df = df.iloc[rows_lost:].reset_index(drop=True)
    bars = valid_df[["open", "high", "low", "close"]].values.astype(np.float32)

    # ── Spreads ─────────────────────────────
    if "spread" in valid_df.columns:
        spreads = valid_df["spread"].values.astype(np.float32)
    else:
        spreads = np.full(len(valid_df), 2.0, dtype=np.float32)

    # ── ATR (aligned) ──────────────────────
    atrs = atr_series.iloc[rows_lost:].reset_index(drop=True).values.astype(np.float32)
    atrs = np.nan_to_num(atrs, nan=1.0)

    # ── Regimes ─────────────────────────────
    logger.info("Computing market regimes...")
    regimes = np.full(len(valid_df), 3, dtype=np.int32)  # Default: UNCERTAIN

    # Classify regime using sliding window
    window = max(config.regime.volatility_window, config.regime.trend_window) + 50
    for i in range(window, len(df)):
        chunk = df.iloc[max(0, i - window):i]
        regime, _ = regime_classifier.classify(chunk)
        aligned_idx = i - rows_lost
        if 0 <= aligned_idx < len(regimes):
            regimes[aligned_idx] = list(MarketRegime).index(regime)

    regime_counts = np.bincount(regimes, minlength=4)
    logger.info("Regimes: TRENDING=%d, MEAN_REV=%d, HIGH_VOL=%d, UNCERTAIN=%d",
                *regime_counts)

    return bars, features, regimes, spreads, atrs


def create_train_env(
    bars, features, regimes, spreads, atrs,
    config: ApexConfig,
) -> BacktestEnv:
    """Create the backtesting environment for training."""

    # Detect instrument type for contract size
    symbol = config.mt5.symbol.upper()
    if "XAU" in symbol or "GOLD" in symbol:
        contract_size = 100.0     # Gold: 100 oz per lot
        point_value = 0.01
    elif "JPY" in symbol:
        contract_size = 100000.0  # JPY pairs
        point_value = 0.001
    else:
        contract_size = 100000.0  # Standard Forex
        point_value = 0.00001

    env = BacktestEnv(
        bars=bars,
        features=features,
        regimes=regimes,
        spreads=spreads,
        atrs=atrs,
        lookback=10,
        reward_config=config.reward,
        initial_balance=10000.0,
        lot_size=0.01,
        contract_size=contract_size,
        point_value=point_value,
        max_hold_bars=config.reward.max_hold_steps,
        sl_atr_mult=config.risk.atr_multiplier,
        tp_atr_mult=config.risk.atr_multiplier * config.risk.tp_ratio,
    )
    return env


def train_model(env: BacktestEnv, config: ApexConfig, total_timesteps: int):
    """Train PPO on the backtesting environment."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    os.makedirs(config.drl.model_save_path, exist_ok=True)

    logger.info("=" * 60)
    logger.info("   PPO Training Configuration")
    logger.info("   Timesteps: %s", f"{total_timesteps:,}")
    logger.info("   Learning Rate: %s", config.drl.learning_rate)
    logger.info("   Batch Size: %s", config.drl.batch_size)
    logger.info("   GAE Lambda: %s", config.drl.gae_lambda)
    logger.info("   Clip Range: %s", config.drl.clip_range)
    logger.info("=" * 60)

    # ── Cosine LR Schedule ──────────────────
    def cosine_lr_schedule(progress_remaining: float) -> float:
        """Cosine decay: LR decays from 1.0x to 0.1x over training."""
        import math
        return 0.1 + 0.9 * (1 + math.cos(math.pi * (1 - progress_remaining))) / 2

    # ── Wrap env with VecNormalize (reward only) ──
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,       # Features already normalized
        norm_reward=True,     # Normalize rewards → stable value function
        clip_reward=10.0,
        gamma=config.drl.gamma,
    )

    # ── Create PPO Model ────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
    )

    safe_n_steps = max(
        min(config.drl.n_steps, env.max_steps - 1) // config.drl.batch_size, 1
    ) * config.drl.batch_size

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lambda p: config.drl.learning_rate * cosine_lr_schedule(p),
        gamma=config.drl.gamma,
        gae_lambda=config.drl.gae_lambda,
        clip_range=config.drl.clip_range,
        n_steps=safe_n_steps,
        batch_size=config.drl.batch_size,
        n_epochs=config.drl.n_epochs,
        ent_coef=config.drl.ent_coef,
        vf_coef=config.drl.vf_coef,
        max_grad_norm=config.drl.max_grad_norm,
        target_kl=config.drl.target_kl,
        normalize_advantage=config.drl.normalize_advantage,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        verbose=1,
    )

    # ── Callbacks ───────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1000),
        save_path=config.drl.model_save_path,
        name_prefix="ppo_checkpoint",
    )

    # ── Train ───────────────────────────────
    logger.info("Starting PPO training...")
    t0 = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=False,
    )

    elapsed = time.time() - t0
    logger.info("Training complete in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    # ── Save Final Model + VecNormalize stats ──
    save_path = os.path.join(config.drl.model_save_path, "drl_model")
    model.save(save_path)
    vec_env.save(os.path.join(config.drl.model_save_path, "vec_normalize.pkl"))
    logger.info("Model saved to: %s", save_path)

    return model


def train_with_curriculum(
    bars, features, regimes, spreads, atrs,
    config: ApexConfig,
    total_timesteps: int,
    use_imitation: bool = True,
    imitation_epochs: int = 10,
    report: TrainingReport = None,
):
    """
    Full training pipeline with Imitation Learning + Curriculum Learning.

    Flow:
      1. Create initial env (full data)
      2. Imitation pre-training (behavioral cloning from expert oracle)
      3. Curriculum Phase 1: trending only
      4. Curriculum Phase 2: trending + ranging
      5. Curriculum Phase 3: all regimes
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    os.makedirs(config.drl.model_save_path, exist_ok=True)

    # ── Step 1: Create initial model on full data ────────
    logger.info("Creating initial environment on full dataset...")
    full_env = create_train_env(bars, features, regimes, spreads, atrs, config)

    # ── Wrap with VecNormalize (reward only) ─────────────
    vec_env = DummyVecEnv([lambda: full_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=config.drl.gamma,
    )

    # ── Cosine LR Schedule ──────────────────
    def cosine_lr_schedule(progress_remaining: float) -> float:
        """Cosine decay: LR decays from 1.0x to 0.1x over training."""
        import math
        return 0.1 + 0.9 * (1 + math.cos(math.pi * (1 - progress_remaining))) / 2

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
    )

    safe_n_steps = max(
        min(config.drl.n_steps, full_env.max_steps - 1) // config.drl.batch_size, 1
    ) * config.drl.batch_size

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lambda p: config.drl.learning_rate * cosine_lr_schedule(p),
        gamma=config.drl.gamma,
        gae_lambda=config.drl.gae_lambda,
        clip_range=config.drl.clip_range,
        n_steps=safe_n_steps,
        batch_size=config.drl.batch_size,
        n_epochs=config.drl.n_epochs,
        ent_coef=config.drl.ent_coef,
        vf_coef=config.drl.vf_coef,
        max_grad_norm=config.drl.max_grad_norm,
        target_kl=config.drl.target_kl,
        normalize_advantage=config.drl.normalize_advantage,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        verbose=1,
    )

    # ── Step 2: Imitation Pre-Training ───────────────────
    bc_stats = None
    if use_imitation and config.imitation.enabled:
        logger.info("")
        logger.info("🎓 IMITATION LEARNING — Pre-Training from Expert Oracle")
        logger.info("=" * 60)

        config.imitation.epochs = imitation_epochs

        expert_gen = ExpertGenerator(
            lookahead_bars=config.imitation.lookahead_bars,
            sl_atr_mult=config.risk.atr_multiplier,
            tp_atr_mult=config.risk.atr_multiplier * config.risk.tp_ratio,
        )
        expert_labels = expert_gen.generate_labels(bars, atrs)

        pre_trainer = ImitationPreTrainer(config.imitation)
        bc_stats = pre_trainer.pretrain(
            model=model,
            env=full_env,
            expert_labels=expert_labels,
            features=features,
            regimes=regimes,
        )

        if not bc_stats.get("skipped"):
            logger.info("✅ Imitation pre-training complete!")
        else:
            logger.warning("⚠️ Imitation pre-training skipped: %s",
                           bc_stats.get("reason", "unknown"))

        # Log imitation learning results to report
        if report:
            report.log_imitation(bc_stats)
    else:
        logger.info("Imitation learning disabled — skipping.")
        if report:
            report.log_imitation({"skipped": True, "reason": "disabled"})

    # ── Step 3: Curriculum Training ──────────────────────
    scheduler = CurriculumScheduler(config.curriculum)

    if config.curriculum.enabled:
        schedule = scheduler.get_phase_schedule(total_timesteps)

        for phase in schedule:
            logger.info("")
            logger.info("📚 %s — %s", phase["name"], phase["description"])
            logger.info("   Timesteps: %s | Regimes: %s | Vaccine: %s (%.0f%%)",
                         f"{phase['timesteps']:,}", phase["regimes"],
                         phase.get("vaccine_regimes", []),
                         phase.get("vaccine_pct", 0) * 100)
            logger.info("=" * 60)

            # Filter data for this phase (with vaccine for anti-forgetting)
            p_bars, p_feats, p_regimes, p_spreads, p_atrs = \
                scheduler.filter_data_by_regimes(
                    bars, features, regimes, spreads, atrs,
                    allowed_regimes=phase["regimes"],
                    vaccine_regimes=phase.get("vaccine_regimes", []),
                    vaccine_pct=phase.get("vaccine_pct", 0.0),
                )

            if len(p_bars) < 200:
                logger.warning(
                    "Skipping %s — too few bars (%d)",
                    phase["name"], len(p_bars),
                )
                continue

            # Create env for this phase
            phase_env = create_train_env(
                p_bars, p_feats, p_regimes, p_spreads, p_atrs, config
            )

            scheduler.log_phase_stats(phase["name"], p_bars, p_regimes)

            # Compute n_steps for this phase env
            batch_sz = config.drl.batch_size
            max_n = max(phase_env.max_steps - 1, batch_sz)
            desired_n = min(config.drl.n_steps, max_n)
            desired_n = max(desired_n // batch_sz, 1) * batch_sz

            # Wrap phase env with VecNormalize (reward only)
            phase_vec_env = DummyVecEnv([lambda: phase_env])
            phase_vec_env = VecNormalize(
                phase_vec_env,
                norm_obs=False,
                norm_reward=True,
                clip_reward=10.0,
                gamma=config.drl.gamma,
            )

            # Save current model weights, then reload into a fresh PPO
            # with the new env. This properly rebuilds the rollout buffer.
            tmp_path = os.path.join(config.drl.model_save_path, "_phase_tmp")
            model.save(tmp_path)

            model = PPO.load(
                tmp_path,
                env=phase_vec_env,
                n_steps=desired_n,
                batch_size=batch_sz,
                tensorboard_log=None,
                verbose=1,
            )

            # Train this phase with callbacks
            checkpoint_cb = CheckpointCallback(
                save_freq=max(phase["timesteps"] // 5, 1000),
                save_path=config.drl.model_save_path,
                name_prefix=f"ppo_phase{phase['phase_idx']}",
            )

            # PPO metrics callback
            metrics_cb = PPOMetricsCallback(report=report, log_freq=50000)
            callbacks = [checkpoint_cb, metrics_cb]

            t0 = time.time()
            model.learn(
                total_timesteps=phase["timesteps"],
                callback=callbacks,
                progress_bar=False,
                reset_num_timesteps=True,  # Fresh count per phase
            )
            elapsed = time.time() - t0
            logger.info(
                "%s complete in %.1f sec (%.1f min)",
                phase["name"], elapsed, elapsed / 60,
            )

            # Log curriculum phase to report
            if report:
                phase_regime_counts = np.bincount(p_regimes, minlength=4)
                report.log_curriculum_phase(
                    phase_idx=phase["phase_idx"],
                    phase_name=phase["name"],
                    timesteps=phase["timesteps"],
                    bars=len(p_bars),
                    regime_counts={
                        "TRENDING": int(phase_regime_counts[0]),
                        "MEAN_REVERTING": int(phase_regime_counts[1]),
                        "HIGH_VOLATILITY": int(phase_regime_counts[2]),
                        "UNCERTAIN": int(phase_regime_counts[3]),
                    },
                    elapsed_sec=elapsed,
                )
    else:
        logger.info("Curriculum disabled — training on full data.")
        model = train_model(full_env, config, total_timesteps)

    # ── Save Final Model + VecNormalize stats ────────────
    save_path = os.path.join(config.drl.model_save_path, "drl_model")
    model.save(save_path)
    norm_path = os.path.join(config.drl.model_save_path, "vec_normalize.pkl")
    model_env = model.get_env()
    if isinstance(model_env, VecNormalize):
        model_env.save(norm_path)
        logger.info("VecNormalize stats saved to: %s", norm_path)
    logger.info("Model saved to: %s", save_path)

    return model, full_env


def evaluate_model(model, env: BacktestEnv, n_episodes: int = 5):
    """Evaluate trained model and print statistics."""
    logger.info("=" * 60)
    logger.info("   Evaluating Model (%d episodes)", n_episodes)
    logger.info("=" * 60)

    all_stats = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        stats = env.get_stats()
        all_stats.append(stats)
        logger.info(
            "Episode %d | Trades: %d | Win Rate: %.1f%% | PnL: $%.2f | "
            "Balance: $%.2f | Sharpe: %.2f",
            ep + 1, stats["total_trades"], stats["win_rate"] * 100,
            stats["total_pnl"], stats["final_balance"], stats["sharpe_ratio"],
        )

    # ── Summary ─────────────────────────────
    avg_wr = np.mean([s["win_rate"] for s in all_stats])
    avg_pnl = np.mean([s["total_pnl"] for s in all_stats])
    avg_sharpe = np.mean([s["sharpe_ratio"] for s in all_stats])
    avg_trades = np.mean([s["total_trades"] for s in all_stats])

    logger.info("-" * 60)
    logger.info("   AVERAGE RESULTS (%d episodes)", n_episodes)
    logger.info("   Trades/ep: %.0f | Win Rate: %.1f%% | PnL: $%.2f | Sharpe: %.2f",
                avg_trades, avg_wr * 100, avg_pnl, avg_sharpe)
    logger.info("-" * 60)

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Train Apex Predator DRL Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                          # Default training
  python train.py --bars 50000             # More data
  python train.py --timesteps 500000       # Train longer
  python train.py --no-mt5                 # Without MT5
  python train.py --no-curriculum          # Skip curriculum
  python train.py --no-imitation           # Skip imitation
  python train.py --symbol XAUUSDm --tf M5 # Gold on M5
        """,
    )
    parser.add_argument("--bars", type=int, default=50000,
                        help="Number of historical bars to download (default: 50000)")
    parser.add_argument("--timesteps", type=int, default=2000000,
                        help="Training timesteps (default: 2000000)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Symbol to train on (overrides settings.py)")
    parser.add_argument("--tf", type=str, default=None,
                        help="Timeframe (overrides settings.py)")
    parser.add_argument("--no-mt5", action="store_true",
                        help="Use mock data instead of MT5")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning (train on all data)")
    parser.add_argument("--no-imitation", action="store_true",
                        help="Disable imitation pre-training")
    parser.add_argument("--imitation-epochs", type=int, default=10,
                        help="Number of behavioral cloning epochs (default: 10)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes after training")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # ── Configuration ───────────────────────
    config = ApexConfig()
    config.log_level = args.log_level
    if args.symbol:
        config.mt5.symbol = args.symbol
    if args.tf:
        config.mt5.timeframe = args.tf
    if args.no_curriculum:
        config.curriculum.enabled = False
    if args.no_imitation:
        config.imitation.enabled = False

    # ── Setup Logger ────────────────────────
    setup_logger(log_dir=config.log_dir, level=config.log_level)

    logger.info("=" * 60)
    logger.info("   APEX PREDATOR — Training Mode")
    logger.info("   Symbol: %s | TF: %s", config.mt5.symbol, config.mt5.timeframe)
    logger.info("   Bars: %s | Timesteps: %s",
                f"{args.bars:,}", f"{args.timesteps:,}")
    logger.info("   Curriculum: %s | Imitation: %s",
                "ON" if config.curriculum.enabled else "OFF",
                "ON" if config.imitation.enabled else "OFF")
    logger.info("=" * 60)

    # ── Training Report ─────────────────────
    report = new_training_report(
        output_dir="reports/",
        experiment_name=f"{config.mt5.symbol}_{config.mt5.timeframe}",
    )
    report.log_config(config, extra_params={
        "bars": args.bars,
        "timesteps": args.timesteps,
        "use_mt5": not args.no_mt5,
        "curriculum_enabled": config.curriculum.enabled,
        "imitation_enabled": config.imitation.enabled,
        "imitation_epochs": args.imitation_epochs,
        "eval_episodes": args.eval_episodes,
    })
    logger.info("📝 Training report initialized: %s", report.report_path)

    try:
        # ── Step 1: Download Data ───────────────
        df = download_data(config, args.bars, use_mt5=not args.no_mt5)

        # ── Step 2: Preprocess ──────────────────
        bars, features, regimes, spreads, atrs = preprocess_data(df, config)
        logger.info("Preprocessed: %d bars → %d valid samples", len(df), len(bars))

        # Log data info to report
        regime_counts = np.bincount(regimes, minlength=4)
        report.log_data_info(
            total_bars=len(df),
            valid_bars=len(bars),
            start_date=str(df["time"].iloc[0]) if "time" in df.columns else None,
            end_date=str(df["time"].iloc[-1]) if "time" in df.columns else None,
            regime_counts={
                "TRENDING": int(regime_counts[0]),
                "MEAN_REVERTING": int(regime_counts[1]),
                "HIGH_VOLATILITY": int(regime_counts[2]),
                "UNCERTAIN": int(regime_counts[3]),
            },
        )

        # ── Step 3: Train (Imitation + Curriculum) ──
        logger.info("Creating BacktestEnv (raw features, no LSTM)...")
        model, eval_env = train_with_curriculum(
            bars, features, regimes, spreads, atrs,
            config=config,
            total_timesteps=args.timesteps,
            use_imitation=not args.no_imitation,
            imitation_epochs=args.imitation_epochs,
            report=report,
        )

        # ── Step 4: Evaluate ────────────────────
        eval_stats = evaluate_model(model, eval_env, n_episodes=args.eval_episodes)

        # Log evaluation results to report
        for i, stats in enumerate(eval_stats):
            report.log_evaluation_episode(i + 1, stats)
        report.log_evaluation_summary(eval_stats)

        # Finalize report (success)
        report.finalize(status="completed")
        logger.info("📊 Training report saved to: %s", report.report_path)

        logger.info("")
        logger.info("✅ Training complete! Model saved to: %s",
                    os.path.join(config.drl.model_save_path, "drl_model"))
        logger.info("   Run 'python main.py --live' to trade with trained model")

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user (Ctrl+C)")
        report.finalize(status="interrupted")
        logger.info("📊 Partial report saved to: %s", report.report_path)
        raise

    except Exception as e:
        logger.error("❌ Training failed with error: %s", str(e))
        import traceback
        report.report["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        report.finalize(status="failed")
        logger.info("📊 Error report saved to: %s", report.report_path)
        raise


if __name__ == "__main__":
    main()

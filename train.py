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
from core.mt5_connector import MT5Connector
from core.feature_engine import FeatureEngine
from brain.regime_classifier import RegimeClassifier
from brain.backtest_env import BacktestEnv

logger = logging.getLogger("apex_predator.train")


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

    os.makedirs(config.drl.model_save_path, exist_ok=True)

    logger.info("=" * 60)
    logger.info("   PPO Training Configuration")
    logger.info("   Timesteps: %s", f"{total_timesteps:,}")
    logger.info("   Learning Rate: %s", config.drl.learning_rate)
    logger.info("   Batch Size: %s", config.drl.batch_size)
    logger.info("   GAE Lambda: %s", config.drl.gae_lambda)
    logger.info("   Clip Range: %s", config.drl.clip_range)
    logger.info("=" * 60)

    # ── Create PPO Model ────────────────────
    # Note: tensorboard_log=None to avoid requiring tensorboard package
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.drl.learning_rate,
        gamma=config.drl.gamma,
        gae_lambda=config.drl.gae_lambda,
        clip_range=config.drl.clip_range,
        n_steps=min(config.drl.n_steps, env.max_steps // 2),  # Fit within data
        batch_size=config.drl.batch_size,
        n_epochs=config.drl.n_epochs,
        ent_coef=config.drl.ent_coef,
        vf_coef=config.drl.vf_coef,
        max_grad_norm=config.drl.max_grad_norm,
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

    # ── Save Final Model ────────────────────
    save_path = os.path.join(config.drl.model_save_path, "drl_model")
    model.save(save_path)
    logger.info("Model saved to: %s", save_path)

    return model


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
  python train.py --symbol XAUUSDm --tf M5 # Gold on M5
        """,
    )
    parser.add_argument("--bars", type=int, default=10000,
                        help="Number of historical bars to download (default: 10000)")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Training timesteps (default: 100000)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Symbol to train on (overrides settings.py)")
    parser.add_argument("--tf", type=str, default=None,
                        help="Timeframe (overrides settings.py)")
    parser.add_argument("--no-mt5", action="store_true",
                        help="Use mock data instead of MT5")
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

    # ── Setup Logger ────────────────────────
    setup_logger(log_dir=config.log_dir, level=config.log_level)

    logger.info("=" * 60)
    logger.info("   APEX PREDATOR — Training Mode")
    logger.info("   Symbol: %s | TF: %s", config.mt5.symbol, config.mt5.timeframe)
    logger.info("   Bars: %s | Timesteps: %s",
                f"{args.bars:,}", f"{args.timesteps:,}")
    logger.info("=" * 60)

    # ── Step 1: Download Data ───────────────
    df = download_data(config, args.bars, use_mt5=not args.no_mt5)

    # ── Step 2: Preprocess ──────────────────
    bars, features, regimes, spreads, atrs = preprocess_data(df, config)
    logger.info("Preprocessed: %d bars → %d valid samples", len(df), len(bars))

    # ── Step 3: Create Environment ──────────
    logger.info("Creating BacktestEnv (raw features, no LSTM)...")
    env = create_train_env(bars, features, regimes, spreads, atrs, config)

    # Quick sanity check
    obs, _ = env.reset()
    logger.info("Env sanity check: obs_shape=%s, action_space=%s",
                obs.shape, env.action_space)

    # ── Step 5: Train ───────────────────────
    model = train_model(env, config, args.timesteps)

    # ── Step 6: Evaluate ────────────────────
    evaluate_model(model, env, n_episodes=args.eval_episodes)

    logger.info("")
    logger.info("✅ Training complete! Model saved to: %s",
                os.path.join(config.drl.model_save_path, "drl_model"))
    logger.info("   Run 'python main.py --live' to trade with trained model")


if __name__ == "__main__":
    main()

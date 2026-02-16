"""
Apex Predator — Feature Engineering Module
============================================
Transforms raw OHLC/tick data into fixed-dimension
feature vectors ready for neural network input.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import FeatureConfig

logger = logging.getLogger("apex_predator.features")


class FeatureEngine:
    """
    Calculates technical features and normalizes them
    into a fixed-dimension numpy array for the perception module.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config

    # ── Main Entry Point ────────────────────────

    def compute(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Compute all features from OHLC DataFrame.

        Args:
            df: DataFrame with columns [time, open, high, low, close, volume, spread]

        Returns:
            np.ndarray of shape (len(df), num_features) — normalized feature matrix
        """
        if df is None or len(df) < self.config.atr_period + 10:
            logger.warning("Insufficient data for feature computation: %s rows",
                           len(df) if df is not None else 0)
            return None

        features = pd.DataFrame(index=df.index)

        # ── Price Features ──────────────────────
        features["close"] = df["close"]
        features["high"] = df["high"]
        features["low"] = df["low"]
        features["volume"] = df["volume"].astype(float)

        # ── Spread ──────────────────────────────
        if "spread" in df.columns:
            features["spread"] = df["spread"].astype(float)
        else:
            features["spread"] = 0.0

        # ── ATR (Average True Range) ────────────
        features["atr"] = self._compute_atr(df)

        # ── Returns ─────────────────────────────
        if self.config.log_returns:
            features["returns"] = np.log(df["close"] / df["close"].shift(1))
        else:
            features["returns"] = df["close"].pct_change()

        # ── Z-Score Normalization ────────────────
        window = self.config.z_score_window
        rolling_mean = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()
        features["z_score"] = (df["close"] - rolling_mean) / (rolling_std + 1e-10)

        # ── Volatility ──────────────────────────
        features["volatility"] = df["close"].pct_change().rolling(
            window=self.config.atr_period
        ).std()

        # ── Momentum ────────────────────────────
        features["momentum"] = df["close"] - df["close"].shift(self.config.atr_period)

        # ── RSI (Raw — not bounded 0-100) ───────
        features["rsi_raw"] = self._compute_rsi(df["close"], self.config.atr_period)

        # ── EMA Crossover (5/20) ────────────────
        ema5 = df["close"].ewm(span=5, adjust=False).mean()
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        features["ema_cross"] = (ema5 - ema20) / (df["close"] + 1e-10)

        # ── MACD Signal ─────────────────────────
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        features["macd_signal"] = (macd_line - signal_line) / (df["close"] + 1e-10)

        # ── Bollinger Band Position ──────────────
        bb_mid = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # ── 3-bar Price Change ───────────────────
        features["price_change_3"] = df["close"].pct_change(3)

        # ── High-Low Ratio (bar volatility) ──────
        features["high_low_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)

        # ── Body Ratio (candle strength) ─────────
        features["body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)

        # ── Drop NaN rows (from rolling calculations)
        features = features.dropna().reset_index(drop=True)

        if features.empty:
            logger.warning("All features are NaN after computation")
            return None

        # ── Normalize ───────────────────────────
        normalized = self._normalize(features.values)
        logger.debug("Features computed: shape=%s", normalized.shape)
        return normalized

    def get_feature_names(self) -> list:
        """Return ordered list of feature names."""
        return list(self.config.features)

    def get_feature_dim(self) -> int:
        """Return number of features (output dimension)."""
        return len(self.config.features)

    # ── ATR Calculation ─────────────────────────

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Compute Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.config.atr_period).mean()
        return atr

    # ── RSI Calculation ─────────────────────────

    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Compute RSI (raw float, not clipped).
        Returns values roughly in [-1, 1] range after normalization.
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        # Normalize RSI to [-1, 1] range
        return (rsi - 50) / 50

    # ── Normalization ───────────────────────────

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Z-score normalize each feature column.
        Handles zero-variance columns gracefully.
        """
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < 1e-10] = 1.0  # Prevent division by zero
        return (data - mean) / std

    def compute_single_bar(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Compute features and return only the latest bar's features.
        Useful for real-time inference.

        Returns:
            1D numpy array of shape (num_features,)
        """
        full = self.compute(df)
        if full is None or len(full) == 0:
            return None
        return full[-1]

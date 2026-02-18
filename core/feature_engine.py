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

        # ── MOMENTUM SNIPER FEATURES (ตรวจจับจังหวะระเบิด) ────
        # 1. RSI Fast (5-period) - แรงส่งระยะสั้น
        features["rsi_fast"] = self._compute_rsi(df["close"], 5)

        # 2. Bollinger Band Width - ดูกราฟบีบตัว (Squeeze) รอระเบิด
        bb_width_raw = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        features["bb_width"] = bb_width_raw

        # 3. Volume Spike - แรงกระชากของ Volume เทียบกับค่าเฉลี่ย
        vol_ma = df["volume"].rolling(window=20).mean()
        features["vol_spike"] = df["volume"].astype(float) / (vol_ma + 1e-10)

        # ── MULTI-TIMEFRAME CONTEXT (ภาพใหญ่) ────
        # ADX: วัดความแรงของเทรนด์ (0-100, >25 = trending)
        features["adx"] = self._compute_adx(df, period=14) / 100.0  # Normalize to 0-1

        # EMA 50 Distance: ราคาห่างจาก EMA 50 แค่ไหน (medium-term trend)
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        features["ema50_distance"] = (df["close"] - ema50) / (features["atr"] + 1e-10)

        # EMA 200 Distance: ราคาห่างจาก EMA 200 แค่ไหน (long-term trend)
        ema200 = df["close"].ewm(span=200, adjust=False).mean()
        features["ema200_distance"] = (df["close"] - ema200) / (features["atr"] + 1e-10)

        # ── VWAP & Trapped Sentiment (Max Pain Theory) ────
        # VWAP = Volume Weighted Average Price = "ต้นทุนเฉลี่ยของตลาด"
        vwap = self._compute_vwap(df, window=20)
        features["vwap_distance"] = (df["close"] - vwap) / (features["atr"] + 1e-10)

        # Trapped Sentiment: วัดว่า "ฝ่ายไหนกำลังเจ็บปวด"
        # > 0 = Shorts กำลังขาดทุน (ราคาสูงกว่าต้นทุนเฉลี่ย) → พร้อม squeeze ขึ้น
        # < 0 = Longs กำลังติดดอย (ราคาต่ำกว่าต้นทุนเฉลี่ย) → พร้อม dump ลง
        features["trapped_sentiment"] = self._compute_trapped_sentiment(df, vwap)

        # Pain Intensity: ความรุนแรงของความเจ็บปวด (ยิ่งห่างยิ่งเจ็บ)
        features["pain_intensity"] = abs(features["vwap_distance"]) * features["volatility"]

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

    # ── ADX Calculation (Trend Strength) ───────

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX).
        วัดความแรงของเทรนด์ (ไม่บอกทิศทาง):
        - ADX > 25 = Strong trend
        - ADX < 20 = Weak/No trend (ranging)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        return adx

    # ── VWAP Calculation (Max Pain Theory) ───────

    def _compute_vwap(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Volume Weighted Average Price (Rolling).
        ตัวแทนของ "ต้นทุนเฉลี่ยของคนทั้งตลาด"
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        volume = df["volume"].astype(float).replace(0, 1)  # Avoid div by zero

        tp_volume = typical_price * volume
        cumulative_tp_vol = tp_volume.rolling(window=window).sum()
        cumulative_vol = volume.rolling(window=window).sum()

        vwap = cumulative_tp_vol / (cumulative_vol + 1e-10)
        return vwap

    def _compute_trapped_sentiment(self, df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
        """
        Trapped Sentiment Index (Max Pain Theory).

        วัดว่า "ฝ่ายไหนกำลังเจ็บปวด" โดยพิจารณา:
        - ระยะห่างจาก VWAP (normalized by ATR)
        - Volume confirmation (volume สูง = คนเข้ามาเยอะ = trapped เยอะ)

        Returns:
            > 0: Shorts trapped (ราคาสูงกว่าต้นทุน) → expect squeeze UP
            < 0: Longs trapped (ราคาต่ำกว่าต้นทุน) → expect dump DOWN
        """
        atr = self._compute_atr(df)
        distance = (df["close"] - vwap) / (atr + 1e-10)

        # Volume confirmation: ถ้า volume สูงกว่าค่าเฉลี่ย = คนติดเยอะ
        vol_ma = df["volume"].rolling(window=20).mean()
        vol_ratio = df["volume"] / (vol_ma + 1e-10)
        vol_factor = np.clip(vol_ratio, 0.5, 2.0)  # Clamp extreme values

        # Trapped sentiment = distance * volume_factor
        # ยิ่งห่างจาก VWAP + volume สูง = ยิ่งมีคน trapped เยอะ
        trapped = distance * vol_factor

        # Smooth with EMA to reduce noise
        trapped_ema = trapped.ewm(span=5, adjust=False).mean()

        return trapped_ema

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

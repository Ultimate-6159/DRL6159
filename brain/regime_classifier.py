"""
Apex Predator — Market Regime Classifier (Layer 3: Shield)
============================================================
Detects current market state using statistical methods.
No heavy ML — lightweight rolling statistics for speed.

Regimes:
  - TRENDING: Strong directional movement
  - MEAN_REVERTING: Range-bound, oscillating
  - HIGH_VOLATILITY: Crisis / news events
  - UNCERTAIN: Mixed signals → stay out
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from config.settings import RegimeConfig, MarketRegime

logger = logging.getLogger("apex_predator.regime")


class RegimeClassifier:
    """
    Lightweight market regime detection using:
    - Volatility Ratio (current vs historical)
    - Hurst Exponent approximation (trend vs mean-reversion)
    - ADX-like trend strength

    Outputs: (MarketRegime, confidence: float)
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self._current_regime = MarketRegime.UNCERTAIN
        self._confidence = 0.0
        self._bar_counter = 0

    # ── Main Interface ──────────────────────────

    def classify(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Classify the current market regime.

        Args:
            df: OHLC DataFrame with at least `close` column.

        Returns:
            Tuple of (MarketRegime, confidence_score 0-1)
        """
        self._bar_counter += 1

        # Only re-evaluate every N bars for efficiency
        # BUT always evaluate on first call or if UNCERTAIN
        if (self._bar_counter % self.config.update_frequency != 0
                and self._current_regime != MarketRegime.UNCERTAIN
                and self._bar_counter > 1):
            return self._current_regime, self._confidence

        if len(df) < self.config.trend_window + 10:
            logger.warning("Not enough data for regime classification: %d bars (need %d)",
                          len(df), self.config.trend_window + 10)
            return MarketRegime.UNCERTAIN, 0.0

        close = df["close"].values

        # ── Compute indicators ──────────────────
        vol_ratio = self._volatility_ratio(close)
        hurst = self._hurst_exponent(close)
        trend_strength = self._trend_strength(close)

        # ── Decision Logic ──────────────────────
        regime, confidence = self._decide_regime(vol_ratio, hurst, trend_strength)

        # Log raw values for debugging
        logger.debug(
            "Regime indicators: vol_ratio=%.3f hurst=%.3f trend=%.3f -> %s (conf=%.2f)",
            vol_ratio, hurst, trend_strength, regime.value, confidence
        )

        # Only switch regime if confidence exceeds threshold
        # OR if this is the first classification (escape from UNCERTAIN)
        if confidence >= self.config.regime_change_threshold or self._bar_counter <= 2:
            if regime != self._current_regime:
                logger.info(
                    "Regime changed: %s → %s (confidence=%.2f) | "
                    "vol_ratio=%.2f hurst=%.3f trend=%.2f",
                    self._current_regime.value, regime.value, confidence,
                    vol_ratio, hurst, trend_strength,
                )
            self._current_regime = regime
            self._confidence = confidence

        return self._current_regime, self._confidence

    def get_regime(self) -> Tuple[MarketRegime, float]:
        """Get cached regime without recalculation."""
        return self._current_regime, self._confidence

    def get_regime_one_hot(self) -> np.ndarray:
        """Get one-hot encoding of current regime for DRL observation."""
        regimes = list(MarketRegime)
        one_hot = np.zeros(len(regimes), dtype=np.float32)
        idx = regimes.index(self._current_regime)
        one_hot[idx] = 1.0
        return one_hot

    # ── Statistical Indicators ──────────────────

    def _volatility_ratio(self, close: np.ndarray) -> float:
        """
        Ratio of recent volatility to historical volatility.
        > 1.5 = high vol, < 0.8 = low vol (mean reverting).
        """
        returns = np.diff(np.log(close + 1e-10))
        short_window = min(self.config.volatility_window, len(returns))
        long_window = min(self.config.trend_window, len(returns))

        recent_vol = np.std(returns[-short_window:])
        historical_vol = np.std(returns[-long_window:])

        if historical_vol < 1e-10:
            return 1.0
        return recent_vol / historical_vol

    def _hurst_exponent(self, close: np.ndarray) -> float:
        """
        Simplified Hurst Exponent approximation:
        H > 0.5 → Trending (persistent)
        H ≈ 0.5 → Random walk
        H < 0.5 → Mean-reverting (anti-persistent)
        """
        window = min(self.config.hurst_window, len(close) - 1)
        series = close[-window:]

        if len(series) < 20:
            return 0.5

        lags = range(2, min(20, len(series) // 4))
        tau = []
        for lag in lags:
            diffs = series[lag:] - series[:-lag]
            std = np.std(diffs)
            if std > 0:
                tau.append(std)
            else:
                tau.append(1e-10)

        if len(tau) < 2:
            return 0.5

        # Linear regression on log-log plot
        log_lags = np.log(list(lags[:len(tau)]))
        log_tau = np.log(tau)

        try:
            coeffs = np.polyfit(log_lags, log_tau, 1)
            hurst = coeffs[0]  # Slope ≈ Hurst exponent
            return float(np.clip(hurst, 0.0, 1.0))
        except (np.linalg.LinAlgError, ValueError):
            return 0.5

    def _trend_strength(self, close: np.ndarray) -> float:
        """
        ADX-like trend strength metric.
        Returns 0-1, where > 0.6 is strong trend.
        """
        window = min(self.config.trend_window, len(close) - 1)
        series = close[-window:]

        if len(series) < 10:
            return 0.0

        # Linear regression → R² as trend strength
        x = np.arange(len(series))
        try:
            coeffs = np.polyfit(x, series, 1)
            trend_line = np.polyval(coeffs, x)
            ss_res = np.sum((series - trend_line) ** 2)
            ss_tot = np.sum((series - np.mean(series)) ** 2)

            if ss_tot < 1e-10:
                return 0.0

            r_squared = 1 - (ss_res / ss_tot)
            return float(np.clip(r_squared, 0.0, 1.0))
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    # ── Regime Decision ─────────────────────────

    def _decide_regime(
        self, vol_ratio: float, hurst: float, trend_strength: float
    ) -> Tuple[MarketRegime, float]:
        """
        Combine indicators into a regime classification.

        Returns:
            (regime, confidence)
        """
        scores = {
            MarketRegime.HIGH_VOLATILITY: 0.0,
            MarketRegime.TRENDING: 0.0,
            MarketRegime.MEAN_REVERTING: 0.0,
            MarketRegime.UNCERTAIN: 0.2,  # Base score for uncertain
        }

        # ── High Volatility Detection ───────────
        if vol_ratio > self.config.vol_ratio_threshold:
            scores[MarketRegime.HIGH_VOLATILITY] += 0.5
            scores[MarketRegime.HIGH_VOLATILITY] += min(
                (vol_ratio - self.config.vol_ratio_threshold) * 0.3, 0.3
            )

        # ── Trending Detection ──────────────────
        if hurst > 0.55 and trend_strength > self.config.trend_strength_threshold:
            scores[MarketRegime.TRENDING] += 0.4
            scores[MarketRegime.TRENDING] += trend_strength * 0.3
            scores[MarketRegime.TRENDING] += (hurst - 0.5) * 0.3

        # ── Mean Reversion Detection ────────────
        if hurst < 0.45:
            scores[MarketRegime.MEAN_REVERTING] += 0.4
            scores[MarketRegime.MEAN_REVERTING] += (0.5 - hurst) * 0.4

        if vol_ratio < 0.8:
            scores[MarketRegime.MEAN_REVERTING] += 0.2

        # ── Select strongest regime ─────────────
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        # Normalize confidence to [0, 1]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0

        return best_regime, float(np.clip(confidence, 0.0, 1.0))

    def reset(self):
        """Reset regime state."""
        self._current_regime = MarketRegime.UNCERTAIN
        self._confidence = 0.0
        self._bar_counter = 0

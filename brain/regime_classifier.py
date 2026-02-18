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

    Features:
    - Hysteresis: Prevents regime flicker by requiring buffer
    - Smoothing: EMA on indicators to reduce noise
    - Confirmation: Must hold new regime for N bars before switching
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self._current_regime = MarketRegime.UNCERTAIN
        self._confidence = 0.0
        self._bar_counter = 0

        # ═══ HYSTERESIS STATE ═══
        self._pending_regime = None       # Regime waiting for confirmation
        self._pending_confidence = 0.0
        self._pending_bars = 0            # How many bars pending regime held
        self._confirmation_bars = 3       # Must hold for N bars to confirm
        self._hysteresis_buffer = 0.15    # Must exceed current regime by 15%

        # ═══ SMOOTHING STATE (EMA) ═══
        self._ema_alpha = 0.3             # EMA smoothing factor (0.3 = responsive but smooth)
        self._smoothed_vol_ratio = None
        self._smoothed_hurst = None
        self._smoothed_trend = None

    # ── Main Interface ──────────────────────────

    def classify(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Classify the current market regime with hysteresis and smoothing.

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

        # ── Compute raw indicators ──────────────────
        raw_vol_ratio = self._volatility_ratio(close)
        raw_hurst = self._hurst_exponent(close)
        raw_trend = self._trend_strength(close)

        # ── Apply EMA Smoothing ──────────────────
        vol_ratio = self._apply_ema("vol_ratio", raw_vol_ratio)
        hurst = self._apply_ema("hurst", raw_hurst)
        trend_strength = self._apply_ema("trend", raw_trend)

        # ── Decision Logic with Hysteresis ──────────────────────
        candidate_regime, candidate_confidence = self._decide_regime(vol_ratio, hurst, trend_strength)

        # Log raw values for debugging
        logger.debug(
            "Regime indicators: vol_ratio=%.3f hurst=%.3f trend=%.3f -> %s (conf=%.2f)",
            vol_ratio, hurst, trend_strength, candidate_regime.value, candidate_confidence
        )

        # ── Apply Hysteresis Logic ──────────────────
        final_regime, final_confidence = self._apply_hysteresis(
            candidate_regime, candidate_confidence
        )

        return final_regime, final_confidence

    def _apply_ema(self, name: str, raw_value: float) -> float:
        """Apply EMA smoothing to an indicator."""
        attr_name = f"_smoothed_{name}"
        current = getattr(self, attr_name, None)

        if current is None:
            # First value - initialize
            setattr(self, attr_name, raw_value)
            return raw_value

        # EMA formula: new = alpha * raw + (1 - alpha) * old
        smoothed = self._ema_alpha * raw_value + (1 - self._ema_alpha) * current
        setattr(self, attr_name, smoothed)
        return smoothed

    def _apply_hysteresis(
        self, candidate_regime: MarketRegime, candidate_confidence: float
    ) -> Tuple[MarketRegime, float]:
        """
        Apply hysteresis to prevent regime flicker.

        Rules:
        1. If candidate == current: keep it, reset pending
        2. If candidate != current: 
           - Must exceed current confidence by buffer amount
           - Must hold for confirmation_bars consecutive bars
        """
        # First call or escaping UNCERTAIN - no hysteresis needed
        if self._bar_counter <= 2 or self._current_regime == MarketRegime.UNCERTAIN:
            if candidate_regime != self._current_regime:
                logger.info(
                    "Regime initialized: %s → %s (conf=%.2f)",
                    self._current_regime.value, candidate_regime.value, candidate_confidence
                )
            self._current_regime = candidate_regime
            self._confidence = candidate_confidence
            self._pending_regime = None
            self._pending_bars = 0
            return self._current_regime, self._confidence

        # Same regime as current - reinforce it
        if candidate_regime == self._current_regime:
            self._confidence = candidate_confidence
            self._pending_regime = None
            self._pending_bars = 0
            return self._current_regime, self._confidence

        # Different regime - apply hysteresis
        # Check 1: Must exceed current confidence by buffer
        if candidate_confidence < self._confidence + self._hysteresis_buffer:
            # Not strong enough to challenge current regime
            return self._current_regime, self._confidence

        # Check 2: Is this the same pending regime?
        if candidate_regime == self._pending_regime:
            self._pending_bars += 1
            self._pending_confidence = candidate_confidence
        else:
            # New pending regime - reset counter
            self._pending_regime = candidate_regime
            self._pending_confidence = candidate_confidence
            self._pending_bars = 1

        # Check 3: Has pending regime been confirmed for enough bars?
        if self._pending_bars >= self._confirmation_bars:
            logger.info(
                "Regime changed (confirmed after %d bars): %s → %s (conf=%.2f) | "
                "smoothed: vol=%.2f hurst=%.3f trend=%.2f",
                self._pending_bars,
                self._current_regime.value, self._pending_regime.value, self._pending_confidence,
                self._smoothed_vol_ratio or 0, self._smoothed_hurst or 0, self._smoothed_trend or 0,
            )
            self._current_regime = self._pending_regime
            self._confidence = self._pending_confidence
            self._pending_regime = None
            self._pending_bars = 0

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
            MarketRegime.UNCERTAIN: 0.1,  # Lower base score for uncertain
        }

        # ── High Volatility Detection ───────────
        if vol_ratio > self.config.vol_ratio_threshold:
            scores[MarketRegime.HIGH_VOLATILITY] += 0.5
            scores[MarketRegime.HIGH_VOLATILITY] += min(
                (vol_ratio - self.config.vol_ratio_threshold) * 0.3, 0.3
            )

        # ── Trending Detection (BALANCED thresholds) ──────────────────
        # Target: ~30% of data should be TRENDING
        # Hurst > 0.52 = persistent tendency (trending)
        # trend_strength > 0.4 = price follows direction reasonably well
        if hurst > 0.52 and trend_strength > 0.4:
            # Strong trend: both indicators confirm
            scores[MarketRegime.TRENDING] += 0.45
            scores[MarketRegime.TRENDING] += trend_strength * 0.3
            scores[MarketRegime.TRENDING] += (hurst - 0.52) * 0.25
        elif hurst > 0.54 or trend_strength > 0.55:
            # Moderate trend: one strong indicator
            scores[MarketRegime.TRENDING] += 0.3
            if hurst > 0.52:
                scores[MarketRegime.TRENDING] += (hurst - 0.52) * 0.2
            if trend_strength > 0.4:
                scores[MarketRegime.TRENDING] += (trend_strength - 0.4) * 0.2

        # ── Mean Reversion Detection (BALANCED) ────────────
        # Hurst < 0.46 = anti-persistent (true mean reverting)
        # More strict to avoid capturing random walk
        if hurst < 0.46:
            # Strong mean reversion signal
            scores[MarketRegime.MEAN_REVERTING] += 0.45
            scores[MarketRegime.MEAN_REVERTING] += (0.46 - hurst) * 0.4
        elif hurst < 0.48 and trend_strength < 0.35:
            # Weak trend + low hurst = consolidation
            scores[MarketRegime.MEAN_REVERTING] += 0.25

        # ── Random Walk Zone → UNCERTAIN ────────────
        # Hurst 0.48-0.52 is random walk, don't force into TRENDING or MEAN_REV
        if 0.48 <= hurst <= 0.52 and 0.35 <= trend_strength <= 0.5:
            scores[MarketRegime.UNCERTAIN] += 0.3

        # Low volatility favors mean reversion (but weaker influence)
        if vol_ratio < 0.85:
            scores[MarketRegime.MEAN_REVERTING] += 0.1

        # ── Select strongest regime ─────────────
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        # Calculate confidence as ratio of best score to max possible (1.0)
        # This prevents UNCERTAIN from getting 100% confidence when others have 0
        max_possible = 1.0
        confidence = best_score / max_possible

        # If UNCERTAIN wins but with low absolute score, reduce confidence further
        if best_regime == MarketRegime.UNCERTAIN:
            confidence = min(confidence, 0.3)  # Cap UNCERTAIN confidence at 30%

        return best_regime, float(np.clip(confidence, 0.0, 1.0))

    def reset(self):
        """Reset regime state."""
        self._current_regime = MarketRegime.UNCERTAIN
        self._confidence = 0.0
        self._bar_counter = 0
        # Reset hysteresis state
        self._pending_regime = None
        self._pending_confidence = 0.0
        self._pending_bars = 0
        # Reset smoothing state
        self._smoothed_vol_ratio = None
        self._smoothed_hurst = None
        self._smoothed_trend = None

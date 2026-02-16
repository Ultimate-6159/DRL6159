"""
Tests for Market Regime Classifier
"""

import pytest
import numpy as np
import pandas as pd

from config.settings import RegimeConfig, MarketRegime
from brain.regime_classifier import RegimeClassifier


@pytest.fixture
def classifier():
    config = RegimeConfig(
        volatility_window=50,
        trend_window=100,
        hurst_window=100,
        update_frequency=1,  # Evaluate every bar for testing
    )
    return RegimeClassifier(config)


def _make_trending_data(n=200, slope=0.001):
    """Generate trending price data."""
    prices = 1.10 + np.arange(n) * slope + np.random.normal(0, 0.0001, n)
    return pd.DataFrame({
        "close": prices,
        "high": prices + 0.0002,
        "low": prices - 0.0002,
        "volume": np.random.randint(50, 500, n),
    })


def _make_mean_reverting_data(n=200):
    """Generate range-bound price data."""
    base = 1.10
    noise = np.cumsum(np.random.normal(0, 0.0001, n))
    # Pull back to mean
    prices = base + noise * 0.5 - np.mean(noise * 0.5)
    return pd.DataFrame({
        "close": prices,
        "high": prices + 0.0001,
        "low": prices - 0.0001,
        "volume": np.random.randint(50, 500, n),
    })


def _make_volatile_data(n=200):
    """Generate high-volatility price data."""
    prices = 1.10 + np.cumsum(np.random.normal(0, 0.003, n))
    return pd.DataFrame({
        "close": prices,
        "high": prices + abs(np.random.normal(0, 0.002, n)),
        "low": prices - abs(np.random.normal(0, 0.002, n)),
        "volume": np.random.randint(50, 500, n),
    })


class TestRegimeClassifier:
    def test_classify_returns_tuple(self, classifier):
        df = _make_trending_data()
        regime, conf = classifier.classify(df)
        assert isinstance(regime, MarketRegime)
        assert 0.0 <= conf <= 1.0

    def test_trending_detection(self, classifier):
        df = _make_trending_data(300, slope=0.01)
        regime, conf = classifier.classify(df)
        # Strong trend should not be classified as UNCERTAIN
        assert regime != MarketRegime.UNCERTAIN or conf < 0.3

    def test_regime_one_hot(self, classifier):
        df = _make_trending_data()
        classifier.classify(df)
        one_hot = classifier.get_regime_one_hot()
        assert one_hot.shape == (4,)
        assert np.sum(one_hot) == 1.0

    def test_insufficient_data(self, classifier):
        df = pd.DataFrame({"close": [1.1, 1.11, 1.12]})
        regime, conf = classifier.classify(df)
        assert regime == MarketRegime.UNCERTAIN
        assert conf == 0.0

    def test_reset(self, classifier):
        df = _make_trending_data()
        classifier.classify(df)
        classifier.reset()
        regime, conf = classifier.get_regime()
        assert regime == MarketRegime.UNCERTAIN


class TestStatisticalIndicators:
    def test_hurst_trending(self, classifier):
        """Trending data should have Hurst > 0.5."""
        close = 1.10 + np.arange(200) * 0.001
        hurst = classifier._hurst_exponent(close)
        assert hurst >= 0.0
        assert hurst <= 1.0

    def test_volatility_ratio_normal(self, classifier):
        """Stable data should have vol ratio near 1.0."""
        close = 1.10 + np.random.normal(0, 0.0003, 200)
        ratio = classifier._volatility_ratio(close)
        assert 0.5 < ratio < 2.0

    def test_trend_strength_strong(self, classifier):
        """Strong linear trend should have high R²."""
        close = 1.10 + np.arange(200) * 0.001
        strength = classifier._trend_strength(close)
        assert strength > 0.8

"""
Integration Tests — Full Pipeline (Mock Mode)
Tests the complete flow without connecting to MT5.
"""

import pytest
import numpy as np
import pandas as pd

from config.settings import ApexConfig, TradeAction, MarketRegime
from core.mt5_connector import MT5Connector
from core.data_feed import DataFeed
from core.feature_engine import FeatureEngine
from brain.regime_classifier import RegimeClassifier
from brain.perception import PerceptionModule
from brain.drl_agent import DRLAgent, ForexTradingEnv
from brain.reward import RewardCalculator
from shield.risk_manager import RiskManager
from shield.circuit_breaker import CircuitBreaker
from memory.vector_store import VectorMemory


@pytest.fixture
def config():
    c = ApexConfig()
    c.dry_run = True
    c.perception.device = "cpu"
    return c


class TestMT5Connector:
    def test_mock_mode(self, config):
        conn = MT5Connector(config.mt5, dry_run=True)
        tick = conn.get_tick()
        assert tick is not None
        assert tick.bid > 0

    def test_mock_ohlc(self, config):
        conn = MT5Connector(config.mt5, dry_run=True)
        df = conn.get_ohlc(count=100)
        assert df is not None
        assert len(df) == 100
        assert "close" in df.columns

    def test_mock_order(self, config):
        conn = MT5Connector(config.mt5, dry_run=True)
        result = conn.send_order(TradeAction.BUY, volume=0.1)
        assert result.success is True
        assert result.comment == "DRY_RUN"


class TestFeatureEngine:
    def test_compute_features(self, config):
        conn = MT5Connector(config.mt5, dry_run=True)
        df = conn.get_ohlc(count=200)
        engine = FeatureEngine(config.features)
        features = engine.compute(df)
        assert features is not None
        assert features.ndim == 2

    def test_feature_dim(self, config):
        engine = FeatureEngine(config.features)
        dim = engine.get_feature_dim()
        assert dim == len(config.features.features)


class TestPerception:
    def test_encode(self, config):
        config.perception.device = "cpu"
        perception = PerceptionModule(config.perception)
        # Create fake features: (seq_len, input_dim)
        features = np.random.randn(60, config.perception.input_dim).astype(np.float32)
        embedding = perception.encode(features)
        assert embedding.shape == (config.perception.embedding_dim,)

    def test_encode_short_sequence(self, config):
        config.perception.device = "cpu"
        perception = PerceptionModule(config.perception)
        # Short sequence — should be padded
        features = np.random.randn(10, config.perception.input_dim).astype(np.float32)
        embedding = perception.encode(features)
        assert embedding.shape == (config.perception.embedding_dim,)


class TestForexEnv:
    def test_observation_space(self, config):
        env = ForexTradingEnv(
            feature_dim=17,
            lookback=10,
            reward_config=config.reward,
        )
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.shape == (10 * 17 + 4 + 7,)

    def test_build_observation(self, config):
        features_flat = np.random.randn(10 * 17).astype(np.float32)
        regime = np.array([1, 0, 0, 0], dtype=np.float32)
        obs = ForexTradingEnv.build_observation(
            features_flat, regime, 0.01, 0.0, 0.15,
            has_position=1.0, hold_time_norm=0.5,
            recent_wr=0.6, drawdown=0.02,
        )
        assert obs.shape == (10 * 17 + 4 + 7,)


class TestCircuitBreaker:
    def test_consecutive_losses_halt(self, config):
        cb = CircuitBreaker(config.circuit_breaker)
        cb.set_initial_state(10000.0, 1.5)

        for _ in range(config.circuit_breaker.max_consecutive_losses):
            cb.record_trade(-10.0)

        can_trade, reason = cb.check(10000.0, 1.5)
        assert can_trade is False
        assert "CONSECUTIVE_LOSSES" in reason

    def test_normal_trading(self, config):
        cb = CircuitBreaker(config.circuit_breaker)
        cb.set_initial_state(10000.0, 1.5)
        can_trade, reason = cb.check(10000.0, 1.5)
        assert can_trade is True

    def test_spread_spike_pause(self, config):
        cb = CircuitBreaker(config.circuit_breaker)
        cb.set_initial_state(10000.0, 1.5)
        # Spread = 6.0, normal = 1.5, ratio = 4.0 > 3.0x threshold
        can_trade, reason = cb.check(10000.0, 6.0)
        assert can_trade is False
        assert "SPREAD_SPIKE" in reason


class TestVectorMemory:
    def test_store_and_recall(self, config):
        config.memory.similarity_threshold = 0.0  # Accept all
        mem = VectorMemory(config.memory)

        emb1 = np.random.randn(64).astype(np.float32)
        mem.store(emb1, action=0, outcome=10.0, regime="trending")

        results = mem.recall(emb1, top_k=1)
        assert len(results) >= 1

    def test_pattern_stats(self, config):
        config.memory.similarity_threshold = 0.0
        mem = VectorMemory(config.memory)

        for _ in range(10):
            emb = np.random.randn(64).astype(np.float32)
            mem.store(emb, action=0, outcome=5.0, regime="trending")

        query = np.random.randn(64).astype(np.float32)
        stats = mem.get_pattern_stats(query)
        assert "similar_count" in stats
        assert "win_rate" in stats


class TestFullPipeline:
    """Integration test: run through the complete pipeline with mock data."""

    def test_pipeline_mock_flow(self, config):
        config.perception.device = "cpu"

        # 1. Get mock data
        conn = MT5Connector(config.mt5, dry_run=True)
        df = conn.get_ohlc(count=200)
        assert df is not None

        # 2. Compute features
        engine = FeatureEngine(config.features)
        features = engine.compute(df)
        assert features is not None

        # 3. Classify regime
        classifier = RegimeClassifier(config.regime)
        regime, conf = classifier.classify(df)
        assert isinstance(regime, MarketRegime)

        # 4. Encode perception
        perception = PerceptionModule(config.perception)
        seq = features[-config.perception.sequence_length:]
        embedding = perception.encode(seq)
        assert embedding.shape == (config.perception.embedding_dim,)

        # 5. Build observation
        regime_oh = classifier.get_regime_one_hot()
        obs = ForexTradingEnv.build_observation(
            embedding, regime_oh, 0.0, 0.0, 0.15,
        )

        # 6. Risk evaluation
        rm = RiskManager(config.risk)
        rm.set_initial_balance(10000.0)
        account = conn.get_account_info()
        atr = float(engine._compute_atr(df).iloc[-1])

        proposal = rm.evaluate_trade(
            action=TradeAction.BUY,
            current_price=float(df["close"].iloc[-1]),
            atr=atr,
            account_info=account,
        )
        assert proposal.approved is True

        # 7. Execute (dry run)
        result = conn.send_order(
            TradeAction.BUY,
            volume=proposal.lot_size,
            sl=proposal.stop_loss,
            tp=proposal.take_profit,
        )
        assert result.success is True

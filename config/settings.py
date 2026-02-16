"""
Apex Predator DRL Trading System — Central Configuration
=========================================================
All system parameters in one place. No magic numbers scattered across modules.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class MarketRegime(Enum):
    """Market state classification."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    UNCERTAIN = "uncertain"


class TradeAction(Enum):
    """Possible trade actions for the DRL agent."""
    BUY = 0
    SELL = 1
    HOLD = 2


# ──────────────────────────────────────────────
# MT5 Configuration
# ──────────────────────────────────────────────

@dataclass
class MT5Config:
    """MetaTrader 5 connection settings."""
    login: int = 415146568                       # MT5 account number
    password: str = "Ultimate@6159"                   # MT5 account password
    server: str = "Exness-MT5Trial14"                     # Broker server name
    path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    symbol: str = "XAUUSDm"
    timeframe: str = "M1"               # M1, M5, M15, H1 etc.
    magic_number: int = 616159           # Unique EA identifier
    deviation: int = 20                  # Max slippage in points
    fill_type: str = "IOC"              # IOC, FOK, RETURN


# ──────────────────────────────────────────────
# Feature Engineering Configuration
# ──────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Parameters for feature construction."""
    lookback_window: int = 500           # Bars of history for LSTM input
    atr_period: int = 14                 # ATR calculation period
    spread_ma_period: int = 20           # Spread moving average period
    volume_profile_bins: int = 50        # Number of bins for volume profile
    z_score_window: int = 100            # Rolling z-score normalization window
    log_returns: bool = True             # Use log returns vs simple returns
    features: List[str] = field(default_factory=lambda: [
        "close", "high", "low", "volume",
        "spread", "atr", "returns", "z_score",
        "volatility", "momentum", "rsi_raw",
        "ema_cross", "macd_signal", "bb_position",
        "price_change_3", "high_low_ratio", "body_ratio",
    ])


# ──────────────────────────────────────────────
# Regime Classifier Configuration
# ──────────────────────────────────────────────

@dataclass
class RegimeConfig:
    """Parameters for market regime detection."""
    volatility_window: int = 50          # Window for volatility calculation
    trend_window: int = 100              # Window for trend detection
    hurst_window: int = 100              # Window for Hurst exponent
    regime_change_threshold: float = 0.3 # Min confidence for regime switch
    vol_ratio_threshold: float = 1.5     # High vol if ratio > this
    trend_strength_threshold: float = 0.6 # Strong trend if > this
    update_frequency: int = 10           # Re-evaluate regime every N bars


# ──────────────────────────────────────────────
# Perception (LSTM) Configuration
# ──────────────────────────────────────────────

@dataclass
class PerceptionConfig:
    """LSTM / Transformer neural network settings."""
    input_dim: int = 17                  # Number of input features (11 base + 6 scalping)
    hidden_dim: int = 128                # LSTM hidden state size
    num_layers: int = 2                  # Number of LSTM layers
    dropout: float = 0.2                 # Dropout rate
    embedding_dim: int = 64             # Output embedding dimension
    use_attention: bool = True           # Add attention layer on top
    sequence_length: int = 60            # Input sequence length for LSTM
    device: str = "auto"                 # "cpu", "cuda", or "auto"


# ──────────────────────────────────────────────
# DRL Agent Configuration
# ──────────────────────────────────────────────

@dataclass
class DRLConfig:
    """PPO / SAC reinforcement learning settings."""
    algorithm: str = "PPO"               # "PPO" or "SAC"
    learning_rate: float = 1e-4          # Slower LR → more stable
    gamma: float = 0.99                  # Discount factor
    gae_lambda: float = 0.95            # GAE lambda
    clip_range: float = 0.2             # PPO clip range
    n_steps: int = 4096                  # More experience per update
    batch_size: int = 128                # Larger batch → lower variance
    n_epochs: int = 15                   # More learning per update
    ent_coef: float = 0.005             # More exploration early on
    vf_coef: float = 0.5                # Value function coefficient
    max_grad_norm: float = 0.5          # Gradient clipping
    total_timesteps: int = 2_000_000     # Train longer for better convergence
    model_save_path: str = "models/"     # Model checkpoint directory
    tensorboard_log: str = "logs/tb/"    # TensorBoard log directory


# ──────────────────────────────────────────────
# Reward Configuration
# ──────────────────────────────────────────────

@dataclass
class RewardConfig:
    """Reward function parameters (Sharpe-based)."""
    profit_reward: float = 1.0           # Reward per pip profit
    loss_penalty: float = -2.0           # Penalty per pip loss
    hold_penalty: float = -0.05          # Light hold penalty (scalping already short)
    max_hold_steps: int = 60             # Force close faster (scalping)
    drawdown_penalty: float = -3.0       # Moderate drawdown penalty
    sharpe_window: int = 50              # Rolling window for Sharpe calc
    risk_free_rate: float = 0.0          # Risk-free rate for Sharpe
    optimal_close_bonus: float = 0.5     # Bonus for closing at good R:R


# ──────────────────────────────────────────────
# Risk Management Configuration
# ──────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Hard-coded risk management parameters (NO AI — pure math)."""
    max_risk_per_trade: float = 0.01     # Max 1% risk per trade
    max_daily_loss: float = 0.05         # Max 5% daily drawdown
    max_total_drawdown: float = 0.15     # Max 15% total drawdown -> halt
    max_concurrent_trades: int = 3       # Max open positions
    max_lot_size: float = 1.0            # Absolute max lot size
    min_lot_size: float = 0.01           # Minimum lot size
    atr_multiplier: float = 0.8          # Tight SL = ATR * 0.8 (scalping)
    tp_ratio: float = 1.2               # TP = SL * 1.2 (easier to hit → more wins)


# ──────────────────────────────────────────────
# Circuit Breaker Configuration
# ──────────────────────────────────────────────

@dataclass
class CircuitBreakerConfig:
    """Emergency stop parameters."""
    max_consecutive_losses: int = 5      # Halt after N consecutive losses
    cooldown_minutes: int = 60           # Cooldown period after halt
    drawdown_halt_pct: float = 0.10      # Halt if drawdown reaches 10%
    max_daily_trades: int = 200          # High-frequency: allow many trades
    spread_spike_multiplier: float = 3.0 # Halt if spread > 3x normal


# ──────────────────────────────────────────────
# Evolution / Self-Learning Configuration
# ──────────────────────────────────────────────

@dataclass
class EvolutionConfig:
    """Online learning and walk-forward optimization."""
    retrain_interval: int = 100          # Retrain after N completed trades
    replay_buffer_size: int = 50_000     # Max experiences in replay buffer
    min_retrain_samples: int = 500       # Min samples before retraining
    walk_forward_window: int = 5_000     # Bars for walk-forward testing
    walk_forward_step: int = 1_000       # Step forward by N bars
    performance_threshold: float = 0.3   # Min Sharpe to keep model
    model_history_count: int = 10        # Keep N model checkpoints
    checkpoint_interval_hours: int = 4   # Save model every N hours


# ──────────────────────────────────────────────
# Vector Memory Configuration
# ──────────────────────────────────────────────

@dataclass
class MemoryConfig:
    """Vector database for pattern memory."""
    embedding_dim: int = 64              # Must match PerceptionConfig
    max_memories: int = 100_000          # Max stored patterns
    recall_top_k: int = 5               # Top-K similar patterns to retrieve
    similarity_threshold: float = 0.7    # Min cosine similarity for recall
    persist_path: str = "data/memory/"   # Disk persistence path


# ──────────────────────────────────────────────
# Master Configuration
# ──────────────────────────────────────────────

@dataclass
class ApexConfig:
    """
    Master configuration aggregating all sub-configs.
    Instantiate this once and pass it to all modules.
    """
    mt5: MT5Config = field(default_factory=MT5Config)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    drl: DRLConfig = field(default_factory=DRLConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Global
    dry_run: bool = True                 # True = mock mode, no real orders
    log_level: str = "INFO"              # DEBUG, INFO, WARNING, ERROR
    log_dir: str = "logs/"               # Log file directory
    data_dir: str = "data/"              # Data persistence directory

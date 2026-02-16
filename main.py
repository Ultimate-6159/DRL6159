"""
Apex Predator — Main Orchestrator
====================================
Integrates all 4 layers into the main trading loop:
DataFeed → FeatureEngine → RegimeClassifier → PerceptionModule
→ DRLAgent → RiskManager → MT5Connector → OnlineTrainer

Supports: --dry-run mode, graceful shutdown, heartbeat logging
"""

import sys
import os
import time
import signal
import argparse
import logging
from datetime import datetime
from typing import Optional

import numpy as np

# ── Internal Imports ────────────────────────────
from config.settings import ApexConfig, TradeAction, MarketRegime
from utils.logger import setup_logger, get_trade_logger
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
from evolution.online_trainer import OnlineTrainer
from evolution.walk_forward import WalkForwardOptimizer


class ApexPredator:
    """
    The Apex Predator — a 4-layer AI-driven Forex scalping system.

    Main loop:
    1. Fetch data → compute features → detect regime
    2. Encode market state via LSTM → recall similar patterns
    3. DRL agent decides action → risk manager evaluates
    4. Execute order → collect experience → evolve

    Runs indefinitely until stopped or circuit breaker trips.
    """

    def __init__(self, config: ApexConfig):
        self.config = config
        self._running = False
        self._loop_count = 0
        self._trade_history: list = []

        # ── Setup Logger ────────────────────────
        self.logger = setup_logger(
            log_dir=config.log_dir,
            level=config.log_level,
        )
        self.trade_logger = get_trade_logger()

        # ── Layer 0: Core ───────────────────────
        self.connector = MT5Connector(config.mt5, dry_run=config.dry_run)
        self.data_feed = DataFeed(self.connector, config.features)
        self.feature_engine = FeatureEngine(config.features)

        # ── Layer 1: Perception ─────────────────
        self.perception = PerceptionModule(config.perception)

        # ── Layer 2: Brain ──────────────────────
        self.regime_classifier = RegimeClassifier(config.regime)
        self.drl_agent = DRLAgent(config.drl, config.perception, config.reward)
        self.reward_calc = RewardCalculator(config.reward)

        # ── Layer 3: Shield ─────────────────────
        self.risk_manager = RiskManager(config.risk)
        self.circuit_breaker = CircuitBreaker(config.circuit_breaker)

        # ── Layer 4: Evolution ──────────────────
        self.online_trainer = OnlineTrainer(config.evolution)
        self.walk_forward = WalkForwardOptimizer(config.evolution)

        # ── Memory ──────────────────────────────
        self.memory = VectorMemory(config.memory)

        # ── State Tracking ──────────────────────
        self._initial_equity = 0.0
        self._last_observation: Optional[np.ndarray] = None

    # ═══════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════

    def start(self):
        """Initialize all modules and start the trading loop."""
        self.logger.info("=" * 60)
        self.logger.info("   APEX PREDATOR — Starting Up")
        self.logger.info("   Mode: %s", "DRY RUN" if self.config.dry_run else "⚡ LIVE")
        self.logger.info("   Symbol: %s | TF: %s",
                         self.config.mt5.symbol, self.config.mt5.timeframe)
        self.logger.info("=" * 60)

        # ── Connect to MT5 ─────────────────────
        connected = self.connector.connect()
        if not connected and not self.config.dry_run:
            self.logger.error("Cannot connect to MT5 — aborting")
            return

        # ── Initialize Systems ──────────────────
        if not self._initialize():
            self.logger.error("Initialization failed — aborting")
            self.shutdown()
            return

        # ── Signal handlers ─────────────────────
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # ── Main Loop ───────────────────────────
        self._running = True
        self.logger.info("System armed — entering main loop")
        self._main_loop()

    def _initialize(self) -> bool:
        """Initialize all subsystems."""
        # Load data
        if not self.data_feed.initialize():
            self.logger.warning("DataFeed init with mock data")

        # Get initial account state
        account = self.connector.get_account_info()
        self._initial_equity = account.get("equity", 10000.0)
        self.risk_manager.set_initial_balance(self._initial_equity)
        self.circuit_breaker.set_initial_state(
            equity=self._initial_equity,
            normal_spread=self.data_feed.get_spread(),
        )

        # Initialize DRL agent
        self.drl_agent.initialize()

        # Load saved memory & model
        self.memory.load()
        try:
            self.drl_agent.load()
            self.logger.info("Loaded existing DRL model")
        except Exception:
            self.logger.info("No existing model — starting fresh")

        self.logger.info(
            "Initialization complete | Equity: $%.2f | Memory: %d patterns",
            self._initial_equity, self.memory.size(),
        )
        return True

    def shutdown(self):
        """Graceful shutdown — save state and disconnect."""
        self._running = False
        self.logger.info("Shutting down Apex Predator...")

        # Close all positions
        self.connector.close_all_positions()

        # Save state
        self.drl_agent.save()
        self.memory.save()

        # Disconnect
        self.connector.disconnect()

        self.logger.info("Shutdown complete. Goodbye.")

    def _signal_handler(self, signum, frame):
        """Handle OS signals for graceful shutdown."""
        self.logger.info("Signal %d received — initiating shutdown", signum)
        self.shutdown()
        sys.exit(0)

    # ═══════════════════════════════════════════
    # MAIN TRADING LOOP
    # ═══════════════════════════════════════════

    def _main_loop(self):
        """
        The beating heart of Apex Predator.
        Each iteration:
        1. Update data
        2. Compute features & detect regime
        3. Encode state & build observation
        4. Agent decides → Risk evaluates → Execute
        5. Learn from result
        """
        while self._running:
            try:
                self._loop_count += 1

                # ── Step 1: Update Data ─────────
                new_data = self.data_feed.update()
                if not new_data and self._loop_count > 1:
                    time.sleep(1)  # Wait for new bar
                    continue

                if not self.data_feed.is_ready():
                    time.sleep(1)
                    continue

                # ── Step 2: Circuit Breaker Check
                account = self.connector.get_account_info()
                equity = account.get("equity", self._initial_equity)
                spread = self.data_feed.get_spread()

                can_trade, cb_reason = self.circuit_breaker.check(equity, spread)
                if not can_trade:
                    self.logger.warning("Circuit Breaker: %s", cb_reason)
                    time.sleep(30)
                    continue

                # ── Step 3: Feature Engineering ─
                # Compute on FULL buffer (need enough bars for rolling windows)
                # then slice the last sequence_length rows for LSTM
                full_buffer = self.data_feed.get_buffer()
                if full_buffer is None or full_buffer.empty:
                    continue

                features = self.feature_engine.compute(full_buffer)
                if features is None:
                    continue

                # Take last sequence_length for LSTM input
                seq_len = self.config.perception.sequence_length
                if len(features) > seq_len:
                    features = features[-seq_len:]

                # ── Step 4: Regime Classification
                regime, regime_conf = self.regime_classifier.classify(full_buffer)

                # Skip trading in uncertain regime
                if regime == MarketRegime.UNCERTAIN and regime_conf > 0.5:
                    if self._loop_count % 60 == 0:
                        self.logger.info(
                            "Regime: UNCERTAIN (conf=%.2f) — holding cash",
                            regime_conf,
                        )
                    time.sleep(5)
                    continue

                # ── Step 5: Perception Encoding ─
                embedding = self.perception.encode(features)

                # ── Step 6: Memory Recall ───────
                memory_stats = self.memory.get_pattern_stats(embedding)

                # ── Step 7: Build Observation ───
                regime_one_hot = self.regime_classifier.get_regime_one_hot()
                balance_norm = (equity / self._initial_equity) - 1.0
                positions = self.connector.get_open_positions()
                pnl_norm = sum(p["profit"] for p in positions) / max(equity, 1.0)
                spread_norm = spread / 10.0  # Normalize spread

                observation = ForexTradingEnv.build_observation(
                    perception_embedding=embedding,
                    regime_one_hot=regime_one_hot,
                    balance_norm=balance_norm,
                    pnl_norm=pnl_norm,
                    spread_norm=spread_norm,
                )

                # ── Step 8: Agent Decision ──────
                action, confidence = self.drl_agent.predict(observation)

                # ── Step 9: Risk Evaluation ─────
                if action != TradeAction.HOLD:
                    atr_values = self.feature_engine._compute_atr(full_buffer)
                    current_atr = float(atr_values.iloc[-1]) if not atr_values.empty else 0.0005
                    current_price = self.data_feed.get_latest_price() or 0.0
                    symbol_info = self.connector.get_symbol_info()

                    proposal = self.risk_manager.evaluate_trade(
                        action=action,
                        current_price=current_price,
                        atr=current_atr,
                        account_info=account,
                        open_positions=len(positions),
                        point=symbol_info.get("point", 0.00001),
                        contract_size=symbol_info.get("trade_contract_size", 100000),
                    )

                    if proposal.approved:
                        self._execute_trade(action, proposal, observation, embedding, regime)
                    else:
                        self.logger.debug(
                            "Trade REJECTED: %s | Reason: %s",
                            action.name, proposal.reason,
                        )

                # ── Step 10: Manage Open Positions
                self._manage_positions(observation)

                # ── Step 11: Evolution Check ────
                if self.online_trainer.should_retrain():
                    self.online_trainer.retrain(self.drl_agent)

                if self.walk_forward.should_evaluate():
                    self.walk_forward.evaluate(self._trade_history[-100:])

                # ── Heartbeat ───────────────────
                if self._loop_count % 300 == 0:
                    self._log_heartbeat(equity, regime, regime_conf)

                # Save last observation
                self._last_observation = observation

                # Small sleep to prevent CPU hogging
                time.sleep(0.5)

            except KeyboardInterrupt:
                self.shutdown()
                break
            except Exception as e:
                self.logger.error("Main loop error: %s", e, exc_info=True)
                time.sleep(10)

    # ═══════════════════════════════════════════
    # TRADE EXECUTION
    # ═══════════════════════════════════════════

    def _execute_trade(self, action, proposal, observation, embedding, regime):
        """Execute a trade and record the experience."""
        result = self.connector.send_order(
            action=action,
            volume=proposal.lot_size,
            sl=proposal.stop_loss,
            tp=proposal.take_profit,
            comment=f"apex_{regime.value}",
        )

        if result.success:
            self.trade_logger.info(
                "OPEN | %s | %.2f lots @ %.5f | SL=%.5f TP=%.5f | Risk=$%.2f",
                action.name, proposal.lot_size, result.price,
                proposal.stop_loss, proposal.take_profit, proposal.risk_amount,
            )
            self.reward_calc.on_trade_open(result.price)

            # Store in memory
            self.memory.store(
                embedding=embedding,
                action=action.value,
                outcome=0.0,  # Will be updated on close
                regime=regime.value,
                timestamp=time.time(),
            )

    def _manage_positions(self, current_observation):
        """Check and manage open positions — force close if needed."""
        positions = self.connector.get_open_positions()

        for pos in positions:
            pnl = pos["profit"]
            hold_time = (datetime.now() - pos["time"]).total_seconds()

            # Force close if holding too long (scalping)
            max_hold_seconds = self.config.reward.max_hold_steps * 60
            if hold_time > max_hold_seconds:
                self.logger.info(
                    "Force closing position %d — held %.0fs (max: %.0fs)",
                    pos["ticket"], hold_time, max_hold_seconds,
                )
                close_result = self.connector.close_position(pos["ticket"])

                if close_result.success:
                    self._on_trade_closed(pnl, current_observation)

    def _on_trade_closed(self, pnl: float, observation: np.ndarray):
        """Handle post-trade cleanup and learning."""
        # Record in systems
        self.risk_manager.record_trade_result(pnl)
        self.circuit_breaker.record_trade(pnl)
        self.reward_calc.on_trade_close()

        # Compute reward
        account = self.connector.get_account_info()
        equity = account.get("equity", self._initial_equity)
        reward = self.reward_calc.compute(
            pnl_pips=pnl,
            equity=equity,
            is_trade_closed=True,
        )

        # Store experience for online learning
        self.online_trainer.add(
            state=self._last_observation if self._last_observation is not None else observation,
            action=0,
            reward=reward,
            next_state=observation,
            done=False,
        )

        # Trade history for walk-forward
        self._trade_history.append({
            "pnl": pnl,
            "reward": reward,
            "timestamp": time.time(),
        })

        self.trade_logger.info(
            "CLOSE | PnL: $%.2f | Reward: %.3f | Sharpe: %.2f",
            pnl, reward, self.reward_calc.get_rolling_sharpe(),
        )

    # ═══════════════════════════════════════════
    # HEARTBEAT & STATUS
    # ═══════════════════════════════════════════

    def _log_heartbeat(self, equity, regime, regime_conf):
        """Periodic status log."""
        reward_stats = self.reward_calc.get_stats()
        cb_stats = self.circuit_breaker.get_stats()
        buf_size = self.data_feed.get_buffer_size()

        self.logger.info(
            "♥ HEARTBEAT | Loop: %d | Equity: $%.2f | Regime: %s (%.0f%%) | "
            "WR: %.0f%% | Sharpe: %.2f | Memory: %d | Buffer: %d | "
            "CB: %s",
            self._loop_count, equity, regime.value, regime_conf * 100,
            reward_stats["win_rate"] * 100, reward_stats["rolling_sharpe"],
            self.memory.size(), buf_size,
            "HALTED" if cb_stats["is_halted"] else "OK",
        )


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apex Predator DRL Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Dry-run mode (default)
  python main.py --live             # Live trading (CAREFUL!)
  python main.py --symbol GBPUSD    # Trade different pair
  python main.py --timeframe M5     # Use M5 timeframe
        """,
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading (default: dry-run mode)",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Trading symbol (overrides settings.py)",
    )
    parser.add_argument(
        "--timeframe", type=str, default=None,
        help="Trading timeframe (overrides settings.py)",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Build configuration — settings.py is the base,
    # CLI args only override when explicitly provided
    config = ApexConfig()
    config.dry_run = not args.live
    if args.symbol is not None:
        config.mt5.symbol = args.symbol
    if args.timeframe is not None:
        config.mt5.timeframe = args.timeframe
    if args.log_level is not None:
        config.log_level = args.log_level

    # Create and run
    predator = ApexPredator(config)
    predator.start()


if __name__ == "__main__":
    main()

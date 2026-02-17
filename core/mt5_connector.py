"""
Apex Predator — Layer 0: MetaTrader 5 Connector
=================================================
Wraps MT5 Python API with error handling, retry logic,
and clean interfaces for the rest of the system.
"""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config.settings import MT5Config, TradeAction


logger = logging.getLogger("apex_predator.mt5")


# ──────────────────────────────────────────────
# Data Containers
# ──────────────────────────────────────────────

@dataclass
class TickData:
    """Single tick snapshot."""
    time: datetime
    bid: float
    ask: float
    spread: float
    volume: float


@dataclass
class OrderResult:
    """Trade execution result."""
    success: bool
    ticket: int
    price: float
    volume: float
    comment: str
    retcode: int


# ──────────────────────────────────────────────
# MT5 Timeframe Mapping
# ──────────────────────────────────────────────

TIMEFRAME_MAP = {}
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M5": mt5.TIMEFRAME_M5,
        "M10": mt5.TIMEFRAME_M10,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }


class MT5Connector:
    """
    MetaTrader 5 connection manager.
    Handles connection lifecycle, data retrieval, and order execution.
    """

    def __init__(self, config: MT5Config, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self._connected = False
        self._symbol_info = None

    # ── Connection ──────────────────────────────

    def connect(self) -> bool:
        """Initialize MT5 connection."""
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 package not installed — running in mock mode")
            self._connected = False
            return False

        if not mt5.initialize(path=self.config.path):
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            return False

        if self.config.login and self.config.password:
            authorized = mt5.login(
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
            )
            if not authorized:
                logger.error("MT5 login failed: %s", mt5.last_error())
                mt5.shutdown()
                return False

        # Validate symbol
        self._symbol_info = mt5.symbol_info(self.config.symbol)
        if self._symbol_info is None:
            logger.error("Symbol %s not found", self.config.symbol)
            mt5.shutdown()
            return False

        if not self._symbol_info.visible:
            mt5.symbol_select(self.config.symbol, True)

        self._connected = True
        logger.info("MT5 connected — Account: %s | Symbol: %s",
                     mt5.account_info().login if mt5.account_info() else "N/A",
                     self.config.symbol)
        logger.info("Symbol info — Point: %s | Digits: %d",
                     self._symbol_info.point, self._symbol_info.digits)
        return True

    def disconnect(self):
        """Shutdown MT5 connection."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Market Data ─────────────────────────────

    def get_tick(self) -> Optional[TickData]:
        """Get latest tick data."""
        if not self._connected:
            return self._mock_tick()

        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            logger.warning("Failed to get tick for %s", self.config.symbol)
            return None

        return TickData(
            time=datetime.fromtimestamp(tick.time),
            bid=tick.bid,
            ask=tick.ask,
            spread=round((tick.ask - tick.bid) / self._get_point(), 1),
            volume=tick.volume,
        )

    def get_ohlc(self, count: int = 500) -> Optional[pd.DataFrame]:
        """
        Get OHLC bars from MT5.

        Args:
            count: Number of bars to retrieve.

        Returns:
            DataFrame with columns: time, open, high, low, close, tick_volume, spread
        """
        if not self._connected:
            return self._mock_ohlc(count)

        tf = TIMEFRAME_MAP.get(self.config.timeframe)
        if tf is None:
            logger.error("Invalid timeframe: %s", self.config.timeframe)
            return None

        rates = mt5.copy_rates_from_pos(self.config.symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No OHLC data returned for %s", self.config.symbol)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df[["time", "open", "high", "low", "close", "volume", "spread"]]

    def get_ticks(self, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get recent tick history."""
        if not self._connected:
            return None

        ticks = mt5.copy_ticks_from_pos(self.config.symbol, 0, count, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return None

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # ── Account Info ────────────────────────────

    def get_account_info(self) -> Dict[str, Any]:
        """Get current account state."""
        if not self._connected:
            return self._mock_account()

        info = mt5.account_info()
        if info is None:
            return {}

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "leverage": info.leverage,
            "currency": info.currency,
        }

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions for current symbol."""
        if not self._connected:
            return []

        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "type": "BUY" if pos.type == 0 else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "profit": pos.profit,
                "sl": pos.sl,
                "tp": pos.tp,
                "time": datetime.fromtimestamp(pos.time),
                "magic": pos.magic,
            })
        return [p for p in result if p["magic"] == self.config.magic_number]

    # ── Order Execution ─────────────────────────

    def send_order(
        self,
        action: TradeAction,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "apex",
    ) -> OrderResult:
        """
        Send a market order.

        Args:
            action: BUY or SELL
            volume: Lot size
            sl: Stop Loss price
            tp: Take Profit price
            comment: Order comment

        Returns:
            OrderResult with execution details
        """
        if self.dry_run:
            logger.info("[DRY RUN] %s %.2f lots | SL=%.5f TP=%.5f",
                        action.name, volume, sl, tp)
            return OrderResult(
                success=True, ticket=0, price=0.0,
                volume=volume, comment="DRY_RUN", retcode=0,
            )

        if not self._connected:
            return OrderResult(
                success=False, ticket=0, price=0.0,
                volume=0.0, comment="NOT_CONNECTED", retcode=-1,
            )

        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            return OrderResult(
                success=False, ticket=0, price=0.0,
                volume=0.0, comment="NO_TICK", retcode=-1,
            )

        order_type = mt5.ORDER_TYPE_BUY if action == TradeAction.BUY else mt5.ORDER_TYPE_SELL
        price = tick.ask if action == TradeAction.BUY else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResult(
                success=False, ticket=0, price=0.0,
                volume=0.0, comment=str(mt5.last_error()), retcode=-1,
            )

        success = result.retcode == mt5.TRADE_RETCODE_DONE
        if not success:
            logger.warning("Order failed: retcode=%d comment=%s",
                           result.retcode, result.comment)

        return OrderResult(
            success=success,
            ticket=result.order if success else 0,
            price=result.price if success else 0.0,
            volume=volume,
            comment=result.comment,
            retcode=result.retcode,
        )

    def close_position(self, ticket: int) -> OrderResult:
        """Close a specific position by ticket number."""
        if self.dry_run:
            logger.info("[DRY RUN] Closing position ticket=%d", ticket)
            return OrderResult(
                success=True, ticket=ticket, price=0.0,
                volume=0.0, comment="DRY_RUN_CLOSE", retcode=0,
            )

        if not self._connected:
            return OrderResult(
                success=False, ticket=0, price=0.0,
                volume=0.0, comment="NOT_CONNECTED", retcode=-1,
            )

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(
                success=False, ticket=ticket, price=0.0,
                volume=0.0, comment="POSITION_NOT_FOUND", retcode=-1,
            )

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.config.symbol)
        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": "apex_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(
            success=success,
            ticket=ticket,
            price=result.price if success else 0.0,
            volume=pos.volume,
            comment=result.comment if result else "SEND_FAILED",
            retcode=result.retcode if result else -1,
        )

    def close_all_positions(self) -> int:
        """Close all positions for current symbol/magic. Returns count closed."""
        positions = self.get_open_positions()
        closed = 0
        for pos in positions:
            result = self.close_position(pos["ticket"])
            if result.success:
                closed += 1
        logger.info("Closed %d/%d positions", closed, len(positions))
        return closed

    # ── Helpers ──────────────────────────────────

    def _get_point(self) -> float:
        """Get symbol's point size."""
        if self._symbol_info:
            return self._symbol_info.point
        return 0.00001  # Default for 5-digit Forex

    def get_point(self) -> float:
        """Public accessor for point size."""
        return self._get_point()

    def get_symbol_info(self) -> Dict[str, Any]:
        """Get symbol trading specifications."""
        if not self._connected or self._symbol_info is None:
            return {"point": 0.00001, "digits": 5, "trade_contract_size": 100000}

        return {
            "point": self._symbol_info.point,
            "digits": self._symbol_info.digits,
            "trade_contract_size": self._symbol_info.trade_contract_size,
            "volume_min": self._symbol_info.volume_min,
            "volume_max": self._symbol_info.volume_max,
            "volume_step": self._symbol_info.volume_step,
        }

    # ── Mock Data (for dry-run / testing) ───────

    def _mock_tick(self) -> TickData:
        """Generate mock tick for testing."""
        base_price = 1.10000
        noise = np.random.normal(0, 0.0005)
        bid = base_price + noise
        ask = bid + 0.00015
        return TickData(
            time=datetime.now(),
            bid=round(bid, 5),
            ask=round(ask, 5),
            spread=1.5,
            volume=float(np.random.randint(1, 100)),
        )

    def _mock_ohlc(self, count: int) -> pd.DataFrame:
        """Generate mock OHLC data for testing."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=count, freq="min")
        price = 1.10000
        data = []
        for dt in dates:
            change = np.random.normal(0, 0.0003)
            o = price
            h = price + abs(np.random.normal(0, 0.0002))
            l = price - abs(np.random.normal(0, 0.0002))
            c = price + change
            price = c
            data.append({
                "time": dt, "open": round(o, 5), "high": round(h, 5),
                "low": round(l, 5), "close": round(c, 5),
                "volume": np.random.randint(50, 500),
                "spread": np.random.randint(1, 5),
            })
        return pd.DataFrame(data)

    def _mock_account(self) -> Dict[str, Any]:
        """Mock account info for testing."""
        return {
            "balance": 10000.0,
            "equity": 10000.0,
            "margin": 0.0,
            "free_margin": 10000.0,
            "profit": 0.0,
            "leverage": 500,
            "currency": "USD",
        }

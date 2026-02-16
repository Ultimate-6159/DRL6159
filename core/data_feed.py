"""
Apex Predator — Data Feed Manager
===================================
Rolling buffer for real-time tick/OHLC data ingestion.
"""

import logging
from typing import Optional
from collections import deque

import numpy as np
import pandas as pd

from config.settings import MT5Config, FeatureConfig
from core.mt5_connector import MT5Connector

logger = logging.getLogger("apex_predator.data_feed")


class DataFeed:
    """
    Manages streaming data into a rolling buffer.
    Provides normalized snapshots for downstream modules.
    """

    def __init__(
        self,
        connector: MT5Connector,
        feature_config: FeatureConfig,
    ):
        self.connector = connector
        self.config = feature_config
        self._buffer: Optional[pd.DataFrame] = None
        self._tick_buffer: deque = deque(maxlen=10_000)
        self._last_bar_time = None

    # ── Initialization ──────────────────────────

    def initialize(self) -> bool:
        """Load initial historical data into buffer."""
        df = self.connector.get_ohlc(count=self.config.lookback_window)
        if df is None or df.empty:
            logger.error("Failed to load initial OHLC data")
            return False

        self._buffer = df.copy()
        self._last_bar_time = df["time"].iloc[-1]
        logger.info(
            "DataFeed initialised — %d bars loaded | Latest: %s",
            len(df), self._last_bar_time,
        )
        return True

    # ── Update ──────────────────────────────────

    def update(self) -> bool:
        """
        Fetch new bars and append to buffer.
        Maintains rolling window of lookback_window size.

        Returns:
            True if new data was added.
        """
        df = self.connector.get_ohlc(count=10)  # Fetch last 10 bars
        if df is None or df.empty:
            return False

        if self._buffer is None:
            self._buffer = df
            self._last_bar_time = df["time"].iloc[-1]
            return True

        # Append only new bars
        new_bars = df[df["time"] > self._last_bar_time]
        if new_bars.empty:
            return False

        self._buffer = pd.concat(
            [self._buffer, new_bars], ignore_index=True
        )

        # Trim to lookback window
        if len(self._buffer) > self.config.lookback_window:
            self._buffer = self._buffer.iloc[-self.config.lookback_window:]
            self._buffer.reset_index(drop=True, inplace=True)

        self._last_bar_time = self._buffer["time"].iloc[-1]
        logger.debug("DataFeed updated: +%d bars | Total: %d",
                      len(new_bars), len(self._buffer))
        return True

    def update_tick(self):
        """Capture latest tick into tick buffer."""
        tick = self.connector.get_tick()
        if tick is not None:
            self._tick_buffer.append(tick)

    # ── Accessors ───────────────────────────────

    def get_buffer(self) -> Optional[pd.DataFrame]:
        """Get the full OHLC buffer as DataFrame."""
        return self._buffer

    def get_latest_bars(self, n: int = 60) -> Optional[pd.DataFrame]:
        """Get last N bars from the buffer."""
        if self._buffer is None or len(self._buffer) < n:
            return self._buffer
        return self._buffer.iloc[-n:].reset_index(drop=True)

    def get_latest_price(self) -> Optional[float]:
        """Get the most recent close price."""
        if self._buffer is not None and not self._buffer.empty:
            return float(self._buffer["close"].iloc[-1])
        return None

    def get_spread(self) -> float:
        """Get current spread from latest tick or buffer."""
        if self._tick_buffer:
            return self._tick_buffer[-1].spread
        if self._buffer is not None and "spread" in self._buffer.columns:
            return float(self._buffer["spread"].iloc[-1])
        return 0.0

    def get_buffer_size(self) -> int:
        """Return current buffer size."""
        return len(self._buffer) if self._buffer is not None else 0

    def is_ready(self) -> bool:
        """Check if buffer has enough data for processing."""
        return (
            self._buffer is not None
            and len(self._buffer) >= self.config.lookback_window
        )

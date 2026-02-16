"""
Apex Predator — Structured Logging System
==========================================
Trade logs, error logs, performance logs with rotation.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "apex_predator",
    log_dir: str = "logs/",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB per log file
    backup_count: int = 20,               # Keep 20 rotated files
) -> logging.Logger:
    """
    Create a structured logger with file rotation and console output.

    Args:
        name: Logger name (also used as log filename prefix).
        log_dir: Directory for log files.
        level: Logging level string.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of backup files to keep.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on re-init
    if logger.handlers:
        return logger

    # ── Format ──────────────────────────────────
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(funcName)s:%(lineno)d | %(message)s"
    )
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # ── Console Handler ─────────────────────────
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── File Handler (rotated) ──────────────────
    log_path = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ── Trade-specific log ──────────────────────
    trade_path = os.path.join(log_dir, f"{name}_trades.log")
    trade_handler = RotatingFileHandler(
        trade_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    trade_fmt = logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    trade_handler.setFormatter(trade_fmt)
    trade_logger = logging.getLogger(f"{name}.trades")
    trade_logger.addHandler(trade_handler)
    trade_logger.setLevel(logging.INFO)

    logger.info("Logger initialised — log_dir=%s level=%s", log_dir, level)
    return logger


def get_trade_logger(parent_name: str = "apex_predator") -> logging.Logger:
    """Get the trade-specific sub-logger."""
    return logging.getLogger(f"{parent_name}.trades")

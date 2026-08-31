"""Overnight gap signal package."""

from .data_loader import RollingOptionsDataLoader
from .overnight_signal import (
    OvernightConfig,
    OvernightSignalEngine,
    run_backtest,
    summarize,
)

__all__ = [
    "RollingOptionsDataLoader",
    "OvernightConfig",
    "OvernightSignalEngine",
    "run_backtest",
    "summarize",
]

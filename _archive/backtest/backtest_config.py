from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestConfig:
    # Isolated storage root for all backtest artifacts.
    data_root: Path = Path("data") / "backtest"
    cache_dir: Path = Path("data") / "backtest" / "cache"
    outputs_dir: Path = Path("data") / "backtest" / "outputs"

    # Existing live logs used as one replay source (read-only input).
    predictions_csv: Path = Path("data") / "tracker" / "predictions.csv"
    outcomes_csv: Path = Path("data") / "tracker" / "outcomes.csv"

    # Execution assumptions for future extensions.
    slippage_bps: float = 5.0
    brokerage_per_lot: float = 40.0


from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .data_loader import BacktestDataLoader


class BacktestReporter:
    def __init__(self, loader: BacktestDataLoader):
        self.loader = loader

    def write(self, summary: Dict, events: List[Dict], tag: str) -> Dict[str, Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.loader.write_json_output(f"{tag}_summary_{ts}.json", summary)
        events_path = self.loader.write_csv_output(f"{tag}_events_{ts}.csv", events)
        return {"summary": summary_path, "events": events_path}


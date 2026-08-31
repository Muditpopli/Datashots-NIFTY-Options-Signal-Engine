"""Compatibility launcher for the data-backed live strategy pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.data_backed.live_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())

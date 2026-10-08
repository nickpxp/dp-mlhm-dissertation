"""Run the three research questions in order, then the tables and figure."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("experiments/rq1_privacy.py", "experiments/rq2_tradeoff.py", "experiments/rq3_performance.py",
         "tools/make_tables.py", "tools/make_figure.py")

for step in STEPS:
    print("\n" + "=" * 80 + f"\n{step}\n" + "=" * 80)
    sys.argv = [str(ROOT / step)]
    runpy.run_path(str(ROOT / step), run_name="__main__")

"""Shared runner plumbing: import path, logging, CSV output."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlhm.config import RESULTS_DIR  # noqa: E402


def setup_logging(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def setting(eps: float | None) -> str:
    return "non-private" if eps is None else f"eps_{eps:.1f}"

"""Load the synthetic cardiac dataset and build the scaled, stratified split.

Features are min-max scaled to [0, 1] with bounds from the training split,
treated as public, so every released statistic and every noised feature has
sensitivity 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlhm.config import DATA_PATH, FEATURES, SPLIT_SEED, TARGET, TEST_SIZE


@dataclass(frozen=True)
class Split:
    """Scaled train/test arrays, the raw frame, and the public bounds."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    features: tuple[str, ...] = FEATURES

    def drop(self, feature: str) -> "Split":
        """The same split without one column, for the attribute attack."""
        j = self.features.index(feature)
        keep = [i for i in range(len(self.features)) if i != j]
        return Split(self.x_train[:, keep], self.x_test[:, keep], self.y_train, self.y_test,
                     self.lo[keep], self.hi[keep], tuple(self.features[i] for i in keep))

    def column(self, feature: str, part: str) -> np.ndarray:
        x = self.x_train if part == "train" else self.x_test
        return x[:, self.features.index(feature)]


def load_dataframe(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read ddata.csv and fail loudly on a wrong schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; place ddata.csv in data/")
    df = pd.read_csv(path)
    missing = [c for c in (*FEATURES, TARGET) if c not in df.columns]
    if missing:
        raise ValueError(f"ddata.csv is missing columns: {missing}")
    if not set(np.unique(df[TARGET])) <= {0, 1}:
        raise ValueError(f"{TARGET} must be binary")
    return df


def minmax(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] with public bounds; constant features map to 0."""
    width = np.where(hi - lo == 0, 1.0, hi - lo)
    return np.clip((np.asarray(x, float) - lo) / width, 0.0, 1.0)


def make_split(df: pd.DataFrame | None = None) -> Split:
    """80/20 stratified split, seed 42, then train-bound min-max scaling."""
    df = load_dataframe() if df is None else df
    x = df[list(FEATURES)].to_numpy(float)
    y = df[TARGET].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    lo, hi = x_tr.min(axis=0), x_tr.max(axis=0)
    return Split(minmax(x_tr, lo, hi), minmax(x_te, lo, hi), y_tr, y_te, lo, hi)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Table 8: mean, SD and range per feature, plus label prevalence."""
    rows = [{"variable": c, "mean": df[c].mean(), "sd": df[c].std(), "min": df[c].min(), "max": df[c].max()}
            for c in (*FEATURES, TARGET)]
    return pd.DataFrame(rows)

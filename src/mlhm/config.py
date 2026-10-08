"""Every constant the dissertation states (Chapters 3 and 4), in one place."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PATH: Path = REPO_ROOT / "data" / "ddata.csv"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = REPO_ROOT / "figures"

# ---------------------------------------------------------------------------
# Data (Chapter 4, Data Results)
# ---------------------------------------------------------------------------
FEATURES: tuple[str, ...] = (
    "age", "af_diagnosis", "ecg_reading", "ppg_reading",
    "signal_quality", "af_context_30min", "confidence_score",
)
TARGET: str = "af_detected"
TEST_SIZE: float = 0.2              # 80,000 / 20,000
SPLIT_SEED: int = 42                # stratified on the label

# ---------------------------------------------------------------------------
# Privacy grid (Algorithm: privacy parameter selection)
# ---------------------------------------------------------------------------
EPSILONS: tuple[float, ...] = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
FOCUS_EPSILON: float = 1.0          # the recommended operating point
DELTA: float = 1e-5                 # delta << 1/n for n = 100,000
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)    # n = 5 repetitions per setting
MECHANISMS: tuple[str, ...] = ("laplace", "gaussian")
PLACEMENTS: tuple[str, ...] = ("central", "featurewise")

# ---------------------------------------------------------------------------
# Models and attacks
# ---------------------------------------------------------------------------
VAR_FLOOR: float = 1e-3             # floor on reconstructed NB variances
LR_LAMBDA: float = 1e-3             # L2 regularization, output-perturbation LR
RF_TREES: int = 100
MIA_SAMPLE: int = 2000              # members and non-members per attack
ATTRIBUTE_TARGET: str = "ecg_reading"    # Table 10: ECG discretized into quartiles
ATTRIBUTE_BINS: int = 4
ATTACK_SPLIT: float = 0.8           # share of the held-out fold the adversary trains on

# ---------------------------------------------------------------------------
# Performance (RQ3)
# ---------------------------------------------------------------------------
TIMING_TRIALS: int = 50

"""The Laplace and Gaussian mechanisms at three points of release.

* central: noise on Gaussian NB sufficient statistics (Vaidya et al. 2013),
  k(1 + 2d) released statistics, each with sensitivity 1, the budget split by
  sequential composition (Dwork & Roth 2014, Thm 3.16).
* featurewise: noise on every training record's features before fitting, the
  preprocessing placement of the Chapter 3 noise-addition algorithm, with the
  per-record budget split evenly over the d features.
* statistical query: noise on a feature mean, sensitivity range / n (Figures 25-26).

Laplace: b = 1 / eps_q. Gaussian: sigma = sqrt(2 ln(1.25 / delta_q)) / eps_q,
valid for eps_q < 1 (Dwork & Roth 2014, Thm A.1). Both arms of each
placement share one code path and differ only in the distribution drawn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlhm.config import DELTA, MECHANISMS, VAR_FLOOR


def noise_scale(eps: float, mechanism: str, sensitivity: float = 1.0, delta: float = DELTA) -> float:
    """Laplace b or Gaussian sigma for one query at budget eps."""
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    if mechanism == "laplace":
        return sensitivity / eps
    if mechanism == "gaussian":
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / eps
    raise ValueError(f"mechanism must be one of {MECHANISMS}, got {mechanism!r}")


def draw(rng: np.random.Generator, mechanism: str, scale: float, size) -> np.ndarray:
    return rng.laplace(0.0, scale, size) if mechanism == "laplace" else rng.normal(0.0, scale, size)


# ---------------------------------------------------------------------------
# Central: Gaussian NB on privatized sufficient statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussianNB:
    """Fitted Gaussian NB; predict and predict_proba need nothing else."""

    classes: np.ndarray
    log_priors: np.ndarray
    means: np.ndarray        # (k, d)
    variances: np.ndarray    # (k, d)

    def joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.means.shape[1]:
            raise ValueError(f"expected (n, {self.means.shape[1]}) input, got {x.shape}")
        return np.stack([self.log_priors[c] - 0.5 * (np.log(2 * np.pi * self.variances[c])
                         + (x - self.means[c]) ** 2 / self.variances[c]).sum(axis=1)
                         for c in range(len(self.classes))], axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        jll = self.joint_log_likelihood(x)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.joint_log_likelihood(x), axis=1)]


def fit_central_nb(x: np.ndarray, y: np.ndarray, eps: float | None, mechanism: str = "laplace",
                   seed: int | None = None, delta: float = DELTA) -> GaussianNB:
    """Privatize count, sums and sums of squares per class; eps=None fits without noise."""
    x, y = np.asarray(x, float), np.asarray(y)
    if x.min() < 0 or x.max() > 1:
        raise ValueError("x must be scaled to [0, 1]; sensitivity 1 depends on it")
    classes = np.unique(y)
    k, d = len(classes), x.shape[1]
    n_q = k * (1 + 2 * d)
    rng = np.random.default_rng(seed)
    scale = 0.0 if eps is None else noise_scale(eps / n_q, mechanism, delta=delta / n_q)

    def noise(size: int) -> np.ndarray:
        return np.zeros(size) if eps is None else draw(rng, mechanism, scale, size)

    counts, means, variances = np.zeros(k), np.zeros((k, d)), np.zeros((k, d))
    for c, label in enumerate(classes):   # draw order fixed: count, d sums, d sums of squares
        xc = x[y == label]
        n = max(len(xc) + float(noise(1)[0]), 1.0)
        mu = np.clip((xc.sum(axis=0) + noise(d)) / n, 0.0, 1.0)
        var = np.maximum(((xc ** 2).sum(axis=0) + noise(d)) / n - mu ** 2, VAR_FLOOR)
        counts[c], means[c], variances[c] = n, mu, var
    return GaussianNB(classes, np.log(counts / counts.sum()), means, variances)


# ---------------------------------------------------------------------------
# Featurewise: noise on the training records
# ---------------------------------------------------------------------------
def perturb_features(x: np.ndarray, eps: float, mechanism: str, seed: int | None = None,
                     delta: float = DELTA) -> np.ndarray:
    """Noise every [0, 1] feature at budget eps / d, clipped back to [0, 1]."""
    x = np.asarray(x, float)
    d = x.shape[1]
    scale = noise_scale(eps / d, mechanism, delta=delta / d)
    return np.clip(x + draw(np.random.default_rng(seed), mechanism, scale, x.shape), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Statistical query: noisy feature means (Figures 25 and 26)
# ---------------------------------------------------------------------------
def noisy_means(x_raw: np.ndarray, eps: float, mechanism: str, seed: int | None = None,
                delta: float = DELTA) -> np.ndarray:
    """Each feature mean released alone at budget eps, sensitivity (max - min) / n."""
    x_raw = np.asarray(x_raw, float)
    n = len(x_raw)
    sens = (x_raw.max(axis=0) - x_raw.min(axis=0)) / n
    rng = np.random.default_rng(seed)
    return np.array([x_raw[:, j].mean() + draw(rng, mechanism, noise_scale(eps, mechanism, s, delta), 1)[0]
                     for j, s in enumerate(sens)])

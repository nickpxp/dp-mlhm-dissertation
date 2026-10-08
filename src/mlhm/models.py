"""One dispatcher for every model, placement and mechanism the codebook evaluates.

fit(model, placement, mechanism, eps, x, y, seed) returns an object with
predict and predict_proba. eps=None is the non-private reference.

| model | central                                         | featurewise          |
|-------|-------------------------------------------------|----------------------|
| nb    | sufficient statistics, Laplace or Gaussian      | sklearn GaussianNB   |
| lr    | output perturbation (Chaudhuri et al. 2011)     | sklearn LR           |
| rf    | diffprivlib forest, Laplace only (Fletcher &    | sklearn RF           |
|       | Islam 2017); Gaussian raises NotApplicable      |                      |
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB as SkGaussianNB

from mlhm.config import DELTA, LR_LAMBDA, RF_TREES
from mlhm.mechanisms import fit_central_nb, noise_scale, perturb_features

MODELS: tuple[str, ...] = ("nb", "lr", "rf")


class NotApplicable(Exception):
    """A model/mechanism pair the library does not offer."""


@dataclass(frozen=True)
class OutputPerturbedLR:
    """Logistic regression whose weight vector was noised once after fitting."""

    weights: np.ndarray
    row_norm: float
    classes: np.ndarray

    def _augment(self, x: np.ndarray) -> np.ndarray:
        return np.hstack([np.asarray(x, float), np.ones((len(x), 1))]) / self.row_norm

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-np.clip(self._augment(x) @ self.weights, -500, 500)))
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes[(self.predict_proba(x)[:, 1] >= 0.5).astype(int)]


def fit_output_perturbed_lr(x: np.ndarray, y: np.ndarray, eps: float, mechanism: str,
                            seed: int | None = None, lam: float = LR_LAMBDA,
                            delta: float = DELTA) -> OutputPerturbedLR:
    """L2-regularized LR with ||x|| <= 1; L2 sensitivity of the minimizer is 2 / (n lambda)."""
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("output perturbation is implemented for binary labels")
    xa = np.hstack([np.asarray(x, float), np.ones((len(x), 1))])
    row_norm = float(np.linalg.norm(xa, axis=1).max())
    xa /= row_norm
    n, d = xa.shape
    w = LogisticRegression(C=1.0 / (n * lam), fit_intercept=False, max_iter=2000).fit(xa, y).coef_.ravel()
    sens = 2.0 / (n * lam)
    rng = np.random.default_rng(seed)
    if mechanism == "gaussian":
        b = rng.normal(0.0, noise_scale(eps, "gaussian", sens, delta), d)
    else:   # spherical Laplace: Gamma(d, sens / eps) radius, uniform direction
        g = rng.normal(size=d)
        b = rng.gamma(d, sens / eps) * g / np.linalg.norm(g)
    return OutputPerturbedLR(w + b, row_norm, classes)


def _plain(model: str, seed: int | None):
    if model == "nb":
        return SkGaussianNB()
    if model == "lr":
        return LogisticRegression(max_iter=1000)
    if model == "rf":
        return RandomForestClassifier(n_estimators=RF_TREES, random_state=seed, n_jobs=-1)
    raise ValueError(f"model must be one of {MODELS}")


def fit(model: str, placement: str, mechanism: str, eps: float | None,
        x: np.ndarray, y: np.ndarray, seed: int | None = None):
    """Fit one configuration; eps=None ignores placement and mechanism."""
    if eps is None:
        return fit_central_nb(x, y, None) if model == "nb" else _plain(model, seed).fit(x, y)
    if placement == "featurewise":
        return _plain(model, seed).fit(perturb_features(x, eps, mechanism, seed), y)
    if placement != "central":
        raise ValueError("placement must be 'central' or 'featurewise'")
    if model == "nb":
        return fit_central_nb(x, y, eps, mechanism, seed)
    if model == "lr":
        return fit_output_perturbed_lr(x, y, eps, mechanism, seed)
    if model == "rf":
        if mechanism != "laplace":
            raise NotApplicable("diffprivlib's forest has no Gaussian mechanism")
        from diffprivlib.models import RandomForestClassifier as DPRF   # lazy: optional dependency
        d = x.shape[1]
        return DPRF(n_estimators=RF_TREES, epsilon=eps, bounds=(np.zeros(d), np.ones(d)),
                    classes=np.unique(y), random_state=seed).fit(x, y)
    raise ValueError(f"model must be one of {MODELS}")

"""Utility metrics, prediction consistency, timing and memory, and the statistics."""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mlhm.config import TIMING_TRIALS


def utility(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall and F1 on the positive class (Tables 11-12)."""
    p = model.predict(x)
    return {"accuracy": float(accuracy_score(y, p)), "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)), "f1": float(f1_score(y, p, zero_division=0))}


def consistency(prob_runs: list[np.ndarray], pred_runs: list[np.ndarray],
                reference_pred: np.ndarray | None = None) -> dict[str, float]:
    """Table 14 from real model outputs across repeated DP fits.

    CV is the per-record SD / mean of P(AF) across runs, in percent; decision
    agreement is the share of records whose predicted label never changes,
    and agreement with the non-private model when a reference is given.
    """
    p = np.stack(prob_runs)
    mean = p.mean(axis=0)
    cv = np.where(mean > 1e-12, p.std(axis=0) / np.maximum(mean, 1e-12) * 100.0, 0.0)
    preds = np.stack(pred_runs)
    out = {"mean_cv_pct": float(cv.mean()), "sd_cv_pct": float(cv.std()), "median_cv_pct": float(np.median(cv)),
           "p95_cv_pct": float(np.percentile(cv, 95)), "max_cv_pct": float(cv.max()),
           "decision_agreement_runs": float((preds == preds[0]).all(axis=0).mean())}
    if reference_pred is not None:
        out["decision_agreement_nondp"] = float((preds == reference_pred).mean())
    return out


def inference_cost(model, x: np.ndarray, trials: int = TIMING_TRIALS) -> dict[str, float]:
    """ms per sample, samples per second, and peak traced memory of predict (Table 15)."""
    model.predict(x[:10])                      # warm-up
    times, peaks = [], []
    for _ in range(trials):
        tracemalloc.start()
        t0 = time.perf_counter()
        model.predict(x)
        times.append(time.perf_counter() - t0)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    dt = float(np.median(times))
    return {"ms_per_sample": dt / len(x) * 1000.0, "throughput": len(x) / dt, "peak_mb": max(peaks) / 1e6}


def compare_runs(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired t-test, 95% t intervals and Pearson r on matched-seed runs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("need two matched samples of at least two runs")
    t, p = stats.ttest_rel(a, b)
    ci = lambda v: stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))  # noqa: E731
    r, rp = stats.pearsonr(a, b) if a.std() > 0 and b.std() > 0 else (float("nan"), float("nan"))
    return {"mean_a": a.mean(), "mean_b": b.mean(), "ci_a_lo": ci(a)[0], "ci_a_hi": ci(a)[1],
            "ci_b_lo": ci(b)[0], "ci_b_hi": ci(b)[1], "t": float(t), "p": float(p),
            "pearson_r": float(r), "pearson_p": float(rp)}

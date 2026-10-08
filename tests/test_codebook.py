"""Unit tests for the mechanisms, models, attacks and evaluation helpers.

Toy data only; runs in a few seconds without data/ddata.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlhm.attacks import attribute_inference, membership_rule_based, quartile_bins  # noqa: E402
from mlhm.evaluation import compare_runs, consistency, utility  # noqa: E402
from mlhm.mechanisms import fit_central_nb, noise_scale, noisy_means, perturb_features  # noqa: E402
from mlhm.models import MODELS, NotApplicable, fit  # noqa: E402


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n, d = 4000, 7
    y = rng.binomial(1, 0.4, n)
    x = np.clip(rng.normal(0.4 + 0.2 * y[:, None], 0.15, (n, d)), 0, 1)
    return x[:3000], y[:3000], x[3000:], y[3000:]


# --- mechanisms -------------------------------------------------------------
def test_noise_scales():
    assert noise_scale(0.5, "laplace") == pytest.approx(2.0)
    assert noise_scale(1.0, "laplace", sensitivity=3.0) == pytest.approx(3.0)
    assert noise_scale(1.0, "gaussian", delta=1e-5) == pytest.approx(np.sqrt(2 * np.log(1.25e5)))
    for bad in ((0, "laplace"), (1.0, "cauchy")):
        with pytest.raises(ValueError):
            noise_scale(*bad)


def test_central_nb_without_noise_matches_sklearn(toy):
    from sklearn.naive_bayes import GaussianNB
    xtr, ytr, xte, _ = toy
    ours = fit_central_nb(xtr, ytr, None)
    ref = GaussianNB(var_smoothing=0).fit(xtr, ytr)
    np.testing.assert_allclose(ours.means, ref.theta_, atol=1e-10)
    np.testing.assert_allclose(ours.variances, ref.var_, rtol=1e-8)
    assert (ours.predict(xte) == ref.predict(xte)).mean() > 0.999


def test_central_nb_is_seeded_and_rejects_unscaled(toy):
    xtr, ytr, xte, _ = toy
    a, b = (fit_central_nb(xtr, ytr, 1.0, "gaussian", seed=7) for _ in range(2))
    np.testing.assert_array_equal(a.predict_proba(xte), b.predict_proba(xte))
    with pytest.raises(ValueError):
        fit_central_nb(xtr * 2, ytr, 1.0)


def test_featurewise_stays_in_unit_box(toy):
    xtr = toy[0]
    for mech in ("laplace", "gaussian"):
        z = perturb_features(xtr, 0.1, mech, seed=1)
        assert z.shape == xtr.shape and z.min() >= 0 and z.max() <= 1
        assert not np.allclose(z, xtr)


def test_noisy_means_error_shrinks_with_epsilon():
    x = np.random.default_rng(3).uniform(0, 100, (5000, 4))
    err = lambda e: np.mean([np.abs(noisy_means(x, e, "laplace", s) - x.mean(0)).mean() for s in range(50)])  # noqa: E731
    assert err(5.0) < err(0.1)


# --- models -----------------------------------------------------------------
@pytest.mark.parametrize("model", MODELS)
def test_every_configuration_fits(toy, model):
    xtr, ytr, xte, yte = toy
    configs = [(None, None, None)] + [(p, m, 1.0) for p in ("central", "featurewise") for m in ("laplace", "gaussian")]
    for placement, mech, eps in configs:
        if (model, placement, mech) == ("rf", "central", "gaussian"):
            with pytest.raises(NotApplicable):
                fit(model, placement, mech, eps, xtr, ytr, 0)
            continue
        m = fit(model, placement, mech, eps, xtr, ytr, 0)
        proba = m.predict_proba(xte)
        assert proba.shape == (len(xte), 2) and np.allclose(proba.sum(1), 1)
        assert 0 <= utility(m, xte, yte)["accuracy"] <= 1


# --- attacks ----------------------------------------------------------------
def test_membership_is_chance_on_identical_folds(toy):
    xtr, ytr, _, _ = toy
    m = fit("nb", None, None, None, xtr, ytr)
    assert membership_rule_based(m, xtr, ytr, xtr, ytr, n=1000) == pytest.approx(0.5)


def test_quartile_bins_use_training_edges():
    train = np.arange(100.0)
    b = quartile_bins(train, np.array([-1.0, 10.0, 30.0, 60.0, 99.0, 500.0]))
    assert b.tolist() == [0, 0, 1, 2, 3, 3]
    assert np.bincount(quartile_bins(train, train)).tolist() == [25, 25, 25, 25]


def test_attribute_inference_reports_control(toy):
    xtr, ytr, xte, _ = toy
    m = fit("nb", None, None, None, xtr[:, 1:], ytr)
    target = quartile_bins(xtr[:, 0], xte[:, 0])
    r = attribute_inference(m, xte[:, 1:], target, seed=0)
    assert r["model_contribution"] == pytest.approx(r["attack_accuracy"] - r["known_features_only"])
    assert 0.25 <= r["majority_rate"] <= 1 and len(r["importances"]) == xte.shape[1] - 1 + 3


# --- evaluation -------------------------------------------------------------
def test_consistency_on_toy_runs():
    same = consistency([np.full(4, 0.3)] * 3, [np.zeros(4, int)] * 3, np.zeros(4, int))
    assert same["mean_cv_pct"] == 0 and same["decision_agreement_runs"] == 1 and same["decision_agreement_nondp"] == 1
    probs = [np.array([0.2, 0.5]), np.array([0.4, 0.5])]
    r = consistency(probs, [np.array([0, 1]), np.array([1, 1])])
    assert r["max_cv_pct"] == pytest.approx(100 * 0.1 / 0.3) and r["decision_agreement_runs"] == 0.5


def test_compare_runs():
    a, b = np.array([0.70, 0.71, 0.72, 0.73, 0.74]), np.array([0.69, 0.71, 0.70, 0.72, 0.72])
    r = compare_runs(a, b)
    assert r["mean_a"] == pytest.approx(0.72) and r["ci_a_lo"] < 0.72 < r["ci_a_hi"]
    assert r["t"] > 0 and 0 < r["p"] < 1 and -1 <= r["pearson_r"] <= 1
    with pytest.raises(ValueError):
        compare_runs(a, b[:3])

"""Membership and attribute inference, measured against the DP model itself.

Membership: the rule-based black-box attack (member iff classified correctly;
Yeom et al. 2018, shipped in ART as MembershipInferenceBlackBoxRuleBased),
scored as balanced accuracy on equal numbers of members and non-members.
Chance is 0.5; a value below 0.5 is chance plus sampling noise, not protection.

Attribute: the adversary holds a held-out patient's other features and can
query the model (Chapter 3 algorithm). An attack model learns the target
feature's quartile from the known features plus the model's prediction and
class probabilities. The same attack without the model's outputs is reported
alongside: the difference is what the released model adds.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from mlhm.config import ATTACK_SPLIT, ATTRIBUTE_BINS, MIA_SAMPLE, RF_TREES


def membership_rule_based(model, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray, n: int = MIA_SAMPLE) -> float:
    """(train accuracy + test error) / 2 on the first n records of each fold."""
    n = min(n, len(x_train), len(x_test))
    if n == 0:
        raise ValueError("attack needs members and non-members")
    tr = (model.predict(x_train[:n]) == y_train[:n]).mean()
    te = (model.predict(x_test[:n]) == y_test[:n]).mean()
    return float((tr + 1.0 - te) / 2.0)


def quartile_bins(train_values: np.ndarray, values: np.ndarray, bins: int = ATTRIBUTE_BINS) -> np.ndarray:
    """Quantile bins with edges from the training fold, so the bins are public."""
    edges = np.quantile(train_values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(values, edges)


def attribute_inference(model, x_known: np.ndarray, target_bin: np.ndarray, seed: int = 42,
                        split: float = ATTACK_SPLIT) -> dict[str, float | np.ndarray]:
    """Train the attack on part of the held-out fold, score it on the rest.

    Returns attack accuracy with and without the model's outputs, the
    majority-class rate, and the attack's feature importances.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_known))
    cut = int(split * len(idx))
    tr, ev = idx[:cut], idx[cut:]
    outputs = np.column_stack([model.predict(x_known), model.predict_proba(x_known)])
    with_model = np.column_stack([x_known, outputs])

    def score(features: np.ndarray) -> tuple[float, np.ndarray]:
        atk = RandomForestClassifier(n_estimators=RF_TREES, random_state=seed, n_jobs=-1)
        atk.fit(features[tr], target_bin[tr])
        return float((atk.predict(features[ev]) == target_bin[ev]).mean()), atk.feature_importances_

    acc_model, importances = score(with_model)
    acc_known, _ = score(np.asarray(x_known, float))
    majority = float(np.bincount(target_bin[ev]).max() / len(ev))
    return {"attack_accuracy": acc_model, "known_features_only": acc_known,
            "model_contribution": acc_model - acc_known, "majority_rate": majority,
            "importances": importances}

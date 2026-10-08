"""RQ1: how well do DP-trained models resist inference attacks? (Tables 9-10, Figures 21-23)

For both placements and both mechanisms across the epsilon grid, five seeds
each, the Gaussian NB target is attacked directly:

* membership: rule-based attack on 2,000 members and 2,000 non-members;
* attribute: ECG quartile inferred from the other six features plus the
  model's outputs, trained on 80% of the held-out fold and scored on 20%,
  with the known-features-only attack alongside.

Writes results/rq1_privacy.csv and results/rq1_attribute_importance.csv.
"""
from __future__ import annotations

import pandas as pd

from _common import setting, setup_logging, write_csv
from mlhm.attacks import attribute_inference, membership_rule_based, quartile_bins
from mlhm.config import ATTRIBUTE_TARGET, EPSILONS, FOCUS_EPSILON, MECHANISMS, PLACEMENTS, SEEDS
from mlhm.data import make_split
from mlhm.models import fit

log = setup_logging("rq1")


def main() -> None:
    split = make_split()
    known = split.drop(ATTRIBUTE_TARGET)   # the attribute model never sees ECG
    target = quartile_bins(split.column(ATTRIBUTE_TARGET, "train"), split.column(ATTRIBUTE_TARGET, "test"))
    feature_names = list(known.features) + ["model_prediction", "p_no_af", "p_af"]

    configs = [(None, None, None)] + [(p, m, e) for p in PLACEMENTS for m in MECHANISMS for e in EPSILONS]
    rows, importance = [], []
    for placement, mech, eps in configs:
        for seed in (SEEDS[:1] if eps is None else SEEDS):      # the non-private fit has no noise to vary
            mia_model = fit("nb", placement, mech, eps, split.x_train, split.y_train, seed)
            attr_model = fit("nb", placement, mech, eps, known.x_train, known.y_train, seed)
            att = attribute_inference(attr_model, known.x_test, target, seed=seed)
            rows.append({"placement": placement or "none", "mechanism": mech or "none", "setting": setting(eps),
                         "epsilon": eps, "seed": seed,
                         "membership_success": membership_rule_based(mia_model, split.x_train, split.y_train,
                                                                     split.x_test, split.y_test),
                         **{k: v for k, v in att.items() if k != "importances"}})
            if seed == SEEDS[0] and (eps is None or (placement == "central" and eps == FOCUS_EPSILON)):
                importance += [{"setting": setting(eps), "mechanism": mech or "none", "feature": f, "importance": w}
                               for f, w in zip(feature_names, att["importances"])]
        r = rows[-1]
        log.info(f"{r['placement']:<11} {r['mechanism']:<8} {r['setting']:<11} membership {r['membership_success']:.3f} "
                 f"attribute {r['attack_accuracy']:.3f} (known only {r['known_features_only']:.3f})")

    write_csv(pd.DataFrame(importance), "rq1_attribute_importance")
    log.info(f"wrote {write_csv(pd.DataFrame(rows), 'rq1_privacy')}")


if __name__ == "__main__":
    main()

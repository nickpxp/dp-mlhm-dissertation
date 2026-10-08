"""RQ3: what do DP configurations cost in consistency, compute and across models? (Tables 14-16)

* consistency: P(AF) and predicted labels from the five seeded fits at
  eps = 1.0, per record, for both mechanisms (central placement);
* cost: inference time, throughput and peak memory, non-private vs DP;
* architectures: Gaussian NB, logistic regression and random forest at
  eps = 1.0, both placements and both mechanisms, against each model's own
  non-private accuracy, on the cardiac data.

Writes results/rq3_consistency.csv, rq3_cost.csv, rq3_architectures.csv.
"""
from __future__ import annotations

import platform

import pandas as pd
import sklearn

from _common import setting, setup_logging, write_csv
from mlhm.config import FOCUS_EPSILON, MECHANISMS, PLACEMENTS, SEEDS
from mlhm.data import make_split
from mlhm.evaluation import consistency, inference_cost, utility
from mlhm.models import MODELS, NotApplicable, fit

log = setup_logging("rq3")
ENV = f"{platform.machine()} | {platform.processor() or platform.system()} | Python {platform.python_version()} | sklearn {sklearn.__version__}"


def main() -> None:
    split = make_split()
    base = fit("nb", None, None, None, split.x_train, split.y_train)
    base_pred = base.predict(split.x_test)

    cons, cost = [], [{"implementation": "non-private", **inference_cost(base, split.x_test), "environment": ENV}]
    for mech in MECHANISMS:
        fits = [fit("nb", "central", mech, FOCUS_EPSILON, split.x_train, split.y_train, s) for s in SEEDS]
        cons.append({"mechanism": mech, "epsilon": FOCUS_EPSILON, "n_runs": len(fits),
                     **consistency([m.predict_proba(split.x_test)[:, 1] for m in fits],
                                   [m.predict(split.x_test) for m in fits], base_pred)})
        cost.append({"implementation": f"{mech} eps={FOCUS_EPSILON}", **inference_cost(fits[0], split.x_test),
                     "environment": ENV})
    write_csv(pd.DataFrame(cons), "rq3_consistency")
    write_csv(pd.DataFrame(cost), "rq3_cost")

    rows = []
    for model in MODELS:
        nondp = utility(fit(model, None, None, None, split.x_train, split.y_train, SEEDS[0]), split.x_test, split.y_test)
        rows.append({"model": model, "placement": "none", "mechanism": "none", "setting": "non-private",
                     "seed": SEEDS[0], **nondp})
        for placement in PLACEMENTS:
            for mech in MECHANISMS:
                for seed in SEEDS:
                    try:
                        m = fit(model, placement, mech, FOCUS_EPSILON, split.x_train, split.y_train, seed)
                    except NotApplicable:
                        break
                    rows.append({"model": model, "placement": placement, "mechanism": mech,
                                 "setting": setting(FOCUS_EPSILON), "seed": seed, **utility(m, split.x_test, split.y_test)})
        log.info(f"{model}: done")
    log.info(f"wrote {write_csv(pd.DataFrame(rows), 'rq3_architectures')}")


if __name__ == "__main__":
    main()

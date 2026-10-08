"""RQ2: which (epsilon, delta) balance privacy and utility? (Tables 11-13, Figures 24-26)

* utility: accuracy, precision, recall, F1 of the Gaussian NB for both
  placements and mechanisms across the grid, five seeds;
* mechanism tests: paired t-test, 95% intervals and Pearson r on the
  matched-seed accuracies, Laplace vs Gaussian, at every epsilon;
* statistical queries: noisy feature means, absolute error by mechanism.

Writes results/rq2_utility.csv, rq2_mechanism_tests.csv, rq2_noisy_means.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import setting, setup_logging, write_csv
from mlhm.config import EPSILONS, FEATURES, MECHANISMS, PLACEMENTS, SEEDS
from mlhm.data import load_dataframe, make_split
from mlhm.evaluation import compare_runs, utility
from mlhm.mechanisms import noisy_means
from mlhm.models import fit

log = setup_logging("rq2")


def main() -> None:
    split = make_split()
    rows = [{"placement": "none", "mechanism": "none", "setting": "non-private", "epsilon": None, "seed": SEEDS[0],
             **utility(fit("nb", None, None, None, split.x_train, split.y_train), split.x_test, split.y_test)}]
    for placement in PLACEMENTS:
        for mech in MECHANISMS:
            for eps in EPSILONS:
                for seed in SEEDS:
                    m = fit("nb", placement, mech, eps, split.x_train, split.y_train, seed)
                    rows.append({"placement": placement, "mechanism": mech, "setting": setting(eps),
                                 "epsilon": eps, "seed": seed, **utility(m, split.x_test, split.y_test)})
            log.info(f"{placement} {mech}: done")
    util = pd.DataFrame(rows)
    write_csv(util, "rq2_utility")

    tests = []
    for placement in PLACEMENTS:
        for eps in EPSILONS:
            g = util[(util.placement == placement) & (util.epsilon == eps)]
            lap = g[g.mechanism == "laplace"].sort_values("seed").accuracy.to_numpy()
            gau = g[g.mechanism == "gaussian"].sort_values("seed").accuracy.to_numpy()
            if np.ptp(lap - gau) == 0:   # identical runs (e.g. both collapsed to the majority class)
                tests.append({"placement": placement, "epsilon": eps, "mean_laplace": lap.mean(),
                              "mean_gaussian": gau.mean(), "note": "no variation; test undefined"})
                continue
            r = compare_runs(lap, gau)
            tests.append({"placement": placement, "epsilon": eps, "mean_laplace": r["mean_a"],
                          "mean_gaussian": r["mean_b"], **{k: v for k, v in r.items() if not k.startswith("mean")}})
    write_csv(pd.DataFrame(tests), "rq2_mechanism_tests")

    raw = load_dataframe()[list(FEATURES)].to_numpy(float)
    truth = raw.mean(axis=0)
    q = [{"mechanism": mech, "epsilon": eps, "seed": seed, "feature": f, "true_mean": t, "noisy_mean": v,
          "abs_error": abs(v - t)}
         for mech in MECHANISMS for eps in EPSILONS for seed in SEEDS
         for f, t, v in zip(FEATURES, truth, noisy_means(raw, eps, mech, seed))]
    log.info(f"wrote {write_csv(pd.DataFrame(q), 'rq2_noisy_means')}")


if __name__ == "__main__":
    main()

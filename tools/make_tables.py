"""Measured counterparts of dissertation Tables 8-16, from results/*.csv. No model is fit.

Prints each table and writes them all to results/tables.md. Cells are
mean (SD) over the five seeds unless a table says otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlhm.config import EPSILONS, FOCUS_EPSILON, RESULTS_DIR  # noqa: E402
from mlhm.data import describe, load_dataframe  # noqa: E402

ARMS = (("central", "laplace"), ("central", "gaussian"), ("featurewise", "laplace"), ("featurewise", "gaussian"))
OUT: list[str] = []


def load(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"results/{name}.csv missing; run experiments/run_all.py")
    return pd.read_csv(path)


def ms(s: pd.Series, d: int = 3) -> str:
    return f"{s.mean():.{d}f} ({s.std(ddof=1):.{d}f})" if len(s) > 1 else f"{s.mean():.{d}f}"


def emit(title: str, df: pd.DataFrame, note: str = "") -> None:
    print("\n" + title + "\n" + df.to_string(index=False) + (f"\n{note}" if note else ""))
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    OUT.append(f"### {title}\n\n" + "\n".join(lines) + (f"\n\n{note}" if note else "") + "\n")


def by_eps(df: pd.DataFrame, col: str, ref: float | None = None) -> pd.DataFrame:
    rows = []
    for eps in EPSILONS:
        row = {"epsilon": eps}
        if ref is not None:
            row["non-private"] = f"{ref:.3f}"
        for p, m in ARMS:
            row[f"{p} {m}"] = ms(df[(df.placement == p) & (df.mechanism == m) & (df.epsilon == eps)][col])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    d = describe(load_dataframe())
    emit("Table 8. Distribution of variables in the synthetic dataset",
         d.assign(mean=d["mean"].round(2), sd=d.sd.round(2), range=d["min"].round(2).astype(str) + "-" + d["max"].round(2).astype(str))
          [["variable", "mean", "sd", "range"]])

    p = load("rq1_privacy")
    nd = p[p.setting == "non-private"].iloc[0]
    emit("Table 9. Membership inference success (rule-based attack, balanced accuracy)",
         by_eps(p, "membership_success", nd.membership_success),
         "Chance = 0.500. Values near or below 0.500 are chance plus sampling noise. Feature-wise rows are flat "
         "because the per-feature budget eps/7 leaves the NB predicting the majority class (Table 16).")
    emit("Table 10. Attribute inference success (ECG quartile from the other features plus model outputs)",
         by_eps(p, "attack_accuracy", nd.attack_accuracy),
         f"Majority-class rate {nd.majority_rate:.3f}; the same attack without model outputs reaches "
         f"{p.known_features_only.mean():.3f} on average, so the model adds {p.model_contribution.mean():+.3f}.")
    imp = load("rq1_attribute_importance")
    imp["config"] = imp.setting.where(imp.mechanism == "none", imp.mechanism + " " + imp.setting)
    emit("Figure 21 data. Attack feature importance",
         imp.pivot_table(index="feature", columns="config", values="importance").round(3)
            .sort_values("non-private", ascending=False).reset_index().rename_axis(columns=None))

    u = load("rq2_utility")
    base = u[u.setting == "non-private"].iloc[0]
    for mech, num in (("laplace", 11), ("gaussian", 12)):
        rows = [{"epsilon": e, **{k: ms(u[(u.placement == "central") & (u.mechanism == mech) & (u.epsilon == e)][k], 4)
                                  for k in ("accuracy", "precision", "recall", "f1")}} for e in EPSILONS]
        emit(f"Table {num}. Utility, central {mech.title()} NB", pd.DataFrame(rows),
             f"Non-private: accuracy {base.accuracy:.4f}, F1 {base.f1:.4f}.")

    rows = []
    for e in EPSILONS:
        row = {"epsilon": e}
        for m in ("gaussian", "laplace"):
            g, a = u[(u.placement == "central") & (u.mechanism == m) & (u.epsilon == e)], p[(p.placement == "central") & (p.mechanism == m) & (p.epsilon == e)]
            row |= {f"{m} membership": f"{a.membership_success.mean():.3f}", f"{m} attribute": f"{a.attack_accuracy.mean():.3f}",
                    f"{m} attribute adv.": f"{(a.attack_accuracy - a.majority_rate).mean():+.3f}", f"{m} accuracy": f"{g.accuracy.mean():.4f}"}
        rows.append(row)
    emit("Table 13. Privacy-utility trade-off, central placement", pd.DataFrame(rows))

    t = load("rq2_mechanism_tests")
    t = t[t.placement == "central"]
    emit("Statistical significance. Laplace vs Gaussian accuracy, matched seeds (central)",
         t.assign(**{"Laplace (95% CI)": t.apply(lambda r: f"{r.mean_laplace:.4f} ({r.ci_a_lo:.4f}-{r.ci_a_hi:.4f})", axis=1),
                     "Gaussian (95% CI)": t.apply(lambda r: f"{r.mean_gaussian:.4f} ({r.ci_b_lo:.4f}-{r.ci_b_hi:.4f})", axis=1)})
          [["epsilon", "Laplace (95% CI)", "Gaussian (95% CI)", "t", "p", "pearson_r", "pearson_p"]].round(3))

    q = load("rq2_noisy_means")
    qm = q.groupby(["epsilon", "mechanism"]).abs_error.mean().unstack()
    emit("Figures 25-26 data. Mean absolute error of noisy feature means",
         qm.assign(gaussian_over_laplace=qm.gaussian / qm.laplace).round(5).reset_index())

    c = load("rq3_consistency")
    emit(f"Table 14. Prediction consistency across five fits at eps = {FOCUS_EPSILON}",
         c[["mechanism", "mean_cv_pct", "sd_cv_pct", "median_cv_pct", "p95_cv_pct", "max_cv_pct",
            "decision_agreement_runs", "decision_agreement_nondp"]].round(4),
         "CV of P(AF) per record across the five seeded fits; non-private CV is 0 by construction.")
    k = load("rq3_cost")
    emit("Table 15. Inference cost", k[["implementation", "ms_per_sample", "throughput", "peak_mb"]].round({"ms_per_sample": 5, "peak_mb": 2})
         .astype({"throughput": int}),
         f"Median of {50} full passes over the 20,000-record test fold. {k.environment.iloc[0]}")

    a = load("rq3_architectures")
    rows = []
    for model in ("nb", "lr", "rf"):
        g = a[a.model == model]
        nd_acc = g[g.setting == "non-private"].accuracy.iloc[0]
        row = {"model": model, "non-private acc": f"{nd_acc:.4f}"}
        for pl, mech in ARMS:
            h = g[(g.placement == pl) & (g.mechanism == mech)]
            row[f"{pl} {mech}"] = "n/a" if h.empty else f"{h.accuracy.mean():.4f} / F1 {h.f1.mean():.3f}"
        rows.append(row)
    emit(f"Table 16. Architectures at eps = {FOCUS_EPSILON}: accuracy / F1", pd.DataFrame(rows),
         "n/a: diffprivlib's random forest offers only the Laplace mechanism.")

    (RESULTS_DIR / "tables.md").write_text("# Measured tables\n\nGenerated by tools/make_tables.py from results/*.csv.\n\n" + "\n".join(OUT))
    print(f"\nwrote {RESULTS_DIR / 'tables.md'}")


if __name__ == "__main__":
    main()

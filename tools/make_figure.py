"""One four-panel figure from results/*.csv: utility, membership, attribute, query error vs epsilon.

Measured counterparts of dissertation Figures 17, 22, 23 and 26. Writes
figures/privacy_utility.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlhm.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402

STYLE = {("central", "laplace"): ("#1f4e79", "o", "-"), ("central", "gaussian"): ("#3b7dd8", "s", "-"),
         ("featurewise", "laplace"): ("#b8341f", "o", "--"), ("featurewise", "gaussian"): ("#e07a5f", "s", "--")}


def main() -> None:
    u, p, q = (pd.read_csv(RESULTS_DIR / f"{n}.csv") for n in ("rq2_utility", "rq1_privacy", "rq2_noisy_means"))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    panels = ((ax[0, 0], u, "accuracy", "Test accuracy", "a  Utility (Fig. 17, 24)"),
              (ax[0, 1], p, "membership_success", "Attack balanced accuracy", "b  Membership inference (Fig. 22)"),
              (ax[1, 0], p, "attack_accuracy", "Attack accuracy (ECG quartile)", "c  Attribute inference (Fig. 23)"))
    for a, df, col, ylabel, title in panels:
        for (pl, mech), (c, mk, ls) in STYLE.items():
            g = df[(df.placement == pl) & (df.mechanism == mech)].groupby("epsilon")[col].agg(["mean", "std"])
            a.plot(g.index, g["mean"], color=c, marker=mk, ls=ls, lw=2, ms=5, label=f"{pl}, {mech}")
            a.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.12)
        ref = df[df.setting == "non-private"][col].mean()
        a.axhline(ref, color="gray", ls=":", lw=1.3, label=f"non-private ({ref:.3f})")
        a.set_ylabel(ylabel)
        a.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax[0, 1].set_ylim(0.47, 0.53)
    ax[0, 1].axhline(0.5, color="green", ls=":", lw=1.3, label="chance (0.500)")
    known = p.known_features_only.mean()
    ax[1, 0].axhline(known, color="black", ls="-.", lw=1.2, label=f"no model access ({known:.3f})")
    ax[1, 0].axhline(p.majority_rate.mean(), color="green", ls=":", lw=1.3, label="majority quartile")

    qm = q.groupby(["epsilon", "mechanism"]).abs_error.mean().unstack()
    for mech, c in (("laplace", "#1f4e79"), ("gaussian", "#3b7dd8")):
        ax[1, 1].plot(qm.index, qm[mech], color=c, marker="o" if mech == "laplace" else "s", lw=2, label=mech)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel("Mean |noisy mean - true mean|")
    ax[1, 1].set_title("d  Statistical query error (Fig. 25-26)", loc="left", fontsize=10, fontweight="bold")

    for a in ax.ravel():
        a.set_xscale("log")
        a.set_xticks([0.1, 0.5, 1, 2, 5])
        a.set_xticklabels(["0.1", "0.5", "1", "2", "5"])
        a.set_xlabel(r"$\epsilon$")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "privacy_utility.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

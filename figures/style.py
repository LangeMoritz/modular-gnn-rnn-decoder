"""Shared plotting style and helpers for the paper figures (sized 1:1 for revtex columns)."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import scienceplots  # noqa: F401
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS, OUT = ROOT / "results", ROOT / "figures" / "pdf"
COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]
COLUMN, TEXT = 246 / 72, 510 / 72  # widths in inches
NN, MWPM = dict(marker="^", ls="-"), dict(marker="o", ls=":")

plt.style.use("science")
plt.rcParams.update({"figure.dpi": 300, "axes.prop_cycle": plt.cycler(color=COLORS),
                     "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.bbox": "standard"})


def rows(name):
    """Rows {rounds, p, decoder, p_L, std, shots} written by scripts/evaluate.py."""
    return json.loads((RESULTS / f"{name}.json").read_text())


def curve(rows_, decoder, p=None):
    """Sorted (rounds, p_L, std) arrays of one decoder (and noise strength)."""
    sel = sorted((r["rounds"], r["p_L"], r["std"]) for r in rows_
                 if r["decoder"] == decoder and (p is None or r["p"] == p))
    return tuple(np.array(c) for c in zip(*sel))


def p_l_model(t, eps):
    """p_L(t) = [1 - (1 - 2 eps)^t] / 2 for a logical error per round eps."""
    return 0.5 * (1 - (1 - 2 * eps) ** np.asarray(t, float))


def fit_eps(t, p_l):
    """Fit eps over all durations; returns (eps, standard error)."""
    (eps,), cov = curve_fit(p_l_model, t, p_l, p0=[0.01], bounds=(0, 0.5))
    return eps, float(np.sqrt(cov[0, 0]))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    print(f"saved figures/pdf/{name}.pdf")


def panel_label(ax, text, x=-0.18):
    ax.text(x, 1.03, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")

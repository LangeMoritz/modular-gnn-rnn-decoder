"""Fig. 5: Google 105-qubit memory data, (a) p_L vs. t, (b) eps_d vs. d with Lambda fits."""
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from style import COLORS, RESULTS, TEXT, curve, fit_eps, p_l_model, panel_label, plt, rows, save

# AlphaQubit on the same data, Z basis like our decoder: mean over patches of the
# arXiv:2408.13687 Table S7 values, uncertainty sqrt(sum sigma_i^2) / n
ALPHAQUBIT = {3: (6.16e-3, 0.08e-3), 5: (2.76e-3, 0.05e-3), 7: (1.30e-3, 0.04e-3)}
COLOR = {3: COLORS[0], 5: COLORS[1], 7: COLORS[2]}
FILES = {3: "google_d3_q*.json", 5: "google_d5_q*.json", 7: "google_d7_q6_7_wide.json"}


def per_patch(d, decoder):
    """t, p_L per patch [patches x t], and the mean eps over patches with its error."""
    curves = [curve(rows(Path(f).stem), decoder) for f in sorted(RESULTS.glob(FILES[d]))]
    t = curves[0][0]
    p_l, std = np.array([c[1] for c in curves]), np.array([c[2] for c in curves])
    fits = [fit_eps(t, p) for p in p_l]
    eps = np.mean([e for e, _ in fits])
    err = np.sqrt(sum(s**2 for _, s in fits)) / len(fits)
    return t, p_l, std, eps, err


def fit_lambda(ds, eps):
    model = lambda d, e0, lam: e0 * lam ** (-(np.asarray(d, float) - ds[0]) / 2)  # noqa: E731
    popt, pcov = curve_fit(model, ds, [eps[d] for d in ds], p0=[eps[ds[0]], 2], bounds=([0, 1], [1, 10]))
    return model, popt, np.sqrt(pcov[1, 1])


fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT, 3.3), gridspec_kw={"width_ratios": [2.2, 1]})
eps = {"nn": {}, "mwpm": {}}
t_fit = np.linspace(1, 260, 500)
for d in (3, 5, 7):
    c = COLOR[d]
    t, p_l, std, eps["nn"][d], err_nn = per_patch(d, "nn")
    for p, s in zip(p_l, std):
        ax.errorbar(t, p, yerr=s, fmt="^", ms=4, mfc="none", mec=c, ecolor=c, elinewidth=0.6, capsize=1.5, alpha=0.6)
    ax.errorbar(t, p_l.mean(0), yerr=p_l.std(0) / np.sqrt(len(p_l)), fmt="^", ms=7, color=c, capsize=2.5,
                label=f"$d={d}$ mean", zorder=4)
    ax.plot(t_fit, p_l_model(t_fit, eps["nn"][d]), "-", color=c, lw=1.5)
    t, p_l, _, eps["mwpm"][d], err_mw = per_patch(d, "mwpm")
    ax.errorbar(t, p_l.mean(0), yerr=p_l.std(0) / np.sqrt(len(p_l)), fmt="o", ms=5, mfc="none", mec=c, ecolor=c,
                capsize=2)
    ax.plot(t_fit, p_l_model(t_fit, eps["mwpm"][d]), "--", color=c, lw=1.1)
    bx.errorbar([d], [eps["nn"][d]], yerr=[err_nn], fmt="^", ms=7, color=c, capsize=3, zorder=4)
    bx.errorbar([d], [eps["mwpm"][d]], yerr=[err_mw], fmt="o", ms=6, mfc="none", mec=c, ecolor=c, capsize=2)
    bx.errorbar([d], [ALPHAQUBIT[d][0]], yerr=[ALPHAQUBIT[d][1]], fmt="s", ms=6, mfc="none", mec=c, ecolor=c, capsize=2)
    print(f"d={d}: eps GNN-RNN = {100 * eps['nn'][d]:.3f} +- {100 * err_nn:.3f} %, MWPM = {100 * eps['mwpm'][d]:.3f} %")

ax.set(xlabel=r"QEC cycle, $t$", ylabel=r"Logical error probability, $p_L$", xlim=(0, 260), ylim=(0, 0.55))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [Line2D([], [], color="k", marker="^", mfc="none", ls="none", ms=5),
                     Line2D([], [], color="k", marker="o", mfc="none", ls="--", ms=5)],
          labels + ["individual", "MWPM mean"], loc="upper left", frameon=True)
ax.text(0.97, 0.04, "105Q", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")

d_fit = np.linspace(2.5, 7.5, 200)
for decoder, ls in (("nn", "-"), ("mwpm", "--")):
    model, popt, err = fit_lambda([3, 5, 7], eps[decoder])
    bx.plot(d_fit, model(d_fit, *popt), ls, color="k", lw=1.5 if ls == "-" else 1.1)
    print(f"Lambda {decoder} = {popt[1]:.2f} +- {err:.2f}")
aq_ratio = ALPHAQUBIT[5][0] / ALPHAQUBIT[7][0]
bx.plot(d_fit, ALPHAQUBIT[7][0] * aq_ratio ** ((7 - d_fit) / 2), ":", color="k", lw=1.2)
bx.legend([Line2D([], [], color="k", marker="^", ls="-", ms=6),
           Line2D([], [], color="k", marker="o", mfc="none", ls="--", ms=6),
           Line2D([], [], color="k", marker="s", mfc="none", ls=":", ms=6)],
          ["GNN-RNN", "MWPM", "AlphaQubit"], loc="upper right", frameon=True)
bx.set(yscale="log", xlabel=r"Code distance, $d$", ylabel=r"Logical error per cycle, $\varepsilon_d$", xticks=[3, 5, 7])
bx.xaxis.set_minor_locator(ticker.NullLocator())
bx.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: rf"${100 * y:.2g}\%$"))
panel_label(ax, "(a)", -0.1)
panel_label(bx, "(b)", -0.3)
plt.tight_layout()
save(fig, "fig05_google_epsilon_scaling")

"""Fig. 6: d=3 X-type stability experiment on ibm_berlin.
(a) p_L vs. t at native noise with fits p_L(t) = p_L(t0) Lambda_t^{-(t - t0)/4};
(b) p_L vs. the coherent R_X(alpha) rotation angle at t = 20 and 100."""
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from style import COLORS, COLUMN, MWPM, NN, panel_label, plt, rows, save

data = rows("ibm_stability_d3")


def point(rounds, decoder):
    r = next(r for r in data if str(r["rounds"]) == str(rounds) and r["decoder"] == decoder)
    return r["p_L"], r["std"]


# (a) native noise
fig, ax = plt.subplots(figsize=(COLUMN, 3.0))
t = np.array([5, 10, 20, 50, 100])
for decoder, style in (("mwpm", MWPM), ("nn", NN)):
    p_l, std = np.array([point(r, decoder) for r in t]).T
    model = lambda t_, p0, lam: p0 * lam ** (-(np.asarray(t_, float) - t[0]) / 4)  # noqa: E731
    popt, pcov = curve_fit(model, t, p_l, sigma=std, absolute_sigma=True, p0=[p_l[0], 2], bounds=([0, 1], [1, 50]))
    print(f"{decoder}: Lambda_t = {popt[1]:.3f} +- {np.sqrt(pcov[1, 1]):.3f}")
    ax.errorbar(t, p_l, yerr=std, color="k", marker=style["marker"], ms=5, capsize=3, ls="none")
    t_fit = np.linspace(t[0], t[-1], 200)
    ax.plot(t_fit, model(t_fit, *popt), color="k", ls=style["ls"], lw=1, alpha=0.6)
ax.set(yscale="log", xlabel="QEC cycle, $t$", ylabel="Logical error probability, $p_L$")
ax.margins(x=0.08, y=0.15)
ax.legend([Line2D([], [], color="k", **MWPM), Line2D([], [], color="k", **NN)], ["MWPM", "GNN-RNN"],
          loc="lower left", frameon=True)
panel_label(ax, "(a)")
plt.tight_layout(rect=[0.03, 0, 1, 1])
save(fig, "fig06a_stability_pL_vs_T")

# (b) coherent rotation sweep
fig, ax = plt.subplots(figsize=(COLUMN, 3.0))
alphas = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
for color, t_ in zip(COLORS, (20, 100)):
    for decoder, style in (("mwpm", MWPM), ("nn", NN)):
        p_l, std = np.array([point(t_ if a == 0 else f"{t_}_alpha{a}", decoder) for a in alphas]).T
        ax.errorbar(alphas, p_l, yerr=std, color=color, marker=style["marker"], ms=5, capsize=3, ls="none")
ax.set(yscale="log", xlabel=r"$R_X$ rotation angle, $\alpha$ (rad)", ylabel="Logical error probability, $p_L$")
ax.margins(x=0.05, y=0.15)
ax.legend([Line2D([], [], color=c, marker="s", ls="none") for c in COLORS[:2]]
          + [Line2D([], [], color="k", marker=s["marker"], ls="none") for s in (MWPM, NN)],
          ["$t=20$", "$t=100$", "MWPM", "GNN-RNN"], ncol=2, loc="upper left", frameon=True)
panel_label(ax, "(b)")
plt.tight_layout(rect=[0.03, 0, 1, 1])
save(fig, "fig06b_stability_pL_vs_alpha")

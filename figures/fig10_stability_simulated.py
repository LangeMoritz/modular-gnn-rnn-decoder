"""Fig. 10: simulated d=3 X-type stability experiment, p_L vs. t for several p."""
from matplotlib.lines import Line2D
from style import COLORS, COLUMN, MWPM, NN, curve, plt, rows, save

data = rows("stability_simulated")
ps = sorted({r["p"] for r in data})
fig, ax = plt.subplots(figsize=(COLUMN, 3.0))
for p, color in zip(ps, COLORS):
    for decoder, style in (("mwpm", MWPM), ("nn", NN)):
        t, p_l, std = curve(data, decoder, p)
        keep = p_l > 0
        ax.errorbar(t[keep], p_l[keep], yerr=std[keep], color=color, ms=4, lw=1, capsize=2, **style)
ax.set(yscale="log", xlabel="QEC cycle, $t$", ylabel="Logical error probability, $p_L$")
ax.margins(x=0.05)
ax.set_ylim(bottom=1e-8)  # room for the legends
legend = ax.legend([Line2D([], [], color="k", **MWPM), Line2D([], [], color="k", **NN)], ["MWPM", "GNN-RNN"],
                   loc="lower left", frameon=True)
ax.add_artist(legend)
ax.legend([Line2D([], [], color=c, lw=2) for c in COLORS[:len(ps)]], [f"$p={p:g}$" for p in ps],
          loc="lower right", ncol=2, frameon=True, handlelength=1.2, columnspacing=0.8)
plt.tight_layout()
save(fig, "fig10_stability_sim_nn_vs_mwpm")

"""Fig. 9: simulated memory experiments, p_L vs. t for d = 5, 7, 9 and p = 0.001 ... 0.005."""
from matplotlib.lines import Line2D
from style import COLORS, COLUMN, MWPM, NN, curve, plt, rows, save

PS = [0.001, 0.002, 0.003, 0.004, 0.005]
fig, axes = plt.subplots(3, 1, figsize=(COLUMN, 8.9), sharex=True)
for ax, d in zip(axes, (5, 7, 9)):
    data = rows(f"memory_d{d}")
    for p, color in zip(PS, COLORS):
        for decoder, style, alpha in (("nn", NN, 1), ("mwpm", MWPM, 0.7)):
            t, p_l, std = curve(data, decoder, p)
            keep = p_l > 0
            ax.errorbar(t[keep], p_l[keep], yerr=std[keep], fmt=style["marker"] + style["ls"], color=color,
                        ms=4, lw=1, capsize=3, alpha=alpha)
    ax.set(xscale="log", yscale="log", ylabel="Logical error probability, $p_L$", title=f"$d={d}$")
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
axes[2].set_ylim(bottom=1e-7)
t_all = sorted(set(curve(rows("memory_d9"), "nn", PS[0])[0]))
axes[-1].set(xticks=t_all, xlim=(0.85 * t_all[0], 1.15 * t_all[-1]))
axes[-1].set_xlabel("QEC cycle, $t$")
plt.tight_layout(rect=[0, 0.13, 1, 1])
gap = axes[1].get_position().y0 - axes[2].get_position().y1
fig.legend([Line2D([], [], color=c, lw=2) for c in COLORS[:5]]
           + [Line2D([], [], color="k", ms=4, **NN), Line2D([], [], color="k", ms=4, alpha=0.7, **MWPM)],
           [f"$p={p}$" for p in PS] + ["GNN-RNN", "MWPM"], loc="upper center",
           bbox_to_anchor=(0.5, axes[2].get_position().y0 - gap), ncol=3, frameon=True,
           handlelength=1.3, columnspacing=0.9, handletextpad=0.4)
save(fig, "fig09_simulated_pL_vs_t")

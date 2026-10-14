"""Fig. 11: simulated ZZ merge (rounds 3-3-3), p_L of Z1 xor Z2 xor M vs. physical error rate."""
import numpy as np
from style import COLUMN, MWPM, NN, plt, rows, save

data = rows("merge_simulated")
fig, ax = plt.subplots(figsize=(COLUMN, 3.1))
for decoder, style, label in (("mwpm", MWPM, "MWPM"), ("nn", NN, "GNN-RNN")):
    p, p_l, std = np.array(sorted((r["p"], r["p_L"], r["std"]) for r in data if r["decoder"] == decoder)).T
    ax.errorbar(p, p_l, yerr=std, color="k", ms=4, capsize=3, elinewidth=1, label=label, **style)
ax.set(xscale="log", yscale="log", xlabel=r"Physical error rate, $p$", ylabel=r"Logical error probability, $p_L$")
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
save(fig, "fig11_merge_threshold_mwpm_vs_nn")

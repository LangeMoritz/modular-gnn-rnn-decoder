"""Fig. 4 and Table I: logical error rate vs. duration on ibm_berlin memory experiments."""
import numpy as np
from matplotlib.lines import Line2D
from style import COLORS, COLUMN, MWPM, NN, curve, fit_eps, p_l_model, plt, rows, save

PATCHES = [("ibm_memory_d3_q36", "$d=3$ (q36)"), ("ibm_memory_d3_q85", "$d=3$ (q85)"),
           ("ibm_memory_d5_q85", "$d=5$ (q85)")]

fig, ax = plt.subplots(figsize=(COLUMN, 3.1))
handles = []
for (name, label), color in zip(PATCHES, COLORS):
    data = rows(name)
    for decoder, style in (("mwpm", MWPM), ("nn", NN)):
        t, p_l, std = curve(data, decoder)
        eps, err = fit_eps(t, p_l)
        print(f"{label} {decoder}: eps = {100 * eps:.2f} +- {100 * err:.2f} %")
        h = ax.errorbar(t, p_l, yerr=std, fmt=style["marker"], ms=4, color=color, capsize=3, elinewidth=1)
        t_fit = np.linspace(1, 56, 400)
        ax.plot(t_fit, p_l_model(t_fit, eps), style["ls"], color=color, lw=1.5)
    handles.append(h)
ax.set(xlabel=r"QEC cycle, $t$", ylabel=r"Logical error probability, $p_L$", xlim=(0, 56), ylim=(0, 0.55))
handles += [Line2D([], [], color="k", marker=s["marker"], ls=s["ls"], ms=4) for s in (NN, MWPM)]
ax.legend(handles, [label for _, label in PATCHES] + ["GNN-RNN", "MWPM"], loc="lower right", frameon=True)
plt.tight_layout()
save(fig, "fig04_ibm_memory_pL_vs_t")

"""Fig. 8: Bell-state histogram of the ibm_berlin ZZ merge (from scripts/bell_histogram.py)."""
import json

import numpy as np
from style import COLORS, RESULTS, TEXT, plt, save

d = json.loads((RESULTS / "ibm_zz_merge_bell.json").read_text())
ideal = {"plus": [0.5, 0, 0, 0.5], "minus": [0, 0.5, 0.5, 0]}  # |++> merged in ZZ
bars = [("nominal", COLORS[0]), ("mwpm", COLORS[1]), ("nn", COLORS[2])]
names = {"nominal": "nominal", "mwpm": "MWPM", "nn": "GNN-RNN"}
plt.rcParams["legend.fontsize"] = 8

fig, axes = plt.subplots(1, 2, figsize=(506 / 72, 2.6), sharey=True)
w = 0.2
for ax, sign, title in ((axes[0], "plus", "+1"), (axes[1], "minus", "-1")):
    for i in range(4):  # outcome; the ideal bar only where its probability is non-zero
        series = bars + ([("ideal", COLORS[7])] if ideal[sign][i] else [])
        for k, (name, color) in enumerate(series):
            height = ideal[sign][i] if name == "ideal" else d[f"{name}_{sign}"][i]
            label = None
            if i == 0 and name != "ideal":
                label = f"{names[name]} ({100 * d[f'{name}_accuracy']:.1f}\\% acc.)"
            elif name == "ideal" and i == (0 if sign == "plus" else 1):
                label = "Ideal"
            ax.bar(i + (k - (len(series) - 1) / 2) * w, height, w, color=color, edgecolor="k", lw=0.4, label=label)
    ax.set_xticks(np.arange(4), ["00", "01", "10", "11"])
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.set(xlabel=r"Logical outcome $Z_1, Z_2$", ylim=(0, 0.55),
           title=f"nominal $ZZ = {title}$  ($n$={d[f'nominal_{sign}_shots']})")
axes[0].set_ylabel("Relative count")
axes[0].legend(frameon=True, loc="upper left")
plt.tight_layout()
save(fig, "fig08_bell_hist_mwpm_vs_nn")

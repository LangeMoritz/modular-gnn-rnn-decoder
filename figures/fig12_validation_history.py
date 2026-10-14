"""Fig. 12: training and real-data validation accuracy through Stages 2 and 3 for the d=3
memory patch q36 on ibm_berlin (history of the paper's run, results/ibm_d3_q36_history.json)."""
import json

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator
from style import COLORS, RESULTS, plt, save

plt.rcParams.update({"legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.01})
h = json.loads((RESULTS / "ibm_d3_q36_history.json").read_text())
stage2, stage3 = h["stage2_dem"], h["stage3_real"]
train = [e["accuracy"] for e in stage2 + stage3]
val = [e["val_accuracy"] for e in stage2 + stage3]
epochs = list(range(1, len(train) + 1))

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.plot(epochs, train, color=COLORS[1], lw=1.2, label="Train")
ax.plot(epochs, val, color=COLORS[0], lw=1.6, label="Validation (real data)")
ax.axvline(len(stage2) + 0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
ax.text(0, 1.02, "Stage 2: $p_{ij}$ DEM", ha="left", va="bottom", fontsize=8, transform=ax.transAxes)
ax.text(1, 1.02, "Stage 3: real data", ha="right", va="bottom", fontsize=8, transform=ax.transAxes)
ax.set(xlabel="Epoch", ylabel="Accuracy")
ax.yaxis.set_major_locator(MaxNLocator(5))
lo, hi = min(train + val), max(train + val)
ax.set_ylim(lo - 0.45 * (hi - lo), hi + 0.25 * (hi - lo))
ax.legend(loc="upper left", frameon=True)

n = max(len(stage3), 10) + 5  # inset: the last epochs, where the curves flatten
inset = inset_axes(ax, width="58%", height="33%", loc="lower left", bbox_to_anchor=(0.2, 0.06, 1, 1),
                   bbox_transform=ax.transAxes)
inset.plot(epochs[-n:], train[-n:], color=COLORS[1], lw=1.0)
inset.plot(epochs[-n:], val[-n:], color=COLORS[0], lw=1.4)
inset.axvline(len(stage2) + 0.5, color="gray", ls="--", lw=0.7, alpha=0.7)
zoom = train[-n:] + val[-n:]
pad = 0.01 * (max(zoom) - min(zoom) + 1e-6)
inset.set(xlim=(epochs[-n], epochs[-1]), ylim=(min(zoom) - pad, max(zoom) + pad))
inset.tick_params(labelsize=7)
inset.yaxis.set_major_locator(MaxNLocator(3))
inset.xaxis.set_major_locator(MaxNLocator(3, integer=True))
mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.6", lw=0.6)
plt.tight_layout()
save(fig, "fig12_ibm_val_from_start")
print(f"validation accuracy: start {val[0]:.4f}, end of Stage 2 {stage2[-1]['val_accuracy']:.4f}, "
      f"end of Stage 3 {stage3[-1]['val_accuracy']:.4f}")

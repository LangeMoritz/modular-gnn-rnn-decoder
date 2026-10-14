"""The experiments of the paper: which shots to use and which decoder architecture fits."""
from pathlib import Path

from . import data

# code distance -> tile grids of the modular decoder. The same list goes to grid_tiles
# (which splits the outermost level first) and to Decoder (which merges the innermost
# level first); this works because every list here reads the same in both directions.
TILING = {3: (), 5: [2], 7: [3], 9: [2, 2]}


def distance(experiment, patch=None, d=None):
    if experiment in ("stability", "merge", "ibm-merge"):
        return 3
    if experiment == "google":
        return int(Path(patch).name[1])  # d5_at_q4_7
    if experiment == "ibm":
        return int(Path(patch).name.split("_d")[1][0])  # memory_d5_q85
    return d


def architecture(experiment, d, hidden=256):
    """Constructor arguments of gnnrnn.model.Decoder."""
    if experiment in ("merge", "ibm-merge"):
        return dict(grids=[(1, 3)], n_gnns=3, hidden=hidden)
    return dict(grids=TILING[d], hidden=hidden)


def sources(experiment, *, d=None, rounds=None, ps=None, patch=None, stage="real", part="train"):
    """Shot sources: one per duration for training (simulated sources mix all ps);
    pass a single p to get the source for evaluating that noise strength."""
    if experiment == "memory":
        return [data.memory(d, t, ps, TILING[d]) for t in rounds]
    if experiment == "stability":
        return [data.stability(t, ps) for t in rounds]
    if experiment == "merge":
        return [data.merge(ps)]
    if experiment == "google":
        return data.google(patch, stage, rounds or data.GOOGLE_ROUNDS, part, TILING[distance("google", patch)])
    if experiment == "ibm":
        return data.ibm(patch, stage, rounds or data.IBM_ROUNDS, part, TILING[distance("ibm", patch)])
    if experiment == "ibm-merge":
        return data.ibm_merge(patch, stage, part)
    raise ValueError(experiment)

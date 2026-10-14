"""Syndrome graphs (paper App. B.1) and the tiling of a code into d=3 tiles (App. B.4).

Every detector has integer space-time coordinates (x, y, t). Detection events are
cut into sliding windows of `dt` rounds; inside a window every event becomes a node
with features (x, y, t - window start), and all nodes are connected with weight
1 / ||r_i - r_j||_inf^2.
"""
from dataclasses import dataclass
from functools import cache

import numpy as np
import stim
import torch


@dataclass
class Graphs:
    """All window graphs of one tile for a batch of shots."""
    x: torch.Tensor           # [n_nodes, 3]  node features (x, y, local time)
    edge_index: torch.Tensor  # [2, n_edges]  all ordered node pairs within a graph
    edge_attr: torch.Tensor   # [n_edges]     edge weights
    node_graph: torch.Tensor  # [n_nodes]     graph index of every node
    graph_slot: torch.Tensor  # [n_graphs]    shot * n_windows + window of every graph

    def to(self, device):
        return Graphs(*(v.to(device) for v in vars(self).values()))


def build_graphs(events: np.ndarray, coords: np.ndarray, n_windows: int, dt: int) -> Graphs:
    """events: [shots, n_det] bool, coords: [n_det, 3] int (x, y, t)."""
    shot, det = np.nonzero(events)
    t = coords[det, 2]
    # A detector at time t belongs to the windows that start at t-dt+1, ..., t.
    first = np.maximum(t - dt + 1, 0)
    n_copies = np.minimum(t, n_windows - 1) - first + 1
    offset = np.arange(n_copies.sum()) - np.repeat(np.cumsum(n_copies) - n_copies, n_copies)
    window = np.repeat(first, n_copies) + offset
    shot, det = np.repeat(shot, n_copies), np.repeat(det, n_copies)

    slot = shot * n_windows + window
    order = np.argsort(slot, kind="stable")
    slot, det, window = slot[order], det[order], window[order]
    x = np.column_stack([coords[det, :2], coords[det, 2] - window])
    graph_slot, node_graph, sizes = np.unique(slot, return_inverse=True, return_counts=True)

    src, dst = _all_pairs(sizes)
    dist = np.abs(x[src] - x[dst]).max(axis=1).astype(np.float32)
    weight = np.divide(1.0, dist**2, out=np.zeros_like(dist), where=dist > 0)
    return Graphs(
        x=torch.from_numpy(x.astype(np.float32)),
        edge_index=torch.from_numpy(np.stack([src, dst])),
        edge_attr=torch.from_numpy(weight),
        node_graph=torch.from_numpy(node_graph),
        graph_slot=torch.from_numpy(graph_slot),
    )


def _all_pairs(sizes):
    """Indices of all ordered pairs (i != j) of nodes inside each consecutive group."""
    starts = np.cumsum(sizes) - sizes
    src, dst = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for s in np.unique(sizes[sizes > 1]):
        i, j = np.nonzero(~np.eye(s, dtype=bool))
        first = starts[sizes == s, None]
        src.append((first + i).ravel())
        dst.append((first + j).ravel())
    return np.concatenate(src), np.concatenate(dst)


# ── Tiling ──────────────────────────────────────────────────────────────────

@cache
def _standalone_positions(k: int) -> frozenset:
    """(x, y) positions of the stabilizers of a standalone distance-k rotated surface code."""
    dem = stim.Circuit.generated("surface_code:rotated_memory_z", distance=k, rounds=3,
                                 after_clifford_depolarization=0.001).detector_error_model()
    xy = np.array([c[:2] for c in dem.get_detector_coordinates().values()])
    return frozenset(map(tuple, (xy - xy.min(axis=0)).astype(int).tolist()))


def _crop(coords, idx, x_range, y_range, k=None):
    """Detectors of `idx` inside the box, translated to the origin.

    With k given, only positions that also exist in a standalone distance-k code are
    kept, so that every tile looks to the GNN exactly like a d=k code."""
    x, y = coords[idx, 0], coords[idx, 1]
    inside = np.ones(len(idx), bool)
    for v, (lo, hi) in ((x, x_range), (y, y_range)):
        if lo is not None:
            inside &= v >= lo
        if hi is not None:
            inside &= v <= hi
    idx = idx[inside]
    local = coords[idx].copy()
    local[:, :2] -= local[:, :2].min(axis=0)
    if k is not None:
        valid = _standalone_positions(k)
        keep = np.array([(a, b) in valid for a, b in local[:, :2].tolist()])
        idx, local = idx[keep], local[keep]
    return idx, local


def _grid_ranges(v, n):
    """Overlapping coordinate ranges of an n x n grid along one axis (stabilizer spacing 2)."""
    lo, hi = int(v.min()), int(v.max())
    if n == 2:
        mid = (lo + hi) / 2
        return [(None, mid + 1), (mid - 1, None)]
    step = (hi - lo - 6) // 2  # n == 3: three d=3 tiles
    cut = [lo + step + 1, lo + 2 * step + 1]
    return [(None, cut[0] + 1), (cut[0] - 1, cut[1] + 1), (cut[1] - 1, None)]


def grid_tiles(coords, d, grids):
    """Split a distance-d patch into tiles, recursively for nested grids (outermost first).

    A 2 x 2 grid splits d=2k-1 into four d=k tiles, a 3 x 3 grid splits d=7 into nine
    d=3 tiles. Returns a list of (detector indices, local coords), row-major in y then x
    and outer-tile-major for nested grids (grids=[2, 2] splits d=9 into 4 x 4 d=3 tiles).
    """
    if not grids:
        return [(np.arange(len(coords)), coords)]
    n, inner = grids[0], grids[1:]
    assert n == 2 or (n == 3 and d == 7), "3 x 3 tiling is defined for d=7 only"
    k = (d + 1) // 2 if n == 2 else 3
    idx, tiles = np.arange(len(coords)), []
    for y_range in _grid_ranges(coords[:, 1], n):
        for x_range in _grid_ranges(coords[:, 0], n):
            sub, local = _crop(coords, idx, x_range, y_range, k)
            tiles += [(sub[i], c) for i, c in grid_tiles(local, k, inner)]
    return tiles


def row_tiles(coords, d):
    """Split the merged ZZ strip into [patch 1, bridge, patch 2] along x (unit spacing)."""
    idx, x_max = np.arange(len(coords)), int(coords[:, 0].max())
    mid = (x_max - d) // 2
    return [_crop(coords, idx, r, (None, None)) for r in ((None, d), (mid, mid + d), (x_max - d, None))]

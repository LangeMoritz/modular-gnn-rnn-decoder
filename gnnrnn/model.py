"""The (modular) GNN-RNN decoder, paper Sec. III and App. B.

A GNN encodes every window graph of every d=3 tile into one vector. For the modular
decoder, CNN levels then merge the tile vectors of each window (a 2x2, 3x3 or 1x3
kernel spanning the tile grid, then a 1x1 kernel), level by level. A GRU integrates
the resulting sequence over windows and a linear head predicts the logical flip.

    d=3 base decoder        Decoder()
    d=5 / d=9 memory        Decoder(grids=[2]) / Decoder(grids=[2, 2])
    d=7 memory              Decoder(grids=[3])   (hidden=1024 for the wide variant)
    ZZ merge                Decoder(grids=[(1, 3)], n_gnns=3)
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, global_mean_pool

from .graphs import Graphs


class GNN(nn.Module):
    """Graph convolutions + mean pooling; windows without events get a learned vector."""

    def __init__(self, widths=(3, 64, 256)):
        super().__init__()
        self.convs = nn.ModuleList(GraphConv(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.empty = nn.Parameter(torch.zeros(widths[-1]))

    def forward(self, g: Graphs, n_slots: int):
        x = g.x
        for conv in self.convs:
            x = torch.relu(conv(x, g.edge_index, g.edge_attr))
        out = self.empty.expand(n_slots, -1).clone()
        out[g.graph_slot] = global_mean_pool(x, g.node_graph, size=len(g.graph_slot))
        return out


class Decoder(nn.Module):
    def __init__(self, grids=(), n_gnns=1, hidden=256, gru_layers=4, n_obs=1):
        """grids: tile-grid shapes from the innermost level outwards, e.g. [2, 2] for d=9."""
        super().__init__()
        self.config = dict(grids=list(grids), n_gnns=n_gnns, hidden=hidden,
                           gru_layers=gru_layers, n_obs=n_obs)
        self.grids = [g if isinstance(g, (tuple, list)) else (g, g) for g in grids]
        self.gnns = nn.ModuleList(GNN() for _ in range(n_gnns))
        dim = 256
        self.aggregators = nn.ModuleList()
        for rows, cols in self.grids:
            self.aggregators.append(nn.Sequential(
                nn.Conv2d(dim, hidden, (rows, cols)), nn.ReLU(),
                nn.Conv2d(hidden, hidden, 1), nn.ReLU()))
            dim = hidden
        self.rnn = nn.GRU(dim, hidden, gru_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_obs)

    def forward(self, tiles: list[Graphs], n_shots: int, n_windows: int):
        """Returns logits [n_shots, n_obs] for the logical flip(s) at the end of the experiment."""
        slots = n_shots * n_windows
        h = torch.stack([self.gnns[i % len(self.gnns)](g, slots) for i, g in enumerate(tiles)])
        for (rows, cols), aggregate in zip(self.grids, self.aggregators):
            n, _, dim = h.shape  # n = groups * rows * cols tiles, grouped consecutively
            h = h.view(-1, rows, cols, slots, dim).permute(0, 3, 4, 1, 2).reshape(-1, dim, rows, cols)
            h = aggregate(h).view(n // (rows * cols), slots, -1)
        # cuDNN's GRU cannot run fewer windows than layers (e.g. t=2 stability)
        with torch.backends.cudnn.flags(enabled=n_windows >= self.rnn.num_layers):
            out, _ = self.rnn(h.view(n_shots, n_windows, -1))
        return self.head(out[:, -1])


def save(model: Decoder, path, **info):
    torch.save({"config": model.config, "state_dict": model.state_dict(), **info}, path)


def load(path, device="cpu") -> Decoder:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Decoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def init_from(model: Decoder, path):
    """Warm start: copy every parameter whose name and shape match, e.g. the GNN and GRU
    of a d=3 decoder into a fresh d=5 decoder, or both into a d=9 decoder from d=5."""
    source = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    target = model.state_dict()
    copied = [k for k, v in source.items() if k in target and target[k].shape == v.shape]
    model.load_state_dict({k: source[k] for k in copied}, strict=False)
    print(f"initialized {len(copied)}/{len(target)} tensors from {path}")

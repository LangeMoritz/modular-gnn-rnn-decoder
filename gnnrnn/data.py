"""Shot sources: simulated shots from stim detector error models (DEMs) and recorded
hardware shots (Google, IBM). Every source yields (detection events, logical flips)
together with the tiled detector layout needed to turn events into graphs.
"""
from pathlib import Path

import numpy as np
import stim

from . import circuits
from .graphs import build_graphs, grid_tiles, row_tiles

# Durations of the Google experiments (t=1 has no complete window of dt=2 rounds; unused)
GOOGLE_ROUNDS = [10, 13, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
IBM_ROUNDS = [5, 10, 20, 30, 50]


class Layout:
    """Detector coordinates of one experiment, split into the tiles the model expects:
    `tiles` lists (detector indices, local coords) per tile."""

    def __init__(self, coords, tiling=(), d=3, dt=2):
        """tiling: () for the base decoder, a list of grid sizes for square patches
        ([2] for d=5, [3] for d=7, [2, 2] for d=9) or "row" for the ZZ merge."""
        self.dt = dt
        self.n_windows = int(coords[:, 2].max()) - dt + 2
        self.tiles = row_tiles(coords, d) if tiling == "row" else grid_tiles(coords, d, list(tiling))

    def graphs(self, events, device="cpu"):
        return [build_graphs(events[:, idx], c, self.n_windows, self.dt).to(device)
                for idx, c in self.tiles]


class Simulated:
    """Unlimited shots from one or more DEMs, mixed in equal shares within a batch
    (e.g. several physical error rates)."""

    def __init__(self, dems, layout, seed=None):
        self.dems, self.layout = dems, layout
        self.samplers = [dem.compile_sampler(seed=seed) for dem in dems]

    def sample(self, n, nontrivial=True):
        """n shots; with nontrivial=True only shots with at least one detection event."""
        events, labels = [], []
        for i, sampler in enumerate(self.samplers):
            share = n // len(self.samplers) + (i < n % len(self.samplers))
            got = 0
            while got < share:
                det, obs, _ = sampler.sample(max(2 * (share - got), 64))
                if nontrivial:
                    keep = det.any(axis=1)
                    det, obs = det[keep], obs[keep]
                det, obs = det[:share - got], obs[:share - got]
                events.append(det)
                labels.append(obs)
                got += len(det)
        return np.concatenate(events), np.concatenate(labels).astype(np.float32)


class Recorded:
    """A fixed set of recorded shots. Shots without any detection event are kept aside:
    the decoder predicts "no flip" for them. `part` selects the reproducible random
    80/10/10 train/val/test split (taken separately among trivial and non-trivial shots)."""

    def __init__(self, events, labels, layout, part="all", seed=42, dem_path=None):
        nontrivial = events.any(axis=1)
        events, trivial_labels, labels = events[nontrivial], labels[~nontrivial], labels[nontrivial]
        if part != "all":
            keep = split_indices(len(events), part, seed)
            events, labels = events[keep], labels[keep]
            trivial_labels = trivial_labels[split_indices(len(trivial_labels), part, seed)]
        self.events, self.labels = events, labels.astype(np.float32)
        self.trivial_labels = trivial_labels.astype(np.float32)
        self.layout, self.rng = layout, np.random.default_rng()
        self.dem_path = dem_path  # calibrated detector error model of these shots (for MWPM)

    def sample(self, n):
        """n random non-trivial shots (with replacement), for training."""
        i = self.rng.integers(len(self.events), size=n)
        return self.events[i], self.labels[i]

    def batches(self, n):
        """All non-trivial shots in order, in chunks of n, for evaluation."""
        for i in range(0, len(self.events), n):
            yield self.events[i:i + n], self.labels[i:i + n]


def split_indices(n, part, seed, fractions=(0.8, 0.1)):
    """Indices (in file order) of one part of a seeded random train/val/test split."""
    order = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(n * fractions[0]), int(n * fractions[1])
    bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}
    return np.sort(order[slice(*bounds[part])])


def _read_b8(path, n_bits):
    return stim.read_shot_data_file(path=str(path), format="b8", num_measurements=n_bits)


def _stim_coords(dem):
    """Detector coordinates of a stim DEM, shifted to start at 0 (half-integers become integers)."""
    coords = np.array(list(dem.get_detector_coordinates().values()))
    return (coords - coords.min(axis=0)).astype(np.int64)


# ── Simulated experiments ────────────────────────────────────────────────────

def memory(d, rounds, ps, tiling=()):
    dems = [circuits.memory_circuit(d, rounds, p).detector_error_model(decompose_errors=True) for p in ps]
    return Simulated(dems, Layout(_stim_coords(dems[0]), tiling, d))


def stability(rounds, ps):
    dems = [circuits.stability_circuit(rounds, p).detector_error_model(decompose_errors=True) for p in ps]
    return Simulated(dems, Layout(_stim_coords(dems[0])))


def merge(ps, rounds=(3, 3, 3)):
    dems = [circuits.merge_circuit(rounds, p).detector_error_model(decompose_errors=True) for p in ps]
    return Simulated(dems, Layout(_stim_coords(dems[0]), "row"))


# ── Google 105-qubit processor (Acharya et al., Nature 2025) ─────────────────

def google(patch_dir, stage, rounds=GOOGLE_ROUNDS, part="train", tiling=()):
    """One source per experiment duration for a patch directory such as .../d5_at_q4_7.

    stage "si1000": the patch's SI1000 circuit with all error rates x3 (50 rounds only),
          "dem":    the patch's calibrated p_ij detector error model,
          "real":   the recorded detection events."""
    patch_dir = Path(patch_dir)
    d = int(patch_dir.name[1])
    if stage == "si1000":
        rounds = [50]
    sources = []
    for t in rounds:
        r_dir = patch_dir / "Z" / f"r{t:02d}"
        ideal = stim.Circuit.from_file(r_dir / "circuit_ideal.stim")
        layout = Layout(_google_coords(ideal.detector_error_model()), tiling, d)
        if stage == "si1000":
            noisy = r_dir / "circuit_noisy_si1000_p1.stim"  # Google's unscaled SI1000 model
            sources.append(Simulated([circuits.scale_noise(stim.Circuit.from_file(noisy), 3)
                                      .detector_error_model()], layout))
        elif stage == "dem":
            dem = stim.DetectorErrorModel.from_file(r_dir / _GOOGLE_DEM)
            sources.append(Simulated([dem], layout))
        else:
            events = _read_b8(r_dir / "detection_events.b8", ideal.num_detectors)
            labels = _read_b8(r_dir / "obs_flips_actual.b8", 1)
            sources.append(Recorded(events, labels, layout, part, dem_path=r_dir / _GOOGLE_DEM))
    return sources


_GOOGLE_DEM = "decoding_results/pij_model/error_model.dem"


def _google_coords(dem):
    """Google's detectors carry (x, y, t) in the bulk and longer tuples for the final data
    qubit layer; rotate the grid by 45 degrees (u = x+y, v = x-y) to stim's convention."""
    rows = []
    for c in dem.get_detector_coordinates().values():
        if len(c) in (3, 6):
            rows.append(c[:3])
        else:  # final layer: last triple, one time step later
            rows.append([c[-3], c[-2], c[-1] + 1])
    xyt = np.array(rows)
    xyt -= xyt.min(axis=0)
    uv = np.stack([xyt[:, 0] + xyt[:, 1], xyt[:, 0] - xyt[:, 1]], axis=1)
    return np.column_stack([np.round(uv - uv.min(axis=0)), xyt[:, 2]]).astype(np.int64)


# ── IBM ibm_berlin ───────────────────────────────────────────────────────────

# Physical measure qubits (row-major index q = 10 * row + col) and the positions in that
# list of the X-type ones; detector order: Z checks, then all checks per bulk round,
# then Z checks again from the final data-qubit measurement.
IBM_PATCHES = {
    "memory_d3_q36": ([17, 48, 37, 26, 46, 35, 24, 55], {0, 2, 5, 7}),
    "memory_d3_q85": ([66, 97, 86, 75, 95, 84, 73, 104], {0, 2, 5, 7}),
    "memory_d5_q85": ([68, 46, 99, 88, 77, 66, 55, 97, 86, 75, 64, 53, 117, 106, 95, 84, 73,
                       115, 104, 93, 82, 71, 102], {0, 1, 3, 5, 8, 10, 13, 15, 18, 20, 22}),
    "stability_d3": ([73, 75, 84, 93, 77, 86, 95, 97], {0, 1, 3, 4, 6, 7}),
}


def ibm(data_dir, stage, rounds=IBM_ROUNDS, part="train", tiling=()):
    """One source per duration for an ibm_berlin experiment directory such as
    data/ibm_berlin/memory_d3_q36. Durations may carry a variant suffix, e.g. "20_alpha0.15".

    stage "dem": the calibrated p_ij detector error model, "real": the recorded shots."""
    data_dir = Path(data_dir)
    ancillas, x_type = IBM_PATCHES[data_dir.name]
    d = int(data_dir.name.split("_d")[1][0])
    sources = []
    for entry in rounds:
        t = int(str(entry).split("_")[0])
        t_dir = data_dir / f"T{entry}"
        # d=3 decoders see the chip's (row, col) grid; the tiled d=5 decoder sees stim's rotated frame
        layout = Layout(_ibm_coords(ancillas, x_type, t, rotate=bool(tiling)), tiling, d)
        if stage == "dem":
            sources.append(Simulated([stim.DetectorErrorModel.from_file(t_dir / "error_model.dem")], layout))
        else:
            n_det = 2 * (len(ancillas) - len(x_type)) + (t - 1) * len(ancillas)
            events = _read_b8(t_dir / "detection_events.b8", n_det)
            sources.append(Recorded(events, _read_b8(t_dir / "logical_flips.b8", 1), layout, part,
                                    dem_path=t_dir / "error_model.dem"))
    return sources


def _ibm_coords(ancillas, x_type, t, rotate=False):
    row, col = np.array(ancillas) // 10, np.array(ancillas) % 10
    x, y = (row + col, row - col) if rotate else (row, col)
    xy = np.stack([x - x.min(), y - y.min()], axis=1)
    z = [i for i in range(len(ancillas)) if i not in x_type]
    layers = [(z, 0)] + [(range(len(ancillas)), r) for r in range(1, t)] + [(z, t)]
    return np.array([[*xy[i], r] for idx, r in layers for i in idx], dtype=np.int64)


def ibm_merge(data_dir, stage, part="train"):
    """The 1-3-1 ZZ-merge run on ibm_berlin (data/ibm_berlin/zz_merge_1-3-1)."""
    data_dir = Path(data_dir)
    circuit = circuits.merge_circuit((1, 3, 1), p=0.01)  # defines the detector structure
    layout = Layout(_stim_coords(circuit.detector_error_model()), "row")
    if stage == "dem":
        return [Simulated([stim.DetectorErrorModel.from_file(data_dir / "error_model.dem")], layout)]
    events = _read_b8(data_dir / "detection_events.b8", circuit.num_detectors)
    return [Recorded(events, _read_b8(data_dir / "obs_flips.b8", 1), layout, part,
                     dem_path=data_dir / "error_model.dem")]

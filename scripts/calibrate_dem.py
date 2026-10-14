"""Calibrated p_ij detector error model (paper Sec. III C, Stage 2).

Takes the error structure (which detectors and observables each error mechanism flips)
from a template circuit and estimates every probability from the correlations of
recorded detection events with the closed-form expressions of Appendix E of
arXiv:2502.17722, for errors flipping up to four detectors.

  python scripts/calibrate_dem.py --circuit structure.stim --events detection_events.b8 --out error_model.dem
"""
import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
import stim


def calibrate_dem(circuit: stim.Circuit, events: np.ndarray) -> stim.DetectorErrorModel:
    errors = []  # (detectors, observables) of every error mechanism of the template
    for inst in circuit.detector_error_model().flattened():
        if inst.type == "error":
            t = inst.targets_copy()
            errors.append((tuple(sorted(x.val for x in t if x.is_relative_detector_id())),
                           [x.val for x in t if x.is_logical_observable_id()]))
    by_size = defaultdict(set)
    for dets, _ in errors:
        by_size[len(dets)].add(dets)
    spins = 1.0 - 2.0 * np.asarray(events, dtype=np.float64)  # detection event -> -1
    cache = {}

    def corr(dets):  # <prod_i (1 - 2 x_i)>
        if dets not in cache:
            cache[dets] = float(np.prod(spins[:, list(dets)], axis=1).mean())
        return cache[dets]

    def subsets(dets, k):
        return [tuple(sorted(c)) for c in combinations(dets, k)]

    probs = {}

    def prob(dets):
        """Eqs. (E1)-(E4): ratio of correlators, corrected for all larger errors containing dets."""
        if dets in probs:
            return probs[dets]
        n = len(dets)  # odd-size sub-correlators over even-size ones
        num = np.prod([corr(s) for k in range(1, n + 1, 2) for s in subsets(dets, k)])
        den = np.prod([corr(s) for k in range(2, n + 1, 2) for s in subsets(dets, k)])
        if n == 1:
            ratio = num
        elif den == 0 or num / den < 0:
            probs[dets] = 0.0
            return 0.0
        else:
            ratio = (num / den) ** (1 / 2 ** (n - 1))
        correction = 1.0
        for size in range(n + 1, 5):
            for larger in by_size[size]:
                if set(dets) <= set(larger) and (v := prob(larger)) < 0.5:
                    correction /= 1 - 2 * v
        probs[dets] = 0.5 - 0.5 * ratio * correction
        return probs[dets]

    dem = stim.DetectorErrorModel()
    for dets, obs in errors:
        p = prob(dets) if 1 <= len(dets) <= 4 else 0.0
        targets = [stim.target_relative_detector_id(d) for d in dets]
        dem.append("error", min(max(p, 1e-10), 0.5 - 1e-10),
                   targets + [stim.target_logical_observable_id(o) for o in obs])
    return dem


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--circuit", required=True, help="template circuit with the right detectors")
    p.add_argument("--events", required=True, help="detection_events.b8")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    circuit = stim.Circuit.from_file(args.circuit)
    events = stim.read_shot_data_file(path=args.events, format="b8", num_detectors=circuit.num_detectors)
    calibrate_dem(circuit, events).to_file(args.out)

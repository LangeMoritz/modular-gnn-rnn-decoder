"""Bell-state histogram of the ibm_berlin ZZ merge (paper Fig. 8): joint (Z1, Z2) outcome
frequencies on the held-out test shots, split by the merge outcome M, before correction
and after the MWPM or GNN-RNN correction of M (only Z1 xor Z2 xor M is decodable).

  python scripts/bell_histogram.py --data data/ibm_berlin/zz_merge_1-3-1 \\
      --model models/ibm_zz_merge.pt --out results/ibm_zz_merge_bell.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stim
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gnnrnn import circuits, data  # noqa: E402
from gnnrnn.model import load  # noqa: E402
from gnnrnn.train import mwpm_decoder, nn_decoder  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", required=True)
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

root = Path(args.data)
n_det = circuits.merge_circuit((1, 3, 1)).num_detectors
events = stim.read_shot_data_file(path=str(root / "detection_events.b8"), format="b8", num_detectors=n_det)
labels = stim.read_shot_data_file(path=str(root / "obs_flips.b8"), format="b8", num_measurements=1)[:, 0]
z1, z2, m = stim.read_shot_data_file(path=str(root / "z1_z2_m.b8"), format="b8", num_measurements=3).T

# the held-out test shots: the same split as gnnrnn.data.Recorded, among shots with and without events
nontrivial = events.any(axis=1)
test = np.sort(np.concatenate([np.flatnonzero(pool)[data.split_indices(pool.sum(), "test", 42)]
                               for pool in (nontrivial, ~nontrivial)]))
events, labels, z1, z2, m = events[test], labels[test], z1[test], z2[test], m[test]

device = "cuda" if torch.cuda.is_available() else "cpu"
layout = data.ibm_merge(root, "real", part="test")[0].layout
flips = {"nominal": np.zeros(len(test), bool),
         "mwpm": mwpm_decoder(stim.DetectorErrorModel.from_file(root / "error_model.dem"))(events)[:, 0],
         "nn": nn_decoder(load(args.model, device), layout, device)(events)[:, 0]}
result = {"shots": len(test)}
for name, flip in flips.items():
    m_corrected = m ^ flip
    result[f"{name}_accuracy"] = float((flip == labels).mean())
    for sign, mask in (("plus", ~m_corrected), ("minus", m_corrected)):
        outcome = 2 * z1[mask].astype(int) + z2[mask].astype(int)  # 00, 01, 10, 11
        result[f"{name}_{sign}"] = (np.bincount(outcome, minlength=4) / mask.sum()).tolist()
        result[f"{name}_{sign}_shots"] = int(mask.sum())
    print(f"{name}: accuracy {result[f'{name}_accuracy']:.4f}")
Path(args.out).write_text(json.dumps(result, indent=1))

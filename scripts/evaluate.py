"""Logical error rates of a trained decoder and/or MWPM, written as a JSON list of rows
{rounds, p, decoder, p_L, std, shots}. Hardware experiments are evaluated on the held-out
test shots, simulated ones on fresh samples shared by both decoders. Examples:

  python scripts/evaluate.py --experiment google --patch data/google/d5_at_q4_7 \\
      --model models/google_d5_q4_7.pt --mwpm --out results/google_d5_q4_7.json
  python scripts/evaluate.py --experiment memory --d 9 --rounds 5 10 20 50 100 200 500 1000 \\
      --p 0.001 0.003 0.005 --shots 1000000 --model models/memory_d9.pt --mwpm --out results/memory_d9.json
"""
import argparse
import json
import sys
from pathlib import Path

import stim
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gnnrnn import experiments  # noqa: E402
from gnnrnn.data import GOOGLE_ROUNDS, IBM_ROUNDS, Recorded  # noqa: E402
from gnnrnn.model import load  # noqa: E402
from gnnrnn.train import logical_error_rates, mwpm_decoder, nn_decoder  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--experiment", required=True,
               choices=["memory", "stability", "merge", "google", "ibm", "ibm-merge"])
p.add_argument("--d", type=int, default=3)
p.add_argument("--rounds", nargs="+")
p.add_argument("--p", type=float, nargs="+", default=[None])
p.add_argument("--patch")
p.add_argument("--model", help="trained decoder checkpoint")
p.add_argument("--mwpm", action="store_true", help="also evaluate MWPM on the same shots")
p.add_argument("--shots", type=int, default=1_000_000, help="maximum shots per point (simulated experiments)")
p.add_argument("--rel-error", type=float, default=0.01,
               help="simulated: stop once every decoder's p_L has this relative standard error")
p.add_argument("--batch-size", type=int, default=4096)
p.add_argument("--out", required=True)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load(args.model, device) if args.model else None
rounds = [int(r) if r.isdigit() else r for r in args.rounds] if args.rounds else None
d = experiments.distance(args.experiment, args.patch, args.d)
rows = []
for p_phys in args.p:  # one noise strength at a time (hardware: a single None)
    spec = dict(d=d, rounds=rounds, ps=[p_phys], patch=args.patch, stage="real", part="test")
    sources = experiments.sources(args.experiment, **spec)
    labels = rounds or {"google": GOOGLE_ROUNDS, "ibm": IBM_ROUNDS}.get(args.experiment, [None])
    for t, source in zip(labels, sources):
        decoders = {}
        if model:
            decoders["nn"] = nn_decoder(model, source.layout, device)
        if args.mwpm:
            dem = stim.DetectorErrorModel.from_file(source.dem_path) if isinstance(source, Recorded) else source.dems[0]
            decoders["mwpm"] = mwpm_decoder(dem)
        rounds_ = int(str(t).split("_")[0]) if t else 1
        batch = min(args.batch_size, max(256, 400_000 // rounds_))  # long experiments: fewer shots per batch
        for name, (p_l, std, n) in logical_error_rates(decoders, source, args.shots, batch, args.rel_error).items():
            rows.append(dict(rounds=t, p=p_phys, decoder=name, p_L=p_l, std=std, shots=n))
            print(f"rounds={t} p={p_phys} {name}: p_L = {p_l:.5f} +- {std:.5f} ({n} shots)", flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(rows, indent=1))

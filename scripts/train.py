"""Train or fine-tune a decoder. Examples (see README for the full recipes):

  # Stage 1: d=3 base decoder on simulated memory experiments
  python scripts/train.py --experiment memory --d 3 --rounds 50 --p 0.001 0.002 0.003 0.004 0.005 \\
      --epochs 500 --lr 1e-3 --out models/memory_d3.pt
  # Modular d=5 decoder, warm-started from the d=3 GNN and GRU
  python scripts/train.py --experiment memory --d 5 ... --init models/memory_d3.pt --out models/memory_d5.pt
  # Stages 2 and 3 on hardware: calibrated detector error model, then recorded shots
  python scripts/train.py --experiment google --patch data/google/d5_at_q4_7 --stage dem  --init ... --lr 1e-4
  python scripts/train.py --experiment google --patch data/google/d5_at_q4_7 --stage real --init ... --lr 1e-5
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gnnrnn import experiments  # noqa: E402
from gnnrnn.model import Decoder, init_from  # noqa: E402
from gnnrnn.train import train  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--experiment", required=True,
               choices=["memory", "stability", "merge", "google", "ibm", "ibm-merge"])
p.add_argument("--d", type=int, default=3, help="code distance (simulated memory)")
p.add_argument("--rounds", nargs="+", help="experiment durations; repeat one to weight it more")
p.add_argument("--p", type=float, nargs="+", help="physical error rates (simulated experiments)")
p.add_argument("--patch", help="hardware data directory (google, ibm, ibm-merge)")
p.add_argument("--stage", choices=["si1000", "dem", "real"], default="real",
               help="hardware experiments: train on the SI1000 circuit, the calibrated DEM or real shots")
p.add_argument("--init", help="checkpoint to start from (matching parameters are copied)")
p.add_argument("--hidden", type=int, default=256, help="GRU/aggregation width (1024 for the wide d=7)")
p.add_argument("--epochs", type=int, required=True)
p.add_argument("--steps", type=int, default=256, help="optimizer steps per epoch")
p.add_argument("--batch-size", type=int, default=2048)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--min-lr", type=float, help="learning-rate floor (default lr / 10)")
p.add_argument("--out", required=True)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
rounds = [int(r) if r.isdigit() else r for r in args.rounds] if args.rounds else None
d = experiments.distance(args.experiment, args.patch, args.d)
spec = dict(d=d, rounds=rounds, ps=args.p, patch=args.patch, stage=args.stage)
train_sources = experiments.sources(args.experiment, part="train", **spec)
val_sources = None
if args.experiment in ("google", "ibm", "ibm-merge"):  # validate on held-out real shots
    val_sources = experiments.sources(args.experiment, **{**spec, "stage": "real"}, part="val")

model = Decoder(**experiments.architecture(args.experiment, d, args.hidden)).to(device)
if args.init:
    init_from(model, args.init)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(train_sources)} training sources")

history = train(model, train_sources, epochs=args.epochs, steps=args.steps, batch_size=args.batch_size,
                lr=args.lr, min_lr=args.min_lr, val_sources=val_sources, save_path=args.out, device=device)
Path(args.out).with_suffix(".json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))

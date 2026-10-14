# Modular GNN-RNN decoder

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23335425.svg)](https://doi.org/10.5281/zenodo.23335425)

Code, trained models and data for

> M. Lange, O. Fjeldså, G. Jonasson Johansson, M. Granath,
> *Modular Neural Decoder for Surface Code Memory and Logic on Superconducting Processors*,
> [arXiv:2610.03869](https://arxiv.org/abs/2610.03869).

A graph neural network (GNN) encodes the detection events of every window of Δt = 2
rounds of a distance-3 surface code into a vector, and a GRU integrates these vectors
over time to predict the logical outcome. Larger codes are split into overlapping d=3
tiles whose encodings a small CNN merges level by level (2×2 tiles for d=5, 2×2 of 2×2
for d=9, 3×3 for d=7, a row of 1×3 tiles for a ZZ lattice-surgery merge). The same
decoder handles memory, stability and lattice-surgery experiments, in simulation and on
Google and IBM hardware.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training needs a CUDA GPU (all paper models were trained on one NVIDIA GH200);
evaluating the released models also works on a CPU.

## Code

| | |
|---|---|
| `gnnrnn/graphs.py` | detection events → window graphs; tiling of a code into d=3 tiles |
| `gnnrnn/model.py` | the decoder (GNN, tile aggregation, GRU, head) |
| `gnnrnn/data.py` | shot sources: simulated (stim) and recorded (Google, IBM) shots |
| `gnnrnn/circuits.py` | memory, stability and ZZ-merge circuits with circuit-level noise |
| `gnnrnn/train.py` | training loop, logical error rates, MWPM baseline |
| `gnnrnn/experiments.py` | the experiments of the paper and their decoder architectures |
| `scripts/train.py` | train / fine-tune a decoder (any stage, any experiment) |
| `scripts/evaluate.py` | logical error rates of a decoder and MWPM → `results/*.json` |
| `scripts/bell_histogram.py` | Bell-state histogram of the hardware ZZ merge |
| `scripts/calibrate_dem.py` | p_ij detector error model from recorded detection events |
| `scripts/download_models.py` | fetch the trained models |
| `figures/` | one script per result figure of the paper, reading `results/` |
| `circuits/` | noiseless stability and ZZ-merge circuits (noise is added in code) |

## Data

* **Google** (Fig. 5): the public data of Google Quantum AI,
  *Quantum error correction below the surface code threshold*, Nature 638, 920 (2025),
  [doi:10.5281/zenodo.13273331](https://doi.org/10.5281/zenodo.13273331), unpacked so that
  `data/google/d5_at_q4_7/Z/r50/detection_events.b8` exists. Stages 2 and 3 additionally
  need our calibrated p_ij detector error models
  `data/google/<patch>/Z/r<t>/decoding_results/pij_model/error_model.dem`: unzip
  `google_pij_dems.zip` from our [Zenodo record](https://doi.org/10.5281/zenodo.23335425) into `data/google/`, or rebuild them
  with `scripts/calibrate_dem.py` from Google's `circuit_noisy_si1000*.stim` and
  `detection_events.b8`.
* **IBM `ibm_berlin`** (Figs. 4, 6, 8, 12): our memory, stability and ZZ-merge runs,
  [doi:10.5281/zenodo.23335425](https://doi.org/10.5281/zenodo.23335425) (`ibm_berlin.zip`, unzip into `data/`):
  `memory_d3_q36`, `memory_d3_q85`, `memory_d5_q85`, `stability_d3` (one directory per
  duration `T<t>`, with `_alpha<angle>` for the coherent-rotation sweep) and
  `zz_merge_1-3-1`, each with `detection_events.b8`, the logical flips and the calibrated
  `error_model.dem`; `raw_jobs/` holds the raw IBM Runtime results.

Every recorded data set is split 80/10/10 into training, validation and test shots by a
fixed random permutation (seed 42); all reported hardware numbers use the test shots.

## Models

`python scripts/download_models.py` fetches all models from our
[Zenodo record](https://doi.org/10.5281/zenodo.23335425) into `models/` and checks their SHA-256 hashes:

| Model | Figure | Architecture |
|---|---|---|
| `memory_d3.pt` | – (base of the memory and hardware decoders) | d=3 GNN-RNN |
| `memory_d5.pt`, `memory_d7.pt`, `memory_d9.pt` | 9 | 2×2, 3×3, 2×2 of 2×2 tiles |
| `stability_d3.pt` | 10 | d=3 GNN-RNN |
| `merge_d3.pt` | 11 | 1×3 tiles, three GNNs |
| `google_si1000_d3.pt` | – (Stage 1 for Google d=3 and IBM d=3) | d=3 GNN-RNN |
| `google_d3_q<patch>.pt` (9), `google_d5_q<patch>.pt` (4) | 5 | d=3 / 2×2 tiles |
| `google_d7_q6_7.pt`, `google_d7_q6_7_wide.pt` | 5 (text), 5 | 3×3 tiles, GRU width 256 / 1024 |
| `ibm_memory_d3_q36.pt`, `ibm_memory_d3_q85.pt`, `ibm_memory_d5_q85.pt` | 4, Table I | d=3 / 2×2 tiles |
| `ibm_stability_d3.pt` | 6 | d=3 GNN-RNN |
| `ibm_zz_merge.pt` | 8 | 1×3 tiles, three GNNs |

## Reproducing the figures

`results/` contains the evaluation output behind every figure, so the figures can be
redrawn directly, e.g. `python figures/fig05_google.py` (PDFs go to `figures/pdf/`).
To regenerate the results from the models and data:

```bash
# hardware: decoder and MWPM on the held-out test shots
for m in models/google_d[35]_q*.pt models/google_d7_q6_7.pt models/google_d7_q6_7_wide.pt; do
  n=$(basename $m .pt); d=${n:8:1}; q=${n#google_d?_}; q=${q%_wide}
  python scripts/evaluate.py --experiment google --patch data/google/d${d}_at_${q} --model $m --mwpm --out results/$n.json
done
for p in memory_d3_q36 memory_d3_q85 memory_d5_q85; do
  python scripts/evaluate.py --experiment ibm --patch data/ibm_berlin/$p --model models/ibm_$p.pt --mwpm --out results/ibm_$p.json
done
python scripts/evaluate.py --experiment ibm --patch data/ibm_berlin/stability_d3 --model models/ibm_stability_d3.pt --mwpm \
    --rounds 5 10 20 50 100 $(for t in 20 100; do for a in 0.15 0.3 0.45 0.6 0.8 1.0; do echo ${t}_alpha$a; done; done) \
    --out results/ibm_stability_d3.json
python scripts/bell_histogram.py --data data/ibm_berlin/zz_merge_1-3-1 --model models/ibm_zz_merge.pt --out results/ibm_zz_merge_bell.json

# simulation (GPU recommended)
for d in 5 7 9; do
  python scripts/evaluate.py --experiment memory --d $d --model models/memory_d$d.pt --mwpm \
      --rounds 5 10 20 50 100 200 500 1000 --p 0.001 0.002 0.003 0.004 0.005 --out results/memory_d$d.json
done
python scripts/evaluate.py --experiment stability --model models/stability_d3.pt --mwpm \
    --rounds 2 5 10 20 50 --p 0.001 0.003 0.005 0.01 0.02 0.03 --out results/stability_simulated.json
python scripts/evaluate.py --experiment merge --model models/merge_d3.pt --mwpm \
    --p 0.001 0.001292 0.001668 0.002154 0.002783 0.003594 0.004642 0.005995 0.007743 0.01 --out results/merge_simulated.json
```

The logical error rate is the fraction of all shots whose logical outcome is predicted
wrongly; shots without any detection event count as "no flip" predictions. MWPM uses
PyMatching on the same shots, with the calibrated p_ij models on hardware. Fig. 12 shows
the training history of `ibm_memory_d3_q36` (`results/ibm_d3_q36_history.json`).

## Training

`scripts/train.py` runs one stage of the pipeline of Sec. III C. Each optimizer step
averages the loss over one batch per experiment duration (`--rounds`; a duration given
twice counts twice), simulated batches mix all `--p` values in equal shares, the learning
rate decays as `lr · max(0.95^epoch, min_lr/lr)` (`--min-lr`, default lr/10), and the
checkpoint with the best accuracy on the validation shots (hardware) or the training
batches (simulation) is kept. `--init` copies all matching weights from a checkpoint,
e.g. the GNN and GRU of a d=3 decoder into a new d=5 decoder.

```bash
# Stage 1, simulated: d=3 base decoder, then the modular decoders
python scripts/train.py --experiment memory --d 3 --rounds 50 --p 0.001 0.002 0.003 0.004 0.005 \
    --epochs 500 --lr 1e-3 --out models/memory_d3.pt
python scripts/train.py --experiment memory --d 5 --rounds 50 --p 0.001 0.002 0.003 0.004 0.005 \
    --init models/memory_d3.pt --epochs 1000 --lr 1e-3 --out models/memory_d5.pt
python scripts/train.py --experiment memory --d 7 --rounds 50 --p 0.001 0.002 0.003 0.004 0.005 \
    --init models/memory_d3.pt --epochs 1500 --lr 1e-3 --out models/memory_d7.pt
python scripts/train.py --experiment memory --d 9 --rounds 50 100 200 --p 0.001 0.002 0.003 0.004 0.005 \
    --init models/memory_d5.pt --epochs 1000 --lr 1e-4 --out models/memory_d9.pt
python scripts/train.py --experiment stability --rounds 50 --p 0.001 0.003 0.005 0.01 0.02 0.03 \
    --epochs 200 --lr 1e-3 --out models/stability_d3_t50.pt
python scripts/train.py --experiment stability --rounds 2 5 10 20 50 50 --p 0.001 0.003 0.005 0.01 0.02 0.03 \
    --init models/stability_d3_t50.pt --epochs 200 --lr 1e-4 --out models/stability_d3.pt
python scripts/train.py --experiment merge --p 0.001 0.002 0.004 0.007 0.01 \
    --epochs 10000 --steps 128 --batch-size 512 --lr 1e-3 --out models/merge_d3.pt

# Stage 1 on Google's SI1000 circuits (noise x3), then Stages 2 and 3 for one patch
python scripts/train.py --experiment google --patch data/google/d3_at_q6_7 --stage si1000 \
    --init models/memory_d3.pt --epochs 200 --lr 1e-3 --out models/google_si1000_d3.pt
python scripts/train.py --experiment google --patch data/google/d3_at_q2_7 --stage dem \
    --init models/google_si1000_d3.pt --epochs 400 --lr 1e-4 --min-lr 1e-4 --out models/stage2.pt
python scripts/train.py --experiment google --patch data/google/d3_at_q2_7 --stage real \
    --init models/stage2.pt --epochs 100 --lr 1e-5 --min-lr 1e-5 --out models/google_d3_q2_7.pt

# IBM: Stages 2 and 3 (d=5 uses the tiled decoder automatically)
python scripts/train.py --experiment ibm --patch data/ibm_berlin/memory_d3_q36 --stage dem \
    --init models/google_si1000_d3.pt --epochs 800 --lr 1e-4 --out models/stage2.pt
python scripts/train.py --experiment ibm --patch data/ibm_berlin/memory_d3_q36 --stage real \
    --init models/stage2.pt --epochs 25 --lr 1e-5 --out models/ibm_memory_d3_q36.pt
```

The d=7 decoders use `--experiment google --patch data/google/d7_at_q6_7` with the default
width or `--hidden 1024`; the stability and ZZ-merge hardware decoders start from
`stability_d3_t50.pt` (the first of the two stability runs) and `merge_d3.pt`
(`--experiment ibm --patch data/ibm_berlin/stability_d3` with the durations listed above,
`--experiment ibm-merge --patch data/ibm_berlin/zz_merge_1-3-1`).
The released models were trained in several continuation runs of these stages; the epoch
counts above are representative, and each checkpoint records its origin under `source`.

## Citation

```bibtex
@article{lange2026modular,
  title   = {Modular Neural Decoder for Surface Code Memory and Logic on Superconducting Processors},
  author  = {Lange, Moritz and Fjelds{\aa}, Ole and Jonasson Johansson, Gustaf and Granath, Mats},
  journal = {arXiv:2610.03869},
  year    = {2026}
}
```

## License

Code: MIT (`LICENSE`). Trained models and the IBM data: CC-BY-4.0 ([Zenodo record](https://doi.org/10.5281/zenodo.23335425)).

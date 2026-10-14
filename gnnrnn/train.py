"""Training (paper App. B.6) and evaluation."""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pymatching
import torch
import torch.nn.functional as F

from .data import Recorded
from .model import save


def train(model, sources, *, epochs, steps=256, batch_size=2048, lr=1e-3, min_lr=None,
          val_sources=None, save_path=None, device="cpu"):
    """Adam with lr * max(0.95^epoch, min_lr / lr). Every optimizer step averages the
    loss over one batch from each source (e.g. every experiment duration). The checkpoint
    with the best validation accuracy (training accuracy if there is no validation set)
    is saved to save_path. Returns the per-epoch history."""
    min_lr = min_lr or lr / 10
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(optim, lambda e: max(0.95**e, min_lr / lr))
    batches = _prefetch(lambda: [_batch(s, batch_size) for s in sources])
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, correct = time.time(), 0.0, 0.0
        for _ in range(steps):
            optim.zero_grad()
            for graphs, labels, n_windows in next(batches):
                labels = labels.to(device)
                logits = model([g.to(device) for g in graphs], len(labels), n_windows)
                loss = F.binary_cross_entropy_with_logits(logits, labels) / len(sources)
                loss.backward()
                loss_sum += loss.item()
                correct += ((logits > 0) == labels).float().mean().item() / len(sources)
            optim.step()
        schedule.step()
        log = dict(epoch=epoch, loss=loss_sum / steps, accuracy=correct / steps,
                   lr=optim.param_groups[0]["lr"], seconds=time.time() - t0)
        if val_sources:
            log["val_accuracy"] = float(np.mean(
                [1 - logical_error_rates({"nn": nn_decoder(model, s.layout, device)}, s)["nn"][0]
                 for s in val_sources]))
        history.append(log)
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in log.items()), flush=True)
        score = log.get("val_accuracy", log["accuracy"])
        if score > best and save_path:
            best = score
            save(model, save_path, history=history)
    return history


def _batch(source, batch_size):
    events, labels = source.sample(batch_size)
    return source.layout.graphs(events), torch.from_numpy(labels), source.layout.n_windows


def _prefetch(make):
    """Yield make() forever, preparing the next result in a background thread."""
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(make)
        while True:
            result, future = future.result(), pool.submit(make)
            yield result


def nn_decoder(model, layout, device="cpu"):
    """events -> predicted logical flips; shots without detection events predict no flip."""
    @torch.no_grad()
    def decode(events):
        model.eval()
        flips = np.zeros((len(events), model.config["n_obs"]), dtype=bool)
        nontrivial = events.any(axis=1)
        if nontrivial.any():
            logits = model(layout.graphs(events[nontrivial], device), int(nontrivial.sum()), layout.n_windows)
            flips[nontrivial] = (logits > 0).cpu().numpy()
        return flips
    return decode


def mwpm_decoder(dem):
    matching = pymatching.Matching.from_detector_error_model(dem)
    return lambda events: matching.decode_batch(events).astype(bool)


def logical_error_rates(decoders, source, shots=None, batch_size=4096, rel_error=None):
    """For each decoder (name -> function events -> predicted flips), the fraction of all
    shots (including those without detection events) whose logical outcome it gets wrong.
    Recorded sources are evaluated on every shot once, simulated ones on up to `shots`
    fresh samples shared by all decoders, stopping early once every decoder's relative
    standard error is below `rel_error`. Returns {name: (p_L, standard error, shots)}."""
    errors, n = dict.fromkeys(decoders, 0), 0
    if isinstance(source, Recorded):
        chunks, rel_error = source.batches(batch_size), None
    else:
        chunks = (source.sample(min(batch_size, shots - i), nontrivial=False) for i in range(0, shots, batch_size))
    for events, labels in chunks:
        for name, decode in decoders.items():
            errors[name] += int((decode(events) != labels).any(axis=1).sum())
        n += len(events)
        if rel_error and all(e > 0 and np.sqrt((1 - e / n) / e) < rel_error for e in errors.values()):
            break
    if isinstance(source, Recorded):  # shots without detection events: every decoder predicts no flip
        for name in decoders:
            errors[name] += int((source.trivial_labels != 0).any(axis=1).sum())
        n += len(source.trivial_labels)
    return {name: (e / n, float(np.sqrt(e / n * (1 - e / n) / n)), n) for name, e in errors.items()}

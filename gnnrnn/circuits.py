"""Stim circuits for the simulated experiments (memory, stability, ZZ merge).

The stability and ZZ-merge circuits are stored noiseless in circuits/ (generated with
the tile-based circuit builder of Gidney's stability-experiment code) and get uniform
circuit-level depolarizing noise from `add_noise`.
"""
from collections import defaultdict
from pathlib import Path

import stim

CIRCUITS = Path(__file__).parent.parent / "circuits"

# Noise channel applied to the targets of each gate after it acts (strength p).
_NOISE_AFTER = {"H": "DEPOLARIZE1", "CX": "DEPOLARIZE2", "CZ": "DEPOLARIZE2", "R": "X_ERROR",
                "RX": "Z_ERROR", "M": "DEPOLARIZE1", "MX": "DEPOLARIZE1"}


def memory_circuit(d: int, rounds: int, p: float) -> stim.Circuit:
    """Rotated surface code Z-memory under circuit-level depolarizing noise."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def stability_circuit(rounds: int, p: float) -> stim.Circuit:
    """d=3 X-type stability experiment (Gidney 2022); the template repeats its bulk round."""
    assert rounds >= 2, "the stability circuit has at least a first and a last round"
    template = stim.Circuit.from_file(CIRCUITS / "stability_d3_X.stim")
    circuit = stim.Circuit()
    for op in template:
        if not isinstance(op, stim.CircuitRepeatBlock):
            circuit.append(op)
        elif rounds > 2:  # first and last round are written out
            circuit.append(stim.CircuitRepeatBlock(rounds - 2, op.body_copy()))
    return add_noise(circuit, p)


def merge_circuit(rounds=(3, 3, 3), p: float = 0.0) -> stim.Circuit:
    """d=3 ZZ lattice-surgery merge (gap-1 bridge) with |+> initialization.

    rounds: (before, during, after) the merge; available: (3, 3, 3) and (1, 3, 1).
    The observable is the combined check Z1 xor Z2 xor M (Z1 and Z2 alone are random)."""
    name = "merge_d3_X_{}-{}-{}.stim".format(*rounds)
    return add_noise(stim.Circuit.from_file(CIRCUITS / name), p)


def add_noise(circuit: stim.Circuit, p: float, qubits=None) -> stim.Circuit:
    """Every gate is followed by a depolarizing (or flip) channel of strength p on its
    qubits, measurement results flip with probability p, and qubits idling during a
    moment (between two TICKs) depolarize with strength p."""
    qubits = set(range(circuit.num_qubits)) if qubits is None else qubits
    out = stim.Circuit()
    for i, moment in enumerate(_moments(circuit)):
        if i > 0 and not (len(out) and isinstance(out[-1], stim.CircuitRepeatBlock)):
            out.append("TICK")
        if isinstance(moment, stim.CircuitRepeatBlock):
            body = add_noise(moment.body_copy(), p, qubits)
            body.append("TICK")
            out.append(stim.CircuitRepeatBlock(moment.repeat_count, body))
            continue
        after, busy = defaultdict(stim.Circuit), set()
        for op in moment:
            if op.name not in _NOISE_AFTER:  # annotations: DETECTOR, QUBIT_COORDS, ...
                out.append(op)
                continue
            targets = [t.value for t in op.targets_copy()]
            busy.update(targets)
            out.append(op.name, targets, [p] if op.name in ("M", "MX") else [])
            channel = _NOISE_AFTER[op.name]
            after[channel, p].append(channel, targets, p)
        for key in sorted(after):
            out += after[key]
        if idle := sorted(qubits - busy):
            out.append("DEPOLARIZE1", idle, p)
    return out


def _moments(circuit):
    moment = []
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            yield op
        elif op.name == "TICK":
            yield moment
            moment = []
        else:
            moment.append(op)
    if moment:
        yield moment


def scale_noise(circuit: stim.Circuit, factor: float) -> stim.Circuit:
    """Multiply every noise probability (including measurement flips) by `factor`."""
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(op.repeat_count, scale_noise(op.body_copy(), factor)))
        elif op.gate_args_copy() and op.name not in ("DETECTOR", "QUBIT_COORDS", "SHIFT_COORDS",
                                                       "OBSERVABLE_INCLUDE"):
            out.append(op.name, op.targets_copy(), [a * factor for a in op.gate_args_copy()])
        else:
            out.append(op)
    return out

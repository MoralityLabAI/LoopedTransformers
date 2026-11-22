"""Independent scalar checks and whole-context controls for NEC-04H."""

from collections import Counter, defaultdict
from hashlib import sha256
from math import fsum, sqrt

from mechinterp.neuron_communities.nec04a_independent_oracle import oracle


def source_labels(world, bits):
    selector, first, second = world["context_positions"][:3]
    return int(bits[selector]), int(bits[first] != bits[second])


def verify_labels(world, row):
    bits = row["history_bits12"]
    selector, phase = source_labels(world, bits)
    program = world["canonical_program"]
    focal = world["focal_positions"]
    zero = [0 if i in focal else value for i, value in enumerate(bits)]
    raw_xor = sum(bits[i] for i in focal) % 2
    if (program[:2] != ["MUX", ["VAR", world["context_positions"][0]]]
            or phase != int(oracle(program[2], zero))
            or (raw_xor ^ phase) != int(oracle(program[2], bits))
            or raw_xor != row["xor_prefix"]["5"] or selector != row["mux_selector"]
            or int(oracle(program, bits)) != row["composed_label"]):
        raise ValueError("the independent original MUX, XOR or phase identity failed")
    return selector, phase


def shuffled_phases(root, groups):
    """Permute one label per source context inside each selector stratum."""
    assignments = {}
    for selector in (0, 1):
        keys = sorted(group for group, (sel, _) in groups.items() if sel == selector)
        ordered = sorted(keys, key=lambda g: (sha256(f"NEC04H/phase-control/v1|{root}|{selector}|{g}".encode()).digest(), g))
        assignments.update((dest, groups[donor][1]) for dest, donor in zip(keys, ordered, strict=True))
    return assignments


def verify_shuffle(rows, labels):
    if len(rows) != len(labels):
        raise ValueError("incomplete phase control")
    groups = defaultdict(set)
    counts = Counter()
    expected = Counter()
    for (group, selector, phase), label in zip(rows, labels, strict=True):
        groups[group].add(label)
        counts[selector, label] += 1
        expected[selector, phase] += 1
    if counts != expected or any(len(values) != 1 for values in groups.values()):
        raise ValueError("the control changed counts or split a source context group")


def scalar_route(xor, phase, selector, native):
    """Enumerate the latent outcomes independently of the tensor composer."""
    parity = [0., 0.]
    for bit in (0, 1):
        for offset in (0, 1):
            parity[(bit + offset) % 2] += xor[bit] * phase[offset]
    return [fsum((selector[1] * parity[y], selector[0] * native[y])) for y in (0, 1)]


def arrival_stratum(world):
    fifth = max(world["focal_positions"])
    return "/".join("before" if i < fifth else "after" for i in world["context_positions"][1:3])


def fifth_walsh(margins, phase):
    """Return the teacher-aligned coefficient of class-one minus class-zero."""
    if len(margins) != 32:
        raise ValueError("the Walsh contrast requires a complete focal context")
    mean = fsum(margins) / 32
    raw = fsum(value * (1 - 2 * (i.bit_count() % 2)) for i, value in enumerate(margins)) / 32
    energy = fsum((value - mean) ** 2 for value in margins) / 32
    signed = raw * (2 * phase - 1)
    return {"signed": signed, "normalized": signed / sqrt(energy) if energy > 1e-24 else None}

"""SUD-02: exact uniform-cube observation channels for reversible pointer loops.

Only Python stdlib is needed. All x,y pairs are weighted exactly using the
known bit-flip channel; semantic uncertainty is conditioned on a fixed graph.

This module evaluates iid *episodes at a fixed depth*, not the Hausdorff
dimension of a single dependent recurrent trace.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def binary_entropy(p: float) -> float:
    if not -1e-12 <= p <= 1 + 1e-12:
        raise ValueError("invalid probability")
    p = min(1.0, max(0.0, p))
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def validate(permutation: Sequence[int], depth: int, p: float, bit: int) -> tuple[int, ...]:
    perm = tuple(permutation)
    n = len(perm)
    if n < 2 or n > 16 or n & (n - 1) or sorted(perm) != list(range(n)):
        raise ValueError("permutation must be a bijection of 2, 4, 8 or 16 nodes")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    if not (0 <= p <= .5):
        raise ValueError("BSC crossover must be in [0,1/2]")
    if isinstance(bit, bool) or not isinstance(bit, int) or not 0 <= bit < n.bit_length() - 1:
        raise ValueError("semantic bit outside node register")
    return perm


def iterated_nodes(permutation: Sequence[int], depth: int) -> list[int]:
    nodes = list(range(len(permutation)))
    for _ in range(depth):
        nodes = [permutation[node] for node in nodes]
    return nodes


def hamming_channel(n: int, p: float) -> list[list[float]]:
    """W[x][y] = Pr[Y=y | X=x]; all x and y are n-bit node values."""
    return [
        [p ** ((x ^ y).bit_count()) * (1 - p) ** (n.bit_length() - 1 - (x ^ y).bit_count())
         for y in range(n)]
        for x in range(n)
    ]


def spectral_mass(labels: Sequence[int]) -> list[float]:
    """Normalized Walsh-Fourier energy by degree, sign g=(-1)^S."""
    n = len(labels)
    degree_mass = [0.0] * n.bit_length()
    for mask in range(n):
        coefficient = sum(
            (1 if label == 0 else -1) *
            (1 if (mask & x).bit_count() % 2 == 0 else -1)
            for x, label in enumerate(labels)
        ) / n
        degree_mass[mask.bit_count()] += coefficient * coefficient
    return degree_mass


def _output_metrics(labels: Sequence[int], w: list[list[float]],
                    output_by_y: Sequence[int]) -> dict:
    n = len(labels)
    if len(output_by_y) != n or any(v not in (0, 1) for v in output_by_y):
        raise ValueError("output_by_y must provide one bit per noisy start")
    joint = [[0.0, 0.0], [0.0, 0.0]]
    for x in range(n):
        for y in range(n):
            joint[labels[x]][output_by_y[y]] += w[x][y] / n
    entropy_cond = 0.0
    optimal_error_with_output = 0.0
    for a in (0, 1):
        mass = joint[0][a] + joint[1][a]
        if mass > 0:
            entropy_cond += mass * binary_entropy(joint[1][a] / mass)
            optimal_error_with_output += min(joint[0][a], joint[1][a])
    return {
        "mutual_information_bits": 1.0 - entropy_cond,
        "conditional_entropy_bits": entropy_cond,
        "accuracy_vs_clean_target": joint[0][0] + joint[1][1],
        "bayes_error_given_output": optimal_error_with_output,
        "joint_S_output": joint,
    }


def evaluate_channel(permutation: Sequence[int], depth: int, p: float = 0.1,
                     bit: int = 0, model_nodes_by_y: Sequence[int] | None = None,
                     ratio: float = 1 / 3) -> dict:
    """Exact conditional graph channel + optional trained-head binary projection.

    X uniform on graph nodes, Y=X xor independent BSC(p) bits, target
    S=bit(perm^depth(X)). The model may only receive (perm,Y). Its native
    predicted node at depth is supplied for each possible observed start Y.
    No clean X is ever given to the neural model.
    """
    perm = validate(permutation, depth, p, bit)
    if not (0 < ratio < .5):
        raise ValueError("Cantor ratio must be strictly between 0 and 1/2")
    n = len(perm)
    successor_by_x = iterated_nodes(perm, depth)
    labels = [(v >> bit) & 1 for v in successor_by_x]
    assert sum(labels) == n // 2, "bijections preserve a balanced semantic bit"
    weights = hamming_channel(n, p)
    q = [sum(labels[x] * weights[x][y] for x in range(n)) for y in range(n)]
    conditional_entropy = sum(binary_entropy(v) for v in q) / n
    bayes_error = sum(min(v, 1 - v) for v in q) / n
    info = 1 - conditional_entropy
    degree_mass = spectral_mass(labels)
    rho = 1 - 2 * p
    spectral_v = sum(m * rho ** (2 * k) for k, m in enumerate(degree_mass))
    posterior_v = sum((2 * x - 1) ** 2 for x in q) / n
    tol = 1e-9
    if abs(spectral_v - posterior_v) > tol:
        raise AssertionError("Walsh posterior second-moment identity failed")
    if (conditional_entropy + tol < binary_entropy(p) or
            bayes_error + tol < p or
            info + tol < spectral_v / (2 * math.log(2)) or
            info > spectral_v + tol):
        raise AssertionError("Courtade-Kumar / Bayes / Fourier bounds violated")
    oracle_bits = labels
    oracle_head = _output_metrics(labels, weights, oracle_bits)
    optimal_bits = [int(x > .5) for x in q]
    optimal_head = _output_metrics(labels, weights, optimal_bits)
    result = {
        "depth": depth,
        "semantic_bit": bit,
        "noise_p": p,
        "conditional_entropy_bits": conditional_entropy,
        "source_information_bits": info,
        "source_bayes_error": bayes_error,
        "iid_episode_dimension_reference": conditional_entropy / math.log2(1 / ratio),
        "CK_conditional_entropy_floor_bits": binary_entropy(p),
        "spectral_mass_by_degree": degree_mass,
        "fourier_second_moment": spectral_v,
        "fourier_info_lower_bits": spectral_v / (2 * math.log(2)),
        "fourier_info_upper_bits": spectral_v,
        "raw_corrupted_oracle_bit": oracle_head,
        "bayes_optimal_bit": optimal_head,
        "clean_semantic_labels_by_start": labels,
    }
    if model_nodes_by_y is not None:
        if len(model_nodes_by_y) != n or any(
            isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n
            for v in model_nodes_by_y
        ):
            raise ValueError("model readout must return one valid node for each noisy start")
        model_bits = [(v >> bit) & 1 for v in model_nodes_by_y]
        neural = _output_metrics(labels, weights, model_bits)
        neural["clean_native_cursor_accuracy"] = sum(
            model_nodes_by_y[y] == successor_by_x[y] for y in range(n)
        ) / n
        neural["clean_semantic_bit_accuracy"] = sum(
            model_bits[y] == labels[y] for y in range(n)
        ) / n
        neural["additional_conditional_entropy_bits"] = (
            neural["conditional_entropy_bits"] - conditional_entropy
        )
        if neural["additional_conditional_entropy_bits"] < -tol:
            raise AssertionError("deterministic head violated data processing")
        result["native_head"] = neural
    return result


def reversible_echo_control(p: float = .1, ratio: float = 1 / 3) -> dict:
    """Order-three reversible XOR map: 0->0, 1->3, 2->1, 3->2."""
    permutation = (0, 3, 1, 2)
    entries = [evaluate_channel(permutation, depth, p, ratio=ratio)
               for depth in range(7)]
    if abs(entries[0]["source_information_bits"] -
           entries[3]["source_information_bits"]) > 1e-10:
        raise AssertionError("period-three information echo failed")
    if not entries[1]["source_information_bits"] < entries[0]["source_information_bits"]:
        raise AssertionError("parity mixing failed to lower information")
    return {"status": "CONTROL_ONLY", "permutation": list(permutation),
            "p": p, "ratio": ratio, "depths": entries}

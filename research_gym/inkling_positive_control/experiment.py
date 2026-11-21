"""CPU empirical positive control for recurrent state patching.

One learned scalar gate mixes a fixed, task-provided permutation operator with
the previous state. The permutation and hop budget are available explicitly.
This instrument verifies tracing/dose plumbing, not neural graph discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NODES = 17
TRAIN_SEED = 23
DEVELOPMENT_SEED = 20261121
TRAIN_ROOTS = 48
DEVELOPMENT_ROOTS = 64
TRAIN_HOPS = 4
DEVELOPMENT_HOPS = 8
EPOCHS = 70
LEARNING_RATE = 0.75
INTERVENTION_DOSE = 0.85
INTERVENTION_VISIT = 2
DEPTHS = (1, 2, 3, 5, 7, 10)
ARMS = ("clean", "zero", "active_coordinate", "random_direction", "random_coordinate")
SCHEMA = "rmp.inkling_positive_control.v1"


@dataclass(frozen=True)
class Task:
    root_id: str
    case_id: str
    permutation: tuple[int, ...]
    start: int
    hops: int
    target: int


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def generate_roots(seed: int, split: str, roots: int, max_hops: int) -> list[list[Task]]:
    """A source-root is one independently drawn directed single-cycle graph."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    if split not in {"train", "development"}:
        raise ValueError("protected splits cannot be generated here")
    if type(roots) is not int or not 1 <= roots <= 256:
        raise ValueError("root count must be in [1,256]")
    if type(max_hops) is not int or not 1 <= max_hops < NODES:
        raise ValueError("hop count is out of bounds")
    groups = []
    for root in range(roots):
        identity = f"{split}:{seed}:{root}"
        digest = hashlib.sha256(identity.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        order = rng.permutation(NODES)
        permutation = [0] * NODES
        for a, b in zip(order, np.roll(order, -1), strict=True):
            permutation[int(a)] = int(b)
        start = int(rng.integers(NODES))
        current = start
        group = []
        root_id = f"{split}-{seed}-{root:04d}"
        for hops in range(1, max_hops + 1):
            current = permutation[current]
            group.append(
                Task(root_id, f"{root_id}-h{hops}", tuple(permutation), start, hops, current)
            )
        groups.append(group)
    return groups


def exact_verifier(task: Task, candidate: int) -> bool:
    """Recompute the graph walk independently of the model prediction."""
    if len(task.permutation) != NODES or sorted(task.permutation) != list(range(NODES)):
        raise ValueError("task permutation must be a bijection")
    current = task.start
    for _ in range(task.hops):
        current = task.permutation[current]
    if current != task.target:
        raise ValueError("corrupted oracle target")
    return type(candidate) is int and candidate == current


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _advance(state: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    moved = np.zeros_like(state)
    moved[list(permutation)] = state
    return moved


def _intervention(
    state: np.ndarray, arm: str, *, dose: float, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a physical raw-state perturbation at a named loop boundary."""
    n = state.size
    direction = np.zeros(n, dtype=np.float64)
    selected = int(state.argmax())
    if arm == "active_coordinate":
        direction[selected] = -1.0 / math.sqrt(2)
        direction[(selected + 1) % n] = 1.0 / math.sqrt(2)
    elif arm == "random_direction":
        direction = np.random.default_rng(seed).normal(size=n)
        direction /= np.linalg.norm(direction)
    elif arm == "random_coordinate":
        random_index = int(np.random.default_rng(seed).integers(n))
        direction[random_index] = -1.0 / math.sqrt(2)
        direction[(random_index + 1) % n] = 1.0 / math.sqrt(2)
    elif arm == "zero":
        dose = 0.0
    else:
        raise ValueError("unrecognized intervention arm")
    after = state + dose * direction
    displacement = float(np.linalg.norm(after - state))
    if arm != "zero" and not math.isclose(displacement, dose, abs_tol=1e-10):
        raise ValueError("physical patch dose mismatch")
    return after, {
        "selected_pre_patch_coordinate": selected,
        "realized_raw_l2_displacement": displacement,
        "pre_patch_argmax": selected,
        "post_patch_argmax": int(after.argmax()),
    }


def forward(
    task: Task,
    theta: float,
    visits: int,
    *,
    intervention: str = "clean",
    patch_visit: int = INTERVENTION_VISIT,
    dose: float = INTERVENTION_DOSE,
    seed: int = 0,
    derivative: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    if type(visits) is not int or not 1 <= visits <= 64:
        raise ValueError("visits must be in [1,64]")
    if not math.isfinite(theta) or not math.isfinite(dose) or dose < 0:
        raise ValueError("invalid gate or dose")
    state = np.zeros(NODES, dtype=np.float64)
    state[task.start] = 1.0
    derivative_state = np.zeros(NODES, dtype=np.float64)
    gate = sigmoid(theta)
    gate_gradient = gate * (1.0 - gate)
    receipt = None
    for visit in range(1, visits + 1):
        if visit <= task.hops:
            message = _advance(state, task.permutation)
            dmessage = _advance(derivative_state, task.permutation)
            updated = (1 - gate) * state + gate * message
            new_derivative = (
                (1 - gate) * derivative_state
                + gate * dmessage + gate_gradient * (message - state)
            )
            state, derivative_state = updated, new_derivative
        # The hop count is supplied by the task. After that count, state is held.
        if intervention != "clean" and visit == patch_visit:
            if derivative:
                raise ValueError("derivative through an intervention is not supported")
            state, receipt = _intervention(state, intervention, dose=dose, seed=seed)
    return state, derivative_state, receipt


def train_gate(
    *,
    seed: int = TRAIN_SEED,
    roots: int = TRAIN_ROOTS,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> tuple[float, list[dict[str, Any]]]:
    """Train a real scalar gate via analytic gradients of cross-entropy."""
    if type(epochs) is not int or not 1 <= epochs <= 500:
        raise ValueError("epochs out of bounds")
    if not 0 < learning_rate <= 2 or not math.isfinite(learning_rate):
        raise ValueError("learning rate out of bounds")
    examples = [t for group in generate_roots(seed, "train", roots, TRAIN_HOPS) for t in group]
    theta = 0.0
    trace = []
    for epoch in range(epochs):
        gradients = []
        losses = []
        for task in examples:
            state, derivative_state, _ = forward(task, theta, task.hops, derivative=True)
            probability = max(float(state[task.target]), 1e-20)
            losses.append(-math.log(probability))
            gradients.append(-float(derivative_state[task.target]) / probability)
        mean_gradient = float(np.mean(gradients))
        mean_loss = float(np.mean(losses))
        theta -= learning_rate * mean_gradient
        if epoch in {0, 1, 4, 9, 19, 39, epochs - 1}:
            trace.append({
                "epoch": epoch, "cross_entropy": mean_loss,
                "gate": sigmoid(theta), "gradient": mean_gradient,
            })
    return theta, trace


def _seed(task: Task, arm: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{task.case_id}:{arm}".encode()).digest()[:8], "big")


def evaluate(
    theta: float,
    *,
    seed: int = DEVELOPMENT_SEED,
    roots: int = DEVELOPMENT_ROOTS,
    dose: float = INTERVENTION_DOSE,
    depths: tuple[int, ...] = DEPTHS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(set(depths)) != len(depths) or not depths:
        raise ValueError("depths must be distinct and nonempty")
    groups = generate_roots(seed, "development", roots, DEVELOPMENT_HOPS)
    checkpoint_hash = _hash({"theta": theta, "arch": "task_supplied_permutation_with_learned_gate_v1"})
    depth_rows = []
    causal_rows = []
    for group in groups:
        for task in group:
            task_world = {
                "start": task.start, "permutation": list(task.permutation),
                "hops": task.hops,
            }
            task_digest = _hash({"root_id": task.root_id, "world": task_world})
            for depth in depths:
                state, _, _ = forward(task, theta, depth)
                predicted = int(state.argmax())
                depth_rows.append({
                    "case_id": task.case_id, "root_id": task.root_id,
                    "task_world": task_world, "task_digest": task_digest,
                    "verifier_id": "rmp.ink_pc01.exact_cycle.v1",
                    "family": "explicit_single_cycle_pointer_chase",
                    "model_sha": checkpoint_hash, "cache_mode": "none",
                    "split": "development", "depth": depth, "hops": task.hops,
                    "verified": exact_verifier(task, predicted),
                    "predicted": predicted, "target": task.target,
                    "output_state_rms": float(np.sqrt(np.mean(state * state))),
                    "reasoning_tokens": None, "total_tokens": None,
                    "wall_ms": None, "joules": None,
                    "architectural_passes": depth,
                })
            if task.hops >= 4:
                for arm in ARMS:
                    output, _, patch = forward(
                        task, theta, max(depths), intervention=arm,
                        patch_visit=INTERVENTION_VISIT,
                        dose=dose, seed=_seed(task, arm),
                    )
                    causal_rows.append({
                        "case_id": task.case_id, "root_id": task.root_id,
                        "task_world": task_world, "task_digest": task_digest,
                        "verifier_id": "rmp.ink_pc01.exact_cycle.v1",
                        "family": "explicit_single_cycle_pointer_chase",
                        "model_sha": checkpoint_hash,
                        "split": "development", "hops": task.hops,
                        "depth": max(depths), "arm": arm,
                        "patch_visit": INTERVENTION_VISIT,
                        "nominal_l2_dose": 0.0 if arm in {"clean", "zero"} else dose,
                        "realized_raw_l2_displacement": 0.0 if patch is None else patch[
                            "realized_raw_l2_displacement"
                        ],
                        "selected_pre_patch_coordinate": None if patch is None else patch[
                            "selected_pre_patch_coordinate"
                        ],
                        "pre_patch_argmax": None if patch is None else patch["pre_patch_argmax"],
                        "post_patch_argmax": None if patch is None else patch["post_patch_argmax"],
                        "verified": exact_verifier(task, int(output.argmax())),
                        "predicted": int(output.argmax()), "target": task.target,
                    })
    accuracy = {
        str(depth): {
            "correct": sum(row["verified"] for row in depth_rows if row["depth"] == depth),
            "total": sum(row["depth"] == depth for row in depth_rows),
        }
        for depth in depths
    }
    intervention_accuracy = {
        arm: {
            "correct": sum(row["verified"] for row in causal_rows if row["arm"] == arm),
            "total": sum(row["arm"] == arm for row in causal_rows),
        }
        for arm in ARMS
    }
    per_root = {}
    for group in groups:
        root_id = group[0].root_id
        arm_correct = {
            arm: sum(
                row["verified"] for row in causal_rows
                if row["root_id"] == root_id and row["arm"] == arm
            ) for arm in ARMS
        }
        per_root[root_id] = {
            "active_minus_random_direction": (
                arm_correct["active_coordinate"] - arm_correct["random_direction"]
            ) / (DEVELOPMENT_HOPS - 3),
            "active_minus_random_coordinate": (
                arm_correct["active_coordinate"] - arm_correct["random_coordinate"]
            ) / (DEVELOPMENT_HOPS - 3),
        }
    result = {
        "schema_version": SCHEMA, "evidence_class": "synthetic_trained_gate_development_positive_control",
        "checkpoint_sha256": checkpoint_hash, "gate": sigmoid(theta),
        "source_roots": roots, "cases": roots * DEVELOPMENT_HOPS,
        "train_hop_exposure": [1, 2, 3, 4],
        "evaluation_hops": list(range(1, DEVELOPMENT_HOPS + 1)),
        "depth_accuracy": accuracy, "intervention_accuracy": intervention_accuracy,
        "patch_visit": INTERVENTION_VISIT, "nominal_dose": dose,
        "root_macro_active_minus_random_direction": float(np.mean([
            item["active_minus_random_direction"] for item in per_root.values()
        ])),
        "root_macro_active_minus_random_coordinate": float(np.mean([
            item["active_minus_random_coordinate"] for item in per_root.values()
        ])),
        "positive_control": {
            "learned_gate_above_point_9": sigmoid(theta) >= 0.9,
            "full_depth_exact": accuracy[str(max(depths))]["correct"] == roots * DEVELOPMENT_HOPS,
            "dose_matched_shams_present": (
                all(intervention_accuracy[arm]["total"] == roots * (DEVELOPMENT_HOPS - 3)
                    for arm in ARMS)
            ),
        },
        "limitations": [
            "permutation_transition_is_explicitly_supplied",
            "hop_budget_is_explicitly_supplied",
            "one_scalar_gate_learned_not_an_end_to_end_transformer",
            "selected_active_coordinate_is_a_contrived_positive_control",
            "dose_was_selected_on_prior_development_pilot",
            "no_future_or_sealed_roots_used",
            "random_directions_are_not_a_neural_routing_mechanism",
        ],
    }
    return result, depth_rows, causal_rows


def run(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    started = time.perf_counter()
    theta, history = train_gate()
    training_seconds = time.perf_counter() - started
    result, depth_rows, causal_rows = evaluate(theta)
    eval_seconds = time.perf_counter() - started - training_seconds
    out.mkdir(parents=True)
    checkpoint = {
        "architecture": "explicit_permutation_one_parameter_recurrent_gate",
        "theta": theta, "alpha": sigmoid(theta),
        "training_seed": TRAIN_SEED, "training_roots": TRAIN_ROOTS,
        "training_hops": TRAIN_HOPS, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "trace": history, "sha256": result["checkpoint_sha256"],
    }
    (out / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _jsonl(out / "depth_matrix.jsonl", depth_rows)
    _jsonl(out / "interventions.jsonl", causal_rows)
    result["training_seconds_measured"] = training_seconds
    result["evaluation_seconds_measured"] = eval_seconds
    result["files_sha256"] = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("checkpoint.json", "depth_matrix.jsonl", "interventions.jsonl")
    }
    result["module_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({
        "out": str(args.out),
        "gate": result["gate"],
        "source_roots": result["source_roots"],
        "depth_accuracy": result["depth_accuracy"],
        "intervention_accuracy": result["intervention_accuracy"],
        "evidence_class": result["evidence_class"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

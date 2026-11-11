"""RMP-MG-01 fixed-target probes with graph-isomorphism-disjoint splits.

NumPy handles graph construction, scoring and clustered summaries. Probe fitting
loads PyTorch lazily. Actor weights are never optimized by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_hash(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode() + b"\0" + value.dtype.str.encode())
        digest.update(repr(value.shape).encode() + b"\0" + value.tobytes())
    return digest.hexdigest()


def permutation_signature(permutation: Sequence[int]) -> tuple[int, ...]:
    """Complete isomorphism invariant for a directed permutation graph."""
    p = np.asarray(permutation)
    if (p.ndim != 1 or not len(p) or not np.issubdtype(p.dtype, np.integer)
            or not np.array_equal(np.sort(p), np.arange(len(p)))):
        raise ValueError("expected an integer permutation of 0..n-1")
    seen: set[int] = set()
    lengths = []
    for start in range(len(p)):
        if start in seen:
            continue
        node, length = start, 0
        while node not in seen:
            seen.add(node)
            node = int(p[node])
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths))


def partition_types(n: int) -> list[tuple[int, ...]]:
    """Enumerate cycle types, with a deliberate resource bound."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 32:
        raise ValueError("cycle-type enumeration supports integer node counts 1..32")

    def recur(total: int, lower: int):
        if total == 0:
            yield ()
        for first in range(lower, total + 1):
            for rest in recur(total - first, first):
                yield (first,) + rest

    return list(recur(int(n), 1))


def canonical_permutation(cycle_lengths: Sequence[int]) -> np.ndarray:
    if not cycle_lengths or any(isinstance(k, bool) or not isinstance(k, (int, np.integer))
                                or k < 1 for k in cycle_lengths):
        raise ValueError("cycle lengths must be positive integers")
    p = np.empty(sum(cycle_lengths), dtype=np.int64)
    offset = 0
    for length in cycle_lengths:
        p[offset:offset + length] = np.roll(np.arange(offset, offset + length), -1)
        offset += length
    return p


def rename_permutation(permutation: Sequence[int], renaming: Sequence[int]) -> np.ndarray:
    p, rename = np.asarray(permutation), np.asarray(renaming)
    permutation_signature(p)
    permutation_signature(rename)
    if p.shape != rename.shape:
        raise ValueError("renaming and permutation sizes differ")
    return rename[p[np.argsort(rename)]]


def fixed_target(permutation: Sequence[int], start: int, horizon: int) -> int:
    permutation_signature(permutation)
    if (isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 0
            or isinstance(start, bool) or not isinstance(start, (int, np.integer))
            or not 0 <= start < len(permutation)):
        raise ValueError("start must be a valid node and horizon a nonnegative integer")
    node = int(start)
    for _ in range(horizon):
        node = int(permutation[node])
    return node


def make_graph_splits(n_nodes: int, group_counts: Sequence[int], items_per_group: int,
                      seed: int) -> dict[str, list[dict]]:
    """Sample cycle types uniformly, then random renamings/starts/edge orders.

    The resulting estimand weights selected structural families equally. It is
    a structural holdout distribution, not uniform sampling over permutations.
    """
    if (len(group_counts) != 3 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1
                                    for x in group_counts)
            or isinstance(items_per_group, bool) or not isinstance(items_per_group, int)
            or items_per_group < 1):
        raise ValueError("three positive group counts and positive items_per_group required")
    signatures = partition_types(n_nodes)
    if sum(group_counts) > len(signatures):
        raise ValueError(f"requested {sum(group_counts)} groups; n={n_nodes} has only {len(signatures)} cycle types")
    rng = np.random.default_rng(seed)
    choice = rng.permutation(len(signatures))[:sum(group_counts)]
    splits: dict[str, list[dict]] = {}
    offset = 0
    for split, count in zip(("train", "validation", "test"), group_counts):
        rows = []
        for index in choice[offset:offset + count]:
            signature = signatures[int(index)]
            canonical = canonical_permutation(signature)
            group_id = ":".join(map(str, signature))
            for item in range(items_per_group):
                p = rename_permutation(canonical, rng.permutation(n_nodes))
                order = rng.permutation(n_nodes)
                start = int(rng.integers(n_nodes))
                tokens = np.stack([order, p[order]], axis=-1).reshape(-1).tolist() + [start]
                rows.append({"id": f"{split}/{group_id}/{item}", "split": split,
                             "group_id": group_id, "cycle_type": list(signature),
                             "graph_hash": json_hash(p.tolist()), "permutation": p.tolist(),
                             "start": start, "edge_order": order.tolist(), "tokens": tokens})
        splits[split] = rows
        offset += count
    validate_graph_splits(splits)
    return splits


def validate_graph_splits(splits: Mapping[str, Sequence[dict]]) -> dict:
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("expected train, validation, and test splits")
    seen_groups: dict[str, str] = {}
    seen_graphs: dict[str, str] = {}
    ids: set[str] = set()
    counts = {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"empty split {split}")
        local_groups = set()
        for row in rows:
            signature = permutation_signature(row["permutation"])
            key = ":".join(map(str, signature))
            graph_hash = json_hash(row["permutation"])
            if (row["group_id"] != key or tuple(row["cycle_type"]) != signature
                    or row["graph_hash"] != graph_hash or row["split"] != split):
                raise ValueError("graph metadata does not match its permutation or split")
            n = len(row["permutation"])
            order = np.asarray(row["edge_order"])
            permutation_signature(order)
            if len(order) != n or not 0 <= row["start"] < n:
                raise ValueError("invalid edge order or start")
            expected = np.stack([order, np.asarray(row["permutation"])[order]], -1).reshape(-1).tolist() + [row["start"]]
            if expected != row["tokens"]:
                raise ValueError("tokens disagree with graph metadata")
            if row["id"] in ids:
                raise ValueError("duplicate item ID")
            ids.add(row["id"])
            if key in seen_groups and seen_groups[key] != split:
                raise ValueError("renamed/isomorphic graphs cross splits")
            if graph_hash in seen_graphs and seen_graphs[graph_hash] != split:
                raise ValueError("a labelled graph crosses splits")
            seen_groups[key], seen_graphs[graph_hash] = split, split
            local_groups.add(key)
        counts[split] = {"items": len(rows), "structural_groups": len(local_groups),
                         "unique_labelled_graphs": len({r["graph_hash"] for r in rows})}
    return {"passed": True, "grouping": "exact_permutation_cycle_type",
            "split_overlap_groups": 0, "counts": counts}


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training_features: np.ndarray) -> "Standardizer":
        x = _features(training_features)
        flat = x.reshape(-1, x.shape[-1])
        mean, scale = flat.mean(0), flat.std(0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean, scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = _features(features)
        if x.shape[-1] != len(self.mean):
            raise ValueError("feature width differs from fitted standardizer")
        return (x - self.mean) / self.scale

    def receipt(self) -> dict:
        return {"fit_scope": "training_graphs_all_selected_visits_including_zero",
                "width": len(self.mean),
                "sha256": array_hash({"mean": self.mean, "scale": self.scale})}


def _features(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("features must be finite [visits, nonempty items, width]")
    return x


def _labels(value: np.ndarray, n_items: int, n_classes: int) -> np.ndarray:
    y = np.asarray(value)
    if (y.shape != (n_items,) or not np.issubdtype(y.dtype, np.integer)
            or n_classes < 2 or (y < 0).any() or (y >= n_classes).any()):
        raise ValueError("labels must be valid integer classes, one per item")
    return y.astype(np.int64)


def group_means(values: np.ndarray, groups: Sequence[str]) -> tuple[list[str], np.ndarray]:
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups, dtype=str)
    if values.ndim != 1 or groups.shape != values.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite values and groups must be equally sized nonempty vectors")
    unique = sorted(set(groups.tolist()))
    return unique, np.array([values[groups == key].mean() for key in unique])


def paired_summary(baseline_losses: np.ndarray, visit_losses: np.ndarray,
                   groups: Sequence[str], *, seed: int, bootstrap: int = 2000) -> dict:
    before, after = np.asarray(baseline_losses), np.asarray(visit_losses)
    if before.shape != after.shape or bootstrap < 1:
        raise ValueError("paired loss shapes must match and bootstrap must be positive")
    keys, means = group_means(before - after, groups)
    interval = None
    if len(keys) > 1:
        rng = np.random.default_rng(seed)
        samples = means[rng.integers(0, len(keys), size=(bootstrap, len(keys)))].mean(1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    return {"mean_gain_nats": float(means.mean()), "bootstrap_percentile_95": interval,
            "independent_groups": len(keys), "items": len(before),
            "group_gains_nats": dict(zip(keys, means.tolist())),
            "uncertainty_scope": "selected_cycle_types_conditional_on_this_frozen_actor",
            "ci_status": "EXPLORATORY_CLUSTER_BOOTSTRAP" if interval is not None else "INSUFFICIENT_GROUPS"}


def scores(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError("logits must be a nonempty finite matrix")
    labels = _labels(labels, len(logits), logits.shape[-1])
    centered = logits - logits.max(1, keepdims=True)
    loss = np.log(np.exp(centered).sum(1)) - centered[np.arange(len(labels)), labels]
    return loss, logits.argmax(1) == labels


@dataclass(frozen=True)
class ProbeSpec:
    family: str = "affine"
    hidden: int = 32
    steps: int = 200
    batch_size: int = 128
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    dtype: str = "float64"

    def validate(self) -> None:
        if (self.family not in ("affine", "mlp") or self.dtype not in ("float32", "float64")
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                       for v in (self.hidden, self.steps, self.batch_size))
                or not np.isfinite([self.learning_rate, self.weight_decay]).all()
                or self.learning_rate <= 0 or self.weight_decay < 0):
            raise ValueError("invalid probe specification")


def _network(width: int, n_classes: int, spec: ProbeSpec):
    import torch
    spec.validate()
    if spec.family == "affine":
        net = torch.nn.Sequential(torch.nn.Linear(width, n_classes))
    else:
        net = torch.nn.Sequential(torch.nn.Linear(width, spec.hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(spec.hidden, n_classes))
    return net.to(dtype=getattr(torch, spec.dtype), device="cpu")


@dataclass
class FittedProbe:
    spec: ProbeSpec
    width: int
    n_classes: int
    weights: dict[str, np.ndarray]
    training: dict

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        import torch
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.width or not np.isfinite(x).all():
            raise ValueError("prediction features have invalid shape or values")
        with torch.random.fork_rng(devices=[]):
            net = _network(self.width, self.n_classes, self.spec)
        net.load_state_dict({key: torch.as_tensor(value.copy()) for key, value in self.weights.items()}, strict=True)
        net.eval()
        with torch.no_grad():
            out = net(torch.as_tensor(x, dtype=getattr(torch, self.spec.dtype)))
        return out.double().numpy()

    def receipt(self) -> dict:
        n_params = sum(value.size for value in self.weights.values())
        macs = (self.width * self.n_classes if self.spec.family == "affine"
                else self.width * self.spec.hidden + self.spec.hidden * self.n_classes)
        return {"spec": asdict(self.spec), "parameters": n_params,
                "linear_multiply_accumulates_per_item": macs,
                "weights_sha256": array_hash(self.weights), "training": self.training}


def fit_probe(features: np.ndarray, labels: np.ndarray, n_classes: int,
              spec: ProbeSpec, seed: int) -> FittedProbe:
    """Exactly steps AdamW updates, each with exactly batch_size sampled rows."""
    import torch
    spec.validate()
    x = _features(features)
    y = _labels(labels, x.shape[1], n_classes)
    flat = x.reshape(-1, x.shape[-1])
    targets = np.tile(y, x.shape[0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = _network(x.shape[-1], n_classes, spec)
        initial_hash = array_hash({k: v.detach().numpy() for k, v in net.state_dict().items()})
        tx = torch.as_tensor(flat, dtype=getattr(torch, spec.dtype))
        ty = torch.as_tensor(targets, dtype=torch.long)
        optimizer = torch.optim.AdamW(net.parameters(), lr=spec.learning_rate,
                                      weight_decay=spec.weight_decay)
        rng = np.random.default_rng(seed + 230001)
        net.train()
        for step in range(spec.steps):
            index = torch.as_tensor(rng.integers(0, len(flat), size=spec.batch_size), dtype=torch.long)
            logits = net(tx[index])
            loss = torch.nn.functional.cross_entropy(logits, ty[index])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite probe objective at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(not bool(torch.isfinite(p.grad).all()) for p in net.parameters()):
                raise RuntimeError(f"nonfinite probe gradient at step {step}")
            optimizer.step()
        with torch.no_grad():
            train_loss = float(torch.nn.functional.cross_entropy(net(tx), ty))
        weights = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    if not all(np.isfinite(v).all() for v in weights.values()):
        raise RuntimeError("fitted probe has nonfinite parameters")
    return FittedProbe(spec, x.shape[-1], n_classes, weights,
                       {"seed": seed, "optimizer": "AdamW", "steps": spec.steps,
                        "examples_processed": spec.steps * spec.batch_size,
                        "available_feature_rows": len(flat), "final_train_ce_nats": train_loss,
                        "initial_weights_sha256": initial_hash,
                        "training_labels_sha256": array_hash({"labels": y})})


@dataclass
class SelectedProbes:
    family: str
    mode: str
    shuffled_training_labels: bool
    visits: tuple[int, ...]
    probes: list[FittedProbe]
    candidates: list[dict]
    normalizer: Standardizer

    def receipt(self) -> dict:
        return {"family": self.family, "mode": self.mode,
                "shuffled_training_labels": self.shuffled_training_labels,
                "visits": list(self.visits), "normalizer": self.normalizer.receipt(),
                "probes": [probe.receipt() for probe in self.probes],
                "candidates": self.candidates,
                "selection_scope": "training_and_validation_only"}


def select_probes(training_x: np.ndarray, training_y: np.ndarray,
                  validation_x: np.ndarray, validation_y: np.ndarray,
                  validation_groups: Sequence[str], visits: Sequence[int],
                  n_classes: int, specs: Sequence[ProbeSpec], *, mode: str,
                  seed: int, shuffle_labels: bool = False) -> SelectedProbes:
    """Fit and select without accepting any test data or test labels."""
    train, val = _features(training_x), _features(validation_x)
    if (mode not in ("shared", "separate") or train.shape[0] != val.shape[0]
            or train.shape[-1] != val.shape[-1] or len(visits) != train.shape[0]
            or not visits or visits[0] != 0 or len(set(visits)) != len(visits)
            or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in visits)
            or not specs or len({spec.family for spec in specs}) != 1):
        raise ValueError("invalid visits, split shapes, probe family, or fit mode")
    ytrain = _labels(training_y, train.shape[1], n_classes).copy()
    yval = _labels(validation_y, val.shape[1], n_classes)
    group_means(np.zeros(len(yval)), validation_groups)
    if shuffle_labels:
        ytrain = ytrain[np.random.default_rng(seed + 91081).permutation(len(ytrain))]
    normalizer = Standardizer.fit(train)
    train, val = normalizer.transform(train), normalizer.transform(val)
    selected, all_candidates = [], []
    visit_sets = [tuple(range(len(visits)))] if mode == "shared" else [(v,) for v in range(len(visits))]
    for indices in visit_sets:
        candidates = []
        for config, spec in enumerate(specs):
            probe = fit_probe(train[list(indices)], ytrain, n_classes, spec, seed)
            visit_losses = []
            for i in indices:
                loss, _ = scores(probe.predict_logits(val[i]), yval)
                visit_losses.append(float(group_means(loss, validation_groups)[1].mean()))
            val_loss = float(np.mean(visit_losses))
            entry = {"visits": [visits[i] for i in indices], "config_index": config,
                     "validation_group_equal_ce_nats": val_loss,
                     "per_visit_validation_ce_nats": visit_losses, "probe": probe.receipt()}
            all_candidates.append(entry)
            candidates.append((val_loss, config, probe))
        selected.append(min(candidates, key=lambda a: (a[0], a[1]))[2])
    return SelectedProbes(specs[0].family, mode, shuffle_labels, tuple(visits), selected,
                          all_candidates, normalizer)


def evaluate_selected(selected: SelectedProbes, test_x: np.ndarray, test_y: np.ndarray,
                      test_groups: Sequence[str], *, seed: int,
                      bootstrap: int = 2000) -> dict:
    x = selected.normalizer.transform(test_x)
    if len(x) != len(selected.visits):
        raise ValueError("test visit axis does not match the selected study")
    y = _labels(test_y, x.shape[1], selected.probes[0].n_classes)
    losses, accuracy = [], []
    for index in range(len(x)):
        probe = selected.probes[0] if selected.mode == "shared" else selected.probes[index]
        loss, correct = scores(probe.predict_logits(x[index]), y)
        losses.append(loss)
        accuracy.append(correct)
    rows = []
    for index, visit in enumerate(selected.visits):
        _, group_loss = group_means(losses[index], test_groups)
        _, group_acc = group_means(accuracy[index].astype(float), test_groups)
        rows.append({"visit": visit, "test_group_equal_ce_nats": float(group_loss.mean()),
                     "test_group_equal_accuracy": float(group_acc.mean()),
                     "gain_from_input_only": paired_summary(losses[0], losses[index], test_groups,
                                                             seed=seed, bootstrap=bootstrap),
                     "item_ce_nats": losses[index].tolist(),
                     "item_correct": accuracy[index].tolist()})
    receipt = selected.receipt()
    return {"selection_sha256": json_hash(receipt), "family": selected.family,
            "mode": selected.mode, "shuffled_training_labels": selected.shuffled_training_labels,
            "rows": rows}

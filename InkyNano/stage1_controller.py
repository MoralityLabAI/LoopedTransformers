"""Pass-conditioned predictors and cost-accounted physical residency replay."""
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np
import torch
from torch import nn

from expert_lattice import ExpertLattice
from stage1_data import ACTIVE, EXPERTS, LAYERS, PASSES, collapse

CAPACITIES = (8, 16, 32)
BUDGETS = (0, 2, 4, 8)
CONTROLLERS = {
    "independent": (False, False),
    "layer_mesh": (True, False),
    "pass_mesh": (True, True),
}
LATTICE = ExpertLattice(EXPERTS)


class PassController(nn.Module):
    """Identical parameterization for private, layer-mesh and inter-pass arms."""

    def __init__(self, width, layer_messages, pass_messages):
        super().__init__()
        self.layer_messages = layer_messages
        self.pass_messages = pass_messages
        self.encoder = nn.Linear(2 * EXPERTS, width)
        self.layer_id = nn.Embedding(LAYERS, width)
        self.pass_id = nn.Embedding(PASSES, width)
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                     nn.Linear(width, EXPERTS))
        self.register_buffer("active", torch.tensor(ACTIVE), persistent=False)

    def forward(self, x):
        h = torch.tanh(self.encoder(x) + self.layer_id.weight[:, None] + self.pass_id.weight)
        block, shared, previous_pass = h, h, h
        if self.layer_messages:
            # Same-pass layer/mesh messages cannot convey other passes' observations.
            mask = self.active.reshape(4, 6, PASSES)
            pooled = (h.reshape(len(h), 4, 6, PASSES, -1) * mask[..., None]).sum(dim=2)
            block = (pooled / mask.sum(dim=1).clamp_min(1)[..., None]).repeat_interleave(6, dim=1)
            shared = (pooled.sum(dim=1) / self.active.sum(dim=0)[..., None])[:, None].expand_as(h)
        if self.pass_messages:
            # Prior-pass encoding of prompt and completed-prefix features at this boundary.
            previous_pass = torch.cat((h[:, :, :1], h[:, :, :-1]), dim=2)
        return self.readout(torch.cat((h, block, shared, previous_pass), dim=-1))


def fit_or_restore(path, dataset, train, name, settings, device, guard):
    torch.manual_seed(settings["seed"])
    model = PassController(settings["width"], *CONTROLLERS[name]).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 250_000:
        raise ValueError("controller exceeds the Stage 1 parameter cap")
    if path.is_file():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["settings"] != settings or checkpoint["controller"] != name:
            raise ValueError(f"incompatible checkpoint: {path}")
        model.load_state_dict(checkpoint["state"])
        return model, checkpoint["training"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = perf_counter()
    counts = Counter(dataset.windows[i].turn.root for i in train)
    weights = np.array([len(train) / (len(counts) * counts[dataset.windows[i].turn.root])
                        for i in train], dtype=np.float32)
    rng = np.random.default_rng(settings["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(settings["epochs"]):
        order = rng.permutation(len(train))
        for start in range(0, len(order), settings["batch"]):
            guard.check()
            offsets = order[start:start + settings["batch"]]
            samples = train[offsets]
            x = torch.from_numpy(dataset.x[samples]).to(device)
            y = torch.from_numpy(dataset.y[samples]).to(device)
            weight = torch.from_numpy(weights[offsets]).to(device)
            logits = model(x)
            loss = -(y * logits.log_softmax(dim=-1)).sum(dim=(-1, -2, -3)) / int(ACTIVE.sum())
            loss = (loss * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            guard.check()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stats = {"parameters": parameters, "training_seconds": perf_counter() - before,
             "memory": guard.receipt()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "controller": name, "settings": settings, "training": stats,
    }, temporary)
    temporary.replace(path)
    return model, stats


@torch.inference_mode()
def predict(model, dataset, dev, batch, device, guard):
    model.eval()
    scores = np.empty((len(dev), LAYERS, EXPERTS), dtype=np.float32)
    for start in range(0, len(dev), batch):
        samples = dev[start:start + batch]
        x = torch.from_numpy(dataset.x[samples]).to(device)
        probabilities = model(x).softmax(dim=-1) * model.active[..., None]
        scores[start:start + len(samples)] = collapse(probabilities.cpu().numpy())
        guard.check()
    return scores


def refresh_decision(score, previous, old, proposed, expert_bytes, threshold, cost):
    """Gate a feasible proposal using shift, expected mass gain, entropy and bytes."""
    if old == proposed:
        return False
    drift = 0.5 * np.abs(score - previous).sum()
    nonzero = score[score > 0]
    certainty = max(0.0, 1 + float(nonzero @ np.log(nonzero)) / np.log(EXPERTS))
    gain = float(score[sorted(proposed - old)].sum() - score[sorted(old - proposed)].sum())
    transfer = expert_bytes * len(proposed - old)
    return bool(drift >= threshold and gain * certainty > cost * transfer / (64 * (1 << 20)))


def evaluate(dataset, dev, scores, capacity, budget, expert_bytes, guard, adaptive=None):
    """Replay one cache per physical layer, with cold starts at turn boundaries."""
    roots = defaultdict(lambda: {
        "mass": np.zeros(PASSES, dtype=np.float64), "pass_visits": np.zeros(PASSES, dtype=np.int64),
        "admissions": 0, "cold_admissions": 0, "estimated_transfer_bytes": 0,
        "eligible_layer_boundaries": 0, "changed_layer_boundaries": 0,
    })
    last_turn, old, previous = None, None, None
    for j, i in enumerate(dev):
        window = dataset.windows[i]
        stats = roots[window.turn.root]
        tokens = window.end - window.start
        update = old is not None and window.turn == last_turn
        selected = []
        for layer in range(LAYERS):
            prior = old[layer] if update else None
            score = scores[j, layer]
            chosen = LATTICE.admit(score, prior, capacity, budget)
            if update and budget > 0:
                stats["eligible_layer_boundaries"] += 1
                if adaptive and not refresh_decision(
                    score, previous[layer], prior, chosen, expert_bytes[layer], *adaptive
                ):
                    chosen = prior
                stats["changed_layer_boundaries"] += chosen != prior
            admitted = len(chosen - (prior or LATTICE.bottom))
            stats["admissions"] += admitted
            stats["cold_admissions"] += admitted if not update else 0
            stats["estimated_transfer_bytes"] += admitted * expert_bytes[layer]
            stats["mass"] += dataset.y[i, layer][:, sorted(chosen)].sum(axis=-1) * tokens
            selected.append(chosen)
        stats["pass_visits"] += ACTIVE.sum(axis=0) * tokens
        last_turn, old, previous = window.turn, selected, scores[j]
        if j % 8 == 0:
            guard.check()
    for stats in roots.values():
        mass, visits = stats.pop("mass"), stats.pop("pass_visits")
        stats["retained_mass"] = float(mass.sum() / visits.sum())
        stats["retained_mass_by_pass"] = (mass / visits).tolist()
        stats["visits"] = int(visits.sum())
        stats["replacement_admissions"] = stats["admissions"] - stats["cold_admissions"]
    return {
        "root_mean_mass": float(np.mean([stats["retained_mass"] for stats in roots.values()])),
        "admissions": sum(stats["admissions"] for stats in roots.values()),
        "cold_admissions": sum(stats["cold_admissions"] for stats in roots.values()),
        "estimated_transfer_bytes": sum(stats["estimated_transfer_bytes"] for stats in roots.values()),
        "eligible_layer_boundaries": sum(stats["eligible_layer_boundaries"] for stats in roots.values()),
        "changed_layer_boundaries": sum(stats["changed_layer_boundaries"] for stats in roots.values()),
        "n_roots": len(roots), "roots": dict(roots),
    }


def controls(dataset, dev, frequency):
    prompt = collapse(dataset.x[dev, ..., :EXPERTS])
    truth = collapse(dataset.y[dev])
    previous = prompt.copy()
    static_oracle = np.empty_like(truth)
    spans = defaultdict(list)
    for j, i in enumerate(dev):
        window = dataset.windows[i]
        spans[window.turn].append(j)
        if window.start != window.turn.prompt:
            previous[j] = truth[j - 1]
    for indices in spans.values():
        weights = [dataset.windows[dev[j]].end - dataset.windows[dev[j]].start for j in indices]
        static_oracle[indices] = np.average(truth[indices], axis=0, weights=weights)
    return {"global_frequency": np.broadcast_to(frequency, truth.shape), "prompt": prompt,
            "previous_window": previous}, truth, static_oracle

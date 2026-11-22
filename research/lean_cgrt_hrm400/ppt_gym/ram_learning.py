"""FP32 online learners and a differentiable support/query encoder update."""
from __future__ import annotations

import hashlib
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from .core import Config, FastHead, PlasticProofTransformer, RetrievalControl
from .data import canonical

# Widths chosen to make the two plans approximately parameter matched at each size.
WIDTHS = {"micro": (96, 56), "tiny": (320, 192),
          "small": (640, 384), "medium": (1152, 704)}
DEPTHS = (1, 2, 4, 8, 16)
DOSES = (0, 1, 2, 4, 8, 16)
ARMS = ("sgd", "rls", "rank8", "retrieval", "frozen", "shuffled",
        "extra_compute", "static", "no_rollback", "zero_messages", "rolled_messages", "auto8", "horn")


def build_model(plan, preset):
    width = WIDTHS[preset][plan == "b"]
    # Plan A medium needs additional distinct capacity in its tied cell.
    ratio = 4 if plan == "a" and preset == "medium" else 2
    return PlasticProofTransformer(Config(plan=plan, width=width, mlp_ratio=ratio)).cpu().float().eval()


def tensor_bytes(tensor):
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def tensor_hash(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(canonical([key, list(tensor.shape), str(tensor.dtype)]).encode())
        h.update(tensor_bytes(tensor))
    return h.hexdigest()


class Online:
    def __init__(self, width):
        self.sgd = FastHead(width, learning_rate=2.0)
        self.shuffled = FastHead(width, learning_rate=2.0)
        self.retrieval = RetrievalControl(width, limit=128)
        self.w = torch.zeros(width, dtype=torch.float32)
        self.cov = torch.eye(width, dtype=torch.float32) * 4
        generator = torch.Generator().manual_seed(804)
        self.adapter = nn.Sequential(nn.Linear(width, 8, bias=False), nn.Tanh(),
                                     nn.Linear(8, 1, bias=False))
        with torch.no_grad():
            self.adapter[0].weight.copy_(torch.randn((8, width), generator=generator) * 0.1)
            self.adapter[2].weight.zero_()
        self.optimizer = torch.optim.SGD(self.adapter.parameters(), lr=0.4, momentum=0.8)
        self.version = 0

    def state(self):
        return {"sgd": self.sgd.weights, "shuffled": self.shuffled.weights,
                "w": self.w, "cov": self.cov, "adapter": self.adapter.state_dict(),
                "optimizer": self.optimizer.state_dict(), "retrieval": self.retrieval.items,
                "version": self.version}

    def restore(self, state):
        self.sgd.weights.copy_(state["sgd"])
        self.shuffled.weights.copy_(state["shuffled"])
        self.w.copy_(state["w"])
        self.cov.copy_(state["cov"])
        self.adapter.load_state_dict(state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.retrieval.items = state["retrieval"]
        self.version = state["version"]

    def weights(self):
        return {"sgd": self.sgd.weights, "shuffled": self.shuffled.weights, "rls": self.w,
                **{f"rank8.{key}": val for key, val in self.adapter.state_dict().items()}}

    def score(self, arm, features):
        if arm == "rls":
            return F.normalize(features, dim=-1) @ self.w
        if arm == "rank8":
            return self.adapter(F.normalize(features, dim=-1)).squeeze(-1)
        if arm in ("sgd", "shuffled", "retrieval"):
            return getattr(self, arm).predict(features)
        return features.new_zeros(len(features))

    def update(self, feature, label, shuffled):
        feature = feature.detach()
        before = tensor_hash(self.weights())
        self.sgd.update(feature, label)
        self.shuffled.update(feature, shuffled)
        self.retrieval.update(feature, label)
        x = F.normalize(feature, dim=0)
        px = self.cov @ x
        denom = 1 + (x @ px)
        self.w.add_((2 * label - 1 - x @ self.w) * px / denom)
        self.cov.sub_(torch.outer(px, px) / denom)
        self.optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(self.adapter(x).squeeze(), x.new_tensor(float(label)))
        loss.backward()
        self.optimizer.step()
        self.version += 1
        weights = self.weights()
        if any(not torch.isfinite(tensor).all() for tensor in weights.values()):
            raise FloatingPointError("nonfinite DRAM adaptation state")
        return {"version": self.version, "label": label, "shuffled_label": shuffled,
                "feature_fp32": tensor_bytes(feature).hex(), "before": before,
                "after": tensor_hash(weights)}


def encode_packet(text, menu):
    packet = json.loads(text)
    goals = packet["state"]["goals"]
    goal = goals[0]
    hypotheses = {hyp["name"]: hyp["type"] for hyp in goal["hypotheses"]}
    actions = [canonical(hypotheses.get(tactic.removeprefix("apply "))) + " " + tactic for tactic in menu]
    goal_text = canonical({"pending": len(goals), "target": goal["target"],
                           "failed": packet["failed_choices"]})
    return goal_text, actions


def read_features(model, text, menu, remaining, depth, message_mode="normal", pass_costs=None):
    goal, actions = encode_packet(text, menu)
    return model(goal, actions, depth=depth, checks_remaining=remaining, message_mode=message_mode,
                 pass_costs=pass_costs)[-1]


def model_macs(model, depth, candidates, message_mode="normal"):
    """Dense multiply-accumulate count for convs, attention, linear maps and readout."""
    d, slots, cfg = model.cfg.width, model.cfg.slots, model.cfg
    block = (4 + 2 * cfg.mlp_ratio) * slots * d * d + 2 * slots * slots * d
    fixed = (candidates + 1) * cfg.max_bytes * 3 * d * d + (2 * d + 1) * d
    readout = candidates * (3 * d * d + 2 * d) + d
    if cfg.plan == "b":
        return fixed + readout + 2 * block + depth * (2 * block + 6 * slots * d * d)
    return fixed + readout + depth * block


def credit(episode):
    """Return action indices and targets from verified paths and exhausted branches."""
    labels, path = {}, []
    for i, step in enumerate(episode["events"]):
        if step["kind"] == "rollback":
            at = step["restore_script_length"]
            for abandoned in path[at:]:
                labels[abandoned] = (0, "abandoned_branch")
            path = path[:at]
        elif step["status"] == "rejected":
            labels[i] = (0, "lean_rejection")
        else:
            path.append(i)
    if episode["verified_closed"]:
        for index in path:
            labels[index] = (1, "independently_certified_path")
    return sorted(labels.items())


def fit_encoder(model, optimizer, support, query, depth):
    """Backpropagate held-out training-root loss through one differentiable SGD step."""
    def batch(episode):
        pairs = []
        for index, (label, _) in credit(episode):
            event = episode["events"][index]
            out = read_features(model, canonical(event["input"]), event["candidate_menu"],
                                event["input"]["checks_remaining"], depth)
            pairs.append((out["features"][event["selected_index"]], label))
        x, y = zip(*pairs)
        return F.normalize(torch.stack(x), dim=-1), torch.tensor(y, dtype=torch.float32)

    if not credit(support) or not credit(query):
        return None
    start = time.perf_counter()
    x, y = batch(support)
    q, target = batch(query)
    origin = x.new_zeros(model.cfg.width, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(x @ origin, y)
    (gradient,) = torch.autograd.grad(loss, origin, create_graph=True)
    meta_loss = F.binary_cross_entropy_with_logits(q @ (origin - 2 * gradient), target)
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()
    return {"loss": float(meta_loss.detach()), "wall_seconds": time.perf_counter() - start}

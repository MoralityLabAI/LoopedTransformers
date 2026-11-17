"""MIRROR-02: counterfactual operator-composition stress test.

A self-contained synthetic gym. PyTorch is imported only for neural operations.
The evaluator owns hidden persona/base bits. Policies receive a frozen public View.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Optional

# First four contexts are supplied to both training and OOD evaluation.
# Remaining unscoped contexts are compositions of the training operators.
TRAIN_CONTEXTS = ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0),
                  (0, 0, 1, 0), (1, 1, 1, 1))
OOD_CONTEXTS = TRAIN_CONTEXTS[:4] + ((1, 1, 0, 0), (1, 0, 1, 0),
                                     (0, 1, 1, 0), (1, 1, 1, 0), TRAIN_CONTEXTS[4])
TRAIN_PERSONAS = (0, 1, 2, 4)
OOD_PERSONAS = (0, 3, 5, 6, 7)


def parity(n: int) -> int:
    return n.bit_count() & 1


@dataclass(frozen=True)
class World:
    features: tuple[tuple[int, int, int], ...]
    contexts: tuple[tuple[int, int, int, int], ...]
    bases: tuple[int, ...]
    persona: int
    scope_flips: tuple[int, ...]

    @classmethod
    def sample(cls, rng: random.Random, split: str = "train") -> "World":
        if split not in ("train", "iid", "ood"):
            raise ValueError(split)
        family_count = 6 if split != "ood" else 9
        contexts = OOD_CONTEXTS if split == "ood" else TRAIN_CONTEXTS
        personas = OOD_PERSONAS if split == "ood" else TRAIN_PERSONAS
        return cls(tuple(tuple(rng.randrange(2) for _ in range(3))
                         for _ in range(family_count)), contexts,
                   tuple(rng.randrange(2) for _ in range(family_count)),
                   rng.choice(personas),
                   tuple(rng.randrange(2) for _ in range(family_count)))

    def answer(self, family: int, context: int) -> int:
        x = self.features[family]
        t = self.contexts[context]
        mask = sum((x[k] & t[k]) << k for k in range(3))
        return self.bases[family] ^ parity(mask & self.persona) ^ (self.scope_flips[family] if t[3] else 0)

    def inconsistent(self) -> bool:
        for f in range(len(self.features)):
            if len({self.answer(f, c) for c, t in enumerate(self.contexts) if not t[3]}) > 1:
                return True
        return False


@dataclass(frozen=True)
class View:
    """Only public context/features and acquired answers cross the policy boundary."""
    features: tuple[tuple[int, int, int], ...]
    contexts: tuple[tuple[int, int, int, int], ...]
    observations: tuple[tuple[int, ...], ...]

    def candidates(self) -> tuple[tuple[int, int], ...]:
        return tuple((f, c) for f, row in enumerate(self.observations)
                     for c, t in enumerate(self.contexts) if row[c] == -1 and not t[3])

    def known(self, family: int) -> tuple[int, ...]:
        return tuple(self.observations[family][c] for c, t in enumerate(self.contexts)
                     if not t[3] and self.observations[family][c] >= 0)


def witness(view: View) -> Optional[dict]:
    for f, row in enumerate(view.observations):
        for a, av in enumerate(row):
            if av == -1 or view.contexts[a][3]:
                continue
            for b in range(a + 1, len(row)):
                if row[b] != -1 and not view.contexts[b][3] and row[b] != av:
                    return {"family": f, "first": {"context": a, "answer": av},
                            "second": {"context": b, "answer": row[b]},
                            "certificate": "declared_unscoped_semantic_equivalence"}
    return None


class Dialogue:
    def __init__(self, world: World):
        self._world = world
        self._observations = [[-1] * len(world.contexts) for _ in world.features]
        self.trace: list[dict] = []

    def view(self) -> View:
        return View(self._world.features, self._world.contexts,
                    tuple(tuple(row) for row in self._observations))

    def revise(self, family: int, context: int, value: int) -> None:
        """An explicit target correction supersedes the active answer."""
        if self._observations[family][context] == -1:
            raise ValueError("cannot revise an unasked question")
        if value not in (0, 1):
            raise ValueError("correction must be binary")
        self._observations[family][context] = value
        self.trace.append({"family": family, "context": context, "answer": value, "revision": True})

    def ask(self, family: int, context: int) -> int:
        if (family, context) not in self.view().candidates():
            raise ValueError("question is unavailable or outside declared equivalence class")
        reply = self._world.answer(family, context)
        self._observations[family][context] = reply
        self.trace.append({"family": family, "context": context, "answer": reply})
        return reply


def paired(view: View) -> tuple[int, int]:
    # Treats the problem as searching candidate paired questions by feature density.
    for f in sorted(range(len(view.features)), key=lambda i: (-sum(view.features[i]), i)):
        for c, t in enumerate(view.contexts):
            if not t[3] and view.observations[f][c] == -1:
                return f, c
    raise ValueError("exhausted")


def exact_bayes(view: View) -> tuple[int, int]:
    """Information-rich analytic baseline that knows the public synthetic grammar.

    Inference enumerates all eight persona masks but never accesses true hidden bits.
    Base answer is integrated out by conditioning on the earliest observed reply.
    """
    possible = []
    for persona in range(8):
        valid = True
        for f, row in enumerate(view.observations):
            known = [(c, a) for c, a in enumerate(row)
                     if a >= 0 and not view.contexts[c][3]]
            if not known:
                continue
            anchor_c, anchor_a = known[0]
            x = view.features[f]
            for c, a in known[1:]:
                diff = sum((x[k] & (view.contexts[anchor_c][k] ^ view.contexts[c][k])) << k for k in range(3))
                if (anchor_a ^ a) != parity(diff & persona):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            possible.append(persona)
    assert possible, "synthetic evidence must be compatible with some persona"
    scored = []
    for f, c in view.candidates():
        known = [(a, v) for a, v in enumerate(view.observations[f]) if v >= 0 and not view.contexts[a][3]]
        x = view.features[f]
        if known:
            anchor_c, anchor_a = known[0]
            diff = sum((x[k] & (view.contexts[anchor_c][k] ^ view.contexts[c][k])) << k for k in range(3))
            pmismatch = sum(parity(diff & persona) for persona in possible) / len(possible)
            information = 4 * pmismatch * (1 - pmismatch)
            score = pmismatch + 0.05 * information
        else:
            # Value of having an anchor in this family before a second question.
            diffs = []
            for d, t in enumerate(view.contexts):
                if d == c or t[3] or view.observations[f][d] >= 0:
                    continue
                diff = sum((x[k] & (view.contexts[c][k] ^ t[k])) << k for k in range(3))
                diffs.append(sum(parity(diff & persona) for persona in possible) / len(possible))
            score = (max(diffs, default=0.0) * 0.55) + 0.01 * sum(x)
        scored.append((score, -f, -c, (f, c)))
    return max(scored)[-1]


def play(world: World, policy, budget: int = 8, seed: int = 0) -> dict:
    d = Dialogue(world)
    rng = random.Random(seed)
    for _ in range(budget):
        view = d.view()
        if not view.candidates():
            break
        if policy == "random":
            choice = rng.choice(view.candidates())
        elif policy == "paired":
            choice = paired(view)
        elif policy == "bayes":
            choice = exact_bayes(view)
        else:
            choice = policy.choose(view)
        d.ask(*choice)
        if witness(d.view()) is not None:
            break
    w = witness(d.view())
    return {"found": bool(w), "queries": len(d.trace), "ground_truth": world.inconsistent(),
            "witness": w, "trace": d.trace}


# Neural module is defined lazily to keep the gym usable without PyTorch.
def torch_components():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class TransportNet(nn.Module):
        def __init__(self, width=48, loops=2, transport="reflection", rank=12):
            super().__init__()
            if transport not in ("reflection", "free", "none"):
                raise ValueError(transport)
            self.width, self.loops, self.transport = width, loops, transport
            self.feature = nn.Linear(3, width)
            self.context = nn.Linear(4, width)
            self.observation = nn.Embedding(3, width)
            if transport == "reflection":
                self.axes = nn.Parameter(torch.randn(width, 3) / math.sqrt(width))
            elif transport == "free":
                self.u = nn.Linear(width, rank, bias=False)
                self.gate = nn.Linear(4, rank, bias=False)
                self.v = nn.Linear(rank, width, bias=False)
            self.block = nn.TransformerEncoderLayer(width, 4, 2 * width, batch_first=True,
                                                    norm_first=True, dropout=0.0, activation="gelu")
            self.global_projection = nn.Linear(width, width)
            self.norm = nn.LayerNorm(width)
            self.reply_head = nn.Linear(width, 1)
            self.risk_head = nn.Linear(width, 1)
            self.action_head = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

        def transport_state(self, base, contexts):
            if self.transport == "none":
                return base[:, :, None, :].expand(-1, -1, contexts.shape[1], -1)
            if self.transport == "free":
                u = self.u(base)[:, :, None, :]
                gate = self.gate(contexts)[:, None, :, :]
                return base[:, :, None, :] + self.v(u * gate)
            # Orthogonal commuting Householder reflections implement (Z_2)^3.
            axes, _ = torch.linalg.qr(self.axes, mode="reduced")
            proj = torch.einsum("bfw,wk->bfk", base, axes)
            signed = proj[:, :, None, :] * contexts[:, None, :, :3]
            displacement = torch.einsum("bfck,wk->bfcw", signed, axes)
            return base[:, :, None, :] - 2 * displacement

        def forward(self, features, contexts, observed):
            b, f, _ = features.shape
            c = contexts.shape[1]
            if observed.shape != (b, f, c):
                raise ValueError("observation shape mismatch")
            base = self.feature(features)
            x = (self.transport_state(base, contexts)
                 + self.context(contexts)[:, None, :, :]
                 + self.observation(observed + 1))
            mask = (observed >= 0).float()
            for _ in range(self.loops):
                x = self.block(x.reshape(b * f, c, self.width)).reshape(b, f, c, self.width)
                # Globally shared latent persona, conditional on observed replies.
                evidence = (x * mask[..., None]).sum(dim=(1, 2))
                denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)[..., None]
                x = x + self.global_projection(evidence / denominator)[:, None, None, :]
            x = self.norm(x)
            reply = self.reply_head(x).squeeze(-1)
            valid = (1 - contexts[:, None, :, 3]).expand(b, f, c)
            family = (x * valid[..., None]).sum(dim=2) / valid.sum(dim=2).clamp_min(1)[..., None]
            risk = self.risk_head(family).squeeze(-1)
            return reply, risk, self.action_head(x).squeeze(-1)

        def composition_error(self, features, a, b):
            """Measured group-action error, independent of decoder behavior."""
            base = self.feature(features)
            zero = torch.zeros(features.shape[0], 1, 1, device=features.device)
            def as_context(bits):
                return torch.cat((bits[:, None, :], zero), dim=-1)
            ta = as_context(a)
            tb = as_context(b)
            tab = as_context((a.long() ^ b.long()).float())
            first = self.transport_state(base, ta)
            chained = self.transport_state(first.squeeze(2), tb).squeeze(2)
            direct = self.transport_state(base, tab).squeeze(2)
            return float((chained - direct).square().mean().item())

    class Policy:
        def __init__(self, model, use_actor=True):
            self.model = model.eval()
            self.use_actor = use_actor
        @torch.no_grad()
        def choose(self, view: View):
            features = torch.tensor([view.features], dtype=torch.float32)
            contexts = torch.tensor([view.contexts], dtype=torch.float32)
            observations = torch.tensor([view.observations], dtype=torch.long)
            logits, risk_logit, actions = self.model(features, contexts, observations)
            if self.use_actor:
                return max(view.candidates(), key=lambda pair: (float(actions[0, pair[0], pair[1]]), -pair[0], -pair[1]))
            p = logits[0].sigmoid().tolist()
            risks = risk_logit[0].sigmoid().tolist()
            scores = []
            for f, c in view.candidates():
                known = view.known(f)
                if known:
                    mismatch = max((1 - p[f][c] if y else p[f][c]) for y in known)
                    score = mismatch + 0.07 * risks[f]
                else:
                    entropy = 4 * p[f][c] * (1 - p[f][c])
                    score = 0.5 * risks[f] + 0.05 * entropy + (0.10 if c == 0 else 0)
                scores.append((score, -f, -c, (f, c)))
            return max(scores)[-1]

    def batch(rng, size, split="train"):
        worlds = [World.sample(rng, split) for _ in range(size)]
        features = torch.tensor([w.features for w in worlds], dtype=torch.float32)
        contexts = torch.tensor([w.contexts for w in worlds], dtype=torch.float32)
        answers = torch.tensor([[[w.answer(f, c) for c in range(len(w.contexts))]
                                 for f in range(len(w.features))] for w in worlds], dtype=torch.float32)
        obs = torch.full_like(answers, -1, dtype=torch.long)
        for i, w in enumerate(worlds):
            ids = [(f, c) for f in range(len(w.features)) for c in range(len(w.contexts))]
            rng.shuffle(ids)
            for f, c in ids[:rng.randrange(1, 8)]:
                obs[i, f, c] = int(answers[i, f, c])
        risk = torch.tensor([[int(len({w.answer(f, c) for c, t in enumerate(w.contexts)
                                       if not t[3]}) > 1) for f in range(len(w.features))]
                             for w in worlds], dtype=torch.float32)
        return features, contexts, obs, answers, risk

    def imitation_batch(rng, size):
        worlds = [World.sample(rng, "train") for _ in range(size)]
        features = torch.tensor([w.features for w in worlds], dtype=torch.float32)
        contexts = torch.tensor([w.contexts for w in worlds], dtype=torch.float32)
        answers = torch.tensor([[[w.answer(f, c) for c in range(len(w.contexts))]
                                 for f in range(len(w.features))] for w in worlds], dtype=torch.float32)
        views = []
        for w in worlds:
            d = Dialogue(w)
            for _ in range(rng.randrange(0, 6)):
                available = d.view().candidates()
                if not available or witness(d.view()):
                    break
                choice = exact_bayes(d.view()) if rng.random() < 0.65 else rng.choice(available)
                d.ask(*choice)
            if witness(d.view()):
                d = Dialogue(w)  # A terminal state has no valid next action.
            views.append(d.view())
        obs = torch.tensor([v.observations for v in views], dtype=torch.long)
        risk = torch.tensor([[int(len({w.answer(f, c) for c, t in enumerate(w.contexts)
                                       if not t[3]}) > 1) for f in range(len(w.features))]
                             for w in worlds], dtype=torch.float32)
        labels = torch.tensor([f * len(w.contexts) + c for w, view in zip(worlds, views)
                               for f, c in [exact_bayes(view)]], dtype=torch.long)
        allowed = torch.tensor([[[int((f, c) in view.candidates()) for c in range(len(view.contexts))]
                                 for f in range(len(view.features))] for view in views], dtype=torch.bool)
        return features, contexts, obs, answers, risk, labels, allowed

    def train(model, seed=7, steps=120, batch_size=32, imitation_weight=0.0):
        rng = random.Random(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
        for _ in range(steps):
            if imitation_weight:
                features, contexts, obs, answer, target_risk, action_labels, allowed = imitation_batch(rng, batch_size)
            else:
                features, contexts, obs, answer, target_risk = batch(rng, batch_size)
            logits, risk, actions = model(features, contexts, obs)
            mask = (obs == -1).float()
            bce = (F.binary_cross_entropy_with_logits(logits, answer, reduction="none") * mask).sum() / mask.sum()
            loss = bce + .2 * F.binary_cross_entropy_with_logits(risk, target_risk)
            if imitation_weight:
                masked = actions.flatten(1).masked_fill(~allowed.flatten(1), -1e9)
                imitation_loss = F.cross_entropy(masked, action_labels)
                loss = loss + imitation_weight * imitation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        return {"last_loss": float(loss.item()), "steps": steps, "imitation_weight": imitation_weight}

    return TransportNet, Policy, batch, train


def predictive_compositions(model, seed=10007, episodes=160):
    """Zero-shot accuracy on composite transformations after single-transform probes."""
    import torch
    import torch.nn.functional as F
    rng = random.Random(seed)
    worlds = [World.sample(rng, "ood") for _ in range(episodes)]
    views = []
    for w in worlds:
        d = Dialogue(w)
        for f in range(len(w.features)):
            d.ask(f, 0)  # known family anchor
        for f, c in ((0, 1), (1, 2), (2, 3)):
            d.ask(f, c)  # singleton transformations only
        views.append(d.view())
    features = torch.tensor([w.features for w in worlds], dtype=torch.float32)
    contexts = torch.tensor([w.contexts for w in worlds], dtype=torch.float32)
    observations = torch.tensor([v.observations for v in views], dtype=torch.long)
    answers = torch.tensor([[[w.answer(f, c) for c in range(4, 8)]
                             for f in range(len(w.features))] for w in worlds], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(features, contexts, observations)[0][:, :, 4:8]
        brier = (logits.sigmoid() - answers).square().mean()
        nll = F.binary_cross_entropy_with_logits(logits, answers)
        accuracy = ((logits > 0).float() == answers).float().mean()
    return {"n_predictions": int(answers.numel()), "composed_brier": float(brier),
            "composed_nll": float(nll), "composed_accuracy": float(accuracy)}


def benchmark(model=None, episodes=200, budget=8, seed=10007, split="ood", include_heuristic=False):
    rng = random.Random(seed)
    worlds = [World.sample(rng, split) for _ in range(episodes)]
    policies = {"random": "random", "paired": "paired", "bayes": "bayes"}
    if model is not None:
        _, Policy, _, _ = torch_components()
        policies["mirror"] = Policy(model, use_actor=True)
        if include_heuristic:
            policies["mirror_heuristic"] = Policy(model, use_actor=False)
    results = {}
    for name, policy in policies.items():
        episodes_out = [play(w, policy, budget=budget, seed=seed + i) for i, w in enumerate(worlds)]
        positive = sum(r["ground_truth"] for r in episodes_out)
        found = sum(r["found"] and r["ground_truth"] for r in episodes_out)
        false_positives = sum(r["found"] and not r["ground_truth"] for r in episodes_out)
        results[name] = {"positives": positive, "found": found,
                         "false_positives": false_positives,
                         "mean_queries": sum(r["queries"] for r in episodes_out) / episodes}
    return {"split": split, "seed": seed, "budget": budget, "episodes": episodes, "results": results}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--transport", choices=("reflection", "free", "none"), default="reflection")
    a.add_argument("--loops", type=int, default=2)
    a.add_argument("--steps", type=int, default=120)
    a.add_argument("--imitation-weight", type=float, default=0.0)
    a.add_argument("--budgets", default="4,8,12")
    a.add_argument("--episodes", type=int, default=200)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--output", type=Path)
    args = a.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    Model, _, _, train = torch_components()
    net = Model(loops=args.loops, transport=args.transport)
    trained = train(net, steps=args.steps, seed=args.seed, imitation_weight=args.imitation_weight)
    checks = []
    for budget in [int(x) for x in args.budgets.split(",")]:
        checks.append(benchmark(net, budget=budget, episodes=args.episodes,
                                seed=args.seed + 10000, split="ood", include_heuristic=True))
    features = torch.tensor([World.sample(random.Random(17)).features], dtype=torch.float32)
    ae = torch.tensor([[1., 0., 0.]])
    be = torch.tensor([[0., 1., 0.]])
    composition_error = net.composition_error(features, ae, be)
    report = {"seed": args.seed, "transport": args.transport, "loops": args.loops,
              "heldout_compositions": predictive_compositions(net, seed=args.seed + 10000, episodes=args.episodes),
              "parameters": sum(p.numel() for p in net.parameters()), "training": trained,
              "composition_mse": composition_error, "evaluations": checks}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

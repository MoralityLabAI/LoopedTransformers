"""Tied-weight counterfactual belief field; requires optional torch dependency."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from .gym import CONTEXTS, INVARIANT_CONTEXTS


class CounterfactualBeliefField(nn.Module):
    """Family beliefs transported to contexts, refined by a shared transformer.

    h[f,c] = E(features[f]) + E(f) + E(c) + T_c(E(features[f]))
             + E(observed_reply[f,c]); h <- shared_block(h) for K passes.
    Observation code: -1 unknown, 0 false, 1 true. No target hidden
    persona, bases, or unobserved answers are passed to forward().
    """
    def __init__(self, families=6, width=64, heads=4, loops=2, rank=16):
        super().__init__()
        if width % heads or loops < 1:
            raise ValueError("width must divide heads; loops >= 1")
        self.families = families
        self.loops = loops
        self.feature = nn.Linear(2, width)
        self.family = nn.Embedding(families, width)
        self.context = nn.Embedding(len(CONTEXTS), width)
        self.reply = nn.Embedding(3, width)
        self.transport_u = nn.Linear(width, rank, bias=False)
        self.transport_gate = nn.Embedding(len(CONTEXTS), rank)
        self.transport_v = nn.Linear(rank, width, bias=False)
        self.shared_block = nn.TransformerEncoderLayer(
            width, heads, width * 2, dropout=0.0, batch_first=True,
            norm_first=True, activation="gelu")
        self.norm = nn.LayerNorm(width)
        self.prediction = nn.Linear(width, 1)
        self.family_risk = nn.Linear(width, 1)

    def forward(self, features, observed):
        b, f, d = features.shape
        if f != self.families or d != 2 or observed.shape != (b, f, len(CONTEXTS)):
            raise ValueError("invalid feature or observation shape")
        if torch.any((observed < -1) | (observed > 1)):
            raise ValueError("observations must be -1, 0, or 1")
        base = self.feature(features) + self.family.weight[None, :, :]
        transported = self.transport_v(
            self.transport_u(base)[:, :, None, :]
            * self.transport_gate.weight[None, None, :, :])
        tokens = (base[:, :, None, :] + transported
                  + self.context.weight[None, None, :, :]
                  + self.reply(observed.long() + 1))
        x = tokens.reshape(b, f * len(CONTEXTS), -1)
        for _ in range(self.loops):
            x = self.shared_block(x)
        x = self.norm(x).reshape(b, f, len(CONTEXTS), -1)
        logits = self.prediction(x).squeeze(-1)
        risk = self.family_risk(x[:, :, :3].mean(dim=2)).squeeze(-1)
        return logits, risk


class MirrorPolicy:
    def __init__(self, model, device="cpu"):
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def choose(self, dialogue):
        feats = torch.tensor([dialogue.target.features], dtype=torch.float32, device=self.device)
        obs = torch.tensor([dialogue.observations()], dtype=torch.long, device=self.device)
        probabilities, risks = self.model(feats, obs)
        p = probabilities[0].sigmoid().cpu().tolist()
        risk = risks[0].sigmoid().cpu().tolist()
        candidates = []
        for f, c in dialogue.available():
            if c not in INVARIANT_CONTEXTS:
                continue  # scoped proposition lacks an equivalence certificate
            known = [dialogue.commitments[(f, k)].answer for k in INVARIANT_CONTEXTS
                     if (f, k) in dialogue.commitments]
            mismatch = max((1 - p[f][c] if y == 1 else p[f][c]
                            for y in known), default=0.0)
            entropy = -(p[f][c] * math.log(max(p[f][c], 1e-8))
                        + (1 - p[f][c]) * math.log(max(1 - p[f][c], 1e-8)))
            # A witness requires two observed answers; first question is exploration.
            score = (mismatch if known else 0.0) + 0.1 * entropy + 0.15 * risk[f]
            if not known and c == 0:
                score += 0.10
            candidates.append((score, -f, -c, (f, c)))
        if not candidates:
            return dialogue.available()[0]
        return max(candidates)[-1]


def sample_batch(rng, families, batch_size, device):
    from .gym import Target
    worlds = [Target.sample(rng, families) for _ in range(batch_size)]
    feature = torch.tensor([w.features for w in worlds], device=device, dtype=torch.float32)
    labels = torch.tensor([[[w.answer(f, c) for c in range(len(CONTEXTS))]
                            for f in range(families)] for w in worlds], device=device,
                          dtype=torch.float32)
    observed = torch.full_like(labels, -1, dtype=torch.long)
    for b in range(batch_size):
        indices = list(range(families * len(CONTEXTS)))
        rng.shuffle(indices)
        for flat in indices[:rng.randrange(1, min(len(indices) - 1, 10) + 1)]:
            f, c = divmod(flat, len(CONTEXTS))
            observed[b, f, c] = int(labels[b, f, c].item())
    risk = torch.tensor([[int(len({w.answer(f, c) for c in INVARIANT_CONTEXTS}) > 1)
                          for f in range(families)] for w in worlds],
                        device=device, dtype=torch.float32)
    return feature, observed, labels, risk


def train(model, steps=300, batch_size=32, lr=0.001, seed=7, device="cpu"):
    from random import Random
    rng = Random(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    last = None
    for step in range(steps):
        feat, observed, labels, risk_true = sample_batch(rng, model.families,
                                                         batch_size, device)
        logits, risk_logits = model(feat, observed)
        mask = (observed == -1).float()
        prediction_loss = (F.binary_cross_entropy_with_logits(logits, labels,
                                                               reduction="none") * mask).sum() / mask.sum()
        risk_loss = F.binary_cross_entropy_with_logits(risk_logits, risk_true)
        loss = prediction_loss + 0.25 * risk_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = (float(prediction_loss.item()), float(risk_loss.item()))
    return {"prediction_loss": last[0], "risk_loss": last[1],
            "steps": steps, "batch_size": batch_size, "train_seed": seed}

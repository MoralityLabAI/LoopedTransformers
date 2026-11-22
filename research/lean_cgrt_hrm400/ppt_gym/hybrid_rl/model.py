"""A single tied proof cell with gated recall from a typed message bank."""
from __future__ import annotations

import json

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock

from ..core import ByteEncoder, ROLES, block
from ..data import canonical


class ProofPolicy(nn.Module):
    """Shared observations, workspace, policy and critic for the hybrid and TRM."""

    def __init__(self, architecture: str, width: int = 256):
        super().__init__()
        if architecture not in ("hybrid", "trm") or width < 16 or width % 4:
            raise ValueError("expected hybrid or TRM with width divisible by four")
        self.architecture = architecture
        self.width = width
        self.text = ByteEncoder(width, 112)
        self.roles = nn.Parameter(torch.randn(len(ROLES), width) * 0.02)
        self.context = nn.Linear(2 * width + 1, width)
        self.norm = nn.LayerNorm(width)
        self.feature = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.actor = nn.Linear(width, 1)
        self.critic = nn.Linear(width, 1)
        if architecture == "hybrid":
            self.cell = block(width, 4, 2)
            self.recall = nn.MultiheadAttention(width, 4, batch_first=True)
            self.read_gate = nn.Linear(2 * width, width)
            self.write_gate = nn.Linear(width, width)
            self.write_content = nn.Linear(width, width)
        else:
            self.trm = nn.Sequential(PostBlock(width, 4, 2), PostBlock(width, 4, 2))
            self.register_buffer("answer_init", torch.randn(width) * 0.02)
            self.register_buffer("latent_init", torch.randn(width) * 0.02)

    def parameter_count(self):
        return sum(weight.numel() for weight in self.parameters())

    def macs(self, candidates, passes):
        """Dense MAC estimate, including encoders, policy and TRM latent steps."""
        d, n = self.width, len(ROLES)
        core = 8 * n * d * d + 2 * n * n * d
        shared = (1 + candidates) * self.text.max_bytes * 3 * d * d + (2 * d + 1) * d
        shared += candidates * (3 * d * d + d) + d
        if self.architecture == "trm":
            return shared + passes * 6 * core
        reads = 6 * n * d * d + 2 * n * n * d
        return shared + passes * core + (passes - 1) * (reads + 2 * n * d * d)

    def forward(self, packet, actions, passes, *, intervention=None, trace=None):
        if not actions or not 1 <= passes <= 16:
            raise ValueError("invalid proof action menu or recurrent depth")
        if intervention is not None:
            kind, at = intervention
            if self.architecture != "hybrid" or kind not in ("zero", "sham") or not 2 <= at <= passes:
                raise ValueError("message interventions require a carried message on a valid hybrid pass")
        packet = json.loads(packet)
        goals = packet["state"]["goals"]
        goal = goals[0]
        hypotheses = {item["name"]: item["type"] for item in goal["hypotheses"]}
        candidate_texts = [canonical(hypotheses.get(action.removeprefix("apply "))) + " " + action
                           for action in actions]
        goal_text = canonical({"pending": [item["target"] for item in goals[1:]],
                               "failures": packet["failed_choices"], "target": goal["target"]})
        text = self.text([goal_text, *candidate_texts])
        target, options = text[0], text[1:]
        summary = options.mean(dim=0)
        context = self.context(torch.cat((target, summary, target.new_tensor([packet["checks_remaining"] / 8]))))
        cue = self.roles + context
        cue = torch.cat(((cue[0] + target)[None], cue[1:4], (cue[4] + summary)[None], cue[5:]))
        cue = self.norm(cue.unsqueeze(0))
        if self.architecture == "trm":
            y, z = self.answer_init.expand_as(cue), self.latent_init.expand_as(cue)
            for _ in range(passes):
                for _ in range(2):
                    z = self.trm(cue + y + z)
                y = self.trm(y + z)
                if trace is not None:
                    trace.append({"state": y.detach()})
            workspace = y
        else:
            workspace, bank = cue, torch.zeros_like(cue)
            for index in range(passes):
                sent = bank
                if intervention is not None and index + 1 == intervention[1]:
                    sent = torch.zeros_like(bank) if intervention[0] == "zero" else bank.roll(1, dims=-1)
                if index:
                    recalled, _ = self.recall(workspace, sent, sent, need_weights=False)
                    gate = torch.sigmoid(self.read_gate(torch.cat((workspace, recalled), dim=-1)))
                    workspace = workspace + gate * recalled
                workspace = self.cell(workspace)
                if index + 1 < passes:
                    write = torch.sigmoid(self.write_gate(workspace))
                    bank = torch.lerp(bank, torch.tanh(self.write_content(workspace)), write)
                if trace is not None:
                    trace.append({"state": workspace.detach(), "received_bank": sent.detach()})
        anchor = workspace[:, 0].expand_as(options)
        features = self.feature(torch.cat((anchor, options, anchor * options), dim=-1))
        return self.actor(features).squeeze(-1), self.critic(workspace[0, 6]).squeeze(-1)

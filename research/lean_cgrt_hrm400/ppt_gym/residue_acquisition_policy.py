"""A tied policy that carries a source-feedback slot into query selection."""

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock

from .live_quote_policy import FEATURES, WIDTH, pad_features
from .residue_recovery import MODULUS


def tokens(pair, first_action, observed, support, warm, allowance, menu, anchor):
    first, target = pair["first"], pair["second"]
    paid = [pair["base"] / 8, pair["depth"] / 64, pair["family"] == "BPD.geom",
            observed is not None, bool(observed), *(first_action == index for index in range(3)),
            *support, *(warm[first_action] if first_action < 3 else (0.,) * 8)]
    costs = [cost / 7 for route in target["routes"] for cost in (route["pass_cost"], route["fail_cost"])]
    present = {value for value, ok in enumerate(support) if ok}
    groups = [set(route["residues"]) & present for route in target["routes"]]
    rejected = first["routes"][first_action]["fail_cost"] / 7 if first_action < 3 else 0.
    rows = []
    for action in range(len(menu) + 1):
        members = set(menu[action - 1]["members"]) if action else set()
        hits = [len(group & members) / MODULUS for group in groups]
        target_slot = [allowance / 7, *costs, *(len(group) / MODULUS for group in groups), *hits]
        context = [bool(action), action in anchor, anchor.get(action, 0.), rejected,
                   *(value in members for value in range(MODULUS)), len(members & present) / MODULUS]
        rows.append([pad_features(paid), pad_features(target_slot), pad_features(context)])
    return rows


class FeedbackSlotActor(nn.Module):
    """Use the same recurrent block to encode feedback and to route the two slots."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(FEATURES, WIDTH)
        self.roles = nn.Parameter(torch.randn(3, WIDTH) * .02)
        self.loop = PostBlock(WIDTH, 4, 2)
        self.message = nn.Linear(WIDTH, WIDTH, bias=False)
        self.gate = nn.Linear(2 * WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, features, *, reset=False):
        cue = self.embed(features) + self.roles
        paid, target = cue[:, :1], cue[:, 1:]
        for _ in range(2):
            paid = self.loop(cue[:, :1] + paid)
        paid_cue = self.roles[:1].expand_as(paid)
        cue = torch.cat((paid_cue, target), dim=1)
        state = torch.cat((paid_cue if reset else paid, target), dim=1)
        for _ in range(3):
            message = self.message(state[:, [1, 0, 2]])
            gate = self.gate(torch.cat((cue, state), dim=-1)).sigmoid()
            state = self.loop(cue + state + message * gate)
        return self.head(state[:, -1]).squeeze(-1)

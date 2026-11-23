"""Learn proof values and paid information acquisition from quote fragments."""

from itertools import product

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock

from .private_quotes import COMMITS, PENALTY, QUOTES


ACTIONS = (*COMMITS, *QUOTES)
HISTORIES = tuple(product((-1, 0, 1), repeat=len(QUOTES)))
HISTORY_INDEX = {history: index for index, history in enumerate(HISTORIES)}
HYPOTHESES = tuple(product((0, 1), repeat=len(QUOTES)))
CONTEXT_WIDTH = 3
WIDTH = 24


def history_index(history):
    observations = dict(history)
    return HISTORY_INDEX[tuple(observations.get(action, -1) for action in QUOTES)]


def decision_mask(fees, slack, history):
    left = slack - (fees * (history >= 0)).sum(dim=-1, keepdim=True)
    quotes = (history < 0) & (fees <= left)
    commits = torch.ones_like(quotes[..., :len(COMMITS)])
    return torch.cat((commits, quotes), -1), left


class BudgetQuoteBackup(nn.Module):
    def __init__(self):
        super().__init__()
        histories = torch.tensor(HISTORIES)
        hypotheses = torch.tensor(HYPOTHESES)
        compatible = ((histories[:, None, :] == hypotheses) | (histories[:, None, :] == -1)).all(-1)
        successors = [[[HISTORY_INDEX[(*history[:axis], bit, *history[axis + 1:])] for bit in (0, 1)]
                       for axis in range(len(QUOTES))] for history in HISTORIES]
        self.register_buffer("histories", histories)
        self.register_buffer("hypotheses", hypotheses.float())
        self.register_buffer("compatible", compatible)
        self.register_buffer("successors", torch.tensor(successors))
        for count in range(len(QUOTES) + 1):
            ids = ((histories >= 0).sum(-1) == count).nonzero().flatten()
            self.register_buffer(f"layer_{count}", ids)

    def posterior(self, log_weights):
        return log_weights[:, None].masked_fill(~self.compatible, -torch.inf).softmax(-1)

    def backup(self, commits, fees, mask, quote_probability, history_ids):
        table = torch.cat((commits, commits.new_full((*commits.shape[:-1], len(QUOTES)), -torch.inf)), -1)
        for depth in reversed(range(len(QUOTES))):
            indices = getattr(self, f"layer_{depth}")
            next_value = table.max(dim=-1).values[:, self.successors[indices]]
            p = quote_probability[:, indices]
            quotes = (1 - p) * next_value[..., 0] + p * next_value[..., 1] - fees[:, None] * PENALTY
            scores = torch.cat((commits[:, indices], quotes), dim=-1).masked_fill(~mask[:, indices], -torch.inf)
            table = table.index_copy(dim=1, index=indices, source=scores)
        rows = torch.arange(len(commits), device=commits.device)
        return table[rows, history_ids], quote_probability[rows, history_ids]


class BeliefQuotePolicy(BudgetQuoteBackup):
    """Two tied plan readouts under a posterior over possible paid feedback."""

    def __init__(self, plan_width):
        super().__init__()
        self.forecast = nn.Sequential(nn.Linear(CONTEXT_WIDTH + 2 * plan_width + 3, WIDTH), nn.GELU(),
                                      nn.Linear(WIDTH, len(HYPOTHESES)))
        self.commit = nn.Sequential(nn.Linear(CONTEXT_WIDTH + plan_width + 9, WIDTH), nn.GELU(),
                                    nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))
        self.register_buffer("plan_ids", torch.eye(len(COMMITS)))

    def forward(self, plans, context, fees, slack, history_ids):
        batch = plans.shape[0]
        histories = self.histories.expand(batch, -1, -1)
        mask, left = decision_mask(fees[:, None], slack[:, None], histories)
        log_weights = self.forecast(torch.cat((plans.flatten(1), context, fees / 3), -1))
        posterior = self.posterior(log_weights)
        quote_probability = posterior @ self.hypotheses

        dims = (batch, len(HISTORIES), len(HYPOTHESES), len(COMMITS))
        inputs = torch.cat((
            plans[:, None, None].expand(*dims, -1),
            context[:, None, None, None].expand(*dims, -1),
            fees[:, None, None, None].expand(*dims, -1) / 3,
            self.hypotheses[None, None, :, None].expand(*dims, -1),
            self.plan_ids.expand(*dims, -1),
            left[:, :, None, None].expand(*dims, -1) / 3,
        ), -1)
        # Per-plan success under each hypothetical feedback word. No word is
        # given by the environment until its corresponding quotes are bought.
        hypothetical = self.commit(inputs).squeeze(-1).sigmoid()
        commits = (hypothetical * posterior[..., None]).sum(dim=2)
        commits = commits * (left >= 0)

        return self.backup(commits, fees, mask, quote_probability, history_ids)


class RuleQuotePolicy(BudgetQuoteBackup):
    """Compose learned tariff forecasts along both plans using literal rule incidence."""

    def __init__(self, plan_width):
        super().__init__()
        self.word_weight = nn.Parameter(torch.zeros(len(QUOTES) + 1))
        self.charge = nn.Sequential(nn.Linear(2, 12), nn.GELU(), nn.Linear(12, 1))
        self.residual = nn.Linear(plan_width + CONTEXT_WIDTH, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.register_buffer("word_count", self.hypotheses.sum(-1).long())

    def forward(self, plans, context, fees, slack, history_ids, links):
        mask, left = decision_mask(fees[:, None], slack[:, None], self.histories.expand(len(plans), -1, -1))
        log_weights = self.word_weight[self.word_count].expand(len(plans), -1)
        posterior = self.posterior(log_weights)
        high = 8 * self.charge(links[..., :2]).sigmoid()
        direct, inverse = links[..., 2:4], links[..., 4:]
        billed = torch.einsum("iq,bqp->bip", self.hypotheses, high * direct)
        billed = billed + torch.einsum("iq,bqp->bip", 1 - self.hypotheses, high * inverse)
        typed = torch.cat((plans, context[:, None].expand(-1, len(COMMITS), -1)), -1)
        offset = .25 * self.residual(typed).squeeze(-1).tanh()
        conditional = ((left[:, :, None] - billed[:, None] - offset[:, None, None] + .5) / .30).sigmoid()
        commits = (conditional * posterior[..., None]).sum(2) * (left >= 0)
        return self.backup(commits, fees, mask, posterior @ self.hypotheses, history_ids)


class FullHistoryQuoteTRM(nn.Module):
    """Tied TRM updates over both proof plans, all quote channels, and public budget."""

    def __init__(self, plan_width, *, bound=False):
        super().__init__()
        self.plan = nn.Linear(plan_width + 2, WIDTH)
        self.quote = nn.Linear(13 if bound else 7, WIDTH)
        self.context = nn.Linear(CONTEXT_WIDTH + 2, WIDTH)
        self.recursion = nn.Sequential(PostBlock(WIDTH, 4, 2), PostBlock(WIDTH, 4, 2))
        self.readout = nn.Linear(WIDTH, 1)
        self.forecast = nn.Linear(WIDTH, 1)
        self.register_buffer("histories", torch.tensor(HISTORIES))
        self.register_buffer("plan_ids", torch.eye(len(COMMITS)))
        self.register_buffer("quote_ids", torch.eye(len(QUOTES)))
        self.register_buffer("answer_init", torch.randn(WIDTH) * .02)
        self.register_buffer("latent_init", torch.randn(WIDTH) * .02)

    def forward(self, plans, context, fees, slack, history_ids, links=None):
        history = self.histories[history_ids]
        mask, left = decision_mask(fees, slack, history)
        batch = len(plans)
        symbols = nn.functional.one_hot(history + 1, 3).to(plans.dtype)
        quote_tokens = torch.cat((symbols, fees[..., None] / 3, self.quote_ids.expand(batch, -1, -1)), -1)
        if links is not None:
            quote_tokens = torch.cat((quote_tokens, links), -1)
        plan_tokens = torch.cat((plans, self.plan_ids.expand(batch, -1, -1)), -1)
        global_token = self.context(torch.cat((context, slack / 3, left / 3), -1))[:, None]
        cue = torch.cat((self.plan(plan_tokens), self.quote(quote_tokens), global_token), dim=1)
        answer, latent = self.answer_init.expand_as(cue), self.latent_init.expand_as(cue)
        for _ in range(2):
            latent = self.recursion(cue + answer + latent)
        answer = self.recursion(answer + latent)
        values = self.readout(answer[:, :len(ACTIONS)]).squeeze(-1).masked_fill(~mask, -torch.inf)
        probabilities = self.forecast(answer[:, len(COMMITS):len(ACTIONS)]).squeeze(-1).sigmoid()
        return values, probabilities

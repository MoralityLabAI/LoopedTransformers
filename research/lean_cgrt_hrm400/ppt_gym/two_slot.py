"""A tied two-token proof cell with explicit carry between Lean tactics."""

from dataclasses import dataclass

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock

from .core import ByteEncoder, block
from .data import canonical
from .dependent_witness import action_projection, proof_hypotheses
from .rewrite import rewrite_hypothesis
from .slot_bindings import BINDING_CHANNELS, action_bindings, binding_features
from .slot_path_bank import SlotPathBank
from .slot_witness_bank import SlotWitnessBank, WitnessWorkspace
from .two_slot_fixture import CHECKS


MAX_CONTEXT_CHUNKS = 8


def goal_text(goal, *, witness=False, anonymous_witness=False):
    hypotheses = proof_hypotheses(goal) if witness else goal["hypotheses"]
    return canonical(goal["target"]) + " " + " ".join(
        f"{'hyp' if anonymous_witness else hyp['name']}:{hyp['type'].get('raw', canonical(hyp['type']))}"
        for hyp in hypotheses)


def text_chunks(text, limit):
    """Split at UTF-8 boundaries so the byte encoder sees the whole context."""
    if limit < 4:
        raise ValueError("a text chunk must hold any UTF-8 character")
    data = text.encode("utf-8")
    chunks, start = [], 0
    while start < len(data):
        end = min(start + limit, len(data))
        while end < len(data) and data[end] & 0xc0 == 0x80:
            end -= 1
        chunks.append(data[start:end].decode("utf-8"))
        start = end
    return chunks or [""]


def proof_bank(width, paths, witnesses, witness_workspace):
    if paths and witnesses:
        raise ValueError("select one typed proof bank")
    if witness_workspace and not witnesses:
        raise ValueError("persistent witness support requires the witness bank")
    if witnesses:
        return SlotWitnessBank(width)
    return SlotPathBank(width) if paths else None


@dataclass(frozen=True)
class WitnessState:
    slots: torch.Tensor
    workspace: WitnessWorkspace


def slot_tensor(state):
    return state.slots if isinstance(state, WitnessState) else state


def candidate_state(state, index):
    slots = slot_tensor(state)[index]
    return WitnessState(slots, state.workspace) if isinstance(state, WitnessState) else slots


class TwoSlotInputs(nn.Module):
    def __init__(self, width, passes, *, bindings, witness_workspace, anonymous_witness):
        super().__init__()
        self.width = width
        self.passes = passes
        self.text = ByteEncoder(width, 384)
        self.roles = nn.Parameter(torch.randn(2, width) * .02)
        self.budget = nn.Linear(1, width)
        self.binding = nn.Linear(BINDING_CHANNELS, width, bias=False) if bindings else None
        self.witness_workspace = witness_workspace
        self.anonymous_witness = anonymous_witness

    def start_bank(self, goals, menu, carry):
        if self.witness_workspace:
            return self.path_bank.resume(goals, menu, carry.workspace if carry is not None else None)
        view, bank, gates = self.path_bank.start(goals, menu)
        return view, bank, gates, None

    def encode(self, goals):
        if not 1 <= len(goals) <= 2:
            raise ValueError("two-slot policy requires one or two open goals")
        chunks = self.context_chunks(goals)
        embeddings = self.text([text for parts in chunks for text in parts])
        if any(len(parts) > 1 for parts in chunks):
            embeddings = torch.stack([item.mean(0) for item in embeddings.split([len(parts) for parts in chunks])])
        return embeddings + self.roles[:len(goals)]

    def context_chunks(self, goals):
        witness = isinstance(self.path_bank, SlotWitnessBank)
        chunks = [text_chunks(goal_text(goal, witness=witness, anonymous_witness=self.anonymous_witness),
                              self.text.max_bytes) for goal in goals]
        if any(len(parts) > MAX_CONTEXT_CHUNKS for parts in chunks):
            raise ValueError("two-slot goal exceeds the registered text budget")
        return chunks

    def extra_context_macs(self, goals):
        # The normal estimate already includes up to two context encodings.
        extra = max(0, sum(map(len, self.context_chunks(goals))) - 2)
        return extra * self.text.max_bytes * 3 * self.width * self.width

    def cues(self, goals, menu, checks_left, carry=None):
        observed = self.encode(goals)
        hypotheses = {hyp["name"]: hyp["type"] for hyp in goals[0]["hypotheses"]}
        witness_fork = (
            self.anonymous_witness and len(goals) == 2 and len(menu) > 1
            and all(action_projection(goals, action)[1] != goals[1]["target"] for action in menu)
        )
        actions = []
        for action in menu:
            name = (rewrite_hypothesis(action.removeprefix("all_goals "))
                    or action.removeprefix("exact ").removeprefix("apply "))
            # At a witness fork the typed bank carries each candidate's substitution.
            actions.append("exact witness" if witness_fork else f"{action} {canonical(hypotheses.get(name))}")
        cue = self.text(actions) + self.budget(self.roles.new_tensor([checks_left / CHECKS]))
        if carry is not None:
            observed = observed + carry[:len(goals)]
        slots = observed.unsqueeze(0) + cue.unsqueeze(1)
        if self.binding is not None:
            features = slots.new_tensor([action_bindings(goals, action) for action in menu])
            slots = slots + self.binding(features)
        return slots


class TwoSlotPolicy(TwoSlotInputs):
    def __init__(self, width=32, passes=2, *, bindings=False, paths=False, witnesses=False, residual_carry=False,
                 witness_workspace=False, anonymous_witness=False):
        super().__init__(width, passes, bindings=bindings, witness_workspace=witness_workspace,
                         anonymous_witness=anonymous_witness)
        if anonymous_witness and not witnesses:
            raise ValueError("an anonymous witness scorer requires the typed witness bank")
        self.cell = block(width, 4, 2)
        self.write = nn.Linear(2 * width, width)
        self.actor = nn.Linear(width, 1)
        self.path_bank = proof_bank(width, paths, witnesses, witness_workspace)
        self.residual_carry = residual_carry

    def macs(self, actions):
        d = self.width
        text = (4 + actions) * self.text.max_bytes * 3 * d * d
        binding = 2 * (actions + 1) * BINDING_CHANNELS * d if self.binding is not None else 0
        paths = self.path_bank.macs(actions, self.passes) if self.path_bank is not None else 0
        return text + actions * self.passes * (16 * d * d + 8 * d) + 8 * d * d + (actions + 1) * d + binding + paths

    def forward(self, goals, menu, checks_left, carry=None):
        slots = self.cues(goals, menu, checks_left, slot_tensor(carry))
        if self.path_bank is not None:
            view, bank, gates, keys = self.start_bank(goals, menu, carry)
        for _ in range(self.passes):
            if self.path_bank is not None:
                bank, evidence = self.path_bank.advance(view, bank, gates)
                slots = slots + evidence
            slots = self.cell(slots)
        state = WitnessState(slots, WitnessWorkspace(keys, bank)) if self.witness_workspace else slots
        return self.actor(slots[:, 0]).squeeze(-1), state

    def advance(self, predicted, goals):
        observed = self.encode(goals)
        if self.binding is not None:
            observed = observed + self.binding(observed.new_tensor([binding_features(goal) for goal in goals]))
        slots = slot_tensor(predicted)
        if len(slots) == 2 and len(goals) == 1:
            slots = slots[1:]
        elif len(slots) == 1 and len(goals) == 2:
            slots = torch.cat((slots, torch.zeros_like(slots)))
        carry = torch.lerp(slots, observed, self.write(torch.cat((slots, observed), dim=-1)).sigmoid())
        if self.residual_carry:
            carry = carry - observed
        return WitnessState(carry, predicted.workspace) if isinstance(predicted, WitnessState) else carry


class TwoSlotTRM(TwoSlotInputs):
    """TRM's tied answer/latent updates on the same two public goal tokens."""

    def __init__(self, width=28, passes=1, *, bindings=False, paths=False, witnesses=False, witness_workspace=False,
                 anonymous_witness=False):
        super().__init__(width, passes, bindings=bindings, witness_workspace=witness_workspace,
                         anonymous_witness=anonymous_witness)
        if anonymous_witness and not witnesses:
            raise ValueError("an anonymous witness scorer requires the typed witness bank")
        self.trm = nn.Sequential(PostBlock(width, 4, 2), PostBlock(width, 4, 2))
        self.actor = nn.Linear(width, 1)
        self.register_buffer("answer_init", torch.randn(width) * .02)
        self.register_buffer("latent_init", torch.randn(width) * .02)
        self.path_bank = proof_bank(width, paths, witnesses, witness_workspace)

    def macs(self, actions):
        d = self.width
        text = (2 + actions) * self.text.max_bytes * 3 * d * d
        binding = 2 * actions * BINDING_CHANNELS * d if self.binding is not None else 0
        paths = self.path_bank.macs(actions, 2 * self.passes) if self.path_bank is not None else 0
        return text + 6 * actions * self.passes * (16 * d * d + 8 * d) + (actions + 1) * d + binding + paths

    def forward(self, goals, menu, checks_left, carry=None):
        cue = self.cues(goals, menu, checks_left)
        answer, latent = self.answer_init.expand_as(cue), self.latent_init.expand_as(cue)
        if self.path_bank is not None:
            view, bank, gates, keys = self.start_bank(goals, menu, carry)
        for _ in range(self.passes):
            for _ in range(2):
                latent_input = cue + answer + latent
                if self.path_bank is not None:
                    bank, evidence = self.path_bank.advance(view, bank, gates)
                    latent_input = latent_input + evidence
                latent = self.trm(latent_input)
            answer = self.trm(answer + latent)
        state = WitnessState(answer, WitnessWorkspace(keys, bank)) if self.witness_workspace else answer
        return self.actor(answer[:, 0]).squeeze(-1), state

    def advance(self, predicted, goals):
        return predicted if isinstance(predicted, WitnessState) else None

"""MIRROR-01: synthetic epistemic dialogue with auditable contradiction witnesses.

Only assertions in an explicitly declared equivalence class can conflict.
The environment has hidden target personas; the examiner sees only questions,
answers, public family features, and the equivalence contract.
"""
from dataclasses import dataclass
import random
from typing import Optional

CONTEXTS = ("neutral", "role_swap", "consequence", "qualified_scope")
INVARIANT_CONTEXTS = frozenset((0, 1, 2))
PERSONAS = ("stable", "role_asymmetry", "consequence_asymmetry", "scope_only", "mixed")


@dataclass(frozen=True)
class Target:
    features: tuple[tuple[int, int], ...]
    bases: tuple[int, ...]
    persona: int

    @classmethod
    def sample(cls, rng: random.Random, families: int = 6) -> "Target":
        if families < 2:
            raise ValueError("families must be at least 2")
        return cls(
            tuple((rng.randrange(2), rng.randrange(2)) for _ in range(families)),
            tuple(rng.randrange(2) for _ in range(families)),
            rng.randrange(len(PERSONAS)),
        )

    def answer(self, family: int, context: int) -> int:
        if context not in range(len(CONTEXTS)) or family not in range(len(self.bases)):
            raise IndexError("unknown question")
        x, y = self.features[family]
        flip = (
            (context == 1 and self.persona in (1, 4) and x == 1)
            or (context == 2 and self.persona in (2, 4) and y == 1)
            or (context == 3 and (x != y or self.persona == 3))
        )
        return self.bases[family] ^ int(flip)

    def is_inconsistent(self) -> bool:
        return any(len({self.answer(f, c) for c in INVARIANT_CONTEXTS}) > 1
                   for f in range(len(self.bases)))


@dataclass(frozen=True)
class Commitment:
    family: int
    context: int
    answer: int
    turn: int


@dataclass(frozen=True)
class Witness:
    first: Commitment
    second: Commitment
    relation: str = "declared_context_invariance"


class Dialogue:
    def __init__(self, target: Target):
        self.target = target
        self.commitments: dict[tuple[int, int], Commitment] = {}
        self.history: list[Commitment] = []

    def observe(self, family: int, context: int, answer: Optional[int] = None) -> Commitment:
        if (family, context) in self.commitments:
            raise ValueError("duplicate question")
        value = self.target.answer(family, context) if answer is None else answer
        if value not in (0, 1):
            raise ValueError("answer must be binary")
        item = Commitment(family, context, value, len(self.history))
        self.history.append(item)
        self.commitments[(family, context)] = item
        return item

    def revise(self, family: int, context: int, value: int) -> Commitment:
        """An explicit correction replaces the prior active commitment."""
        if (family, context) not in self.commitments:
            raise ValueError("no prior commitment to revise")
        if value not in (0, 1):
            raise ValueError("answer must be binary")
        item = Commitment(family, context, value, len(self.history))
        self.history.append(item)
        self.commitments[(family, context)] = item
        return item

    def witness(self) -> Optional[Witness]:
        for f in range(len(self.target.bases)):
            items = [self.commitments[(f, c)] for c in sorted(INVARIANT_CONTEXTS)
                     if (f, c) in self.commitments]
            for i, left in enumerate(items):
                for right in items[i + 1:]:
                    if left.answer != right.answer:
                        return Witness(left, right)
        return None

    def available(self) -> list[tuple[int, int]]:
        return [(f, c) for f in range(len(self.target.bases))
                for c in range(len(CONTEXTS)) if (f, c) not in self.commitments]

    def observations(self) -> list[list[int]]:
        result = [[-1] * len(CONTEXTS) for _ in self.target.bases]
        for (f, c), item in self.commitments.items():
            result[f][c] = item.answer
        return result


def choose_paired(dialogue: Dialogue) -> tuple[int, int]:
    """Strong, deterministic paired-question baseline without hidden target access."""
    for f, features in sorted(enumerate(dialogue.target.features),
                              key=lambda pair: -sum(pair[1])):
        for c in (0, 1, 2):
            if (f, c) not in dialogue.commitments:
                return f, c
    return dialogue.available()[0]


def choose_random(dialogue: Dialogue, rng: random.Random) -> tuple[int, int]:
    return rng.choice(dialogue.available())


def run_episode(target: Target, policy: str, budget: int = 12,
                rng: Optional[random.Random] = None, model=None) -> dict:
    if policy not in ("random", "paired", "mirror", "oracle"):
        raise ValueError("unknown policy")
    rng = rng or random.Random(0)
    dialogue = Dialogue(target)
    for _ in range(min(budget, len(target.bases) * len(CONTEXTS))):
        if policy == "random":
            question = choose_random(dialogue, rng)
        elif policy == "paired":
            question = choose_paired(dialogue)
        elif policy == "oracle":
            # A diagnostic ceiling: the oracle alone may inspect hidden answers.
            candidates = [(f, c) for f, c in dialogue.available()
                          if c in INVARIANT_CONTEXTS]
            pairs = [(f, a, b) for f in range(len(target.bases))
                     for a in (0, 1, 2) for b in range(a + 1, 3)
                     if target.answer(f, a) != target.answer(f, b)]
            if pairs:
                f, a, b = pairs[0]
                question = (f, a) if (f, a) in candidates else (f, b)
                if question not in candidates:
                    question = candidates[0]
            else:
                question = candidates[0] if candidates else dialogue.available()[0]
        else:
            if model is None:
                raise ValueError("mirror policy requires a trained model")
            question = model.choose(dialogue)
        dialogue.observe(*question)
        if dialogue.witness():
            break
    witness = dialogue.witness()
    return {
        "found": witness is not None,
        "ground_truth": target.is_inconsistent(),
        "queries": len(dialogue.history),
        "witness": None if witness is None else {
            "first": vars(witness.first), "second": vars(witness.second),
            "relation": witness.relation,
        },
        "trace": [vars(item) for item in dialogue.history],
    }

"""Synthetic algorithmic tasks (finite-state tracking, arithmetic, multistep variable chains).

Each task has two formats sharing the same question:
  direct  : prompt ends in "A:"/"State:" and the target is only the final answer
  scratch : prompt ends in "C:"/"Trace:" and the target spells out the intermediate states
            (FSM states after each move, carries / partial products, every chain value) then the answer.
Training on `scratch` tokens is the *intermediate-state supervision* at the data level.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

# Fixed S5 generators (a: 5-cycle, b: transposition, c: 3-cycle) -> non-solvable group word problem.
_PERMS = {
    "a": [1, 2, 3, 4, 0],
    "b": [1, 0, 2, 3, 4],
    "c": [1, 2, 0, 3, 4],
}
FAMILIES = ("fsm", "add", "mul", "chain")

# training ranges (inclusive) and evaluation levels
TRAIN_RANGE = {"fsm": (4, 16), "add": (2, 6), "mul": (2, 3), "chain": (2, 6)}
EVAL_LEVELS = {  # (id levels), (ood levels)
    "fsm": ([8, 16], [32, 48]),
    "add": ([3, 6], [9, 12]),
    "mul": ([(2, 1), (3, 2)], [(4, 2), (5, 3)]),
    "chain": ([3, 6], [9, 12]),
}


@dataclass
class Example:
    family: str
    level: object
    fmt: str
    prompt: str
    target: str
    final_len: int          # number of trailing target tokens that are the final answer (before "\n")


def _rand_num(rng: random.Random, digits: int) -> int:
    lo = 10 ** (digits - 1) if digits > 1 else 0
    return rng.randint(lo, 10 ** digits - 1)


def gen_fsm(rng, n, fmt):
    moves = [rng.choice("abc") for _ in range(n)]
    s, trace = 0, []
    for m in moves:
        s = _PERMS[m][s]
        trace.append(s)
    q = "Moves: " + " ".join(moves) + "\n"
    if fmt == "direct":
        return Example("fsm", n, fmt, q + "State:", f"{s}\n", 1)
    return Example("fsm", n, fmt, q + "Trace:", ",".join(map(str, trace)) + f"\nState:{s}\n", 1)


def gen_add(rng, d, fmt):
    a, b = _rand_num(rng, d), _rand_num(rng, d)
    q = f"Q:{a}+{b}="
    ans = str(a + b)
    if fmt == "direct":
        return Example("add", d, fmt, q + "A:", ans + "\n", len(ans))
    xa, xb = str(a)[::-1], str(b)[::-1]
    carry, cs = 0, []
    for i in range(d):
        carry = (int(xa[i]) + int(xb[i]) + carry) // 10
        cs.append(carry)
    return Example("add", d, fmt, q + "C:", ",".join(map(str, cs)) + "|" + ans + "\n", len(ans))


def gen_mul(rng, lv, fmt):
    da, db = lv if isinstance(lv, tuple) else (lv, rng.randint(1, 2))
    a, b = _rand_num(rng, da), _rand_num(rng, db)
    q = f"Q:{a}*{b}="
    ans = str(a * b)
    if fmt == "direct":
        return Example("mul", (da, db), fmt, q + "A:", ans + "\n", len(ans))
    parts = [str(a * int(ch) * 10 ** i) for i, ch in enumerate(reversed(str(b)))]
    return Example("mul", (da, db), fmt, q + "C:", ",".join(parts) + "|" + ans + "\n", len(ans))


def gen_chain(rng, depth, fmt):
    names = rng.sample("abcdefghijklmnopqrstuvwxyz", depth + 1)
    v = rng.randint(0, 99)
    vals, stmts = [v], [f"{names[0]}={v}"]
    for i in range(1, depth + 1):
        op = rng.choice("+-*")
        k = rng.randint(2, 4) if op == "*" else rng.randint(1, 9)
        v = (vals[-1] + k if op == "+" else vals[-1] - k if op == "-" else vals[-1] * k) % 100
        vals.append(v)
        stmts.append(f"{names[i]}={names[i-1]}{op}{k}")
    q = "Q:" + ";".join(stmts) + f";{names[-1]}?"
    if fmt == "direct":
        return Example("chain", depth, fmt, q + "A:", f"{v}\n", len(str(v)))
    return Example("chain", depth, fmt, q + "C:", ",".join(map(str, vals)) + f"|{v}\n", len(str(v)))


_GEN = {"fsm": gen_fsm, "add": gen_add, "mul": gen_mul, "chain": gen_chain}


def train_example(rng: random.Random, p_scratch: float, families=FAMILIES) -> Example:
    fam = rng.choice(families)
    lo, hi = TRAIN_RANGE[fam]
    lv = rng.randint(lo, hi)
    if fam == "mul":
        lv = (rng.randint(2, 3), rng.randint(1, 2))
    fmt = "scratch" if rng.random() < p_scratch else "direct"
    return _GEN[fam](rng, lv, fmt)


def eval_examples(family: str, level, fmt: str, n: int, seed: int) -> list[Example]:
    rng = random.Random(f"{family}-{level}-{fmt}-{seed}")
    return [_GEN[family](rng, level, fmt) for _ in range(n)]


class TaskTokenizer:
    def __init__(self, tok_file: str, pad_id: int = 151643):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tok_file)
        self.pad_id = pad_id

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def collate(self, exs: list[Example], device="cpu"):
        P = self.encode_batch([e.prompt for e in exs])
        T = self.encode_batch([e.target for e in exs])
        n = max(len(p) + len(t) for p, t in zip(P, T))
        ids = torch.full((len(exs), n), self.pad_id, dtype=torch.long)
        tmask = torch.zeros(len(exs), n, dtype=torch.bool)      # target tokens (incl. trailing "\n")
        fmask = torch.zeros(len(exs), n, dtype=torch.bool)      # final-answer tokens only
        for i, (p, t, e) in enumerate(zip(P, T, exs)):
            ids[i, :len(p) + len(t)] = torch.tensor(p + t)
            tmask[i, len(p):len(p) + len(t)] = True
            end = len(p) + len(t) - 1                           # position of trailing "\n" token
            fmask[i, end - e.final_len:end] = True
        return ids.to(device), tmask.to(device), fmask.to(device)

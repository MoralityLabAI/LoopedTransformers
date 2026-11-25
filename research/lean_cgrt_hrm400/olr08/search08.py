"""Cost-ordered exact search over the OLR-08 action space (referee probe).

Only the original proposal enumeration and counted-prior pricing helpers are retained. Every action is one tactic run on the main
goal by the session server; Lean enumerates the arguments (local declarations, definitions in the
goal, lemmas whose side matches a subterm, lemmas whose conclusion matches the goal). The search
is best-first on a fixed hand prior, so it is an exact symbolic control. Nothing is learned here.
"""
from __future__ import annotations
import heapq, json, math, re, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import probe
from probe import read, write, key

NAME = re.compile(r"[^\W\d][\w'!?₀-₉]*")
CONST = re.compile(r"[\w.'!?₀-₉]+")

def actions(info, prem, k_rw=24, k_apply=6, max_locals=6, max_defs=5):
    """(cost, kind, argument, tactic) for the main goal."""
    out = [(1.0, "close", "", "simp_all"), (1.0, "close", "", "omega"), (1.5, "close", "", "simp [*]"),
           (1.5, "close", "", "rfl"), (2.0, "norm", "", "simp"), (3.0, "close", "", "exact?"),
           (2.5, "struct", "", "intros"), (2.5, "struct", "", "split"), (2.5, "struct", "", "constructor"),
           (3.0, "struct", "", "ext"), (3.0, "struct", "", "subst_vars"), (3.0, "norm", "", "simp at *")]
    hyps = [h for h in (info or {}).get("hyps", []) if h["acc"] and NAME.fullmatch(h["name"])]
    data = [h for h in hyps if not h["prop"]][-max_locals:]; props = [h for h in hyps if h["prop"]][-max_locals:]
    for h in data:
        c = 3.0 if h["ind"] else 4.5
        out.append((c, "induction", h["name"], f"induction {h['name']}")); out.append((c + .25, "cases", h["name"], f"cases {h['name']}"))
    for h in props:
        out += [(3.0, "rw_hyp", h["name"], f"rw [{h['name']}]"), (3.5, "rw_hyp_rev", h["name"], f"rw [← {h['name']}]"),
                (3.5, "cases_hyp", h["name"], f"cases {h['name']}"), (3.5, "apply_hyp", h["name"], f"apply {h['name']}")]
    for i, f in enumerate([d for d in (info or {}).get("defs", []) if CONST.fullmatch(d)][:max_defs]):
        out += [(3.0 + .25 * i, "simp_def", f, f"simp [{f}]"), (3.5 + .25 * i, "unfold", f, f"unfold {f}")]
    n = 0
    for x in (prem or {}).get("rw", []):
        if not CONST.fullmatch(x["name"]): continue
        if n >= k_rw: break
        if x["symm"]: out.append((4.0 + n / 6, "rw_lemma_rev", x["name"], f"rw [← {x['name']}]"))
        else:
            out.append((3.0 + n / 6, "simp_lemma", x["name"], f"simp [{x['name']}]"))
            out.append((3.75 + n / 6, "rw_lemma", x["name"], f"rw [{x['name']}]"))
        n += 1
    for i, x in enumerate([x for x in (prem or {}).get("apply", []) if CONST.fullmatch(x["name"])][:k_apply]):
        out.append((4.0 + i / 3, "apply_lemma", x["name"], f"apply {x['name']}"))
    return out

FREE = ("close", "norm", "struct")
RANKED = ("simp_lemma", "rw_lemma", "rw_lemma_rev", "apply_lemma", "simp_def", "unfold")

def reprice(acts, prior, depth):
    """Counted source prior: on-path rate of each template at this depth, times a monotone rank
    decay, normalised over the state's own action list. Cost is minus log2 of that probability."""
    rk = prior["rank"]; mono = [max(rk[: b + 1]) - rk[0] for b in range(len(rk))]
    seen = {}; w = []
    for c, kind, arg, tac in acts:                      # acts arrive in index order within a kind
        n = seen.get(kind, 0); seen[kind] = n + 1
        t = tac if kind in FREE else kind
        base = prior["weight"].get(f"{min(depth, 1)}|{t}", 0.002)
        w.append(base * 2.0 ** -(mono[min(n // 4, len(mono) - 1)] if kind in RANKED else 0.0))
    z = sum(w)
    return [(-math.log2(x / z), kind, arg, tac) for x, (c, kind, arg, tac) in zip(w, acts)]


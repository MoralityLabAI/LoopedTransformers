"""INK-BUDGET-01 offline policy evaluation on receipts from budget_eval.py.

Grid semantics (temperature 0, checked by budget_eval.prefix_check): for task t, effort e and cap c,
    success(t,e,c) = long run correct and finished (finish_reason=stop) within c tokens
    spent(t,e,c)   = completion tokens of the long run if it finished within c, else c
    seconds(t,e,c) = long-run seconds x spent / long-run completion tokens   (estimate, labelled)
Arms
    A/B/C  static low / medium / high effort at each cap
    D      verifier-triggered escalation ladders; every failed rung's tokens are charged
    E      learned difficulty router: per-effort success model, effort chosen by expected value - lam * cost
    F      learned token-budget allocator: low effort, cap from a predicted completion length, then escalation
    G      search with compact external memory: needs new calls; not evaluated offline
    H      oracle ceiling: cheapest (effort, cap) that succeeds
Learned arms (E, F) are scored leave-one-family-out: the evaluated family is never in training.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EFFORTS = ("low", "medium", "high")
CAPS = (256, 512, 1024, 2048, 4096, 8192)
LADDERS = {
    "D1 low1024>med4096>high8192": [("low", 1024), ("medium", 4096), ("high", 8192)],
    "D2 low2048>low8192": [("low", 2048), ("low", 8192)],
    "D3 low512>low2048>low8192": [("low", 512), ("low", 2048), ("low", 8192)],
    "D4 low1024>low4096>high8192": [("low", 1024), ("low", 4096), ("high", 8192)],
}


def load(records: Path) -> dict:
    """{task_id: {"family", "prompt_features", effort: {"ok", "ct", "fin", "sec"}}}"""
    grid: dict = {}
    for line in Path(records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = grid.setdefault(r["task_id"], {"family": r["family"]})
        g[r["effort"]] = {"ok": bool(r["correct"]), "fin": r["finish_reason"] == "stop",
                          "ct": int((r.get("usage") or {}).get("completion_tokens") or 0),
                          "sec": float(r.get("seconds") or 0.0)}
    return {k: v for k, v in grid.items() if all(e in v for e in EFFORTS)}


def outcome(cell: dict, cap: int) -> tuple[bool, int, float]:
    if cell["fin"] and cell["ct"] <= cap:
        return cell["ok"], cell["ct"], cell["sec"]
    spent = cap
    sec = cell["sec"] * spent / max(1, cell["ct"])
    return False, spent, sec


def run_static(grid, effort, cap):
    rows = {}
    for t, g in grid.items():
        ok, spent, sec = outcome(g[effort], cap)
        rows[t] = {"ok": ok, "tokens": spent, "seconds": sec, "truncated": not (g[effort]["fin"] and g[effort]["ct"] <= cap),
                   "attempts": 1}
    return rows


def run_ladder(grid, ladder):
    rows = {}
    for t, g in grid.items():
        tok = sec = 0.0
        ok = False
        trunc_final = False
        for i, (e, c) in enumerate(ladder):
            o, spent, s = outcome(g[e], c)
            tok += spent
            sec += s
            trunc_final = not (g[e]["fin"] and g[e]["ct"] <= c)
            if o:
                ok = True
                break
        rows[t] = {"ok": ok, "tokens": tok, "seconds": sec, "truncated": trunc_final, "attempts": i + 1}
    return rows


def run_oracle(grid):
    rows = {}
    for t, g in grid.items():
        best = None
        for e in EFFORTS:
            for c in CAPS:
                o, spent, s = outcome(g[e], c)
                if o and (best is None or spent < best[0]):
                    best = (spent, s)
        rows[t] = ({"ok": True, "tokens": best[0], "seconds": best[1], "truncated": False, "attempts": 1} if best else
                   {"ok": False, "tokens": min(outcome(g["low"], CAPS[0])[1], CAPS[0]), "seconds": 0.0,
                    "truncated": True, "attempts": 1})
    return rows


# ---- features and learned arms ---------------------------------------------------------------------
def features(prompt: str) -> np.ndarray:
    nums = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    return np.array([1.0, math.log1p(len(prompt)), math.log1p(prompt.count("\n")), math.log1p(len(nums)),
                     math.log1p(len(set(nums))), prompt.count(",") / max(1, len(prompt)) * 100], dtype=float)


def fit_logistic(X, y, l2=1.0, iters=400, lr=0.1):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * g
    return w


def fit_ridge(X, y, l2=1.0):
    A = X.T @ X + l2 * np.diag(np.r_[0, np.ones(X.shape[1] - 1)])
    return np.linalg.solve(A, X.T @ y)


def run_router(grid, prompts, lam=0.5, families=None):
    """Arm E: leave-one-family-out effort router; cap fixed at 8192 so only effort is chosen."""
    fams = sorted({g["family"] for g in grid.values()}) if families is None else families
    rows, choices = {}, defaultdict(int)
    for held in fams:
        tr = [t for t, g in grid.items() if g["family"] != held]
        te = [t for t, g in grid.items() if g["family"] == held]
        X = np.stack([features(prompts[t]) for t in tr])
        models = {}
        for e in EFFORTS:
            y = np.array([float(outcome(grid[t][e], 8192)[0]) for t in tr])
            yt = np.log1p(np.array([outcome(grid[t][e], 8192)[1] for t in tr], dtype=float))
            models[e] = (fit_logistic(X, y), fit_ridge(X, yt))
        for t in te:
            x = features(prompts[t])
            score = {}
            for e, (wp, wc) in models.items():
                p = 1 / (1 + math.exp(-float(x @ wp)))
                cost_k = math.expm1(float(x @ wc)) / 1000.0
                score[e] = p - lam * cost_k / 8.192
            e = max(score, key=score.get)
            choices[e] += 1
            ok, spent, sec = outcome(grid[t][e], 8192)
            rows[t] = {"ok": ok, "tokens": spent, "seconds": sec, "attempts": 1,
                       "truncated": not (grid[t][e]["fin"] and grid[t][e]["ct"] <= 8192), "choice": e}
    return rows, dict(choices)


def run_allocator(grid, prompts, quantile_margin=0.5, families=None):
    """Arm F: low effort; cap = smallest grid cap above the predicted log-length + margin; escalate x4 on failure."""
    fams = sorted({g["family"] for g in grid.values()}) if families is None else families
    rows = {}
    for held in fams:
        tr = [t for t, g in grid.items() if g["family"] != held and grid[t]["low"]["fin"]]
        te = [t for t, g in grid.items() if g["family"] == held]
        X = np.stack([features(prompts[t]) for t in tr])
        w = fit_ridge(X, np.log(np.array([max(1, grid[t]["low"]["ct"]) for t in tr], dtype=float)))
        for t in te:
            pred = math.exp(float(features(prompts[t]) @ w) + quantile_margin)
            start = next((c for c in CAPS if c >= pred), CAPS[-1])
            ladder = [("low", c) for c in CAPS if c >= start][::2] or [("low", CAPS[-1])]
            if ladder[-1][1] != CAPS[-1]:
                ladder.append(("low", CAPS[-1]))
            rows[t] = run_ladder({t: grid[t]}, ladder)[t] | {"start_cap": start}
    return rows


# ---- summaries ---------------------------------------------------------------------------------------
def summarize(rows: dict) -> dict:
    n = len(rows)
    succ = sum(r["ok"] for r in rows.values())
    tok = sum(r["tokens"] for r in rows.values())
    sec = sum(r["seconds"] for r in rows.values())
    fin = [r for r in rows.values() if not r["truncated"]]
    esc = [r for r in rows.values() if r["attempts"] > 1]
    return {"n": n, "success": succ, "tokens": int(tok), "success_per_1k_tokens": 1000 * succ / tok if tok else None,
            "success_per_second_est": succ / sec if sec else None, "truncation_rate": 1 - len(fin) / n if n else None,
            "correct_given_complete": (sum(r["ok"] for r in fin) / len(fin)) if fin else None,
            "escalated": len(esc), "escalation_rescues": sum(r["ok"] for r in esc),
            "failed_escalation_tokens": int(sum(r["tokens"] for r in esc if not r["ok"]))}


def evaluate(records: Path, tasks: Path) -> dict:
    grid = load(records)
    prompts = {}
    for line in Path(tasks).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            prompts[r["task_id"]] = r["prompt"]
    grid = {t: g for t, g in grid.items() if t in prompts}
    arms, paired = {}, {}
    for e, name in zip(EFFORTS, "ABC"):
        for c in CAPS:
            rows = run_static(grid, e, c)
            arms[f"{name} {e}@{c}"] = summarize(rows)
            paired[f"{name} {e}@{c}"] = rows
    for name, lad in LADDERS.items():
        rows = run_ladder(grid, lad)
        arms[name] = summarize(rows)
        paired[name] = rows
    for lam in (0.25, 1.0):
        rows, ch = run_router(grid, prompts, lam=lam)
        arms[f"E router lam={lam}"] = summarize(rows) | {"choices": ch}
        paired[f"E router lam={lam}"] = rows
    rows = run_allocator(grid, prompts)
    arms["F allocator"] = summarize(rows)
    paired["F allocator"] = rows
    rows = run_oracle(grid)
    arms["H oracle"] = summarize(rows)
    paired["H oracle"] = rows
    arms["G memory search"] = {"status": "not evaluated offline: requires new calls with an external memory"}
    # matched budget: best static arm whose total tokens do not exceed each adaptive arm's tokens
    static = {k: v for k, v in arms.items() if k[:1] in "ABC"}
    matched = {}
    for k, v in arms.items():
        if k[:1] in "DEF" and "tokens" in v:
            ok = [(s["success"], -s["tokens"], sk) for sk, s in static.items() if s["tokens"] <= v["tokens"]]
            best = max(ok) if ok else None
            matched[k] = {"adaptive_success": v["success"], "adaptive_tokens": v["tokens"],
                          "best_static_within_budget": None if best is None else best[2],
                          "static_success": None if best is None else best[0],
                          "static_tokens": None if best is None else -best[1]}
    fam = defaultdict(lambda: defaultdict(int))
    for t, g in grid.items():
        for e in EFFORTS:
            fam[g["family"]][e] += outcome(g[e], 8192)[0]
        fam[g["family"]]["n"] += 1
    return {"n_tasks": len(grid), "arms": arms, "matched_budget": matched,
            "families_at_8192": {f: dict(v) for f, v in sorted(fam.items())}, "paired": paired,
            "grid_semantics": __doc__.split("Arms")[0].strip()}

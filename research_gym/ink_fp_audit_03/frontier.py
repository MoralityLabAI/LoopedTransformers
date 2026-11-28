"""INK-RESEARCH-FRONTIER-01: harder verifier-backed research families.

Every task carries an exact verifier and, where useful, a compact certificate. Families
register into the audit-02 curriculum registry, so partitioning, leakage audits, exports and
the Inky evaluator work unchanged. Model-generated text is never trusted: answers are parsed
and checked against the task's own executable ground truth.

Families
  storyworld_v2         plan to a hidden goal under irreversible transitions, or NONE with a
                        compact unreachability certificate (max reachable value of a variable)
  experiment_choice     pick the experiment with the highest exact expected information gain per
                        unit cost among options that include confounded and underpowered ones
  power_design          minimum independent roots for one-sided exact McNemar power >= target
  graph_certificate     bipartite? answer a 2-colouring, or an odd cycle
  stability_certificate x_{t+1} = A x_t stable? (spectral radius < 1, exact rational A)
  sorting_network       comparator network that sorts n inputs, checked on all 2^n 0/1 inputs;
                        credit requires size <= registered bound
  hillclimb_diagnosis   from a real or synthetic hill-climb history: name the proposal rejected
                        more than once, or say whether the incumbent is a local optimum
  research_memory       choose the next experiment not already tested that addresses an open
                        question; rendered as full history or as a structured summary
"""
from __future__ import annotations

import itertools
import json
import math
import random
from fractions import Fraction

from research_gym.ink_fp_audit_02.curriculum import VERIFIERS, register
from research_gym.ink_fp_audit_02.gyms import SUFFIX, _task

SRC = "LoopedTransformers/ink_fp_audit_03 (new)"


def _t(family, root, partition, prompt, answer, content, verifier, difficulty, certificate=None, cost_ms=0.5):
    return _task(family, root, partition, prompt, answer, content, verifier=verifier,
                 generator=f"ink03.frontier.{family}", version="v1", source_repo=SRC,
                 difficulty=difficulty, cost_ms=cost_ms, certificate=certificate or {})


# ---- A. long-horizon storyworld -------------------------------------------------------------
SW_VARS = ("trust", "evidence", "heat", "keys")
SW_ACTIONS = {
    "befriend": {"trust": 1, "heat": -1},
    "investigate": {"evidence": 1, "heat": 1},
    "defuse": {"heat": -2},
    "forge": {"keys": 1, "trust": -1},       # irreversible: keys never decrease
    "burn": {"evidence": -2, "heat": -1},    # destroys evidence
    "wait": {},
}
SW_BOUNDS = {"trust": (-2, 3), "evidence": (0, 4), "heat": (0, 4), "keys": (0, 2)}


def sw_step(state: dict, action: str, banned: frozenset) -> dict | None:
    if action in banned:
        return None
    s = dict(state)
    for k, d in SW_ACTIONS[action].items():
        lo, hi = SW_BOUNDS[k]
        s[k] = max(lo, min(hi, s[k] + d))
    if s["heat"] >= 4:  # alarm: irreversible failure state
        s["alarm"] = 1
    s["t"] += 1
    return s


def sw_goal_ok(s: dict, goal: dict) -> bool:
    return not s.get("alarm") and all(s[k] >= v for k, v in goal.items())


def sw_search(start: dict, goal: dict, horizon: int, banned: frozenset):
    """BFS over (state) up to horizon; returns (plan or None, max reachable value per goal var)."""
    key = lambda s: tuple(s[k] for k in SW_VARS) + (s.get("alarm", 0),)
    frontier = {key(start): (start, [])}
    seen = set(frontier)
    best = {k: start[k] for k in goal}
    if sw_goal_ok(start, goal):
        return [], best
    for _ in range(horizon):
        nxt = {}
        for s, plan in frontier.values():
            for a in SW_ACTIONS:
                n = sw_step(s, a, banned)
                if n is None or n.get("alarm"):
                    continue
                for k in goal:
                    best[k] = max(best[k], n[k])
                if sw_goal_ok(n, goal):
                    return plan + [a], best
                kk = key(n)
                if kk not in seen:
                    seen.add(kk)
                    nxt[kk] = (n, plan + [a])
        frontier = nxt
    return None, best


def storyworld_v2_check(task, answer: str) -> bool:
    c = task.certificate
    a = answer.strip()
    start, goal, horizon, banned = c["start"], c["goal"], c["horizon"], frozenset(c["banned"])
    if not c["reachable"]:
        # Certificate: "NONE <var>=<max reachable value>"; must name a goal var whose max < target.
        parts = a.replace(":", "=").split()
        if not parts or parts[0].upper() != "NONE" or len(parts) != 2 or "=" not in parts[1]:
            return False
        var, val = parts[1].split("=", 1)
        var = var.strip().lower()
        return (var in c["max_reachable"] and val.strip().lstrip("-").isdigit()
                and int(val) == c["max_reachable"][var] and c["max_reachable"][var] < goal[var])
    plan = [x.strip().lower() for x in a.split(",") if x.strip()]
    if not plan or len(plan) > horizon:
        return False
    s = dict(start)
    for act in plan:
        if act not in SW_ACTIONS:
            return False
        s = sw_step(s, act, banned)
        if s is None or s.get("alarm"):
            return False
    return sw_goal_ok(s, goal)


VERIFIERS["storyworld_v2.v1"] = storyworld_v2_check


@register("storyworld_v2")
def storyworld_v2(root, partition):
    r = random.Random(root)
    want_reach = r.random() < 0.5
    for _ in range(200):
        start = {"trust": r.randint(-2, 1), "evidence": r.randint(0, 2), "heat": r.randint(0, 2),
                 "keys": 0, "t": 0}
        goal_vars = r.sample(["trust", "evidence", "keys"], r.randint(2, 3))
        goal = {k: r.randint(1, SW_BOUNDS[k][1]) for k in goal_vars}
        banned = frozenset(r.sample(["befriend", "investigate", "defuse", "forge", "burn"], r.randint(0, 2)))
        horizon = r.randint(5, 9)
        plan, best = sw_search(start, goal, horizon, banned)
        # Unreachable instances must have a single-variable certificate (some goal var's max < target).
        if (plan is not None) == want_reach and (plan is not None or any(best[k] < goal[k] for k in goal)):
            break
    reachable = plan is not None
    if reachable:
        answer = ",".join(plan)
    else:
        blocked = sorted(k for k in goal if best[k] < goal[k])
        answer = f"NONE {blocked[0]}={best[blocked[0]]}" if blocked else "NONE"
    rules = "; ".join(f"{a}: " + (", ".join(f"{k}{d:+d}" for k, d in v.items()) or "no change")
                      for a, v in SW_ACTIONS.items())
    bounds = ", ".join(f"{k} in [{lo},{hi}]" for k, (lo, hi) in SW_BOUNDS.items())
    prompt = (f"A covert operation. State variables ({bounds}) clamp to their bounds after each action. "
              f"Actions: {rules}. If heat reaches 4 an alarm fires and the operation fails permanently. "
              f"Unavailable actions this time: {', '.join(sorted(banned)) or 'none'}.\n"
              f"Start: {{trust: {start['trust']}, evidence: {start['evidence']}, heat: {start['heat']}, keys: 0}}. "
              f"Goal (all at least): {json.dumps(goal)}. At most {horizon} actions.\n"
              "If the goal is reachable, answer a comma-separated action plan. If it is not, answer NONE "
              "followed by one goal variable that cannot reach its target and the maximum value it can "
              "reach within the horizon, e.g. NONE keys=1." + SUFFIX.format(fmt="<plan> | NONE <var>=<max>"))
    return _t("storyworld_v2", root, partition, prompt, answer, [start, goal, horizon, sorted(banned)],
              "storyworld_v2.v1", 0.8, certificate={"reachable": reachable, "start": start, "goal": goal,
              "horizon": horizon, "banned": sorted(banned), "max_reachable": best, "witness": answer}, cost_ms=5.0)


# ---- B. research experiment choice ----------------------------------------------------------------
def entropy(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def eig(prior, lik):
    """Exact expected information gain of a binary experiment; lik[h] = P(positive | h)."""
    pp = sum(prior[h] * lik[h] for h in prior)
    post_pos = [prior[h] * lik[h] / pp for h in prior] if pp > 0 else [0] * len(prior)
    post_neg = [prior[h] * (1 - lik[h]) / (1 - pp) for h in prior] if pp < 1 else [0] * len(prior)
    return entropy(list(prior.values())) - pp * entropy(post_pos) - (1 - pp) * entropy(post_neg)


@register("experiment_choice")
def experiment_choice(root, partition):
    r = random.Random(root)
    for _ in range(200):  # redraw until the best option wins by a clear margin
        out = _experiment_choice_draw(r)
        if out[2] >= 0.02:
            break
    prior, opts, margin, scores, hyps = out
    best = max(scores, key=scores.get)
    lines = [f"{o['id']}: {o['kind']}, cost {o['cost']}, P(positive result | hypothesis) = "
             + ", ".join(f"{h}: {o['lik'][h]}" for h in hyps) for o in opts]
    prompt = ("A research campaign has prior beliefs " + ", ".join(f"{h} {p}" for h, p in prior.items())
              + ". Each candidate experiment gives a positive or negative result with the stated probabilities.\n"
              + "\n".join(lines) + "\nChoose the experiment with the highest expected information gain about the "
              "hypotheses (in bits) per unit cost." + SUFFIX.format(fmt="<letter>"))
    return _t("experiment_choice", root, partition, prompt, best, [prior, opts], "exact_token.v1",
              0.7, certificate={"eig_per_cost": scores, "margin": margin})


def _experiment_choice_draw(r):
    hyps = ["H1 contraction", "H2 routing", "H3 artifact"][: r.randint(2, 3)]
    w = [r.uniform(0.2, 1.0) for _ in hyps]
    prior = {h: round(x / sum(w), 3) for h, x in zip(hyps, w)}
    s = sum(prior.values())
    prior[hyps[-1]] = round(prior[hyps[-1]] + 1 - s, 3)
    kinds = [("matched-sham intervention", 1.0), ("unmatched intervention (confounded)", 1.0),
             ("replication on fresh roots", 1.5), ("adaptive re-test on reused validation roots", 0.5),
             ("underpowered pilot (4 roots)", 0.3), ("mechanistic probe with causal control", 2.0)]
    opts = []
    for i, (kind, cost_scale) in enumerate(r.sample(kinds, 4)):
        if "confounded" in kind or "reused" in kind:
            lik = {h: round(r.uniform(0.55, 0.65), 2) for h in hyps}          # barely separates
        elif "underpowered" in kind:
            base = r.uniform(0.3, 0.7)
            lik = {h: round(min(0.95, max(0.05, base + r.uniform(-0.1, 0.1))), 2) for h in hyps}
        else:
            lik = {h: round(r.uniform(0.05, 0.95), 2) for h in hyps}
        cost = round(cost_scale * r.uniform(0.8, 1.6), 2)
        opts.append({"id": "ABCD"[i], "kind": kind, "lik": lik, "cost": cost})
    scores = {o["id"]: eig(prior, o["lik"]) / o["cost"] for o in opts}
    margin = sorted(scores.values())[-1] - sorted(scores.values())[-2]
    return prior, opts, margin, scores, hyps


def _log_pmf(n, k, p):
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _critical_counts(n_max, alpha):
    """k_d = smallest k with P(Bin(d, 1/2) >= k) <= alpha, for d = 0..n_max (one-sided exact test)."""
    ks = []
    for d in range(n_max + 1):
        tail, k = 0.0, d + 1
        for j in range(d, -1, -1):
            t = tail + math.exp(_log_pmf(d, j, 0.5))
            if t > alpha:
                break
            tail, k = t, j
        ks.append(k)
    return ks


def _upper_tail(d, k, q):
    return sum(math.exp(_log_pmf(d, i, q)) for i in range(k, d + 1)) if k <= d else 0.0


def mcnemar_power(n, p_gain, p_loss, alpha, _cache={}):
    """Exact power of the one-sided McNemar test: sum over the discordant count D ~ Bin(n, pd)."""
    pd = p_gain + p_loss
    q = p_gain / pd
    key = (alpha, q)
    ks, tails = _cache.get(key, (None, None))
    if ks is None or len(ks) <= n:
        ks = _critical_counts(max(n, 512), alpha)
        tails = [_upper_tail(d, ks[d], q) for d in range(len(ks))]
        _cache[key] = (ks, tails)
    return sum(math.exp(_log_pmf(n, d, pd)) * tails[d] for d in range(1, n + 1))


@register("power_design")
def power_design(root, partition):
    r = random.Random(root)
    effect = r.choice([0.15, 0.2, 0.25, 0.3])
    disc = r.choice([0.3, 0.35, 0.4])
    alpha = r.choice([0.05, 0.1])
    target = r.choice([0.7, 0.8])
    pg, pl = (disc + effect) / 2, (disc - effect) / 2
    n = next(n for n in range(5, 400) if mcnemar_power(n, pg, pl, alpha) >= target)
    prompt = (f"A paired study compares a candidate against a control on independent roots. Each root is "
              f"discordant with probability {disc}; among discordant roots the candidate wins with probability "
              f"{pg / disc:.4f} (net effect {effect}). The analysis is a one-sided exact McNemar (binomial sign) "
              f"test at alpha {alpha}. What is the smallest number of roots n giving power at least {target}? "
              "Power is exact: sum over the discordant count." + SUFFIX.format(fmt="<integer>"))
    return _t("power_design", root, partition, prompt, n, [effect, disc, alpha, target], "exact_int.v1", 0.8,
              certificate={"power_at_n": mcnemar_power(n, pg, pl, alpha), "power_at_n_minus_1": mcnemar_power(n - 1, pg, pl, alpha)})


# ---- D. mathematical certificates -------------------------------------------------------------
def graph_check(task, answer: str) -> bool:
    c = task.certificate
    edges = [tuple(e) for e in c["edges"]]
    n = c["n"]
    a = answer.strip().upper()
    if a.startswith("BIPARTITE"):
        body = a[len("BIPARTITE"):].strip()
        cols = [x for x in body.replace(",", " ").split() if x]
        if len(cols) != n or any(x not in ("0", "1") for x in cols):
            return False
        return c["bipartite"] and all(cols[u] != cols[v] for u, v in edges)
    if a.startswith("ODD"):
        body = a[len("ODD"):].strip()
        cyc = [int(x) for x in body.replace(",", " ").split() if x.isdigit()]
        es = {frozenset(e) for e in edges}
        return (not c["bipartite"] and len(cyc) % 2 == 1 and len(cyc) >= 3 and len(set(cyc)) == len(cyc)
                and all(frozenset((cyc[i], cyc[(i + 1) % len(cyc)])) in es for i in range(len(cyc))))
    return False


VERIFIERS["graph_certificate.v1"] = graph_check


@register("graph_certificate")
def graph_certificate(root, partition):
    r = random.Random(root)
    n = r.randint(7, 11)
    want_bip = r.random() < 0.5
    side = [i % 2 for i in range(n)]  # both sides non-empty, then shuffled
    r.shuffle(side)
    cross = [(u, v) for u in range(n) for v in range(u + 1, n) if side[u] != side[v]]
    edges = set(r.sample(cross, min(len(cross), r.randint(n, 2 * n))))
    if not want_bip:
        same = [(u, v) for u in range(n) for v in range(u + 1, n) if side[u] == side[v]]
        edges.add(r.choice(same))
    edges = sorted(edges)
    # exact bipartiteness by BFS
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    col, ok, odd = {}, True, None
    parent, depth = {}, {}
    for s in range(n):
        if s in col:
            continue
        col[s], parent[s], depth[s] = 0, None, 0
        q = [s]
        while q:
            x = q.pop(0)  # BFS: tree paths are shortest, so the lca construction below is valid
            for y in adj[x]:
                if y not in col:
                    col[y], parent[y], depth[y] = 1 - col[x], x, depth[x] + 1
                    q.append(y)
                elif col[y] == col[x] and odd is None:
                    ok = False
                    a, b, pa, pb = x, y, [x], [y]
                    while depth[a] > depth[b]:
                        a = parent[a]; pa.append(a)
                    while depth[b] > depth[a]:
                        b = parent[b]; pb.append(b)
                    while a != b:
                        a, b = parent[a], parent[b]
                        pa.append(a); pb.append(b)
                    odd = pa + pb[:-1][::-1]  # x .. lca .. y, closed by edge (y, x)
    answer = ("BIPARTITE " + ",".join(str(col[i]) for i in range(n)) if ok
              else "ODD " + ",".join(str(v) for v in odd))
    prompt = (f"Undirected graph on vertices 0..{n - 1} with edges {edges}. If it is bipartite, answer BIPARTITE "
              "followed by a 0/1 colour for each vertex in order (adjacent vertices must differ). Otherwise answer "
              "ODD followed by the vertices of an odd cycle in cycle order." + SUFFIX.format(fmt="BIPARTITE c0,c1,... | ODD v1,v2,..."))
    return _t("graph_certificate", root, partition, prompt, answer, [n, edges], "graph_certificate.v1", 0.6,
              certificate={"n": n, "edges": edges, "bipartite": ok})


def spectral_radius_lt1(A) -> bool:
    import numpy as np
    return bool(max(abs(np.linalg.eigvals(np.array(A, dtype=float)))) < 1 - 1e-12)


@register("stability_certificate")
def stability_certificate(root, partition):
    r = random.Random(root)
    n = r.choice([2, 3])
    den = r.choice([2, 4, 5, 10])
    for _ in range(100):
        A = [[Fraction(r.randint(-den, den), den) for _ in range(n)] for _ in range(n)]
        import numpy as np
        rho = max(abs(np.linalg.eigvals(np.array(A, dtype=float))))
        if abs(rho - 1) > 0.05:
            break
    stable = rho < 1
    mat = "[" + "; ".join(" ".join(str(x) for x in row) for row in A) + "]"
    prompt = (f"Linear recurrence x_(t+1) = A x_t with A = {mat} (rows separated by ';'). Is the zero state "
              "asymptotically stable for every initial x_0, i.e. is the spectral radius of A below 1? Do not "
              "infer stability from row norms or cosine similarity; compute the eigenvalues."
              + SUFFIX.format(fmt="STABLE | UNSTABLE"))
    return _t("stability_certificate", root, partition, prompt, "STABLE" if stable else "UNSTABLE",
              [[str(x) for x in row] for row in A], "exact_token.v1", 0.6, certificate={"spectral_radius": float(rho)})


# ---- E. algorithm discovery: sorting networks -----------------------------------------------------
OPT_SIZE = {4: 5, 5: 9, 6: 12, 7: 16}
# Known optimal-size networks (Knuth TAOCP vol. 3 §5.3.4); each is verified exhaustively in tests.
REFERENCE_NETWORK = {
    4: "0-1,2-3,0-2,1-3,1-2",
    5: "0-1,3-4,2-4,2-3,0-3,0-2,1-4,1-3,1-2",
    6: "1-2,4-5,0-2,3-5,0-1,3-4,2-5,0-3,1-4,2-4,1-3,2-3",
}


def network_check(task, answer: str) -> bool:
    n, bound = task.certificate["n"], task.certificate["max_comparators"]
    try:
        pairs = [tuple(int(x) for x in p.strip(" ()").split("-")) for p in answer.split(",") if p.strip()]
    except ValueError:
        return False
    if not pairs or len(pairs) > bound or any(len(p) != 2 or not (0 <= p[0] < n and 0 <= p[1] < n) or p[0] == p[1] for p in pairs):
        return False
    prefix = [tuple(p) for p in task.certificate.get("prefix", [])]
    if [tuple(sorted(p)) for p in pairs[:len(prefix)]] != [tuple(sorted(p)) for p in prefix]:
        return False
    for bits in itertools.product((0, 1), repeat=n):
        v = list(bits)
        for i, j in pairs:
            i, j = min(i, j), max(i, j)
            if v[i] > v[j]:
                v[i], v[j] = v[j], v[i]
        if any(v[k] > v[k + 1] for k in range(n - 1)):
            return False
    return True


VERIFIERS["sorting_network.v1"] = network_check


@register("sorting_network")
def sorting_network(root, partition):
    r = random.Random(root)
    n = r.choice([4, 5, 6])
    slack = r.choice([0, 1, 2])
    prefix = [tuple(sorted(r.sample(range(n), 2))) for _ in range(r.randint(1, 3))]
    bound = OPT_SIZE[n] + len(prefix) + slack
    pre = ",".join(f"{i}-{j}" for i, j in prefix)
    prompt = (f"Design a comparator network that sorts any {n} inputs (wires 0..{n - 1}) into ascending order. "
              f"A comparator i-j swaps wires i<j when wire i holds the larger value. The network must begin with "
              f"these comparators, in this order: {pre}. Use at most {bound} comparators in total, including them. "
              "Answer the whole network in application order." + SUFFIX.format(fmt="i-j,i-j,..."))
    # Reference: the fixed prefix followed by a known optimal network (sorting is unaffected by the prefix).
    return _t("sorting_network", root, partition, prompt, pre + "," + REFERENCE_NETWORK[n], [n, bound, prefix],
              "sorting_network.v1", 0.5 + 0.2 * (2 - slack),
              certificate={"n": n, "max_comparators": bound, "optimal_size": OPT_SIZE[n], "prefix": prefix})


# ---- C. hill-climb diagnosis ---------------------------------------------------------------------------
@register("hillclimb_diagnosis")
def hillclimb_diagnosis(root, partition):
    r = random.Random(root)
    fields = {"lr": [0.04, 0.08, 0.12, 0.2], "steps": [12, 16, 24, 32], "stop_policy": ["action_class", "train_calibrated"]}
    cur = {"lr": 0.2, "steps": 16, "stop_policy": "action_class"}
    kind = r.choice(["repeat", "local_opt"])
    hist = []
    singles = [(f, v) for f, vals in fields.items() for v in vals if cur[f] != v]
    if kind == "repeat":
        dup = r.choice(singles)
        seq = r.sample(singles, min(len(singles), r.randint(4, 6)))
        if dup not in seq:
            seq[0] = dup
        seq.insert(r.randint(seq.index(dup) + 1, len(seq)), dup)
        for f, v in seq:
            hist.append((f, v, "reject_no_gain" if (f, v) != dup else "reject_safety_regression"))
        answer = f"{dup[0]}={dup[1]}"
    else:
        exhaust = r.random() < 0.5
        seq = list(singles) if exhaust else r.sample(singles, len(singles) - r.randint(1, 3))
        r.shuffle(seq)
        hist = [(f, v, "reject_no_gain") for f, v in seq]
        answer = "LOCAL_OPTIMUM" if exhaust else "UNTRIED " + ",".join(sorted(f"{f}={v}" for f, v in singles if (f, v) not in seq))
    lines = [f"turn {i}: set {f} to {v} -> {d}" for i, (f, v, d) in enumerate(hist)]
    if kind == "repeat":
        q = "Which edit was proposed more than once? Answer field=value."
        fmt = "field=value"
    else:
        q = ("Every listed edit changes one field of the incumbent and was rejected; the incumbent never changed. "
             "Is the incumbent a local optimum under single-field edits of this grammar? If every single-field edit "
             "has been tried, answer LOCAL_OPTIMUM; otherwise answer UNTRIED followed by every untried edit, "
             "comma-separated, sorted alphabetically.")
        fmt = "LOCAL_OPTIMUM | UNTRIED field=value,..."
    prompt = (f"Hill-climb grammar: {json.dumps(fields)}. Incumbent config: {json.dumps(cur)}.\nHistory:\n"
              + "\n".join(lines) + "\n" + q + SUFFIX.format(fmt=fmt))
    return _t("hillclimb_diagnosis", root, partition, prompt, answer, [kind, hist], "hillclimb_diag.v1",
              0.5, certificate={"kind": kind})


def _diag_norm(s: str):
    s = s.strip()
    head, _, rest = s.partition(" ")
    if head.upper() == "UNTRIED":
        return ("UNTRIED", frozenset(x.strip().replace(" ", "").lower() for x in rest.split(",") if x.strip()))
    return (s.replace(" ", "").lower(), frozenset())


VERIFIERS["hillclimb_diag.v1"] = lambda task, answer: _diag_norm(answer) == _diag_norm(task.answer)


# ---- F. research memory --------------------------------------------------------------------------------
@register("research_memory")
def research_memory(root, partition):
    r = random.Random(root)
    qs = ["Q1 does the stop head cause premature stops", "Q2 does bundle size move solved cases",
          "Q3 does the null-space update leak", "Q4 does high effort rescue failures"]
    exps = {"E1": ("Q1", "oracle-stop rollout"), "E2": ("Q1", "stop-head ablation"), "E3": ("Q2", "24-root bundle"),
            "E4": ("Q2", "wide grammar"), "E5": ("Q3", "float32 residual"), "E6": ("Q3", "literal pinv"),
            "E7": ("Q4", "paired effort grid"), "E8": ("Q4", "8192-token frontier")}
    for _ in range(200):  # redraw until some untested experiment addresses an open question
        done = r.sample(sorted(exps), r.randint(4, 6))
        resolved = {exps[e][0] for e in done if r.random() < 0.6}
        open_q = [q.split()[0] for q in qs if q.split()[0] not in resolved]
        cand = [e for e in sorted(exps) if e not in done and exps[e][0] in open_q]
        if cand:
            break
    target = sorted(cand)[0]
    render = r.choice(["full_history", "structured_summary"])
    if render == "full_history":
        log = []
        for e in done:
            q, name = exps[e]
            log.append(f"ran {e} ({name}) for {q}; result " + ("settled the question" if q in resolved else "inconclusive"))
            if r.random() < 0.4:
                log.append(f"note: reconsidered {e}, no new information")
        r.shuffle(log)
        memory = "Lab notebook (chronological, unedited):\n" + "\n".join(log)
    else:
        memory = ("Structured memory: tested=" + ",".join(sorted(done)) + "; resolved=" + ",".join(sorted(resolved))
                  + "; open=" + ",".join(open_q))
    menu = "\n".join(f"{e}: {name} (addresses {q})" for e, (q, name) in sorted(exps.items()))
    prompt = (f"Research questions: {'; '.join(qs)}.\n{memory}\nAvailable experiments:\n{menu}\n"
              "Choose the lowest-numbered experiment that has not been run yet and addresses a question that is "
              "still open." + SUFFIX.format(fmt="<experiment id>"))
    return _t("research_memory", root, partition, prompt, target, [sorted(done), sorted(resolved), render],
              "exact_token.v1", 0.5, certificate={"render": render, "tested": sorted(done), "open": open_q})


FAMILIES = ("storyworld_v2", "experiment_choice", "power_design", "graph_certificate", "stability_certificate",
            "sorting_network", "hillclimb_diagnosis", "research_memory")

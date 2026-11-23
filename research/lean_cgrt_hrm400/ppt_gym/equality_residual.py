"""LSA-02M: native-certified off-path residual ordering, development only."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import heapq
from itertools import combinations, product
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import digest, canonical, sha256_file
from .equality_suite import ALPHABET, BANDS, RULE_COUNT, EqCase, choose_shell, neighbors, system_lineage
from .equality_search import BUDGET, CAPS, EXACT_CAP, run_live, replay, path_moves
from .equality_preflight import compile_scripts, _native_backend, _rows, _write
from .equality_ordering import (
    SUITE_SHA, SEEDS, EPOCHS, EDGE_CAPS, DEV_N, _suite, _jsonl,
    training_rows, hamming, best_first, _public_plan, freeze_dev,
)

DEV_SEED = 2026112403
OLD_DEV_SHA = "afe98f4f6a8af265379d2a2f2c82477497727a07512a5aecf16e75e290da6444"
DEPTH, TEACH_CAP = 12, 4096
PAIRS = tuple("".join(x) for x in product(ALPHABET, repeat=2))
WIDTH, FEATURES, WINDOWS = 24, 11, 9
PUBLIC = ("hamming_greedy", "bidirectional", "pair_astar", "pair_greedy", "local_greedy")
LEARNED = ("residual", "sham", "one_step", "reference_only")


def excluded(suite, old):
    cases, meta = _suite(suite)
    if sha256_file(old / "manifest.json") != OLD_DEV_SHA:
        raise ValueError("previous development roster manifest mismatch")
    m = json.loads((old / "manifest.json").read_text())
    if len(m["lineages"]) != DEV_N or len(set(m["lineages"])) != DEV_N:
        raise ValueError("previous development lineage census mismatch")
    seen = {lin for shard in meta["shards"].values() for lin in shard["lineages"]}
    if seen.intersection(m["lineages"]):
        raise ValueError("previous development lineage overlap")
    return cases, seen | set(m["lineages"])


def retarget(case, word, split=None):
    split = split or case.split
    root = digest([case.lineage, split, case.equations, case.source, word])[:24]
    return EqCase(root, case.lineage, split, case.family, case.equations,
                  case.source, word).validate()


def freeze(suite, old, out):
    _, seen = excluded(suite, old)
    rng = random.Random(DEV_SEED)
    available = list(combinations(PAIRS, 2))
    rows = []
    attempts = 0
    while len(rows) < DEV_N:
        attempts += 1
        if attempts > DEV_N * 500:
            raise RuntimeError("development generation failed")
        rules = tuple((b, a) if rng.randrange(2) else (a, b)
                      for a, b in rng.sample(available, RULE_COUNT))
        lin = system_lineage(rules)
        if lin in seen:
            continue
        i = len(rows)
        source = "".join(rng.choice(ALPHABET) for _ in range(7 + (i // 3) % 3))
        target = choose_shell(source, rules, BANDS[i % 3], rng)
        if target is None:
            continue
        root = digest([lin, "calibration", rules, source, target])[:24]
        c = EqCase(root, lin, "calibration", "indexed_unary_equations", rules,
                   source, target).validate()
        seen.add(lin)
        rows.append(dict(vars(c)))
    out.mkdir(parents=True, exist_ok=False)
    _rows(out / "roots.jsonl", rows)
    m = {"schema": "lsa02m.fresh_development.v1", "seed": DEV_SEED, "roots": DEV_N,
         "suite_sha256": SUITE_SHA, "prior_dev_sha256": OLD_DEV_SHA,
         "lineages": [r["lineage"] for r in rows],
         "rows_sha256": sha256_file(out / "roots.jsonl"),
         "generator_sha256": sha256_file(Path(__file__))}
    _write(out / "manifest.json", m)
    return m


def load_dev(suite, old, directory):
    _, seen = excluded(suite, old)
    m = json.loads((directory / "manifest.json").read_text())
    if (m["schema"] != "lsa02m.fresh_development.v1" or m["seed"] != DEV_SEED
        or m["roots"] != DEV_N or m["suite_sha256"] != SUITE_SHA
        or m["prior_dev_sha256"] != OLD_DEV_SHA
        or m["rows_sha256"] != sha256_file(directory / "roots.jsonl")
        or m["generator_sha256"] != sha256_file(Path(__file__))):
        raise ValueError("development manifest mismatch")
    cases = []
    for row in _jsonl(directory / "roots.jsonl"):
        if set(row) != set(EqCase.__dataclass_fields__):
            raise ValueError("private metadata in development row")
        row["equations"] = tuple(tuple(p) for p in row["equations"])
        c = EqCase(**row).validate()
        if c.lineage in seen or c.split != "calibration":
            raise ValueError("development lineage/split violation")
        seen.add(c.lineage)
        cases.append(c)
    if len(cases) != DEV_N or [c.lineage for c in cases] != m["lineages"]:
        raise ValueError("development census mismatch")
    return cases


def reverse_teacher(case, cap=TEACH_CAP, depth=DEPTH):
    """FIFO discovery certifies exact graph distance of every discovered word."""
    d, parent = {case.source: 0}, {case.source: None}
    todo = deque([case.source])
    expansions = edges = 0
    while todo and expansions < cap:
        w = todo.popleft()
        if d[w] >= depth:
            break
        moves = neighbors(w, case.equations)
        expansions += 1
        edges += len(moves)
        for m in moves:
            v = m.next_word
            if v not in d:
                d[v] = d[w] + 1
                parent[v] = w
                todo.append(v)
    return d, parent, expansions, edges


def teacher_path(word, parent):
    path = [word]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path


def escape_depth(case, word, d):
    """Minimum distance-descending steps to strictly smaller Hamming."""
    h = hamming(word, case.source)
    if h == 0:
        return 0
    frontier = {word}
    for k in range(1, d[word]+1):
        frontier = {m.next_word for w in frontier for m in neighbors(w, case.equations)
                    if d.get(m.next_word) == d[w]-1}
        if any(hamming(w, case.source) < h for w in frontier):
            return k
    raise ValueError("distance-descending path did not reach witness")


def groups(case, d, reference):
    rng = random.Random(int(digest([case.root, DEV_SEED, "offpath"])[:16], 16))
    strata = [[], []]
    for w in sorted(d):
        if w in reference or d[w] < 2:
            continue
        ns = sorted({m.next_word for m in neighbors(w, case.equations)})
        good = [v for v in ns if d.get(v) == d[w]-1]
        other = [v for v in ns if v in d and d[v] >= d[w]]
        if not good or not other:
            continue
        h = hamming(w, case.source)
        detour = all(hamming(v, case.source) >= h for v in good)
        local_min = all(hamming(v, case.source) >= h for v in ns)
        strata[int(detour)].append((w, good, other, detour, local_min))
    for s in strata:
        rng.shuffle(s)
    selected = strata[0][:2] + strata[1][:2]
    selected += (strata[0][2:] + strata[1][2:])[:4-len(selected)]
    return [{"word": w, "good": rng.choice(good), "alternate": rng.choice(other),
             "shortest_path_detour": detour, "local_hamming_minimum": local,
             "escape_depth": escape_depth(case, w, d)}
            for w, good, other, detour, local in selected]


class Features:
    """Finite, alpha-invariant pair lookup; no whole-word neighbor lookahead."""
    def __init__(self, case):
        self.case = case
        self.tables = []
        self.setup_states = self.rule_checks = self.abstract_expansions = self.abstract_edges = 0
        n = len(case.source)-1
        oriented = [(a, b) for a, b in case.equations] + [(b, a) for a, b in case.equations]
        for i in range(n):
            target = case.source[i:i+2]
            adj, direct = {}, {}
            for w in PAIRS:
                self.setup_states += 1
                opts, exact = set(), set()
                for a, b in oriented:
                    self.rule_checks += 1
                    if w == a:
                        opts.add(b)
                        exact.add(b)
                    if i > 0 and w[0] == a[1]:
                        opts.add(b[1]+w[1])
                    if i < n-1 and w[1] == a[0]:
                        opts.add(w[0]+b[0])
                adj[w] = opts - {w}
                direct[w] = exact
            d = {target: 0}
            q = deque([target])
            while q:
                w = q.popleft()
                self.abstract_expansions += 1
                self.abstract_edges += len(adj[w])
                for v in sorted(adj[w]):
                    if v not in d:
                        d[v] = d[w]+1
                        q.append(v)
            table = {}
            for w in PAIRS:
                h = hamming(w, target)
                delta = min([0] + [hamming(v, target)-h for v in direct[w]])
                dist = d.get(w, math.inf)
                x = [float(w[0] != target[0]), float(w[1] != target[1]),
                     float(w[0] == w[1]), float(target[0] == target[1]),
                     float(w[0] == target[1]), float(w[1] == target[0]),
                     min(dist, 16)/16, len(direct[w])/12, delta/2,
                     float(i == 0), float(i == n-1)]
                table[w] = (x, dist, delta)
            self.tables.append(table)
        self.mask = [True]*n + [False]*(WINDOWS-n)

    def entries(self, word):
        return [t[word[i:i+2]] for i,t in enumerate(self.tables)]

    def encode(self, word):
        x = [e[0] for e in self.entries(word)]
        return x + [[0.0]*FEATURES for _ in range(WINDOWS-len(x))]

    def public_score(self, word, arm):
        h = hamming(word, self.case.source)
        entries = self.entries(word)
        if arm == "local_greedy":
            return h + min(e[2] for e in entries)
        vals = [e[1] for e in entries]
        bound = max(max(vals), math.ceil(sum(vals)/3)) if all(math.isfinite(v) for v in vals) else math.inf
        return h + 0.25*bound


def build_pairs(suite, teacher, out):
    # Inherited checker authenticates the original native certificates and all frames.
    training_rows(suite, teacher)
    cases, _ = _suite(suite)
    eps = _jsonl(teacher / "reference_episodes.jsonl")
    out.mkdir(parents=True, exist_ok=False)
    cert_dir = out / "certificates"
    cert_dir.mkdir()
    samples, group_rows, cert_rows, items = [], [], [], []
    reference_count = 0
    with LeanRepl(imports=("Lean",), timeout=60, exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR, env=config.lean_env()) as repl:
        backend = _native_backend(repl)
        for case, ep in zip(cases, eps):
            words = [e["word_before"] for e in ep["events"]]
            ref = {w: len(words)-i-1 for i,w in enumerate(words)}
            if words[-1] != case.source:
                raise ValueError("original reference terminal mismatch")
            d, parents, expands, edges = reverse_teacher(case)
            for w, distance in ref.items():
                if w in d and d[w] != distance:
                    raise ValueError("reverse teacher disagrees with certified reference distance")
            selected = groups(case, d, ref)
            fe = Features(case)
            certified = set(ref)

            def add_pair(hard, easy, gap, kind):
                nonlocal reference_count
                if hard not in certified or easy not in certified or gap not in (1, 2):
                    raise ValueError("uncertified or invalid pair")
                samples.append({"root": case.root, "lineage": case.lineage,
                                "hard": hard, "easy": easy, "gap": gap, "kind": kind,
                                "hard_features": fe.encode(hard), "easy_features": fe.encode(easy),
                                "mask": fe.mask, "hard_hamming": hamming(hard, case.source),
                                "easy_hamming": hamming(easy, case.source)})
                reference_count += kind == "reference"

            for hard, easy in zip(words, words[1:]):
                add_pair(hard, easy, 1, "reference")
            for g in selected:
                for w in (g["word"], g["good"], g["alternate"]):
                    if w in certified:
                        continue
                    c = retarget(case, w)
                    path = teacher_path(w, parents)
                    if len(path)-1 != d[w]:
                        raise ValueError("teacher distance/path disagreement")
                    t = time.perf_counter()
                    native = run_live(repl, c, "bidirectional_bfs", words=path)
                    if not native["live_kernel"] or not replay(repl, c, native):
                        raise ValueError("off-path native certificate/replay failed")
                    items.append((c, "offpath", native["script"]))
                    cert_rows.append({"training_root": case.root, "word": w, "distance": d[w],
                                      "episode": native, "seconds": time.perf_counter()-t})
                    certified.add(w)
                w, good, alt = g["word"], g["good"], g["alternate"]
                add_pair(w, good, d[w]-d[good], "offpath_parent")
                add_pair(alt, good, d[alt]-d[good], "offpath_sibling")
                group_rows.append(dict(g, root=case.root, lineage=case.lineage,
                                       parent_distance=d[w], good_distance=d[good],
                                       alternate_distance=d[alt], teacher_expansions=expands,
                                       teacher_edges=edges))
    if not items or not group_rows:
        raise ValueError("no additional certified supervision")
    # Independent compilation in deterministic chunks; every proof is checked.
    compiled = 0
    for i in range(0, len(items), 128):
        chunk = cert_dir / str(i//128)
        chunk.mkdir()
        receipt = compile_scripts(items[i:i+128], chunk, timeout=600)
        compiled += receipt["passed"]
    if compiled != len(items):
        raise ValueError("off-path compile census mismatch")
    for row in cert_rows:
        row["episode"]["standalone_verified"] = True
    _rows(out / "pairs.jsonl", samples)
    _rows(out / "groups.jsonl", group_rows)
    _rows(out / "native_episodes.jsonl", cert_rows)
    m = {"schema": "lsa02m.certified_pairs.v1", "backend": backend,
         "train_roots": len(cases), "original_frames": 546,
         "reference_pairs": reference_count, "offpath_pairs": len(samples)-reference_count,
         "offpath_groups": len(group_rows), "offpath_native_compiled": compiled,
         "shortest_path_detours": sum(g["shortest_path_detour"] for g in group_rows),
         "local_hamming_minima": sum(g["local_hamming_minimum"] for g in group_rows),
         "max_escape_depth": max(g["escape_depth"] for g in group_rows),
         "teacher_sha256": sha256_file(teacher / "manifest.json"),
         "artifacts": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}}
    _write(out / "manifest.json", m)
    return samples, m


class Residual(nn.Module):
    def __init__(self, steps=4):
        super().__init__()
        self.steps = steps
        self.project = nn.Linear(FEATURES, WIDTH)
        self.loop = nn.Linear(WIDTH, WIDTH)
        self.readout = nn.Linear(2*WIDTH, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x, mask):
        valid = mask.unsqueeze(-1)
        z = self.project(x)*valid
        h = torch.tanh(z)*valid
        for _ in range(self.steps):
            left = F.pad(h[:, :-1], (0, 0, 1, 0))
            right = F.pad(h[:, 1:], (0, 0, 0, 1))
            h = torch.tanh(self.loop(h) + z + (left+right)/2)*valid
        mean = h.sum(1)/valid.sum(1)
        mx = h.masked_fill(~valid, -torch.inf).max(1).values
        return 2*torch.tanh(self.readout(torch.cat((mean, mx), dim=-1)).squeeze(-1))


def fit(samples, arm, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed+101)
    net = Residual(1 if arm == "one_step" else 4)
    opt = torch.optim.AdamW(net.parameters(), lr=0.003)
    hard = torch.tensor([r["hard_features"] for r in samples])
    easy = torch.tensor([r["easy_features"] for r in samples])
    mask = torch.tensor([r["mask"] for r in samples], dtype=torch.bool)
    hh = torch.tensor([r["hard_hamming"] for r in samples], dtype=torch.float32)
    eh = torch.tensor([r["easy_hamming"] for r in samples], dtype=torch.float32)
    gap = torch.tensor([r["gap"] for r in samples], dtype=torch.float32)
    if arm == "sham":
        rng = random.Random(seed+1601)
        gap *= torch.tensor([rng.choice((-1, 1)) for _ in samples], dtype=torch.float32)
    refs = torch.tensor([i for i,r in enumerate(samples) if r["kind"] == "reference"])
    updates = 0
    for _ in range(EPOCHS):
        order = (refs[torch.randint(len(refs), (len(samples),), generator=gen)]
                 if arm == "reference_only" else torch.randperm(len(samples), generator=gen))
        for ix in order.split(64):
            a = net(hard[ix], mask[ix])*(hh[ix] > 0)
            b = net(easy[ix], mask[ix])*(eh[ix] > 0)
            score_gap = hh[ix]+a-eh[ix]-b
            loss = F.smooth_l1_loss(score_gap, gap[ix]) + 0.02*(a.square()+b.square()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
    sha = digest({k: v.detach().tolist() for k,v in net.state_dict().items()})
    return net.eval(), {"state_digest": sha, "updates": updates,
                        "params": sum(p.numel() for p in net.parameters()),
                        "steps": net.steps, "last_loss": float(loss.detach())}


def residual_search(case, cap, arm, net=None):
    if cap < 1 or arm not in ("hamming_greedy", "pair_greedy", "local_greedy", *LEARNED):
        raise ValueError("unsupported residual ordering")
    if (net is not None) != (arm in LEARNED):
        raise ValueError("model/arm mismatch")
    t0 = time.perf_counter()
    fe = Features(case) if arm != "hamming_greedy" else None
    cache = {case.source: 0.0}
    batches = scored = 0
    infer = 0.0

    def score(words):
        nonlocal batches, scored, infer
        missing = list(dict.fromkeys(w for w in words if w not in cache))
        if not missing:
            return
        if net is not None:
            t = time.perf_counter()
            with torch.inference_mode():
                x = torch.tensor([fe.encode(w) for w in missing])
                mask = torch.tensor([fe.mask]*len(missing), dtype=torch.bool)
                residuals = net(x, mask).tolist()
            infer += time.perf_counter()-t
            batches += 1
            scored += len(missing)
            vals = [hamming(w, case.source)+v for w,v in zip(missing, residuals)]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("nonfinite residual priority")
        elif fe is not None:
            vals = [fe.public_score(w, arm) for w in missing]
        else:
            vals = [hamming(w, case.source) for w in missing]
        cache.update(zip(missing, vals))

    start = case.target
    score([start])
    heap = [(cache[start], hamming(start, case.source), 0, start, (start,))]
    best = {start: 0}
    expansions = edges = 0
    path = ()
    while heap:
        _, _, g, w, trace = heapq.heappop(heap)
        if best[w] != g:
            continue
        if w == case.source:
            path = trace
            break
        if g >= BUDGET-1:
            continue
        if expansions >= cap:
            break
        moves = neighbors(w, case.equations)
        expansions += 1
        edges += len(moves)
        children = []
        for child in sorted({m.next_word for m in moves}):
            if g+1 < best.get(child, math.inf):
                best[child] = g+1
                children.append(child)
        score(children)
        for child in children:
            heapq.heappush(heap, (cache[child], hamming(child, case.source), g+1,
                                  child, (*trace, child)))
    return {"found": bool(path), "words": list(path), "expansion_cap": cap,
            "expansions": expansions, "candidate_edges_enumerated": edges,
            "planning_seconds": time.perf_counter()-t0,
            "setup_states": fe.setup_states if fe else 0,
            "setup_rule_checks": fe.rule_checks if fe else 0,
            "setup_abstract_expansions": fe.abstract_expansions if fe else 0,
            "setup_abstract_edges": fe.abstract_edges if fe else 0,
            "model_batches": batches, "model_states_scored": scored,
            "inference_seconds": infer}


def evaluate(cases, models, out):
    records = []
    out.mkdir(parents=True, exist_ok=False)
    case_map = {c.root: c for c in cases}
    for i, case in enumerate(cases):
        arms = [(a, None) for a in PUBLIC] + [(a,s) for a in LEARNED for s in SEEDS]
        arms = arms[i % len(arms):] + arms[:i % len(arms)]
        for arm, seed in arms:
            for cap in CAPS:
                if arm == "bidirectional":
                    result = _public_plan(case, cap)
                elif arm == "pair_astar":
                    result = best_first(case, cap, arm)
                else:
                    result = residual_search(case, cap, arm, models.get((arm,seed)))
                records.append(dict(result, arm=arm, seed=seed, root=case.root,
                                    lineage=case.lineage))
        records.append(dict(_public_plan(case, EXACT_CAP), arm="exact_ceiling",
                            seed=None, root=case.root, lineage=case.lineage))
    _rows(out / "proposed_paths.jsonl", records)
    certs, episodes, items = {}, [], []
    certdir = out / "certificates"
    certdir.mkdir()
    with LeanRepl(imports=("Lean",), timeout=60, exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR, env=config.lean_env()) as repl:
        backend = _native_backend(repl)
        for row in records:
            row.update(certified=False, native_tactics=0, certificate_id=None)
            if not row["found"]:
                continue
            case = case_map[row["root"]]
            key = (case.root, tuple(row["words"]))
            if key not in certs:
                path_moves(case, row["words"])
                label = f"p{len(certs)}"
                t = time.perf_counter()
                ep = run_live(repl, case, "bidirectional_bfs", words=row["words"])
                if not ep["live_kernel"] or not replay(repl, case, ep):
                    raise ValueError("evaluation proof/replay failed")
                certs[key] = label, ep["effort"]
                items.append((case, label, ep["script"]))
                episodes.append({"certificate_id": label, "episode": ep,
                                 "native_and_replay_seconds": time.perf_counter()-t})
            row["certificate_id"], row["native_tactics"] = certs[key]
    compiled = 0
    for i in range(0, len(items), 128):
        chunk = certdir / str(i//128)
        chunk.mkdir()
        compiled += compile_scripts(items[i:i+128], chunk, timeout=600)["passed"]
    if compiled != len(certs):
        raise ValueError("evaluation certificate census mismatch")
    for e in episodes:
        e["episode"]["standalone_verified"] = True
    for row in records:
        row["certified"] = row["certificate_id"] is not None
    depths = {r["root"]: len(r["words"])-1 for r in records
              if r["arm"] == "exact_ceiling" and r["certified"]}
    if len(depths) != DEV_N:
        raise ValueError("exact ceiling failed")
    for row in records:
        row["reference_rewrite_count"] = depths[row["root"]]
    _rows(out / "root_results.jsonl", records)
    _rows(out / "native_episodes.jsonl", episodes)
    return records, backend, compiled


def summarize(records):
    summary = {}
    for arm in (*PUBLIC, *LEARNED, "exact_ceiling"):
        summary[arm] = {}
        for cap in (CAPS if arm != "exact_ceiling" else (EXACT_CAP,)):
            rows = [r for r in records if r["arm"] == arm and r["expansion_cap"] == cap]
            n = len(rows)
            if n != DEV_N*(len(SEEDS) if arm in LEARNED else 1):
                raise ValueError("registered arm/cap census mismatch")
            s = {"episodes": n, "certified": sum(r["certified"] for r in rows),
                 "closure": sum(r["certified"] for r in rows)/n,
                 "edge_threshold_certified": {str(e): sum(r["certified"] and r["candidate_edges_enumerated"] <= e
                                                          for r in rows) for e in EDGE_CAPS},
                 "joint_cap_certified": sum(r["certified"] and r["expansions"]+r["native_tactics"] <= cap for r in rows),
                 "by_distance": {}}
            for key in ("expansions", "candidate_edges_enumerated", "planning_seconds", "native_tactics",
                        "setup_states", "setup_rule_checks", "setup_abstract_expansions", "setup_abstract_edges",
                        "model_states_scored", "model_batches", "inference_seconds"):
                s["mean_"+key] = sum(r.get(key,0) for r in rows)/n
            for lo,hi in BANDS:
                band = [r for r in rows if lo <= r["reference_rewrite_count"] <= hi]
                s["by_distance"][f"{lo}-{hi}"] = {"episodes":len(band),
                                                   "certified":sum(r["certified"] for r in band)}
            summary[arm][str(cap)] = s
    return summary


def gate(records):
    grouped = defaultdict(list)
    for row in records:
        if row["expansion_cap"] == 64:
            grouped[row["root"], row["arm"]].append(int(row["certified"]))
    roots = sorted({r for r,a in grouped})
    if len(roots) != DEV_N:
        raise ValueError("primary root census mismatch")
    means = {}
    for r in roots:
        for a in (*PUBLIC,*LEARNED):
            v = grouped[r,a]
            if len(v) != (len(SEEDS) if a in LEARNED else 1):
                raise ValueError("primary seed/arm mismatch")
            means[r,a] = sum(v)/len(v)

    def mean(rs, arm):
        return sum(means[r,arm] for r in rs)/len(rs)

    def delta(rs):
        return mean(rs,"residual") - max(mean(rs,a) for a in PUBLIC)

    rng = random.Random(DEV_SEED)
    boot = sorted(delta([rng.choice(roots) for _ in roots]) for _ in range(2000))
    seed_diffs = {str(s): sum((1 if r["arm"]=="residual" else -1)*int(r["certified"])
                             for r in records if r["expansion_cap"]==64 and r["seed"]==s
                             and r["arm"] in ("residual","sham"))/DEV_N for s in SEEDS}
    sham = mean(roots,"residual")-mean(roots,"sham")
    return {"residual_minus_strongest_public": delta(roots),
            "strongest_public": max(PUBLIC,key=lambda a: mean(roots,a)),
            "bootstrap95": [boot[50],boot[1949]], "residual_minus_sham":sham,
            "seed_differences":seed_diffs,
            "residual_minus_one_step":mean(roots,"residual")-mean(roots,"one_step"),
            "residual_minus_reference_only":mean(roots,"residual")-mean(roots,"reference_only"),
            "developmental_gate":delta(roots)>=0.09375 and boot[50]>0 and sham>=0.0625
                                 and min(seed_diffs.values())>0}


def run(suite, old, teacher, dev, out):
    cases = load_dev(suite, old, dev)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    samples, source = build_pairs(suite, teacher, out / "teacher")
    models, fits = {}, []
    for arm in LEARNED:
        for seed in SEEDS:
            t = time.perf_counter()
            a, ra = fit(samples, arm, seed)
            b, rb = fit(samples, arm, seed)
            if ra["state_digest"] != rb["state_digest"] or ra["updates"] != rb["updates"]:
                raise ValueError("duplicate training mismatch")
            path = out / f"{arm}_{seed}.pt"
            torch.save(a.state_dict(), path)
            models[arm,seed] = a
            fits.append(dict(ra, arm=arm, seed=seed, duplicate_state_digest=rb["state_digest"],
                             seconds=time.perf_counter()-t, checkpoint_sha256=sha256_file(path)))
    if len({r["updates"] for r in fits}) != 1 or len({r["params"] for r in fits}) != 1:
        raise ValueError("matched update/parameter census failed")
    _write(out / "training.json", fits)
    records, backend, compiled = evaluate(cases, models, out / "evaluation")
    m = {"schema":"lsa02m.residual_ordering_development.v1","backend":backend,
         "roots":len(cases),"teacher":source,"unique_eval_native_proofs":compiled,
         "summary":summarize(records),"gate":gate(records),
         "development_sha256":sha256_file(dev / "manifest.json"),
         "sealed_contents_parsed":False,"scientific_eligible":False,
         "architecture_promotion_eligible":False,
         "root_results_sha256":sha256_file(out / "evaluation/root_results.jsonl")}
    _write(out / "manifest.json",m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("freeze","run"))
    p.add_argument("--suite",type=Path,required=True)
    p.add_argument("--old-dev",type=Path,required=True)
    p.add_argument("--teacher",type=Path)
    p.add_argument("--dev",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    if a.command == "freeze":
        result = freeze(a.suite,a.old_dev,a.out)
    else:
        if a.teacher is None or a.dev is None:
            p.error("run requires --teacher and --dev")
        result = run(a.suite,a.old_dev,a.teacher,a.dev,a.out)
    print(canonical(result))


if __name__ == "__main__":
    main()

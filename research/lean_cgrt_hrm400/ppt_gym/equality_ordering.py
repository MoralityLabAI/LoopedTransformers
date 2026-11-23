"""LSA-02L development-only learned ordering of public equality search."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from itertools import combinations, permutations, product
import heapq
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
from .data import canonical, digest, sha256_file
from .equality_suite import (ALPHABET, BANDS, RULE_COUNT, EqCase, choose_shell,
                             load_split, neighbors, system_lineage)
from .equality_search import BUDGET, CAPS, EXACT_CAP, plan, path_moves, run_live, replay
from .equality_preflight import compile_scripts, _native_backend, _rows, _write

SUITE_SHA = "d928d8b5ffbbd9760e02fc9af903704bdac0cea7adc0b6e5be9103e7f2153874"
DEV_SEED = 2026112402
DEV_N = 32
SEEDS = (107, 109)
EPOCHS = 30
WIDTH = 48
DIM = 196
EDGE_CAPS = (128, 512, 2048, 8192)
PUBLIC = ("bidirectional", "hamming_greedy", "hamming_astar", "edit_astar",
          "positional_astar", "pair_astar")
LEARNED = ("learned", "sham", "one_step")


def _suite(suite):
    if sha256_file(suite / "manifest.json") != SUITE_SHA:
        raise ValueError("LSA-02K frozen suite manifest mismatch")
    return load_split(suite, "train")


def freeze_dev(suite, out):
    _, meta = _suite(suite)
    used = {lin for shard in meta["shards"].values() for lin in shard["lineages"]}
    rng = random.Random(DEV_SEED)
    pairs = ["".join(x) for x in product(ALPHABET, repeat=2)]
    available = list(combinations(pairs, 2))
    rows = []
    attempts = 0
    while len(rows) < DEV_N:
        attempts += 1
        if attempts > DEV_N * 500:
            raise RuntimeError("fresh development roster generation failed")
        rules = tuple((b, a) if rng.randrange(2) else (a, b)
                      for a, b in rng.sample(available, RULE_COUNT))
        lin = system_lineage(rules)
        if lin in used:
            continue
        i = len(rows)
        source = "".join(rng.choice(ALPHABET) for _ in range(7 + (i // 3) % 3))
        target = choose_shell(source, rules, BANDS[i % 3], rng)
        if target is None:
            continue
        root = digest([lin, "calibration", rules, source, target])[:24]
        c = EqCase(root, lin, "calibration", "indexed_unary_equations", rules,
                   source, target).validate()
        used.add(lin)
        rows.append(dict(vars(c)))
    out.mkdir(parents=True, exist_ok=False)
    _rows(out / "roots.jsonl", rows)
    manifest = {"schema": "lsa02l.fresh_development.v1", "seed": DEV_SEED,
                "suite_sha256": SUITE_SHA, "roots": DEV_N,
                "lineages": [r["lineage"] for r in rows],
                "rows_sha256": sha256_file(out / "roots.jsonl"),
                "generator_sha256": sha256_file(Path(__file__)),
                "scientific_eligible": False}
    _write(out / "manifest.json", manifest)
    return manifest


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_dev(suite, directory):
    _, original = _suite(suite)
    m = json.loads((directory / "manifest.json").read_text())
    if (m["schema"] != "lsa02l.fresh_development.v1" or m["seed"] != DEV_SEED
        or m["suite_sha256"] != SUITE_SHA or m["roots"] != DEV_N
        or m["generator_sha256"] != sha256_file(Path(__file__))
        or m["rows_sha256"] != sha256_file(directory / "roots.jsonl")):
        raise ValueError("fresh development manifest mismatch")
    result = []
    seen = {lin for shard in original["shards"].values() for lin in shard["lineages"]}
    for x in _jsonl(directory / "roots.jsonl"):
        if set(x) != set(EqCase.__dataclass_fields__):
            raise ValueError("development row carries private fields")
        x["equations"] = tuple(tuple(p) for p in x["equations"])
        c = EqCase(**x).validate()
        if c.split != "calibration" or c.lineage in seen:
            raise ValueError("development lineage collision")
        seen.add(c.lineage)
        result.append(c)
    if len(result) != DEV_N or [c.lineage for c in result] != m["lineages"]:
        raise ValueError("development roster mismatch")
    return result


class Encoder:
    def __init__(self, case):
        variants = []
        for p in permutations(ALPHABET):
            tr = str.maketrans(ALPHABET, "".join(p))
            rules = tuple(sorted(tuple(sorted((a.translate(tr), b.translate(tr))))
                                 for a, b in case.equations))
            variants.append(((rules, case.source.translate(tr), case.target.translate(tr)), tr))
        key = min(k for k, _ in variants)
        self.maps = [tr for k, tr in variants if k == key]
        rules, source, _ = key
        self.base = self.word(source) + self.onehot("".join(a + b for a, b in rules), ALPHABET)

    @staticmethod
    def onehot(chars, alphabet):
        return [float(c == a) for c in chars for a in alphabet]

    @classmethod
    def word(cls, word):
        return cls.onehot(word.ljust(10, "_"), ALPHABET + "_")

    def __call__(self, word):
        x = self.word(min(word.translate(tr) for tr in self.maps)) + self.base
        if len(x) != DIM:
            raise ValueError("encoding width mismatch")
        return x


def training_rows(suite, teacher):
    cases, _ = _suite(suite)
    m = json.loads((teacher / "manifest.json").read_text())
    if (m["schema"] != "lsa02k.native_equality_training_source.v1"
        or m["suite_manifest_sha256"] != SUITE_SHA or m["training_roots"] != 64
        or m["reference_proofs_compiled"] != 64 or m["actual_native_replays"] != 64
        or m["reference_frames"] != 546 or not m["train_only"]
        or m["heldout_labels_opened"] or not m["backend"]["actual_lean_process"]):
        raise ValueError("native training custody failure")
    required = {"reference_episodes.jsonl", "root_census.jsonl", "train_reference_frames.jsonl",
                "standalone.lean", "compiler_receipt.json", "compiler.log"}
    if set(m["artifacts"]) != required:
        raise ValueError("native source artifact roster mismatch")
    for name, sha in m["artifacts"].items():
        if name != Path(name).name or sha256_file(teacher / name) != sha:
            raise ValueError("native source hash mismatch")
    receipt = json.loads((teacher / "compiler_receipt.json").read_text())
    if receipt["passed"] != 64 or not receipt["no_extra_axioms"]:
        raise ValueError("standalone teacher receipt mismatch")
    episodes = _jsonl(teacher / "reference_episodes.jsonl")
    census = _jsonl(teacher / "root_census.jsonl")
    frames = _jsonl(teacher / "train_reference_frames.jsonl")
    if len(episodes) != 64 or len(census) != 64 or len(frames) != 546:
        raise ValueError("teacher row count mismatch")
    rows = []
    for case, ep, root in zip(cases, episodes, census):
        if (ep["root"] != case.root or root["root"] != case.root
            or ep["lineage"] != case.lineage or ep["split"] != "train"
            or not ep["standalone_verified"] or not ep["live_kernel"]
            or ep["backtracks"] or ep["effort"] != len(ep["events"])
            or root["minimum_public_rewrite_count"] + 1 != ep["effort"]):
            raise ValueError("reference episode mismatch")
        enc = Encoder(case)
        word = case.target
        for i, event in enumerate(ep["events"]):
            frame = frames[len(rows)]
            moves = neighbors(word, case.equations)
            menu = [m.tactic(word) for m in moves]
            if word == case.source:
                menu.append("exact w")
            if (event["kind"] != "TACTIC" or event["word_before"] != word
                or frame["root"] != case.root or frame["lineage"] != case.lineage
                or frame["split"] != "train" or frame["current_index"] != word
                or frame["witness_index"] != case.source
                or tuple(tuple(p) for p in frame["equations"]) != case.equations
                or frame["label_status"] != "selected_native_certified_path_only"
                or frame["public_state_sha256"] != event["state_before_sha256"]
                or frame["candidate_tactics"] != menu
                or frame["selected_native_reference_action"] != event["tactic"]):
                raise ValueError("frame does not match certified reference event")
            terminal = word == case.source
            distance = len(ep["events"]) - i - 1
            if terminal:
                if event["tactic"] != "exact w" or distance != 0 or event["word_after"] is not None:
                    raise ValueError("invalid terminal reference")
                candidates, label = [word], 0
            else:
                next_word = event["word_after"]
                if not any(m.next_word == next_word and m.tactic(word) == event["tactic"] for m in moves):
                    raise ValueError("invalid certified successor")
                candidates = sorted({m.next_word for m in moves})
                label = candidates.index(next_word)
            rows.append((enc(word), [enc(w) for w in candidates], label, distance,
                         not terminal, len(word)))
            word = event["word_after"]
        if word is not None:
            raise ValueError("reference did not terminate")
    if len(rows) != len(frames):
        raise ValueError("unconsumed reference frames")
    return rows


class Net(nn.Module):
    def __init__(self, steps=4):
        super().__init__()
        self.steps = steps
        self.project = nn.Linear(DIM, WIDTH)
        self.loop = nn.Linear(WIDTH, WIDTH)
        self.readout = nn.Linear(WIDTH, 1)

    def forward(self, x):
        z = self.project(x)
        h = torch.tanh(z)
        for _ in range(self.steps):
            h = torch.tanh(self.loop(h) + z)
        return F.softplus(self.readout(h).squeeze(-1))


def _tensors(rows, sham, seed):
    n = len(rows)
    k = max(len(r[1]) for r in rows)
    state = torch.tensor([r[0] for r in rows], dtype=torch.float32)
    candidates = torch.zeros(n, k, DIM)
    mask = torch.zeros(n, k, dtype=torch.bool)
    labels = [r[2] for r in rows]
    distances = [r[3] for r in rows]
    active = torch.tensor([r[4] for r in rows], dtype=torch.bool)
    rng = random.Random(seed + 8001)
    bylength = defaultdict(list)
    for i, r in enumerate(rows):
        candidates[i, :len(r[1])] = torch.tensor(r[1])
        mask[i, :len(r[1])] = True
        bylength[r[5]].append(i)
        if sham and r[4]:
            labels[i] = rng.randrange(len(r[1]))
    if sham:
        for indices in bylength.values():
            shuffled = [distances[i] for i in indices]
            rng.shuffle(shuffled)
            for i, value in zip(indices, shuffled):
                distances[i] = value
    return state, candidates, mask, torch.tensor(labels), torch.tensor(distances, dtype=torch.float32), active


def fit(rows, arm, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 93)
    net = Net(1 if arm == "one_step" else 4)
    state, candidates, mask, labels, distances, active = _tensors(rows, arm == "sham", seed)
    opt = torch.optim.AdamW(net.parameters(), lr=0.003)
    updates = 0
    for _ in range(EPOCHS):
        for ix in torch.randperm(len(rows), generator=gen).split(64):
            pred = net(state[ix])
            loss = F.smooth_l1_loss(pred, distances[ix])
            logits = -net(candidates[ix].reshape(-1, DIM)).reshape(len(ix), -1)
            logits = logits.masked_fill(~mask[ix], -1e9)
            if active[ix].any():
                loss = loss + F.cross_entropy(logits[active[ix]], labels[ix][active[ix]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
    sha = digest({k: v.detach().tolist() for k, v in net.state_dict().items()})
    return net.eval(), {"state_digest": sha, "updates": updates,
                        "params": sum(p.numel() for p in net.parameters()),
                        "steps": net.steps, "last_loss": float(loss.detach())}


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def edit(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j-1] + (x != y)))
        row = nxt
    return row[-1]


def positional_bound(case):
    tables = []
    for i in range(len(case.source)):
        d = {(a, b): (0 if a == b else math.inf) for a in ALPHABET for b in ALPHABET}
        for a, b in case.equations:
            for j in ((0,) if i == 0 else (1,) if i == len(case.source)-1 else (0, 1)):
                if a[j] != b[j]:
                    d[a[j], b[j]] = d[b[j], a[j]] = 1
        for k in ALPHABET:
            for a in ALPHABET:
                for b in ALPHABET:
                    d[a, b] = min(d[a, b], d[a, k] + d[k, b])
        tables.append(d)

    def h(word):
        vals = [d[a, b] for d, a, b in zip(tables, word, case.source)]
        return max(max(vals), math.ceil(sum(vals) / 2)) if all(math.isfinite(v) for v in vals) else math.inf
    return h


def pair_bound(case):
    """Relax each adjacent window, allowing either outside letter to vary freely."""
    tables = []
    expansions = edges = 0
    for i in range(len(case.source)-1):
        goal = case.source[i:i+2]
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            w = queue.popleft()
            expansions += 1
            options = set()
            for a, b in case.equations:
                for lhs, rhs in ((a, b), (b, a)):
                    if w == lhs:
                        options.add(rhs)
                    if i > 0 and w[0] == lhs[1]:
                        options.add(rhs[1] + w[1])
                    if i < len(case.source)-2 and w[1] == lhs[0]:
                        options.add(w[0] + rhs[0])
            options.discard(w)
            edges += len(options)
            for child in sorted(options):
                if child not in dist:
                    dist[child] = dist[w] + 1
                    queue.append(child)
        tables.append(dist)
    pos = positional_bound(case)

    def h(word):
        vals = [d.get(word[i:i+2], math.inf) for i, d in enumerate(tables)]
        if not all(math.isfinite(v) for v in vals):
            return math.inf
        return max(pos(word), max(vals), math.ceil(sum(vals) / 3))
    return h, expansions, edges


def best_first(case, cap, arm, net=None):
    if cap < 1 or arm not in (*PUBLIC[1:], *LEARNED):
        raise ValueError("invalid ordering or cap")
    if (net is None) != (arm in PUBLIC):
        raise ValueError("ordering/model mismatch")
    t0 = time.perf_counter()
    setup_expansions = setup_edges = batches = scored = 0
    inference_seconds = 0.0
    cache = {case.source: 0.0}
    greedy = arm == "hamming_greedy"
    if net is not None:
        enc = Encoder(case)
    elif arm == "pair_astar":
        heuristic, setup_expansions, setup_edges = pair_bound(case)
    elif arm == "positional_astar":
        heuristic = positional_bound(case)
    elif arm == "edit_astar":
        heuristic = lambda w: math.ceil(edit(w, case.source) / 2)
    elif greedy:
        heuristic = lambda w: hamming(w, case.source)
    else:
        heuristic = lambda w: math.ceil(hamming(w, case.source) / 2)

    def score(words):
        nonlocal batches, scored, inference_seconds
        missing = list(dict.fromkeys(w for w in words if w not in cache))
        if not missing:
            return
        if net is None:
            vals = [heuristic(w) for w in missing]
        else:
            t = time.perf_counter()
            with torch.inference_mode():
                vals = net(torch.tensor([enc(w) for w in missing], dtype=torch.float32)).tolist()
            inference_seconds += time.perf_counter() - t
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("nonfinite learned score")
            batches += 1
            scored += len(missing)
        cache.update(zip(missing, vals))

    start = case.target
    score([start])
    todo = [(cache[start], hamming(start, case.source), 0, start, (start,))]
    best = {start: 0}
    expansions = edges = 0
    result = ()
    while todo:
        _, _, g, w, path = heapq.heappop(todo)
        if best[w] != g:
            continue
        if w == case.source:
            result = path
            break
        if g >= BUDGET - 1:
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
            h = cache[child]
            heapq.heappush(todo, (h if greedy else g+1+h, hamming(child, case.source),
                                  g+1, child, (*path, child)))
    return {"found": bool(result), "words": list(result),
            "expansions": expansions, "candidate_edges_enumerated": edges,
            "planning_seconds": time.perf_counter()-t0, "expansion_cap": cap,
            "setup_abstract_expansions": setup_expansions, "setup_abstract_edges": setup_edges,
            "model_batches": batches, "model_states_scored": scored,
            "inference_seconds": inference_seconds}


def _public_plan(case, cap):
    r = plan(case, cap=cap)
    if r["found"] and len(r["words"]) > BUDGET:
        r.update(found=False, words=[])
    r.update(setup_abstract_expansions=0, setup_abstract_edges=0,
             model_batches=0, model_states_scored=0, inference_seconds=0.0)
    return r


def _summary(records):
    out = {}
    for arm in (*PUBLIC, *LEARNED, "exact_ceiling"):
        out[arm] = {}
        for cap in (CAPS if arm != "exact_ceiling" else (EXACT_CAP,)):
            rows = [r for r in records if r["arm"] == arm and r["expansion_cap"] == cap]
            n = len(rows)
            if n == 0:
                raise ValueError("missing registered arm/cap")
            out[arm][str(cap)] = {
                "episodes": n, "certified": sum(r["certified"] for r in rows),
                "closure": sum(r["certified"] for r in rows)/n,
                "mean_expansions": sum(r["expansions"] for r in rows)/n,
                "mean_edges": sum(r["candidate_edges_enumerated"] for r in rows)/n,
                "mean_planning_seconds": sum(r["planning_seconds"] for r in rows)/n,
                "mean_model_states_scored": sum(r["model_states_scored"] for r in rows)/n,
                "mean_model_batches": sum(r["model_batches"] for r in rows)/n,
                "mean_inference_seconds": sum(r["inference_seconds"] for r in rows)/n,
                "mean_setup_abstract_expansions": sum(r["setup_abstract_expansions"] for r in rows)/n,
                "mean_setup_abstract_edges": sum(r["setup_abstract_edges"] for r in rows)/n,
                "joint_cap_certified": sum(r["certified"] and r["expansions"]+r["native_tactics"] <= cap for r in rows),
                "edge_threshold_certified": {str(e): sum(r["certified"] and r["candidate_edges_enumerated"] <= e
                                                        for r in rows) for e in EDGE_CAPS},
            }
    return out


def _gate(records):
    grouped = defaultdict(list)
    for r in records:
        if r["expansion_cap"] == 64:
            grouped[r["root"], r["arm"]].append(int(r["certified"]))
    roots = sorted({r for r, _ in grouped})
    if len(roots) != DEV_N:
        raise ValueError("primary root count changed")
    means = {}
    for root in roots:
        for arm in (*PUBLIC, *LEARNED):
            vals = grouped[root, arm]
            if len(vals) != (len(SEEDS) if arm in LEARNED else 1):
                raise ValueError("primary arm/seed census mismatch")
            means[root, arm] = sum(vals)/len(vals)

    def mean(sample, arm):
        return sum(means[r, arm] for r in sample)/len(sample)

    def delta(sample):
        return mean(sample, "learned") - max(mean(sample, a) for a in PUBLIC)

    rng = random.Random(DEV_SEED)
    boot = sorted(delta([rng.choice(roots) for _ in roots]) for _ in range(2000))
    seed_diffs = {}
    for seed in SEEDS:
        seed_diffs[str(seed)] = sum(
            (1 if r["arm"] == "learned" else -1) * int(r["certified"])
            for r in records if r["expansion_cap"] == 64 and r["seed"] == seed
            and r["arm"] in ("learned", "sham"))/DEV_N
    sham = mean(roots, "learned") - mean(roots, "sham")
    recurrence = mean(roots, "learned") - mean(roots, "one_step")
    return {"learned_minus_strongest_public": delta(roots),
            "strongest_public": max(PUBLIC, key=lambda a: mean(roots, a)),
            "root_paired_bootstrap95": [boot[50], boot[1949]],
            "learned_minus_sham": sham, "seed_differences": seed_diffs,
            "learned_minus_single_step": recurrence,
            "developmental_gate": delta(roots) >= 0.125 and boot[50] > 0
                and sham >= 0.0625 and min(seed_diffs.values()) > 0,
            "recurrence_direction_positive": recurrence > 0,
            "scientific_eligible": False, "architecture_promotion_eligible": False}


def run(suite, teacher, dev, out):
    if out.exists():
        raise ValueError("fresh output required")
    # Authenticate the frozen development roster before fitting.
    dev_cases = load_dev(suite, dev)
    rows = training_rows(suite, teacher)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = {}
    fits = []
    for arm in LEARNED:
        for seed in SEEDS:
            t = time.perf_counter()
            a, ra = fit(rows, arm, seed)
            b, rb = fit(rows, arm, seed)
            if ra["state_digest"] != rb["state_digest"] or ra["updates"] != rb["updates"]:
                raise ValueError("same-worker duplicate training failed")
            path = out / f"{arm}_{seed}.pt"
            torch.save(a.state_dict(), path)
            models[arm, seed] = a
            fits.append(dict(ra, arm=arm, seed=seed, duplicate_state_digest=rb["state_digest"],
                             seconds=time.perf_counter()-t, checkpoint_sha256=sha256_file(path)))
    _write(out / "training.json", fits)

    records, cases_by_root = [], {c.root: c for c in dev_cases}
    for i, case in enumerate(dev_cases):
        arms = [(a, None) for a in PUBLIC] + [(a, s) for a in LEARNED for s in SEEDS]
        # Fixed rotation balances deterministic run order across roots.
        arms = arms[i % len(arms):] + arms[:i % len(arms)]
        for arm, seed in arms:
            for cap in CAPS:
                r = (_public_plan(case, cap) if arm == "bidirectional" else
                     best_first(case, cap, arm, models.get((arm, seed))))
                records.append(dict(r, arm=arm, seed=seed, root=case.root, lineage=case.lineage))
        records.append(dict(_public_plan(case, EXACT_CAP), arm="exact_ceiling",
                            seed=None, root=case.root, lineage=case.lineage))
    _rows(out / "proposed_paths.jsonl", records)
    # All proposals are frozen before any native certificate pass.
    certs = {}
    episodes = []
    items = []
    certdir = out / "certificates"
    certdir.mkdir()
    with LeanRepl(imports=("Lean",), timeout=60, exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR, env=config.lean_env()) as repl:
        backend = _native_backend(repl)
        for r in records:
            r.update(certified=False, native_tactics=0, certificate_id=None)
            if not r["found"]:
                continue
            case = cases_by_root[r["root"]]
            key = (case.root, tuple(r["words"]))
            if key not in certs:
                path_moves(case, r["words"])
                label = f"p{len(certs)}"
                t = time.perf_counter()
                # The existing path-based native adapter uses this policy name;
                # algorithm identity remains in the outer planner receipt.
                ep = run_live(repl, case, "bidirectional_bfs", words=r["words"])
                if not ep["live_kernel"] or not replay(repl, case, ep):
                    raise ValueError("proposed equality path failed native proof/replay")
                certs[key] = (label, ep["effort"])
                items.append((case, label, ep["script"]))
                episodes.append({"certificate_id": label, "episode": ep,
                                 "native_and_replay_seconds": time.perf_counter()-t})
            r["certificate_id"], r["native_tactics"] = certs[key]
    receipt = compile_scripts(items, certdir, timeout=600)
    for item in episodes:
        item["episode"]["standalone_verified"] = True
    for r in records:
        r["certified"] = r["certificate_id"] is not None
    if not all(r["certified"] for r in records if r["arm"] == "exact_ceiling"):
        raise ValueError("fresh reachable cohort failed exact ceiling")
    _rows(out / "native_episodes.jsonl", episodes)
    _rows(out / "root_results.jsonl", records)
    report = {"schema": "lsa02l.learned_ordering_development.v1", "backend": backend,
              "suite_sha256": SUITE_SHA, "teacher_sha256": sha256_file(teacher / "manifest.json"),
              "development_sha256": sha256_file(dev / "manifest.json"),
              "roots": len(dev_cases), "training_frames": len(rows),
              "unique_native_proofs": receipt["passed"], "summary": _summary(records),
              "gate": _gate(records), "sealed_contents_parsed": False,
              "scientific_eligible": False, "architecture_promotion_eligible": False,
              "artifacts": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}}
    _write(out / "manifest.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze-dev", "run"))
    p.add_argument("--suite", required=True, type=Path)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--dev", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.command == "freeze-dev":
        result = freeze_dev(a.suite, a.out)
    else:
        if a.teacher is None or a.dev is None:
            p.error("run requires --teacher and --dev")
        result = run(a.suite, a.teacher, a.dev, a.out)
    print(canonical(result))


if __name__ == "__main__":
    main()

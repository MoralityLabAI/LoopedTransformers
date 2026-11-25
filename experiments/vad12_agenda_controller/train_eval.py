"""VAD-12 neural advancement gate (CPU): recurrent research-agenda controller vs matched baselines.

Trains on exact-oracle decision sequences (gen_data.py) and scores policies by exact forward evaluation on
FRESH structural families (wide graphs, planted complementary pairs, planted chains; seeds 95000+) and on an
in-distribution held-out set. Utility regret is measured against the exact optimum; costs are measured
(wall time, analytic FLOPs, Q-evaluations) and training cost is amortized. Run with the bluebeam venv.
"""
import argparse, hashlib, json, os, sys, time
from fractions import Fraction as F
from pathlib import Path
from random import Random

EV = Path(os.environ.get("EVALMATH_VAD", r"C:\Users\patri\OneDrive\Documents\GitHub\EvalMath-vad")) / "scripts"
sys.path.insert(0, str(EV))
import numpy as np
import torch
import torch.nn as nn
import vad_agenda as g
import vad_agenda_families as fm

S, FD, T = fm.N_SLOT, fm.FDIM, 6
HERE = Path(__file__).parent


# ---- models -----------------------------------------------------------------------------------

class Slotnet(nn.Module):
    """Weight-tied refinement over action slots (K inner steps) with optional episode memory (GRU)."""

    def __init__(self, d=32, K=4, mem=True, oracle=False):
        super().__init__()
        self.d, self.K, self.mem, self.oracle = d, K, mem, oracle
        self.enc = nn.Linear(FD, d)
        self.step = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.logit = nn.Linear(d, 1)
        self.qhead = nn.Linear(d, 1)
        if mem:
            self.gru = nn.GRUCell(d + S + 1, d)

    def forward(self, x, mask, h, prev_slot, prev_out):
        B = x.shape[0]
        if self.mem:
            oh = torch.zeros(B, S + 1)
            oh[torch.arange(B), prev_slot + 1] = 1.0  # index 0 = no previous action
            h = self.gru(torch.cat([self.pool_cache, oh[:, 1:], prev_out[:, None]], 1), h) if self.pool_cache is not None else h
        z = torch.relu(self.enc(x))
        m = mask.float()[..., None]
        for _ in range(self.K):
            pooled = (z * m).sum(1) / m.sum(1).clamp(min=1)
            hh = h if self.mem else torch.zeros_like(pooled)
            z = z + self.step(torch.cat([z, pooled[:, None].expand_as(z), hh[:, None].expand_as(z)], -1))
        pooled = (z * m).sum(1) / m.sum(1).clamp(min=1)
        logits = self.logit(z).squeeze(-1).masked_fill(~mask, -1e9)
        return logits, self.qhead(z).squeeze(-1), pooled, h

    pool_cache = None


class FF(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(S * FD, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * S))

    def forward(self, x, mask, h, prev_slot, prev_out):
        o = self.net(x.flatten(1))
        return o[:, :S].masked_fill(~mask, -1e9), o[:, S:], None, h


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def matched(build, target, lo, hi):
    best = min(range(lo, hi), key=lambda w: abs(nparams(build(w)) - target))
    return best


def make_models():
    trm = Slotnet(32, 4, True)
    P = nparams(trm)
    arms = {
        "trm": trm,
        "memoryless": Slotnet(matched(lambda d: Slotnet(d, 4, False), P, 8, 80), 4, False),
        "fixed_depth": Slotnet(matched(lambda d: Slotnet(d, 1, True), P, 8, 80), 1, True),
        "ff": FF(matched(FF, P, 8, 200)),
        "trm_oracle_features": Slotnet(32, 4, True, oracle=True),
    }
    return arms


# ---- training ---------------------------------------------------------------------------------

def load(path):
    z = np.load(path)
    return {k: torch.tensor(z[k][:, :T] if z[k].ndim >= 2 else z[k]) for k in z.files}


def run_sequence(model, X, M, PS, PO, V, oracle_G=None, shuffle_memory=None):
    """Unroll the model over T steps for a batch of sequences; returns logits/q lists."""
    B = X.shape[0]
    d = getattr(model, "d", 1)
    h = torch.zeros(B, d)
    logits, qs = [], []
    last_pool = None
    for t in range(T):
        x = X[:, t].clone()
        if oracle_G is not None:
            x[:, :, 13] = oracle_G[:, t]
        if isinstance(model, Slotnet) and model.mem:
            model.pool_cache = last_pool
        lg, q, pooled, h = model(x, M[:, t], h, PS[:, t], PO[:, t])
        if shuffle_memory is not None and isinstance(model, Slotnet) and model.mem:
            h = h[shuffle_memory]
        last_pool = pooled
        logits.append(lg)
        qs.append(q)
    return torch.stack(logits, 1), torch.stack(qs, 1)


def train(model, tr, ca, use_oracle, epochs, seed, log):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = tr["X"].shape[0]
    best, best_state = -1, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            lg, q = run_sequence(model, tr["X"][idx], tr["M"][idx], tr["PS"][idx], tr["PO"][idx], tr["V"][idx], tr["G"][idx] if use_oracle else None)
            v = tr["V"][idx]
            ce = nn.functional.cross_entropy(lg[v], tr["Y"][idx][v])
            mse = (((q - tr["Q"][idx]) ** 2) * tr["M"][idx]).sum() / tr["M"][idx].sum()
            loss = ce + 0.5 * mse
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lg, _ = run_sequence(model, ca["X"], ca["M"], ca["PS"], ca["PO"], ca["V"], ca["G"] if use_oracle else None)
            acc = (lg.argmax(-1)[ca["V"]] == ca["Y"][ca["V"]]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            log(f"  epoch {ep} calib action accuracy {acc:.4f}")
    model.load_state_dict(best_state)
    return best, time.perf_counter() - t0, epochs * n * T


def flops_per_decision(model):
    if isinstance(model, FF):
        return 2 * sum(p.numel() for n_, p in model.named_parameters() if n_.endswith("weight"))
    f = 2 * S * (FD * model.d)
    f += model.K * 2 * S * (3 * model.d * model.d + model.d * model.d)
    f += 2 * S * 2 * model.d
    if model.mem:
        f += 2 * 3 * (model.d + S + 1 + model.d) * model.d
    return f


# ---- exact evaluation with memory -------------------------------------------------------------

class NNPolicy:
    def __init__(self, model, oracle=False, pool=None, seed=0):
        self.m, self.oracle, self.pool, self.rng = model, oracle, pool, Random(seed)
        self.calls = 0
        self.seconds = 0.0

    def initial(self):
        return (torch.zeros(1, getattr(self.m, "d", 1)), -1, 0.0, None, 0)

    def act(self, orc, struct, L, pi, b, mem):
        t0 = time.perf_counter()
        rows, mask = fm.features(orc, L, pi, b, struct, oracle_features=self.oracle)
        x = torch.tensor([rows], dtype=torch.float32)
        mk = torch.tensor([mask])
        h, ps, po, pool, step = mem
        if isinstance(self.m, Slotnet) and self.m.mem:
            self.m.pool_cache = pool
            if self.pool is not None and step > 0:  # shuffled-memory ablation: another episode's state at this step
                bank = self.pool[min(step, len(self.pool) - 1)]
                h = bank[self.rng.randrange(len(bank))][None]
        with torch.no_grad():
            lg, _, pooled, h2 = self.m(x, mk, h, torch.tensor([ps]), torch.tensor([po], dtype=torch.float32))
        a = fm.action_of_slot(orc.i, int(lg.argmax(-1)))
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return a, (h2, ps, po, pooled, step)

    def update(self, mem, a, outcome):
        h, _, _, pooled, step = mem
        return (h, fm.slot_of_action(a), float(outcome), pooled, step + 1)


def evaluate_mem(inst, orc, struct, pol):
    """Exact expected utility of an NNPolicy; mirrors vad_agenda.evaluate with memory threaded through."""
    T_ = g.Oracle(inst)
    A = orc

    def go(L, b, tw, aw, mem):
        mass = sum(tw)
        if not mass:
            return F(0)
        a, mem2 = pol.act(A, struct, L, aw, b, mem)
        if a not in A.actions(L, aw, b):
            a = g.STOP
        if a == g.STOP:
            return sum(t * T_.J(L, w) for w, t in enumerate(tw) if t)
        if a[0] == "att":
            k = a[1]; e = inst.edges[k]
            ts = [t if k in inst.worlds[w] else F(0) for w, t in enumerate(tw)]
            tf = [t if k not in inst.worlds[w] else F(0) for w, t in enumerate(tw)]
            ps = sum(p for w, p in enumerate(aw) if k in inst.worlds[w])
            a_s = tuple(p / ps if k in inst.worlds[w] else F(0) for w, p in enumerate(aw)) if ps else aw
            a_f = tuple(p / (1 - ps) if k not in inst.worlds[w] else F(0) for w, p in enumerate(aw)) if ps != 1 else aw
            return (sum(ts) * inst.reward[e.concl] - inst.lam * e.cost * mass
                    + go(L | 1 << e.concl, b - e.cost, ts, a_s, pol.update(mem2, a, 1))
                    + go(L, b - e.cost, tf, a_f, pol.update(mem2, a, 0)))
        x = inst.experiments[a[1]]
        tot = -inst.lam * x.cost * mass
        for y in range(len(x.lik)):
            tws = [t * x.lik[y][w] for w, t in enumerate(tw)]
            awu = [p * x.lik[y][w] for w, p in enumerate(aw)]
            z = sum(awu)
            tot += go(L, b - x.cost, tws, tuple(p / z for p in awu) if z else aw, pol.update(mem2, a, y))
        return tot

    return go(A.start, inst.budget, list(inst.prior), inst.prior, pol.initial())


# ---- driver -----------------------------------------------------------------------------------

def paired_bootstrap(d, seed=0, B=2000):
    r = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = [d[r.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fresh", type=int, default=80)
    ap.add_argument("--indist", type=int, default=120)
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    a = ap.parse_args()
    torch.set_num_threads(1)
    lines = []
    log = lambda s: (print(s, flush=True), lines.append(s))
    tr, ca = load(HERE / "data/train.npz"), load(HERE / "data/calib.npz")
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    res = {"protocol": "VAD-12", "data_manifest": manifest, "epochs": a.epochs, "models": {}, "test": {}}
    trained = {}
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        for name, model in make_models().items():
            use_o = name.endswith("oracle_features")
            log(f"seed {seed} arm {name} params {nparams(model)} flops/decision {flops_per_decision(model)}")
            acc, secs, steps = train(model, tr, ca, use_o, a.epochs, seed, log)
            model.eval()
            trained[(name, seed)] = (model, use_o)
            res["models"].setdefault(name, []).append(dict(seed=seed, params=nparams(model), calib_accuracy=acc, train_seconds=secs,
                                                           train_flops=3 * flops_per_decision(model) * steps, decision_flops=flops_per_decision(model)))
    # frozen: build test families only now
    torch.set_num_threads(1)
    families = {"in_distribution": [fm.train_instance(96000 + i) for i in range(a.indist)]}
    for kind in ("wide", "complementary", "chain"):
        families["fresh_" + kind] = [fm.fresh(95000 + i, kind) for i in range(a.fresh)]
    pools = {}
    for s_ in range(a.seeds):
        m_ = trained[("trm", s_)][0]
        with torch.no_grad():
            h = torch.zeros(ca["X"].shape[0], m_.d); bank = []; last = None
            for t in range(T):
                m_.pool_cache = last
                _, _, last, h = m_(ca["X"][:, t], ca["M"][:, t], h, ca["PS"][:, t], ca["PO"][:, t])
                bank.append(h.clone())
        pools[s_] = bank
    # sanity: the memory-threading evaluator equals the library evaluator for a memoryless policy
    ff = trained[("ff", 0)][0]
    for i in range(6):
        inst = fm.fresh(95900 + i, "wide"); st = fm.structure(inst)
        def lib_pol(orc, L, pi, b, ff=ff, st=st):
            rows, mask = fm.features(orc, L, pi, b, st)
            with torch.no_grad():
                lg, _, _, _ = ff(torch.tensor([rows], dtype=torch.float32), torch.tensor([mask]), None, None, None)
            return [(fm.action_of_slot(orc.i, int(lg.argmax(-1))), F(1))]
        o_ = g.Oracle(inst)
        assert g.evaluate(inst, o_, lib_pol) == evaluate_mem(inst, g.Oracle(inst), st, NNPolicy(ff)), "memory evaluator mismatch"
    log("memory-threading evaluator agrees with library evaluator on 6 instances")
    for fam, insts in families.items():
        rec = {k: [] for k in ("optimal", "stop", "greedy0", "greedy1", "greedy2", "direct", "random")}
        cost = {k: 0.0 for k in ("greedy1", "greedy2", "optimal_solve")}
        nn_rows = {(n, s): [] for (n, s) in trained}
        nn_rows.update({("shuffled_memory", s): [] for s in range(a.seeds)})
        nn_secs = {k: 0.0 for k in nn_rows}
        nn_calls = {k: 0 for k in nn_rows}
        for inst in insts:
            struct = fm.structure(inst)
            t0 = time.perf_counter(); o, v = g.solve(inst); cost["optimal_solve"] += time.perf_counter() - t0
            rec["optimal"].append(v); rec["stop"].append(o.terminal(o.start, inst.prior))
            for nm, pol in (("greedy0", g.policy_lookahead(0)), ("greedy1", g.policy_lookahead(1)), ("greedy2", g.policy_lookahead(2)),
                            ("direct", g.policy_direct), ("random", g.policy_random)):
                o2 = g.Oracle(inst)
                t0 = time.perf_counter()
                val = g.evaluate(inst, o2, pol)
                if nm in ("greedy1", "greedy2"):
                    cost[nm] += time.perf_counter() - t0
                rec[nm].append(val)
            for (nm, s), (model, use_o) in trained.items():
                if True:
                    pol = NNPolicy(model, oracle=use_o)
                    o3 = g.Oracle(inst)
                    val = evaluate_mem(inst, o3, struct, pol)
                    nn_rows[(nm, s)].append(val); nn_secs[(nm, s)] += pol.seconds; nn_calls[(nm, s)] += pol.calls
                    if nm == "trm":
                        polS = NNPolicy(model, pool=pools[s], seed=s)
                        vS = evaluate_mem(inst, g.Oracle(inst), struct, polS)
                        nn_rows[("shuffled_memory", s)].append(vS); nn_secs[("shuffled_memory", s)] += polS.seconds; nn_calls[("shuffled_memory", s)] += polS.calls
        # sanity: memory-threading evaluator agrees with the library evaluator for a memoryless 'policy' (greedy via NN not needed)
        out = {"instances": len(insts)}
        opt = rec["optimal"]
        reg = lambda xs: [float(o_ - x) for o_, x in zip(opt, xs)]
        for k in ("stop", "greedy0", "greedy1", "greedy2", "direct", "random"):
            out[k] = dict(mean_regret=float(np.mean(reg(rec[k]))), optimal_fraction=float(np.mean([x == o_ for x, o_ in zip(rec[k], opt)])))
        out["greedy1"]["seconds_per_instance"] = cost["greedy1"] / len(insts)
        out["greedy2"]["seconds_per_instance"] = cost["greedy2"] / len(insts)
        out["optimal_solve_seconds_per_instance"] = cost["optimal_solve"] / len(insts)
        arms = {}
        for (nm, s), vals in nn_rows.items():
            if not vals:
                continue
            arms.setdefault(nm, []).append(dict(seed=s, mean_regret=float(np.mean(reg(vals))),
                                                optimal_fraction=float(np.mean([x == o_ for x, o_ in zip(vals, opt)])),
                                                decisions=nn_calls[(nm, s)], seconds=nn_secs[(nm, s)]))
        out["neural"] = arms
        g1 = reg(rec["greedy1"])
        for nm, rows in arms.items():
            seed_vals = [np.mean(reg(nn_rows[(nm, r["seed"])])) for r in rows]
            out.setdefault("paired_vs_greedy1", {})[nm] = paired_bootstrap([np.mean([reg(nn_rows[(nm, r["seed"])])[i] for r in rows]) - g1[i] for i in range(len(insts))])
        res["test"][fam] = out
        log(f"{fam}: " + json.dumps({k: (round(v['mean_regret'], 4) if isinstance(v, dict) and 'mean_regret' in v else None) for k, v in out.items() if k in ('stop', 'greedy0', 'greedy1', 'greedy2', 'direct', 'random')})
            + " nn: " + json.dumps({nm: round(float(np.mean([r['mean_regret'] for r in rows])), 4) for nm, rows in arms.items()}))
    res["log"] = lines
    a.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

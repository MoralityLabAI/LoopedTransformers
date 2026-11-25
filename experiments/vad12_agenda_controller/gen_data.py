"""VAD-12 data: exact-oracle decision sequences from EvalMath VAD-10 instances (CPU, multiprocessing).

Set EVALMATH_VAD to the EvalMath checkout holding scripts/vad_agenda*.py. Writes train.npz / calib.npz plus a
manifest with split hashes. The fresh test families are never touched here; they are generated and scored by
train_eval.py only after the model is frozen.
"""
import argparse, hashlib, json, os, sys
from multiprocessing import Pool
from pathlib import Path
from random import Random

EV = Path(os.environ.get("EVALMATH_VAD", r"C:\Users\patri\OneDrive\Documents\GitHub\EvalMath-vad")) / "scripts"
sys.path.insert(0, str(EV))
import numpy as np
import vad_agenda as g
import vad_agenda_families as fm

T = 8


def episode(inst, o, rng, behavior):
    w = rng.choices(range(len(inst.worlds)), weights=[float(p) for p in inst.prior])[0]
    L, pi, b = o.start, inst.prior, inst.budget
    steps, prev_slot, prev_out = [], -1, 0.0
    for _ in range(T):
        acts = o.actions(L, pi, b)
        best, _ = o.best_action(L, pi, b)
        rows, mask = fm.features(o, L, pi, b, STRUCT[0])
        gain_rows, _ = fm.features(o, L, pi, b, STRUCT[0], oracle_features=True)
        qv = [0.0] * fm.N_SLOT
        stop_val = o.terminal(L, pi)
        for a in acts:
            qv[fm.slot_of_action(a)] = float(o.q(L, pi, b, a) - stop_val) / 10
        steps.append((rows, mask, fm.slot_of_action(best), qv, [r[13] for r in gain_rows], prev_slot, prev_out))
        if best == g.STOP:
            break
        if behavior == "optimal":
            a = best
        elif behavior == "greedy0":
            a = o.best_action(L, pi, b, 0)[0]
        else:
            a = rng.choice(acts[1:] or [g.STOP])
        if a == g.STOP:
            break
        br = o.branches(L, pi, b, a)
        if a[0] == "att":
            ok = a[1] in inst.worlds[w]
            br = [x for x in br if (x[2] != L) == ok] or br
            prob, _, L, pi, b = br[0]
            prev_out = float(ok)
        else:
            x = inst.experiments[a[1]]
            ys = [y for y in range(len(x.lik)) if x.lik[y][w] > 0]
            y = rng.choices(ys, weights=[float(x.lik[y][w]) for y in ys])[0]
            row = x.lik[y]
            pz = [p * row[i] for i, p in enumerate(pi)]
            z = sum(pz)
            pi = tuple(p / z for p in pz)
            b -= x.cost
            prev_out = float(y)
        prev_slot = fm.slot_of_action(a)
    return steps


STRUCT = [None]


def work(seed):
    inst = fm.train_instance(seed)
    STRUCT[0] = fm.structure(inst)
    o, _ = g.solve(inst)
    rng = Random(seed)
    out = []
    for behavior in ("optimal", "optimal", "greedy0", "random", "random", "random"):
        out.append(episode(inst, o, rng, behavior))
    return seed, out


def pack(results, path):
    n = sum(len(r[1]) for r in results)
    S, F_ = fm.N_SLOT, fm.FDIM
    X = np.zeros((n, T, S, F_), np.float32); M = np.zeros((n, T, S), bool); Y = -np.ones((n, T), np.int64)
    Q = np.zeros((n, T, S), np.float32); G = np.zeros((n, T, S), np.float32)
    PS = -np.ones((n, T), np.int64); PO = np.zeros((n, T), np.float32); V = np.zeros((n, T), bool); SD = np.zeros(n, np.int64)
    i = 0
    for seed, eps in results:
        for ep in eps:
            SD[i] = seed
            for t, (rows, mask, y, qv, gain, ps, po) in enumerate(ep):
                X[i, t] = rows; M[i, t] = mask; Y[i, t] = y; Q[i, t] = qv; G[i, t] = gain; PS[i, t] = ps; PO[i, t] = po; V[i, t] = True
            i += 1
    np.savez_compressed(path, X=X, M=M, Y=Y, Q=Q, G=G, PS=PS, PO=PO, V=V, SD=SD)
    return int(n), int(V.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--calib", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    man = {"evalmath_sources": {n: hashlib.sha256((EV / n).read_bytes()).hexdigest() for n in ("vad_agenda.py", "vad_agenda_families.py")}}
    with Pool(6) as pool:
        for name, lo, cnt in (("train", 90000, a.train), ("calib", 91000, a.calib)):
            res = pool.map(work, range(lo, lo + cnt), chunksize=4)
            seqs, steps = pack(res, a.out / f"{name}.npz")
            man[name] = {"seeds": [lo, lo + cnt - 1], "sequences": seqs, "decision_steps": steps,
                         "sha256": hashlib.sha256((a.out / f"{name}.npz").read_bytes()).hexdigest()}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))

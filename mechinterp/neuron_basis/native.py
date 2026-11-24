"""NBR engineering runner: local transported neuron writes on frozen TRM.

See PROTOCOL.md. No prospective gate is evaluated by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

STEPS = (3, 6, 12)
SEED = 20261125
EPS = 1e-8
EXPECTED_SHA = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_model():
    from mechinterp.loopscope02.scripts import trm_sudoku as trm
    return trm


def capture_tail(model, zh, zl, x, site, edit=None):
    """One last L update and one H commit. Differentiate through both.

    Add a separate zero leaf to the selected token's actual down_proj input
    on each call. Do not detach the incoming activation: later leaves must
    preserve upstream gradient paths. edit=(role, layer, neuron, delta) is
    optional and used for finite differences only.
    """
    if zh.shape[0] != 1:
        raise ValueError("capture_tail requires batch one")
    token = model.puzzle_emb_len + site
    mask = zh.new_zeros((1, zh.shape[1], 1))
    mask[:, token] = 1
    active = {"role": "L"}
    saved, handles = {}, []

    def make_hook(layer):
        def hook(_mod, args):
            a = args[0]
            key = f"{active['role']}{layer}"
            if key in saved:
                raise RuntimeError("duplicate physical call identity")
            delta = torch.zeros_like(a[0, token], requires_grad=True)
            saved[key] = (a[0, token].detach().clone(), delta)
            extra = delta
            if edit is not None and key == f"{edit[0]}{edit[1]}":
                shift = torch.zeros_like(delta)
                shift[edit[2]] = edit[3]
                extra = extra + shift
            return (a + mask * extra.view(1, 1, -1),)
        return hook

    for li, layer in enumerate(model.layers):
        handles.append(layer.down_proj.register_forward_pre_hook(make_hook(li)))
    try:
        with torch.enable_grad():
            zl = model.run_l_level(zl, zh + x)
            active["role"] = "H"
            zh = model.run_l_level(zh, zl)
            y = model.digit_logits(zh)[0, site]
            centered = y - y.mean()
            if len(saved) != 2 * len(model.layers):
                raise RuntimeError("missing native calls")
            keys = list(saved)
            leaves = [saved[k][1] for k in keys]
            jac = {k: [] for k in keys}
            for d in range(centered.numel()):
                gs = torch.autograd.grad(centered[d], leaves,
                                         retain_graph=d+1 < centered.numel())
                for k, g in zip(keys, gs):
                    jac[k].append(g.detach().cpu().numpy())
            out = {}
            for k in keys:
                a = saved[k][0].cpu().numpy()
                j = np.stack(jac[k], axis=-1)
                if not np.isfinite(a).all() or not np.isfinite(j).all():
                    raise ArithmeticError("nonfinite native neuron receipt")
                out[k] = (a, j)
            return zh.detach(), zl.detach(), y.detach().cpu().numpy(), out
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def trace(model, question, site, encode):
    """Native schedule, with explicit parity checks at captured ACTs."""
    x = model.encode_inputs(encode([question], "cpu"))
    zh, zl = model.init_states(1)
    predictions, receipts = [], {}
    for step in range(1, 17):
        if step not in STEPS:
            zh, zl = model.act_step(zh, zl, x)
        else:
            ref_h, ref_l = model.act_step(zh, zl, x)
            for hc in range(model.h_cycles):
                for lc in range(model.l_cycles):
                    last = hc == model.h_cycles - 1 and lc == model.l_cycles - 1
                    if last:
                        zh, zl, y, captures = capture_tail(model, zh, zl, x, site)
                        receipts[step] = {"y": y, "calls": captures}
                    else:
                        zl = model.run_l_level(zl, zh + x)
                if hc != model.h_cycles - 1:
                    zh = model.run_l_level(zh, zl)
            torch.testing.assert_close(zh, ref_h, rtol=0, atol=0)
            torch.testing.assert_close(zl, ref_l, rtol=0, atol=0)
        logits = model.digit_logits(zh)[0].cpu().numpy()
        if not np.isfinite(logits).all():
            raise ArithmeticError("nonfinite trajectory")
        predictions.append(logits.argmax(-1) + 1)
    return np.stack(predictions), receipts


def unit_rows(a):
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a, axis=1)
    valid = np.isfinite(a).all(1) & np.isfinite(n) & (n > EPS)
    return np.divide(a, n[:, None], out=np.zeros_like(a),
                     where=valid[:, None]), valid


def residualize(j, static):
    s, good = unit_rows(static)
    return j - np.sum(j * s, axis=1, keepdims=True) * s


def stability(a, b, write_norms, seed):
    """Board-level same-identity advantage vs norm-stratified derangements."""
    aa, va = unit_rows(a)
    bb, vb = unit_rows(b)
    both = va & vb
    rng = np.random.default_rng(seed)
    order = np.argsort(write_norms, kind="stable")
    groups = [g[both[g]] for g in np.array_split(order, 8)]
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        return None
    ids = np.concatenate(groups)
    same = float(np.sum(aa[ids] * bb[ids], axis=1).mean())
    null = []
    for _ in range(32):
        gsum = 0.0
        for group in groups:
            p = rng.permutation(group)
            q = np.roll(p, 1)
            gsum += float(np.sum(aa[p] * bb[q]))
        null.append(gsum / len(ids))
    return {"same": same, "null": float(np.mean(null)),
            "advantage": same - float(np.mean(null)),
            "coverage": len(ids) / len(a), "n_valid": len(ids)}


def auc(y, scores):
    """Exact tie-aware AUROC for scalar or matrix of candidate scores."""
    y = np.asarray(y, dtype=bool)
    x = np.asarray(scores, dtype=np.float64)
    p, n = x[y], x[~y]
    if not len(p) or not len(n):
        return None
    return ((p[:, None] > n[None, :]).mean(axis=(0, 1))
            + 0.5 * (p[:, None] == n[None, :]).mean(axis=(0, 1)))


def pick(y, features):
    a = auc(y, features)
    if a is None:
        return None
    strength = np.maximum(a, 1 - a)
    i = int(np.argmax(strength))
    return {"neuron": i, "sign": 1 if a[i] >= 0.5 else -1,
            "cal_auc": float(strength[i])}


def bootstrap_mean(v):
    if not v:
        return None
    a = np.array(v, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws = a[rng.integers(len(a), size=(2000, len(a)))].mean(1)
    return {"n": len(a), "mean": float(a.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def recovery_report(rows):
    arms = ("write", "activation")
    cal = [r for r in rows if r["split"] == "cal" and r["eligible"]]
    def labels(rs):
        return np.asarray([r["recovered"] for r in rs], dtype=bool)
    def features(rs, arm):
        return np.stack([r[arm] for r in rs])
    selected = {a: pick(labels(cal), features(cal, a)) if cal else None
                for a in arms}
    out = {"cal_n": len(cal), "cal_positive": int(labels(cal).sum()),
           "selected": selected, "families": {}}
    for split in ("eval", "rep"):
        rs = [r for r in rows if r["split"] == split and r["eligible"]]
        y = labels(rs)
        fam = {"n": len(rs), "positive": int(y.sum()), "auc": {}}
        scores = {}
        for arm in arms:
            s = selected[arm]
            if rs and s is not None:
                scores[arm] = features(rs, arm)[:, s["neuron"]] * s["sign"]
                v = auc(y, scores[arm])
                fam["auc"][arm] = None if v is None else float(v)
            else:
                fam["auc"][arm] = None
        if len(scores) == 2 and all(v is not None for v in fam["auc"].values()):
            rng = np.random.default_rng(SEED)
            ds = []
            for _ in range(2000):
                idx = rng.integers(len(rs), size=len(rs))
                if y[idx].all() or not y[idx].any():
                    continue
                ds.append(float(auc(y[idx], scores["write"][idx])
                                - auc(y[idx], scores["activation"][idx])))
            fam["delta_auc"] = fam["auc"]["write"] - fam["auc"]["activation"]
            fam["delta_ci95"] = np.quantile(ds, [0.025, 0.975]).tolist() if ds else None
            fam["bootstrap_discarded"] = 2000 - len(ds)
        out["families"][split] = fam
    return out


def load_roster(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("roster must be a nonempty list")
    ids, puzzles, solutions = set(), set(), set()
    units = [[r*9+c for c in range(9)] for r in range(9)]
    units += [[r*9+c for r in range(9)] for c in range(9)]
    units += [[(r+i)*9+c+j for i in range(3) for j in range(3)]
              for r in (0, 3, 6) for c in (0, 3, 6)]
    for r in rows:
        q, a = r["question"].replace("0", "."), r["answer"]
        if (len(q) != 81 or len(a) != 81 or not set(q) <= set(".123456789")
                or set(a) - set("123456789") or "." not in q
                or any(v != "." and v != a[i] for i, v in enumerate(q))
                or any(set(a[i] for i in u) != set("123456789") for u in units)):
            raise ValueError("invalid puzzle or completion")
        if r["split"] not in ("cal", "eval", "rep"):
            raise ValueError("invalid split")
        if r["id"] in ids or q in puzzles or a in solutions:
            raise ValueError("duplicate ID, puzzle or completion")
        ids.add(r["id"])
        puzzles.add(q)
        solutions.add(a)
        r["question"] = q
    return rows


def run(roster, out):
    rows = load_roster(roster)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    trm = native_model()
    model, digest = trm.load_trm_checkpoint("gaoxin492", "cpu", dtype=torch.float32)
    if digest != EXPECTED_SHA:
        raise RuntimeError("checkpoint mismatch")
    model.eval().requires_grad_(False)
    with torch.no_grad():
        head = model.lm_head.weight[2:11].detach().cpu().numpy().astype(np.float64)
        head -= head.mean(axis=0)
        weights = {}
        for li, layer in enumerate(model.layers):
            w = layer.down_proj.weight.detach().cpu().numpy().astype(np.float64)
            weights[li] = (np.linalg.norm(w, axis=0), w.T @ head.T)

    board_rows, detection = [], []
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        for idx, r in enumerate(rows):
            blanks = [i for i, ch in enumerate(r["question"]) if ch == "."]
            site = blanks[len(blanks)//2]
            pred, rec = trace(model, r["question"], site, trm.encode_questions)
            arrays = {"pred": pred}
            stats = {}
            for step in STEPS:
                arrays[f"s{step}_logits"] = rec[step]["y"]
                for key, (a, j) in rec[step]["calls"].items():
                    arrays[f"s{step}_{key}_a"] = a
                    arrays[f"s{step}_{key}_j"] = j
            for s, t in ((3, 6), (6, 12)):
                for key in rec[s]["calls"]:
                    norm, static = weights[int(key[1:])]
                    j1 = rec[s]["calls"][key][1].astype(np.float64)
                    j2 = rec[t]["calls"][key][1].astype(np.float64)
                    tag = f"{s}-{t}/{key}"
                    stats[tag] = {
                        "raw": stability(j1, j2, norm, SEED),
                        "residual": stability(residualize(j1, static),
                                              residualize(j2, static), norm, SEED)}
                    u, vu = unit_rows(j1)
                    v, vv = unit_rows(static)
                    good = vu & vv
                    stats[tag]["static_cos"] = float((u[good]*v[good]).sum(1).mean()) if good.any() else None

            y = rec[6]["y"]
            winner, runner = np.argsort(-y, kind="stable")[:2]
            a, j = rec[6]["calls"]["L0"]
            target = int(r["answer"][site])
            eligible = bool(pred[5, site] != target)
            recovered = bool(np.all(pred[6:, site] == target))
            detection.append({"split": r["split"], "eligible": eligible,
                              "recovered": recovered,
                              "write": a * (j[:, winner] - j[:, runner]),
                              "activation": a})
            file = out / f"board_{idx:04d}.npz"
            np.savez_compressed(file, **arrays)
            board_rows.append({"id": r["id"], "split": r["split"], "site": site,
                               "receipt": file.name, "sha256": sha(file),
                               "eligible": eligible, "recovered": recovered,
                               "stability": stats})
            print("NBR", idx, r["id"], r["split"], flush=True)

    aggregate = {}
    for split in ("cal", "eval", "rep"):
        subset = [r for r in board_rows if r["split"] == split]
        aggregate[split] = {}
        for tag in board_rows[0]["stability"]:
            aggregate[split][tag] = {}
            for kind in ("raw", "residual"):
                vals = [r["stability"][tag][kind]["advantage"]
                        for r in subset if r["stability"][tag][kind] is not None]
                aggregate[split][tag][kind] = bootstrap_mean(vals)
    report = {"schema": "NBR/native-engineering-v1", "engineering_only": True,
              "checkpoint_sha256": digest, "roster_sha256": sha(roster),
              "python": sys.version, "torch": torch.__version__,
              "numpy": np.__version__, "platform": platform.platform(),
              "processor": platform.processor(), "threads": torch.get_num_threads(),
              "seed": SEED, "boards": board_rows, "aggregate": aggregate,
              "recovery_detection": recovery_report(detection)}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                    encoding="utf-8")


def self_test():
    """Real TRM block implementation, random toy weights, no checkpoint."""
    trm = native_model()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    m = trm.PureSudokuTRM(hidden_size=16, num_heads=2, l_layers=2,
                         h_cycles=1, l_cycles=2).float().eval().requires_grad_(False)
    q = "." * 81
    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        x = m.encode_inputs(trm.encode_questions([q], "cpu"))
        h, l = m.init_states(1)
        l = m.run_l_level(l, h+x)
        nl = m.run_l_level(l, h+x)
        nh = m.run_l_level(h, nl)
        hh, ll, y, caps = capture_tail(m, h, l, x, 40)
        torch.testing.assert_close(hh, nh, rtol=0, atol=0)
        torch.testing.assert_close(ll, nl, rtol=0, atol=0)
        assert set(caps) == {"L0","L1","H0","H1"}
        for key, (_, j) in caps.items():
            i = int(np.argmax(np.linalg.norm(j, axis=1)))
            e = 0.002
            yp = capture_tail(m,h,l,x,40,(key[0],int(key[1:]),i,e))[2]
            ym = capture_tail(m,h,l,x,40,(key[0],int(key[1:]),i,-e))[2]
            fd = (yp-ym)/(2*e)
            fd -= fd.mean()
            np.testing.assert_allclose(fd, j[i], rtol=0.05, atol=0.002)
            np.testing.assert_allclose(j.sum(1), 0, atol=1e-5)
        pred, receipts = trace(m,q,40,trm.encode_questions)
        assert pred.shape == (16,81) and set(receipts) == set(STEPS)
    rng = np.random.default_rng(7)
    a = rng.normal(size=(64,9))
    st = stability(a,a,np.arange(64),SEED)
    assert st["same"] > 0.999 and st["advantage"] > 0.5
    np.testing.assert_allclose(auc([False,True],[[0,1],[1,0]]),[1,0])
    assert pick([False,True],[[0,1],[1,0]])["neuron"] == 0
    print("NBR self-test PASS (toy; no checkpoint inference)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--roster", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.self_test:
        self_test()
    elif a.roster and a.out:
        run(a.roster,a.out)
    else:
        p.error("supply --self-test or --roster and --out")


if __name__ == "__main__":
    main()

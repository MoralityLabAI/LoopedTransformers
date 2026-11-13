"""LOOPSCOPE-01 Q11 (amendment 11): the hop code as an intervention-preserving compactification.

pi = the model's own belief z_t = softmax(readout(h_t^c)); G_a = advance one hop along item a's permutation (L = 1).
rho_t = ||z_{t+1} - G_a z_t||; eps_hat = max rho_t, t = 4..7; P_hat = 1[8 eps_hat >= m(z_8)/sqrt2]; E_hat = mean P_hat.
Phase obstruction Phi on cycle items: same pi at loops t and t+c (both successors <= loop 8), successors differ.

    run:   all 24 arms -> <out>/<arm>.json
    score: registered rules -> <out>/q11_report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechinterp.loopscope01.scripts import loopscope_toy as L  # noqa: E402
from mechinterp.loopscope01.scripts import loopscope_toy_q7 as Q  # noqa: E402

OUT = Path("mechinterp/loopscope01/results/q11")
D16_SEED, D8_SEED, BOOT_SEED = 940016, 940008, 941008
N16, N8, N_BOOT, CHUNK = 4096, 2048, 200, 1024
DESYNC = ("lam0_s1", "lam0_s4", "lam0_s5", "lam0_s6", "lam0_s7")
CLOSS = ("lam0_s3", "lam0.01_s3")
S2 = math.sqrt(2)


def beliefs(model, tokens, T, device):
    """z [T, n, 16]: softmax readout of the cursor after loops 1..T."""
    out = []
    with torch.no_grad():
        for i in range(0, len(tokens), CHUNK):
            tk = tokens[i:i + CHUNK].to(device)
            hs = Q.trajectory(model, Q.embed(model, tk), T)
            out.append(torch.stack([torch.softmax(model.readout(h[:, -1]), -1) for h in hs]).float().cpu())
    return torch.cat(out, 1)


def sigma_of(tokens, n_nodes=16):
    """sigma[a, src] = dst from the edge tokens."""
    e = tokens[:, :2 * n_nodes].view(-1, n_nodes, 2)
    sig = torch.empty(len(tokens), n_nodes, dtype=torch.long)
    sig.scatter_(1, e[..., 0], e[..., 1])
    return sig


def advance(z, sig):
    """(G z)_{sigma(j)} = z_j."""
    return torch.zeros_like(z).scatter_(1, sig, z)


def margin(z):
    t2 = z.topk(2, -1).values
    return t2[..., 0] - t2[..., 1]


def cycle_len(hops):
    n, K = hops.shape
    c = torch.full((n,), K, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    for k in range(1, K):
        per = (hops[:, :K - k] == hops[:, k:]).all(1) & ~done
        c[per] = k
        done |= per
    return c


def rho_curve(z, sig):
    """rho[t-1] = ||z_{t+1} - G z_t||, t = 1..T-1  -> [T-1, n]."""
    return torch.stack([(z[t] - advance(z[t - 1], sig)).norm(dim=-1) for t in range(1, len(z))])


def e_hat(rho, m8, loops=(4, 5, 6, 7), factor=8):
    eps = rho[[t - 1 for t in loops]].max(0).values
    return (factor * eps >= m8 / S2).float(), eps


def phase_obstruction(z, cyc, t_min=1):
    pi = z[:8].argmax(-1)  # loops 1..8
    near, obs, by_t = 0, 0, {}
    for c in range(1, 7):
        idx = (cyc == c).nonzero().flatten()
        for t in range(max(1, t_min), 8 - c):  # t + c + 1 <= 8
            nf = pi[t - 1, idx] == pi[t + c - 1, idx]
            o = (pi[t, idx] != pi[t + c, idx]) & nf
            near += int(nf.sum())
            obs += int(o.sum())
            s = by_t.setdefault(t, [0, 0])
            s[0] += int(nf.sum())
            s[1] += int(o.sum())
    cand = sum(int((cyc == c).sum()) * max(0, 7 - c) for c in range(1, 7))
    phi = obs / near if near >= 200 else None
    return {"phi": phi, "near_fiber": near, "candidates": cand, "V": near / max(cand, 1),
            "phi_by_t": {t: (o / n if n else None) for t, (n, o) in by_t.items()}}


def cmd_run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok16, hops16 = L.make_batch(16, N16, D16_SEED, 16)
    tok8, hops8 = L.make_batch(16, N8, D8_SEED, 16)
    sig = sigma_of(tok16)
    assert torch.equal(torch.gather(sig, 1, hops16[:, :-1]), hops16[:, 1:]), "sigma decode"
    cyc = cycle_len(hops16)
    g = torch.Generator().manual_seed(BOOT_SEED)
    perm = torch.randperm(N16, generator=g)
    shuf = torch.empty(N16, dtype=torch.long)
    shuf[perm] = perm.roll(-1)  # derangement a -> a'
    boots = [torch.randint(0, N16, (N16,), generator=g) for _ in range(N_BOOT)]
    for s, arm, toy, _tca in Q.arms():
        dst = out / f"{arm}.json"
        if dst.exists() and not args.force:
            print(f"[skip] {arm}")
            continue
        ck = toy / "ckpt" / f"{arm}.final.pt"
        st = torch.load(ck, map_location="cpu")
        model = L.build_model(L.model_args_from(st["cfg"]))
        model.load_state_dict(st["model"])
        model.to(args.device).eval()
        z8 = beliefs(model, tok8, 8, args.device)
        unit_acc8 = float((z8[7].argmax(-1) == hops8[:, 7]).float().mean())
        z = beliefs(model, tok16, 16, args.device)
        rho = rho_curve(z, sig)
        m8 = margin(z[7])
        P, eps = e_hat(rho, m8)
        E_obs_i = (z[15].argmax(-1) != hops16[:, 15]).float()
        Eh, Eo = float(P.mean()), float(E_obs_i.mean())
        bE = np.array([(float(P[b].mean()), float(E_obs_i[b].mean())) for b in boots])
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
        rbar = rho.mean(1).numpy()
        q8 = json.loads((Q.EXP / "loopscope_toy_q8" / f"{arm}.json").read_text())
        q9 = json.loads((Q.EXP / "loopscope_toy_q9" / f"{arm}.json").read_text())
        res = {"arm": arm, "set": s, "lam": st["cfg"]["lam"], "seed": st["cfg"]["seed"], "checkpoint_sha256": L.sha256_file(ck),
               "heldout": q9["heldout"], "lock_in": q9["lock_in"], "f1_over_f8": q9["f1_over_f8"],
               "unit_acc8_d8": unit_acc8, "unit_margin8_d8": float(margin(z8[7]).mean()),
               "acc16_ondisk": q8["acc_by_loop"][15], "E_hat": Eh, "E_obs": Eo, "Vio": Eo - Eh,
               "E_hat_ci": ci(bE[:, 0]), "E_obs_ci": ci(bE[:, 1]), "Vio_ci": ci(bE[:, 1] - bE[:, 0]),
               "E_hat_boot": bE[:, 0].tolist(),
               "E_hat_T": {T: float(e_hat(rho, m8, factor=T - 8)[0].mean()) for T in range(9, 17)},
               "rho_bar": rbar.tolist(), "eps_hat_mean": float(eps.mean()), "margin8_mean": float(m8.mean()),
               "N": float(abs(np.log(rbar[6] / rbar[3]))), "N_sign": float(np.sign(rbar[6] - rbar[3])),
               "phase": phase_obstruction(z, cyc), "phase_late": phase_obstruction(z, cyc, t_min=5),
               "controls": {"E_hat_shuffled": float(e_hat(rho_curve(z, sig[shuf]), m8)[0].mean()),
                            "E_hat_h567": float(e_hat(rho, m8, loops=(5, 6, 7))[0].mean()),
                            "E_hat_h2_7": float(e_hat(rho, m8, loops=(2, 3, 4, 5, 6, 7))[0].mean())}}
        dst.write_text(json.dumps(res, indent=1))
        print(f"[{arm}] held-out={res['heldout']:.3f} E_hat={Eh:.3f} E_obs={Eo:.3f} Vio={Eo - Eh:+.3f} "
              f"phi={res['phase']['phi']} nf={res['phase']['near_fiber']} shuf={res['controls']['E_hat_shuffled']:.3f} "
              f"unit8={unit_acc8:.3f} rho={np.round(rbar, 3).tolist()}", flush=True)
    return 0


def cmd_score(args):
    out = Path(args.out)
    rows = [json.loads((out / f"{a}.json").read_text()) for _, a, _, _ in Q.arms()]
    assert len(rows) == 24, len(rows)
    by = {r["arm"]: r for r in rows}
    fails = np.array([r["heldout"] < Q.EXTRAP for r in rows])
    col = lambda f: np.array([f(r) for r in rows], float)  # noqa: E731
    Eh = col(lambda r: r["E_hat"])
    a_E = Q.auc(Eh, fails)
    boot = np.array([r["E_hat_boot"] for r in rows])
    a_boot = [Q.auc(boot[:, b], fails) for b in range(boot.shape[1])]
    aucs = {"E_hat": a_E, "E_hat_shuffled": Q.auc(col(lambda r: r["controls"]["E_hat_shuffled"]), fails),
            "neg_margin8": Q.auc(-col(lambda r: r["margin8_mean"]), fails),
            "eps_hat_mean": Q.auc(col(lambda r: r["eps_hat_mean"]), fails),
            "lock_in": Q.auc(col(lambda r: r["lock_in"]), fails), "F1/F8": Q.auc(col(lambda r: r["f1_over_f8"]), fails),
            "E_hat_h567": Q.auc(col(lambda r: r["controls"]["E_hat_h567"]), fails),
            "E_hat_h2_7": Q.auc(col(lambda r: r["controls"]["E_hat_h2_7"]), fails),
            "N": Q.auc(col(lambda r: r["N"]), fails)}
    ext = [r for r in rows if r["heldout"] >= Q.EXTRAP]
    fl = [r for r in rows if r["heldout"] < Q.EXTRAP]
    n_b = sum(r["Vio"] <= 0.05 for r in ext)
    cls = {r["arm"]: ("flagged" if r["E_hat"] >= 0.10 else "violating" if r["Vio"] > 0.10 else "neither") for r in fl}
    neither = sum(v == "neither" for v in cls.values())
    match = sum((a in DESYNC and v == "flagged") or (a in CLOSS and v == "violating") for a, v in cls.items())
    phis_ext = [r["phase"]["phi"] for r in ext if r["phase"]["phi"] is not None]
    pmax = max(phis_ext) if phis_ext else 0.0
    d_ok = sum(by[a]["phase"]["phi"] is not None and by[a]["phase"]["phi"] > pmax for a in DESYNC)
    miss = {"b": max(0, 16 - n_b), "c": neither + max(0, 6 - match), "d": len(DESYNC) - d_ok}
    e_ok = aucs["E_hat_shuffled"] <= 0.65
    if a_E >= 0.90 and all(v == 0 for v in miss.values()) and e_ok:
        dec = "supported"
    elif a_E < 0.80 or len(ext) - n_b >= 3 or not e_ok:
        dec = "not supported"
    else:
        dec = "weak"
    rep = {"decision": dec, "auc": aucs, "auc_E_hat_ci": [float(np.percentile(a_boot, 2.5)), float(np.percentile(a_boot, 97.5))],
           "b_extrap_vio_le_0.05": f"{n_b}/{len(ext)}", "c_classes": cls, "c_match_q8": match,
           "d_desync_above_ext_max": f"{d_ok}/5", "phi_ext_max": pmax, "e_shuffled_auc_ok": e_ok, "misses": miss}
    (out / "q11_report.json").write_text(json.dumps(rep, indent=1))
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    md = ["# LOOPSCOPE-01 Q11: the hop code as a compactification (amendment 11)", "",
          f"Decision: **{dec}**. AUC(E_hat) = {a_E:.3f} (95% item-bootstrap CI {rep['auc_E_hat_ci'][0]:.3f}-{rep['auc_E_hat_ci'][1]:.3f}).",
          f"(b) extrapolators with Vio <= 0.05: {rep['b_extrap_vio_le_0.05']}. (c) failure classes {cls}; matches Q8 modes {match}/7. "
          f"(d) desync arms with Phi above the extrapolator max ({pmax:.3f}): {rep['d_desync_above_ext_max']}. "
          f"(e) shuffled-operation AUC {aucs['E_hat_shuffled']:.3f} ({'ok' if e_ok else 'fails'}).", "",
          "AUC (higher = fails): " + ", ".join(f"{k} {v:.3f}" for k, v in aucs.items()) + ".", "",
          "| arm | held-out | E_hat | E_obs | Vio | Phi | near-fiber | Phi late | E_hat shuffled | margin8 | eps_hat | N | unit acc8 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["E_hat"]):
        md.append(f"| {r['arm']} ({r['set']}) | {r['heldout']:.3f} | {r['E_hat']:.3f} | {r['E_obs']:.3f} | {r['Vio']:+.3f} | "
                  f"{f3(r['phase']['phi'])} | {r['phase']['near_fiber']} | {f3(r['phase_late']['phi'])} | "
                  f"{r['controls']['E_hat_shuffled']:.3f} | {r['margin8_mean']:.3f} | {r['eps_hat_mean']:.3f} | {r['N']:.2f} | "
                  f"{r['unit_acc8_d8']:.3f} |")
    (out / "q11_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    return {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

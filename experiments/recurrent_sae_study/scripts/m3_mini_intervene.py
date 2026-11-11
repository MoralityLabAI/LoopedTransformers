"""Mini-LaCo replica, step 3: causal interventions on held-out inputs (RQ5, RQ6).

(1) Within the recovered student: ablate features selected on ana_train (answer-correct / about-to-lose) at loop t0 of
    the recurrent site, read the answer after the full T loops and after extra loops (T+3).  Controls: count-matched
    random active features; energy-matched random direction.
(2) Cross-model functional test of 'span-written' teacher features (defined on sae_train only): ablate the same SAE_T
    features in the TEACHER (at H[a+span]) and in the recovered STUDENT (at its loop-T state); if recurrence rebuilt the
    pruned computation, the downstream dependence should be comparable.  Random-feature controls for both.
All effects are paired on identical held-out inputs (fresh RNG stream, disjoint from SAE/selection splits).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import DATA, mini
from recsae.io import fig_path, res_dir, safe, sae_prefix
from recsae.sae import TopKSAE
from recsae.stats import bootstrap_ci

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pci(x, B=400):
    lo, hi = bootstrap_ci(lambda i: float(np.mean(x[i])), len(x), B=B, seed=0)
    return [float(np.mean(x)), lo, hi]


@torch.no_grad()
def edit_state(sae, x, S, mode, g, ref_norm=None):
    """Return edited last-token vector(s) and per-example removed norm / count."""
    n = sae.cfg.n_latents
    z = sae.encode(x)
    W, scale = sae.W_dec, float(sae.scale)
    tmask = torch.zeros(n, dtype=torch.bool); tmask[S] = True
    if mode == "target":
        sel = (z > 0) & tmask[None]
    elif mode == "rand_feat":
        n_i = ((z > 0) & tmask[None]).sum(1)
        cand = (z > 0) & ~tmask[None]
        r = torch.rand(z.shape, generator=g) * cand.float() - (~cand).float()
        sel = (r.argsort(1, descending=True).argsort(1) < n_i[:, None]) & cand
    delta = (z * sel.float()) @ W * scale
    xn = x - delta
    xn = xn * (x.norm(dim=1, keepdim=True) / xn.norm(dim=1, keepdim=True).clamp_min(1e-6))
    return xn, delta.norm(dim=1), sel.sum(1)


@torch.no_grad()
def rand_dir_edit(x, ref_norm, g):
    r = torch.randn(x.shape, generator=g)
    r = r / r.norm(dim=1, keepdim=True) * ref_norm[:, None]
    xn = x - r
    return xn * (x.norm(dim=1, keepdim=True) / xn.norm(dim=1, keepdim=True).clamp_min(1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="dfa")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    T = 5
    build = json.load(open(os.path.join(res_dir(), f"mini_s{a.seed}__build.json")))
    anchor, span = build["anchor"], build["span"]
    name = f"mini:{a.task}_s{a.seed}:rec"
    st, teacher, task, _ = mini.load_student(a.seed, a.task, "rec")
    sae_s = TopKSAE.load(sae_prefix(name, "all"))
    sae_t = TopKSAE.load(sae_prefix(f"mini:{a.task}_s{a.seed}:teacher", "span"))
    ev = np.load(os.path.join(DATA, f"{safe(name)}__event_auc.npz"))
    # ---- span-written features of SAE_T (from sae_train only)
    z = np.load(os.path.join(mini.mdir(a.seed, a.task), "acts_sae_train.npz"))
    with torch.no_grad():
        c_in = sae_t.encode(torch.from_numpy(z["t_last"][:, 0].astype(np.float32))).numpy()
        c_out = sae_t.encode(torch.from_numpy(z["t_last"][:, span].astype(np.float32))).numpy()
    eff = np.abs((c_out - c_in).mean(0)) / (np.sqrt((c_out ** 2).mean(0) + (c_in ** 2).mean(0)) + 1e-6)
    freq = (c_out > 0).mean(0)
    elig = np.where(freq > 0.02)[0]
    written = elig[np.argsort(-eff[elig])]
    freq_s = None
    R = {"seed": a.seed, "T": T, "anchor": anchor, "span": span}
    rng = np.random.default_rng(9_000_000 + a.seed)
    sets = {"id": ([8, 12, 16], a.n), "ood": ([24], a.n)}
    # feature sets for (1)
    zc = np.load(os.path.join(mini.mdir(a.seed, a.task), "acts_ana_train.npz"))
    Xm = torch.from_numpy(zc["rec__last"][:, 2:6].reshape(-1, zc["rec__last"].shape[-1]).astype(np.float32))
    with torch.no_grad():
        Zm = sae_s.encode(Xm)
    ok = (Zm > 0).float().mean(0).numpy() > 0.003
    fsets = {"answer_feats": np.argsort(-np.where(ok, ev["now"] - 0.5, -1))[:8], "loss_feats": np.argsort(-np.where(ok, ev["harm"] - 0.5, -1))[:8]}
    R["feature_sets"] = {k: v.tolist() for k, v in fsets.items()}
    for sp, (lens, n) in sets.items():
        R[sp] = {}
        xs, ys = [], []
        for L in lens:
            x, y = task.sample(rng, n, L)
            xs.append(torch.from_numpy(x)); ys.append(y[:, -1])
        ids = torch.cat([x for x in xs]) if len(set(lens)) == 1 else None
        for Tr in (T, T + 3):
            pass
        def run(hook_fn, t0, Tr):
            out = []
            for x in xs:
                with torch.no_grad():
                    hh, _, _ = mini.student_forward(st, x, T=Tr, hook=hook_fn, collect_site=False)
                    out.append((st.logits(hh)[:, -1].argmax(-1) - task.n_sym).numpy())
            return np.concatenate(out)
        y_all = np.concatenate(ys)
        clean = {Tr: run(None, 0, Tr) == y_all for Tr in (T, T + 3)}
        R[sp]["clean_acc"] = {str(k): float(v.mean()) for k, v in clean.items()}
        rows = []
        for sname, S_all in fsets.items():
            for t0 in (2, 3, 4):
                for m in (2, 4, 8):
                    S = torch.tensor(S_all[:m], dtype=torch.long)
                    res = {"set": sname, "t0": t0, "m": m}
                    rm_all = []
                    arms = {}
                    for arm in ("target", "rand_feat", "rand_dir"):
                        g = torch.Generator().manual_seed(7)
                        preds = {Tr: [] for Tr in (T, T + 3)}
                        edited = []
                        pos = 0
                        for x in xs:
                            for Tr in (T, T + 3):
                                g2 = torch.Generator().manual_seed(7)
                                state = {}

                                def hook(k, h, arm=arm, g2=g2, state=state, x=x):
                                    if k != t0:
                                        return h
                                    xx = h[:, -1]
                                    if arm == "rand_dir":
                                        # energy-matched: norm of the target arm's removal
                                        _, rn, _ = edit_state(sae_s, xx, S, "target", g2)
                                        xn = rand_dir_edit(xx, rn, g2)
                                        state["n"] = None
                                    else:
                                        xn, rn, nr = edit_state(sae_s, xx, S, arm, g2)
                                        state["n"] = nr
                                    h = h.clone(); h[:, -1] = xn
                                    return h
                                with torch.no_grad():
                                    hh, _, _ = mini.student_forward(st, x, T=Tr, hook=hook, collect_site=False)
                                preds[Tr].append((st.logits(hh)[:, -1].argmax(-1) - task.n_sym).numpy())
                                if arm == "target" and Tr == T:
                                    edited.append((state["n"] > 0).numpy())
                        for Tr in (T, T + 3):
                            arms[(arm, Tr)] = np.concatenate(preds[Tr]) == y_all
                        if arm == "target":
                            res["frac_edited"] = float(np.concatenate(edited).mean())
                    for Tr in (T, T + 3):
                        for arm in ("target", "rand_feat", "rand_dir"):
                            res[f"{arm}_d_T{Tr}"] = pci(arms[(arm, Tr)].astype(float) - clean[Tr].astype(float))
                        res[f"target_minus_randfeat_T{Tr}"] = pci(arms[("target", Tr)].astype(float) - arms[("rand_feat", Tr)].astype(float))
                        res[f"target_minus_randdir_T{Tr}"] = pci(arms[("target", Tr)].astype(float) - arms[("rand_dir", Tr)].astype(float))
                    rows.append(res)
                    if m == 8:
                        print(f"[{sp}] {sname} t0={t0} m=8 edited={res['frac_edited']:.2f} dT: tgt={res[f'target_d_T{T}'][0]:+.3f} rf={res[f'rand_feat_d_T{T}'][0]:+.3f} rd={res[f'rand_dir_d_T{T}'][0]:+.3f}")
        R[sp]["rows"] = rows
        # ---- (2) cross-model: ablate span-written teacher features in teacher vs student
        cross = []
        for m in (4, 8, 16):
            W = torch.tensor(written[:m], dtype=torch.long)
            row = {"m": m, "n_written_eligible": int(len(elig))}
            for who in ("teacher", "student"):
                accs = {}
                for arm in ("clean", "target", "rand_feat", "rand_dir"):
                    g2 = torch.Generator().manual_seed(11)
                    preds = []
                    for x in xs:
                        def edit(xx, arm=arm):
                            if arm == "clean":
                                return xx
                            if arm == "rand_dir":
                                _, rn, _ = edit_state(sae_t, xx, W, "target", g2)
                                return rand_dir_edit(xx, rn, g2)
                            xn, _, _ = edit_state(sae_t, xx, W, arm, g2)
                            return xn
                        if who == "teacher":
                            def fn(h, edit=edit):
                                h = h.clone(); h[:, -1] = edit(h[:, -1]); return h
                            with torch.no_grad():
                                hh = mini.teacher_forward(teacher, x, hook_depth=anchor + span, fn=fn)
                                lg = teacher.logits(hh)[:, -1]
                        else:
                            def hook(k, h, edit=edit):
                                if k != T:
                                    return h
                                h = h.clone(); h[:, -1] = edit(h[:, -1]); return h
                            with torch.no_grad():
                                hh, _, _ = mini.student_forward(st, x, T=T, hook=hook, collect_site=False)
                                lg = st.logits(hh)[:, -1]
                        preds.append((lg.argmax(-1) - task.n_sym).numpy())
                    accs[arm] = np.concatenate(preds) == y_all
                for arm in ("target", "rand_feat", "rand_dir"):
                    row[f"{who}_{arm}_d"] = pci(accs[arm].astype(float) - accs["clean"].astype(float))
                row[f"{who}_clean_acc"] = float(accs["clean"].mean())
            cross.append(row)
            print(f"[{sp}] cross m={m}: teacher d tgt={row['teacher_target_d'][0]:+.3f} rf={row['teacher_rand_feat_d'][0]:+.3f} | student d tgt={row['student_target_d'][0]:+.3f} rf={row['student_rand_feat_d'][0]:+.3f}")
        R[sp]["cross_model"] = cross
    json.dump(R, open(os.path.join(res_dir(), f"mini_s{a.seed}__interventions.json"), "w"), indent=1)
    plot(R, a.seed)


def plot(R, seed):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sp in zip(axs, ("id", "ood")):
        cr = R[sp]["cross_model"]
        ms = [c["m"] for c in cr]
        for who, col in (("teacher", "tab:blue"), ("student", "tab:red")):
            ax.plot(ms, [c[f"{who}_target_d"][0] for c in cr], c=col, marker="o", label=f"{who}: span-written features")
            ax.plot(ms, [c[f"{who}_rand_feat_d"][0] for c in cr], c=col, ls="--", marker="x", label=f"{who}: random features")
            ax.plot(ms, [c[f"{who}_rand_dir_d"][0] for c in cr], c=col, ls=":", marker="s", ms=3, label=f"{who}: random direction")
        ax.axhline(0, c="k", lw=0.5); ax.set_title(f"{sp}: ablate SAE_T span-written features"); ax.set_xlabel("# features"); ax.set_ylabel("Δ accuracy")
    axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig_path(f"mini_s{seed}", "m3_cross_ablation"), dpi=130); plt.close()


if __name__ == "__main__":
    main()

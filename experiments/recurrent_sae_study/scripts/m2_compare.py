"""Mini-LaCo replica, step 2: do pruned layers contain features the recurrent adapters reconstruct? (RQ2, RQ5)

Teacher SAE (SAE_T) is trained on teacher residual states across the span that LaCo removes, using ONLY `sae_train`.
Everything below is evaluated on held-out `test_id` / `test_ood` last-token states of identical inputs:
  A. SAE_T reconstruction error on student states vs matched teacher depth          (feature-basis fit)
  B. SAE_T code agreement teacher-output vs student-output (cosine, Jaccard, per-feature correlation) with controls
  C. 'span-written' features: those whose activation changes across the removed span - reproduced by the student?
  D. loop <-> teacher-depth alignment (code cosine and CKA heatmaps), principal angles
  E. representational similarity vs functional equivalence (CKA/angles/code agreement vs top-1 agreement with teacher)
  F. dictionary-atom matching between SAE_T and a SAE trained on the student's own loop states
  G. functional test: replace the site-output last-token state with its SAE_T reconstruction
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
from recsae.analysis import dense_codes, encode_loops
from recsae.io import fig_path, res_dir, sae_deliver_prefix, sae_prefix
from recsae.sae import SAEConfig, TopKSAE, recon_stats, train_sae
from recsae.stats import auroc_cols, linear_cka, principal_angles, subspace_overlap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["laco", "rec0", "rec"]


def load_acts(tag, split):
    return dict(np.load(os.path.join(recsae.DATA, "mini", tag, f"acts_{split}.npz")))


def teacher_sae(tag, seed, span, args):
    pref = sae_prefix(f"mini:{tag}:teacher", "span")
    if os.path.exists(pref + ".pt"):
        return TopKSAE.load(pref)
    z = load_acts(tag, "sae_train")
    zv = load_acts(tag, "sae_val")
    d = z["t_last"].shape[-1]
    X = np.concatenate([z["t_last"].reshape(-1, d), z["t_pos"].reshape(-1, d)]).astype(np.float32)
    Xv = np.concatenate([zv["t_last"].reshape(-1, d), zv["t_pos"].reshape(-1, d)]).astype(np.float32)
    cfg = SAEConfig(d_in=d, n_latents=args.n_latents, k=args.k, seed=seed, steps=args.steps,
                    train_tag=f"teacher states H[a..a+{span}] last+random positions (sae_train)")
    sae, hist = train_sae(torch.from_numpy(X), cfg, torch.from_numpy(Xv))
    sae.save(pref, extra={"history": hist})
    sae.save(sae_deliver_prefix(f"mini:{tag}:teacher", "span"), extra={"history": hist})
    return sae


@torch.no_grad()
def codes(sae, X):
    out = []
    for i in range(0, X.shape[0], 20000):
        out.append(sae.encode(torch.from_numpy(X[i:i + 20000].astype(np.float32))).numpy())
    return np.concatenate(out)


def code_cos(A, B):
    return (A * B).sum(1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-9)


def jacc(A, B):
    a, b = A > 0, B > 0
    return (a & b).sum(1) / np.maximum((a | b).sum(1), 1)


def feat_corr(A, B, min_freq=0.02):
    """Pearson correlation of each feature's activation across examples between code sets A and B."""
    fa, fb = (A > 0).mean(0), (B > 0).mean(0)
    keep = np.where((fa >= min_freq) | (fb >= min_freq))[0]
    a, b = A[:, keep] - A[:, keep].mean(0), B[:, keep] - B[:, keep].mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    r = (a * b).sum(0) / np.maximum(den, 1e-9)
    return keep, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="dfa")
    ap.add_argument("--tag", default=None, help="run tag under DATA/mini (default <task>_s<seed>); Qwen runs use their q1 tag")
    ap.add_argument("--n_latents", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tag = a.tag or f"{a.task}_s{a.seed}"
    qwen = a.tag is not None
    if qwen:
        meta = json.load(open(os.path.join(recsae.DATA, "mini", tag, "meta.json")))
        anchor, span, T = meta["anchor"], meta["span"], meta["T"]
        class _N:  # class tokens are 0..10 already in the q1 layout
            n_sym = 0
        task = _N()
    else:
        build = json.load(open(os.path.join(res_dir(), f"mini_s{a.seed}__build.json")))
        anchor, span = build["anchor"], build["span"]
        T = 5
        teacher, task, _ = mini.load_teacher(a.seed, a.task)
    sae_t = teacher_sae(tag, a.seed, span, a)
    n = sae_t.cfg.n_latents
    R = {"seed": a.seed, "tag": tag, "anchor": anchor, "span": span, "T": T}
    ztr = load_acts(tag, "sae_train")
    c_in_tr, c_out_tr = codes(sae_t, ztr["t_last"][:, 0].astype(np.float32)), codes(sae_t, ztr["t_last"][:, span].astype(np.float32))
    eff_tr = np.abs((c_out_tr - c_in_tr).mean(0)) / (np.sqrt((c_out_tr ** 2).mean(0) + (c_in_tr ** 2).mean(0)) + 1e-6)
    elig_tr = np.where((c_out_tr > 0).mean(0) > 0.02)[0]
    top_global = elig_tr[np.argsort(-eff_tr[elig_tr])[:max(10, len(elig_tr) // 5)]]
    rest_global = np.setdiff1d(elig_tr, top_global)
    for sp in ("test_id", "test_ood"):
        z = load_acts(tag, sp)
        y = z["y"]
        N = len(y)
        Ht = z["t_last"].astype(np.float32)                                   # [N, span+1, d]  depth j = 0..span
        Ct = [codes(sae_t, Ht[:, j]) for j in range(span + 1)]
        tl = z["t_logits"].argmax(-1)
        res = {}
        # ---- A. reconstruction NMSE of SAE_T on teacher depths and student loops
        A = {"teacher_depth": [recon_stats(sae_t, torch.from_numpy(Ht[:, j]))["nmse"] for j in range(span + 1)]}
        for v in VARIANTS:
            S = z[f"{v}__last"].astype(np.float32)
            A[v] = [recon_stats(sae_t, torch.from_numpy(S[:, t]), ref_mu=None)["nmse"] for t in range(S.shape[1])]
        res["A_nmse_by_state"] = A
        # ---- B. code agreement of student output with teacher output state H[a+span]
        out_t = Ct[span]
        rng = np.random.default_rng(0)
        grp = (y * 100 + z["N"]).astype(int)
        perm = np.arange(N)
        for g in np.unique(grp):
            m = np.where(grp == g)[0]
            perm[m] = rng.permutation(m)
        B = {"null_label_matched": {"cos": float(code_cos(out_t, out_t[perm]).mean()), "jacc": float(jacc(out_t, out_t[perm]).mean())},
             "adjacent_teacher_layer": {"cos": float(code_cos(out_t, Ct[span - 1]).mean()), "jacc": float(jacc(out_t, Ct[span - 1]).mean())},
             "teacher_input_state": {"cos": float(code_cos(out_t, Ct[0]).mean()), "jacc": float(jacc(out_t, Ct[0]).mean())}}
        keepA, rA = feat_corr(out_t, Ct[span - 1])
        B["adjacent_teacher_layer"]["feat_corr_mean"] = float(rA.mean())
        for v in VARIANTS:
            S = z[f"{v}__last"].astype(np.float32)
            by_t = []
            for t in range(1, T + 6 if S.shape[1] > T + 5 else S.shape[1]):
                Cs = codes(sae_t, S[:, t])
                keep, r = feat_corr(out_t, Cs)
                by_t.append({"t": t, "cos": float(code_cos(out_t, Cs).mean()), "jacc": float(jacc(out_t, Cs).mean()),
                             "feat_corr_mean": float(r.mean()), "feat_corr_gt0.5": float((r > 0.5).mean())})
            B[v] = by_t
        res["B_output_agreement"] = B
        # ---- C. span-written features
        top, rest = top_global, rest_global           # span-written features were chosen on sae_train only
        Cc = {}
        for v in VARIANTS:
            S = z[f"{v}__last"].astype(np.float32)
            t_last = 1 if v == "laco" else T
            Cs = codes(sae_t, S[:, t_last])
            keepv, rv = feat_corr(out_t, Cs, min_freq=0.0)
            rr = np.zeros(n)
            rr[keepv] = rv
            Cc[v] = {"written_corr": float(rr[top].mean()), "other_corr": float(rr[rest].mean()), "n_written": int(len(top)), "n_other": int(len(rest))}
        # teacher adjacent layer reference
        keepv, rv = feat_corr(out_t, Ct[span - 1], min_freq=0.0)
        rr = np.zeros(n); rr[keepv] = rv
        Cc["teacher_adjacent_layer"] = {"written_corr": float(rr[top].mean()), "other_corr": float(rr[rest].mean())}
        res["C_span_written"] = Cc
        # ---- D. loop <-> teacher depth alignment
        heat_cos = np.zeros((T + 1, span + 1))
        heat_cka = np.zeros((T + 1, span + 1))
        S = z["rec__last"].astype(np.float32)
        for t in range(T + 1):
            Cs = codes(sae_t, S[:, t])
            for j in range(span + 1):
                heat_cos[t, j] = code_cos(Cs, Ct[j]).mean()
                heat_cka[t, j] = linear_cka(S[:, t], Ht[:, j])
        res["D_heat_cos"], res["D_heat_cka"] = heat_cos.tolist(), heat_cka.tolist()
        k = 8
        res["D_principal_angles_deg_rec_T_vs_teacher_out"] = principal_angles(S[:, T], Ht[:, span], k).tolist()
        res["D_subspace_overlap"] = {v: subspace_overlap(z[f"{v}__last"][:, (1 if v == "laco" else T)].astype(np.float32), Ht[:, span], k) for v in VARIANTS}
        res["D_subspace_overlap"]["teacher_adjacent"] = subspace_overlap(Ht[:, span - 1], Ht[:, span], k)
        # ---- E. representation vs function
        E = []
        for v in ("rec0", "rec"):
            Sv = z[f"{v}__last"].astype(np.float32)
            lg = z[f"{v}__logits"].argmax(-1)
            for t in range(1, Sv.shape[1]):
                E.append({"variant": v, "t": t, "cka_to_teacher_out": linear_cka(Sv[:, t], Ht[:, span]),
                          "angle_overlap_k8": subspace_overlap(Sv[:, t], Ht[:, span], k),
                          "code_cos": float(code_cos(codes(sae_t, Sv[:, t]), out_t).mean()),
                          "top1_agree_teacher": float((lg[:, t - 1] == tl).mean()),
                          "acc": float((lg[:, t - 1] == y).mean())})
        lg = z["laco__logits"].argmax(-1)
        Sl = z["laco__last"].astype(np.float32)
        E.append({"variant": "laco", "t": 1, "cka_to_teacher_out": linear_cka(Sl[:, 1], Ht[:, span]),
                  "angle_overlap_k8": subspace_overlap(Sl[:, 1], Ht[:, span], k),
                  "code_cos": float(code_cos(codes(sae_t, Sl[:, 1]), out_t).mean()),
                  "top1_agree_teacher": float((lg[:, 0] == tl).mean()), "acc": float((lg[:, 0] == y).mean())})
        res["E_repr_vs_function"] = E
        res["teacher_acc"] = float((tl == y).mean())
        R[sp] = res
        print(f"[{sp}] teacher acc {res['teacher_acc']:.3f}; rec@T5 acc {[e['acc'] for e in E if e['variant']=='rec' and e['t']==5]}, "
              f"laco acc {E[-1]['acc']:.3f}; code-cos(out): rec={B['rec'][T-1]['cos']:.3f} rec0={B['rec0'][T-1]['cos']:.3f} "
              f"laco={B['laco'][0]['cos']:.3f} null={B['null_label_matched']['cos']:.3f} adj-layer={B['adjacent_teacher_layer']['cos']:.3f}")
    # ---- G functional: SAE_T-reconstruction substitution at site output (test_id subset)
    G = {}
    rng = np.random.default_rng(5)
    for sp, lens in (() if qwen else (("test_id", [8, 16]), ("test_ood", [24]))):
        G[sp] = {}
        for v in ("rec", "rec0"):
            st, _, task_, _ = mini.load_student(a.seed, a.task, v)
            for mode in ("clean", "sae_t_recon", "zero_top_written"):
                accs = []
                for L in lens:
                    x, y = task_.sample(rng, 400, L)
                    ids = torch.from_numpy(x)

                    def hook(k, h, mode=mode):
                        if k != T or mode == "clean":
                            return h
                        xx = h[:, -1]
                        z_ = sae_t.encode(xx)
                        if mode == "zero_top_written":
                            z_[:, top_idx] = 0
                        h = h.clone(); h[:, -1] = sae_t.decode(z_) + 0.0
                        return h
                    top_idx = torch.from_numpy(top_global)
                    with torch.no_grad():
                        hh, _, _ = mini.student_forward(st, ids, T=T, hook=hook, collect_site=False)
                        pr = st.logits(hh)[:, -1].argmax(-1) - task_.n_sym
                    accs.append(float((pr.numpy() == y[:, -1]).mean()))
                G[sp][f"{v}/{mode}"] = float(np.mean(accs))
        print(f"[G {sp}]", G[sp])
    R["G_functional"] = G
    json.dump(R, open(os.path.join(res_dir(), f"{tag}__compare.json" if qwen else f"mini_s{a.seed}__compare.json"), "w"), indent=1)
    plot(R, tag if qwen else f"mini_s{a.seed}", T, span)


def plot(R, name, T, span):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for r, sp in enumerate(("test_id", "test_ood")):
        res = R[sp]
        ax = axs[r, 0]
        im = ax.imshow(np.array(res["D_heat_cos"]), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("teacher depth j (H[a+j])"); ax.set_ylabel("recovered loop t"); ax.set_title(f"{sp}: SAE_T code cosine")
        plt.colorbar(im, ax=ax)
        ax = axs[r, 1]
        for v, c in (("rec", "tab:red"), ("rec0", "tab:blue")):
            ax.plot(range(1, T + 6), [b["cos"] for b in res["B_output_agreement"][v]], c=c, marker="o", label=f"{v} code-cos")
        b = res["B_output_agreement"]
        ax.axhline(b["null_label_matched"]["cos"], c="gray", ls=":", label="label-matched null")
        ax.axhline(b["adjacent_teacher_layer"]["cos"], c="k", ls="--", label="teacher adjacent layer")
        ax.axhline(b["laco"][0]["cos"], c="tab:green", ls="-.", label="LaCo only")
        ax.set_xlabel("loop t"); ax.set_title("code agreement with teacher output"); ax.legend(fontsize=7)
        ax = axs[r, 2]
        E = res["E_repr_vs_function"]
        for v, c, m_ in (("rec", "tab:red", "o"), ("rec0", "tab:blue", "s"), ("laco", "tab:green", "^")):
            xs = [e["cka_to_teacher_out"] for e in E if e["variant"] == v]
            ys = [e["top1_agree_teacher"] for e in E if e["variant"] == v]
            ax.scatter(xs, ys, c=c, marker=m_, label=v)
        ax.set_xlabel("linear CKA to teacher output"); ax.set_ylabel("top-1 agreement with teacher"); ax.set_title("similarity vs function"); ax.legend()
    plt.tight_layout(); plt.savefig(fig_path(name, "m2_compare"), dpi=130); plt.close()


if __name__ == "__main__":
    main()

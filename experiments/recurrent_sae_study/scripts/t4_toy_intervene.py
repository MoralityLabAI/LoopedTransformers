"""Toy path step 4: controlled activation interventions on HELD-OUT examples (RQ3/RQ5).

Features are selected on `ana_train` (loop-binned AUROC saved by t3); effects are measured on test_id / test_ood.
Arms per intervention (paired on identical inputs):
  target   : edit the selected SAE features (those active in the example)
  rand_feat: edit the same NUMBER of other active features per example, drawn at random (matched dose in count)
  rand_dir : subtract a random-direction vector with the SAME norm as the target edit (matched dose in energy)
Edit is applied to the last-token residual state right after loop t0 and the state is renormalised to its original norm,
then the model keeps looping to K.  We read answers after every later loop.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import DATA, toy
from recsae.io import fig_path, load_split, res_dir, safe, sae_prefix
from recsae.sae import TopKSAE
from recsae.stats import bootstrap_ci

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def paired_ci(delta, B=400, seed=0):
    lo, hi = bootstrap_ci(lambda i: float(np.mean(delta[i])), len(delta), B=B, seed=seed)
    return float(np.mean(delta)), lo, hi


class Editor:
    def __init__(self, sae, t0, mode, S=None, m=4, alpha=0.0, target_dir=None, seed=0):
        self.sae, self.t0, self.mode, self.S, self.m, self.alpha = sae, t0, mode, S, m, alpha
        self.target_dir = target_dir
        self.g = torch.Generator().manual_seed(seed)
        self.removed_norm = None

    @torch.no_grad()
    def __call__(self, k, h):
        if k != self.t0:
            return h
        x = h[:, -1]
        n = self.sae.cfg.n_latents
        z = self.sae.encode(x)                                                    # [B, n]
        scale = float(self.sae.scale)
        W = self.sae.W_dec
        if self.mode == "target":
            mask = torch.zeros(n, dtype=torch.bool)
            mask[self.S] = True
            sel = (z > 0) & mask[None]
        elif self.mode == "rand_feat":
            tmask = torch.zeros(n, dtype=torch.bool)
            tmask[self.S] = True
            n_i = ((z > 0) & tmask[None]).sum(1)                                  # same count as the target arm
            cand = (z > 0) & ~tmask[None]
            r = torch.rand(z.shape, generator=self.g) * cand.float() - (~cand).float()
            order = r.argsort(1, descending=True)
            ranks = order.argsort(1)
            sel = (ranks < n_i[:, None]) & cand
        else:
            raise ValueError(self.mode)
        delta = (z * sel.float()) @ W * scale                                      # energy removed from the state
        self.removed_norm = delta.norm(dim=1)
        xn = x - delta
        xn = xn * (x.norm(dim=1, keepdim=True) / xn.norm(dim=1, keepdim=True).clamp_min(1e-6))
        h = h.clone()
        h[:, -1] = xn
        self._n_removed = sel.sum(1)
        return h


class RandDirEditor:
    """Subtract a random-direction vector whose norm equals a reference per-example norm."""

    def __init__(self, t0, ref_norm, seed=0):
        self.t0, self.ref, self.g = t0, ref_norm, torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def __call__(self, k, h):
        if k != self.t0:
            return h
        x = h[:, -1]
        r = torch.randn(x.shape, generator=self.g)
        r = r / r.norm(dim=1, keepdim=True) * self.ref[:, None]
        xn = x - r
        xn = xn * (x.norm(dim=1, keepdim=True) / xn.norm(dim=1, keepdim=True).clamp_min(1e-6))
        h = h.clone()
        h[:, -1] = xn
        return h


class SteerEditor:
    """Add `alpha` x (summed decoder directions of class-writing features S, scaled by their mean activation)."""

    def __init__(self, sae, t0, S, mean_act, alpha, seed=0, random_dir=False):
        self.sae, self.t0, self.alpha = sae, t0, alpha
        v = (sae.W_dec[S] * mean_act[S][:, None]).sum(0) * float(sae.scale)
        if random_dir:
            g = torch.Generator().manual_seed(seed)
            r = torch.randn(v.shape, generator=g)
            v = r / r.norm() * v.norm()
        self.v = v

    @torch.no_grad()
    def __call__(self, k, h):
        if k != self.t0:
            return h
        x = h[:, -1]
        xn = x + self.alpha * self.v[None]
        xn = xn * (x.norm(dim=1, keepdim=True) / xn.norm(dim=1, keepdim=True).clamp_min(1e-6))
        h = h.clone()
        h[:, -1] = xn
        return h


@torch.no_grad()
def run_edit(model, split, K, hook_factory, bs=500, subset=None):
    """hook_factory(batch_index)->hook.  Returns logits [N, K, C] for the (sub)split in split order."""
    out = []
    off = 0
    for x in split["x"]:
        for i in range(0, x.shape[0], bs):
            xb = torch.from_numpy(x[i:i + bs])
            hook = hook_factory(off + i, xb)
            r = toy.rollout(model, xb, K, hook=hook)
            out.append(r["logits"].cpu().numpy())
        off += x.shape[0]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--t0s", default="4,8,16,24")
    ap.add_argument("--m", default="1,2,4,8")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--n_eval", type=int, default=600, help="examples per length in each test split")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tag = a.tag or safe(a.name)
    model, task, ck = toy.load_toy(a.name)
    sae = TopKSAE.load(sae_prefix(a.name, "all"))
    ev = np.load(os.path.join(DATA, f"{safe(a.name)}__event_auc.npz"))
    auc_now, auc_harm = ev["now"], ev["harm"]
    n = sae.cfg.n_latents
    K = 32
    t0s = [int(t) for t in a.t0s.split(",")]
    ms = [int(m) for m in a.m.split(",")]
    # test splits as lists of per-length arrays, reconstructed from the saved data (x was not stored: regenerate)
    from recsae.io import SPLITS
    results = {"name": a.name, "t0s": t0s, "ms": ms, "arms": {}}
    # mean activation per feature (for steering strength) from ana_train at mid loops
    tr = load_split(a.name, "ana_train")
    Xmid = torch.from_numpy(tr["last"][:, 8:20].reshape(-1, tr["last"].shape[-1]).astype(np.float32))
    with torch.no_grad():
        Zm = sae.encode(Xmid)
    mean_act = torch.where((Zm > 0).any(0), (Zm.sum(0) / (Zm > 0).sum(0).clamp_min(1)), torch.zeros(n))
    freq = (Zm > 0).float().mean(0).numpy()
    ok = freq > 0.003
    # class written by each feature
    W_dec = sae.W_dec.detach() * float(sae.scale)
    dla = ((W_dec * model.ln_f.weight.detach()) @ model.head.weight.detach().t()).numpy()
    dla_c = dla - dla.mean(1, keepdims=True)
    for sp in ("test_id", "test_ood"):
        lengths, npl, off = SPLITS[sp]
        rng = np.random.default_rng(1_000_000 * (ck["seed"] + 1) + off)          # same stream as the collected split
        split = toy.make_split(task, rng, lengths, npl)
        sub = {"x": [x[: a.n_eval] for x in split["x"]]}
        keep = np.concatenate([np.arange(len(x))[: a.n_eval] + o for o, x in zip(np.cumsum([0] + [len(x) for x in split["x"]])[:-1], split["x"])])
        y = split["y"][keep]
        Nn = split["N"][keep]
        clean = run_edit(model, sub, K, lambda i, xb: None)
        c_clean = clean.argmax(-1) == y[:, None]
        results["arms"][sp] = {"clean_acc_by_loop": c_clean.mean(0).tolist(), "n": int(len(y)), "rows": []}
        print(f"[{sp}] n={len(y)} clean acc@8={c_clean[:, 7].mean():.3f} @16={c_clean[:, 15].mean():.3f} @32={c_clean[:, 31].mean():.3f}")
        # ------- feature sets selected on ana_train only
        sets = {
            "answer_feats": np.argsort(-np.where(ok, auc_now - 0.5, -1))[: max(ms)],       # most 'answer-correct' features
            "loss_feats": np.argsort(-np.where(ok, auc_harm - 0.5, -1))[: max(ms)],         # most 'about-to-lose' features
        }
        for sname, S_all in sets.items():
            for t0 in t0s:
                for m in ms:
                    S = torch.tensor(S_all[:m], dtype=torch.long)
                    ed = Editor(sae, t0, "target", S=S)
                    norms = []

                    def fac_t(i, xb, ed=ed, norms=norms):
                        return ed
                    # target arm (also records removed norm per example for matching)
                    removed = []

                    class Rec(Editor):
                        pass
                    tg_logits, rm_list, nrem = [], [], []
                    off_i = 0
                    for xs in sub["x"]:
                        for i in range(0, len(xs), 500):
                            xb = torch.from_numpy(xs[i:i + 500])
                            e1 = Editor(sae, t0, "target", S=S)
                            r1 = toy.rollout(model, xb, K, hook=e1)
                            tg_logits.append(r1["logits"].numpy()); rm_list.append(e1.removed_norm.clone()); nrem.append(e1._n_removed.clone())
                    tg_logits = np.concatenate(tg_logits); rm = torch.cat(rm_list); nr = torch.cat(nrem)
                    # random-feature arm (count-matched) and random-direction arm (energy-matched)
                    rf_logits, rd_logits = [], []
                    pos = 0
                    for xs in sub["x"]:
                        for i in range(0, len(xs), 500):
                            xb = torch.from_numpy(xs[i:i + 500])
                            e2 = Editor(sae, t0, "rand_feat", S=S, seed=11)
                            rf_logits.append(toy.rollout(model, xb, K, hook=e2)["logits"].numpy())
                            e3 = RandDirEditor(t0, rm[pos:pos + len(xb)], seed=13)
                            rd_logits.append(toy.rollout(model, xb, K, hook=e3)["logits"].numpy())
                            pos += len(xb)
                    rf_logits = np.concatenate(rf_logits); rd_logits = np.concatenate(rd_logits)
                    c_t, c_rf, c_rd = (tg_logits.argmax(-1) == y[:, None]), (rf_logits.argmax(-1) == y[:, None]), (rd_logits.argmax(-1) == y[:, None])
                    active = (nr > 0).numpy()
                    row = {"set": sname, "t0": t0, "m": m, "frac_examples_edited": float(active.mean()),
                           "mean_removed_features": float(nr.float().mean())}
                    jidx = {"at_t0": t0 - 1, "t0+4": min(t0 + 3, K - 1), "final": K - 1}
                    for lab, j in jidx.items():
                        for arm, cc in (("target", c_t), ("rand_feat", c_rf), ("rand_dir", c_rd)):
                            d_ = cc[:, j].astype(float) - c_clean[:, j].astype(float)
                            mu_, lo_, hi_ = paired_ci(d_)
                            row[f"{arm}_d_{lab}"] = [mu_, lo_, hi_]
                            if active.any():
                                row[f"{arm}_d_{lab}_edited_only"] = float(d_[active].mean())
                        # difference-in-differences: target minus controls (paired per example)
                        dd = (c_t[:, j].astype(float) - c_rf[:, j].astype(float))
                        row[f"target_minus_randfeat_{lab}"] = list(paired_ci(dd))
                        dd2 = (c_t[:, j].astype(float) - c_rd[:, j].astype(float))
                        row[f"target_minus_randdir_{lab}"] = list(paired_ci(dd2))
                    # average over all loops after t0
                    for arm, cc in (("target", c_t), ("rand_feat", c_rf), ("rand_dir", c_rd)):
                        d_ = (cc[:, t0 - 1:].astype(float) - c_clean[:, t0 - 1:].astype(float)).mean(1)
                        row[f"{arm}_d_mean_after"] = list(paired_ci(d_))
                    # harm rescue: among examples that lose the answer later in the clean run, does editing fix it?
                    lose = c_clean[:, t0 - 1] & ~c_clean[:, min(t0 + 3, K - 1)]
                    row["n_clean_harm"] = int(lose.sum())
                    if lose.sum() > 20:
                        for arm, cc in (("target", c_t), ("rand_feat", c_rf), ("rand_dir", c_rd)):
                            d_ = (cc[lose][:, min(t0 + 3, K - 1)].astype(float))
                            row[f"{arm}_rescue_rate"] = list(paired_ci(d_))
                    results["arms"][sp]["rows"].append(row)
                    if m == max(ms):
                        print(f"  [{sp}] {sname} t0={t0} m={m}: edited={row['frac_examples_edited']:.2f} "
                              f"d@final target={row['target_d_final'][0]:+.3f} rf={row['rand_feat_d_final'][0]:+.3f} rd={row['rand_dir_d_final'][0]:+.3f} "
                              f"| d@t0 target={row['target_d_at_t0'][0]:+.3f} rf={row['rand_feat_d_at_t0'][0]:+.3f} rd={row['rand_dir_d_at_t0'][0]:+.3f}")
        # ------- steering (sufficiency): push wrong-class examples toward the true class
        steer_rows = []
        for t0 in t0s:
            for alpha in (0.5, 1.0, 2.0):
                num_ok = 0
                tot = 0
                res = {"t0": t0, "alpha": alpha}
                per_class_hits = {"target": [], "random": []}
                for c in range(task.n_classes):
                    cand = np.argsort(-np.where(ok, dla_c[:, c], -1e9))[:4]
                    S = torch.tensor(cand, dtype=torch.long)
                    idx_c = np.where(y == c)[0]
                    if len(idx_c) < 10:
                        continue
                    sub_c = {"x": []}
                    # run per-length groups containing class-c examples
                    for arm, rdir in (("target", False), ("random", True)):
                        rows = []
                        for L_ in np.unique(Nn[idx_c]):
                            ids_ = idx_c[Nn[idx_c] == L_]
                            pos_in_len = ids_ - np.where(Nn == L_)[0][0]
                            xs = sub["x"][list(sorted(set(Nn))).index(L_)][pos_in_len]
                            xb = torch.from_numpy(xs)
                            ed = SteerEditor(sae, t0, S, mean_act, alpha, seed=c, random_dir=rdir)
                            lg = toy.rollout(model, xb, K, hook=ed)["logits"].numpy()
                            was_wrong = ~c_clean[ids_, t0 - 1]
                            rows.append((lg.argmax(-1) == c, was_wrong, ids_))
                        hit_t0 = np.concatenate([r[0][:, t0 - 1][r[1]] for r in rows]) if rows else np.array([])
                        hit_fin = np.concatenate([r[0][:, -1][r[1]] for r in rows]) if rows else np.array([])
                        per_class_hits[arm].append((hit_t0, hit_fin))
                for arm in ("target", "random"):
                    h0 = np.concatenate([h[0] for h in per_class_hits[arm]]) if per_class_hits[arm] else np.array([])
                    hf = np.concatenate([h[1] for h in per_class_hits[arm]]) if per_class_hits[arm] else np.array([])
                    res[f"{arm}_flip_to_true_at_t0"] = float(h0.mean()) if len(h0) else None
                    res[f"{arm}_flip_to_true_final"] = float(hf.mean()) if len(hf) else None
                    res[f"{arm}_n_wrong"] = int(len(h0))
                steer_rows.append(res)
                print(f"  [{sp}] steer t0={t0} alpha={alpha}: flip@t0 target={res['target_flip_to_true_at_t0']} rand={res['random_flip_to_true_at_t0']}; final target={res['target_flip_to_true_final']} rand={res['random_flip_to_true_final']}")
        results["arms"][sp]["steer"] = steer_rows
    json.dump(results, open(os.path.join(res_dir(), f"{tag}__interv.json"), "w"), indent=1)
    plot(results, tag)


def plot(res, tag):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for r, sp in enumerate(("test_id", "test_ood")):
        rows = res["arms"][sp]["rows"]
        for c, sname in enumerate(("answer_feats", "loss_feats")):
            ax = axs[r, c]
            for arm, col in (("target", "tab:red"), ("rand_feat", "tab:blue"), ("rand_dir", "tab:gray")):
                for t0, ls in zip(res["t0s"][:3], ("-", "--", ":")):
                    xs = [x["m"] for x in rows if x["set"] == sname and x["t0"] == t0]
                    ys = [x[f"{arm}_d_final"][0] for x in rows if x["set"] == sname and x["t0"] == t0]
                    ax.plot(xs, ys, c=col, ls=ls, marker="o", ms=3, label=f"{arm} t0={t0}" if r == 0 and c == 0 else None)
            ax.axhline(0, c="k", lw=0.5)
            ax.set_title(f"{sp}: ablate {sname}"); ax.set_xlabel("# features in set"); ax.set_ylabel("Δ final accuracy vs clean")
    axs[0, 0].legend(fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(fig_path(tag, "f4_interventions"), dpi=130); plt.close()


if __name__ == "__main__":
    main()

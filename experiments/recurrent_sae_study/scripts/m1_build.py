"""Mini-LaCo replica, step 1: teacher -> LaCo collapse -> recurrent recovery -> activation collection.

Variants (all share the teacher's embedding, head and the layers outside the collapsed span):
  laco  : LaCo-collapsed block, single pass, no recovery  (the 'pruned' model)
  rec0  : same collapsed block looped T times, adapters at init (B=0)  (control: loops without recovery training)
  rec   : looped site + loop-specific LoRA + loop gates, distilled from the teacher (hidden-state + KL + CE)
Outputs: DATA/mini/dfa_s{seed}/{teacher,laco,rec0,rec}.pt, acts_{split}.npz, and out/res/mini_s{seed}__build.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import mini
from recsae.io import res_dir

MSPL = {          # name: (lengths, n_per_length, seed_offset)
    "sae_train": ([4, 8, 12, 16], 1000, 1),
    "sae_val": ([4, 8, 12, 16], 200, 2),
    "ana_train": ([4, 6, 8, 10, 12, 14, 16], 500, 3),
    "ana_val": ([4, 6, 8, 10, 12, 14, 16], 150, 4),
    "test_id": ([4, 8, 12, 16], 500, 5),
    "test_ood": ([20, 24, 32], 500, 6),
}
TMAX = 10          # loops evaluated (trained at T)
NPOS = 6           # extra random positions per sequence kept for SAE training (sae_train / sae_val)


@torch.no_grad()
def collect_split(teacher, students, task, split_rng, lengths, n_per, anchor, span, T, bs=250, extra_pos=0):
    """Returns dict of arrays.  Teacher states H[anchor .. anchor+span] at the last position (and extra positions).
    Per student: last [N, 1+TMAX, d] (loop 0 = state entering the site), logits [N, TMAX, C]."""
    out = {"y": [], "N": [], "t_last": [], "t_logits": [], "t_pos": []}
    for nm in students:
        out[f"{nm}__last"], out[f"{nm}__logits"], out[f"{nm}__pos"] = [], [], []
    for L in lengths:
        for i in range(0, n_per, bs):
            b = min(bs, n_per - i)
            x, y = task.sample(split_rng, b, L)
            ids = torch.from_numpy(x)
            th, ts = teacher.forward_hidden(ids, collect=True)
            H = {k: h for k, h, _ in ts}
            Hs = torch.stack([H[anchor + j] for j in range(span + 1)], 2)          # [b, L, span+1, d]
            tl = teacher.logits(th)[:, -1, task.n_sym:]          # logits restricted to the class tokens
            out["y"].append(y[:, -1]); out["N"].append(np.full(b, L))
            out["t_last"].append(Hs[:, -1].half().numpy()); out["t_logits"].append(tl.numpy())
            if extra_pos:
                pidx = torch.from_numpy(split_rng.integers(1, L, size=(b, extra_pos)))
                out["t_pos"].append(torch.gather(Hs, 1, pidx[:, :, None, None].expand(-1, -1, span + 1, Hs.shape[-1]))
                                    .reshape(-1, span + 1, Hs.shape[-1]).half().numpy())
            for nm, st in students.items():
                if st.site() is None:
                    # LaCo-only: pseudo-site with one 'loop' = the merged block
                    sh, sstates = st.forward_hidden(ids, collect=True)
                    SA = {k: h for k, h, tag in sstates if tag in ("merged", "plain")}
                    h_in = SA[anchor]
                    h_out = SA[anchor + span + 1 - 1 + 1] if (anchor + span + 1) in SA else None
                    # entry order: find the merged entry
                    mi = [j for j, e in enumerate(st.entries) if getattr(e, "merged", False)][0]
                    states = [x_ for (k, x_, tag) in sstates if tag != "embed"]
                    h_in_ = (sstates[mi][1])                                      # entries start at index 1 (after embed)
                    h_out_ = sstates[mi + 1][1]
                    last = torch.stack([h_in_[:, -1], h_out_[:, -1]], 1)
                    lg = st.logits(sh)[:, -1:, task.n_sym:]
                    pad = TMAX - 1
                    last = torch.cat([last, last[:, -1:].expand(-1, pad, -1)], 1)
                    lg = torch.cat([lg, lg[:, -1:].expand(-1, pad, -1)], 1)
                    pos_arr = None
                    if extra_pos:
                        p3 = torch.stack([h_in_, h_out_], 2)
                        p3 = torch.cat([p3, p3[:, :, -1:].expand(-1, -1, pad, -1)], 2)
                        pos_arr = torch.gather(p3, 1, pidx[:, :, None, None].expand(-1, -1, p3.shape[2], p3.shape[-1])).reshape(-1, p3.shape[2], p3.shape[-1])
                else:
                    lgs = []
                    for t in range(1, TMAX + 1):
                        h_t, _, _ = mini.student_forward(st, ids, T=t, collect_site=False)
                        lgs.append(st.logits(h_t)[:, -1, task.n_sym:])
                    lg = torch.stack(lgs, 1)
                    _, site, h_in = mini.student_forward(st, ids, T=TMAX)
                    full = torch.stack([h_in] + site, 2)                            # [b, L, 1+TMAX, d]
                    last = full[:, -1]
                    pos_arr = None
                    if extra_pos:
                        pos_arr = torch.gather(full, 1, pidx[:, :, None, None].expand(-1, -1, full.shape[2], full.shape[-1])).reshape(-1, full.shape[2], full.shape[-1])
                out[f"{nm}__last"].append(last.half().numpy()); out[f"{nm}__logits"].append(lg.numpy())
                if extra_pos and pos_arr is not None:
                    out[f"{nm}__pos"].append(pos_arr.half().numpy())
    return {k: np.concatenate(v) for k, v in out.items() if len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="dfa")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n_remove", type=int, default=4)
    ap.add_argument("--T", type=int, default=5, help="loops used in recovery training")
    ap.add_argument("--teacher_steps", type=int, default=1800)
    ap.add_argument("--rec_steps", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    task = mini.PrefixTask(a.task)
    md = mini.mdir(a.seed, a.task)
    log = {"seed": a.seed, "task": a.task}
    # ---- teacher
    if not os.path.exists(os.path.join(md, "teacher.pt")):
        t = mini.train_teacher(task, layers=a.layers, d=a.d, seed=a.seed, steps=a.teacher_steps)
        mini.save_teacher(t, a.seed, task, a.layers, a.d)
    teacher, task, _ = mini.load_teacher(a.seed, a.task)
    rng = np.random.default_rng(31_000 + a.seed)
    log["teacher_acc"] = mini.eval_last_acc(teacher, task, rng, [4, 8, 12, 16, 20, 24, 32], n=500)
    print("teacher acc", log["teacher_acc"])
    # ---- LaCo
    merges, table = mini.laco_plan(teacher, task, a.n_remove, seed=a.seed)
    anchor, absorbed = merges[0]["anchor"], merges[0]["absorbed"]
    span = len(absorbed) + 1
    log.update({"merges": merges, "laco_table": table, "anchor": anchor, "span": span})
    print("LaCo merges", merges)
    # ---- variants
    spec_rec = dict(kind="rec", T=a.T, rank=16, alpha=32.0, adapter="loop", norm="frozen", init="laco")
    variants = {}
    lac = mini.build_laco_student(teacher, merges, "laco")
    mini.save_student(lac, a.seed, a.task, "laco", merges, dict(kind="laco"))
    variants["laco"] = lac
    r0 = mini.build_laco_student(teacher, merges, "rec", T=a.T)
    mini.save_student(r0, a.seed, a.task, "rec0", merges, spec_rec)
    variants["rec0"] = r0
    path_rec = os.path.join(md, "rec.pt")
    if os.path.exists(path_rec):
        rec, _, _, _ = mini.load_student(a.seed, a.task, "rec")
    else:
        rec = mini.build_laco_student(teacher, merges, "rec", T=a.T)
        t0 = time.time()
        mini.train_recovery(rec, teacher, task, a.T, seed=a.seed, steps=a.rec_steps, tag=f"s{a.seed}")
        log["rec_train_s"] = time.time() - t0
        mini.save_student(rec, a.seed, a.task, "rec", merges, spec_rec)
    variants["rec"] = rec
    # reload check: the saved student must reproduce the in-memory one exactly
    chk, _, _, ckpt_ = mini.load_student(a.seed, a.task, "rec")
    assert len(ckpt_["train_sd"]) > 0, "empty adapter checkpoint"
    xx, _ = task.sample(np.random.default_rng(1), 64, 12)
    with torch.no_grad():
        l1 = rec.logits(rec.forward_hidden(torch.from_numpy(xx), n_loops=a.T)[0])
        l2 = chk.logits(chk.forward_hidden(torch.from_numpy(xx), n_loops=a.T)[0])
    assert torch.allclose(l1, l2, atol=1e-4), float((l1 - l2).abs().max())
    print("checkpoint reload check OK", len(ckpt_["train_sd"]), "tensors")
    # ---- accuracy tables
    rng = np.random.default_rng(32_000 + a.seed)
    acc = {"teacher": mini.eval_last_acc(teacher, task, rng, [4, 8, 12, 16, 24, 32])}
    acc["laco"] = mini.eval_last_acc(lac, task, rng, [4, 8, 12, 16, 24, 32])
    for nm in ("rec0", "rec"):
        acc[nm] = {t: mini.eval_last_acc(variants[nm], task, rng, [4, 8, 12, 16, 24, 32], n_loops=t) for t in (1, 2, 3, 4, 5, 6, 8, 10)}
    log["acc"] = acc
    print(json.dumps({k: (v if k in ("teacher", "laco") else {t: v[t] for t in (1, 3, 5, 8, 10)}) for k, v in acc.items()}))
    log["trainable"] = {k: mini.model_summary(v) for k, v in variants.items()}
    # ---- collect activations (disjoint splits)
    for sp, (lengths, n, off) in MSPL.items():
        p = os.path.join(md, f"acts_{sp}.npz")
        if os.path.exists(p):
            continue
        r = np.random.default_rng(2_000_000 * (a.seed + 1) + off)
        t0 = time.time()
        d = collect_split(teacher, variants, task, r, lengths, n, anchor, span, a.T, extra_pos=NPOS if sp in ("sae_train", "sae_val") else 0)
        np.savez(p, **d)
        print(f"  collected {sp}: {d['t_last'].shape} in {time.time()-t0:.0f}s")
    json.dump(log, open(os.path.join(res_dir(), f"mini_s{a.seed}__build.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

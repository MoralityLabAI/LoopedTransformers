"""Qwen3 path, step 1: collect activations from a LaCo-compressed + recurrently-recovered Qwen3 run produced by the
'Recurrent recovery' / 'LaCo' sibling sessions (looprec2 snapshot), on algorithmic prompts with disjoint splits.

  python q1_collect.py --sib_root <sibling root> --run runs/main_D --tag q06_main --device cuda

Writes DATA/mini/<tag>/acts_<split>.npz in the same layout the mini replica uses, so s1_train_sae.py / t3 / m2 / t5
run unchanged on  `mini:<tag>:rec`.

State convention: residual stream at the LAST PROMPT TOKEN (the position that predicts the first answer token).
  t_last   [N, span+1, d]  teacher H[anchor .. anchor+span]
  rec__last [N, 1+Tmax, d] student: state entering the recurrent site, then after loop 1..Tmax
  rec__logits [N, Tmax, 11] readout after running the whole student with t loops: logits over the digit tokens '0'..'9'
                           plus 'other' (logsumexp of all remaining vocabulary).   y = the first answer digit.
Only digit-first answers are used (all four task families produce a digit as the first answer token).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import DATA
from looprec2.laco import build_student, load_laco_manifest
from looprec2.qwen3_core import RecurrentSite, load_qwen3, resolve_model_dir
from looprec2.tasks import EVAL_LEVELS, FAMILIES, TRAIN_RANGE, TaskTokenizer, eval_examples, train_example

SPLITS = {          # name: (kind, n examples, seed)
    "sae_train": ("train", 2400, 101),
    "sae_val": ("train", 400, 102),
    "ana_train": ("train", 2400, 103),
    "ana_val": ("train", 600, 104),
    "test_id": ("eval_id", 1200, 105),
    "test_ood": ("eval_ood", 1200, 106),
}
NPOS = 6


def build_examples(kind, n, seed, families):
    if kind == "train":
        rng = random.Random(seed)
        return [train_example(rng, 0.0, families) for _ in range(n)]
    split_i = 0 if kind == "eval_id" else 1
    out = []
    per = max(1, n // (len(families) * 2))
    for fam in families:
        for lv in EVAL_LEVELS[fam][split_i]:
            out += eval_examples(fam, lv, "direct", per, seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sib_root", required=True)
    ap.add_argument("--run", required=True, help="run dir relative to sib_root (contains manifest.json, trainable.pt)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--Tmax", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply split sizes (use <1 for smoke tests)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = a.device
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    root = a.sib_root
    man = json.load(open(os.path.join(root, a.run, "manifest.json")))
    args = man["args"]
    teacher_dir = man["teacher_dir"] if os.path.isabs(man["teacher_dir"]) else os.path.join(root, man["teacher_dir"])
    laco_path = man["laco_manifest"] if os.path.isabs(man["laco_manifest"]) else os.path.join(root, man["laco_manifest"])
    teacher, tdir = load_qwen3(teacher_dir, dtype=dt, device="cpu")
    lm = load_laco_manifest(laco_path)
    merges, deleted, modes = lm["merges"], lm.get("deleted", []), lm.get("modes")
    alpha = args["alpha"] if args["alpha"] is not None else 2.0 * args["rank"]
    rec = dict(max_loops=args["loops"], adapter=args["adapter"], rank=args["rank"], alpha=alpha, norm=args["norm"],
               gate=not args["no_gate"], init=args["rec_init"], extrap=args["extrap"])
    Tmax = a.Tmax or max(2 * args["loops"], 8)
    variants = {}
    variants["laco"] = build_student(teacher, merges, modes=modes, deleted=deleted)
    variants["rec0"] = build_student(teacher, merges, rec=rec, modes=modes, deleted=deleted)
    rec_model = build_student(teacher, merges, rec=rec, modes=modes, deleted=deleted)
    sd = torch.load(os.path.join(root, a.run, "trainable.pt"), map_location="cpu")
    miss = rec_model.load_state_dict(sd, strict=False)
    assert not miss.unexpected_keys, miss.unexpected_keys
    variants["rec"] = rec_model
    teacher = teacher.to(dev)
    for k in variants:
        variants[k] = variants[k].to(dev).eval()
    site = variants["rec"].site()
    anchor, span = site.anchor, site.span
    tok_file = os.path.join(tdir, "tokenizer.json")
    tt = TaskTokenizer(tok_file)
    digit_ids = [tt.tok.encode(str(i), add_special_tokens=False).ids[0] for i in range(10)]
    out_dir = os.path.join(DATA, "mini", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(tag=a.tag, run=a.run, anchor=anchor, span=span, T=args["loops"], Tmax=Tmax, merges=merges, deleted=deleted,
                modes=modes, rec=rec, teacher_dir=teacher_dir, d=teacher.cfg.hidden_size, n_layers=teacher.cfg.num_layers,
                families=a.families, classes=11)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    dig = torch.tensor(digit_ids, device=dev)

    def restrict(h_last, model):
        lg = (h_last.to(model.head_weight().dtype) @ model.head_weight().t()).float()
        other = torch.logsumexp(lg.index_fill(1, dig, float("-inf")), dim=1, keepdim=True)
        return torch.cat([lg[:, dig], other], 1)

    families = a.families.split(",")
    for sp in a.splits.split(","):
        kind, n, seed = SPLITS[sp]
        p = os.path.join(out_dir, f"acts_{sp}.npz")
        if os.path.exists(p):
            continue
        exs = build_examples(kind, max(8, int(n * a.scale)), seed, families)
        keep = [e for e in exs if e.target[0].isdigit()]
        enc = tt.encode_batch([e.prompt for e in keep])
        order = np.argsort([len(x) for x in enc])
        extra = sp in ("sae_train", "sae_val")
        rngp = np.random.default_rng(seed)
        out = {"y": [], "N": [], "t_last": [], "t_logits": [], "t_pos": []}
        for nm in variants:
            out[f"{nm}__last"], out[f"{nm}__logits"], out[f"{nm}__pos"] = [], [], []
        t0 = time.time()
        for i in range(0, len(order), a.bs):
            bi = order[i:i + a.bs]
            seqs = [enc[j] for j in bi]
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), tt.pad_id, dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, :len(s)] = torch.tensor(s)
            ids = ids.to(dev)
            lastpos = torch.tensor([len(s) - 1 for s in seqs], device=dev)
            ar = torch.arange(len(seqs), device=dev)
            pos_idx = None
            if extra:
                pos_idx = torch.from_numpy(np.stack([rngp.integers(1, max(2, len(s) - 1), size=NPOS) for s in seqs])).to(dev)
            with torch.no_grad():
                th, ts = teacher.forward_hidden(ids, collect=True)
                H = {k: h for k, h, _ in ts}
                Hs = torch.stack([H[anchor + j] for j in range(span + 1)], 2)           # [b, L, span+1, d]
                out["t_last"].append(Hs[ar, lastpos].float().half().cpu().numpy())
                out["t_logits"].append(restrict(th[ar, lastpos], teacher).cpu().numpy())
                if extra:
                    out["t_pos"].append(torch.gather(Hs, 1, pos_idx[:, :, None, None].expand(-1, -1, span + 1, Hs.shape[-1])).reshape(-1, span + 1, Hs.shape[-1]).float().half().cpu().numpy())
                out["y"].append(np.array([int(keep[j].target[0]) for j in bi])); out["N"].append(np.array([len(s) for s in seqs]))
                for nm, st in variants.items():
                    if st.site() is None:
                        sh, sst = st.forward_hidden(ids, collect=True)
                        mi = [k for k, e in enumerate(st.entries) if getattr(e, "merged", False)][0]
                        h_in, h_out = sst[mi][1], sst[mi + 1][1]
                        full = torch.stack([h_in, h_out], 2)
                        full = torch.cat([full, full[:, :, -1:].expand(-1, -1, Tmax - 1, -1)], 2)
                        lg = restrict(sh[ar, lastpos], st)[:, None].expand(-1, Tmax, -1)
                    else:
                        lgs = []
                        for t in range(1, Tmax + 1):
                            sh, _ = st.forward_hidden(ids, n_loops=t)
                            lgs.append(restrict(sh[ar, lastpos], st))
                        lg = torch.stack(lgs, 1)
                        _, sst = st.forward_hidden(ids, n_loops=Tmax, collect=True)
                        first = [k for k, (_, _, tag) in enumerate(sst) if tag in ("step", "site_final")][0]
                        h_in = sst[first - 1][1]
                        steps = [x[1] for x in sst if x[2] in ("step", "site_final")]
                        full = torch.stack([h_in] + steps, 2)
                    out[f"{nm}__last"].append(full[ar, lastpos].float().half().cpu().numpy())
                    out[f"{nm}__logits"].append(lg.float().cpu().numpy())
                    if extra:
                        out[f"{nm}__pos"].append(torch.gather(full, 1, pos_idx[:, :, None, None].expand(-1, -1, full.shape[2], full.shape[-1])).reshape(-1, full.shape[2], full.shape[-1]).float().half().cpu().numpy())
            if (i // a.bs) % 20 == 0:
                print(f"  {sp}: {i}/{len(order)} ({time.time()-t0:.0f}s)", flush=True)
        np.savez(p, **{k: np.concatenate(v) for k, v in out.items() if len(v)})
        print(f"saved {sp}: {len(order)} examples in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

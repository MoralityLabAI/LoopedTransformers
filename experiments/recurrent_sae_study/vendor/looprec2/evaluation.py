"""Evaluation harness: accuracy, teacher-logit KL, latency, memory, FLOPs, longer sequences, per loop count."""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from .qwen3_core import Qwen3LM
from .tasks import EVAL_LEVELS, FAMILIES, TaskTokenizer, eval_examples


@torch.no_grad()
def token_stats(h_s, h_t, ids, pred_mask, W, chunk=512):
    """Per-predicted-token stats.  Returns dict of 1-D tensors aligned with pred_mask[:,1:].nonzero()."""
    m = pred_mask[:, 1:]
    rows = m.nonzero()[:, 0]
    hs = h_s[:, :-1][m]
    ht = h_t[:, :-1][m] if h_t is not None else None
    lab = ids[:, 1:][m]
    kl, ce, corr, agree = [], [], [], []
    for i in range(0, lab.numel(), chunk):
        ls = (hs[i:i + chunk] @ W.t()).float()
        lp = F.log_softmax(ls, -1)
        ce.append(-lp.gather(-1, lab[i:i + chunk, None]).squeeze(-1))
        corr.append(ls.argmax(-1) == lab[i:i + chunk])
        if ht is not None:
            lt = (ht[i:i + chunk] @ W.t()).float()
            lq = F.log_softmax(lt, -1)
            kl.append((lq.exp() * (lq - lp)).sum(-1))
            agree.append(ls.argmax(-1) == lt.argmax(-1))
    cat = lambda x: torch.cat(x) if x else None
    return {"rows": rows, "kl": cat(kl), "ce": cat(ce), "correct": cat(corr), "agree": cat(agree)}


def _batches(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]


def build_eval_sets(text, tt: TaskTokenizer, device, n_text=64, n_long=16, n_gsm=64, n_task=64, seed=0,
                    families=FAMILIES):
    """Returns list of dicts: name, kind, ids, pred_mask, (final_mask), meta."""
    sets = []
    ids = text.heldout_chunks(n_text, 256)
    sets.append(dict(name="text256", kind="text", ids=ids.to(device), pred=torch.ones_like(ids, dtype=torch.bool).to(device)))
    if n_long:
        ids = text.heldout_chunks(n_long, 1024)
        sets.append(dict(name="text1024", kind="text", ids=ids.to(device),
                         pred=torch.ones_like(ids, dtype=torch.bool).to(device)))
    g = text.gsm_batches(n_gsm)
    T = max(x.shape[1] for x in g)
    gi = torch.full((len(g), T), tt.pad_id, dtype=torch.long)
    gm = torch.zeros(len(g), T, dtype=torch.bool)
    for i, x in enumerate(g):
        gi[i, :x.shape[1]] = x[0]
        gm[i, :x.shape[1]] = True
    sets.append(dict(name="gsm8k_text", kind="text", ids=gi.to(device), pred=gm.to(device)))
    for fam in families:
        id_lv, ood_lv = EVAL_LEVELS[fam]
        for split, lvs in (("id", id_lv), ("ood", ood_lv)):
            for lv in lvs:
                for fmt in ("direct", "scratch"):
                    exs = eval_examples(fam, lv, fmt, n_task, seed)
                    ids, tmask, fmask = tt.collate(exs, device)
                    sets.append(dict(name=f"{fam}|{lv}|{split}|{fmt}", kind="task", fam=fam, level=str(lv),
                                     split=split, fmt=fmt, ids=ids, pred=tmask, final=fmask))
    return sets


@torch.no_grad()
def teacher_hidden(teacher: Qwen3LM, ids):
    return teacher.forward_hidden(ids)[0]


@torch.no_grad()
def eval_suite(student: Qwen3LM, teacher: Qwen3LM | None, sets: list[dict], loops: list[int | None],
               bs_text=16, bs_long=4, bs_task=32, ref_ctx=256, latency=True, log=print) -> dict:
    dev = next(student.parameters()).device
    W = student.head_weight()
    out = {}
    student.eval()
    for T in loops:
        out[str(T)] = {"sets": {}}
    for s in sets:
        bs = bs_long if s["name"] == "text1024" else (bs_text if s["kind"] == "text" else bs_task)
        acc = {str(T): dict(kl=[], ce=[], correct=[], agree=[], rows=[], off=0) for T in loops}
        n = s["ids"].shape[0]
        for i in range(0, n, bs):
            ids, pm = s["ids"][i:i + bs], s["pred"][i:i + bs]
            ht = teacher_hidden(teacher, ids) if teacher is not None else None
            for T in loops:
                hs, _ = student.forward_hidden(ids, n_loops=T)
                st = token_stats(hs, ht, ids, pm, W)
                a = acc[str(T)]
                a["rows"].append(st["rows"] + i)
                for k in ("kl", "ce", "correct", "agree"):
                    if st[k] is not None:
                        a[k].append(st[k])
        for T in loops:
            a = acc[str(T)]
            cat = lambda k: torch.cat(a[k]) if a[k] else None
            rows, kl, ce, corr, agree = cat("rows"), cat("kl"), cat("ce"), cat("correct"), cat("agree")
            r = {"ce": ce.mean().item(), "tok_acc": corr.float().mean().item()}
            if kl is not None:
                r["kl"] = kl.mean().item()
                r["top1_agree"] = agree.float().mean().item()
            if s["kind"] == "task":
                nrow = n
                per_ex = torch.zeros(nrow, device=dev).index_add_(0, rows, (~corr).float())
                r["exact"] = (per_ex == 0).float().mean().item()
                # final-answer-only accuracy (teacher-forced prefix)
                fm = s["final"][:, 1:]
                fm_flat = fm[s["pred"][:, 1:]]                       # aligned with the flattened target tokens
                bad = torch.zeros(nrow, device=dev).index_add_(0, rows[fm_flat], (~corr[fm_flat]).float())
                r["final_exact"] = (bad == 0).float().mean().item()
            out[str(T)]["sets"][s["name"]] = r
    # ---- cost metrics (per loop count) ----
    for T in loops:
        rec = out[str(T)]
        rec["flops_per_token_256"] = student.flops_per_token(ref_ctx, T)
        rec["flops_per_token_1024"] = student.flops_per_token(1024, T)
        rec["layer_evals"] = student.n_layer_evals(T)
        if latency and dev.type == "cuda":
            ids = torch.randint(0, 1000, (8, ref_ctx), device=dev)
            for _ in range(3):
                student.forward_hidden(ids, n_loops=T)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(10):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                student.forward_hidden(ids, n_loops=T)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            rec["latency_ms_b8_s256"] = statistics.median(ts)
            rec["act_peak_mb_b8_s256"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    return out


def summarize(res_T: dict) -> dict:
    """Aggregate a per-T record into headline numbers."""
    sets = res_T["sets"]
    g = lambda pred, key: [v[key] for n, v in sets.items() if pred(n) and key in v]
    mean = lambda x: float(np.mean(x)) if x else float("nan")
    s = {
        "text_kl": sets["text256"].get("kl", float("nan")),
        "text_ce": sets["text256"]["ce"],
        "text_top1": sets["text256"].get("top1_agree", float("nan")),
        "text1024_kl": sets.get("text1024", {}).get("kl", float("nan")),
        "gsm_kl": sets["gsm8k_text"].get("kl", float("nan")),
    }
    for fam in FAMILIES:
        for split in ("id", "ood"):
            for fmt in ("direct", "scratch"):
                pre = lambda n, fam=fam, split=split, fmt=fmt: n.startswith(fam + "|") and f"|{split}|{fmt}" in n
                s[f"{fam}_{split}_{fmt}"] = mean(g(pre, "exact"))
    for split in ("id", "ood"):
        for fmt in ("direct", "scratch"):
            s[f"algo_{split}_{fmt}"] = mean([s[f"{f}_{split}_{fmt}"] for f in FAMILIES])
    mc = res_T.get("mc")
    if mc:
        for k, v in mc.items():
            s[f"{k}_acc_norm"] = v["acc_norm"]
            s[f"{k}_acc"] = v["acc"]
        s["mc_acc_norm"] = float(np.mean([v["acc_norm"] for v in mc.values()]))
        s["mc_acc"] = float(np.mean([v["acc"] for v in mc.values()]))
    s["task_kl_direct"] = mean([v["kl"] for n, v in sets.items() if "|direct" in n and "kl" in v])
    return s

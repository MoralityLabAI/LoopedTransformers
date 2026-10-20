"""LaCo (Layer Collapse, Yang et al. 2024) structured pruning + student construction.

LaCo merges a run of consecutive layers into its first one by parameter-difference summation
    theta*_l = theta_l + sum_{k=1..m} (theta_{l+k} - theta_l)
and deletes layers l+1..l+m.  The merge range is picked by the cosine similarity between the final
representation of the merged and the original model on calibration text.

Two search modes are implemented:
  single : exhaustive search for ONE collapsed range of exactly n_remove layers (default in the
           experiments - gives one clean site for the recurrent block)
  paper  : the paper's top-down scan with threshold T, max merge span C and gap I; T is bisected so
           that exactly n_remove layers are removed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import install_lora, quantize_blocks
from .qwen3_core import Block, Qwen3LM, RecurrentSite, StepCtx, rope_cos_sin


# --------------------------------------------------------------- merging ----

@torch.no_grad()
def collapse_blocks(blocks: list[Block], anchor: int, absorbed: list[int]) -> Block:
    base = blocks[anchor]
    out = copy.deepcopy(base)
    sd_a = dict(base.named_parameters())
    for name, p in out.named_parameters():
        acc = sd_a[name].float().clone()
        for k in absorbed:
            acc += dict(blocks[k].named_parameters())[name].float() - sd_a[name].float()
        p.data = acc.to(sd_a[name].dtype)
    out.merged = len(absorbed) > 0
    out.t_end = (absorbed[-1] if absorbed else anchor)
    return out


@torch.no_grad()
def mean_blocks(blocks: list[Block], idxs: list[int]) -> Block:
    out = copy.deepcopy(blocks[idxs[0]])
    named = [dict(blocks[i].named_parameters()) for i in idxs]
    for name, p in out.named_parameters():
        p.data = (sum(n[name].float() for n in named) / len(idxs)).to(p.dtype)
    out.merged = len(idxs) > 1
    out.t_end = idxs[-1]
    return out


# ---------------------------------------------------------------- search ----

@torch.no_grad()
def _teacher_states(teacher: Qwen3LM, ids: torch.Tensor):
    h = teacher.embed(ids)
    cos, sin = rope_cos_sin(ids.shape[1], teacher.cfg.head_dim, teacher.cfg.rope_theta, ids.device, h.dtype)
    Hs = [h]
    for b in teacher.entries:
        h = b(h, cos, sin)
        Hs.append(h)
    return Hs, cos, sin


@torch.no_grad()
def _sim(teacher: Qwen3LM, Hs, cos, sin, merges_above: list[tuple[int, list[int]]],
         cand: tuple[int, list[int]] | None, ref_final: torch.Tensor) -> float:
    """cosine similarity of final hidden (post-norm, token 0 skipped) of the plan vs the teacher."""
    plan = sorted(merges_above + ([cand] if cand else []), key=lambda x: x[0])
    start = plan[0][0] if plan else teacher.cfg.num_layers
    blocks = teacher.entries
    h = Hs[start]
    i = start
    mp = {a: ab for a, ab in plan}
    L = teacher.cfg.num_layers
    while i < L:
        if i in mp:
            b = collapse_blocks(blocks, i, mp[i])
            h = b(h, cos, sin)
            i = (mp[i][-1] if mp[i] else i) + 1
        else:
            h = blocks[i](h, cos, sin)
            i += 1
    hn = teacher.norm(h)
    return F.cosine_similarity(hn[:, 1:].float(), ref_final[:, 1:].float(), dim=-1).mean().item()


@torch.no_grad()
def laco_search_single(teacher: Qwen3LM, calib: torch.Tensor, n_remove: int,
                       protect_first: int = 2, protect_last: int = 1, bs: int = 8):
    L = teacher.cfg.num_layers
    table = []
    sims_by_anchor: dict[int, float] = {}
    chunks = calib.split(bs)
    cache = []
    for c in chunks:
        Hs, cos, sin = _teacher_states(teacher, c)
        cache.append((Hs, cos, sin, teacher.norm(Hs[-1])))
    for l in range(protect_first, L - protect_last - n_remove):
        cand = (l, list(range(l + 1, l + 1 + n_remove)))
        s = sum(_sim(teacher, Hs, cos, sin, [], cand, rf) * c.shape[0]
                for (Hs, cos, sin, rf), c in zip(cache, chunks)) / calib.shape[0]
        sims_by_anchor[l] = s
        table.append({"anchor": l, "absorbed": cand[1], "cos_sim": s})
    best = max(sims_by_anchor, key=sims_by_anchor.get)
    merges = [{"anchor": best, "absorbed": list(range(best + 1, best + 1 + n_remove))}] if n_remove > 0 else []
    return merges, table


@torch.no_grad()
def laco_search_paper(teacher: Qwen3LM, calib: torch.Tensor, n_remove: int, max_span: int = 8, gap: int = 1,
                      protect_first: int = 2, protect_last: int = 1, bs: int = 8, iters: int = 10):
    L = teacher.cfg.num_layers
    chunks = calib.split(bs)
    cache = []
    for c in chunks:
        Hs, cos, sin = _teacher_states(teacher, c)
        cache.append((Hs, cos, sin, teacher.norm(Hs[-1])))

    def sim_of(above, cand):
        return sum(_sim(teacher, Hs, cos, sin, above, cand, rf) * c.shape[0]
                   for (Hs, cos, sin, rf), c in zip(cache, chunks)) / calib.shape[0]

    def scan(T):
        above: list[tuple[int, list[int]]] = []
        removed, l, ceil = 0, L - protect_last - 2, L - protect_last - 1   # ceil: first protected idx
        while l >= protect_first and removed < n_remove:
            mmax = min(max_span, ceil - 1 - l, n_remove - removed)
            hit = None
            for m in range(mmax, 0, -1):
                if sim_of(above, (l, list(range(l + 1, l + 1 + m)))) >= T:
                    hit = m
                    break
            if hit:
                above.append((l, list(range(l + 1, l + 1 + hit))))
                removed += hit
                ceil = l
                l -= gap + 1
            else:
                l -= 1
        return above, removed

    lo, hi = 0.0, 1.0
    best = None
    for _ in range(iters):
        T = (lo + hi) / 2
        above, removed = scan(T)
        if removed >= n_remove:
            best, lo = (T, above), T
        else:
            hi = T
    if best is None:
        T, above = lo, scan(lo)[0]
        best = (T, above)
    T, above = best
    merges = [{"anchor": a, "absorbed": ab} for a, ab in sorted(above)]
    return merges, {"threshold": T, "removed": sum(len(m["absorbed"]) for m in merges)}


@torch.no_grad()
def plan_similarity(teacher: Qwen3LM, calib: torch.Tensor, merges: list[dict], bs: int = 8) -> float:
    tot = 0.0
    for c in calib.split(bs):
        Hs, cos, sin = _teacher_states(teacher, c)
        tot += _sim(teacher, Hs, cos, sin, [(m["anchor"], m["absorbed"]) for m in merges], None,
                    teacher.norm(Hs[-1])) * c.shape[0]
    return tot / calib.shape[0]


# ------------------------------------------------------------- manifests ----

def sha256_file(path: str, n: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()


def write_laco_manifest(path: str, *, model_dir: str, num_layers: int, merges: list[dict], extra: dict):
    removed = sorted(i for m in merges for i in m["absorbed"])
    anchors = [m["anchor"] for m in merges]
    man = {"schema": "laco_manifest/v1", "model_dir": model_dir, "num_layers": num_layers, "merges": merges,
           "removed_layers": removed, "kept_layer_count": num_layers - len(removed),
           "anchors": anchors, **extra}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(man, open(path, "w"), indent=2)
    return man


def load_laco_manifest(path: str) -> dict:
    """Accepts our schema, or any JSON with `merges` [{anchor, absorbed}] ; or a bare list of removed layers
    (then each maximal run of removed layers is attributed to the kept layer immediately before it)."""
    m = json.load(open(path))
    if "merges" in m:
        return m
    removed = sorted(m.get("removed_layers") or m.get("pruned_layers") or [])
    merges, run = [], []
    for i in removed:
        if run and i == run[-1] + 1:
            run.append(i)
        else:
            if run:
                merges.append({"anchor": run[0] - 1, "absorbed": run})
            run = [i]
    if run:
        merges.append({"anchor": run[0] - 1, "absorbed": run})
    m["merges"] = merges
    return m


# --------------------------------------------------------------- student ----

def build_student(teacher: Qwen3LM, merges: list[dict], *, rec: dict | None = None, lora: dict | None = None,
                  quant: str = "none") -> Qwen3LM:
    """teacher: plain Qwen3LM on CPU (weights only).  Returns a student sharing embed/norm/head with teacher.

    rec  = dict(max_loops, adapter ('loop'|'shared'|'none'), rank, alpha, norm ('frozen'|'shared'|'loop'),
                gate, init ('laco'|'anchor'|'mean'), extrap)
    lora = dict(rank, alpha)   ordinary LoRA on every block (condition C); n=1
    """
    blocks = list(teacher.entries)
    absorbed = {i for m in merges for i in m["absorbed"]}
    by_anchor = {m["anchor"]: m["absorbed"] for m in merges}
    site_merge = max(merges, key=lambda m: (len(m["absorbed"]), -m["anchor"])) if (rec and merges) else None
    if rec and not merges:
        raise ValueError("recurrent student needs at least one collapsed range")

    entries: list[nn.Module] = []
    for i, b in enumerate(blocks):
        if i in absorbed:
            continue
        if i in by_anchor:
            ab = by_anchor[i]
            if rec and site_merge is not None and i == site_merge["anchor"]:
                if rec.get("init", "laco") == "laco":
                    nb = collapse_blocks(blocks, i, ab)
                elif rec["init"] == "anchor":
                    nb = copy.deepcopy(b)
                elif rec["init"] == "mean":
                    nb = mean_blocks(blocks, [i] + ab)
                else:
                    raise ValueError(rec["init"])
                nb.t_end = ab[-1]
                entries.append(RecurrentSite(nb, anchor=i, span=len(ab) + 1, max_loops=rec["max_loops"],
                                             gate=rec.get("gate", True), extrap=rec.get("extrap", "clamp")))
            else:
                entries.append(collapse_blocks(blocks, i, ab))
        else:
            nb = copy.deepcopy(b)
            nb.t_end = i
            entries.append(nb)

    with torch.device("meta"):
        st = Qwen3LM(teacher.cfg, entries=entries)
    st.embed, st.norm = teacher.embed, teacher.norm
    if not teacher.cfg.tie:
        st.lm_head = teacher.lm_head
    for p in st.parameters():
        p.requires_grad_(False)

    quantize_blocks(st, quant)

    if lora:
        ctx = StepCtx()
        for e in st.entries:
            if isinstance(e, Block):
                install_lora(e, lora["rank"], lora["alpha"], 1, ctx)
    if rec:
        site = st.site()
        n_ad = {"loop": rec["max_loops"], "shared": 1, "none": 0}[rec["adapter"]]
        if n_ad:
            install_lora(site.block, rec["rank"], rec["alpha"], n_ad, site.ctx)
        if rec["norm"] != "frozen":
            n_nm = rec["max_loops"] if rec["norm"] == "loop" else 1
            for nm in Block.NORMS:
                getattr(site.block, nm).enable_trainable(n_nm, site.ctx)
        st.default_loops = rec["max_loops"]
    return st


def trainable_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for _, p in trainable_params(model))


def trainable_state_dict(model: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in trainable_params(model)}

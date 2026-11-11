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
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import install_lora, quantize_blocks
from .qwen3_core import Block, Qwen3LM, RecurrentSite, StepCtx, rope_cos_sin


# --------------------------------------------------------------- merging ----

DEFAULT_MODES = {"weight": "last", "norm": "base", "qknorm": "follow"}


def _cat(name: str) -> str:
    """parameter category of a Block parameter name (mirrors laco-lab `classify`)"""
    if name.startswith(("ln1.", "ln2.")):
        return "norm"
    if name.startswith(("q_norm.", "k_norm.")):
        return "qknorm"
    return "proj"


def _combine(ts: list[torch.Tensor], mode: str) -> torch.Tensor:
    """Group [t0..tK] -> one tensor.  last: tK (what the official LaCo notebook computes via in-place
    base.add_(t_k - base)); first: t0; mean; rdsc: t0 + sum(t_k - t0) (formula printed in the paper);
    rdsc_mean: t0 + mean(t_k - t0)."""
    if mode == "base":
        mode = "first"
    if mode == "last":
        return ts[-1].clone()
    if mode == "first":
        return ts[0].clone()
    f = [t.float() for t in ts]
    if mode == "mean":
        out = torch.stack(f).mean(0)
    elif mode == "rdsc":
        out = f[0] + sum((x - f[0]) for x in f[1:])
    elif mode == "rdsc_mean":
        out = f[0] + torch.stack([x - f[0] for x in f[1:]]).mean(0)
    else:
        raise ValueError(mode)
    return out.to(ts[0].dtype)


@torch.no_grad()
def combine_blocks(blocks: list[Block], idxs: list[int], modes: dict | None = None) -> Block:
    modes = {**DEFAULT_MODES, **(modes or {})}
    out = copy.deepcopy(blocks[idxs[0]])
    named = [dict(blocks[i].named_parameters()) for i in idxs]
    qk = modes["weight"] if modes["qknorm"] == "follow" else modes["qknorm"]
    for name, p in out.named_parameters():
        mode = {"proj": modes["weight"], "norm": modes["norm"], "qknorm": qk}[_cat(name)]
        p.data = _combine([n[name] for n in named], mode).to(p.dtype)
    out.merged = len(idxs) > 1
    out.t_end = idxs[-1]
    return out


def collapse_blocks(blocks, anchor: int, absorbed: list[int], modes: dict | None = None) -> Block:
    return combine_blocks(blocks, [anchor] + list(absorbed), modes)


def mean_blocks(blocks, idxs):
    return combine_blocks(blocks, idxs, {"weight": "mean", "norm": "mean", "qknorm": "mean"})


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
         cand: tuple[int, list[int]] | None, ref_final: torch.Tensor, modes: dict | None = None) -> float:
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
            b = collapse_blocks(blocks, i, mp[i], modes)
            h = b(h, cos, sin)
            i = (mp[i][-1] if mp[i] else i) + 1
        else:
            h = blocks[i](h, cos, sin)
            i += 1
    hn = teacher.norm(h)
    return F.cosine_similarity(hn[:, 1:].float(), ref_final[:, 1:].float(), dim=-1).mean().item()


@torch.no_grad()
def laco_search_single(teacher: Qwen3LM, calib: torch.Tensor, n_remove: int,
                       protect_first: int = 2, protect_last: int = 1, bs: int = 8, modes: dict | None = None):
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
        s = sum(_sim(teacher, Hs, cos, sin, [], cand, rf, modes) * c.shape[0]
                for (Hs, cos, sin, rf), c in zip(cache, chunks)) / calib.shape[0]
        sims_by_anchor[l] = s
        table.append({"anchor": l, "absorbed": cand[1], "cos_sim": s})
    best = max(sims_by_anchor, key=sims_by_anchor.get)
    merges = [{"anchor": best, "absorbed": list(range(best + 1, best + 1 + n_remove))}] if n_remove > 0 else []
    return merges, table


@torch.no_grad()
def laco_search_paper(teacher: Qwen3LM, calib: torch.Tensor, n_remove: int, max_span: int = 8, gap: int = 1,
                      protect_first: int = 2, protect_last: int = 1, bs: int = 8, iters: int = 10, modes: dict | None = None):
    L = teacher.cfg.num_layers
    chunks = calib.split(bs)
    cache = []
    for c in chunks:
        Hs, cos, sin = _teacher_states(teacher, c)
        cache.append((Hs, cos, sin, teacher.norm(Hs[-1])))

    def sim_of(above, cand):
        return sum(_sim(teacher, Hs, cos, sin, above, cand, rf, modes) * c.shape[0]
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
def plan_similarity(teacher: Qwen3LM, calib: torch.Tensor, merges: list[dict], bs: int = 8, modes: dict | None = None) -> float:
    tot = 0.0
    for c in calib.split(bs):
        Hs, cos, sin = _teacher_states(teacher, c)
        tot += _sim(teacher, Hs, cos, sin, [(m["anchor"], m["absorbed"]) for m in merges], None,
                    teacher.norm(Hs[-1]), modes) * c.shape[0]
    return tot / calib.shape[0]


# ------------------------------------------------------------- manifests ----

def sha256_file(path: str, n: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()


def write_laco_manifest(path: str, *, model_dir: str, num_layers: int, merges: list[dict], extra: dict,
                        modes: dict | None = None, deleted: list[int] | None = None):
    deleted = sorted(deleted or [])
    removed = sorted([i for m in merges for i in m["absorbed"]] + deleted)
    man = {"schema": "laco_manifest/v1", "model_dir": model_dir, "num_layers": num_layers, "merges": merges,
           "deleted": deleted, "modes": {**DEFAULT_MODES, **(modes or {})},
           "removed_layers": removed, "kept_layer_count": num_layers - len(removed),
           "anchors": [m["anchor"] for m in merges], **extra}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(man, open(path, "w"), indent=2)
    return man


def load_laco_manifest(path: str) -> dict:
    """Accepts, in order of preference:
      * a laco-lab recipe (laco_recipe.json / `search` output: has `layout` or `ops`+`source_layers`)  <- the LaCo
        reproduction project's output; groups of >1 original layers become merges (anchor = first), original
        layers that appear in no group are `deleted`;
      * a directory containing laco_recipe.json (a compressed checkpoint written by `lacolab compress`);
      * our own manifest (`merges` [{anchor, absorbed}]);
      * a bare `removed_layers` list (each maximal run is attributed to the kept layer right before it)."""
    if os.path.isdir(path):
        path = os.path.join(path, "laco_recipe.json")
    m = json.load(open(path))
    if "layout" in m or ("ops" in m and "source_layers" in m):
        if "layout" not in m:
            sys.path.insert(0, os.environ.get("LACOLAB_SRC", ""))
            from lacolab.recipe import Recipe
            m["layout"] = Recipe.from_dict(m).layout()
        n = m["source_layers"]
        merges = [{"anchor": g[0], "absorbed": g[1:]} for g in m["layout"] if len(g) > 1]
        kept = {i for g in m["layout"] for i in g}
        deleted = sorted(set(range(n)) - kept)
        modes = {"weight": m.get("weight_mode", "last"), "norm": m.get("norm_mode", "base"),
                 "qknorm": m.get("qknorm_mode", "follow")}
        removed = sorted([i for mm in merges for i in mm["absorbed"]] + deleted)
        return {"schema": "laco_manifest/v1", "num_layers": n, "merges": merges, "deleted": deleted, "modes": modes,
                "removed_layers": removed, "kept_layer_count": n - len(removed), "source": "laco-lab recipe",
                "recipe_method": m.get("method"), "recipe_meta": m.get("meta", {})}
    if "merges" in m:
        m.setdefault("deleted", [])
        m.setdefault("modes", dict(DEFAULT_MODES))
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
    m.setdefault("deleted", [])
    m.setdefault("modes", dict(DEFAULT_MODES))
    return m


# --------------------------------------------------------------- student ----

def shallow_block(b: Block) -> Block:
    """New Block object sharing all weight tensors / sub-modules with `b`, but with its own module table, so that
    installing LoRA wrappers on the copy never touches `b` (the frozen teacher)."""
    c = copy.copy(b)
    c._modules = dict(b._modules)
    c._parameters = dict(b._parameters)
    c._buffers = dict(b._buffers)
    return c


def build_student(teacher: Qwen3LM, merges: list[dict], *, rec: dict | None = None, lora: dict | None = None,
                  quant: str = "none", modes: dict | None = None, deleted: list[int] | None = None) -> Qwen3LM:
    """teacher: plain Qwen3LM (weights only).  Returns a student sharing embed/norm/head with teacher.

    merges  : [{anchor, absorbed}]  LaCo collapses (merge arithmetic per `modes`)
    deleted : original layers removed outright (laco-lab delete recipes)
    rec  = dict(max_loops, adapter ('loop'|'shared'|'none'), rank, alpha, norm ('frozen'|'shared'|'loop'),
                gate, init ('laco'|'anchor'|'mean'), extrap)
            The recurrent site replaces the largest collapsed group; if the recipe is deletion-only it replaces the
            layer just before the largest deleted run (initialised from that layer) so that the loop re-spends the
            deleted depth.
    lora = dict(rank, alpha)   ordinary LoRA on every block (condition C); n=1
    """
    blocks = list(teacher.entries)
    deleted = sorted(deleted or [])
    absorbed = {i for m in merges for i in m["absorbed"]} | set(deleted)
    by_anchor = {m["anchor"]: m["absorbed"] for m in merges}
    site_anchor, site_span, site_init = None, 0, None
    if rec:
        if merges:
            sm = max(merges, key=lambda m: (len(m["absorbed"]), -m["anchor"]))
            site_anchor, site_span, site_init = sm["anchor"], len(sm["absorbed"]) + 1, rec.get("init", "laco")
        elif deleted:
            runs, run = [], [deleted[0]]
            for i in deleted[1:]:
                if i == run[-1] + 1:
                    run.append(i)
                else:
                    runs.append(run)
                    run = [i]
            runs.append(run)
            best = max(runs, key=len)
            if best[0] == 0:
                raise ValueError("deleted run starts at layer 0; no anchor layer for the recurrent site")
            site_anchor, site_span, site_init = best[0] - 1, len(best) + 1, "anchor"
        else:
            raise ValueError("recurrent student needs at least one collapsed/deleted range")

    entries: list[nn.Module] = []
    for i, b in enumerate(blocks):
        if i in absorbed:
            continue
        if rec and i == site_anchor:
            ab = by_anchor.get(i, [])
            if site_init == "laco" and ab:
                nb = collapse_blocks(blocks, i, ab, modes)
            elif site_init == "mean" and ab:
                nb = mean_blocks(blocks, [i] + ab)
            else:
                nb = copy.deepcopy(b)
            nb.t_end = i + site_span - 1
            entries.append(RecurrentSite(nb, anchor=i, span=site_span, max_loops=rec["max_loops"],
                                         gate=rec.get("gate", True), extrap=rec.get("extrap", "clamp")))
        elif i in by_anchor:
            entries.append(collapse_blocks(blocks, i, by_anchor[i], modes))
        else:
            nb = shallow_block(b) if quant == "none" else copy.deepcopy(b)
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
        if site.gate is not None:
            site.gate.requires_grad_(True)           # (the blanket freeze above also caught the gate)
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

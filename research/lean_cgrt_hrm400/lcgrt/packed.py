"""Pre-encoded training states: flat numpy arrays + offsets, built once per split.

Features come from encode.state_features (the function the search uses on live Lean
states), with no failure history; the failure-history augmentation is re-applied per
sample at batch time, vectorised. Quantised to uint8 (features lie in [0, 1]).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from .dataset import iter_records
from .encode import (E_SELF, F_FAILED, F_PREM_FAILED, N_FEATS, atom_labels, build_graph,
                     path_states, state_features, tactic_token)

INF = 10 ** 9


def pack(name: str, split: str, out: Path | None = None) -> Path:
    out = out or (config.DATA_ROOT / name / "packed" / f"{split}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    T = {k: [] for k in ("n_tok", "tok_type", "edges", "edges_off", "prem", "prem_off", "label",
                         "dead", "depth", "family", "tid")}
    S = {k: [] for k in ("thm", "x", "main", "legal", "legal_off", "opt", "opt_off", "n_legal", "n_opt")}
    tok_off = edge_off = prem_off = 0
    leg_off = opt_off = 0
    T_tok_off, T_edge_off, T_prem_off = [], [], []
    S_leg_off, S_opt_off, S_x_off = [], [], []
    x_off = 0
    for ti, rec in enumerate(iter_records(name, split)):
        g = build_graph(rec)
        n = g.n_tok
        T_tok_off.append(tok_off); T_edge_off.append(edge_off); T_prem_off.append(prem_off)
        T["n_tok"].append(n); T["tok_type"].append(g.tok_type)
        T["edges"].append(g.edges); edge_off += len(g.edges)
        prem = np.full((n, 3), -1, np.int16)
        for t, ps in g.hyp_prem.items():
            prem[t, :len(ps)] = ps
        T["prem"].append(prem)
        lab = atom_labels(g, rec)
        T["label"].append(lab)
        T["dead"].append(lab == 0)
        T["depth"].append(rec["depth"]); T["family"].append(rec["family"]); T["tid"].append(rec["tid"])
        tok_off += n
        size = rec["size"]
        for goals, anc, tac in path_states(rec):
            children = rec["table"][goals[0]]
            x, main = state_features(g, goals, anc, children, set())
            cost = {t2: 1 + sum(size[k] if size[k] is not None else INF for k in new)
                    for t2, new in children.items()}
            best = min(cost.values())
            legal = np.array(sorted(tactic_token(g, t2) for t2 in children), np.int16)
            opt = np.array(sorted(tactic_token(g, t2) for t2, c in cost.items() if c == best), np.int16)
            S["thm"].append(ti); S["main"].append(main)
            S["x"].append(np.round(x * 255).astype(np.uint8)); S_x_off.append(x_off); x_off += n
            S["legal"].append(legal); S_leg_off.append(leg_off); leg_off += len(legal)
            S["opt"].append(opt); S_opt_off.append(opt_off); opt_off += len(opt)
            S["n_legal"].append(len(legal)); S["n_opt"].append(len(opt))
    arrays = {
        "t_n_tok": np.array(T["n_tok"], np.int32), "t_tok_off": np.array(T_tok_off, np.int64),
        "t_tok_type": np.concatenate(T["tok_type"]), "t_edges": np.concatenate(T["edges"]),
        "t_edge_off": np.array(T_edge_off + [edge_off], np.int64),
        "t_prem": np.concatenate(T["prem"]), "t_label": np.concatenate(T["label"]),
        "t_dead": np.concatenate(T["dead"]), "t_depth": np.array(T["depth"], np.int16),
        "t_family": np.array(T["family"]), "t_tid": np.array(T["tid"]),
        "s_thm": np.array(S["thm"], np.int32), "s_main": np.array(S["main"], np.int16),
        "s_x": np.concatenate(S["x"]), "s_x_off": np.array(S_x_off, np.int64),
        "s_legal": np.concatenate(S["legal"]), "s_legal_off": np.array(S_leg_off + [leg_off], np.int64),
        "s_opt": np.concatenate(S["opt"]), "s_opt_off": np.array(S_opt_off + [opt_off], np.int64),
        "s_n_legal": np.array(S["n_legal"], np.int16), "s_n_opt": np.array(S["n_opt"], np.int16),
    }
    tmp = out.with_suffix(".part.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out)
    return out


class Packed:
    def __init__(self, path: Path):
        z = np.load(path)
        self.__dict__.update({k: z[k] for k in z.files})
        self.n_states = len(self.s_thm)
        self.n_theorems = len(self.t_n_tok)

    def batch(self, idx, rng: np.random.Generator, fail_aug: float, device):
        thm = self.s_thm[idx]
        ns = self.t_n_tok[thm]
        B, N = len(idx), int(ns.max())
        x = np.zeros((B, N, N_FEATS), np.float32)
        ttype = np.zeros((B, N), np.int64)
        mask = np.zeros((B, N), bool)
        legal = np.zeros((B, N), bool); opt = np.zeros((B, N), bool)
        alab = np.full((B, N), -1, np.int64)
        e_b, e_s, e_d, e_t = [], [], [], []
        for i, (s, t, n) in enumerate(zip(idx, thm, ns)):
            to = self.t_tok_off[t]
            xs = self.s_x[self.s_x_off[s]: self.s_x_off[s] + n].astype(np.float32) / 255.0
            dead = self.t_dead[to: to + n]
            if fail_aug and dead.any() and rng.random() < fail_aug:
                f = dead & (rng.random(n) < 0.4)
                xs[:, F_FAILED] = f
                prem = self.t_prem[to: to + n].astype(np.int64)
                hit = np.where(prem >= 0, f[np.clip(prem, 0, None)], False)
                xs[:, F_PREM_FAILED] = hit.sum(1) / 3.0
            x[i, :n] = xs
            ttype[i, :n] = self.t_tok_type[to: to + n]
            mask[i, :n] = True
            legal[i, self.s_legal[self.s_legal_off[s]: self.s_legal_off[s + 1]]] = True
            opt[i, self.s_opt[self.s_opt_off[s]: self.s_opt_off[s + 1]]] = True
            alab[i, :n] = self.t_label[to: to + n]
            ed = self.t_edges[self.t_edge_off[t]: self.t_edge_off[t + 1]]
            e_b.append(np.full(len(ed), i)); e_s.append(ed[:, 0]); e_d.append(ed[:, 1]); e_t.append(ed[:, 2])
            if n < N:
                pad = np.arange(n, N)
                e_b.append(np.full(len(pad), i)); e_s.append(pad); e_d.append(pad); e_t.append(np.full(len(pad), E_SELF))
        dev = lambda a, dt=None: torch.from_numpy(a).to(device, non_blocking=True)
        etype = torch.zeros(B, N, N, dtype=torch.long, device=device)
        etype[dev(np.concatenate(e_b).astype(np.int64)), dev(np.concatenate(e_s).astype(np.int64)),
              dev(np.concatenate(e_d).astype(np.int64))] = dev(np.concatenate(e_t).astype(np.int64))
        return {"x": dev(x), "ttype": dev(ttype), "etype": etype, "tok_mask": dev(mask),
                "legal": dev(legal), "optimal": dev(opt), "atom_label": dev(alab),
                "main": dev(self.s_main[idx].astype(np.int64))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="lcgrt-v1")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    a = ap.parse_args()
    for s in a.splits:
        t = time.time(); p = pack(a.data, s); z = Packed(p)
        print(json.dumps({"split": s, "theorems": z.n_theorems, "states": z.n_states,
                          "mb": round(p.stat().st_size / 2 ** 20, 1), "seconds": round(time.time() - t, 1),
                          "states_with_choice": int((z.s_n_legal > 1).sum()),
                          "states_where_choice_matters": int((z.s_n_opt < z.s_n_legal).sum())}), flush=True)


if __name__ == "__main__":
    main()

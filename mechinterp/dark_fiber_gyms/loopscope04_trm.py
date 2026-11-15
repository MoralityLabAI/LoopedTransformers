#!/usr/bin/env python3
"""LOOPSCOPE-04 local gym: audited TRM sphere-fiber swaps and small cache baselines.

This uses the exact pinned PureSudokuTRM decoder from LOOPSCOPE-02. The
cache comparison is a capacity-matched *cellwise ridge diagnostic*, not a
replication of the full 97-token C5 transformer from LOOPSCOPE-03.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .core import (
    project_visible, row_basis, sha256_file, trm_manifold_swap, write_json,
)


def import_trm(root: Path):
    sys.path.insert(0, str(root.resolve()))
    from mechinterp.loopscope02.scripts import trm_sudoku as TS
    return TS


def canonical_digit_relabel(question: str) -> str:
    mapping = {}
    next_id = 1
    chars = []
    for c in question:
        if c in ".0":
            chars.append(".")
        else:
            if c not in mapping:
                mapping[c] = str(next_id)
                next_id += 1
            chars.append(mapping[c])
    return "".join(chars)


def read_boards(path: Path, limit: int = 0) -> list[dict]:
    found = []
    with path.open(newline="", encoding="utf-8") as handle:
        rd = csv.DictReader(handle)
        for idx,row in enumerate(rd):
            q = row.get("question") or row.get("puzzle")
            a = row.get("answer") or row.get("solution")
            if not q or not a:
                raise ValueError(f"{path}:{idx+2}: expected question/answer columns")
            q = q.replace("0", ".")
            if len(q)!=81 or len(a)!=81 or any(c not in ".123456789" for c in q) or any(c not in "123456789" for c in a):
                raise ValueError(f"{path}:{idx+2}: invalid 81-cell puzzle or answer")
            if any(q[i] not in (".", a[i]) for i in range(81)):
                raise ValueError("clue/answer mismatch")
            found.append({"question":q,"answer":a,"id":sha256(q.encode()).hexdigest()})
            if limit and len(found)>=limit:
                break
    if not found:
        raise ValueError(f"empty dataset: {path}")
    return found


def audit_boards(train: list[dict], test: list[dict], excluded: list[Path]) -> dict:
    banned_exact, banned_relabel = set(), set()
    for path in excluded:
        # Reads exclusions completely, including rows beyond pilot limit.
        for r in read_boards(path):
            banned_exact.add(r["question"])
            banned_relabel.add(canonical_digit_relabel(r["question"]))
    tr = [r["question"] for r in train]
    te = [r["question"] for r in test]
    if len(set(tr)) != len(tr) or len(set(te)) != len(te):
        raise ValueError("duplicate puzzle in a split")
    can_tr = {canonical_digit_relabel(q) for q in tr}
    can_te = {canonical_digit_relabel(q) for q in te}
    if len(can_tr) != len(tr) or len(can_te) != len(te):
        raise ValueError("digit-relabel duplicate within a split")
    if can_tr & can_te:
        raise ValueError("digit-relabel overlap between train/eval")
    if any(q in banned_exact or canonical_digit_relabel(q) in banned_relabel for q in tr+te):
        raise ValueError("puzzle overlaps excluded prior experimental data")
    return {"train":len(tr),"test":len(te),"excluded_files":{str(p):sha256_file(p) for p in excluded},
            "audit_level":"exact plus digit-relabel, not full Sudoku row/column/group automorphisms",
            "train_sha256": sha256("\n".join(tr).encode()).hexdigest(),
            "test_sha256": sha256("\n".join(te).encode()).hexdigest()}


def load_model(TS, checkpoint_file: Path, tag: str, device: str):
    expected = TS.TRM_CHECKPOINTS[tag]["sha256"]
    found = sha256_file(checkpoint_file)
    if found != expected:
        raise AssertionError(f"{tag}: checkpoint SHA mismatch {found} != {expected}")
    sd = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model = TS.PureSudokuTRM.from_state_dict(sd, dtype=torch.float32).to(device).eval()
    return model, found


@torch.inference_mode()
def solve_logits(model, zh):
    return model.digit_logits(zh).argmax(dim=-1)[0].add(1).cpu().tolist()


@torch.inference_mode()
def run_swap_case(TS, model, row, *, patch_step: int, horizon: int, seed: int, sigma: float):
    device = model.H_init.device
    x = model.encode_inputs(TS.encode_questions([row["question"]], device))
    zh_a, zl_a = model.init_states(1)
    zh_b, zl_b = model.init_states(1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    def perturb(z):
        noise = torch.randn(tuple(z.shape), generator=g, dtype=torch.float32).to(device)
        return z + sigma * z.float().square().mean().sqrt() * noise
    zh_b, zl_b = perturb(zh_b), perturb(zl_b)
    for _ in range(patch_step):
        zh_a,zl_a = model.act_step(zh_a,zl_a,x)
        zh_b,zl_b = model.act_step(zh_b,zl_b,x)
    h_swap, cert = trm_manifold_swap(zh_a, zh_b, model.lm_head.weight, board_start=model.puzzle_emb_len)
    g_rand = torch.Generator(device="cpu").manual_seed(seed+9137)
    rand = torch.randn(tuple(zh_a.shape),generator=g_rand,dtype=torch.float32).to(device)
    h_random, cert_random = trm_manifold_swap(zh_a, rand, model.lm_head.weight,
                                              board_start=model.puzzle_emb_len)
    h_self, cert_self = trm_manifold_swap(zh_a, zh_a, model.lm_head.weight,
                                         board_start=model.puzzle_emb_len)
    states = {"clean":(zh_a,zl_a), "swapped":(h_swap,zl_a.clone()),
              "random_orth":(h_random,zl_a.clone()), "self_control":(h_self,zl_a.clone())}
    # Crucial: at the *intervention instant* all 11 logits per board cell match,
    # including the 2 unused readout classes.
    readouts = {name: model.lm_head(zh)[:,model.puzzle_emb_len:,:].float()
                for name,(zh,zl) in states.items()}
    errors = {name:float((q-readouts["clean"]).abs().max().item()) for name,q in readouts.items()}
    if max(errors.values())>1e-3:
        raise AssertionError(f"pre-intervention full-head mismatch: {errors}")
    first_step = {}
    for step in range(patch_step+1,horizon+1):
        for name,(zh,zl) in states.items():
            states[name] = model.act_step(zh,zl,x)
        if step == patch_step+1:
            ref = model.digit_logits(states["clean"][0])
            first_step = {name:float((model.digit_logits(zh)-ref).norm().item())
                          for name,(zh,zl) in states.items()}
    answer = [int(c) for c in row["answer"]]
    result = {}
    for name,(zh,zl) in states.items():
        pred = solve_logits(model,zh)
        result[name] = {"exact":pred==answer,"correct_cells":sum(a==b for a,b in zip(pred,answer)),
                        "first_successor_logit_l2":first_step[name],
                        "pre_intervention_full_head_max_error":errors[name]}
    return {"case_id":row["id"],"patch_step":patch_step, "horizon":horizon,
            "sigma":sigma, "board_start":model.puzzle_emb_len, "head_rank":cert["head_rank"],
            "orthogonal_dim":cert["orthogonal_dim"], "certificates":{**cert, "random":cert_random,"self":cert_self},
            "arms":result}


@torch.inference_mode()
def collect_cache_rows(TS, model, boards: list[dict], n_steps: int, basis: torch.Tensor):
    """Construct matched-width snapshot/trace features and full board-cell hidden targets.

    Diagnostic target = [zH_board orthogonal, zL_board] (2*512 dimensions).
    The 16 prefix tokens are outside this baseline and require a future
    full-state sequence-level reconstructor.
    """
    X_now, X_trace, Y = [], [], []
    device = model.H_init.device
    dtype = model.H_init.dtype
    eye10 = np.eye(10,dtype=np.float32)
    eye9 = np.eye(9,dtype=np.float32)
    for row in boards:
        xb = model.encode_inputs(TS.encode_questions([row["question"]],device))
        zh,zl = model.init_states(1)
        v_prev = (zh[:,model.puzzle_emb_len:,:].float() @ basis).squeeze(0).cpu().numpy()
        clue = np.array([0 if c=="." else int(c) for c in row["question"]], dtype=np.int64)
        base = np.concatenate([eye10[clue],eye9[np.arange(81)//9],
                               eye9[np.arange(81)%9]],axis=1)
        for t in range(1,n_steps+1):
            zh,zl = model.act_step(zh,zl,xb)
            board = zh[:,model.puzzle_emb_len:,:].float()
            v_t = (board @ basis).squeeze(0).cpu().numpy()
            orth = (board - project_visible(board,basis)).squeeze(0).cpu().numpy()
            zl_board = zl[:,model.puzzle_emb_len:,:].float().squeeze(0).cpu().numpy()
            step = np.full((81,1),t/n_steps,dtype=np.float32)
            # Identical feature width for both regressors. Snapshot uses V_t
            # twice; trace uses V_{t-1} as its additional informative block.
            prefix = np.concatenate([base, step, v_t],axis=1)
            X_now.append(np.concatenate([prefix,v_t],axis=1))
            X_trace.append(np.concatenate([prefix,v_prev],axis=1))
            Y.append(np.concatenate([orth,zl_board],axis=1))
            v_prev=v_t
    return np.concatenate(X_now),np.concatenate(X_trace),np.concatenate(Y)


def ridge_fit(x: np.ndarray,y: np.ndarray,alpha:float) -> np.ndarray:
    # Fit intercept without penalizing it; float64 solves 52 x 52 system.
    x64=x.astype(np.float64)
    y64=y.astype(np.float64)
    xb=np.column_stack([np.ones((len(x64),1)),x64])
    penalty=np.eye(xb.shape[1])*alpha
    penalty[0,0]=0.0
    return np.linalg.solve(xb.T@xb+penalty,xb.T@y64)


def ridge_predict(x,w):
    return np.column_stack([np.ones((len(x),1)),x.astype(np.float64)])@w


@torch.inference_mode()
def cache_compare(TS, model, train, test, steps, alpha):
    basis = row_basis(model.lm_head.weight).to(model.H_init.device)
    xn,xt,yt = collect_cache_rows(TS,model,train,steps,basis)
    vn,vt,yv = collect_cache_rows(TS,model,test,steps,basis)
    m_now = ridge_fit(xn,yt,alpha)
    m_trace = ridge_fit(xt,yt,alpha)
    den = max(float(((yv-yv.mean(0))**2).sum()),1e-12)
    result = {}
    for name,x,w in (("snapshot",vn,m_now),("visible_trace",vt,m_trace)):
        mse = float(((yv-ridge_predict(x,w))**2).mean())
        r2 = 1.0 - float(((yv-ridge_predict(x,w))**2).sum())/den
        result[name]={"mse":mse,"r2":r2,"feature_width":int(x.shape[1]),"parameters":int(w.size),
                      "training_cells":len(xn),"validation_cells":len(vn)}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--checkpoint-file",type=Path,required=True)
    p.add_argument("--tag",choices=("gaoxin492","yagizdevre"),default="gaoxin492")
    p.add_argument("--mode",choices=("swap","cache"),default="swap")
    p.add_argument("--train-csv",type=Path,help="Required for cache mode")
    p.add_argument("--eval-csv",type=Path,required=True,help="Fresh puzzles, disjoint from prior runs")
    p.add_argument("--exclude-csv",type=Path,action="append",default=[],
                   help="Prior evaluated puzzles; repeat for each data file")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--limit",type=int,default=4,help="pilot limit; 0 means all")
    p.add_argument("--train-limit",type=int,default=24)
    p.add_argument("--patch-step",type=int,default=4)
    p.add_argument("--horizon",type=int,default=16)
    p.add_argument("--steps",type=int,default=8,help="cache collection depth")
    p.add_argument("--alpha",type=float,default=50.0)
    p.add_argument("--sigma",type=float,default=0.35)
    p.add_argument("--seed",type=int,default=20261115)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args(argv)
    if args.limit<0 or args.train_limit<1 or args.steps<2 or not(1<=args.patch_step<args.horizon):
        p.error("invalid limits/steps/horizon")
    if args.mode=="cache" and args.train_csv is None:
        p.error("--train-csv required in cache mode")
    if not args.exclude_csv:
        p.error("explicit --exclude-csv of previously evaluated boards is required")
    evaluation=read_boards(args.eval_csv,args.limit)
    train=read_boards(args.train_csv,args.train_limit) if args.mode=="cache" else []
    audit=audit_boards(train,evaluation,args.exclude_csv)
    manifest={"schema":"darkfibers.loopscope04.local.v1","audit":audit,
              "mode":args.mode,"checkpoint_tag":args.tag,
              "checkpoint_path":str(args.checkpoint_file),
              "checkpoint_sha256":sha256_file(args.checkpoint_file),
              "code_sha256":sha256_file(Path(__file__)), "device":args.device,
              "settings":{k:getattr(args,k) for k in ("patch_step","horizon","steps","alpha","sigma","seed")}}
    if args.dry_run:
        print(json.dumps(manifest,indent=2))
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable: use --device cpu")
    TS=import_trm(args.repo_root)
    start=time.perf_counter()
    model,pin=load_model(TS,args.checkpoint_file,args.tag,args.device)
    if args.mode=="cache":
        manifest["cache"] = cache_compare(TS,model,train,evaluation,args.steps,args.alpha)
        manifest["scope"]="cellwise ridge snapshot-vs-one-step-visible-history; not full C5"
    else:
        manifest["rows"]=[]
        for index,row in enumerate(evaluation,1):
            seed=args.seed ^ int(row["id"][:8],16)
            result=run_swap_case(TS,model,row,patch_step=args.patch_step,horizon=args.horizon,
                                 seed=seed,sigma=args.sigma)
            manifest["rows"].append(result)
            print(f"{index}/{len(evaluation)} {row['id'][:12]}: clean {result['arms']['clean']['exact']} swap {result['arms']['swapped']['exact']}",flush=True)
    manifest["elapsed_seconds"]=round(time.perf_counter()-start,3)
    if args.device.startswith("cuda"):
        manifest["peak_vram_bytes"]=torch.cuda.max_memory_allocated()
    write_json(args.out,manifest)
    print("wrote",args.out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

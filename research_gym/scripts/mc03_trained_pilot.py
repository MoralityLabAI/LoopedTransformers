"""MC-03: bounded CPU-trained native LoopedChase development pilot.

The fixed recipe was recorded before training. This creates new checkpoints;
it never claims to reproduce the external Q4/Q5 checkpoint measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from research_gym.analysis.mechanistic_certificate import Statement, digest, encode, relation
from research_gym.analysis.intervention_certificate import select_pairs, validate_manifest
from research_gym.scripts import loopscope_semantic_echo as native
from research_gym.scripts.loopscope_toy import LoopedChase
from research_gym.scripts.loopscope_intervention_certificate import execute_pair, summarize
from research_gym.scripts.mechanistic_certificate import file_hash

ROOT = Path(__file__).resolve().parents[2]
PLAN_COMMIT = "92c1613c052dd1c04b8ac1b236f2713b0feaec5f"
MODEL_SHA256 = "0e0d5108de88f5415e6953482646f4679d1087b479b46621e6ca64679c1b1400"
NATIVE_SHA256 = "a7e95d61a3814f0b44b556eecf52e8f5484867535af287cb27be3ec153119e2f"
SEEDS = (201, 202)
RECIPE = dict(n_nodes=6, d=48, layers=1, heads=4, loops=4,
              max_train_depth=4, fixed_depth=True, lam=1.0, steps=3000,
              batch=64, lr=0.001, warmup=100, weight_decay=0.01,
              betas=[0.9, 0.98], gradient_clip=1.0, wall_limit_seconds=240,
              cpu_threads=1, study="MC03_CPU_TRAINED_PILOT")


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def graph_split() -> dict[str, list[tuple[int, ...]]]:
    graphs = sorted(itertools.permutations(range(6)),
                    key=lambda g: digest(["mc03.graph_split.v1", list(g)]))
    return {"train": graphs[:480], "calibration": graphs[480:600], "test": graphs[600:]}


def render(graphs: torch.Tensor, starts: torch.Tensor, orders: torch.Tensor,
           steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the native token contract for a caller-declared graph panel."""
    if graphs.ndim != 2 or starts.shape != (len(graphs),) or orders.shape != graphs.shape:
        raise ValueError("graph/start/order shapes disagree")
    b, n = graphs.shape
    if any(t.dtype != torch.long for t in (graphs, starts, orders)) or not 1 <= steps <= 64:
        raise ValueError("integer tensors and 1..64 steps required")
    target = torch.arange(n).expand(b, n)
    if not torch.equal(graphs.sort(-1).values, target) or not torch.equal(orders.sort(-1).values, target):
        raise ValueError("graphs and edge orders must be permutations")
    if bool((starts < 0).any() or (starts >= n).any()):
        raise ValueError("start is outside graph")
    edges = torch.stack((orders, graphs.gather(1, orders)), -1).reshape(b, 2*n)
    tokens = torch.cat((edges, starts[:, None]), 1)
    labels, cur = [], starts
    for _ in range(steps):
        cur = graphs.gather(1, cur[:, None])[:, 0]
        labels.append(cur)
    return tokens, torch.stack(labels, 1)


def sampled_batch(graphs: torch.Tensor, count: int, rng: torch.Generator, steps: int):
    chosen = graphs[torch.randint(len(graphs), (count,), generator=rng)]
    starts = torch.randint(graphs.shape[1], (count,), generator=rng)
    orders = torch.rand(count, graphs.shape[1], generator=rng).argsort(-1)
    return render(chosen, starts, orders, steps)


def heldout_panel(graphs, seed: int = 940311):
    """Every start on every test graph, then fixed edge orders, independent of models."""
    gs = torch.tensor(graphs, dtype=torch.long).repeat_interleave(6, dim=0)
    starts = torch.arange(6).repeat(len(graphs))
    rng = torch.Generator().manual_seed(seed)
    orders = torch.rand(len(gs), 6, generator=rng).argsort(-1)
    tokens, targets = render(gs, starts, orders, 16)
    # The existing MC-02 selector is quadratic in its input panel size.
    # Fix its 64 task items once, before training or any endpoint observation.
    selected = torch.randperm(len(gs), generator=rng)[:64]
    return gs, starts, tokens, targets, selected


def lr_at(step: int) -> float:
    if step < RECIPE["warmup"]:
        return RECIPE["lr"] * (step + 1) / RECIPE["warmup"]
    phase = (step - RECIPE["warmup"]) / (RECIPE["steps"] - RECIPE["warmup"])
    return RECIPE["lr"] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * phase)))


def source_identity() -> dict:
    paths = [Path(__file__), ROOT/"research_gym/scripts/loopscope_toy.py",
             ROOT/"research_gym/scripts/loopscope_semantic_echo.py",
             ROOT/"research_gym/analysis/semantic_echo.py",
             ROOT/"research_gym/analysis/mechanistic_certificate.py",
             ROOT/"research_gym/analysis/intervention_certificate.py",
             ROOT/"research_gym/scripts/loopscope_intervention_certificate.py"]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    if hashes["research_gym/scripts/loopscope_toy.py"] != MODEL_SHA256:
        raise ValueError("original model source pin failed")
    if hashes["research_gym/scripts/loopscope_semantic_echo.py"] != NATIVE_SHA256:
        raise ValueError("original adapter source pin failed")
    return hashes


def train_seed(seed: int, directory: Path, graphs: torch.Tensor) -> dict:
    """Single fixed trajectory; evaluation is deferred until training is complete."""
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = LoopedChase(6, 48, 1, 4)
    config = {**RECIPE, "seed": seed, "training_status": "TRAINED_CPU_PILOT"}
    initial = directory/"initial.pt"
    torch.save({"model": model.state_dict(), "cfg": {**config, "training_status": "RANDOM_INITIALIZATION"}}, initial)
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                           betas=tuple(RECIPE["betas"]), weight_decay=RECIPE["weight_decay"])
    rng = torch.Generator().manual_seed(seed * 1000003)
    model.train()
    start = time.monotonic()
    records = []
    status = "COMPLETED"
    for step in range(RECIPE["steps"]):
        if time.monotonic() - start >= RECIPE["wall_limit_seconds"]:
            status = "BUDGET_STOP"
            break
        tokens, target = sampled_batch(graphs, RECIPE["batch"], rng, RECIPE["loops"])
        rate = lr_at(step)
        opt.param_groups[0]["lr"] = rate
        logits = model.readout(model(tokens, 4))
        # All four visits have unit loss weight under the frozen lambda=1 recipe.
        loss = F.cross_entropy(logits.reshape(-1, 6), target.T.reshape(-1)) * 4
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise RuntimeError("nonfinite gradient")
        opt.step()
        records.append({"update": step+1, "loss": float(loss.detach()), "lr": rate,
                        "gradient_norm_before_clip": grad_norm})
        if (step+1) % 250 == 0:
            print(json.dumps({"seed":seed,"update":step+1,"training_loss":records[-1]["loss"],
                              "elapsed_seconds":round(time.monotonic()-start,2)}), flush=True)
    elapsed = time.monotonic() - start
    final = directory/"final.pt"
    config["completed_updates"] = len(records)
    config["budget_status"] = status
    torch.save({"model": model.state_dict(), "cfg": config}, final)
    # Optimizer and RNG are kept separately from the inference-compatible checkpoint.
    torch.save({"optimizer": opt.state_dict(), "batch_rng": rng.get_state(),
                "torch_rng": torch.get_rng_state(), "completed_updates":len(records)}, directory/"training_state.pt")
    dump(directory/"losses.json", records)
    receipt = {"seed":seed,"recipe":config,"status":status,"updates":len(records),
               "wall_seconds":elapsed,"params":sum(p.numel() for p in model.parameters()),
               "initial_sha256":file_hash(initial),"final_sha256":file_hash(final),
               "losses_sha256":file_hash(directory/"losses.json"),
               "training_state_sha256":file_hash(directory/"training_state.pt"),
               "last_loss":records[-1]["loss"] if records else None}
    dump(directory/"training.json", receipt)
    return receipt


def trace_metrics(predictions: np.ndarray, truth: np.ndarray, graphs, starts) -> dict:
    """Cross-check complete observed trajectories with the MC-01 R1CS relation."""
    if predictions.shape != truth.shape or predictions.ndim != 2 or predictions.shape[0] != 16:
        raise ValueError("expected sixteen visit rows of equally shaped labels")
    same = predictions == truth
    circuit = relation(6, 16)
    accepts = []
    first_error = []
    for i, (g, start) in enumerate(zip(graphs, starts)):
        statement = Statement(tuple(map(int,g)), int(start), 16, int(truth[-1,i]))
        observed = (int(start),) + tuple(map(int,predictions[:,i]))
        valid = not circuit.failures(encode(statement, observed))
        if valid != bool(same[:,i].all()):
            raise RuntimeError("R1CS verdict differs from exact oracle path equality")
        accepts.append(valid)
        wrong = np.flatnonzero(~same[:,i])
        first_error.append(int(wrong[0]+1) if len(wrong) else None)
    return {"items":same.shape[1],"accuracy_by_visit":same.mean(1).tolist(),
            "correct_by_visit":same.sum(1).tolist(),
            "full_trace_4_accept_count":int(same[:4].all(0).sum()),
            "full_trace_16_accept_count":sum(accepts),
            "first_wrong_visit":first_error,
            "learning_gate_depth4":float(same[3].mean())>=0.90,
            "proof_system":"exact Python R1CS relation, no cryptographic proof"}


@torch.inference_mode()
def evaluate_checkpoint(path: Path, pin: str, directory: Path, panel, *, arm: str) -> dict:
    if file_hash(path) != pin:
        raise ValueError("checkpoint changed after training receipt")
    directory.mkdir(parents=True, exist_ok=False)
    model, cfg = native.load_model(path, smoke=False, seed=cfg_seed(path), device="cpu", dtype="float32")
    gs, starts, tokens, targets, selected = panel
    all_logits = []
    for begin in range(0,len(tokens),64):
        tok = tokens[begin:begin+64]
        embedding, states = native.native_trace(model,tok,16)
        native.verify_native_trace(model,tok,states,rtol=2e-5,atol=2e-6)
        logits = model.readout(states[1:,:,-1])
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite evaluation logits")
        all_logits.append(logits.cpu().numpy())
    logits = np.concatenate(all_logits,axis=1)
    predictions = logits.argmax(-1)
    metrics = trace_metrics(predictions,targets.T.numpy(),gs.tolist(),starts.tolist())
    np.savez_compressed(directory/"heldout.npz",tokens=tokens.numpy(),graphs=gs.numpy(),
                        starts=starts.numpy(),targets=targets.T.numpy(),logits=logits,
                        predictions=predictions,intervention_item_indices=selected.numpy())
    visits = [1,2,4,8,12]
    tok = tokens[selected]
    embedding, states = native.native_trace(model,tok,16)
    parity = native.verify_native_trace(model,tok,states,rtol=2e-5,atol=2e-6)
    endpoints = model.readout(states[visits,:,-1]).argmax(-1).tolist()
    manifest = select_pairs(gs[selected].tolist(),starts[selected].tolist(),endpoints,visits,
                            horizon=4,max_pairs=32,seed=940312)
    dump(directory/"selection.json",manifest)
    manifest_pin = file_hash(directory/"selection.json")
    pairs, arrays = [], {}
    for i,row in enumerate(manifest["selected_pairs"]):
        result, values = execute_pair(model,native,states,embedding,row,horizon=4,
                                     seed=940313+i,random_controls=8)
        pairs.append(result)
        arrays.update({f"pair_{i}_{k}":v for k,v in values.items()})
    if file_hash(directory/"selection.json") != manifest_pin:
        raise RuntimeError("selection changed during replay")
    validate_manifest(manifest)
    np.savez_compressed(directory/"replays.npz",**arrays)
    np.savez_compressed(directory/"clean_trace.npz",states=states.numpy(),
                        embeddings=embedding.numpy(),tokens=tok.numpy())
    summary = summarize(pairs,status="MEASURED_CHECKPOINT",visits=visits)
    scores = {}
    for name in (["cursor_transfer","sham","native_donor_replay","cursor_constant_shift"]+
                 [f"random_cursor_{i}" for i in range(8)]):
        subset = [next(a for a in row["arms"] if a["name"]==name) for row in pairs]
        scores[name] = {"pairs":len(subset),
            "trace_match":sum(a["counterfactual_trace_match"] for a in subset),
            "future_match":sum(a["counterfactual_future_match"] for a in subset),
            "receiver_future_preserved":sum(a["receiver_oracle_future_match"] for a in subset)}
    result = {"study":"MC03_CPU_TRAINED_PILOT","arm":arm,"checkpoint_sha256":pin,
              "config":cfg,"task_metrics":metrics,"native_parity":parity,
              "causal_summary":summary,"arm_counts":scores,"pairs":pairs,
              "coverage":{k:manifest[k] for k in ("receiver_slots","endpoint_correct_slots",
                        "eligible_receiver_slots","candidate_ordered_pairs")},
              "q4_q5_replication":False,"cryptographic_source_binding":False,
              "evidence":"finite empirical learned-model pilot" if arm=="trained" else "random-initialization control",
              "file_sha256":{n:file_hash(directory/n) for n in (
                  "heldout.npz","selection.json","replays.npz","clean_trace.npz")}}
    dump(directory/"metrics.json",result)
    return {"arm":arm,"checkpoint_sha256":pin,"task_metrics":metrics,
            "causal_summary":summary,"arm_counts":scores,"coverage":result["coverage"]}


def cfg_seed(path: Path) -> int:
    return int(torch.load(path,map_location="cpu",weights_only=True)["cfg"]["seed"])


def run(out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; no automatic retraining/resume")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    source = source_identity()
    splits = graph_split()
    panel = heldout_panel(splits["test"])
    manifest = {"study":"MC03_CPU_TRAINED_PILOT","plan_commit":PLAN_COMMIT,
                "recipe":RECIPE,"seeds":SEEDS,"splits":splits,
                "intervention_item_indices":panel[-1].tolist(),"source_sha256":source,
                "python":platform.python_version(),"torch":str(torch.__version__),
                "numpy":np.__version__,"created_utc":datetime.now(timezone.utc).isoformat()}
    dump(out/"frozen_manifest.json",manifest)
    manifest_pin=file_hash(out/"frozen_manifest.json")
    train_graphs=torch.tensor(splits["train"],dtype=torch.long)
    trained=[]
    for seed in SEEDS:
        trained.append(train_seed(seed,out/f"seed_{seed}",train_graphs))
    # All seeds finish their fixed training budget before any held-out score is read.
    evaluations=[]
    for receipt in trained:
        directory=out/f"seed_{receipt['seed']}"
        for arm,filename,pin in (("initial","initial.pt",receipt["initial_sha256"]),
                                 ("trained","final.pt",receipt["final_sha256"])):
            evaluation=evaluate_checkpoint(directory/filename,pin,directory/arm,panel,arm=arm)
            evaluations.append({"seed":receipt["seed"],**evaluation})
            print(json.dumps({"seed":receipt["seed"],"arm":arm,
                 "accuracy_by_visit":evaluation["task_metrics"]["accuracy_by_visit"],
                 "trace16":evaluation["task_metrics"]["full_trace_16_accept_count"],
                 "transfer":evaluation["arm_counts"]["cursor_transfer"]}),flush=True)
    if file_hash(out/"frozen_manifest.json")!=manifest_pin or source_identity()!=source:
        raise RuntimeError("manifest or execution sources changed during experiment")
    result={"study":"MC03_CPU_TRAINED_PILOT","manifest_sha256":manifest_pin,
            "training":trained,"evaluations":evaluations,"q4_q5_replication":False,
            "source_binding":"trusted local execution with exact source/checkpoint hashes"}
    dump(out/"summary.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.out)

"""OLR-01C: train/evaluate tiny guard scorers on native Lean source labels.

This is an exploratory source-only training pilot. Same-structure paired worlds
share a root lineage. Only source ('train') outcomes may update model weights;
all development and OOD predictions are made before their native replay.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import re
import time
import platform
from pathlib import Path

import torch

from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .olr_guards import SCHEMA as SOURCE_SCHEMA, GuardWorld, certified_standalone, proof_attempt
from .olr_guard_model import GuardScorer, ranking_loss, choose, exact_arithmetic_reference

SCHEMA = "ppt.olr-guard-fit.v1"
SEED_ROSTER = (17, 29, 43)
ARMS = ("cross", "isolated", "direct", "frozen_cross")
CONTROLS = ("fixed_r0", "exact_arithmetic")
STMT = re.compile(
    r"^∀ \(P : Prop\) \(r0 : \((\d+) \+ (\d+) = (\d+)\) → P\) "
    r"\(r1 : \((\d+) \+ (\d+) = (\d+)\) → P\), P$"
)


def load_bank(directory: Path):
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SOURCE_SCHEMA or manifest.get("model_training_uses_train_split_only") is not True:
        raise ValueError("not a verifier-grounded OLR source bank")
    shards = {}
    for split in ("train", "dev", "ood"):
        info=manifest["splits"][split]
        path=directory/(split+".jsonl")
        if sha256_file(path)!=info["sha256"]:
            raise ValueError("tampered native source shard")
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != info["worlds"]:
            raise ValueError("native bank lost a world")
        shards[split]=rows
    source = {r["pair_id"] for r in shards["train"]}
    if any(r["pair_id"] in source for split in ("dev", "ood") for r in shards[split]):
        raise ValueError("source lineage leaked into evaluation")
    ids = [r["root"] for rows in shards.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate theorem root across splits")
    for split,rows in shards.items():
        grouped=defaultdict(list)
        for r in rows:
            grouped[r["pair_id"]].append(r)
            if (r["split"] != split or len(r["menu"]) != 2 or
                    r["menu"] != ["apply r0","apply r1"] or
                    len(r["features"]) != 2 or r["labels"] not in ([1,0],[0,1])):
                raise ValueError("invalid source root / candidate label")
            if len(r["attempts"]) != 2 or sum(a["kernel_ok"] for a in r["attempts"]) != 1:
                raise ValueError("missing native proof receipts")
        if any(sorted(r["world"] for r in group) != [0,1]
               for group in grouped.values()):
            raise ValueError("unpaired worlds in registered root family")
    return manifest,shards


def reconstruct(row: dict):
    """Reconstruction for native theorem execution, outside model input."""
    match=STMT.fullmatch(row["statement"])
    if match is None:
        raise ValueError("unrecognized bounded guard statement")
    a,b,c,d,e,f=map(int,match.groups())
    if (a,b)!=(d,e):
        raise ValueError("root guard operands disagree")
    return GuardWorld(row["split"],row["pair_id"],row["world"],a,b,(c,f))


def tensors(rows):
    x=torch.tensor([r["features"] for r in rows],dtype=torch.float32)
    y=torch.tensor([r["labels"] for r in rows],dtype=torch.float32)
    if x.shape!=(len(rows),2,3) or y.shape!=(len(rows),2):
        raise ValueError("invalid source candidate tensor")
    return x,y


def fit(rows: list[dict], *, seed: int, arm: str,
        steps: int, depth: int, width: int):
    if arm not in ARMS or not 1 <= steps <= 5000:
        raise ValueError("unsupported fitted arm or step count")
    torch.manual_seed(seed)
    mode = "cross" if arm == "frozen_cross" else arm
    model=GuardScorer(width=width, mode=mode)
    if arm == "frozen_cross":
        # Source encoder and readout train; the tied recurrence itself stays
        # at its seed-initialized values. A stronger readout-only control.
        for name,param in model.named_parameters():
            if name.startswith(("core.","gate.","norm.","unknown_evidence")):
                param.requires_grad_(False)
    model.train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=.003,weight_decay=.005)
    x,y=tensors(rows)
    # Same minibatch stream, seed and source ordering across all arms.
    rng=torch.Generator().manual_seed(seed+7000)
    batches=torch.randint(len(rows),(steps,min(32,len(rows))),generator=rng)
    first_loss=last_loss=0.
    for t,ind in enumerate(batches):
        pred=model(x[ind],depth=depth)
        loss=ranking_loss(pred,y[ind])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite OLR reward-model loss")
        if t==0:
            first_loss=float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       1.,error_if_nonfinite=True)
        optimizer.step()
        last_loss=float(loss.detach())
    model.eval()
    train_correct=0
    with torch.no_grad():
        for row in rows:
            train_correct+=choose(model,row["features"],depth=depth)==row["labels"].index(1)
    return model, {
        "seed":seed,"arm":arm,"mode":mode,
        "steps":steps,"depth":depth,"width":width,
        "parameters":model.parameter_count(),
        "trained_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "first_loss":first_loss,"last_loss":last_loss,
        "source_correct":train_correct,"source_worlds":len(rows),
    }


def evaluate(lean: LeanRepl, rows: list[dict], models: dict[tuple,int],
             *, depth: int, split: str):
    """Choose without labels and execute each selected tactic on a new Lean root."""
    output=[]
    for row in rows:
        world=reconstruct(row)
        controls={
            "fixed_r0":0,
            "exact_arithmetic":exact_arithmetic_reference(row["features"]),
        }
        actions={}
        diagnostics={}
        for (seed,arm),model in models.items():
            actions[(seed,arm)]=choose(model,row["features"],depth=depth)
            if arm == "cross":
                # The checkpoint and source fit are frozen. Patch only the
                # cross-message input at inference, without changing logits
                # through a trained alternative. The sham changes the action
                # margin at the same absolute dose with an independent sign.
                x=torch.tensor([row["features"]],dtype=torch.float32)
                with torch.no_grad():
                    normal=model(x,depth=depth)[0]
                    model.mode="isolated"
                    try:
                        no_message=model(x,depth=depth)[0]
                    finally:
                        model.mode="cross"
                margin=float(normal[0]-normal[1])
                removed_margin=float(no_message[0]-no_message[1])
                delta=abs(margin-removed_margin)
                sign=1 if int(digest([row["root"],seed,"margin-sham-v1"])[:8],16)%2 else -1
                actions[(seed,"cross_zero_message")]=int(no_message.argmax().item())
                actions[(seed,"cross_margin_sham")]=0 if margin+sign*delta >= 0 else 1
                diagnostics[seed]={
                    "natural_margin":round(margin,7),
                    "zero_message_margin":round(removed_margin,7),
                    "matched_margin_sham_dose":round(delta,7),
                    "sham_sign":sign,
                }
        for arm,index in controls.items():
            actions[(0,arm)]=index
        # Deduplicate *identical native action requests* within a root to
        # make CI cheaper. Every selected action has a new live Lean witness;
        # artifact records retain which policy reused that checked outcome.
        native = {}
        for index in set(actions.values()):
            outcome=proof_attempt(lean,world,"apply r"+str(index))
            if (outcome["numeric_features"]!=row["features"] or
                    outcome["public_state_sha256"]!=row["source"] or
                    int(outcome["kernel_ok"])!=row["labels"][index]):
                raise ValueError("evaluation tactic outcomes or public inputs drifted")
            native[index]=outcome
        for (seed,arm),index in actions.items():
            outcome=native[index]
            output.append({
                "seed":seed,"arm":arm,"split":split,"root":row["root"],
                "pair_id":row["pair_id"],"world":row["world"],
                "action":"apply r"+str(index),"kernel_closed":outcome["kernel_ok"],
                "native_tactic_calls":outcome["native_tactic_calls"],
                "native_result_sha256":digest(outcome),
                "intervention":diagnostics.get(seed) if arm in (
                    "cross","cross_zero_message","cross_margin_sham") else None,
            })
    return output


def scores(records):
    groups=defaultdict(list)
    for r in records:
        groups[(r["split"],r["seed"],r["arm"])].append(r)
    report={}
    for (split,seed,arm),rows in groups.items():
        bypair=defaultdict(list)
        for r in rows:
            bypair[r["pair_id"]].append(r)
        if any(len(v)!=2 or sorted(x["world"] for x in v)!=[0,1]
               for v in bypair.values()):
            raise ValueError("paired evaluation lost a root")
        key=f"{split}:{seed}:{arm}"
        report[key]={
            "worlds":len(rows), "pairs":len(bypair),
            "proofs":sum(int(x["kernel_closed"]) for x in rows),
            "complete_pairs":sum(all(x["kernel_closed"] for x in v)
                                 for v in bypair.values()),
            "native_tactic_calls":sum(x["native_tactic_calls"] for x in rows),
        }
    return report


def run(bank: Path, out: Path, *, steps: int, depth: int,
        width: int, seeds: tuple[int,...], max_seconds: float = 900.):
    if out.exists() or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("fresh output and unique model seeds required")
    if not 1 <= depth <= 16 or not 8 <= width <= 128:
        raise ValueError("invalid bounded recurrent architecture")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    source,shards=load_bank(bank)
    train=shards["train"]
    models={}
    training={}
    for seed in seeds:
        for arm in ARMS:
            model,receipt=fit(train,seed=seed,arm=arm,steps=steps,depth=depth,width=width)
            models[(seed,arm)]=model
            # Same-worker exact replay distinguishes stochastic code from
            # CPU-hardware-dependent cross-run numerical drift.
            if seed==seeds[0] and arm in ("cross","direct"):
                replay_model,_=fit(train,seed=seed,arm=arm,steps=steps,
                                   depth=depth,width=width)
                if not all(torch.equal(a,b) for
                           (ka,a),(kb,b) in zip(model.state_dict().items(),
                                                 replay_model.state_dict().items())
                           if ka==kb):
                    raise ValueError("same-worker seeded training replay changed weights")
                receipt["same_worker_bit_exact_replay"]=True
            training[f"{seed}:{arm}"]=receipt
            if time.perf_counter()-start>max_seconds:
                raise TimeoutError("bounded CPU training budget exhausted")
    # All query labels are source-file data used solely for postchoice checks.
    scores_by_shard={}
    records=[]
    with LeanRepl(timeout=60) as lean:
        for split in ("dev","ood"):
            rows=shards[split]
            if not rows:
                continue
            records.extend(evaluate(lean,rows,models,depth=depth,split=split))
    score=scores(records)
    # Recompile exactly those distinct query declarations which some policy
    # closed. The generator's reference proof is excluded from model input.
    passed = {(r["root"],r["action"]) for r in records if r["kernel_closed"]}
    certified=[]
    for split in ("dev","ood"):
        for row in shards[split]:
            for action in ("apply r0","apply r1"):
                if (row["root"],action) in passed:
                    certified.append({**row,"reference_proof":[action,"rfl"]})
    out.mkdir(parents=True)
    proof_receipt=certified_standalone(certified,out/"standalone.lean")
    with (out/"models.pt").open("xb") as stream:
        torch.save({f"{s}:{arm}":model.state_dict()
                    for (s,arm),model in models.items()},stream)
    (out/"decisions.jsonl").write_text(
        "".join(canonical(r)+"\n" for r in records),encoding="utf-8")
    receipt={
        "schema":SCHEMA,"source_manifest_sha256":sha256_file(bank/"manifest.json"),
        "source_lean_sha256":source["lean_exe_sha256"],
        "seed_roster":list(seeds),"training":training,"evaluation":score,
        "new_native_query_choices":len(records),
        "native_query_replays":sum(len({r["action"] for r in records if r["root"]==x["root"]})
                                   for x in shards["dev"]+shards["ood"]),
        "independent_query_certificates":proof_receipt,
        "architecture_promotion_eligible":False,
        "training_classification":"native-source-labelled_CPU_research_pilot",
        "cpu_environment":{
            "torch":torch.__version__,
            "platform":platform.platform(),
            "machine":platform.machine(),
            "cpu_count":__import__("os").cpu_count(),
            "threads":torch.get_num_threads(),
            "mkldnn":torch.backends.mkldnn.enabled,
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        },
        "runtime_seconds":round(time.perf_counter()-start,3),
        "model_causal_evidence":"none; next-stage comparison requires prospective matched compute and sham",
    }
    receipt["artifacts"]={name:sha256_file(out/name)
                          for name in ("models.pt","decisions.jsonl","standalone.lean")}
    (out/"manifest.json").write_text(canonical(receipt)+"\n",encoding="utf-8")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--depth",type=int,default=3)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--seeds",type=int,nargs="+",default=list(SEED_ROSTER))
    p.add_argument("--max-seconds",type=float,default=900.)
    a=p.parse_args(argv)
    print(canonical(run(a.bank,a.out,steps=a.steps,depth=a.depth,
                        width=a.width,seeds=tuple(a.seeds),max_seconds=a.max_seconds)))


if __name__=="__main__":
    main()

"""OLR-METTA-04D: frozen carry intervention on fresh native dependent roots.

The archived training bank and weights are exact-hash pinned. There are no
optimizer updates. Public observations are unchanged; only the previous
decision's carry is altered. This is a developmental trajectory intervention.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical,digest,sha256_file
from .live_lean import public_state,candidate_menu
from .olr_dependent import ROOT_MENU,compile_certified
from .olr_metta04_campaign import read_bank,load_fit,SEEDS
from .olr_metta04_world import collect,load_world
from .olr_metta04_policy import act
from .olr_metta04_episode import run_episode,replay_episode

BANK_SHA="4e7f0b5c298f85dc82ec17be0b5b4e5e943059f9585bab01579607fdafdc3dc6"
FIT_SHA="9e331941a41736ca719509a368f531b6b1c0f881efaf6c498414d0a910ebfed6"
WEIGHTS_SHA="89050773cf3d63904f08149904201e177c3d2bdef4558513b4b414de4fbaae24"
REGISTRY="ppt.olr-metta04d-carry-registry-v1"
SCHEMA="ppt.olr-metta04d-carry-score-v1"
MODES=("natural","zero","sham")
CONTROLS=("fixed_r0","majority_correct","source_likelihood","exact")
SPLITS=("dev","ood")


def _read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _write(path,rows):
    path.write_text("".join(canonical(r)+"\n" for r in rows),encoding="utf-8")


def _manifest(path,m):
    path.write_text(canonical(m)+"\n",encoding="utf-8")


def pinned_source(bank:Path,fit:Path):
    """Strict byte custody; reject source data or weights from any other run."""
    for file,sha in ((bank/"manifest.json",BANK_SHA),
                     (fit/"manifest.json",FIT_SHA),
                     (fit/"models.pt",WEIGHTS_SHA)):
        if sha256_file(file)!=sha:
            raise ValueError("04D archived source identity changed: "+file.name)
    registry,rows=read_bank(bank)
    fitted,models=load_fit(bank,fit)
    return rows,models


def lineage(row):
    p=load_world(row).proof
    return p.a,p.b,p.delta


def freeze(bank:Path,fit:Path,out:Path):
    if out.exists():
        raise ValueError("create-only 04D registry required")
    original,_=pinned_source(bank,fit)
    excluded=frozenset(lineage(r) for split in original.values() for r in split)
    seed=20261204
    groups={
        "dev":collect("dev",seed=seed,count=12,low=17,high=30,excluded=excluded),
        "ood":collect("ood",seed=seed+41,count=12,low=31,high=44,excluded=excluded),
    }
    prepared={}
    seen=set()
    with LeanRepl(timeout=60) as lean:
        if not str(lean.version).startswith("4.19.0"):
            raise ValueError("requires native Lean 4.19")
        for split,worlds in groups.items():
            rows=[]
            for w in worlds:
                r=w.public()
                if lineage(r) in excluded or r["root_id"] in seen:
                    raise ValueError("04D fresh lineage or root overlapped source")
                seen.add(r["root_id"])
                lean.reset()
                root=lean.init(w.proof.statement(),pp=True)
                if candidate_menu(lean,root)!=ROOT_MENU:
                    raise ValueError("04D root grammar drift")
                r["root_public_sha256"]=digest(public_state(lean,root))
                rows.append(r)
            if len({lineage(r) for r in rows})!=12:
                raise ValueError("04D distinct-pair count changed")
            prepared[split]=rows
    if {lineage(r) for r in prepared["dev"]}&{lineage(r) for r in prepared["ood"]}:
        raise ValueError("04D evaluation bands overlapped")
    out.mkdir(parents=True)
    for split,rows in prepared.items():
        _write(out/(split+".jsonl"),rows)
    m={"schema":REGISTRY,"seed":seed,"root_worlds":{"dev":24,"ood":24},
       "excluded_original_lineages":len(excluded),
       "source_bank_sha256":BANK_SHA,"source_fit_sha256":FIT_SHA,
       "source_weights_sha256":WEIGHTS_SHA,
       "lean_sha256":sha256_file(config.LEAN_EXE),
       "repl_sha256":sha256_file(config.REPL_EXE),
       "files":{s+".jsonl":sha256_file(out/(s+".jsonl")) for s in SPLITS}}
    _manifest(out/"manifest.json",m)
    return m


def read_fresh(path:Path,original:dict):
    m=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if (m["schema"]!=REGISTRY or m["source_bank_sha256"]!=BANK_SHA or
            m["source_weights_sha256"]!=WEIGHTS_SHA or
            m["lean_sha256"]!=sha256_file(config.LEAN_EXE) or
            m["repl_sha256"]!=sha256_file(config.REPL_EXE)):
        raise ValueError("04D registry identity or native toolchain changed")
    excluded={lineage(r) for rows in original.values() for r in rows}
    records={}
    seen=set()
    for split in SPLITS:
        file=path/(split+".jsonl")
        if sha256_file(file)!=m["files"][file.name]:
            raise ValueError("04D registered bytes changed")
        rows=_read(file)
        pairs=defaultdict(list)
        for r in rows:
            if r["split"]!=split or r["root_id"] in seen or lineage(r) in excluded:
                raise ValueError("04D root leaked or changed split")
            seen.add(r["root_id"])
            pairs[r["pair_id"]].append(r)
        if (len(rows)!=24 or len(pairs)!=12 or
                any(sorted(x["world"] for x in pair)!=[0,1] for pair in pairs.values())):
            raise ValueError("04D paired cohort changed")
        records[split]=rows
    return records


def signed_majority(features):
    """Public feature sign: negative denotes r0; positive denotes r1."""
    return int(sum(float(x[0])*float(x[1]) for x in features)>0.)


def pattern(features):
    """The public strengths of admitted claims, not generator mode or truth."""
    return tuple(round(3*float(x[1])) if x[2] else 0 for x in features)


def calibrate(train:list,labels:list):
    """Laplace-smoothed conditional vote reliability on archived TRAIN only."""
    if len(train)!=48 or len(labels)!=48:
        raise ValueError("wrong source-only calibration size")
    by_id={r["root_id"]:r["source_first_winner"] for r in labels}
    if set(by_id)!={r["root_id"] for r in train}:
        raise ValueError("source-native calibration IDs changed")
    counts=defaultdict(lambda:[1,2])
    for r in train:
        f=r["model_features"]
        p=pattern(f)
        winner=by_id[r["root_id"]]
        if winner not in (0,1):
            raise ValueError("bad source winner")
        for j,x in enumerate(f):
            if x[2]:
                key=(p,j)
                counts[key][0]+=int(int(x[0]>0)==winner)
                counts[key][1]+=1
    return {k:tuple(v) for k,v in counts.items()}


def likelihood(features,counts):
    p=pattern(features)
    logodds=0.
    for j,x in enumerate(features):
        if x[2]:
            n,total=counts.get((p,j),(1,2))
            logodds+=float(x[0])*math.log(n/(total-n))
    return int(logodds>0.)


def perturb(carry,mode,*,root_id,seed,decision):
    """Zero or equal-L2-dose random direction; root carry is never perturbed."""
    if carry is None or mode=="natural":
        return carry,0.
    if mode not in ("zero","sham"):
        raise ValueError("unknown frozen carry intervention")
    size=float(torch.linalg.vector_norm(carry))
    if mode=="zero":
        return torch.zeros_like(carry),size
    # Hash-based sign vector avoids dependence on ambient RNG state.
    key=f"04D/{root_id}/{seed}/{decision}".encode()
    rng=random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8],"big"))
    direction=torch.tensor([rng.choice((-1.,1.)) for _ in range(carry.numel())],
                           dtype=carry.dtype,device=carry.device).reshape_as(carry)
    delta=direction*(size/float(torch.linalg.vector_norm(direction)))
    return carry+delta,float(torch.linalg.vector_norm(delta))


def model_callback(model,*,arm,mode,features,macro,root_id,seed,frozen,trace):
    carry=None
    decision=0
    def callback(phase,remaining,branch):
        nonlocal carry,decision
        inp,dose=perturb(carry,mode,root_id=root_id,seed=seed,decision=decision)
        action,carry,_=act(model,features,phase=phase,remaining=remaining,
                           branch=branch,macro_allowed=macro,carry=inp,arm=arm)
        if phase=="root" and action!=frozen:
            raise ValueError("frozen first decision drift")
        trace.append({"decision":decision,"phase":phase,"action":action,
                      "mode":mode,"carry_perturbation_l2":round(dose,8)})
        decision+=1
        return action,3 if arm=="loop3" else 1
    return callback


def scripted(first,macro):
    def callback(phase,remaining,branch):
        if phase=="root":
            return "root_r"+str(first),0
        if phase=="retry":
            return "root_r"+str(1-branch),0
        if phase=="open":
            return "probe_guard",0
        if phase=="failed":
            return ("restore" if macro and remaining>=4 else "stop"),0
        if phase=="guard_ok":
            return ("metta_repair" if macro else "manual"),0
        return "stop",0
    return callback


def summarize(records):
    groups=defaultdict(list)
    for r in records:
        groups[r["split"],r["seed"],r["policy"]].append(r)
    out={}
    for (split,seed,arm),rows in sorted(groups.items()):
        pairs=defaultdict(list)
        for r in rows:pairs[r["pair_id"]].append(r)
        if len(rows)!=24 or len(pairs)!=12 or any(
                sorted(r["world"] for r in pair)!=[0,1] for pair in pairs.values()):
            raise ValueError("04D unpaired outcome group")
        out[f"{split}:{seed}:{arm}"]={
            "worlds":len(rows),"pairs":len(pairs),
            "certified":sum(int(r["kernel_closed"]) for r in rows),
            "complete_pairs":sum(all(r["kernel_closed"] for r in p) for p in pairs.values()),
            "utility":round(sum(r["utility"] for r in rows)/len(rows),8),
            **{k:sum(r[k] for r in rows) for k in ("tactic_calls","restores",
                  "false_stop","vetoed_restores","metta_queries","model_passes",
                  "model_decisions")}}
    return out


def paired_contrast(records,split,left,right,metric,*,repetitions=10000):
    """Average model seeds within each pair, then resample independent pairs."""
    rows=[r for r in records if r["split"]==split]
    index={(r["pair_id"],r["world"],r["seed"],r["policy"]):r for r in rows}
    pairs=sorted({r["pair_id"] for r in rows})
    values=[]
    seeds={s:[] for s in SEEDS}
    for p in pairs:
        local=[]
        for seed in SEEDS:
            for world in (0,1):
                a=index[p,world,seed,left][metric]
                b=index[p,world,seed,right][metric]
                d=float(a)-float(b)
                local.append(d)
                seeds[seed].append(d)
        values.append(sum(local)/len(local))
    rng=random.Random("04D/"+split+"/"+left+"/"+right+"/"+metric)
    n=len(values)
    means=sorted(sum(rng.choice(values) for _ in range(n))/n for _ in range(repetitions))
    return {"effect":sum(values)/n,"ci95":[means[int(.025*repetitions)],
              means[min(repetitions-1,int(.975*repetitions))]],
            "pair_clusters":n,"per_seed":{str(s):sum(v)/len(v) for s,v in seeds.items()}}


def score(bank:Path,fit:Path,registry:Path,out:Path,*,max_seconds=650.):
    if out.exists():
        raise ValueError("04D score output must be new")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    original,models=pinned_source(bank,fit)
    roots=read_fresh(registry,original)
    calibration=calibrate(original["train"],_read(fit/"source_native_labels.jsonl"))
    # Freeze every root choice and model identity before any evaluation outcome.
    pending=[]
    frozen=[]
    for split,rows in roots.items():
        for row in rows:
            w=load_world(row)
            f=row["model_features"]
            refs={"fixed_r0":0,"majority_correct":signed_majority(f),
                  "source_likelihood":likelihood(f,calibration),
                  "exact":w.proof.reference_branch()}
            for name,first in refs.items():
                pending.append((row,w,0,name,None,None,"root_r"+str(first)))
            for seed in SEEDS:
                for arm in ("loop3","loop1","direct"):
                    model=models[seed,arm]
                    first,_,_=act(model,f,phase="root",remaining=6,branch=None,
                                 macro_allowed=w.macro_allowed,carry=None,arm=arm)
                    for mode in (MODES if arm=="loop3" else ("natural",)):
                        name="loop3_"+mode if arm=="loop3" else arm
                        pending.append((row,w,seed,name,arm,mode,first))
    for row,w,seed,name,arm,mode,first in pending:
        frozen.append({"root_id":w.proof.root_id,"pair_id":w.proof.pair_id,
            "world":w.proof.world,"split":w.proof.split,"seed":seed,
            "policy":name,"root_action":first,"packet_sha256":digest(row["model_features"])})
    out.mkdir(parents=True)
    _write(out/"frozen_choices.jsonl",frozen)
    _manifest(out/"source_calibration.json",{str(k):list(v) for k,v in calibration.items()})
    records=[];episodes=[];successes=[]
    start=time.monotonic()
    with LeanRepl(timeout=60) as lean:
        for row,w,seed,name,arm,mode,first in pending:
            doses=[]
            callback=(scripted(int(first[-1]),w.macro_allowed) if arm is None else
                model_callback(models[seed,arm],arm=arm,mode=mode,
                   features=row["model_features"],macro=w.macro_allowed,
                   root_id=w.proof.root_id,seed=seed,frozen=first,trace=doses))
            ep=run_episode(lean,w.proof,claims=row["model_features"],
                           macro_allowed=w.macro_allowed,policy=callback)
            if not replay_episode(lean,w.proof,ep,macro_allowed=w.macro_allowed):
                raise ValueError("04D native replay failed")
            if ep["initial_public_sha256"]!=row["root_public_sha256"]:
                raise ValueError("04D policy source state drift")
            record={"root_id":w.proof.root_id,"pair_id":w.proof.pair_id,
                    "world":w.proof.world,"split":w.proof.split,"seed":seed,"policy":name,
                    **{k:ep[k] for k in ("kernel_closed","utility","tactic_calls",
                           "restores","false_stop","vetoed_restores","metta_queries",
                           "model_passes","model_decisions")},
                    "episode_sha256":digest(ep)}
            records.append(record)
            episodes.append({"identity":record,"episode":ep,"carry_interventions":doses})
            if ep["kernel_closed"]:
                script=[f"apply r{ep['verified_branch']}","rfl"]
                script+=["apply h0","exact h1 _ h0"] if ep["tail"]=="fast" else [
                        "apply h0","apply h1","apply h0"]
                successes.append((w.proof,{"root_id":w.proof.root_id,
                                  "kernel_closed":True,"script":script}))
            if time.monotonic()-start>max_seconds:
                raise TimeoutError("04D native evaluation wall cap")
    cert=compile_certified(successes,out/"standalone.lean")
    _write(out/"score_rows.jsonl",records)
    _write(out/"native_episodes.jsonl",episodes)
    contrasts={}
    for split in SPLITS:
        for a,b in (("loop3_natural","loop3_zero"),("loop3_natural","loop3_sham"),
                    ("loop3_zero","loop3_sham"),("loop3_natural","loop1")):
            for metric in ("kernel_closed","utility"):
                contrasts[f"{split}:{a}-{b}:{metric}"]=paired_contrast(
                    records,split,a,b,metric)
    names=("frozen_choices.jsonl","source_calibration.json","score_rows.jsonl",
           "native_episodes.jsonl","standalone.lean")
    report={"schema":SCHEMA,"registry_sha256":sha256_file(registry/"manifest.json"),
            "source_fit_sha256":FIT_SHA,"source_weights_sha256":WEIGHTS_SHA,
            "optimizer_updates":0,"native_episodes":len(episodes),
            "native_replays":len(episodes),"standalone":cert,
            "root_worlds":48,"scorecard":summarize(records),"contrasts":contrasts,
            "carry_dose_matched_l2_per_decision":True,
            "total_episode_flops_matched":False,"architecture_promotion_eligible":False,
            "artifacts":{n:sha256_file(out/n) for n in names}}
    _manifest(out/"manifest.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="phase",required=True)
    for mode in ("freeze","score"):
        p=sub.add_parser(mode)
        p.add_argument("--bank",type=Path,required=True)
        p.add_argument("--fit",type=Path,required=True)
        p.add_argument("--out",type=Path,required=True)
        if mode=="score":
            p.add_argument("--registry",type=Path,required=True)
    a=parser.parse_args(argv)
    m=(freeze(a.bank,a.fit,a.out) if a.phase=="freeze" else
       score(a.bank,a.fit,a.registry,a.out))
    if a.phase=="score":
        print(canonical({k:m[k] for k in ("schema","native_episodes","native_replays",
                          "standalone","scorecard","contrasts")}))
    else:
        print(canonical(m))


if __name__=="__main__":
    main()

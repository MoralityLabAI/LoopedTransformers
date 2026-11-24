"""Independently qualify and score closed two-pass tangent observations."""

import argparse
from collections import Counter
import json
from math import isclose
from pathlib import Path
import re
from statistics import fmean
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import bound
from experiments.nan_routematch86.geometry import ARMS, COSINE, DOSE, DOSE_TOLERANCE
from experiments.nan_radial76.geometry import MAX_NORM_GAP, MAX_RADIAL_FRACTION, MAX_TANGENT_COSINE
from experiments.nan_vocab82.heads import partition_sha
from experiments.nan_vocab82.host import annotated, effects, reduce_contrast
from experiments.nan_vocab83.custody import HOST_BUDGET, load_baseline, qualification, vocabulary

from .source import EXPERIMENT, config, load


MODES=(*ARMS,"learned_k_rescue")
CONTRASTS=(("learned","cal_static"),("learned","seeded"),("learned","clean"),("learned","learned_k_rescue"))


def finite(number):
    from math import isfinite

    return type(number) in (int,float) and isfinite(number)


def check_geometry(p):
    nominal=p["nominal_l2"]
    measured=p["measured"]
    cosines=p["realized_sham_remainder_cosines"]
    if (p["status"]!="matched" or p["same_learned_and_static_bytes"] is not False
            or not finite(nominal) or nominal<=0 or set(measured)!=set(ARMS)
            or len(cosines)!=2 or any(not finite(v) or abs(v)>MAX_TANGENT_COSINE for v in cosines)):
        raise ValueError("Invalid matched phase-1 MLP geometry")
    r=nominal/DOSE
    for arm in measured.values():
        l,b,n,c=(arm[k] for k in ("realized_l2","radial_l2","relative_reached_norm","planned_direction_cosine"))
        if (any(not finite(v) for v in (l,b,n,c)) or abs(l/nominal-1)>DOSE_TOLERANCE
                or not COSINE<=c<=1+1e-12 or n<=0 or abs(b)>l+1e-10
                or not isclose(n*n,1+2*b/r+(l/r)**2,rel_tol=1e-12)):
            raise ValueError("Reached MLP displacement misses its orbit or dose")
    for key,field,scale,limit in (("dose_gap_fraction","realized_l2",nominal,DOSE_TOLERANCE),
                                  ("radial_gap_fraction","radial_l2",nominal,MAX_RADIAL_FRACTION),
                                  ("relative_norm_gap","relative_reached_norm",1,MAX_NORM_GAP)):
        values=[arm[field] for arm in measured.values()]
        actual=(max(values)-min(values))/scale
        if not finite(p[key]) or p[key]>limit or not isclose(actual,p[key],rel_tol=0,abs_tol=1e-13):
            raise ValueError("The three rounded MLP arms are geometrically unmatched")


def gate_report(source, rows, baseline_sha, classes, tok, cfg):
    return qualification(source,rows,baseline_sha,classes,tok,cfg)|{"experiment":EXPERIMENT}


def score(source, baseline, episodes, gate, classes):
    gold={(r["root"],r["world"]):r for r in source["gold"] if r["arm"]=="focused"}
    clean={r["id"]:r for r in baseline}
    bare=source["tokens"]["bare_label_token_ids"]
    qualified=set(gate["qualified_focused_roots"])
    declared={root for root,_ in gold}
    seen=set()
    status=Counter()
    private=[]
    for ep in episodes:
        root=ep["root"]
        twins=[gold[root,w] for w in ("A","B")]
        if root in seen or ep["source_ids"]!=[p["id"] for p in twins]:
            raise ValueError("Edited result lacks the frozen A/B source binding")
        seen.add(root)
        allowed=("native_gate_failed",) if not gate["passed"] else (
            ("closed","geometry_unqualified","mixed_duplicate_unqualified") if root in qualified
            else ("native_root_unqualified",))
        state=ep["status"]
        if state not in allowed:
            raise ValueError("Edited result contradicts native competence qualification")
        status[state]+=1
        if state!="closed":
            continue
        if len(ep["plans"])!=2 or len(ep["worlds"])!=2:
            raise ValueError("The edited episode needs both qualified physical worlds")
        worlds=[]
        for i,(world,p,plan) in enumerate(zip(ep["worlds"],twins,ep["plans"],strict=True)):
            check_geometry(plan)
            if (set(world["arms"])!=set(MODES) or set(world["class_maxima"])!=set(MODES)
                    or not all(world[k] is True for k in ("rescue_k_exact","zero_hook_exact","clean_repeat_exact"))
                    or world["arms"]["learned"][0]!=world["arms"]["learned_k_rescue"][0]
                    or world["class_maxima"]["learned"][0]!=world["class_maxima"]["learned_k_rescue"][0]):
                raise ValueError("The scoped edit or pass-2 K rescue failed custody")
            reader=world["reader_l2"]
            if set(reader)!=set(ARMS) or any(set(r)!={"q","k","v"} or any(
                    not finite(x) or x<0 for x in r.values()) for r in reader.values()):
                raise ValueError("Pass-2 reader displacement is missing or invalid")
            controls=clean[p["id"]]
            target,foil=p["answer"],twins[1-i]["answer"]
            reads={}
            for mode in ("clean",*MODES):
                pair,classes_seen=(controls["phases"],controls["class_maxima"]) if mode=="clean" else (
                    world["arms"][mode],world["class_maxima"][mode])
                if len(pair)!=2 or len(classes_seen)!=2:
                    raise ValueError("A physical pass is missing from the head record")
                reads[mode]=[annotated(q,n,classes,bare,target,foil)
                             for q,n in zip(classes_seen,pair,strict=True)]
            worlds.append({"id":p["id"],"heads":reads,"phase2_reader_l2":reader})
        private.append({"root":root,"worlds":worlds})
    if seen!=declared:
        raise ValueError("Terminal does not close every registered graph root")
    contrasts={}
    for a,b in CONTRASTS:
        terms=[[tuple(effects(w["heads"][a][p],w["heads"][b][p]) for p in range(2))
                for w in root["worlds"]] for root in private]
        passes={f"pass{p+1}":reduce_contrast([[pair[p] for pair in world] for world in terms],len(declared))
                for p in range(2)}
        for metric in ("full_margin","paired_margin"):
            passes[f"pass2_minus_pass1_{metric}"]=fmean(
                fmean(pair[1][metric]-pair[0][metric] for pair in world) for world in terms) if terms else None
        contrasts[f"{a}_minus_{b}"]=passes
    floor=config()["minimum_meaningful_delayed_margin_logit"]
    primary=contrasts["learned_minus_cal_static"]
    return {"experiment":EXPERIMENT,"registered_roots":len(declared),"closed_roots":len(private),
            "unknown_worlds":2*(len(declared)-len(private)),"root_status":dict(status),
            "engineering_floor_logit":floor,"contrasts":contrasts,
            "primary_crosses_engineering_floor":None if not private else
                abs(primary["pass2"]["root_macro_delta"]["full_margin"])>=floor,
            "formal_comparisons":0,"sealed_test_accessed":False},private


def collect(work,source,tok,classes,frozen,*,mode):
    t=work/"terminal.json"
    terminal_sha=sha(t)
    end=json.loads(bound(t,terminal_sha))
    target=runtime.TARGETS["nf4"]
    if ((end["experiment"],end["mode"],end["backend"],end["status"],end["runtime_sha256"]) !=
            (EXPERIMENT,mode,"nf4","completed",frozen)
            or end["checkpoint_verified_after"] is not True
            or end["target_before"]!=target or end["target_after"]!=target
            or end["quant_before"]!=end["quant_after"] or end["partition_sha256"]!=partition_sha(classes)
            or end["vocab"]!=len(classes)):
        raise ValueError("The terminal does not close the pinned NF4 condition")
    admission=json.loads(bound(work/"admission.json",end["admission_sha256"]))
    if (re.fullmatch(r"[0-9a-f]{40}",end["published_commit"]) is None
            or tuple(admission[k] for k in ("runtime_sha256","mode","published_commit")) !=
            (frozen,mode,end["published_commit"])):
        raise ValueError("Admission and closed run differ")
    preflight=json.loads(bound(work/"preflight.json",end["preflight_sha256"]))
    if (preflight["native_B2_heads_exact"] is not True or preflight["resident_and_streamed_exact"] is not True
            or preflight["partition_sha256"]!=partition_sha(classes)):
        raise ValueError("Preflight lacks clean or resident-head parity")
    baseline,baseline_sha=load_baseline(work/"baseline.json",end["baseline_sha256"])
    gate=gate_report(source,baseline,baseline_sha,classes,tok,config())
    if gate!=json.loads(bound(work/"qualification.json",end["qualification_sha256"])):
        raise ValueError("Independent competence gate differs from the archived receipt")
    roots=sorted({p["root"] for p in source["public"]})
    journal=bound(work/"closed.jsonl",end["journal_sha256"]).splitlines()
    if len(journal)!=len(roots) or end["root_records"]!=len(roots) or 2*end["b2_baseline_batches"]!=len(baseline):
        raise ValueError("The closed run has incomplete batch or root coverage")
    episodes=[]
    for i,(root,line) in enumerate(zip(roots,journal,strict=True)):
        entry=json.loads(line)
        path=work/f"root_{i:03d}.json"
        if entry["file"]!=path.name:
            raise ValueError("Root evidence is out of canonical order")
        episode=json.loads(bound(path,entry["sha256"]))
        if episode["root"]!=root:
            raise ValueError("The episode root differs from its registered position")
        episodes.append(episode)
    aggregate,private=score(source,baseline,episodes,gate["gate"],classes)
    return aggregate|{"runtime_sha256":frozen,"terminal_sha256":terminal_sha,"mode":mode},private


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir",type=Path,required=True)
    cli.add_argument("--snapshot",type=Path,required=True)
    cli.add_argument("--work",type=Path,required=True)
    cli.add_argument("--qualification",action="store_true")
    cli.add_argument("--fixture",action="store_true")
    args=cli.parse_args()
    from .lane import verified

    frozen=verified()
    if "torch" in sys.modules or args.snapshot.name!=config()["checkpoint_revision"]:
        cli.error("Require the lightweight host and the pinned tokenizer")
    mode="synthetic_fixture" if args.fixture else "pretrained_cuda"
    with lease(HOST_BUDGET) as (check,samples):
        source,receipt=load(args.source_dir,("public","gold","tokens"))
        tok,classes=vocabulary(args.snapshot,check,receipt["tokenizer_pins"])
        if args.qualification:
            a=json.loads((args.work/"admission.json").read_bytes())
            if (a["runtime_sha256"],a["mode"])!=(frozen,mode):
                raise ValueError("Baseline lacks matching admission")
            rows,sha_rows=load_baseline(args.work/"baseline.json")
            write_new(args.work/"qualification.json",gate_report(source,rows,sha_rows,classes,tok,config()))
        else:
            result,private=collect(args.work,source,tok,classes,frozen,mode=mode)
            write_new(args.work/"private_outcomes.json",private)
        if "torch" in sys.modules:
            raise RuntimeError("Independent host imported Torch")
    if not args.qualification:
        write_new(args.work/"aggregate.json",result|{"private_outcomes_sha256":sha(args.work/"private_outcomes.json"),
          "max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
          "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples),
          "max_sampled_working_set_bytes":max(s["working_set_bytes"] for s in samples)})


if __name__=="__main__":
    main()

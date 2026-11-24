"""Score a closed writer pilot using the separate source oracle and tokenizer."""
import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_comp14.parser import label
from experiments.nan_delay39.source import digest
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer
from experiments.nan_relabel68.pipeline import audited_score

from experiments.nan_cidose74.geometry import COSINE, DOSE_TOLERANCE
from experiments.nan_radial76.geometry import ARMS, MAX_NORM_GAP, MAX_RADIAL_FRACTION, MAX_TANGENT_COSINE

from .pipeline import MODES
from .source import CONFIG,HERE,load


def reading(scores,expected,tokens,tok,vocab):
    winner = audited_score(scores,tokens,vocab)
    text = tok.decode([winner],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    return label(text)==expected


def qualify(source,baseline,labels,tok,vocab,cfg):
    public = {p["id"]:p for p in source["public"]}
    gold = {p["id"]:p["answer"] for p in source["gold"]}
    if set(gold)!=set(public) or len(baseline)!=len(public):
        raise ValueError("The baseline omitted a registered source world")
    observed = {}
    roots_by_arm = {}
    for row in baseline:
        key = row["id"]
        expected = source["tokens"]["row_bindings"][key]["sha256"]
        if key in observed or row["token_sha256"]!=expected or len(row["phases"])!=2:
            raise ValueError("Wrong source binding in the baseline observation")
        p = public[key]
        phases = [reading(score,gold[key],labels,tok,vocab) for score in row["phases"]]
        observed[key] = {"correct":phases,"phases":row["phases"]}
        roots_by_arm.setdefault((p["root"],p["arm"]),[]).append(phases)
    if set(observed)!=set(public) or any(len(v)!=2 for v in roots_by_arm.values()):
        raise ValueError("The baseline lacks a complete physical pair")
    focus = sorted(root for (root,arm),v in roots_by_arm.items() if arm=="focused" and all(p[1] for p in v))
    copy = sum(arm=="copy" and all(p[1] for p in v) for (_,arm),v in roots_by_arm.items())
    counts = {arm:{"first_correct":sum(v[0] for (_,other),rows in roots_by_arm.items() if other==arm for v in rows),
                   "second_correct":sum(v[1] for (_,other),rows in roots_by_arm.items() if other==arm for v in rows)}
              for arm in cfg["arms"]}
    criteria = len(focus)>=cfg["minimum_clean_both_world_roots"] and copy>=cfg["minimum_literal_copy_both_world_roots"]
    return {"passed":criteria,"qualified_focused_roots":focus,"both_world_copy_roots":copy,"baseline_counts":counts,
            "baseline_digest":digest(baseline),"registered_roots":cfg["source_roots"]},observed


def check_plan(plan):
    arms = plan["arms"]
    nominal = plan["nominal_l2"]
    if set(arms)!=set(ARMS) or not isfinite(nominal) or nominal<=0:
        raise ValueError("The matched-geometry receipt omits its doses")
    for arm in arms.values():
        if (any(not isfinite(arm[k]) for k in ("l2","radial_l2","direction_cosine","relative_norm_drift"))
                or abs(arm["l2"]/nominal-1)>DOSE_TOLERANCE or not COSINE<=arm["direction_cosine"]<=1+1e-12):
            raise ValueError("The reached edit missed its physical dose or angle gate")
    lengths = [p["l2"] for p in arms.values()]
    active,sham = (arms[a] for a in ("active","matched_seeded"))
    norm_gap = abs(sham["relative_norm_drift"]-active["relative_norm_drift"])
    if (max(lengths)-min(lengths)>DOSE_TOLERANCE*nominal
            or abs(sham["radial_l2"]-active["radial_l2"])>MAX_RADIAL_FRACTION*nominal
            or norm_gap>MAX_NORM_GAP
            or not abs(plan["realized_tangent_cosine"])<=MAX_TANGENT_COSINE):
        raise ValueError("The joint sham geometry does not close")


def score_roots(rows,gate,lookup,labels,tok,vocab):
    disposition = Counter()
    known = []
    qualified = set(gate["qualified_focused_roots"])
    seen = set()
    for episode in rows:
        root,status = episode["root"],episode["status"]
        if root in seen or root not in lookup:
            raise ValueError("Unknown or repeated root in the closed journal")
        seen.add(root)
        if episode["source_ids"]!=[ref[0] for ref in lookup[root]]:
            raise ValueError("Edited worlds do not bind the registered A/B order")
        allowed = ("native_gate_failed",) if not gate["passed"] else (
            ("closed","geometry_unqualified","mixed_duplicate_unqualified") if root in qualified
            else ("native_root_unqualified",))
        if status not in allowed:
            raise ValueError("World intervention contradicts its registered gate")
        disposition[status] += 1
        if status!="closed":
            continue
        if (len(episode["worlds"])!=2 or len(episode["plans"])!=2
                or any(p["status"]!="matched" for p in episode["plans"])):
            raise ValueError("Closed root omitted one scoped world or reached-dose plan")
        for plan in episode["plans"]:
            check_plan(plan)
        effects = {}
        for world,ref in zip(episode["worlds"],lookup[root],strict=True):
            _,expected,other,clean = ref
            if (set(world["arms"])!=set(MODES) or not world["rescue_k_exact"]
                    or not world["zero_hook_exact"] or not world["clean_repeat_exact"]):
                raise ValueError("Closed edited world did not pass its custody gates")
            for mode,phases in world["arms"].items():
                if len(phases)!=2:
                    raise ValueError("Edited outcome lacks a physical pass")
                effect = []
                for t,p in enumerate(phases):
                    correct = reading(p,expected,labels,tok,vocab)
                    base = clean["phases"][t]["labels"]
                    margin_change = p["labels"][expected] - p["labels"][other] - base[expected] + base[other]
                    effect.append({"correct":correct,"delta_margin":margin_change})
                effects.setdefault(mode,[]).append(effect)
            if world["arms"]["active"][0]!=world["arms"]["active_k_rescue"][0]:
                raise ValueError("Clean K rescue changed the first-pass head")
        known.append({"effects":effects,"clean_correct":[r[3]["correct"] for r in lookup[root]]})
    if seen!=set(lookup):
        raise ValueError("Missing roots in the terminal journal")
    modes = {}
    for mode in MODES:
        arms = [(phases,clean) for root in known
                for phases,clean in zip(root["effects"][mode],root["clean_correct"],strict=True)]
        modes[mode] = {"worlds":len(arms),"both_world_pass2_correct_roots":sum(
                           all(p[1]["correct"] for p in root["effects"][mode]) for root in known),
                       "clean_repair_worlds":sum(not c[0] and c[1] for _,c in arms),
                       "clean_repair_worlds_retained":sum(not c[0] and c[1] and p[1]["correct"] for p,c in arms),
                       "correct_worlds_by_pass":[sum(p[t]["correct"] for p,_ in arms) for t in (0,1)],
                       "mean_label_margin_delta_by_pass":[fmean(p[t]["delta_margin"] for p,_ in arms) if arms else None
                                                          for t in (0,1)]}
    return {"registered_roots":len(lookup),"closed_roots":len(known),"unresolved_roots":len(lookup)-len(known),
            "root_status":dict(disposition),"modes":modes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--vocab",type=int,required=True)
    p.add_argument("--qualification",type=Path)
    p.add_argument("--closed",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not __debug__ or "torch" in sys.modules:
        p.error("Use assertions and a lightweight host process")
    cfg=json.loads(CONFIG.read_bytes())
    source,proof=load(args.source_dir,("public","gold","tokens"))
    tok,pins=tokenizer(args.snapshot)
    if pins!=proof["tokenizer_pins"]:
        raise RuntimeError("Host readout tokenizer differs from the verified source")
    tok=tok.tok
    labels=source["tokens"]["bare_label_token_ids"]
    raw=json.loads(args.baseline.read_bytes())
    gate,obs=qualify(source,raw,labels,tok,args.vocab,cfg)
    if args.closed is None:
        write_new(args.output,gate)
        return
    terminal=json.loads((args.closed/"terminal.json").read_bytes())
    if (terminal["status"]!="completed" or not terminal["checkpoint_verified_after"]
            or not terminal["quant_unchanged"] or terminal["vocab"]!=args.vocab
            or terminal["runtime_sha256"]!=sha(HERE/"RUNTIME_MANIFEST.json")
            or terminal["baseline_sha256"]!=sha(args.baseline)
            or terminal["qualification_sha256"]!=sha(args.qualification)
            or json.loads(args.qualification.read_bytes())!=gate):
        raise ValueError("No integrity-closed matching worker terminal")
    gold={(r["root"],r["world"]):(r["id"],r["answer"],obs[r["id"]]) for r in source["gold"] if r["arm"]=="focused"}
    lookup={}
    for root in {key[0] for key in gold}:
        pairs=[gold[root,w] for w in ("A","B")]
        lookup[root]=[(key,answer,pairs[1-i][1],clean) for i,(key,answer,clean) in enumerate(pairs)]
    index=args.closed/"closed.jsonl"
    if sha(index)!=terminal["journal_sha256"]:
        raise ValueError("Closed journal digest changed")
    episodes=[]
    for i,line in enumerate(index.read_bytes().splitlines()):
        row=json.loads(line)
        path=args.closed/f"root_{i:03d}.json"
        if row["file"]!=path.name or sha(path)!=row["sha256"]:
            raise ValueError("Closed root digest or ordinal changed")
        episodes.append(json.loads(path.read_bytes()))
    output=score_roots(episodes,gate,lookup,labels,tok,args.vocab)
    write_new(args.output,output | {"experiment":"NAN-WRITER-77","formal_comparisons":0,
                                   "native_gate":gate["baseline_counts"],"terminal_sha256":sha(args.closed/"terminal.json")})


if __name__=="__main__":
    main()

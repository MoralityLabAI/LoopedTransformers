"""LBF-12: independently compiled rewrite witnesses, decision-local fresco and actual gate probes.

The pinned LSA-02K public simulator is a dependency, not a live Lean tactic
engine. Only standalone compiled closed scripts count as certified proofs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import torch

from lbf_gym import SCHEMA_OBS, ROLES, canonical, digest, encode, file_sha, fresco_svg, validate_observation, write_jsonl

LCGRT=Path(__file__).resolve().parents[1]/"lean_cgrt_hrm400"
sys.path.insert(0,str(LCGRT))
from lcgrt import config
from ppt_gym.core import Config,PlasticProofTransformer
from ppt_gym.native_train import _model_fingerprint

UPSTREAM_COMMIT="cd9ac784c5804ec2fe595bb6a01d2697fc016610"
UPSTREAM_BLOB="204a1d012a4ebe8e80d8bd722590742b7e7f02ba"
GENERATOR_SEED=20261205
COUNTS={"train":24,"calibration":16,"ordinary":16,"ood":16}
SEED=107
STEPS=360
DEPTH=16
WORK=128
MODES=("normal","zero","roll")
RANDOM_SEEDS=(17,31,47,61,83)


def load_upstream(path):
    raw=path.read_bytes()
    sha=hashlib.sha1(("blob "+str(len(raw))+"\0").encode()+raw).hexdigest()
    if sha!=UPSTREAM_BLOB:
        raise ValueError("rewrite simulator differs from registered pinned source")
    spec=importlib.util.spec_from_file_location("lbf12_pinned_rewrite",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def write(path,obj):
    path.write_text(canonical(obj)+"\n",encoding="utf-8")


def render_proof(rg,case,path,name):
    """Explicit Eq.trans composition avoids dependence on calc layout parsing."""
    case.validate()
    if not re.fullmatch(r"[a-z][a-z0-9_]*",name):
        raise ValueError("invalid certificate declaration")
    moves=[x if isinstance(x,rg.Move) else rg.Move(**x) for x in path]
    states=rg.replay(case,moves)
    if not moves:
        raise ValueError("certificate needs a nonempty witness")
    term=rg.term
    src,dst=term(case.source),term(case.target)
    binders=["(A : Type)","("+" ".join(f"f_{c}" for c in rg.ALPHABET)+" : A → A)","(x : A)"]
    for i,(a,b) in enumerate(case.rules):
        binders.append(f"(h{i} : ∀ z : A, {term(a,'z')} = {term(b,'z')})")
    transport=case.family=="dependent_transport"
    if transport:
        binders += ["(P : A → Type)",f"(witness : P {src})"]
    target=f"P {dst}" if transport else f"{src} = {dst}"
    declaration="def" if transport else "theorem"
    lines=[f"{declaration} {name} {' '.join(binders)} : {target} := by"]
    for i,(current,move) in enumerate(zip(states,moves)):
        left,right=case.rules[move.rule]
        if move.direction:
            left,right=right,left
        suffix=current[move.position+len(left):]
        e=f"(h{move.rule} {term(suffix)})"
        if move.direction:
            e=f"(Eq.symm {e})"
        if move.position:
            e=f"(congrArg (fun z : A => {term(current[:move.position],'z')}) {e})"
        lines.append(f"  have step_{i} : {term(current)} = {term(move.successor)} := {e}")
    proof="step_0"
    for i in range(1,len(moves)):
        proof=f"(Eq.trans {proof} step_{i})"
    lines.append(f"  exact Eq.mp (congrArg P {proof}) witness" if transport else f"  exact {proof}")
    return "\n".join(lines)


def certify(rg,items,out):
    if out.exists() or not items:
        raise ValueError("fresh nonempty native certificate destination required")
    names=[n for _,_,n in items]
    if len(set(names))!=len(names):
        raise ValueError("duplicate certificate name")
    out.mkdir(parents=True)
    path=out/"Proofs.lean"
    path.write_text("namespace LBF12Cert\n\n"+
        "\n\n".join(render_proof(rg,c,p,n) for c,p,n in items)+
        "\n\nend LBF12Cert\n"+
        "\n".join(f"#print axioms LBF12Cert.{n}" for n in names)+"\n",encoding="utf-8")
    version=subprocess.run([str(config.LEAN_EXE),"--version"],capture_output=True,text=True,
                           check=True,env=config.lean_env()).stdout.strip()
    if "version 4.19.0" not in version:
        raise ValueError("pinned Lean 4.19 required")
    result=subprocess.run([str(config.LEAN_EXE),str(path)],capture_output=True,text=True,
                          timeout=420,env=config.lean_env())
    log=result.stdout+result.stderr
    (out/"compiler.log").write_text(log,encoding="utf-8")
    passed=set(re.findall(r"'LBF12Cert\.([a-z0-9_]+)' does not depend on any axioms",log))
    if result.returncode or passed!=set(names):
        raise RuntimeError("independent standalone Lean certificate rejected: "+log[-3000:])
    m={"schema":"lbf12.explicit_eq_trans_cert.v1","actual_lean_compiler":True,
       "lean_version":version,"lean_exe_sha256":file_sha(config.LEAN_EXE),
       "source_sha256":file_sha(path),"log_sha256":file_sha(out/"compiler.log"),
       "compiled_axiom_free":len(passed),"declarations":names,"native_online_tactic_calls":0}
    write(out/"manifest.json",m)
    return m


def teacher(rg,suite,out):
    cases,manifest,witnesses=rg.load_split(suite,"train",witnesses=True)
    if len(cases)!=24 or manifest["seed"]!=GENERATOR_SEED:
        raise ValueError("registered source roster changed")
    items=[(c,witnesses[c.root],"train_"+c.root) for c in cases]
    receipt=certify(rg,items,out/"certificates")
    frames=[]
    for c,path,_ in items:
        states=rg.replay(c,path)
        for i in range(len(path)):
            menu=rg.neighbors(c,states[i])
            labels=[int(m.successor==states[i+1]) for m in menu]
            if not any(labels):
                raise ValueError("witness successor absent from public menu")
            frames.append({"root":c.root,"lineage":c.lineage,"step":i,
                           "packet":rg.public_packet(c,states[i],DEPTH-i),
                           "menu":[rg.action_token(m) for m in menu],"remaining":min(8,DEPTH-i),
                           "witness_labels":labels})
    write_jsonl(out/"train_frames.jsonl",frames)
    m={"schema":"lbf12.train_witness_frames.v1","suite_manifest_sha256":file_sha(suite/"manifest.json"),
       "certificates_sha256":file_sha(out/"certificates"/"manifest.json"),
       "roots":len(cases),"frames":len(frames),"compiled_axiom_free":receipt["compiled_axiom_free"],
       "train_frames_sha256":file_sha(out/"train_frames.jsonl"),
       "label_semantics":"same witnessed successor; all unselected actions unknown utility",
       "calibration_outcomes_opened":False,"shortest_claimed":False}
    write(out/"manifest.json",m)
    return frames,m


def fit(frames,teacher_sha,out):
    if out.exists():
        raise ValueError("fresh model destination required")
    torch.manual_seed(SEED)
    model=PlasticProofTransformer(Config(plan="b",width=32,heads=4,max_bytes=384)).cpu().float()
    if model.parameter_count()!=57475:
        raise ValueError("model parameter budget changed")
    init=_model_fingerprint(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    byroot=defaultdict(list)
    for f in frames: byroot[f["root"]].append(f)
    roster=sorted(byroot)
    rng=random.Random(SEED); draws=[]; t0=time.monotonic()
    for i in range(STEPS):
        if time.monotonic()-t0>300:
            raise TimeoutError("fixed CPU training cap reached")
        root=rng.choice(roster); row=rng.choice(byroot[root])
        outputs=model(row["packet"],row["menu"],depth=2,checks_remaining=row["remaining"],
                      return_all=True,middle_order="ab")
        mask=torch.tensor(row["witness_labels"],dtype=torch.bool)
        loss=torch.stack([torch.logsumexp(p["logits"],0)-torch.logsumexp(p["logits"][mask],0)
                          for p in outputs]).mean()
        if not bool(torch.isfinite(loss)): raise ValueError("nonfinite witness loss")
        opt.zero_grad(set_to_none=True); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)): raise ValueError("nonfinite gradient")
        opt.step();draws.append({"step":i+1,"root":root,"source_step":row["step"],"loss":float(loss.detach())})
    model.eval();out.mkdir(parents=True)
    torch.save({"schema":"lbf12.rewrite_ranker.v1","config":asdict(model.cfg),"weights":model.state_dict(),
                "seed":SEED,"steps":STEPS,"teacher_manifest_sha256":teacher_sha},out/"checkpoint.pt")
    write_jsonl(out/"optimizer.jsonl",draws)
    m={"schema":"lbf12.fit_receipt.v1","seed":SEED,"steps":STEPS,"parameters":57475,
       "initial":init,"final":_model_fingerprint(model),"teacher_manifest_sha256":teacher_sha,
       "checkpoint":file_sha(out/"checkpoint.pt"),"trace":file_sha(out/"optimizer.jsonl"),
       "calibration_optimizer_updates":0}
    write(out/"manifest.json",m)
    return model,m


def rms(x):
    return float(x.detach().float().square().mean().sqrt())


def capture(model,packet,menu,remaining,mode):
    """Observe exact execution boundaries. All captured data are predecision."""
    buffers=defaultdict(list); handles=[]
    def clone(x):
        return x.detach().cpu().clone()
    def attach(module,name,gate=False):
        def hook(_module,args,out):
            row={"out":clone(out)}
            if gate: row["state_in"],row["bank_in"]=clone(args[0]),clone(args[1])
            buffers[name].append(row)
        handles.append(module.register_forward_hook(hook))
    for name in ("gate_a","gate_b"):
        module=getattr(model,name)
        attach(module,name,True);attach(module.gate,name+"_strength");attach(module.proj,name+"_projection")
    for name in ("middle_a","middle_b","message_norm","coda"):
        attach(getattr(model,name),name)
    try:
        with torch.no_grad():
            outputs=model(packet,menu,depth=2,checks_remaining=remaining,return_all=True,
                          middle_order="ab",message_mode=mode)
    finally:
        for handle in handles: handle.remove()
    if set(buffers)!={"gate_a","gate_b","gate_a_strength","gate_b_strength","gate_a_projection",
                     "gate_b_projection","middle_a","middle_b","message_norm","coda"} or any(len(x)!=2 for x in buffers.values()):
        raise ValueError("missing or duplicated actual execution boundary")
    with torch.no_grad():
        baseline=model(packet,menu,depth=2,checks_remaining=remaining,return_all=True,
                       middle_order="ab",message_mode=mode)
    for i,(a,b) in enumerate(zip(outputs,baseline)):
        if any(not torch.equal(a[k],b[k]) for k in ("logits","outcome","state","features")):
            raise ValueError("observation hooks altered the model forward")
        ga,gb=buffers["gate_a"][i],buffers["gate_b"][i]
        if not torch.equal(ga["bank_in"],gb["bank_in"]):
            raise ValueError("A/B incoming message disagrees")
        bank=ga["bank_in"]
        expected=torch.zeros_like(bank) if i==0 or mode=="zero" else buffers["message_norm"][i-1]["out"]
        if i and mode=="roll": expected=expected.roll(1,dims=1)
        torch.testing.assert_close(bank,expected,rtol=0,atol=0)
        for name in ("gate_a","gate_b"):
            contribution=buffers[name][i]["out"]
            predicted=buffers[name+"_strength"][i]["out"].sigmoid()*buffers[name+"_projection"][i]["out"]
            torch.testing.assert_close(contribution,predicted,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(a["state"].cpu(),buffers["coda"][i]["out"],rtol=0,atol=0)
    return outputs,buffers


def observe(model,case,packet,menu,remaining,checkpoint_sha,ordinal):
    """Current menu only, with no labels or outcome supplied to this function."""
    if not menu or len(menu)>64 or len(set(menu))!=len(menu):
        raise ValueError("invalid decision-local public menu")
    normal,bank=capture(model,packet,menu,remaining,"normal")
    sham={mode:capture(model,packet,menu,remaining,mode) for mode in ("zero","roll")}
    identity=digest([case.root,ordinal,packet,menu])
    source_sha=digest([packet,menu,remaining])
    record={"schema":SCHEMA_OBS,"root_id":"rewrite:"+identity,"lineage":case.lineage,
            "split":"calibration","source":{"kind":"ppt_replay","source_sha256":source_sha,
            "checkpoint_sha256":checkpoint_sha},"candidates":list(menu),"visits":[]}
    measurements=[]
    for i,o in enumerate(normal):
        coda=o["state"][0].detach().cpu()
        incoming=bank["gate_a"][i]["bank_in"]
        def squash(x): return x/(1+x)
        tiles=[]
        for j,role in enumerate(ROLES):
            value=rms(incoming) if role=="message" else rms(coda[j])
            tiles.append({"candidate_id":"shared","role":role,"frame":"memory" if role=="message" else "state",
                          "activation":squash(value),"polarity":0})
        for candidate,z in zip(menu,o["logits"]):
            tiles.append({"candidate_id":candidate,"role":"value","frame":"action",
                          "activation":float(z.sigmoid()),"polarity":0})
        record["visits"].append({"attempt":1,"pass":i+1,"tiles":tiles})
        row={"decision_id":record["root_id"],"root":case.root,"lineage":case.lineage,
             "pass":i+1,"menu_sha256":digest(menu),"incoming_bank_rms":rms(incoming),
             "outgoing_bank_rms":rms(bank["message_norm"][i]["out"]),
             "gate_a_strength_mean":float(bank["gate_a_strength"][i]["out"].sigmoid().mean()),
             "gate_b_strength_mean":float(bank["gate_b_strength"][i]["out"].sigmoid().mean()),
             "gate_a_contribution_rms":rms(bank["gate_a"][i]["out"]),
             "gate_b_contribution_rms":rms(bank["gate_b"][i]["out"]),
             "middle_a_rms":rms(bank["middle_a"][i]["out"]),
             "middle_b_rms":rms(bank["middle_b"][i]["out"]),
             "coda_rms":rms(coda)}
        for mode,(alt,abank) in sham.items():
            diff=float((o["logits"]-alt[i]["logits"]).abs().max())
            if i==0 and diff>1e-5: raise ValueError("first pass changed under message sham")
            if mode=="roll" and i:
                # At pass 2 the input is the rotated first outgoing bank;
                # norm matching follows from the identical first-pass states.
                torch.testing.assert_close(incoming.norm(),abank["gate_a"][i]["bank_in"].norm(),rtol=1e-6,atol=1e-6)
            row[mode+"_max_logit_delta"]=diff
            row[mode+"_top_changed"]=int(o["logits"].argmax())!=int(alt[i]["logits"].argmax())
        measurements.append(row)
    validate_observation(record)
    if record["candidates"]!=menu or any(v["attempt"]!=1 for v in record["visits"]):
        raise ValueError("decision-local observation contract failed")
    return record,measurements,normal[-1]["logits"].tolist()


def bootstrap(vals):
    rng=random.Random(20261123)
    samples=sorted(sum(rng.choice(vals) for _ in vals)/len(vals) for _ in range(2000))
    return [samples[50],samples[1949]]


def campaign(rg,out):
    if out.exists(): raise ValueError("fresh LBF-12 output required")
    out.mkdir(parents=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
    # Real Lean smoke includes nested forward, inverse and dependent transport.
    a=rg.Case("native_transport_fixture","fixture","train","dependent_transport","aabc","abcc",(("ab","bc"),)).validate()
    b=rg.Case("native_inverse_fixture","fixture","train","equality","abcc","aabc",(("ab","bc"),)).validate()
    fixture=certify(rg,[(a,[rg.Move(0,0,1,"abcc")],"dependent_fixture"),
                      (b,[rg.Move(0,1,1,"aabc")],"inverse_fixture")],out/"native_fixtures")
    suite=out/"suite";sm=rg.freeze(suite,GENERATOR_SEED,COUNTS)
    if sm["counts"]!=COUNTS or sm["unique_rule_system_lineages"]!=sum(COUNTS.values()):
        raise ValueError("registered frozen rule-system census changed")
    frames,teacher_m=teacher(rg,suite,out/"teacher")
    teacher_sha=file_sha(out/"teacher"/"manifest.json")
    model,receipt=fit(frames,teacher_sha,out/"fit")
    duplicate,repeat=fit(frames,teacher_sha,out/"duplicate")
    if receipt!=repeat or any(not torch.equal(x,duplicate.state_dict()[name])
                              for name,x in model.state_dict().items()):
        raise ValueError("same-worker duplicate fit mismatch")
    del duplicate
    for p in model.parameters(): p.requires_grad_(False)
    cases,_=rg.load_split(suite,"calibration")
    if len(cases)!=16 or {c.lineage for c in cases}&{f["lineage"] for f in frames}:
        raise ValueError("calibration lineage overlaps train")
    frozen=_model_fingerprint(model)
    observations=[];measurements=[];rows=[];proofs=[]
    for c in cases:
        for policy in ("astar","bidir"):
            r=rg.search(c,policy,budget=WORK,depth=DEPTH)
            rows.append({**r,"condition":policy,"stream":None})
        for seed in RANDOM_SEEDS:
            r=rg.search(c,"uniform",budget=WORK,depth=DEPTH,seed=int(digest([seed,c.root])[:16],16))
            rows.append({**r,"condition":"uniform","stream":seed})
        for mode in MODES:
            decisions=0
            def ranker(case,word,moves,remaining):
                nonlocal decisions
                decisions+=1
                menu=[rg.action_token(m) for m in moves]
                packet=rg.public_packet(case,word,remaining)
                remain=min(8,max(1,remaining))
                with torch.no_grad():
                    if mode=="normal" and decisions<=4:
                        ob,ms,scores=observe(model,case,packet,menu,remain,receipt["checkpoint"],decisions)
                        observations.append(ob);measurements.extend(ms)
                        return scores
                    return model(packet,menu,depth=2,checks_remaining=remain,middle_order="ab",
                                 message_mode=mode)[-1]["logits"].tolist()
            r=rg.search(c,"learned",budget=WORK,depth=DEPTH,ranker=ranker)
            if r["model_calls"]!=decisions:
                raise ValueError("policy forward accounting differs")
            rows.append({**r,"condition":mode,"stream":None})
    if frozen!=_model_fingerprint(model):
        raise ValueError("frozen model weights changed during query evaluation")
    lookup={c.root:c for c in cases}
    for i,row in enumerate(rows):
        row["certificate_name"]=None
        if row["closed"]:
            rg.replay(lookup[row["root"]],row["path"])
            name=f"policy_{i}"
            proofs.append((lookup[row["root"]],row["path"],name))
            row["certificate_name"]=name
    policy_proofs=len(proofs)
    # Independent reference outcomes are opened only after the frozen policies.
    _,_,ref=rg.load_split(suite,"calibration",witnesses=True)
    proofs.extend((c,ref[c.root],"reference_"+c.root) for c in cases)
    cert=certify(rg,proofs,out/"calibration_certificates")
    passed=set(cert["declarations"])
    for row in rows:
        row["native_certified"]=row["closed"] and row["certificate_name"] in passed
    if cert["compiled_axiom_free"]!=policy_proofs+16 or len(rows)!=16*10:
        raise ValueError("incomplete native policy/reference receipt")
    if len(observations)!=len({o["root_id"] for o in observations}) or len(measurements)!=2*len(observations):
        raise ValueError("decision-local clock/identity mismatch")
    write_jsonl(out/"search_episodes.jsonl",rows)
    write_jsonl(out/"observations.jsonl",observations)
    write_jsonl(out/"boundary_measurements.jsonl",measurements)
    encoded=[encode(o) for o in observations]
    write_jsonl(out/"encoded.jsonl",encoded)
    if encoded:
        (out/"fresco.svg").write_text(fresco_svg(encoded[0]),encoding="utf-8")
    conditions=("normal","zero","roll","astar","bidir","uniform")
    byroot=defaultdict(list)
    for row in rows: byroot[row["root"]].append(row)
    pairs={};rates={};cost={}
    for name in conditions:
        rr=[r for r in rows if r["condition"]==name]
        rates[name]=sum(r["native_certified"] for r in rr)/len(rr)
        cost[name]={"episodes":len(rr),"proofs":sum(r["native_certified"] for r in rr),
                    "work":sum(r["work"] for r in rr),"model_calls":sum(r["model_calls"] for r in rr),
                    "expanded":sum(r["expanded"] for r in rr)}
        if any(len([r for r in byroot[c.root] if r["condition"]==name])!=(5 if name=="uniform" else 1) for c in cases):
            raise ValueError("root paired condition incomplete")
    def rootrate(root,condition):
        rr=[r for r in byroot[root] if r["condition"]==condition]
        return sum(r["native_certified"] for r in rr)/len(rr)
    for name in conditions[1:]:
        vals=[rootrate(c.root,"normal")-rootrate(c.root,name) for c in cases]
        pairs["normal_minus_"+name]={"mean":sum(vals)/16,"descriptive_95":bootstrap(vals),
                                     "wins":sum(v>0 for v in vals),"losses":sum(v<0 for v in vals),
                                     "ties":sum(v==0 for v in vals)}
    result={"schema":"lbf12.decision_local_rewrite_pilot.v1",
        "upstream_commit":UPSTREAM_COMMIT,"upstream_blob":UPSTREAM_BLOB,
        "suite_manifest_sha256":file_sha(suite/"manifest.json"),"teacher_manifest_sha256":teacher_sha,
        "train_roots":24,"train_frames":teacher_m["frames"],"train_certificates":teacher_m["compiled_axiom_free"],
        "native_fixture_certificates":fixture["compiled_axiom_free"],"same_worker_repeat":True,
        "checkpoint":receipt["checkpoint"],"parameters":57475,"updates":STEPS,"seed":SEED,
        "calibration_roots":16,"calibration_reference_certificates":16,"policy_certificates":policy_proofs,
        "calibration_certificate_manifest_sha256":file_sha(out/"calibration_certificates"/"manifest.json"),
        "work_budget":WORK,"path_cap":DEPTH,"rates":rates,"costs":cost,"paired":pairs,
        "memory_screen":all(pairs["normal_minus_"+m]["mean"]>=.05 for m in ("zero","roll")),
        "numerical_competence":rates["normal"]>=.4 and rates["normal"]-max(rates[k] for k in ("astar","bidir","uniform"))>=.1,
        "decision_observations":len(observations),"actual_boundary_pass_records":len(measurements),
        "max_zero_logit_delta":max((r["zero_max_logit_delta"] for r in measurements),default=0),
        "max_roll_logit_delta":max((r["roll_max_logit_delta"] for r in measurements),default=0),
        "zero_top_changes":sum(r["zero_top_changed"] for r in measurements),
        "roll_top_changes":sum(r["roll_top_changed"] for r in measurements),
        "hook_free_equivalence":True,"actual_gate_and_bank_checks":True,
        "labels_exported_to_observations":False,"online_native_tactic_calls":0,
        "calibration_optimizer_updates":0,"ordinary_ood_outcomes_opened":False,
        "scientific_eligible":False,"architecture_promotion_eligible":False,
        "artifacts":{name:file_sha(out/name) for name in ("search_episodes.jsonl","observations.jsonl","boundary_measurements.jsonl","encoded.jsonl","fresco.svg")}}
    write(out/"manifest.json",result)
    print("LBF12_RESULT",canonical(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    campaign(load_upstream(a.upstream),a.out)


if __name__=="__main__":main()

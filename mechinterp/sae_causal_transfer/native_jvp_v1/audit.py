"""Independent stdlib-only LS-SAE03 saved numerical receipt and gate replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from mechinterp.sae_causal_transfer.native_trm_v1 import families as OLD
from mechinterp.sae_causal_transfer.native_jvp_v1 import families as NEW

CHECKPOINT="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
FIT_SHA="f217453dc17e078b055e99f3cebc3cf736af7e7cfae2ab120353c91bf2554c9d"
DICT_SHA="decf90fd73137bd232ba40104cf387e894c4a3d1f4f007036c591d173f3e238c"
PROTO="07490150318b94ca5d035d928ff3255c9d6b60ff"


def close(a,b,atol=2e-5,rtol=3e-5):
    if a is None or b is None:
        return a is None and b is None
    return (math.isfinite(a) and math.isfinite(b)
            and abs(a-b)<=atol+rtol*max(abs(a),abs(b)))


def dot(x,y):
    if len(x)!=len(y):raise ValueError("vector shape drift")
    return sum(a*b for a,b in zip(x,y))


def norm(x):
    return math.sqrt(sum(y*y for y in x))


def _event(condition):
    if condition is None:return False
    s0=condition["0"]["task_margin_slope"]
    s2=condition["2"]["task_margin_slope"]
    gain=condition["2"]["full_state_gain"]
    return bool(abs(s0)>=.05 and abs(s2)/(abs(s0)+.000001)<=.10 and gain>=.25)


def validate_roster(raw,mode):
    rosters=raw["new_test_rosters"]
    assert set(rosters)==set(NEW.NEW_BASES)
    old,_=OLD.build_rosters(pilot=False)
    original_full={tuple(r["target"]) for rows in old.values() for r in rows}
    original_input={tuple(r["x"]) for rows in old.values() for r in rows}
    all_full=set();all_input=set()
    for family,base in NEW.NEW_BASES.items():
        rows=rosters[family]
        if len(rows)!=(2 if mode=="pilot" else 12):
            raise ValueError("new roster cardinality changed")
        if OLD.fingerprint(rows)!=raw["new_test_roster_sha256"][family]:
            raise ValueError("new roster SHA256 mismatch")
        for idx,row in enumerate(rows):
            if row["seed"]!=base+idx or row["holes"]!=NEW.NEW_HOLES[family]:
                raise ValueError("seed or hole count changed")
            y=tuple(row["target"]);x=tuple(row["x"])
            if (len(x)!=81 or x.count(0)!=row["holes"]
                or not OLD.valid_solution(y) or OLD.rectangles(y)!=row["rectangles"]
                or row["rectangles"]<=0 or OLD.count_solutions(x)!=1
                or any(v and v!=y[i] for i,v in enumerate(x))):
                raise ValueError("heldout Sudoku invalid/ambiguous")
            if (y in original_full or x in original_input
                or y in all_full or x in all_input):
                raise ValueError("heldout Sudoku was already used for fit/test")
            all_full.add(y);all_input.add(x)


def validate_direction(name,observed,unit,anchor_pos,gradient,readouts,rank):
    if observed is None:
        if unit is not None:raise ValueError("declared direction unscored")
        return
    if unit is None or len(unit)!=512 or not close(norm(unit),1,atol=2e-6):
        raise ValueError("JVP direction lacks 512D unit norm")
    if set(observed)!={"0","1","2"}:
        raise ValueError("ACT depth grid was changed")
    for k in range(3):
        row=observed[str(k)]
        tnorms=row["all97_token_norms"]
        g=row["goal_tangent_512"]
        w=row["readout_row_512"]
        if len(tnorms)!=97 or len(g)!=512 or len(w)!=512:
            raise ValueError("incomplete 97-token full-state JVP diagnostics")
        if not all(math.isfinite(v) and v>=0 for v in tnorms):
            raise ValueError("nonfinite/negative token tangent norm")
        if row["full_tangent_float32_sha256"] and len(row["full_tangent_float32_sha256"])!=64:
            raise ValueError("bad original tangent-state SHA256")
        total=norm(tnorms)
        goalnorm=norm(g)
        off=math.sqrt(max(0,total*total-goalnorm*goalnorm))
        if (not close(total,row["full_state_gain"]) or
            not close(goalnorm,tnorms[anchor_pos]) or
            not close(goalnorm,row["goal_state_gain"]) or
            not close(off,row["other_tokens_gain"])):
            raise ValueError("full-state JVP energy decomposition changed")
        ratio=goalnorm*goalnorm/(total*total) if total>1e-12 else None
        if not close(ratio,row["target_energy_fraction"]):
            raise ValueError("goal-vs-off-target signal partition differs")
        slope=dot(g,w)
        if not close(slope,row["task_margin_slope"]):
            raise ValueError("readout projected JVP task slope modified")
        den=norm(w)*goalnorm
        cosine=abs(slope)/den if den>1e-12 else None
        if not close(cosine,row["task_readout_abs_cosine"]):
            raise ValueError("readout alignment cosine changed")
        if len(readouts[k]["readout"])!=512:
            raise ValueError("clean depth-specific native head rows missing")
        if any(not close(x,y) for x,y in zip(w,readouts[k]["readout"])):
            raise ValueError("JVP used a different clean target-rival row")
        sec=(row["finite_plus_margin"]-row["finite_minus_margin"])/.4
        if (not close(sec,row["symmetric_finite_slope"]) or
            not close(sec-slope,row["finite_minus_jvp_slope"])):
            raise ValueError("finite causal secant changed")
        if k==0:
            if (not close(total,1) or not close(goalnorm,1)
                or not close(off,0)):
                raise ValueError("injection tangent is no longer a single-token unit vector")
            if any(not close(x,y,atol=1e-6) for x,y in zip(g,unit)):
                raise ValueError("k0 JVP differs from the selected physical 512D direction")
        if k==2:
            reverse=dot(gradient,unit)
            if not close(reverse,row["reverse_mode_slope"],atol=.02,rtol=.05):
                raise ValueError("original full-BPTT gradient scalar changed")
            if abs(reverse-slope)>max(.02,.05*abs(reverse))+1e-5:
                raise ValueError("native reverse-mode / forward-mode derivative discrepancy")
    if name=="head_null" and abs(observed["0"]["task_margin_slope"])>.02:
        raise ValueError("head-null vector affects the immediate digit readout")


def validate_case(case,source,family,index):
    if case["puzzle_seed"]!=source["seed"] or case["puzzle_index"]!=index:
        raise ValueError("test puzzle order changed")
    position=NEW.median_blank(source["x"])
    if case["goal_position"]!=position:
        raise ValueError("goal token selected from Sudoku target")
    if case["true_digit_zero_based"]!=source["target"][position]-1:
        raise ValueError("ground-truth score digit differs")
    if set(case["directions"])!={"sae","task_head","head_null","random"}:
        raise ValueError("missing registered control arm")
    if set(case["source_directions"])!=set(case["directions"]):
        raise ValueError("physical direction set changed")
    y=case["true_digit_zero_based"]
    readouts=case["clean_readouts"]
    if len(readouts)!=3:
        raise ValueError("ACT 0/1/2 clean task readouts missing")
    for k,row in enumerate(readouts):
        raw=row["clean_logits9"]
        if (row["depth"]!=k or row["target"]!=y or len(raw)!=9
            or any(not math.isfinite(z) for z in raw)):
            raise ValueError("invalid clean native logits")
        other=raw[:];other[y]=-float("inf")
        rival=max(range(9),key=lambda i:other[i])
        if row["rival"]!=rival or rival==y:
            raise ValueError("non-clean rival or incorrect target")
    meta=case["source_direction_metadata"]
    clean0=readouts[0]["clean_logits9"]
    winning=max(range(9),key=lambda i:clean0[i])
    if meta["clean_winner"]!=winning:
        raise ValueError("source decision uses non-clean predicted winner")
    grad=case["full_bptt_gradient_512"]
    if len(grad)!=512 or not all(math.isfinite(v) for v in grad):
        raise ValueError("missing full source gradient")
    for name,row in case["directions"].items():
        validate_direction(name,row,case["source_directions"][name],
                           16+position,grad,readouts,case["rank_one_nulls"])
    is_admitted=case["directions"]["sae"] is not None
    if is_admitted!=case["sae_admitted"]:
        raise ValueError("SAE pre-outcome feature-selection changed")
    ranks=case["rank_one_nulls"]
    if is_admitted and norm(grad)>1e-12:
        cos=abs(dot(grad,case["source_directions"]["sae"]))/norm(grad)
        null=ranks["random_cosines"]
        if len(null)!=128 or any(not math.isfinite(q) or not 0<=q<=1+1e-7 for q in null):
            raise ValueError("unregistered 128 rank-one observer null controls")
        p=(1+sum(z>=cos-1e-15 for z in null))/129
        if not close(cos,ranks["sae_cosine"]) or not close(p,ranks["rank_p"]):
            raise ValueError("rank-observer empirical probability differs")
    elif ranks["rank_p"] is not None:
        raise ValueError("ineligible SAE received nonnull rank score")
    if case["sae_event"]!=_event(case["directions"]["sae"]) or case["random_event"]!=_event(case["directions"]["random"]):
        raise ValueError("published attenuation event changed")


def summary(rows,family):
    events=[(r["sae_event"],r["random_event"]) for r in rows]
    saeadmit=sum(r["sae_admitted"] for r in rows)
    a=sum(p for p,_ in events);b=sum(q for _,q in events)
    rng=random.Random(202622130+len(family))
    bootstrap=[]
    for _ in range(2000):
        selected=[events[rng.randrange(len(events))] for _ in events]
        bootstrap.append(sum(int(x)-int(y) for x,y in selected)/len(selected))
    bootstrap.sort()
    ci=[bootstrap[50],bootstrap[1949]]
    return {"n_puzzles":len(rows),"sae_admitted":saeadmit,
            "sae_attenuation_with_survival_events":a,
            "random_attenuation_with_survival_events":b,
            "paired_event_fraction_difference":(a-b)/len(rows),
            "paired_puzzle_bootstrap95":ci,
            "valid_bootstrap_draws":len(bootstrap),
            "registered_success":len(rows)==12 and saeadmit>=10 and a>=8 and ci[0]>0}


def audit(receipt):
    if (receipt["schema"]!="LS-SAE03/full-native-autograd-JVP-512-v1"
        or receipt["preregistration_commit"]!=PROTO
        or receipt["source"]["original_checkpoint_file_sha256"]!=CHECKPOINT
        or receipt["source"]["old_fit_PCA_SAE_sha256"]!=FIT_SHA
        or receipt["source"]["old_physical_dictionary_sha256"]!=DICT_SHA
        or receipt["source"]["old_fit_roster_sha256"]!="915194003397d0fbc43bea30f13f6161af36b579fb9c1928d4d4fafc78c3128d"
        or receipt["radius"]!=.2
        or receipt["depths"]!=[0,1,2]
        or receipt["random_rank_one_observer_budget"]!=128):
        raise ValueError("native LS-SAE03 source, budget or registration differs")
    if receipt["mode"] not in ("pilot","full") or receipt["full_protocol"]!=(receipt["mode"]=="full"):
        raise ValueError("pilot/full status changed")
    validate_roster(receipt,receipt["mode"])
    for family,group in receipt["heldout"].items():
        rows=group["per_puzzle"]
        if len(rows)!=len(receipt["new_test_rosters"][family]):
            raise ValueError("output family roster differs")
        for i,(case,source) in enumerate(zip(rows,receipt["new_test_rosters"][family])):
            validate_case(case,source,family,i)
        recomputed=summary(rows,family)
        for key,val in recomputed.items():
            old=group["statistics"][key]
            if isinstance(val,list):
                if len(val)!=len(old) or any(not close(x,y) for x,y in zip(val,old)):
                    raise ValueError("bootstrap interval differs")
            elif isinstance(val,float):
                if not close(val,old):raise ValueError("paired puzzle score changed "+key)
            elif old!=val:
                raise ValueError("registered count/decision changed "+key)
    joint=(receipt["full_protocol"] and
           all(group["statistics"]["registered_success"] for group in receipt["heldout"].values()))
    if joint!=receipt["registered_success_both_families"]:
        raise ValueError("two-family scientific gate modified")
    return {
      "schema":"LS-SAE03/independent-stdlib-numerical-replay-v1",
      "source_fingerprint_and_fresh_roster_checked":True,
      "all97_token_norms_and512_goal_jvp_readouts_checked":True,
      "native_512D_vjp_and_symmetric_secants_replayed":True,
      "128_rank_null_controls_and_puzzle_cluster_gate_replayed":True,
      "puzzles":sum(len(x["per_puzzle"]) for x in receipt["heldout"].values()),
      "full_protocol":receipt["full_protocol"],
      "registered_success_both_families":joint,
      "scope":"Audits archived numerical statements; does not reexecute original pretrained model or independently verify tangent tensor SHA contents."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data=a.input.read_bytes()
    result=audit(json.loads(data))
    result["raw_receipt_sha256"]=hashlib.sha256(data).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()

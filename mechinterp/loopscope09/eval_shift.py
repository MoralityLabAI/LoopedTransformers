"""LGA-06: original frozen E(h) under matched source shifts and 12 actual loops.

Original teacher weights and scalar potential never change. After pretest
source lock, fresh 31058 IID input is transformed into three deterministic
source-matched OOD distributions, with one correct cyclic-copy label per arm.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import torch
from torch.nn import functional as F
from mechinterp.loopscope04.gradient_flow import (
    fixed_data,per_token_ce,tensor_int64_sha,head_rowspace
)
from mechinterp.loopscope05.fit_and_freeze import digest,load_frozen
from mechinterp.loopscope05.terminal_sae_test import make_model,make_sae,active_components
from mechinterp.loopscope08.fit_potential import load_potentials,potential_gradient,OLD_SOURCES
from mechinterp.loopscope08.eval_energy import radial_gradient,radial_energy

SCHEMA="LGA-06/frozen-shifted-12-pass-native-energy-v1"
LOCK_SCHEMA="LGA-06/frozen-pretest-source-and-hypothesis-v1"
TEACHERS=(103,211,307)
ARMS=("iid","period9","block9","zero20")
N_LOOPS=12
TEST_SEED=31058
SMOKE_SEED=61058
N_TEST=256
N_NATIVE=64
GOALS=(0,10,20,30,40,50,60,70)
NATIVE_DEPTHS=(2,6)
EDIT_ARMS=("shared_down","init_down","radial_down","native",
           "sae_active","sae_remainder","head_null","random")
CONTRASTS=(("shared_down","random"),("shared_down","native"),
           ("shared_down","init_down"),("sae_active","sae_remainder"),
           ("head_null","random"))
BOOT=2000
ALL_PREVIOUS=(*OLD_SOURCES,(31055,256),(31056,64),(31057,256))


def protocol_sha():
    return digest((Path(__file__).resolve().parent/"PROTOCOL.md").read_bytes())


def stable_seed(*parts):
    x="LGA06|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(x.encode()).digest()[:8],"big")&((1<<63)-1)


def create_source(seed=TEST_SEED,n=N_TEST):
    """Deterministic matched OOD transforms, with 256 independent base IDs."""
    x,y=fixed_data(n,seed)
    seen_prior=set()
    for sd,size in ALL_PREVIOUS:
        old,_=fixed_data(size,sd)
        seen_prior.update(tuple(r) for r in old.tolist())
    if any(tuple(r) in seen_prior for r in x.tolist()):
        raise ValueError("new IID source overlaps prior data")
    zero=x.clone()
    zero[:,::5]=0
    variations={
        "iid":x,
        "period9":x[:,:9].repeat(1,9),
        "block9":x[:,::9].repeat_interleave(9,dim=1),
        "zero20":zero,
    }
    manifest={
        "seed":seed,"n_base_independent_sequences":n,
        "base_x_sha256":tensor_int64_sha(x),
        "families":{}
    }
    all_sources={}
    for name in ARMS:
        a=variations[name]
        b=a.roll(1,1)
        if a.shape!=(n,81) or (not bool(((a>=0)&(a<=9)).all())):
            raise ValueError("wrong shifted 81-token supervised alphabet")
        if len(set(tuple(r) for r in a.tolist()))!=n:
            raise ValueError("transformation collapsed independent input-source identities")
        # An exact prior source collision is prevented before running the teacher.
        if any(tuple(r) in seen_prior for r in a.tolist()):
            raise ValueError("shifted source reused an earlier sample")
        all_sources[name]=(a,b)
        manifest["families"][name]={
            "n":n,"x_sha256":tensor_int64_sha(a),
            "y_sha256":tensor_int64_sha(b),
            "n_zero_input_tokens":int((a==0).sum()),
            "reference_base_indices":list(range(n)),
        }
    return x,all_sources,manifest


def cosine(v,g):
    return (v*g).sum(-1)/(
        v.norm(dim=-1)*g.norm(dim=-1)).clamp_min(1e-12)


def bootstrap(values,root,draws=BOOT):
    rng=random.Random(root)
    n=len(values)
    if n<2 or any(not math.isfinite(float(x)) for x in values):
        raise ValueError("whole-base-source bootstrap values invalid")
    samples=sorted(sum(values[rng.randrange(n)] for __ in range(n))/n
                   for _ in range(draws))
    return [samples[int(.025*draws)],samples[int(.975*draws)-1]]


def twelve_states(model,x):
    with torch.no_grad():
        state=model.initial_state(x)
        states=[state]
        for i in range(N_LOOPS):
            state=model.block(state)
            states.append(state)
    if len(states)!=13 or any(not bool(torch.isfinite(s).all()) for s in states):
        raise ArithmeticError("native 12-loop extrapolation produced nonfinite state")
    return states


def trajectories(model,potentials,initial,mu,std,x,y,chunk=16):
    """All four arms share model/potentials; one per-sequence row at every k."""
    raw={str(k):[] for k in range(N_LOOPS)}
    basis,rank=head_rowspace(model)
    for start in range(0,len(x),chunk):
        xb=x[start:start+chunk];yb=y[start:start+chunk]
        states=twelve_states(model,xb)
        with torch.no_grad():
            logits=[model.head(s) for s in states]
            losses=[per_token_ce(z,yb).mean(-1) for z in logits]
            acc=[(z.argmax(-1)==yb).float().mean(-1) for z in logits]
            shared=[potentials["shared"](s,0,mu,std).double().sum(-1) for s in states]
            init=[initial(s,0,mu,std).double().sum(-1) for s in states]
            radial=[radial_energy(s,mu,std).double() for s in states]
            # Only the ORIGINAL conditioned h0..h4 horizon embeddings exist.
            conditioned=[
                (potentials["conditioned"](states[k],k,mu,std).double().sum(-1),
                 potentials["conditioned"](states[k+1],k,mu,std).double().sum(-1))
                for k in range(4)
            ]
        for k in range(N_LOOPS):
            h=states[k]
            v=states[k+1]-h
            _,gshared=potential_gradient(potentials["shared"],h,0,mu,std)
            _,ginit=potential_gradient(initial,h,0,mu,std)
            grad_rad=radial_gradient(h,mu,std)
            c_shared=cosine(v,-gshared)
            c_init=cosine(v,-ginit)
            c_rad=cosine(v,-grad_rad)
            fraction=(v.double()@basis).square().sum(-1) / (
                v.double().square().sum(-1).clamp_min(1e-18))
            for i in range(len(xb)):
                before=float(shared[k][i])
                after=float(shared[k+1][i])
                ci=float(c_shared[i].mean())
                cn=float(c_init[i].mean())
                cr=float(c_rad[i].mean())
                row={
                    "shared_E_before":before,
                    "shared_E_after":after,
                    "shared_energy_drop":(before-after)/81,
                    "shared_positive_drop":float(before>after),
                    "initial_E_before":float(init[k][i]),
                    "initial_E_after":float(init[k+1][i]),
                    "initial_energy_drop":float((init[k][i]-init[k+1][i])/81),
                    "radial_E_before":float(radial[k][i]),
                    "radial_E_after":float(radial[k+1][i]),
                    "radial_energy_drop":float((radial[k][i]-radial[k+1][i])/81),
                    "shared_cosine":ci,
                    "initial_cosine":cn,
                    "radial_cosine":cr,
                    "gain_vs_initial":ci-cn,
                    "gain_vs_radial":ci-cr,
                    "CE_before":float(losses[k][i]),
                    "CE_after":float(losses[k+1][i]),
                    "accuracy_before":float(acc[k][i]),
                    "accuracy_after":float(acc[k+1][i]),
                    "mean_update_l2":float(v[i].norm(dim=-1).mean()),
                    "head_visible_update_energy":float(fraction[i].mean()),
                }
                if k<4:
                    p,q=conditioned[k]
                    row.update({
                        "conditioned_E_before":float(p[i]),
                        "conditioned_E_after":float(q[i]),
                        "conditioned_energy_drop":float((p[i]-q[i])/81),
                    })
                raw[str(k)].append(row)
    if any(len(rows)!=len(x) for rows in raw.values()):
        raise ValueError("original independent source sequence count changed")
    return raw


def summarize(raw,sid,arm,smoke=False):
    """Preregistered geometric and task-coupled transfer gates."""
    curves={}
    n=len(raw["0"])
    early=late=0
    n_boot=80 if smoke else BOOT
    for k in range(N_LOOPS):
        rows=raw[str(k)]
        means={key:sum(z[key] for z in rows)/n for key in rows[0]}
        fields=("shared_energy_drop","shared_positive_drop")
        if k<4:
            fields=fields+("gain_vs_initial","gain_vs_radial")
        ci={
            key:bootstrap([z[key] for z in rows],
                          stable_seed(46058,sid,arm,k,key),draws=n_boot)
            for key in fields
        }
        drop_ok=ci["shared_energy_drop"][0]>0 and ci["shared_positive_drop"][0]>.90
        cosine_ok=bool(k>=4 or
            (ci["gain_vs_initial"][0]>0 and ci["gain_vs_radial"][0]>0))
        passed=bool(drop_ok and cosine_ok)
        if k<4:early+=int(passed)
        else:late+=int(passed)
        curves[str(k)]={"n_original_sequences":n,"means":means,
                        "bootstrap95":ci,"pass":passed,"per_sequence":rows}
    early_all=sum(int(all(raw[str(k)][i]["shared_positive_drop"]==1. for k in range(4)))
                  for i in range(n))/n
    late_all=sum(int(all(raw[str(k)][i]["shared_positive_drop"]==1. for k in range(4,12)))
                 for i in range(n))/n
    all12=sum(int(all(raw[str(k)][i]["shared_positive_drop"]==1. for k in range(12)))
              for i in range(n))/n
    four=curves["3"]["means"]
    twelve=curves["11"]["means"]
    task_gate=bool(
        four["accuracy_after"]>=.60 and
        twelve["accuracy_after"]>=four["accuracy_after"]-.05 and
        twelve["CE_after"]<=four["CE_after"]+.05)
    geometry_gate=bool(early>=3 and late>=6)
    return {
        "n_original_sequences":n,
        "n_original_steps":N_LOOPS,
        "n_early_passes":early,"n_extra_passes":late,
        "fraction_early_all_positive":early_all,
        "fraction_extra_all_positive":late_all,
        "fraction_all12_positive":all12,
        "readout4_accuracy":four["accuracy_after"],
        "readout12_accuracy":twelve["accuracy_after"],
        "readout4_CE":four["CE_after"],
        "readout12_CE":twelve["CE_after"],
        "registered_geometric_pass":geometry_gate,
        "registered_task_preservation_pass":task_gate,
        "registered_cell_pass":bool(geometry_gate and task_gate),
        "curves":curves,
    }


def normalized(z,radius):
    norm=float(z.norm())
    if norm<=1e-8 or radius<=1e-8:return None
    result=(z*(radius/norm)).float().detach()
    if abs(float(result.norm())-radius)>1e-5:
        raise ArithmeticError("source physical edit not same L2 dose")
    return result


def edit_sha(z):
    return digest(struct.pack("<"+"f"*z.numel(),*z.reshape(-1).detach().float().tolist()))


def native_edits(model,shared,initial,mu,std,sae,x,y,sid,arm,k,smoke=False):
    """Source-only planned edits, with native remaining-loop task endpoints.

    At k2 readout 4; at k6 readout 12. The original SAE2 dictionary is
    intentionally re-used unchanged at both k2 and k6 (extrapolation control).
    """
    samples=min(4 if smoke else N_NATIVE,len(x))
    basis,_=head_rowspace(model)
    horizon=4 if k==2 else 12
    cases=[]
    for i in range(samples):
        xb=x[i:i+1];yb=y[i:i+1]
        state=twelve_states(model,xb)
        h=state[k][0].detach()
        native=state[k+1][0]-h
        _,gshared=potential_gradient(shared,h.unsqueeze(0),0,mu,std)
        _,ginit=potential_gradient(initial,h.unsqueeze(0),0,mu,std)
        rad=radial_gradient(h,mu,std)
        # Reuse the original source-fitted k2 SAE even on the extrapolated k6 state.
        dictionary,saemu,saestd=sae
        parts=active_components(dictionary,saemu,saestd,h,native)
        headnull=native.double()-(native.double()@basis)@basis.T
        chosen=[]
        planned=[]
        for j,tok in enumerate(GOALS):
            rho=.25*float(native[tok].norm())
            gen=torch.Generator().manual_seed(stable_seed("random",sid,arm,k,i,j))
            rand=torch.randn(48,dtype=torch.float32,generator=gen)
            directions={
                "shared_down":-gshared[0,tok],
                "init_down":-ginit[0,tok],
                "radial_down":-rad[tok],
                "native":native[tok],
                "sae_active":parts["active"][tok],
                "sae_remainder":parts["remainder"][tok],
                "head_null":headnull[tok].float(),
                "random":rand,
            }
            admit=bool(parts["admitted"][tok]) and rho>1e-8 and all(
                float(v.norm())>1e-8 for v in directions.values())
            note={"slot":j,"token":tok,"dose":rho,"source_admitted":admit,"arms":{}}
            if admit:
                for name,v in directions.items():
                    e=normalized(v,rho)
                    if e is None:raise ArithmeticError("source admitted direction unexpectedly degenerate")
                    chosen.append((j,tok,name,e))
            else:note["reason"]="source-degenerate-SAE-gradient-or-headnull"
            planned.append(note)
        # Every physical direction was frozen WITHOUT reading task outcomes.
        all_states=[h]
        for j,tok,name,e in chosen:
            z=h.clone();z[tok]+=e
            all_states.append(z)
        with torch.no_grad():
            z=torch.stack(all_states)
            source_edit_energies=shared(z,0,mu,std).double().sum(-1)
            for _ in range(horizon-k):
                z=model.block(z)
            logits=model.head(z)
            losses=per_token_ce(logits,yb.expand(len(z),-1))
            baseline=losses[0]
            clean_logits=logits[0]
            original=model.head(state[horizon])[0]
            if float((clean_logits-original).abs().max())>1e-5:
                raise ArithmeticError("source clean/zero injection native loop closure failed")
        for idx,(j,tok,name,edit) in enumerate(chosen,1):
            truth=int(yb[0,tok])
            ref=clean_logits[tok].clone()
            ref[truth]=-float("inf")
            rival=int(ref.argmax())
            out=logits[idx,tok]
            # Energy evaluated in one batch on *source* edited states,
            # before the future recurrent continuation and label scoring.
            eb=float(source_edit_energies[0])
            ea=float(source_edit_energies[idx])
            planned[j]["arms"][name]={
                "edit_L2":float(edit.norm()),"edit_sha256":edit_sha(edit),
                "truth_digit":truth,"clean_rival":rival,
                "goal_logits":out.tolist(),"clean_goal_logits":clean_logits[tok].tolist(),
                "all_final_token_CE":losses[idx].tolist(),
                "clean_all_final_token_CE":baseline.tolist(),
                "global_CE_change":float(losses[idx].mean()-baseline.mean()),
                "goal_CE_change":float(losses[idx,tok]-baseline[tok]),
                "goal_margin_change":float((out[truth]-out[rival])-
                                           (clean_logits[tok,truth]-clean_logits[tok,rival])),
                "shared_E_before":eb,"shared_E_after":ea,
                "edit_shared_energy_drop":(eb-ea)/81,
                "shared_firstorder_energy_drop":float(
                    -(gshared[0,tok].double()@edit.double())/81),
            }
        cases.append({"source_index":i,"original_goals":planned,
                      "native_endpoint":horizon})
    return cases


def causal_summary(cases,sid,arm,k,smoke=False):
    rows=[]
    for i,sample in enumerate(cases):
        valid=[g for g in sample["original_goals"] if g["source_admitted"]]
        c={}
        for a,b in CONTRASTS:
            key=f"{a}_minus_{b}"
            v=[g["arms"][a]["global_CE_change"]-
               g["arms"][b]["global_CE_change"] for g in valid]
            c[key]=sum(v)/len(v) if v else 0.
        rows.append({"source_index":i,"n_original_goals":8,
                     "n_admitted_goals":len(valid),"contrasts":c})
    comparisons={}
    for a,b in CONTRASTS:
        key=f"{a}_minus_{b}"
        v=[row["contrasts"][key] for row in rows]
        ci=bootstrap(v,stable_seed("causal",sid,arm,k,key),
                     draws=80 if smoke else BOOT)
        comparisons[key]={"mean":sum(v)/len(v),"bootstrap95":ci,
                          "direction":"POSITIVE" if ci[0]>0 else
                          "NEGATIVE" if ci[1]<0 else "INCONCLUSIVE"}
    return {"source_sequences":len(rows),"original_goals":len(rows)*8,
            "admitted_goals":sum(r["n_admitted_goals"] for r in rows),
            "native_target_readout":4 if k==2 else 12,
            "contrasts":comparisons,"per_sequence":rows,"raw_cases":cases}


def curl_witness(model,x,sid,arm,k,smoke=False):
    samples=min(4 if smoke else N_NATIVE,len(x))
    finite=copy.deepcopy(model).double().eval()
    eps=.001
    rows=[]
    with torch.no_grad():
        for i in range(samples):
            h=twelve_states(model,x[i:i+1])[k].double()
            gen=torch.Generator().manual_seed(stable_seed("curl",sid,arm,k,i))
            u=torch.randn(h.shape,dtype=torch.float64,generator=gen)
            w=torch.randn(h.shape,dtype=torch.float64,generator=gen)
            u=u/u.norm()
            w=w-(w*u).sum()*u
            w=w/w.norm()
            z=torch.cat([h+eps*u,h-eps*u,h+eps*w,h-eps*w])
            f=finite.block(z)-z
            Ju=(f[0]-f[1])/(2*eps)
            Jw=(f[2]-f[3])/(2*eps)
            a=float((u*Jw).sum())
            b=float((w*Ju).sum())
            rows.append({
                "source_index":i,"u_dot_J_w":a,"w_dot_J_u":b,
                "skew":a-b,
                "skew_ratio":abs(a-b)/(abs(a)+abs(b)+1e-8)
            })
    return {"n_source_sequences":len(rows),
            "mean_skew_ratio":sum(q["skew_ratio"] for q in rows)/len(rows),
            "per_sequence":rows}


def run(source_bundle,source_meta,energy_bundle,energy_meta,lock_path,out,smoke=False):
    torch.set_num_threads(2)
    old=[Path(z).read_bytes() for z in
         (source_bundle,source_meta,energy_bundle,energy_meta)]
    frozen=json.loads(Path(lock_path).read_bytes())
    if (frozen["schema"]!=LOCK_SCHEMA or frozen["protocol_sha256"]!=protocol_sha()
        or frozen["new_test_generated"] is not False or
        frozen["test_labels_read"] is not False or
        [frozen[k] for k in ("teacher_sha256","teacher_manifest_sha256",
                             "potential_sha256","potential_manifest_sha256")]!=[digest(z) for z in old]):
        raise ValueError("prospective source lock differs from original frozen trained models")
    source_meta_data,source=load_frozen(source_bundle,source_meta)
    base,families,manifest=create_source(SMOKE_SEED if smoke else TEST_SEED,
                                        8 if smoke else N_TEST)
    seeds=(103,) if smoke else TEACHERS
    result={}
    for sid in seeds:
        teacher=make_model(source,sid)
        potentials,initial,mu,std,rms=load_potentials(energy_bundle,energy_meta,sid)
        sae=make_sae(source,sid,2)
        result[str(sid)]={}
        for arm in ARMS:
            x,y=families[arm]
            profile=trajectories(teacher,potentials,initial,mu,std,x,y)
            measured=summarize(profile,sid,arm,smoke)
            measured["native_interventions"]={}
            measured["native_curl"]={}
            for k in (2,6):
                cases=native_edits(teacher,potentials["shared"],initial,mu,std,
                                   sae,x,y,sid,arm,k,smoke)
                measured["native_interventions"][str(k)]=causal_summary(
                    cases,sid,arm,k,smoke)
                measured["native_curl"][str(k)]=curl_witness(
                    teacher,x,sid,arm,k,smoke)
            result[str(sid)][arm]=measured
            print("LGA06_SHIFT",sid,arm,"geo",measured["registered_geometric_pass"],
                  "task",measured["registered_task_preservation_pass"],
                  "early",measured["n_early_passes"],
                  "late",measured["n_extra_passes"],
                  "acc4",round(measured["readout4_accuracy"],4),
                  "acc12",round(measured["readout12_accuracy"],4),flush=True)
    per_teacher={str(sid):{
        "all_four_geometric":all(result[str(sid)][arm]["registered_geometric_pass"] for arm in ARMS),
        "all_four_task":all(result[str(sid)][arm]["registered_task_preservation_pass"] for arm in ARMS),
        "all_four_joint":all(result[str(sid)][arm]["registered_cell_pass"] for arm in ARMS),
    } for sid in seeds}
    geometric=sum(x["all_four_geometric"] for x in per_teacher.values())
    robust=sum(x["all_four_joint"] for x in per_teacher.values())
    verdict=("PILOT_ONLY" if smoke else "PASS" if robust>=2 else "FAIL")
    source_digest=[digest(v) for v in old]
    record={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "source_lock_sha256":digest(Path(lock_path).read_bytes()),
        "source_artifact_sha256":source_digest,
        "new_test":manifest,"base_source_x":base.tolist(),
        "teacher_seeds":list(seeds),"distributions":list(ARMS),
        "n_recurrent_steps":N_LOOPS,
        "n_base_independent_sequences":len(base),
        "registered_primary_verdict":verdict,
        "n_all_arm_geometric_teacher_passes":geometric,
        "n_all_arm_robust_teacher_passes":robust,
        "teacher_summary":per_teacher,"results":result,
        "interpretation_limits":"Shared external potential trained to original native updates on old calibration. Structural shifts and extra loops are new test; task performance and curl remain independent controls."
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print("LGA06_ORIGINAL_FRESH_TEST_DONE",verdict,geometric,robust,flush=True)
    return record


def main():
    p=argparse.ArgumentParser()
    for name in ("source-bundle","source-meta","energy-bundle","energy-meta","lock","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.source_bundle,a.source_meta,a.energy_bundle,a.energy_meta,a.lock,a.out,a.smoke)


if __name__=="__main__":
    main()

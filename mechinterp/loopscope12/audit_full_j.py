"""LGA-09 independent Python-stdlib first-science audit, no Torch/NumPy.

Verify source-only SHA lock, every original legal donor, all 47,616 raw
physically applied pair edits, complete ten future logit first derivatives,
matched K/dose full-J shams, true ten-class outputs, 2,000 base-cluster
bootstraps, and registered development verdict.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import struct

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SCHEMA="LGA09/first-native-full-ten-logit-jacobian-pair-development-v1"
LOCK_SCHEMA="LGA09/pretest-source-and-all-ten-logit-code-lock-v1"
ORIGINAL={
  "teacher_bundle_sha256":"d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
  "teacher_meta_sha256":"7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d",
  "potential_bundle_sha256":"24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53",
  "potential_meta_sha256":"69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d",
}
GOALS=(0,10,20,30,40,50,60,70)
TEACHERS=(103,211,307)
N=64
SHAMS=8
ARMS=("pair_donor_k",*(f"full_j_sham_{i}" for i in range(SHAMS)),
      *(f"margin_sham_{i}" for i in range(SHAMS)),
      *(f"random_k_sham_{i}" for i in range(SHAMS)),
      "pair_head_only","natural_pair","goal_only_natural",
      "predecessor_only_natural","pair_quarter","full_donor_state")
EDIT_BYTES=96*4
SOURCE_BYTES=(96*3+3888+960)*4
BOOTS=2000
PROTO=("PROTOCOL.md","POWER_PREFLIGHT.md")
CODES=("source_lock.py","full_jacobian.py","native_full_j.py",
       "audit_full_j.py","tests/test_full_jacobian.py","tests/test_native_full_j.py")
PARENTS=(
 "mechinterp/loopscope11/pair_fiber.py",
 "mechinterp/loopscope10/donor_source.py",
 "mechinterp/loopscope10/global_energy_fiber.py",
 "mechinterp/loopscope10/energy_fiber.py",
 "mechinterp/loopscope04/gradient_flow.py",
 "mechinterp/loopscope05/fit_and_freeze.py",
 "mechinterp/loopscope08/fit_potential.py",
 ".github/workflows/loopscope12-source-only.yml",
 ".github/workflows/loopscope12-first-native.yml"
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    return sha(Path(path).read_bytes())


def close(a,b,what,atol=3e-5,rtol=4e-5):
    x=float(a);y=float(b)
    if not (math.isfinite(x) and math.isfinite(y)) or abs(x-y)>atol+rtol*max(abs(x),abs(y)):
        raise ValueError(f"first-original audit: {what}: {x} vs {y}")


def floats(raw,n):
    if len(raw)!=n*4:
        raise ValueError("physical float32 raw vector size mismatch")
    x=struct.unpack("<"+str(n)+"f",raw)
    if not all(math.isfinite(z) for z in x):
        raise ValueError("nonfinite physical source/edit vector")
    return x


def l2(v):
    return math.sqrt(math.fsum(float(a)*float(a) for a in v))


def dot(a,b):
    return math.fsum(float(x)*float(y) for x,y in zip(a,b))


def project_pair(v,p):
    return [math.fsum(v[t*48+i]*p[i][j] for i in range(48))
            for t in range(2) for j in range(48)]


def norm_to(v,r):
    n=l2(v)
    return [float(x)*r/n for x in v] if n>1e-10 and r>1e-10 else [0.]*len(v)


def head_pair_l2(v,w):
    return math.sqrt(math.fsum(dot(v[t*48:(t+1)*48],row)**2
        for t in range(2) for row in w))


def head_full_l2(v,w):
    return math.sqrt(math.fsum(dot(v[t*48:(t+1)*48],row)**2
        for t in range(81) for row in w))


def apply_ten_j(J,v):
    return [dot(J[c*96:(c+1)*96],v) for c in range(10)]


def source_seed(*parts):
    data=("LGA09|"+"|".join(map(str,parts))).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8],"big")&((1<<63)-1)


def bootstrap(rows,sid,contrast):
    if len(rows)!=N or any(not math.isfinite(float(a)) for a in rows):
        raise ValueError("64 independent base-cluster effects required")
    rng=random.Random(source_seed("bootstrap",sid,contrast))
    z=sorted(sum(rows[rng.randrange(N)] for _ in range(N))/N for _ in range(BOOTS))
    return [z[int(.025*BOOTS)],z[int(.975*BOOTS)-1]]


def median8(values):
    ordered=sorted(float(v) for v in values)
    if len(ordered)!=8 or any(z not in (0.,1.) for z in ordered):
        raise ValueError("8 binary per-case sham outcomes missing")
    return (ordered[3]+ordered[4])/2.


def logsumexp(a):
    mx=max(float(v) for v in a)
    return mx+math.log(math.fsum(math.exp(float(v)-mx) for v in a))


def ordered_sha(rows):
    flat=[int(v) for row in rows for v in row]
    return sha(struct.pack("<"+str(len(flat))+"q",*flat))


def verify_roster(roster):
    if (roster["seed"]!=31063 or roster["n_independent_bases"]!=N or
        roster["source_base_ids"]!=list(range(N)) or len(roster["cases"])!=N):
        raise ValueError("new source roster did not follow first preregistration")
    xs=[];ys=[];donor_keys=set()
    for i,source in enumerate(roster["cases"]):
        x=source["x"];y=source["y"]
        if (source["base_index"]!=i or len(x)!=81 or len(y)!=81 or
            any(type(v)!=int or not 1<=v<=9 for v in x) or
            y!=x[-1:]+x[:-1] or len(source["donors"])!=len(GOALS)):
            raise ValueError("first source legal cyclic sequence or labels changed")
        xs.append(x);ys.append(y)
        for j,goal in enumerate(GOALS):
            d=source["donors"][j]
            p=(goal-1)%81
            donor=x.copy()
            donor[p]=donor[p]%9+1
            changed=[t for t,(a,b) in enumerate(zip(y,donor[-1:]+donor[:-1])) if a!=b]
            if (d["goal"]!=goal or d["donor_x"]!=donor or
                d["predecessor_position"]!=p or d["old_digit"]!=x[p] or
                d["new_digit"]!=donor[p] or
                d["original_goal_label"]!=y[goal] or
                d["donor_goal_label"]!=donor[p] or
                d["changed_targets"]!=[goal] or changed!=[goal]):
                raise ValueError("legal donor predecessor change/source target invalid")
            donor_keys.add(tuple(donor))
    if len(set(map(tuple,xs)))!=N or len(donor_keys)!=N*len(GOALS):
        raise ValueError("nonunique original 31063 legal sources")
    if ordered_sha(xs)!=roster["x_sha256"] or ordered_sha(ys)!=roster["y_sha256"]:
        raise ValueError("first source x/y little-endian int64 hashes differ")


def verify_head(w,p):
    if len(w)!=10 or any(len(r)!=48 for r in w) or len(p)!=48 or any(len(r)!=48 for r in p):
        raise ValueError("invalid physical full 10x48 head and right-null projector")
    if abs(sum(p[i][i] for i in range(48))-38.)>1e-6:
        raise ValueError("current head kernel dimension changed")
    for i in range(48):
        for j in range(48):
            if abs(p[i][j]-p[j][i])>1e-8:
                raise ValueError("head-right-kernel P asymmetric")
            if abs(math.fsum(p[i][k]*p[k][j] for k in range(48))-p[i][j])>2e-8:
                raise ValueError("head-right-kernel P not idempotent")
    for row in w:
        for j in range(48):
            if abs(math.fsum(row[k]*p[k][j] for k in range(48)))>3e-7:
                raise ValueError("reported right kernel is not head-silent")


def audit_case(c,source,w,p,source_raw,edit_stream,edit_digest,edit_index):
    goal=c["goal"];slot=GOALS.index(goal)
    pair=((goal-1)%81,goal)
    if (c["base_index"]!=source["base_index"] or c["goal_slot"]!=slot or
        c["pair_positions"]!=list(pair) or
        c["donor_target"]!=source["donors"][slot]["donor_goal_label"] or
        c["original_target"]!=source["y"][goal] or
        set(c["arms"])!=set(ARMS) or
        len(c["clean_goal_logits"])!=10 or
        len(c["clean_all_predicted_classes"])!=81 or
        len(c["clean_donor_all_logits"])!=81):
        raise ValueError("registered source-goal identity or arm family changed")
    if len(source_raw)!=SOURCE_BYTES or c["source_index"]<0:
        raise ValueError("original full-ten-J source raw record invalid")
    chunks=[
        source_raw[:EDIT_BYTES],
        source_raw[EDIT_BYTES:2*EDIT_BYTES],
        source_raw[2*EDIT_BYTES:3*EDIT_BYTES],
        source_raw[3*EDIT_BYTES:3*EDIT_BYTES+3888*4],
        source_raw[3*EDIT_BYTES+3888*4:]
    ]
    keys=("pair_delta","energy_pair_gradient","signed_future_margin_gradient",
          "full_donor_delta","full_ten_logit_jacobian")
    if any(sha(raw)!=c["source_sha256"][key] for key,raw in zip(keys,chunks)):
        raise ValueError("first registered source physical delta/grad/J SHA changed")
    u=floats(chunks[0],96)
    g=floats(chunks[1],96)
    gm=floats(chunks[2],96)
    full=floats(chunks[3],3888)
    J=floats(chunks[4],960)
    for t,j in enumerate(pair):
        if max(abs(a-b) for a,b in zip(u[t*48:(t+1)*48],
                                       full[j*48:(j+1)*48]))>2e-5:
            raise ValueError("source full donor and two-token donor disagree")
    truth=c["donor_target"];old=c["original_target"]
    if max(abs(J[truth*96+i]-J[old*96+i]-gm[i]) for i in range(96))>5e-5:
        raise ValueError("source signed-margin gradient is not difference of full-ten J")
    uh=project_pair(u,p)
    gh=project_pair(g,p)
    active=l2(gh)>max(1e-12,1e-9*l2(g))
    uk=[a-(dot(uh,gh)/dot(gh,gh))*b for a,b in zip(uh,gh)] if active else uh
    radius=l2(u)
    close(c["pair_radius"],radius,"natural pair dose")
    if (c["current_head_rank"]!=10 or c["pair_fiber_dimension"]!=76-int(active)
        or c["energy_row_active"]!=active):
        raise ValueError("pair K rank/null diagnostic changed")
    close(c["projected_pair_fraction"],l2(uk)/max(radius,1e-30),
          "source donor pair projected fraction",atol=2e-6,rtol=2e-6)
    if c["admitted"] and not active:
        raise ValueError("admitted full-J K must have independent energy row")
    selected=norm_to(uk,radius) if c["admitted"] else [0.]*96
    original_margin=dot(selected,gm)
    close(c["selected_margin_derivative"],original_margin,
          "selected donor original signed future margin",atol=2e-4,rtol=4e-5)
    if c["admitted"]:
        rank=c["full_future_jacobian_rank"]
        if (not isinstance(rank,int) or not 1<=rank<=10 or
            c["future_dark_dimension"]!=75-rank or c["future_dark_dimension"]<65 or
            len(c["future_jacobian_singular_values"])!=10):
            raise ValueError("original full-ten-J rank/dark dimension invalid")
    else:
        if c["reason"] not in ("INELIGIBLE_FULL_J_MATCH",
                              "DEGENERATE_PAIR_TANGENT_OR_RADIUS",
                              "DEGENERATE_GRADIENT_MATCH"):
            raise ValueError("source failed admission without frozen reason")
    if not isinstance(c["admitted"],bool):
        raise ValueError("source admission must be boolean")
    rec_logits=c["clean_goal_logits"]
    if c["clean_all_predicted_classes"][goal]!=max(range(10),key=lambda k:rec_logits[k]):
        raise ValueError("clean goal prediction differs from original clean logits")
    clean_donor=c["clean_donor_all_logits"]
    if any(len(row)!=10 for row in clean_donor):
        raise ValueError("clean donor original full 81x10 logits absent")
    if (c["clean_recipient_correct"]!=(max(range(10),key=lambda k:rec_logits[k])==old) or
        c["clean_donor_correct"]!=(max(range(10),key=lambda k:clean_donor[goal][k])==truth) or
        c["joint_clean_competence"]!=(c["clean_recipient_correct"] and c["clean_donor_correct"])):
        raise ValueError("true independent legal donor/recipient clean competence invalid")
    rival=max((k for k in range(10) if k!=truth),key=lambda k:rec_logits[k])
    original_selected_output=None
    max_full_parity=0.
    for name in ARMS:
        d=c["arms"][name]
        raw=edit_stream.read(EDIT_BYTES)
        if len(raw)!=EDIT_BYTES or d["edit_index"]!=edit_index:
            raise ValueError("truncated or reordered first native physical pair vector")
        edit_index+=1;edit_digest.update(raw)
        if sha(raw)!=d["edit_sha256"]:
            raise ValueError("actual applied FP32 pair SHA differs")
        v=floats(raw,96)
        dose=l2(v)
        close(dose,d["actual_pair_l2"],"realized physical pair L2")
        if name=="full_donor_state":
            close(d["actual_full_l2"],l2(full),"full natural whole-state L2",atol=2e-4)
        else:
            close(d["actual_full_l2"],dose,"all pair arms have only two-token support")
        if c["admitted"] and (name=="pair_donor_k" or
            name.startswith("full_j_sham_") or name.startswith("margin_sham_") or
            name.startswith("random_k_sham_") or name=="pair_head_only"):
            close(dose,radius,"all K physical sham and candidate equal natural pair L2",
                  atol=2e-4,rtol=6e-5)
        if c["admitted"] and name=="pair_quarter":
            close(dose,radius*.25,"quarter natural pair physical dose",atol=2e-4)
        if not c["admitted"] and (name=="pair_donor_k" or
            name.startswith("full_j_sham_") or name.startswith("margin_sham_") or
            name.startswith("random_k_sham_") or name in ("pair_head_only","pair_quarter")):
            if dose>1e-7:
                raise ValueError("degenerate source had nonzero matched edit")
        expected=None
        if name=="pair_donor_k":expected=selected
        if name=="pair_head_only":expected=norm_to(uh,radius) if c["admitted"] else [0.]*96
        if name=="natural_pair" or name=="full_donor_state":expected=u
        if name=="goal_only_natural":expected=[0.]*48+list(u[48:])
        if name=="predecessor_only_natural":expected=list(u[:48])+[0.]*48
        if name=="pair_quarter":expected=[.25*z for z in selected]
        if expected is not None:
            if max(abs(a-b) for a,b in zip(v,expected))>3e-5+3e-5*max(1.,radius):
                raise ValueError("original physically applied donor/pair/source edit mismatch "+name)
        if name=="full_donor_state":
            derivative=head_full_l2(full,w)
        else:
            derivative=head_pair_l2(v,w)
        close(derivative,d["current_alltoken_head_l2"],
              "current all-token linear head delta",atol=8e-4,rtol=4e-4)
        if c["admitted"] and (name=="pair_donor_k" or
            name.startswith("full_j_sham_") or name.startswith("margin_sham_") or
            name.startswith("random_k_sham_") or name=="pair_head_only"):
            if derivative>4e-4*max(1.,radius):
                raise ValueError("original K/head-null edit has nonzero head visibility")
        if c["admitted"] and (name=="pair_donor_k" or name.startswith("full_j_sham_") or
            name.startswith("margin_sham_") or name.startswith("random_k_sham_")):
            if abs(dot(v,g))>5e-5*max(1.,l2(g)*radius):
                raise ValueError("first-order energy tangent violation")
        first=apply_ten_j(J,v)
        if len(d["ten_logit_linear_response"])!=10:
            raise ValueError("raw ten-logit first-order response not preserved")
        for a,b in zip(first,d["ten_logit_linear_response"]):
            close(a,b,"all-class future logit response",atol=3e-4,rtol=4e-5)
        close(dot(gm,v),d["signed_margin_linear_response"],
              "donor-vs-original signed derivative",atol=4e-4,rtol=4e-5)
        if name=="pair_donor_k":
            original_selected_output=first
        if c["admitted"] and name.startswith("full_j_sham_"):
            if any(abs(a-b)>2e-4+5e-5*max(abs(a),abs(b))
                   for a,b in zip(original_selected_output,first)):
                raise ValueError("one of ten first-order digit logits is unmatched")
        if c["admitted"] and name.startswith("margin_sham_"):
            if abs(dot(gm,v)-original_margin)>2e-4+5e-5*max(1.,abs(original_margin)):
                raise ValueError("signed donor-vs-original margin control mismatch")
        logits=d["goal_logits"]
        if len(logits)!=10 or len(d["all_predicted_classes"])!=81:
            raise ValueError("true native all-token predictions/ten digit logits missing")
        predicted=max(range(10),key=lambda k:logits[k])
        if (d["all_predicted_classes"][goal]!=predicted or
            d["donor_correct"]!=(predicted==truth) or d["clean_rival"]!=rival):
            raise ValueError("actual donor-target task scoring or clean rival changed")
        close(logsumexp(logits)-logits[truth],d["donor_goal_CE"],
              "true ten-class donor CE",atol=3e-6,rtol=1e-7)
        close(logits[truth]-logits[rival],d["donor_goal_margin_clean_rival"],
              "true donor-vs-clean-rival margin",atol=3e-6,rtol=1e-7)
        if not math.isfinite(float(d["finite_shared_E_change"])):
            raise ValueError("nonfinite finite E telemetry")
        if name=="full_donor_state":
            full_logits=d["all_final_logits"]
            if len(full_logits)!=81 or any(len(row)!=10 for row in full_logits):
                raise ValueError("natural full donor 81x10 positive control missing")
            max_full_parity=max(abs(float(a)-float(b))
                for row,clean in zip(full_logits,clean_donor)
                for a,b in zip(row,clean))
    close(max_full_parity,c["full_donor_max_logit_error"],
          "natural full donor whole-head native parity",atol=1e-6)
    return edit_index


def audit(teacher_bundle,teacher_meta,potential_bundle,potential_meta,
          lock_path,full_path,edit_path,source_path,out):
    model_files={
        "teacher_bundle_sha256":teacher_bundle,
        "teacher_meta_sha256":teacher_meta,
        "potential_bundle_sha256":potential_bundle,
        "potential_meta_sha256":potential_meta
    }
    if {k:sha_file(path) for k,path in model_files.items()}!=ORIGINAL:
        raise ValueError("original frozen pretrained model/potential source hash changed")
    lock_raw=Path(lock_path).read_bytes()
    locked=json.loads(lock_raw)
    expected_proto={p:sha_file(HERE/p) for p in PROTO}
    expected_code={
       **{p:sha_file(HERE/p) for p in CODES},
       **{p:sha_file(ROOT/p) for p in PARENTS}
    }
    if (locked["schema"]!=LOCK_SCHEMA or locked["original_sha256"]!=ORIGINAL or
        locked["protocol_sha256"]!=expected_proto or
        locked["implementation_sha256"]!=expected_code or
        locked["reserved_new_seed"]!=31063 or
        locked["teacher_seeds"]!=list(TEACHERS) or
        locked["n_independent_bases"]!=N or
        locked["goals"]!=list(GOALS) or
        locked["native_intervention_depth"]!=2 or
        locked["native_final_depth"]!=4 or
        locked["new_test_generated"] is not False or
        locked["new_test_labels_read"] is not False or
        locked["first_native_scored"] is not False or
        locked["canonical_original_tensor_loader_status"]!="PASS"):
        raise ValueError("source-only frozen model/code/contract chronology altered")
    data=json.loads(Path(full_path).read_bytes())
    if (data["schema"]!=SCHEMA or
        data["evidence_class"]!="EXPOSED_OLD_TEACHER_DEVELOPMENT_ONLY" or
        data["original_artifact_sha256"]!=ORIGINAL or
        data["source_lock_sha256"]!=sha(lock_raw) or
        data["protocol_sha256"]!=expected_proto or
        data["implementation_sha256"]!=expected_code or
        data["teacher_seeds"]!=list(TEACHERS) or
        data["n_base"]!=N or data["goals"]!=list(GOALS) or
        data["arms"]!=list(ARMS) or data["n_arms"]!=len(ARMS) or
        set(data["teachers"])!=set(map(str,TEACHERS))):
        raise ValueError("first original science schema/source/arms changed")
    verify_roster(data["source"])
    for field,path in (("physical_pair_edits",edit_path),
                       ("source_jacobians",source_path)):
        if data[field]["gzip_sha256"]!=sha_file(path):
            raise ValueError("first original compressed physical vector SHA drift")
    edit_digest=hashlib.sha256();source_digest=hashlib.sha256()
    edit_count=source_count=passed=0
    teacher_results={}
    with gzip.open(edit_path,"rb") as edits, gzip.open(source_path,"rb") as sources:
        for sid in TEACHERS:
            root=data["teachers"][str(sid)]
            w=root["head_weight"];p=root["head_projector"]
            verify_head(w,p)
            cases=root["cases"]
            if len(cases)!=N*len(GOALS):
                raise ValueError("original 512 legal donor goals missing")
            groups={k:{i:[] for i in range(N)} for k in ("full_j","margin","random")}
            correct={arm:0 for arm in ARMS}
            n_admitted=n_joint=n_full=n_pair=0
            max_parity=0.
            for i in range(N):
                source=data["source"]["cases"][i]
                for j,goal in enumerate(GOALS):
                    c=cases[i*len(GOALS)+j]
                    if c["source_index"]!=source_count or c["goal_slot"]!=j:
                        raise ValueError("first original source-goal vector order changed")
                    raw=sources.read(SOURCE_BYTES)
                    if len(raw)!=SOURCE_BYTES:
                        raise ValueError("original full-J source vectors truncated")
                    source_digest.update(raw);source_count+=1
                    edit_count=audit_case(c,source,w,p,raw,edits,edit_digest,edit_count)
                    selected=float(c["arms"]["pair_donor_k"]["donor_correct"])
                    for key,prefix in (("full_j","full_j_sham_"),
                                       ("margin","margin_sham_"),
                                       ("random","random_k_sham_")):
                        matched=median8([c["arms"][prefix+str(k)]["donor_correct"]
                                         for k in range(SHAMS)])
                        groups[key][i].append(selected-matched)
                    competent=int(c["joint_clean_competence"])
                    n_joint+=competent;n_admitted+=int(c["admitted"])
                    n_full+=competent*int(c["arms"]["full_donor_state"]["donor_correct"])
                    n_pair+=competent*int(c["arms"]["natural_pair"]["donor_correct"])
                    max_parity=max(max_parity,float(c["full_donor_max_logit_error"]))
                    for name in ARMS:
                        correct[name]+=int(c["arms"][name]["donor_correct"])
            result=root["summary"]
            if (result["n_original_source_goals"]!=N*len(GOALS) or
                result["n_admitted"]!=n_admitted or
                result["n_joint_clean_competent"]!=n_joint or
                result["arm_correct_counts"]!=correct):
                raise ValueError("registered teacher source/correct-target totals altered")
            comparisons={}
            for kind in ("full_j","margin","random"):
                if any(len(groups[kind][i])!=len(GOALS) for i in range(N)):
                    raise ValueError("missing paired donor goals within independent source")
                effects=[sum(groups[kind][i])/len(GOALS) for i in range(N)]
                mean=sum(effects)/N;ci=bootstrap(effects,sid,kind)
                summary=result["contrasts"][kind]
                close(mean,summary["paired_ITT_effect"],
                      "first source independent teacher-mean task effect",atol=1e-12)
                for a,b in zip(ci,summary["whole_base_bootstrap95"]):
                    close(a,b,"fixed 2000 base-cluster percentile bootstrap",atol=1e-12)
                for a,b in zip(effects,summary["base_level_effects"]):
                    close(a,b,"original per-source repeated-goal ITT effect",atol=1e-12)
                if summary["n_independent_bases"]!=N or summary["bootstrap_draws"]!=BOOTS:
                    raise ValueError("old registered source unit/bootstrap size changed")
                comparisons[kind]={"mean":mean,"bootstrap95":ci}
            competence=n_joint/(N*len(GOALS))
            full=n_full/max(1,n_joint);pair=n_pair/max(1,n_joint)
            for name,val in (("joint_competence_rate",competence),
                             ("full_donor_correct_among_competent",full),
                             ("natural_pair_correct_among_competent",pair),
                             ("max_original_full_donor_head_parity_error",max_parity)):
                close(val,result[name],"source-level clean model positive control "+name,atol=1e-12)
            gate=bool(
                n_admitted/(N*len(GOALS))>=.95 and competence>=.8 and
                full>=.95 and pair>=.90 and max_parity<=1e-4 and
                comparisons["full_j"]["mean"]>=.10 and comparisons["full_j"]["bootstrap95"][0]>0 and
                comparisons["margin"]["mean"]>=.10 and comparisons["margin"]["bootstrap95"][0]>0 and
                comparisons["random"]["mean"]>=.15 and comparisons["random"]["bootstrap95"][0]>0)
            if result["teacher_strong_full_J_gate"]!=gate:
                raise ValueError("registered per-teacher full-J strict gate changed")
            passed+=int(gate)
            teacher_results[str(sid)]={
                "n_admitted":n_admitted,
                "n_joint_competent":n_joint,
                "full_donor_rate":full,"natural_pair_rate":pair,
                "max_full_donor_parity":max_parity,
                "correct_counts":correct,"contrasts":comparisons,
                "teacher_full_J_gate":gate
            }
        if edits.read(1)!=b"" or sources.read(1)!=b"":
            raise ValueError("unexpected trailing raw physical source/edited vectors")
    if (edit_count!=3*N*len(GOALS)*len(ARMS) or
        source_count!=3*N*len(GOALS) or
        edit_count!=data["physical_pair_edits"]["n_records"] or
        source_count!=data["source_jacobians"]["n_records"] or
        edit_digest.hexdigest()!=data["physical_pair_edits"]["raw_sha256"] or
        source_digest.hexdigest()!=data["source_jacobians"]["raw_sha256"]):
        raise ValueError("first registered full physical pair/J vector count or SHA mismatch")
    verdict=("DEVELOPMENT_STRONG_FULL_J_PASS" if passed>=2
             else "DEVELOPMENT_STRONG_FULL_J_FAIL")
    if (data["n_old_teachers_strong_full_J"]!=passed or
        data["first_registered_development_verdict"]!=verdict):
        raise ValueError("scientific first original preregistered development verdict changed")
    result={
      "schema":"LGA09/first-native-independent-stdlib-audit-v1",
      "audit_status":"PASS","evidence_class":"DEVELOPMENT_ONLY",
      "original_full_receipt_sha256":sha_file(full_path),
      "original_source_lock_sha256":sha(lock_raw),
      "n_original_edit_vectors":edit_count,"n_original_source_jacobians":source_count,
      "edit_raw_sha256":edit_digest.hexdigest(),"source_raw_sha256":source_digest.hexdigest(),
      "n_old_teachers_strict_pass":passed,
      "registered_development_verdict":verdict,
      "teachers":teacher_results,"torch_imported":False
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("LGA09_FIRST_INDEPENDENT_STDLIB_AUDIT_PASS",verdict,flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    for key in ("teacher-bundle","teacher-meta","potential-bundle","potential-meta",
                "source-lock","full","edits","sources","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    audit(a.teacher_bundle,a.teacher_meta,a.potential_bundle,a.potential_meta,
          a.source_lock,a.full,a.edits,a.sources,a.out)


if __name__=="__main__":
    main()

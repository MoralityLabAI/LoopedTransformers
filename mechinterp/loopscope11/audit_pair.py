"""LGA-08 independent stdlib audit of FIRST frozen 31062 pair-fiber receipt.

No Torch/NumPy imports, no new model execution, no task sources regenerated.
Every 96-f32 edit is checked against archived first-source pair geometry and
future signed-gradient matching, with source-cluster bootstrap and verdict replay.
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
SCHEMA="LGA08/first-native-pair-gradient-matched-development-v1"
LOCK_SCHEMA="LGA08/first-source-pretest-frozen-lock-v1"
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
ARMS=("pair_energy_tangent",*(f"pair_random_k_{i}" for i in range(SHAMS)),
      *(f"pair_gradient_match_{i}" for i in range(SHAMS)),
      "pair_head_only","pair_natural","goal_natural",
      "predecessor_natural","pair_tangent_quarter","full_donor_state")
PAIR=96
FULL=3888
EDIT_BYTES=PAIR*4
SOURCE_BYTES=(PAIR*3+FULL)*4
BOOT=2000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(p):
    return sha(Path(p).read_bytes())


def dot(a,b):
    return math.fsum(float(x)*float(y) for x,y in zip(a,b))


def l2(a):
    return math.sqrt(dot(a,a))


def close(a,b,what,*,atol=3e-5,rtol=2e-5):
    a=float(a);b=float(b)
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a-b)>atol+rtol*max(abs(a),abs(b)):
        raise ValueError("first native pair arithmetic mismatch "+what+f": {a} vs {b}")


def vec(raw,n):
    if len(raw)!=4*n:
        raise ValueError("physical source/edited pair raw byte count incorrect")
    z=struct.unpack("<"+str(n)+"f",raw)
    if any(not math.isfinite(float(q)) for q in z):
        raise ValueError("nonfinite physical vector")
    return z


def project(x,p):
    return [math.fsum(x[token*48+k]*p[k][j] for k in range(48))
            for token in range(2) for j in range(48)]


def normalized(x,r):
    size=l2(x)
    return [float(z)*r/size for z in x] if size>1e-10 and r>1e-10 else [0.]*len(x)


def margin_gradient_seed(*parts):
    message=("LGA08|"+"|".join(map(str,parts))).encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8],"big") & ((1<<63)-1)


def boot(values,sid,kind):
    if len(values)!=N or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("wrong 64 independent source bootstrap values")
    rng=random.Random(margin_gradient_seed("bootstrap",sid,kind))
    sims=sorted(sum(values[rng.randrange(N)] for _ in range(N))/N for _ in range(BOOT))
    return [sims[int(.025*BOOT)],sims[int(.975*BOOT)-1]]


def median_binary(values):
    values=sorted(float(v) for v in values)
    if len(values)!=8 or any(v not in (0.,1.) for v in values):
        raise ValueError("first-source strict gradient matched sham binary family changed")
    return (values[3]+values[4])/2.


def logsumexp(x):
    m=max(float(v) for v in x)
    return m+math.log(math.fsum(math.exp(float(v)-m) for v in x))


def int64_sha(rows):
    nums=[int(z) for row in rows for z in row]
    return sha(struct.pack("<"+"q"*len(nums),*nums))


def verify_source(roster):
    if (roster["seed"]!=31062 or
        roster["n_independent_bases"]!=N or
        roster["ordered_base_indices"]!=list(range(N)) or
        len(roster["sources"])!=N):
        raise ValueError("original source roster changed")
    xs,ys,donors=[],[],set()
    for i,case in enumerate(roster["sources"]):
        x=case["x"];y=case["y"]
        if (case["base_index"]!=i or len(x)!=81 or len(y)!=81 or
            any(type(v)!=int or not 1<=v<=9 for v in x) or
            y!=x[-1:]+x[:-1] or len(case["donors"])!=len(GOALS)):
            raise ValueError("new legal IID cyclic source invalid")
        xs.append(x);ys.append(y)
        for j,goal in enumerate(GOALS):
            d=case["donors"][j]
            original=x.copy();p=(goal-1)%81
            original[p]=x[p]%9+1
            if (d["goal"]!=goal or d["donor_x"]!=original or
                d["predecessor_position"]!=p or
                d["old_digit"]!=x[p] or d["new_digit"]!=original[p] or
                d["donor_goal_label"]!=original[p] or
                d["original_goal_label"]!=y[goal] or
                d["changed_targets"]!=[goal]):
                raise ValueError("legal donor predecessor/source target altered")
            donors.add(tuple(original))
    if len(set(map(tuple,xs)))!=N or len(donors)!=N*len(GOALS):
        raise ValueError("source/donor duplicate original inputs")
    if int64_sha(xs)!=roster["x_sha256"] or int64_sha(ys)!=roster["y_sha256"]:
        raise ValueError("original source SHA256 differs")


def verify_physical_head(head,p):
    if len(head)!=10 or any(len(z)!=48 for z in head) or len(p)!=48 or any(len(z)!=48 for z in p):
        raise ValueError("native 10x48 head or 48x48 orthogonal projector missing")
    if abs(sum(p[i][i] for i in range(48))-38)>1e-6:
        raise ValueError("native current-head right kernel dimension changed")
    for i in range(48):
        for j in range(48):
            if abs(float(p[i][j])-float(p[j][i]))>1e-8:
                raise ValueError("current-head projector asymmetry")
            if abs(math.fsum(p[i][k]*p[k][j] for k in range(48))-p[i][j])>1e-8:
                raise ValueError("physical current-head projector not idempotent")
    for row in head:
        for j in range(48):
            if abs(math.fsum(float(row[k])*float(p[k][j]) for k in range(48)))>3e-7:
                raise ValueError("original current-head projector allows visible edits")


def head_l2(edit,head):
    total=0.
    for token in range(2):
        x=edit[token*48:(token+1)*48]
        for row in head:
            v=dot(row,x)
            total+=v*v
    return math.sqrt(total)


def full_head_l2(edit,head):
    total=0.
    for token in range(81):
        x=edit[token*48:(token+1)*48]
        for row in head:
            v=dot(row,x)
            total+=v*v
    return math.sqrt(total)


def audit_case(c,source_case,head,p,raw_source,edit_stream,edit_digest,idx):
    if (c["base_index"]!=source_case["base_index"] or
        c["goal_slot"]!=GOALS.index(c["goal"]) or
        c["pair_positions"]!=[(c["goal"]-1)%81,c["goal"]] or
        set(c["arms"])!=set(ARMS) or
        c["donor_target"]!=source_case["donors"][c["goal_slot"]]["donor_goal_label"] or
        c["original_target"]!=source_case["y"][c["goal"]]):
        raise ValueError("native case legal target/source/arm association failed")
    if len(raw_source)!=SOURCE_BYTES:
        raise ValueError("original source pair+full vector source bytes truncated")
    parts=(
        raw_source[:EDIT_BYTES],
        raw_source[EDIT_BYTES:2*EDIT_BYTES],
        raw_source[2*EDIT_BYTES:3*EDIT_BYTES],
        raw_source[3*EDIT_BYTES:],
    )
    hashes=("pair_delta_sha256","energy_pair_gradient_sha256",
            "future_margin_pair_gradient_sha256","full_donor_state_delta_sha256")
    if any(sha(raw)!=c[k] for k,raw in zip(hashes,parts)):
        raise ValueError("pre-edit source donor/energy/signed-margin gradients SHA invalid")
    u=vec(parts[0],PAIR)
    egrad=vec(parts[1],PAIR)
    future=vec(parts[2],PAIR)
    full=vec(parts[3],FULL)
    old=(c["goal"]-1)%81;goal=c["goal"]
    for token,index in enumerate((old,goal)):
        src=full[index*48:(index+1)*48]
        a=u[token*48:(token+1)*48]
        if max(abs(x-y) for x,y in zip(src,a))>2e-5:
            raise ValueError("naturally reached full and two-token donor source disagree")
    r=l2(u)
    close(r,c["natural_pair_radius"],"original natural pair radius")
    head_u=project(u,p)
    head_g=project(egrad,p)
    active=l2(head_g)>max(1e-12,1e-9*l2(egrad))
    if active:
        a=dot(head_u,head_g)/(dot(head_g,head_g))
        selected=[x-a*y for x,y in zip(head_u,head_g)]
    else:
        selected=head_u
    dim=76-int(active)
    if (c["head_rank"]!=10 or c["fiber_dimension"]!=dim or
        c["energy_row_active"]!=active):
        raise ValueError("registered pair tangent numerical dimension changed")
    close(l2(selected)/max(r,1e-30),c["projected_pair_fraction"],
          "actual donor pair tangent mass",atol=3e-6,rtol=2e-6)
    close(l2(head_u)/max(r,1e-30),c["head_only_pair_fraction"],
          "actual donor pair head-null mass",atol=3e-6,rtol=2e-6)
    close(dot(selected,egrad),c["tangent_energy_residual"],
          "native pair original E tangent",atol=1e-4,rtol=2e-5)
    eligible=bool(r>1e-10 and l2(selected)>1e-10 and l2(head_u)>1e-10 and active and dim==75)
    gf=project(future,p)
    if active:
        b=dot(gf,head_g)/(dot(head_g,head_g))
        gf=[x-b*y for x,y in zip(gf,head_g)]
    # Ineligible status is allowed if signed oracle gradient is degenerate.
    gradient_matched=bool(l2(gf)>1e-10)
    if c["admitted"] and (not eligible or not gradient_matched):
        raise ValueError("native admitted case lacks legal 75D K or future margin sensitivity")
    if not c["admitted"] and c["exclusion_reason"] not in (
        "DEGENERATE_GRADIENT_MATCH","DEGENERATE_PAIR_TANGENT_OR_RADIUS"):
        raise ValueError("unregistered geometric case exclusion")
    expected_selected=normalized(selected,r) if c["admitted"] else [0.]*PAIR
    expected_head=normalized(head_u,r) if c["admitted"] else [0.]*PAIR
    expected={
        "pair_energy_tangent":expected_selected,
        "pair_head_only":expected_head,
        "pair_natural":u,
        "goal_natural":[0.]*48+list(u[48:]),
        "predecessor_natural":list(u[:48])+[0.]*48,
        "pair_tangent_quarter":[x*.25 for x in expected_selected],
        "full_donor_state":u,
    }
    signed_selected=dot(expected_selected,gf)
    close(signed_selected,c["selected_signed_derivative"],
          "source-selected signed first-order future margin",atol=2e-4,rtol=2e-5)
    if c["clean_future_margin"] is not None and not math.isfinite(float(c["clean_future_margin"])):
        raise ValueError("nonfinite model clean margin")
    # Independently check all original 10-class source goal labels before edits.
    truth=int(c["donor_target"])
    clean=c["clean_goal_logits"]
    if len(clean)!=10 or len(c["clean_predicted_classes"])!=81 or len(c["clean_donor_all_logits"])!=81:
        raise ValueError("complete clean model head source missing")
    clean_pred=max(range(10),key=lambda x:clean[x])
    donor_pred=max(range(10),key=lambda x:c["clean_donor_all_logits"][goal][x])
    if (c["clean_predicted_classes"][goal]!=clean_pred or
        c["clean_recipient_correct"]!=(clean_pred==c["original_target"]) or
        c["clean_donor_correct"]!=(donor_pred==truth) or
        c["joint_clean_competence"]!=(c["clean_recipient_correct"] and c["clean_donor_correct"])):
        raise ValueError("native clean competence labels/logits inconsistent")
    rival=max((i for i in range(10) if i!=truth),key=lambda i:clean[i])
    fullparity=0.
    for name in ARMS:
        a=c["arms"][name]
        raw=edit_stream.read(EDIT_BYTES)
        if len(raw)!=EDIT_BYTES or a["edit_index"]!=idx:
            raise ValueError("first native pair-edited vector archive order changed")
        idx+=1;edit_digest.update(raw)
        if sha(raw)!=a["pair_edit_sha256"]:
            raise ValueError("physical FP32 actual pair edit SHA mismatch")
        v=vec(raw,PAIR)
        dose=l2(v)
        close(dose,a["actual_pair_l2"],"physical applied pair edit L2")
        if name in expected:
            error=max(abs(x-y) for x,y in zip(expected[name],v))
            if error>3e-5+3e-5*max(1.,r):
                raise ValueError("original actual pair edit differs from its source-only planned direction: "+name)
        if name=="full_donor_state":
            close(l2(full),a["actual_full_l2"],
                  "natural full-state larger positive-control dose",atol=2e-4,rtol=3e-5)
        else:
            close(dose,a["actual_full_l2"],
                  "pair-only edits have zero support outside two source tokens")
        if c["admitted"] and (name=="pair_energy_tangent" or
            name=="pair_head_only" or name.startswith("pair_random_k_") or
            name.startswith("pair_gradient_match_")):
            close(dose,r,"registered natural pair physical L2 dose",atol=2e-4,rtol=3e-5)
        if name=="pair_tangent_quarter" and c["admitted"]:
            close(dose,.25*r,"registered quarter natural pair dose",atol=2e-4,rtol=3e-5)
        if not c["admitted"] and (name=="pair_energy_tangent" or
            name=="pair_head_only" or name=="pair_tangent_quarter" or
            name.startswith("pair_random_k_") or name.startswith("pair_gradient_match_")):
            if dose>2e-7:
                raise ValueError("ineligible ITT source was replaced, not zero-edited")
        if name=="full_donor_state":
            current_head=full_head_l2(full,head)
        else:
            current_head=head_l2(v,head)
        close(current_head,a["current_head_alltoken_l2"],
              "applied native physical current-head delta",atol=8e-4,rtol=3e-4)
        if c["admitted"] and (name=="pair_energy_tangent" or
            name=="pair_head_only" or name.startswith("pair_random_k_") or
            name.startswith("pair_gradient_match_")):
            if current_head>4e-4*max(1.,r):
                raise ValueError("current head tangent physical edit is visible")
        if c["admitted"] and (name=="pair_energy_tangent" or
            name.startswith("pair_random_k_") or name.startswith("pair_gradient_match_")):
            if abs(dot(v,egrad))>4e-5*max(1.,l2(egrad)*r):
                raise ValueError("original energy tangent physical edit changed derivative")
        actual_signed=dot(v,future)
        close(actual_signed,a["signed_future_margin_derivative"],
              "applied exact future model derivative",atol=4e-4,rtol=4e-5)
        if c["admitted"] and name.startswith("pair_gradient_match_"):
            if abs(actual_signed-signed_selected)>5e-5*max(1.,abs(signed_selected),l2(future)*r):
                raise ValueError("strict sham does not match signed donor-target future derivative")
        if name=="pair_energy_tangent" and c["admitted"]:
            if abs(actual_signed-signed_selected)>5e-5*max(1.,abs(signed_selected),l2(future)*r):
                raise ValueError("native selected pair changed first-order signed derivative")
        if not math.isfinite(float(a["finite_shared_E_change"])):
            raise ValueError("nonfinite measured finite external E")
        logits=a["goal_logits"]
        if len(logits)!=10 or len(a["predicted_classes"])!=81:
            raise ValueError("native full head 81 predicted classes or 10 digit logits missing")
        pred=max(range(10),key=lambda i:logits[i])
        if (pred!=a["predicted_classes"][goal] or
            a["donor_correct"]!=(pred==truth) or
            a["clean_rival"]!=rival):
            raise ValueError("native corrected donor-target label/clean rival invalid")
        close(logsumexp(logits)-logits[truth],a["donor_goal_CE"],
              "original legal donor goal CE",atol=3e-6,rtol=1e-7)
        close(logits[truth]-logits[rival],a["donor_margin_clean_rival"],
              "original clean rival margin",atol=3e-6,rtol=1e-7)
        if name=="full_donor_state":
            full_logits=a["all_final_logits"]
            if len(full_logits)!=81 or any(len(z)!=10 for z in full_logits):
                raise ValueError("full natural donor oracle must archive 81x10 logits")
            fullparity=max(abs(float(z)-float(t)) for predrow,cleanrow in
                           zip(full_logits,c["clean_donor_all_logits"])
                           for z,t in zip(predrow,cleanrow))
    close(fullparity,c["full_donor_max_logit_error"],
          "independent natural whole donor/source native function parity",atol=1e-6,rtol=1e-6)
    return idx


def audit(teacher_bundle,teacher_meta,potential_bundle,potential_meta,
          lock,full_path,edit_path,source_path,out):
    original={
        "teacher_bundle_sha256":teacher_bundle,
        "teacher_meta_sha256":teacher_meta,
        "potential_bundle_sha256":potential_bundle,
        "potential_meta_sha256":potential_meta,
    }
    if {k:file_sha(v) for k,v in original.items()}!=ORIGINAL:
        raise ValueError("original frozen teacher/potential artifact bytes changed")
    lock_raw=Path(lock).read_bytes()
    receipt=json.loads(lock_raw)
    # Independently rebuild original prospective code/protocol fingerprints.
    # Importing source_lock would import Torch and invalidate audit isolation.
    root=HERE.parents[1]
    expected_proto={
        p:file_sha(HERE/p) for p in ("PROTOCOL.md","POWER_PREFLIGHT.md")
    }
    code_names=("source_lock.py","pair_fiber.py","native_pair.py",
                "audit_pair.py","tests/test_pair_fiber.py",
                "tests/test_native_pair.py")
    parent_names=(
        "mechinterp/loopscope10/donor_source.py",
        "mechinterp/loopscope10/global_energy_fiber.py",
        "mechinterp/loopscope10/energy_fiber.py",
        "mechinterp/loopscope04/gradient_flow.py",
        "mechinterp/loopscope05/fit_and_freeze.py",
        "mechinterp/loopscope08/fit_potential.py",
        ".github/workflows/loopscope11-pair-fiber.yml",
        ".github/workflows/loopscope11-first-native.yml"
    )
    expected_code={
        **{p:file_sha(HERE/p) for p in code_names},
        **{p:file_sha(root/p) for p in parent_names}
    }
    if (receipt["protocol_sha256"]!=expected_proto or
        receipt["implementation_sha256"]!=expected_code):
        raise ValueError("original source-only protocol/code SHA differs from independent audit")
    if (receipt["schema"]!=LOCK_SCHEMA or receipt["original_sha256"]!=ORIGINAL or
        receipt["reserved_seed"]!=31062 or receipt["new_test_generated"] is not False or
        receipt["new_test_labels_read"] is not False or
        receipt["prospective_first_test_scored"] is not False or
        receipt["teacher_seeds"]!=list(TEACHERS) or receipt["n_independent_bases"]!=N or
        receipt["canonical_original_tensor_loaders"]!="PASS"):
        raise ValueError("original pretest first-source lock invalid")
    data=json.loads(Path(full_path).read_bytes())
    if (data["schema"]!=SCHEMA or
        data["evidence_class"]!="EXPOSED_OLD_TEACHERS_DEVELOPMENT_ONLY" or
        data["original_artifact_sha256"]!=ORIGINAL or
        data["first_pretest_lock_sha256"]!=sha(lock_raw) or
        data["protocol_sha256"]!=receipt["protocol_sha256"] or
        data["implementation_sha256"]!=receipt["implementation_sha256"] or
        data["teacher_seeds"]!=list(TEACHERS) or
        data["n_independent_bases"]!=N or data["goals"]!=list(GOALS) or
        data["n_arms"]!=len(ARMS) or data["arms"]!=list(ARMS)):
        raise ValueError("first test source/protocol/code/arms receipt changed")
    verify_source(data["source"])
    for key,file in (("physical_pair_edits",edit_path),("source_vectors",source_path)):
        if file_sha(file)!=data[key]["gzip_sha256"]:
            raise ValueError("first native complete raw vector artifact SHA changed")
    edit_digest=hashlib.sha256();source_digest=hashlib.sha256()
    edit_index=source_index=pass_count=0
    root_results={}
    with gzip.open(edit_path,"rb") as edit_file,gzip.open(source_path,"rb") as source_file:
        for sid in TEACHERS:
            model=data["teachers"][str(sid)]
            head=model["head_weight"];projector=model["head_projector"]
            verify_physical_head(head,projector)
            cases=model["cases"]
            if len(cases)!=N*len(GOALS):
                raise ValueError("teacher source repeated goal count changed")
            sums={"random":{i:[] for i in range(N)},
                  "gradient":{i:[] for i in range(N)}}
            counts={arm:0 for arm in ARMS}
            joint=full_success=pair_success=admitted=0
            maxparity=0.
            for base in range(N):
                source=data["source"]["sources"][base]
                for j,goal in enumerate(GOALS):
                    c=cases[base*len(GOALS)+j]
                    if c["source_index"]!=source_index or c["goal_slot"]!=j:
                        raise ValueError("source-to-goal first vector order changed")
                    raw=source_file.read(SOURCE_BYTES)
                    if len(raw)!=SOURCE_BYTES:
                        raise ValueError("source physical vectors truncated")
                    source_digest.update(raw);source_index+=1
                    edit_index=audit_case(c,source,head,projector,raw,edit_file,
                                          edit_digest,edit_index)
                    sel=float(c["arms"]["pair_energy_tangent"]["donor_correct"])
                    medgrad=median_binary([c["arms"][f"pair_gradient_match_{i}"]["donor_correct"]
                                            for i in range(SHAMS)])
                    medrand=median_binary([c["arms"][f"pair_random_k_{i}"]["donor_correct"]
                                            for i in range(SHAMS)])
                    sums["gradient"][base].append(sel-medgrad)
                    sums["random"][base].append(sel-medrand)
                    admitted+=int(c["admitted"])
                    iscomp=int(c["joint_clean_competence"])
                    joint+=iscomp
                    full_success+=iscomp*int(c["arms"]["full_donor_state"]["donor_correct"])
                    pair_success+=iscomp*int(c["arms"]["pair_natural"]["donor_correct"])
                    maxparity=max(maxparity,float(c["full_donor_max_logit_error"]))
                    for arm in ARMS:
                        counts[arm]+=int(c["arms"][arm]["donor_correct"])
            root_summary=model["summary"]
            if (root_summary["n_admitted"]!=admitted or
                root_summary["n_jointly_competent"]!=joint or
                root_summary["arm_correct_counts"]!=counts or
                root_summary["n_original_source_goals"]!=N*len(GOALS)):
                raise ValueError("registered source admission/goal/arm counts changed")
            compare={}
            for kind in ("gradient","random"):
                base_means=[sum(sums[kind][i])/len(GOALS) for i in range(N)]
                effect=sum(base_means)/N
                ci=boot(base_means,sid,kind)
                s=root_summary["contrasts"][kind]
                close(effect,s["paired_ITT_effect"],"registered source mean contrast",atol=1e-12)
                if s["n_independent_base_sources"]!=N or s["bootstrap_draws"]!=BOOT:
                    raise ValueError("registered source-bootstrap units changed")
                for a,b in zip(s["whole_base_bootstrap95"],ci):
                    close(a,b,"original 2000-draw source-bootstrap interval",atol=1e-12)
                for a,b in zip(s["base_level_paired_effects"],base_means):
                    close(a,b,"original source-level goal grouped effect",atol=1e-12)
                compare[kind]={"paired_ITT_effect":effect,"bootstrap95":ci}
            full_rate=full_success/max(1,joint)
            pair_rate=pair_success/max(1,joint)
            close(full_rate,root_summary["full_donor_correct_among_competent"],
                  "natural full source donor accuracy",atol=1e-12)
            close(pair_rate,root_summary["natural_pair_correct_among_competent"],
                  "natural pair source donor accuracy",atol=1e-12)
            close(joint/(N*len(GOALS)),root_summary["joint_competence_fraction"],
                  "original clean recipient/donor competence",atol=1e-12)
            close(maxparity,root_summary["max_full_donor_parity_error"],
                  "registered native four-pass donor parity",atol=1e-8)
            passed=bool(joint/(N*len(GOALS))>=.8 and maxparity<=1e-4 and
                        full_rate>=.95 and pair_rate>=.90 and
                        compare["gradient"]["paired_ITT_effect"]>=.10 and
                        compare["gradient"]["bootstrap95"][0]>0 and
                        compare["random"]["paired_ITT_effect"]>=.15 and
                        compare["random"]["bootstrap95"][0]>0)
            if passed!=root_summary["registered_strong_teacher_gate"]:
                raise ValueError("preregistered strict incremental teacher gate changed")
            pass_count+=int(passed)
            root_results[str(sid)]={
                "strict_gradient_matched":compare["gradient"],
                "simple_random_K":compare["random"],
                "full_donor_control_correct":full_rate,
                "natural_pair_control_correct":pair_rate,
                "n_admitted":admitted,"n_competent":joint,
                "registered_strong_teacher_gate":passed,
            }
        if edit_file.read(1)!=b"" or source_file.read(1)!=b"":
            raise ValueError("unexpected surplus hidden physical raw vector data")
    if (edit_index!=data["physical_pair_edits"]["n_records"] or
        source_index!=data["source_vectors"]["n_records"] or
        edit_index!=len(TEACHERS)*N*len(GOALS)*len(ARMS) or
        source_index!=len(TEACHERS)*N*len(GOALS) or
        edit_digest.hexdigest()!=data["physical_pair_edits"]["raw_sha256"] or
        source_digest.hexdigest()!=data["source_vectors"]["raw_sha256"]):
        raise ValueError("all 96D physical pair edit and source vector hashes/counts differ")
    verdict=("DEVELOPMENT_STRONG_INCREMENTAL_PASS" if pass_count>=2 else
             "DEVELOPMENT_STRONG_INCREMENTAL_FAIL")
    if (data["n_teachers_passing_strong_incremental"]!=pass_count or
        data["first_registered_development_verdict"]!=verdict):
        raise ValueError("original registered scientific development verdict changed")
    result={
        "schema":"LGA08/first-native-independent-stdlib-audit-v1",
        "audit_status":"PASS","evidence_class":"EXPOSED_TEACHER_DEVELOPMENT_ONLY",
        "original_full_receipt_sha256":file_sha(full_path),
        "original_source_lock_sha256":sha(lock_raw),
        "source_vectors":source_index,"physical_pair_edits":edit_index,
        "source_vectors_sha256":source_digest.hexdigest(),
        "physical_pair_edits_sha256":edit_digest.hexdigest(),
        "n_passing_old_teachers":pass_count,
        "registered_first_development_verdict":verdict,
        "teachers":root_results,
        "torch_imported":False,
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("LGA08_FIRST_INDEPENDENT_STDLIB_AUDIT_PASS",verdict,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    for arg in ("teacher-bundle","teacher-meta","potential-bundle","potential-meta",
                "source-lock","full","edits","sources","out"):
        parser.add_argument("--"+arg,required=True,type=Path)
    x=parser.parse_args()
    audit(x.teacher_bundle,x.teacher_meta,x.potential_bundle,x.potential_meta,
          x.source_lock,x.full,x.edits,x.sources,x.out)


if __name__=="__main__":
    main()

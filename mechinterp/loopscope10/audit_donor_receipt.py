"""Independent Python-STDLIB audit of first frozen LGA-07B native receipt.

No Torch, NumPy, model import or new source inference. All arithmetic checks
use the originally archived inputs, edit vectors and genuine final logits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

HERE=Path(__file__).resolve().parent
SCHEMA="LGA07B/first-native-reachable-donor-development-v1"
LOCK_SCHEMA="LGA07B/first-native-source-lock-v1"
SOURCE_COUNT=64
GOALS=(0,10,20,30,40,50,60,70)
TEACHERS=(103,211,307)
DEPTHS=(2,6)
ARMS=("projected_donor","inverse_projected_donor","unprojected_donor",
      "full_donor_token",*(f"same_k_sham_{i}" for i in range(8)))
SHAS={
    "teacher_bundle_sha256":"d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
    "teacher_meta_sha256":"7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d",
    "potential_bundle_sha256":"24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53",
    "potential_meta_sha256":"69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d",
}
PROTOCOLS=("DONOR_PROTOCOL.md","LGA07B_FIRST_NATIVE_RELEASE.md","POWER_PREFLIGHT.md")
CODE=("source_lock_donor.py","native_donor_eval.py",
      "audit_donor_receipt.py","donor_source.py","energy_fiber.py")
BOOT=2000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(p):
    return sha(Path(p).read_bytes())


def stable_seed(*parts):
    word=("LGA07B|"+"|".join(map(str,parts))).encode()
    return int.from_bytes(hashlib.sha256(word).digest()[:8],"big")&((1<<63)-1)


def sha_int64(rows):
    v=[int(d) for row in rows for d in row]
    return sha(struct.pack("<"+"q"*len(v),*v))


def sha_float32(v):
    return sha(struct.pack("<"+"f"*len(v),*[float(x) for x in v]))


def close(a,b,name,*,absolute=3e-5,relative=5e-6):
    a=float(a);b=float(b)
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a-b)>absolute+relative*max(abs(a),abs(b)):
        raise ValueError("independent numerical replay mismatch: "+name)


def logsumexp(v):
    high=max(v)
    return high+math.log(sum(math.exp(z-high) for z in v))


def vec_norm(v):
    return math.sqrt(sum(z*z for z in v))


def median_binary(v):
    a=sorted(float(x) for x in v)
    if len(a)!=8 or any(x not in (0.,1.) for x in a):
        raise ValueError("not eight binary same-K shams")
    return (a[3]+a[4])/2


def bootstrap(values,seed):
    if len(values)!=SOURCE_COUNT:
        raise ValueError("independent base-cluster count changed")
    rng=random.Random(seed)
    a=sorted(sum(values[rng.randrange(SOURCE_COUNT)]
                 for _ in range(SOURCE_COUNT))/SOURCE_COUNT
             for _ in range(BOOT))
    return [a[int(.025*BOOT)],a[int(.975*BOOT)-1]]


def verify_roster(source):
    if (source["seed"]!=31060 or
        source["n_independent_base_sequences"]!=SOURCE_COUNT or
        source["paired_base_ids"]!=list(range(SOURCE_COUNT)) or
        len(source["cases"])!=SOURCE_COUNT):
        raise ValueError("original LGA07B source roster shape or provenance changed")
    bases=[];targets=[];all_donors=set()
    for i,case in enumerate(source["cases"]):
        if case["base_index"]!=i or len(case["donors"])!=8:
            raise ValueError("source order or exactly eight donors violated")
        x=case["x"];y=case["y"]
        if len(x)!=81 or len(y)!=81 or any(type(v)!=int or not 1<=v<=9 for v in x):
            raise ValueError("input alphabet/length differs")
        if y!=x[-1:]+x[:-1]:
            raise ValueError("source cyclic-copy labels differ")
        for goal,donor in zip(GOALS,case["donors"]):
            predecessor=(goal-1)%81
            d=x.copy()
            d[predecessor]=d[predecessor]%9+1
            dy=d[-1:]+d[:-1]
            if (donor["goal"]!=goal or donor["donor_x"]!=d or
                donor["predecessor_position"]!=predecessor or
                donor["old_digit"]!=x[predecessor] or
                donor["new_digit"]!=d[predecessor] or
                donor["changed_targets"]!=[goal] or
                donor["original_goal_label"]!=y[goal] or
                donor["donor_goal_label"]!=dy[goal]):
                raise ValueError("donor source legal predecessor mutation invalid")
            all_donors.add(tuple(d))
        bases.append(tuple(x))
        targets.append(y)
    if len(set(bases))!=SOURCE_COUNT or len(all_donors)!=SOURCE_COUNT*len(GOALS):
        raise ValueError("new base sources or donor inputs are not unique")
    if sha_int64(bases)!=source["x_sha256"] or sha_int64(targets)!=source["y_sha256"]:
        raise ValueError("source tensor little-endian SHA mismatch")
    return targets


def verify_case(row,source_case,teacher_head,depth,goal):
    idx=source_case["base_index"]
    donor=source_case["donors"][GOALS.index(goal)]
    truth=donor["donor_goal_label"]
    if (row["base_index"]!=idx or row["goal"]!=goal or
        row["donor_goal_target"]!=truth or
        row["original_goal_target"]!=source_case["y"][goal] or
        row["source_goal_slot"]!=GOALS.index(goal) or
        set(row["arms"])!=set(ARMS)):
        raise ValueError("case lost original source/goal/arm provenance")
    if not isinstance(row["admitted"],bool):
        raise ValueError("source admission must be a bool")
    if len(row["energy_gradient_goal"])!=48 or any(
            not math.isfinite(float(z)) for z in row["energy_gradient_goal"]):
        raise ValueError("missing local physical energy gradient")
    if row["fiber_dimension"]<0 or row["fiber_dimension"]>48:
        raise ValueError("invalid original joint-kernel dimension")
    if row["fiber_joint_rank"]+row["fiber_dimension"]!=48:
        raise ValueError("kernel rank-nullity violation")
    if row["fiber_joint_rank"]-row["fiber_head_rank"] not in (0,1):
        raise ValueError("energy row changed rank by more than one")
    if row["fiber_constraint_residual"]>1e-5 or row["fiber_projector_error"]>1e-5:
        raise ValueError("original computed kernel constraints not closed")
    truth=int(truth)
    clean=row["clean"]
    if (len(clean["goal_logits"])!=10 or len(clean["predicted_classes"])!=81 or
        len(row["clean_recipient4_goal_logits"])!=10 or
        len(row["clean_donor4_goal_logits"])!=10):
        raise ValueError("wrong 10-class full-head / 81-token readout")
    if int(max(range(10),key=lambda c:clean["goal_logits"][c]))!=clean["predicted_classes"][goal]:
        raise ValueError("clean native goal argmax contradicts recorded head")
    if clean["donor_target_correct"]!=(clean["predicted_classes"][goal]==truth):
        raise ValueError("clean donor-target rate arithmetic invalid")
    close(clean["donor_goal_CE"],
          logsumexp(clean["goal_logits"])-clean["goal_logits"][truth],
          "clean CE")
    rec_pred=max(range(10),key=lambda c:row["clean_recipient4_goal_logits"][c])
    donor_pred=max(range(10),key=lambda c:row["clean_donor4_goal_logits"][c])
    if row["clean_recipient_correct"]!=(rec_pred==row["original_goal_target"]):
        raise ValueError("clean original-target competence mismatch")
    if row["clean_donor_correct"]!=(donor_pred==truth):
        raise ValueError("clean legal-donor competence mismatch")
    for arm in ARMS:
        data=row["arms"][arm]
        v=data["edit_vector_float32"]
        if len(v)!=48 or len(data["goal_logits"])!=10 or len(data["predicted_classes"])!=81:
            raise ValueError("source raw edit, goal logits or full predicted classes missing")
        if data["edit_sha256"]!=sha_float32(v):
            raise ValueError("actual physical edit vector SHA mismatch")
        norm=vec_norm(v)
        close(norm,data["edit_l2"],"edit physical dose",absolute=2e-5)
        close(norm,data["realized_edit_l2"],"realized physical edit dose",absolute=2e-5)
        if arm.startswith("same_k_sham") or arm in ("projected_donor","inverse_projected_donor"):
            if row["admitted"]:
                close(norm,row["radius"],"admitted same-K radius",absolute=2e-5)
                head_dots=[sum(a*b for a,b in zip(w,v)) for w in teacher_head]
                gdot=sum(a*b for a,b in zip(row["energy_gradient_goal"],v))
                if max(abs(x) for x in head_dots)>2e-5:
                    raise ValueError("source physical head-tangent constraint violated")
                # FP32 physical additions and gradient floats introduce scale-dependent
                # residuals; this checks the normalized first-order energy tangent.
                if abs(gdot)>2e-5*max(1.,vec_norm(row["energy_gradient_goal"])*norm):
                    raise ValueError("source energy tangent constraint violated")
                if data["finite_current_head_l2"]>2e-4:
                    raise ValueError("head-null edit changed current linear head too much")
            elif norm>1e-8 or data["donor_target_correct"]!=clean["donor_target_correct"]:
                raise ValueError("unavailable ITT intervention improperly replaced")
        if arm=="full_donor_token":
            close(norm,row["source_donor_goal_delta_l2"],
                  "natural donor-token physical radius",absolute=2e-5)
        if any(type(z)!=int or z<0 or z>9 for z in data["predicted_classes"]):
            raise ValueError("final logits predicted invalid class")
        predicted=max(range(10),key=lambda c:data["goal_logits"][c])
        if data["predicted_classes"][goal]!=predicted or data["donor_target_correct"]!=(predicted==truth):
            raise ValueError("final model donor-target correctness not from full logits")
        close(data["donor_goal_CE"],
              logsumexp(data["goal_logits"])-data["goal_logits"][truth],
              "native donor goal CE")
        rival=max((c for c in range(10) if c!=truth),
                  key=lambda c:clean["goal_logits"][c])
        if rival!=data["clean_competitor_class"]:
            raise ValueError("clean competitor was selected from edited logits")
        close(data["donor_goal_margin"],
              data["goal_logits"][truth]-data["goal_logits"][rival],
              "native donor margin")
        if not all(math.isfinite(float(data[k])) for k in (
            "finite_current_head_l2","finite_shared_E_change",
            "final_shared_E_change")):
            raise ValueError("finite native head/energy telemetry missing")


def audit(teacher_bundle,teacher_meta,potential_bundle,potential_meta,lock_path,
          result_path,out):
    binaries=[Path(p).read_bytes() for p in (
        teacher_bundle,teacher_meta,potential_bundle,potential_meta)]
    if dict(zip(SHAS,[sha(z) for z in binaries]))!=SHAS:
        raise ValueError("frozen first-science artifact SHA changed")
    lock_raw=Path(lock_path).read_bytes()
    lock=json.loads(lock_raw)
    expected_protocol={p:sha_file(HERE/p) for p in PROTOCOLS}
    expected_code={p:sha_file(HERE/p) for p in CODE}
    if (lock["schema"]!=LOCK_SCHEMA or
        lock["original_artifact_shas"]!=SHAS or
        lock["protocol_sha256"]!=expected_protocol or
        lock["implementation_sha256"]!=expected_code or
        lock["test_seed_reserved"]!=31060 or
        lock["teacher_seeds"]!=list(TEACHERS) or
        lock["n_independent_base_sequences"]!=SOURCE_COUNT or
        lock["depths"]!=list(DEPTHS) or
        lock["primary_depth"]!=2 or
        lock["goals"]!=list(GOALS) or
        lock["new_test_generated"] is not False or
        lock["test_labels_read"] is not False or
        lock["source_generation_complete"] is not False or
        lock["experiment_scored"] is not False or
        lock["verified_tensor_loaders"] is not True):
        raise ValueError("source-only lock or prospective chronology changed")
    data=json.loads(Path(result_path).read_bytes())
    if (data["schema"]!=SCHEMA or
        data["source_lock_sha256"]!=sha(lock_raw) or
        data["original_artifact_shas"]!=SHAS or
        data["protocol_sha256"]!=expected_protocol or
        data["implementation_sha256"]!=expected_code or
        data["teacher_seeds"]!=list(TEACHERS) or
        data["native_depths"]!=list(DEPTHS) or
        data["n_independent_base_sources"]!=SOURCE_COUNT or
        data["n_cases_per_teacher"]!=SOURCE_COUNT*len(GOALS) or
        data["arms"]!=list(ARMS) or
        data["evidence_class"]!="EXPOSED_TEACHERS_DEVELOPMENT_ONLY" or
        set(data["teachers"])!=set(map(str,TEACHERS))):
        raise ValueError("original receipt contract or teacher roster drift")
    verify_roster(data["test_source"])
    original_roster=data["test_source"]["cases"]
    teacher_summaries={}
    n_passes=0
    for sid in TEACHERS:
        root=data["teachers"][str(sid)]
        if set(root)!={"2","6","head_weight"}:
            raise ValueError("native teacher result keys changed")
        weight=root["head_weight"]
        if len(weight)!=10 or any(len(z)!=48 for z in weight):
            raise ValueError("full original teacher readout weight missing")
        teacher_summaries[str(sid)]={}
        for depth in DEPTHS:
            cell=root[str(depth)]
            cases=cell["cases"]
            if len(cases)!=SOURCE_COUNT*len(GOALS):
                raise ValueError("entire registered case roster incomplete")
            base_values=[];sel=sham=joint=full=eligible=0.
            for i in range(SOURCE_COUNT):
                vals=[]
                for j,goal in enumerate(GOALS):
                    row=cases[i*len(GOALS)+j]
                    verify_case(row,original_roster[i],weight,depth,goal)
                    x=float(row["arms"]["projected_donor"]["donor_target_correct"])
                    sham_binary=[float(row["arms"][f"same_k_sham_{s}"]["donor_target_correct"])
                                 for s in range(8)]
                    med=median_binary(sham_binary)
                    vals.append(x-med);sel+=x;sham+=med
                    eligible+=int(row["admitted"])
                    competent=int(row["clean_recipient_correct"] and row["clean_donor_correct"])
                    joint+=competent
                    full+=competent*int(row["arms"]["full_donor_token"]["donor_target_correct"])
                base_values.append(sum(vals)/len(vals))
            effect=sum(base_values)/SOURCE_COUNT
            ci=bootstrap(base_values,stable_seed("bootstrap",sid,depth))
            donor_rate=full/joint if joint else 0.
            gate=bool(depth==2 and effect>=.15 and ci[0]>0 and
                      joint/(SOURCE_COUNT*len(GOALS))>=.8 and donor_rate>=.95)
            check=cell["summary"]
            for key,value in {
                "n_independent_base_sources":SOURCE_COUNT,
                "n_original_goals":SOURCE_COUNT*len(GOALS),
                "n_admitted":eligible,
                "n_jointly_competent":joint,
                "competence_fraction":joint/(SOURCE_COUNT*len(GOALS)),
                "selected_ITT_rate":sel/(SOURCE_COUNT*len(GOALS)),
                "sham_median_ITT_rate":sham/(SOURCE_COUNT*len(GOALS)),
                "ITT_paired_effect":effect,
                "full_donor_token_correct_among_jointly_competent":donor_rate,
            }.items():
                close(check[key],value,key,absolute=1e-8,relative=1e-9)
            for a,b in zip(check["whole_base_bootstrap95"],ci):
                close(a,b,"2000 original whole-source bootstrap",absolute=1e-9)
            for a,b in zip(check["base_level_paired_effects"],base_values):
                close(a,b,"all 64 paired base-root effects",absolute=1e-9)
            if check["teacher_engineering_gate"]!=gate or check["bootstrap_repetitions"]!=BOOT:
                raise ValueError("frozen teacher development gate changed")
            if depth==2:n_passes+=int(gate)
            teacher_summaries[str(sid)][str(depth)]={
                "mean_ITT_effect":effect,"bootstrap95":ci,
                "source_admission":eligible,"joint_competent":joint,
                "larger_dose_donor_token_rate":donor_rate,
                "development_gate":gate,
            }
    verdict=("DEVELOPMENT_ENGINEERING_POSITIVE" if n_passes>=2
             else "DEVELOPMENT_NULL_OR_INELIGIBLE")
    if (data["n_teachers_passing_engineering_gate"]!=n_passes or
        data["registered_development_verdict"]!=verdict):
        raise ValueError("first registered development verdict changed")
    summary={
        "schema":"LGA07B/independent-stdlib-first-native-audit-v1",
        "audit_status":"PASS","evidence_class":"DEVELOPMENT_ONLY",
        "new_test_seed":31060,
        "verified_original_test_sha256":sha_file(result_path),
        "verified_original_source_lock_sha256":sha(lock_raw),
        "n_original_source_goal_cases":len(TEACHERS)*SOURCE_COUNT*len(GOALS),
        "n_native_depths":len(DEPTHS),
        "n_native_edited_arm_records":
            len(TEACHERS)*SOURCE_COUNT*len(GOALS)*len(DEPTHS)*len(ARMS),
        "n_teachers_passing_engineering_gate":n_passes,
        "registered_development_verdict":verdict,
        "teachers":teacher_summaries,
        "torch_imported":False,
    }
    target=Path(out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    print("LGA07B_INDEPENDENT_AUDIT_PASS",verdict,flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    for arg in ("teacher-bundle","teacher-meta","potential-bundle",
                "potential-meta","source-lock","test-receipt","out"):
        p.add_argument("--"+arg,required=True,type=Path)
    x=p.parse_args()
    audit(x.teacher_bundle,x.teacher_meta,x.potential_bundle,
          x.potential_meta,x.source_lock,x.test_receipt,x.out)


if __name__=="__main__":
    main()

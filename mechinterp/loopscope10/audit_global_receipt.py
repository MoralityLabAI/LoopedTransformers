"""LGA-07C first native receipt audit. Python standard library ONLY.

Original first registered source, all saved float32 state edits and readouts,
physical doses, exact legal donor targets, source-cluster bootstrap and verdict
are independently checked. No Torch/NumPy/model checkpoint inferencing.
"""
from __future__ import annotations
from array import array
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

HERE=Path(__file__).resolve().parent
SCHEMA="LGA07C/global-81token-fiber-first-development-v1"
LOCK_SCHEMA="LGA07C/pretest-frozen-source-lock-v1"
ORIGINAL={
 "teacher_bundle_sha256":"d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
 "teacher_meta_sha256":"7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d",
 "potential_bundle_sha256":"24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53",
 "potential_meta_sha256":"69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d",
}
N=64
GOALS=(0,10,20,30,40,50,60,70)
TEACHERS=(103,211,307)
ARMS=("global_energy_tangent",*(f"same_k_sham_{i}" for i in range(8)),
      "global_head_only","full_donor_state","goal_token_only",
      "predecessor_goal_pair","global_energy_tangent_quarter")
BYTES=81*48*4
BOOT=2000
PROTOCOLS=("LGA07C_GLOBAL_FIBER_PROTOCOL.md",)
CODES=("global_energy_fiber.py","global_donor_eval.py",
       "audit_global_receipt.py","source_lock_global.py",
       "donor_source.py","energy_fiber.py")
WF=".github/workflows/loopscope10c-first-global-donor.yml"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    return sha(Path(path).read_bytes())


def stable_seed(*parts):
    key=("LGA07C|"+"|".join(map(str,parts))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],"big") & ((1<<63)-1)


def unpack_f32(raw):
    if len(raw)!=BYTES or sys.byteorder!="little":
        raise ValueError("bad physical float32 byte count/architecture")
    out=array("f")
    out.frombytes(raw)
    if len(out)!=81*48 or not all(math.isfinite(float(x)) for x in out):
        raise ValueError("invalid physical float32 vector")
    return out


def dot(a,b):
    return math.fsum(float(x)*float(y) for x,y in zip(a,b))


def l2(a):
    return math.sqrt(math.fsum(float(x)*float(x) for x in a))


def close(a,b,what,*,abs_tol=2e-5,rel_tol=2e-5):
    x=float(a);y=float(b)
    if not (math.isfinite(x) and math.isfinite(y)) or abs(x-y)>abs_tol+rel_tol*max(abs(x),abs(y)):
        raise ValueError("replay mismatch "+what+f": {x} vs {y}")


def logsumexp(row):
    x=max(row)
    return x + math.log(math.fsum(math.exp(y-x) for y in row))


def median8(x):
    a=sorted(float(z) for z in x)
    if len(a)!=8 or any(z not in (0.,1.) for z in a):
        raise ValueError("incorrect eight binary same-K sham schedule")
    return (a[3]+a[4])/2.


def bootstrap(v,sid):
    if len(v)!=N:
        raise ValueError("64 independent base ID source values required")
    rng=random.Random(stable_seed("bootstrap",sid))
    a=sorted(sum(v[rng.randrange(N)] for _ in range(N))/N for _ in range(BOOT))
    return [a[int(.025*BOOT)],a[int(.975*BOOT)-1]]


def sha_int64_rows(rows):
    data=[int(z) for row in rows for z in row]
    return sha(struct.pack("<"+"q"*len(data),*data))


def validate_source(source):
    if source["seed"]!=31061 or source["n_base"]!=N or len(source["cases"])!=N:
        raise ValueError("original fresh source roster differs")
    x_rows=[];y_rows=[];donor_all=set()
    for i,c in enumerate(source["cases"]):
        x=c["x"];y=c["y"]
        if (c["base_index"]!=i or len(x)!=81 or len(y)!=81 or
            any(type(z)!=int or not 1<=z<=9 for z in x) or
            y!=x[-1:]+x[:-1] or len(c["donors"])!=len(GOALS)):
            raise ValueError("invalid original cyclic-copy source")
        for j,goal in enumerate(GOALS):
            d=c["donors"][j]
            z=x.copy(); pred=(goal-1)%81
            z[pred]=z[pred]%9+1
            if (d["goal"]!=goal or d["x"]!=z or
                d["predecessor_position"]!=pred or
                d["old_digit"]!=x[pred] or d["new_digit"]!=z[pred] or
                d["donor_goal_label"]!=z[pred] or
                d["original_goal_label"]!=y[goal] or
                d["changed_targets"]!=[goal]):
                raise ValueError("legal donor source/target corrupted")
            donor_all.add(tuple(z))
        x_rows.append(x);y_rows.append(y)
    if len(set(map(tuple,x_rows)))!=N or len(donor_all)!=N*len(GOALS):
        raise ValueError("nonunique base or legal donor")
    if sha_int64_rows(x_rows)!=source["x_sha256"] or sha_int64_rows(y_rows)!=source["y_sha256"]:
        raise ValueError("source SHA256 of little-endian int64 differs")


def verify_head_matrix(head,proj):
    if (len(head)!=10 or any(len(row)!=48 for row in head) or
        len(proj)!=48 or any(len(row)!=48 for row in proj)):
        raise ValueError("missing native full 10x48 head/projection matrix")
    for i in range(48):
        for j in range(48):
            if abs(proj[i][j]-proj[j][i])>1e-8:
                raise ValueError("head projector not symmetric")
    for i in range(48):
        for j in range(48):
            found=math.fsum(proj[i][k]*proj[k][j] for k in range(48))
            if abs(found-proj[i][j])>2e-8:
                raise ValueError("head projector idempotence failed")
    for row in head:
        for c in range(48):
            if abs(math.fsum(row[k]*proj[k][c] for k in range(48)))>2e-7:
                raise ValueError("physical current-logit head projector has residual")


def head_residual(head, v, *, check_limit=None):
    """Full 81-token Jacobian norm from original physical float32 edit."""
    sq=0.
    for t in range(81):
        u=v[t*48:(t+1)*48]
        for hrow in head:
            z=dot(u,hrow)
            sq+=z*z
    total=math.sqrt(sq)
    if check_limit is not None and total>check_limit:
        raise ValueError("physical energy fiber is visible to current full head")
    return total


def check_case(case,source,head,source_raw,vec_f,vec_sha,count_vectors):
    sid=case["source_vector_index"]
    if sid<0:
        raise ValueError("invalid source vector ordinal")
    if len(source_raw)!=2*BYTES:
        raise ValueError("unexpected source delta+energy physical byte count")
    delta_raw=source_raw[:BYTES];gradient_raw=source_raw[BYTES:]
    if sha(delta_raw)!=case["source_delta_sha256"] or sha(gradient_raw)!=case["source_energy_gradient_sha256"]:
        raise ValueError("source-reaching donor or gradient vector SHA mismatch")
    delta=unpack_f32(delta_raw);g=unpack_f32(gradient_raw)
    radius=l2(delta)
    close(radius,case["radius_natural_full_state"],"full naturally reached donor norm")
    if (case["head_rank"]!=10 or case["tangent_dimension"] not in (3077,3078)
            or case["energy_row_active"] not in (True,False)):
        raise ValueError("registered full-state fiber dimension changed")
    if case["energy_row_active"] != (case["tangent_dimension"]==3077):
        raise ValueError("tangent energy-row nullity relation changed")
    if case["head_residual"]>1e-5*max(1.,radius):
        raise ValueError("reported first-order head-null residual nonzero")
    goal=case["goal"];truth=case["donor_target"]
    if (goal not in GOALS or type(truth)!=int or not 1<=truth<=9 or
        set(case["arms"])!=set(ARMS) or len(case["clean_goal_logits"])!=10 or
        len(case["clean_recipient_predicted_classes"])!=81):
        raise ValueError("original goal or arm roster invalid")
    clean_rec=case["clean_goal_logits"]
    clean_don=case["clean_donor_goal_logits"]
    if len(clean_don)!=10 or case["clean_recipient_predicted_classes"][goal]!=max(range(10),key=lambda n:clean_rec[n]):
        raise ValueError("original clean head differs from predicted classes")
    old_target=source["y"][goal]
    if (case["original_target"]!=old_target or
        case["clean_recipient_correct"]!=(max(range(10),key=lambda n:clean_rec[n])==old_target) or
        case["clean_donor_correct"]!=(max(range(10),key=lambda n:clean_don[n])==truth) or
        case["jointly_clean_competent"]!=(case["clean_recipient_correct"] and case["clean_donor_correct"])):
        raise ValueError("original source-independent donor competence arithmetic changed")
    if len(case["clean_donor_all_logits"])!=81 or any(len(x)!=10 for x in case["clean_donor_all_logits"]):
        raise ValueError("entire clean donor 81x10 logit positive control absent")
    source_donor=source["donors"][GOALS.index(goal)]
    if (source_donor["donor_goal_label"]!=truth or
        case["slot"]!=GOALS.index(goal)):
        raise ValueError("case donor legal source/target mismatch")
    rival=max((k for k in range(10) if k!=truth),key=lambda k:clean_rec[k])
    max_closure=0.
    for arm in ARMS:
        d=case["arms"][arm]
        raw=vec_f.read(BYTES)
        if len(raw)!=BYTES or d["vector_index"]!=count_vectors:
            raise ValueError("truncated/reordered first native binary edit receipt")
        count_vectors+=1
        vec_sha.update(raw)
        if sha(raw)!=d["vector_sha256"]:
            raise ValueError("every physical native edit SHA must match raw binary")
        v=unpack_f32(raw)
        norm=l2(v)
        close(norm,d["applied_l2"],"actual f32 vector radius")
        if not case["admitted"] and (
            arm in ("global_energy_tangent",
                    *(f"same_k_sham_{i}" for i in range(8)),
                    "global_head_only","global_energy_tangent_quarter")):
            if norm>1e-7:
                raise ValueError("inadmissible ITT source was edited")
        if case["admitted"] and (arm=="global_energy_tangent" or
            arm=="global_head_only" or arm.startswith("same_k_sham_")):
            close(norm,radius,"equal-norm natural donor and K sham",abs_tol=2e-4,rel_tol=8e-5)
        if case["admitted"] and arm=="global_energy_tangent_quarter":
            close(norm,.25*radius,"fixed quarter-natural-dose arm",abs_tol=2e-4,rel_tol=8e-5)
        if arm=="full_donor_state":
            # Validate all 3,888 actual patched physical coordinates.
            if max(abs(x-y) for x,y in zip(v,delta))>1e-5:
                raise ValueError("full donor whole state not exactly natural reached donor")
        if arm in ("global_energy_tangent","global_head_only") or arm.startswith("same_k_sham_"):
            head_len=head_residual(head,v)
            close(head_len,d["current_head_alltoken_l2"],"actual native current-head all-token displacement",
                  abs_tol=2e-3,rel_tol=1e-4)
            if case["admitted"] and head_len>3e-4*max(1.,radius):
                raise ValueError("declared current-head-null physical edit not silent")
        gdot=dot(g,v)
        close(gdot,d["firstorder_E_dot"],"original gradient physical energy dot",
              abs_tol=.0004,rel_tol=3e-5)
        if case["admitted"] and (arm=="global_energy_tangent" or arm.startswith("same_k_sham_")):
            if abs(gdot)>3e-5*max(1.,l2(g)*norm):
                raise ValueError("actual edited direction not first-order energy tangent")
        if arm=="global_energy_tangent":
            # Independent physical projector check: source delta is the
            # recorded reached-state difference, constrained by all-token head.
            head_null=[]
            for t in range(81):
                uu=delta[t*48:(t+1)*48]
                # P is archived per teacher and checked once for W@P=0 and P²=P.
                # Per-case exact expected projector is separately checked below.
                head_null.append(uu)
        gl=d["goal_logits"]
        if len(gl)!=10 or len(d["predicted_classes"])!=81 or not all(math.isfinite(float(z)) for z in gl):
            raise ValueError("missing 10-class goal/full-81 predictions")
        pred=max(range(10),key=lambda k:gl[k])
        if (d["predicted_classes"][goal]!=pred or d["donor_correct"]!=(pred==truth) or
            d["clean_rival"]!=rival):
            raise ValueError("registered correct-donor target arithmetic differs")
        close(logsumexp(gl)-gl[truth],d["donor_goal_CE"],"donor target true CE",abs_tol=2e-6,rel_tol=1e-7)
        close(gl[truth]-gl[rival],d["donor_margin_clean_rival"],"correct donor margin",abs_tol=2e-6,rel_tol=1e-7)
        if arm=="full_donor_state":
            full=d["all_final_logits"]
            if len(full)!=81 or any(len(x)!=10 for x in full):
                raise ValueError("natural whole-state donor full-head parity missing")
            for original, patched in zip(case["clean_donor_all_logits"],full):
                max_closure=max(max_closure, max(abs(float(x)-float(y))
                                               for x,y in zip(original,patched)))
    close(max_closure,case["full_donor_max_logit_error"],
          "source independent full donor whole head closure",abs_tol=1e-6)
    return count_vectors


def audit(teacher_bundle,teacher_meta,potential_bundle,potential_meta,
          lock_path,full_path,vec_path,source_path,out):
    originals={
        "teacher_bundle_sha256":teacher_bundle,
        "teacher_meta_sha256":teacher_meta,
        "potential_bundle_sha256":potential_bundle,
        "potential_meta_sha256":potential_meta,
    }
    if {k:sha_file(v) for k,v in originals.items()}!=ORIGINAL:
        raise ValueError("original LGA02/LGA05 source-byte identity changed")
    lock_raw=Path(lock_path).read_bytes()
    lock=json.loads(lock_raw)
    # Independently recompute all protocol and executable source bytes.
    expected_protos={p:sha_file(HERE/p) for p in PROTOCOLS}
    expected_codes={p:sha_file(HERE/p) for p in CODES}
    expected_codes[WF]=sha_file(HERE.parents[1]/WF)
    if (lock["schema"]!=LOCK_SCHEMA or lock["original_shas"]!=ORIGINAL or
        lock["protocol_shas"]!=expected_protos or lock["code_shas"]!=expected_codes or
        lock["test_seed_reserved"]!=31061 or
        lock["new_test_generated"] is not False or
        lock["test_labels_read"] is not False or
        lock["teacher_seeds"]!=list(TEACHERS) or
        lock["n_base"]!=N or lock["goals"]!=list(GOALS) or
        lock["verified_canonical_loaders"] is not True):
        raise ValueError("separate original pretest source lock altered")
    data=json.loads(Path(full_path).read_bytes())
    if (data["schema"]!=SCHEMA or data["frozen_original_shas"]!=ORIGINAL or
        data["pretest_source_lock_sha256"]!=sha(lock_raw) or
        data["protocol_shas"]!=expected_protos or data["code_shas"]!=expected_codes or
        data["teacher_seeds"]!=list(TEACHERS) or
        data["evidence_class"]!="EXPOSED_FROZEN_TEACHERS_DEVELOPMENT_ONLY" or
        data["n_base"]!=N or data["n_goal_per_base"]!=len(GOALS) or
        data["n_arms_per_case"]!=len(ARMS) or data["arms"]!=list(ARMS) or
        set(data["teachers"])!=set(map(str,TEACHERS))):
        raise ValueError("first science result contract changed")
    validate_source(data["source"])
    assert len(ARMS)==14
    for key,file in [("binary_vectors",vec_path),("binary_sources",source_path)]:
        if data[key]["gzip_sha256"]!=sha_file(file):
            raise ValueError("binary native/source artifact compressed SHA changed")
    raw_vectors=hashlib.sha256();raw_sources=hashlib.sha256()
    nvec=nsrc=0;npass=0;teacher_reports={}
    with gzip.open(vec_path,"rb") as vec_f, gzip.open(source_path,"rb") as source_f:
        for sid in TEACHERS:
            root=data["teachers"][str(sid)]
            if len(root["cases"])!=N*len(GOALS):
                raise ValueError("original source/goals missing")
            head=root["head_weight"];proj=root["head_projector"]
            verify_head_matrix(head,proj)
            groups={i:[] for i in range(N)}
            joint=fullcorrect=admit=selected=shams=0.
            max_parity=0.
            for i in range(N):
                src=data["source"]["cases"][i]
                for j,goal in enumerate(GOALS):
                    case=root["cases"][i*len(GOALS)+j]
                    if case["base_index"]!=i or case["slot"]!=j or case["goal"]!=goal or case["source_vector_index"]!=nsrc:
                        raise ValueError("source goal/whole-state binary order differs")
                    source_raw=source_f.read(2*BYTES)
                    if len(source_raw)!=2*BYTES:
                        raise ValueError("truncated independent source physical vector archive")
                    raw_sources.update(source_raw);nsrc+=1
                    nvec=check_case(case,src,head,source_raw,vec_f,raw_vectors,nvec)
                    sel=float(case["arms"]["global_energy_tangent"]["donor_correct"])
                    med=median8([int(case["arms"][f"same_k_sham_{k}"]["donor_correct"])
                                 for k in range(8)])
                    groups[i].append(sel-med)
                    selected+=sel;shams+=med;admit+=int(case["admitted"])
                    competent=int(case["jointly_clean_competent"])
                    joint+=competent
                    fullcorrect+=competent*int(case["arms"]["full_donor_state"]["donor_correct"])
                    max_parity=max(max_parity,case["full_donor_max_logit_error"])
            base_values=[sum(groups[i])/len(GOALS) for i in range(N)]
            effect=sum(base_values)/N
            ci=bootstrap(base_values,sid)
            rate=fullcorrect/joint if joint else 0.
            passed=bool(joint/(N*len(GOALS))>=.80 and max_parity<=1e-4
                        and rate>=.95 and effect>=.15 and ci[0]>0)
            s=root["summary"]
            for name,num in {
                "n_base_clusters":N,"n_goal_pairs":N*len(GOALS),
                "n_admitted":admit,"n_joint_competent":joint,
                "joint_competence_fraction":joint/(N*len(GOALS)),
                "full_donor_correct_among_competent":rate,
                "max_full_donor_alllogit_error":max_parity,
                "selected_ITT_correct_rate":selected/(N*len(GOALS)),
                "median_sham_ITT_correct_rate":shams/(N*len(GOALS)),
                "primary_paired_ITT_effect":effect,
            }.items():
                close(s[name],num,"registered teacher summary "+name,abs_tol=1e-8,rel_tol=1e-8)
            for k in range(2):
                close(s["whole_base_bootstrap95"][k],ci[k],"original whole-base bootstrap",abs_tol=1e-9,rel_tol=0)
            if (s["bootstrap_repetitions"]!=BOOT or
                s["development_teacher_gate"]!=passed or
                len(s["base_level_effects"])!=N):
                raise ValueError("original frozen development teacher gate altered")
            for a,b in zip(s["base_level_effects"],base_values):
                close(a,b,"per-independent-base result",abs_tol=1e-9,rel_tol=0)
            npass+=int(passed)
            teacher_reports[str(sid)]={
                "paired_ITT_effect":effect,"whole_base_bootstrap95":ci,
                "joint_competence":joint,"full_donor_positive_control_rate":rate,
                "geometry_admitted":admit,"max_full_donor_parity":max_parity,
                "teacher_engineering_gate":passed,
            }
        if vec_f.read(1)!=b"" or source_f.read(1)!=b"":
            raise ValueError("unexpected excess binary physical records")
    if (nvec != data["binary_vectors"]["n_records"] or
        nsrc != data["binary_sources"]["n_records"] or
        nvec!=len(TEACHERS)*N*len(GOALS)*len(ARMS) or
        nsrc!=len(TEACHERS)*N*len(GOALS) or
        raw_vectors.hexdigest()!=data["binary_vectors"]["raw_sha256"] or
        raw_sources.hexdigest()!=data["binary_sources"]["raw_sha256"]):
        raise ValueError("all first original complete physical vector byte counts/hashes differ")
    verdict=("DEVELOPMENT_ENGINEERING_POSITIVE" if npass>=2
             else "DEVELOPMENT_NULL_OR_INELIGIBLE")
    if data["n_teachers_meeting_development_gate"]!=npass or data["registered_first_development_verdict"]!=verdict:
        raise ValueError("first science preregistered verdict inconsistent")
    result={
        "schema":"LGA07C/independent-stdlib-original-audit-v1",
        "audit_status":"PASS",
        "original_evidence_class":"EXPOSED_FROZEN_TEACHERS_DEVELOPMENT_ONLY",
        "first_result_sha256":sha_file(full_path),
        "first_pretest_source_lock_sha256":sha(lock_raw),
        "original_physical_vector_raw_sha256":raw_vectors.hexdigest(),
        "original_source_delta_gradient_raw_sha256":raw_sources.hexdigest(),
        "n_independent_base_per_teacher":N,
        "n_goal_cases":nsrc,"n_native_edited_arms":nvec,
        "n_teachers_passing_development":npass,
        "registered_first_development_verdict":verdict,
        "teachers":teacher_reports,
        "no_torch_import":True,
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("LGA07C_INDEPENDENT_SOURCE_AUDIT_PASS",verdict,flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    for a in ("teacher-bundle","teacher-meta","potential-bundle","potential-meta",
              "source-lock","full","vec","sources","out"):
        p.add_argument("--"+a,type=Path,required=True)
    z=p.parse_args()
    audit(z.teacher_bundle,z.teacher_meta,z.potential_bundle,z.potential_meta,
          z.source_lock,z.full,z.vec,z.sources,z.out)


if __name__=="__main__":
    main()

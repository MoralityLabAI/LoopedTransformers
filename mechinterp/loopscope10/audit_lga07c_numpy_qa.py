"""Post-result LGA-07C QA-01, accelerated independent NumPy first-file audit.

This is a separate descriptive implementation audit of the immutable original
Actions #44283810760 artifacts. It is NOT the prospectively registered stdlib
auditor, does not rerun models, does not regenerate seed 31061 and cannot
change the original first-science verdict.
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
import numpy as np

SCHEMA="LGA07C/post-result-independent-numpy-qa-v1"
HASHES={
 "lock.json":"79478f82dc7064650a89f031e15895c472d09d842b37c315641acbc60e42aac9",
 "full.json":"e48861f1834b7995be3aee1da91d84affd10218d92c54c4dbcead3a8349ce25e",
 "edits_f32.gz":"c4969a9651274cb116e4ea5f813253a334c7de722d230becb6c70f086a3d1605",
 "source_grad_f32.gz":"4facab71876c8d670246aa7c78474b28b66dc1fe376f9498baf19905a68c6f75",
}
GOALS=(0,10,20,30,40,50,60,70)
TEACHERS=(103,211,307)
ARMS=("global_energy_tangent",*(f"same_k_sham_{j}" for j in range(8)),
      "global_head_only","full_donor_state","goal_token_only",
      "predecessor_goal_pair","global_energy_tangent_quarter")
N=64
BYTES=81*48*4
BOOT=2000


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha(Path(p).read_bytes())


def seed(*parts):
    return int.from_bytes(hashlib.sha256(
        ("LGA07C|"+"|".join(map(str,parts))).encode()).digest()[:8],"big") & ((1<<63)-1)


def checkclose(x,y,name,atol=4e-5,rtol=2e-5):
    if not np.isfinite(x) or abs(float(x)-float(y))>atol+rtol*max(abs(float(x)),abs(float(y))):
        raise ValueError(f"{name}: {x} != {y}")


def boot(v,sid):
    rng=random.Random(seed("bootstrap",sid))
    s=sorted(sum(v[rng.randrange(N)] for _ in range(N))/N for _ in range(BOOT))
    return [s[int(.025*BOOT)],s[int(.975*BOOT)-1]]


def logsumexp(a):
    m=float(np.max(a))
    return m+math.log(float(np.exp(a-m).sum()))


def do_audit(lockpath,fullpath,vecpath,sourcepath,outpath):
    paths={"lock.json":lockpath,"full.json":fullpath,
           "edits_f32.gz":vecpath,"source_grad_f32.gz":sourcepath}
    for name,p in paths.items():
        if sha_file(p)!=HASHES[name]:
            raise ValueError("Original first-science artifact byte hash changed: "+name)
    lock=json.loads(Path(lockpath).read_bytes())
    d=json.loads(Path(fullpath).read_bytes())
    if (lock["new_test_generated"] is not False or
        lock["test_seed_reserved"]!=31061 or
        d["source"]["seed"]!=31061 or
        d["evidence_class"]!="EXPOSED_FROZEN_TEACHERS_DEVELOPMENT_ONLY" or
        d["pretest_source_lock_sha256"]!=HASHES["lock.json"] or
        d["teacher_seeds"]!=list(TEACHERS) or
        d["arms"]!=list(ARMS) or
        d["n_base"]!=N):
        raise ValueError("original source release chronology or schema changed")
    source=d["source"]
    rows=source["cases"]
    assert len(rows)==N
    xx=[];yy=[];donor_keys=set()
    for i,c in enumerate(rows):
        x=c["x"];y=c["y"]
        if c["base_index"]!=i or len(x)!=81 or y!=x[-1:]+x[:-1] or len(c["donors"])!=8:
            raise ValueError("source independent cyclic input/label invalid")
        for j,goal in enumerate(GOALS):
            donor=c["donors"][j]
            altered=x[:]
            altered[(goal-1)%81]=x[(goal-1)%81]%9+1
            dy=altered[-1:]+altered[:-1]
            if donor["x"]!=altered or donor["goal"]!=goal or donor["donor_goal_label"]!=dy[goal]:
                raise ValueError("wrong legal source donor/matched goal")
            donor_keys.add(tuple(altered))
        xx.extend(x);yy.extend(y)
    if len(set(tuple(x["x"]) for x in rows))!=N or len(donor_keys)!=N*8:
        raise ValueError("source roster duplicates")
    if sha(struct.pack("<"+"q"*len(xx),*xx))!=source["x_sha256"] or sha(
        struct.pack("<"+"q"*len(yy),*yy))!=source["y_sha256"]:
        raise ValueError("source independent int64 SHA differs")
    vector_index=source_index=0
    vec_digest=hashlib.sha256();src_digest=hashlib.sha256()
    reports={}
    with gzip.open(vecpath,"rb") as vectors, gzip.open(sourcepath,"rb") as sources:
        for sid in TEACHERS:
            root=d["teachers"][str(sid)]
            w=np.asarray(root["head_weight"],np.float64)
            P=np.asarray(root["head_projector"],np.float64)
            if (w.shape!=(10,48) or P.shape!=(48,48) or np.linalg.matrix_rank(w)!=10
                or not np.allclose(P,P.T,atol=1e-9,rtol=0)
                or not np.allclose(P@P,P,atol=1e-9,rtol=0)
                or not np.allclose(w@P,0,atol=1e-7,rtol=0)
                or abs(float(np.trace(P))-38)>1e-6):
                raise ValueError("original readout head/kernel projector not physical")
            rootcases=root["cases"]
            if len(rootcases)!=N*8:raise ValueError("missing raw source-goal rows")
            base_groups={i:[] for i in range(N)}
            n_admitted=n_joint=full_correct=selected_correct=median_shams=0
            maxparity=0.
            for i in range(N):
                for j,goal in enumerate(GOALS):
                    case=rootcases[i*8+j]
                    if (case["base_index"]!=i or case["slot"]!=j or
                        case["goal"]!=goal or case["source_vector_index"]!=source_index):
                        raise ValueError("first binary source order invalid")
                    source_index+=1
                    raw=sources.read(2*BYTES)
                    if len(raw)!=2*BYTES:raise ValueError("truncated source vector bytes")
                    src_digest.update(raw)
                    delta=np.frombuffer(raw[:BYTES],dtype="<f4").astype(np.float64).reshape(81,48)
                    g=np.frombuffer(raw[BYTES:],dtype="<f4").astype(np.float64).reshape(81,48)
                    if sha(raw[:BYTES])!=case["source_delta_sha256"] or sha(raw[BYTES:])!=case["source_energy_gradient_sha256"]:
                        raise ValueError("frozen donor/energy derivative source hash changed")
                    radius=float(np.linalg.norm(delta))
                    checkclose(radius,case["radius_natural_full_state"],"native donor displacement")
                    u=delta@P
                    gh=g@P
                    active=float(np.linalg.norm(gh))>max(1e-12,1e-9*float(np.linalg.norm(g)))
                    tang=u-((u*gh).sum()/(gh*gh).sum())*gh if active else u
                    if case["tangent_dimension"]!=81*(48-10)-int(active) or case["energy_row_active"]!=active:
                        raise ValueError("registered head/energy rank changed")
                    checkclose(float(np.linalg.norm(tang))/radius,case["energy_tangent_fraction"],
                               "tangent donor fraction",atol=1e-6,rtol=1e-6)
                    norm=float(np.linalg.norm(tang))
                    chosen=tang*radius/norm if norm>1e-10 else np.zeros_like(tang)
                    natural=np.asarray(case["clean_donor_all_logits"],np.float64)
                    n_admitted+=int(case["admitted"])
                    n_joint+=int(case["jointly_clean_competent"])
                    selected=int(case["arms"]["global_energy_tangent"]["donor_correct"])
                    shams=[int(case["arms"][f"same_k_sham_{z}"]["donor_correct"]) for z in range(8)]
                    median=(sorted(shams)[3]+sorted(shams)[4])/2
                    base_groups[i].append(selected-median)
                    selected_correct+=selected;median_shams+=median
                    full_correct+=int(case["jointly_clean_competent"])*int(case["arms"]["full_donor_state"]["donor_correct"])
                    maxparity=max(maxparity,float(case["full_donor_max_logit_error"]))
                    truth=case["donor_target"]
                    for name in ARMS:
                        a=case["arms"][name]
                        vraw=vectors.read(BYTES)
                        if len(vraw)!=BYTES or a["vector_index"]!=vector_index or sha(vraw)!=a["vector_sha256"]:
                            raise ValueError("all original raw physical vector SHA/order must match")
                        vector_index+=1;vec_digest.update(vraw)
                        v=np.frombuffer(vraw,dtype="<f4").astype(np.float64).reshape(81,48)
                        checkclose(float(np.linalg.norm(v)),a["applied_l2"],"applied native physical dose")
                        if name=="full_donor_state" and not np.array_equal(v,delta):
                            raise ValueError("natural full donor source state lost")
                        if name=="global_energy_tangent" and np.max(np.abs(v-chosen))>2e-5:
                            raise ValueError("selected native edit differs from independent K projection")
                        if name in ("global_head_only","global_energy_tangent") or name.startswith("same_k_sham_"):
                            if float(np.linalg.norm(v@w.T))>4e-4*max(1.,radius):
                                raise ValueError("global physical state edit is not current-head silent")
                        if name=="global_energy_tangent" or name.startswith("same_k_sham_"):
                            if abs(float(np.sum(v*g)))>4e-5*max(1.,float(np.linalg.norm(v))*float(np.linalg.norm(g))):
                                raise ValueError("actual physical edit outside energy tangent")
                        logits=np.asarray(a["goal_logits"],dtype=np.float64)
                        if (logits.shape!=(10,) or int(logits.argmax())!=a["predicted_classes"][goal]
                            or bool(int(logits.argmax())==truth)!=a["donor_correct"]):
                            raise ValueError("original native output correct-target arithmetic differs")
                        checkclose(logsumexp(logits)-float(logits[truth]),a["donor_goal_CE"],"goal CE",atol=1e-6,rtol=1e-7)
                        if name=="full_donor_state":
                            z=np.asarray(a["all_final_logits"],np.float64)
                            if z.shape!=(81,10):raise ValueError("original full donor head omitted")
                            if float(np.max(np.abs(z-natural)))>1e-5:
                                raise ValueError("whole reached donor head parity invalid")
            root_s=root["summary"]
            effects=[sum(base_groups[i])/8 for i in range(N)]
            effect=sum(effects)/N;ci=boot(effects,sid)
            for a,b in zip(ci,root_s["whole_base_bootstrap95"]):
                checkclose(a,b,"registered source-cluster bootstrap",atol=1e-10,rtol=0.)
            checkclose(effect,root_s["primary_paired_ITT_effect"],"registered mean effect",atol=1e-12,rtol=0.)
            positive=bool(n_joint/(N*8)>=.8 and maxparity<=1e-4
                          and full_correct/n_joint>=.95 and effect>=.15 and ci[0]>0)
            if positive!=root_s["development_teacher_gate"]:
                raise ValueError("original development teacher status changed")
            reports[str(sid)]={"selected_minus_median_sham":effect,
                               "bootstrap95":ci,"n_competent":n_joint,
                               "geometry_admitted":n_admitted,
                               "full_donor_parity":maxparity,
                               "engineering_gate":positive}
        if vectors.read(1)!=b"" or sources.read(1)!=b"":
            raise ValueError("trailing binary records after full roster")
    if (vector_index!=21504 or source_index!=1536 or
        vec_digest.hexdigest()!=d["binary_vectors"]["raw_sha256"] or
        src_digest.hexdigest()!=d["binary_sources"]["raw_sha256"]):
        raise ValueError("source/full physical edit binary archive count/hash changed")
    verdict="DEVELOPMENT_ENGINEERING_POSITIVE" if sum(x["engineering_gate"] for x in reports.values())>=2 else "DEVELOPMENT_NULL_OR_INELIGIBLE"
    if verdict!=d["registered_first_development_verdict"]:
        raise ValueError("original prospective teacher/whole-study gate changed")
    result={"schema":SCHEMA,"audit_status":"PASS","evidence_class":"POST_RESULT_QA_ONLY",
            "original_first_native_SHA":HASHES["full.json"],
            "original_pretest_lock_SHA":HASHES["lock.json"],
            "vectors_audited":vector_index,"sources_audited":source_index,
            "all_vector_SHAs_pass":True,
            "first_registered_development_verdict":verdict,"teachers":reports}
    outpath=Path(outpath);outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("LGA07C_POST_RESULT_NUMPY_QA_PASS",verdict,flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("lock","full","vectors","sources","out"):
        p.add_argument("--"+n,required=True,type=Path)
    a=p.parse_args()
    do_audit(a.lock,a.full,a.vectors,a.sources,a.out)
